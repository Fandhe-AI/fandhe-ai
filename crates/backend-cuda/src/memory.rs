//! CUDA バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `fandhe_ai_tensor_core::buffer::MemoryOps` の CUDA 実装。既存の GEMM 実装
//! （`gemm.rs`）に埋め込まれていたホスト⇔デバイス転送（`clone_htod`/
//! `alloc_zeros`/`clone_dtoh`）を、演算から独立した「確保・転送・解放」
//! 抽象として切り出す（`docs/public-api-design.md` §4.2）。
//!
//! `CudaMemory` は [`CudaDevice`] の `Arc<CudaStream>` を共有するのみで、
//! `CudaDevice::new` が経由する `is_culib_present()` パニック回避ゲート
//! （`device.rs` モジュールコメント参照）は `CudaMemory::new` 呼び出し
//! 時点で既に通過済みの `CudaDevice` を要求することで間接的に共有する
//! （`CudaMemory` 自身は driver API を新たに直接呼ばない）。
//!
//! 解放は [`CudaSlice`] の `Drop` に一本化する（`cudarc-0.19.8` の
//! `CudaSlice<T>` は内部で `Arc<CudaStream>` を co-own しており、`Drop`
//! 実装がストリーム上で `cuMemFreeAsync`/`cuMemFree` を呼ぶ。
//! `cudarc-0.19.8/src/driver/safe/core.rs` の `impl<T> Drop for
//! CudaSlice<T>` 参照）。本モジュールは明示 `free()` を持たない
//! （`fandhe_ai_tensor_core::buffer` モジュールコメント「解放方針」と同じ RAII
//! 一本化方針）。[`UnifiedSlice`]（下記「配置（managed 拡張）」節）も
//! 同じ RAII 一本化方針だが、`Drop` の中身が異なる（`event.synchronize()`
//! による同期 free。`CudaStorage` ドキュメンテーションコメント参照）。
//!
//! ## 配置（managed 拡張。イシュー #1352）
//!
//! `alloc_zeroed`／`upload` が確保する実バッファは、既定では
//! `cuMemAlloc`（[`CudaSlice`]）だが、`crate::placement::
//! managed_placement_enabled()` が `true` の opt-in 時は
//! `cuMemAllocManaged`（[`UnifiedSlice`]）へ切り替わる（DGX Spark GB10
//! のような物理統合メモリ環境向け。`crate::placement` モジュール冒頭
//! コメントの契約参照）。`CudaStorage`（crate 内部限定型）がこの 2 配置を
//! crate 内部で統一的に扱う列挙型で、`CudaArg`／`CudaArgMut` が両配置を
//! 同一のカーネル起動経路（`PushKernelArg`）へ橋渡しする。既定（フラグ OFF）
//! では常に `CudaStorage::Device` のみが生成されるため、本イシュー
//! 導入前との出力 bit 同一性は経路の分岐自体が発生しないことにより
//! 機構として保証される。

use std::any::Any;
use std::mem::size_of;
use std::ops::RangeBounds;
use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, DevicePtr, DeviceRepr, LaunchArgs, PushKernelArg,
    UnifiedSlice, UnifiedView,
};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::placement;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
use fandhe_ai_tensor_core::pool::PoolZeroFill;

/// CUDA バッファの具体ハンドル。
///
/// `numel == 0`（空テンソルの契約。`fandhe_ai_tensor_core::buffer` モジュール
/// コメント参照）では `slice` を `None` とし、`cuMemAlloc` 自体を呼ばない
/// （一部環境の driver は 0 バイト確保を拒否する。`gemm.rs` の `k == 0`
/// 早期 return コメントと同じ理由）。`CudaSlice<T>` は `#[derive(Debug)]`
/// されているため本型も `Debug` を導出できる。
///
/// `_alloc`（[`TrackedAllocation`]）は TASK-14.1b（#175）で追加した。
/// `slice` より後に宣言しているため、フィールドは宣言順に drop される
/// Rust の規則により `slice`（`CudaSlice::drop` が `cuMemFreeAsync`／
/// `cuMemFree` をストリーム上で発行する。モジュール冒頭コメント「解放は
/// `CudaSlice` の `Drop` に一本化する」参照）の後に `_alloc` が drop
/// される。`TrackedAllocation::drop` は `slice` の中身を参照せず、確保時に
/// 記録したバイト数を `AllocationTracker` へ返すだけ（`backend-cpu::
/// CpuBufferHandle` の `_alloc` と同型のコメント。`memory_stats.rs`
/// モジュールコメント「トラッカーの共有範囲」参照）であるため、
/// `cuMemFreeAsync` の実処理が非同期であっても計測上の問題にはならない
/// （計測は「ハンドル Drop 時点の論理解放」を数える。CPU と同一の
/// 「確保済みバイト数」セマンティクス）。
/// `pub(crate)`（イシュー #935・`docs/device-resident-update-design.md`
/// §3.2 で `ops.rs::CudaBackendOps::sgd_step_device`／`sgd.rs::CudaSgd::run`
/// が `DeviceBuffer::downcast_handle_mut` 経由で in-place 書き換えを行う
/// ために `crate::memory::CudaBufferHandle` として参照する必要があり、
/// 可視性を crate 内に広げた。`backend-cpu::CpuBufferHandle` と同じ判断）。
#[derive(Debug)]
pub(crate) struct CudaBufferHandle {
    pub(crate) storage: Option<CudaStorage>,
    _alloc: TrackedAllocation,
    /// 確保時点の ordinal 世代（`buffer.rs::DeviceBuffer::generation` と
    /// 同じ値をハンドル自身にも刻印する。codex-review P0 指摘・PR #1064
    /// 追補）。`DeviceBuffer::generation` は `PooledMemory`（`tensor-core::
    /// pool`）がプールから再利用したバッファを新規 `DeviceBuffer::new`
    /// （既定世代 0）で包み直す際に失われる（`PoolZeroFill::zero_fill`
    /// は `&mut dyn BufferHandle` のみを受け取り `DeviceBuffer` を経由
    /// しないため、プール再利用時の世代情報を運ぶ手段が
    /// `DeviceBuffer::generation` には存在しない）。ハンドル自身に世代を
    /// 持たせることで、プール経由で再利用されても
    /// `zero_fill`（本ファイル下部 `PoolZeroFill` 実装）が正しい世代
    /// 検査を行える。
    pub(crate) generation: u64,
}

impl BufferHandle for CudaBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// `CudaBufferHandle` が実際に保持する確保済みメモリの配置（イシュー
/// #1352。モジュール冒頭コメント「配置（managed 拡張）」参照）。
///
/// `Device`（既定・`cuMemAlloc`／[`CudaSlice`]）と `Managed`（opt-in・
/// `cuMemAllocManaged`／[`UnifiedSlice`]）の 2 通り。`crate::placement::
/// managed_placement_enabled()` が `false`（既定）の間は `alloc_zeroed_inner`／
/// `upload_inner` が `Managed` を生成することはなく、`Device` 一択の経路は
/// 本イシュー導入前と完全に同一（分岐そのものが発生しない）。
///
/// `Drop` の差分（呼び出し元が意識すべき唯一の非対称性）: `CudaSlice::drop`
/// は該当ストリーム上に `cuMemFreeAsync`（デバイス側の完了を待つのみ）を
/// 発行する非同期解放だが、`UnifiedSlice::drop`（cudarc-0.19.8
/// `unified_memory.rs:46-53`）は `event.synchronize()` の後に同期
/// `cuMemFree` を呼ぶ**同期解放**である。いずれも cudarc 内部の
/// `record_err` でエラーを記録するのみで、本クレートの `with_driver_call`
/// （poison 状態機械）を経由しない点は両者で対称（既存の `CudaSlice::drop`
/// と同じ既知のギャップであり、本イシューが新たに導入するものではない）。
#[derive(Debug)]
pub(crate) enum CudaStorage {
    Device(CudaSlice<f32>),
    Managed(UnifiedSlice<f32>),
}

impl CudaStorage {
    /// 要素数（配置に依らない）。
    pub(crate) fn len(&self) -> usize {
        match self {
            CudaStorage::Device(s) => s.len(),
            CudaStorage::Managed(s) => s.len(),
        }
    }

    /// カーネル起動の読み取り専用引数として渡せる形に変換する
    /// （[`CudaArg`]。`ops.rs`／`gemm.rs`／`sgd.rs` の `*_arg` 系入口が
    /// 使う）。
    pub(crate) fn as_arg(&self) -> CudaArg<'_> {
        match self {
            CudaStorage::Device(s) => CudaArg::Slice(s),
            CudaStorage::Managed(s) => CudaArg::Unified(s),
        }
    }

    /// カーネル起動の書き込み可能引数として渡せる形に変換する
    /// （[`CudaArgMut`]）。
    pub(crate) fn as_arg_mut(&mut self) -> CudaArgMut<'_> {
        match self {
            CudaStorage::Device(s) => CudaArgMut::SliceMut(s),
            CudaStorage::Managed(s) => CudaArgMut::UnifiedMut(s),
        }
    }

    /// `bounds`（要素インデックス範囲）の部分ビューを読み取り専用引数
    /// として返す（`DeviceParamStore` の連結バッファから個々のパラメータ
    /// を切り出す `ops.rs::CudaBackendOps::gemm_resident_rhs` 等が使う。
    /// `DeviceBufferView::new`〈tensor-core〉が offset+numel の範囲検査を
    /// 構築時に済ませているため、ここでの追加検証は不要）。
    pub(crate) fn view(&self, bounds: impl RangeBounds<usize>) -> CudaArg<'_> {
        match self {
            CudaStorage::Device(s) => CudaArg::View(s.slice(bounds)),
            CudaStorage::Managed(s) => CudaArg::UnifiedView(s.slice(bounds)),
        }
    }
}

/// 配置非依存の読み取り専用カーネル引数（イシュー #1352）。
///
/// `cudarc` の `PushKernelArg` はカーネル起動直前に引数フィールドの
/// アドレスをそのまま `LaunchArgs::args` へ積む実装（`cudarc-0.19.8
/// src/driver/safe/launch.rs` の各 `PushKernelArg` 実装参照）のため、
/// `View`／`UnifiedView`（値として保持する `CudaView`／`UnifiedView`）を
/// 積んだ本列挙体自身が `.launch()` 呼び出しまで（ムーブされず）生存し
/// 続けなければならない。呼び出し元は本値を `launch_builder` 呼び出しの
/// 前に名前付きローカル変数として宣言し、`.push()` 呼び出しチェーンの中で
/// その場限りの一時値として構築しない（`LaunchArgs<'a>` は `'a` について
/// 不変であるため、遅延構築は借用エラーになる）。
pub(crate) enum CudaArg<'a> {
    Slice(&'a CudaSlice<f32>),
    View(CudaView<'a, f32>),
    Unified(&'a UnifiedSlice<f32>),
    UnifiedView(UnifiedView<'a, f32>),
}

impl<'d> CudaArg<'d> {
    /// 要素数（配置・部分ビューに依らない）。`gemm.rs`／`sgd.rs` の
    /// `*_arg` 系入口が既存の境界検証（`validate_gemm_dims` 等。REQ-8）を
    /// そのまま適用できるよう、`CudaSlice::len`／`CudaView::len`／
    /// `UnifiedSlice::len`／`UnifiedView::len` へ委譲する。
    pub(crate) fn len(&self) -> usize {
        match self {
            CudaArg::Slice(s) => s.len(),
            CudaArg::View(v) => v.len(),
            CudaArg::Unified(s) => s.len(),
            CudaArg::UnifiedView(v) => v.len(),
        }
    }

    /// `builder` へ本引数を積む。`CudaArg` のバリアントに応じて
    /// cudarc 側の対応する `PushKernelArg` 実装（`&CudaSlice`／
    /// `&CudaView`／`&UnifiedSlice`／`&UnifiedView`）へ委譲するだけの
    /// 薄い分岐であり、カーネル本体・起動 config は配置に依らず完全に
    /// 共有する（出力 bit 同一契約の根拠）。
    pub(crate) fn push<'a>(&'a self, builder: &mut LaunchArgs<'a>)
    where
        'd: 'a,
    {
        match self {
            CudaArg::Slice(s) => {
                builder.arg(*s);
            }
            CudaArg::View(v) => {
                builder.arg(v);
            }
            CudaArg::Unified(s) => {
                builder.arg(*s);
            }
            CudaArg::UnifiedView(v) => {
                builder.arg(v);
            }
        }
    }
}

/// 配置非依存の書き込み可能カーネル引数（[`CudaArg`] の可変版）。
/// 呼び出し規約は [`CudaArg::push`] と同一。
///
/// `View`／`UnifiedView` 相当の可変部分ビュー variant は持たない
/// （`CudaStorage::as_arg_mut` が常にバッファ全体を返す契約のため。
/// 出力バッファ〈`c_dev`〉はいずれも `CudaMemory::alloc_zeroed` が
/// 新規確保した全体バッファであり、`DeviceParamStore` の連結バッファの
/// 部分範囲へ書き込む呼び出し元は本イシュー時点で存在しない。必要に
/// なった時点で [`CudaArg::View`]／`UnifiedView` と対称な variant を
/// 追加する）。
pub(crate) enum CudaArgMut<'a> {
    SliceMut(&'a mut CudaSlice<f32>),
    UnifiedMut(&'a mut UnifiedSlice<f32>),
}

impl<'d> CudaArgMut<'d> {
    /// [`CudaArg::len`] の可変版。
    pub(crate) fn len(&self) -> usize {
        match self {
            CudaArgMut::SliceMut(s) => s.len(),
            CudaArgMut::UnifiedMut(s) => s.len(),
        }
    }

    /// [`CudaArg::push`] の可変版。`'d: 'a` の理由は同じ（`self` が持つ
    /// 参照・値の生存期間 `'d` は、`builder`〈`LaunchArgs<'a>`〉が要求する
    /// 借用期間 `'a` より長い必要がある）。
    pub(crate) fn push<'a>(&'a mut self, builder: &mut LaunchArgs<'a>)
    where
        'd: 'a,
    {
        match self {
            CudaArgMut::SliceMut(s) => {
                builder.arg(&mut **s);
            }
            CudaArgMut::UnifiedMut(s) => {
                builder.arg(&mut **s);
            }
        }
    }
}

/// `MemoryOps` の CUDA 実装。`CudaDevice::new` を経由して初期化済みの
/// ハンドルからのみ構築できる（受け入れ条件「CUDA 非搭載環境で実行時に
/// panic せず型付きエラーが返る」を、確保・転送呼び出し前の構築段階から
/// 一貫させるため）。
///
/// `tracker`（TASK-14.1b・#175）は `Arc` で共有されるため `Clone` は
/// 「同一計測系列への参照複製」を意味する（`backend-cpu::CpuMemory` の
/// `Clone` doc コメントと同型の契約）。`stream`（`Arc<CudaStream>`）・
/// `ordinal`（`usize`）はいずれも安価に複製できるため、`derive(Clone)`
/// で構造体全体を複製しても新たな driver リソースは確保されない。
#[derive(Clone)]
pub struct CudaMemory {
    stream: Arc<CudaStream>,
    ordinal: usize,
    tracker: Arc<AllocationTracker>,
    /// `device.managed_memory_supported()` の複製（イシュー #1352）。
    /// `alloc_zeroed_inner`／`upload_inner` が `crate::placement::
    /// managed_placement_enabled()` の opt-in 時に、driver 呼び出し前の
    /// fail-closed 事前検査として参照する（`CudaContext` は
    /// `self.stream.context()` から取得できるため、別途フィールドとして
    /// 保持しない）。
    managed_supported: bool,
}

impl CudaMemory {
    /// 初期化済みの [`CudaDevice`] から `CudaMemory` を構築する。
    /// `device.stream()` を `Arc` クローンで共有する（`gemm.rs::CudaGemm::new`
    /// と同じ共有契約）。新規の計測系列を持つトラッカーを生成する
    /// （`backend-cpu::CpuMemory::new` と同型。同一プロセス内でピークを
    /// 集約したい場合は `clone()` でトラッカーを共有する）。
    pub fn new(device: &CudaDevice) -> Self {
        Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            tracker: Arc::new(AllocationTracker::new()),
            managed_supported: device.managed_memory_supported(),
        }
    }
}

/// [`MemoryStats`] の CUDA 実装（TASK-14.1b・#175）。`backend-cpu::
/// CpuMemory` と同一シグネチャで `tracker` へ委譲する。REQ-14 の受け入れ
/// 条件（CPU/CUDA/Metal で同一 API からピーク値が取得できる）を満たす。
impl MemoryStats for CudaMemory {
    fn allocated_bytes(&self) -> u64 {
        self.tracker.allocated_bytes()
    }

    fn peak_allocated_bytes(&self) -> u64 {
        self.tracker.peak_allocated_bytes()
    }

    fn reset_peak(&self) {
        self.tracker.reset_peak();
    }
}

/// カーネル起動直後の都度 `synchronize()` を除去した非同期実行契約
/// （イシュー #1013・`docs/backend-cuda-async-execution-design.md` §3〜
/// §4）の下で、ホストへ結果を読み戻す全ての readback 経路が共有する
/// 唯一の同期点。`clone_dtoh` は `cuMemcpyDtoHAsync` を発行する非同期
/// コピー（`cudarc-0.19.8` `core.rs::memcpy_dtoh`）のため、呼び出し
/// 直後にホスト側データが確定していることを保証するには `clone_dtoh`
/// → `synchronize` の順が必須（逆順ではコピー自体の完了を待てない）。
/// 起動元のカーネルが `unsafe { stream.launch(..) }` を経て投入した
/// 非同期作業も、同一ストリーム上の FIFO 順序保証により本関数の
/// `synchronize` で合わせて完了が確定する（`CudaDevice` は ordinal ごとに
/// 単一ストリームを共有する。設計文書 §3「実行モデル」）。
/// `download_inner`（本ファイル）・`gemm.rs`／`gemm_wmma.rs`／
/// `gemm_mma.rs`／`gemm_mma_tf32.rs` の `download_f32`／`download_f16`・
/// 各演算のホスト `Tensor` 返却ラッパーはすべて本関数を経由し、
/// 「同期点は D2H 境界のみ」という契約を単一箇所に集約する。
pub(crate) fn readback<T, Src>(stream: &Arc<CudaStream>, dev: &Src) -> Result<Vec<T>, CudaError>
where
    T: DeviceRepr,
    Src: DevicePtr<T>,
{
    let host = stream.clone_dtoh(dev)?;
    stream.synchronize()?;
    Ok(host)
}

/// `CudaStorage::Managed`（[`UnifiedSlice`]）専用の readback（イシュー
/// #1352）。[`readback`] と異なり `cuMemcpyDtoHAsync` を発行しない
/// （managed memory は既にホストから直接アクセス可能なアドレス空間に
/// あるため、`clone_dtoh` 経由のコピーは managed 配置の目的〈ゼロコピー〉
/// を損なう）。
///
/// **同期契約（`UnifiedSlice::as_slice` だけでは不十分な理由）**:
/// `UnifiedSlice::as_slice`（cudarc-0.19.8 `unified_memory.rs:447-450`）は
/// 内部の `self.event.synchronize()` のみを待つが、この `event` は
/// `LaunchArgs::launch`（`cudarc-0.19.8 launch.rs:100-135`）が
/// `self.stream.context().is_managing_stream_synchronization()`（複数
/// ストリームを跨ぐ場合のみ true）の場合にだけ記録する。本クレートは
/// `CudaDevice` が ordinal ごとに単一ストリームしか持たない構成
/// （`docs/backend-cuda-async-execution-design.md` §3「実行モデル」）の
/// ため、この event には何も記録されず、`as_slice()` だけでは直前に
/// 投入したカーネルの完了を待てない。そのため本関数は `readback` と
/// 同じく明示的に `stream.synchronize()` を先に呼んでから
/// `as_slice()` でホストスライスを取得する（`with_driver_call` の中で
/// 呼ばれるため、`synchronize` の sticky エラーは通常の poison 経路で
/// 観測される）。
fn host_readback(stream: &Arc<CudaStream>, dev: &UnifiedSlice<f32>) -> Result<Vec<f32>, CudaError> {
    stream.synchronize()?;
    Ok(dev.as_slice()?.to_vec())
}

/// `numel` 分の `f32` 確保が消費するバイト数を検査付きで計算する
/// （TASK-14.1b・#175。`backend-cpu::memory::checked_byte_len` と同型の
/// checked 乗算。`checked_numel` の後段検証として配置する。外部由来の
/// shape がこの経路へ流入しうるための OWASP A03 対策）。計測専用の
/// ヘルパーであり、確保サイズ自体は `checked_numel`／`cudarc` 側の検証を
/// 経由済みのため、本関数はオーバーフロー時のみ `CudaError` を返す。
fn checked_byte_len(numel: usize) -> Result<u64, CudaError> {
    let bytes = numel
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("allocation byte length overflows usize: numel={numel}"),
        })?;
    Ok(bytes as u64)
}

/// shape の要素数積を検査付きで計算する（`gemm.rs::validate_gemm_dims` と
/// 同種の OWASP A03 前段検証。外部由来の shape がこの経路へ流入しうる）。
fn checked_numel(shape: &[usize]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("shape element count overflows usize: {shape:?}"),
        })
}

/// `CudaError` を `BackendError` へ変換する（転送系呼び出し用）。
///
/// `TransferFailed`（TASK-1.9b で追加。`fandhe_ai_tensor_core::device` 参照）は
/// 確保済みバッファへのコピー（`clone_htod`/`clone_dtoh`）の失敗を表す。
/// `CudaError::Driver` は `clone_htod`/`clone_dtoh`（`upload`/`download`）
/// と `alloc_zeros`（`alloc_zeroed`）の両方から生じうるが、同じ
/// `driver::result::DriverError` にラップされ区別できないため、
/// `alloc_zeroed` 側は本関数を使わず `map_cuda_alloc_error` を使う
/// （Bugbot 指摘: `alloc_zeros` 失敗が `TransferFailed` に化けていた
/// バグの修正）。`CudaError` は `#[non_exhaustive]` のため、将来の
/// variant 追加に対しても構造上フォールバックできるよう
/// `KernelLaunchFailed` を wildcard の受け皿とする（`Compile`/
/// `TensorCoreUnsupported` はこのモジュールの呼び出し経路からは発生
/// しないが、`non_exhaustive` ゆえに網羅的 match は書けない）。
pub(crate) fn map_cuda_error(err: CudaError) -> BackendError {
    match err {
        CudaError::DriverUnavailable { detail } => BackendError::CudaUnavailable(detail),
        CudaError::NvrtcUnavailable { detail } => BackendError::CudaUnavailable(detail),
        CudaError::InvalidShape { detail } => BackendError::DeviceAllocationFailed(detail),
        CudaError::Driver(e) => BackendError::TransferFailed(format!("{e:?}")),
        // managed 配置 opt-in（`crate::placement`）が非対応デバイスで
        // 要求された場合の fail-closed 拒否（イシュー #1352）。driver
        // 呼び出しに到達していないため `Unsupported`（`CudaUnavailable`
        // ほど致命的ではなく、呼び出し側の設定ミスに近い）へマップする。
        CudaError::ManagedMemoryUnsupported { detail } => BackendError::Unsupported(detail),
        other => BackendError::KernelLaunchFailed(format!("{other}")),
    }
}

/// `CudaError` を `BackendError` へ変換する（`alloc_zeroed` 専用）。
///
/// `DeviceAllocationFailed` は確保そのものの失敗（`alloc_zeros` 由来）を
/// 表す契約（`fandhe_ai_tensor_core::device::BackendError` ドキュメンテーション
/// コメント参照）。`map_cuda_error` と異なり `CudaError::Driver` を
/// `DeviceAllocationFailed` にマップする点のみが差分である
/// （`alloc_zeroed_inner` 内で `CudaError::Driver` を生じさせるのは
/// `alloc_zeros` 呼び出しのみであり、転送系呼び出しを含まないため
/// 区別が付く）。
fn map_cuda_alloc_error(err: CudaError) -> BackendError {
    match err {
        CudaError::Driver(e) => BackendError::DeviceAllocationFailed(format!("{e:?}")),
        other => map_cuda_error(other),
    }
}

impl CudaMemory {
    /// managed 配置 opt-in（`crate::placement::managed_placement_enabled()`）
    /// が要求されている場合の driver 呼び出し前 fail-closed 事前検査
    /// （イシュー #1352。`crate::error::CudaError::ManagedMemoryUnsupported`
    /// ドキュメンテーションコメント参照）。
    ///
    /// **呼び出し契約**: 本関数は呼び出し元が既に `placement::
    /// managed_placement_enabled()` を読んで managed 分岐へ入った後にのみ
    /// 呼び出すこと（`alloc_zeroed_inner`／`upload_inner` 参照）。ここで
    /// フラグを再読しない（codex-review 指摘。PR #1395）: フラグはプロセス
    /// グローバル（`AtomicBool` 等）であり、外側の分岐判定と本関数呼び出し
    /// の間に別スレッドが OFF へ変更すると、再読した場合は
    /// `enabled() && !managed_supported` が短絡評価で `false` になり
    /// `managed_supported == false`（非対応デバイス）でも検査を素通りして
    /// `alloc_unified` に到達してしまう（MANAGED_MEMORY=1・
    /// CONCURRENT_MANAGED_ACCESS=0 のデバイスで安全条件を満たさないまま
    /// 確保する fail-open バグ）。よってここでは呼び出し元が確定させた
    /// 分岐に従い `self.managed_supported` のみを無条件に検査する
    /// （flag が OFF の間は本関数自体が呼ばれない設計のため、
    /// `managed_placement_enabled()` の値に依存しない）。
    fn check_managed_placement_supported(&self) -> Result<(), CudaError> {
        if !self.managed_supported {
            return Err(CudaError::ManagedMemoryUnsupported {
                detail: format!(
                    "managed memory placement is opt-in enabled but device (ordinal={}) does not \
                     support CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY / \
                     CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
                    self.ordinal
                ),
            });
        }
        Ok(())
    }

    fn alloc_zeroed_inner(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, CudaError> {
        let numel = checked_numel(shape)?;
        // イシュー #1013 設計文書 §9 item 7: 確保時点の ordinal 世代を
        // 先に確定させ、`DeviceBuffer`（`new_with_generation`）と
        // `CudaBufferHandle`（`generation` フィールド。codex-review P0
        // 指摘・PR #1064 追補。`CudaBufferHandle` ドキュメンテーション
        // コメント参照）の両方へ同じ値を刻印する。
        let generation = context_cache::current_generation(self.ordinal);
        // 計測（`TrackedAllocation::new`）は確保成功後に行う。確保が
        // 失敗しうる `?` の前でカウントすると、失敗した確保が一時的に
        // ピークへ計上されてしまう（`backend-cpu::CpuMemory` と同じ順序
        // 契約。TASK-14.1b・#175）。
        let handle: Box<dyn BufferHandle> = if numel == 0 {
            // 空テンソルの契約（`fandhe_ai_tensor_core::buffer` モジュールコメント）:
            // FFI を呼ばず空ハンドルを返す。0 バイトの `TrackedAllocation`
            // は current・peak いずれも変化させない no-op（`memory_stats`
            // モジュールコメント参照）だが、他バックエンドと契約を対称に
            // 保つため明示的に保持する。
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            Box::new(CudaBufferHandle {
                storage: None,
                _alloc: alloc,
                generation,
            })
        } else if placement::managed_placement_enabled() {
            self.check_managed_placement_supported()?;
            // SAFETY: `alloc_unified::<f32>` は「T が任意ビットパターンで
            // 有効か cudarc 側で保証しない」ことのみを理由に unsafe
            // （cudarc-0.19.8 `unified_memory.rs:88-93`）。ここでは `f32`
            // を確保しており、`f32` に無効なビットパターンは存在しない
            // （NaN／inf を含め全ビットパターンが有効な浮動小数点表現に
            // なる）。加えて確保直後の内容は本節直後の `memset_zeros`
            // でゼロ埋めしてから初めて呼び出し元へ公開する（`pool.rs::
            // CudaAllocator::alloc_uninit` の `unsafe { stream.alloc }` と
            // 同一クラスの安全性根拠。呼び出し元へ渡す前に必ず全域を
            // 書き切る）。`attach_global`（`true`）は本クレートが
            // `CudaDevice` ごとに単一ストリームしか使わない構成
            // （`docs/backend-cuda-async-execution-design.md` §3）のため、
            // 複数ストリーム間の所有権譲渡を要する `CU_MEM_ATTACH_HOST`／
            // `CU_MEM_ATTACH_SINGLE` は不要。
            let mut unified = unsafe { self.stream.context().alloc_unified::<f32>(numel, true)? };
            self.stream.memset_zeros(&mut unified)?;
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
            Box::new(CudaBufferHandle {
                storage: Some(CudaStorage::Managed(unified)),
                _alloc: alloc,
                generation,
            })
        } else {
            let slice = self.stream.alloc_zeros::<f32>(numel)?;
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
            Box::new(CudaBufferHandle {
                storage: Some(CudaStorage::Device(slice)),
                _alloc: alloc,
                generation,
            })
        };
        Ok(DeviceBuffer::new_with_generation(
            Device::Cuda(self.ordinal),
            shape.to_vec(),
            handle,
            generation,
        ))
    }

    fn upload_inner(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, CudaError> {
        let shape = tensor.shape().to_vec();
        let generation = context_cache::current_generation(self.ordinal);
        if tensor.numel() == 0 {
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                storage: None,
                _alloc: alloc,
                generation,
            });
            return Ok(DeviceBuffer::new_with_generation(
                Device::Cuda(self.ordinal),
                shape,
                handle,
                generation,
            ));
        }
        // 非 contiguous な入力は実体化してから転送する（`MemoryOps::upload`
        // の契約。`fandhe_ai_tensor_core::buffer` モジュールコメント参照）。
        let contiguous = tensor.contiguous();
        let data = contiguous
            .as_slice()
            .ok_or_else(|| CudaError::InvalidShape {
                detail: "contiguous() の直後にもかかわらず as_slice が None を返した \
                     （tensor-core 側のロジック不整合。到達しないはずの防御経路）"
                    .to_string(),
            })?;
        let bytes = checked_byte_len(data.len())?;
        let storage = if placement::managed_placement_enabled() {
            self.check_managed_placement_supported()?;
            // SAFETY: 上記 `alloc_zeroed_inner` の SAFETY コメントと同一
            // 根拠（f32 は全ビットパターン有効）。ここでは新規確保
            // 直後に `data` 全域を `copy_from_slice` で上書きするため
            // （`alloc_unified` 直後の未初期化内容が露出することはない）、
            // ゼロ埋め（`memset_zeros`）は不要。
            let mut unified = unsafe {
                self.stream
                    .context()
                    .alloc_unified::<f32>(data.len(), true)?
            };
            // managed memory はホストから直接書き込めるため、
            // `cuMemcpyHtoD`（`clone_htod`）を発行しない（H2D 往復を
            // 避ける本イシューの目的）。新規確保のバッファであり在飛
            // カーネル作業は存在しないため、`as_mut_slice` の内部
            // `event.synchronize()`（何もしていない新規 event）はコスト
            // にならない。
            unified.as_mut_slice()?.copy_from_slice(data);
            CudaStorage::Managed(unified)
        } else {
            CudaStorage::Device(self.stream.clone_htod(data)?)
        };
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
            storage: Some(storage),
            _alloc: alloc,
            generation,
        });
        Ok(DeviceBuffer::new_with_generation(
            Device::Cuda(self.ordinal),
            shape,
            handle,
            generation,
        ))
    }

    fn download_inner(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, CudaError> {
        let handle = buffer
            .downcast_handle::<CudaBufferHandle>()
            .ok_or_else(|| CudaError::InvalidShape {
                detail: "buffer handle is not a CudaBufferHandle (device mismatch)".to_string(),
            })?;
        let data = match &handle.storage {
            None => Vec::new(),
            Some(CudaStorage::Device(slice)) => {
                // 同期点は本モジュール共通の `readback` ヘルパーへ集約
                // 済み（#1013。`fandhe_ai_tensor_core::buffer` モジュール
                // コメント「download の同期契約」参照）。
                readback(&self.stream, slice)?
            }
            Some(CudaStorage::Managed(unified)) => {
                // managed 配置は `cuMemcpyDtoHAsync` を発行しない専用の
                // readback を使う（`host_readback` ドキュメンテーション
                // コメント参照。イシュー #1352）。
                host_readback(&self.stream, unified)?
            }
        };
        Tensor::new(data, buffer.shape()).map_err(|err| CudaError::InvalidShape {
            detail: format!("download produced a shape-inconsistent tensor: {err}"),
        })
    }
}

impl CudaMemory {
    /// `MemoryOps` 実装の各公開メソッド（本 impl 直後の `impl MemoryOps for
    /// CudaMemory`）が唯一の driver 呼び出し境界として使う共通ヘルパー
    /// （イシュー #1013 設計文書 §9 item 9「TOCTOU 回避のため事前検査を
    /// 別ステップにしない」・PR #1064 の Phase C 結線）。
    ///
    /// `begin_driver_call` を演算入口で 1 回だけ呼び（`resource_generations`
    /// に、当該演算が読み書きする既存 `DeviceBuffer` の
    /// [`fandhe_ai_tensor_core::buffer::DeviceBuffer::generation`] を渡す。
    /// 新規確保〈`alloc_zeroed`／`upload`〉には検査対象の既存バッファが
    /// ないため空スライスでよい）、`f` の内部で行われる 1 回以上の driver
    /// 呼び出し（`clone_htod`／`alloc_zeros`／`clone_dtoh`／`synchronize`。
    /// いずれも `?` で直結しているため、最初に失敗した 1 回だけが
    /// `CudaError::Driver` として `f` の戻り値に現れる）の結果を
    /// `observe_cuda_result` で観測し、sticky エラーなら ordinal を
    /// poison する（`context_cache::observe_cuda_result` ドキュメンテー
    /// ションコメント参照）。最終的な `CudaError` は呼び出し元が渡す
    /// `map` で `BackendError` へ変換する（`alloc_zeroed` は
    /// `map_cuda_alloc_error`、`upload`／`download` は `map_cuda_error`
    /// と、呼び出し元ごとに異なる variant 割り当てを保つため）。
    fn with_driver_call<T>(
        &self,
        resource_generations: &[u64],
        map: impl FnOnce(CudaError) -> BackendError,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, BackendError> {
        let token = context_cache::begin_driver_call(self.ordinal, resource_generations)?;
        context_cache::observe_cuda_result(self.ordinal, &token, f()).map_err(map)
    }
}

impl MemoryOps for CudaMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        self.with_driver_call(&[], map_cuda_alloc_error, || self.alloc_zeroed_inner(shape))
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        self.with_driver_call(&[], map_cuda_error, || self.upload_inner(tensor))
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        // ハンドル型不一致（他バックエンドの `DeviceBuffer` を誤って
        // 渡した場合）は、CPU 実装（`backend-cpu/src/memory.rs`）と
        // 同じ `BackendError::DeviceMismatch` に統一する。`CudaError` を
        // 経由すると `map_cuda_error` で実態と異なるエラー種別
        // （`DeviceAllocationFailed`）に化けてしまうため、ここで直接
        // 判定する（3 バックエンド共通のハンドル型不一致検出。レビュー
        // 指摘対応）。
        if buffer.downcast_handle::<CudaBufferHandle>().is_none() {
            return Err(BackendError::DeviceMismatch);
        }
        // ハンドル型（CudaBufferHandle）が一致しても、複数 GPU 環境では
        // 別 ordinal 上で確保された `CudaSlice` を受理してしまいうる
        // （`CudaBufferHandle` 自体は ordinal を保持しないため、型検査
        // だけでは他デバイス由来のバッファを判別できない）。`self.ordinal`
        // と `buffer.device()` の ordinal が一致することを、実際の
        // driver API 呼び出し（`clone_dtoh`）の前に検証する
        // （Bugbot 指摘: device ordinal 不一致が無視され誤ったストリーム
        // 上でコピーが実行されうるバグの修正）。
        if buffer.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        // `buffer.generation()`（確保時点で刻印済み。`alloc_zeroed_inner`／
        // `upload_inner` 参照）を渡し、`invalidate` による回復後の新世代に
        // 対して旧世代のバッファが誤って読まれることを検出する
        // （イシュー #1013 設計文書 §9 item 7）。
        self.with_driver_call(&[buffer.generation()], map_cuda_error, || {
            self.download_inner(buffer)
        })
    }
}

impl CudaMemory {
    /// **`internal-diagnostics` feature（既定 off）限定の診断専用入口**。
    /// イシュー #1353（codex-review 指摘）: `download()`（`MemoryOps` 実装。
    /// 本ファイル上部）は managed 配置でも `host_readback` が
    /// `UnifiedSlice::as_slice().to_vec()` で通常ホストメモリへコピーして
    /// から `Tensor` を返すため、`tests/managed_placement_bandwidth_
    /// real_device.rs` が測っていた「readback」区間はこのコピー後の
    /// 通常 `Vec<f32>` を読むだけになり、managed ページへの CPU 直接
    /// アクセス帯域を計測できていなかった。本関数は `UnifiedSlice::
    /// as_slice()` が返す借用スライスを**コピーせずそのまま**逐次合計
    /// して読み取り時間（秒）を返すことで、managed ページ自体への CPU
    /// アクセス帯域を計測可能にする。同期契約は `host_readback` と同一
    /// （`stream.synchronize()` を先に呼ぶ理由は同関数のドキュメンテー
    /// ションコメント参照）。`Device`（device-only）配置のバッファは
    /// ホストから直接アクセス可能なアドレスを持たないため
    /// `BackendError::Unsupported` を返す。
    #[cfg(feature = "internal-diagnostics")]
    pub fn measure_managed_direct_read_seconds(
        &self,
        buffer: &DeviceBuffer<f32>,
    ) -> Result<f64, BackendError> {
        let handle = buffer
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if buffer.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        self.with_driver_call(&[buffer.generation()], map_cuda_error, || {
            let unified = match &handle.storage {
                Some(CudaStorage::Managed(unified)) => unified,
                Some(CudaStorage::Device(_)) => {
                    return Err(CudaError::ManagedMemoryUnsupported {
                        detail: "measure_managed_direct_read_seconds は Managed 配置限定"
                            .to_string(),
                    });
                }
                None => return Ok(0.0),
            };
            self.stream.synchronize()?;
            let slice = unified.as_slice()?;
            let t0 = std::time::Instant::now();
            let mut acc = 0.0f64;
            for &v in slice {
                acc += v as f64;
            }
            std::hint::black_box(acc);
            Ok(t0.elapsed().as_secs_f64())
        })
    }
}

/// `fandhe_ai_tensor_core::pool::PooledMemory<CudaMemory>`（TASK-#201・REQ-14 14-3）
/// が再利用バッファを返す前に呼ぶゼロ初期化フック。プール保持中も
/// `CudaBufferHandle::_alloc`（`TrackedAllocation`）は生存し続けるため、
/// 「返却されたが未解放のバッファ」も `allocated_bytes()` に自然に計上
/// され続ける（リークではなく意図した挙動。`fandhe_ai_tensor_core::pool` モジュール
/// の `MemoryStats for PooledMemory<M>` 転送実装〈`pool.rs`〉参照）。
/// 実機でのピーク計測の裏取りは TASK-14.2（#177）で実施する。
/// `CudaStream::memset_zeros`（`cudarc-0.19.8/src/driver/safe/core.rs`）で
/// デバイス側のメモリを直接ゼロクリアする（ホスト往復なし。`alloc_zeros`
/// と同じストリーム上の非同期メモリ操作）。
impl PoolZeroFill for CudaMemory {
    fn zero_fill(&self, handle: &mut dyn BufferHandle) -> Result<(), BackendError> {
        let Some(cuda_handle) = handle.as_any_mut().downcast_mut::<CudaBufferHandle>() else {
            return Err(BackendError::DeviceMismatch);
        };
        // `CudaBufferHandle::generation`（確保時点に刻印済み。
        // `CudaBufferHandle` ドキュメンテーションコメント参照）を通常経路
        // と同じ generation 検査へ渡す（codex-review P0 指摘・PR #1064
        // 追補・`memory.rs:348` 相当: `PoolZeroFill::zero_fill` は
        // `MemoryOps::{alloc_zeroed,upload,download}` と異なり
        // `with_driver_call` の外側で `self.stream.memset_zeros` を直接
        // 呼んでいたため、(a) poison 済み ordinal でもプール再利用時に
        // 拒否されない (b) ここで初めて観測しうる sticky
        // `DriverError` が `observe_cuda_result` に渡らない (c)
        // `invalidate` 後の旧世代 allocation の世代検査もない、という
        // fail-closed 状態機械の迂回経路になっていた）。
        let generation = cuda_handle.generation;
        // 空ハンドル（`numel == 0`）は `pool.rs::PooledMemory::alloc_zeroed`
        // が空テンソル契約によりそもそもプールを介さない経路で扱うため
        // 到達しない想定だが、`CudaBufferHandle::slice` が `None` の場合に
        // 備えて no-op として安全に振る舞う（`buffer.rs` モジュールコメント
        // 「空テンソルの契約」と同じ扱い）。空入力の早期 return でも
        // poison・世代検査は fail-closed に行う（`ops.rs::
        // gemm_resident_rhs`／`gemm_resident_lhs` の空 shape 早期 return
        // と同じ方針。codex-review P1 指摘・PR #1064 追補）。
        let Some(storage) = cuda_handle.storage.as_mut() else {
            context_cache::begin_driver_call(self.ordinal, &[generation])?;
            return Ok(());
        };
        self.with_driver_call(&[generation], map_cuda_error, || match storage {
            CudaStorage::Device(slice) => self.stream.memset_zeros(slice).map_err(CudaError::from),
            CudaStorage::Managed(unified) => {
                self.stream.memset_zeros(unified).map_err(CudaError::from)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 受け入れ条件「CUDA 非搭載環境で実行時に panic せず型付きエラーが
    /// 返る」の `CudaMemory` 版。`CudaDevice::new` が失敗する環境
    /// （self-hosted CI 想定）では `CudaMemory` の構築自体を試みられず、
    /// `CudaMemory::new` を呼ぶ経路そのものに到達しない設計であることを
    /// 確認する（`device_init.rs` の `new_does_not_panic_and_returns_typed_result`
    /// と同じ環境適応パターン）。
    #[test]
    fn cuda_memory_construction_follows_device_init_gate() {
        match CudaDevice::new(0) {
            Ok(device) => {
                // CUDA 搭載環境: CudaMemory を構築できる（panic しない）。
                let _mem = CudaMemory::new(&device);
            }
            Err(_) => {
                // 非搭載環境: CudaDevice::new 自体が型付きエラーで止まる
                // ため、CudaMemory::new を呼ぶ経路に到達しない。
                // panic しないことそのものが検証対象。
            }
        }
    }

    #[test]
    fn map_cuda_error_covers_driver_unavailable() {
        let err = map_cuda_error(CudaError::DriverUnavailable {
            detail: "no libcuda".to_string(),
        });
        assert!(matches!(err, BackendError::CudaUnavailable(msg) if msg.contains("no libcuda")));
    }

    #[test]
    fn map_cuda_error_covers_invalid_shape() {
        let err = map_cuda_error(CudaError::InvalidShape {
            detail: "bad shape".to_string(),
        });
        assert!(
            matches!(err, BackendError::DeviceAllocationFailed(msg) if msg.contains("bad shape"))
        );
    }

    #[test]
    fn map_cuda_alloc_error_labels_driver_failure_as_allocation_failed() {
        // `alloc_zeros` の失敗（`CudaError::Driver`）は、転送系の
        // `map_cuda_error`（`TransferFailed` にマップする）ではなく
        // `map_cuda_alloc_error` で `DeviceAllocationFailed` にマップ
        // されるべきことを検証する（Bugbot 指摘の再発防止）。
        //
        // `cudarc::driver::result::DriverError` を直接構築する公開 API が
        // ないため、`CudaError::InvalidShape` 経由で `map_cuda_alloc_error`
        // が `map_cuda_error` へ委譲するフォールバック経路を確認しつつ、
        // `CudaError::Driver` の分岐そのものはコード上の match アームで
        // `DeviceAllocationFailed` を返すことを構造的に保証している
        // （本関数の定義参照）。
        let err = map_cuda_alloc_error(CudaError::InvalidShape {
            detail: "bad alloc shape".to_string(),
        });
        assert!(
            matches!(err, BackendError::DeviceAllocationFailed(msg) if msg.contains("bad alloc shape"))
        );
    }

    #[test]
    fn download_rejects_mismatched_device_ordinal() {
        // 別 ordinal 上で確保された `DeviceBuffer`（ハンドル型は
        // `CudaBufferHandle` で一致するが device ordinal が異なる）を
        // `download` に渡すと `DeviceMismatch` で拒否されることを検証する
        // （Bugbot 指摘: device ordinal 不一致が無視されるバグの修正）。
        //
        // 実 GPU ドライバ呼び出しは行わない（`numel == 0` の空バッファは
        // `slice: None` で `cuMemcpyDtoHAsync` 等を経由しないため、
        // CUDA 非搭載環境でも到達可能。`CudaMemory::new` 自体は
        // 初期化済み `CudaDevice` を要求するため、既存の
        // `cuda_memory_construction_follows_device_init_gate` と同じ
        // 環境適応ゲートで守る）。
        match CudaDevice::new(0) {
            Ok(device) => {
                let mem = CudaMemory::new(&device);
                // `mem.ordinal` とは異なる ordinal を持つバッファを構築する
                // （実機の ordinal が 0 の場合を考慮し 0 以外を採用）。
                let other_ordinal = mem.ordinal + 1;
                let alloc = TrackedAllocation::new(Arc::clone(&mem.tracker), 0);
                let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                    storage: None,
                    _alloc: alloc,
                    generation: 0,
                });
                let buffer: DeviceBuffer<f32> =
                    DeviceBuffer::new(Device::Cuda(other_ordinal), vec![0], handle);
                let err = mem.download(&buffer).unwrap_err();
                assert!(matches!(err, BackendError::DeviceMismatch));
            }
            Err(_) => {
                // 非搭載環境: CudaDevice::new 自体が型付きエラーで止まる
                // ため、本テストの主張には到達しない（panic しないことが
                // 検証対象）。
            }
        }
    }

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn checked_numel_accepts_ordinary_shape() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel(&[0, 3]).unwrap(), 0);
    }

    #[test]
    fn checked_byte_len_rejects_overflow() {
        let err = checked_byte_len(usize::MAX).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn checked_byte_len_accepts_ordinary_numel() {
        assert_eq!(checked_byte_len(1024).unwrap(), 4096);
        assert_eq!(checked_byte_len(0).unwrap(), 0);
    }

    /// コンパイル時の静的検査。`fn(): T where T: MemoryStats` が
    /// `CudaMemory`／`PooledMemory<CudaMemory>` に対して呼び出せること
    /// 自体が、「CPU/CUDA/Metal で同一 API（同一シグネチャの trait）から
    /// ピーク値が取得できる」という REQ-14 の受け入れ条件を Linux
    /// self-hosted CI（実機非搭載）でも機械検証する（TASK-14.1b・#175。
    /// 実機でのピーク実測は TASK-14.2・#177 で裏取りする）。
    fn assert_memory_stats<T: MemoryStats>() {}

    #[test]
    fn cuda_memory_and_pooled_cuda_memory_implement_memory_stats() {
        assert_memory_stats::<CudaMemory>();
        assert_memory_stats::<fandhe_ai_tensor_core::pool::PooledMemory<CudaMemory>>();
    }

    // ---------------------------------------------------------------
    // `PoolZeroFill::zero_fill` の poison／世代検査回帰テスト
    // （codex-review P0 指摘・`memory.rs:348` 相当・PR #1064 追補）。
    //
    // `CudaMemory` は `stream: Arc<CudaStream>` を必須フィールドに持ち、
    // `CudaMemory::new` は初期化済みの実 `CudaDevice`（実 driver）を
    // 要求する。本ファイルの他テスト（`download_rejects_mismatched_
    // device_ordinal` 等）と同じ理由で、`zero_fill` を実際に呼び出す
    // エンドツーエンドテストは CUDA 搭載環境が必要（`match
    // CudaDevice::new(0) { Ok(_) => .., Err(_) => 空搭載環境としてスキップ
    // }` の環境適応パターンでのみ組める）。
    //
    // 一方 `zero_fill` の poison／世代検査そのもの（`context_cache::
    // begin_driver_call(self.ordinal, &[cuda_handle.generation])` の
    // 早期 return 分岐、および `self.with_driver_call` 経由の
    // `context_cache::observe_cuda_result` 分類）は `self.stream` に
    // 一切触れずに完結する（`zero_fill` の実装本体を参照。poison 済み
    // ordinal では `self.stream.memset_zeros` へ到達する前に
    // `begin_driver_call` が拒否する）。そのためこれらのテストは
    // `zero_fill` が実際に呼ぶのと同じ `context_cache` API を同じ引数
    // 形状（`&[generation]`・`CudaError::from(DriverError)`）で直接
    // 検証することで、実機なしに wiring の正しさを確認する
    // （`ops.rs::tests::with_driver_call_poisons_ordinal_when_construction_
    // closure_returns_sticky_error` と同じ「hardware 非依存プリミティブ
    // レベル検証」方針）。

    fn unique_zero_fill_test_ordinal() -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(40_000);
        NEXT.fetch_add(1, Ordering::SeqCst)
    }

    /// `zero_fill` の空ハンドル早期 return 分岐
    /// （`context_cache::begin_driver_call(self.ordinal, &[generation])`）
    /// は、poison 済み ordinal を fail-closed に拒否する。
    #[test]
    fn zero_fill_early_return_poison_check_rejects_poisoned_ordinal() {
        let ordinal = unique_zero_fill_test_ordinal();
        let generation = context_cache::current_generation(ordinal);

        // context_cache の poison 状態機械を直接操作して poison 化する
        // （`context_cache::poison_state_tests` と同じ手法）。
        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(cudarc::driver::result::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            ))),
        );
        drop(token);

        // `zero_fill` の早期 return 分岐と同一の呼び出し
        // （`self.ordinal` → `ordinal`、`cuda_handle.generation` →
        // `generation`）。
        let result = context_cache::begin_driver_call(ordinal, &[generation]);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では zero_fill の早期 return 分岐も              fail-closed に拒否されるはず: {result:?}"
        );
    }

    /// `zero_fill` の空ハンドル早期 return 分岐は、`invalidate` 後の
    /// 旧世代ハンドル（`cuda_handle.generation` が現行世代と不一致）を
    /// `StaleDeviceGeneration` で拒否する。
    #[test]
    fn zero_fill_early_return_generation_check_rejects_stale_generation() {
        let ordinal = unique_zero_fill_test_ordinal();
        let current = context_cache::current_generation(ordinal);
        assert_eq!(current, 0, "新規 ordinal の現行世代は既定 0 のはず");

        // `cuda_handle.generation` が現行世代（0）と異なる旧世代
        // ハンドルを模す。
        let stale_generation = 1;
        let result = context_cache::begin_driver_call(ordinal, &[stale_generation]);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代ハンドルは zero_fill の早期 return 分岐でも              StaleDeviceGeneration で拒否されるはず: {result:?}"
        );
    }

    /// `zero_fill` の実処理分岐（`self.with_driver_call` 経由の
    /// `memset_zeros` 呼び出し）が sticky な driver エラーを観測した
    /// 場合、対象 ordinal は poison 化され、以降の呼び出し（同一
    /// ordinal 上の `zero_fill`／他の演算いずれも）は
    /// `begin_driver_call` の拒否により fail-closed になる
    /// （`with_driver_call` の実装は `CudaBackendOps::with_driver_call`
    /// と同一パターンのため、`memset_zeros` の代わりに直接
    /// `context_cache::observe_cuda_result` を同じ形状で呼んで検証する）。
    #[test]
    fn zero_fill_real_path_poisons_ordinal_on_sticky_driver_error() {
        let ordinal = unique_zero_fill_test_ordinal();
        let generation = context_cache::current_generation(ordinal);

        // `zero_fill` の `self.with_driver_call(&[generation], ...)` と
        // 同一の呼び出し形状。
        let token =
            context_cache::begin_driver_call(ordinal, &[generation]).expect("begin succeeds");
        let observed = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(cudarc::driver::result::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            ))),
        );
        assert!(observed.is_err());

        let rejected = context_cache::begin_driver_call(ordinal, &[generation]);
        assert!(
            matches!(rejected, Err(BackendError::DeviceContextPoisoned(_))),
            "zero_fill 経路で観測された sticky エラーにより ordinal は poison され、             以降の呼び出しは fail-closed に拒否されるはず: {rejected:?}"
        );
    }

    // -----------------------------------------------------------
    // managed 配置（イシュー #1352）: GPU 非依存の契約テスト。
    // -----------------------------------------------------------

    #[test]
    fn map_cuda_error_covers_managed_memory_unsupported() {
        let err = map_cuda_error(CudaError::ManagedMemoryUnsupported {
            detail: "device does not support managed memory".to_string(),
        });
        assert!(matches!(err, BackendError::Unsupported(msg) if msg.contains("managed memory")));
    }

    #[test]
    fn map_cuda_alloc_error_also_covers_managed_memory_unsupported() {
        // `map_cuda_alloc_error` は `Driver` 以外を `map_cuda_error` へ
        // 委譲するため、`ManagedMemoryUnsupported` も同じ
        // `BackendError::Unsupported` にマップされる（`alloc_zeroed`
        // 経由の managed 確保拒否も `upload` 経由と同じ variant になる
        // ことを確認する）。
        let err = map_cuda_alloc_error(CudaError::ManagedMemoryUnsupported {
            detail: "device does not support managed memory".to_string(),
        });
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    /// `check_managed_placement_supported` は呼び出し元が既に opt-in
    /// フラグを確認して managed 分岐へ入った後の事前検査であり、
    /// **フラグを再読しない**（codex-review 指摘。PR #1395）。この契約を
    /// GPU 非依存に検証する: `managed_supported` フィールドのみで
    /// 判定され、opt-in フラグの現在値（テスト実行順序に依存しうる
    /// プロセスグローバル）には一切影響されないことを、フラグを
    /// 変更しないまま確認する（フラグ非依存の関数であるため
    /// `crate::placement::tests` の直列化ガードは不要）。
    #[test]
    fn check_managed_placement_supported_depends_only_on_managed_supported_field() {
        // `CudaMemory::new` は実 driver 初期化済みの `CudaDevice` を
        // 要求するため、`cuda_memory_construction_follows_device_init_gate`
        // と同じ環境適応パターンで守る（CUDA 非搭載環境では本テストの
        // 主張自体に到達しない。panic しないことが検証対象）。
        if let Ok(device) = CudaDevice::new(0) {
            let mut mem = CudaMemory::new(&device);

            // `managed_supported == true`（デバイスが対応）なら
            // opt-in フラグの値に関わらず常に `Ok(())`。
            mem.managed_supported = true;
            assert!(mem.check_managed_placement_supported().is_ok());

            // `managed_supported == false`（デバイスが非対応）なら
            // opt-in フラグの値に関わらず常に拒否する（フラグを
            // 再読していれば、フラグが OFF に見える瞬間だけこの
            // 拒否がすり抜けてしまう回帰を検出する）。
            mem.managed_supported = false;
            let err = mem
                .check_managed_placement_supported()
                .expect_err("managed_supported=false は無条件で拒否されるべき");
            assert!(matches!(err, CudaError::ManagedMemoryUnsupported { .. }));
        }
    }
}
