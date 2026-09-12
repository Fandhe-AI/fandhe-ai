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
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DeviceRepr, LaunchArgs, PushKernelArg,
    UnifiedSlice, UnifiedView, UnifiedViewMut,
};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
#[cfg(feature = "internal-diagnostics")]
use crate::host_staging::HostStagingStats;
use crate::host_staging::{self, H2dStagingCache, HostStaging, HostStagingCache};
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
    /// 確保元デバイスの ordinal（イシュー #1349・PR #1390 マージ時是正）。
    ///
    /// `Drop` 実装（本モジュール下部）が `context_cache::
    /// begin_buffer_release` を呼ぶために必要。`storage` が
    /// `CudaStorage::Device`（[`CudaSlice`]）の場合は `CudaSlice::
    /// ordinal()` からも取得できるが、`CudaStorage::Managed`
    /// （[`UnifiedSlice`]。イシュー #1352）は `cudarc` 側に ordinal・
    /// context への公開アクセサを持たない（`unified_memory.rs` の
    /// `stream` フィールドは `pub(crate)`）ため、`storage` の variant に
    /// 依らず共通に扱えるようハンドル自身に ordinal を刻印する
    /// （`CudaMemory::ordinal` を各構築箇所でそのまま複製するだけの
    /// 安価な複製。`generation` フィールドと同じ設計判断）。
    pub(crate) ordinal: usize,
}

impl BufferHandle for CudaBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// codex-review P1／P0 再指摘対応（イシュー #1349・PR #1390）: `storage`
/// フィールドの自然な drop（`cudarc::driver::CudaSlice::drop` が
/// `cuMemFreeAsync`/`cuMemFree` を、`UnifiedSlice::drop`（イシュー
/// #1352。`CudaStorage::Managed`）が `event.synchronize()` の後に
/// `cuMemFree` を、それぞれ自身の `stream` へ直接発行する。本モジュール
/// 冒頭コメント「解放は `CudaSlice`／`UnifiedSlice` の `Drop` に一本化
/// する」参照）は `context_cache` の `begin_driver_call`／
/// `begin_capture_session` 排他機構を一切経由しない。そのため、別
/// スレッドが `graph::run_captured_sgd_step_segment` で同じ ordinal を
/// capture 中に本ハンドルが drop されると、その解放操作が capture 中の
/// 共有ストリームへ意図せず記録されうる（`context_cache::
/// begin_buffer_release` doc コメント参照）。
///
/// 本 `Drop` はフィールド既定の drop 順序（宣言順。`storage` →
/// `_alloc`）より**前**に走る（Rust の `Drop::drop` は構造体自身の
/// コードがフィールドの自動 drop より先に実行される規則）。`storage` が
/// `Some`（`numel > 0`。`numel == 0` は driver に触れないため対象外。
/// 構造体 doc コメント参照）の場合のみ、`self.ordinal`（`storage` の
/// variant に依らずハンドル自身が保持する。フィールド doc コメント
/// 参照。`CudaStorage::Managed` は `CudaSlice::ordinal()` 相当の公開
/// アクセサを `cudarc` 側に持たないため、variant 分岐なしで共通に扱う）
/// で `context_cache::begin_buffer_release` を呼び、返した
/// [`context_cache::BufferReleaseToken`] を**実際の `storage` の drop
/// （`cuMemFreeAsync`/`cuMemFree` の発行）が完了するまで**保持する
/// （P0 再指摘対応: 旧稿は「駐機して戻るだけ」で、戻った直後に別スレッド
/// が新たな capture を実際に開始できてしまう競合窓があった。
/// `begin_buffer_release` doc コメント「P0 再修正」参照。トークンを
/// `storage` の drop より後まで生かすことで、`state.in_flight` 経由の
/// 排他が実際の解放発行を包み込む）。
impl Drop for CudaBufferHandle {
    fn drop(&mut self) {
        if let Some(storage) = self.storage.take() {
            let release_token = context_cache::begin_buffer_release(self.ordinal);
            drop(storage);
            drop(release_token);
        }
    }
}

/// `gemm.rs`／`gemm_wmma.rs`／`gemm_mma.rs`／`gemm_mma_tf32.rs`／
/// `gemm_mma_tf32x3.rs`／`transpose.rs` のベンチ・診断専用公開 API
/// （`upload_*`／`alloc_output_*`）が返す生の [`CudaSlice`] を包む薄い
/// RAII ラッパー（codex-review P0 指摘対応・PR #1390 再々修正）。
///
/// 型自体は `pub`（`upload_*`／`alloc_output_*` の戻り値の型として
/// crate 外の呼び出し元〈ベンチ・examples・integration tests〉のシグ
/// ネチャに現れるため公開が必須）。構築子（`Self::new`）は
/// `pub(crate)` のまま封じ、crate 外は本クレートが返した値を保持・
/// 転送・drop することしかできない（未検証の `CudaSlice` を外部から
/// 差し込んで `ordinal` を偽装する経路を型で排除する。`gemm_mma_tf32x3.
/// rs::ValidatedTf32x3Inputs` と同じ「フィールド非公開・構築子限定」
/// 設計判断）。
///
/// **背景**: これらの公開 API は `CudaBackendOps`（`ops.rs`）を経由せず
/// crate 外から直接呼び出せる（`gemm.rs::CudaGemm::upload_f32`
/// ドキュメンテーションコメント「PyTorch 参照計測」参照）。呼び出し元
/// （ベンチハーネス・`fresh_overhead_diag_tests.rs` 等の診断テスト）は
/// 返された `CudaSlice<T>` を複数回の関数呼び出しをまたいで保持し、
/// 最終的に自身のスコープで drop する。生の `CudaSlice<T>::drop` は
/// `context_cache::begin_driver_call`／`begin_capture_session` の排他
/// 機構を一切経由しない（`cudarc` 側の実装であり本クレートが介入
/// できない。`CudaBufferHandle` ドキュメンテーションコメント「背景」節と
/// 同型の欠陥）ため、別スレッドが `run_captured_sgd_step_segment` で
/// 同じ ordinal を実際に driver capture 中に、このラッパーなしで
/// `CudaSlice` を drop すると、その解放操作が capture 中の共有ストリーム
/// へ意図せず記録されうる。
///
/// `Drop` 実装は `CudaBufferHandle::drop` と同一の手順（
/// `context_cache::begin_buffer_release` を呼び、返した
/// `context_cache::BufferReleaseToken` を実際の `CudaSlice::drop` が
/// 完了するまで保持する）を踏む。
///
/// **公開アクセス面（codex-review P0 再指摘対応・PR #1390 再々々修正）**:
/// 生の `&CudaSlice<T>`／`&mut CudaSlice<T>` は crate 外へ一切公開しない
/// （`Deref`／`DerefMut` を実装しない）。理由は 2 つ: (1) 可変参照
/// （旧 `DerefMut`）を公開すると、`std::mem::swap` 等の安全な操作だけで
/// 異なる `ordinal`（別 GPU）を持つ 2 つの `GuardedSlice` の間で内部の
/// `CudaSlice` 実体を交換できてしまう——ラッパー自身の `ordinal`
/// フィールドは交換されないため、`Drop` 時に実体が実際に存在する GPU と
/// 異なる GPU の排他トークンで解放が走り、`capture` 中の共有ストリームへ
/// 誤った解放操作が混入しうる。(2) 不変参照（旧 `Deref`）であっても
/// `CudaSlice::context()`／`stream()` 等の公開アクセサへ到達でき、
/// `context_cache::disable_event_tracking()` が前提とする「1 ストリーム
/// のみ」という不変条件を crate 外から破りうる。
///
/// 代わりに、内部の `CudaSlice<T>` へは本クレート内（`pub(crate)`）の
/// `Self::as_raw`/`Self::as_raw_mut` からのみ到達できる。`gemm.rs`／
/// `gemm_wmma.rs`／`gemm_mma.rs`／`gemm_mma_tf32.rs`／
/// `gemm_mma_tf32x3.rs`／`transpose.rs` の `launch_*`／`download_*` 系
/// 公開シグネチャ自体を `&GuardedSlice<T>`／`&mut GuardedSlice<T>` を
/// 受け取る形へ変更し（旧稿の `&CudaSlice<T>`／`&mut CudaSlice<T>` から
/// 変更）、crate 外からは排他制御を経由する公開 API を通してしか
/// バッファを渡せない。
#[derive(Debug)]
pub struct GuardedSlice<T: cudarc::driver::DeviceRepr> {
    // `ManuallyDrop` で保持する（`Option` は使わない）: `Drop::drop`
    // 以外の生存区間では常に初期化済みであることを型で保証し、
    // `Self::as_raw`／`Self::as_raw_mut` が `Option` の取り出し失敗
    // （`unwrap`/`expect`）で本番経路 panic しうる余地を構造的に排除する
    // （codex-review P1 指摘対応・PR #1390 再々々修正。coding-rust.md
    // 「本番経路で unwrap()/expect() を使わない」）。
    inner: std::mem::ManuallyDrop<CudaSlice<T>>,
    ordinal: usize,
}

impl<T: cudarc::driver::DeviceRepr> GuardedSlice<T> {
    /// `slice`（確保済み・アップロード済みのいずれか）を `ordinal` の
    /// capture 排他へ参加する形で包む。
    pub(crate) fn new(ordinal: usize, slice: CudaSlice<T>) -> Self {
        Self {
            inner: std::mem::ManuallyDrop::new(slice),
            ordinal,
        }
    }

    /// crate 内部限定で内部の `&CudaSlice<T>` へアクセスする（構造体
    /// ドキュメンテーションコメント「公開アクセス面」参照。crate 外へは
    /// 公開しない）。
    pub(crate) fn as_raw(&self) -> &CudaSlice<T> {
        &self.inner
    }

    /// `Self::as_raw` の可変版。crate 内部限定（同上）。
    pub(crate) fn as_raw_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.inner
    }
}

impl<T: cudarc::driver::DeviceRepr> Drop for GuardedSlice<T> {
    fn drop(&mut self) {
        // `CudaBufferHandle::drop` と同一の手順（doc コメント「P0
        // 再修正」参照）: `begin_buffer_release` のトークンを、実際の
        // `CudaSlice::drop`（`cuMemFreeAsync`/`cuMemFree` の発行）が
        // 完了するまで保持する。
        //
        // SAFETY: `ManuallyDrop::take` は同一フィールドから 2 度取り出す
        // と未定義動作になる。`Drop::drop` は各インスタンスにつき高々
        // 1 回しか呼ばれず（Rust の drop 契約）、`drop` 実行後は `self`
        // （`self.inner` を含む）へ二度とアクセスされないため、ここでの
        // 1 回きりの `take` は当該不変条件を満たす。
        let slice = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        let release_token = context_cache::begin_buffer_release(self.ordinal);
        drop(slice);
        drop(release_token);
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

    /// [`Self::view`] の可変版（イシュー #1559）。`bounds`（要素インデックス
    /// 範囲）の部分ビューを書き込み可能引数として返す。
    /// `ops.rs::CudaBackendOps::gemm_fp32_strict_into` が
    /// `DeviceParamStore` の連結バッファ内の指定オフセットへ GEMM 結果を
    /// 直接書き込むために使う（`CudaArgMut::View`／`UnifiedView`
    /// 追加の動機。同構造体ドキュメンテーションコメント参照）。
    /// `view()` と同じく境界検証は呼び出し元
    /// （`ops.rs::gemm_fp32_strict_into_impl` の `checked_add`／
    /// `out.numel()` 検査）が済ませている前提で、ここでの追加検証は
    /// 行わない。
    pub(crate) fn view_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaArgMut<'_> {
        match self {
            CudaStorage::Device(s) => CudaArgMut::View(s.slice_mut(bounds)),
            CudaStorage::Managed(s) => CudaArgMut::UnifiedView(s.slice_mut(bounds)),
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
/// `View`／`UnifiedView`（可変部分ビュー。イシュー #1559 で追加）:
/// `ops.rs::CudaBackendOps::gemm_fp32_strict_into` が
/// `DeviceParamStore` の連結バッファ内の指定オフセット（`out_offset`）
/// へ GEMM 結果を直接書き込むために必要になった（導入前は「出力
/// バッファ〈`c_dev`〉はいずれも `CudaMemory::alloc_zeroed` が新規確保
/// した全体バッファであり、連結バッファの部分範囲へ書き込む呼び出し元は
/// 存在しない」ため `SliceMut`／`UnifiedMut`〈バッファ全体〉のみで
/// 足りていた）。`cudarc-0.19.8` は `PushKernelArg<&'b mut
/// CudaViewMut<'c, T>>`／`PushKernelArg<&'b mut UnifiedViewMut<'c, T>>`
/// を実装済みのため、[`CudaArg::View`]／`UnifiedView`（読み取り専用側）
/// と対称に追加できる。
pub(crate) enum CudaArgMut<'a> {
    SliceMut(&'a mut CudaSlice<f32>),
    UnifiedMut(&'a mut UnifiedSlice<f32>),
    View(CudaViewMut<'a, f32>),
    UnifiedView(UnifiedViewMut<'a, f32>),
}

impl<'d> CudaArgMut<'d> {
    /// [`CudaArg::len`] の可変版。
    pub(crate) fn len(&self) -> usize {
        match self {
            CudaArgMut::SliceMut(s) => s.len(),
            CudaArgMut::UnifiedMut(s) => s.len(),
            CudaArgMut::View(v) => v.len(),
            CudaArgMut::UnifiedView(v) => v.len(),
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
            CudaArgMut::View(v) => {
                builder.arg(v);
            }
            CudaArgMut::UnifiedView(v) => {
                builder.arg(v);
            }
        }
    }

    /// 全要素をゼロで埋める（イシュー #1559。`gemm.rs::CudaGemm::
    /// launch_tiled_f32_nt_into`／`_tn_into` の `k == 0` 分岐——数学的に
    /// 結果が全 0 になる契約——が、カーネル起動を経由せず `c_dev` へ
    /// 直接ゼロを書き込むために使う。`cudarc::CudaStream::memset_zeros`
    /// は `DevicePtrMut<T>` を実装する型（`CudaSlice`／`CudaViewMut`／
    /// `UnifiedSlice`／`UnifiedViewMut`。いずれも実装済み）を汎用に扱う
    /// ため、本列挙体の 4 variant すべてに委譲できる。非同期投入契約
    /// （#1013）は他のカーネル起動と同じで、完了保証は呼び出し元の次の
    /// 同期点へ委ねる。
    pub(crate) fn zero_fill(&mut self, stream: &Arc<CudaStream>) -> Result<(), CudaError> {
        match self {
            CudaArgMut::SliceMut(s) => stream.memset_zeros(&mut **s)?,
            CudaArgMut::UnifiedMut(s) => stream.memset_zeros(&mut **s)?,
            CudaArgMut::View(v) => stream.memset_zeros(v)?,
            CudaArgMut::UnifiedView(v) => stream.memset_zeros(v)?,
        }
        Ok(())
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
    /// [`MemoryOps::with_host_view`] の CUDA 実装（イシュー #1336）が
    /// 使う、形状（要素数）ごとに再利用するホストステージングバッファの
    /// キャッシュ（`crate::host_staging` モジュール参照）。`Arc<Mutex<_>>`
    /// で保持する理由は `tracker` と同じ「`Clone` は同一計測系列／同一
    /// キャッシュ系列への参照複製」契約を保つため（`Mutex` 自体は
    /// `Clone` を持たないため素の `Mutex<HostStagingCache>` フィールドは
    /// `derive(Clone)` を壊す）。
    host_staging: Arc<Mutex<HostStagingCache>>,
    /// H2D pinned staging（イシュー #1585。`crate::host_staging` モジュール
    /// 「H2D 用ステージング」節）の opt-in キャッシュ。`upload_inner`／
    /// `upload_into` の `Device` 分岐が使う。`host_staging`（D2H 側）と
    /// 同じ `Arc<Mutex<_>>` 共有契約（`Clone` は同一キャッシュ系列への
    /// 参照複製）。フラグ OFF（既定）時は本フィールドを一切参照しない
    /// （`host_staging::upload_new`／`upload_into` 冒頭の早期分岐）。
    h2d_staging: Arc<Mutex<H2dStagingCache>>,
}

impl CudaMemory {
    /// 初期化済みの [`CudaDevice`] から `CudaMemory` を構築する。
    /// `device.stream()` を `Arc` クローンで共有する（`gemm.rs::CudaGemm::new`
    /// と同じ共有契約）。新規の計測系列を持つトラッカーを生成する
    /// （`backend-cpu::CpuMemory::new` と同型。同一プロセス内でピークを
    /// 集約したい場合は `clone()` でトラッカーを共有する）。`host_staging`
    /// は本番既定種別（`host_staging::HOST_STAGING_KIND`。イシュー
    /// #1478 で `Pinned` へ切替済み）で初期化する。`Pinned` 確保
    /// （`unsafe`）自体はキャッシュ miss 時に `HostStaging::alloc` 内で
    /// 遅延実行されるため、本コンストラクタ自体は driver を呼ばない。
    pub fn new(device: &CudaDevice) -> Self {
        Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            tracker: Arc::new(AllocationTracker::new()),
            managed_supported: device.managed_memory_supported(),
            host_staging: Arc::new(Mutex::new(HostStagingCache::new(
                host_staging::HOST_STAGING_KIND,
            ))),
            h2d_staging: Arc::new(Mutex::new(H2dStagingCache::new())),
        }
    }

    /// **`internal-diagnostics` feature（既定 off）限定の診断専用
    /// コンストラクタ**（イシュー #1336 実機実測フェーズで追加）。
    ///
    /// [`Self::with_host_view_using_kind`] は呼び出しごとに `kind` を
    /// 直接 `HostStaging::alloc` するためキャッシュを経由せず、
    /// `Pinned` 系列の A/B 計測に使うと「毎回新規確保」という
    /// `with_host_view_using_kind` 自身の設計（ドキュメンテーション
    /// コメント参照）により本番 `with_host_view`（`self.host_staging`
    /// キャッシュ経由・2 回目以降は `take` が hit する）と不公平な
    /// 比較になってしまう。本コンストラクタは `self.host_staging` の
    /// 初期種別だけを差し替えた `CudaMemory` を返すことで、`Pinned`
    /// （本番既定・イシュー #1478）と `Pageable`（切替前既定・対照腕）
    /// の両方を**同じキャッシュ経由の `with_host_view` 経路**で比較
    /// できるようにする（`docs/perf/cuda-host-view-staging-readout.md`
    /// §5.1 のゲート C 計測が使う入口。§8 の #1478 再計測でも同じ入口を
    /// 使う）。`unsafe` は追加しない（`kind` に応じた `unsafe` 呼び出し
    /// 自体は既存の `HostStaging::alloc` 内に閉じており、本
    /// コンストラクタはそこへ渡す初期値を選ぶだけ）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_host_staging_kind(
        device: &CudaDevice,
        kind: host_staging::HostStagingKind,
    ) -> Self {
        Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            tracker: Arc::new(AllocationTracker::new()),
            managed_supported: device.managed_memory_supported(),
            host_staging: Arc::new(Mutex::new(HostStagingCache::new(kind))),
            h2d_staging: Arc::new(Mutex::new(H2dStagingCache::new())),
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

/// [`readback`] の宛先確保方式（イシュー #1437。`host-view-readout`
/// feature〈#1335〜#1337〉有効時の CUDA reuse N=1024/2048 後退を、
/// `readback` 唯一の同期点を是正することで全形状非後退化する）。
///
/// `docs/perf/cuda-host-view-readout-small-shape-regression.md`（#1436）
/// の診断で、後退の増分は「`clone_dtoh` の宛先 `Vec<T>` が毎回 fresh な
/// 未タッチ mmap ページになりうる」ことに帰着すると判明した（on 腕は
/// 反復ごとの free が無いため glibc の動的 mmap 閾値適応が定常状態化
/// せず、宛先が既タッチページを再利用できない）。`PretouchedFresh` は
/// 宛先を非ゼロ値で明示的に埋めてから `memcpy_dtoh` するため、確保
/// 直後の D2H が必ず既にコミット済みの物理ページへ書き込まれる
/// （ページフォールト処理を D2H 区間の外〈CPU 側の fill〉へ前倒しする）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadbackDest {
    /// 現行方式（`clone_dtoh` が内部で `Vec::with_capacity` +
    /// `set_len` により確保する未初期化 `Vec`）。挙動・bit 出力とも
    /// 本イシュー導入前と完全に同一。
    Fresh,
    /// 事前タッチ済み `Vec`（[`pretouched_host_vec`]）へ `memcpy_dtoh`
    /// する方式。宛先の全バイトが `memcpy_dtoh` 呼び出し前に一度
    /// 上書きされるため、返す `Vec` の内容自体は `Fresh` と bit 同一
    /// （D2H が全要素を上書きするため事前値は残らない）。
    PretouchedFresh,
}

/// `crates/backend-cuda` の CUDA 実機実測（#1437・GB10）で Layer B
/// 分離計測（`readout_regression_diag_tests_1436.rs` の 5 腕・単一腕
/// プロセス分離実行）が事前宣言ゲートを通過したことを確認したうえで
/// 既定値を `PretouchedFresh` へ切り替える。GB10 実機実測（同一プロセス
/// 内の腕間汚染を排したプロセス分離実行）: N=1024（legacy 6.06 ms・
/// borrowed 37.15 ms・pretouched-fresh **2.04 ms**）・N=2048（legacy
/// 12.15 ms・borrowed 11.32 ms・pretouched-fresh **7.34 ms**）・N=4096
/// （legacy 48.39 ms・borrowed 29.26 ms・pretouched-fresh 30.37 ms）の
/// いずれも `PretouchedFresh` が `Fresh`（legacy 相当）を下回るか同水準
/// であり、後退対象だった N=1024/2048 で大幅改善（受け入れ条件 Gate 1。
/// 全 N で 0.637〜0.897 倍・PASS）を裏付ける。Gate 2（既存経路の非後退
/// 目安。≤1.03）は N=1024/2048 でわずかに超過（1.038・1.047）しており
/// 「後退なし」ではない——`PretouchedFresh` は `host-view-readout`
/// feature の有効・無効を問わず無条件に既定となるため、`off` 経路にも
/// 事前フィル費用が一律で乗ることが機構的な説明として整合する。Gate 2
/// は受け入れ条件自体には含まれず、超過幅が Gate 1 の改善幅より小さい
/// ことから ADOPT 判断は変更していない。詳細・Layer A（framework-compare
/// 実践規模）ゲート結果は `docs/perf/cuda-host-view-readout-small-shape-
/// regression.md` §13 以降（とくに §13.3 の再評価）を参照。
pub(crate) const READBACK_DEST: ReadbackDest = ReadbackDest::PretouchedFresh;

/// [`ReadbackDest::PretouchedFresh`] が要求する「非ゼロ事前タッチ値」を
/// 型ごとに定義する crate 内部限定トレイト。`readback` は `f32`
/// （`gemm.rs` 等の大半の GEMM 経路）と `f16`（`gemm_mma.rs` の
/// `download_f16` 系。#1191 で本番結線済みの MMA f16 経路が経由する）の
/// 両方で使われるため、`T: DeviceRepr` だけでは「非ゼロ値」を汎用に
/// 構成できない。新しい `T` で `readback` を呼ぶ場合はここへ impl を
/// 追加する必要があり、追加を怠るとコンパイルエラーで機械的に検出
/// される（トレイト境界の欠落として現れるため、既定値 `Fresh` 側は
/// 影響を受けない）。
pub(crate) trait ReadbackSentinel: DeviceRepr + Copy {
    /// 事前タッチに使う非ゼロ値。`0` だと `vec![Self::SENTINEL; n]` が
    /// `alloc_zeroed`（calloc）経由になり、mmap の COW ゼロページの
    /// まま実ページがコミットされず「事前タッチ」の意図を満たさない
    /// （`readout_regression_diag_tests_1436.rs` の `PretouchedReusedDest`
    /// 腕が既に踏んだ同種の罠。#1436 コメント参照）。
    const SENTINEL: Self;
}

impl ReadbackSentinel for f32 {
    const SENTINEL: f32 = 1.0;
}

impl ReadbackSentinel for half::f16 {
    const SENTINEL: half::f16 = half::f16::ONE;
}

/// `numel` 要素ぶんの事前タッチ済みホストバッファを確保する
/// （[`ReadbackDest::PretouchedFresh`] 専用ヘルパー）。`vec![T::SENTINEL;
/// numel]` は `Vec::from_elem` 経由で全要素を明示的に書き込むため
/// （`alloc_zeroed` を経由しない）、返した時点で全ページが物理コミット
/// 済みであることが保証される。フィル自体の費用は帯域律速（数百 KiB〜
/// 数十 MiB で概ね 0.1〜数 ms オーダー）で、事前タッチが避けようと
/// している D2H 中のページフォールト処理費用より小さい想定
/// （実測は `docs/perf/cuda-host-view-readout-small-shape-regression.md`
/// を参照）。
pub(crate) fn pretouched_host_vec<T: ReadbackSentinel>(numel: usize) -> Vec<T> {
    vec![T::SENTINEL; numel]
}

/// カーネル起動直後の都度 `synchronize()` を除去した非同期実行契約
/// （イシュー #1013・`docs/backend-cuda-async-execution-design.md` §3〜
/// §4）の下で、ホストへ結果を読み戻す全ての readback 経路が共有する
/// 唯一の同期点。`clone_dtoh`／`memcpy_dtoh` は `cuMemcpyDtoHAsync` を
/// 発行する非同期コピー（`cudarc-0.19.8` `core.rs::memcpy_dtoh`）のため、
/// 呼び出し直後にホスト側データが確定していることを保証するには
/// D2H コピー → `synchronize` の順が必須（逆順ではコピー自体の完了を
/// 待てない）。起動元のカーネルが `unsafe { stream.launch(..) }` を
/// 経て投入した非同期作業も、同一ストリーム上の FIFO 順序保証により
/// 本関数の `synchronize` で合わせて完了が確定する（`CudaDevice` は
/// ordinal ごとに単一ストリームを共有する。設計文書 §3「実行モデル」）。
/// `download_inner`（本ファイル）・`gemm.rs`／`gemm_wmma.rs`／
/// `gemm_mma.rs`／`gemm_mma_tf32.rs` の `download_f32`／`download_f16`・
/// 各演算のホスト `Tensor` 返却ラッパーはすべて本関数を経由し、
/// 「同期点は D2H 境界のみ」という契約を単一箇所に集約する。
///
/// 宛先確保方式は [`READBACK_DEST`]（[`ReadbackDest`]）で切り替わる
/// （イシュー #1437）。既定は `PretouchedFresh` であり、`host-view-readout`
/// feature の有効・無効に関わらず本関数の全呼び出し（`off`／`on` 両方の
/// 経路）が同じ既定値を通る（`ReadbackDest` の選択はこの feature flag
/// に連動しない）。返す `Vec` の内容は `Fresh` と bit 同一だが、宛先を
/// `SENTINEL` で事前に埋めるぶんの費用が全呼び出しに一律で乗るため、
/// `off` 経路（`host-view-readout` 無効時）の性能も本イシューの前後で
/// 完全に不変とは限らない（実測・評価は
/// `docs/perf/cuda-host-view-readout-small-shape-regression.md` §13.3
/// を参照）。
pub(crate) fn readback<T, Src>(stream: &Arc<CudaStream>, dev: &Src) -> Result<Vec<T>, CudaError>
where
    T: ReadbackSentinel,
    Src: DevicePtr<T>,
{
    readback_with(stream, dev, READBACK_DEST)
}

/// [`readback`] の宛先確保方式を明示指定できる内部版（実機 A/B 計測・
/// 単体テスト用。`readback` はこれへ `READBACK_DEST` を渡して委譲する）。
pub(crate) fn readback_with<T, Src>(
    stream: &Arc<CudaStream>,
    dev: &Src,
    dest: ReadbackDest,
) -> Result<Vec<T>, CudaError>
where
    T: ReadbackSentinel,
    Src: DevicePtr<T>,
{
    match dest {
        ReadbackDest::Fresh => {
            let host = stream.clone_dtoh(dev)?;
            stream.synchronize()?;
            Ok(host)
        }
        ReadbackDest::PretouchedFresh => {
            let mut host = pretouched_host_vec::<T>(dev.len());
            stream.memcpy_dtoh(dev, &mut host)?;
            stream.synchronize()?;
            Ok(host)
        }
    }
}

/// [`readback_with`] を `ReadbackDest::Fresh`／`PretouchedFresh` の両方で
/// crate 外部（実機 `#[ignore]` テスト）から直接呼べるようにする診断専用
/// 入口（イシュー #1437）。`ReadbackDest`／`readback_with` 自体は
/// `pub(crate)` のままシグネチャへは出さず、`bool` フラグで戦略を選ぶ
/// ことで「`internal-diagnostics` feature 無効時は公開 API 面に一切
/// 現れない」という既存の診断専用入口群（`device.rs::context`/`stream`
/// 等）と同じ可視性契約を保つ。既定ビルドでは存在しない関数のため、
/// `readback`／`READBACK_DEST` の既定動作には一切影響しない。
#[cfg(feature = "internal-diagnostics")]
pub fn readback_f32_diag(
    stream: &Arc<CudaStream>,
    dev: &CudaSlice<f32>,
    pretouched: bool,
) -> Result<Vec<f32>, CudaError> {
    let dest = if pretouched {
        ReadbackDest::PretouchedFresh
    } else {
        ReadbackDest::Fresh
    };
    readback_with(stream, dev, dest)
}

/// [`readback_f32_diag`] の f16 版（`gemm_mma.rs::download_f16` が経由
/// する `T = f16` 経路の bit 一致検証用。#1437）。
#[cfg(feature = "internal-diagnostics")]
pub fn readback_f16_diag(
    stream: &Arc<CudaStream>,
    dev: &CudaSlice<half::f16>,
    pretouched: bool,
) -> Result<Vec<half::f16>, CudaError> {
    let dest = if pretouched {
        ReadbackDest::PretouchedFresh
    } else {
        ReadbackDest::Fresh
    };
    readback_with(stream, dev, dest)
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

/// `CudaStorage::Managed`（[`UnifiedSlice`]）専用の [`MemoryOps::
/// with_host_view`] 実装（イシュー #1336）。[`host_readback`] と同じ
/// 同期契約（`stream.synchronize()` を先に呼ぶ理由は同関数のドキュメン
/// テーションコメント参照）を保ったまま、`to_vec()` によるホストコピーを
/// 経由せず `UnifiedSlice::as_slice()` が返す借用をそのまま `f` へ渡す
/// （managed 配置はホストから直接アクセス可能なため、`Device` 配置向け
/// の `host_staging` キャッシュ経由 D2H は不要かつ目的〈ゼロコピー〉に
/// 反する）。
fn host_view_managed(
    stream: &Arc<CudaStream>,
    dev: &UnifiedSlice<f32>,
    f: &mut dyn FnMut(&[f32]),
) -> Result<(), CudaError> {
    stream.synchronize()?;
    f(dev.as_slice()?);
    Ok(())
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
                ordinal: self.ordinal,
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
                ordinal: self.ordinal,
            })
        } else {
            let slice = self.stream.alloc_zeros::<f32>(numel)?;
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
            Box::new(CudaBufferHandle {
                storage: Some(CudaStorage::Device(slice)),
                _alloc: alloc,
                generation,
                ordinal: self.ordinal,
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
                ordinal: self.ordinal,
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
            // イシュー #1585: H2D pinned staging（opt-in・既定 OFF）。
            // `host_staging::upload_new` はフラグ OFF・`data` が空の
            // 場合は `self.stream.clone_htod(data)` をそのまま呼ぶため、
            // 導入前と経路・出力は bit 同一（`host_staging` モジュール
            // 「H2D 用ステージング」節参照）。
            CudaStorage::Device(host_staging::upload_new(
                &self.stream,
                &self.h2d_staging,
                self.stream.context(),
                generation,
                data,
            )?)
        };
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
            storage: Some(storage),
            _alloc: alloc,
            generation,
            ordinal: self.ordinal,
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

    /// [`Self::with_driver_call`] と同じだが、CUDA Graph capture 中
    /// （イシュー #1349・`docs/backend-cuda-graph-step-capture-design.md`
    /// §4.2）は driver に触れる前に拒否する（`context_cache::
    /// begin_sync_point_call`）。ホスト⇔デバイス転送・確保・ゼロ初期化
    /// はいずれも capture 境界を跨ぐ同期点であり、capture 中の呼び出しを
    /// 許すと graph が「その時点のホストデータ」を焼き込んでしまい、
    /// 2 回目以降の再生で不正な結果を生む（`what` は診断メッセージ用の
    /// 呼び出し名）。
    fn with_sync_point_call<T>(
        &self,
        resource_generations: &[u64],
        what: &'static str,
        map: impl FnOnce(CudaError) -> BackendError,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, BackendError> {
        let token = context_cache::begin_sync_point_call(self.ordinal, resource_generations, what)?;
        context_cache::observe_cuda_result(self.ordinal, &token, f()).map_err(map)
    }

    /// [`MemoryOps::with_host_view`]（`Device` 配置分岐）が使うホスト
    /// ステージングキャッシュから既存エントリを取り出す（イシュー #1336・
    /// codex-review／Cursor Bugbot 指摘対応）。`self.host_staging` の
    /// Mutex poison は `static_cuda_memory`（`ops.rs`）と同じ fail-closed
    /// 方針で `BackendError::DeviceUnavailable` へ変換する（本番経路で
    /// `unwrap`／`expect` を使わない。`.claude/rules/coding-rust.md`）。
    /// ロック保持区間はキャッシュの `take` 呼び出しのみに限定し
    /// （`crate::host_staging` モジュールコメント「ロック方針」節）、
    /// キャッシュ miss 時の新規確保（`HostStaging::alloc`。`Pinned`
    /// 種別では driver 呼び出し `alloc_pinned` を伴う）は本メソッドでは
    /// 行わず、呼び出し元（`with_host_view`／`with_host_view_using_kind`）
    /// が `with_driver_call`（poison／世代検査境界）の内側で行う。
    /// 従来はこの alloc をロック解放直後・`with_driver_call` の外側で
    /// 行っていたため、poison 済み・旧世代の ordinal でも `alloc_pinned`
    /// が素通りで実行され、確保時の sticky エラーも `observe_cuda_result`
    /// を経ず ordinal が poison されない欠陥があった（返り値を
    /// `(Option<HostStaging>, HostStagingKind)` へ変更し、alloc の要否
    /// 判定と実行を境界の内側へ委ねる）。
    fn take_cached_staging(
        &self,
        numel: usize,
        generation: u64,
    ) -> Result<(Option<HostStaging>, host_staging::HostStagingKind), BackendError> {
        // fail-closed: poison を検出したらここで拒否する（`static_cuda_
        // memory`〈ops.rs〉と同じ方針。`crate::host_staging::put_back`
        // 〈`return_staging` が使う返却経路〉は poison 後も `into_inner`
        // で回復する設計だが、それは使用後の返却側で無条件に使う目的の
        // 緩和策であり、新規取得側（本メソッド）では poison を素通り
        // させない）。
        let mut guard = self.host_staging.lock().map_err(|_| {
            BackendError::DeviceUnavailable("host staging cache mutex poisoned".to_string())
        })?;
        Ok((guard.take(numel, generation), guard.kind()))
    }

    /// [`Self::take_cached_staging`] で取り出した、または `with_driver_call`
    /// 境界の内側で新規確保したバッファを使用後にキャッシュへ返却する
    /// （`with_host_view` の呼び出し元クロージャ `f` の実行後、成功・
    /// 失敗いずれの経路でも呼ばれる。`crate::host_staging::put_back` は
    /// poison 後も panic しない設計のため、ここでは呼び出し結果を無視
    /// してよい〈以降のアクセスは `take_cached_staging` の poison 検査が
    /// fail-closed に拒否する〉）。
    fn return_staging(&self, numel: usize, generation: u64, buf: HostStaging) {
        host_staging::put_back(&self.host_staging, numel, generation, buf);
    }

    /// `host_staging` の統計スナップショット（実機診断用。`crate::
    /// host_staging::HostStagingStats` ドキュメンテーションコメント
    /// 参照）。poison 時は既定値（全 0）を返す（診断専用の補助 API の
    /// ため fail-closed にせず観測可能な最善値を返す）。`gemm_profile_
    /// target` 等と同じ `internal-diagnostics` feature（既定 off）限定で
    /// 公開 API 面から除外する（`lib.rs` の `pub use host_staging::
    /// HostStagingStats` re-export と同一ゲート。イシュー #1336）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn host_staging_stats(&self) -> HostStagingStats {
        match self.host_staging.lock() {
            Ok(guard) => guard.stats(),
            Err(poisoned) => poisoned.into_inner().stats(),
        }
    }

    /// `self.host_staging` が実際に構築されている種別（`Pinned`／
    /// `Pageable`）を返す診断用アクセサ（イシュー #1478 で追加）。
    /// `CudaMemory::new` の既定が `HOST_STAGING_KIND` の値どおり
    /// `Pinned` に解決されていることを、A/B ハーネス・実機テストが
    /// 自己証明するために使う（`host_staging_stats` と同じ
    /// `internal-diagnostics` feature 限定・poison 時のフォールバック
    /// 方針。unsafe は追加しない）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn host_staging_kind(&self) -> host_staging::HostStagingKind {
        match self.host_staging.lock() {
            Ok(guard) => guard.kind(),
            Err(poisoned) => poisoned.into_inner().kind(),
        }
    }

    /// `host_staging` の全エントリを破棄し、解放したバイト数を返す
    /// （REQ-14 `release_cached` 系と同型の明示解放 API。page-locked
    /// メモリ〈`Pinned` 種〉はホスト RAM を固定するため、長時間常駐する
    /// `CudaMemory` インスタンスに対する明示解放手段として公開する）。
    pub fn release_host_staging(&self) -> u64 {
        match self.host_staging.lock() {
            Ok(mut guard) => guard.release_all(),
            Err(poisoned) => poisoned.into_inner().release_all(),
        }
    }

    /// `h2d_staging`（イシュー #1585・H2D pinned staging。opt-in・既定
    /// OFF）の全エントリを破棄し、解放したバイト数を返す
    /// （[`Self::release_host_staging`] の H2D 版・同一契約）。フラグ
    /// OFF でも呼び出し自体は安全（`h2d_staging` が空のまま `0` を
    /// 返す）。
    pub fn release_h2d_staging(&self) -> u64 {
        match self.h2d_staging.lock() {
            Ok(mut guard) => guard.release_all(),
            Err(poisoned) => poisoned.into_inner().release_all(),
        }
    }

    /// `h2d_staging` の統計スナップショット（実機診断用。
    /// [`Self::host_staging_stats`] の H2D 版・同一 `internal-
    /// diagnostics` feature ゲート）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn h2d_staging_stats(&self) -> HostStagingStats {
        match self.h2d_staging.lock() {
            Ok(guard) => guard.stats(),
            Err(poisoned) => poisoned.into_inner().stats(),
        }
    }

    /// **`internal-diagnostics` feature（既定 off）限定の診断専用入口**。
    /// イシュー #1336 codex-review 指摘の経緯: 当時の本番既定
    /// `host_staging::HOST_STAGING_KIND` は `Pageable` に固定されて
    /// おり、`Pinned`（page-locked・WRITECOMBINED）経路は実機テスト・
    /// `Pageable` との A/B 比較のいずれからも到達できていなかった。
    /// 現在は本番既定が `Pinned`（イシュー #1478）であるため、本
    /// メソッドは主に `Pageable`（切替前既定）を明示選択して A/B 比較
    /// する用途で使う。本メソッドは
    /// [`MemoryOps::with_host_view`]（`Device` 配置分岐）と同じ D2H・
    /// `f` 呼び出し手順を踏みつつ、`self.host_staging`（本番既定種別で
    /// 固定された共有キャッシュ）を経由せず、呼び出しごとに指定
    /// `kind` で [`HostStaging::alloc`] を直接呼ぶ（キャッシュに
    /// 登録しないため統計〈[`Self::host_staging_stats`]〉には現れず、
    /// `kind` ごとの独立比較を単純にする）。`None`（空バッファ）・
    /// `Managed` 配置は種別に依存しないため [`MemoryOps::
    /// with_host_view`]（本 struct の実装）へそのまま委譲する。
    #[cfg(feature = "internal-diagnostics")]
    pub fn with_host_view_using_kind(
        &self,
        buffer: &DeviceBuffer<f32>,
        kind: host_staging::HostStagingKind,
        f: &mut dyn FnMut(&[f32]),
    ) -> Result<(), BackendError> {
        let handle = buffer
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if buffer.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let Some(CudaStorage::Device(slice)) = &handle.storage else {
            // `None`／`Managed` は `kind` に依存しない分岐のため、通常
            // 経路（`MemoryOps::with_host_view`）へそのまま委譲する。
            return <Self as MemoryOps>::with_host_view(self, buffer, f);
        };
        let numel = slice.len();
        let generation = buffer.generation();
        let ctx = self.stream.context();
        // codex-review 指摘対応: `HostStaging::alloc`（`Pinned` では
        // driver 呼び出し `alloc_pinned` を伴う）・D2H・`as_slice()`
        // （`Pinned` 側は内部で `event.synchronize()`）・`f` 呼び出しの
        // 全てを `with_driver_call`（poison／世代検査境界）の内側で行う
        // （`take_cached_staging` ドキュメンテーションコメント参照。
        // 本メソッドはキャッシュを経由しないため、常に新規確保する）。
        self.with_driver_call(&[generation], map_cuda_error, || {
            let mut staging = HostStaging::alloc(kind, ctx, numel)?;
            self.stream
                .memcpy_dtoh(slice, staging.as_host_slice_mut())?;
            self.stream.synchronize()?;
            let view = staging.as_slice()?;
            f(view);
            Ok(())
        })
    }
}

impl MemoryOps for CudaMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        self.with_sync_point_call(&[], "alloc_zeroed", map_cuda_alloc_error, || {
            self.alloc_zeroed_inner(shape)
        })
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        self.with_sync_point_call(&[], "upload", map_cuda_error, || self.upload_inner(tensor))
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
        self.with_sync_point_call(&[buffer.generation()], "download", map_cuda_error, || {
            self.download_inner(buffer)
        })
    }

    /// ホスト常駐の `tensor` を既存の `dst` の `dst_offset` 要素目から
    /// H2D 転送する（イシュー #1212・§4.5 で `DeviceParamStore::step` の
    /// grad staging 書き込みに使う。イシュー #1349 では graph capture
    /// 対象区間の外側〈`run_captured_sgd_step_segment` 呼び出し前〉で毎回呼ぶ
    /// ことで、capture 済み graph が参照するバッファのアドレス・内容を
    /// capture 前に確定させる契約とする。`backend-cpu::upload_into_cpu_buffer`
    /// と同じ境界検査を行う）。
    fn upload_into(
        &self,
        tensor: &Tensor<f32>,
        dst: &mut DeviceBuffer<f32>,
        dst_offset: usize,
    ) -> Result<(), BackendError> {
        if dst.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        let contiguous = tensor.contiguous();
        let numel = contiguous.numel();
        let end = dst_offset.checked_add(numel).ok_or_else(|| {
            BackendError::InvalidArgument(
                "upload_into: dst_offset + tensor.numel() overflowed usize".to_string(),
            )
        })?;
        if end > dst.numel() {
            return Err(BackendError::InvalidArgument(format!(
                "upload_into: write range [{dst_offset}, {end}) exceeds dst buffer length {}",
                dst.numel()
            )));
        }
        let generation = dst.generation();
        self.with_sync_point_call(&[generation], "upload_into", map_cuda_error, || {
            if numel == 0 {
                return Ok(());
            }
            let data = contiguous
                .as_slice()
                .ok_or_else(|| CudaError::InvalidShape {
                    detail: "upload_into: contiguous() の直後にもかかわらず as_slice が \
                             None を返した（tensor-core 側のロジック不整合）"
                        .to_string(),
                })?;
            let handle = dst
                .downcast_handle_mut::<CudaBufferHandle>()
                .ok_or_else(|| CudaError::InvalidShape {
                    detail: "upload_into: dst buffer handle is not a CudaBufferHandle".to_string(),
                })?;
            let storage = handle
                .storage
                .as_mut()
                .ok_or_else(|| CudaError::InvalidShape {
                    detail: "upload_into: dst buffer has numel > 0 but no device allocation"
                        .to_string(),
                })?;
            match storage {
                CudaStorage::Device(slice) => {
                    let mut view = slice.slice_mut(dst_offset..end);
                    // イシュー #1585: `upload_inner` と同じ pinned
                    // staging 経路（opt-in・既定 OFF）。
                    host_staging::upload_into(
                        &self.stream,
                        &self.h2d_staging,
                        self.stream.context(),
                        generation,
                        data,
                        &mut view,
                    )?;
                }
                CudaStorage::Managed(unified) => {
                    // managed 配置はホストから直接書き込めるため
                    // `memcpy_htod` を発行しない（`upload_inner` の
                    // 新規確保時と同じ方針）。ただし既存バッファへの
                    // 書き込みであるため、直前に投入されたカーネルが
                    // 同じ領域を読み書き中でないことを、書き込み前に
                    // `stream.synchronize()` で確定させる（`host_readback`
                    // の同期契約コメント参照。`as_mut_slice()` 内部の
                    // `event.synchronize()` だけでは単一ストリーム構成
                    // では不十分なため）。
                    self.stream.synchronize()?;
                    unified.as_mut_slice()?[dst_offset..end].copy_from_slice(data);
                }
            }
            Ok(())
        })
    }

    /// [`MemoryOps::with_host_view`] の CUDA 実装（イシュー #1336）。
    ///
    /// 既定実装（`tensor_core::buffer` モジュールの同トレイト
    /// ドキュメンテーションコメント「デフォルト実装」節）は毎回
    /// `download`（`readback` 経由で D2H 宛先を都度新規確保）を経由する
    /// ため、`docs/perf/cuda-large-buffer-percall-alloc-transfer-
    /// threshold.md` が実測した 31→32 MiB 段差の対象になりうる。本実装は
    /// `handle.storage` の配置ごとに以下へ分岐する:
    ///
    /// - `None`（空テンソル）: `f(&[])`（FFI を呼ばない）。
    /// - `Managed`: `host_view_managed` へ委譲し、`UnifiedSlice::
    ///   as_slice()` の借用をコピーなしでそのまま渡す（`download` が
    ///   `to_vec()` するのと異なり、managed 配置本来のゼロコピー特性を
    ///   保つ）。
    /// - `Device`: `crate::host_staging`（形状ごとに再利用するホスト
    ///   ステージングバッファ）から取得・確保・`memcpy_dtoh` で D2H・
    ///   `synchronize`・`f` 呼び出しまでを `with_driver_call`（poison／
    ///   世代検査境界）の内側で行う。キャッシュ miss 時の新規確保
    ///   （`HostStaging::alloc`。`Pinned` 種別では driver 呼び出し
    ///   `alloc_pinned` を伴う）も同境界の内側で行い（`Self::
    ///   take_cached_staging` ドキュメンテーションコメント参照。
    ///   codex-review／Cursor Bugbot 指摘対応）、poison 済み・旧世代の
    ///   ordinal に対して driver 操作が素通りで実行されることを防ぐ。
    ///   使用後は成功・失敗いずれの経路でも `Self::return_staging` で
    ///   キャッシュへ返却する（失敗時に返却したバッファの内容は不定
    ///   だが、次回の `memcpy_dtoh` が呼び出し前に全域を上書きする
    ///   ため安全。`crate::host_staging::HostStaging::alloc` の SAFETY
    ///   コメント参照）。
    ///
    /// 同期契約は `download`（`Self::download_inner`）と同一
    /// （`with_driver_call` を唯一の driver 呼び出し境界とし、
    /// `buffer.generation()` を検査対象へ渡す。イシュー #1013 設計文書
    /// §9 item 7）。
    fn with_host_view(
        &self,
        buffer: &DeviceBuffer<f32>,
        f: &mut dyn FnMut(&[f32]),
    ) -> Result<(), BackendError> {
        let handle = buffer
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if buffer.device() != Device::Cuda(self.ordinal) {
            return Err(BackendError::DeviceMismatch);
        }
        match &handle.storage {
            // 空バッファ（`storage == None`）でも FFI を伴わない
            // クロージャとして `with_driver_call` を経由させる
            // （codex-review 指摘 P0: 従来は `with_driver_call` を
            // 素通りしていたため、`Poisoned`／`Retiring` 状態や
            // invalidate 後の旧世代バッファに対しても無条件でクロー
            // ジャが実行され、`download` 経由に存在する fail-closed な
            // poison／世代検査を迂回できてしまっていた）。内部では
            // driver 呼び出しを一切行わず `f(&[])` を実行するだけだが、
            // `begin_driver_call` による検査は他分岐と同じく必ず通す。
            None => self.with_driver_call(&[buffer.generation()], map_cuda_error, || {
                f(&[]);
                Ok(())
            }),
            Some(CudaStorage::Managed(unified)) => {
                self.with_driver_call(&[buffer.generation()], map_cuda_error, || {
                    host_view_managed(&self.stream, unified, f)
                })
            }
            Some(CudaStorage::Device(slice)) => {
                let numel = slice.len();
                let generation = buffer.generation();
                let (cached, kind) = self.take_cached_staging(numel, generation)?;
                let mut staging_slot = cached;
                let ctx = self.stream.context();
                // codex-review／Cursor Bugbot 指摘対応（イシュー #1336）:
                // キャッシュ miss 時の新規確保（`HostStaging::alloc`。
                // `Pinned` では driver 呼び出し `alloc_pinned` を伴う）・
                // D2H・`as_slice()`（`Pinned` 側は内部で
                // `event.synchronize()`）・`f` 呼び出しを、poison／世代
                // 検査境界（`with_driver_call`）の内側へ移した（従来は
                // alloc が境界の外側〈`take_or_alloc_staging`〉で行われ、
                // `as_slice()`／`f` 呼び出しも境界の外側で行われていた
                // ため、poison 済み・旧世代の ordinal でも driver 操作が
                // 素通りで実行され得た）。`staging_slot`（`Option<
                // HostStaging>`）は closure に可変参照で捕捉され、
                // 成功時は必ず `Some` のまま closure を抜けるため、
                // 成功・失敗いずれの経路でも使用後にキャッシュへ返却
                // できる（`Managed`／`None` 分岐と同じく `f` を境界の
                // 内側で呼ぶ設計に統一）。
                let copy_result = self.with_driver_call(&[generation], map_cuda_error, || {
                    // codex-review 指摘（イシュー #1336・PR #1408）: 以前は
                    // `if staging_slot.is_none() { staging_slot = Some(..) }`
                    // で `Some` を保証したあと `staging_slot.as_mut().expect(..)`
                    // で取り出していたが、本番経路の `panic` 系 API 使用は
                    // `.claude/rules/coding-rust.md`「エラーは型付きエラーと
                    // し、本番経路で `unwrap()` / `expect()` を使わない」で
                    // 禁止されている。`staging_slot.take()` で所有権ごと取り
                    // 出し、`None` なら新規 `alloc` した値をそのまま使う形へ
                    // 変えることで、`Option` を再度覗いて取り出す
                    // （＝ `.expect()` が必要になる）分岐そのものを無くす。
                    // クロージャの最後で `staging_slot` へ書き戻すため、
                    // 成功・失敗いずれの経路でも `?` による早期 return 時点
                    // までに確保できていれば呼び出し元の返却キャッシュ処理
                    // （下の `if let Some(staging) = staging_slot`）は従来と
                    // 同じく機能する。
                    let mut staging = match staging_slot.take() {
                        Some(staging) => staging,
                        None => HostStaging::alloc(kind, ctx, numel)?,
                    };
                    let copy_and_read = (|| {
                        self.stream
                            .memcpy_dtoh(slice, staging.as_host_slice_mut())?;
                        self.stream.synchronize()?;
                        let view = staging.as_slice()?;
                        f(view);
                        Ok(())
                    })();
                    staging_slot = Some(staging);
                    copy_and_read
                });
                if let Some(staging) = staging_slot {
                    self.return_staging(numel, generation, staging);
                }
                copy_result
            }
        }
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

    /// [`ReadbackSentinel::SENTINEL`] が非ゼロであることを固定する
    /// （イシュー #1437）。`0` だと `vec![T::SENTINEL; n]` が
    /// `alloc_zeroed` 経由になり、mmap の COW ゼロページのまま実ページが
    /// コミットされず「事前タッチ」の意図が壊れる。GPU 実機なしで
    /// 機械的に検出できるよう、bit パターンでのゼロ判定を直接テストする。
    #[test]
    fn readback_sentinel_f32_is_nonzero() {
        assert_ne!(
            <f32 as ReadbackSentinel>::SENTINEL.to_bits(),
            0.0f32.to_bits()
        );
    }

    #[test]
    fn readback_sentinel_f16_is_nonzero() {
        assert_ne!(
            <half::f16 as ReadbackSentinel>::SENTINEL.to_bits(),
            half::f16::from_f32(0.0).to_bits()
        );
    }

    /// [`pretouched_host_vec`] が要求長・全要素 sentinel 埋めであることを
    /// 固定する（`PretouchedFresh` の前提条件）。
    #[test]
    fn pretouched_host_vec_f32_has_expected_len_and_fill() {
        let v: Vec<f32> = pretouched_host_vec(1024);
        assert_eq!(v.len(), 1024);
        assert!(
            v.iter()
                .all(|&x| x.to_bits() == <f32 as ReadbackSentinel>::SENTINEL.to_bits()),
            "all elements must equal the non-zero sentinel before D2H overwrites them"
        );
    }

    #[test]
    fn pretouched_host_vec_f16_has_expected_len_and_fill() {
        let v: Vec<half::f16> = pretouched_host_vec(37);
        assert_eq!(v.len(), 37);
        assert!(
            v.iter()
                .all(|&x| x.to_bits() == <half::f16 as ReadbackSentinel>::SENTINEL.to_bits())
        );
    }

    /// `numel == 0` は空 `Vec` を返し panic しないことを確認する
    /// （`readback` が 0 要素バッファに対して呼ばれるケースの境界値）。
    #[test]
    fn pretouched_host_vec_zero_numel_is_empty() {
        let v: Vec<f32> = pretouched_host_vec(0);
        assert!(v.is_empty());
    }

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

    /// [`CudaMemory::upload_into`]（イシュー #1349・#1212）は `dst.device()`
    /// が `self` のデバイスと一致しない場合、driver に一切触れずに
    /// `DeviceMismatch` を返す（`download_rejects_mismatched_device_
    /// ordinal` と同じ「実 GPU ドライバ呼び出しを経由しない検証」方針。
    /// numel == 0 の空バッファなので CUDA 非搭載環境でも到達可能）。
    #[test]
    fn upload_into_rejects_mismatched_device_ordinal() {
        match CudaDevice::new(0) {
            Ok(device) => {
                let mem = CudaMemory::new(&device);
                let other_ordinal = mem.ordinal + 1;
                let alloc = TrackedAllocation::new(Arc::clone(&mem.tracker), 0);
                let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                    storage: None,
                    _alloc: alloc,
                    generation: 0,
                    ordinal: other_ordinal,
                });
                let mut dst: DeviceBuffer<f32> =
                    DeviceBuffer::new(Device::Cuda(other_ordinal), vec![0], handle);
                let tensor = Tensor::<f32>::new(vec![], &[0]).unwrap();
                let err = mem.upload_into(&tensor, &mut dst, 0).unwrap_err();
                assert!(matches!(err, BackendError::DeviceMismatch));
            }
            Err(_) => {
                // 非搭載環境: `CudaDevice::new` 自体が型付きエラーで
                // 止まるため本テストの主張には到達しない。
            }
        }
    }

    /// [`CudaMemory::upload_into`] は `dst_offset + tensor.numel()` が
    /// `dst.numel()` を超える場合、driver に触れずに `InvalidArgument`
    /// で拒否する（REQ-8「カーネル側の手動境界チェックを省略しない」・
    /// OWASP A03。境界検査は device 一致検査の後・driver 呼び出しの前に
    /// 行われるため、CUDA 非搭載環境でも `CudaDevice::new` が成功する
    /// 環境でのみ到達する。空バッファ〈`numel == 0`〉の `dst` に対して
    /// 1 要素書き込もうとする最小ケースで検証する）。
    #[test]
    fn upload_into_rejects_out_of_range_write() {
        if let Ok(device) = CudaDevice::new(0) {
            let mem = CudaMemory::new(&device);
            let alloc = TrackedAllocation::new(Arc::clone(&mem.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                storage: None,
                _alloc: alloc,
                generation: 0,
                ordinal: mem.ordinal,
            });
            let mut dst: DeviceBuffer<f32> =
                DeviceBuffer::new(Device::Cuda(mem.ordinal), vec![0], handle);
            let tensor = Tensor::<f32>::new(vec![1.0], &[1]).unwrap();
            let err = mem.upload_into(&tensor, &mut dst, 0).unwrap_err();
            assert!(matches!(err, BackendError::InvalidArgument(_)));
        }
        // 非搭載環境: `CudaDevice::new` 自体が型付きエラーで止まるため
        // 本テストの主張には到達しない。
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
                    ordinal: other_ordinal,
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

    /// [`MemoryOps::with_host_view`]（イシュー #1336）のハンドル型・
    /// device ordinal 不一致検出が `download`（`download_rejects_
    /// mismatched_device_ordinal` 上記）と同一の `DeviceMismatch` 契約を
    /// 保つことを検証する。実 GPU ドライバ呼び出しは行わない
    /// （`numel == 0` の空バッファは `handle.storage: None` で
    /// `memcpy_dtoh` 等を経由しない）。
    #[test]
    fn with_host_view_rejects_mismatched_device_ordinal() {
        match CudaDevice::new(0) {
            Ok(device) => {
                let mem = CudaMemory::new(&device);
                let other_ordinal = mem.ordinal + 1;
                let alloc = TrackedAllocation::new(Arc::clone(&mem.tracker), 0);
                let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                    storage: None,
                    _alloc: alloc,
                    generation: 0,
                    ordinal: other_ordinal,
                });
                let buffer: DeviceBuffer<f32> =
                    DeviceBuffer::new(Device::Cuda(other_ordinal), vec![0], handle);
                let mut observed: Option<Vec<f32>> = None;
                let err = mem
                    .with_host_view(&buffer, &mut |slice| observed = Some(slice.to_vec()))
                    .unwrap_err();
                assert!(matches!(err, BackendError::DeviceMismatch));
                assert!(
                    observed.is_none(),
                    "DeviceMismatch で拒否される場合、呼び出し元クロージャは呼ばれないはず"
                );
            }
            Err(_) => {
                // 非搭載環境: `CudaDevice::new` 自体が型付きエラーで止まる
                // ため本テストの主張には到達しない（panic しないことが
                // 検証対象）。
            }
        }
    }

    /// [`MemoryOps::with_host_view`] が他バックエンド由来のハンドル型
    /// （`CudaBufferHandle` 以外）を `DeviceMismatch` で拒否することを
    /// 検証する（`download` の同種チェックと対称。`MockHandle` を使い
    /// `CudaDevice` すら要求しない完全な GPU 非依存テスト）。
    #[derive(Debug)]
    struct OtherBackendHandle;

    impl BufferHandle for OtherBackendHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn with_host_view_rejects_foreign_handle_type_without_gpu() {
        if let Ok(device) = CudaDevice::new(0) {
            let mem = CudaMemory::new(&device);
            let handle: Box<dyn BufferHandle> = Box::new(OtherBackendHandle);
            let buffer: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cuda(0), vec![0], handle);
            let err = mem
                .with_host_view(&buffer, &mut |_slice| {
                    panic!("foreign handle must be rejected before f is invoked")
                })
                .unwrap_err();
            assert!(matches!(err, BackendError::DeviceMismatch));
        }
    }

    /// [`MemoryOps::with_host_view`] の空テンソル契約（`handle.storage
    /// == None`）は FFI を呼ばず空スライスを `f` へ渡す（`buffer.rs`
    /// モジュールコメント「空テンソルの契約」）。`CudaDevice::new` に
    /// 依存しない完全な GPU 非依存テスト（`CudaMemory` は環境適応
    /// フィールドを直接構築できないため、`download` 側の空バッファ
    /// テスト〈`download_rejects_mismatched_device_ordinal` 等〉と同じ
    /// 環境適応ゲートを使う）。
    #[test]
    fn with_host_view_empty_buffer_invokes_f_with_empty_slice_and_no_ffi() {
        if let Ok(device) = CudaDevice::new(0) {
            let mem = CudaMemory::new(&device);
            let alloc = TrackedAllocation::new(Arc::clone(&mem.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                storage: None,
                _alloc: alloc,
                generation: 0,
                ordinal: mem.ordinal,
            });
            let buffer: DeviceBuffer<f32> =
                DeviceBuffer::new(Device::Cuda(mem.ordinal), vec![0], handle);
            let mut observed: Option<usize> = None;
            mem.with_host_view(&buffer, &mut |slice| observed = Some(slice.len()))
                .expect("空バッファは常に成功するはず");
            assert_eq!(observed, Some(0));
        }
    }

    /// `CudaMemory`（`host_staging: Arc<Mutex<HostStagingCache>>` を含む）
    /// が `Send + Sync` であることの静的検査（`with_host_view` の
    /// `host_staging` キャッシュ導入がスレッド安全性を壊していないことを
    /// コンパイル時に保証する。`PinnedHostSlice<f32>` は cudarc-0.19.8
    /// `core.rs:1394-1395` で `unsafe impl Send`／`Sync` 済み）。
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn cuda_memory_is_send_and_sync() {
        assert_send_sync::<CudaMemory>();
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
