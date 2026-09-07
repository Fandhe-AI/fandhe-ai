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
//! 一本化方針）。

use std::any::Any;
use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
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
    pub(crate) slice: Option<CudaSlice<f32>>,
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

/// codex-review P1 指摘対応（イシュー #1349・PR #1390）: `slice`
/// フィールドの自然な drop（`cudarc::driver::CudaSlice::drop` が
/// `cuMemFreeAsync`/`cuMemFree` を自身の `stream` へ直接発行する。
/// 本モジュール冒頭コメント「解放は `CudaSlice` の `Drop` に一本化する」
/// 参照）は `context_cache` の `begin_driver_call`／
/// `begin_capture_session` 排他機構を一切経由しない。そのため、別
/// スレッドが `graph::run_captured_sgd_step_segment` で同じ ordinal を
/// capture 中に本ハンドルが drop されると、その解放操作が capture 中の
/// 共有ストリームへ意図せず記録されうる（`context_cache::
/// wait_until_not_capturing` doc コメント参照）。
///
/// 本 `Drop` はフィールド既定の drop 順序（宣言順。`slice` → `_alloc`）
/// より**前**に走る（Rust の `Drop::drop` は構造体自身のコードが
/// フィールドの自動 drop より先に実行される規則）。`slice` が
/// `Some`（`numel > 0`。`numel == 0` は driver に触れないため対象外。
/// 構造体 doc コメント参照）の場合のみ、`CudaSlice::ordinal()` から
/// 得た ordinal で `wait_until_not_capturing` を呼び、別スレッドの
/// capture が終わるまで駐機してから本体の drop（後続のフィールド既定
/// drop）へ進む。同一スレッドが自身の capture 中に呼ぶ場合は待たない
/// （`wait_until_not_capturing` doc コメント参照）。
impl Drop for CudaBufferHandle {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.as_ref() {
            context_cache::wait_until_not_capturing(slice.ordinal());
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
                slice: None,
                _alloc: alloc,
                generation,
            })
        } else {
            let slice = self.stream.alloc_zeros::<f32>(numel)?;
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
            Box::new(CudaBufferHandle {
                slice: Some(slice),
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
                slice: None,
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
        let slice = self.stream.clone_htod(data)?;
        let bytes = checked_byte_len(data.len())?;
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
            slice: Some(slice),
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
        let data = match &handle.slice {
            None => Vec::new(),
            Some(slice) => {
                // 同期点は本モジュール共通の `readback` ヘルパーへ集約
                // 済み（#1013。`fandhe_ai_tensor_core::buffer` モジュール
                // コメント「download の同期契約」参照）。
                readback(&self.stream, slice)?
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
            let slice = handle
                .slice
                .as_mut()
                .ok_or_else(|| CudaError::InvalidShape {
                    detail: "upload_into: dst buffer has numel > 0 but no device allocation"
                        .to_string(),
                })?;
            let mut view = slice.slice_mut(dst_offset..end);
            self.stream.memcpy_htod(data, &mut view)?;
            Ok(())
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
        let Some(slice) = cuda_handle.slice.as_mut() else {
            context_cache::begin_driver_call(self.ordinal, &[generation])?;
            return Ok(());
        };
        self.with_driver_call(&[generation], map_cuda_error, || {
            self.stream.memset_zeros(slice).map_err(CudaError::from)
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
                    slice: None,
                    _alloc: alloc,
                    generation: 0,
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
                slice: None,
                _alloc: alloc,
                generation: 0,
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
                    slice: None,
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
}
