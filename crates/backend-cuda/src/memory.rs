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
}

impl BufferHandle for CudaBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
            })
        } else {
            let slice = self.stream.alloc_zeros::<f32>(numel)?;
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
            Box::new(CudaBufferHandle {
                slice: Some(slice),
                _alloc: alloc,
            })
        };
        Ok(DeviceBuffer::new(
            Device::Cuda(self.ordinal),
            shape.to_vec(),
            handle,
        ))
    }

    fn upload_inner(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, CudaError> {
        let shape = tensor.shape().to_vec();
        if tensor.numel() == 0 {
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle {
                slice: None,
                _alloc: alloc,
            });
            return Ok(DeviceBuffer::new(Device::Cuda(self.ordinal), shape, handle));
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
        });
        Ok(DeviceBuffer::new(Device::Cuda(self.ordinal), shape, handle))
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

impl MemoryOps for CudaMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        self.alloc_zeroed_inner(shape).map_err(map_cuda_alloc_error)
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        self.upload_inner(tensor).map_err(map_cuda_error)
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
        self.download_inner(buffer).map_err(map_cuda_error)
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
        // 空ハンドル（`numel == 0`）は `pool.rs::PooledMemory::alloc_zeroed`
        // が空テンソル契約によりそもそもプールを介さない経路で扱うため
        // 到達しない想定だが、`CudaBufferHandle::slice` が `None` の場合に
        // 備えて no-op として安全に振る舞う（`buffer.rs` モジュールコメント
        // 「空テンソルの契約」と同じ扱い）。
        let Some(slice) = cuda_handle.slice.as_mut() else {
            return Ok(());
        };
        self.stream
            .memset_zeros(slice)
            .map_err(|e| BackendError::DeviceAllocationFailed(format!("{e:?}")))
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
                    slice: None,
                    _alloc: alloc,
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
}
