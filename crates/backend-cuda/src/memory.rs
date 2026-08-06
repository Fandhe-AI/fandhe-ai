//! CUDA バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `tensor_core::buffer::MemoryOps` の CUDA 実装。既存の GEMM 実装
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
//! （`tensor_core::buffer` モジュールコメント「解放方針」と同じ RAII
//! 一本化方針）。

use std::any::Any;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use crate::device::CudaDevice;
use crate::error::CudaError;
use tensor_core::Tensor;
use tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use tensor_core::device::{BackendError, Device};

/// CUDA バッファの具体ハンドル。
///
/// `numel == 0`（空テンソルの契約。`tensor_core::buffer` モジュール
/// コメント参照）では `slice` を `None` とし、`cuMemAlloc` 自体を呼ばない
/// （一部環境の driver は 0 バイト確保を拒否する。`gemm.rs` の `k == 0`
/// 早期 return コメントと同じ理由）。`CudaSlice<T>` は `#[derive(Debug)]`
/// されているため本型も `Debug` を導出できる。
#[derive(Debug)]
struct CudaBufferHandle {
    slice: Option<CudaSlice<f32>>,
}

impl BufferHandle for CudaBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `MemoryOps` の CUDA 実装。`CudaDevice::new` を経由して初期化済みの
/// ハンドルからのみ構築できる（受け入れ条件「CUDA 非搭載環境で実行時に
/// panic せず型付きエラーが返る」を、確保・転送呼び出し前の構築段階から
/// 一貫させるため）。
pub struct CudaMemory {
    stream: Arc<CudaStream>,
    ordinal: usize,
}

impl CudaMemory {
    /// 初期化済みの [`CudaDevice`] から `CudaMemory` を構築する。
    /// `device.stream()` を `Arc` クローンで共有する（`gemm.rs::CudaGemm::new`
    /// と同じ共有契約）。
    pub fn new(device: &CudaDevice) -> Self {
        Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
        }
    }
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
/// `TransferFailed`（TASK-1.9b で追加。`tensor_core::device` 参照）は
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
fn map_cuda_error(err: CudaError) -> BackendError {
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
/// 表す契約（`tensor_core::device::BackendError` ドキュメンテーション
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
        let handle: Box<dyn BufferHandle> = if numel == 0 {
            // 空テンソルの契約（`tensor_core::buffer` モジュールコメント）:
            // FFI を呼ばず空ハンドルを返す。
            Box::new(CudaBufferHandle { slice: None })
        } else {
            let slice = self.stream.alloc_zeros::<f32>(numel)?;
            Box::new(CudaBufferHandle { slice: Some(slice) })
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
            let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle { slice: None });
            return Ok(DeviceBuffer::new(Device::Cuda(self.ordinal), shape, handle));
        }
        // 非 contiguous な入力は実体化してから転送する（`MemoryOps::upload`
        // の契約。`tensor_core::buffer` モジュールコメント参照）。
        let contiguous = tensor.contiguous();
        let data = contiguous
            .as_slice()
            .ok_or_else(|| CudaError::InvalidShape {
                detail: "contiguous() の直後にもかかわらず as_slice が None を返した \
                     （tensor-core 側のロジック不整合。到達しないはずの防御経路）"
                    .to_string(),
            })?;
        let slice = self.stream.clone_htod(data)?;
        let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle { slice: Some(slice) });
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
                // `clone_dtoh` は内部で `cuMemcpyDtoHAsync` を発行する
                // 非同期コピーのため（`cudarc-0.19.8/src/driver/safe/
                // core.rs::memcpy_dtoh`）、`download` 復帰時点でホスト
                // データが確定していることを保証するため
                // `synchronize()` を後段に挟む（`tensor_core::buffer`
                // モジュールコメント「download の同期契約」参照。
                // カーネル起動直後の `gemm.rs` はカーネル完了待ちとして
                // 起動 → synchronize → clone_dtoh の順だが、本関数は
                // 「clone_dtoh 自体の非同期完了待ち」が主目的のため
                // clone_dtoh → synchronize の順になる）。
                let host = self.stream.clone_dtoh(slice)?;
                self.stream.synchronize()?;
                host
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
                let handle: Box<dyn BufferHandle> = Box::new(CudaBufferHandle { slice: None });
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
}
