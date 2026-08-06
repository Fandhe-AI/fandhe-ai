//! Metal バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `tensor_core::buffer::MemoryOps` の Metal 実装。既存の
//! [`crate::buffer::MetalBuffer`]（TASK-1.8a・#38。`new_with_data`／
//! `new_zeroed`／`read_to_vec`）をそのまま再利用し、新規 `unsafe` を
//! 追加しない（`.claude/rules/security.md` の「unsafe は必要最小限」
//! 方針。FFI 境界の safety 根拠は `buffer.rs` 側に既に記載済み）。
//!
//! `StorageModeShared`（Apple Silicon の UMA。`buffer.rs` モジュール
//! コメント参照）を用いるため、CUDA のような明示的な非同期転送・
//! `synchronize()` は不要（`MetalBuffer::read_to_vec` は `contents()` の
//! CPU 可視アドレスを直接読むため、呼び出し復帰時点でホストデータは
//! 既に確定している）。

use std::any::Any;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use tensor_core::Tensor;
use tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use tensor_core::device::{BackendError, Device};

/// Metal バッファの具体ハンドル。
///
/// `numel == 0`（空テンソルの契約。`tensor_core::buffer` モジュール
/// コメント参照）では `buffer` を `None` とする。`MetalBuffer::new_with_data`／
/// `new_zeroed` はいずれも長さ 0 を `MetalError::ZeroLengthAllocation`
/// として FFI 呼び出し前に拒否するため（`buffer.rs::checked_byte_len`）、
/// 空テンソルはこのハンドル自体を経由して Metal 側の拒否を回避する。
#[derive(Debug)]
struct MetalBufferHandle {
    buffer: Option<MetalBuffer>,
}

impl BufferHandle for MetalBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `MemoryOps` の Metal 実装。[`MetalContext`] を保持し、確保・転送の
/// たびに `MetalBuffer::new_with_data`／`new_zeroed` へ委譲する。
pub struct MetalMemory {
    context: MetalContext,
}

impl MetalMemory {
    /// 初期化済みの [`MetalContext`] から `MetalMemory` を構築する。
    pub fn new(context: MetalContext) -> Self {
        Self { context }
    }
}

/// `MetalError` を `BackendError` へ変換する。
///
/// `ZeroLengthAllocation`/`AllocationSizeOverflow` は形状検証系の失敗
/// （呼び出し前の shape 由来）のため `DeviceAllocationFailed` に、
/// `CommandBufferExecutionFailed` 等の実行時失敗は 4.4 の
/// `KernelLaunchFailed` に寄せる（本モジュールは GEMM ディスパッチを
/// 行わないため実際には到達しないが、`MetalError` は `#[non_exhaustive]`
/// であり網羅的 match が書けないため wildcard の受け皿として用意する）。
fn map_metal_error(err: MetalError) -> BackendError {
    match err {
        MetalError::ZeroLengthAllocation => {
            BackendError::DeviceAllocationFailed("zero-length allocation requested".to_string())
        }
        MetalError::AllocationSizeOverflow { len } => BackendError::DeviceAllocationFailed(
            format!("buffer byte length overflows usize for len={len} elements"),
        ),
        MetalError::BufferAllocation { bytes } => {
            BackendError::DeviceAllocationFailed(format!("allocation failed for {bytes} bytes"))
        }
        MetalError::DeviceUnavailable | MetalError::CommandQueueCreation => {
            BackendError::DeviceUnavailable(err.to_string())
        }
        // CUDA 実装（`backend-cuda/src/memory.rs::map_cuda_error`）の
        // `CudaError::InvalidShape { detail } => BackendError::
        // DeviceAllocationFailed(detail)` と同じ変換先に揃える
        // （レビュー指摘対応。detail を保持したまま伝播する）。
        MetalError::ShapeMismatch { detail } => BackendError::DeviceAllocationFailed(detail),
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// shape の要素数積を検査付きで計算する（`checked_byte_len`
/// （`buffer.rs`）が要素数からバイト長を検査するのに対し、本関数は
/// その手前の shape → 要素数の積を検証する。外部由来の shape がこの
/// 経路へ流入しうるための前段検証。OWASP A03）。
fn checked_numel(shape: &[usize]) -> Result<usize, MetalError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            // エラー値構築のために `shape.iter().product()`（checked 乗算）を
            // 再計算しない。`try_fold` が既に検知したのと同じオーバーフロー
            // 乗算を `product()` で踏むと、debug プロファイル
            // （overflow-checks 既定 ON）ではここ自体が
            // `attempt to multiply with overflow` で panic し、本関数が
            // 防御対象とする入力（shape 由来のオーバーフロー）で目的を
            // 果たせなくなる（CPU/CUDA 版の `ok_or_else` + 遅延評価パターンに
            // 揃える）。`wrapping_mul` は overflow-checks の影響を受けず
            // panic しないため、参考値としての近似 len を安全に算出できる。
            MetalError::AllocationSizeOverflow {
                len: shape.iter().fold(1usize, |acc, &dim| acc.wrapping_mul(dim)),
            }
        })
}

impl MetalMemory {
    fn alloc_zeroed_inner(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, MetalError> {
        let numel = checked_numel(shape)?;
        let handle: Box<dyn BufferHandle> = if numel == 0 {
            Box::new(MetalBufferHandle { buffer: None })
        } else {
            let buf = MetalBuffer::new_zeroed(&self.context, numel)?;
            Box::new(MetalBufferHandle { buffer: Some(buf) })
        };
        Ok(DeviceBuffer::new(Device::Metal, shape.to_vec(), handle))
    }

    fn upload_inner(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, MetalError> {
        let shape = tensor.shape().to_vec();
        if tensor.numel() == 0 {
            let handle: Box<dyn BufferHandle> = Box::new(MetalBufferHandle { buffer: None });
            return Ok(DeviceBuffer::new(Device::Metal, shape, handle));
        }
        // 非 contiguous な入力は実体化してから転送する（`MemoryOps::upload`
        // の契約。`tensor_core::buffer` モジュールコメント参照）。
        let contiguous = tensor.contiguous();
        let data = contiguous.as_slice().ok_or(MetalError::BufferAllocation {
            bytes: 0, // contiguous() 直後に as_slice が None を返す到達不能パス。
        })?;
        let buf = MetalBuffer::new_with_data(&self.context, data)?;
        let handle: Box<dyn BufferHandle> = Box::new(MetalBufferHandle { buffer: Some(buf) });
        Ok(DeviceBuffer::new(Device::Metal, shape, handle))
    }

    fn download_inner(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, MetalError> {
        let handle = buffer
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(MetalError::DeviceUnavailable)?;
        let data = match &handle.buffer {
            // StorageModeShared のため read_to_vec 復帰時点でホストデータは
            // 既に確定している（同期不要。モジュール冒頭コメント参照）。
            None => Vec::new(),
            Some(buf) => buf.read_to_vec(),
        };
        // shape 不整合（通常到達しない防御的経路）を `BufferAllocation
        // { bytes: 0 }` のような実態と異なる variant に化けさせず、
        // 元の `ShapeError` の詳細を `MetalError::ShapeMismatch` として
        // 保持する（CUDA 実装の `CudaError::InvalidShape { detail:
        // format!(...) }` と同型の防御的経路。レビュー指摘対応）。
        Tensor::new(data, buffer.shape()).map_err(|err| MetalError::ShapeMismatch {
            detail: format!("download produced a shape-inconsistent tensor: {err}"),
        })
    }
}

impl MemoryOps for MetalMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        self.alloc_zeroed_inner(shape).map_err(map_metal_error)
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        self.upload_inner(tensor).map_err(map_metal_error)
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        // ハンドル型不一致（他バックエンドの `DeviceBuffer` を誤って
        // 渡した場合）は、CPU 実装（`backend-cpu/src/memory.rs`）と
        // 同じ `BackendError::DeviceMismatch` に統一する。`MetalError`
        // には「デバイス確保失敗」「デバイス利用不可」等の既存 variant
        // しかなく、`map_metal_error` を経由すると実態と異なるエラー種別
        // （`DeviceUnavailable`）に化けてしまうため、ここで直接判定する
        // （3 バックエンド共通のハンドル型不一致検出。レビュー指摘対応）。
        if buffer.downcast_handle::<MetalBufferHandle>().is_none() {
            return Err(BackendError::DeviceMismatch);
        }
        self.download_inner(buffer).map_err(map_metal_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, MetalError::AllocationSizeOverflow { .. }));
    }

    #[test]
    fn checked_numel_accepts_ordinary_shape() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel(&[0, 3]).unwrap(), 0);
    }

    #[test]
    fn map_metal_error_covers_zero_length_allocation() {
        let err = map_metal_error(MetalError::ZeroLengthAllocation);
        assert!(matches!(err, BackendError::DeviceAllocationFailed(_)));
    }

    #[test]
    fn map_metal_error_covers_device_unavailable() {
        let err = map_metal_error(MetalError::DeviceUnavailable);
        assert!(matches!(err, BackendError::DeviceUnavailable(_)));
    }
}
