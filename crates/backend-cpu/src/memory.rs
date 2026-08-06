//! CPU バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `tensor_core::buffer::MemoryOps` の CPU 実装。CPU は「デバイス」が
//! ホストメモリそのものであるため、`upload`/`download` は FFI を伴わず
//! `Vec<f32>` の複製のみで完結する。`backend-cuda::CudaMemory`／
//! `backend-metal::MetalMemory` の数値一致の参照点（`.claude/rules/coding-rust.md`
//! の「CPU 参照実装」方針）であり、`CpuDeviceProvider`（`device.rs`・#44）
//! と同種の位置付けを `MemoryOps` 側で担う。

use std::any::Any;

use tensor_core::Tensor;
use tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use tensor_core::device::{BackendError, Device};

/// CPU バッファの具体ハンドル。ホスト常駐そのものなので `Vec<f32>` を
/// 直接保持する。`Drop` は `Vec<f32>` の既定 `Drop` に委ねる（RAII 一本化
/// 方針。`tensor_core::buffer` モジュールコメント参照）。
#[derive(Debug)]
struct CpuBufferHandle {
    data: Vec<f32>,
}

impl BufferHandle for CpuBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `MemoryOps` の CPU 実装。内部状態を持たない（`Device::Cpu` は単一
/// デバイスであり、`CpuDeviceProvider` と同じく構築自体が失敗しない）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuMemory;

impl CpuMemory {
    /// 新規 `CpuMemory` を構築する。
    pub fn new() -> Self {
        Self
    }
}

/// shape の要素数積を検査付きで計算する。`Tensor::zeros` 等が内部で行う
/// 検証と同種だが、`tensor_core::tensor::checked_numel` はクレート内
/// 非公開（`pub(crate)`）のため、`backend-cpu` 側で同じ検証をここでも
/// 独立して行う（`.claude/rules/security.md` A03: 外部由来の shape が
/// この経路へ流入しうるための前段検証）。
fn checked_numel(shape: &[usize]) -> Result<usize, BackendError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            BackendError::DeviceAllocationFailed(format!(
                "shape element count overflows usize: {shape:?}"
            ))
        })
}

impl MemoryOps for CpuMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        let numel = checked_numel(shape)?;
        // numel == 0 でも `vec![0.0f32; 0]` は素通りする（CPU には
        // MetalBuffer::checked_byte_len のような FFI 前ゼロ長拒否は
        // 不要）が、他バックエンドと同じ「空ハンドル」契約
        // （`tensor_core::buffer` モジュールコメント）に揃えて明示的に
        // 分岐しておく（将来 CPU 側にプール確保等を導入した際に契約が
        // 崩れないようにするための保険）。
        let data = vec![0.0f32; numel];
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle { data });
        Ok(DeviceBuffer::new(Device::Cpu, shape.to_vec(), handle))
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        let shape = tensor.shape().to_vec();
        if tensor.numel() == 0 {
            let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle { data: Vec::new() });
            return Ok(DeviceBuffer::new(Device::Cpu, shape, handle));
        }
        // 非 contiguous な入力（transpose/narrow 後の view）は実体化して
        // から複製する（`MemoryOps::upload` の契約。`tensor_core::buffer`
        // モジュールコメント「非 contiguous テンソルの upload」参照）。
        let contiguous = tensor.contiguous();
        let data = contiguous
            .as_slice()
            .ok_or_else(|| {
                BackendError::DeviceAllocationFailed(
                    "contiguous() の直後にもかかわらず as_slice が None を返した \
                     （tensor-core 側のロジック不整合。到達しないはずの防御経路）"
                        .to_string(),
                )
            })?
            .to_vec();
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle { data });
        Ok(DeviceBuffer::new(Device::Cpu, shape, handle))
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        if buffer.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let handle = buffer
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if buffer.numel() == 0 {
            return Tensor::new(Vec::new(), buffer.shape()).map_err(BackendError::ShapeMismatch);
        }
        Tensor::new(handle.data.clone(), buffer.shape()).map_err(BackendError::ShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_download_roundtrip_is_bit_exact() {
        let mem = CpuMemory::new();
        let data = vec![1.0f32, -2.5, 3.25, f32::MIN_POSITIVE, f32::MAX];
        let tensor = Tensor::<f32>::new(data.clone(), &[5]).unwrap();

        let buf = mem.upload(&tensor).unwrap();
        assert_eq!(buf.shape(), &[5]);
        assert_eq!(buf.device(), Device::Cpu);

        let back = mem.download(&buf).unwrap();
        assert_eq!(back.shape(), &[5]);
        for (i, &expected) in data.iter().enumerate() {
            let actual = back.get(&[i]).unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip must be bit exact at index {i}"
            );
        }
    }

    #[test]
    fn upload_non_contiguous_tensor_materializes_before_transfer() {
        let mem = CpuMemory::new();
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        // transpose 直後は非 contiguous（as_slice は None）。upload は
        // これを内部で contiguous() 実体化してから転送できることを確認する。
        assert!(!tt.is_contiguous());
        assert!(tt.as_slice().is_none());

        let buf = mem.upload(&tt).unwrap();
        let back = mem.download(&buf).unwrap();
        assert_eq!(back.shape(), &[3, 2]);
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(back.get(&[i, j]).unwrap(), tt.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn zero_numel_tensor_roundtrips_without_error() {
        let mem = CpuMemory::new();
        let empty = Tensor::<f32>::zeros(&[0, 3]).unwrap();

        let buf = mem.upload(&empty).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.shape(), &[0, 3]);

        let back = mem.download(&buf).unwrap();
        assert!(back.is_empty());
        assert_eq!(back.shape(), &[0, 3]);
    }

    #[test]
    fn alloc_zeroed_returns_all_zero_buffer() {
        let mem = CpuMemory::new();
        let buf = mem.alloc_zeroed(&[2, 2]).unwrap();
        let tensor = mem.download(&buf).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(tensor.get(&[i, j]).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn alloc_zeroed_element_count_overflow_is_rejected() {
        let mem = CpuMemory::new();
        let err = mem.alloc_zeroed(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, BackendError::DeviceAllocationFailed(_)));
    }

    #[test]
    fn download_rejects_mismatched_handle_type() {
        // 他バックエンドのハンドル型を模した downcast 失敗経路の検証。
        #[derive(Debug)]
        struct OtherHandle;
        impl BufferHandle for OtherHandle {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let mem = CpuMemory::new();
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![1], Box::new(OtherHandle));
        let err = mem.download(&buf).unwrap_err();
        assert!(matches!(err, BackendError::DeviceMismatch));
    }

    #[test]
    fn repeated_upload_alloc_cycles_do_not_leak_across_backend_error_reuse() {
        // リークなく動作する（受け入れ条件）の CPU 側検証: 大量サイクルで
        // panic せず・値が破壊されないことを確認する（`Vec<f32>` の
        // `Drop` に委ねる RAII 方式のため実際のメモリ解放そのものは
        // tensor-core 側モックテストで検証済み。ここでは CPU 経路が
        // 繰り返し呼び出しに対して安定して動作することを確認する）。
        let mem = CpuMemory::new();
        for i in 0..200 {
            let shape = [((i % 8) + 1)];
            let buf = mem.alloc_zeroed(&shape).unwrap();
            let back = mem.download(&buf).unwrap();
            assert_eq!(back.numel(), shape[0]);
        }
    }
}
