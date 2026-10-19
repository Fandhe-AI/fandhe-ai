//! ROCm 専用の `BufferHandle`（イシュー #3433・分解案 R2）。
//!
//! 役割: `BackendOps::binary_elementwise_device`／`unary_elementwise_device` が受け渡す
//! [`DeviceBuffer`](fandhe_ai_tensor_core::DeviceBuffer) の実体。trait doc はホストへ download して
//! 委譲する暗黙のフォールバックを禁じているため、デバイス常駐のハンドルが必要になる。
//! 呼び出し元は `ops/ew.rs` の device 系メソッドだけ。
//!
//! # 判別手段
//! `RocmBackendOps::device()` は #3157 まで仮置きの `Device::Cpu` を返し、`CpuMemory` のバッファも
//! `Device::Cpu` を名乗る。このため **`DeviceBuffer::device()` の比較では CPU 由来のバッファと区別
//! できない**。#3157 で `Device::Rocm` が入るまでは、`downcast_handle::<RocmBufferHandle>()` が
//! 唯一の判別手段であり、`None` なら `DeviceMismatch` にする（fail-closed）。

use fandhe_ai_tensor_core::BufferHandle;
use std::any::Any;
use std::fmt;

use crate::typed_buffer::TypedDeviceBuffer;

/// ROCm デバイス常駐バッファの実体。
pub(crate) enum RocmBufferHandle {
    /// 要素数 0（HIP に触れずに作れる。確保しない）。
    Empty,
    /// HIP デバイスバッファ（所有。Drop で `hipFree`）。
    Hip(TypedDeviceBuffer<f32>),
    /// テスト専用: ROCm 非搭載環境で組み立て経路を検証するホスト側の模擬デバイスメモリ。
    #[cfg(test)]
    Host(Vec<f32>),
}

impl RocmBufferHandle {
    /// 保持している要素数。
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Hip(b) => b.len(),
            #[cfg(test)]
            Self::Host(v) => v.len(),
        }
    }
}

impl From<TypedDeviceBuffer<f32>> for RocmBufferHandle {
    /// デバイスカーネルの出力（所有バッファ）を常駐ハンドルへ包む。
    fn from(buf: TypedDeviceBuffer<f32>) -> Self {
        Self::Hip(buf)
    }
}

impl fmt::Debug for RocmBufferHandle {
    /// `TypedDeviceBuffer` は `Debug` を持たない（生ポインタを出さない）ため、種別と要素数だけ出す。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::Empty => "Empty",
            Self::Hip(_) => "Hip",
            #[cfg(test)]
            Self::Host(_) => "Host",
        };
        write!(f, "RocmBufferHandle::{kind}(len={})", self.len())
    }
}

impl BufferHandle for RocmBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_shows_kind_and_len_only() {
        assert_eq!(
            format!("{:?}", RocmBufferHandle::Empty),
            "RocmBufferHandle::Empty(len=0)"
        );
        assert_eq!(
            format!("{:?}", RocmBufferHandle::Host(vec![1.0, 2.0])),
            "RocmBufferHandle::Host(len=2)"
        );
    }

    #[test]
    fn downcast_distinguishes_from_other_handle_types() {
        use fandhe_ai_tensor_core::{Device, DeviceBuffer};
        let b: DeviceBuffer<f32> = DeviceBuffer::new(
            Device::Cpu,
            vec![2],
            Box::new(RocmBufferHandle::Host(vec![0.0; 2])),
        );
        assert!(b.downcast_handle::<RocmBufferHandle>().is_some());
    }
}
