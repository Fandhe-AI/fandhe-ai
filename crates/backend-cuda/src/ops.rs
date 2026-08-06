//! CUDA バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `tensor_core::backend_ops::BackendOps` の CUDA 実装。GEMM は
//! `gemm::CudaGemm::run_tiled_f32` へ委譲する（既存カーネル・許容誤差・
//! 境界検査には触れない）。CUDA は本イシュー時点で GEMM カーネルのみ
//! 実装済みのため、elementwise・reduction は
//! [`tensor_core::device::BackendError::Unsupported`] を返す（GPU 側
//! カーネルの実装自体は本イシューのスコープ外。out-of-scope-tracking.md
//! 対象）。
//!
//! `device.rs` の「動的ロード panic 回避ゲート」方針をそのまま踏襲する:
//! `CudaDevice::new` は driver 不在を `Err(CudaError::DriverUnavailable)`
//! で返す non-panicking な入口であり、本実装はこれを経由してから
//! `BackendError::CudaUnavailable` へ変換する（panic しない。
//! `.claude/rules/coding-rust.md`）。

use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, Tensor};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::CudaGemm;

/// CUDA バックエンドの `BackendOps` 実装。`ordinal` は `Device::Cuda(_)`
/// の一致判定に使う `cudarc` のデバイス番号
/// （`CudaContext::new(ordinal)` に対応。`tensor_core::device::Device`
/// の doc コメント参照）。
///
/// `CudaDevice`／`CudaGemm` は各メソッド呼び出し時に都度構築する
/// （TASK-1.9b の `DeviceBuffer`／デバイスハンドル常駐が未着地のため。
/// モジュール冒頭 `backend_ops` の突合コメント参照）。ハンドル常駐化・
/// 再利用による初期化コスト削減は TASK-1.9b／1.9d 以降の最適化対象。
#[derive(Debug, Clone, Copy)]
pub struct CudaBackendOps {
    ordinal: usize,
}

impl CudaBackendOps {
    /// 指定した `ordinal` に対応する `CudaBackendOps` を構築する。
    /// 構築自体は driver 初期化を行わないため常に成功する（実際の
    /// driver 呼び出しは各メソッドが `CudaDevice::new` を経由した時点）。
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// `CudaDevice::new` を経由してデバイスハンドルを取得する。
    /// driver 不在・初期化失敗は `BackendError::CudaUnavailable` へ
    /// 変換する（panic 回避ゲートは `CudaDevice::new` 内部で完結する。
    /// `device.rs` 参照）。
    fn device_handle(&self) -> Result<CudaDevice, BackendError> {
        CudaDevice::new(self.ordinal)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))
    }
}

impl BackendOps for CudaBackendOps {
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        // `run_tiled_f32` は contiguous な `&[f32]` を要求する（CPU 実装
        // と同じ契約。`ops.rs`（backend-cpu）参照）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let device = self.device_handle()?;
        let gemm = CudaGemm::new(&device)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
        let out = gemm
            .run_tiled_f32(a_slice, b_slice, m, n, k)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::add: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::mul: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::relu: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::exp: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::tanh: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::sum: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::max: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }
}
