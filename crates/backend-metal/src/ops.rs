//! Metal バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `tensor_core::backend_ops::BackendOps` の Metal 実装。GEMM は
//! `gemm::MetalGemm::dispatch_auto`（動的タイル選択済み。TASK-1.8c・#40）
//! へ委譲する（既存カーネル・許容誤差・境界検査には触れない）。Metal は
//! 本イシュー時点で GEMM カーネルのみ実装済みのため、elementwise・
//! reduction は [`tensor_core::device::BackendError::Unsupported`] を
//! 返す（GPU 側カーネルの実装自体は本イシューのスコープ外。
//! out-of-scope-tracking.md 対象）。
//!
//! `cfg(target_os = "macos")` 限定（`objc2`／`objc2-foundation`／
//! `objc2-metal` と同じ cfg 境界。`.claude/rules/deps-policy.md`）。
//! 非 macOS 環境ではこのファイル自体がコンパイル対象に入らない
//! （`lib.rs` の cfg 境界と整合。`device.rs` と同方針）。

use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, Tensor};

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::gemm::MetalGemm;

/// Metal バックエンドの `BackendOps` 実装。`Device::Metal` は ordinal を
/// 持たない単一 variant のため（`docs/public-api-design.md` §4.1・
/// `device.rs::MetalDeviceProvider` と同じ位置付け）、本実装は複数 GPU の
/// 個別選択をサポートしない（システムデフォルトの Metal デバイスに
/// 対応する）。
///
/// `MetalContext`／`MetalGemm` は各メソッド呼び出し時に都度構築する
/// （`backend-cuda::ops::CudaBackendOps` と同じ設計判断。TASK-1.9b の
/// デバイスハンドル常駐が未着地のため。ハンドル常駐化は TASK-1.9b／1.9d
/// 以降の最適化対象）。
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackendOps;

impl MetalBackendOps {
    /// 新規 `MetalBackendOps` を構築する。構築自体はデバイス初期化を
    /// 行わないため常に成功する（実際の初期化は各メソッドが
    /// `MetalContext::new` を経由した時点）。
    pub fn new() -> Self {
        Self
    }
}

impl BackendOps for MetalBackendOps {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        // `dispatch_auto` は contiguous な `&[f32]` を要求する（CPU／CUDA
        // 実装と同じ契約）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let ctx = MetalContext::new()
            .map_err(|e: MetalError| BackendError::DeviceAllocationFailed(e.to_string()))?;
        let gemm = MetalGemm::new(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gemm
            .dispatch_auto(&ctx, a_slice, b_slice, m, n, k)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::add: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::mul: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::relu: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::exp: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::tanh: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::sum: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::max: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }
}
