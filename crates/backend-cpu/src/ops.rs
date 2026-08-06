//! CPU バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `tensor_core::backend_ops::BackendOps` の CPU 実装。既存カーネル
//! （`gemm_blis::gemm_blis_parallel`・`elementwise::{add,mul,relu,exp,tanh}`・
//! `reduction::{sum,max}`）への薄い委譲に徹し、カーネル本体・許容誤差・
//! 境界検査には一切触れない（`.claude/rules/delegation-impl.md` の
//! 実装フロー標準どおり、本ファイルはディスパッチ層のみを追加する）。
//! CPU は常に利用可能なため（`device::CpuDeviceProvider` と同じ位置付け）
//! 全 8 演算とも `Unsupported` を返す経路は持たない（TASK-1.9c の受け入れ
//! 条件「3 バックエンドが呼び分けられる」の参照実装として、CPU は常に
//! 実カーネルを実行できることを保証する）。

use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, Tensor};

use crate::gemm_blis::gemm_blis_parallel;
use crate::{elementwise, reduction};

/// CPU バックエンドの `BackendOps` 実装。状態を持たないゼロサイズ型
/// （CPU カーネルはホストメモリのみを扱い、CUDA `CudaDevice`／Metal
/// `MetalContext` のようなデバイスハンドルを必要としないため）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackendOps;

impl CpuBackendOps {
    /// 新規 `CpuBackendOps` を構築する。
    pub fn new() -> Self {
        Self
    }
}

/// `Tensor::contiguous()` 実体化後もなお `as_slice()` が `None` を返す
/// （契約上到達しないはずだが、`Tensor` 実装のバグに対する fail-safe と
/// して型付きエラーで受ける）場合の変換ヘルパー。shape 不一致ではなく
/// 実行時の契約違反であるため `BackendError::KernelLaunchFailed` を返す
/// （命名を実際のエラー種別に合わせ `gemm_shape_mismatch` から改名。
/// Review 指摘対応）。
fn gemm_contiguity_fail_safe(msg: impl std::fmt::Display) -> BackendError {
    BackendError::KernelLaunchFailed(msg.to_string())
}

impl BackendOps for CpuBackendOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        // `gemm_blis_parallel` は contiguous な `&[f32]` を要求する。
        // 非 contiguous（view・broadcast 由来）な入力は `contiguous()` で
        // 実体化してから渡す（`Tensor::as_slice` は非 contiguous では
        // `None` を返す契約。`crates/tensor-core/src/tensor.rs` 参照）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm: lhs not contiguous after contiguous()")
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm: rhs not contiguous after contiguous()")
        })?;

        let mut out = vec![0.0f32; m * n];
        gemm_blis_parallel(a_slice, b_slice, &mut out, m, n, k)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::add(a, b).map_err(BackendError::ShapeMismatch)
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::mul(a, b).map_err(BackendError::ShapeMismatch)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::relu(a).map_err(BackendError::ShapeMismatch)
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::exp(a).map_err(BackendError::ShapeMismatch)
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::tanh(a).map_err(BackendError::ShapeMismatch)
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        reduction::sum(a, dim).map_err(reduce_error_to_backend_error)
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        reduction::max(a, dim).map_err(reduce_error_to_backend_error)
    }
}

/// `reduction::ReduceError`（`Shape`／`EmptyReduction` の 2 variant）を
/// `BackendError` へ写像する。`EmptyReduction` は shape 由来ではない
/// 実行時失敗のため `KernelLaunchFailed` に寄せる（`BackendError` に
/// reduction 専用 variant は設けない。§4.4 の 5 variant + TASK-1.9a/1.9c
/// 拡張の範囲に収める）。
fn reduce_error_to_backend_error(err: reduction::ReduceError) -> BackendError {
    match err {
        reduction::ReduceError::Shape(shape_err) => BackendError::ShapeMismatch(shape_err),
        reduction::ReduceError::EmptyReduction { op } => {
            BackendError::KernelLaunchFailed(format!("empty reduction for op \"{op}\""))
        }
    }
}
