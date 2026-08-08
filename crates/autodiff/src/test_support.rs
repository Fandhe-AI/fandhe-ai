//! テスト専用の naive `BackendOps` フィクスチャ（TASK-12.1d・#164）。
//!
//! `autodiff` は `backend-cpu`／`backend-cuda`／`backend-metal` のいずれ
//! にも依存しない（`docs/fusion-graph-design.md` §3.4「`autodiff` は
//! 具体クレートへの依存を一切持たない」）。`Tape::new(ops)` が必須所有値
//! `ops: Box<dyn BackendOps + Send>` を要求するため、クレート内の
//! `#[cfg(test)]` テスト（`src/` 内ユニットテスト。統合テストは
//! `tests/common/mod.rs` に別途同型のフィクスチャを持つ）はこのモジュール
//! の `test_ops()` を使う。実装は既存の `eval.rs`（クレート非公開の
//! 参照実装。FMA 契約〈`f32::mul_add`〉・広義ブロードキャスト等の意味論を
//! forward と共有）へそのまま委譲し、数式の実体を二重管理しない。
//!
//! `#[cfg(test)]` 限定モジュール（`lib.rs` の `mod` 宣言も同様に
//! `#[cfg(test)]`）のため、本番ビルドには一切含まれない。

#![cfg(test)]

use tensor_core::{BackendError, BackendOps, Device, Tensor};

/// `eval.rs` の naive 参照実装へ委譲するテスト専用 `BackendOps`。
/// `gemm`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/`max` はいずれも
/// 構造的に失敗しない（`eval.rs` 側が非 fallible なため）が、
/// `BackendOps` の契約に合わせ `Result` で包む。
pub(crate) struct TestOps;

impl BackendOps for TestOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::matmul(a, b))
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::add(a, b))
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::mul(a, b))
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::relu(a))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::exp(a))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::tanh(a))
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let shape = a.shape().to_vec();
        let out_shape =
            tensor_core::reduce_out_shape(&shape, dim).map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::sum(a, dim, &out_shape))
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let shape = a.shape().to_vec();
        let out_shape =
            tensor_core::reduce_out_shape(&shape, dim).map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::max(a, dim, &out_shape))
    }
}

/// `Tape::new(test_ops())` の形で使う（`src/` 内 `#[cfg(test)]` 専用）。
pub(crate) fn test_ops() -> Box<dyn BackendOps + Send> {
    Box::new(TestOps)
}
