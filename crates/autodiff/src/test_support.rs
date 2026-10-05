//! テスト専用の naive `BackendOps` フィクスチャ（TASK-12.1d・#164）。
//!
//! `autodiff` は `backend-cpu`／`backend-cuda`／`backend-metal` のいずれ
//! にも依存しない（`docs/fusion-graph-design.md` §3.4「`autodiff` は
//! 具体クレートへの依存を一切持たない」）。`Tape::new_with_ops(ops)` が必須所有値
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

use fandhe_ai_tensor_core::{
    BackendError, BackendOps, Device, Tensor, TypedOps, bf16, f16, gemm_out_shape,
};

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
        // `gemm` は 2 次元カーネル入口のため `gemm_out_shape`（2 次元
        // 厳密版）で検査する（イシュー #1715。バッチ対応は
        // `BackendOps::gemm_batched` の既定合成実装が本メソッドを
        // 2 次元ずつ呼ぶことで自動的に得られる）。
        gemm_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
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
        let out_shape = fandhe_ai_tensor_core::reduce_out_shape(&shape, dim)
            .map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::sum(a, dim, &out_shape))
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let shape = a.shape().to_vec();
        let out_shape = fandhe_ai_tensor_core::reduce_out_shape(&shape, dim)
            .map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::max(a, dim, &out_shape))
    }
}

/// `Tape::new_with_ops(test_ops())` の形で使う（`src/` 内 `#[cfg(test)]` 専用）。
pub(crate) fn test_ops() -> Box<dyn BackendOps + Send> {
    Box::new(TestOps)
}

/// 低精度 forward（`TypedOps<f16>`／`TypedOps<bf16>`）を持つテスト専用
/// `BackendOps`（イシュー #2628）。f32 側は [`TestOps`] へ委譲し、
/// `TypedOps<T>` の 8 演算は「f32 へ昇格 → `TestOps` の f32 演算 → 1 回丸め」
/// で実装する（`backend-cpu` の `typed_f16.rs`／`typed_bf16.rs` と同じ数値方式。
/// `autodiff` は `backend-cpu` に依存しないため独立に持つ）。
/// `low_precision_ops` の単体テストが、typed ops を持たない [`TestOps`]
/// では検証できない成功経路（丸めオラクル一致・backward）に使う。
pub(crate) struct LowPrecisionTestOps;

macro_rules! impl_typed_test_ops {
    ($t:ty, $up:ident, $down:ident) => {
        fn $up(t: &Tensor<$t>) -> Tensor<f32> {
            let data: Vec<f32> = t.host_slice().iter().map(|v| v.to_f32()).collect();
            Tensor::new(data, t.shape()).expect("test: shape 不変")
        }
        fn $down(t: &Tensor<f32>) -> Tensor<$t> {
            let data: Vec<$t> = t.host_slice().iter().map(|&v| <$t>::from_f32(v)).collect();
            Tensor::new(data, t.shape()).expect("test: shape 不変")
        }
        impl TypedOps<$t> for LowPrecisionTestOps {
            fn gemm(&self, a: &Tensor<$t>, b: &Tensor<$t>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.gemm(&$up(a), &$up(b))?))
            }
            fn add(&self, a: &Tensor<$t>, b: &Tensor<$t>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.add(&$up(a), &$up(b))?))
            }
            fn mul(&self, a: &Tensor<$t>, b: &Tensor<$t>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.mul(&$up(a), &$up(b))?))
            }
            fn relu(&self, a: &Tensor<$t>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.relu(&$up(a))?))
            }
            fn exp(&self, a: &Tensor<$t>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.exp(&$up(a))?))
            }
            fn tanh(&self, a: &Tensor<$t>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.tanh(&$up(a))?))
            }
            fn sum(&self, a: &Tensor<$t>, dim: Option<usize>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.sum(&$up(a), dim)?))
            }
            fn max(&self, a: &Tensor<$t>, dim: Option<usize>) -> Result<Tensor<$t>, BackendError> {
                Ok($down(&TestOps.max(&$up(a), dim)?))
            }
        }
    };
}

impl_typed_test_ops!(f16, up_f16, down_f16);
impl_typed_test_ops!(bf16, up_bf16, down_bf16);

impl BackendOps for LowPrecisionTestOps {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        TestOps.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        TestOps.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        TestOps.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        TestOps.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        TestOps.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        TestOps.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        TestOps.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        TestOps.max(a, dim)
    }
    fn typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>> {
        Some(self)
    }
    fn typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>> {
        Some(self)
    }
}

/// `Tape::new_with_ops(low_precision_test_ops())` の形で使う。
pub(crate) fn low_precision_test_ops() -> Box<dyn BackendOps + Send> {
    Box::new(LowPrecisionTestOps)
}
