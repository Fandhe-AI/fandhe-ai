//! `TypedOps<f64>` の CUDA 実装（イシュー #1703・親 #1650・
//! `docs/backend-dtype-dispatch-design.md` §4.2「最小集合 8 演算」）。
//!
//! # 方針: fail-closed `Unsupported`（driver 非接触）
//!
//! CUDA の f64 GEMM／elementwise／reduction カーネルは性能上の目的が
//! なく本イシューの対象外（設計 §8「f64 GPU カーネル（CUDA SIMT）」・
//! 承認事項 6）であるため、8 演算すべてを `crate::ops::CudaBackendOps`
//! の他メソッド（`gemm`／`add` 等）と同じ `BackendError::Unsupported`
//! で即座に拒否する。`crate::ops::CudaBackendOps::linalg_*`
//! オーバーライド（未実装の線形代数演算）と同じ設計判断であり、
//! **`context_cache::cached_device` 等の driver 呼び出しに一切触れない**
//! （`CudaDevice::new`／`with_driver_call` を経由しないため、driver
//! 不在の CI 環境でも本モジュールの契約テストは実行できる）。
//!
//! 将来 CUDA SIMT による f64 GEMM／elementwise／reduction カーネルを
//! 追加する場合、本モジュールの各メソッド本体を「driver 非接触の
//! `Unsupported`」から実装へ差し替える入口として扱う（`typed_f16.rs`
//! の f32 昇格方式とは異なり、f64 は f32 より広い表現域を持つため
//! 昇格による代替はできない。SIMT カーネル本体の新規実装が必要）。
//!
//! bf16（#1704）・Metal（#1705）は対象外。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{Tensor, TypedOps};

use crate::ops::CudaBackendOps;

fn unsupported(op: &str) -> BackendError {
    BackendError::Unsupported(format!(
        "cuda typed f64 {op}: not implemented (fail-closed; f64 GPU kernels have no \
         performance objective. docs/backend-dtype-dispatch-design.md §8)"
    ))
}

impl TypedOps<f64> for CudaBackendOps {
    fn gemm(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("gemm"))
    }

    fn add(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("add"))
    }

    fn mul(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("mul"))
    }

    fn relu(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("relu"))
    }

    fn exp(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("exp"))
    }

    fn tanh(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("tanh"))
    }

    fn sum(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("sum"))
    }

    fn max(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        Err(unsupported("max"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops() -> CudaBackendOps {
        // ordinal は driver へ一切触れないため任意の値でよい（本モジュール
        // の全メソッドが driver 非接触で `Unsupported` を返すため）。
        CudaBackendOps::new(0)
    }

    fn t(shape: &[usize]) -> Tensor<f64> {
        let numel: usize = shape.iter().product();
        Tensor::new(vec![0.0f64; numel], shape).unwrap()
    }

    #[test]
    fn gemm_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2, 2]);
        let b = t(&[2, 2]);
        let err = TypedOps::<f64>::gemm(&c, &a, &b).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("gemm")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn add_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2]);
        let b = t(&[2]);
        let err = TypedOps::<f64>::add(&c, &a, &b).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("add")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn mul_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2]);
        let b = t(&[2]);
        let err = TypedOps::<f64>::mul(&c, &a, &b).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("mul")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn relu_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2]);
        let err = TypedOps::<f64>::relu(&c, &a).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("relu")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn exp_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2]);
        let err = TypedOps::<f64>::exp(&c, &a).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("exp")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn tanh_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2]);
        let err = TypedOps::<f64>::tanh(&c, &a).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("tanh")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn sum_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2, 2]);
        let err = TypedOps::<f64>::sum(&c, &a, None).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("sum")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn max_is_unsupported_and_mentions_op_name() {
        let c = ops();
        let a = t(&[2, 2]);
        let err = TypedOps::<f64>::max(&c, &a, None).unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("max")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// 8 演算すべてが `Unsupported` を返すことを一括で確認する
    /// （`typed_ops_f64()` accessor 経由。`BackendOps` 側の結線テストは
    /// `crate::ops::tests` に置く）。
    #[test]
    fn all_eight_ops_return_unsupported_via_accessor() {
        use fandhe_ai_tensor_core::BackendOps;
        let c = ops();
        let typed = BackendOps::typed_ops_f64(&c).expect("typed_ops_f64 must be Some");
        let a = t(&[2, 2]);
        let b = t(&[2, 2]);
        assert!(matches!(
            typed.gemm(&a, &b),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            typed.add(&a, &b),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            typed.mul(&a, &b),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(typed.relu(&a), Err(BackendError::Unsupported(_))));
        assert!(matches!(typed.exp(&a), Err(BackendError::Unsupported(_))));
        assert!(matches!(typed.tanh(&a), Err(BackendError::Unsupported(_))));
        assert!(matches!(
            typed.sum(&a, None),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            typed.max(&a, None),
            Err(BackendError::Unsupported(_))
        ));
    }
}
