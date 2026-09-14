//! `TypedOps<f64>` の CUDA 契約テスト（イシュー #1703・親 #1650）。
//!
//! `crates/backend-cuda/src/typed_f64.rs` 本体は 8 演算すべて driver
//! 非接触の `Unsupported` を返す fail-closed 実装であるため、本ファイルの
//! テストは GPU・CUDA driver を一切必要とせず常時 CI 実行可能である
//! （`crate::ops::CudaBackendOps::new` はコンストラクタのみで driver へ
//! 触れない）。

use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

fn zeros(shape: &[usize]) -> Tensor<f64> {
    let numel: usize = shape.iter().product();
    Tensor::new(vec![0.0f64; numel], shape).unwrap()
}

/// `BackendOps::typed_ops_f64()` accessor が `Some` を返す
/// （`crate::ops::CudaBackendOps` が `TypedOps<f64>` を実装し `self` を
/// そのまま返す結線。CPU 側 `typed_ops_accessors_default_to_none` 等と
/// 対になる肯定側の契約テスト）。
#[test]
fn typed_ops_f64_accessor_is_some() {
    let ops = CudaBackendOps::new(0);
    assert!(BackendOps::typed_ops_f64(&ops).is_some());
}

/// 8 演算すべてが driver に触れず `Unsupported` を返すことを、
/// `typed_ops_f64()` accessor 経由で確認する。
#[test]
fn all_eight_ops_are_unsupported_via_accessor() {
    let ops = CudaBackendOps::new(0);
    let typed = BackendOps::typed_ops_f64(&ops).expect("typed_ops_f64 must be Some");
    let a = zeros(&[2, 2]);
    let b = zeros(&[2, 2]);

    for (name, result) in [
        ("gemm", typed.gemm(&a, &b).map(|_| ())),
        ("add", typed.add(&a, &b).map(|_| ())),
        ("mul", typed.mul(&a, &b).map(|_| ())),
        ("relu", typed.relu(&a).map(|_| ())),
        ("exp", typed.exp(&a).map(|_| ())),
        ("tanh", typed.tanh(&a).map(|_| ())),
        ("sum", typed.sum(&a, None).map(|_| ())),
        ("max", typed.max(&a, None).map(|_| ())),
    ] {
        assert!(
            matches!(result, Err(BackendError::Unsupported(_))),
            "{name} は Unsupported を返すはず: {result:?}"
        );
    }
}

/// `TypedOps::<f64>::gemm` を直接呼んだ場合も同様に `Unsupported`
/// （trait メソッドとして呼べることの型検査を兼ねる）。
#[test]
fn gemm_direct_trait_call_is_unsupported() {
    let ops = CudaBackendOps::new(0);
    let a = zeros(&[2, 2]);
    let b = zeros(&[2, 2]);
    let err = TypedOps::<f64>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)));
}
