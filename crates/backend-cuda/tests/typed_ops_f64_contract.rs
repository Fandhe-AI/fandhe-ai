//! `TypedOps<f64>` の CUDA 契約テスト（イシュー #2060・親 #1650）。
//!
//! #2060 で `crates/backend-cuda/src/typed_f64.rs` はネイティブ CUDA
//! カーネル実装へ差し替わった（旧: 8 演算すべて driver 非接触の
//! `Unsupported`）。本ファイルは新しい契約——
//! (a) shape 不整合は driver に一切触れる前に `ShapeMismatch` を返す、
//! (b) 有効な shape でも driver 不在環境（本 CI）では
//! `BackendError::CudaUnavailable`（`KernelLaunchFailed` にはならない。
//! shape 検証を通過した呼び出しが最初に触れる driver 呼び出しは
//! `context_cache::cached_typed_f64` 取得であり、これは
//! `CudaUnavailable` へ写像される）を返す——を検証する。GPU・CUDA
//! driver を一切必要とせず常時 CI 実行可能である（`typed_ops_f16_parity.rs`
//! の非 `#[ignore]` 部分と同型の構成）。実機での数値正しさは
//! `typed_ops_f64_parity.rs` の `#[ignore]` テストへ引き継ぐ。

use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

fn zeros(shape: &[usize]) -> Tensor<f64> {
    let numel: usize = shape.iter().product();
    Tensor::new(vec![0.0f64; numel], shape).unwrap()
}

/// `BackendOps::typed_ops_f64()` accessor が `Some` を返す
/// （`crate::ops::CudaBackendOps` が `TypedOps<f64>` を実装し `self` を
/// そのまま返す結線）。
#[test]
fn typed_ops_f64_accessor_is_some() {
    let ops = CudaBackendOps::new(0);
    assert!(BackendOps::typed_ops_f64(&ops).is_some());
}

/// `gemm` の shape 不一致は driver に触れる前に `ShapeMismatch` を返す
/// （ordinal 0 が driver 不在環境でも本テストは成立する）。
#[test]
fn gemm_rejects_shape_mismatch_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let a = zeros(&[1, 3]);
    let b = zeros(&[2, 1]);
    let err = TypedOps::<f64>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `add`／`mul` の shape 不一致（ブロードキャスト不能）は driver に
/// 触れる前に `ShapeMismatch` を返す。
#[test]
fn add_mul_reject_shape_mismatch_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let a = zeros(&[3]);
    let b = zeros(&[2]);
    let add_err = TypedOps::<f64>::add(&ops, &a, &b).unwrap_err();
    assert!(matches!(add_err, BackendError::ShapeMismatch(_)));
    let mul_err = TypedOps::<f64>::mul(&ops, &a, &b).unwrap_err();
    assert!(matches!(mul_err, BackendError::ShapeMismatch(_)));
}

/// `sum`／`max` の範囲外 `dim` は driver に触れる前に `ShapeMismatch` を
/// 返す。
#[test]
fn sum_max_reject_out_of_range_dim_before_touching_driver() {
    let ops = CudaBackendOps::new(0);
    let a = zeros(&[2]);
    let sum_err = TypedOps::<f64>::sum(&ops, &a, Some(5)).unwrap_err();
    assert!(matches!(sum_err, BackendError::ShapeMismatch(_)));
    let max_err = TypedOps::<f64>::max(&ops, &a, Some(5)).unwrap_err();
    assert!(matches!(max_err, BackendError::ShapeMismatch(_)));
}

/// 有効な shape の 8 演算すべてが、driver 不在環境（本 CI）では
/// `CudaUnavailable` を返す（`Unsupported`／`KernelLaunchFailed` には
/// ならない。`typed_ops_f64()` accessor 経由）。
#[test]
fn all_eight_ops_return_cuda_unavailable_via_accessor_in_ci() {
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
            matches!(result, Err(BackendError::CudaUnavailable(_))),
            "{name} は CudaUnavailable を返すはず: {result:?}"
        );
    }
}

/// `TypedOps::<f64>::gemm` を直接呼んだ場合も同様に `CudaUnavailable`
/// （trait メソッドとして呼べることの型検査を兼ねる）。
#[test]
fn gemm_direct_trait_call_returns_cuda_unavailable_in_ci() {
    let ops = CudaBackendOps::new(0);
    let a = zeros(&[2, 2]);
    let b = zeros(&[2, 2]);
    let err = TypedOps::<f64>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::CudaUnavailable(_)));
}
