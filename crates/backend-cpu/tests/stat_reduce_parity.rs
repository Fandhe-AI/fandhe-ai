//! `CpuBackendOps::stat_kthvalue`／`stat_median_dim`／`stat_median_all`／
//! `stat_quantile`／`stat_nansum`／`stat_nanmean`（イシュー #2637）の直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::stat_reduce`。順序規則・
//! 数値契約の正）を呼ぶだけの薄い配線のため、ここでは解析値・非連続（strided）
//! 入力・型付きエラー・run-to-run 決定性を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/stat_reduce_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/stat_reduce_ops_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, QuantileInterpolation, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

#[test]
fn kthvalue_and_median_return_values_and_indices() {
    let ops = CpuBackendOps::new();
    let x = t(vec![4.0, 1.0, 3.0, 2.0], &[4]);
    let (v, i) = ops.stat_kthvalue(&x, 3, 0).unwrap();
    assert_eq!(v.shape(), &[] as &[usize]);
    assert_eq!(v.host_slice().into_owned(), [3.0]);
    assert_eq!(i.host_slice().into_owned(), [2]);
    // 偶数個の下側中央値。
    let (v, i) = ops.stat_median_dim(&x, 0).unwrap();
    assert_eq!(v.host_slice().into_owned(), [2.0]);
    assert_eq!(i.host_slice().into_owned(), [3]);
    assert_eq!(ops.stat_median_all(&x).unwrap().host_slice()[0], 2.0);
}

#[test]
fn quantile_and_nan_reductions_have_analytic_values() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0], &[6]);
    let q = ops
        .stat_quantile(&x, 0.3, None, QuantileInterpolation::Linear)
        .unwrap();
    assert_eq!(q.host_slice().into_owned(), [2.5]);
    let y = t(vec![f32::NAN, 2.0, 4.0, 1.0, f32::NAN, f32::NAN], &[2, 3]);
    assert_eq!(
        ops.stat_nansum(&y, Some(1))
            .unwrap()
            .host_slice()
            .into_owned(),
        [6.0, 1.0]
    );
    let m = ops
        .stat_nanmean(&y, Some(1))
        .unwrap()
        .host_slice()
        .into_owned();
    assert_eq!(m[0], 3.0);
    assert_eq!(m[1], 1.0);
    assert_eq!(ops.stat_nansum(&y, None).unwrap().host_slice()[0], 7.0);
    assert_eq!(ops.stat_nanmean(&y, Some(0)).unwrap().shape(), &[3]);
}

#[test]
fn strided_transposed_input_is_reduced_in_logical_order() {
    let ops = CpuBackendOps::new();
    // [2,3] を転置した [3,2] の dim 0: 論理順の列 {1,4,2}・{5,3,0}。
    let base = t(vec![1.0, 4.0, 2.0, 5.0, 3.0, 0.0], &[2, 3]);
    let tr = base.transpose(0, 1).unwrap();
    let (v, i) = ops.stat_median_dim(&tr, 0).unwrap();
    assert_eq!(v.host_slice().into_owned(), [2.0, 3.0]);
    assert_eq!(i.host_slice().into_owned(), [2, 1]);
    assert_eq!(
        ops.stat_nansum(&tr, Some(0))
            .unwrap()
            .host_slice()
            .into_owned(),
        [7.0, 8.0]
    );
}

#[test]
fn results_are_bit_deterministic() {
    let ops = CpuBackendOps::new();
    let data: Vec<f32> = (0..60)
        .map(|i| ((i * 37 % 29) as f32) * 0.21 - 3.0)
        .collect();
    let x = t(data, &[3, 4, 5]);
    let bits = |t: &Tensor<f32>| {
        t.host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    for dim in 0..3 {
        let a = ops
            .stat_quantile(&x, 0.37, Some(dim), QuantileInterpolation::Linear)
            .unwrap();
        let b = ops
            .stat_quantile(&x, 0.37, Some(dim), QuantileInterpolation::Linear)
            .unwrap();
        assert_eq!(bits(&a), bits(&b));
        let a = ops.stat_nanmean(&x, Some(dim)).unwrap();
        let b = ops.stat_nanmean(&x, Some(dim)).unwrap();
        assert_eq!(bits(&a), bits(&b));
    }
}

#[test]
fn invalid_arguments_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0], &[2]);
    for r in [
        ops.stat_kthvalue(&x, 1, 1).map(|_| ()),
        ops.stat_median_dim(&x, 1).map(|_| ()),
        ops.stat_quantile(&x, 0.5, Some(1), QuantileInterpolation::Linear)
            .map(|_| ()),
        ops.stat_nansum(&x, Some(1)).map(|_| ()),
        ops.stat_nanmean(&x, Some(1)).map(|_| ()),
    ] {
        assert!(matches!(
            r,
            Err(BackendError::ShapeMismatch(ShapeError::AxisOutOfRange {
                axis: 1,
                rank: 1
            }))
        ));
    }
    assert!(matches!(
        ops.stat_kthvalue(&x, 0, 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.stat_kthvalue(&x, 3, 0),
        Err(BackendError::InvalidArgument(_))
    ));
    for q in [-0.5_f32, 1.5, f32::NAN] {
        assert!(matches!(
            ops.stat_quantile(&x, q, None, QuantileInterpolation::Linear),
            Err(BackendError::InvalidArgument(_))
        ));
    }
}

#[test]
fn empty_axis_rules() {
    let ops = CpuBackendOps::new();
    let x = t(vec![], &[0]);
    assert!(matches!(
        ops.stat_median_dim(&x, 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(ops.stat_median_all(&x).unwrap().host_slice()[0].is_nan());
    assert_eq!(ops.stat_nansum(&x, None).unwrap().host_slice()[0], 0.0);
    assert!(ops.stat_nanmean(&x, Some(0)).unwrap().host_slice()[0].is_nan());
}
