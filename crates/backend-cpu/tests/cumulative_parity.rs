//! `CpuBackendOps::scan_cummax`／`scan_cummin`／`scan_logcumsumexp`（イシュー
//! #2636）の直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::cumulative`。走査規則・
//! 数値契約の正）を呼ぶだけの薄い配線のため、ここでは解析値・非連続（strided）
//! 入力・型付きエラー・run-to-run 決定性を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/cumulative_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/cumulative_ops_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

#[test]
fn cummax_and_cummin_return_values_and_indices() {
    let ops = CpuBackendOps::new();
    let x = t(vec![3.0, 1.0, 3.0, 2.0, 5.0], &[5]);
    let (v, i) = ops.scan_cummax(&x, 0).unwrap();
    assert_eq!(v.host_slice().into_owned(), [3.0, 3.0, 3.0, 3.0, 5.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 0, 2, 2, 4]);
    let (v, i) = ops.scan_cummin(&x, 0).unwrap();
    assert_eq!(v.host_slice().into_owned(), [3.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 1, 1, 1, 1]);
}

#[test]
fn strided_transposed_input_is_scanned_in_logical_order() {
    let ops = CpuBackendOps::new();
    // [2,3] を転置した [3,2] の dim 0 走査: 論理順（転置後の行方向）で走査する。
    let base = t(vec![1.0, 4.0, 2.0, 5.0, 3.0, 0.0], &[2, 3]);
    let tr = base.transpose(0, 1).unwrap();
    // tr = [[1,5],[4,3],[2,0]]
    let (v, i) = ops.scan_cummax(&tr, 0).unwrap();
    assert_eq!(v.shape(), &[3, 2]);
    assert_eq!(v.host_slice().into_owned(), [1.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 0, 1, 0, 1, 0]);
}

#[test]
fn logcumsumexp_matches_analytic_values_and_is_deterministic() {
    let ops = CpuBackendOps::new();
    let x = t(vec![0.0, 0.0, 0.0, 0.0], &[4]);
    let y = ops.scan_logcumsumexp(&x, 0).unwrap();
    for (k, &v) in y.host_slice().iter().enumerate() {
        let want = ((k + 1) as f64).ln();
        assert!((f64::from(v) - want).abs() < 1e-6, "k={k}: {v} vs {want}");
    }
    let again = ops.scan_logcumsumexp(&x, 0).unwrap();
    let bits = |t: &Tensor<f32>| {
        t.host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&y), bits(&again));
}

#[test]
fn invalid_dim_is_a_typed_error() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0], &[2]);
    for r in [
        ops.scan_cummax(&x, 1).map(|_| ()),
        ops.scan_cummin(&x, 1).map(|_| ()),
        ops.scan_logcumsumexp(&x, 1).map(|_| ()),
    ] {
        assert!(matches!(
            r,
            Err(BackendError::ShapeMismatch(ShapeError::AxisOutOfRange {
                axis: 1,
                rank: 1
            }))
        ));
    }
}

#[test]
fn empty_axis_returns_empty_outputs() {
    let ops = CpuBackendOps::new();
    let x = t(vec![], &[0]);
    let (v, i) = ops.scan_cummax(&x, 0).unwrap();
    assert_eq!(v.shape(), &[0]);
    assert_eq!(i.shape(), &[0]);
    assert_eq!(ops.scan_logcumsumexp(&x, 0).unwrap().shape(), &[0]);
}
