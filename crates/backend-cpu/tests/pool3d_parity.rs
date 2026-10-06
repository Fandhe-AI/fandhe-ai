//! `CpuBackendOps::pool3d_max`／`pool3d_avg`（イシュー #2643）の直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::pool3d`。走査規則・数値契約の正）を
//! 呼ぶだけの薄い配線のため、ここでは解析値・非連続（strided）入力・型付きエラー・
//! run-to-run 決定性を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/pool3d_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/pool3d_ops_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::pool3d::Pool3dParams;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn p(k: [usize; 3], s: Option<[usize; 3]>, pad: [usize; 3], d: [usize; 3]) -> Pool3dParams {
    Pool3dParams::new(k, s, pad, d).expect("test fixture: params 有効")
}

#[test]
fn max_returns_values_and_flat_indices() {
    let ops = CpuBackendOps::new();
    let x = t((0..8).map(|i| i as f32).collect(), &[1, 1, 2, 2, 2]);
    let (v, i) = ops
        .pool3d_max(&x, &p([2, 2, 2], None, [0; 3], [1; 3]))
        .unwrap();
    assert_eq!(v.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(v.host_slice().into_owned(), [7.0]);
    assert_eq!(i.host_slice().into_owned(), [7]);
}

#[test]
fn avg_matches_analytic_values_for_both_pad_modes() {
    let ops = CpuBackendOps::new();
    let x = t(vec![2.0, 4.0], &[1, 1, 1, 1, 2]);
    let params = p([1, 1, 2], Some([1; 3]), [0, 0, 1], [1; 3]);
    let a = ops.pool3d_avg(&x, &params, true).unwrap();
    assert_eq!(a.host_slice().into_owned(), [1.0, 3.0, 2.0]);
    let b = ops.pool3d_avg(&x, &params, false).unwrap();
    assert_eq!(b.host_slice().into_owned(), [2.0, 3.0, 4.0]);
}

#[test]
fn strided_transposed_input_is_pooled_in_logical_order() {
    let ops = CpuBackendOps::new();
    // [1,1,2,2,2] の H と W を入れ替えた非連続 view。論理順で窓を取る。
    let base = t((0..8).map(|i| i as f32).collect(), &[1, 1, 2, 2, 2]);
    let tr = base.transpose(3, 4).unwrap();
    let params = p([1, 1, 2], None, [0; 3], [1; 3]);
    let (v, i) = ops.pool3d_max(&tr, &params).unwrap();
    // tr[d,h,w] = base[d,w,h]。W 窓 {0,1} の最大 = base[d,1,h]。
    assert_eq!(v.host_slice().into_owned(), [2.0, 3.0, 6.0, 7.0]);
    assert_eq!(i.host_slice().into_owned(), [1, 3, 5, 7]);
    let a = ops.pool3d_avg(&tr, &params, true).unwrap();
    assert_eq!(a.host_slice().into_owned(), [1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn invalid_shapes_and_dilation_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let x4 = t(vec![0.0; 8], &[1, 2, 2, 2]);
    let params = p([2, 2, 2], None, [0; 3], [1; 3]);
    assert!(matches!(
        ops.pool3d_max(&x4, &params),
        Err(BackendError::ShapeMismatch(_))
    ));
    let x = t(vec![0.0; 8], &[1, 1, 2, 2, 2]);
    let dil = p([1, 1, 2], None, [0; 3], [1, 1, 2]);
    assert!(matches!(
        ops.pool3d_avg(&x, &dil, true),
        Err(BackendError::InvalidArgument(_))
    ));
    // 空間軸 0。
    let z = t(vec![], &[1, 1, 0, 2, 2]);
    assert!(ops.pool3d_max(&z, &params).is_err());
}

#[test]
fn results_are_deterministic_run_to_run() {
    let ops = CpuBackendOps::new();
    let data: Vec<f32> = (0..2 * 3 * 4 * 5 * 4)
        .map(|i| ((i * 37) % 13) as f32 * 0.5)
        .collect();
    let x = t(data, &[2, 3, 4, 5, 4]);
    let params = p([2, 3, 2], Some([1, 1, 1]), [1, 1, 1], [1; 3]);
    let a = ops.pool3d_avg(&x, &params, false).unwrap();
    let b = ops.pool3d_avg(&x, &params, false).unwrap();
    let bits = |t: &Tensor<f32>| {
        t.host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
    let (v1, i1) = ops.pool3d_max(&x, &params).unwrap();
    let (v2, i2) = ops.pool3d_max(&x, &params).unwrap();
    assert_eq!(bits(&v1), bits(&v2));
    assert_eq!(i1.host_slice().into_owned(), i2.host_slice().into_owned());
}
