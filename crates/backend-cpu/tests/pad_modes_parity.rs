//! `CpuBackendOps::pad_modes_forward`（イシュー #2642）の直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::pad_modes`。添字写像・
//! 数値契約の正）を呼ぶだけの薄い配線のため、ここでは解析値・非連続（strided）
//! 入力・型付きエラー・run-to-run bit 一致を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/pad_modes_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/pad_modes_ops_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, PadMode, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.host_slice().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn three_modes_match_analytic_values() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let cases = [
        (PadMode::Reflect, vec![3., 2., 1., 2., 3., 4., 3., 2., 1.]),
        (PadMode::Replicate, vec![1., 1., 1., 2., 3., 4., 4., 4., 4.]),
        (PadMode::Circular, vec![3., 4., 1., 2., 3., 4., 1., 2., 3.]),
    ];
    for (mode, want) in cases {
        let y = ops.pad_modes_forward(&x, &[(2, 3)], mode).unwrap();
        assert_eq!(y.shape(), &[9]);
        assert_eq!(y.host_slice().into_owned(), want, "{mode:?}");
    }
}

#[test]
fn strided_transposed_input_is_padded_in_logical_order() {
    let ops = CpuBackendOps::new();
    let base = t(vec![1.0, 4.0, 2.0, 5.0, 3.0, 0.0], &[2, 3]);
    // tr = [[1,5],[4,3],[2,0]]
    let tr = base.transpose(0, 1).unwrap();
    let y = ops
        .pad_modes_forward(&tr, &[(0, 0), (1, 1)], PadMode::Replicate)
        .unwrap();
    assert_eq!(y.shape(), &[3, 4]);
    assert_eq!(
        y.host_slice().into_owned(),
        [1., 1., 5., 5., 4., 4., 3., 3., 2., 2., 0., 0.]
    );
}

#[test]
fn invalid_arguments_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0, 3.0], &[3]);
    assert!(matches!(
        ops.pad_modes_forward(&x, &[(3, 0)], PadMode::Reflect),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.pad_modes_forward(&x, &[(4, 0)], PadMode::Circular),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.pad_modes_forward(&x, &[(1, 1), (1, 1)], PadMode::Replicate),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn forward_is_bit_identical_across_runs_and_preserves_nonfinite() {
    let ops = CpuBackendOps::new();
    let x = t(vec![f32::NAN, -0.0, f32::NEG_INFINITY, 1.5], &[2, 2]);
    let a = ops
        .pad_modes_forward(&x, &[(1, 1), (1, 1)], PadMode::Circular)
        .unwrap();
    let b = ops
        .pad_modes_forward(&x, &[(1, 1), (1, 1)], PadMode::Circular)
        .unwrap();
    assert_eq!(bits(&a), bits(&b));
    assert!(bits(&a).contains(&(-0.0f32).to_bits()));
}
