//! `CpuBackendOps::indexed_scatter_reduce`（イシュー #2641）の直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::indexed_update`。走査規則・
//! 数値契約の正）を呼ぶだけの薄い配線のため、ここでは解析値・非連続（strided）
//! 入力・型付きエラー・run-to-run 決定性を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/indexed_update_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/indexed_update_ops_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScatterReduceMode, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

#[test]
fn analytic_values_for_each_mode() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0, 3.0], &[3]);
    let idx = ti(vec![0, 0, 2], &[3]);
    let s = t(vec![4.0, 5.0, 6.0], &[3]);
    let run = |m, inc| {
        ops.indexed_scatter_reduce(&x, 0, &idx, &s, m, inc)
            .unwrap()
            .host_slice()
            .into_owned()
    };
    assert_eq!(run(ScatterReduceMode::Sum, true), [10.0, 2.0, 9.0]);
    assert_eq!(run(ScatterReduceMode::Prod, false), [20.0, 2.0, 6.0]);
    assert_eq!(run(ScatterReduceMode::Mean, false), [4.5, 2.0, 6.0]);
    assert_eq!(run(ScatterReduceMode::Amax, true), [5.0, 2.0, 6.0]);
    assert_eq!(run(ScatterReduceMode::Amin, true), [1.0, 2.0, 3.0]);
}

#[test]
fn strided_transposed_input_is_read_in_logical_order() {
    let ops = CpuBackendOps::new();
    let base = t(vec![1.0, 4.0, 2.0, 5.0, 3.0, 0.0], &[2, 3]);
    let tr = base.transpose(0, 1).unwrap(); // [[1,5],[4,3],[2,0]]
    let idx = ti(vec![0, 2], &[1, 2]);
    let s = t(vec![10.0, 20.0], &[1, 2]);
    let out = ops
        .indexed_scatter_reduce(&tr, 0, &idx, &s, ScatterReduceMode::Sum, true)
        .unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(
        out.host_slice().into_owned(),
        [11.0, 5.0, 4.0, 3.0, 2.0, 20.0]
    );
}

#[test]
fn invalid_inputs_return_typed_errors() {
    let ops = CpuBackendOps::new();
    let x = t(vec![0.0; 3], &[3]);
    let s = t(vec![1.0], &[1]);
    for bad in [3, -1] {
        let r = ops.indexed_scatter_reduce(
            &x,
            0,
            &ti(vec![bad], &[1]),
            &s,
            ScatterReduceMode::Sum,
            true,
        );
        assert!(matches!(r, Err(BackendError::ShapeMismatch(_))), "{bad}");
    }
    let r = ops.indexed_scatter_reduce(&x, 1, &ti(vec![0], &[1]), &s, ScatterReduceMode::Sum, true);
    assert!(matches!(r, Err(BackendError::ShapeMismatch(_))));
}

#[test]
fn repeated_calls_are_bit_identical() {
    let ops = CpuBackendOps::new();
    let x = t((0..8).map(|i| i as f32 * 0.37 - 1.0).collect(), &[8]);
    let idx = ti(vec![0, 1, 0, 1, 7, 7], &[6]);
    let s = t(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[6]);
    let bits = || -> Vec<u32> {
        ops.indexed_scatter_reduce(&x, 0, &idx, &s, ScatterReduceMode::Prod, true)
            .unwrap()
            .host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(bits(), bits());
}
