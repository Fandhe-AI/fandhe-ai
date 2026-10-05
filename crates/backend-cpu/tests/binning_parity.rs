//! `CpuBackendOps::binning_histc`／`binning_bincount`／`binning_bincount_weighted`／
//! `binning_searchsorted`（イシュー #2638）の直接呼び出しテスト。
//!
//! CPU 実装は共有ホストカーネル（`fandhe_ai_tensor_core::binning`。範囲決定・添字算術・
//! 探索手順の正）を呼ぶだけの薄い配線のため、ここでは解析値・非連続（strided）入力・
//! 型付きエラー・空入力・run-to-run 決定性を固定する。PyTorch fixture との突合は
//! `crates/autodiff/tests/binning_parity.rs`、`Tape` 経由の CPU 対 naive 比較は
//! `crates/facade/tests/binning_ops_backend_parity.rs` が担当する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

#[test]
fn histc_has_analytic_values_and_flattens_input() {
    let ops = CpuBackendOps::new();
    let x = t(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5], &[2, 3]);
    let h = ops.binning_histc(&x, 2, 0.0, 2.5).unwrap();
    assert_eq!(h.shape(), &[2]);
    assert_eq!(h.host_slice().into_owned(), [3.0, 3.0]);
    // 既定範囲（min == max == 0）は入力の最小・最大。
    let h = ops.binning_histc(&x, 5, 0.0, 0.0).unwrap();
    assert_eq!(h.host_slice().into_owned(), [1.0, 1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn bincount_and_weighted_have_analytic_values() {
    let ops = CpuBackendOps::new();
    let i = ti(vec![0, 1, 1, 3], &[4]);
    let c = ops.binning_bincount(&i, 0).unwrap();
    assert_eq!(c.shape(), &[4]);
    assert_eq!(c.host_slice().into_owned(), [1, 2, 0, 1]);
    let w = t(vec![0.5, 1.0, 2.0, -1.0], &[4]);
    let s = ops.binning_bincount_weighted(&i, &w, 5).unwrap();
    assert_eq!(s.host_slice().into_owned(), [0.5, 3.0, 0.0, -1.0, 0.0]);
    let e = ops.binning_bincount(&ti(vec![], &[0]), 3).unwrap();
    assert_eq!(e.host_slice().into_owned(), [0, 0, 0]);
}

#[test]
fn searchsorted_uses_lower_and_upper_bound_and_batches() {
    let ops = CpuBackendOps::new();
    let seq = t(vec![1.0, 2.0, 2.0, 5.0], &[4]);
    let v = t(vec![0.0, 2.0, 3.0, 9.0], &[2, 2]);
    let l = ops.binning_searchsorted(&seq, &v, false).unwrap();
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(l.host_slice().into_owned(), [0, 1, 3, 4]);
    let r = ops.binning_searchsorted(&seq, &v, true).unwrap();
    assert_eq!(r.host_slice().into_owned(), [0, 3, 3, 4]);
    let bseq = t(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let bv = t(vec![3.0, 0.0, 4.0, 5.0], &[2, 2]);
    let b = ops.binning_searchsorted(&bseq, &bv, false).unwrap();
    assert_eq!(b.host_slice().into_owned(), [1, 0, 1, 2]);
}

#[test]
fn strided_transposed_inputs_are_read_in_logical_order() {
    let ops = CpuBackendOps::new();
    let base = t(vec![0.0, 2.0, 1.0, 3.0], &[2, 2]);
    let tr = base.transpose(0, 1).unwrap(); // 論理順 [0, 1, 2, 3]
    let h = ops.binning_histc(&tr, 2, 0.0, 3.0).unwrap();
    assert_eq!(h.host_slice().into_owned(), [2.0, 2.0]);
    let bseq = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
        .transpose(0, 1)
        .unwrap(); // [[1,3],[2,4]]
    let v = t(vec![2.0, 3.0, 2.0, 5.0], &[2, 2]);
    let r = ops.binning_searchsorted(&bseq, &v, false).unwrap();
    assert_eq!(r.host_slice().into_owned(), [1, 1, 0, 2]);
}

#[test]
fn invalid_arguments_are_typed_errors() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0], &[2]);
    assert!(matches!(
        ops.binning_histc(&x, 0, 0.0, 1.0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_histc(&x, 2, 2.0, 1.0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_histc(&x, 2, 0.0, f32::INFINITY),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_histc(&x, 2, f32::NAN, 1.0),
        Err(BackendError::InvalidArgument(_))
    ));
    // 空入力は重みの rank を見ず minlength 個の零を返す。
    let z = ops
        .binning_bincount_weighted(&ti(vec![], &[0]), &t(vec![1.0], &[1, 1]), 3)
        .unwrap();
    assert_eq!(z.shape(), [3]);
    assert!(matches!(
        ops.binning_histc(&x, usize::MAX, 0.0, 1.0),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        ops.binning_bincount(&ti(vec![0, -1], &[2]), 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_bincount(&ti(vec![0, 1, 1, 2], &[2, 2]), 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_bincount_weighted(&ti(vec![0, 1], &[2]), &t(vec![1.0], &[1]), 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_searchsorted(&t(vec![1.0], &[]), &x, false),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_searchsorted(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), &x, false),
        Err(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn empty_inputs_follow_the_documented_rules() {
    let ops = CpuBackendOps::new();
    let e = t(vec![], &[0]);
    assert_eq!(
        ops.binning_histc(&e, 3, 0.0, 0.0)
            .unwrap()
            .host_slice()
            .into_owned(),
        [0.0, 0.0, 0.0]
    );
    let seq = t(vec![], &[0]);
    let v = t(vec![1.0, 2.0], &[2]);
    assert_eq!(
        ops.binning_searchsorted(&seq, &v, true)
            .unwrap()
            .host_slice()
            .into_owned(),
        [0, 0]
    );
    let r = ops.binning_searchsorted(&v, &e, false).unwrap();
    assert_eq!(r.shape(), &[0]);
}

#[test]
fn results_are_bit_deterministic() {
    let ops = CpuBackendOps::new();
    let data: Vec<f32> = (0..500)
        .map(|i| ((i * 37 % 29) as f32) * 0.21 - 3.0)
        .collect();
    let x = t(data, &[500]);
    let a = ops.binning_histc(&x, 17, -2.0, 2.0).unwrap();
    let b = ops.binning_histc(&x, 17, -2.0, 2.0).unwrap();
    assert_eq!(a.host_slice().into_owned(), b.host_slice().into_owned());
    let idx = ti((0..500).map(|i| i % 13).collect(), &[500]);
    let w1 = ops.binning_bincount_weighted(&idx, &x, 0).unwrap();
    let w2 = ops.binning_bincount_weighted(&idx, &x, 0).unwrap();
    let bits = |t: &Tensor<f32>| {
        t.host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&w1), bits(&w2));
}

/// 巨大な broadcast view（形状だけ大きく実体化すれば TB 級）を渡しても、検証違反は実体化より
/// 前に型付きエラーで返る（`contiguous` の確保で panic／abort しない。PR #2782 レビュー）。
/// 検証が実体化の後ろに回ると本テストはプロセスごと落ちる（fail-closed の回帰固定）。
#[test]
fn violations_are_rejected_before_materializing_huge_views() {
    let ops = CpuBackendOps::new();
    let huge = 1usize << 40;
    let big_f = t(vec![1.0], &[1]).broadcast_to(&[huge]).unwrap();
    let small_i = ti(vec![0, 1], &[2]);

    // 重みの長さ不一致（rank 1）・rank 2 の巨大 view。
    assert!(matches!(
        ops.binning_bincount_weighted(&small_i, &big_f, 0),
        Err(BackendError::InvalidArgument(_))
    ));
    let big_f2 = t(vec![1.0], &[1, 1]).broadcast_to(&[2, huge]).unwrap();
    assert!(matches!(
        ops.binning_bincount_weighted(&small_i, &big_f2, 0),
        Err(BackendError::InvalidArgument(_))
    ));
    // 入力側の rank 違反（rank 2 の巨大 view）。
    let big_i2 = ti(vec![1], &[1, 1]).broadcast_to(&[2, huge]).unwrap();
    assert!(matches!(
        ops.binning_bincount(&big_i2, 0),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        ops.binning_bincount_weighted(&big_i2, &t(vec![1.0], &[1]), 0),
        Err(BackendError::InvalidArgument(_))
    ));
    // histc: bins == 0 は巨大入力の実体化より前に拒否する。
    assert!(matches!(
        ops.binning_histc(&big_f, 0, 0.0, 1.0),
        Err(BackendError::InvalidArgument(_))
    ));
    // searchsorted: rank 不一致（seq は 2-d・values は 1-d の巨大 view）。
    let seq = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(matches!(
        ops.binning_searchsorted(&seq, &big_f, false),
        Err(BackendError::InvalidArgument(_))
    ));
}
