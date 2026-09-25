//! `nn::Conv3d`（イシュー #2158）の受け入れ条件検証（`tests/nn_conv.rs`
//! の Conv2d 節と同型）。
//!
//! - 構築検査: `in_channels == 0`／`in % groups != 0`／`out % groups !=
//!   0`／`out < groups`（`Conv3d::new` 経由）・`stride/dilation/groups
//!   == 0`（`Conv3dParams::new` 経由）・`from_parameters` の rank／
//!   `weight[1]==0`／bias shape 不一致が全て `Err`。
//! - 決定性: 同 seed 同重み・`bias=false` で `bias().is_none()`。
//! - `Conv3dVars::forward` が `conv3d_ops::conv3d` 直呼びと bit 完全
//!   一致（同一 tape 上）。
//! - `Module::forward`（tape 経路）と `forward_host`（tape 不要経路）が
//!   bit 完全一致。
//! - `named_parameters` の順序（weight → bias）・`set_parameter` の
//!   shape 不一致／未知名／bias なし層拒否。
//! - `Module::as_conv3d`／`as_conv3d_mut` が `Some(&self)` を返す。

mod common;

use fandhe_ai_autodiff::conv3d_ops::conv3d;
use fandhe_ai_autodiff::nn::{Conv3d, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor
        .contiguous()
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

/// `Result<T, AutodiffError>::unwrap_err()` の代替（`Conv3d` は `Debug`
/// を実装しないため。`tests/nn_conv.rs::err_of` と同型）。
fn err_of<T>(result: Result<T, AutodiffError>) -> AutodiffError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected Err"),
    }
}

// --- 1. 構築検査 ---

#[test]
fn conv3d_new_rejects_in_channels_zero() {
    let err = err_of(Conv3d::new(
        0,
        4,
        [3, 3, 3],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv3d_new_rejects_in_channels_not_divisible_by_groups() {
    let err = err_of(Conv3d::new(
        3,
        4,
        [3, 3, 3],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        2,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv3d_new_rejects_out_channels_not_divisible_by_groups() {
    let err = err_of(Conv3d::new(
        4,
        3,
        [3, 3, 3],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        2,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv3d_new_rejects_out_channels_less_than_groups() {
    let err = err_of(Conv3d::new(
        4,
        2,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        4,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv3d_new_rejects_zero_stride_dilation_groups() {
    let err = err_of(Conv3d::new(
        2,
        2,
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        true,
        1,
    ));
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn conv3d_from_parameters_rejects_wrong_rank() {
    let w = t(vec![0.0; 4], &[1, 1, 2, 2]);
    let err = err_of(Conv3d::from_parameters(
        w,
        None,
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
    ));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn conv3d_from_parameters_rejects_zero_cin_g() {
    let w = t(vec![], &[1, 0, 1, 1, 1]);
    let err = err_of(Conv3d::from_parameters(
        w,
        None,
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv3d_from_parameters_rejects_bias_shape_mismatch() {
    let w = t(vec![0.0; 16], &[2, 1, 2, 2, 2]);
    let b = t(vec![0.0; 3], &[3]);
    let err = err_of(Conv3d::from_parameters(
        w,
        Some(b),
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
    ));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

// --- 2. 決定性 ---

#[test]
fn conv3d_new_without_bias_has_no_bias() {
    let conv = Conv3d::new(
        2,
        2,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        false,
        1,
    )
    .unwrap();
    assert!(conv.bias().is_none());
}

#[test]
fn conv3d_new_is_deterministic_and_seed_dependent() {
    let a = Conv3d::new(
        2,
        3,
        [2, 2, 2],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        true,
        42,
    )
    .unwrap();
    let b = Conv3d::new(
        2,
        3,
        [2, 2, 2],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        true,
        42,
    )
    .unwrap();
    let c = Conv3d::new(
        2,
        3,
        [2, 2, 2],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        true,
        43,
    )
    .unwrap();
    assert_eq!(dense(a.weight()), dense(b.weight()));
    assert_ne!(dense(a.weight()), dense(c.weight()));
}

// --- 3. Conv3dVars::forward ≡ conv3d_ops::conv3d 直呼び ---

#[test]
fn conv3d_vars_forward_matches_direct_call_bit_exact() {
    let conv = Conv3d::new(2, 3, [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, true, 3).unwrap();
    let x = t(
        (0..2 * 2 * 3 * 3 * 3)
            .map(|i| (i as f32) * 0.01 - 0.2)
            .collect(),
        &[2, 2, 3, 3, 3],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let vars = conv.bind(&tape);
    let via_vars = vars.forward(&xv).unwrap().to_tensor();

    let wv = tape.var(conv.weight());
    let bv = conv.bias().map(|b| tape.var(b));
    let via_direct = conv3d(&xv, &wv, bv.as_ref(), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1)
        .unwrap()
        .to_tensor();

    assert_eq!(dense(&via_vars), dense(&via_direct));
}

// --- 4. Module::forward ≡ forward_host bit 一致 ---

#[test]
fn conv3d_module_forward_matches_forward_host_bit_exact() {
    let conv = Conv3d::new(2, 3, [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], 1, true, 5).unwrap();
    let x = t(
        (0..2 * 2 * 4 * 4 * 4)
            .map(|i| (i as f32) * 0.02 - 0.5)
            .collect(),
        &[2, 2, 4, 4, 4],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = Module::forward(&conv, &tape, &xv).unwrap().to_tensor();

    let ops = common::naive_ops();
    let via_host = conv.forward_host(&*ops, &x).unwrap();

    assert_eq!(dense(&via_module), dense(&via_host));
}

// --- 5. named_parameters／set_parameter ---

#[test]
fn conv3d_named_parameters_order_is_weight_then_bias() {
    let conv = Conv3d::new(2, 2, [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, true, 1).unwrap();
    let named = conv.named_parameters();
    assert_eq!(named.len(), 2);
    assert_eq!(named[0].0, "weight");
    assert_eq!(named[1].0, "bias");
}

#[test]
fn conv3d_named_parameters_excludes_bias_when_none() {
    let conv = Conv3d::new(
        2,
        2,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        false,
        1,
    )
    .unwrap();
    let named = conv.named_parameters();
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].0, "weight");
}

#[test]
fn conv3d_set_parameter_rejects_shape_mismatch() {
    let mut conv =
        Conv3d::new(2, 2, [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, true, 1).unwrap();
    let bad = t(vec![0.0; 3], &[3]);
    let err = err_of(Module::set_parameter(&mut conv, "weight", bad));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn conv3d_set_parameter_rejects_unknown_name() {
    let mut conv =
        Conv3d::new(2, 2, [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, true, 1).unwrap();
    let v = t(vec![0.0], &[1]);
    let err = err_of(Module::set_parameter(&mut conv, "unknown", v));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv3d_set_parameter_rejects_bias_when_layer_has_no_bias() {
    let mut conv = Conv3d::new(
        2,
        2,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        false,
        1,
    )
    .unwrap();
    let v = t(vec![0.0; 2], &[2]);
    let err = err_of(Module::set_parameter(&mut conv, "bias", v));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- 6. as_conv3d／as_conv3d_mut ---

#[test]
fn as_conv3d_and_as_conv3d_mut_return_some_for_conv3d_layer() {
    let mut conv =
        Conv3d::new(2, 2, [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, true, 1).unwrap();
    assert!(Module::as_conv3d(&conv).is_some());
    assert!(Module::as_conv3d_mut(&mut conv).is_some());
}

// --- 7. requires_grad の凍結フラグ ---

#[test]
fn conv3d_set_requires_grad_round_trips() {
    let mut conv =
        Conv3d::new(2, 2, [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, true, 1).unwrap();
    assert!(Module::requires_grad(&conv));
    Module::set_requires_grad(&mut conv, false).unwrap();
    assert!(!Module::requires_grad(&conv));
}
