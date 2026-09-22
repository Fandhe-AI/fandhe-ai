//! `nn::Conv2d`／`nn::Conv1d`（イシュー #1770・親 #1645）の受け入れ条件
//! 検証。
//!
//! - 構築検査: `in_channels == 0`／`in % groups != 0`／`out % groups !=
//!   0`／`out < groups`（`Conv2d::new` 経由）・`stride/dilation/groups
//!   == 0`（`Conv2dParams::new` 経由）・`from_parameters` の rank／
//!   `weight[1]==0`／bias shape 不一致が全て `Err`（variant を
//!   `matches!` で固定）。
//! - 決定性: 同 seed 同重み・異 seed 相違・値域 `|w| <= 1/√fan_in`・
//!   `bias=false` で `bias().is_none()`。
//! - `Conv2dVars::forward`／`Conv1dVars::forward` が `Var::conv2d`／
//!   `Var::conv1d` 直呼びと bit 完全一致（同一 tape 上）。
//! - `Module::forward`（tape 経路）と `forward_host`（tape 不要経路）が
//!   bit 完全一致（`conv2d_with_fallback` の同一関数呼び出しにより
//!   構造的に成立することの回帰）。
//! - `Conv1d` ≡ 手動 reshape `Conv2d`（forward／forward_host）bit 一致。
//! - `named_parameters` の順序（weight → bias）・`set_parameter` の
//!   shape 不一致／未知名／bias なし層拒否。
//! - 数値微分（中央差分）との突合（`tests/conv2d.rs::numeric_conv2d_grad`
//!   と同じ判定閾値。新規の許容誤差緩和はしない）。

mod common;

use fandhe_ai_autodiff::nn::{Conv1d, Conv2d, ConvTranspose2d, Module};
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

/// `Result<T, AutodiffError>::unwrap_err()` の代替。`Conv2d`／`Conv1d`
/// は `Debug` を実装しないため `unwrap_err()`（`Ok` 側にも `Debug` を
/// 要求する）は使えない（`compat/sequential.rs` テストの `Sequential`
/// と同じ理由）。
fn err_of<T>(result: Result<T, AutodiffError>) -> AutodiffError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected Err"),
    }
}

// --- 1. 構築検査 ---

#[test]
fn conv2d_new_rejects_in_channels_zero() {
    let err = err_of(Conv2d::new(
        0,
        4,
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
        1,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv2d_new_rejects_in_channels_not_divisible_by_groups() {
    let err = err_of(Conv2d::new(
        3,
        4,
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
        2,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv2d_new_rejects_out_channels_not_divisible_by_groups() {
    let err = err_of(Conv2d::new(
        4,
        3,
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
        2,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv2d_new_rejects_out_channels_less_than_groups() {
    let err = err_of(Conv2d::new(
        4,
        2,
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        4,
        true,
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv2d_new_rejects_zero_stride_dilation_groups() {
    let base = |stride: [usize; 2], dilation: [usize; 2], groups: usize| {
        Conv2d::new(2, 2, [3, 3], stride, [0, 0], dilation, groups, true, 1)
    };
    assert!(matches!(
        err_of(base([0, 1], [1, 1], 1)),
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        err_of(base([1, 1], [0, 1], 1)),
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        err_of(base([1, 1], [1, 1], 0)),
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn conv2d_from_parameters_rejects_wrong_rank() {
    let w = t(vec![0.0; 6], &[2, 3]);
    let err = err_of(Conv2d::from_parameters(w, None, [1, 1], [0, 0], [1, 1], 1));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn conv2d_from_parameters_rejects_zero_cin_g() {
    let w = t(vec![], &[2, 0, 3, 3]);
    let err = err_of(Conv2d::from_parameters(w, None, [1, 1], [0, 0], [1, 1], 1));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv2d_from_parameters_rejects_bias_shape_mismatch() {
    let w = t(vec![0.1; 2 * 3 * 3], &[2, 1, 3, 3]);
    let bad_bias = t(vec![0.0; 3], &[3]);
    let err = err_of(Conv2d::from_parameters(
        w,
        Some(bad_bias),
        [1, 1],
        [0, 0],
        [1, 1],
        1,
    ));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn conv1d_new_rejects_in_channels_zero() {
    let err = err_of(Conv1d::new(0, 4, 3, 1, 0, 1, 1, true, 1));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv1d_from_parameters_rejects_wrong_rank() {
    let w = t(vec![0.0; 6], &[2, 3]);
    let err = err_of(Conv1d::from_parameters(w, None, 1, 0, 1, 1));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

/// codex-review P1 指摘（イシュー #1770 PR #1880）の回帰: `checked_mul`
/// のみでは `usize` オーバーフローしか検査できず、`Vec<f32>` の
/// `isize::MAX` バイト制限を超える確保は `uniform_init` 内の
/// `collect()` が capacity overflow で panic していた。`weight_numel =
/// out_channels * cin_g * kh * kw = 1 * (1<<61) * 1 * 1 = 1<<61` は
/// `checked_mul` を素通りするが、f32 4 バイト換算で `1<<63` バイトと
/// なり 64-bit の `isize::MAX`（`2^63 - 1`）を超える。本番経路 panic
/// 禁止（`.claude/rules/coding-rust.md`）のため `Err` を返す契約を
/// 固定する（panic せず `Err` が返ること自体が検証対象）。
#[test]
fn conv2d_new_rejects_weight_allocation_exceeding_isize_max() {
    let err = err_of(Conv2d::new(
        1usize << 61,
        1,
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        1,
        false,
        0,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

/// [`conv2d_new_rejects_weight_allocation_exceeding_isize_max`] の
/// `Conv1d` 版。
#[test]
fn conv1d_new_rejects_weight_allocation_exceeding_isize_max() {
    let err = err_of(Conv1d::new(1usize << 61, 1, 1, 1, 0, 1, 1, false, 0));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- 2. 決定性・初期化 ---

#[test]
fn conv2d_new_is_deterministic_and_seed_dependent() {
    let a = Conv2d::new(3, 4, [3, 3], [1, 1], [0, 0], [1, 1], 1, true, 7).unwrap();
    let b = Conv2d::new(3, 4, [3, 3], [1, 1], [0, 0], [1, 1], 1, true, 7).unwrap();
    let c = Conv2d::new(3, 4, [3, 3], [1, 1], [0, 0], [1, 1], 1, true, 8).unwrap();

    assert_eq!(dense(a.weight()), dense(b.weight()));
    assert_ne!(dense(a.weight()), dense(c.weight()));

    let fan_in = 3 * 3 * 3;
    let bound = 1.0f32 / (fan_in as f32).sqrt();
    for &v in dense(a.weight()).iter() {
        assert!(v.abs() <= bound, "|{v}| <= {bound}");
    }
}

#[test]
fn conv2d_new_without_bias_has_no_bias() {
    let conv = Conv2d::new(2, 2, [1, 1], [1, 1], [0, 0], [1, 1], 1, false, 1).unwrap();
    assert!(conv.bias().is_none());
}

// --- 3. Conv2dVars::forward ≡ Var::conv2d 直呼び ---

#[test]
fn conv2d_vars_forward_matches_var_conv2d_bit_exact() {
    let conv = Conv2d::new(2, 3, [2, 2], [1, 1], [0, 0], [1, 1], 1, true, 3).unwrap();
    let x = t(
        (0..2 * 2 * 4 * 4)
            .map(|i| (i as f32) * 0.01 - 0.2)
            .collect(),
        &[2, 2, 4, 4],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let vars = conv.bind(&tape);
    let via_vars = vars.forward(&xv).unwrap().to_tensor();

    let wv = tape.var(conv.weight());
    let bv = conv.bias().map(|b| tape.var(b));
    let via_direct = xv
        .conv2d(&wv, bv.as_ref(), [1, 1], [0, 0], [1, 1], 1)
        .unwrap()
        .to_tensor();

    assert_eq!(dense(&via_vars), dense(&via_direct));
}

// --- 4. Module::forward ≡ forward_host bit 一致 ---

#[test]
fn conv2d_module_forward_matches_forward_host_bit_exact() {
    let conv = Conv2d::new(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 5).unwrap();
    let x = t(
        (0..2 * 4 * 5 * 5)
            .map(|i| (i as f32) * 0.02 - 0.5)
            .collect(),
        &[2, 4, 5, 5],
    );
    // groups=1・in_channels=2 と conv 生成時の in_channels=2 を一致させる。
    let x = t(dense(&x)[..2 * 2 * 5 * 5].to_vec(), &[2, 2, 5, 5]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = Module::forward(&conv, &tape, &xv).unwrap().to_tensor();

    let ops = common::naive_ops();
    let via_host = conv.forward_host(&*ops, &x).unwrap();

    assert_eq!(dense(&via_module), dense(&via_host));
}

#[test]
fn conv1d_module_forward_matches_forward_host_bit_exact() {
    let conv = Conv1d::new(2, 3, 3, 1, 1, 1, 1, true, 6).unwrap();
    let x = t(
        (0..2 * 2 * 9).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 2, 9],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = Module::forward(&conv, &tape, &xv).unwrap().to_tensor();

    let ops = common::naive_ops();
    let via_host = conv.forward_host(&*ops, &x).unwrap();

    assert_eq!(dense(&via_module), dense(&via_host));
}

// --- 5. Conv1d ≡ 手動 reshape Conv2d ---

#[test]
fn conv1d_forward_matches_manual_reshaped_conv2d_bit_exact() {
    let conv1d = Conv1d::new(2, 4, 3, 1, 1, 1, 1, true, 9).unwrap();
    let x = t(
        (0..3 * 2 * 8).map(|i| (i as f32) * 0.01 - 0.1).collect(),
        &[3, 2, 8],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let vars1d = conv1d.bind(&tape);
    let out1d = vars1d.forward(&xv).unwrap().to_tensor();

    // 手動 reshape: [N,C,L] -> [N,C,1,L]・weight [Cout,Cin,k] -> [Cout,Cin,1,k]。
    let w4 = t(dense(conv1d.weight()), &[4, 2, 1, 3]);
    let x4 = t(dense(&x), &[3, 2, 1, 8]);
    let x4v = tape.var(&x4);
    let w4v = tape.var(&w4);
    let b4v = conv1d.bias().map(|b| tape.var(b));
    let out4 = x4v
        .conv2d(&w4v, b4v.as_ref(), [1, 1], [0, 1], [1, 1], 1)
        .unwrap()
        .to_tensor();

    assert_eq!(out1d.shape(), &[3, 4, 8]);
    assert_eq!(dense(&out1d), dense(&out4));
}

// --- 6. named_parameters／set_parameter ---

#[test]
fn conv2d_named_parameters_order_is_weight_then_bias() {
    let conv = Conv2d::new(2, 3, [1, 1], [1, 1], [0, 0], [1, 1], 1, true, 1).unwrap();
    let params = Module::named_parameters(&conv);
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
}

#[test]
fn conv2d_named_parameters_excludes_bias_when_none() {
    let conv = Conv2d::new(2, 3, [1, 1], [1, 1], [0, 0], [1, 1], 1, false, 1).unwrap();
    let params = Module::named_parameters(&conv);
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["weight"]);
}

#[test]
fn conv1d_named_parameters_order_is_weight_then_bias() {
    let conv = Conv1d::new(2, 3, 3, 1, 0, 1, 1, true, 1).unwrap();
    let params = Module::named_parameters(&conv);
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
}

#[test]
fn conv2d_set_parameter_rejects_shape_mismatch() {
    let mut conv = Conv2d::new(2, 3, [1, 1], [1, 1], [0, 0], [1, 1], 1, true, 1).unwrap();
    let wrong = t(vec![0.0; 4], &[2, 2]);
    let err = Module::set_parameter(&mut conv, "weight", wrong).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn conv2d_set_parameter_rejects_unknown_name() {
    let mut conv = Conv2d::new(2, 3, [1, 1], [1, 1], [0, 0], [1, 1], 1, true, 1).unwrap();
    let dummy = conv.weight().clone();
    let err = Module::set_parameter(&mut conv, "bogus", dummy).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv2d_set_parameter_rejects_bias_when_layer_has_none() {
    let mut conv = Conv2d::new(2, 3, [1, 1], [1, 1], [0, 0], [1, 1], 1, false, 1).unwrap();
    let bias = t(vec![0.0; 3], &[3]);
    let err = Module::set_parameter(&mut conv, "bias", bias).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv1d_set_parameter_replaces_weight_in_place() {
    let mut conv = Conv1d::new(2, 3, 3, 1, 0, 1, 1, true, 1).unwrap();
    let new_weight = t(vec![9.0f32; 3 * 2 * 3], &[3, 2, 3]);
    Module::set_parameter(&mut conv, "weight", new_weight.clone()).unwrap();
    assert_eq!(dense(conv.weight()), dense(&new_weight));
}

// --- 7. 数値微分突合（groups>1 の 1 ケースのみ。厳密な forward/backward
// 突合は `tests/conv2d.rs`／`tests/conv1d.rs` の本体で網羅済みのため、
// 本ファイルでは「`nn::Conv2d` 層経由でも同じ勾配が出る」ことのみ確認
// する）。

const H: f64 = 1e-3;
const TAU: f64 = 1e-4;
const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-3;

fn assert_grad_close(label: &str, analytic: &[f32], numeric: &[f64]) {
    assert_eq!(analytic.len(), numeric.len(), "{label}: 要素数不一致");
    for (i, (&av, &nv)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let av64 = av as f64;
        let diff = (av64 - nv).abs();
        let rel = diff / av64.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{i}]: analytic={av64} numeric={nv} diff={diff} rel={rel}"
        );
    }
}

#[test]
fn conv2d_layer_weight_grad_matches_numeric_grad_with_groups() {
    let conv = Conv2d::new(4, 4, [3, 3], [1, 1], [1, 1], [1, 1], 2, true, 11).unwrap();
    let x = t(
        (0..4 * 5 * 5).map(|i| (i as f32) * 0.02 - 0.4).collect(),
        &[1, 4, 5, 5],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let vars = conv.bind(&tape);
    let y = vars.forward(&xv).unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(
        (0..out_shape.iter().product::<usize>())
            .map(|i| ((i % 5) as f32) * 0.1 - 0.2)
            .collect(),
        &out_shape,
    );
    let sv = tape.var(&s);
    let loss = y.mul(&sv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let weight_grad = grads.get(&vars.weight).unwrap().unwrap();

    let numeric = numeric_conv2d_layer_weight_grad(
        &x,
        conv.weight(),
        conv.bias(),
        &s,
        [1, 1],
        [1, 1],
        [1, 1],
        2,
    );
    assert_grad_close("weight_grad", dense(weight_grad).as_slice(), &numeric);
}

#[allow(clippy::too_many_arguments)]
fn numeric_conv2d_layer_weight_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Vec<f64> {
    let forward = |w: &Tensor<f32>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let wv = tape.var(w);
        let bv = b.map(|bt| tape.var(bt));
        let y = xv
            .conv2d(&wv, bv.as_ref(), stride, padding, dilation, groups)
            .unwrap();
        let out = y.to_tensor();
        dense(&out)
            .iter()
            .zip(dense(s).iter())
            .map(|(&yv, &sv)| yv as f64 * sv as f64)
            .sum()
    };

    let shape = w.shape().to_vec();
    let mut data = dense(w);
    let mut grad = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = forward(&t(data.clone(), &shape));
        data[i] = (orig - H) as f32;
        let lm = forward(&t(data.clone(), &shape));
        data[i] = orig as f32;
        grad[i] = (lp - lm) / (2.0 * H);
    }
    grad
}

// --- 8. `nn::ConvTranspose2d`（イシュー #2067）---

#[test]
fn conv_transpose2d_new_rejects_in_channels_zero() {
    let err = err_of(ConvTranspose2d::new(
        0,
        4,
        [3, 3],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        1,
        false,
        0,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose2d_new_rejects_output_padding_ge_stride() {
    let err = err_of(ConvTranspose2d::new(
        2,
        4,
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 0],
        [1, 1],
        1,
        false,
        0,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose2d_new_rejects_in_channels_not_divisible_by_groups() {
    let err = err_of(ConvTranspose2d::new(
        3,
        4,
        [3, 3],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        2,
        false,
        0,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose2d_from_parameters_rejects_wrong_rank() {
    let err = err_of(ConvTranspose2d::from_parameters(
        t(vec![0.0; 4], &[2, 2]),
        None,
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        1,
    ));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn conv_transpose2d_from_parameters_rejects_in_channels_zero() {
    // weight.shape()[0] == 0（in_channels = 0）は forward の GEMM が
    // Cin_g = weight.shape()[0] / groups で縮約するため zero-K GEMM を
    // サイレントに構築させないよう拒否する（cursor Bugbot 指摘・#2209）。
    let weight = t(vec![], &[0, 3, 1, 1]);
    let err = err_of(ConvTranspose2d::from_parameters(
        weight,
        None,
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        1,
    ));
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose2d_from_parameters_rejects_bias_shape_mismatch() {
    let weight = t(vec![0.0; 2 * 3], &[2, 3, 1, 1]);
    let bad_bias = t(vec![0.0; 2], &[2]);
    let err = err_of(ConvTranspose2d::from_parameters(
        weight,
        Some(bad_bias),
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        1,
    ));
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn conv_transpose2d_new_is_deterministic_and_seed_dependent() {
    let a = ConvTranspose2d::new(3, 4, [3, 3], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 7).unwrap();
    let b = ConvTranspose2d::new(3, 4, [3, 3], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 7).unwrap();
    let c = ConvTranspose2d::new(3, 4, [3, 3], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 8).unwrap();

    assert_eq!(dense(a.weight()), dense(b.weight()));
    assert_ne!(dense(a.weight()), dense(c.weight()));

    // fan_in = Cout_g * kH * kW = 4 * 3 * 3（`ConvTranspose2d::new` doc
    // 参照。`Conv2d` の `Cin_g * kH * kW` とは異なる軸を使う）。
    let fan_in = 4 * 3 * 3;
    let bound = 1.0f32 / (fan_in as f32).sqrt();
    for &v in dense(a.weight()).iter() {
        assert!(v.abs() <= bound, "|{v}| <= {bound}");
    }
}

#[test]
fn conv_transpose2d_new_without_bias_has_no_bias() {
    let conv =
        ConvTranspose2d::new(2, 2, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, false, 1).unwrap();
    assert!(conv.bias().is_none());
}

#[test]
fn conv_transpose2d_vars_forward_matches_var_conv_transpose2d_bit_exact() {
    let conv =
        ConvTranspose2d::new(2, 3, [2, 2], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 3).unwrap();
    let x = t(
        (0..2 * 2 * 4 * 4)
            .map(|i| (i as f32) * 0.01 - 0.2)
            .collect(),
        &[2, 2, 4, 4],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let vars = conv.bind(&tape);
    let via_vars = vars.forward(&xv).unwrap().to_tensor();

    let wv = tape.var(conv.weight());
    let bv = conv.bias().map(|b| tape.var(b));
    let via_direct = xv
        .conv_transpose2d(&wv, bv.as_ref(), [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap()
        .to_tensor();

    assert_eq!(dense(&via_vars), dense(&via_direct));
}

#[test]
fn conv_transpose2d_module_forward_matches_forward_host_bit_exact() {
    let conv =
        ConvTranspose2d::new(2, 3, [3, 3], [1, 1], [1, 1], [0, 0], [1, 1], 1, true, 5).unwrap();
    let x = t(
        (0..2 * 2 * 5 * 5)
            .map(|i| (i as f32) * 0.02 - 0.5)
            .collect(),
        &[2, 2, 5, 5],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = Module::forward(&conv, &tape, &xv).unwrap().to_tensor();

    let ops = common::naive_ops();
    let via_host = conv.forward_host(&*ops, &x).unwrap();

    assert_eq!(dense(&via_module), dense(&via_host));
}

#[test]
fn conv_transpose2d_named_parameters_order_is_weight_then_bias() {
    let conv =
        ConvTranspose2d::new(2, 3, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 1).unwrap();
    let params = Module::named_parameters(&conv);
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
}

#[test]
fn conv_transpose2d_named_parameters_excludes_bias_when_none() {
    let conv =
        ConvTranspose2d::new(2, 3, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, false, 1).unwrap();
    let params = Module::named_parameters(&conv);
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["weight"]);
}

#[test]
fn conv_transpose2d_set_parameter_rejects_shape_mismatch() {
    let mut conv =
        ConvTranspose2d::new(2, 3, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 1).unwrap();
    let wrong = t(vec![0.0; 4], &[2, 2]);
    let err = Module::set_parameter(&mut conv, "weight", wrong).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn conv_transpose2d_set_parameter_rejects_unknown_name() {
    let mut conv =
        ConvTranspose2d::new(2, 3, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, true, 1).unwrap();
    let dummy = conv.weight().clone();
    let err = Module::set_parameter(&mut conv, "bogus", dummy).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose2d_set_parameter_rejects_bias_when_layer_has_none() {
    let mut conv =
        ConvTranspose2d::new(2, 3, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, false, 1).unwrap();
    let bias = t(vec![0.0; 3], &[3]);
    let err = Module::set_parameter(&mut conv, "bias", bias).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn conv_transpose2d_layer_weight_grad_matches_numeric_grad_with_groups() {
    let conv =
        ConvTranspose2d::new(4, 4, [3, 3], [1, 1], [1, 1], [0, 0], [1, 1], 2, true, 11).unwrap();
    let x = t(
        (0..4 * 5 * 5).map(|i| (i as f32) * 0.02 - 0.4).collect(),
        &[1, 4, 5, 5],
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let vars = conv.bind(&tape);
    let y = vars.forward(&xv).unwrap();
    let out_shape = y.to_tensor().shape().to_vec();
    let s = t(
        (0..out_shape.iter().product::<usize>())
            .map(|i| ((i % 5) as f32) * 0.1 - 0.2)
            .collect(),
        &out_shape,
    );
    let sv = tape.var(&s);
    let loss = y.mul(&sv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let weight_grad = grads.get(&vars.weight).unwrap().unwrap();

    let numeric = numeric_conv_transpose2d_layer_weight_grad(
        &x,
        conv.weight(),
        conv.bias(),
        &s,
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        2,
    );
    assert_grad_close("weight_grad", dense(weight_grad).as_slice(), &numeric);
}

#[allow(clippy::too_many_arguments)]
fn numeric_conv_transpose2d_layer_weight_grad(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    s: &Tensor<f32>,
    stride: [usize; 2],
    padding: [usize; 2],
    output_padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Vec<f64> {
    let forward = |w: &Tensor<f32>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let wv = tape.var(w);
        let bv = b.map(|bt| tape.var(bt));
        let y = xv
            .conv_transpose2d(
                &wv,
                bv.as_ref(),
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            )
            .unwrap();
        let out = y.to_tensor();
        dense(&out)
            .iter()
            .zip(dense(s).iter())
            .map(|(&yv, &sv)| yv as f64 * sv as f64)
            .sum()
    };

    let shape = w.shape().to_vec();
    let mut data = dense(w);
    let mut grad = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = forward(&t(data.clone(), &shape));
        data[i] = (orig - H) as f32;
        let lm = forward(&t(data.clone(), &shape));
        data[i] = orig as f32;
        grad[i] = (lp - lm) / (2.0 * H);
    }
    grad
}
