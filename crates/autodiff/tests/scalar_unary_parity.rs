//! `fandhe_ai_autodiff::scalar_unary_ops`（イシュー #2145）の forward・
//! backward を NaiveOps（`common::naive_ops()`。`scalar_unary` が
//! `BackendError::Unsupported` を返すためホスト参照実装
//! フォールバック経路のみを通る）で検証する統合テスト。
//!
//! forward は [`fandhe_ai_tensor_core::ScalarUnaryOp::apply`] の逐次
//! 適用と bit 一致する契約（`crate::eval::scalar::unary` が同じ
//! `apply` へ委譲するため）。backward は中央差分（微分可能点のみ）・
//! 解析式の直接検証（区分定数・境界値）で確認する。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::scalar_unary_ops::{
    ceil, erf, floor, pow_scalar, reciprocal, round, rsqrt, sign,
};
use fandhe_ai_tensor_core::{ScalarUnaryOp, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

// ---------------------------------------------------------------------
// 1. forward が ScalarUnaryOp::apply の逐次適用と bit 一致する
// ---------------------------------------------------------------------

#[test]
fn forward_matches_apply_bit_for_bit() {
    let data = vec![-2.7_f32, -0.5, 0.0, -0.0, 1.5, 2.5, f32::NAN, f32::INFINITY];
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(data.clone(), &[data.len()]));

    type UnaryFn =
        for<'t> fn(
            &fandhe_ai_autodiff::Var<'t>,
        )
            -> Result<fandhe_ai_autodiff::Var<'t>, fandhe_ai_autodiff::AutodiffError>;
    let cases: [(UnaryFn, ScalarUnaryOp); 6] = [
        (floor, ScalarUnaryOp::Floor),
        (ceil, ScalarUnaryOp::Ceil),
        (round, ScalarUnaryOp::Round),
        (sign, ScalarUnaryOp::Sign),
        (rsqrt, ScalarUnaryOp::Rsqrt),
        (erf, ScalarUnaryOp::Erf),
    ];
    for (f, op) in cases {
        let out = f(&x).unwrap().to_tensor();
        let expected: Vec<f32> = data.iter().map(|&v| op.apply(v)).collect();
        assert_eq!(
            bits(&out),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{op:?}: forward が ScalarUnaryOp::apply と bit 一致しない"
        );
    }

    // `reciprocal`（0 を含む入力で inf/NaN が出る）も同じ契約。
    let out = reciprocal(&x).unwrap().to_tensor();
    let expected: Vec<f32> = data
        .iter()
        .map(|&v| ScalarUnaryOp::Reciprocal.apply(v))
        .collect();
    assert_eq!(
        bits(&out),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    // `pow_scalar`（exponent=3.0）。
    let out = pow_scalar(&x, 3.0).unwrap().to_tensor();
    let expected: Vec<f32> = data
        .iter()
        .map(|&v| ScalarUnaryOp::PowScalar { exponent: 3.0 }.apply(v))
        .collect();
    assert_eq!(
        bits(&out),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------
// 2. floor/ceil/round/sign backward は恒等的に 0（区分定数）
// ---------------------------------------------------------------------

#[test]
fn piecewise_constant_backward_is_exactly_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-2.7, -0.5, 0.0, 1.5, 2.5], &[5]));

    for (name, y) in [
        ("floor", floor(&x).unwrap()),
        ("ceil", ceil(&x).unwrap()),
        ("round", round(&x).unwrap()),
        ("sign", sign(&x).unwrap()),
    ] {
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        for i in 0..5 {
            assert_eq!(
                dx.get(&[i]).unwrap(),
                0.0,
                "{name}: backward[{i}] は 0 のはず"
            );
        }
    }
}

/// upstream が `inf` を含んでいても区分定数の backward は `0 * inf =
/// NaN` に汚染されず有限の `0` を保つ（PR #1823 の教訓・イシュー
/// #2145 §4 回帰）。
#[test]
fn piecewise_constant_backward_stays_zero_with_inf_upstream() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-2.7, 0.0, 2.5], &[3]));
    let inf = tape.var(&t(vec![f32::INFINITY; 3], &[3]));

    for y in [
        floor(&x).unwrap(),
        ceil(&x).unwrap(),
        round(&x).unwrap(),
        sign(&x).unwrap(),
    ] {
        let loss = y.mul(&inf).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        for i in 0..3 {
            let g = dx.get(&[i]).unwrap();
            assert!(
                g.is_finite() && g == 0.0,
                "backward[{i}] = {g} は有限の 0 のはず"
            );
        }
    }
}

// ---------------------------------------------------------------------
// 3. round の偶数丸め・sign の ±0／NaN
// ---------------------------------------------------------------------

#[test]
fn round_ties_to_even() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.5, 1.5, 2.5, -0.5], &[4]));
    let out = round(&x).unwrap().to_tensor();
    assert_eq!(out.get(&[0]).unwrap(), 0.0);
    assert_eq!(out.get(&[1]).unwrap(), 2.0);
    assert_eq!(out.get(&[2]).unwrap(), 2.0);
    assert_eq!(out.get(&[3]).unwrap(), 0.0);
    assert!(
        out.get(&[3]).unwrap().is_sign_negative(),
        "round(-0.5) は -0.0 のはず"
    );
}

#[test]
fn sign_zero_and_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, -0.0, f32::NAN], &[3]));
    let out = sign(&x).unwrap().to_tensor();
    assert_eq!(out.get(&[0]).unwrap(), 0.0);
    assert_eq!(out.get(&[1]).unwrap(), 0.0);
    assert!(out.get(&[2]).unwrap().is_nan());
}

// ---------------------------------------------------------------------
// 4. 定義域外でも panic しない（rsqrt(-1)/reciprocal(0)）
// ---------------------------------------------------------------------

#[test]
fn domain_edges_do_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 0.0], &[2]));
    let r = rsqrt(&x).unwrap().to_tensor();
    assert!(r.get(&[0]).unwrap().is_nan(), "rsqrt(-1) は NaN のはず");
    assert!(r.get(&[1]).unwrap().is_infinite(), "rsqrt(0) は inf のはず");

    let recip = reciprocal(&x).unwrap().to_tensor();
    assert_eq!(recip.get(&[0]).unwrap(), -1.0, "reciprocal(-1) == -1");
    assert!(
        recip.get(&[1]).unwrap().is_infinite(),
        "reciprocal(0) は inf のはず"
    );
}

// ---------------------------------------------------------------------
// 5. reciprocal／rsqrt／erf／pow_scalar backward: 中央差分と突合
// ---------------------------------------------------------------------

fn numeric_grad<'t>(
    tape: &'t Tape,
    x_data: &[f32],
    f: impl Fn(&fandhe_ai_autodiff::Var<'t>) -> fandhe_ai_autodiff::Var<'t>,
) -> Vec<f32> {
    const H: f32 = 1e-3;
    let n = x_data.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut plus = x_data.to_vec();
        plus[i] += H;
        let mut minus = x_data.to_vec();
        minus[i] -= H;
        let yp = f(&tape.var(&t(plus, &[n]))).to_tensor().get(&[i]).unwrap();
        let ym = f(&tape.var(&t(minus, &[n]))).to_tensor().get(&[i]).unwrap();
        out[i] = (yp - ym) / (2.0 * H);
    }
    out
}

fn assert_close_vec(name: &str, analytic: &[f32], numeric: &[f32]) {
    for (i, (&a, &n)) in analytic.iter().zip(numeric).enumerate() {
        let diff = (a - n).abs();
        let tol = 1e-2 * n.abs().max(1.0);
        assert!(
            diff <= tol,
            "{name}[{i}]: analytic={a} numeric={n} diff={diff} tol={tol}"
        );
    }
}

#[test]
fn reciprocal_backward_matches_numeric_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xs = vec![0.5_f32, 1.3, -2.1, -0.7];
    let x = tape.var(&t(xs.clone(), &[4]));
    let y = reciprocal(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    let analytic: Vec<f32> = (0..4).map(|i| dx.get(&[i]).unwrap()).collect();
    let numeric = numeric_grad(&tape, &xs, |v| reciprocal(v).unwrap());
    assert_close_vec("reciprocal", &analytic, &numeric);
}

#[test]
fn rsqrt_backward_matches_numeric_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xs = vec![0.5_f32, 1.3, 2.1, 4.0];
    let x = tape.var(&t(xs.clone(), &[4]));
    let y = rsqrt(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    let analytic: Vec<f32> = (0..4).map(|i| dx.get(&[i]).unwrap()).collect();
    let numeric = numeric_grad(&tape, &xs, |v| rsqrt(v).unwrap());
    assert_close_vec("rsqrt", &analytic, &numeric);
}

#[test]
fn erf_backward_matches_numeric_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xs = vec![-1.7_f32, -0.3, 0.4, 1.9];
    let x = tape.var(&t(xs.clone(), &[4]));
    let y = erf(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    let analytic: Vec<f32> = (0..4).map(|i| dx.get(&[i]).unwrap()).collect();
    let numeric = numeric_grad(&tape, &xs, |v| erf(v).unwrap());
    assert_close_vec("erf", &analytic, &numeric);
}

#[test]
fn pow_scalar_backward_matches_numeric_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xs = vec![0.5_f32, 1.3, 2.1, 3.0];
    let x = tape.var(&t(xs.clone(), &[4]));
    let y = pow_scalar(&x, 2.5).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    let analytic: Vec<f32> = (0..4).map(|i| dx.get(&[i]).unwrap()).collect();
    let numeric = numeric_grad(&tape, &xs, |v| pow_scalar(v, 2.5).unwrap());
    assert_close_vec("pow_scalar", &analytic, &numeric);
}

/// `pow_scalar(x, 0.0)` は forward が定数関数（`x^0 = 1`）になるため
/// 勾配は常に 0（`x == 0.0` でも `0 * inf = NaN` にならない。#1634／
/// #1686 是正の踏襲）。
#[test]
fn pow_scalar_exponent_zero_grad_is_masked() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 3.0], &[2]));
    let y = pow_scalar(&x, 0.0).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.get(&[0]).unwrap(), 0.0);
    assert_eq!(dx.get(&[1]).unwrap(), 0.0);
}

// ---------------------------------------------------------------------
// 6. requires_grad 伝播（forward 値が正しい shape・値で得られること）
// ---------------------------------------------------------------------

#[test]
fn forward_preserves_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let y = floor(&x).unwrap();
    assert_eq!(y.to_tensor().shape(), &[2, 3]);
}
