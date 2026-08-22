//! 受け入れ条件「MSE 損失（mean/sum）の forward／backward が数値微分と
//! 一致する」を直接検証する統合テスト（#190。親イシュー #189）。
//!
//! - forward: mean/sum の解析値突合（固定 fixture）。
//! - backward: `matmul → mse_loss_with` の合成関数に対し
//!   `Tape::backward` の解析勾配を中央差分（数値微分）と突合する
//!   （`tests/backward.rs`・`tests/nn_activation.rs` と同じ構成・
//!   許容誤差を再利用。`H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`。
//!   新設・緩和はしない）。
//! - `nn::loss::MseLoss` が `Var::mse_loss_with` 直接呼び出しと同一の
//!   値・勾配を返す（薄いラッパー性）。
//! - エラー経路（shape 不一致・クロステープ）。

mod common;

use fandhe_ai_autodiff::nn::loss::MseLoss;
use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. forward 解析値突合（pred/target 固定 fixture） ---
//
// diff = pred − target = [0.5, -1.0, 0.5, -0.5] → 二乗誤差 = [0.25, 1.0, 0.25, 0.25]
// sum = 1.75、mean = 1.75 / 4 = 0.4375

#[test]
fn mse_loss_forward_mean_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&t(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]));

    let loss = pred.mse_loss_with(&target, Reduction::Mean).unwrap();
    assert!((scalar(&loss.to_tensor()) - 0.4375).abs() < 1e-6);
}

#[test]
fn mse_loss_forward_sum_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&t(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]));

    let loss = pred.mse_loss_with(&target, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 1.75).abs() < 1e-6);
}

#[test]
fn mse_loss_default_mse_loss_equals_mean_reduction() {
    // 既存 `Var::mse_loss`（mean 固定）は `mse_loss_with(.., Mean)` への
    // 委譲であり、値が完全一致することを確認する（公開 API 非破壊の
    // 直接検証）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&t(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]));

    let via_default = pred.mse_loss(&target).unwrap();
    let via_with = pred.mse_loss_with(&target, Reduction::Mean).unwrap();
    assert_eq!(
        via_default.to_tensor().get(&[]),
        via_with.to_tensor().get(&[])
    );
}

// --- 2. end-to-end backward: matmul → mse_loss_with（mean/sum） ---

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for flat in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{flat:?} idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

struct Fixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    target: Tensor<f32>,
}

fn fixture() -> Fixture {
    Fixture {
        x: t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]),
        w: t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]),
        target: t(vec![0.2, 0.6, 0.1, 0.4], &[2, 2]),
    }
}

fn forward_loss(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    reduction: Reduction,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let y = xv.matmul(&wv).unwrap();
    let loss = y.mse_loss_with(&tv, reduction).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn mse_loss_mean_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap();
    let loss = y.mse_loss_with(&tv, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, Reduction::Mean));
    assert_grad_close("mse(mean) e2e dW", dw, &num_dw);
}

#[test]
fn mse_loss_sum_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap();
    let loss = y.mse_loss_with(&tv, Reduction::Sum).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, Reduction::Sum));
    assert_grad_close("mse(sum) e2e dW", dw, &num_dw);
}

// --- 3. nn::loss::MseLoss の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_loss_mse_loss_matches_var_mse_loss_with_end_to_end() {
    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let pred_a = xv_a.matmul(&wv_a).unwrap();
    let loss_a = MseLoss::new(Reduction::Sum)
        .forward(&pred_a, &tv_a)
        .unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let pred_b = xv_b.matmul(&wv_b).unwrap();
    let loss_b = pred_b.mse_loss_with(&tv_b, Reduction::Sum).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

// --- 4. エラー経路 ---

#[test]
fn mse_loss_with_shape_mismatch_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));

    let err = pred.mse_loss_with(&target, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn mse_loss_with_cross_tape_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let pred = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape_b.var(&t(vec![1.0, 2.0], &[2]));

    let err = pred.mse_loss_with(&target, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}
