//! `Var::silu`／`leaky_relu`／`elu`／`hardswish`（イシュー #1714）の
//! forward 既知値・end-to-end backward（`matmul → 活性化 → mse_loss`
//! 合成に対する解析勾配と数値微分の突合）を検証する統合テスト。
//!
//! `tests/nn_activation.rs`（TASK-9.1b・#92）と同じ構成方針（`H=1e-3`・
//! 相対 1e-2 または絶対 1e-3・`τ=1e-4` の許容誤差）を再利用するが、
//! 対象 4 メソッドは `Var::scalar_unary` の eager 実体化契約により
//! `Result` を返す（`sqrt`／`log` 等と同型。`relu`／`sigmoid`／`tanh`
//! と異なり `?` で連鎖する）ため別ファイルとして分離する（#1713
//! 〈GELU／Softplus〉との並行編集衝突を避ける目的も兼ねる）。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. forward 既知値突合 ---

#[test]
fn silu_forward_matches_analytic_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 1.0], &[2]));
    let y = x.silu().unwrap();
    let out = y.to_tensor();

    assert!((out.get(&[0]).unwrap() - 0.0).abs() < 1e-6);
    assert!(
        (out.get(&[1]).unwrap() - 0.731_058_6).abs() < 1e-6,
        "silu(1) = x * sigmoid(x)"
    );
}

#[test]
fn leaky_relu_forward_matches_analytic_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 2.0], &[2]));
    let y = x.leaky_relu(0.1).unwrap();
    let out = y.to_tensor();

    assert!((out.get(&[0]).unwrap() - (-0.1)).abs() < 1e-6);
    assert_eq!(out.get(&[1]).unwrap(), 2.0);
}

#[test]
fn elu_forward_matches_analytic_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 0.5], &[2]));
    let y = x.elu(1.0).unwrap();
    let out = y.to_tensor();

    assert!(
        (out.get(&[0]).unwrap() - (-0.632_121)).abs() < 1e-5,
        "elu(-1, alpha=1.0) = exp(-1) - 1"
    );
    assert_eq!(out.get(&[1]).unwrap(), 0.5);
}

#[test]
fn hardswish_forward_matches_analytic_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-4.0, 4.0, 1.0], &[3]));
    let y = x.hardswish().unwrap();
    let out = y.to_tensor();

    assert_eq!(out.get(&[0]).unwrap(), 0.0, "hardswish(-4) = 0（飽和域）");
    assert_eq!(out.get(&[1]).unwrap(), 4.0, "hardswish(4) = x（恒等域）");
    assert!(
        (out.get(&[2]).unwrap() - 0.666_666_7).abs() < 1e-6,
        "hardswish(1) = 1 * clamp(4,0,6) / 6"
    );
}

// --- 2. end-to-end backward: matmul → 活性化 → mse_loss ---

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

/// 折れ点（`LeakyReLU` の `x=0`・`Hardswish` の `x=±3`・`Elu` の `x=0`）
/// を避けた `x @ w` の固定値（`tests/nn_activation.rs::fixture` と
/// 同方針で個別に選定）。
fn fixture() -> Fixture {
    Fixture {
        x: t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]),
        w: t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]),
        target: t(vec![0.2, 0.6, 0.1, 0.4], &[2, 2]),
    }
}

/// `活性化(x @ w)` の forward loss（`mse_loss` に対する `target` 突合）
/// を再評価する。`activation` は `Result` を返す本イシュー対象 4
/// メソッド用に `tests/nn_activation.rs::forward_loss` を fallible 版
/// へ調整したもの。
fn forward_loss(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    activation: impl for<'a> Fn(
        &'a fandhe_ai_autodiff::Var<'a>,
    ) -> Result<
        fandhe_ai_autodiff::Var<'a>,
        fandhe_ai_autodiff::AutodiffError,
    >,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let pre_activation = xv.matmul(&wv).unwrap();
    let y = activation(&pre_activation).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn silu_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().silu().unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, |v| v.silu()));
    assert_grad_close("silu e2e dW", dw, &num_dw);
}

#[test]
fn leaky_relu_end_to_end_grad_matches_numeric() {
    // pre-activation の各要素が h=1e-3 の摂動で `x=0`（キンク）を
    // 踏まないよう固定値を選ぶ（`tests/nn_activation.rs::relu_end_to_end`
    // と同方針）。
    let x = t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
    let w = t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]);
    let target = t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let tv = tape.var(&target);
    let y = xv.matmul(&wv).unwrap().leaky_relu(0.1).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&w, |w| forward_loss(&x, &w, &target, |v| v.leaky_relu(0.1)));
    assert_grad_close("leaky_relu e2e dW", dw, &num_dw);
}

#[test]
fn elu_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().elu(1.3).unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, |v| v.elu(1.3)));
    assert_grad_close("elu e2e dW", dw, &num_dw);
}

#[test]
fn hardswish_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().hardswish().unwrap();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss(&f.x, &w, &f.target, |v| v.hardswish())
    });
    assert_grad_close("hardswish e2e dW", dw, &num_dw);
}

// --- 3. nn::activation の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_activation_silu_matches_var_silu_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Silu;

    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = Silu.forward(&xv_a.matmul(&wv_a).unwrap()).unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().silu().unwrap();
    let loss_b = y_b.mse_loss(&tv_b).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

#[test]
fn nn_activation_hardswish_matches_var_hardswish_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Hardswish;

    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = Hardswish.forward(&xv_a.matmul(&wv_a).unwrap()).unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().hardswish().unwrap();
    let loss_b = y_b.mse_loss(&tv_b).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

#[test]
fn nn_activation_leaky_relu_matches_var_leaky_relu_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::LeakyRelu;

    let x = t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
    let w = t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]);
    let target = t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]);

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let wv_a = tape_a.var(&w);
    let tv_a = tape_a.var(&target);
    let y_a = LeakyRelu::new(0.1)
        .forward(&xv_a.matmul(&wv_a).unwrap())
        .unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&x);
    let wv_b = tape_b.var(&w);
    let tv_b = tape_b.var(&target);
    let y_b = xv_b.matmul(&wv_b).unwrap().leaky_relu(0.1).unwrap();
    let loss_b = y_b.mse_loss(&tv_b).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

#[test]
fn nn_activation_elu_matches_var_elu_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Elu;

    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = Elu::new(1.3).forward(&xv_a.matmul(&wv_a).unwrap()).unwrap();
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().elu(1.3).unwrap();
    let loss_b = y_b.mse_loss(&tv_b).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}
