//! 受け入れ条件「各活性化の forward／backward が期待値と一致する」を
//! 直接検証する統合テスト（TASK-9.1b・イシュー #92）。
//!
//! - forward: 解析値との突合（ReLU・Sigmoid・Tanh）。Sigmoid は加えて
//!   `x = ±30` 級の入力で NaN/Inf を出さず 0/1 へ飽和することを検証
//!   する（`eval::sigmoid` の数値安定形の受け入れ条件）。
//! - backward: `matmul → 活性化 → mse_loss` の合成関数に対し
//!   `Tape::backward` の解析勾配を中央差分（数値微分）と突合する
//!   （`tests/backward.rs` の MLP テストと同じ構成・許容誤差を再利用。
//!   `H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`）。

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

// --- 1. forward 解析値突合 ---

#[test]
fn sigmoid_forward_matches_analytic_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 1.0, -1.0], &[3]));
    let y = x.sigmoid();
    let out = y.to_tensor();

    let sigma0 = out.get(&[0]).unwrap();
    let sigma1 = out.get(&[1]).unwrap();
    let sigma_neg1 = out.get(&[2]).unwrap();

    assert!((sigma0 - 0.5).abs() < 1e-6, "sigmoid(0) = {sigma0}");
    assert!((sigma1 - 0.731_058_6).abs() < 1e-6, "sigmoid(1) = {sigma1}");
    assert!(
        (sigma_neg1 - 0.268_941_4).abs() < 1e-6,
        "sigmoid(-1) = {sigma_neg1}"
    );
}

#[test]
fn tanh_forward_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0], &[1]));
    let y = x.tanh();
    let out = y.to_tensor().get(&[0]).unwrap();
    assert!((out - 0.761_594_2).abs() < 1e-6, "tanh(1) = {out}");
}

#[test]
fn relu_forward_matches_analytic_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-2.0, 0.0, 3.0], &[3]));
    let y = x.relu();
    let out = y.to_tensor();
    assert_eq!(out.get(&[0]).unwrap(), 0.0);
    assert_eq!(out.get(&[1]).unwrap(), 0.0);
    assert_eq!(out.get(&[2]).unwrap(), 3.0);
}

#[test]
fn sigmoid_forward_saturates_without_nan_or_inf_for_large_magnitude_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![30.0, -30.0, 100.0, -100.0], &[4]));
    let y = x.sigmoid();
    let out = y.to_tensor();

    for i in 0..4 {
        let v = out.get(&[i]).unwrap();
        assert!(
            v.is_finite(),
            "sigmoid[{i}] = {v} は NaN/Inf であってはならない"
        );
        assert!((0.0..=1.0).contains(&v), "sigmoid[{i}] = {v} は [0,1] の外");
    }
    // x=30 は 0 桁近似で 1 へ、x=-30 は 0 へ飽和する。
    assert!(out.get(&[0]).unwrap() > 0.999_999);
    assert!(out.get(&[1]).unwrap() < 0.000_001);
    assert!(out.get(&[2]).unwrap() > 0.999_999);
    assert!(out.get(&[3]).unwrap() < 0.000_001);
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

fn fixture() -> Fixture {
    Fixture {
        x: t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]),
        w: t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]),
        target: t(vec![0.2, 0.6, 0.1, 0.4], &[2, 2]),
    }
}

/// `活性化(x @ w)` の forward loss（`mse_loss` に対する `target` 突合）
/// を再評価する。`activation` は `eval.rs` の非公開関数と同じ数式の
/// クレート内テスト用ラッパー（`fandhe_ai_autodiff::Tape`/`Var` 経由のみを公開
/// API サーフェスとして使う）。
fn forward_loss(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    activation: impl for<'a> Fn(&'a fandhe_ai_autodiff::Var<'a>) -> fandhe_ai_autodiff::Var<'a>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let pre_activation = xv.matmul(&wv).unwrap();
    let y = activation(&pre_activation);
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn sigmoid_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().sigmoid();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, |v| v.sigmoid()));
    assert_grad_close("sigmoid e2e dW", dw, &num_dw);
}

#[test]
fn tanh_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().tanh();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, |v| v.tanh()));
    assert_grad_close("tanh e2e dW", dw, &num_dw);
}

#[test]
fn relu_end_to_end_grad_matches_numeric() {
    // pre-relu = x @ w の各要素絶対値が h=1e-3 の摂動でキンクを
    // 踏まないよう固定値を選ぶ（`tests/backward.rs` の mlp_fixture と
    // 同方針）。
    let x = t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
    let w = t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]);
    let target = t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let tv = tape.var(&target);
    let y = xv.matmul(&wv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&w, |w| forward_loss(&x, &w, &target, |v| v.relu()));
    assert_grad_close("relu e2e dW", dw, &num_dw);
}

// --- 3. nn::activation の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_activation_sigmoid_matches_var_sigmoid_end_to_end() {
    use fandhe_ai_autodiff::nn::activation::Sigmoid;

    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let y_a = Sigmoid.forward(&xv_a.matmul(&wv_a).unwrap());
    let loss_a = y_a.mse_loss(&tv_a).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let y_b = xv_b.matmul(&wv_b).unwrap().sigmoid();
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
