//! Huber／SmoothL1 損失（イシュー #1739）の受け入れ条件「forward／
//! backward が手計算の参照値・数値微分と一致する」を直接検証する統合
//! テスト（`nn_loss.rs` の MSE 版と同型の構成）。
//!
//! - forward: kind×reduction×delta の手計算参照値突合（固定 fixture・
//!   実装計画 §5.2）。
//! - `dTarget == −dPred`（損失が `d = pred − target` のみに依存する
//!   構造的性質の直接検証）。
//! - backward: `matmul → huber_loss` の合成関数に対し `Tape::backward`
//!   の解析勾配を中央差分（数値微分）と突合する（`nn_loss.rs` と同じ
//!   許容誤差 `H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`）。
//! - `delta <= 0`／非有限は `AutodiffError::InvalidArgument`。
//! - shape 不一致・クロステープのエラー経路。
//! - `n == 0`（空テンソル）は forward が `0.0` を返す。

mod common;

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

// --- 1. forward 手計算参照値突合（実装計画 §5.2） ---
//
// pred = [0.0, 1.5, −3.0, 0.25]、target = [0, 0, 0, 0]（d = pred）。

#[test]
fn huber_delta1_sum_matches_hand_computed_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.huber_loss(&target, 1.0, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 3.53125).abs() < 1e-6);
}

#[test]
fn huber_delta1_mean_matches_hand_computed_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.huber_loss(&target, 1.0, Reduction::Mean).unwrap();
    assert!((scalar(&loss.to_tensor()) - 0.8828125).abs() < 1e-6);
}

#[test]
fn smooth_l1_beta1_matches_huber_delta1() {
    // 実装計画 §5.2: SmoothL1(beta=1) は Huber(delta=1) と一致する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.smooth_l1_loss(&target, 1.0, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 3.53125).abs() < 1e-6);
}

#[test]
fn huber_delta2_sum_matches_hand_computed_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.huber_loss(&target, 2.0, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 5.15625).abs() < 1e-6);
}

#[test]
fn huber_delta2_mean_matches_hand_computed_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.huber_loss(&target, 2.0, Reduction::Mean).unwrap();
    assert!((scalar(&loss.to_tensor()) - 1.2890625).abs() < 1e-6);
}

#[test]
fn smooth_l1_beta2_sum_matches_hand_computed_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.smooth_l1_loss(&target, 2.0, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 2.578125).abs() < 1e-6);
}

#[test]
fn smooth_l1_beta2_mean_matches_hand_computed_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![0.0, 1.5, -3.0, 0.25], &[4]));
    let target = tape.var(&t(vec![0.0; 4], &[4]));

    let loss = pred.smooth_l1_loss(&target, 2.0, Reduction::Mean).unwrap();
    assert!((scalar(&loss.to_tensor()) - 0.64453125).abs() < 1e-6);
}

// --- 1b. 境界（線形・二次の折れ点。実装計画 §5.2「境界ケース」） ---

#[test]
fn huber_delta1_boundary_d_equals_delta_is_continuous_with_quadratic_side() {
    // pred=1.0, delta=1 → d=1.0（境界。線形分岐が適用される `<` 判定）。
    // 線形分岐の値 = delta*(|d|-0.5*delta) = 1*(1-0.5) = 0.5、
    // 二次分岐の値（境界での極限）= 0.5*d^2 = 0.5 と一致（連続性）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0], &[1]));
    let target = tape.var(&t(vec![0.0], &[1]));

    let loss = pred.huber_loss(&target, 1.0, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 0.5).abs() < 1e-6);

    let grads = Tape::new_with_ops(common::naive_ops());
    let pred2 = grads.var(&t(vec![1.0], &[1]));
    let target2 = grads.var(&t(vec![0.0], &[1]));
    let loss2 = pred2.huber_loss(&target2, 1.0, Reduction::Sum).unwrap();
    let g = grads.backward(&loss2).unwrap();
    let dpred = g.get(&pred2).unwrap().expect("到達する");
    // 線形分岐の勾配 = copysign(delta, d) = 1.0（d>0 側からの境界）。
    assert!((dpred.get(&[0]).unwrap() - 1.0).abs() < 1e-6);
}

// --- 2. dTarget == −dPred（構造的性質） ---

#[test]
fn huber_dtarget_equals_negated_dpred() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.5, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&t(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]));

    let loss = pred.huber_loss(&target, 1.0, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dpred = grads.get(&pred).unwrap().expect("pred は loss に到達する");
    let dtarget = grads
        .get(&target)
        .unwrap()
        .expect("target は loss に到達する");

    for i in 0..2 {
        for j in 0..2 {
            let dp = dpred.get(&[i, j]).unwrap();
            let dt = dtarget.get(&[i, j]).unwrap();
            assert!((dp + dt).abs() < 1e-6, "dp={dp} dt={dt} should sum to 0");
        }
    }
}

// --- 3. end-to-end backward: matmul → huber_loss（mean/sum） ---

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

/// `nn_loss.rs::fixture` と同一の値（`x @ w` の `d = y − target` が
/// `delta = 1.0` の折れ点から十分離れている——y=[[-0.02,-0.5],
/// [0.87,-0.03]], target=[[0.2,0.6],[0.1,0.4]] → d=[[-0.22,-1.1],
/// [0.77,-0.43]]。`|d|=1.1` は境界からマージン `0.1` あり、`H=1e-3`
/// 摂動〈`|Δy| ≈ |x|·H` 程度〉より十分大きいため中央差分が折れ点を
/// 跨がない）。
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
    let loss = y.huber_loss(&tv, 1.0, reduction).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn huber_loss_mean_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap();
    let loss = y.huber_loss(&tv, 1.0, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, Reduction::Mean));
    assert_grad_close("huber(mean) e2e dW", dw, &num_dw);
}

#[test]
fn huber_loss_sum_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap();
    let loss = y.huber_loss(&tv, 1.0, Reduction::Sum).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.target, Reduction::Sum));
    assert_grad_close("huber(sum) e2e dW", dw, &num_dw);
}

// --- 4. エラー経路 ---

#[test]
fn huber_loss_with_shape_mismatch_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));

    let err = pred.huber_loss(&target, 1.0, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn huber_loss_with_cross_tape_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let pred = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape_b.var(&t(vec![1.0, 2.0], &[2]));

    let err = pred.huber_loss(&target, 1.0, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn huber_loss_with_zero_delta_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));

    let err = pred.huber_loss(&target, 0.0, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn huber_loss_with_negative_delta_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));

    let err = pred.huber_loss(&target, -1.0, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn huber_loss_with_non_finite_delta_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));

    let err_nan = pred
        .huber_loss(&target, f32::NAN, Reduction::Sum)
        .unwrap_err();
    assert!(matches!(err_nan, AutodiffError::InvalidArgument(_)));

    let err_inf = pred
        .huber_loss(&target, f32::INFINITY, Reduction::Sum)
        .unwrap_err();
    assert!(matches!(err_inf, AutodiffError::InvalidArgument(_)));
}

#[test]
fn smooth_l1_loss_with_zero_beta_returns_invalid_argument() {
    // 実装計画 §2.1: `beta = 0`（PyTorch `nn.L1Loss` 相当への退化）は
    // 本実装の対象外——他の値と同じ「有限かつ `> 0`」検査で拒否する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));

    let err = pred
        .smooth_l1_loss(&target, 0.0, Reduction::Sum)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- 5. n == 0（空テンソル） ---

#[test]
fn huber_loss_empty_input_returns_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&t(vec![], &[0]));
    let target = tape.var(&t(vec![], &[0]));

    let loss_mean = pred.huber_loss(&target, 1.0, Reduction::Mean).unwrap();
    assert_eq!(scalar(&loss_mean.to_tensor()), 0.0);

    let loss_sum = pred.huber_loss(&target, 1.0, Reduction::Sum).unwrap();
    assert_eq!(scalar(&loss_sum.to_tensor()), 0.0);
}
