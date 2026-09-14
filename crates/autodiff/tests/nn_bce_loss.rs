//! 受け入れ条件「BCELoss／BCEWithLogitsLoss の forward／backward が
//! 数値微分と一致する」を直接検証する統合テスト（イシュー #1737。
//! 親イシュー #1609「損失関数の拡張」。`nn_loss.rs`〈#190〉と同型構成）。
//!
//! - forward: `Probabilities`／`Logits` 両 kind の解析値突合（手計算
//!   `f64` 定数。`docs/compat-api-scope.md` §1.2 の数式を用いる）。
//! - backward: `matmul → sigmoid → bce_loss`（`Probabilities`）・
//!   `matmul → bce_with_logits_loss`（`Logits`）の合成関数に対し
//!   `Tape::backward` の解析勾配を中央差分（数値微分）と突合する
//!   （`nn_loss.rs` と同じ許容誤差 `H=1e-3`・相対 1e-2 または絶対
//!   1e-3・`τ=1e-4`。新設・緩和はしない）。`target` 側勾配も同様に
//!   数値微分で検証する。
//! - `nn::loss::BceLoss`／`BceWithLogitsLoss` が `Var` 直接呼び出しと
//!   同一の値・勾配を返す（薄いラッパー性）。
//! - エラー経路（shape 不一致・クロステープ・`Probabilities` の
//!   `input`／`target` 範囲外・NaN）。

mod common;

use fandhe_ai_autodiff::nn::loss::{BceLoss, BceWithLogitsLoss};
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

// --- 1. forward 解析値突合（f64 手計算定数） ---

#[test]
fn bce_loss_forward_mean_matches_analytic_value() {
    // input=[0.2,0.8,0.5,0.9], target=[0.0,1.0,1.0,0.0]
    // l0 = -ln(1-0.2) = -ln(0.8) = 0.2231435513
    // l1 = -ln(0.8)   = 0.2231435513
    // l2 = -ln(0.5)   = 0.6931471806
    // l3 = -ln(1-0.9) = -ln(0.1) = 2.3025850930
    // mean = (0.2231435513+0.2231435513+0.6931471806+2.3025850930)/4
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![0.2, 0.8, 0.5, 0.9], &[2, 2]));
    let target = tape.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));

    let loss = input.bce_loss(&target, Reduction::Mean).unwrap();
    let l0 = -(0.8f64.ln());
    let l1 = -(0.8f64.ln());
    let l2 = -(0.5f64.ln());
    let l3 = -(0.1f64.ln());
    let expected = (l0 + l1 + l2 + l3) / 4.0;
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

#[test]
fn bce_loss_forward_clamps_extreme_probabilities() {
    // input=1.0, target=1.0（クランプ無しなら l=-ln(1)=0）だが、対照的に
    // input=0.0, target=0.0（同様に l=0）。クランプが効くケース:
    // input=1e-45（f32 の denormal 近傍。ln(1e-45) は非常に大きな負値だが
    // クランプ -100 で有限に抑えられる）を確認する（`docs/compat-api-
    // scope.md` §1.2「ログクランプ」節）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![1e-45f32], &[1]));
    let target = tape.var(&t(vec![1.0f32], &[1]));

    let loss = input.bce_loss(&target, Reduction::Sum).unwrap();
    let value = scalar(&loss.to_tensor());
    assert!(value.is_finite(), "クランプにより有限値になるはず: {value}");
    assert!(
        (value - 100.0).abs() < 1e-3,
        "クランプ下限 -100 相当の値になるはず: {value}"
    );
}

#[test]
fn bce_with_logits_loss_forward_mean_matches_analytic_value() {
    // input(logits)=[-2.0,1.5,0.0,3.0], target=[0.0,1.0,1.0,0.0]
    // l(x,y) = max(x,0) - x*y + ln(1+exp(-|x|))
    // l0: x=-2.0,y=0.0 -> max(-2,0)=0 - 0 + ln(1+exp(-2.0)) = ln(1+0.1353352832)=0.1269280110
    // l1: x=1.5, y=1.0 -> max(1.5,0)=1.5 - 1.5 + ln(1+exp(-1.5))=ln(1+0.2231301601)=0.2014133...
    // l2: x=0.0, y=1.0 -> 0 - 0 + ln(1+exp(0))=ln(2)=0.6931471806
    // l3: x=3.0, y=0.0 -> max(3,0)=3 - 0 + ln(1+exp(-3.0))=3+ln(1+0.0497870684)=3.0486004...
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-2.0, 1.5, 0.0, 3.0], &[2, 2]));
    let target = tape.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));

    let loss = input
        .bce_with_logits_loss(&target, Reduction::Mean)
        .unwrap();
    let l0 = (1.0f64 + (-2.0f64).exp()).ln();
    let l1 = (1.0f64 + (-1.5f64).exp()).ln();
    let l2 = (1.0f64 + 0.0f64.exp()).ln();
    let l3 = 3.0f64 + (1.0f64 + (-3.0f64).exp()).ln();
    let expected = (l0 + l1 + l2 + l3) / 4.0;
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

#[test]
fn bce_with_logits_loss_forward_extreme_logits_is_finite() {
    // x=±100 のような極端な logits でも `ln(1+exp(-|x|))` の合成式
    // （数値安定形）により有限値になることを確認する（素朴な
    // `-ln(sigmoid(x))` では x=-100 のとき exp(100) が overflow する）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![100.0, -100.0], &[2]));
    let target = tape.var(&t(vec![1.0, 0.0], &[2]));

    let loss = input.bce_with_logits_loss(&target, Reduction::Sum).unwrap();
    let value = scalar(&loss.to_tensor());
    assert!(
        value.is_finite(),
        "極端な logits でも有限値になるはず: {value}"
    );
    assert!(
        value < 1e-3,
        "y と符号一致する極端な logits では損失は 0 近傍: {value}"
    );
}

// --- 2. end-to-end backward（数値微分突合） ---

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
    /// `Probabilities` の `target` は `(0, 1)` 開区間に十分余裕を持たせ、
    /// `H=1e-3` の摂動でも範囲外検査に引っかからないようにする。
    target: Tensor<f32>,
}

fn fixture() -> Fixture {
    Fixture {
        x: t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]),
        w: t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]),
        target: t(vec![0.3, 0.6, 0.2, 0.5], &[2, 2]),
    }
}

fn forward_loss_probabilities(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    reduction: Reduction,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let pred = xv.matmul(&wv).unwrap().sigmoid();
    let loss = pred.bce_loss(&tv, reduction).unwrap();
    scalar(&loss.to_tensor())
}

fn forward_loss_logits(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    reduction: Reduction,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let logits = xv.matmul(&wv).unwrap();
    let loss = logits.bce_with_logits_loss(&tv, reduction).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn bce_loss_probabilities_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let pred = xv.matmul(&wv).unwrap().sigmoid();
    let loss = pred.bce_loss(&tv, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss_probabilities(&f.x, &w, &f.target, Reduction::Mean)
    });
    assert_grad_close("bce(probabilities,mean) e2e dW", dw, &num_dw);
}

#[test]
fn bce_loss_logits_end_to_end_grad_matches_numeric() {
    let f = fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let tv = tape.var(&f.target);
    let logits = xv.matmul(&wv).unwrap();
    let loss = logits.bce_with_logits_loss(&tv, Reduction::Sum).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| {
        forward_loss_logits(&f.x, &w, &f.target, Reduction::Sum)
    });
    assert_grad_close("bce(logits,sum) e2e dW", dw, &num_dw);
}

/// `target` 側勾配も `Var::mse_loss_with` の `dPred = −dTarget` のような
/// 単純合成ではなく非対称式（`docs/compat-api-scope.md` §1.2）のため、
/// 独立に数値微分で検証する（`Probabilities` の直接呼び出し。matmul
/// 合成は不要）。
#[test]
fn bce_loss_probabilities_target_grad_matches_numeric() {
    let input_data = vec![0.3f32, 0.7, 0.5, 0.6];
    let target_data = vec![0.4f32, 0.6, 0.2, 0.8];

    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(input_data.clone(), &[2, 2]));
    let target = tape.var(&t(target_data.clone(), &[2, 2]));
    let loss = input.bce_loss(&target, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dtarget = grads
        .get(&target)
        .unwrap()
        .expect("target は loss に到達する");

    let num_dtarget = numeric_grad(&t(target_data, &[2, 2]), |tgt| {
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let i2 = tape2.var(&t(input_data.clone(), &[2, 2]));
        let t2 = tape2.var(&tgt);
        let loss2 = i2.bce_loss(&t2, Reduction::Mean).unwrap();
        scalar(&loss2.to_tensor())
    });
    assert_grad_close("bce(probabilities,mean) dTarget", dtarget, &num_dtarget);
}

#[test]
fn bce_with_logits_loss_target_grad_matches_numeric() {
    let input_data = vec![-1.0f32, 2.0, 0.5, -3.0];
    let target_data = vec![0.4f32, 0.6, 0.2, 0.8];

    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(input_data.clone(), &[2, 2]));
    let target = tape.var(&t(target_data.clone(), &[2, 2]));
    let loss = input.bce_with_logits_loss(&target, Reduction::Sum).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dtarget = grads
        .get(&target)
        .unwrap()
        .expect("target は loss に到達する");

    let num_dtarget = numeric_grad(&t(target_data, &[2, 2]), |tgt| {
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let i2 = tape2.var(&t(input_data.clone(), &[2, 2]));
        let t2 = tape2.var(&tgt);
        let loss2 = i2.bce_with_logits_loss(&t2, Reduction::Sum).unwrap();
        scalar(&loss2.to_tensor())
    });
    assert_grad_close("bce(logits,sum) dTarget", dtarget, &num_dtarget);
}

// --- 3. nn::loss の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_loss_bce_loss_matches_var_bce_loss_end_to_end() {
    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let pred_a = xv_a.matmul(&wv_a).unwrap().sigmoid();
    let loss_a = BceLoss::new(Reduction::Sum)
        .forward(&pred_a, &tv_a)
        .unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let pred_b = xv_b.matmul(&wv_b).unwrap().sigmoid();
    let loss_b = pred_b.bce_loss(&tv_b, Reduction::Sum).unwrap();
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
fn nn_loss_bce_with_logits_loss_matches_var_end_to_end() {
    let f = fixture();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&f.x);
    let wv_a = tape_a.var(&f.w);
    let tv_a = tape_a.var(&f.target);
    let logits_a = xv_a.matmul(&wv_a).unwrap();
    let loss_a = BceWithLogitsLoss::default()
        .forward(&logits_a, &tv_a)
        .unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&f.x);
    let wv_b = tape_b.var(&f.w);
    let tv_b = tape_b.var(&f.target);
    let logits_b = xv_b.matmul(&wv_b).unwrap();
    let loss_b = logits_b
        .bce_with_logits_loss(&tv_b, Reduction::Mean)
        .unwrap();
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
fn bce_loss_with_shape_mismatch_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![0.2, 0.8], &[2]));
    let target = tape.var(&t(vec![0.0, 1.0, 1.0], &[3]));

    let err = input.bce_loss(&target, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn bce_loss_with_cross_tape_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let input = tape_a.var(&t(vec![0.2, 0.8], &[2]));
    let target = tape_b.var(&t(vec![0.0, 1.0], &[2]));

    let err = input.bce_loss(&target, Reduction::Sum).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn bce_loss_input_out_of_range_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![0.5, 1.5], &[2]));
    let target = tape.var(&t(vec![0.0, 1.0], &[2]));

    let err = input.bce_loss(&target, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn bce_loss_target_out_of_range_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![0.5, 0.5], &[2]));
    let target = tape.var(&t(vec![0.0, -0.5], &[2]));

    let err = input.bce_loss(&target, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn bce_loss_nan_input_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![f32::NAN, 0.5], &[2]));
    let target = tape.var(&t(vec![0.0, 1.0], &[2]));

    let err = input.bce_loss(&target, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn bce_with_logits_loss_accepts_out_of_range_input() {
    // Logits kind は範囲制約なし（PyTorch `BCEWithLogitsLoss` と同じ）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![10.0, -10.0], &[2]));
    let target = tape.var(&t(vec![0.0, 1.0], &[2]));

    let loss = input.bce_with_logits_loss(&target, Reduction::Mean);
    assert!(loss.is_ok());
}
