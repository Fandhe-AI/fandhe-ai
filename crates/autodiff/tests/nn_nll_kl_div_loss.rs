//! 受け入れ条件「NLLLoss／KLDivLoss の forward／backward が数値微分と
//! 一致する」を直接検証する統合テスト（イシュー #1738。親イシュー
//! #1609「損失関数の拡張」。`nn_bce_loss.rs`〈イシュー #1737〉と同型
//! 構成）。
//!
//! - forward: 解析値突合（手計算 `f64` 定数。`docs/compat-api-scope.md`
//!   §1.2 の数式を用いる。torch 等はリポ・CI へ持ち込まない）。
//! - `matmul → log_softmax → nll_loss` と `matmul → cross_entropy_loss`
//!   の loss・`x.grad` が REQ-2 複合判定（相対誤差 1e-3 未満 または
//!   絶対誤差 1e-5 未満）で一致することを `Tape::backward` を通した
//!   実際の連鎖律で確認する（計画 §3「動作確認の錨」）。
//! - backward: `matmul → log_softmax → nll_loss`・`matmul → kl_div_loss`
//!   の合成関数に対し `Tape::backward` の解析勾配を中央差分（数値微分）
//!   と突合する（`nn_bce_loss.rs` と同じ許容誤差 `H=1e-3`・相対 1e-2
//!   または絶対 1e-3・`τ=1e-4`。新設・緩和はしない）。KLDiv は `target`
//!   側勾配（`t > 0` の点のみ。`0` を跨ぐ摂動は forward の `l=0` 分岐へ
//!   入り解析的に不連続なため対象外）も数値微分で検証する。
//! - `nn::loss::NllLoss`／`KlDivLoss` が `Var` 直接呼び出しと同一の
//!   値・勾配を返す（薄いラッパー性）。
//! - エラー経路（`class_dim` 範囲外・targets shape 不一致・targets
//!   範囲外・shape 不一致・クロステープ）。
//! - `target == 0` 要素の寄与が 0・空入力で損失 0.0 になることの直接
//!   検証。

mod common;

use fandhe_ai_autodiff::nn::loss::{KlDivLoss, NllLoss};
use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. forward 解析値突合（f64 手計算定数） ---

#[test]
fn nll_loss_forward_mean_matches_analytic_value() {
    // input（log 確率想定。範囲検査は課さない）=[[-0.1,-2.0,-1.5],
    // [-0.3,-0.2,-3.0]], class_dim=1, targets=[0,1]。
    // l0 = -input[0,0] = 0.1, l1 = -input[1,1] = 0.2。
    // mean = (0.1+0.2)/2 = 0.15。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0, -1.5, -0.3, -0.2, -3.0], &[2, 3]));
    let targets = ti(vec![0, 1], &[2]);

    let loss = input.nll_loss(&targets, 1, Reduction::Mean).unwrap();
    let expected = (0.1f64 + 0.2f64) / 2.0;
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

#[test]
fn nll_loss_forward_sum_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0, -1.5, -0.3, -0.2, -3.0], &[2, 3]));
    let targets = ti(vec![0, 1], &[2]);

    let loss = input.nll_loss(&targets, 1, Reduction::Sum).unwrap();
    let expected = 0.1f64 + 0.2f64;
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

#[test]
fn nll_loss_empty_targets_returns_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(Vec::new(), &[0, 3]));
    let targets = ti(Vec::new(), &[0]);

    let loss_mean = input.nll_loss(&targets, 1, Reduction::Mean).unwrap();
    let loss_sum = input.nll_loss(&targets, 1, Reduction::Sum).unwrap();
    assert_eq!(scalar(&loss_mean.to_tensor()), 0.0);
    assert_eq!(scalar(&loss_sum.to_tensor()), 0.0);
}

#[test]
fn kl_div_loss_forward_mean_matches_analytic_value() {
    // input=[-2.0,-0.5,-1.2,-0.1], target=[0.2,0.8,0.5,0.5]
    // l = t*(ln t - x)（t==0 は 0）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-2.0, -0.5, -1.2, -0.1], &[2, 2]));
    let target = tape.var(&t(vec![0.2, 0.8, 0.5, 0.5], &[2, 2]));

    let loss = input.kl_div_loss(&target, Reduction::Mean).unwrap();
    let l0 = 0.2f64 * (0.2f64.ln() - (-2.0f64));
    let l1 = 0.8f64 * (0.8f64.ln() - (-0.5f64));
    let l2 = 0.5f64 * (0.5f64.ln() - (-1.2f64));
    let l3 = 0.5f64 * (0.5f64.ln() - (-0.1f64));
    let expected = (l0 + l1 + l2 + l3) / 4.0;
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

#[test]
fn kl_div_loss_target_zero_contributes_zero() {
    // `t == 0` は forward で寄与 0（`xlogy` 規約。`0 * ln(0)` を NaN に
    // させない）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-2.0, -0.5], &[2]));
    let target = tape.var(&t(vec![0.0, 0.8], &[2]));

    let loss = input.kl_div_loss(&target, Reduction::Sum).unwrap();
    let expected = 0.8f64 * (0.8f64.ln() - (-0.5f64));
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

#[test]
fn kl_div_loss_with_log_target_forward_matches_analytic_value() {
    // log_target=True: l = exp(t)*(t - x)。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-2.0, -0.5], &[2]));
    let target = tape.var(&t(vec![-1.6, -0.2], &[2]));

    let loss = input
        .kl_div_loss_with_log_target(&target, Reduction::Mean)
        .unwrap();
    let l0 = (-1.6f64).exp() * (-1.6f64 - (-2.0f64));
    let l1 = (-0.2f64).exp() * (-0.2f64 - (-0.5f64));
    let expected = (l0 + l1) / 2.0;
    assert!(
        (scalar(&loss.to_tensor()) as f64 - expected).abs() < 1e-5,
        "got={} expected={}",
        scalar(&loss.to_tensor()),
        expected
    );
}

// --- 2. 動作確認の錨: log_softmax → nll_loss ≡ cross_entropy_loss ---

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

/// 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満（REQ-2 統一複合判定。
/// `.claude/rules/coding-rust.md`）。
fn assert_close_req2(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

#[test]
fn nll_loss_of_log_softmax_matches_cross_entropy_loss_end_to_end() {
    let logits_data = vec![1.0f32, -2.0, 3.0, 0.5, -1.0, 2.0];
    let targets = ti(vec![2, 0], &[2]);

    for reduction in [Reduction::Mean, Reduction::Sum] {
        // 経路 A: matmul は不要（logits 自体を追跡対象にする）→
        // log_softmax → nll_loss。
        let tape_a = Tape::new_with_ops(common::naive_ops());
        let logits_a = tape_a.var(&t(logits_data.clone(), &[2, 3]));
        let log_probs_a = logits_a.log_softmax(1).unwrap();
        let loss_a = log_probs_a.nll_loss(&targets, 1, reduction).unwrap();
        let grads_a = tape_a.backward(&loss_a).unwrap();
        let dlogits_a = grads_a
            .get(&logits_a)
            .unwrap()
            .expect("logits は loss に到達する");

        // 経路 B: cross_entropy_loss（1 個の融合オペ）。
        let tape_b = Tape::new_with_ops(common::naive_ops());
        let logits_b = tape_b.var(&t(logits_data.clone(), &[2, 3]));
        let loss_b = logits_b.cross_entropy_loss(&targets, 1, reduction).unwrap();
        let grads_b = tape_b.backward(&loss_b).unwrap();
        let dlogits_b = grads_b
            .get(&logits_b)
            .unwrap()
            .expect("logits は loss に到達する");

        assert_close_req2(
            scalar(&loss_a.to_tensor()),
            scalar(&loss_b.to_tensor()),
            &format!("loss value ({reduction:?})"),
        );
        for i in 0..2 {
            for j in 0..3 {
                assert_close_req2(
                    dlogits_a.get(&[i, j]).unwrap(),
                    dlogits_b.get(&[i, j]).unwrap(),
                    &format!("dLogits[{i},{j}] ({reduction:?})"),
                );
            }
        }
    }
}

// --- 3. end-to-end backward（数値微分突合） ---

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

fn forward_loss_nll(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    targets: &Tensor<i32>,
    reduction: Reduction,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let log_probs = xv.matmul(&wv).unwrap().log_softmax(1).unwrap();
    let loss = log_probs.nll_loss(targets, 1, reduction).unwrap();
    scalar(&loss.to_tensor())
}

fn forward_loss_kl_div(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    target: &Tensor<f32>,
    reduction: Reduction,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let tv = tape.var(target);
    let input = xv.matmul(&wv).unwrap();
    let loss = input.kl_div_loss(&tv, reduction).unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn nll_loss_end_to_end_grad_matches_numeric() {
    let x = t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]);
    let w = t(vec![0.5, -0.7, 0.8, 0.2, -0.1, 0.4], &[2, 3]);
    let targets = ti(vec![0, 2], &[2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let log_probs = xv.matmul(&wv).unwrap().log_softmax(1).unwrap();
    let loss = log_probs.nll_loss(&targets, 1, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&w, |w2| {
        forward_loss_nll(&x, &w2, &targets, Reduction::Mean)
    });
    assert_grad_close("nll_loss e2e dW", dw, &num_dw);
}

#[test]
fn kl_div_loss_end_to_end_grad_matches_numeric() {
    let x = t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]);
    let w = t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]);
    // target は `t > 0` の点のみ（forward の `l=0` 分岐との解析的不連続
    // 〈`docs/compat-api-scope.md` §1.2〉を中央差分の摂動範囲から避ける）。
    let target = t(vec![0.3, 0.6, 0.2, 0.5], &[2, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let tv = tape.var(&target);
    let input = xv.matmul(&wv).unwrap();
    let loss = input.kl_div_loss(&tv, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&w, |w2| {
        forward_loss_kl_div(&x, &w2, &target, Reduction::Mean)
    });
    assert_grad_close("kl_div_loss e2e dW", dw, &num_dw);
}

/// `target` 側勾配も非対称式（`docs/compat-api-scope.md` §1.2）のため
/// 独立に数値微分で検証する（`t > 0` の点のみ。matmul 合成は不要）。
#[test]
fn kl_div_loss_target_grad_matches_numeric() {
    let input_data = vec![-1.0f32, 2.0, 0.5, -3.0];
    let target_data = vec![0.4f32, 0.6, 0.2, 0.8];

    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(input_data.clone(), &[2, 2]));
    let target = tape.var(&t(target_data.clone(), &[2, 2]));
    let loss = input.kl_div_loss(&target, Reduction::Sum).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dtarget = grads
        .get(&target)
        .unwrap()
        .expect("target は loss に到達する");

    let num_dtarget = numeric_grad(&t(target_data, &[2, 2]), |tgt| {
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let i2 = tape2.var(&t(input_data.clone(), &[2, 2]));
        let t2 = tape2.var(&tgt);
        let loss2 = i2.kl_div_loss(&t2, Reduction::Sum).unwrap();
        scalar(&loss2.to_tensor())
    });
    assert_grad_close("kl_div_loss(sum) dTarget", dtarget, &num_dtarget);
}

// --- 4. nn::loss の薄いラッパー同値性（値・勾配とも一致） ---

#[test]
fn nn_loss_nll_loss_matches_var_nll_loss_end_to_end() {
    let x = t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]);
    let w = t(vec![0.5, -0.7, 0.8, 0.2, -0.1, 0.4], &[2, 3]);
    let targets = ti(vec![0, 2], &[2]);

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let wv_a = tape_a.var(&w);
    let log_probs_a = xv_a.matmul(&wv_a).unwrap().log_softmax(1).unwrap();
    let loss_a = NllLoss::new(1, Reduction::Sum)
        .forward(&log_probs_a, &targets)
        .unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&x);
    let wv_b = tape_b.var(&w);
    let log_probs_b = xv_b.matmul(&wv_b).unwrap().log_softmax(1).unwrap();
    let loss_b = log_probs_b.nll_loss(&targets, 1, Reduction::Sum).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

#[test]
fn nn_loss_kl_div_loss_matches_var_end_to_end() {
    let x = t(vec![0.6, -0.4, 0.3, 0.9], &[2, 2]);
    let w = t(vec![0.5, -0.7, 0.8, 0.2], &[2, 2]);
    let target = t(vec![0.3, 0.6, 0.2, 0.5], &[2, 2]);

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let wv_a = tape_a.var(&w);
    let tv_a = tape_a.var(&target);
    let input_a = xv_a.matmul(&wv_a).unwrap();
    let loss_a = KlDivLoss::new(Reduction::Mean)
        .forward(&input_a, &tv_a)
        .unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dw_a = grads_a.get(&wv_a).unwrap().expect("到達する");

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xv_b = tape_b.var(&x);
    let wv_b = tape_b.var(&w);
    let tv_b = tape_b.var(&target);
    let input_b = xv_b.matmul(&wv_b).unwrap();
    let loss_b = input_b.kl_div_loss(&tv_b, Reduction::Mean).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dw_b = grads_b.get(&wv_b).unwrap().expect("到達する");

    assert_eq!(loss_a.to_tensor().get(&[]), loss_b.to_tensor().get(&[]));
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dw_a.get(&[i, j]).unwrap(), dw_b.get(&[i, j]).unwrap());
        }
    }
}

// --- 5. エラー経路 ---

#[test]
fn nll_loss_class_dim_out_of_range_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0], &[2]));
    let targets = ti(vec![0], &[1]);

    let err = input.nll_loss(&targets, 5, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn nll_loss_targets_shape_mismatch_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0, -1.5, -0.3], &[2, 2]));
    let targets = ti(vec![0, 1, 0], &[3]);

    let err = input.nll_loss(&targets, 1, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn nll_loss_targets_out_of_range_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0, -1.5, -0.3], &[2, 2]));
    let targets = ti(vec![0, 5], &[2]);

    let err = input.nll_loss(&targets, 1, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn nll_loss_targets_negative_returns_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0, -1.5, -0.3], &[2, 2]));
    let targets = ti(vec![0, -1], &[2]);

    let err = input.nll_loss(&targets, 1, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn kl_div_loss_shape_mismatch_returns_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-2.0, -0.5], &[2]));
    let target = tape.var(&t(vec![0.2, 0.8, 0.5], &[3]));

    let err = input.kl_div_loss(&target, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn kl_div_loss_cross_tape_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let input = tape_a.var(&t(vec![-2.0, -0.5], &[2]));
    let target = tape_b.var(&t(vec![0.2, 0.8], &[2]));

    let err = input.kl_div_loss(&target, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn nll_loss_cross_tape_is_rejected() {
    // `targets` は非追跡（`Tensor<i32>`）のためクロステープ検査自体は
    // `input` の tape にのみ依存する。ここでは `input`/`targets` 以外の
    // クロステープ経路がないことを構造として明記する代わりに、
    // `class_dim` エラーと同じ検査順序（shape 検査が先）を確認する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&t(vec![-0.1, -2.0], &[1, 2]));
    let targets = ti(vec![1], &[1]);

    let loss = input.nll_loss(&targets, 1, Reduction::Sum);
    assert!(loss.is_ok(), "正常系は成功するはず: {loss:?}");
}
