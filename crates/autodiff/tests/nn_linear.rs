//! 受け入れ条件「Linear の forward／backward が期待値と一致する」を
//! 直接検証する統合テスト（TASK-9.1a・イシュー #91）。
//!
//! - forward: 小型固定行列で手計算期待値と突合（bias 有無の両方）。
//! - backward: `Linear::forward` → `sum` → `Tape::backward` の解析勾配を
//!   f64 中央差分（`tests/backward.rs` の確立パターンと同じ
//!   `H`/`TAU`/`REL_TOL`/`ABS_TOL`。新規の許容誤差緩和はしない）と突合。
//!   weight・bias・input の 3 系統。bias 勾配が batch 軸縮約
//!   （`grad.rs` の `reduce_to_shape`）で `[out_features]` に落ちることを
//!   ここで確認する。
//! - 初期化: 同一シード → 同一重み、異なるシード → 異なる重み、
//!   値域 `|w| <= 1/√in_features`（`nn/init.rs` の単体テストと役割が
//!   重複しない、公開 API（`Linear::new`）経由の検証）。
//! - エラー経路: `in_features` 不一致の入力・bias 長不一致・rank 不正・
//!   `in_features == 0` で、型付きエラー（返る variant まで固定）が
//!   返る（panic しない）ことと、`out_features == 0` は両コンストラクタ
//!   経路とも受理する非対称性が意図的であることを確認する
//!   （review 指摘 #91: Low「エラー variant を固定できていない」
//!   「`out_features == 0` の境界が非対称」への対応）。
//! - シード導出: 連番シードで複数層を構築しても層 i の bias と
//!   層 i+1 の weight が相関しないことを確認する
//!   （review 指摘 #91: Medium「シードストリーム衝突」への対応。
//!   `nn/init.rs::derive_seed` の単体テストと役割が重複しない、
//!   公開 API（`Linear::new`）経由の統合検証）。

use autodiff::nn::Linear;
use autodiff::{AutodiffError, Tape};
use tensor_core::{ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. forward 期待値一致 ---

#[test]
fn forward_matches_hand_computed_expectation_with_bias() {
    // x[2,3] @ w[3,2] + b[2]
    let x = t(vec![1.0, 2.0, 3.0, -1.0, 0.0, 2.0], &[2, 3]);
    let w = t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    let b = t(vec![0.5, -0.5], &[2]);
    let linear = Linear::from_parameters(w, Some(b)).unwrap();

    let tape = Tape::new();
    let vars = linear.bind(&tape);
    let xv = tape.var(&x);
    let y = vars.forward(&xv).unwrap();
    let out = y.to_tensor();

    // w の行は [w[k,0], w[k,1]] = row0:[1,0]・row1:[0,1]・row2:[1,1]。
    // row0: [1,2,3] -> [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5] + b = [4.5, 4.5]
    // row1: [-1,0,2] -> [-1*1+0*0+2*1, -1*0+0*1+2*1] = [1, 2] + b = [1.5, 1.5]
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.get(&[0, 0]).unwrap(), 4.5);
    assert_eq!(out.get(&[0, 1]).unwrap(), 4.5);
    assert_eq!(out.get(&[1, 0]).unwrap(), 1.5);
    assert_eq!(out.get(&[1, 1]).unwrap(), 1.5);
}

#[test]
fn forward_matches_hand_computed_expectation_without_bias() {
    let x = t(vec![1.0, 2.0], &[1, 2]);
    let w = t(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let linear = Linear::from_parameters(w, None).unwrap();

    let tape = Tape::new();
    let vars = linear.bind(&tape);
    let xv = tape.var(&x);
    let y = vars.forward(&xv).unwrap();
    let out = y.to_tensor();

    // [1,2] @ [[2,0],[0,3]] = [2, 6]
    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(out.get(&[0, 1]).unwrap(), 6.0);
}

// --- 2. backward 期待値一致（数値微分突合） ---
//
// `tests/backward.rs` と同一の許容誤差設定を再利用する
// （coding-rust.md: バックエンド間・回帰テストの許容誤差を単独で
// 緩和しない方針を、新規テストが独自に緩い値へ寄せない形で踏襲）。

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

struct LinearFixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    b: Tensor<f32>,
}

fn linear_fixture() -> LinearFixture {
    LinearFixture {
        x: t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]),
        w: t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]),
        b: t(vec![0.1, -0.2], &[2]),
    }
}

/// `Linear::forward` → `sum` のスカラー loss を新規 `Tape` で再評価する。
/// 中央差分の各サンプル点でテープを 1 回使い捨てる（`backward.rs` と同じ
/// パターン。`tape.rs` の学習ループ運用契約）。
fn forward_loss_sum(x: &Tensor<f32>, w: &Tensor<f32>, b: &Tensor<f32>) -> f32 {
    let tape = Tape::new();
    let linear = Linear::from_parameters(w.clone(), Some(b.clone())).unwrap();
    let vars = linear.bind(&tape);
    let xv = tape.var(x);
    let y = vars.forward(&xv).unwrap();
    let loss = y.sum(None).unwrap();
    scalar(&loss.to_tensor())
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

#[test]
fn backward_grad_w_matches_numeric() {
    let f = linear_fixture();
    let tape = Tape::new();
    let linear = Linear::from_parameters(f.w.clone(), Some(f.b.clone())).unwrap();
    let vars = linear.bind(&tape);
    let xv = tape.var(&f.x);
    let y = vars.forward(&xv).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads
        .get(&vars.weight)
        .unwrap()
        .expect("weight は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss_sum(&f.x, &w, &f.b));
    assert_grad_close("linear dW", dw, &num_dw);
}

#[test]
fn backward_grad_b_matches_numeric_and_is_reduced_over_batch() {
    let f = linear_fixture();
    let tape = Tape::new();
    let linear = Linear::from_parameters(f.w.clone(), Some(f.b.clone())).unwrap();
    let vars = linear.bind(&tape);
    let xv = tape.var(&f.x);
    let y = vars.forward(&xv).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let db = grads
        .get(&vars.bias.unwrap())
        .unwrap()
        .expect("bias は loss に到達する");

    // batch=2 の入力に対し bias 勾配は [out_features]（= [2]）へ縮約される。
    assert_eq!(db.shape(), &[2]);

    let num_db = numeric_grad(&f.b, |b| forward_loss_sum(&f.x, &f.w, &b));
    assert_grad_close("linear dB", db, &num_db);
}

#[test]
fn backward_grad_x_matches_numeric() {
    let f = linear_fixture();
    let tape = Tape::new();
    let linear = Linear::from_parameters(f.w.clone(), Some(f.b.clone())).unwrap();
    let vars = linear.bind(&tape);
    let xv = tape.var(&f.x);
    let y = vars.forward(&xv).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&f.x, |x| forward_loss_sum(&x, &f.w, &f.b));
    assert_grad_close("linear dX", dx, &num_dx);
}

// --- 3. 初期化の決定性 ---

#[test]
fn new_with_same_seed_produces_same_weights() {
    let a = Linear::new(8, 4, true, 42).unwrap();
    let b = Linear::new(8, 4, true, 42).unwrap();
    assert_eq!(a.weight().as_slice(), b.weight().as_slice());
    assert_eq!(a.bias().unwrap().as_slice(), b.bias().unwrap().as_slice());
}

#[test]
fn new_with_different_seed_diverges() {
    let a = Linear::new(8, 4, true, 1).unwrap();
    let b = Linear::new(8, 4, true, 2).unwrap();
    assert_ne!(a.weight().as_slice(), b.weight().as_slice());
}

#[test]
fn new_weight_values_are_within_bound() {
    let in_features = 8usize;
    let bound = 1.0 / (in_features as f32).sqrt();
    let linear = Linear::new(in_features, 4, true, 7).unwrap();
    for v in linear.weight().as_slice().unwrap() {
        assert!(v.abs() <= bound, "out of bound: {v}");
    }
    for v in linear.bias().unwrap().as_slice().unwrap() {
        assert!(v.abs() <= bound, "out of bound: {v}");
    }
}

#[test]
fn new_without_bias_has_no_bias() {
    let linear = Linear::new(4, 2, false, 1).unwrap();
    assert!(linear.bias().is_none());
}

// --- 4. エラー経路 ---

#[test]
fn from_parameters_rejects_weight_rank_mismatch() {
    let w = t(vec![1.0, 2.0, 3.0], &[3]);
    let result = Linear::from_parameters(w, None);
    assert!(matches!(
        result,
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: 1,
        }))
    ));
}

#[test]
fn from_parameters_rejects_bias_rank_mismatch() {
    let w = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let bias = t(vec![1.0, 2.0], &[1, 2]);
    let result = Linear::from_parameters(w, Some(bias));
    assert!(matches!(
        result,
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: 2,
        }))
    ));
}

#[test]
fn from_parameters_rejects_bias_length_mismatch() {
    let w = t(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[2, 3]);
    let bias = t(vec![1.0, 2.0], &[2]); // out_features=3 のはずが 2
    let result = Linear::from_parameters(w, Some(bias));
    assert!(matches!(
        result,
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    if let Err(AutodiffError::Shape(ShapeError::ShapeMismatch { lhs, rhs })) = result {
        assert_eq!(lhs, vec![2]);
        assert_eq!(rhs, vec![3]);
    }
}

#[test]
fn forward_rejects_in_features_mismatch() {
    let w = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let linear = Linear::from_parameters(w, None).unwrap();
    let tape = Tape::new();
    let vars = linear.bind(&tape);
    // in_features=2 のはずが x は shape [1, 3]。
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let result = vars.forward(&x);
    assert!(matches!(result, Err(AutodiffError::Shape(_))));
}

#[test]
fn new_rejects_zero_in_features() {
    let result = Linear::new(0, 4, true, 1);
    assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn new_accepts_zero_out_features_symmetrically_via_both_constructors() {
    // `in_features == 0` は拒否するが `out_features == 0` はテンソル生成
    // まで到達する意図（サイズ 0 軸は妥当な shape。review 指摘 #91 の
    // 非対称性 Low 指摘に対する境界固定テスト）ことを、`new`/
    // `from_parameters` の両経路で確認する。
    let via_new = Linear::new(4, 0, true, 1).unwrap();
    assert_eq!(via_new.weight().shape(), &[4, 0]);
    assert_eq!(via_new.bias().unwrap().shape(), &[0]);

    let w = t(Vec::<f32>::new(), &[4, 0]);
    let b = t(Vec::<f32>::new(), &[0]);
    let via_from_parameters = Linear::from_parameters(w, Some(b)).unwrap();
    assert_eq!(via_from_parameters.weight().shape(), &[4, 0]);
}

// --- 5. weight/bias シード導出の独立性（review 指摘 #91: Medium） ---

#[test]
fn sequential_layer_seeds_do_not_produce_correlated_bias_and_next_weight() {
    // 連番シードで 2 層を構築する自然な使い方
    // （`Linear::new(a, b, true, 1)` に続けて `Linear::new(b, c, true, 2)`）
    // で、層 1 の bias と層 2 の weight が同一乱数列の使い回し
    // （スケール違いなだけの完全相関）にならないことを確認する。
    let l1 = Linear::new(4, 3, true, 1).unwrap();
    let l2 = Linear::new(3, 2, true, 2).unwrap();

    let bound1 = 1.0 / (4f32).sqrt();
    let bound2 = 1.0 / (3f32).sqrt();
    let l1_bias_normalized: Vec<f32> = l1
        .bias()
        .unwrap()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v / bound1)
        .collect();
    let l2_weight_normalized: Vec<f32> = l2
        .weight()
        .as_slice()
        .unwrap()
        .iter()
        .take(l1_bias_normalized.len())
        .map(|v| v / bound2)
        .collect();

    assert_ne!(
        l1_bias_normalized, l2_weight_normalized,
        "l1.bias() と l2.weight() が正規化後に完全一致（同一乱数系列の使い回し）"
    );
}
