//! 受け入れ条件「合成関数の end-to-end 勾配が期待値と一致する」を直接
//! 検証する統合テスト（TASK-1.5c・イシュー #18）。
//!
//! - 手計算できる小さな合成関数（`mul` の自己参照・`add`）で解析解と
//!   厳密一致を確認する（`mul` は勾配蓄積〈複数経路からの合算〉の検証を
//!   兼ねる）。
//! - MLP 1 層相当（`matmul → add(bias) → relu → mse_loss`）の合成関数で
//!   `Tape::backward` の解析勾配と中央差分（数値微分）を突合する。判定
//!   閾値は `grad.rs` の grad-check テスト（#17）が用いた値をそのまま
//!   再利用し、新しい許容誤差は導入しない（`H=1e-3`・相対 1e-2 または
//!   絶対 1e-3・`τ=1e-4`。承認追跡は Issue #223）。
//! - `Tape::backward`/`Gradients::get` の API 契約（クロステープ検査・
//!   未到達ノード・境界外アクセス・非スカラー loss の暗黙総和射影）を
//!   検証する。
//!
//! `Tape`/`Var` を経由する end-to-end 経路のみを対象とし、PoC-v2-2 の
//! 実測ケース網羅・回帰テスト化は #19（TASK-1.5d）のスコープのため
//! 含めない。

mod common;

use autodiff::{AutodiffError, Tape};
use tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. mul の自己参照（勾配蓄積の検証）: loss = sum(x * x) → dx = 2x ---

#[test]
fn mul_self_reference_accumulates_gradient() {
    let tape = Tape::new(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));

    // 同一 `Var` を `mul` の両引数に渡す。x のノードへは
    // `Mul(x, x)` の VJP から dA・dB 両方の寄与が流入するため、
    // `backward()` の蓄積（合算）ロジックを直接検証する。
    let y = x.mul(&x).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");

    // d/dx sum(x*x) = 2x
    assert_eq!(dx.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[0, 1]).unwrap(), -4.0);
    assert_eq!(dx.get(&[1, 0]).unwrap(), 6.0);
    assert_eq!(dx.get(&[1, 1]).unwrap(), 1.0);
}

// --- 2. add: loss = sum(a + b) → da = db = ones ---

#[test]
fn add_grad_is_ones_for_both_operands() {
    let tape = Tape::new(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![0.5, 1.5, -1.0], &[3]));

    let y = a.add(&b).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    let db = grads.get(&b).unwrap().expect("b は loss に到達する");

    for i in 0..3 {
        assert_eq!(da.get(&[i]).unwrap(), 1.0);
        assert_eq!(db.get(&[i]).unwrap(), 1.0);
    }
}

// --- 3. MLP 1 層相当の合成関数: 数値微分との end-to-end 突合 ---
//
// `loss = mse_loss(relu(x.matmul(w) + b), target)`。ReLU 入力は
// `|value| >= 10h`（`h = 1e-3`）の固定値でキンクを回避する（#17 と同方針）。

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

/// `Tape`/`Var` を新規構築して forward を再評価し、スカラー loss 値
/// （f32）を返す。中央差分の各サンプル点でテープを 1 回使い捨てる
/// （`tape.rs` が前提とする学習ループ運用と同じパターン）。
fn forward_loss(x: &Tensor<f32>, w: &Tensor<f32>, b: &Tensor<f32>, target: &Tensor<f32>) -> f32 {
    let tape = Tape::new(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let bv = tape.var(b);
    let tv = tape.var(target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

/// 指定テンソルの各要素を中央差分で摂動し、`forward_loss` に対する
/// 数値勾配を計算する。f64 で集計し丸め誤差を抑える
/// （`grad.rs` の `numeric_grad_unary` と同方針）。
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

struct MlpFixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    b: Tensor<f32>,
    target: Tensor<f32>,
}

fn mlp_fixture() -> MlpFixture {
    // pre-relu = x @ w + b の各要素の絶対値は 0.1 以上
    // （[-0.15, -1.3, 3.25, -0.1]）。h=1e-3 の摂動で符号が変わらず
    // ReLU のキンクを踏まない。
    MlpFixture {
        x: t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]),
        w: t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]),
        b: t(vec![0.1, -0.2], &[2]),
        target: t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]),
    }
}

#[test]
fn mlp_grad_w_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.b, &f.target));
    assert_grad_close("mlp dW", dw, &num_dw);
}

#[test]
fn mlp_grad_b_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_db = numeric_grad(&f.b, |b| forward_loss(&f.x, &f.w, &b, &f.target));
    assert_grad_close("mlp dB", db, &num_db);
}

#[test]
fn mlp_grad_x_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&f.x, |x| forward_loss(&x, &f.w, &f.b, &f.target));
    assert_grad_close("mlp dX", dx, &num_dx);
}

// --- 4. API 契約テスト ---

#[test]
fn unreachable_leaf_returns_ok_none() {
    let tape = Tape::new(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    // loss に一切関与しない葉ノード。
    let unused = tape.var(&t(vec![9.0], &[1]));
    let loss = x.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&unused).unwrap().is_none());
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn backward_with_foreign_tape_var_returns_tape_mismatch() {
    let tape_a = Tape::new(common::naive_ops());
    let tape_b = Tape::new(common::naive_ops());
    let loss_a = tape_a.var(&t(vec![1.0], &[1])).sum(None).unwrap();
    // `tape_b` から `backward` を呼びつつ、`tape_a` の loss を渡す。
    let x_b = tape_b.var(&t(vec![2.0], &[1]));
    let _ = x_b.sum(None).unwrap(); // tape_b にも何かノードを積んでおく

    let result = tape_b.backward(&loss_a);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn gradients_get_with_foreign_tape_var_returns_tape_mismatch() {
    let tape_a = Tape::new(common::naive_ops());
    let tape_b = Tape::new(common::naive_ops());
    let x_a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let loss_a = x_a.sum(None).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();

    let x_b = tape_b.var(&t(vec![3.0], &[1]));
    let result = grads_a.get(&x_b);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn get_for_var_added_after_backward_returns_ok_none() {
    let tape = Tape::new(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = x.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    // backward 完了後に同一テープへ新規ノードを追加する
    // （`grads.grads.len()` を超える `NodeId` になる）。
    let after = tape.var(&t(vec![9.0], &[1]));
    assert!(grads.get(&after).unwrap().is_none());
}

#[test]
fn non_scalar_loss_seed_is_implicit_sum_projection() {
    // 非スカラー loss（shape [2]）は「暗黙の総和射影」
    // （`sum(loss).backward()` 相当）として扱われ、シードは全要素 1。
    // ここでは loss = x（恒等）とし、seed = ones と直接一致することを
    // 確認する（各要素が独立にそのまま出力へ伝わるため）。
    let tape = Tape::new(common::naive_ops());
    let x = tape.var(&t(vec![3.0, -1.0], &[2]));
    let grads = tape.backward(&x).unwrap();
    let dx = grads.get(&x).unwrap().expect("x 自身が loss である");

    assert_eq!(dx.get(&[0]).unwrap(), 1.0);
    assert_eq!(dx.get(&[1]).unwrap(), 1.0);
}
