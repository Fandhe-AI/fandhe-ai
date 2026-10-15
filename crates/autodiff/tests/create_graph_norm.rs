//! `Op::LayerNorm`・`Op::RmsNorm` の `create_graph`（`Tape::backward_create_graph`）対応
//! （イシュー #3078・親 #3075・決定記録 `docs/autodiff-higher-order-grad-decision.md` §21.3・§27）
//! の受け入れ条件を直接検証する統合テスト。
//!
//! - Hessian を `Tape::backward` の中央差分（`create_graph` を経由しない独立経路）と REQ-2 統一
//!   複合判定（`common::req2_close`）で突合する（weight・bias の有無・rank 1／2／3・微分対象
//!   `x`／`w`／`b`・`x`×`w` の混合項）。
//! - `cg.first_order()` が `Tape::backward` と bit 同一、`cg.grad(..)` が REQ-2 判定で一致する。
//! - §21.3 の必須ベクトル（中心化の丸め回帰・有限大振幅・小スケール・定数行・全ゼロ行）で
//!   1 階・`cg.grad`・孫テープ再生（`child.backward_create_graph`）の 3 段がすべて有限かつ
//!   REQ-2 判定で一致する。
//! - 入口検査の拒否条件（非有限・overflow 域・`eps=0` の退化行）が `Err(Backward)` かつ
//!   `child.is_empty()` を保つこと、隣接する fail-closed Op が拒否のままであることを固定する。
//! - 構造段の上界による総括的な適用判定（決定記録 §27.3）を正のプローブで確かめる: レビューで
//!   指摘された入力は拒否されて `child` が空になり、掃引した全ケースで「受理なら 1 階・`cg.grad`・
//!   二階・孫テープ再生が有限かつ一致、拒否なら `Err(Backward)` で `child` が空」が成り立つ。
//!
//! 損失は `sum(norm(x) ⊙ c)`（`c` は非一様な固定定数）または `tanh` 合成を使う。affine なしの
//! `sum(norm(x)²)` はほぼ定数で Hessian が 0 に近く検出力が弱いため避ける。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::lrn_ops::local_response_norm;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// 非一様な固定定数（損失の重み `c`・方向ベクトル `v`）。
fn pattern(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.9).sin() + 0.35 * (i as f32 + 1.0).cos() + 0.2)
        .collect()
}

fn dense(v: &Tensor<f32>) -> Vec<f32> {
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

type Built<'t> = Result<Var<'t>, AutodiffError>;

/// 有限差分の刻み。f32 の 1 階勾配の丸め雑音が小さな Hessian 成分の REQ-2 判定を乱さないよう、
/// 2 点中央差分（h=1e-3）ではなく 4 次精度の 5 点中央差分（h=1e-2）で測定側の誤差を抑える。
/// 判定式・tolerance は変更しない。
const FD_STEP: f64 = 1e-2;

/// 5 点中央差分 `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)` を、第 `i` 成分の摂動について
/// 勾配ベクトル全成分分まとめて返す。
fn five_point_column(
    grad_at: &dyn Fn(&[f32]) -> Vec<f64>,
    x0: &[f32],
    i: usize,
    h: f64,
) -> Vec<f64> {
    let at = |d: f64| {
        let mut x = x0.to_vec();
        x[i] += d as f32;
        grad_at(&x)
    };
    let (p2, p1, m1, m2) = (at(2.0 * h), at(h), at(-h), at(-2.0 * h));
    (0..p1.len())
        .map(|j| (-p2[j] + 8.0 * p1[j] - 8.0 * m1[j] + m2[j]) / (12.0 * h))
        .collect()
}

/// `Tape::backward` を中央差分した数値 Hessian（`hessian[j][i] = d(grad_j)/dx_i`）。
fn finite_diff_hessian<F>(build: &F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("forward 構築");
        let grads = tape.backward(&loss).expect("1 階 backward");
        let dx = grads.get(&xv).expect("get").expect("x は loss に到達する");
        dense(dx).into_iter().map(f64::from).collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let col = five_point_column(&grad_at, x0, i, h);
        for j in 0..n {
            hessian[j][i] = col[j];
        }
    }
    hessian
}

/// `v` から平坦添字 `flat_idx` の 1 要素をスカラー `Var` として取り出す。
fn extract_scalar<'c>(v: &Var<'c>, shape: &[usize], flat_idx: usize) -> Built<'c> {
    let mut rem = flat_idx;
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

/// 子テープ上の 1 階勾配をさらに `child.backward` で微分した full Hessian。
fn analytic_hessian<F>(build: &F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let n = numel(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).expect("grad").expect("x は loss に到達する");
    let cx = cg.child_var(&x).expect("child_var").expect("写しあり");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).expect("extract_scalar");
        let row_grads = child.backward(&gj).expect("child.backward");
        let row = dense(
            row_grads
                .get(&cx)
                .expect("get")
                .expect("2 階勾配は x へ到達する"),
        );
        for i in 0..n {
            hessian[j][i] = f64::from(row[i]);
        }
    }
    hessian
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>], ctx: &str) {
    assert_eq!(analytic.len(), numeric.len());
    for (j, (ra, rn)) in analytic.iter().zip(numeric.iter()).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn.iter()).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
}

/// Hessian（有限差分）と 1 階（bit 同一・REQ-2）を一括で検証する。
fn check_all<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Built<'t>,
{
    let numeric = finite_diff_hessian(&build, x0, shape, FD_STEP);
    let analytic = analytic_hessian(&build, x0, shape);
    assert_hessian_close(&analytic, &numeric, ctx);

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward 構築");
    let plain = dense(
        tape.backward(&loss)
            .expect("backward")
            .get(&x)
            .expect("get")
            .expect("到達"),
    );
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let first = dense(cg.first_order().get(&x).expect("get").expect("到達"));
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{ctx}: first_order は Tape::backward と bit 同一"
    );
    let gx = dense(&cg.grad(&x).expect("grad").expect("到達").to_tensor());
    for (i, (&a, &b)) in gx.iter().zip(plain.iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(b)),
            "{ctx}: cg.grad[{i}]={a} vs backward={b}"
        );
    }
}

// --- 構築ヘルパー（微分対象を 1 つ受け、他は定数） -----------------------------------------

const EPS: f32 = 1e-5;

fn weighted_sum<'t>(y: &Var<'t>, tape: &'t Tape, shape: &[usize]) -> Built<'t> {
    let c = tape.var_no_grad(&t(pattern(numel(shape), 1.0), shape));
    y.mul(&c)?.sum(None)
}

fn w_const(tape: &Tape, h: usize) -> Var<'_> {
    tape.var_no_grad(&t(pattern(h, 3.0), &[h]))
}

fn b_const(tape: &Tape, h: usize) -> Var<'_> {
    tape.var_no_grad(&t(pattern(h, 5.0), &[h]))
}

// --- 1. RmsNorm: x について（weight なし／あり・rank 1／2／3） ------------------------------

fn x_data(shape: &[usize]) -> Vec<f32> {
    pattern(numel(shape), 7.0)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * 1.3 + (i % 3) as f32 * 0.4 - 0.5)
        .collect()
}

#[test]
fn rms_norm_hessian_wrt_x_matches_finite_difference() {
    for shape in [vec![3usize], vec![2, 3], vec![2, 2, 3]] {
        let h = *shape.last().unwrap();
        let s = shape.clone();
        check_all(
            &format!("rms/no-w/{shape:?}"),
            |tape, x| weighted_sum(&x.rms_norm(None, EPS)?, tape, &s),
            &x_data(&shape),
            &shape,
        );
        let s = shape.clone();
        check_all(
            &format!("rms/w/{shape:?}"),
            |tape, x| {
                let w = w_const(tape, h);
                weighted_sum(&x.rms_norm(Some(&w), EPS)?, tape, &s)
            },
            &x_data(&shape),
            &shape,
        );
    }
}

#[test]
fn rms_norm_hessian_wrt_weight_matches_finite_difference() {
    let shape = [2usize, 3];
    let xdat = t(x_data(&shape), &shape);
    let w0 = pattern(3, 3.0);
    check_all(
        "rms/dw",
        |tape, w| {
            let x = tape.var_no_grad(&xdat);
            // dw の二階微分は dy（upstream）が w に依存する下流を要するため tanh を挟む。
            let y = x.rms_norm(Some(w), EPS)?.tanh();
            weighted_sum(&y, tape, &shape)
        },
        &w0,
        &[3],
    );
}

// --- 2. LayerNorm: weight・bias の 4 通り（x について）・rank 1／2／3 -----------------------

#[test]
fn layer_norm_hessian_wrt_x_matches_finite_difference_all_affine_combos() {
    for shape in [vec![3usize], vec![2, 3], vec![2, 2, 3]] {
        let h = *shape.last().unwrap();
        for (use_w, use_b) in [(false, false), (true, false), (false, true), (true, true)] {
            let s = shape.clone();
            check_all(
                &format!("ln/w={use_w}/b={use_b}/{shape:?}"),
                move |tape, x| {
                    let w = w_const(tape, h);
                    let b = b_const(tape, h);
                    let y = x.layer_norm(use_w.then_some(&w), use_b.then_some(&b), EPS)?;
                    weighted_sum(&y, tape, &s)
                },
                &x_data(&shape),
                &shape,
            );
        }
    }
}

#[test]
fn layer_norm_hessian_wrt_weight_and_bias_matches_finite_difference() {
    let shape = [2usize, 3];
    let xdat = t(x_data(&shape), &shape);
    let p0 = pattern(3, 3.0);
    check_all(
        "ln/dw",
        |tape, w| {
            let x = tape.var_no_grad(&xdat);
            let b = b_const(tape, 3);
            let y = x.layer_norm(Some(w), Some(&b), EPS)?.tanh();
            weighted_sum(&y, tape, &shape)
        },
        &p0,
        &[3],
    );
    check_all(
        "ln/db",
        |tape, b| {
            let x = tape.var_no_grad(&xdat);
            let w = w_const(tape, 3);
            let y = x.layer_norm(Some(&w), Some(b), EPS)?.tanh();
            weighted_sum(&y, tape, &shape)
        },
        &p0,
        &[3],
    );
}

#[test]
fn layer_norm_nonlinear_downstream_matches_finite_difference() {
    let shape = [2usize, 4];
    check_all(
        "ln/tanh",
        |tape, x| {
            let w = w_const(tape, 4);
            let b = b_const(tape, 4);
            let y = x.layer_norm(Some(&w), Some(&b), EPS)?.tanh();
            weighted_sum(&y, tape, &shape)
        },
        &x_data(&shape),
        &shape,
    );
}

// --- 3. 混合項: x と w が両方追跡ありのとき d(cg.grad(x))/dw ---------------------------------

#[test]
fn layer_norm_mixed_x_w_term_matches_finite_difference() {
    let shape = [2usize, 3];
    let x0 = x_data(&shape);
    let w0 = pattern(3, 3.0);
    // FD: d(grad_x_j)/dw_i を w の中央差分で求める。
    let grad_x_at = |w: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let wv = tape.var(&t(w.to_vec(), &[3]));
        let y = x.layer_norm(Some(&wv), None, EPS).unwrap().tanh();
        let loss = weighted_sum(&y, &tape, &shape).unwrap();
        dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap())
            .into_iter()
            .map(f64::from)
            .collect()
    };
    let h = FD_STEP;
    let n = numel(&shape);
    let mut numeric = vec![vec![0.0f64; 3]; n];
    for i in 0..3 {
        let col = five_point_column(&grad_x_at, &w0, i, h);
        for j in 0..n {
            numeric[j][i] = col[j];
        }
    }

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.clone(), &shape));
    let wv = tape.var(&t(w0.clone(), &[3]));
    let y = x.layer_norm(Some(&wv), None, EPS).unwrap().tanh();
    let loss = weighted_sum(&y, &tape, &shape).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = cg.grad(&x).unwrap().unwrap();
    let cw = cg.child_var(&wv).unwrap().unwrap();
    for j in 0..n {
        let gj = extract_scalar(&gx, &shape, j).unwrap();
        let row = dense(
            child
                .backward(&gj)
                .unwrap()
                .get(&cw)
                .unwrap()
                .expect("混合項は w へ到達する"),
        );
        for i in 0..3 {
            assert!(
                common::req2_close(f64::from(row[i]), numeric[j][i]),
                "mixed[{j}][{i}] analytic={} numeric={}",
                row[i],
                numeric[j][i]
            );
        }
    }
}

// --- 4. §21.3 必須ベクトル: 1 階・cg.grad・孫テープ再生の 3 段が有限かつ一致 -------------------

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rms,
    Layer,
}

fn build_norm<'t>(
    tape: &'t Tape,
    x: &Var<'t>,
    kind: Kind,
    eps: f32,
    affine: bool,
    c: &[f32],
    shape: &[usize],
) -> Built<'t> {
    let h = *shape.last().unwrap();
    let w = w_const(tape, h);
    let b = b_const(tape, h);
    let y = match kind {
        Kind::Rms => x.rms_norm(affine.then_some(&w), eps)?,
        Kind::Layer => x.layer_norm(affine.then_some(&w), affine.then_some(&b), eps)?,
    };
    let cc = tape.var_no_grad(&t(c.to_vec(), shape));
    y.mul(&cc)?.sum(None)
}

fn assert_finite(v: &[f32], ctx: &str) {
    assert!(v.iter().all(|a| a.is_finite()), "{ctx}: 非有限 {v:?}");
}

fn assert_req2(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(x), f64::from(y)),
            "{ctx}[{i}]: {x} vs {y}"
        );
    }
}

fn check_three_stage(ctx: &str, kind: Kind, eps: f32, affine: bool, x0: &[f32], shape: &[usize]) {
    let n = numel(shape);
    let c = pattern(n, 1.0);
    let v = pattern(n, 9.0);

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build_norm(&tape, &x, kind, eps, affine, &c, shape).expect("forward");
    let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    assert_finite(&plain, &format!("{ctx}: 1 階"));

    let cg = tape
        .backward_create_graph(&loss, &child)
        .unwrap_or_else(|e| panic!("{ctx}: backward_create_graph 失敗 {e:?}"));
    let first = dense(cg.first_order().get(&x).unwrap().unwrap());
    assert_eq!(first, plain, "{ctx}: first_order は bit 同一");
    let gx = cg.grad(&x).unwrap().unwrap();
    let gx_v = dense(&gx.to_tensor());
    assert_finite(&gx_v, &format!("{ctx}: cg.grad"));
    assert_req2(&gx_v, &plain, &format!("{ctx}: cg.grad vs 1 階"));

    // 2 段目: child 上で sum(gx ⊙ v) を微分（HVP）。
    let cx = cg.child_var(&x).unwrap().unwrap();
    let vc = child.var_no_grad(&t(v, shape));
    let prod = gx.mul(&vc).unwrap().sum(None).unwrap();
    let hv = dense(child.backward(&prod).unwrap().get(&cx).unwrap().unwrap());
    assert_finite(&hv, &format!("{ctx}: HVP（child.backward）"));

    // 3 段目: 孫テープ再生。
    let grand = Tape::new_with_ops(common::naive_ops());
    let cg2 = child
        .backward_create_graph(&prod, &grand)
        .unwrap_or_else(|e| panic!("{ctx}: 孫テープ再生失敗 {e:?}"));
    let hv_grand = dense(&cg2.grad(&cx).unwrap().unwrap().to_tensor());
    assert_finite(&hv_grand, &format!("{ctx}: 孫テープ再生"));
    assert_req2(&hv_grand, &hv, &format!("{ctx}: 孫 vs child HVP"));
}

fn run_vectors(kind: Kind, name: &str, vectors: &[(&str, f32, Vec<f32>)]) {
    for (label, eps, row) in vectors {
        let shape = [1usize, row.len()];
        for affine in [false, true] {
            check_three_stage(
                &format!("{name}/{label}/affine={affine}"),
                kind,
                *eps,
                affine,
                row,
                &shape,
            );
        }
    }
}

#[test]
fn layer_norm_required_vectors_three_stages_finite() {
    run_vectors(
        Kind::Layer,
        "ln",
        &[
            ("centering-roundoff", 1e-5, vec![16777216.0, 16777218.0]),
            ("large-1e20", 1e-5, vec![1e20, 2e20, -1e20]),
            ("large-single-1e20", 1e-5, vec![1e20, 1.0, 0.5]),
            ("constant-1e30", 1e-5, vec![1e30, 1e30]),
            ("all-zero", 1e-5, vec![0.0, 0.0, 0.0]),
            ("small-scale", 1e-5, vec![1e-30, 3e-30]),
        ],
    );
}

#[test]
fn rms_norm_required_vectors_three_stages_finite() {
    run_vectors(
        Kind::Rms,
        "rms",
        &[
            ("large-1e20", 1e-5, vec![1e20, 2e20, -1e20]),
            ("small-scale", 1e-5, vec![1e-30, 1e-30]),
            ("constant-1e30", 1e-5, vec![1e30, 1e30]),
            ("all-zero", 1e-5, vec![0.0, 0.0, 0.0]),
            ("roundoff", 1e-5, vec![16777216.0, 16777218.0]),
        ],
    );
}

#[test]
fn rms_norm_small_scale_row_keeps_rstd_316() {
    // x=[1e-30,1e-30]・eps=1e-5 の 1 階 dx は dy·rstd ≈ 316.23 倍の項を含む。cg.grad が一致する
    // ことで、子テープ側の s=√eps 規則（eps/s² の overflow 回避）が効いていることを確認する。
    let shape = [1usize, 2];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e-30, 1e-30], &shape));
    let loss = build_norm(&tape, &x, Kind::Rms, 1e-5, false, &[1.0, 2.0], &shape).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gx = dense(&cg.grad(&x).unwrap().unwrap().to_tensor());
    let plain = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
    assert_req2(&gx, &plain, "small-scale cg.grad");
    assert!(
        plain.iter().any(|v| v.abs() > 100.0),
        "rstd≈316 の項: {plain:?}"
    );
}

// --- 5. 入口検査の拒否（child を空のまま保つ） --------------------------------------------

fn assert_rejected(kind: Kind, eps: f32, x0: &[f32], shape: &[usize]) {
    let c = pattern(numel(shape), 1.0);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build_norm(&tape, &x, kind, eps, false, &c, shape).expect("forward");
    let err = tape
        .backward_create_graph(&loss, &child)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{x0:?}: {err:?}");
    assert!(child.is_empty(), "{x0:?}: 拒否時 child は空のまま");
}

#[test]
fn layer_norm_rejects_unsafe_inputs_and_keeps_child_empty() {
    assert_rejected(Kind::Layer, 1e-5, &[3e38, -3e38], &[1, 2]); // (b) x_i − x_0 が f32 で非有限
    assert_rejected(Kind::Layer, 1e-5, &[-2e38, 1e38, 1e38], &[1, 3]); // (c) Mean の部分和 overflow
    assert_rejected(Kind::Layer, 0.0, &[2.0, 2.0, 2.0], &[1, 3]); // (d) eps=0 の定数行
    assert_rejected(Kind::Layer, 1e-5, &[1.0, f32::NAN, 2.0], &[1, 3]); // (a) 非有限
}

#[test]
fn rms_norm_rejects_unsafe_inputs_and_keeps_child_empty() {
    assert_rejected(Kind::Rms, 0.0, &[0.0, 0.0, 0.0], &[1, 3]); // (d') eps=0 の全ゼロ行
    assert_rejected(Kind::Rms, 1e-5, &[1.0, f32::INFINITY], &[1, 2]); // (a) 非有限
}

#[test]
fn eps_zero_nondegenerate_rows_are_accepted() {
    check_three_stage(
        "ln/eps0",
        Kind::Layer,
        0.0,
        true,
        &[1.0, 2.5, -0.5],
        &[1, 3],
    );
    check_three_stage("rms/eps0", Kind::Rms, 0.0, true, &[1.0, 2.5, -0.5], &[1, 3]);
}

#[test]
fn neighbouring_unsupported_norm_op_stays_fail_closed() {
    // LocalResponseNorm は §21.2 の C 区分で fail-closed のまま（BatchNorm は #3079 が反転する）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(pattern(12, 2.0), &[1, 3, 2, 2]));
    let y = local_response_norm(&x, 3, 1e-4, 0.75, 1.0).unwrap();
    let loss = y.sum(None).unwrap();
    let err = tape
        .backward_create_graph(&loss, &child)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)));
    assert!(child.is_empty());
}

// --- 6. 総括的な適用判定（PR #3346 のレビュー指摘の全類型）: 受理なら有限・一致、拒否なら child 空 ---
//
// 指摘はいずれも「有限な入力・upstream で、構造（cg.grad・二階の逆伝播・孫テープ再生）の
// 中間値が f32 を超える」類型だった（g=3e38 の縮約、hidden·g 規模の cotangent 和、低分散行の
// rstd 増幅、w=0 で g·w 検査を素通りする dw 経路）。経路ごとの個別条件ではなく、構造の各段の
// 上界を一括で判定する（`create_graph_norm::check_structure_stages`。決定記録 §27.3）。ここでは
// 判定の契約「受理された入力は 1 階・cg.grad・二階（seed の各要素の絶対値 ≤ 1）・孫テープ再生が
// 有限かつ一致し、拒否された入力は `Err(Backward)` で child が空」を正のプローブで確かめる。

const PROBE_EPS: f32 = 1e-5;

/// 正のプローブの結果。
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Accepted,
    Rejected,
}

/// 二階の seed を掛ける 1 階勾配（`cg.grad` の対象）。
#[derive(Clone, Copy, Debug)]
enum Wrt {
    X,
    W,
    B,
}

/// `w`／`b` の扱い。`Tracked` は追跡ありの葉、`Const` は追跡なしの定数。
#[derive(Clone, Debug)]
enum Affine {
    Tracked(Vec<f32>),
    Const(Vec<f32>),
}

/// 正のプローブの 1 ケース。損失は `sum(norm(x; w, b) ⊙ c)`（`c` が upstream `g`）、
/// `quadratic` なら `sum(norm ⊙ norm ⊙ c)`（upstream `g = 2c·norm` が入力に依存する）。二階は
/// `sum(cg.grad(wrt) ⊙ seed)` を child 上で逆伝播する。
#[derive(Clone, Debug)]
struct Probe {
    kind: Kind,
    quadratic: bool,
    x0: Vec<f32>,
    shape: Vec<usize>,
    c: Vec<f32>,
    w: Option<Affine>,
    b: Option<Affine>,
    wrt: Wrt,
    seed: Vec<f32>,
}

fn affine_var<'t>(tape: &'t Tape, a: &Option<Affine>, h: usize) -> Option<Var<'t>> {
    a.as_ref().map(|a| match a {
        Affine::Tracked(v) => tape.var(&t(v.clone(), &[h])),
        Affine::Const(v) => tape.var_no_grad(&t(v.clone(), &[h])),
    })
}

/// 判定の契約を 1 ケースについて確かめ、受理／拒否を返す。
fn probe(p: &Probe) -> Outcome {
    let ctx = format!("{p:?}");
    let h = *p.shape.last().unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(p.x0.clone(), &p.shape));
    let w = affine_var(&tape, &p.w, h);
    let b = affine_var(&tape, &p.b, h);
    let y = match p.kind {
        Kind::Rms => x.rms_norm(w.as_ref(), PROBE_EPS),
        Kind::Layer => x.layer_norm(w.as_ref(), b.as_ref(), PROBE_EPS),
    }
    .expect("forward");
    let cc = tape.var_no_grad(&t(p.c.clone(), &p.shape));
    let yc = if p.quadratic { y.mul(&y).unwrap() } else { y };
    let loss = yc.mul(&cc).unwrap().sum(None).unwrap();
    let (target, target_shape) = match p.wrt {
        Wrt::X => (x, p.shape.clone()),
        Wrt::W => (w.expect("wrt=W には w が要る"), vec![h]),
        Wrt::B => (b.expect("wrt=B には b が要る"), vec![h]),
    };

    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: {e:?}");
            assert!(child.is_empty(), "{ctx}: 拒否時に child が空へ戻る");
            return Outcome::Rejected;
        }
    };

    // 1 階と cg.grad。
    let plain = dense(tape.backward(&loss).unwrap().get(&target).unwrap().unwrap());
    assert_finite(&plain, &format!("{ctx}: 1 階"));
    let gt = cg.grad(&target).unwrap().unwrap();
    let gt_v = dense(&gt.to_tensor());
    assert_finite(&gt_v, &format!("{ctx}: cg.grad"));
    assert_req2(&gt_v, &plain, &format!("{ctx}: cg.grad vs 1 階"));

    // 線形損失の db = Σ_rows g は追跡対象に依存しない（二階は恒等ゼロで、seed 付きの和は
    // 追跡なしになり child.backward が拒否する）。db 構造の二階は quadratic 側で確かめる。
    if !p.quadratic && matches!(p.wrt, Wrt::B) {
        return Outcome::Accepted;
    }

    // 二階: 追跡ありの境界ノード（x・w・b の写し）すべてへの勾配が有限。
    let sv = child.var_no_grad(&t(p.seed.clone(), &target_shape));
    let prod = gt.mul(&sv).unwrap().sum(None).unwrap();
    let mut leaves = vec![cg.child_var(&x).unwrap().unwrap()];
    for v in [w, b].into_iter().flatten() {
        // 追跡なしの定数（`Affine::Const`）は写しを持たない。
        if let Ok(Some(cv)) = cg.child_var(&v) {
            leaves.push(cv);
        }
    }
    let grads = child
        .backward(&prod)
        .unwrap_or_else(|e| panic!("{ctx}: child.backward の失敗 {e:?}"));
    let hv: Vec<Option<Vec<f32>>> = leaves
        .iter()
        .map(|l| grads.get(l).unwrap().map(dense))
        .collect();
    for v in hv.iter().flatten() {
        assert_finite(v, &format!("{ctx}: 二階（child.backward）"));
    }

    // 孫テープ再生: 同じ二階勾配が有限かつ child と一致。child 上の二階のグラフが正規化の写し
    // （quadratic の `g = 2c·norm` 等）を含むと、孫テープ側でも同じ判定が二階の cotangent を
    // upstream として走る。そこで拒否された場合も契約どおり `Err(Backward)` で grand が空になる。
    let grand = Tape::new_with_ops(common::naive_ops());
    let cg2 = match child.backward_create_graph(&prod, &grand) {
        Ok(cg2) => cg2,
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: 孫 {e:?}");
            assert!(
                grand.is_empty(),
                "{ctx}: 孫テープ再生の拒否時に grand が空へ戻る"
            );
            return Outcome::Accepted;
        }
    };
    for (l, h1) in leaves.iter().zip(&hv) {
        if let (Some(g2), Some(h1)) = (cg2.grad(l).unwrap(), h1) {
            let v = dense(&g2.to_tensor());
            assert_finite(&v, &format!("{ctx}: 孫テープ再生"));
            assert_req2(&v, h1, &format!("{ctx}: 孫 vs child"));
        }
    }
    Outcome::Accepted
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn alternating(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn scaled(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|a| a * s).collect()
}

fn row_probe(kind: Kind, x0: Vec<f32>, c: Vec<f32>, seed: Vec<f32>) -> Probe {
    let shape = vec![1, x0.len()];
    Probe {
        kind,
        quadratic: false,
        x0,
        shape,
        c,
        w: None,
        b: None,
        wrt: Wrt::X,
        seed,
    }
}

/// レビューで指摘された入力そのもの（または同型の LayerNorm 版）。いずれも構造の中間値が f32 を
/// 超えるので拒否され、child は空へ戻る。
fn review_examples() -> Vec<(&'static str, Probe)> {
    let alt4 = alternating(4);
    let lowvar_ln = vec![1.0f32, 1.001, 1.0, 1.001];
    let lowvar_rms = vec![1e-3f32; 4];
    vec![
        (
            "一巡目: RmsNorm x=[1,1]・g=[3e38,3e38] の Mean(dxhat⊙xhat)",
            row_probe(Kind::Rms, vec![1.0, 1.0], vec![3e38, 3e38], ones(2)),
        ),
        (
            "二巡目: LayerNorm x=[1,-1,1,-1]・g=±1e38・v=[1,-1,1,-1] の gs 復元",
            row_probe(Kind::Layer, alt4.clone(), scaled(&alt4, 1e38), alt4.clone()),
        ),
        (
            "Bugbot: LayerNorm 低分散行の rstd 増幅（g=±1e36）",
            row_probe(Kind::Layer, lowvar_ln, scaled(&alt4, 1e36), ones(4)),
        ),
        (
            "Bugbot: RmsNorm 低分散行の rstd 増幅（g=±1e36）",
            row_probe(Kind::Rms, lowvar_rms, scaled(&alt4, 1e36), ones(4)),
        ),
        (
            "四巡目: RmsNorm x=[0.75×4]・w=0（追跡）・g=8e37 の dw 経路",
            Probe {
                w: Some(Affine::Tracked(vec![0.0; 4])),
                wrt: Wrt::W,
                ..row_probe(Kind::Rms, vec![0.75; 4], vec![8e37; 4], ones(4))
            },
        ),
        (
            "四巡目（:539 同型）: LayerNorm x=[1,-1,1,-1]・w=0（追跡）・g=8e37 の dw 経路",
            Probe {
                w: Some(Affine::Tracked(vec![0.0; 4])),
                wrt: Wrt::W,
                ..row_probe(Kind::Layer, alt4.clone(), vec![8e37; 4], ones(4))
            },
        ),
        (
            "先回り: LayerNorm [3,4]・b（追跡）・g=1e38 の db 行をまたぐ和",
            Probe {
                kind: Kind::Layer,
                quadratic: true,
                x0: pattern(12, 7.0),
                shape: vec![3, 4],
                c: vec![1e38; 12],
                w: Some(Affine::Tracked(pattern(4, 3.0))),
                b: Some(Affine::Tracked(pattern(4, 5.0))),
                wrt: Wrt::B,
                seed: ones(4),
            },
        ),
    ]
}

#[test]
fn review_examples_are_rejected_with_empty_child() {
    for (label, p) in review_examples() {
        assert_eq!(probe(&p), Outcome::Rejected, "{label}");
    }
}

#[test]
fn review_examples_at_moderate_scale_are_accepted_and_finite() {
    // 同じ入力で upstream を 1e30 倍へ下げると判定を通り、全段が有限かつ一致する
    // （拒否が過剰に広くないことの正のプローブ）。
    for (label, mut p) in review_examples() {
        let peak = p.c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        p.c = scaled(&p.c, 1e30 / peak);
        assert_eq!(probe(&p), Outcome::Accepted, "{label}（1e30 倍）");
    }
}

/// 掃引用の 1 行ぶんの `x`（幅 `h`）。定数行・交互符号・低分散・小スケール・大振幅を含む。
fn sweep_rows(h: usize) -> Vec<Vec<f32>> {
    vec![
        vec![0.75; h],
        alternating(h),
        (0..h)
            .map(|i| if i % 2 == 0 { 1.0 } else { 1.001 })
            .collect(),
        vec![1e-3; h],
        (0..h)
            .map(|i| [1e20, 2e20, -1e20].get(i).copied().unwrap_or(0.5))
            .collect(),
    ]
}

/// kind ごとの affine 構成と、二階の seed を掛けられる微分対象。
fn sweep_affines(kind: Kind, w0: &[f32]) -> Vec<(Option<Affine>, Option<Affine>, Vec<Wrt>)> {
    let mut v = vec![
        (None, None, vec![Wrt::X]),
        (
            Some(Affine::Tracked(w0.to_vec())),
            None,
            vec![Wrt::X, Wrt::W],
        ),
        (Some(Affine::Const(w0.to_vec())), None, vec![Wrt::X]),
    ];
    if let Kind::Layer = kind {
        v.push((
            Some(Affine::Tracked(w0.to_vec())),
            Some(Affine::Tracked(pattern(w0.len(), 5.0))),
            vec![Wrt::X, Wrt::W, Wrt::B],
        ));
    }
    v
}

#[test]
fn sweep_every_structure_path_is_finite_or_rejected() {
    // dx／dw／db 構造・weight なし／あり（0 と非ゼロ）／定数・bias・幅 4／32・行数 1／3・
    // upstream の大きさ・seed の符号パターンを掛け合わせ、どのケースも「受理なら全段が有限で一致」
    // 「拒否なら child 空」を満たすことを確かめる。個別経路の見落としがあれば受理側の有限性検査が
    // 落ちる（幅 32 は行方向和が forward の各段より大きくなり、cotangent 側の判定が効く範囲）。
    let scales = [1.0f32, 1e30, 1e36, 3e36, 1e37, 3e37, 8e37, 1e38, 3e38];
    // quadratic（g = 2c·norm）は上限を下げる。判定が守るのは正規化の構造と、そこから g へ出る
    // cotangent までで、g を作る上流の Mul の逆伝播（cotangent × 2c）と、その先の写しの 1 階 VJP
    // は対象外（決定記録 §27.3）。真の二階微分は 2c·rstd²（低分散行・定数行で rstd ≈ 316）規模に
    // なり、c ≳ 1e33 では真値そのものが f32 を超えて inf が正解になるため、有限性を問える範囲に限る。
    let quadratic_scales = [1.0f32, 1e20, 1e30];
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (kind, h) in [Kind::Rms, Kind::Layer]
        .into_iter()
        .flat_map(|k| [(k, 4usize), (k, 32)])
    {
        for w0 in [vec![0.0f32; h], pattern(h, 3.0)] {
            for (w, b, wrts) in sweep_affines(kind, &w0) {
                for row in sweep_rows(h) {
                    // 行をまたぐ和（dw／db の Sum(0)）は幅 4 で確かめ、幅 32 は実行時間を抑えて 1 行のみ。
                    let row_counts: &[usize] = if h == 4 { &[1, 3] } else { &[1] };
                    for &rows in row_counts {
                        let x0: Vec<f32> = (0..rows)
                            .flat_map(|r| row.iter().map(move |v| v * (1.0 + 0.25 * r as f32)))
                            .collect();
                        let n = x0.len();
                        let modes = scales
                            .iter()
                            .map(|&s| (s, false))
                            .chain(quadratic_scales.iter().map(|&s| (s, true)));
                        for (scale, quadratic) in modes {
                            for cpat in [ones(n), alternating(n)] {
                                for &wrt in &wrts {
                                    let sn = if let Wrt::X = wrt { n } else { h };
                                    for seed in [ones(sn), alternating(sn)] {
                                        let p = Probe {
                                            kind,
                                            quadratic,
                                            x0: x0.clone(),
                                            shape: vec![rows, h],
                                            c: scaled(&cpat, scale),
                                            w: w.clone(),
                                            b: b.clone(),
                                            wrt,
                                            seed,
                                        };
                                        match probe(&p) {
                                            Outcome::Accepted => accepted += 1,
                                            Outcome::Rejected => rejected += 1,
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // 判定が全拒否・全受理に退化していないこと。
    assert!(
        accepted > 0 && rejected > 0,
        "accepted={accepted} rejected={rejected}"
    );
}

// --- 7. 拒否時の child の空性と再試行（PR #3346 三巡目指摘） --------------------------------

fn layer_hvp_with_upstream(c: Vec<f32>) -> Result<Vec<f32>, AutodiffError> {
    let shape = [1usize, 4];
    let x0 = vec![1.0f32, -1.0, 1.0, -1.0];
    let v = vec![1.0f32, -1.0, 1.0, -1.0];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0, &shape));
    let loss = build_norm(&tape, &x, Kind::Layer, 1e-5, false, &c, &shape).unwrap();
    let cg = tape.backward_create_graph(&loss, &child)?;
    let gx = cg.grad(&x).unwrap().unwrap();
    let cx = cg.child_var(&x).unwrap().unwrap();
    let vc = child.var_no_grad(&t(v, &shape));
    let prod = gx.mul(&vc).unwrap().sum(None).unwrap();
    Ok(dense(
        child.backward(&prod).unwrap().get(&cx).unwrap().unwrap(),
    ))
}

#[test]
fn layer_norm_upstream_1e38_is_rejected_instead_of_nan_hvp() {
    // x=[1,-1,1,-1]・g=±1e38。判定導入前は HVP が NaN になった（真の HVP は約 ±3e33 で有限）。
    let err = layer_hvp_with_upstream(vec![1e38, -1e38, 1e38, -1e38]).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
}

#[test]
fn layer_norm_moderate_upstream_hvp_stays_finite() {
    let hv = layer_hvp_with_upstream(vec![1e30, -1e30, 1e30, -1e30]).expect("受理される範囲");
    assert_finite(&hv, "g=1e30: HVP");
}

/// `x` と upstream `c`（shape `[1, h]`）で create_graph を試み、HVP を返す。
fn try_build(
    kind: Kind,
    x0: Vec<f32>,
    c: Vec<f32>,
    child: &Tape,
) -> Result<Vec<f32>, AutodiffError> {
    let shape = [1usize, x0.len()];
    let v = vec![1.0f32; x0.len()];
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0, &shape));
    let loss = build_norm(&tape, &x, kind, 1e-5, false, &c, &shape).unwrap();
    let cg = tape.backward_create_graph(&loss, child)?;
    let gx = cg.grad(&x).unwrap().unwrap();
    let cx = cg.child_var(&x).unwrap().unwrap();
    let vc = child.var_no_grad(&t(v, &shape));
    let prod = gx.mul(&vc).unwrap().sum(None).unwrap();
    Ok(dense(
        child.backward(&prod).unwrap().get(&cx).unwrap().unwrap(),
    ))
}

#[test]
fn rejected_upstream_leaves_child_empty_and_allows_retry() {
    for kind in [Kind::Layer, Kind::Rms] {
        let child = Tape::new_with_ops(common::naive_ops());
        let x0 = vec![1.0f32, -1.0, 1.0, -1.0];
        // upstream 拒否（build_mirror・中心化ノードの後で判明する経路）。
        let err = try_build(kind, x0.clone(), vec![1e38, -1e38, 1e38, -1e38], &child).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty(), "拒否後に child が空へ戻る");
        // 非有限 upstream でも同様。
        let err = try_build(kind, x0.clone(), vec![f32::NAN, 1.0, 1.0, 1.0], &child).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty());
        // 同じ child で再試行できる。
        let hv = try_build(kind, x0, vec![1.0, -1.0, 1.0, -1.0], &child).expect("再試行は成功");
        assert_finite(&hv, "retry");
    }
}
