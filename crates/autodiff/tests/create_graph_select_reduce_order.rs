//! 選択縮約 3 Op（`OrderSelect`〈`kthvalue`／`median(Some(d))`／`median_with_indices`〉・
//! `MedianAll`〈`median(None)`〉・`Quantile`）の `create_graph` 対応（イシュー #3790・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §34）の受け入れ条件を確かめる統合テスト。
//!
//! - Hessian を独立な有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で突合する。選択だけでは
//!   二階が 0 になるため、選択の前後に二乗・三乗を挟む合成形を使う。入力は順位が切り替わらない
//!   （タイなし・最小間隔が摂動幅の 2 倍以上）値に限る。
//! - `first_order()` と `cg.grad` が 1 階 `Tape::backward` と bit 一致すること。
//! - 孫テープ（`jvp_create_graph`）が独立な方向微分と REQ-2 で一致すること。
//! - 回帰ベクトル（タイ・`±0`・NaN lane・`3e38` 上流・要素数 0）。
//! - 隣接する fail-closed（`Op::Custom` との合成）が拒否され `child` が空に保たれること。
//!
//! GPU 実機の parity は `crates/facade/tests/create_graph_gpu_parity.rs` に登録し、実測は別イシュー
//! （#3856・#3858）。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
use fandhe_ai_tensor_core::{QuantileInterpolation, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn flat_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        out[d] = idx % shape[d];
        idx /= shape[d];
    }
    out
}

fn shp(v: &Var<'_>) -> Vec<usize> {
    v.value().shape().to_vec()
}

fn dense(v: impl std::borrow::Borrow<Tensor<f32>>) -> Vec<f32> {
    let v = v.borrow();
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
}

fn extract_scalar<'c>(
    v: &Var<'c>,
    shape: &[usize],
    flat_idx: usize,
) -> Result<Var<'c>, AutodiffError> {
    let idx = unravel(flat_idx, shape);
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

fn finite_diff_hessian<F>(build: F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = flat_len(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        (0..n)
            .map(|i| dx.get(&unravel(i, shape)).unwrap() as f64)
            .collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let gp = grad_at(&xp);
        let gm = grad_at(&xm);
        for j in 0..n {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

fn analytic_hessian<F>(build: F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = flat_len(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).unwrap().expect("1 階勾配");
    let cx = cg.child_var(&x).unwrap().expect("写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).unwrap();
        let row_grads = child.backward(&gj).expect("2 階");
        if let Some(row) = row_grads.get(&cx).unwrap() {
            for i in 0..n {
                hessian[j][i] = row.get(&unravel(i, shape)).unwrap() as f64;
            }
        }
    }
    hessian
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>], ctx: &str) {
    assert_eq!(analytic.len(), numeric.len());
    for (j, (ra, rn)) in analytic.iter().zip(numeric).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
}

fn hessian_case<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let analytic = analytic_hessian(build, x0, shape);
    let numeric = finite_diff_hessian(build, x0, shape, 0.01);
    assert_hessian_close(&analytic, &numeric, ctx);
}

/// 値の重複がなく最小間隔が 0.3 の正の入力（`x ⊙ x` でも順位が保たれる）。
fn distinct_pos(n: usize) -> Vec<f32> {
    let p = [29usize, 53, 101, 211, 401]
        .into_iter()
        .find(|&p| p > n)
        .unwrap();
    (0..n)
        .map(|i| ((i * 37 + 5) % p) as f32 * 0.3 + 0.5)
        .collect()
}

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let shape = shp(y);
    let c = tape.var_no_grad(&t(common::det_seq(flat_len(&shape), 0.7), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

const INTERPS: [QuantileInterpolation; 5] = [
    QuantileInterpolation::Linear,
    QuantileInterpolation::Lower,
    QuantileInterpolation::Higher,
    QuantileInterpolation::Midpoint,
    QuantileInterpolation::Nearest,
];

// --- Hessian と有限差分の突合 ---------------------------------------------------------

#[test]
fn hessian_kthvalue_matches_finite_difference() {
    let shape = [3usize, 4];
    let x0 = distinct_pos(12);
    for k in [1usize, 2, 4] {
        for dim in [0usize, 1] {
            let k = k.min(shape[dim]);
            hessian_case(
                &format!("kthvalue k={k} dim={dim}"),
                move |tp, x| {
                    let c = x.mul(x)?.mul(x)?;
                    quad(tp, &c.kthvalue(k, dim)?.0)
                },
                &x0,
                &shape,
            );
        }
    }
}

#[test]
fn hessian_median_dim_matches_finite_difference() {
    let shape = [2usize, 3, 2];
    let x0 = distinct_pos(12);
    for dim in 0..3 {
        hessian_case(
            &format!("median dim={dim}"),
            move |tp, x| quad(tp, &x.mul(x)?.median(Some(dim))?),
            &x0,
            &shape,
        );
        hessian_case(
            &format!("median_with_indices dim={dim}"),
            move |tp, x| quad(tp, &x.mul(x)?.median_with_indices(dim)?.0),
            &x0,
            &shape,
        );
    }
}

#[test]
fn hessian_median_all_matches_finite_difference() {
    let shape = [3usize, 3];
    let x0 = distinct_pos(9);
    hessian_case(
        "median all (x*x)",
        |tp, x| quad(tp, &x.mul(x)?.median(None)?),
        &x0,
        &shape,
    );
    hessian_case(
        "median all (x)",
        |tp, x| quad(tp, &x.median(None)?),
        &x0,
        &shape,
    );
}

#[test]
fn hessian_quantile_matches_finite_difference_for_all_interpolations() {
    let shape = [3usize, 5];
    let x0 = distinct_pos(15);
    for interp in INTERPS {
        for q in [0.3f32, 0.5, 1.0] {
            hessian_case(
                &format!("quantile None q={q} {interp:?}"),
                move |tp, x| quad(tp, &x.mul(x)?.quantile(q, None, interp)?),
                &x0,
                &shape,
            );
            for dim in [0usize, 1] {
                hessian_case(
                    &format!("quantile dim={dim} q={q} {interp:?}"),
                    move |tp, x| quad(tp, &x.mul(x)?.quantile(q, Some(dim), interp)?),
                    &x0,
                    &shape,
                );
            }
        }
    }
}

// --- 1 階との bit 一致 --------------------------------------------------------------

/// `first_order()` と `cg.grad` がいずれも素の `Tape::backward` と bit 一致する。
fn assert_first_order_bit_identical<F>(build: F, x0: &[f32], shape: &[usize], ctx: &str)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let reference = tape.backward(&loss).unwrap();
    let want = dense(reference.get(&x).unwrap().expect("x は到達"));
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let first = dense(cg.first_order().get(&x).unwrap().expect("x は到達"));
    assert_eq!(bits(&first), bits(&want), "{ctx}: first_order()");
    let got = dense(cg.grad(&x).unwrap().expect("x は到達").to_tensor());
    assert_eq!(bits(&got), bits(&want), "{ctx}: cg.grad");
}

/// 桁落ちする上流 `c`（f32 逐次和では 1 階と一致しない列）を掛けた損失。
fn cancel_loss<'t>(tp: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let n = flat_len(&shp(y));
    let data: Vec<f32> = (0..n)
        .map(|i| [1.0e8f32, 1.0, -1.0e8, 1.0, 3.0][i % 5])
        .collect();
    y.mul(&tp.var_no_grad(&t(data, &shp(y))))?.sum(None)
}

#[test]
fn value_is_bit_identical_to_first_order() {
    let shape = [3usize, 5];
    let x0 = common::det_seq(15, 0.3);
    assert_first_order_bit_identical(
        |tp, x| cancel_loss(tp, &x.kthvalue(2, 1)?.0),
        &x0,
        &shape,
        "kthvalue",
    );
    assert_first_order_bit_identical(
        |tp, x| cancel_loss(tp, &x.median(Some(0))?),
        &x0,
        &shape,
        "median dim",
    );
    assert_first_order_bit_identical(
        |tp, x| cancel_loss(tp, &x.median(None)?),
        &x0,
        &shape,
        "median all",
    );
    for interp in INTERPS {
        for dim in [None, Some(0), Some(1)] {
            assert_first_order_bit_identical(
                |tp, x| cancel_loss(tp, &x.quantile(0.37, dim, interp)?),
                &x0,
                &shape,
                &format!("quantile dim={dim:?} {interp:?}"),
            );
        }
    }
}

// --- 回帰ベクトル ---------------------------------------------------------------------

#[test]
fn median_all_tie_distributes_evenly_in_second_order() {
    // [2,2,2,5] の下側中央値は 2。等しい 3 要素へ 1/3 ずつ。L = m^2 の Hessian は 3 要素ブロックが 2/9。
    let x0 = [2.0f32, 2.0, 2.0, 5.0];
    let h = analytic_hessian(
        |_tp, x| {
            let m = x.median(None)?;
            m.mul(&m)?.sum(None)
        },
        &x0,
        &[4],
    );
    for j in 0..4 {
        for i in 0..4 {
            let want = if i < 3 && j < 3 { 2.0 / 9.0 } else { 0.0 };
            assert!(common::req2_close(h[j][i], want), "h[{j}][{i}]={}", h[j][i]);
        }
    }
}

#[test]
fn kthvalue_tie_uses_stable_order() {
    // [1,1,0] の 2 番目は安定順で index 0。L = y^2 の Hessian は h[0][0] = 2 のみ。
    let x0 = [1.0f32, 1.0, 0.0];
    let h = analytic_hessian(
        |_tp, x| {
            let y = x.kthvalue(2, 0)?.0;
            y.mul(&y)?.sum(None)
        },
        &x0,
        &[3],
    );
    for j in 0..3 {
        for i in 0..3 {
            let want = if i == 0 && j == 0 { 2.0 } else { 0.0 };
            assert!(common::req2_close(h[j][i], want), "h[{j}][{i}]={}", h[j][i]);
        }
    }
}

#[test]
fn signed_zero_and_nan_lanes_match_first_order() {
    assert_first_order_bit_identical(
        |_tp, x| x.median(None)?.sum(None),
        &[0.0, -0.0, 1.0, 0.0],
        &[4],
        "median ±0",
    );
    // NaN を含む lane: quantile は最後の NaN へ、median(None) は NaN 同士へ均等分配。
    let nan_x = [1.0f32, f32::NAN, 3.0, f32::NAN, 5.0, 4.0];
    for interp in INTERPS {
        assert_first_order_bit_identical(
            |_tp, x| x.quantile(0.5, Some(1), interp)?.sum(None),
            &nan_x,
            &[2, 3],
            &format!("quantile NaN {interp:?}"),
        );
    }
    assert_first_order_bit_identical(
        |_tp, x| x.median(None)?.sum(None),
        &nan_x,
        &[6],
        "median NaN",
    );
}

#[test]
fn quantile_single_and_pair_lanes_both_covered() {
    // q=0.5, n=5 の Linear は rank=2 で整数（Single）、q=0.3 は Pair。
    let x0 = distinct_pos(5);
    for q in [0.5f32, 0.3] {
        hessian_case(
            &format!("single/pair q={q}"),
            move |tp, x| {
                quad(
                    tp,
                    &x.mul(x)?.quantile(q, None, QuantileInterpolation::Linear)?,
                )
            },
            &x0,
            &[5],
        );
    }
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|a| a.is_finite())
}

fn three_stage_finite<F>(build: F, x0: &[f32], shape: &[usize], ctx: &str)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape.backward_create_graph(&loss, &child).expect("cg");
    let first = dense(cg.first_order().get(&x).unwrap().expect("x"));
    assert!(all_finite(&first), "{ctx}: 1 階");
    let gx = cg.grad(&x).unwrap().expect("x");
    assert!(all_finite(&dense(gx.to_tensor())), "{ctx}: cg.grad");
    let seed = child.var_no_grad(&t(common::det_seq(flat_len(shape), 0.2), shape));
    let s = gx.mul(&seed).unwrap().sum(None).unwrap();
    let grads = child.backward(&s).expect("二階");
    let cx = cg.child_var(&x).unwrap().expect("写し");
    if let Some(d) = grads.get(&cx).unwrap() {
        assert!(all_finite(&dense(d)), "{ctx}: 二階");
    }
}

/// `sum(y ⊙ c) + sum(y ⊙ y)`（`c = 3e38`）。上流が f32 上限近傍でも 3 段が有限であること。
fn big_loss<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let c = tape.var_no_grad(&t(vec![3.0e38f32; flat_len(&shp(y))], &shp(y)));
    y.mul(&c)?.sum(None)?.add(&y.mul(y)?.sum(None)?)
}

#[test]
fn three_stages_are_finite_with_large_upstream() {
    let x0 = distinct_pos(15);
    three_stage_finite(
        |tp, x| big_loss(tp, &x.kthvalue(2, 1)?.0),
        &x0,
        &[3, 5],
        "kthvalue",
    );
    three_stage_finite(
        |tp, x| big_loss(tp, &x.median(None)?),
        &x0,
        &[3, 5],
        "median all",
    );
    for interp in INTERPS {
        three_stage_finite(
            |tp, x| big_loss(tp, &x.quantile(0.3, Some(1), interp)?),
            &x0,
            &[3, 5],
            &format!("quantile {interp:?}"),
        );
    }
}

#[test]
fn median_all_of_empty_input_does_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 3]));
    let Ok(y) = x.median(None) else {
        return;
    };
    let loss = y.sum(None).unwrap();
    // 型付きエラーまたは空の勾配のどちらでもよいが panic しない。
    if let Ok(cg) = tape.backward_create_graph(&loss, &child)
        && let Some(g) = cg.grad(&x).unwrap()
    {
        assert_eq!(g.to_tensor().shape(), &[0, 3]);
    }
}

// --- 3 段（jvp_create_graph の孫テープ） ---------------------------------------------

fn jvp_case<F>(ctx: &str, build: F, p0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let tangent: Vec<f32> = (0..p0.len())
        .map(|i| ((i * 3 + 1) % 7) as f32 * 0.1 - 0.3)
        .collect();
    // 独立な参照: 1 階 `Tape::backward` の勾配と接ベクトルの f64 内積。
    let numeric = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let p = tape.var(&t(p0.to_vec(), shape));
        let loss = build(&tape, &p).unwrap();
        let g = dense(tape.backward(&loss).unwrap().get(&p).unwrap().expect("p"));
        g.iter()
            .zip(&tangent)
            .map(|(a, v)| f64::from(*a) * f64::from(*v))
            .sum::<f64>()
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let p = tape.var(&t(p0.to_vec(), shape));
    let loss = build(&tape, &p).unwrap();
    let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
        &tape,
        &loss,
        &p,
        &t(tangent, shape),
        &child,
        &grand,
    )
    .expect("選択縮約を含む関数でも jvp_create_graph が成立する");
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(
        common::req2_close(jv, numeric),
        "{ctx}: jvp={jv} numeric={numeric}"
    );
}

#[test]
fn jvp_create_graph_through_select_reduce_matches_directional_derivative() {
    let shape = [3usize, 4];
    let x = distinct_pos(12);
    jvp_case(
        "kthvalue",
        |tp, x| quad(tp, &x.kthvalue(2, 1)?.0),
        &x,
        &shape,
    );
    jvp_case(
        "median dim",
        |tp, x| quad(tp, &x.median(Some(0))?),
        &x,
        &shape,
    );
    jvp_case("median all", |tp, x| quad(tp, &x.median(None)?), &x, &shape);
    jvp_case(
        "quantile",
        |tp, x| {
            quad(
                tp,
                &x.quantile(0.4, Some(1), QuantileInterpolation::Linear)?,
            )
        },
        &x,
        &shape,
    );
}

// --- 隣接する fail-closed ------------------------------------------------------------

struct IdentityCustomFn;

impl CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity_custom_fn"
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }

    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].contiguous())
    }

    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        Ok(vec![Some(upstream.contiguous())])
    }
}

#[test]
fn quantile_composed_with_custom_is_rejected_and_child_stays_empty() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(distinct_pos(6), &[6]));
    let q = x
        .quantile(0.4, None, QuantileInterpolation::Linear)
        .unwrap()
        .reshape(&[1])
        .unwrap();
    let c = tape.custom(Arc::new(IdentityCustomFn), &[q]).unwrap();
    let loss = c.sum(None).unwrap();
    let len_before = tape.len();
    let err = tape
        .backward_create_graph(&loss, &child)
        .expect_err("Op::Custom は fail-closed で拒否されるはず");
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty());
    assert_eq!(tape.len(), len_before);
}
