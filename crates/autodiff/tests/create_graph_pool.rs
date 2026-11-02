//! プーリング族（`MaxPool2d`・`MaxPool3d`・`AvgPool2d`・`AvgPool3d`・`AdaptiveAvgPool2d`）の
//! `create_graph` 対応（イシュー #3085・決定記録 `docs/autodiff-higher-order-grad-decision.md` §29）の
//! 受け入れ条件を確かめる統合テスト。
//!
//! - 平均系・最大系の Hessian を独立な有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で突合する。
//!   損失は `sum(y ⊙ y ⊙ c)`（`y` が入力の線形または区分線形のため Hessian は定数で、h を大きく取れる）。
//!   最大系は argmax がタイにならない入力に限る（タイの点では摂動で勝者が入れ替わる）。
//! - 先勝ちタイ規則が子テープでも 1 階と一致すること（解析的な曲率と 1 階との bit 一致）。
//! - 平均系の値が 1 階と bit 一致すること（f32 逐次和では桁落ちする打ち消し入力）。
//! - 1 階・`cg.grad`・二階・孫テープ（`jvp_create_graph`）の 3 段がすべて有限で、方向微分が独立な
//!   中央差分と一致すること。
//! - `MaxUnpool` はイシュー #3793 で対象になり、プーリングと組み合わせても成功すること
//!   （隣接する fail-closed Op の拒否は `create_graph_structural.rs`）。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::max_unpool_ops::max_unpool2d;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

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
        let row = row_grads.get(&cx).unwrap().expect("2 階勾配は x へ到達");
        for i in 0..n {
            hessian[j][i] = row.get(&unravel(i, shape)).unwrap() as f64;
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
    let numeric = finite_diff_hessian(build, x0, shape, 0.05);
    assert_hessian_close(&analytic, &numeric, ctx);
}

/// 値の重複がなく最小間隔が 0.3 の入力（argmax がタイにならない。素数 `p > n` の剰余で並べ替える）。
fn distinct(n: usize) -> Vec<f32> {
    let p = [29usize, 53, 101, 211, 401, 809, 1601]
        .into_iter()
        .find(|&p| p > n)
        .unwrap();
    (0..n)
        .map(|i| ((i * 37 + 5) % p) as f32 * 0.3 - 2.0)
        .collect()
}

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let shape = shp(y);
    let c = tape.var_no_grad(&t(common::det_seq(flat_len(&shape), 0.7), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

// --- Hessian と有限差分の突合（受け入れ条件 1・最大系は 2） -------------------------

#[test]
fn hessian_avg_pool2d_matches_finite_difference() {
    let shape = [1usize, 2, 5, 5];
    let x0 = common::det_seq(flat_len(&shape), 0.3);
    hessian_case(
        "avg2d k3 s1 overlap",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool2d([3, 3], Some([1, 1]), [0, 0], false, true)?,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "avg2d k2 s2",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool2d([2, 2], Some([2, 2]), [0, 0], false, true)?,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "avg2d pad cip=true",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool2d([3, 3], Some([2, 2]), [1, 1], false, true)?,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "avg2d pad cip=false",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool2d([3, 3], Some([2, 2]), [1, 1], false, false)?,
            )
        },
        &x0,
        &shape,
    );
}

#[test]
fn hessian_adaptive_avg_pool2d_matches_finite_difference() {
    let x0 = common::det_seq(2 * 16, 0.1);
    hessian_case(
        "adaptive 4->2",
        |tp, x| quad(tp, &x.adaptive_avg_pool2d([2, 2])?),
        &x0,
        &[1, 2, 4, 4],
    );
    let x0 = common::det_seq(2 * 25, 0.2);
    hessian_case(
        "adaptive 5->3",
        |tp, x| quad(tp, &x.adaptive_avg_pool2d([3, 3])?),
        &x0,
        &[1, 2, 5, 5],
    );
}

#[test]
fn hessian_avg_pool3d_matches_finite_difference() {
    let shape = [1usize, 2, 3, 4, 4];
    let x0 = common::det_seq(flat_len(&shape), 0.4);
    hessian_case(
        "avg3d k2 s1 overlap",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], false, true)?,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "avg3d k2 s2",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool3d([2, 2, 2], Some([2, 2, 2]), [0; 3], false, true)?,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "avg3d pad cip=true",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool3d([2, 2, 2], Some([2, 2, 2]), [1, 1, 1], false, true)?,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "avg3d pad cip=false",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool3d([2, 2, 2], Some([2, 2, 2]), [1, 1, 1], false, false)?,
            )
        },
        &x0,
        &shape,
    );
}

#[test]
fn hessian_max_pool_matches_finite_difference_without_ties() {
    let shape = [1usize, 2, 5, 5];
    let x0 = distinct(flat_len(&shape));
    hessian_case(
        "max2d k3 s1 overlap",
        |tp, x| {
            quad(
                tp,
                &x.max_pool2d([3, 3], Some([1, 1]), [0, 0], [1, 1], false)?.0,
            )
        },
        &x0,
        &shape,
    );
    hessian_case(
        "max2d pad dilation2",
        |tp, x| {
            quad(
                tp,
                &x.max_pool2d([2, 2], Some([1, 1]), [1, 1], [2, 2], false)?.0,
            )
        },
        &x0,
        &shape,
    );
    let shape3 = [1usize, 2, 3, 4, 4];
    let x3 = distinct(flat_len(&shape3));
    hessian_case(
        "max3d k2 s1 overlap",
        |tp, x| {
            quad(
                tp,
                &x.max_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], [1; 3], false)?
                    .0,
            )
        },
        &x3,
        &shape3,
    );
}

// --- 先勝ちタイ規則・重複索引（受け入れ条件 2） -------------------------------------

/// `loss = sum(maxpool(x))` の `cg.grad(x)` を 1 階 `Tape::backward` と bit 比較し、`first_order()` も
/// 同じ bit であることを確かめる。
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

#[test]
fn max_pool_tie_first_wins_matches_first_order_and_curvature() {
    // [1,1,2,2] = [3,3,1,3] を k2 で 1 窓。勝者は先頭（index 0）。loss = sum(y*y) の Hessian は
    // 先頭要素の対角 2 のみ。
    let shape = [1usize, 1, 2, 2];
    let x0 = [3.0f32, 3.0, 1.0, 3.0];
    fn build<'t>(_tp: &'t Tape, x: &Var<'t>) -> BuildResult<'t> {
        let y = x.max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)?.0;
        y.mul(&y)?.sum(None)
    }
    let h = analytic_hessian(build, &x0, &shape);
    for j in 0..4 {
        for i in 0..4 {
            let want = if i == 0 && j == 0 { 2.0 } else { 0.0 };
            assert!(common::req2_close(h[j][i], want), "h[{j}][{i}]={}", h[j][i]);
        }
    }
    assert_first_order_bit_identical(
        |_tp, x| {
            x.max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)?
                .0
                .sum(None)
        },
        &x0,
        &shape,
        "tie",
    );
}

#[test]
fn max_pool_duplicate_index_matches_first_order_under_cancellation() {
    // 中央が最大で 4 つの重なり窓すべてから選ばれる（重複索引）。上流 c は f32 逐次和で桁落ちする列。
    let mut x0 = vec![0.0f32; 9];
    x0[4] = 5.0;
    for (i, v) in x0.iter_mut().enumerate() {
        if i != 4 {
            *v = i as f32 * 0.1;
        }
    }
    assert_first_order_bit_identical(
        |tp, x| {
            let y = x.max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)?.0;
            let c = tp.var_no_grad(&t(vec![1.0e8, 1.0, -1.0e8, 1.0], &[1, 1, 2, 2]));
            y.mul(&c)?.sum(None)
        },
        &x0,
        &[1, 1, 3, 3],
        "dup index",
    );
    assert_first_order_bit_identical(
        |tp, x| {
            let y = x
                .max_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], [1; 3], false)?
                .0;
            let n = flat_len(&shp(&y));
            let data: Vec<f32> = (0..n)
                .map(|i| [1.0e8f32, 1.0, -1.0e8, 1.0][i % 4])
                .collect();
            let c = tp.var_no_grad(&t(data, &shp(&y)));
            y.mul(&c)?.sum(None)
        },
        &{
            let mut v: Vec<f32> = (0..27).map(|i| i as f32 * 0.1).collect();
            v[13] = 9.0;
            v
        },
        &[1, 1, 3, 3, 3],
        "dup index 3d",
    );
}

// --- 平均系の値が 1 階と bit 一致（V3） -----------------------------------------

#[test]
fn avg_family_value_is_bit_identical_to_first_order_under_cancellation() {
    let cancel = |i: usize| [1.0e8f32, 1.0, -1.0e8, 1.0, 1.0, -1.0e8, 1.0e8][i % 7];
    let x2 = common::det_seq(25, 0.5);
    assert_first_order_bit_identical(
        |tp, x| {
            let y = x.avg_pool2d([3, 3], Some([1, 1]), [0, 0], false, true)?;
            let data: Vec<f32> = (0..flat_len(&shp(&y))).map(cancel).collect();
            y.mul(&tp.var_no_grad(&t(data, &shp(&y))))?.sum(None)
        },
        &x2,
        &[1, 1, 5, 5],
        "avg2d",
    );
    assert_first_order_bit_identical(
        |tp, x| {
            let y = x.adaptive_avg_pool2d([3, 3])?;
            let data: Vec<f32> = (0..flat_len(&shp(&y))).map(cancel).collect();
            y.mul(&tp.var_no_grad(&t(data, &shp(&y))))?.sum(None)
        },
        &x2,
        &[1, 1, 5, 5],
        "adaptive",
    );
    let x3 = common::det_seq(27, 0.9);
    assert_first_order_bit_identical(
        |tp, x| {
            let y = x.avg_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], false, true)?;
            let data: Vec<f32> = (0..flat_len(&shp(&y))).map(cancel).collect();
            y.mul(&tp.var_no_grad(&t(data, &shp(&y))))?.sum(None)
        },
        &x3,
        &[1, 1, 3, 3, 3],
        "avg3d",
    );
}

// --- 3 段（1 階・cg.grad・二階・孫テープ）の有限性と方向微分 ----------------------------

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|a| a.is_finite())
}

/// 振幅 `amp` の上流を持つ損失 `sum(pool(x) ⊙ c)` で、1 階・`cg.grad`・二階がいずれも有限であること。
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

/// `sum(y ⊙ c) + sum(y ⊙ y)`（`c` は 3e38 の定数）。上流 `c + 2y` が f32 上限近傍で、かつ入力に
/// 依存するため二階が非自明になる。
fn big_loss<'t>(tape: &'t Tape, y: &Var<'t>, n: usize) -> BuildResult<'t> {
    let c = tape.var_no_grad(&t(vec![3.0e38f32; n], &shp(y)));
    y.mul(&c)?.sum(None)?.add(&y.mul(y)?.sum(None)?)
}

#[test]
fn three_stages_are_finite_with_large_upstream() {
    let x2 = distinct(25);
    three_stage_finite(
        |tp, x| {
            let y = x.avg_pool2d([3, 3], Some([1, 1]), [0, 0], false, true)?;
            big_loss(tp, &y, 9)
        },
        &x2,
        &[1, 1, 5, 5],
        "avg2d big",
    );
    three_stage_finite(
        |tp, x| {
            let y = x.adaptive_avg_pool2d([3, 3])?;
            big_loss(tp, &y, 9)
        },
        &x2,
        &[1, 1, 5, 5],
        "adaptive big",
    );
    three_stage_finite(
        |tp, x| {
            // 重なりなし（勝者の重複で 1 階自体が f32 上限を超えないようにする）。
            let y = x.max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)?.0;
            big_loss(tp, &y, 4)
        },
        &distinct(16),
        &[1, 1, 4, 4],
        "max2d big",
    );
    let x3 = distinct(27);
    three_stage_finite(
        |tp, x| {
            let y = x.avg_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], false, true)?;
            big_loss(tp, &y, 8)
        },
        &x3,
        &[1, 1, 3, 3, 3],
        "avg3d big",
    );
}

fn jvp_case<F>(ctx: &str, build: F, p0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let tangent: Vec<f32> = (0..p0.len())
        .map(|i| ((i * 3 + 1) % 7) as f32 * 0.1 - 0.3)
        .collect();
    // 独立な参照: 1 階 `Tape::backward`（create_graph 非依存）の勾配と接ベクトルの f64 内積。
    // loss の中央差分は f32 の丸め雑音（loss の大きさ × 2^-24 ÷ 2h）が判定幅を超えうるため使わない。
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
    .expect("プーリングを含む関数でも jvp_create_graph が成立する");
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(
        common::req2_close(jv, numeric),
        "{ctx}: jvp={jv} numeric={numeric}"
    );
}

#[test]
fn jvp_create_graph_through_pooling_matches_directional_derivative() {
    let shape = [1usize, 2, 5, 5];
    let x = distinct(flat_len(&shape));
    jvp_case(
        "avg2d",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool2d([3, 3], Some([2, 2]), [1, 1], false, true)?,
            )
        },
        &x,
        &shape,
    );
    jvp_case(
        "adaptive",
        |tp, x| quad(tp, &x.adaptive_avg_pool2d([3, 3])?),
        &x,
        &shape,
    );
    jvp_case(
        "max2d",
        |tp, x| {
            quad(
                tp,
                &x.max_pool2d([3, 3], Some([1, 1]), [0, 0], [1, 1], false)?.0,
            )
        },
        &x,
        &shape,
    );
    let s3 = [1usize, 1, 3, 4, 4];
    let x3 = distinct(flat_len(&s3));
    jvp_case(
        "avg3d",
        |tp, x| {
            quad(
                tp,
                &x.avg_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], false, true)?,
            )
        },
        &x3,
        &s3,
    );
    jvp_case(
        "max3d",
        |tp, x| {
            quad(
                tp,
                &x.max_pool3d([2, 2, 2], Some([1, 1, 1]), [0; 3], [1; 3], false)?
                    .0,
            )
        },
        &x3,
        &s3,
    );
}

// --- MaxUnpool はイシュー #3793 で対象になった（プーリングと組み合わせても成功する） ------------

#[test]
fn create_graph_accepts_max_unpool_after_pooling() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(distinct(16), &[1, 1, 4, 4]));
    let (y, idx) = x
        .max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)
        .unwrap();
    let up = max_unpool2d(&y, &idx, [2, 2], Some([2, 2]), [0, 0], None).unwrap();
    let loss = up.sum(None).unwrap();
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("MaxUnpool は #3793 で create_graph 対象");
    let plain = tape.backward(&loss).unwrap();
    let a = dense(cg.first_order().get(&x).unwrap().expect("x は到達"));
    let b = dense(plain.get(&x).unwrap().expect("x は到達"));
    assert_eq!(bits(&a), bits(&b));
}
