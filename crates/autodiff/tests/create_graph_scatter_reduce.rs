//! `Op::IndexedScatterReduce`（`Var::scatter_reduce`。Sum・Mean・Amax・Amin）の `create_graph` 対応
//! （イシュー #3802・決定記録 `docs/autodiff-higher-order-grad-decision.md` §41、実装記録 §61）の
//! 受け入れ条件を確かめる統合テスト。
//!
//! - `first_order()` と `cg.grad` が 1 階 `Tape::backward` と bit 一致する（4 mode × `include_self` ×
//!   重複索引 × 非末尾 dim × index の非 dim 軸が input より小さい形状）。`input` 側・`src` 側の両方。
//! - Hessian を独立な有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で突合する。
//! - 孫テープ（`jvp_create_graph`）の出力を独立参照と突合する。
//! - Amax／Amin のタイは均等分配、NaN の結果は勾配 0 で二階も有限。
//! - 総括判定: ±inf 入力・非有限／過大な上流・`Prod` は子テープへ書く前に拒否され、`child` は空のまま
//!   再試行できる。空の index は 1 階に揃う。
//!
//! GPU parity は `crates/facade/tests/create_graph_gpu_parity.rs`（実機待ち・`#[ignore]`）。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::indexed_update_ops::scatter_reduce;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::ScatterReduceMode as Mode;
use fandhe_ai_tensor_core::Tensor;

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::<i32>::new(data, shape).expect("index fixture")
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
    v.borrow()
        .contiguous()
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|a| {
            if a.is_nan() {
                f32::NAN.to_bits()
            } else {
                a.to_bits()
            }
        })
        .collect()
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
        // 構造が x に依存しない成分は勾配が無い（None）ことがある。その行は 0。
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

fn distinct_vals(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + 5) % 29) as f32) * 0.21 - 2.0)
        .collect()
}

/// `distinct_vals` と値が重ならないよう半ステップずらした定数側の値（Amax／Amin のタイを避ける）。
fn shifted_vals(n: usize) -> Vec<f32> {
    distinct_vals(n).iter().map(|v| v + 0.105).collect()
}

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let shape = shp(y);
    let c = tape.var_no_grad(&t(common::det_pos(flat_len(&shape).max(1), 0.7), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

struct Cfg {
    name: &'static str,
    x_shape: Vec<usize>,
    s_shape: Vec<usize>,
    dim: usize,
    index: Vec<i32>,
}

/// 重複索引・未タッチ位置・非末尾 dim・index の非 dim 軸が小さい形状を含む。
fn cfgs() -> Vec<Cfg> {
    vec![
        Cfg {
            name: "1d dup",
            x_shape: vec![5],
            s_shape: vec![6],
            dim: 0,
            index: vec![0, 0, 2, 2, 2, 4],
        },
        Cfg {
            name: "2d dim1 small axis0",
            x_shape: vec![3, 4],
            s_shape: vec![2, 3],
            dim: 1,
            index: vec![1, 1, 3, 0, 2, 2],
        },
        Cfg {
            name: "2d dim0",
            x_shape: vec![3, 2],
            s_shape: vec![4, 2],
            dim: 0,
            index: vec![0, 2, 0, 2, 1, 1, 0, 2],
        },
    ]
}

const MODES: [Mode; 4] = [Mode::Sum, Mode::Mean, Mode::Amax, Mode::Amin];

/// 結合パラメータ `z = [x; src]` から `scatter_reduce` を作り二乗和損失へ流す。
fn joint<'t>(tape: &'t Tape, z: &Var<'t>, c: &Cfg, mode: Mode, inc: bool) -> BuildResult<'t> {
    let nx = flat_len(&c.x_shape);
    let ns = flat_len(&c.s_shape);
    let x = z.narrow(0, 0, nx)?.reshape(&c.x_shape)?;
    let s = z.narrow(0, nx, ns)?.reshape(&c.s_shape)?;
    let y = scatter_reduce(&x, c.dim, &ti(c.index.clone(), &c.s_shape), &s, mode, inc)?;
    quad(tape, &y)
}

/// `input` だけを微分対象にする（src は定数）。
fn x_only<'t>(tape: &'t Tape, x: &Var<'t>, c: &Cfg, mode: Mode, inc: bool) -> BuildResult<'t> {
    let s = tape.var_no_grad(&t(shifted_vals(flat_len(&c.s_shape)), &c.s_shape));
    let y = scatter_reduce(x, c.dim, &ti(c.index.clone(), &c.s_shape), &s, mode, inc)?;
    quad(tape, &y)
}

/// `src` だけを微分対象にする（input は定数）。
fn s_only<'t>(tape: &'t Tape, s: &Var<'t>, c: &Cfg, mode: Mode, inc: bool) -> BuildResult<'t> {
    let x = tape.var_no_grad(&t(shifted_vals(flat_len(&c.x_shape)), &c.x_shape));
    let y = scatter_reduce(&x, c.dim, &ti(c.index.clone(), &c.s_shape), s, mode, inc)?;
    quad(tape, &y)
}

fn assert_first_order_bit_identical<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
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
    let got = dense(cg.grad(&x).unwrap().expect("x は到達").to_tensor());
    assert_eq!(bits(&first), bits(&want), "{ctx}: first_order()");
    assert_eq!(bits(&got), bits(&want), "{ctx}: cg.grad");
}

#[test]
fn first_order_and_cg_grad_are_bit_identical() {
    for c in cfgs() {
        let nx = flat_len(&c.x_shape);
        let ns = flat_len(&c.s_shape);
        for mode in MODES {
            for inc in [false, true] {
                let ctx = format!("{} {mode:?} inc={inc}", c.name);
                assert_first_order_bit_identical(
                    &format!("{ctx} x"),
                    |tp, x| x_only(tp, x, &c, mode, inc),
                    &distinct_vals(nx),
                    &c.x_shape,
                );
                assert_first_order_bit_identical(
                    &format!("{ctx} src"),
                    |tp, s| s_only(tp, s, &c, mode, inc),
                    &distinct_vals(ns),
                    &c.s_shape,
                );
                assert_first_order_bit_identical(
                    &format!("{ctx} joint"),
                    |tp, z| joint(tp, z, &c, mode, inc),
                    &distinct_vals(nx + ns),
                    &[nx + ns],
                );
            }
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
    // 二階が自明に 0 の検査を弾く（H != 0）。
    assert!(
        analytic.iter().flatten().any(|v| v.abs() > 1e-3),
        "{ctx}: Hessian が全ゼロ（検査が空振り）"
    );
}

#[test]
fn hessian_matches_finite_difference() {
    for c in cfgs() {
        let nx = flat_len(&c.x_shape);
        let ns = flat_len(&c.s_shape);
        for mode in MODES {
            for inc in [false, true] {
                let ctx = format!("{} {mode:?} inc={inc}", c.name);
                hessian_case(
                    &format!("{ctx} joint"),
                    |tp, z| joint(tp, z, &c, mode, inc),
                    &distinct_vals(nx + ns),
                    &[nx + ns],
                );
                hessian_case(
                    &format!("{ctx} x"),
                    |tp, x| x_only(tp, x, &c, mode, inc),
                    &distinct_vals(nx),
                    &c.x_shape,
                );
                hessian_case(
                    &format!("{ctx} src"),
                    |tp, s| s_only(tp, s, &c, mode, inc),
                    &distinct_vals(ns),
                    &c.s_shape,
                );
            }
        }
    }
}

/// 未タッチ位置は上流を素通しするため、そこの曲率（loss の 2 階）が `x` へ流れる。
#[test]
fn untouched_positions_pass_curvature_through() {
    let c = &cfgs()[0];
    // 位置 1・3 は未タッチ。loss = Σ y²c の対角 Hessian = 2c が x[1]・x[3] に現れる。
    let h = analytic_hessian(
        |tp, x| x_only(tp, x, c, Mode::Sum, false),
        &distinct_vals(5),
        &[5],
    );
    assert!(h[1][1].abs() > 1e-3 && h[3][3].abs() > 1e-3, "{h:?}");
}

fn jvp_case<F>(ctx: &str, build: F, p0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tangent: Vec<f32> = common::det_seq(p0.len(), 0.9);
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
    .expect("scatter_reduce を含む関数でも jvp_create_graph が成立する");
    let jv = res.output().value().get(&[]).unwrap() as f64;
    assert!(
        common::req2_close(jv, numeric),
        "{ctx}: jvp={jv} numeric={numeric}"
    );
}

#[test]
fn jvp_create_graph_matches_independent_reference() {
    for c in cfgs() {
        let nx = flat_len(&c.x_shape);
        let ns = flat_len(&c.s_shape);
        for mode in MODES {
            for inc in [false, true] {
                jvp_case(
                    &format!("{} {mode:?} inc={inc}", c.name),
                    |tp, z| joint(tp, z, &c, mode, inc),
                    &distinct_vals(nx + ns),
                    &[nx + ns],
                );
            }
        }
    }
}

/// 1 次元 `scatter_reduce`（`dim = 0`）の `d_src`（1 階・cg.grad）を返し、二階の有限性も確かめる。
fn src_grads_1d(
    x0: &[f32],
    s0: &[f32],
    idx: &[i32],
    mode: Mode,
    inc: bool,
) -> (Vec<f32>, Vec<f32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &[x0.len()]));
    let s = tape.var(&t(s0.to_vec(), &[s0.len()]));
    let c = tape.var_no_grad(&t(common::det_pos(x0.len(), 0.7), &[x0.len()]));
    let y = scatter_reduce(&x, 0, &ti(idx.to_vec(), &[idx.len()]), &s, mode, inc).unwrap();
    let loss = y.mul(&c).unwrap().sum(None).unwrap();
    let want = tape.backward(&loss).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gs = cg.grad(&s).unwrap().unwrap();
    // 二階が有限（NaN の結果でも分母を 0 にしない）。
    let probe = gs.mul(&gs).unwrap().sum(None).unwrap();
    let cs = cg.child_var(&s).unwrap().unwrap();
    if let Ok(grads) = child.backward(&probe)
        && let Some(r) = grads.get(&cs).unwrap()
    {
        assert!(dense(r).iter().all(|v| v.is_finite()), "二階が非有限");
    }
    (dense(want.get(&s).unwrap().unwrap()), dense(gs.to_tensor()))
}

/// Amax／Amin のタイは均等分配（1 階 VJP と同規則）。cg.grad は 1 階と bit 一致。
#[test]
fn scatter_reduce_extremum_ties_follow_first_order_rule() {
    for (mode, s0, x0) in [
        (Mode::Amax, [2.0f32, 2.0, 1.0], [0.0f32, 5.0]),
        (Mode::Amin, [1.0f32, 1.0, 2.0], [3.0f32, 5.0]),
    ] {
        // include_self = false のみ: 自己寄与の値は極値と等しくならないが念のため両方回す。
        for inc in [false, true] {
            let (first, cg) = src_grads_1d(&x0, &s0, &[0, 0, 0], mode, inc);
            assert_eq!(bits(&first), bits(&cg), "{mode:?} inc={inc}");
            let c0 = common::det_pos(2, 0.7)[0];
            // include_self の x[0] は極値（Amax=2／Amin=1）に負けるのでタイは src の 2 つ。
            assert!(
                (cg[0] - c0 / 2.0).abs() <= 1e-6 && (cg[1] - c0 / 2.0).abs() <= 1e-6,
                "{mode:?} inc={inc}: {cg:?}"
            );
            assert_eq!(cg[2], 0.0, "非選択寄与の勾配は 0");
        }
    }
}

/// NaN の結果は ties = 0 で勾配 0。1 階と bit 一致し、二階も有限。
#[test]
fn nan_result_has_zero_gradient_and_finite_second_order() {
    for mode in [Mode::Amax, Mode::Amin] {
        let (first, cg) = src_grads_1d(&[0.0, 1.0], &[f32::NAN, 1.0], &[0, 0], mode, false);
        assert_eq!(bits(&first), bits(&cg));
        assert_eq!(cg, vec![0.0, 0.0]);
    }
}

/// 寄与数が大きい Mean（`g/m` は f64 割り算の 1 階値に揃う）。
#[test]
fn mean_with_many_contributions_matches_first_order() {
    let n = 40;
    let s0: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let idx = vec![0i32; n];
    for inc in [false, true] {
        let (first, cg) = src_grads_1d(&[0.5, 1.0], &s0, &idx, Mode::Mean, inc);
        assert_eq!(bits(&first), bits(&cg), "inc={inc}");
    }
}

fn tape_var<'t>(tape: &'t Tape, data: &[f32]) -> Var<'t> {
    tape.var_no_grad(&t(data.to_vec(), &[data.len()]))
}

fn assert_rejected<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{ctx}: {err:?}");
    assert!(child.is_empty(), "{ctx}: child が空でない");
}

#[test]
fn inf_inputs_and_prod_are_rejected_with_empty_child() {
    let idx = ti(vec![0, 0, 1], &[3]);
    let s = [1.0f32, 2.0, 3.0];
    for mode in MODES {
        assert_rejected(
            &format!("{mode:?} +inf input"),
            |tp, x| {
                let sv = tape_var(tp, &s);
                scatter_reduce(x, 0, &idx, &sv, mode, true)?.sum(None)
            },
            &[0.0, f32::INFINITY],
            &[2],
        );
        assert_rejected(
            &format!("{mode:?} -inf src"),
            |tp, x| {
                let sv = tape_var(tp, &[1.0, f32::NEG_INFINITY, 3.0]);
                scatter_reduce(x, 0, &idx, &sv, mode, true)?.sum(None)
            },
            &[0.0, 1.0],
            &[2],
        );
    }
    // Prod は #3803 の scan 補助待ちで fail-closed のまま。
    assert_rejected(
        "Prod",
        |tp, x| {
            let sv = tape_var(tp, &s);
            scatter_reduce(x, 0, &idx, &sv, Mode::Prod, true)?.sum(None)
        },
        &[1.5, 2.5],
        &[2],
    );
}

#[test]
fn unsafe_upstream_is_rejected_child_stays_empty_and_retry_succeeds() {
    for mode in MODES {
        for scale in [3e38f32, f32::INFINITY, f32::NAN] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![0.5, 1.5, -0.5], &[3]));
            let s = tape.var(&t(vec![1.0, 2.0], &[2]));
            let c_bad = tape.var_no_grad(&t(vec![scale], &[1]));
            let c_ok = tape.var_no_grad(&t(vec![1e3], &[1]));
            let y = scatter_reduce(&x, 0, &ti(vec![0, 0], &[2]), &s, mode, true).unwrap();
            let bad = y.mul(&c_bad).unwrap().sum(None).unwrap();
            let ok = y.mul(&c_ok).unwrap().sum(None).unwrap();
            let err = tape.backward_create_graph(&bad, &child).unwrap_err();
            assert!(
                matches!(err, AutodiffError::Backward(_)),
                "{mode:?} {scale}: {err:?}"
            );
            assert!(child.is_empty(), "{mode:?} {scale}: child が空でない");
            let cg = tape.backward_create_graph(&ok, &child).unwrap();
            let g = dense(cg.grad(&x).unwrap().unwrap().to_tensor());
            assert!(g.iter().all(|v| v.is_finite()), "{mode:?}: {g:?}");
        }
    }
}

#[test]
fn moderate_upstream_is_accepted_and_finite() {
    for mode in MODES {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.5, 1.5, -0.5], &[3]));
        let s = tape.var(&t(vec![1.0, 2.0], &[2]));
        let c = tape.var_no_grad(&t(vec![1e30], &[1]));
        let y = scatter_reduce(&x, 0, &ti(vec![0, 0], &[2]), &s, mode, true).unwrap();
        let loss = y.mul(&c).unwrap().sum(None).unwrap();
        let first = tape.backward(&loss).unwrap();
        assert!(
            dense(first.get(&s).unwrap().unwrap())
                .iter()
                .all(|v| v.is_finite())
        );
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gs = cg.grad(&s).unwrap().unwrap();
        assert!(
            dense(gs.to_tensor()).iter().all(|v| v.is_finite()),
            "{mode:?}"
        );
    }
}

/// index／src が空: 全位置が未タッチで 1 階は `d_input = upstream`・`d_src = []`。
#[test]
fn empty_index_matches_first_order() {
    for mode in MODES {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let s = tape.var(&t(vec![], &[0]));
        let y = scatter_reduce(&x, 0, &ti(vec![], &[0]), &s, mode, false).unwrap();
        let loss = quad(&tape, &y).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&x).unwrap().unwrap().to_tensor());
        assert_eq!(want, got, "{mode:?}");
    }
}

/// 空 index でも非有限／過大な上流を拒否する（index の要素数に依らず fail-closed。
/// 子テープは空のまま、安全な上流での再試行は成功する）。
#[test]
fn empty_index_rejects_unsafe_upstream() {
    for mode in MODES {
        for scale in [3e38f32, f32::INFINITY, f32::NAN] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![0.5, 1.5, -0.5], &[3]));
            let s = tape.var(&t(vec![], &[0]));
            let c_bad = tape.var_no_grad(&t(vec![scale], &[1]));
            let c_ok = tape.var_no_grad(&t(vec![1e3], &[1]));
            let y = scatter_reduce(&x, 0, &ti(vec![], &[0]), &s, mode, true).unwrap();
            let bad = y.mul(&c_bad).unwrap().sum(None).unwrap();
            let ok = y.mul(&c_ok).unwrap().sum(None).unwrap();
            let err = tape.backward_create_graph(&bad, &child).unwrap_err();
            assert!(
                matches!(err, AutodiffError::Backward(_)),
                "{mode:?} {scale}: {err:?}"
            );
            assert!(child.is_empty(), "{mode:?} {scale}: child が空でない");
            let cg = tape.backward_create_graph(&ok, &child).unwrap();
            let g = dense(cg.grad(&x).unwrap().unwrap().to_tensor());
            assert!(g.iter().all(|v| v.is_finite()), "{mode:?}: {g:?}");
        }
    }
}
