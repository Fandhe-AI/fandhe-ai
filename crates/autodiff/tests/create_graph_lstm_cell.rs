//! LSTM セル 2 Op（`LstmCell`・`LstmHidden`）の `create_graph` 対応（イシュー #3796・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §37・§61）の受け入れ条件を確かめる統合テスト。
//!
//! 入力 7 種（x・h_prev・c_prev・w_ih・w_hh・b_ih・b_hh）をすべて追跡ありの葉にして次を確かめる。
//! - 1 階 `first_order()` が素の `Tape::backward` と bit 一致し、`cg.grad` が REQ-2 で一致する。
//! - 全パラメータを 1 本の平坦ベクトル `theta` にまとめた Hessian（混合二階を含む）が、primitive の
//!   合成で組んだ参照の解析 Hessian と REQ-2 で一致し、独立な有限差分とも一致する。
//! - 出力部分集合（h のみ・c のみ・両方）と BPTT（重み共有・fan-in）、`nn::Lstm::forward_seq`。
//! - 必須ベクトル（飽和ゲート・巨大な上流・大きな `c_prev`）を 1 階・`cg.grad`・孫テープの 3 段で。
//! - 決定的な入力スケールの掃引で「1 階が有限なら cgrad も有限」。
//! - overflow guard の境界と、隣接する恒久対象外 Op（`Op::Custom`）の fail-closed。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::functional_ops::{hvp, jacfwd, jvp_create_graph};
use fandhe_ai_autodiff::nn::Lstm;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, GateParams, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(v: impl std::borrow::Borrow<Tensor<f32>>) -> Vec<f32> {
    let v = v.borrow();
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
}

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

/// 損失に使う出力の部分集合。`C` のみのとき `LstmHidden` は祖先に無く、子テープに孤立する。
#[derive(Clone, Copy, PartialEq, Debug)]
enum Out {
    H,
    C,
    Both,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    b: usize,
    d: usize,
    h: usize,
    bias: bool,
}

#[derive(Clone, Copy)]
struct Scale {
    x: f32,
    h: f32,
    c: f32,
    w_ih: f32,
    w_hh: f32,
    bias: f32,
}

const MODERATE: Scale = Scale {
    x: 1.0,
    h: 1.0,
    c: 1.0,
    w_ih: 0.5,
    w_hh: 0.5,
    bias: 0.5,
};

impl Layout {
    /// `[x, h_prev, c_prev, w_ih, w_hh, (b_ih, b_hh)]` の shape。
    fn shapes(&self) -> Vec<Vec<usize>> {
        let gh = 4 * self.h;
        let mut v = vec![
            vec![self.b, self.d],
            vec![self.b, self.h],
            vec![self.b, self.h],
            vec![self.d, gh],
            vec![self.h, gh],
        ];
        if self.bias {
            v.push(vec![gh]);
            v.push(vec![gh]);
        }
        v
    }

    fn scales(&self, s: Scale) -> Vec<f32> {
        let mut v = vec![s.x, s.h, s.c, s.w_ih, s.w_hh];
        if self.bias {
            v.push(s.bias);
            v.push(s.bias);
        }
        v
    }

    fn pieces(&self, s: Scale) -> Vec<Vec<f32>> {
        self.shapes()
            .iter()
            .zip(self.scales(s))
            .enumerate()
            .map(|(i, (shape, sc))| {
                let n: usize = shape.iter().product();
                common::det_seq(n, 0.3 + i as f32 * 1.7)
                    .into_iter()
                    .map(|v| v * sc)
                    .collect()
            })
            .collect()
    }

    fn theta0(&self, s: Scale) -> Vec<f32> {
        self.pieces(s).into_iter().flatten().collect()
    }

    fn numel(&self) -> usize {
        self.shapes()
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum()
    }

    fn leaves<'t>(&self, tape: &'t Tape, s: Scale) -> Vec<Var<'t>> {
        self.pieces(s)
            .into_iter()
            .zip(self.shapes())
            .map(|(data, shape)| tape.var(&t(data, &shape)))
            .collect()
    }

    fn unpack<'t>(&self, theta: &Var<'t>) -> Result<Vec<Var<'t>>, AutodiffError> {
        let mut off = 0;
        let mut out = Vec::new();
        for shape in self.shapes() {
            let n: usize = shape.iter().product();
            out.push(theta.narrow(0, off, n)?.reshape(&shape)?);
            off += n;
        }
        Ok(out)
    }

    /// 融合 Op で 1 step 進める。戻り値は `(h, c)`。
    fn step<'t>(&self, p: &[Var<'t>]) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
        self.step_with(&p[0], &p[1], &p[2], p)
    }

    fn step_with<'t>(
        &self,
        x: &Var<'t>,
        h: &Var<'t>,
        c: &Var<'t>,
        p: &[Var<'t>],
    ) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
        x.lstm_cell(
            h,
            c,
            GateParams {
                w_ih: &p[3],
                w_hh: &p[4],
                b_ih: p.get(5),
                b_hh: p.get(6),
            },
        )
    }

    /// 融合 Op を使わず primitive の合成で同じ 1 step を書く（Hessian の独立な参照）。
    fn step_primitives<'t>(
        &self,
        x: &Var<'t>,
        h: &Var<'t>,
        c: &Var<'t>,
        p: &[Var<'t>],
    ) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
        let mut pre = x.matmul(&p[3])?.add(&h.matmul(&p[4])?)?;
        if self.bias {
            pre = pre.add(&p[5])?.add(&p[6])?;
        }
        let hh = self.h;
        let blk = |k: usize| pre.narrow(1, k * hh, hh);
        let (i, f, g, o) = (
            blk(0)?.sigmoid(),
            blk(1)?.sigmoid(),
            blk(2)?.tanh(),
            blk(3)?.sigmoid(),
        );
        let c_new = f.mul(c)?.add(&i.mul(&g)?)?;
        let h_new = o.mul(&c_new.tanh())?;
        Ok((h_new, c_new))
    }
}

/// 出力部分集合に対する損失 `sum(y ⊙ y ⊙ c)`（`c_scale` は上流勾配の大きさを決める）。
fn loss_of<'t>(
    tape: &'t Tape,
    h: &Var<'t>,
    c: &Var<'t>,
    out: Out,
    c_scale: f32,
) -> BuildResult<'t> {
    let q = |y: &Var<'t>, seed: f32| -> BuildResult<'t> {
        let shape = y.value().shape().to_vec();
        let n: usize = shape.iter().product();
        let w: Vec<f32> = common::det_seq(n, seed)
            .into_iter()
            .map(|v| v * c_scale)
            .collect();
        let w = tape.var_no_grad(&t(w, &shape));
        y.mul(y)?.mul(&w)?.sum(None)
    };
    match out {
        Out::H => q(h, 0.7),
        Out::C => q(c, 1.9),
        Out::Both => q(h, 0.7)?.add(&q(c, 1.9)?),
    }
}

fn layouts() -> Vec<Layout> {
    let mut v = Vec::new();
    for bias in [false, true] {
        for (b, d, h) in [(1, 1, 1), (3, 4, 5)] {
            v.push(Layout { b, d, h, bias });
        }
    }
    v
}

// ---------------------------------------------------------------------
// 有限差分／解析 Hessian（平坦ベクトル theta に対する）
// ---------------------------------------------------------------------

fn hr<F>(f: F) -> F
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    f
}

fn finite_diff_hessian<F>(build: &F, x0: &[f32], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = x0.len();
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &[n]));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("theta は loss に到達する");
        (0..n).map(|i| dx.get(&[i]).unwrap() as f64).collect()
    };
    // Richardson 外挿 (4·D(h/2) − D(h)) / 3 で打ち切り誤差を落とす（§58.2 と同じ扱い）。
    let central = |i: usize, step: f64| -> Vec<f64> {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += step as f32;
        xm[i] -= step as f32;
        let gp = grad_at(&xp);
        let gm = grad_at(&xm);
        (0..n).map(|j| (gp[j] - gm[j]) / (2.0 * step)).collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let coarse = central(i, h);
        let fine = central(i, h / 2.0);
        for j in 0..n {
            hessian[j][i] = (4.0 * fine[j] - coarse[j]) / 3.0;
        }
    }
    hessian
}

fn analytic_hessian<F>(build: &F, x0: &[f32]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = x0.len();
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &[n]));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).unwrap().expect("1 階勾配");
    let cx = cg.child_var(&x).unwrap().expect("写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = gx.narrow(0, j, 1).unwrap().sum(None).unwrap();
        let row_grads = match child.backward(&gj) {
            Err(AutodiffError::GradientTrackingDisabled) => continue,
            other => other.expect("2 階"),
        };
        if let Some(row) = row_grads.get(&cx).unwrap() {
            for i in 0..n {
                hessian[j][i] = row.get(&[i]).unwrap() as f64;
            }
        }
    }
    hessian
}

fn hessian_case<F>(ctx: &str, build: F, x0: &[f32], h: f64, floor: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let analytic = analytic_hessian(&build, x0);
    let numeric = finite_diff_hessian(&build, x0, h);
    let mut nonzero = 0usize;
    for (j, (ra, rn)) in analytic.iter().zip(&numeric).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn).enumerate() {
            if a.abs() > 1e-4 {
                nonzero += 1;
            }
            // 差分側の丸め雑音の床を REQ-2 に足す（§58.2 と同じ。tolerance 定数は変えない）。
            // 厳密な REQ-2 突合は `assert_hessian_matches_primitives` が担う。
            assert!(
                common::req2_close(a, n) || (a - n).abs() < floor,
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
    assert!(nonzero > 0, "{ctx}: Hessian が全ゼロ（検査が空）");
}

fn assert_hessian_matches_primitives<F, G>(ctx: &str, fused: F, reference: G, x0: &[f32])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
    G: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let a = analytic_hessian(&fused, x0);
    let b = analytic_hessian(&reference, x0);
    let mut nonzero = 0usize;
    for (j, (ra, rb)) in a.iter().zip(&b).enumerate() {
        for (i, (&x, &y)) in ra.iter().zip(rb).enumerate() {
            if x.abs() > 1e-4 {
                nonzero += 1;
            }
            assert!(
                common::req2_close(x, y),
                "{ctx}: hessian[{j}][{i}] fused={x} primitives={y}"
            );
        }
    }
    assert!(nonzero > 0, "{ctx}: Hessian が全ゼロ（検査が空）");
}

// ---------------------------------------------------------------------
// 1 階の bit 一致と cgrad の REQ-2 一致
// ---------------------------------------------------------------------

#[test]
fn lstm_cells_first_order_bit_identical_and_cgrad_req2() {
    for lay in layouts() {
        for out in [Out::H, Out::C, Out::Both] {
            let ctx = format!("{lay:?} {out:?}");
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let vars = lay.leaves(&tape, MODERATE);
            let (h, c) = lay.step(&vars).unwrap();
            let loss = loss_of(&tape, &h, &c, out, 1.0).unwrap();
            let want = tape.backward(&loss).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            for (i, v) in vars.iter().enumerate() {
                let w = match want.get(v).unwrap() {
                    Some(w) => dense(w),
                    None => {
                        assert!(cg.grad(v).unwrap().is_none(), "{ctx}: 入力 {i}");
                        continue;
                    }
                };
                let f = dense(cg.first_order().get(v).unwrap().expect("first_order"));
                assert_eq!(bits(&f), bits(&w), "{ctx}: first_order() 入力 {i}");
                let g = dense(cg.grad(v).unwrap().expect("cgrad").to_tensor());
                common::assert_all_req2_close(
                    &g,
                    &w.iter().map(|&a| f64::from(a)).collect::<Vec<_>>(),
                    &format!("{ctx}: cg.grad 入力 {i}"),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Hessian と参照・有限差分の突合
// ---------------------------------------------------------------------

#[test]
fn lstm_cells_hessian_matches_finite_difference() {
    for bias in [false, true] {
        for out in [Out::H, Out::C, Out::Both] {
            let lay = Layout {
                b: 2,
                d: 3,
                h: 3,
                bias,
            };
            let build = hr(move |tape, theta| {
                let p = lay.unpack(theta)?;
                let (h, c) = lay.step(&p)?;
                loss_of(tape, &h, &c, out, 1.0)
            });
            let reference = hr(move |tape, theta| {
                let p = lay.unpack(theta)?;
                let (h, c) = lay.step_primitives(&p[0], &p[1], &p[2], &p)?;
                loss_of(tape, &h, &c, out, 1.0)
            });
            assert_hessian_matches_primitives(
                &format!("{lay:?} {out:?} vs primitives"),
                build,
                reference,
                &lay.theta0(MODERATE),
            );
            let build = hr(move |tape, theta| {
                let p = lay.unpack(theta)?;
                let (h, c) = lay.step(&p)?;
                loss_of(tape, &h, &c, out, 1.0)
            });
            hessian_case(
                &format!("{lay:?} {out:?}"),
                build,
                &lay.theta0(MODERATE),
                2e-2,
                5e-5,
            );
        }
    }
}

#[test]
fn lstm_cells_hvp_and_jacfwd_are_accepted() {
    for lay in layouts().into_iter().filter(|l| l.b == 3) {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(lay.theta0(MODERATE), &[lay.numel()]));
        let p = lay.unpack(&theta).unwrap();
        let (h, c) = lay.step(&p).unwrap();
        let loss = loss_of(&tape, &h, &c, Out::Both, 1.0).unwrap();
        let v = t(common::det_seq(lay.numel(), 2.0), &[lay.numel()]);
        let hv = hvp(&tape, &loss, &theta, &v, &child).unwrap_or_else(|e| panic!("{lay:?}: {e:?}"));
        let hv = dense(hv);
        assert!(hv.iter().all(|a| a.is_finite()), "{lay:?}");
        assert!(hv.iter().any(|a| a.abs() > 1e-4), "{lay:?}: Hv が全ゼロ");

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(lay.theta0(MODERATE), &[lay.numel()]));
        let p = lay.unpack(&theta).unwrap();
        let (h, _c) = lay.step(&p).unwrap();
        let jac = jacfwd(&tape, &h, &theta, &child).unwrap_or_else(|e| panic!("{lay:?}: {e:?}"));
        assert!(dense(jac).iter().all(|a| a.is_finite()), "{lay:?}");
    }
}

// ---------------------------------------------------------------------
// BPTT（重み共有・fan-in）と出力部分集合
// ---------------------------------------------------------------------

/// 重みを共有した 3 step の BPTT 損失。`(h, c)` は損失と次 step の両方へ流れる（fan-in）。
fn bptt_loss<'t>(
    lay: Layout,
    xs: &[Vec<f32>],
    tape: &'t Tape,
    theta: &Var<'t>,
    primitives: bool,
    out: Out,
) -> BuildResult<'t> {
    let p = lay.unpack(theta)?;
    let (mut h, mut c) = (p[1], p[2]);
    let mut acc: Option<Var<'t>> = None;
    for (s, xd) in xs.iter().enumerate() {
        let x = tape.var_no_grad(&t(xd.clone(), &[lay.b, lay.d]));
        (h, c) = if primitives {
            lay.step_primitives(&x, &h, &c, &p)?
        } else {
            lay.step_with(&x, &h, &c, &p)?
        };
        if s == 0 || s == xs.len() - 1 {
            let l = loss_of(tape, &h, &c, out, 1.0)?;
            acc = Some(match acc {
                Some(a) => a.add(&l)?,
                None => l,
            });
        }
    }
    acc.ok_or_else(|| AutodiffError::InvalidArgument("steps > 0".into()))
}

#[test]
fn lstm_cells_bptt_fan_in_and_output_subsets() {
    for out in [Out::H, Out::C, Out::Both] {
        let lay = Layout {
            b: 2,
            d: 2,
            h: 3,
            bias: true,
        };
        let xs: Vec<Vec<f32>> = (0..3usize)
            .map(|s| common::det_seq(lay.b * lay.d, 5.0 + s as f32))
            .collect();
        let build = hr(|tape, theta| bptt_loss(lay, &xs, tape, theta, false, out));
        let reference = hr(|tape, theta| bptt_loss(lay, &xs, tape, theta, true, out));
        assert_hessian_matches_primitives(
            &format!("bptt {out:?} vs primitives"),
            build,
            reference,
            &lay.theta0(MODERATE),
        );
        // 3 step 分の勾配は大きく、差分側の丸め雑音の床も大きくなる。
        hessian_case(
            &format!("bptt {out:?}"),
            build,
            &lay.theta0(MODERATE),
            2e-2,
            1.5e-4,
        );
    }
}

#[test]
fn nn_lstm_forward_seq_hessian_wrt_h0_c0() {
    let (b, d, h, t_len) = (1usize, 2usize, 3usize, 2usize);
    let x = t(common::det_seq(t_len * b * d, 9.0), &[t_len, b, d]);
    let build = hr(|tape, z| {
        let h0 = z.narrow(0, 0, b * h)?.reshape(&[b, h])?;
        let c0 = z.narrow(0, b * h, b * h)?.reshape(&[b, h])?;
        let o = Lstm::new(d, h, true, 7)?.forward_seq(tape, &x, Some(&h0), Some(&c0))?;
        loss_of(tape, &o.h_n, &o.c_n, Out::Both, 1.0)
    });
    hessian_case(
        "nn Lstm forward_seq (h0, c0)",
        build,
        &common::det_seq(2 * b * h, 3.0),
        2e-2,
        5e-5,
    );
}

// ---------------------------------------------------------------------
// 必須ベクトル・掃引（1 階・cg.grad・孫テープの 3 段）
// ---------------------------------------------------------------------

fn assert_cgrad_follows_first_order(ctx: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{ctx}");
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        if w.is_finite() {
            assert!(
                g.is_finite() && common::req2_close(f64::from(g), f64::from(w)),
                "{ctx}: cg.grad[{i}] {g} vs 1 階 {w}"
            );
        }
    }
}

/// 型付き拒否（中間積 overflow）が許される入力域の検査。上流 g は高々 `2·c_scale`、
/// オペランドの最大は `max(1, 各 scale)` なので、拒否は大きな積の域でしか起きてはならない。
fn assert_guard_rejection_allowed(
    ctx: &str,
    e: &AutodiffError,
    s: Scale,
    c_scale: f32,
    check_second: bool,
) {
    let msg = format!("{e:?}");
    assert!(
        msg.contains("create_graph:"),
        "{ctx}: 型付きでない拒否 {msg}"
    );
    if !check_second {
        return;
    }
    let m = [1.0f32, s.x, s.h, s.c, s.w_ih, s.w_hh, s.bias]
        .into_iter()
        .fold(1.0f64, |a, v| a.max(f64::from(v)));
    assert!(
        f64::from(c_scale) * m * m * 1e3 >= 2e37,
        "{ctx}: 安全域で拒否された: {msg}"
    );
}

struct Stages {
    first_finite: bool,
}

fn three_stage(ctx: &str, lay: Layout, s: Scale, c_scale: f32, check_second: bool) -> Stages {
    let n = lay.numel();
    let theta0 = lay.theta0(s);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let theta = tape.var(&t(theta0.clone(), &[n]));
    let p = lay.unpack(&theta).unwrap();
    let (h, c) = lay.step(&p).unwrap();
    let loss = loss_of(&tape, &h, &c, Out::Both, c_scale).unwrap();
    let want = dense(
        tape.backward(&loss)
            .unwrap()
            .get(&theta)
            .unwrap()
            .expect("theta"),
    );
    let first_finite = want.iter().all(|a| a.is_finite());
    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(e) => {
            assert_guard_rejection_allowed(ctx, &e, s, c_scale, check_second);
            return Stages { first_finite };
        }
    };
    let first = dense(cg.first_order().get(&theta).unwrap().expect("first_order"));
    assert_eq!(bits(&first), bits(&want), "{ctx}: first_order()");
    let got = dense(cg.grad(&theta).unwrap().expect("cgrad").to_tensor());
    assert_cgrad_follows_first_order(ctx, &want, &got);

    // 段 3: hvp と孫テープ。拒否されないこと。
    let v = t(common::det_seq(n, 2.0), &[n]);
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(theta0.clone(), &[n]));
        let p = lay.unpack(&theta).unwrap();
        let (h, c) = lay.step(&p).unwrap();
        let loss = loss_of(&tape, &h, &c, Out::Both, c_scale).unwrap();
        match hvp(&tape, &loss, &theta, &v, &child) {
            Ok(hv) => {
                if first_finite && check_second {
                    let hv = dense(hv);
                    assert!(
                        hv.iter().all(|a| a.is_finite()),
                        "{ctx}: Hv が非有限 {hv:?}"
                    );
                }
            }
            Err(e) => {
                assert_guard_rejection_allowed(ctx, &e, s, c_scale, check_second);
                return Stages { first_finite };
            }
        }
    }
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(theta0, &[n]));
        let p = lay.unpack(&theta).unwrap();
        let (h, _c) = lay.step(&p).unwrap();
        match jvp_create_graph(&tape, &h, &theta, &v, &child, &grand) {
            Ok(res) => {
                if first_finite && check_second {
                    let o = dense(res.output().to_tensor());
                    assert!(
                        o.iter().all(|a| a.is_finite()),
                        "{ctx}: 孫テープ出力が非有限"
                    );
                }
            }
            Err(e) => assert_guard_rejection_allowed(ctx, &e, s, c_scale, check_second),
        }
    }
    Stages { first_finite }
}

#[test]
fn lstm_cells_mandatory_vectors_three_stages() {
    for bias in [false, true] {
        let lay = Layout {
            b: 2,
            d: 3,
            h: 3,
            bias,
        };
        let cases: [(&str, Scale, f32); 6] = [
            (
                "飽和ゲート pre=1e4",
                Scale {
                    w_ih: 1e4,
                    w_hh: 1e4,
                    ..MODERATE
                },
                1.0,
            ),
            (
                "飽和ゲート pre=1e30",
                Scale {
                    x: 1e30,
                    w_ih: 1.0,
                    w_hh: 1.0,
                    ..MODERATE
                },
                1.0,
            ),
            ("上流 g=3e38", MODERATE, 3e38),
            ("大きな c_prev", Scale { c: 1e6, ..MODERATE }, 1.0),
            (
                "飽和かつ g=3e38",
                Scale {
                    w_ih: 1e4,
                    w_hh: 1e4,
                    ..MODERATE
                },
                3e38,
            ),
            ("通常", MODERATE, 1.0),
        ];
        for (name, s, c) in cases {
            three_stage(&format!("{lay:?} {name}"), lay, s, c, true);
        }
    }
}

#[test]
fn lstm_cells_sweep_first_order_finite_implies_cgrad_finite() {
    let scales = [1e-3f32, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e10, 1e20, 1e30];
    let gs = [1.0f32, 1e10, 1e30, 3e38];
    let lay = Layout {
        b: 2,
        d: 3,
        h: 3,
        bias: true,
    };
    let mut finite_cases = 0usize;
    for &sc in &scales {
        for &g in &gs {
            let s = Scale {
                x: sc,
                h: sc,
                c: sc,
                w_ih: sc,
                w_hh: sc,
                bias: sc,
            };
            // 二階の有限性を問うのは g·scale³ ≤ 1e30 の範囲に限る（§58.2 と同じ）。
            let check_second = f64::from(g) * f64::from(sc).powi(3) <= 1e30;
            let st = three_stage(
                &format!("sweep scale={sc:e} g={g:e}"),
                lay,
                s,
                g,
                check_second,
            );
            if st.first_finite {
                finite_cases += 1;
            }
        }
    }
    assert!(
        finite_cases > 10,
        "掃引で 1 階が有限なケースが少なすぎる: {finite_cases}"
    );
}

// ---------------------------------------------------------------------
// overflow guard の境界
// ---------------------------------------------------------------------

/// B=D=H=1 の 1 要素セルで `loss = c·Σ(h + c_t)` の `backward_create_graph` を試し、`w_ih` の勾配を返す。
fn scalar_cell_cgrad(x: f32, c_prev: f32, w: f32, scale: f32) -> Result<Vec<f32>, AutodiffError> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(vec![x], &[1, 1]));
    let hv = tape.var(&t(vec![0.0], &[1, 1]));
    let cv = tape.var(&t(vec![c_prev], &[1, 1]));
    let wi = tape.var(&t(vec![w; 4], &[1, 4]));
    let wh = tape.var(&t(vec![0.0; 4], &[1, 4]));
    let (h, c) = xv.lstm_cell(
        &hv,
        &cv,
        GateParams {
            w_ih: &wi,
            w_hh: &wh,
            b_ih: None,
            b_hh: None,
        },
    )?;
    let k = tape.var_no_grad(&t(vec![scale], &[1, 1]));
    let loss = h.add(&c)?.mul(&k)?.sum(None)?;
    let cg = tape.backward_create_graph(&loss, &child)?;
    Ok(dense(cg.grad(&wi)?.expect("w_ih").to_tensor()))
}

#[test]
fn lstm_intermediate_overflow_is_rejected_typed() {
    // x = W_ih = 1e10、loss = 1e30·(h + c)。1 階は飽和で 0 でも二階の随伴 g × W が f32 を超える。
    let e = scalar_cell_cgrad(1e10, 0.0, 1e10, 1e30).expect_err("型付きで拒否される");
    assert!(
        matches!(e, AutodiffError::Backward(_)) && format!("{e:?}").contains("2^125"),
        "{e:?}"
    );
    // c_prev が大きい場合も（オペランドに c_prev を含む）。
    let e = scalar_cell_cgrad(0.5, 1e20, 0.5, 1e30).expect_err("c_prev 由来でも型付きで拒否");
    assert!(format!("{e:?}").contains("2^125"), "{e:?}");
}

#[test]
fn lstm_large_but_safe_scale_is_accepted() {
    // 負のプローブ: g × オペランドが安全域なら受理され有限。
    let r = scalar_cell_cgrad(1e5, 1e3, 1e5, 1e15).expect("受理される");
    assert!(r.iter().all(|v| v.is_finite()), "{r:?}");
}

// ---------------------------------------------------------------------
// 隣接する fail-closed Op
// ---------------------------------------------------------------------

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
fn lstm_cells_neighbouring_unsupported_op_stays_fail_closed() {
    let lay = Layout {
        b: 2,
        d: 3,
        h: 3,
        bias: true,
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let vars = lay.leaves(&tape, MODERATE);
    let (h, _c) = lay.step(&vars).unwrap();
    let custom = tape
        .custom(Arc::new(IdentityCustomFn), &[h])
        .expect("Tape::custom 登録成功");
    let loss = custom.sum(None).unwrap();
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty(), "拒否時は child が空のまま");
    // 同じ child で通常の入力を再試行できる。
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let vars2 = lay.leaves(&tape2, MODERATE);
    let (h2, c2) = lay.step(&vars2).unwrap();
    let loss2 = loss_of(&tape2, &h2, &c2, Out::Both, 1.0).unwrap();
    tape2
        .backward_create_graph(&loss2, &child)
        .expect("通常の入力は再試行で成功する");
}
