//! RNN セル 2 Op（`RnnCell`・`GruCell`）の `create_graph` 対応（イシュー #3795・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §37・§58）の受け入れ条件を確かめる統合テスト。
//!
//! 入力 6 種（x・h_prev・w_ih・w_hh・b_ih・b_hh）をすべて追跡ありの葉にして次を確かめる。
//! - 1 階 `first_order()` が素の `Tape::backward` と bit 一致し、`cg.grad` が REQ-2 で一致する。
//! - 全パラメータを 1 本の平坦ベクトル `theta` にまとめた Hessian（混合二階を含む）が、独立な
//!   有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で一致する。
//! - BPTT（重み共有・fan-in）と `nn::Rnn`／`nn::Gru` の `forward_seq`（h0 について）。
//! - 必須ベクトル（飽和ゲート・巨大な上流・GRU の大きな `q`）を 1 階・`cg.grad`・孫テープの 3 段で。
//! - 決定的な入力スケールの掃引で「1 階が有限なら cgrad も有限」。
//! - 隣接する恒久対象外 Op（`Op::Custom`）は fail-closed のままで `child` が空に保たれる。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::functional_ops::{hvp, jacfwd, jvp_create_graph};
use fandhe_ai_autodiff::nn::{Gru, Rnn};
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Rnn,
    Gru,
}

/// セルの形状と bias の有無。ゲート数 `g` は RNN が 1、GRU が 3。
#[derive(Clone, Copy, Debug)]
struct Layout {
    kind: Kind,
    b: usize,
    d: usize,
    h: usize,
    bias: bool,
}

/// 入力ごとの決定的スケール（掃引・必須ベクトル用）。
#[derive(Clone, Copy)]
struct Scale {
    x: f32,
    h: f32,
    w_ih: f32,
    w_hh: f32,
    bias: f32,
}

const MODERATE: Scale = Scale {
    x: 1.0,
    h: 1.0,
    w_ih: 0.5,
    w_hh: 0.5,
    bias: 0.5,
};

impl Layout {
    fn gates(&self) -> usize {
        match self.kind {
            Kind::Rnn => 1,
            Kind::Gru => 3,
        }
    }

    /// `[x, h_prev, w_ih, w_hh, (b_ih, b_hh)]` の shape。
    fn shapes(&self) -> Vec<Vec<usize>> {
        let gh = self.gates() * self.h;
        let mut v = vec![
            vec![self.b, self.d],
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
        let mut v = vec![s.x, s.h, s.w_ih, s.w_hh];
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

    /// 各入力を独立した追跡ありの葉として登録する。
    fn leaves<'t>(&self, tape: &'t Tape, s: Scale) -> Vec<Var<'t>> {
        self.pieces(s)
            .into_iter()
            .zip(self.shapes())
            .map(|(data, shape)| tape.var(&t(data, &shape)))
            .collect()
    }

    /// 平坦ベクトル `theta`（`[numel]`）を各入力へ切り出す。
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

    /// 入力 `p`（`leaves`／`unpack` の戻り値）で 1 step 進める。
    fn step<'t>(&self, p: &[Var<'t>]) -> BuildResult<'t> {
        self.step_with(&p[0], &p[1], p)
    }

    /// 融合 Op を使わず primitive の合成で同じ 1 step を書く（Hessian の独立な参照）。
    fn step_primitives<'t>(&self, tape: &'t Tape, p: &[Var<'t>]) -> BuildResult<'t> {
        let (x, h) = (&p[0], &p[1]);
        let mut pre_i = x.matmul(&p[2])?;
        let mut pre_h = h.matmul(&p[3])?;
        if self.bias {
            pre_i = pre_i.add(&p[4])?;
            pre_h = pre_h.add(&p[5])?;
        }
        let hh = self.h;
        match self.kind {
            Kind::Rnn => Ok(pre_i.add(&pre_h)?.tanh()),
            Kind::Gru => {
                let blk = |v: &Var<'t>, k: usize| v.narrow(1, k * hh, hh);
                let r = blk(&pre_i, 0)?.add(&blk(&pre_h, 0)?)?.sigmoid();
                let z = blk(&pre_i, 1)?.add(&blk(&pre_h, 1)?)?.sigmoid();
                let n = r.mul(&blk(&pre_h, 2)?)?.add(&blk(&pre_i, 2)?)?.tanh();
                let one = tape.var_no_grad(&t(vec![1.0; self.b * hh], &[self.b, hh]));
                z.mul(h)?.add(&one.sub(&z)?.mul(&n)?)
            }
        }
    }

    fn step_with<'t>(&self, x: &Var<'t>, h: &Var<'t>, p: &[Var<'t>]) -> BuildResult<'t> {
        let gp = GateParams {
            w_ih: &p[2],
            w_hh: &p[3],
            b_ih: p.get(4),
            b_hh: p.get(5),
        };
        match self.kind {
            Kind::Rnn => x.rnn_cell(h, gp),
            Kind::Gru => x.gru_cell(h, gp),
        }
    }
}

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。`c_scale` は上流勾配の大きさを決める。
fn quad_scaled<'t>(tape: &'t Tape, y: &Var<'t>, c_scale: f32) -> BuildResult<'t> {
    let shape = y.value().shape().to_vec();
    let n: usize = shape.iter().product();
    let c: Vec<f32> = common::det_seq(n, 0.7)
        .into_iter()
        .map(|v| v * c_scale)
        .collect();
    let c = tape.var_no_grad(&t(c, &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    quad_scaled(tape, y, 1.0)
}

fn layouts() -> Vec<Layout> {
    let mut v = Vec::new();
    for kind in [Kind::Rnn, Kind::Gru] {
        for bias in [false, true] {
            for (b, d, h) in [(1, 1, 1), (3, 4, 5)] {
                v.push(Layout {
                    kind,
                    b,
                    d,
                    h,
                    bias,
                });
            }
        }
    }
    v
}

// ---------------------------------------------------------------------
// 有限差分 Hessian（平坦ベクトル theta に対する）
// ---------------------------------------------------------------------

/// クロージャの引数を高階ライフタイム（`for<'t>`）として推論させるための恒等ヘルパー。
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
    // 中央差分 D(h) の誤差は O(h²)。f32 の勾配では h を小さくすると丸めが支配するため、
    // Richardson 外挿 (4·D(h/2) − D(h)) / 3（誤差 O(h⁴)）で打ち切り誤差を落とす。
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
            // f32 の勾配差分は絶対誤差 ~2e-5 の丸め雑音を持つ（解析 Hessian の対称性
            // |H[j][i] − H[i][j]| ≲ 1e-7 から、雑音は解析側ではなく差分側にある）。差分側の
            // 分解能として REQ-2 に絶対誤差 `floor`（既定 5e-5）の床を足す。厳密な REQ-2 突合は
            // `assert_hessian_matches_primitives`（雑音のない解析同士）が担う。
            assert!(
                common::req2_close(a, n) || (a - n).abs() < floor,
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n} sym_a={} sym_n={}",
                analytic[i][j],
                numeric[i][j]
            );
        }
    }
    assert!(nonzero > 0, "{ctx}: Hessian が全ゼロ（検査が空）");
}

/// 融合 Op（`Op::RnnCell`／`Op::GruCell`）の解析 Hessian が、同じ計算を既存の primitive
/// （`matmul`・`add`・`tanh`・`sigmoid`・`narrow`・`mul`・`sub`）の合成で書いた参照の解析 Hessian と、
/// 雑音なしで REQ-2 統一複合判定に一致する。
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
fn rnn_gru_cells_first_order_bit_identical_and_cgrad_req2() {
    for lay in layouts() {
        let ctx = format!("{lay:?}");
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let vars = lay.leaves(&tape, MODERATE);
        let loss = quad(&tape, &lay.step(&vars).unwrap()).unwrap();
        let want = tape.backward(&loss).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        for (i, v) in vars.iter().enumerate() {
            let w = dense(want.get(v).unwrap().expect("入力は loss に到達する"));
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

// ---------------------------------------------------------------------
// Hessian と有限差分の突合
// ---------------------------------------------------------------------

#[test]
fn rnn_gru_cells_hessian_matches_finite_difference() {
    for kind in [Kind::Rnn, Kind::Gru] {
        for bias in [false, true] {
            let lay = Layout {
                kind,
                b: 2,
                d: 3,
                h: 3,
                bias,
            };
            let build = hr(move |tape, theta| {
                // 借用の都合でクロージャ内で tape と theta の寿命を揃える。
                let p = lay.unpack(theta)?;
                quad(tape, &lay.step(&p)?)
            });
            let reference = hr(move |tape, theta| {
                let p = lay.unpack(theta)?;
                quad(tape, &lay.step_primitives(tape, &p)?)
            });
            assert_hessian_matches_primitives(
                &format!("{lay:?} vs primitives"),
                build,
                reference,
                &lay.theta0(MODERATE),
            );
            let build = hr(move |tape, theta| {
                let p = lay.unpack(theta)?;
                quad(tape, &lay.step(&p)?)
            });
            hessian_case(
                &format!("{lay:?}"),
                build,
                &lay.theta0(MODERATE),
                2e-2,
                5e-5,
            );
        }
    }
}

#[test]
fn rnn_gru_cells_hvp_and_jacfwd_are_accepted() {
    // facade の hvp／jacfwd が RNN・GRU を含むモデルを拒否しない（#3795 の目的）。
    for lay in layouts().into_iter().filter(|l| l.b == 3) {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(lay.theta0(MODERATE), &[lay.numel()]));
        let p = lay.unpack(&theta).unwrap();
        let loss = quad(&tape, &lay.step(&p).unwrap()).unwrap();
        let v = t(common::det_seq(lay.numel(), 2.0), &[lay.numel()]);
        let hv = hvp(&tape, &loss, &theta, &v, &child).unwrap_or_else(|e| panic!("{lay:?}: {e:?}"));
        let hv = dense(hv);
        assert!(hv.iter().all(|a| a.is_finite()), "{lay:?}");
        assert!(hv.iter().any(|a| a.abs() > 1e-4), "{lay:?}: Hv が全ゼロ");

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(lay.theta0(MODERATE), &[lay.numel()]));
        let p = lay.unpack(&theta).unwrap();
        let y = lay.step(&p).unwrap();
        let jac = jacfwd(&tape, &y, &theta, &child).unwrap_or_else(|e| panic!("{lay:?}: {e:?}"));
        assert!(dense(jac).iter().all(|a| a.is_finite()), "{lay:?}");
    }
}

// ---------------------------------------------------------------------
// BPTT（重み共有・fan-in）
// ---------------------------------------------------------------------

/// 重みを共有した 3 step の BPTT 損失。`h_1` は損失と次 step の両方へ流れる（fan-in）。
/// `primitives` が真なら融合 Op を使わず primitive の合成で書く。
fn bptt_loss<'t>(
    lay: Layout,
    xs: &[Vec<f32>],
    tape: &'t Tape,
    theta: &Var<'t>,
    primitives: bool,
) -> BuildResult<'t> {
    let p = lay.unpack(theta)?;
    let mut h = p[1];
    let mut acc: Option<Var<'t>> = None;
    for (s, xd) in xs.iter().enumerate() {
        let x = tape.var_no_grad(&t(xd.clone(), &[lay.b, lay.d]));
        let mut q = p.clone();
        q[0] = x;
        q[1] = h;
        h = if primitives {
            lay.step_primitives(tape, &q)?
        } else {
            lay.step_with(&x, &h, &p)?
        };
        if s == 0 || s == xs.len() - 1 {
            let l = quad(tape, &h)?;
            acc = Some(match acc {
                Some(a) => a.add(&l)?,
                None => l,
            });
        }
    }
    acc.ok_or_else(|| AutodiffError::InvalidArgument("steps > 0".into()))
}

#[test]
fn rnn_gru_cells_bptt_fan_in() {
    for kind in [Kind::Rnn, Kind::Gru] {
        let lay = Layout {
            kind,
            b: 2,
            d: 2,
            h: 3,
            bias: true,
        };
        let steps = 3usize;
        let xs: Vec<Vec<f32>> = (0..steps)
            .map(|s| common::det_seq(lay.b * lay.d, 5.0 + s as f32))
            .collect();
        let build = hr(|tape, theta| bptt_loss(lay, &xs, tape, theta, false));
        let reference = hr(|tape, theta| bptt_loss(lay, &xs, tape, theta, true));
        assert_hessian_matches_primitives(
            &format!("bptt {kind:?} vs primitives"),
            build,
            reference,
            &lay.theta0(MODERATE),
        );
        // 3 step 分の勾配は大きく、差分側の丸め雑音の床も大きくなる（1.5e-4）。
        hessian_case(
            &format!("bptt {kind:?}"),
            build,
            &lay.theta0(MODERATE),
            2e-2,
            1.5e-4,
        );
    }
}

#[test]
fn nn_rnn_gru_forward_seq_hessian_wrt_h0() {
    for kind in [Kind::Rnn, Kind::Gru] {
        let (b, d, h, t_len) = (1usize, 2usize, 3usize, 2usize);
        let x = t(common::det_seq(t_len * b * d, 9.0), &[t_len, b, d]);
        let build = hr(|tape, h0| {
            let h0 = h0.reshape(&[b, h])?;
            let hn = match kind {
                Kind::Rnn => {
                    Rnn::new(d, h, true, 7)?
                        .forward_seq(tape, &x, Some(&h0))?
                        .h_n
                }
                Kind::Gru => {
                    Gru::new(d, h, true, 7)?
                        .forward_seq(tape, &x, Some(&h0))?
                        .h_n
                }
            };
            quad(tape, &hn)
        });
        hessian_case(
            &format!("nn forward_seq {kind:?}"),
            build,
            &common::det_seq(b * h, 3.0),
            2e-2,
            5e-5,
        );
    }
}

// ---------------------------------------------------------------------
// 必須ベクトル・掃引（1 階・cg.grad・孫テープの 3 段）
// ---------------------------------------------------------------------

/// 1 階が有限な要素について、cg.grad も有限で REQ-2 一致することを確かめる。
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

/// 型付き拒否（中間積 overflow・FMA 相殺。§37.12）が許される入力域の検査。
/// 上流 g は高々 `2·c_scale`、オペランドの最大は `max(1, 各 scale)` なので、拒否は
/// `c_scale × max(1, scale) ≥ 2^125 / 2` の域でしか起きてはならない（安全域の取りこぼしを防ぐ）。
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
    // 二階を問わない域（g·scale³ > 1e30）では、高階の入れ子で上流が増幅されるため拒否を許す。
    if !check_second {
        return;
    }
    let m = [1.0f32, s.x, s.h, s.w_ih, s.w_hh, s.bias]
        .into_iter()
        .fold(1.0f64, |a, v| a.max(f64::from(v)));
    // GRU の q = h·W_hh（和は H 項）も overflow 側の寄与になるため余裕を 1e3 倍みる。
    assert!(
        f64::from(c_scale) * m * m * 1e3 >= 2e37,
        "{ctx}: 安全域で拒否された: {msg}"
    );
}

struct Stages {
    first_finite: bool,
}

/// 平坦 theta で 1 階・cg.grad を比べ、孫テープ（jvp_create_graph）と hvp が拒否されないことを確かめる。
fn three_stage(ctx: &str, lay: Layout, s: Scale, c_scale: f32, check_second: bool) -> Stages {
    let n = lay.numel();
    let theta0 = lay.theta0(s);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let theta = tape.var(&t(theta0.clone(), &[n]));
    let p = lay.unpack(&theta).unwrap();
    let y = lay.step(&p).unwrap();
    let loss = quad_scaled(&tape, &y, c_scale).unwrap();
    let want = dense(
        tape.backward(&loss)
            .unwrap()
            .get(&theta)
            .unwrap()
            .expect("theta"),
    );
    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(e) => {
            assert_guard_rejection_allowed(ctx, &e, s, c_scale, check_second);
            return Stages {
                first_finite: want.iter().all(|a| a.is_finite()),
            };
        }
    };
    let first = dense(cg.first_order().get(&theta).unwrap().expect("first_order"));
    assert_eq!(bits(&first), bits(&want), "{ctx}: first_order()");
    let got = dense(cg.grad(&theta).unwrap().expect("cgrad").to_tensor());
    assert_cgrad_follows_first_order(ctx, &want, &got);
    let first_finite = want.iter().all(|a| a.is_finite());

    // 段 3: hvp と孫テープ。拒否（Err）されないこと。
    let v = t(common::det_seq(n, 2.0), &[n]);
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(theta0.clone(), &[n]));
        let p = lay.unpack(&theta).unwrap();
        let loss = quad_scaled(&tape, &lay.step(&p).unwrap(), c_scale).unwrap();
        let hv = match hvp(&tape, &loss, &theta, &v, &child) {
            Ok(hv) => hv,
            Err(e) => {
                assert_guard_rejection_allowed(ctx, &e, s, c_scale, check_second);
                return Stages { first_finite };
            }
        };
        if first_finite && check_second {
            // 1 階が有限なら二階の行列ベクトル積も有限であることを期待する（掃引で確かめる）。
            let hv = dense(hv);
            assert!(
                hv.iter().all(|a| a.is_finite()),
                "{ctx}: Hv が非有限 {hv:?}"
            );
        }
    }
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let theta = tape.var(&t(theta0, &[n]));
        let p = lay.unpack(&theta).unwrap();
        let y = lay.step(&p).unwrap();
        let res = match jvp_create_graph(&tape, &y, &theta, &v, &child, &grand) {
            Ok(res) => res,
            Err(e) => {
                assert_guard_rejection_allowed(ctx, &e, s, c_scale, check_second);
                return Stages { first_finite };
            }
        };
        if first_finite && check_second {
            let out = dense(res.output().to_tensor());
            assert!(
                out.iter().all(|a| a.is_finite()),
                "{ctx}: 孫テープ出力が非有限"
            );
        }
    }
    Stages { first_finite }
}

#[test]
fn rnn_gru_cells_mandatory_vectors_three_stages() {
    for kind in [Kind::Rnn, Kind::Gru] {
        for bias in [false, true] {
            let lay = Layout {
                kind,
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
                (
                    "GRU の大きな q（W_hh 1e4・h_prev 大）",
                    Scale {
                        h: 50.0,
                        w_hh: 1e4,
                        ..MODERATE
                    },
                    1.0,
                ),
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
}

#[test]
fn rnn_gru_cells_sweep_first_order_finite_implies_cgrad_finite() {
    let scales = [1e-3f32, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e10, 1e20, 1e30];
    let gs = [1.0f32, 1e10, 1e30, 3e38];
    let mut finite_cases = 0usize;
    for kind in [Kind::Rnn, Kind::Gru] {
        let lay = Layout {
            kind,
            b: 2,
            d: 3,
            h: 3,
            bias: true,
        };
        for &sc in &scales {
            for &g in &gs {
                let s = Scale {
                    x: sc,
                    h: sc,
                    w_ih: sc,
                    w_hh: sc,
                    bias: sc,
                };
                // 二階の有限性を問うのは g·scale³ ≤ 1e30 の範囲に限る。外側では 1 階が有限でも
                // cotangent × W × x の中間積が f32 を超え、真値が 0（飽和）でも 0 × inf = NaN になる
                // （§27.3 と同じ「真の二階が f32 に収まらない領域」。cg.grad の有限性は全範囲で問う）。
                let check_second = f64::from(g) * f64::from(sc).powi(3) <= 1e30;
                let st = three_stage(
                    &format!("sweep {kind:?} scale={sc:e} g={g:e}"),
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
    }
    assert!(
        finite_cases > 10,
        "掃引で 1 階が有限なケースが少なすぎる: {finite_cases}"
    );
}

// ---------------------------------------------------------------------
// 隣接する fail-closed Op
// ---------------------------------------------------------------------

/// 恒等関数として振る舞う `CustomFunction`（`Op::Custom` を経由させるためだけの最小実装）。
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
fn rnn_gru_cells_neighbouring_unsupported_op_stays_fail_closed() {
    for kind in [Kind::Rnn, Kind::Gru] {
        let lay = Layout {
            kind,
            b: 2,
            d: 3,
            h: 3,
            bias: true,
        };
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let vars = lay.leaves(&tape, MODERATE);
        let y = lay.step(&vars).unwrap();
        let custom = tape
            .custom(Arc::new(IdentityCustomFn), &[y])
            .expect("Tape::custom 登録成功");
        let loss = custom.sum(None).unwrap();
        let err = tape.backward_create_graph(&loss, &child).unwrap_err();
        assert!(
            matches!(err, AutodiffError::Backward(_)),
            "{kind:?}: {err:?}"
        );
        assert!(child.is_empty(), "{kind:?}: 拒否時は child が空のまま");
        // 同じ child で通常の入力を再試行できる。
        let tape2 = Tape::new_with_ops(common::naive_ops());
        let vars2 = lay.leaves(&tape2, MODERATE);
        let loss2 = quad(&tape2, &lay.step(&vars2).unwrap()).unwrap();
        tape2
            .backward_create_graph(&loss2, &child)
            .expect("通常の入力は再試行で成功する");
    }
}
// ---------------------------------------------------------------------
// 型付き拒否のプローブ（PR #3952 codex 指摘。正のプローブ＝拒否、負のプローブ＝受理）
// ---------------------------------------------------------------------

/// B=D=1 の 1 要素セルを作り、`loss = c·Σh` に対する `backward_create_graph` の結果を返す。
fn scalar_cell_cgrad(
    kind: Kind,
    x: f32,
    h_prev: f32,
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    c: f32,
) -> Result<Vec<f32>, AutodiffError> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let g = w_ih.len();
    let xv = tape.var(&t(vec![x], &[1, 1]));
    let hv = tape.var(&t(vec![h_prev], &[1, 1]));
    let wi = tape.var(&t(w_ih, &[1, g]));
    let wh = tape.var(&t(w_hh, &[1, g]));
    let gp = GateParams {
        w_ih: &wi,
        w_hh: &wh,
        b_ih: None,
        b_hh: None,
    };
    let y = match kind {
        Kind::Rnn => xv.rnn_cell(&hv, gp)?,
        Kind::Gru => xv.gru_cell(&hv, gp)?,
    };
    let cv = tape.var_no_grad(&t(vec![c], &[1, 1]));
    let loss = y.mul(&cv)?.sum(None)?;
    let cg = tape.backward_create_graph(&loss, &child)?;
    Ok(dense(cg.grad(&wi)?.expect("w_ih").to_tensor()))
}

#[test]
fn gru_fma_cancellation_is_rejected_typed() {
    // forward の tanh 引数 ≈ 3.76 に対し、積を先に丸める再構築では 0 になる入力。
    let r = scalar_cell_cgrad(
        Kind::Gru,
        1.0,
        1.0,
        vec![1.0, 0.0, -73105856.0],
        vec![0.0, 0.0, 100000000.0],
        1.0,
    );
    let e = r.expect_err("FMA 相殺入力は型付きで拒否される");
    assert!(format!("{e:?}").contains("FMA"), "{e:?}");
}

#[test]
fn gru_without_cancellation_is_accepted() {
    // 負のプローブ: 相殺しない入力は受理される。
    let r = scalar_cell_cgrad(
        Kind::Gru,
        1.0,
        1.0,
        vec![1.0, 0.5, 0.3],
        vec![0.2, 0.1, 0.7],
        1.0,
    )
    .expect("受理される");
    assert!(r.iter().all(|v| v.is_finite()));
}

#[test]
fn rnn_gru_intermediate_overflow_is_rejected_typed() {
    // x = W_ih = 1e10、loss = 1e30·h。1 階は飽和で 0 だが、二階の随伴 g × W が f32 を超える。
    let r = scalar_cell_cgrad(Kind::Rnn, 1e10, 0.0, vec![1e10], vec![0.0], 1e30);
    let e = r.expect_err("RNN の中間積 overflow は型付きで拒否される");
    assert!(format!("{e:?}").contains("2^125"), "{e:?}");
    let r = scalar_cell_cgrad(
        Kind::Gru,
        1e10,
        0.0,
        vec![1e10, 1e10, 1e10],
        vec![0.0, 0.0, 0.0],
        1e30,
    );
    let e = r.expect_err("GRU の中間積 overflow は型付きで拒否される");
    assert!(format!("{e:?}").contains("2^125"), "{e:?}");
}

#[test]
fn rnn_gru_large_but_safe_scale_is_accepted() {
    // 負のプローブ: g × オペランドが安全域（1e20）なら受理され有限。
    for kind in [Kind::Rnn, Kind::Gru] {
        let n = if matches!(kind, Kind::Rnn) { 1 } else { 3 };
        let r = scalar_cell_cgrad(kind, 1e5, 0.0, vec![1e5; n], vec![0.0; n], 1e15)
            .unwrap_or_else(|e| panic!("{kind:?}: {e:?}"));
        assert!(r.iter().all(|v| v.is_finite()), "{kind:?}: {r:?}");
    }
}

#[test]
fn gru_n_rounding_gap_is_not_amplified_into_gradient() {
    // PR #3952 の codex P1: n が 1 と 0.99999976 に割れる入力（REQ-2 の許容内）でも、
    // `1 − n²` と上流 1e4 の増幅で w_ih の勾配がずれないこと（n の値を forward の FMA 値へ補正する）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(vec![1.0], &[1, 1]));
    let hv = tape.var(&t(vec![1.0], &[1, 1]));
    let wi = tape.var(&t(vec![1.0, 0.0, -73105848.0], &[1, 3]));
    let wh = tape.var(&t(vec![0.0, 0.0, 100000000.0], &[1, 3]));
    let y = xv
        .gru_cell(
            &hv,
            GateParams {
                w_ih: &wi,
                w_hh: &wh,
                b_ih: None,
                b_hh: None,
            },
        )
        .unwrap();
    let cv = tape.var_no_grad(&t(vec![1e4], &[1, 1]));
    let loss = y.mul(&cv).unwrap().sum(None).unwrap();
    let want = dense(
        tape.backward(&loss)
            .unwrap()
            .get(&wi)
            .unwrap()
            .expect("w_ih"),
    );
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("拒否ではなく補正で受理される");
    let got = dense(cg.grad(&wi).unwrap().expect("w_ih").to_tensor());
    common::assert_all_req2_close(
        &got,
        &want.iter().map(|&a| f64::from(a)).collect::<Vec<_>>(),
        "n 丸め差の増幅",
    );
}
