//! 3D 畳み込み族（`Conv3d`・`ConvTranspose3d`）の `create_graph` 対応の受け入れテスト
//! （イシュー #3797・決定記録 `docs/autodiff-higher-order-grad-decision.md` §38・§59）。
//!
//! 2D 版（`tests/create_graph.rs` の「2D 畳み込み族」節。§23）と同型の検証を 3D で行う:
//! - 1 階は素の `Tape::backward` と bit 同一（`first_order()`）。`cg.grad` の d_weight・d_bias は
//!   同じホスト関数を通るため CPU で bit 一致、d_input は V1 合成のため REQ-2 判定で一致する。
//! - Hessian を独立な有限差分（`Tape::backward` の中央差分。Richardson 外挿）と突合する。軸ごとに
//!   異なる kernel・stride・padding・dilation・output_padding で D/H/W 軸の取り違えを検出する。
//! - V3（d_weight・d_bias）の前方値が、f32 逐次和では overflow する入力でも 1 階と bit 一致し、
//!   孫テープへの再生でも有限のまま保たれる。
//! - 空入力・output_padding の符号付き導出・fail-closed（隣接する未対応 Op）。
//!
//! GPU 実機 parity は `crates/facade/tests/create_graph_gpu_parity.rs`（族名 `conv3d_family`）。

// Hessian の添字ループは `hessian[j][i]` のように `i`／`j` を複数配列へ同時に使うため
// （`tests/create_graph.rs` と同じ判断）。
#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var};
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

fn bits(v: &Tensor<f32>) -> Vec<u32> {
    v.host_slice().iter().map(|f| f.to_bits()).collect()
}

/// `Tape::backward`（`create_graph` 非依存）を中央差分した数値 Hessian（`h[j][i] = d g_j / d x_i`）。
fn finite_diff_hessian<F>(build: F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let n = flat_len(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("build: forward 構築に失敗");
        let grads = tape.backward(&loss).expect("backward: 1 階勾配計算に失敗");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達するはず");
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
        let (gp, gm) = (grad_at(&xp), grad_at(&xm));
        for j in 0..n {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
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

/// 子テープ上の 1 階勾配をさらに `Tape::backward` で微分した full Hessian。
fn analytic_hessian<F>(build: F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let n = flat_len(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("build: forward 構築に失敗");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達するはず");
    let cx = cg.child_var(&x).unwrap().expect("子テープ上の写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).unwrap();
        let row_grads = child.backward(&gj).expect("child.backward（2 階）に失敗");
        let row = row_grads
            .get(&cx)
            .unwrap()
            .expect("2 階勾配は x へ到達する");
        for i in 0..n {
            hessian[j][i] = row.get(&unravel(i, shape)).unwrap() as f64;
        }
    }
    hessian
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>]) {
    assert_eq!(analytic.len(), numeric.len());
    for (ra, rn) in analytic.iter().zip(numeric) {
        for (&a, &n) in ra.iter().zip(rn) {
            assert!(
                common::req2_close(a, n),
                "hessian mismatch: analytic={a} numeric={n}"
            );
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Conv3d,
    ConvTranspose3d,
}

#[derive(Clone, Copy)]
struct Cfg {
    kind: Kind,
    n: usize,
    cin: usize,
    cout: usize,
    dhw: [usize; 3],
    k: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    output_padding: [usize; 3],
    groups: usize,
    has_bias: bool,
}

impl Cfg {
    fn new(kind: Kind, groups: usize, has_bias: bool) -> Self {
        Self {
            kind,
            n: 1,
            cin: 2,
            cout: 2,
            dhw: [2, 3, 3],
            k: [2, 2, 2],
            stride: [1; 3],
            padding: [0; 3],
            dilation: [1; 3],
            output_padding: [0; 3],
            groups,
            has_bias,
        }
    }

    /// 軸ごとに異なるハイパーパラメータ（D/H/W 軸の取り違えを検出する）。
    fn axis_distinct(kind: Kind, has_bias: bool) -> Self {
        let mut c = Self::new(kind, 1, has_bias);
        c.k = [2, 3, 2];
        c.stride = [1, 2, 2];
        c.padding = [0, 1, 1];
        c.dilation = [2, 1, 1];
        match kind {
            Kind::Conv3d => {
                c.cin = 1;
                c.dhw = [3, 4, 3];
            }
            Kind::ConvTranspose3d => {
                c.cin = 1;
                c.dhw = [2, 2, 2];
                c.output_padding = [0, 1, 0];
            }
        }
        c
    }

    fn x_shape(&self) -> Vec<usize> {
        vec![self.n, self.cin, self.dhw[0], self.dhw[1], self.dhw[2]]
    }

    fn w_shape(&self) -> Vec<usize> {
        match self.kind {
            Kind::Conv3d => vec![
                self.cout,
                self.cin / self.groups,
                self.k[0],
                self.k[1],
                self.k[2],
            ],
            Kind::ConvTranspose3d => vec![
                self.cin,
                self.cout / self.groups,
                self.k[0],
                self.k[1],
                self.k[2],
            ],
        }
    }

    fn sizes(&self) -> (usize, usize, usize) {
        (
            flat_len(&self.x_shape()),
            flat_len(&self.w_shape()),
            if self.has_bias { self.cout } else { 0 },
        )
    }

    fn p_len(&self) -> usize {
        let (a, b, c) = self.sizes();
        a + b + c
    }

    fn p0(&self) -> Vec<f32> {
        (0..self.p_len())
            .map(|i| (((i * 37 + 11) % 17) as f32 - 8.0) * 0.1)
            .collect()
    }

    fn forward<'t>(
        &self,
        x: &Var<'t>,
        w: &Var<'t>,
        b: Option<&Var<'t>>,
    ) -> Result<Var<'t>, AutodiffError> {
        match self.kind {
            Kind::Conv3d => x.conv3d(w, b, self.stride, self.padding, self.dilation, self.groups),
            Kind::ConvTranspose3d => x.conv_transpose3d(
                w,
                b,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            ),
        }
    }

    /// loss = sum(y ⊙ y ⊙ c)（x・w・b を 1 本の葉 `p` から切り出し、純ブロックと混合ブロックを同時に見る）。
    fn build<'t>(&self, tape: &'t Tape, p: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let (nx, nw, nb) = self.sizes();
        let x = p.narrow(0, 0, nx)?.reshape(&self.x_shape())?;
        let w = p.narrow(0, nx, nw)?.reshape(&self.w_shape())?;
        let b = if self.has_bias {
            Some(p.narrow(0, nx + nw, nb)?)
        } else {
            None
        };
        let y = self.forward(&x, &w, b.as_ref())?;
        let ys = y.to_tensor().shape().to_vec();
        let coef: Vec<f32> = (0..flat_len(&ys))
            .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
            .collect();
        let c = tape.var_no_grad(&t(coef, &ys));
        y.mul(&y)?.mul(&c)?.sum(None)
    }
}

fn check_cfg(cfg: Cfg) {
    let p0 = cfg.p0();
    let shape = [p0.len()];
    // 2D 版（§23.4）と同じ: 大きめの h と Richardson 外挿（多項式 loss なので打ち切り誤差が消える）。
    let d1 = finite_diff_hessian(|tp, p| cfg.build(tp, p), &p0, &shape, 1e-1);
    let d2 = finite_diff_hessian(|tp, p| cfg.build(tp, p), &p0, &shape, 2e-1);
    let numeric: Vec<Vec<f64>> = d1
        .iter()
        .zip(&d2)
        .map(|(r1, r2)| {
            r1.iter()
                .zip(r2)
                .map(|(a, b)| (4.0 * a - b) / 3.0)
                .collect()
        })
        .collect();
    let analytic = analytic_hessian(|tp, p| cfg.build(tp, p), &p0, &shape);
    assert_hessian_close(&analytic, &numeric);

    // 1 階: first_order() は bit 同一、cg.grad は REQ-2 で一致。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let p = tape.var(&t(p0.clone(), &shape));
    let loss = cfg.build(&tape, &p).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gp_plain = plain.get(&p).unwrap().unwrap();
    let gp_first = cg.first_order().get(&p).unwrap().unwrap();
    let gp_child = cg.grad(&p).unwrap().unwrap();
    for i in 0..p0.len() {
        assert_eq!(
            gp_plain.get(&[i]).unwrap().to_bits(),
            gp_first.get(&[i]).unwrap().to_bits()
        );
        let expected = gp_plain.get(&[i]).unwrap() as f64;
        let actual = gp_child.value().get(&[i]).unwrap() as f64;
        assert!(
            common::req2_close(actual, expected),
            "cg.grad[{i}]={actual} backward={expected}"
        );
    }
}

#[test]
fn conv3d_family_hessian_matches_finite_difference() {
    for kind in [Kind::Conv3d, Kind::ConvTranspose3d] {
        for groups in [1usize, 2] {
            for has_bias in [false, true] {
                check_cfg(Cfg::new(kind, groups, has_bias));
            }
        }
        for has_bias in [false, true] {
            check_cfg(Cfg::axis_distinct(kind, has_bias));
        }
    }
}

/// x・w・b を別々の葉にして、1 階が bit 同一・`cg.grad` の d_weight／d_bias が CPU で bit 一致
/// （同じホスト関数を通る）・d_input が REQ-2 で一致することを確かめる。
#[test]
fn conv3d_family_first_order_bit_identical_and_cgrad_req2() {
    for kind in [Kind::Conv3d, Kind::ConvTranspose3d] {
        for groups in [1usize, 2] {
            let mut cfg = Cfg::new(kind, groups, true);
            cfg.n = 2;
            let p = cfg.p0();
            let (nx, nw, nb) = cfg.sizes();
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(p[..nx].to_vec(), &cfg.x_shape()));
            let w = tape.var(&t(p[nx..nx + nw].to_vec(), &cfg.w_shape()));
            let b = tape.var(&t(p[nx + nw..nx + nw + nb].to_vec(), &[nb]));
            let y = cfg.forward(&x, &w, Some(&b)).unwrap();
            let ys = y.to_tensor().shape().to_vec();
            let coef: Vec<f32> = (0..flat_len(&ys))
                .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
                .collect();
            let c = tape.var_no_grad(&t(coef, &ys));
            let loss = y.mul(&y).unwrap().mul(&c).unwrap().sum(None).unwrap();
            let plain = tape.backward(&loss).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            for v in [&x, &w, &b] {
                let a = plain.get(v).unwrap().unwrap();
                let f = cg.first_order().get(v).unwrap().unwrap();
                assert_eq!(bits(a), bits(f), "first_order は素の backward と bit 同一");
            }
            for v in [&w, &b] {
                let a = plain.get(v).unwrap().unwrap();
                let gv = cg.grad(v).unwrap().unwrap();
                let g = gv.value();
                assert_eq!(bits(a), bits(&g), "d_weight／d_bias は CPU で bit 一致");
            }
            let a = plain.get(&x).unwrap().unwrap().host_slice().into_owned();
            let g = cg
                .grad(&x)
                .unwrap()
                .unwrap()
                .value()
                .host_slice()
                .into_owned();
            for (i, (&p, &q)) in a.iter().zip(&g).enumerate() {
                assert!(
                    common::req2_close(f64::from(q), f64::from(p)),
                    "d_input[{i}] cg={q} backward={p}"
                );
            }
        }
    }
}

// --- V3: overflow 回帰（f32 逐次和では inf・f64 縮約では有限） -----------------

const BIG: f32 = 3.0e38;

fn tiny_forward<'t>(
    kind: Kind,
    x: &Var<'t>,
    w: &Var<'t>,
    b: Option<&Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match kind {
        Kind::Conv3d => x.conv3d(w, b, [1; 3], [0; 3], [1; 3], 1),
        Kind::ConvTranspose3d => x.conv_transpose3d(w, b, [1; 3], [0; 3], [0; 3], [1; 3], 1),
    }
}

/// N=3・1×1×1 カーネル・`x=[1,1,-1]`・`g=c=[BIG;3]`: d_weight = Σ x·g = BIG（f64 縮約で有限。
/// f32 逐次和なら 2·BIG で inf）。1 階と `cg.grad(w)` が bit 一致し、二階も有限。
fn weight_overflow_case(kind: Kind) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 1.0, -1.0], &[3, 1, 1, 1, 1]));
    let w = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
    let y = tiny_forward(kind, &x, &w, None).unwrap();
    let c = tape.var_no_grad(&t(vec![BIG; 3], &[3, 1, 1, 1, 1]));
    let loss = y.mul(&c).unwrap().sum(None).unwrap();

    let plain = tape.backward(&loss).unwrap();
    let dw_plain = plain.get(&w).unwrap().unwrap().get(&[0; 5]).unwrap();
    assert!(dw_plain.is_finite(), "1 階の d_weight は f64 縮約で有限");

    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gw = cg.grad(&w).unwrap().unwrap();
    assert_eq!(
        dw_plain.to_bits(),
        gw.value().get(&[0; 5]).unwrap().to_bits(),
        "V3: cg.grad(w) の前方値は 1 階と bit 一致"
    );

    // 二階: ∂gw/∂x_i = g_i = BIG（有限）。
    let cx = cg.child_var(&x).unwrap().unwrap();
    let grads2 = child.backward(&gw).unwrap();
    let dx2 = grads2.get(&cx).unwrap().unwrap();
    for i in 0..3 {
        let v = dx2.get(&[i, 0, 0, 0, 0]).unwrap();
        assert!(
            v.is_finite() && (v - BIG).abs() <= BIG * 1e-6,
            "dx2[{i}]={v}"
        );
    }
    // 3 段目（孫テープ）: 子テープ上の勾配はさらに create_graph で微分できる（閉包性）。
    let grand = Tape::new_with_ops(common::naive_ops());
    let cg2 = child
        .backward_create_graph(&gw, &grand)
        .expect("子テープ上の勾配は更に create_graph で微分できる");
    let g_cx = cg2.grad(&cx).unwrap().unwrap();
    for i in 0..3 {
        let v = g_cx.value().get(&[i, 0, 0, 0, 0]).unwrap();
        assert!(v.is_finite(), "3 段目の勾配 [{i}] は有限のはず: {v}");
    }
}

#[test]
fn conv3d_family_weight_bias_grad_overflow_regression() {
    for kind in [Kind::Conv3d, Kind::ConvTranspose3d] {
        weight_overflow_case(kind);

        // d_bias: g=[BIG,BIG,-BIG] は打ち消し合って有限。1 階と bit 一致。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var_no_grad(&t(vec![0.5; 3], &[3, 1, 1, 1, 1]));
        let w = tape.var_no_grad(&t(vec![1.0], &[1, 1, 1, 1, 1]));
        let b = tape.var(&t(vec![0.0], &[1]));
        let y = tiny_forward(kind, &x, &w, Some(&b)).unwrap();
        let c = tape.var_no_grad(&t(vec![BIG, BIG, -BIG], &[3, 1, 1, 1, 1]));
        let loss = y.mul(&c).unwrap().sum(None).unwrap();
        let plain = tape.backward(&loss).unwrap();
        let db_plain = plain.get(&b).unwrap().unwrap().get(&[0]).unwrap();
        assert!(db_plain.is_finite());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let db_child = cg.grad(&b).unwrap().unwrap().value().get(&[0]).unwrap();
        assert_eq!(db_plain.to_bits(), db_child.to_bits());

        // 打ち消さない入力: g=[BIG,BIG,BIG] の d_bias は f32 でも f64 縮約でも 1 階と同じ（inf）で bit 一致。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var_no_grad(&t(vec![0.5; 3], &[3, 1, 1, 1, 1]));
        let w = tape.var_no_grad(&t(vec![1.0], &[1, 1, 1, 1, 1]));
        let b = tape.var(&t(vec![0.0], &[1]));
        let y = tiny_forward(kind, &x, &w, Some(&b)).unwrap();
        let c = tape.var_no_grad(&t(vec![BIG; 3], &[3, 1, 1, 1, 1]));
        let loss = y.mul(&c).unwrap().sum(None).unwrap();
        let plain = tape.backward(&loss).unwrap();
        let db_plain = plain.get(&b).unwrap().unwrap().get(&[0]).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let db_child = cg.grad(&b).unwrap().unwrap().value().get(&[0]).unwrap();
        assert_eq!(db_plain.to_bits(), db_child.to_bits());
        assert!(
            db_plain.is_infinite(),
            "3·BIG は f32 に収まらず両者とも inf"
        );
    }
}

/// 孫テープで `h = (gw·1e-38)²` を作ってさらに微分した x 勾配が、素の `backward` と bit 一致する
/// （補正値 gw=3e38 が孫テープ再生で inf に化けない）。
#[test]
fn conv3d_family_grandchild_replay_stays_finite() {
    for kind in [Kind::Conv3d, Kind::ConvTranspose3d] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0, 1.0, -1.0], &[3, 1, 1, 1, 1]));
        let w = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
        let y = tiny_forward(kind, &x, &w, None).unwrap();
        let c = tape.var_no_grad(&t(vec![BIG; 3], &[3, 1, 1, 1, 1]));
        let loss = y.mul(&c).unwrap().sum(None).unwrap();

        let child = Tape::new_with_ops(common::naive_ops());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gw = cg.grad(&w).unwrap().unwrap();
        let s = child.var_no_grad(&t(vec![1.0e-38], &[1, 1, 1, 1, 1]));
        let scaled = gw.mul(&s).unwrap();
        let h = scaled.mul(&scaled).unwrap().sum(None).unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        let plain_h = child.backward(&h).unwrap();
        let dx_plain = plain_h.get(&cx).unwrap().unwrap();
        let grand = Tape::new_with_ops(common::naive_ops());
        let cg3 = child.backward_create_graph(&h, &grand).unwrap();
        let dx_grand = cg3.grad(&cx).unwrap().unwrap();
        for i in 0..3 {
            let p = dx_plain.get(&[i, 0, 0, 0, 0]).unwrap();
            let q = dx_grand.value().get(&[i, 0, 0, 0, 0]).unwrap();
            assert!(p.is_finite() && q.is_finite(), "[{i}] plain={p} grand={q}");
            assert_eq!(p.to_bits(), q.to_bits(), "[{i}] 孫テープ再生値の維持");
        }
    }
}

// --- 孫テープ: jvp_create_graph ------------------------------------------------

#[test]
fn conv3d_family_jvp_create_graph_matches_central_difference() {
    for kind in [Kind::Conv3d, Kind::ConvTranspose3d] {
        let cfg = Cfg::new(kind, 1, true);
        let p0 = cfg.p0();
        let shape = [p0.len()];
        let tangent: Vec<f32> = (0..p0.len())
            .map(|i| ((i * 3 + 1) % 7) as f32 * 0.1 - 0.3)
            .collect();
        let out_at = |pv: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let p = tape.var(&t(pv.to_vec(), &shape));
            cfg.build(&tape, &p).unwrap().value().get(&[]).unwrap() as f64
        };
        let h = 1e-3f32;
        let plus: Vec<f32> = p0.iter().zip(&tangent).map(|(a, v)| a + h * v).collect();
        let minus: Vec<f32> = p0.iter().zip(&tangent).map(|(a, v)| a - h * v).collect();
        let numeric = (out_at(&plus) - out_at(&minus)) / (2.0 * f64::from(h));

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let p = tape.var(&t(p0.clone(), &shape));
        let loss = cfg.build(&tape, &p).unwrap();
        let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
            &tape,
            &loss,
            &p,
            &t(tangent.clone(), &shape),
            &child,
            &grand,
        )
        .expect("3D 畳み込みを含む関数でも jvp_create_graph が成立する");
        let jv = res.output().value().get(&[]).unwrap() as f64;
        assert!(
            common::req2_close(jv, numeric),
            "jvp={jv} numeric={numeric}"
        );
    }
}

// --- 境界 ------------------------------------------------------------------------

/// 受理される空 shape（N=0、Cin=0）。N=0 は d_bias もゼロ、N>0・Cin=0・Cout>0 は d_bias = sum(g)。
#[test]
fn conv3d_family_empty_inputs_zero_grads() {
    for kind in [Kind::Conv3d, Kind::ConvTranspose3d] {
        // N>0・Cin=0: d_bias = sum(g) = 1。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&Tensor::<f32>::zeros(&[1, 0, 1, 1, 1]).unwrap());
        let w = match kind {
            Kind::Conv3d => tape.var(&Tensor::<f32>::zeros(&[1, 0, 1, 1, 1]).unwrap()),
            Kind::ConvTranspose3d => tape.var(&Tensor::<f32>::zeros(&[0, 1, 1, 1, 1]).unwrap()),
        };
        let b = tape.var(&t(vec![1.0], &[1]));
        let y = tiny_forward(kind, &x, &w, Some(&b)).unwrap();
        let loss = y.sum(None).unwrap();
        let plain = tape.backward(&loss).unwrap();
        let db_plain = plain.get(&b).unwrap().unwrap().get(&[0]).unwrap();
        assert_eq!(db_plain, 1.0);
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let db = cg.grad(&b).unwrap().unwrap().value().get(&[0]).unwrap();
        assert_eq!(db.to_bits(), db_plain.to_bits());

        // N=0: すべてゼロ勾配（forward が N=0 を受理することは実装時に確認済み。決定記録 §59）。
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&Tensor::<f32>::zeros(&[0, 1, 2, 2, 2]).unwrap());
        let w = tape.var(&t(vec![0.5; 8], &[1, 1, 2, 2, 2]));
        let b = tape.var(&t(vec![0.3], &[1]));
        let y = match kind {
            Kind::Conv3d => x.conv3d(&w, Some(&b), [1; 3], [0; 3], [1; 3], 1),
            Kind::ConvTranspose3d => {
                x.conv_transpose3d(&w, Some(&b), [1; 3], [0; 3], [0; 3], [1; 3], 1)
            }
        };
        let y = y.expect("N=0 は forward が受理する");
        let loss = y.sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        for v in [&w, &b] {
            let gv = cg.grad(v).unwrap().unwrap();
            let g = gv.value();
            assert!(g.host_slice().iter().all(|&e| e == 0.0));
        }
    }
}

/// `in=1, k=1, stride=3, padding=1, output_padding=2`（各軸）: `base = −1` と負になるが forward は有効。
/// 再生が output_padding を符号付きで導出でき、孫テープでも再生できること。
#[test]
fn conv_transpose3d_output_padding_derivation_signed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.5], &[1, 1, 1, 1, 1]));
    let w = tape.var(&t(vec![2.0], &[1, 1, 1, 1, 1]));
    let y = x
        .conv_transpose3d(&w, None, [3; 3], [1; 3], [2; 3], [1; 3], 1)
        .unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("負の base でも有効な転置畳み込みは再生できる");
    let dx_plain = plain.get(&x).unwrap().unwrap().get(&[0; 5]).unwrap();
    let dx_child = cg.grad(&x).unwrap().unwrap().value().get(&[0; 5]).unwrap();
    assert_eq!(dx_plain.to_bits(), dx_child.to_bits());
    let grand = Tape::new_with_ops(common::naive_ops());
    let gx = cg.grad(&x).unwrap().unwrap();
    child
        .backward_create_graph(&gx.sum(None).unwrap(), &grand)
        .expect("子テープ上の転置畳み込みも再生できる");

    // 1×1×1 入力・3×3×3 カーネル・stride 3・padding 2・output_padding 2（`(in−1)s + d(k−1) + 1 < 2p`）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.5], &[1, 1, 1, 1, 1]));
    let w = tape.var(&t(
        (0..27).map(|i| i as f32 * 0.1).collect(),
        &[1, 1, 3, 3, 3],
    ));
    let y = x
        .conv_transpose3d(&w, None, [3; 3], [2; 3], [2; 3], [1; 3], 1)
        .unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gw = cg.grad(&w).unwrap().unwrap();
    let grand = Tape::new_with_ops(common::naive_ops());
    child
        .backward_create_graph(&gw.sum(None).unwrap(), &grand)
        .expect("再生で output_padding を導出できる");
}

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

/// 3D 畳み込みと未対応 Op（`Op::Custom`。兄弟イシューが対象化しない）が同居する祖先は、子テープへ
/// 一切書かず `Err(Backward)` で拒否される（`child.is_empty()`）。拒否後も再試行できる。
#[test]
fn conv3d_family_neighbouring_unsupported_op_stays_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.5; 8], &[1, 1, 2, 2, 2]));
    let w = tape.var(&t(vec![0.25; 8], &[1, 1, 2, 2, 2]));
    let y = x.conv3d(&w, None, [1; 3], [0; 3], [1; 3], 1).unwrap();
    let z = tape
        .custom(Arc::new(IdentityCustomFn), &[y])
        .expect("Tape::custom 登録成功");
    let loss = z.sum(None).unwrap();
    let err = tape
        .backward_create_graph(&loss, &child)
        .expect_err("Op::Custom は fail-closed で拒否される");
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty(), "拒否時に child へ書き込まない");
    // 同じ child で、対象 Op だけの loss は再試行できる。
    let loss_ok = y.sum(None).unwrap();
    tape.backward_create_graph(&loss_ok, &child)
        .expect("拒否後の再試行は成功する");
}
