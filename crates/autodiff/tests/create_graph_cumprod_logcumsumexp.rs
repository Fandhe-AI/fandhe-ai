//! 累積 B 区分（`Cumprod`・`Logcumsumexp`）の `create_graph` 対応の受け入れテスト
//! （イシュー #3803・決定記録 `docs/autodiff-higher-order-grad-decision.md` §41・§62）。
//!
//! - 1 階は素の `Tape::backward` と bit 同一（`first_order()`）、`cg.grad` も CPU で 1 階と bit 同一（V3）。
//! - Hessian を独立な有限差分（`Tape::backward` の中央差分）と突合する。`Cumprod` は零が 0／1／2 個の
//!   入力と、最初の零より後ろの混合二階 `H[j,z]`（除算形では取りこぼす）を含む。
//! - 孫テープ（`jvp_create_graph`）の再生が独立の方向微分と一致する。
//! - 必須ベクトル（§41.5）は 3 段（1 階・`cg.grad`・孫テープ）で「有限」か「型付き拒否（`child` 空）」。
//! - 入口規則・空テンソル・隣接する未対応 Op の fail-closed。
//!
//! GPU 実機 parity は `crates/facade/tests/create_graph_gpu_parity.rs`（族名 `cumulative_b`）。

// Hessian の添字ループは `hessian[j][i]` のように `i`／`j` を複数配列へ同時に使うため
// （`tests/create_graph_conv3d.rs` と同じ判断）。
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Cumprod,
    Logcumsumexp,
}

impl Kind {
    fn apply<'t>(self, x: &Var<'t>, dim: usize) -> Result<Var<'t>, AutodiffError> {
        match self {
            Kind::Cumprod => x.cumprod(dim),
            Kind::Logcumsumexp => x.logcumsumexp(dim),
        }
    }
}

/// 重み `c`（定数）付きの線形 loss `Σ c ⊙ f(x)`。上流勾配が `c` そのものになる。
fn linear_loss<'t>(
    tape: &'t Tape,
    kind: Kind,
    x: &Var<'t>,
    dim: usize,
    c: &[f32],
) -> Result<Var<'t>, AutodiffError> {
    let y = kind.apply(x, dim)?;
    let shape = x.value().shape().to_vec();
    let cv = tape.var_no_grad(&t(c.to_vec(), &shape));
    y.mul(&cv)?.sum(None)
}

/// `Σ c ⊙ f(x)³`（勾配が x の非自明な多項式になり、二階が非ゼロになる loss）。
fn cubic_loss<'t>(
    tape: &'t Tape,
    kind: Kind,
    x: &Var<'t>,
    dim: usize,
    c: &[f32],
) -> Result<Var<'t>, AutodiffError> {
    let y = kind.apply(x, dim)?;
    let shape = x.value().shape().to_vec();
    let cv = tape.var_no_grad(&t(c.to_vec(), &shape));
    y.mul(&y)?.mul(&y)?.mul(&cv)?.sum(None)
}

/// 検査対象の loss（クロージャは高階ライフタイムの推論が通らないため構造体で持つ）。
struct Loss<'a> {
    kind: Kind,
    dim: usize,
    c: &'a [f32],
    cubic: bool,
}

impl Loss<'_> {
    fn build<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        if self.cubic {
            cubic_loss(tape, self.kind, x, self.dim, self.c)
        } else {
            linear_loss(tape, self.kind, x, self.dim, self.c)
        }
    }
}

fn finite_diff_hessian(build: &Loss<'_>, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>> {
    let n = flat_len(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build.build(&tape, &xv).expect("build: forward 構築に失敗");
        let grads = tape.backward(&loss).expect("backward: 1 階勾配計算に失敗");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達するはず");
        (0..n)
            .map(|i| f64::from(dx.get(&unravel(i, shape)).unwrap()))
            .collect()
    };
    // 中央差分。刻み `h` と `h/2` の Richardson 外挿で `O(h²)` の打ち切り誤差を消す。
    let central = |i: usize, step: f64| -> Vec<f64> {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += step as f32;
        xm[i] -= step as f32;
        let (gp, gm) = (grad_at(&xp), grad_at(&xm));
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

fn analytic_hessian(build: &Loss<'_>, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>> {
    let n = flat_len(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build.build(&tape, &x).expect("build: forward 構築に失敗");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph に失敗");
    let gx = cg.grad(&x).unwrap().expect("x は loss に到達するはず");
    let cx = cg.child_var(&x).unwrap().expect("子テープ上の写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).unwrap();
        let row_grads = child.backward(&gj).expect("child.backward（2 階）に失敗");
        // 構造が x に依存しない行（Hessian の行が全ゼロ）は x へ到達しない。
        if let Some(row) = row_grads.get(&cx).unwrap() {
            for i in 0..n {
                hessian[j][i] = f64::from(row.get(&unravel(i, shape)).unwrap());
            }
        }
    }
    hessian
}

fn assert_hessian_close(analytic: &[Vec<f64>], numeric: &[Vec<f64>], what: &str) {
    assert_eq!(analytic.len(), numeric.len());
    for (j, (ra, rn)) in analytic.iter().zip(numeric).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{what}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
}

fn pattern(n: usize, scale: f32, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.77 + shift).sin() * scale)
        .collect()
}

// --- 1 階 bit 一致 ---------------------------------------------------------------

#[test]
fn cum_b_first_order_bit_identical_and_cgrad_bit_identical() {
    for kind in [Kind::Cumprod, Kind::Logcumsumexp] {
        for (shape, dim) in [(vec![6usize], 0usize), (vec![2, 5, 3], 1), (vec![3, 4], 1)] {
            let n = flat_len(&shape);
            let x0: Vec<f32> = pattern(n, 0.8, 0.3).iter().map(|v| v + 1.2).collect();
            let c = pattern(n, 1.0, 1.9);
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let loss = cubic_loss(&tape, kind, &x, dim, &c).unwrap();
            let plain = tape.backward(&loss).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let p = plain.get(&x).unwrap().unwrap();
            let f = cg.first_order().get(&x).unwrap().unwrap();
            assert_eq!(bits(p), bits(f), "{kind:?} {shape:?}: first_order");
            let gx = cg.grad(&x).unwrap().unwrap().to_tensor();
            assert_eq!(bits(p), bits(&gx), "{kind:?} {shape:?}: cg.grad (V3)");
        }
    }
}

// --- Hessian 有限差分突合 ---------------------------------------------------------

/// `cubic = true` は `Σ c·f(x)³`、`false` は `Σ c·f(x)`（x の多重線形で、零点でも有限差分が正確）。
fn check_hessian(kind: Kind, x0: &[f32], shape: &[usize], dim: usize, cubic: bool) {
    let n = flat_len(shape);
    let c = pattern(n, 1.0, 0.4);
    let build = Loss {
        kind,
        dim,
        c: &c,
        cubic,
    };
    let a = analytic_hessian(&build, x0, shape);
    // logcumsumexp は滑らかで Hessian の要素が小さいため、f32 の丸め雑音を減らす大きい刻みを使う
    // （Richardson 外挿で打ち切り誤差は O(h⁴)）。
    let h = if kind == Kind::Logcumsumexp {
        8e-2
    } else {
        2e-2
    };
    let num = finite_diff_hessian(&build, x0, shape, h);
    assert_hessian_close(&a, &num, &format!("{kind:?} shape={shape:?} dim={dim}"));
}

#[test]
fn cumprod_hessian_matches_finite_difference() {
    check_hessian(Kind::Cumprod, &[1.1, 0.9, 1.2, 0.8, 1.05], &[5], 0, true);
    // 零が 1／2 個（零点では cubic loss の打ち切り誤差が出るため多重線形な loss で突合する）。
    check_hessian(Kind::Cumprod, &[1.1, 0.0, 1.2, 0.8, 1.05], &[5], 0, false);
    check_hessian(Kind::Cumprod, &[1.1, 0.0, 1.2, 0.0, 1.05], &[5], 0, false);
    check_hessian(Kind::Cumprod, &[0.0, 0.9, 1.2, 0.8], &[4], 0, false);
    check_hessian(Kind::Cumprod, &[1.1, 0.9, 1.2, 0.0], &[4], 0, false);
    // dim が先頭・最終軸・rank 3 の中間軸。
    let x2: Vec<f32> = pattern(12, 0.3, 0.5).iter().map(|v| v + 1.0).collect();
    check_hessian(Kind::Cumprod, &x2, &[3, 4], 0, true);
    check_hessian(Kind::Cumprod, &x2, &[3, 4], 1, true);
    let x3: Vec<f32> = pattern(18, 0.3, 0.9).iter().map(|v| v + 1.0).collect();
    check_hessian(Kind::Cumprod, &x3, &[2, 3, 3], 1, true);
}

/// 最初の零 `z` より後ろの混合二階 `H[j,z]`（`j > z`）。`x=[2,0,3]`・`g=[0,0,1]` では
/// loss = x0·x1·x2 で Hessian は `[[0,3,0],[3,0,2],[0,2,0]]`。除算形＋最初の零マスクは
/// この成分を取りこぼす（§41.10 が除算を使わない scan を採った根拠の回帰）。
#[test]
fn cumprod_mixed_hessian_after_first_zero_matches_finite_difference() {
    let x0 = [2.0f32, 0.0, 3.0];
    let c = [0.0f32, 0.0, 1.0];
    let build = Loss {
        kind: Kind::Cumprod,
        dim: 0,
        c: &c,
        cubic: false,
    };
    let a = analytic_hessian(&build, &x0, &[3]);
    let want = [[0.0, 3.0, 0.0], [3.0, 0.0, 2.0], [0.0, 2.0, 0.0]];
    for j in 0..3 {
        for i in 0..3 {
            assert!(
                common::req2_close(a[j][i], want[j][i]),
                "H[{j}][{i}]={} want={}",
                a[j][i],
                want[j][i]
            );
        }
    }
    let num = finite_diff_hessian(&build, &x0, &[3], 2e-2);
    assert_hessian_close(&a, &num, "mixed-after-zero");
}

#[test]
fn logcumsumexp_hessian_matches_finite_difference() {
    check_hessian(
        Kind::Logcumsumexp,
        &[0.3, -0.5, 1.1, 0.2, -0.9],
        &[5],
        0,
        true,
    );
    // 一定の lane（x=[0,0] を含む）。受理される（オフセット付きの拒否と対）。
    check_hessian(Kind::Logcumsumexp, &[0.0, 0.0, 0.0], &[3], 0, true);
    check_hessian(Kind::Logcumsumexp, &[2.0, 2.0], &[2], 0, true);
    let x2: Vec<f32> = pattern(12, 1.5, 0.2);
    check_hessian(Kind::Logcumsumexp, &x2, &[3, 4], 0, true);
    check_hessian(Kind::Logcumsumexp, &x2, &[3, 4], 1, true);
    let x3: Vec<f32> = pattern(18, 1.5, 0.7);
    check_hessian(Kind::Logcumsumexp, &x3, &[2, 3, 3], 1, true);
}

// --- 孫テープ ---------------------------------------------------------------------

#[test]
fn cum_b_grandchild_replay_matches_directional_derivative() {
    for kind in [Kind::Cumprod, Kind::Logcumsumexp] {
        let shape = [5usize];
        let x0: Vec<f32> = match kind {
            Kind::Cumprod => vec![1.1, 0.0, 1.2, 0.8, 1.05],
            Kind::Logcumsumexp => vec![0.3, -0.5, 1.1, 0.2, -0.9],
        };
        let c = pattern(5, 1.0, 0.4);
        let tangent: Vec<f32> = vec![0.3, -0.2, 0.5, 0.1, -0.4];
        let out_at = |xv: &[f32]| -> f64 {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(xv.to_vec(), &shape));
            let loss = cubic_loss(&tape, kind, &x, 0, &c).unwrap();
            f64::from(loss.value().get(&[]).unwrap())
        };
        let h = 1e-3f32;
        let plus: Vec<f32> = x0.iter().zip(&tangent).map(|(a, v)| a + h * v).collect();
        let minus: Vec<f32> = x0.iter().zip(&tangent).map(|(a, v)| a - h * v).collect();
        let numeric = (out_at(&plus) - out_at(&minus)) / (2.0 * f64::from(h));

        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0.clone(), &shape));
        let loss = cubic_loss(&tape, kind, &x, 0, &c).unwrap();
        let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
            &tape,
            &loss,
            &x,
            &t(tangent.clone(), &shape),
            &child,
            &grand,
        )
        .unwrap_or_else(|e| panic!("{kind:?}: jvp_create_graph が成立しない: {e:?}"));
        let jv = f64::from(res.output().value().get(&[]).unwrap());
        assert!(
            common::req2_close(jv, numeric),
            "{kind:?}: jvp={jv} numeric={numeric}"
        );
    }
}

// --- 必須ベクトル（§41.5。3 段で有限か型付き拒否） ---------------------------------------

enum Expect {
    Accept,
    Reject,
}

/// 1 ケースを 3 段（1 階・`cg.grad`・孫テープ）で検査する。`Accept` は 3 段とも有限、
/// `Reject` は `Err(Backward)` かつ `child` 空で、同じ入力の素の `backward` は有限のまま。
fn run_vector(kind: Kind, x0: &[f32], c: &[f32], expect: Expect, label: &str) {
    let shape = [x0.len()];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), &shape));
    let loss = linear_loss(&tape, kind, &x, 0, c).unwrap();
    let plain = tape.backward(&loss).unwrap();
    let dp = plain.get(&x).unwrap().unwrap();
    let res = tape.backward_create_graph(&loss, &child);
    match expect {
        Expect::Accept => {
            let cg = res.unwrap_or_else(|e| panic!("{label}: 受理されるはず: {e:?}"));
            assert!(
                dp.host_slice().iter().all(|v| v.is_finite()),
                "{label}: 1 階が有限"
            );
            let gx = cg.grad(&x).unwrap().unwrap();
            assert!(
                gx.value().host_slice().iter().all(|v| v.is_finite()),
                "{label}: cg.grad が有限"
            );
            // 3 段目: 孫テープ（`jvp_create_graph`。三階以上の保証は対象外なので jvp の出力だけ見る）。
            let child2 = Tape::new_with_ops(common::naive_ops());
            let grand = Tape::new_with_ops(common::naive_ops());
            let tangent = t(vec![1.0; x0.len()], &shape);
            let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
                &tape, &loss, &x, &tangent, &child2, &grand,
            )
            .unwrap_or_else(|e| panic!("{label}: 孫テープが成立するはず: {e:?}"));
            let jv = res.output().value().get(&[]).unwrap();
            assert!(jv.is_finite(), "{label}: 孫テープの jvp が有限（{jv}）");
            let rgrads = grand
                .backward(&res.output().sum(None).unwrap())
                .unwrap_or_else(|e| panic!("{label}: 孫テープの逆伝播: {e:?}"));
            let rx = res.child_var(&x).unwrap().unwrap();
            if let Some(rg) = rgrads.get(&rx).unwrap() {
                assert!(
                    rg.host_slice().iter().all(|v| v.is_finite()),
                    "{label}: 孫テープの勾配が有限"
                );
            }
        }
        Expect::Reject => {
            let err = res
                .err()
                .unwrap_or_else(|| panic!("{label}: 拒否されるはず"));
            assert!(
                matches!(err, AutodiffError::Backward(_)),
                "{label}: {err:?}"
            );
            assert!(child.is_empty(), "{label}: 拒否時に child へ書かない");
            assert!(
                dp.host_slice().iter().all(|v| !v.is_nan()),
                "{label}: 素の backward は NaN にならない（先頭の零の勾配は inf になりうる）"
            );
        }
    }
}

#[test]
fn cum_b_mandatory_vectors_finite_or_rejected_in_three_stages() {
    // Cumprod 受理: 0 への underflow・零 0／1／2 個。
    let ones = |n: usize| vec![1.0f32; n];
    run_vector(
        Kind::Cumprod,
        &[1e-30, 1e-30, 1e30],
        &ones(3),
        Expect::Accept,
        "cumprod [1e-30,1e-30,1e30]",
    );
    run_vector(
        Kind::Cumprod,
        &[2.0, 3.0, 4.0],
        &ones(3),
        Expect::Accept,
        "cumprod 零0個",
    );
    run_vector(
        Kind::Cumprod,
        &[2.0, 0.0, 3.0],
        &[0.0, 0.0, 1.0],
        Expect::Accept,
        "cumprod 零1個",
    );
    run_vector(
        Kind::Cumprod,
        &[0.0, 2.0, 0.0, 1.0],
        &ones(4),
        Expect::Accept,
        "cumprod 零2個",
    );
    // Cumprod 拒否: backward.rs の overflow×零遮断の入力（40・50 要素の 1e30）と対称ケース。
    let mut v40 = vec![1e30f32; 40];
    v40[0] = 0.0;
    run_vector(
        Kind::Cumprod,
        &v40,
        &ones(40),
        Expect::Reject,
        "cumprod 40 要素 1e30+零",
    );
    let mut v50 = vec![1e30f32; 50];
    v50[2] = 0.0;
    v50[25] = 0.0;
    run_vector(
        Kind::Cumprod,
        &v50,
        &ones(50),
        Expect::Reject,
        "cumprod 50 要素 1e30+零2個",
    );
    let mut v23 = vec![0f32; 23];
    for v in v23.iter_mut().skip(1).take(11) {
        *v = 2f32.powi(-100);
    }
    for v in v23.iter_mut().skip(12).take(11) {
        *v = 2f32.powi(100);
    }
    run_vector(
        Kind::Cumprod,
        &v23,
        &ones(23),
        Expect::Reject,
        "cumprod suffix overflow",
    );
    let mut v22 = vec![0f32; 22];
    for v in v22.iter_mut().take(11) {
        *v = 2f32.powi(100);
    }
    for v in v22.iter_mut().skip(11) {
        *v = 2f32.powi(-100);
    }
    let mut w22 = vec![0f32; 22];
    w22[21] = 1.0;
    run_vector(
        Kind::Cumprod,
        &v22,
        &w22,
        Expect::Reject,
        "cumprod 対称ケース",
    );

    // Logcumsumexp 受理。
    run_vector(
        Kind::Logcumsumexp,
        &[-1000.0, 1000.0],
        &ones(2),
        Expect::Accept,
        "lcse [-1000,1000]",
    );
    run_vector(
        Kind::Logcumsumexp,
        &[-100.0, -100.0, 80.0],
        &ones(3),
        Expect::Accept,
        "lcse [-100,-100,80]",
    );
    run_vector(
        Kind::Logcumsumexp,
        &[0.0, 0.0],
        &[0.0, 1.0],
        Expect::Accept,
        "lcse 一定 lane",
    );
    // Logcumsumexp 拒否: f32 の出力が ln2 の差を失う・係数の相対誤差が 1e-3 超・上流が大きい。
    run_vector(
        Kind::Logcumsumexp,
        &[1e8, 1e8],
        &[0.0, 1.0],
        Expect::Reject,
        "lcse [1e8,1e8]",
    );
    run_vector(
        Kind::Logcumsumexp,
        &[1e8, 1e8 + 1.0, 1e8],
        &ones(3),
        Expect::Reject,
        "lcse [1e8,1e8+1,1e8]",
    );
    run_vector(
        Kind::Logcumsumexp,
        &[5e4, 5e4],
        &ones(2),
        Expect::Reject,
        "lcse [5e4,5e4]",
    );
    run_vector(
        Kind::Logcumsumexp,
        &[0.0, 0.0],
        &[3e38, 1.0],
        Expect::Reject,
        "lcse g=3e38",
    );
    run_vector(
        Kind::Logcumsumexp,
        &[0.0, 0.0],
        &[3e38, -3e38],
        Expect::Reject,
        "lcse g=[3e38,-3e38]",
    );
}

// --- 入口規則 ---------------------------------------------------------------------

#[test]
fn cum_b_unsafe_inputs_rejected_child_empty_and_retry() {
    for kind in [Kind::Cumprod, Kind::Logcumsumexp] {
        let child = Tape::new_with_ops(common::naive_ops());
        // ±inf 入力は拒否。
        for bad in [f32::INFINITY, f32::NEG_INFINITY] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![1.0, bad, 0.5], &[3]));
            let loss = linear_loss(&tape, kind, &x, 0, &[1.0, 1.0, 1.0]).unwrap();
            let err = tape
                .backward_create_graph(&loss, &child)
                .expect_err("±inf 入力は拒否");
            assert!(
                matches!(err, AutodiffError::Backward(_)),
                "{kind:?}: {err:?}"
            );
            assert!(child.is_empty(), "{kind:?}: child 空");
        }
        // 非有限の upstream は拒否。
        for bad in [f32::INFINITY, f32::NAN] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![1.0, 0.5, 0.7], &[3]));
            let loss = linear_loss(&tape, kind, &x, 0, &[1.0, bad, 1.0]).unwrap();
            let err = tape
                .backward_create_graph(&loss, &child)
                .expect_err("非有限 upstream は拒否");
            assert!(
                matches!(err, AutodiffError::Backward(_)),
                "{kind:?}: {err:?}"
            );
            assert!(child.is_empty(), "{kind:?}: child 空");
        }
        // 同じ child で安全な入力を再試行すると成功する。
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0, 0.5, 0.7], &[3]));
        let loss = linear_loss(&tape, kind, &x, 0, &[1.0, 1.0, 1.0]).unwrap();
        tape.backward_create_graph(&loss, &child)
            .expect("拒否後の再試行は成功する");

        // NaN 入力は拒否されず伝播する。
        let child2 = Tape::new_with_ops(common::naive_ops());
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0, f32::NAN, 0.7], &[3]));
        let loss = linear_loss(&tape, kind, &x, 0, &[1.0, 1.0, 1.0]).unwrap();
        let cg = tape
            .backward_create_graph(&loss, &child2)
            .expect("NaN 入力は伝播（拒否しない）");
        let g = cg.grad(&x).unwrap().unwrap().to_tensor();
        assert!(
            g.host_slice().iter().any(|v| v.is_nan()),
            "{kind:?}: NaN が伝播する"
        );
    }
}

#[test]
fn cum_b_empty_tensor_returns_zeros() {
    for kind in [Kind::Cumprod, Kind::Logcumsumexp] {
        for (shape, dim) in [(vec![0usize], 0usize), (vec![2, 0], 1)] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&Tensor::<f32>::zeros(&shape).unwrap());
            let y = kind.apply(&x, dim).unwrap();
            let loss = y.sum(None).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let g = cg.grad(&x).unwrap().unwrap().to_tensor();
            assert_eq!(g.shape(), shape.as_slice());
            assert!(g.host_slice().iter().all(|v| *v == 0.0));
        }
    }
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

/// 累積 Op と未対応 Op（`Op::Custom`）が同居する祖先は、子テープへ一切書かず拒否される。
#[test]
fn cum_b_neighbouring_unsupported_op_stays_fail_closed() {
    for kind in [Kind::Cumprod, Kind::Logcumsumexp] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![0.5, 0.7, 0.9], &[3]));
        let y = kind.apply(&x, 0).unwrap();
        let z = tape
            .custom(Arc::new(IdentityCustomFn), &[y])
            .expect("Tape::custom 登録成功");
        let loss = z.sum(None).unwrap();
        let err = tape
            .backward_create_graph(&loss, &child)
            .expect_err("Op::Custom は fail-closed で拒否される");
        assert!(
            matches!(err, AutodiffError::Backward(_)),
            "{kind:?}: {err:?}"
        );
        assert!(child.is_empty(), "{kind:?}: 拒否時に child へ書き込まない");
        let loss_ok = y.sum(None).unwrap();
        tape.backward_create_graph(&loss_ok, &child)
            .expect("拒否後の再試行は成功する");
    }
}
