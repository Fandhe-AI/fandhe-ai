//! 損失の回帰系 3 variant（`KlDivLoss`・`PoissonNllLoss`・`GaussianNllLoss`）の `create_graph` 対応
//! （イシュー #3798・決定記録 `docs/autodiff-higher-order-grad-decision.md` §39、実装記録 §53）の
//! 受け入れ条件を確かめる統合テスト。
//!
//! - `first_order()` が 1 階 `Tape::backward` と bit 一致し、`cg.grad` は Poisson・Gaussian が bit 一致
//!   （V3）、KlDiv が REQ-2 統一複合判定で一致する。
//! - Hessian を 1 階解析勾配の中央差分と REQ-2 判定で突合する（x・target・var を 1 本の葉から
//!   `narrow` で切り出し、混合二階項も確かめる）。
//! - Gaussian の clamp 区間（`var < eps`）は独立の解析式と突合する。
//! - 総括判定: 必須ベクトルは有限か型付き拒否（拒否時は `child` が空で再試行可能）。
//! - 孫テープ（`jvp_create_graph`）・空テンソル・隣接 `Op::Custom` の fail-closed。
//!
//! GPU parity は `crates/facade/tests/create_graph_gpu_parity.rs`（実機待ち・`#[ignore]`）。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::elementwise_loss_ops::GaussianNllOptions;
use fandhe_ai_autodiff::loss_ops::PoissonNllOptions;
use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(v: impl std::borrow::Borrow<Tensor<f32>>) -> Vec<f32> {
    v.borrow()
        .contiguous()
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn norm_bits(v: &[f32]) -> Vec<u32> {
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

/// 葉 `z`（長さ `k·n`）を `k` 個の `[n]` へ切り出す。
fn parts<'t>(z: &Var<'t>, k: usize, n: usize) -> Vec<Var<'t>> {
    (0..k).map(|i| z.narrow(0, i * n, n).unwrap()).collect()
}

/// 1 要素ずつの Poisson（和）。クロージャの高階ライフタイム推論を返り値型で固定する。
fn poisson_fn(
    log_input: bool,
    full: bool,
) -> impl for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy {
    move |_, z| {
        let p = parts(z, 2, 1);
        p[0].poisson_nll_loss(
            &p[1],
            &PoissonNllOptions::default().log_input(log_input).full(full),
            Reduction::Sum,
        )
    }
}

/// 1 要素ずつの Gaussian（和）。
fn gauss_fn(eps: f32) -> impl for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy {
    move |_, z| {
        let p = parts(z, 3, 1);
        p[0].gaussian_nll_loss(
            &p[1],
            &p[2],
            &GaussianNllOptions::default().eps(eps),
            Reduction::Sum,
        )
    }
}

fn finite_diff_hessian<F>(build: F, z0: &[f32], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = z0.len();
    let grad_at = |z: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let zv = tape.var(&t(z.to_vec(), &[n]));
        let loss = build(&tape, &zv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let d = dense(grads.get(&zv).unwrap().expect("z は loss に到達する"));
        d.iter().map(|&v| f64::from(v)).collect()
    };
    let mut hess = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut zp = z0.to_vec();
        let mut zm = z0.to_vec();
        zp[i] += h as f32;
        zm[i] -= h as f32;
        let (gp, gm) = (grad_at(&zp), grad_at(&zm));
        for j in 0..n {
            hess[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hess
}

fn analytic_hessian<F>(build: F, z0: &[f32]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = z0.len();
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(z0.to_vec(), &[n]));
    let loss = build(&tape, &z).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gz = cg.grad(&z).unwrap().expect("1 階勾配");
    let cz = cg.child_var(&z).unwrap().expect("写し");
    let mut hess = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = gz.narrow(0, j, 1).unwrap().sum(None).unwrap();
        let grads = child.backward(&gj).expect("2 階");
        let row = grads.get(&cz).unwrap().expect("2 階勾配は z へ到達");
        let row = dense(row);
        for i in 0..n {
            hess[j][i] = f64::from(row[i]);
        }
    }
    hess
}

fn hessian_case<F>(ctx: &str, build: F, z0: &[f32])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let analytic = analytic_hessian(build, z0);
    // Richardson 外挿（h と h/2 の中央差分）で打切り誤差を O(h^4) に抑える。
    let (coarse, fine) = (
        finite_diff_hessian(build, z0, 0.04),
        finite_diff_hessian(build, z0, 0.02),
    );
    let numeric: Vec<Vec<f64>> = coarse
        .iter()
        .zip(&fine)
        .map(|(rc, rf)| {
            rc.iter()
                .zip(rf)
                .map(|(c, f)| (4.0 * f - c) / 3.0)
                .collect()
        })
        .collect();
    for (j, (ra, rn)) in analytic.iter().zip(&numeric).enumerate() {
        for (i, (&a, &nv)) in ra.iter().zip(rn).enumerate() {
            assert!(
                common::req2_close(a, nv),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={nv}"
            );
        }
    }
    assert!(
        analytic.iter().flatten().any(|v| v.abs() > 1e-3),
        "{ctx}: Hessian が全ゼロ（検査が空振り）"
    );
}

// ---------------------------------------------------------------------------------
// ケース定義（葉 z を x・target・var へ切り出して損失を作る。root と非 root の両方）
// ---------------------------------------------------------------------------------

const N: usize = 4;

#[derive(Clone, Copy)]
enum Case {
    KlProb(Reduction),
    KlLog(Reduction),
    Poisson {
        log_input: bool,
        full: bool,
        reduction: Reduction,
    },
    Gaussian {
        full: bool,
        reduction: Reduction,
    },
}

fn cases() -> Vec<(String, Case)> {
    let mut v = Vec::new();
    for (rn, r) in [("mean", Reduction::Mean), ("sum", Reduction::Sum)] {
        v.push((format!("kl_prob {rn}"), Case::KlProb(r)));
        v.push((format!("kl_log {rn}"), Case::KlLog(r)));
        for log_input in [true, false] {
            for full in [false, true] {
                v.push((
                    format!("poisson log={log_input} full={full} {rn}"),
                    Case::Poisson {
                        log_input,
                        full,
                        reduction: r,
                    },
                ));
            }
        }
        for full in [false, true] {
            v.push((
                format!("gaussian full={full} {rn}"),
                Case::Gaussian { full, reduction: r },
            ));
        }
    }
    v
}

fn k_of(c: Case) -> usize {
    if matches!(c, Case::Gaussian { .. }) {
        3
    } else {
        2
    }
}

fn build_case<'t>(c: Case, z: &Var<'t>) -> BuildResult<'t> {
    let p = parts(z, k_of(c), N);
    match c {
        Case::KlProb(r) => p[0].kl_div_loss(&p[1], r),
        Case::KlLog(r) => p[0].kl_div_loss_with_log_target(&p[1], r),
        Case::Poisson {
            log_input,
            full,
            reduction,
        } => p[0].poisson_nll_loss(
            &p[1],
            &PoissonNllOptions::default().log_input(log_input).full(full),
            reduction,
        ),
        Case::Gaussian { full, reduction } => p[0].gaussian_nll_loss(
            &p[1],
            &p[2],
            &GaussianNllOptions::default().full(full),
            reduction,
        ),
    }
}

/// 各ケースで非自明な二階を持つ入力（target は正・Poisson の full は t > 1 を含む・Gaussian の var は eps より大）。
fn z0_of(c: Case) -> Vec<f32> {
    let x = match c {
        Case::Poisson {
            log_input: false, ..
        } => vec![0.6, 1.1, 1.7, 0.9],
        _ => vec![-0.7, 0.3, 0.9, -0.2],
    };
    let tg = match c {
        Case::Poisson { full: true, .. } => vec![0.5, 1.5, 2.5, 3.5],
        Case::KlLog(_) => vec![-1.2, -0.6, -0.3, -0.9],
        _ => vec![0.55, 0.7, 0.85, 0.65],
    };
    let mut z = [x, tg].concat();
    if matches!(c, Case::Gaussian { .. }) {
        z.extend([0.4, 0.8, 1.2, 0.6]);
    }
    z
}

#[test]
fn first_order_bit_identical_and_cgrad_req2() {
    for (name, c) in cases() {
        let z0 = z0_of(c);
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.clone(), &[z0.len()]));
        let loss = build_case(c, &z).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let first = dense(cg.first_order().get(&z).unwrap().unwrap());
        let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
        assert_eq!(norm_bits(&first), norm_bits(&want), "{name}: first_order()");
        if matches!(c, Case::KlProb(_) | Case::KlLog(_)) {
            // KlDiv は融合カーネル f32 scale と構造の f32 が一致するとは限らず REQ-2 で比べる。
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    common::req2_close(f64::from(*a), f64::from(*b)),
                    "{name}: {a} vs {b}"
                );
            }
        } else {
            assert_eq!(norm_bits(&got), norm_bits(&want), "{name}: cg.grad");
        }
    }
}

#[test]
fn hessian_matches_finite_difference() {
    for (name, c) in cases() {
        let z0 = z0_of(c);
        // 損失が根の場合。
        hessian_case(&format!("{name} root"), |_, z| build_case(c, z), &z0);
        // 損失が根でない場合（g = 2·loss が二階へ流れる）。
        hessian_case(
            &format!("{name} non-root"),
            |tape, z| {
                // 係数 0.2 で勾配の桁を抑え、f32 の差分ノイズを REQ-2 の閾値内に保つ。
                let l = build_case(c, z)?.mul(&tape.var_no_grad(&Tensor::scalar(0.2f32)))?;
                l.mul(&l)
            },
            &z0,
        );
    }
}

/// Gaussian の clamp 区間（`var < eps`）。1 階は `c = max(var, eps)` を値 `eps`・`var` についての
/// 微分 1 として扱う（素通し形）ため、独立の解析式で突合する。
#[test]
fn gaussian_clamp_region_matches_analytic() {
    let (x, tg, v, eps) = (0.9f32, 0.4f32, 0.05f32, 0.2f32);
    let build = gauss_fn(eps);
    let hess = analytic_hessian(build, &[x, tg, v]);
    let (d, c) = (f64::from(x - tg), f64::from(eps));
    let want_xv = -d / (c * c);
    let want_vv = 0.5 * (-1.0 / (c * c) + 2.0 * d * d / (c * c * c));
    let chk = |a: f64, b: f64, what: &str| assert!(common::req2_close(a, b), "{what}: {a} vs {b}");
    chk(hess[0][0], 1.0 / c, "dx/dx");
    chk(hess[0][2], want_xv, "dx/dvar");
    chk(hess[1][2], -want_xv, "dt/dvar");
    chk(hess[2][2], want_vv, "dvar/dvar");
}

#[test]
fn grandchild_replay_matches_directional_derivative() {
    for (name, c) in cases() {
        let z0 = z0_of(c);
        let dir: Vec<f32> = (0..z0.len()).map(|i| 0.3 + 0.1 * i as f32).collect();
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.clone(), &[z0.len()]));
        let loss = build_case(c, &z).unwrap();
        let want_grad = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let want: f64 = want_grad
            .iter()
            .zip(&dir)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
            &tape,
            &loss,
            &z,
            &t(dir.clone(), &[dir.len()]),
            &child,
            &grand,
        )
        .unwrap();
        let got = f64::from(res.output().value().get(&[]).unwrap());
        assert!(
            common::req2_close(got, want),
            "{name}: jvp={got} want={want}"
        );
    }
}

// ---------------------------------------------------------------------------------
// 総括判定
// ---------------------------------------------------------------------------------

/// 損失を作って create_graph を試み、受理なら 3 段（1 階・`cg.grad`・2 階）が有限、拒否なら
/// 型付きエラーで `child` が空であることを確かめる。`true` = 受理。
fn try_case<F>(ctx: &str, build: F, z0: &[f32]) -> bool
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(z0.to_vec(), &[z0.len()]));
    let loss = build(&tape, &z).unwrap();
    let first = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => {
            assert!(
                first.iter().all(|v| v.is_finite()),
                "{ctx}: 1 階が非有限 {first:?}"
            );
            let gz = cg.grad(&z).unwrap().unwrap();
            assert!(
                dense(gz.to_tensor()).iter().all(|v| v.is_finite()),
                "{ctx}: cg.grad"
            );
            let s = gz.sum(None).unwrap();
            let cz = cg.child_var(&z).unwrap().unwrap();
            let h = child.backward(&s).unwrap();
            if let Some(row) = h.get(&cz).unwrap() {
                assert!(dense(row).iter().all(|v| v.is_finite()), "{ctx}: 2 階");
            }
            true
        }
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: {e:?}");
            assert!(child.is_empty(), "{ctx}: 拒否後に child が空でない");
            false
        }
    }
}

#[test]
fn mandatory_vectors_finite_or_rejected_in_three_stages() {
    let poisson = poisson_fn;
    // Poisson: 大きな x（exp が段上界を超える）は拒否・通常値と t=1 は受理。
    assert!(!try_case(
        "poisson x=100",
        poisson(true, false),
        &[100.0, 1.0]
    ));
    assert!(try_case("poisson x=2", poisson(true, false), &[2.0, 1.0]));
    assert!(!try_case(
        "poisson x+eps=0",
        poisson(false, false),
        &[-1e-8, 1.0]
    ));
    assert!(try_case(
        "poisson t=1 full",
        poisson(true, true),
        &[0.5, 1.0]
    ));
    assert!(try_case(
        "poisson t large full",
        poisson(true, true),
        &[0.5, 1e6]
    ));
    // Gaussian: var=0<eps は受理（c=eps）・巨大な d は拒否・eps=1e-6 で d²/c² が overflow するなら拒否。
    let gauss = gauss_fn;
    assert!(try_case("gaussian var=0", gauss(1e-3), &[0.5, 0.2, 0.0]));
    assert!(!try_case("gaussian d=1e20", gauss(1e-3), &[1e20, 0.0, 1.0]));
    assert!(!try_case(
        "gaussian d/c overflow",
        gauss(1e-6),
        &[1e15, 0.0, 0.0]
    ));
    // KlDiv: t=0 は受理（t_safe）・LogProbabilities の t=90 は exp が overflow で拒否。
    assert!(try_case(
        "kl t=0",
        |_, z| {
            let p = parts(z, 2, 2);
            p[0].kl_div_loss(&p[1], Reduction::Sum)
        },
        &[0.3, -0.2, 0.0, 0.5],
    ));
    assert!(!try_case(
        "kl log t=90",
        |_, z| {
            let p = parts(z, 2, 1);
            p[0].kl_div_loss_with_log_target(&p[1], Reduction::Sum)
        },
        &[0.3, 90.0],
    ));
}

#[test]
fn huge_upstream_scale_is_rejected_or_finite() {
    for scale in [3e38f32, -3e38] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(vec![0.3, 0.7, 0.4, 0.6], &[4]));
        let p = parts(&z, 2, 2);
        let l = p[0].kl_div_loss(&p[1], Reduction::Sum).unwrap();
        let big = tape.var_no_grad(&t(vec![scale], &[1]));
        let loss = l.mul(&big).unwrap().sum(None).unwrap();
        let err = tape.backward_create_graph(&loss, &child).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty());
        // 同じ child で安全な損失を再試行できる。
        let ok = l
            .mul(&tape.var_no_grad(&t(vec![2.0], &[1])))
            .unwrap()
            .sum(None)
            .unwrap();
        assert!(tape.backward_create_graph(&ok, &child).is_ok());
    }
    // 2 つの損失ノードへ ±3e38 を流す組。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![0.3, 0.7, 0.4, 0.6], &[4]));
    let p = parts(&z, 2, 2);
    let a = p[0].kl_div_loss(&p[1], Reduction::Sum).unwrap();
    let b = p[0]
        .kl_div_loss_with_log_target(&p[1], Reduction::Sum)
        .unwrap();
    let c_pos = tape.var_no_grad(&t(vec![3e38], &[1]));
    let c_neg = tape.var_no_grad(&t(vec![-3e38], &[1]));
    let loss = a
        .mul(&c_pos)
        .unwrap()
        .add(&b.mul(&c_neg).unwrap())
        .unwrap()
        .sum(None)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
    assert!(child.is_empty());
}

#[test]
fn piecewise_boundaries_follow_first_order_rule() {
    // KlDiv(Probabilities) の t == 0・Poisson(full) の t == 1。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![0.3, 0.2, 0.0, 0.5], &[4]));
    let p = parts(&z, 2, 2);
    let loss = p[0].kl_div_loss(&p[1], Reduction::Sum).unwrap();
    let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
    for (a, b) in got.iter().zip(&want) {
        assert!(
            common::req2_close(f64::from(*a), f64::from(*b)),
            "kl t=0: {a} vs {b}"
        );
    }

    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![0.3, 0.2, 1.0, 1.0], &[4]));
    let p = parts(&z, 2, 2);
    let loss = p[0]
        .poisson_nll_loss(
            &p[1],
            &PoissonNllOptions::default().full(true),
            Reduction::Sum,
        )
        .unwrap();
    let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
    assert_eq!(norm_bits(&got), norm_bits(&want), "poisson t=1");

    // Gaussian var == eps（c = var = eps で両側の規則が一致する）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![0.9, 0.4, 0.2], &[3]));
    let p = parts(&z, 3, 1);
    let loss = p[0]
        .gaussian_nll_loss(
            &p[1],
            &p[2],
            &GaussianNllOptions::default().eps(0.2),
            Reduction::Sum,
        )
        .unwrap();
    let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    assert_eq!(
        norm_bits(&dense(cg.grad(&z).unwrap().unwrap().to_tensor())),
        norm_bits(&want)
    );
}

#[test]
fn unsafe_inputs_rejected_child_empty_and_retry_nan_propagates() {
    // ±inf の入力は拒否・child は空で、同じ child で通常の損失を再試行できる。
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(vec![bad, 0.5, 0.4, 0.6], &[4]));
        let p = parts(&z, 2, 2);
        let loss = p[0]
            .poisson_nll_loss(&p[1], &PoissonNllOptions::default(), Reduction::Sum)
            .unwrap();
        assert!(tape.backward_create_graph(&loss, &child).is_err());
        assert!(child.is_empty());
        let z2 = tape.var(&t(vec![0.2, 0.5, 0.4, 0.6], &[4]));
        let p2 = parts(&z2, 2, 2);
        let ok = p2[0]
            .poisson_nll_loss(&p2[1], &PoissonNllOptions::default(), Reduction::Sum)
            .unwrap();
        assert!(tape.backward_create_graph(&ok, &child).is_ok());
    }
    // NaN は伝播し、値のクラスが 1 階と一致する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![f32::NAN, 0.5, 0.4, 0.6], &[4]));
    let p = parts(&z, 2, 2);
    let loss = p[0]
        .poisson_nll_loss(&p[1], &PoissonNllOptions::default(), Reduction::Sum)
        .unwrap();
    let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
    assert_eq!(norm_bits(&got), norm_bits(&want));
}

#[test]
fn empty_tensor_returns_zeros() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 3]));
    let tg = tape.var(&t(vec![], &[0, 3]));
    let v = tape.var(&t(vec![], &[0, 3]));
    for loss in [
        x.kl_div_loss(&tg, Reduction::Sum).unwrap(),
        x.poisson_nll_loss(&tg, &PoissonNllOptions::default(), Reduction::Sum)
            .unwrap(),
        x.gaussian_nll_loss(&tg, &v, &GaussianNllOptions::default(), Reduction::Sum)
            .unwrap(),
    ] {
        let child = Tape::new_with_ops(common::naive_ops());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let g = cg.grad(&x).unwrap().unwrap();
        assert_eq!(g.value().shape(), &[0usize, 3]);
    }
}

// ---------------------------------------------------------------------------------
// 隣接の fail-closed
// ---------------------------------------------------------------------------------

struct IdentityCustomFn;

impl fandhe_ai_autodiff::CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok((*inputs[0]).clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![Some(upstream.clone())])
    }
}

#[test]
fn neighbouring_custom_stays_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![0.3, 0.7, 0.4, 0.6], &[4]));
    let p = parts(&z, 2, 2);
    let l = p[0].kl_div_loss(&p[1], Reduction::Mean).unwrap();
    let out = tape.custom(Arc::new(IdentityCustomFn), &[l]).unwrap();
    let err = tape.backward_create_graph(&out, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty());
}
