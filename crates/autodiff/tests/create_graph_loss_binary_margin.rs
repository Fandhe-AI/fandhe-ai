//! 損失の二値・マージン系 6 variant（`BceLoss`・`BceWithLogitsPosWeightLoss`・`SoftMarginLoss`・
//! `HingeEmbeddingLoss`・`MarginRankingLoss`・`SigmoidFocalLoss`）の `create_graph` 対応
//! （イシュー #3799・決定記録 `docs/autodiff-higher-order-grad-decision.md` §39、実装記録 §54）の
//! 受け入れ条件を確かめる統合テスト。
//!
//! - `first_order()` が 1 階 `Tape::backward` と bit 一致し、`cg.grad` は V3 の 5 variant が bit 一致、
//!   Bce（融合カーネル経路・V1）が REQ-2 統一複合判定で一致する。
//! - Hessian を 1 階解析勾配の中央差分（h と h/2 の Richardson 外挿）と REQ-2 判定で突合する。
//! - 混合二階項（追跡する target との `d²L/dx·dt`）を独立の解析式と突合する。
//! - 総括判定: 必須ベクトルは有限か型付き拒否（拒否時は `child` が空で再試行可能）。
//! - 孫テープ（`jvp_create_graph`）・空テンソル・隣接 `Op::Custom` の fail-closed。
//!
//! GPU parity は `crates/facade/tests/create_graph_gpu_parity.rs`（実機待ち・`#[ignore]`）。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::elementwise_loss_ops::BceWithLogitsOptions;
use fandhe_ai_autodiff::margin_focal_loss_ops::SigmoidFocalLossOptions;
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
        // 区分定数な成分（Hinge・MarginRanking の root）は追跡対象の祖先を持たず逆伝播できない（= 全ゼロ）。
        let Ok(grads) = child.backward(&gj) else {
            continue;
        };
        // 区分定数な損失（Hinge・MarginRanking）の root では 2 階が z へ到達しない（全ゼロ）。
        if let Some(row) = grads.get(&cz).unwrap() {
            let row = dense(row);
            for i in 0..n {
                hess[j][i] = f64::from(row[i]);
            }
        }
    }
    hess
}

fn hessian_case<F>(ctx: &str, build: F, z0: &[f32], expect_nonzero: bool, h: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let analytic = analytic_hessian(build, z0);
    let (coarse, fine) = (
        finite_diff_hessian(build, z0, h),
        finite_diff_hessian(build, z0, h / 2.0),
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
    if expect_nonzero {
        assert!(
            analytic.iter().flatten().any(|v| v.abs() > 1e-3),
            "{ctx}: Hessian が全ゼロ（検査が空振り）"
        );
    }
}

// ---------------------------------------------------------------------------------
// ケース定義（葉 z を x・target 等へ切り出して損失を作る。root と非 root の両方）
// ---------------------------------------------------------------------------------

const N: usize = 4;
const Y_PM: [f32; N] = [1.0, -1.0, -1.0, 1.0];
const Y_RANK: [f32; N] = [1.0, -1.0, 1.0, -1.0];

#[derive(Clone, Copy)]
enum Kind {
    BceProb,
    BceLogits,
    PosWeight(f32),
    SoftMargin,
    Hinge,
    Rank,
    Focal { gamma: f32, alpha: Option<f32> },
}

#[derive(Clone, Copy)]
struct Case {
    kind: Kind,
    reduction: Reduction,
}

fn cases() -> Vec<(String, Case)> {
    let mut v = Vec::new();
    for (rn, reduction) in [("mean", Reduction::Mean), ("sum", Reduction::Sum)] {
        let mut kinds = vec![
            ("bce_prob".to_string(), Kind::BceProb),
            ("bce_logits".to_string(), Kind::BceLogits),
            ("soft_margin".to_string(), Kind::SoftMargin),
            ("hinge".to_string(), Kind::Hinge),
            ("rank".to_string(), Kind::Rank),
        ];
        for p in [0.0f32, 1.0, 3.0] {
            kinds.push((format!("pos_weight={p}"), Kind::PosWeight(p)));
        }
        for gamma in [0.0f32, 0.5, 1.0, 2.0] {
            for alpha in [None, Some(0.25f32)] {
                kinds.push((
                    format!("focal gamma={gamma} alpha={alpha:?}"),
                    Kind::Focal { gamma, alpha },
                ));
            }
        }
        for (name, kind) in kinds {
            v.push((format!("{name} {rn}"), Case { kind, reduction }));
        }
    }
    v
}

fn k_of(kind: Kind) -> usize {
    match kind {
        Kind::BceProb | Kind::BceLogits | Kind::PosWeight(_) | Kind::Rank => 2,
        Kind::SoftMargin | Kind::Hinge | Kind::Focal { .. } => 1,
    }
}

fn build_case<'t>(c: Case, z: &Var<'t>) -> BuildResult<'t> {
    let p = parts(z, k_of(c.kind), N);
    let r = c.reduction;
    match c.kind {
        Kind::BceProb => p[0].bce_loss(&p[1], r),
        Kind::BceLogits => p[0].bce_with_logits_loss(&p[1], r),
        Kind::PosWeight(pw) => p[0].bce_with_logits_loss_with(
            &p[1],
            r,
            &BceWithLogitsOptions::default().pos_weight(t(vec![pw], &[1])),
        ),
        Kind::SoftMargin => p[0].soft_margin_loss(&t(Y_PM.to_vec(), &[N]), r),
        Kind::Hinge => p[0].hinge_embedding_loss(&t(Y_PM.to_vec(), &[N]), 1.0, r),
        Kind::Rank => p[0].margin_ranking_loss(&p[1], &t(Y_RANK.to_vec(), &[N]), 0.5, r),
        Kind::Focal { gamma, alpha } => p[0].sigmoid_focal_loss(
            &t(vec![1.0, 0.0, 0.7, 0.2], &[N]),
            &SigmoidFocalLossOptions::default().alpha(alpha).gamma(gamma),
            r,
        ),
    }
}

/// 各ケースで非自明な二階を持つ入力。区分定数のケースは境界から 0.04 超離す。
fn z0_of(kind: Kind) -> Vec<f32> {
    match kind {
        Kind::BceProb => vec![0.2, 0.4, 0.65, 0.85, 0.1, 0.9, 0.5, 0.3],
        Kind::BceLogits | Kind::PosWeight(_) => vec![-0.7, 0.3, 0.9, -0.2, 0.2, 0.8, 0.5, 0.35],
        Kind::SoftMargin => vec![-0.7, 0.3, 0.9, -0.2],
        Kind::Hinge => vec![-0.7, 0.3, 0.6, 1.8],
        Kind::Rank => vec![0.9, 0.2, -0.3, 0.8, 0.1, 0.6, 0.4, 0.1],
        Kind::Focal { .. } => vec![-0.7, 0.3, 0.9, -0.2],
    }
}

/// root で Hessian が非零になりうるか（Hinge・Rank は x について区分定数）。
fn root_hessian_nonzero(kind: Kind) -> bool {
    !matches!(kind, Kind::Hinge | Kind::Rank)
}

/// 中央差分の刻み。`1/(p(1−p))` が急な Bce Probabilities だけ細かくする（打切り誤差の抑制）。
fn step_of(kind: Kind) -> f64 {
    if matches!(kind, Kind::BceProb) {
        0.01
    } else {
        0.04
    }
}

fn is_bce(kind: Kind) -> bool {
    matches!(kind, Kind::BceProb | Kind::BceLogits)
}

#[test]
fn first_order_bit_identical_and_cgrad_req2() {
    for (name, c) in cases() {
        let z0 = z0_of(c.kind);
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.clone(), &[z0.len()]));
        let loss = build_case(c, &z).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let first = dense(cg.first_order().get(&z).unwrap().unwrap());
        let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
        assert_eq!(norm_bits(&first), norm_bits(&want), "{name}: first_order()");
        if is_bce(c.kind) {
            // Bce は融合カーネル f32 scale と構造の f32 が一致するとは限らず REQ-2 で比べる。
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

/// 損失が根でない場合、子テープへ写された損失 Op は forward と bit 一致する値を持つ
/// （再生した `BceWithLogitsPosWeightLoss` の `pos_weight` が親と同じ payload になる確認を含む）。
#[test]
fn replayed_loss_value_matches_forward() {
    for (name, c) in cases() {
        let z0 = z0_of(c.kind);
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.clone(), &[z0.len()]));
        let l = build_case(c, &z).unwrap();
        let loss = l.mul(&l).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let lm = cg.child_var(&l).unwrap().expect("損失の写し");
        assert_eq!(
            norm_bits(&dense(lm.to_tensor())),
            norm_bits(&dense(l.to_tensor())),
            "{name}: 写しの損失値"
        );
    }
}

#[test]
fn hessian_matches_finite_difference() {
    for (name, c) in cases() {
        let z0 = z0_of(c.kind);
        // 損失が根の場合。
        hessian_case(
            &format!("{name} root"),
            |_, z| build_case(c, z),
            &z0,
            root_hessian_nonzero(c.kind),
            step_of(c.kind),
        );
        // 損失が根でない場合（g = 0.2·loss が二階へ流れる。区分定数のケースも非零）。
        hessian_case(
            &format!("{name} non-root"),
            |tape, z| {
                let l = build_case(c, z)?.mul(&tape.var_no_grad(&Tensor::scalar(0.2f32)))?;
                l.mul(&l)
            },
            &z0,
            true,
            step_of(c.kind),
        );
    }
}

/// 追跡する target との混合二階項 `d²L/dx·dt` を独立の解析式と突合する。
#[test]
fn mixed_second_derivatives_with_tracked_target() {
    let x = [-0.7f32, 0.3, 0.9, -0.2];
    let tg = [0.2f32, 0.8, 0.5, 0.35];
    let z0 = [x.to_vec(), tg.to_vec()].concat();
    // BCEWithLogits(Sum): dx = σ(x) − t ⇒ ∂dx/∂t = −1、dt = −x ⇒ ∂dt/∂x = −1。
    fn build<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        let p = parts(z, 2, N);
        p[0].bce_with_logits_loss(&p[1], Reduction::Sum)
    }
    let hess = analytic_hessian(build, &z0);
    for j in 0..N {
        assert!(common::req2_close(hess[j][N + j], -1.0), "logits dx/dt");
        assert!(common::req2_close(hess[N + j][j], -1.0), "logits dt/dx");
    }
    // PosWeight P=3 (Sum): ∂dx/∂t = −1 − (P−1)σ(−x)、∂dt/∂x = −1 + (P−1)·(−σ(−x))·(−1)·… = −1 − (P−1)σ(−x)。
    fn build_pw<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        let p = parts(z, 2, N);
        p[0].bce_with_logits_loss_with(
            &p[1],
            Reduction::Sum,
            &BceWithLogitsOptions::default().pos_weight(t(vec![3.0], &[1])),
        )
    }
    let hess = analytic_hessian(build_pw, &z0);
    for j in 0..N {
        let s = 1.0 / (1.0 + f64::from(x[j]).exp());
        let want = -1.0 - 2.0 * s;
        assert!(
            common::req2_close(hess[j][N + j], want),
            "pos_weight dx/dt: {} vs {want}",
            hess[j][N + j]
        );
        assert!(
            common::req2_close(hess[N + j][j], want),
            "pos_weight dt/dx: {} vs {want}",
            hess[N + j][j]
        );
    }
    // MarginRanking の x1・x2 交差項: 区分定数なので root では 0（非 root の g 経由は Hessian テストで確認）。
    let z_rank = z0_of(Kind::Rank);
    let hess = analytic_hessian(
        |_, z| {
            build_case(
                Case {
                    kind: Kind::Rank,
                    reduction: Reduction::Sum,
                },
                z,
            )
        },
        &z_rank,
    );
    assert!(
        hess.iter().flatten().all(|v| *v == 0.0),
        "rank root hessian"
    );
}

#[test]
fn grandchild_replay_matches_directional_derivative() {
    for (name, c) in cases() {
        let z0 = z0_of(c.kind);
        let dir: Vec<f32> = (0..z0.len()).map(|i| 0.3 + 0.1 * i as f32).collect();
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.clone(), &[z0.len()]));
        let l = build_case(c, &z).unwrap();
        // 非 root にして、写された損失 Op の孫テープでの再生も通す。
        let loss = l.mul(&tape.var_no_grad(&Tensor::scalar(0.5f32))).unwrap();
        let loss = loss.mul(&loss).unwrap();
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

fn one<'t>(z: &Var<'t>, k: usize) -> Vec<Var<'t>> {
    parts(z, k, 1)
}

fn focal_fn(
    gamma: f32,
    target: f32,
) -> impl for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy {
    move |_, z| {
        one(z, 1)[0].sigmoid_focal_loss(
            &t(vec![target], &[1]),
            &SigmoidFocalLossOptions::default().gamma(gamma),
            Reduction::Sum,
        )
    }
}

#[test]
fn mandatory_vectors_finite_or_rejected_in_three_stages() {
    // Bce Probabilities: p = 0・1・1e-30（受理なら 3 段で有限、拒否なら child が空）。
    for p in [0.0f32, 1.0, 1e-30] {
        try_case(
            &format!("bce_prob p={p}"),
            |_, z| {
                let q = one(z, 2);
                q[0].bce_loss(&q[1], Reduction::Sum)
            },
            &[p, 0.5],
        );
    }
    // Logits x = ±1e30、PosWeight x = ±1e30・pos_weight = 0、SoftMargin x = ±1e30。
    for x in [1e30f32, -1e30] {
        try_case(
            &format!("bce_logits x={x}"),
            |_, z| {
                let q = one(z, 2);
                q[0].bce_with_logits_loss(&q[1], Reduction::Sum)
            },
            &[x, 0.5],
        );
        for pw in [0.0f32, 3.0] {
            try_case(
                &format!("pos_weight={pw} x={x}"),
                move |_, z| {
                    let q = one(z, 2);
                    q[0].bce_with_logits_loss_with(
                        &q[1],
                        Reduction::Sum,
                        &BceWithLogitsOptions::default().pos_weight(t(vec![pw], &[1])),
                    )
                },
                &[x, 0.5],
            );
        }
        try_case(
            &format!("soft_margin x={x}"),
            |_, z| one(z, 1)[0].soft_margin_loss(&t(vec![1.0], &[1]), Reduction::Sum),
            &[x],
        );
    }
    // focal の飽和域（f32 で q が 0）。γ<1 と 1<γ<2 は拒否、γ=0 と γ=2 は受理。
    assert!(!try_case(
        "focal gamma=0.5 sat",
        focal_fn(0.5, 0.0),
        &[-200.0]
    ));
    assert!(!try_case(
        "focal gamma=1.5 sat",
        focal_fn(1.5, 0.0),
        &[-200.0]
    ));
    assert!(try_case("focal gamma=0 sat", focal_fn(0.0, 0.0), &[-200.0]));
    assert!(try_case("focal gamma=2 sat", focal_fn(2.0, 0.0), &[-200.0]));
    assert!(try_case("focal gamma=1 sat", focal_fn(1.0, 0.0), &[-200.0]));
    // 通常域は受理。
    assert!(try_case("focal gamma=0.5", focal_fn(0.5, 1.0), &[0.7]));
}

/// 上流が巨大（`g = ±3e38`）な損失は総括判定で拒否し、`child` は空で再試行できる。
#[test]
fn huge_upstream_scale_is_rejected_or_finite() {
    for scale in [3e38f32, -3e38] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(vec![0.3, 0.7, 0.4, 0.6], &[4]));
        let p = parts(&z, 2, 2);
        let l = p[0].bce_with_logits_loss(&p[1], Reduction::Sum).unwrap();
        let big = tape.var_no_grad(&t(vec![scale], &[1]));
        let loss = l.mul(&big).unwrap().sum(None).unwrap();
        let err = tape.backward_create_graph(&loss, &child).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty());
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
    let a = p[0].bce_with_logits_loss(&p[1], Reduction::Sum).unwrap();
    let b = p[0]
        .soft_margin_loss(&t(vec![1.0, -1.0], &[2]), Reduction::Sum)
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
    let cmp =
        |ctx: &str, build: &dyn for<'t> Fn(&Var<'t>) -> BuildResult<'t>, z0: &[f32], bit: bool| {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let z = tape.var(&t(z0.to_vec(), &[z0.len()]));
            let loss = build(&z).unwrap();
            let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
            if bit {
                assert_eq!(norm_bits(&got), norm_bits(&want), "{ctx}");
            } else {
                for (a, b) in got.iter().zip(&want) {
                    assert!(
                        common::req2_close(f64::from(*a), f64::from(*b)),
                        "{ctx}: {a} vs {b}"
                    );
                }
            }
        };
    // Hinge: x == margin（y = −1 で勾配 0、y = +1 は s）。
    cmp(
        "hinge x==margin",
        &|z| z.hinge_embedding_loss(&t(vec![-1.0, 1.0], &[2]), 1.0, Reduction::Sum),
        &[1.0, 1.0],
        true,
    );
    // MarginRanking: raw == 0（x1=1.0・x2=0.5・y=1・margin=0.5 → raw = 0 で通す）。
    cmp(
        "rank raw==0",
        &|z| {
            let p = parts(z, 2, 1);
            p[0].margin_ranking_loss(&p[1], &t(vec![1.0], &[1]), 0.5, Reduction::Sum)
        },
        &[1.0, 0.5],
        true,
    );
    // Bce Probabilities の ln 境界（-100 ちょうど付近: f32 非正規数 26 単位は clamp 側）。
    let clamped = 26.0 * f32::from_bits(1);
    cmp(
        "bce_prob ln clamp side",
        &|z| {
            let p = parts(z, 2, 1);
            p[0].bce_loss(&p[1], Reduction::Sum)
        },
        &[clamped, 0.5],
        false,
    );
    // p(1−p) = 1e-12 付近（D の clamp 境界）。受理なら 1 階と一致、拒否なら child が空。
    for p in [1e-12f32, 1.0000001e-12, 0.9999999e-12] {
        try_case(
            &format!("bce_prob D boundary p={p}"),
            |_, z| {
                let q = one(z, 2);
                q[0].bce_loss(&q[1], Reduction::Sum)
            },
            &[p, 0.5],
        );
    }
    // 非 clamp 側の ln 境界（27 単位）は 1/u が f32 で overflow するため拒否。
    let unclamped = 27.0 * f32::from_bits(1);
    assert!(!try_case(
        "bce_prob ln unclamp side",
        |_, z| {
            let q = one(z, 2);
            q[0].bce_loss(&q[1], Reduction::Sum)
        },
        &[unclamped, 0.5],
    ));
}

#[test]
fn unsafe_inputs_rejected_child_empty_and_retry_nan_propagates() {
    // ±inf の入力は拒否・child は空で、同じ child で通常の損失を再試行できる。
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(vec![bad, 0.5, 0.4, 0.6], &[4]));
        let p = parts(&z, 2, 2);
        let loss = p[0].bce_with_logits_loss(&p[1], Reduction::Sum).unwrap();
        assert!(tape.backward_create_graph(&loss, &child).is_err());
        assert!(child.is_empty());
        let z2 = tape.var(&t(vec![0.2, 0.5, 0.4, 0.6], &[4]));
        let p2 = parts(&z2, 2, 2);
        let ok = p2[0].bce_with_logits_loss(&p2[1], Reduction::Sum).unwrap();
        assert!(tape.backward_create_graph(&ok, &child).is_ok());
    }
    // NaN は伝播し、値のクラスが 1 階と一致する（V3 の PosWeight は bit 一致）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![f32::NAN, 0.5, 0.4, 0.6], &[4]));
    let p = parts(&z, 2, 2);
    let loss = p[0]
        .bce_with_logits_loss_with(
            &p[1],
            Reduction::Sum,
            &BceWithLogitsOptions::default().pos_weight(t(vec![2.0], &[1])),
        )
        .unwrap();
    let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
    assert_eq!(norm_bits(&got), norm_bits(&want));
    // MarginRanking の ±inf も拒否する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![f32::INFINITY, 0.5], &[2]));
    let p = parts(&z, 2, 1);
    let loss = p[0]
        .margin_ranking_loss(&p[1], &t(vec![1.0], &[1]), 0.5, Reduction::Sum)
        .unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
    assert!(child.is_empty());
}

#[test]
fn empty_tensor_returns_zeros() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 3]));
    let tg = tape.var(&t(vec![], &[0, 3]));
    let y = t(vec![], &[0, 3]);
    for loss in [
        x.bce_loss(&tg, Reduction::Sum).unwrap(),
        x.bce_with_logits_loss(&tg, Reduction::Sum).unwrap(),
        x.bce_with_logits_loss_with(
            &tg,
            Reduction::Sum,
            &BceWithLogitsOptions::default().pos_weight(t(vec![2.0], &[1])),
        )
        .unwrap(),
        x.soft_margin_loss(&y, Reduction::Sum).unwrap(),
        x.hinge_embedding_loss(&y, 1.0, Reduction::Sum).unwrap(),
        x.margin_ranking_loss(&tg, &y, 0.5, Reduction::Sum).unwrap(),
        x.sigmoid_focal_loss(&y, &SigmoidFocalLossOptions::default(), Reduction::Sum)
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
    let l = p[0].bce_with_logits_loss(&p[1], Reduction::Mean).unwrap();
    let out = tape.custom(Arc::new(IdentityCustomFn), &[l]).unwrap();
    let err = tape.backward_create_graph(&out, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty());
}

// ---------------------------------------------------------------------------------
// レビュー指摘（PR #3944・codex P1 3 件）の回帰: 上界検査の演算順序と飽和域の曲率保持
// ---------------------------------------------------------------------------------

/// 損失へ `scale` を掛けて create_graph し、入力 `z0[0]` の二階微分（スカラー）を返す。
fn scaled_second_derivative<F>(build: F, z0: &[f32], scale: f32) -> Result<f64, AutodiffError>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    scaled_mixed_derivative(build, z0, scale, 0, 0)
}

/// 損失へ `scale` を掛けて create_graph し、`∂(grad[row]) / ∂z[col]` を返す。
fn scaled_mixed_derivative<F>(
    build: F,
    z0: &[f32],
    scale: f32,
    row: usize,
    col: usize,
) -> Result<f64, AutodiffError>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(z0.to_vec(), &[z0.len()]));
    let l = build(&tape, &z)?;
    let loss = l.mul(&tape.var_no_grad(&t(vec![scale], &[1])))?.sum(None)?;
    let cg = tape.backward_create_graph(&loss, &child)?;
    let gz = cg.grad(&z)?.expect("1 階勾配");
    let cz = cg.child_var(&z)?.expect("写し");
    let h = child.backward(&gz.narrow(0, row, 1)?.sum(None)?)?;
    Ok(f64::from(dense(h.get(&cz)?.expect("2 階"))[col]))
}

/// gamma * ce 自体が f32 で溢れる入力（gamma = 1e30）は、q^(gamma-1) = 0 で積が小さくても拒否する。
#[test]
fn focal_gamma_times_ce_overflow_is_rejected() {
    assert!(!try_case("focal gamma=1e30", focal_fn(1e30, 0.5), &[1e10]));
}

/// pos_weight 付き BCE（x = -20・t = 0・pw = 1・損失 x 1e10）の二階が σ'(x) x 1e10（約 20.6）を保つ。
#[test]
fn bce_pos_weight_keeps_curvature_in_saturated_region() {
    let x = -20.0f64;
    let s = 1.0 / (1.0 + (-x).exp());
    let expect = s * (1.0 - s) * 1e10;
    let got = scaled_second_derivative(
        |_, z| {
            let q = one(z, 2);
            q[0].bce_with_logits_loss_with(
                &q[1],
                Reduction::Sum,
                &BceWithLogitsOptions::default().pos_weight(t(vec![1.0], &[1])),
            )
        },
        &[-20.0, 0.0],
        1e10,
    )
    .expect("受理");
    assert!(
        (got - expect).abs() / expect < 1e-3,
        "got={got} expect={expect}"
    );
}

/// focal（gamma = 0.5・t = 0・x = -15・損失 x 1e10）の二階が f64 参照を保つ（ce の桁落ちなし）。
#[test]
fn focal_keeps_curvature_without_ce_cancellation() {
    let g = 0.5f64;
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let sp = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();
    // t = 0: L = softplus(x)·σ(x)^γ。L' を解析式で書き、中央差分で L'' を得る。
    let d1 = |x: f64| {
        let p = sig(x);
        p.powf(g + 1.0) + g * sp(x) * p.powf(g) * (1.0 - p)
    };
    let h = 1e-4;
    let expect = (d1(-15.0 + h) - d1(-15.0 - h)) / (2.0 * h) * 1e10;
    let got = scaled_second_derivative(
        |_, z| {
            one(z, 1)[0].sigmoid_focal_loss(
                &t(vec![0.0], &[1]),
                &SigmoidFocalLossOptions::default().alpha(None).gamma(0.5),
                Reduction::Sum,
            )
        },
        &[-15.0],
        1e10,
    )
    .expect("受理");
    assert!(
        (got - expect).abs() / expect.abs() < 1e-3,
        "got={got} expect={expect}"
    );
}

fn sigma_prime(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 - s)
}

/// BCE logits（x = 20・t = 0・損失 x 1e10）の二階が σ'(x) x 1e10（約 20.6）を保つ。
#[test]
fn bce_logits_keeps_curvature_in_positive_saturation() {
    let expect = sigma_prime(20.0) * 1e10;
    let got = scaled_second_derivative(
        |_, z| {
            let q = one(z, 2);
            q[0].bce_with_logits_loss(&q[1], Reduction::Sum)
        },
        &[20.0, 0.0],
        1e10,
    )
    .expect("受理");
    assert!(
        (got - expect).abs() / expect < 1e-3,
        "got={got} expect={expect}"
    );
}

/// BCE logits の一階値は大きなスケール（x = 20・t = 1・Sum x 1e10）でも既存 backward と REQ-2 一致する
/// （安定形の値 -2e-9 x 1e10 = -20.6 を返さない）。二階は別テストで保持を確認している。
#[test]
fn bce_logits_first_order_matches_existing_backward_at_large_scale() {
    for (x, tg) in [(20.0f32, 1.0f32), (-20.0, 0.0), (20.0, 0.0), (-20.0, 1.0)] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(vec![x], &[1]));
        let target = tape.var_no_grad(&t(vec![tg], &[1]));
        let loss = z
            .bce_with_logits_loss(&target, Reduction::Sum)
            .unwrap()
            .mul(&tape.var_no_grad(&t(vec![1e10], &[1])))
            .unwrap()
            .sum(None)
            .unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
        for (a, b) in got.iter().zip(&want) {
            assert!(
                common::req2_close(f64::from(*a), f64::from(*b)),
                "x={x} t={tg}: {a} vs {b}"
            );
        }
    }
}

/// SoftMargin（x = -20・y = 1・損失 x 1e10）の二階が σ'(20) x 1e10 を保つ。
#[test]
fn soft_margin_keeps_curvature_in_saturation() {
    let expect = sigma_prime(20.0) * 1e10;
    let got = scaled_second_derivative(
        |_, z| one(z, 1)[0].soft_margin_loss(&t(vec![1.0], &[1]), Reduction::Sum),
        &[-20.0],
        1e10,
    )
    .expect("受理");
    assert!(
        (got - expect).abs() / expect < 1e-3,
        "got={got} expect={expect}"
    );
}

/// pos_weight = 1e-8・t = 1・x = 0（損失 x 1e20）で lw が 0 に丸まらず二階が 2.5e11 を保つ。
#[test]
fn bce_pos_weight_tiny_keeps_curvature() {
    let expect = 1e-8 * 0.25 * 1e20;
    let got = scaled_second_derivative(
        |_, z| {
            let q = one(z, 2);
            q[0].bce_with_logits_loss_with(
                &q[1],
                Reduction::Sum,
                &BceWithLogitsOptions::default().pos_weight(t(vec![1e-8], &[1])),
            )
        },
        &[0.0, 1.0],
        1e20,
    )
    .expect("受理");
    assert!(
        (got - expect).abs() / expect < 1e-3,
        "got={got} expect={expect}"
    );
}

/// pos_weight = 0・x = -21（target 追跡・損失 x 1e10）で ∂dt/∂x が -σ(x) x 1e10（約 -7.58）を保つ。
#[test]
fn bce_pos_weight_zero_keeps_mixed_derivative() {
    let expect = -(1.0 / (1.0 + 21f64.exp())) * 1e10;
    let got = scaled_mixed_derivative(
        |_, z| {
            let q = one(z, 2);
            q[0].bce_with_logits_loss_with(
                &q[1],
                Reduction::Sum,
                &BceWithLogitsOptions::default().pos_weight(t(vec![0.0], &[1])),
            )
        },
        &[-21.0, 0.5],
        1e10,
        1,
        0,
    )
    .expect("受理");
    assert!(
        (got - expect).abs() / expect.abs() < 1e-3,
        "got={got} expect={expect}"
    );
}
