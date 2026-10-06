//! イシュー #2659（親 #2657）の [`fandhe_ai_autodiff::nn::optim::PolynomialLr`]・
//! [`fandhe_ai_autodiff::nn::optim::ChainedScheduler`] の統合テスト。
//!
//! - **実 PyTorch 2.14.0 の実行値 fixture**
//!   （`tests/fixtures/lr-scheduler-poly-chained-pytorch-reference/
//!   lr_scheduler_poly_chained_reference.json`・生成条件は同ディレクトリの `README.md`）と、
//!   統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`common::req2_close`）および
//!   相対 1e-6（期待値 0 は完全一致。#2176・#2658 の先例。スケジューラ値の検査であり REQ-2 の
//!   tolerance 定数とは無関係）の両方で突合する。tolerance 定数は新設・変更しない。
//! - 突合が相対 1e-6 を外れた場合の原因は「非乗算メンバーの混入」か「入力値の f32 丸め」であり、
//!   判定を緩めずケースと対応メンバー集合を見直すこと。
//! - ホスト計算のみで実機（CUDA／Metal）非依存のため `#[ignore]` 分離は行わない。

mod common;

use std::path::PathBuf;

use common::req2_close;
use fandhe_ai_autodiff::nn::optim::{
    ChainedScheduler, CosineAnnealingLr, ExponentialLr, LinearWarmupLr, LrScheduler, MultiStepLr,
    PolynomialLr, SequentialLr, StepLr,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    poly_cases: Vec<PolyCase>,
    chain_cases: Vec<ChainCase>,
    non_multiplicative_probe: Probe,
}

#[derive(Deserialize)]
struct PolyCase {
    name: String,
    base_lr: f32,
    total_iters: usize,
    power: f32,
    lrs: Vec<f64>,
}

#[derive(Deserialize)]
struct ChainCase {
    name: String,
    base_lr: f32,
    members: Vec<Member>,
    lrs: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Member {
    Step {
        step_size: usize,
        gamma: f32,
    },
    Exp {
        gamma: f32,
    },
    Linear {
        warmup_steps: usize,
        start_factor: f32,
    },
    MultiStep {
        milestones: Vec<usize>,
        gamma: f32,
    },
    Poly {
        total_iters: usize,
        power: f32,
    },
}

#[derive(Deserialize)]
struct Probe {
    base_lr: f32,
    t_max: usize,
    eta_min: f32,
    gamma: f32,
    lrs: Vec<f64>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/lr-scheduler-poly-chained-pytorch-reference/\
         lr_scheduler_poly_chained_reference.json",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

/// 相対誤差 1e-6（期待値 0 は完全一致）。スケジューラ値の検査用で REQ-2 tolerance とは無関係。
fn lr_close_1e6(actual: f32, expected: f64) -> bool {
    let a = actual as f64;
    if expected == 0.0 {
        return a == 0.0;
    }
    ((a - expected) / expected).abs() < 1e-6
}

fn build_member(base_lr: f32, m: &Member) -> Box<dyn LrScheduler> {
    match m {
        Member::Step { step_size, gamma } => {
            Box::new(StepLr::new(base_lr, *step_size, *gamma).unwrap())
        }
        Member::Exp { gamma } => Box::new(ExponentialLr::new(base_lr, *gamma).unwrap()),
        Member::Linear {
            warmup_steps,
            start_factor,
        } => Box::new(LinearWarmupLr::new(base_lr, *warmup_steps, *start_factor).unwrap()),
        Member::MultiStep { milestones, gamma } => {
            Box::new(MultiStepLr::new(base_lr, milestones, *gamma).unwrap())
        }
        Member::Poly { total_iters, power } => {
            Box::new(PolynomialLr::new(base_lr, *total_iters, *power).unwrap())
        }
    }
}

fn assert_series(name: &str, sched: &dyn LrScheduler, expected: &[f64]) {
    assert_eq!(expected.len(), 31, "{name}: 系列長は 31");
    for (step, &e) in expected.iter().enumerate() {
        let a = sched.lr_at(step);
        assert!(req2_close(a as f64, e), "{name} step {step}: {a} vs {e}");
        assert!(
            lr_close_1e6(a, e),
            "{name} step {step}: 相対 1e-6 超過 {a} vs {e}"
        );
    }
}

#[test]
fn fixture_was_generated_by_torch_2_14_0() {
    assert!(load_fixture().torch_version.starts_with("2.14.0"));
}

#[test]
fn polynomial_lr_matches_pytorch_fixture() {
    for c in load_fixture().poly_cases {
        let s = PolynomialLr::new(c.base_lr, c.total_iters, c.power).unwrap();
        assert_series(&c.name, &s, &c.lrs);
    }
}

#[test]
fn chained_scheduler_matches_pytorch_fixture() {
    for c in load_fixture().chain_cases {
        let members = c
            .members
            .iter()
            .map(|m| build_member(c.base_lr, m))
            .collect();
        let s = ChainedScheduler::new(c.base_lr, members).unwrap();
        assert_series(&c.name, &s, &c.lrs);
    }
}

/// 非乗算メンバー（`CosineAnnealingLr`）を含むチェーンは PyTorch と一致しない（意図的な制限）。
/// `T_max` 超過後に大きく乖離することを固定し、「直そう」とされるのを防ぐ。
#[test]
fn chain_with_non_multiplicative_member_diverges_from_pytorch_by_design() {
    let p = load_fixture().non_multiplicative_probe;
    let s = ChainedScheduler::new(
        p.base_lr,
        vec![
            Box::new(CosineAnnealingLr::new(p.base_lr, p.t_max, p.eta_min).unwrap()),
            Box::new(ExponentialLr::new(p.base_lr, p.gamma).unwrap()),
        ],
    )
    .unwrap();
    let max_rel = p
        .lrs
        .iter()
        .enumerate()
        .filter(|(_, e)| **e > 0.0)
        .map(|(i, e)| ((s.lr_at(i) as f64 - e) / e).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_rel > 0.1, "乖離が想定より小さい: {max_rel}");
}

#[test]
fn polynomial_lr_boundaries() {
    let s = PolynomialLr::new(0.5, 10, 2.0).unwrap();
    assert_eq!(s.lr_at(0), 0.5);
    assert_eq!(s.lr_at(10), 0.0);
    assert_eq!(s.lr_at(11), 0.0);
    assert_eq!(s.lr_at(usize::MAX), 0.0);
    let mut prev = f32::INFINITY;
    for step in 0..=15 {
        let v = s.lr_at(step);
        assert!(v <= prev, "単調非増加: step {step}");
        prev = v;
    }
}

#[test]
fn polynomial_lr_power_zero_stays_base_lr_because_zero_powf_zero_is_one() {
    assert_eq!(0f64.powf(0.0), 1.0);
    let s = PolynomialLr::new(0.25, 5, 0.0).unwrap();
    for step in [0, 1, 4, 5, 6, 1000, usize::MAX] {
        assert_eq!(s.lr_at(step), 0.25, "step {step}");
    }
}

#[test]
fn polynomial_lr_is_deterministic() {
    let a = PolynomialLr::new(0.1, 7, 1.5).unwrap();
    let b = PolynomialLr::new(0.1, 7, 1.5).unwrap();
    for step in 0..12 {
        assert_eq!(a.lr_at(step).to_bits(), b.lr_at(step).to_bits());
        assert_eq!(a.lr_at(step).to_bits(), a.lr_at(step).to_bits());
    }
}

#[test]
fn polynomial_lr_rejects_invalid_arguments() {
    for base in [0.0, -0.1, f32::NAN, f32::INFINITY] {
        assert!(PolynomialLr::new(base, 5, 1.0).is_err(), "base_lr {base}");
    }
    assert!(PolynomialLr::new(0.1, 0, 1.0).is_err());
    for power in [-1.0, -0.5, f32::NAN, f32::INFINITY] {
        assert!(PolynomialLr::new(0.1, 5, power).is_err(), "power {power}");
    }
    assert!(PolynomialLr::new(0.1, 5, 0.0).is_ok());
}

#[test]
fn chained_scheduler_rejects_invalid_arguments() {
    assert!(ChainedScheduler::new(0.1, vec![]).is_err());
    for base in [0.0, -0.1, f32::NAN, f32::INFINITY] {
        let m: Vec<Box<dyn LrScheduler>> = vec![Box::new(ExponentialLr::new(0.1, 0.9).unwrap())];
        assert!(ChainedScheduler::new(base, m).is_err(), "base_lr {base}");
    }
}

#[test]
fn chained_single_member_is_bit_identical_to_member() {
    let member = || StepLr::new(0.5, 3, 0.5).unwrap();
    let c = ChainedScheduler::new(0.5, vec![Box::new(member())]).unwrap();
    let m = member();
    for step in 0..40 {
        assert_eq!(
            c.lr_at(step).to_bits(),
            m.lr_at(step).to_bits(),
            "step {step}"
        );
    }
}

#[test]
fn chained_order_swap_stays_within_req2() {
    let mk = |rev: bool| {
        let mut v: Vec<Box<dyn LrScheduler>> = vec![
            Box::new(StepLr::new(0.5, 3, 0.5).unwrap()),
            Box::new(ExponentialLr::new(0.5, 0.875).unwrap()),
        ];
        if rev {
            v.reverse();
        }
        ChainedScheduler::new(0.5, v).unwrap()
    };
    let (a, b) = (mk(false), mk(true));
    for step in 0..40 {
        assert!(
            req2_close(a.lr_at(step) as f64, b.lr_at(step) as f64),
            "step {step}"
        );
    }
}

#[test]
fn chained_with_polynomial_member_stays_zero_after_total_iters() {
    let c = ChainedScheduler::new(
        0.5,
        vec![
            Box::new(PolynomialLr::new(0.5, 5, 1.0).unwrap()),
            Box::new(ExponentialLr::new(0.5, 0.9).unwrap()),
        ],
    )
    .unwrap();
    for step in 5..20 {
        assert_eq!(c.lr_at(step), 0.0, "step {step}");
    }
}

#[test]
fn chained_nests_in_sequential_and_coerces_to_dyn() {
    let chain = ChainedScheduler::new(
        0.5,
        vec![
            Box::new(ExponentialLr::new(0.5, 0.5).unwrap()),
            Box::new(PolynomialLr::new(0.5, 4, 1.0).unwrap()),
        ],
    )
    .unwrap();
    let r: &dyn LrScheduler = &chain;
    assert_eq!(r.lr_at(0), 0.5);
    let seq = SequentialLr::new(
        vec![
            Box::new(ExponentialLr::new(0.5, 0.5).unwrap()),
            Box::new(chain),
        ],
        vec![2],
    )
    .unwrap();
    assert_eq!(seq.lr_at(0), 0.5);
    assert_eq!(seq.lr_at(2), 0.5);
    // 段内 local step 1 → 0.5 * 0.5 * (1 - 1/4)
    assert!(req2_close(seq.lr_at(3) as f64, 0.5 * 0.5 * 0.75));
}
