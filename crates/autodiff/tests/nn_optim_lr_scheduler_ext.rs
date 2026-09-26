//! イシュー #2176（親 #2131「PyTorch／TF 置き換えの API 網羅」）:
//! LR scheduler 5 種（[`MultiStepLr`]・[`CosineAnnealingWarmRestarts`]・
//! [`CyclicLr`]・[`LambdaLr`]・[`SequentialLr`]）の受け入れテスト。
//!
//! 既存 6 種（`nn_optim_lr_scheduler.rs`）と同じ規律に加え、実 PyTorch
//! 実行値 fixture（`tests/fixtures/lr-scheduler-ext-pytorch-reference/
//! lr_scheduler_ext_reference.json`。README 参照）との突合を行う。
//!
//! **契約: CI は `docs/spec`（submodule）を checkout しない**。本
//! ファイルは `tests/fixtures/lr-scheduler-ext-pytorch-reference/`
//! （本クレート配下に複製済み）のみを参照し、`docs/spec` 配下のいかなる
//! ファイルにも依存しない。

use std::fs;
use std::path::PathBuf;

use fandhe_ai_autodiff::nn::optim::{
    CosineAnnealingLr, CosineAnnealingWarmRestarts, CyclicLr, ExponentialLr, LambdaLr,
    LinearWarmupLr, LrScheduler, MultiStepLr, SequentialLr,
};
use serde::Deserialize;

// =====================================================================
// fixture の読み込み
// =====================================================================

#[derive(Deserialize)]
struct FixtureFile {
    #[allow(dead_code)]
    torch_version: String,
    #[allow(dead_code)]
    n_epochs: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    base_lr: f32,
    lr: Vec<f64>,
    milestones: Option<Vec<usize>>,
    gamma: Option<f32>,
    t_0: Option<usize>,
    t_mult: Option<usize>,
    eta_min: Option<f32>,
    max_lr: Option<f32>,
    step_size_up: Option<usize>,
    step_size_down: Option<usize>,
    lambda_kind: Option<String>,
    cosine_t_max: Option<usize>,
    cosine_eta_min: Option<f32>,
    multistep_milestones: Option<Vec<usize>>,
    multistep_gamma: Option<f32>,
}

fn load_fixture() -> FixtureFile {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lr-scheduler-ext-pytorch-reference/lr_scheduler_ext_reference.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("fixture のパースに失敗（JSON 構造が壊れている）: {e}"))
}

/// 受入基準の相対誤差 1e-6 判定（期待値 0 の場合は完全一致を要求する。
/// スケジューラ値の検査であり `.claude/rules/coding-rust.md` の
/// tolerance 定数〈`RELATIVE_TOLERANCE` 等〉とは無関係）。
fn assert_matches_pytorch(name: &str, step: usize, got: f32, want: f64) {
    let got = got as f64;
    if want == 0.0 {
        assert_eq!(
            got, 0.0,
            "{name}: step={step} got={got} want=0（期待値 0 は完全一致を要求）"
        );
        return;
    }
    let rel = (got - want).abs() / want.abs();
    assert!(
        rel <= 1e-6,
        "{name}: step={step} got={got} want={want} rel_err={rel}"
    );
}

fn build_sequential_2stage(base_lr: f32) -> SequentialLr {
    let s1: Box<dyn LrScheduler> = Box::new(LinearWarmupLr::new(base_lr, 3, 0.1).unwrap());
    let s2: Box<dyn LrScheduler> = Box::new(ExponentialLr::new(base_lr, 0.9).unwrap());
    SequentialLr::new(vec![s1, s2], vec![3]).unwrap()
}

fn build_sequential_3stage(case: &Case) -> SequentialLr {
    let s1: Box<dyn LrScheduler> = Box::new(LinearWarmupLr::new(case.base_lr, 3, 0.1).unwrap());
    let s2: Box<dyn LrScheduler> = Box::new(
        CosineAnnealingLr::new(
            case.base_lr,
            case.cosine_t_max.unwrap(),
            case.cosine_eta_min.unwrap(),
        )
        .unwrap(),
    );
    let s3: Box<dyn LrScheduler> = Box::new(
        MultiStepLr::new(
            case.base_lr,
            &case.multistep_milestones.clone().unwrap(),
            case.multistep_gamma.unwrap(),
        )
        .unwrap(),
    );
    SequentialLr::new(vec![s1, s2, s3], case.milestones.clone().unwrap()).unwrap()
}

// =====================================================================
// 1. fixture 突合（kind ごとに 1 テスト）
// =====================================================================

#[test]
fn multi_step_lr_matches_pytorch_reference() {
    let fixture = load_fixture();
    for case in fixture.cases.iter().filter(|c| c.kind == "multi_step") {
        let sched = MultiStepLr::new(
            case.base_lr,
            case.milestones.as_ref().unwrap(),
            case.gamma.unwrap(),
        )
        .unwrap();
        for (step, &want) in case.lr.iter().enumerate() {
            assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
        }
    }
}

#[test]
fn cosine_annealing_warm_restarts_matches_pytorch_reference() {
    let fixture = load_fixture();
    for case in fixture
        .cases
        .iter()
        .filter(|c| c.kind == "cosine_annealing_warm_restarts")
    {
        let sched = CosineAnnealingWarmRestarts::new(
            case.base_lr,
            case.t_0.unwrap(),
            case.t_mult.unwrap(),
            case.eta_min.unwrap(),
        )
        .unwrap();
        for (step, &want) in case.lr.iter().enumerate() {
            assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
        }
    }
}

#[test]
fn cyclic_lr_matches_pytorch_reference() {
    let fixture = load_fixture();
    for case in fixture.cases.iter().filter(|c| c.kind == "cyclic") {
        let sched = CyclicLr::new(
            case.base_lr,
            case.max_lr.unwrap(),
            case.step_size_up.unwrap(),
            case.step_size_down,
        )
        .unwrap();
        for (step, &want) in case.lr.iter().enumerate() {
            assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
        }
    }
}

#[test]
fn lambda_lr_matches_pytorch_reference() {
    let fixture = load_fixture();
    for case in fixture.cases.iter().filter(|c| c.kind == "lambda") {
        let base_lr = case.base_lr;
        match case.lambda_kind.as_deref().unwrap() {
            "geometric" => {
                let sched = LambdaLr::new(base_lr, |e: usize| 0.95f64.powi(e as i32)).unwrap();
                for (step, &want) in case.lr.iter().enumerate() {
                    assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
                }
            }
            "harmonic" => {
                let sched = LambdaLr::new(base_lr, |e: usize| 1.0 / (e as f64 + 1.0)).unwrap();
                for (step, &want) in case.lr.iter().enumerate() {
                    assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
                }
            }
            other => panic!("未知の lambda_kind: {other}"),
        }
    }
}

#[test]
fn sequential_lr_matches_pytorch_reference() {
    let fixture = load_fixture();
    for case in fixture
        .cases
        .iter()
        .filter(|c| c.kind == "sequential_2stage")
    {
        let sched = build_sequential_2stage(case.base_lr);
        for (step, &want) in case.lr.iter().enumerate() {
            assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
        }
    }
    for case in fixture
        .cases
        .iter()
        .filter(|c| c.kind == "sequential_3stage")
    {
        let sched = build_sequential_3stage(case);
        for (step, &want) in case.lr.iter().enumerate() {
            assert_matches_pytorch(&case.name, step, sched.lr_at(step), want);
        }
    }
}

// =====================================================================
// 2. 閉形式の bit 一致（2 冪など表現が正確なケース）
// =====================================================================

#[test]
fn multi_step_lr_bit_exact_power_of_two_gamma() {
    let sched = MultiStepLr::new(1.0, &[2, 4], 0.5).unwrap();
    assert_eq!(sched.lr_at(0).to_bits(), 1.0f32.to_bits());
    assert_eq!(sched.lr_at(1).to_bits(), 1.0f32.to_bits());
    assert_eq!(sched.lr_at(2).to_bits(), 0.5f32.to_bits());
    assert_eq!(sched.lr_at(3).to_bits(), 0.5f32.to_bits());
    assert_eq!(sched.lr_at(4).to_bits(), 0.25f32.to_bits());
    assert_eq!(sched.lr_at(100).to_bits(), 0.25f32.to_bits());
}

#[test]
fn multi_step_lr_empty_milestones_is_constant() {
    let sched = MultiStepLr::new(0.3, &[], 0.5).unwrap();
    for step in [0usize, 1, 100, 1_000_000] {
        assert_eq!(sched.lr_at(step).to_bits(), 0.3f32.to_bits());
    }
}

#[test]
fn cyclic_lr_boundary_values_bit_reasonable() {
    // up=2, down=2（対称）: pos=0 -> base, pos=2 -> max, pos=4(=0) -> base。
    let sched = CyclicLr::new(0.1, 1.0, 2, Some(2)).unwrap();
    assert_eq!(sched.lr_at(0).to_bits(), 0.1f32.to_bits());
    assert_eq!(sched.lr_at(2).to_bits(), 1.0f32.to_bits());
    assert_eq!(sched.lr_at(4).to_bits(), 0.1f32.to_bits());
}

// =====================================================================
// 3. fail-closed 入力検証
// =====================================================================

#[test]
fn multi_step_lr_rejects_invalid_arguments() {
    assert!(MultiStepLr::new(0.0, &[1], 0.5).is_err(), "base_lr=0");
    assert!(MultiStepLr::new(-0.1, &[1], 0.5).is_err(), "base_lr<0");
    assert!(
        MultiStepLr::new(f32::NAN, &[1], 0.5).is_err(),
        "base_lr=NaN"
    );
    assert!(MultiStepLr::new(0.1, &[1], 0.0).is_err(), "gamma=0");
    assert!(MultiStepLr::new(0.1, &[1], -0.5).is_err(), "gamma<0");
    assert!(MultiStepLr::new(0.1, &[1], f32::NAN).is_err(), "gamma=NaN");
    assert!(
        MultiStepLr::new(0.1, &[1], f32::INFINITY).is_err(),
        "gamma=Inf"
    );
    // gamma > 1.0 は StepLr／ExponentialLr と同じく拒否しない。
    assert!(MultiStepLr::new(0.1, &[1], 2.0).is_ok());
}

#[test]
fn cosine_annealing_warm_restarts_rejects_invalid_arguments() {
    assert!(
        CosineAnnealingWarmRestarts::new(0.0, 1, 1, 0.0).is_err(),
        "base_lr=0"
    );
    assert!(
        CosineAnnealingWarmRestarts::new(0.1, 0, 1, 0.0).is_err(),
        "t_0=0"
    );
    assert!(
        CosineAnnealingWarmRestarts::new(0.1, 1, 0, 0.0).is_err(),
        "t_mult=0"
    );
    assert!(
        CosineAnnealingWarmRestarts::new(0.1, 1, 1, -0.1).is_err(),
        "eta_min<0"
    );
    assert!(
        CosineAnnealingWarmRestarts::new(0.1, 1, 1, 0.2).is_err(),
        "eta_min>base_lr"
    );
    assert!(
        CosineAnnealingWarmRestarts::new(0.1, 1, 1, f32::NAN).is_err(),
        "eta_min=NaN"
    );
    // eta_min == base_lr は境界値として受理する（CosineAnnealingLr と
    // 同じ規則）。
    assert!(CosineAnnealingWarmRestarts::new(0.1, 1, 1, 0.1).is_ok());
}

#[test]
fn cyclic_lr_rejects_invalid_arguments() {
    assert!(CyclicLr::new(0.0, 0.1, 1, None).is_err(), "base_lr=0");
    assert!(CyclicLr::new(0.1, 0.05, 1, None).is_err(), "max_lr<base_lr");
    assert!(CyclicLr::new(0.1, f32::NAN, 1, None).is_err(), "max_lr=NaN");
    assert!(CyclicLr::new(0.1, 0.2, 0, None).is_err(), "step_size_up=0");
    assert!(
        CyclicLr::new(0.1, 0.2, 1, Some(0)).is_err(),
        "step_size_down=Some(0)"
    );
    // max_lr == base_lr は境界値として受理する（振幅 0 の退化だが
    // 数学的には well-defined）。
    assert!(CyclicLr::new(0.1, 0.1, 1, None).is_ok());
    // step_size_up + step_size_down が usize で overflow するケース。
    assert!(
        CyclicLr::new(0.1, 0.2, usize::MAX, Some(1)).is_err(),
        "usize overflow"
    );
}

#[test]
fn lambda_lr_rejects_invalid_base_lr() {
    assert!(LambdaLr::new(0.0, |_| 1.0).is_err(), "base_lr=0");
    assert!(LambdaLr::new(-0.1, |_| 1.0).is_err(), "base_lr<0");
    assert!(LambdaLr::new(f32::NAN, |_| 1.0).is_err(), "base_lr=NaN");
}

/// `lr_lambda` が非有限値を返しても `lr_at` は panic せずそのまま
/// 伝播する（`lr_scheduler.rs::LambdaLr` doc「非有限な結果について」
/// 節参照。`Result` を返せない契約のため）。
#[test]
fn lambda_lr_propagates_non_finite_lambda_result() {
    let sched = LambdaLr::new(1.0, |_: usize| f64::NAN).unwrap();
    assert!(sched.lr_at(0).is_nan());
    let sched = LambdaLr::new(1.0, |_: usize| f64::INFINITY).unwrap();
    assert!(sched.lr_at(0).is_infinite());
}

#[test]
fn sequential_lr_rejects_invalid_arguments() {
    let make = |base: f32| -> Box<dyn LrScheduler> {
        Box::new(fandhe_ai_autodiff::nn::optim::ConstantLr::new(base).unwrap())
    };
    assert!(
        SequentialLr::new(vec![], vec![]).is_err(),
        "schedulers が空"
    );
    assert!(
        SequentialLr::new(vec![make(0.1)], vec![1]).is_err(),
        "milestones.len() + 1 != schedulers.len()（多い）"
    );
    assert!(
        SequentialLr::new(vec![make(0.1), make(0.2)], vec![]).is_err(),
        "milestones.len() + 1 != schedulers.len()（少ない）"
    );
    assert!(
        SequentialLr::new(vec![make(0.1), make(0.2)], vec![0]).is_err(),
        "milestones の先頭が 0"
    );
    assert!(
        SequentialLr::new(vec![make(0.1), make(0.2), make(0.3)], vec![3, 3]).is_err(),
        "milestones が狭義単調増加でない（等しい）"
    );
    assert!(
        SequentialLr::new(vec![make(0.1), make(0.2), make(0.3)], vec![5, 3]).is_err(),
        "milestones が狭義単調増加でない（減少）"
    );
    // 単一 scheduler・milestones 空は許容する（実質 1 段のみ）。
    assert!(SequentialLr::new(vec![make(0.1)], vec![]).is_ok());
}

// =====================================================================
// 4. stateless 契約（決定性）
// =====================================================================

#[test]
fn all_five_schedulers_are_stateless_across_out_of_order_calls() {
    let multi = MultiStepLr::new(0.1, &[2, 5], 0.5).unwrap();
    let cawr = CosineAnnealingWarmRestarts::new(0.1, 3, 2, 0.01).unwrap();
    let cyclic = CyclicLr::new(0.01, 0.1, 3, Some(4)).unwrap();
    let lambda = LambdaLr::new(0.1, |e: usize| 1.0 / (e as f64 + 1.0)).unwrap();
    let seq = build_sequential_2stage(0.1);

    let steps = [0usize, 7, 3, 100, 1, 3, 7, 0];
    for &s in &steps {
        // 同じ step を任意の順序で繰り返し呼んでも、各 step に対する
        // 初回呼び出しの結果と bit 同一（内部可変状態を持たない）。
        let a = multi.lr_at(s);
        let b = multi.lr_at(s);
        assert_eq!(a.to_bits(), b.to_bits());

        let a = cawr.lr_at(s);
        let b = cawr.lr_at(s);
        assert_eq!(a.to_bits(), b.to_bits());

        let a = cyclic.lr_at(s);
        let b = cyclic.lr_at(s);
        assert_eq!(a.to_bits(), b.to_bits());

        let a = lambda.lr_at(s);
        let b = lambda.lr_at(s);
        assert_eq!(a.to_bits(), b.to_bits());

        let a = seq.lr_at(s);
        let b = seq.lr_at(s);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// =====================================================================
// 5. CAWR: PyTorch 浮動小数 log 経路の逸脱を固定
// =====================================================================

/// `t_0=1, t_mult=10, step=111` は PyTorch の epoch 引数経路
/// （`int(math.log(...))`）では周期を誤判定しうる境界値（`lr_scheduler.
/// rs::CosineAnnealingWarmRestarts` doc「PyTorch との意図的な相違」
/// 節参照）。本実装は整数演算のため常に正しい周期（`t_mult=10` を
/// 111 回のうち何回掛けたか）を返す。`t_0=1` なので `t_mult^k` 個の
/// step ごとに周期が切り替わる: 周期0=[0,1)・周期1=[1,11)・
/// 周期2=[11,111)・周期3=[111,1111)。`step=111` は周期3 の先頭
/// （`t_cur=0`）のため必ず `base_lr` を返す。
#[test]
fn cosine_annealing_warm_restarts_fixes_pytorch_float_log_boundary() {
    let sched = CosineAnnealingWarmRestarts::new(0.1, 1, 10, 0.0).unwrap();
    assert_eq!(sched.lr_at(111).to_bits(), 0.1f32.to_bits());
    // 周期境界の前後も併せて確認する（周期2 の最終 step=110 は
    // `eta_min` 側に近いはず。厳密な bit 一致は主張しないが `base_lr`
    // より小さいことだけ確認する退化しない健全性チェック）。
    assert!(sched.lr_at(110) < 0.1);
}

// =====================================================================
// 6. 巨大 step・周期境界での非 panic
// =====================================================================

#[test]
fn all_five_schedulers_do_not_panic_on_usize_max() {
    let multi = MultiStepLr::new(0.1, &[2, 5], 0.5).unwrap();
    let cawr_simple = CosineAnnealingWarmRestarts::new(0.1, 4, 1, 0.0).unwrap();
    let cawr_growing = CosineAnnealingWarmRestarts::new(0.1, 4, 2, 0.0).unwrap();
    let cyclic = CyclicLr::new(0.01, 0.1, 3, Some(4)).unwrap();
    let lambda = LambdaLr::new(0.1, |e: usize| 1.0 / (e as f64 + 1.0)).unwrap();
    let seq = build_sequential_2stage(0.1);

    for &s in &[usize::MAX, usize::MAX - 1, 0usize] {
        assert!(multi.lr_at(s).is_finite());
        assert!(cawr_simple.lr_at(s).is_finite());
        assert!(cawr_growing.lr_at(s).is_finite());
        assert!(cyclic.lr_at(s).is_finite());
        assert!(lambda.lr_at(s).is_finite());
        assert!(seq.lr_at(s).is_finite());
    }
}

#[test]
fn cosine_annealing_warm_restarts_period_boundaries_are_smooth() {
    // t_0=2, t_mult=2: 周期境界は 0, 2, 6, 14, 30, ...（各周期長
    // t_i は 2, 4, 8, 16, ...）。
    let sched = CosineAnnealingWarmRestarts::new(1.0, 2, 2, 0.0).unwrap();
    let period_lens = [2u64, 4, 8, 16];
    for (&boundary, &t_i) in [2usize, 6, 14, 30].iter().zip(period_lens.iter()) {
        // 境界ちょうどの step は新しい周期の先頭（t_cur=0）のため
        // base_lr（1.0）に戻る。
        assert_eq!(
            sched.lr_at(boundary).to_bits(),
            1.0f32.to_bits(),
            "boundary={boundary}"
        );
        // 直前の step（前周期の最終 t_cur = t_i-1）は閉形式どおりの値
        // になる（bit 同一は主張しない。cos の libm 差を許容する
        // 誤差 1e-6 判定）。
        let t_cur = t_i - 1;
        let phase = std::f64::consts::PI * (t_cur as f64) / (t_i as f64);
        let want = (1.0 + phase.cos()) / 2.0;
        let got = sched.lr_at(boundary - 1) as f64;
        assert!(
            (got - want).abs() < 1e-6,
            "boundary={boundary} got={got} want={want}"
        );
    }
}

// =====================================================================
// 7. `Box<dyn LrScheduler>` への格納・`SequentialLr` の入れ子
// =====================================================================

#[test]
fn all_five_schedulers_can_be_boxed_as_trait_object() {
    let schedulers: Vec<Box<dyn LrScheduler>> = vec![
        Box::new(MultiStepLr::new(0.1, &[2], 0.5).unwrap()),
        Box::new(CosineAnnealingWarmRestarts::new(0.1, 3, 2, 0.0).unwrap()),
        Box::new(CyclicLr::new(0.01, 0.1, 3, None).unwrap()),
        Box::new(LambdaLr::new(0.1, |e: usize| 1.0 / (e as f64 + 1.0)).unwrap()),
        Box::new(build_sequential_2stage(0.1)),
    ];
    for s in &schedulers {
        assert!(s.lr_at(0).is_finite());
        assert!(s.lr_at(5).is_finite());
    }
}

#[test]
fn sequential_lr_supports_nesting() {
    // 外側の SequentialLr の 1 段として、別の SequentialLr（2 段）を
    // 使えることを確認する（milestones ちょうどの境界も併せて確認）。
    let inner_s1: Box<dyn LrScheduler> =
        Box::new(fandhe_ai_autodiff::nn::optim::ConstantLr::new(0.5).unwrap());
    let inner_s2: Box<dyn LrScheduler> =
        Box::new(fandhe_ai_autodiff::nn::optim::ConstantLr::new(0.1).unwrap());
    let inner: Box<dyn LrScheduler> =
        Box::new(SequentialLr::new(vec![inner_s1, inner_s2], vec![2]).unwrap());
    let outer_s1: Box<dyn LrScheduler> =
        Box::new(fandhe_ai_autodiff::nn::optim::ConstantLr::new(1.0).unwrap());
    let outer = SequentialLr::new(vec![outer_s1, inner], vec![3]).unwrap();

    // step 0..3: 外側 1 段目（1.0）。
    for step in 0..3 {
        assert_eq!(outer.lr_at(step).to_bits(), 1.0f32.to_bits(), "step={step}");
    }
    // step 3,4: 外側 2 段目（inner）の局所 epoch 0,1 -> inner 1 段目
    // （0.5。inner の milestone=2 未満）。
    assert_eq!(outer.lr_at(3).to_bits(), 0.5f32.to_bits());
    assert_eq!(outer.lr_at(4).to_bits(), 0.5f32.to_bits());
    // step 5 以降: inner 局所 epoch 2 以降 -> inner 2 段目（0.1）。
    assert_eq!(outer.lr_at(5).to_bits(), 0.1f32.to_bits());
    assert_eq!(outer.lr_at(100).to_bits(), 0.1f32.to_bits());
}
