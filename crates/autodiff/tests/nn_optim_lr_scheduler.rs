//! #195（親 #192）: LR スケジューラ最小セット（constant / step）の
//! 系列テスト・入力検証（fail-closed）テスト。
//!
//! #1745（親 #1611）: CosineAnnealingLr／ExponentialLr／
//! LinearWarmupLr（式ベース 3 種）の参照系列・境界値・入力検証テスト
//! を追加した。参照値は本環境で PyTorch を実行できないため、閉形式の
//! 式（`lr_scheduler.rs` の各 `impl LrScheduler::lr_at` doc 参照）から
//! 手計算した値を用いる。

use fandhe_ai_autodiff::nn::optim::{
    ConstantLr, CosineAnnealingLr, ExponentialLr, LinearWarmupLr, LrScheduler, StepLr,
};

#[test]
fn constant_lr_is_stable_across_steps() {
    let sched = ConstantLr::new(0.01).unwrap();
    for step in [0usize, 1, 10, 1000] {
        assert_eq!(sched.lr_at(step), 0.01);
    }
}

#[test]
fn constant_lr_rejects_non_positive_base_lr() {
    assert!(ConstantLr::new(0.0).is_err());
    assert!(ConstantLr::new(-0.1).is_err());
    assert!(ConstantLr::new(f32::NAN).is_err());
    assert!(ConstantLr::new(f32::INFINITY).is_err());
}

#[test]
fn step_lr_matches_pytorch_step_lr_reference_sequence() {
    // PyTorch StepLR(base_lr=0.1, step_size=2, gamma=0.5) の参照系列:
    // step 0,1 -> 0.1 / step 2,3 -> 0.05 / step 4,5 -> 0.025
    // （lr(step) = base_lr * gamma^(step // step_size)）。
    let sched = StepLr::new(0.1, 2, 0.5).unwrap();
    let expected = [0.1f32, 0.1, 0.05, 0.05, 0.025, 0.025];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "step={step} got={got} want={want}"
        );
    }
}

#[test]
fn step_lr_rejects_invalid_arguments() {
    assert!(StepLr::new(0.0, 2, 0.5).is_err(), "base_lr=0");
    assert!(StepLr::new(-0.1, 2, 0.5).is_err(), "base_lr<0");
    assert!(StepLr::new(f32::NAN, 2, 0.5).is_err(), "base_lr=NaN");
    assert!(StepLr::new(0.1, 0, 0.5).is_err(), "step_size=0");
    assert!(StepLr::new(0.1, 2, 0.0).is_err(), "gamma=0");
    assert!(StepLr::new(0.1, 2, -0.5).is_err(), "gamma<0");
    assert!(StepLr::new(0.1, 2, f32::INFINITY).is_err(), "gamma=Inf");
}
#[test]
fn cosine_annealing_lr_matches_closed_form_reference_sequence() {
    // base_lr=0.1, t_max=4, eta_min=0.0 の閉形式手計算値。
    // lr(step) = eta_min + (base_lr - eta_min) * (1 + cos(pi * step / t_max)) / 2
    let sched = CosineAnnealingLr::new(0.1, 4, 0.0).unwrap();
    let expected = [0.1f32, 0.085_355_34, 0.05, 0.014_644_66, 0.0];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "eta_min=0.0 step={step} got={got} want={want}"
        );
    }

    // eta_min=0.01 の閉形式手計算値。
    let sched_eta = CosineAnnealingLr::new(0.1, 4, 0.01).unwrap();
    let expected_eta = [0.1f32, 0.086_819_81, 0.055, 0.023_180_19, 0.01];
    for (step, &want) in expected_eta.iter().enumerate() {
        let got = sched_eta.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "eta_min=0.01 step={step} got={got} want={want}"
        );
    }
}

#[test]
fn cosine_annealing_lr_is_periodic_beyond_t_max_like_pytorch() {
    // PyTorch の閉形式は周期的（warm restart なし）: step==t_max で
    // eta_min に達し、step==2*t_max で base_lr へ戻る。TensorFlow の
    // ように min(step, t_max) で clamp しないことをここで機械固定する。
    let sched = CosineAnnealingLr::new(0.1, 4, 0.0).unwrap();
    assert!((sched.lr_at(4) - 0.0).abs() < 1e-6, "lr_at(t_max)");
    assert!(
        (sched.lr_at(8) - 0.1).abs() < 1e-6,
        "lr_at(2*t_max) は base_lr へ戻る"
    );
}

#[test]
fn cosine_annealing_lr_is_non_increasing_within_first_period() {
    let sched = CosineAnnealingLr::new(0.1, 8, 0.0).unwrap();
    let mut prev = sched.lr_at(0);
    for step in 1..=8usize {
        let cur = sched.lr_at(step);
        assert!(
            cur <= prev + 1e-6,
            "step={step} cur={cur} prev={prev} が単調非増加でない"
        );
        prev = cur;
    }
}

#[test]
fn cosine_annealing_lr_rejects_invalid_arguments() {
    assert!(CosineAnnealingLr::new(0.0, 4, 0.0).is_err(), "base_lr=0");
    assert!(CosineAnnealingLr::new(-0.1, 4, 0.0).is_err(), "base_lr<0");
    assert!(
        CosineAnnealingLr::new(f32::NAN, 4, 0.0).is_err(),
        "base_lr=NaN"
    );
    assert!(
        CosineAnnealingLr::new(f32::INFINITY, 4, 0.0).is_err(),
        "base_lr=Inf"
    );
    assert!(CosineAnnealingLr::new(0.1, 0, 0.0).is_err(), "t_max=0");
    assert!(CosineAnnealingLr::new(0.1, 4, -0.01).is_err(), "eta_min<0");
    assert!(
        CosineAnnealingLr::new(0.1, 4, 0.2).is_err(),
        "eta_min>base_lr"
    );
    assert!(
        CosineAnnealingLr::new(0.1, 4, f32::NAN).is_err(),
        "eta_min=NaN"
    );
    // 境界値は受理する: eta_min==0.0 と eta_min==base_lr。
    assert!(CosineAnnealingLr::new(0.1, 4, 0.0).is_ok(), "eta_min==0.0");
    assert!(
        CosineAnnealingLr::new(0.1, 4, 0.1).is_ok(),
        "eta_min==base_lr"
    );
}

#[test]
fn exponential_lr_matches_pytorch_reference_sequence() {
    // PyTorch ExponentialLR(base_lr=0.1, gamma=0.5) の参照系列:
    // lr(step) = base_lr * gamma^step
    let sched = ExponentialLr::new(0.1, 0.5).unwrap();
    let expected = [0.1f32, 0.05, 0.025, 0.0125];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "step={step} got={got} want={want}"
        );
    }
}

#[test]
fn exponential_lr_with_gamma_one_is_constant() {
    let sched = ExponentialLr::new(0.1, 1.0).unwrap();
    for step in [0usize, 1, 10, 1000] {
        let got = sched.lr_at(step);
        assert!((got - 0.1).abs() < 1e-6, "step={step} got={got}");
    }
}

#[test]
fn exponential_lr_rejects_invalid_arguments() {
    assert!(ExponentialLr::new(0.0, 0.5).is_err(), "base_lr=0");
    assert!(ExponentialLr::new(-0.1, 0.5).is_err(), "base_lr<0");
    assert!(ExponentialLr::new(f32::NAN, 0.5).is_err(), "base_lr=NaN");
    assert!(ExponentialLr::new(0.1, 0.0).is_err(), "gamma=0");
    assert!(ExponentialLr::new(0.1, -0.5).is_err(), "gamma<0");
    assert!(ExponentialLr::new(0.1, f32::INFINITY).is_err(), "gamma=Inf");
    assert!(ExponentialLr::new(0.1, f32::NAN).is_err(), "gamma=NaN");
    // gamma > 1.0 は拒否しない（StepLr と同一規則）。
    assert!(ExponentialLr::new(0.1, 1.5).is_ok(), "gamma>1.0 は許容");
}

#[test]
fn linear_warmup_lr_ramps_then_holds_base_lr() {
    // base_lr=0.1, warmup_steps=4, start_factor=0.25 の手計算値:
    // lr(step) = base_lr * (start_factor + (1-start_factor) * min(step,4)/4)
    let sched = LinearWarmupLr::new(0.1, 4, 0.25).unwrap();
    let expected = [0.025f32, 0.043_75, 0.0625, 0.081_25, 0.1, 0.1, 0.1];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "step={step} got={got} want={want}"
        );
    }
}

#[test]
fn linear_warmup_lr_with_start_factor_one_is_constant() {
    let sched = LinearWarmupLr::new(0.1, 4, 1.0).unwrap();
    for step in [0usize, 1, 4, 100] {
        let got = sched.lr_at(step);
        assert!((got - 0.1).abs() < 1e-6, "step={step} got={got}");
    }
}

#[test]
fn linear_warmup_lr_rejects_invalid_arguments() {
    assert!(LinearWarmupLr::new(0.0, 4, 0.25).is_err(), "base_lr=0");
    assert!(LinearWarmupLr::new(-0.1, 4, 0.25).is_err(), "base_lr<0");
    assert!(
        LinearWarmupLr::new(f32::NAN, 4, 0.25).is_err(),
        "base_lr=NaN"
    );
    assert!(LinearWarmupLr::new(0.1, 0, 0.25).is_err(), "warmup_steps=0");
    assert!(LinearWarmupLr::new(0.1, 4, 0.0).is_err(), "start_factor=0");
    assert!(LinearWarmupLr::new(0.1, 4, -0.1).is_err(), "start_factor<0");
    assert!(
        LinearWarmupLr::new(0.1, 4, 1.1).is_err(),
        "start_factor>1.0"
    );
    assert!(
        LinearWarmupLr::new(0.1, 4, f32::NAN).is_err(),
        "start_factor=NaN"
    );
    // start_factor==1.0 は境界として受理する。
    assert!(
        LinearWarmupLr::new(0.1, 4, 1.0).is_ok(),
        "start_factor==1.0"
    );
}

#[test]
fn schedulers_are_deterministic_for_same_step() {
    // stateless 契約: 同一 step の 2 回呼び出しは常に同じ値を返す。
    let cosine = CosineAnnealingLr::new(0.1, 4, 0.0).unwrap();
    let exponential = ExponentialLr::new(0.1, 0.5).unwrap();
    let warmup = LinearWarmupLr::new(0.1, 4, 0.25).unwrap();
    for step in [0usize, 1, 3, 4, 10] {
        assert_eq!(cosine.lr_at(step), cosine.lr_at(step), "cosine step={step}");
        assert_eq!(
            exponential.lr_at(step),
            exponential.lr_at(step),
            "exponential step={step}"
        );
        assert_eq!(warmup.lr_at(step), warmup.lr_at(step), "warmup step={step}");
    }
}

#[test]
fn schedulers_are_object_safe_via_dyn_lr_scheduler() {
    let schedulers: Vec<Box<dyn LrScheduler>> = vec![
        Box::new(ConstantLr::new(0.1).unwrap()),
        Box::new(StepLr::new(0.1, 2, 0.5).unwrap()),
        Box::new(CosineAnnealingLr::new(0.1, 4, 0.0).unwrap()),
        Box::new(ExponentialLr::new(0.1, 0.5).unwrap()),
        Box::new(LinearWarmupLr::new(0.1, 4, 0.25).unwrap()),
    ];
    for sched in &schedulers {
        let lr = sched.lr_at(0);
        assert!(lr.is_finite(), "lr_at(0) が非有限: {lr}");
    }
}
