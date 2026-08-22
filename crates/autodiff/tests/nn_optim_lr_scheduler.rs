//! #195（親 #192）: LR スケジューラ最小セット（constant / step）の
//! 系列テスト・入力検証（fail-closed）テスト。

use fandhe_ai_autodiff::nn::optim::{ConstantLr, LrScheduler, StepLr};

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
