//! #195（親 #192）: LR スケジューラ最小セット（constant / step）の
//! 系列テスト・入力検証（fail-closed）テスト。
//!
//! #1745（親 #1611）: CosineAnnealingLr／ExponentialLr／
//! LinearWarmupLr（式ベース 3 種）の参照系列・境界値・入力検証テスト
//! を追加した。参照値は本環境で PyTorch を実行できないため、閉形式の
//! 式（`lr_scheduler.rs` の各 `impl LrScheduler::lr_at` doc 参照）から
//! 手計算した値を用いる。
//!
//! #1747（親 #1611）: OneCycleLr（PyTorch `OneCycleLR` 相当）の参照
//! 系列・境界値・`step` clamp・入力検証テストを追加した（§5）。参照
//! 系列は `OneCycleLr::new`／`lr_at` のアルゴリズム（`lr_scheduler.rs`
//! doc 参照）を python3 で忠実に再現し手計算した値を用いる
//! （max_lr=1.0・total_steps=10・既定値 or 明示値。#1855 の
//! `CosineAnnealingLr` 等と同じ「閉形式手計算」方針）。

use fandhe_ai_autodiff::nn::optim::{
    ConstantLr, CosineAnnealingLr, ExponentialLr, LinearWarmupLr, LrScheduler, OneCycleAnneal,
    OneCycleLr, OneCycleLrConfig, StepLr,
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
    let one_cycle = OneCycleLr::new(OneCycleLrConfig::new(0.1, 10)).unwrap();
    for step in [0usize, 1, 3, 4, 10] {
        assert_eq!(cosine.lr_at(step), cosine.lr_at(step), "cosine step={step}");
        assert_eq!(
            exponential.lr_at(step),
            exponential.lr_at(step),
            "exponential step={step}"
        );
        assert_eq!(warmup.lr_at(step), warmup.lr_at(step), "warmup step={step}");
        assert_eq!(
            one_cycle.lr_at(step),
            one_cycle.lr_at(step),
            "one_cycle step={step}"
        );
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
        Box::new(OneCycleLr::new(OneCycleLrConfig::new(0.1, 10)).unwrap()),
    ];
    for sched in &schedulers {
        let lr = sched.lr_at(0);
        assert!(lr.is_finite(), "lr_at(0) が非有限: {lr}");
    }
}
// ---------------------------------------------------------------------
// §5: OneCycleLr（イシュー #1747）
// ---------------------------------------------------------------------

#[test]
fn one_cycle_lr_matches_pytorch_two_phase_cos_reference_sequence() {
    // max_lr=1.0, total_steps=10, 既定値（pct_start=0.3・Cos・
    // div_factor=25.0・final_div_factor=1e4・three_phase=false）の
    // `OneCycleLr::new`／`lr_at` アルゴリズムを python3 で忠実に再現し
    // 手計算した参照系列（モジュール冒頭 doc 参照）。
    let sched = OneCycleLr::new(OneCycleLrConfig::new(1.0, 10)).unwrap();
    let expected = [
        0.04f32,
        0.52,
        1.0,
        0.950_484_63,
        0.811_745_64,
        0.611_262,
        0.388_741_97,
        0.188_258_35,
        0.049_519_368,
        0.000_004,
    ];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "cos step={step} got={got} want={want}"
        );
    }
}

#[test]
fn one_cycle_lr_matches_pytorch_linear_reference_sequence() {
    let config = OneCycleLrConfig {
        anneal_strategy: OneCycleAnneal::Linear,
        ..OneCycleLrConfig::new(1.0, 10)
    };
    let sched = OneCycleLr::new(config).unwrap();
    let expected = [
        0.04f32,
        0.52,
        1.0,
        0.857_143_4,
        0.714_286_86,
        0.571_430_27,
        0.428_573_73,
        0.285_717_13,
        0.142_860_58,
        0.000_004,
    ];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "linear step={step} got={got} want={want}"
        );
    }
}

#[test]
fn one_cycle_lr_matches_pytorch_three_phase_reference_sequence() {
    let config = OneCycleLrConfig {
        three_phase: true,
        ..OneCycleLrConfig::new(1.0, 10)
    };
    let sched = OneCycleLr::new(config).unwrap();
    let expected = [
        0.04f32,
        0.52,
        1.0,
        0.52,
        0.04,
        0.036_180_72,
        0.026_181_722,
        0.013_822_278,
        0.003_823_278,
        0.000_004,
    ];
    for (step, &want) in expected.iter().enumerate() {
        let got = sched.lr_at(step);
        assert!(
            (got - want).abs() < 1e-6,
            "three_phase step={step} got={got} want={want}"
        );
    }
}

#[test]
fn one_cycle_lr_hits_initial_peak_and_min() {
    // max_lr=1.0, total_steps=10, 既定値: initial_lr=max_lr/25=0.04・
    // 位相境界 step=2 で max_lr・末尾 step=9 で min_lr
    // (=initial_lr/1e4=4e-6) に達する（§2.2 系列参照）。
    let sched = OneCycleLr::new(OneCycleLrConfig::new(1.0, 10)).unwrap();
    assert!(
        (sched.lr_at(0) - 0.04).abs() < 1e-6,
        "lr_at(0) は initial_lr のはず: {}",
        sched.lr_at(0)
    );
    assert!(
        (sched.lr_at(2) - 1.0).abs() < 1e-6,
        "lr_at(2) は max_lr のはず（位相境界）: {}",
        sched.lr_at(2)
    );
    assert!(
        (sched.lr_at(9) - 0.000_004).abs() < 1e-6,
        "lr_at(9) は min_lr のはず: {}",
        sched.lr_at(9)
    );
}

#[test]
fn one_cycle_lr_is_increasing_then_non_increasing() {
    // 上昇フェーズ（step 0..=2）は単調増加、以降（step 2..=9）は
    // 単調非増加であることを固定する。
    let sched = OneCycleLr::new(OneCycleLrConfig::new(1.0, 10)).unwrap();
    let mut prev = sched.lr_at(0);
    for step in 1..=2usize {
        let cur = sched.lr_at(step);
        assert!(
            cur >= prev - 1e-6,
            "step={step} cur={cur} prev={prev} が上昇フェーズで単調増加でない"
        );
        prev = cur;
    }
    for step in 3..=9usize {
        let cur = sched.lr_at(step);
        assert!(
            cur <= prev + 1e-6,
            "step={step} cur={cur} prev={prev} が下降フェーズで単調非増加でない"
        );
        prev = cur;
    }
}

#[test]
fn one_cycle_lr_clamps_step_beyond_total_steps() {
    // PyTorch は `step > total_steps` で ValueError を送出するが、
    // `lr_at` は Result を返せない契約のため `total_steps - 1` へ
    // clamp し続ける（`OneCycleLr::new` doc「`step >= total_steps` の
    // 扱い」節参照）。panic せず・非有限にならないことも併せて固定。
    let sched = OneCycleLr::new(OneCycleLrConfig::new(1.0, 10)).unwrap();
    let at_last = sched.lr_at(9);
    let at_total = sched.lr_at(10);
    let at_far = sched.lr_at(10 * 10);
    assert!(
        (at_total - at_last).abs() < 1e-6,
        "lr_at(total_steps) は lr_at(total_steps-1) と一致するはず: \
         at_total={at_total} at_last={at_last}"
    );
    assert!(
        (at_far - at_last).abs() < 1e-6,
        "lr_at(10*total_steps) も lr_at(total_steps-1) と一致するはず: \
         at_far={at_far} at_last={at_last}"
    );
    assert!(at_total.is_finite(), "at_total が非有限: {at_total}");
    assert!(at_far.is_finite(), "at_far が非有限: {at_far}");
}

#[test]
fn one_cycle_lr_config_new_uses_pytorch_defaults() {
    let config = OneCycleLrConfig::new(0.1, 100);
    assert_eq!(config.max_lr, 0.1);
    assert_eq!(config.total_steps, 100);
    assert_eq!(config.pct_start, 0.3);
    assert_eq!(config.anneal_strategy, OneCycleAnneal::Cos);
    assert_eq!(config.div_factor, 25.0);
    assert_eq!(config.final_div_factor, 1e4);
    assert!(!config.three_phase);
}

#[test]
fn one_cycle_lr_rejects_invalid_arguments() {
    let base = OneCycleLrConfig::new(1.0, 10);

    let mut cfg = base;
    cfg.max_lr = 0.0;
    assert!(OneCycleLr::new(cfg).is_err(), "max_lr=0");
    let mut cfg = base;
    cfg.max_lr = -1.0;
    assert!(OneCycleLr::new(cfg).is_err(), "max_lr<0");
    let mut cfg = base;
    cfg.max_lr = f32::NAN;
    assert!(OneCycleLr::new(cfg).is_err(), "max_lr=NaN");
    let mut cfg = base;
    cfg.max_lr = f32::INFINITY;
    assert!(OneCycleLr::new(cfg).is_err(), "max_lr=Inf");

    let mut cfg = base;
    cfg.total_steps = 0;
    assert!(OneCycleLr::new(cfg).is_err(), "total_steps=0");

    let mut cfg = base;
    cfg.pct_start = 0.0;
    assert!(OneCycleLr::new(cfg).is_err(), "pct_start=0");
    let mut cfg = base;
    cfg.pct_start = 1.0;
    assert!(OneCycleLr::new(cfg).is_err(), "pct_start=1");
    let mut cfg = base;
    cfg.pct_start = -0.1;
    assert!(OneCycleLr::new(cfg).is_err(), "pct_start<0");
    let mut cfg = base;
    cfg.pct_start = 1.1;
    assert!(OneCycleLr::new(cfg).is_err(), "pct_start>1");
    let mut cfg = base;
    cfg.pct_start = f32::NAN;
    assert!(OneCycleLr::new(cfg).is_err(), "pct_start=NaN");

    let mut cfg = base;
    cfg.div_factor = 0.0;
    assert!(OneCycleLr::new(cfg).is_err(), "div_factor=0");
    let mut cfg = base;
    cfg.div_factor = -1.0;
    assert!(OneCycleLr::new(cfg).is_err(), "div_factor<0");
    let mut cfg = base;
    cfg.div_factor = f32::NAN;
    assert!(OneCycleLr::new(cfg).is_err(), "div_factor=NaN");
    // div_factor < 1.0 は PyTorch 自身が拒否しないため、ここでも拒否
    // しない（`new` doc 参照）。
    let mut cfg = base;
    cfg.div_factor = 0.5;
    assert!(OneCycleLr::new(cfg).is_ok(), "div_factor<1.0 は許容");

    let mut cfg = base;
    cfg.final_div_factor = 0.0;
    assert!(OneCycleLr::new(cfg).is_err(), "final_div_factor=0");
    let mut cfg = base;
    cfg.final_div_factor = -1.0;
    assert!(OneCycleLr::new(cfg).is_err(), "final_div_factor<0");
    let mut cfg = base;
    cfg.final_div_factor = f32::NAN;
    assert!(OneCycleLr::new(cfg).is_err(), "final_div_factor=NaN");

    // pct_start が中間値であれば受理する（境界値のみ拒否）。
    let mut cfg = base;
    cfg.pct_start = 0.5;
    assert!(OneCycleLr::new(cfg).is_ok(), "pct_start=0.5 は許容");

    // 退化フェーズ境界: total_steps=3・pct_start=0.3 では
    // `pct*total-1 = -0.1 <= 0` となり最初のフェーズ境界が単調増加の
    // 前提を満たさない。
    let mut cfg = base;
    cfg.total_steps = 3;
    assert!(
        OneCycleLr::new(cfg).is_err(),
        "total_steps=3 は最初のフェーズ境界が退化する"
    );

    // 3 フェーズ形式の退化境界: pct_start=0.6・total_steps=10 では
    // フェーズ 2 の end_step(=2*0.6*10-2=10.0) がフェーズ 3 の
    // end_step(=total_steps-1=9.0) を超える。
    let mut cfg = base;
    cfg.pct_start = 0.6;
    cfg.three_phase = true;
    assert!(
        OneCycleLr::new(cfg).is_err(),
        "three_phase で pct_start=0.6・total_steps=10 は境界が単調増加でない"
    );
}

#[test]
fn one_cycle_lr_rejects_configs_whose_derived_lr_is_not_f32_representable() {
    // codex-review 指摘: `initial_lr`／`min_lr` は `f64` では有限でも
    // `f32` へ変換した時点で overflow（`infinity`）・underflow
    // （`0.0`）しうる。構築時点で検出し fail-closed に拒否する
    // （`OneCycleLr::new` doc「導出値の `f32` 表現可能性」検証節）。

    // overflow: max_lr=3e38・div_factor=0.5 → initial_lr=6e38 は f64
    // では有限だが f32 では infinity に丸められる。
    let mut cfg = OneCycleLrConfig::new(3e38, 10);
    cfg.div_factor = 0.5;
    assert!(
        OneCycleLr::new(cfg).is_err(),
        "initial_lr が f32 overflow する設定は拒否されるべき"
    );

    // underflow: initial_lr（=1.0/1e23）は f32 として表現可能な範囲
    // だが、div_factor・final_div_factor がともに極端に大きく
    // min_lr（=initial_lr/1e23≈1e-46）が f32 の最小正規化数
    // （約 1.4e-45）をも下回り 0.0 に丸められる設定
    // （`final_div_factor` は f32 フィールドのため `f32::MAX`
    // 〈約 3.4e38〉以内に収める必要がある）。
    let mut cfg = OneCycleLrConfig::new(1.0, 10);
    cfg.div_factor = 1e23;
    cfg.final_div_factor = 1e23;
    assert!(
        OneCycleLr::new(cfg).is_err(),
        "min_lr が f32 underflow して 0.0 になる設定は拒否されるべき"
    );

    // 対照: 通常範囲の設定は引き続き受理される。
    assert!(
        OneCycleLr::new(OneCycleLrConfig::new(1.0, 10)).is_ok(),
        "通常範囲の設定は許容されるべき"
    );
}

#[test]
fn one_cycle_lr_returns_exact_min_lr_at_and_beyond_final_step_despite_cancellation() {
    // codex-review 指摘: 線形補間の終端（`p==1.0`）で
    // `(end_lr-start_lr)*p+start_lr` を経由すると、`start_lr` と
    // `end_lr` の大きさが極端に異なる設定（`final_div_factor` が
    // 非常に大きい）では減算時に `end_lr` が丸め落ち、結果が
    // 厳密な `min_lr` を再現しない場合がある（`0.0` になりうる）。
    // `lr_at` は終端で `end_lr` を直接返すため、この桁落ちが起きず
    // 「最終ステップ以降は `min_lr` を返し続ける」契約
    // （`OneCycleLr::new` doc「`step >= total_steps` の扱い」節）を
    // 厳密に満たすことを固定する。
    let mut cfg = OneCycleLrConfig::new(1.0, 10);
    cfg.final_div_factor = 1e20;
    cfg.anneal_strategy = OneCycleAnneal::Linear;
    let sched = OneCycleLr::new(cfg).unwrap();

    // 期待値: min_lr = (max_lr/div_factor) / final_div_factor
    //        = (1.0/25.0) / 1e20 ≈ 4e-22（f32 で表現可能）。
    // `cfg.final_div_factor` は f32 フィールドのため、内部実装
    // （`new` 内の `final_div_factor as f64`）と同じく、ここでも
    // `1e20_f32 as f64` を経由して丸め誤差込みで一致させる
    // （`1e20_f64` を直接使うと f32 往復丸めの分だけ実際の内部計算
    // 結果と食い違う）。
    let expected_min_lr = ((1.0_f64 / 25.0) / (1e20_f32 as f64)) as f32;
    assert!(
        expected_min_lr.is_finite() && expected_min_lr > 0.0,
        "テスト前提が崩れている: expected_min_lr={expected_min_lr}"
    );

    let at_last = sched.lr_at(9);
    assert_eq!(
        at_last, expected_min_lr,
        "lr_at(total_steps-1) は桁落ちなく min_lr と厳密一致するはず: \
         at_last={at_last} expected={expected_min_lr}"
    );
    assert_ne!(
        at_last, 0.0,
        "桁落ちにより 0.0 が返ってはならない（min_lr={expected_min_lr}）"
    );

    // clamp 経由（`step >= total_steps`）でも同じ値を返し続ける。
    let at_clamped = sched.lr_at(100);
    assert_eq!(
        at_clamped, expected_min_lr,
        "step を total_steps 超過させても min_lr を返し続けるはず"
    );
}

#[test]
fn one_cycle_lr_is_object_safe_and_reachable_via_dyn_lr_scheduler() {
    let sched = OneCycleLr::new(OneCycleLrConfig::new(0.1, 4)).unwrap();
    let boxed: Box<dyn LrScheduler> = Box::new(sched);
    let lr = boxed.lr_at(0);
    assert!(lr.is_finite(), "lr_at(0) が非有限: {lr}");
}
