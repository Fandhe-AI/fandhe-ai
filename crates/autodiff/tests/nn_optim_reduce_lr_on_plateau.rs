//! #1746（親 #1611）: `ReduceLrOnPlateau`（状態保持型 LR スケジューラ）の
//! PyTorch 参照系列テスト・fail-closed 入力検証テスト。

use fandhe_ai_autodiff::nn::optim::{
    LrScheduler, PlateauMode, ReduceLrOnPlateau, ReduceLrOnPlateauConfig, ThresholdMode,
};

fn assert_close(got: f32, want: f32, ctx: &str) {
    assert!((got - want).abs() < 1e-6, "{ctx}: got={got} want={want}");
}

/// 既定設定（Min／Rel／patience=10）で単調減少する指標は改善が続く
/// ため lr は不変のまま。
#[test]
fn default_config_no_decay_while_improving() {
    let mut sched = ReduceLrOnPlateau::new(0.1, ReduceLrOnPlateauConfig::default()).unwrap();
    let mut lr = 0.1;
    for i in 0..20 {
        let metric = 1.0 - (i as f32) * 0.01; // 単調減少
        lr = sched.step(metric).unwrap();
    }
    assert_close(lr, 0.1, "単調改善では減衰しないはず");
}

/// 既定設定（patience=10）で同一指標を 11 回連続で渡すと 12 回目の
/// `step` 戻り値で `lr * 0.1` へ減衰する（`num_bad_epochs > patience`
/// の厳密な大なり判定の固定）。
#[test]
fn default_config_decays_after_patience_plus_one_bad_steps() {
    let mut sched = ReduceLrOnPlateau::new(0.1, ReduceLrOnPlateauConfig::default()).unwrap();

    // 1 回目の step で best=1.0（初期 best=+inf からの改善）となり
    // num_bad_epochs=0。以降同じ値 1.0 を渡し続けると threshold=1e-4
    // (rel) により改善とは判定されず num_bad_epochs が増える。
    let first = sched.step(1.0).unwrap();
    assert_close(first, 0.1, "初回観測では減衰しない");

    // 2 回目〜11 回目（10 回）で num_bad_epochs が 1..=10 まで増える。
    // patience=10 のため num_bad_epochs=10 の時点（11 回目の呼び出し）
    // ではまだ発火しない（10 > 10 は false）。
    let mut lr = first;
    for _ in 0..10 {
        lr = sched.step(1.0).unwrap();
    }
    assert_close(
        lr,
        0.1,
        "num_bad_epochs=10（patience と同値）ではまだ発火しない",
    );

    // 12 回目の呼び出しで num_bad_epochs=11 > patience=10 となり発火。
    let lr = sched.step(1.0).unwrap();
    assert_close(
        lr,
        0.01,
        "11 回目の連続悪化で factor=0.1 の減衰が発火するはず",
    );
}

/// `patience=0` では 1 回の悪化で即座に減衰する。
#[test]
fn patience_zero_decays_on_first_bad_step() {
    let config = ReduceLrOnPlateauConfig {
        patience: 0,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut sched = ReduceLrOnPlateau::new(0.1, config).unwrap();

    let first = sched.step(1.0).unwrap();
    assert_close(first, 0.1, "初回観測（改善）では減衰しない");

    // 2 回目: 改善なし -> num_bad_epochs=1 > patience=0 で即発火。
    let second = sched.step(1.0).unwrap();
    assert_close(second, 0.01, "patience=0 では 1 回の悪化で即発火するはず");
}

/// Max モードでは増加が改善として扱われ、停滞で減衰する。
#[test]
fn max_mode_treats_increase_as_improvement() {
    let config = ReduceLrOnPlateauConfig {
        mode: PlateauMode::Max,
        patience: 0,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut sched = ReduceLrOnPlateau::new(0.1, config).unwrap();

    // 単調増加では減衰しない。
    let mut lr = 0.1;
    for i in 0..5 {
        lr = sched.step(1.0 + i as f32).unwrap();
    }
    assert_close(lr, 0.1, "単調増加（Max モードでの改善）では減衰しないはず");

    // 停滞（同値を維持）すると patience=0 のため即発火。
    let lr = sched.step(5.0).unwrap();
    assert_close(lr, 0.01, "Max モードでの停滞は減衰を発火するはず");
}

/// Abs threshold: `best - threshold` を境に改善判定が変わる境界値。
#[test]
fn abs_threshold_mode_boundary() {
    let config = ReduceLrOnPlateauConfig {
        threshold: 0.5,
        threshold_mode: ThresholdMode::Abs,
        patience: 0,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut sched = ReduceLrOnPlateau::new(0.1, config).unwrap();

    // 初回観測: best = +inf からの改善で best=10.0。
    let lr = sched.step(10.0).unwrap();
    assert_close(lr, 0.1, "初回観測では減衰しない");

    // best - threshold = 9.5。9.6 は改善に届かない（10.0 - 9.6 = 0.4 < 0.5）
    // ため 1 回の悪化で patience=0 により即発火。
    let lr = sched.step(9.6).unwrap();
    assert_close(
        lr,
        0.01,
        "9.6 は best-threshold=9.5 を下回らないため悪化扱いのはず",
    );
}

/// cooldown>0: 減衰直後の cooldown 期間中は `num_bad_epochs` が
/// クリアされ再減衰しない。cooldown 経過後に再び patience 分の悪化で
/// 減衰する。
#[test]
fn cooldown_suppresses_immediate_re_decay() {
    let config = ReduceLrOnPlateauConfig {
        patience: 0,
        cooldown: 2,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut sched = ReduceLrOnPlateau::new(0.1, config).unwrap();

    let lr = sched.step(1.0).unwrap();
    assert_close(lr, 0.1, "初回観測では減衰しない");

    // 2 回目: 悪化(patience=0)で発火 -> lr=0.01, cooldown_counter=2。
    let lr = sched.step(1.0).unwrap();
    assert_close(lr, 0.01, "1 回目の悪化で発火するはず");
    assert_eq!(sched.cooldown_counter(), 2);

    // cooldown 中（3, 4 回目）は悪化を観測しても num_bad_epochs が
    // クリアされ続けるため発火しない。
    let lr3 = sched.step(1.0).unwrap();
    assert_close(lr3, 0.01, "cooldown 中は再発火しないはず");
    assert_eq!(sched.cooldown_counter(), 1);
    let lr4 = sched.step(1.0).unwrap();
    assert_close(lr4, 0.01, "cooldown 中は再発火しないはず（2 回目）");
    assert_eq!(sched.cooldown_counter(), 0);

    // cooldown 経過後（5 回目）に悪化を観測すると patience=0 のため
    // 再び発火する。
    let lr5 = sched.step(1.0).unwrap();
    assert_close(lr5, 0.001, "cooldown 経過後は再び発火するはず");
}

/// `min_lr` フロア: `max(lr*factor, min_lr)` で下げ止まり、その後は
/// 変化しない。
#[test]
fn min_lr_floors_decay() {
    let config = ReduceLrOnPlateauConfig {
        patience: 0,
        min_lr: 0.05,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut sched = ReduceLrOnPlateau::new(0.1, config).unwrap();

    let _ = sched.step(1.0).unwrap(); // 初回改善
    let lr = sched.step(1.0).unwrap(); // 0.1 * 0.1 = 0.01 -> min_lr=0.05 でフロア
    assert_close(lr, 0.05, "min_lr でフロアされるはず");

    // それ以降悪化を続けても min_lr のまま（eps ガードにより lr - new_lr
    // が eps=1e-8 以下なので更新自体が起きない）。
    let lr = sched.step(1.0).unwrap();
    assert_close(lr, 0.05, "min_lr 到達後は変化しないはず");
}

/// eps ガード: `lr - new_lr <= eps` のとき lr は据え置かれるが、
/// カウンタは（発火とみなして）リセットされる。
#[test]
fn eps_guard_holds_lr_but_still_resets_counters() {
    // base_lr を非常に小さくし、factor を掛けても eps 未満しか変化
    // しないようにする。
    let config = ReduceLrOnPlateauConfig {
        patience: 0,
        eps: 1e-3,
        ..ReduceLrOnPlateauConfig::default()
    };
    // base_lr=0.01, factor=0.1 -> new_lr=0.001, delta=0.009 > eps(1e-3)
    // ではガードされない。ガードさせるには delta <= eps にする必要が
    // あるため、base_lr を eps の桁に近づける。
    let config = ReduceLrOnPlateauConfig {
        eps: 0.001,
        ..config
    };
    let base_lr = 0.001111; // factor=0.1 -> new_lr=0.0001111, delta=0.0009999 < eps=0.001
    let mut sched = ReduceLrOnPlateau::new(base_lr, config).unwrap();

    let _ = sched.step(1.0).unwrap(); // 初回改善
    let lr_before = sched.current_lr();
    let lr_after = sched.step(1.0).unwrap(); // 悪化 -> 発火判定だが eps ガードで据え置き
    assert_close(
        lr_after,
        lr_before,
        "eps ガードにより lr は据え置かれるはず",
    );
    assert_eq!(
        sched.num_bad_epochs(),
        0,
        "eps ガードで据え置いてもカウンタはリセットされるはず"
    );

    // カウンタがリセットされているため、次の悪化は「発火判定 1 回目」
    // として扱われ、また同じ理由で据え置かれる（無限ループにはならず
    // 収束済みの状態が続くだけ）。
    let lr_after2 = sched.step(1.0).unwrap();
    assert_close(
        lr_after2,
        lr_before,
        "リセット後も同じ eps ガードが働くはず",
    );
}

/// `lr_at(step)` は `step` 値に依らず `current_lr()` と一致する
/// （状態を進める唯一の入口は `step(metric)`。モジュール冒頭 doc
/// 「stateless 契約に対する唯一の例外」節の固定）。
#[test]
fn lr_at_matches_current_lr_regardless_of_step_arg() {
    let sched = ReduceLrOnPlateau::new(0.1, ReduceLrOnPlateauConfig::default()).unwrap();
    assert_close(sched.lr_at(0), 0.1, "構築直後の lr_at は base_lr のはず");
    assert_close(sched.lr_at(9999), 0.1, "lr_at は step 引数を無視するはず");

    // 状態を進めても lr_at は current_lr と一致し続ける。
    let config = ReduceLrOnPlateauConfig {
        patience: 0,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut sched2 = ReduceLrOnPlateau::new(0.1, config).unwrap();
    let _ = sched2.step(1.0).unwrap();
    let after = sched2.step(1.0).unwrap();
    assert_close(
        sched2.lr_at(42),
        after,
        "lr_at は step 後の current_lr と一致するはず",
    );
}

#[test]
fn rejects_invalid_construction_arguments() {
    let base = ReduceLrOnPlateauConfig::default();

    assert!(ReduceLrOnPlateau::new(0.0, base).is_err(), "base_lr=0");
    assert!(ReduceLrOnPlateau::new(-0.1, base).is_err(), "base_lr<0");
    assert!(
        ReduceLrOnPlateau::new(f32::NAN, base).is_err(),
        "base_lr=NaN"
    );
    assert!(
        ReduceLrOnPlateau::new(f32::INFINITY, base).is_err(),
        "base_lr=Inf"
    );

    for factor in [0.0_f32, 1.0, 1.5, f32::NAN, -0.5] {
        let config = ReduceLrOnPlateauConfig { factor, ..base };
        assert!(
            ReduceLrOnPlateau::new(0.1, config).is_err(),
            "factor={factor} は拒否されるはず"
        );
    }

    for threshold in [-0.1_f32, f32::NAN, f32::INFINITY] {
        let config = ReduceLrOnPlateauConfig { threshold, ..base };
        assert!(
            ReduceLrOnPlateau::new(0.1, config).is_err(),
            "threshold={threshold} は拒否されるはず"
        );
    }

    for min_lr in [-0.1_f32, f32::NAN, f32::INFINITY] {
        let config = ReduceLrOnPlateauConfig { min_lr, ..base };
        assert!(
            ReduceLrOnPlateau::new(0.1, config).is_err(),
            "min_lr={min_lr} は拒否されるはず"
        );
    }

    for eps in [-0.1_f32, f32::NAN, f32::INFINITY] {
        let config = ReduceLrOnPlateauConfig { eps, ..base };
        assert!(
            ReduceLrOnPlateau::new(0.1, config).is_err(),
            "eps={eps} は拒否されるはず"
        );
    }

    let config = ReduceLrOnPlateauConfig {
        min_lr: 0.5,
        ..base
    };
    assert!(
        ReduceLrOnPlateau::new(0.1, config).is_err(),
        "min_lr > base_lr は拒否されるはず"
    );
}

#[test]
fn rejects_non_finite_metric() {
    let mut sched = ReduceLrOnPlateau::new(0.1, ReduceLrOnPlateauConfig::default()).unwrap();
    assert!(sched.step(f32::NAN).is_err());
    assert!(sched.step(f32::INFINITY).is_err());
    assert!(sched.step(f32::NEG_INFINITY).is_err());
}
