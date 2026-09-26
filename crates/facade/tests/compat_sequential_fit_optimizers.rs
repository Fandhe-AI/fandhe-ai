//! `compile()` の [`Optimizer`] enum へ追加した `RmsProp`／`Adagrad`／
//! `Lamb` variant（イシュー #2170・親 #2131）の統合テスト。
//! `compat_sequential_fit.rs` と同型: `fandhe_ai` のみを import し、
//! `fit` が「手動ループ（`bind → forward → mse_loss → backward →
//! trainable_grads → <Opt>::step → apply_parameters`）」と**同一の
//! 演算列**であることを bit 完全一致で検証する。
//!
//! **決定的シード**: `compat_sequential_fit.rs` と同じシード方針
//! （`.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード
//! 設定ユーティリティを使う」）。本ファイルはグローバル RNG
//! （`manual_seed`）を消費するテストを含まないため `Mutex` 直列化は
//! 不要（`compat_sequential_fit.rs` の `shuffle(true)` 系との違い）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fit` は CPU `tape()` 固定。`docs/compat-api-scope.md` §1.2）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, FitConfig, Loss, LrSchedule, Optimizer, Sequential,
};
use fandhe_ai::optim::{Adagrad, AdagradConfig, Lamb, LambConfig, RmsProp, RmsPropConfig, StepLr};
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `compat_sequential_fit.rs::gen_regression_data` と同型の決定的生成。
fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

/// `compat_sequential_fit.rs::build_model` と同一構成
/// （784→256(ReLU)→10 の代わりに軽量な 4→8(ReLU)→2 を使う。CI
/// timeout〈`.claude/rules/ci.md` `test-timeout-minutes: 20`〉考慮）。
fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let xd = x
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        let yd = y
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        if xd.len() != yd.len() {
            return false;
        }
        for (xv, yv) in xd.iter().zip(yd.iter()) {
            if xv.to_bits() != yv.to_bits() {
                return false;
            }
        }
    }
    true
}

// =====================================================================
// 1. fit は手動ループ（<Opt>::step 直呼び）と bit 完全一致する
//    （受入基準 3。非既定 config で分岐網羅する）
// =====================================================================

#[test]
fn fit_matches_manual_loop_bit_exact_rmsprop() {
    const STEPS: usize = 5;
    let (x, y) = gen_regression_data(SEED_DATA);
    // 分岐網羅: centered・momentum・weight_decay を全て非既定にする。
    let config = RmsPropConfig {
        lr: 0.05,
        alpha: 0.9,
        eps: 1e-8,
        weight_decay: 0.01,
        momentum: 0.1,
        centered: true,
    };

    let mut manual_model = build_model();
    let mut manual_opt = RmsProp::new(config).unwrap();
    let mut manual_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            manual_losses.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> = param_refs
                .iter()
                .copied()
                .zip(grad_refs.iter().copied())
                .collect();
            manual_opt.step(&pairs).unwrap()
        };
        manual_model.apply_parameters(updated).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::RmsProp(config), Loss::Mse)
        .unwrap();
    let history = fit_model.fit(&x, &y, FitConfig::new(STEPS, N)).unwrap();

    assert_eq!(history.loss.len(), manual_losses.len());
    for (m, f) in manual_losses.iter().zip(history.loss.iter()) {
        assert_eq!(
            m.to_bits(),
            f.to_bits(),
            "loss diverged (RmsProp): manual={m} fit={f}"
        );
    }
    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "final parameters diverged between manual loop and fit() (RmsProp)"
    );
}

#[test]
fn fit_matches_manual_loop_bit_exact_adagrad() {
    const STEPS: usize = 5;
    let (x, y) = gen_regression_data(SEED_DATA);
    // 分岐網羅: lr_decay・weight_decay を非既定にする。
    let config = AdagradConfig {
        lr: 0.1,
        lr_decay: 0.01,
        weight_decay: 0.01,
        initial_accumulator_value: 0.0,
        eps: 1e-10,
    };

    let mut manual_model = build_model();
    let mut manual_opt = Adagrad::new(config).unwrap();
    let mut manual_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            manual_losses.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> = param_refs
                .iter()
                .copied()
                .zip(grad_refs.iter().copied())
                .collect();
            manual_opt.step(&pairs).unwrap()
        };
        manual_model.apply_parameters(updated).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Adagrad(config), Loss::Mse)
        .unwrap();
    let history = fit_model.fit(&x, &y, FitConfig::new(STEPS, N)).unwrap();

    assert_eq!(history.loss.len(), manual_losses.len());
    for (m, f) in manual_losses.iter().zip(history.loss.iter()) {
        assert_eq!(
            m.to_bits(),
            f.to_bits(),
            "loss diverged (Adagrad): manual={m} fit={f}"
        );
    }
    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "final parameters diverged between manual loop and fit() (Adagrad)"
    );
}

#[test]
fn fit_matches_manual_loop_bit_exact_lamb() {
    const STEPS: usize = 5;
    let (x, y) = gen_regression_data(SEED_DATA);
    // 分岐網羅: weight_decay を非既定にする。
    let config = LambConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.01,
    };

    let mut manual_model = build_model();
    let mut manual_opt = Lamb::new(config).unwrap();
    let mut manual_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            manual_losses.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> = param_refs
                .iter()
                .copied()
                .zip(grad_refs.iter().copied())
                .collect();
            manual_opt.step(&pairs).unwrap()
        };
        manual_model.apply_parameters(updated).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Lamb(config), Loss::Mse)
        .unwrap();
    let history = fit_model.fit(&x, &y, FitConfig::new(STEPS, N)).unwrap();

    assert_eq!(history.loss.len(), manual_losses.len());
    for (m, f) in manual_losses.iter().zip(history.loss.iter()) {
        assert_eq!(
            m.to_bits(),
            f.to_bits(),
            "loss diverged (Lamb): manual={m} fit={f}"
        );
    }
    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "final parameters diverged between manual loop and fit() (Lamb)"
    );
}

// =====================================================================
// 2. fit(1) を 2 回 == fit(2)（step_count 依存項が fit をまたいで継続）
// =====================================================================

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_rmsprop() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let config = RmsPropConfig {
        centered: true,
        momentum: 0.1,
        ..RmsPropConfig::default()
    };

    let mut model_twice = build_model();
    model_twice
        .compile(Optimizer::RmsProp(config), Loss::Mse)
        .unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();

    let mut model_once = build_model();
    model_once
        .compile(Optimizer::RmsProp(config), Loss::Mse)
        .unwrap();
    model_once.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    let params_twice = model_twice.trainable_parameters();
    let params_once = model_once.trainable_parameters();
    assert!(
        params_bit_exact(&params_twice, &params_once),
        "fit(1)+fit(1) diverged from fit(2) (RmsProp)"
    );
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_adagrad() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let config = AdagradConfig {
        lr_decay: 0.01,
        ..AdagradConfig::default()
    };

    let mut model_twice = build_model();
    model_twice
        .compile(Optimizer::Adagrad(config), Loss::Mse)
        .unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();

    let mut model_once = build_model();
    model_once
        .compile(Optimizer::Adagrad(config), Loss::Mse)
        .unwrap();
    model_once.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    let params_twice = model_twice.trainable_parameters();
    let params_once = model_once.trainable_parameters();
    assert!(
        params_bit_exact(&params_twice, &params_once),
        "fit(1)+fit(1) diverged from fit(2) (Adagrad)"
    );
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_lamb() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let config = LambConfig::default();

    let mut model_twice = build_model();
    model_twice
        .compile(Optimizer::Lamb(config), Loss::Mse)
        .unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();

    let mut model_once = build_model();
    model_once
        .compile(Optimizer::Lamb(config), Loss::Mse)
        .unwrap();
    model_once.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    let params_twice = model_twice.trainable_parameters();
    let params_once = model_once.trainable_parameters();
    assert!(
        params_bit_exact(&params_twice, &params_once),
        "fit(1)+fit(1) diverged from fit(2) (Lamb)"
    );
}

// =====================================================================
// 3. 再 compile で optimizer 状態が破棄される
// =====================================================================

#[test]
fn recompile_discards_optimizer_state_rmsprop() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let config = RmsPropConfig::default();

    let mut model = build_model();
    model
        .compile(Optimizer::RmsProp(config), Loss::Mse)
        .unwrap();
    model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
    // fit(3) で重み・optimizer 状態（square_avg 等）の両方が進む。
    // 比較対象（`fresh_model`）は同じ出発重みから始めるため、fit(3)
    // 後の重みスナップショットを取っておく。
    let weights_after_first_fit: Vec<Tensor<f32>> =
        model.trainable_parameters().into_iter().cloned().collect();
    // 同一 optimizer で再 compile すると累積状態（square_avg 等）が
    // 破棄される（重みは compile の対象外のため変化しない）。
    model
        .compile(Optimizer::RmsProp(config), Loss::Mse)
        .unwrap();
    let history_after_recompile = model.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    // `fresh_model` は fit(3) 後と同じ重みから出発し、フレッシュな
    // optimizer 状態で fit(2) する（重みは同一・optimizer 状態のみ
    // 「破棄後」と「初回」を突き合わせる比較設計）。
    let mut fresh_model = build_model();
    fresh_model
        .apply_parameters(weights_after_first_fit)
        .unwrap();
    fresh_model
        .compile(Optimizer::RmsProp(config), Loss::Mse)
        .unwrap();
    let history_fresh = fresh_model.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    assert_eq!(history_after_recompile.loss.len(), history_fresh.loss.len());
    for (a, b) in history_after_recompile
        .loss
        .iter()
        .zip(history_fresh.loss.iter())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let params_recompiled = model.trainable_parameters();
    let params_fresh = fresh_model.trainable_parameters();
    assert!(params_bit_exact(&params_recompiled, &params_fresh));
}

// =====================================================================
// 4. 不正設定は fail-closed（compile 前後で compiled 状態が変わらない）
// =====================================================================

#[test]
fn compile_rejects_invalid_rmsprop_config() {
    let mut model = build_model();
    let bad = RmsPropConfig {
        lr: -1.0,
        ..RmsPropConfig::default()
    };
    let err = model
        .compile(Optimizer::RmsProp(bad), Loss::Mse)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(!model.is_compiled());
}

#[test]
fn compile_rejects_invalid_adagrad_config() {
    let mut model = build_model();
    let bad = AdagradConfig {
        lr_decay: f32::NAN,
        ..AdagradConfig::default()
    };
    let err = model
        .compile(Optimizer::Adagrad(bad), Loss::Mse)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(!model.is_compiled());
}

#[test]
fn compile_rejects_invalid_lamb_config() {
    let mut model = build_model();
    let bad = LambConfig {
        lr: f32::INFINITY,
        ..LambConfig::default()
    };
    let err = model.compile(Optimizer::Lamb(bad), Loss::Mse).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(!model.is_compiled());

    // 既に compile 済みの状態から不正設定で再 compile しても、
    // 直前の compiled 状態は変わらないまま維持される
    // （construct-before-assign。`Sequential::compile` doc 参照）。
    model
        .compile(Optimizer::Lamb(LambConfig::default()), Loss::Mse)
        .unwrap();
    assert!(model.is_compiled());
    let err = model.compile(Optimizer::Lamb(bad), Loss::Mse).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

// =====================================================================
// 5. compile_with_amp スモーク（AMP 経路も OptimizerState::step を共有）
// =====================================================================

#[test]
fn compile_with_amp_smoke_rmsprop() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::RmsProp(RmsPropConfig::default()),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap();
    let history = model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
    for l in &history.loss {
        assert!(l.is_finite(), "AMP loss は有限のはず（RmsProp）: {l}");
    }
}

#[test]
fn compile_with_amp_smoke_adagrad() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::Adagrad(AdagradConfig::default()),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap();
    let history = model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
    for l in &history.loss {
        assert!(l.is_finite(), "AMP loss は有限のはず（Adagrad）: {l}");
    }
}

#[test]
fn compile_with_amp_smoke_lamb() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::Lamb(LambConfig::default()),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap();
    let history = model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
    for l in &history.loss {
        assert!(l.is_finite(), "AMP loss は有限のはず（Lamb）: {l}");
    }
}

// =====================================================================
// 6. LrSchedule callback との併用は fail-closed で拒否される
//    （RmsProp／Adagrad／Lamb は set_lr 非対応。イシュー #2170）
// =====================================================================

#[test]
fn fit_with_callbacks_rejects_lr_schedule_for_rmsprop() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::RmsProp(RmsPropConfig::default()), Loss::Mse)
        .unwrap();

    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.05, 1, 0.5).unwrap(),
    ))];
    let params_before = model.trainable_parameters();
    let params_before: Vec<Tensor<f32>> = params_before.into_iter().cloned().collect();

    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(
        model.is_compiled(),
        "拒否後も compile 済み状態は維持されること"
    );
    let params_after = model.trainable_parameters();
    let params_before_refs: Vec<&Tensor<f32>> = params_before.iter().collect();
    assert!(
        params_bit_exact(&params_before_refs, &params_after),
        "拒否された fit でパラメータが変化してはならない（RmsProp × LrSchedule）"
    );
}

#[test]
fn fit_with_callbacks_rejects_lr_schedule_for_adagrad() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Adagrad(AdagradConfig::default()), Loss::Mse)
        .unwrap();

    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.05, 1, 0.5).unwrap(),
    ))];
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

#[test]
fn fit_with_callbacks_rejects_lr_schedule_for_lamb() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Lamb(LambConfig::default()), Loss::Mse)
        .unwrap();

    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.05, 1, 0.5).unwrap(),
    ))];
    let err = model
        .fit_with_callbacks(&x, &y, FitConfig::new(2, N), None, &mut callbacks)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

// =====================================================================
// 7. fit_with_callbacks（callbacks 空）と fit の一致（既存契約の回帰確認）
// =====================================================================

#[test]
fn fit_with_callbacks_empty_matches_fit_lamb() {
    let (x, y) = gen_regression_data(SEED_DATA);

    let mut model_fit = build_model();
    model_fit
        .compile(Optimizer::Lamb(LambConfig::default()), Loss::Mse)
        .unwrap();
    let history_fit = model_fit.fit(&x, &y, FitConfig::new(3, N)).unwrap();

    let mut model_cb = build_model();
    model_cb
        .compile(Optimizer::Lamb(LambConfig::default()), Loss::Mse)
        .unwrap();
    let history_cb = model_cb
        .fit_with_callbacks(&x, &y, FitConfig::new(3, N), None, &mut [])
        .unwrap();

    assert_eq!(history_fit.loss.len(), history_cb.loss.len());
    for (a, b) in history_fit.loss.iter().zip(history_cb.loss.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let params_fit = model_fit.trainable_parameters();
    let params_cb = model_cb.trainable_parameters();
    assert!(params_bit_exact(&params_fit, &params_cb));
}
