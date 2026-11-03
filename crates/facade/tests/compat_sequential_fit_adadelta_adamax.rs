//! `compile()` の [`Optimizer`] enum へ追加した `Adadelta`／`Adamax` variant
//! （イシュー #3839・設計 #3757・承認 #3769）の統合テスト。
//! `compat_sequential_fit_optimizers.rs` と同型: `fandhe_ai` のみを import し、
//! `fit` が手動ループ（`<Opt>::step` 直呼び）と bit 完全一致であること、fit の分割・
//! 再 compile・不正 config・AMP・`LrSchedule` が既存 variant と同じ契約であることを検証する。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない（`fit` は CPU `tape()` 固定）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, FitConfig, Loss, LrSchedule, Optimizer, Sequential,
};
use fandhe_ai::optim::{Adadelta, AdadeltaConfig, Adamax, AdamaxConfig, LrScheduler, StepLr};
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;
const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN]).expect("test fixture: x"),
        Tensor::new(y, &[N, D_OUT]).expect("test fixture: y"),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .expect("test fixture: 層 1")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .expect("test fixture: 層 2")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape []")
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            let xd = x.contiguous().as_slice().expect("contiguous").to_vec();
            let yd = y.contiguous().as_slice().expect("contiguous").to_vec();
            xd.len() == yd.len()
                && xd
                    .iter()
                    .zip(yd.iter())
                    .all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// 非既定 config（分岐網羅のため weight_decay を非 0 にする）。
fn adadelta_config() -> AdadeltaConfig {
    AdadeltaConfig {
        lr: 0.8,
        rho: 0.8,
        eps: 1e-5,
        weight_decay: 0.01,
    }
}

fn adamax_config() -> AdamaxConfig {
    AdamaxConfig {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.99,
        eps: 1e-7,
        weight_decay: 0.01,
    }
}

/// 手動ループ（bind → forward → mse_loss → backward → step → apply_parameters）を
/// 1 バッチ（batch_size = N）で `steps` 回す。`before_step(i)` は各 step 直前のフック。
fn manual_loop(
    steps: usize,
    mut step: impl FnMut(usize, &[(&Tensor<f32>, &Tensor<f32>)]) -> Vec<Tensor<f32>>,
) -> (Vec<f32>, Sequential) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    let mut losses = Vec::with_capacity(steps);
    for i in 0..steps {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);
            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            losses.push(scalar(&loss.to_tensor()));
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> = param_refs
                .iter()
                .copied()
                .zip(grad_refs.iter().copied())
                .collect();
            step(i, &pairs)
        };
        model.apply_parameters(updated).unwrap();
    }
    (losses, model)
}

fn assert_fit_matches_manual(optimizer: Optimizer, manual: (Vec<f32>, Sequential), name: &str) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut fit_model = build_model();
    fit_model.compile(optimizer, Loss::Mse).unwrap();
    let history = fit_model
        .fit(&x, &y, FitConfig::new(manual.0.len(), N))
        .unwrap();
    assert_eq!(history.loss.len(), manual.0.len());
    for (m, f) in manual.0.iter().zip(history.loss.iter()) {
        assert_eq!(m.to_bits(), f.to_bits(), "loss diverged ({name})");
    }
    assert!(
        params_bit_exact(
            &manual.1.trainable_parameters(),
            &fit_model.trainable_parameters()
        ),
        "final parameters diverged between manual loop and fit() ({name})"
    );
}

#[test]
fn fit_matches_manual_loop_bit_exact_adadelta() {
    let mut opt = Adadelta::new(adadelta_config()).unwrap();
    let manual = manual_loop(5, |_, pairs| opt.step(pairs).unwrap());
    assert_fit_matches_manual(Optimizer::Adadelta(adadelta_config()), manual, "Adadelta");
}

#[test]
fn fit_matches_manual_loop_bit_exact_adamax() {
    let mut opt = Adamax::new(adamax_config()).unwrap();
    let manual = manual_loop(5, |_, pairs| opt.step(pairs).unwrap());
    assert_fit_matches_manual(Optimizer::Adamax(adamax_config()), manual, "Adamax");
}

fn assert_fit_twice_equals_once(optimizer: Optimizer, name: &str) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut twice = build_model();
    twice.compile(optimizer, Loss::Mse).unwrap();
    twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    let mut once = build_model();
    once.compile(optimizer, Loss::Mse).unwrap();
    once.fit(&x, &y, FitConfig::new(2, N)).unwrap();
    assert!(
        params_bit_exact(&twice.trainable_parameters(), &once.trainable_parameters()),
        "fit(1)+fit(1) diverged from fit(2) ({name})"
    );
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_adadelta() {
    assert_fit_twice_equals_once(Optimizer::Adadelta(adadelta_config()), "Adadelta");
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_adamax() {
    assert_fit_twice_equals_once(Optimizer::Adamax(adamax_config()), "Adamax");
}

fn assert_recompile_discards_state(optimizer: Optimizer, name: &str) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model.compile(optimizer, Loss::Mse).unwrap();
    model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
    let weights: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();
    model.compile(optimizer, Loss::Mse).unwrap();
    let recompiled = model.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    let mut fresh = build_model();
    fresh.apply_parameters(weights).unwrap();
    fresh.compile(optimizer, Loss::Mse).unwrap();
    let fresh_history = fresh.fit(&x, &y, FitConfig::new(2, N)).unwrap();
    assert_eq!(recompiled.loss.len(), fresh_history.loss.len());
    for (a, b) in recompiled.loss.iter().zip(fresh_history.loss.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "{name}");
    }
    assert!(params_bit_exact(
        &model.trainable_parameters(),
        &fresh.trainable_parameters()
    ));
}

#[test]
fn recompile_discards_optimizer_state_adadelta() {
    assert_recompile_discards_state(Optimizer::Adadelta(adadelta_config()), "Adadelta");
}

#[test]
fn recompile_discards_optimizer_state_adamax() {
    assert_recompile_discards_state(Optimizer::Adamax(adamax_config()), "Adamax");
}

#[test]
fn compile_rejects_invalid_adadelta_config() {
    let mut model = build_model();
    for bad in [
        AdadeltaConfig {
            rho: 1.0,
            ..AdadeltaConfig::default()
        },
        AdadeltaConfig {
            eps: 0.0,
            ..AdadeltaConfig::default()
        },
    ] {
        let err = model
            .compile(Optimizer::Adadelta(bad), Loss::Mse)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(!model.is_compiled());
    }
}

#[test]
fn compile_rejects_invalid_adamax_config() {
    let mut model = build_model();
    let bad = AdamaxConfig {
        beta2: 1.0,
        ..AdamaxConfig::default()
    };
    let err = model
        .compile(Optimizer::Adamax(bad), Loss::Mse)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(!model.is_compiled());
    // compile 済みから不正 config で再 compile しても直前の状態は維持される。
    model
        .compile(Optimizer::Adamax(AdamaxConfig::default()), Loss::Mse)
        .unwrap();
    assert!(model.compile(Optimizer::Adamax(bad), Loss::Mse).is_err());
    assert!(model.is_compiled());
}

#[test]
fn compile_with_amp_smoke_adadelta_adamax() {
    let (x, y) = gen_regression_data(SEED_DATA);
    for optimizer in [
        Optimizer::Adadelta(AdadeltaConfig::default()),
        Optimizer::Adamax(AdamaxConfig::default()),
    ] {
        let mut model = build_model();
        model
            .compile_with_amp(optimizer, Loss::Mse, AmpConfig::new(AmpDType::F16))
            .unwrap();
        let history = model.fit(&x, &y, FitConfig::new(3, N)).unwrap();
        for l in &history.loss {
            assert!(l.is_finite(), "AMP loss は有限のはず: {l}");
        }
    }
}

/// `LrSchedule` を受理し、手動の `set_lr` ループと bit 一致する（Adadelta・Adamax）。
#[test]
fn lr_schedule_adadelta_adamax_match_manual_set_lr_loop() {
    const EPOCHS: usize = 3;
    let (x, y) = gen_regression_data(SEED_DATA);

    let sched = StepLr::new(adadelta_config().lr, 1, 0.5).unwrap();
    let mut ada = Adadelta::new(adadelta_config()).unwrap();
    let manual = manual_loop(EPOCHS, |e, pairs| {
        ada.set_lr(sched.lr_at(e)).unwrap();
        ada.step(pairs).unwrap()
    });
    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Adadelta(adadelta_config()), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(adadelta_config().lr, 1, 0.5).unwrap(),
    ))];
    fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();
    assert!(
        params_bit_exact(
            &manual.1.trainable_parameters(),
            &fit_model.trainable_parameters()
        ),
        "Adadelta × LrSchedule diverged from manual set_lr loop"
    );

    let sched = StepLr::new(adamax_config().lr, 1, 0.5).unwrap();
    let mut amx = Adamax::new(adamax_config()).unwrap();
    let manual = manual_loop(EPOCHS, |e, pairs| {
        amx.set_lr(sched.lr_at(e)).unwrap();
        amx.step(pairs).unwrap()
    });
    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Adamax(adamax_config()), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(adamax_config().lr, 1, 0.5).unwrap(),
    ))];
    fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();
    assert!(
        params_bit_exact(
            &manual.1.trainable_parameters(),
            &fit_model.trainable_parameters()
        ),
        "Adamax × LrSchedule diverged from manual set_lr loop"
    );
}
