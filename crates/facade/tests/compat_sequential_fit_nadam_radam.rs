//! `compile()` の [`Optimizer`] enum へ追加した `NAdam`／`RAdam` variant
//! （イシュー #3840・設計 #3757・承認 #3769）の統合テスト。
//! `compat_sequential_fit_adadelta_adamax.rs` と同型: `fandhe_ai` のみを import し、
//! `fit` が手動ループ（`<Opt>::step` 直呼び）と bit 完全一致であること、fit の分割・
//! 再 compile・不正 config・AMP・`LrSchedule` が既存 variant と同じ契約であることを検証する。
//! `decoupled_weight_decay` の `false`／`true` の両方を通す。
//!
//! RAdam は `rho_t <= 5` の早い step で適応項を使わない分岐に入る
//! （`crates/autodiff/src/nn/optim/radam.rs`: `rho_inf = 2/(1-beta2) - 1`・
//! `rho_t = rho_inf - 2*t*beta2^t/(1-beta2^t)`）。本ファイルは `beta2 = 0.9`
//! （`rho_inf = 19`）を使い、`rho_t` は t=1..5 で約 1.0〜4.6（非適応）・t=6 で約 5.4
//! （適応）となるため、[`RADAM_STEPS`] = 10 の手動ループ／fit は両分岐を通る。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない（`fit` は CPU `tape()` 固定）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, FitConfig, Loss, LrSchedule, Optimizer, Sequential,
};
use fandhe_ai::optim::{LrScheduler, NAdam, NAdamConfig, RAdam, RAdamConfig, StepLr};
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;
const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
/// RAdam の `rho_t <= 5` と `rho_t > 5` の両分岐を通す step 数（モジュール doc 参照）。
const RADAM_STEPS: usize = 10;

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

/// 非既定 config（weight_decay 非 0・momentum_decay は既定 4e-3 と別値）。
fn nadam_config(decoupled: bool) -> NAdamConfig {
    NAdamConfig {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.99,
        eps: 1e-7,
        weight_decay: 0.01,
        momentum_decay: 0.01,
        decoupled_weight_decay: decoupled,
    }
}

fn radam_config(decoupled: bool) -> RAdamConfig {
    RAdamConfig {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.9,
        eps: 1e-7,
        weight_decay: 0.01,
        decoupled_weight_decay: decoupled,
    }
}

/// 手動ループ（bind → forward → mse_loss → backward → step → apply_parameters）を
/// 1 バッチ（batch_size = N）で `steps` 回す。`step(i, pairs)` は各 step の更新。
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
fn fit_matches_manual_loop_bit_exact_nadam() {
    for decoupled in [false, true] {
        let mut opt = NAdam::new(nadam_config(decoupled)).unwrap();
        let manual = manual_loop(RADAM_STEPS, |_, pairs| opt.step(pairs).unwrap());
        assert_fit_matches_manual(
            Optimizer::NAdam(nadam_config(decoupled)),
            manual,
            &format!("NAdam decoupled={decoupled}"),
        );
    }
}

#[test]
fn fit_matches_manual_loop_bit_exact_radam() {
    for decoupled in [false, true] {
        let mut opt = RAdam::new(radam_config(decoupled)).unwrap();
        let manual = manual_loop(RADAM_STEPS, |_, pairs| opt.step(pairs).unwrap());
        assert_fit_matches_manual(
            Optimizer::RAdam(radam_config(decoupled)),
            manual,
            &format!("RAdam decoupled={decoupled}"),
        );
    }
}

fn assert_fit_twice_equals_once(optimizer: Optimizer, name: &str) {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut twice = build_model();
    twice.compile(optimizer, Loss::Mse).unwrap();
    twice.fit(&x, &y, FitConfig::new(5, N)).unwrap();
    twice.fit(&x, &y, FitConfig::new(5, N)).unwrap();
    let mut once = build_model();
    once.compile(optimizer, Loss::Mse).unwrap();
    once.fit(&x, &y, FitConfig::new(10, N)).unwrap();
    assert!(
        params_bit_exact(&twice.trainable_parameters(), &once.trainable_parameters()),
        "fit(5)+fit(5) diverged from fit(10) ({name})"
    );
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_nadam() {
    assert_fit_twice_equals_once(Optimizer::NAdam(nadam_config(true)), "NAdam");
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs_radam() {
    // 5 + 5 step は RAdam の rho_t 分岐の切り替わり（t=6）をまたぐ。
    assert_fit_twice_equals_once(Optimizer::RAdam(radam_config(true)), "RAdam");
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
fn recompile_discards_optimizer_state_nadam() {
    assert_recompile_discards_state(Optimizer::NAdam(nadam_config(false)), "NAdam");
}

#[test]
fn recompile_discards_optimizer_state_radam() {
    assert_recompile_discards_state(Optimizer::RAdam(radam_config(false)), "RAdam");
}

/// 不正 config を拒否し、拒否後も既存の compile 状態が残る（`compile()` は
/// `OptimizerState::new` の成功後にだけ状態を差し替える）。
fn assert_rejects_and_keeps_state(good: Optimizer, bads: Vec<Optimizer>) {
    let mut model = build_model();
    for bad in &bads {
        let err = model.compile(*bad, Loss::Mse).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
        assert!(!model.is_compiled());
    }
    // compile 済みから不正 config で再 compile しても直前の状態は維持される。
    model.compile(good, Loss::Mse).unwrap();
    for bad in &bads {
        assert!(model.compile(*bad, Loss::Mse).is_err());
        assert!(model.is_compiled());
    }
}

#[test]
fn compile_rejects_invalid_nadam_config() {
    let d = NAdamConfig::default();
    let bads = vec![
        Optimizer::NAdam(NAdamConfig { beta1: 1.0, ..d }),
        Optimizer::NAdam(NAdamConfig { beta1: -0.1, ..d }),
        Optimizer::NAdam(NAdamConfig { beta2: 1.0, ..d }),
        Optimizer::NAdam(NAdamConfig { eps: 0.0, ..d }),
        Optimizer::NAdam(NAdamConfig { lr: -1.0, ..d }),
        Optimizer::NAdam(NAdamConfig { lr: f32::NAN, ..d }),
        Optimizer::NAdam(NAdamConfig {
            momentum_decay: -1e-3,
            ..d
        }),
        Optimizer::NAdam(NAdamConfig {
            momentum_decay: f32::INFINITY,
            ..d
        }),
    ];
    assert_rejects_and_keeps_state(Optimizer::NAdam(d), bads);
}

#[test]
fn compile_rejects_invalid_radam_config() {
    let d = RAdamConfig::default();
    let bads = vec![
        Optimizer::RAdam(RAdamConfig { beta1: 1.0, ..d }),
        Optimizer::RAdam(RAdamConfig { beta1: -0.1, ..d }),
        Optimizer::RAdam(RAdamConfig { beta2: 1.0, ..d }),
        Optimizer::RAdam(RAdamConfig { eps: 0.0, ..d }),
        Optimizer::RAdam(RAdamConfig { lr: -1.0, ..d }),
        Optimizer::RAdam(RAdamConfig {
            lr: f32::INFINITY,
            ..d
        }),
    ];
    assert_rejects_and_keeps_state(Optimizer::RAdam(d), bads);
}

#[test]
fn compile_with_amp_smoke_nadam_radam() {
    let (x, y) = gen_regression_data(SEED_DATA);
    for optimizer in [
        Optimizer::NAdam(NAdamConfig::default()),
        Optimizer::RAdam(RAdamConfig::default()),
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

/// `LrSchedule` を受理し、手動の `set_lr` ループと bit 一致する（NAdam・RAdam）。
#[test]
fn lr_schedule_nadam_radam_match_manual_set_lr_loop() {
    const EPOCHS: usize = 8;
    let (x, y) = gen_regression_data(SEED_DATA);

    let sched = StepLr::new(nadam_config(true).lr, 2, 0.5).unwrap();
    let mut nad = NAdam::new(nadam_config(true)).unwrap();
    let manual = manual_loop(EPOCHS, |e, pairs| {
        nad.set_lr(sched.lr_at(e)).unwrap();
        nad.step(pairs).unwrap()
    });
    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::NAdam(nadam_config(true)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(nadam_config(true).lr, 2, 0.5).unwrap(),
    ))];
    fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();
    assert!(
        params_bit_exact(
            &manual.1.trainable_parameters(),
            &fit_model.trainable_parameters()
        ),
        "NAdam × LrSchedule diverged from manual set_lr loop"
    );

    let sched = StepLr::new(radam_config(false).lr, 2, 0.5).unwrap();
    let mut rad = RAdam::new(radam_config(false)).unwrap();
    let manual = manual_loop(EPOCHS, |e, pairs| {
        rad.set_lr(sched.lr_at(e)).unwrap();
        rad.step(pairs).unwrap()
    });
    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::RAdam(radam_config(false)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(radam_config(false).lr, 2, 0.5).unwrap(),
    ))];
    fit_model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap();
    assert!(
        params_bit_exact(
            &manual.1.trainable_parameters(),
            &fit_model.trainable_parameters()
        ),
        "RAdam × LrSchedule diverged from manual set_lr loop"
    );
}
