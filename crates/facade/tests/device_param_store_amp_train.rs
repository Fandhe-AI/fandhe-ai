//! イシュー #2181（AMP の `DeviceParamStore` 常駐更新への結線）の受け入れ
//! 条件検証。
//!
//! `crates/facade/tests/device_param_store_adam_train.rs`（Adam／AdamW の
//! 常駐 step）と同じ `Sequential::forward_resident`／`Tape::
//! backward_device_param_store` ループを使うが、AMP（`Tape::
//! step_device_param_store_amp`／`_adam_amp`／`_adamw_amp`）固有の 2 点を
//! 検証する:
//!
//! 1. **非 skip 経路が既存 optimizer 演算列と bit 完全一致すること**:
//!    `GradScalerConfig { init_scale: 1.0, growth_interval: u64::MAX, .. }`
//!    （scale が学習全体を通じて厳密に `1.0` のまま——`growth_interval`
//!    を極大にして growth を起こさせない）で AMP 経路を回し、同一の
//!    初期値・同一シードで並走させた非 AMP 経路（`step_device_param_store`
//!    等）と per-step で bit 完全一致することを確認する。`Var::mul` に
//!    よる `scale = 1.0` 倍・`unscale_grads` による `/ 1.0` 除算はいずれも
//!    `f32` の丸めを生じない厳密な恒等写像のため、この比較は
//!    `step_amp`／`step_adam_amp`／`step_adamw_amp` のラッパー自体が
//!    既存 `step`／`step_adam`／`step_adamw` の演算列を一切変えずに
//!    委譲していることの直接証拠になる（`crates/autodiff/src/optim/
//!    device_store/amp.rs` モジュール doc「設計方針」参照）。
//! 2. **skip 経路がパラメータ・optimizer 状態を一切進めないこと**:
//!    巨大な target で scale 後の勾配を非有限にし、`step_device_param_
//!    store_amp` が `Ok(true)`（skip）を返すこと・パラメータが不変な
//!    こと・`GradScaler` が backoff することを検証する。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fandhe_ai::tape()` 既定 CPU 経由）。CUDA／Metal 実機での AMP 結線の
//! 申し送りは `docs/perf/logs/device-param-store-amp-2181/README.md`。

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{AdamConfig, AdamWConfig, GradScaler, GradScalerConfig, SgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn assert_bits_eq(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    let a_data = a.as_slice().unwrap_or(&[]);
    let e_data = e.as_slice().unwrap_or(&[]);
    for (i, (av, ev)) in a_data.iter().zip(e_data.iter()).enumerate() {
        assert_eq!(
            av.to_bits(),
            ev.to_bits(),
            "{ctx}: element {i}: actual={av} expected={ev}"
        );
    }
}

/// scale が step をまたいで厳密に `1.0` のまま変化しない設定
/// （`growth_interval` を極大にして `GradScaler::update` の growth 分岐へ
/// 到達させない）。
fn scale_one_config() -> GradScalerConfig {
    GradScalerConfig {
        init_scale: 1.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: u64::MAX,
    }
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap()
}

fn fixed_batch() -> (Tensor<f32>, Tensor<f32>) {
    let x: Vec<f32> = (0..BATCH * D_IN).map(|i| (i as f32 * 0.01).sin()).collect();
    let y: Vec<f32> = (0..BATCH * D_OUT)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

#[test]
fn step_amp_sgd_scale_one_matches_non_amp_per_step_bit_exact() {
    let model = build_model();
    let (x_data, y_data) = fixed_batch();
    let config = SgdConfig::new(0.05).with_momentum(0.9);

    let init_tape_amp = fandhe_ai::tape();
    let mut amp_store = model.init_device_param_store(&init_tape_amp).unwrap();
    drop(init_tape_amp);
    let mut scaler = GradScaler::new(scale_one_config()).unwrap();

    let init_tape_ref = fandhe_ai::tape();
    let mut ref_store = model.init_device_param_store(&init_tape_ref).unwrap();
    drop(init_tape_ref);

    for step in 0..6 {
        let amp_tape = fandhe_ai::tape();
        let x = amp_tape.var(&x_data);
        let y = amp_tape.var(&y_data);
        let pred = model
            .forward_resident(&amp_tape, &x, &mut amp_store)
            .unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let scaled_loss = scaler.scale_loss(&loss).unwrap();
        let grads = amp_tape
            .backward_device_param_store(&scaled_loss, &amp_store)
            .unwrap();
        let skipped = amp_tape
            .step_device_param_store_amp(&mut amp_store, &grads, &config, &mut scaler)
            .unwrap();
        assert!(
            !skipped,
            "step {step}: scale == 1.0 で非有限になるはずがない"
        );
        let amp_synced = amp_tape
            .sync_device_param_store_to_host(&amp_store)
            .unwrap();

        let ref_tape = fandhe_ai::tape();
        let x2 = ref_tape.var(&x_data);
        let y2 = ref_tape.var(&y_data);
        let pred2 = model
            .forward_resident(&ref_tape, &x2, &mut ref_store)
            .unwrap();
        let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
        assert!(scalar(&loss2.to_tensor()).is_finite());
        let grads2 = ref_tape
            .backward_device_param_store(&loss2, &ref_store)
            .unwrap();
        ref_tape
            .step_device_param_store(&mut ref_store, &grads2, &config)
            .unwrap();
        let ref_synced = ref_tape
            .sync_device_param_store_to_host(&ref_store)
            .unwrap();

        for (i, (a, r)) in amp_synced.iter().zip(ref_synced.iter()).enumerate() {
            assert_bits_eq(a, r, &format!("step {step} param slot {i}"));
        }
    }
}

#[test]
fn step_adam_amp_scale_one_matches_non_amp_per_step_bit_exact() {
    let model = build_model();
    let (x_data, y_data) = fixed_batch();
    let config = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };

    let init_tape_amp = fandhe_ai::tape();
    let mut amp_store = model.init_device_param_store(&init_tape_amp).unwrap();
    drop(init_tape_amp);
    let mut scaler = GradScaler::new(scale_one_config()).unwrap();

    let init_tape_ref = fandhe_ai::tape();
    let mut ref_store = model.init_device_param_store(&init_tape_ref).unwrap();
    drop(init_tape_ref);

    for step in 0..6 {
        let amp_tape = fandhe_ai::tape();
        let x = amp_tape.var(&x_data);
        let y = amp_tape.var(&y_data);
        let pred = model
            .forward_resident(&amp_tape, &x, &mut amp_store)
            .unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let scaled_loss = scaler.scale_loss(&loss).unwrap();
        let grads = amp_tape
            .backward_device_param_store(&scaled_loss, &amp_store)
            .unwrap();
        let skipped = amp_tape
            .step_device_param_store_adam_amp(&mut amp_store, &grads, &config, &mut scaler)
            .unwrap();
        assert!(
            !skipped,
            "step {step}: scale == 1.0 で非有限になるはずがない"
        );
        let amp_synced = amp_tape
            .sync_device_param_store_to_host(&amp_store)
            .unwrap();

        let ref_tape = fandhe_ai::tape();
        let x2 = ref_tape.var(&x_data);
        let y2 = ref_tape.var(&y_data);
        let pred2 = model
            .forward_resident(&ref_tape, &x2, &mut ref_store)
            .unwrap();
        let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
        let grads2 = ref_tape
            .backward_device_param_store(&loss2, &ref_store)
            .unwrap();
        ref_tape
            .step_device_param_store_adam(&mut ref_store, &grads2, &config)
            .unwrap();
        let ref_synced = ref_tape
            .sync_device_param_store_to_host(&ref_store)
            .unwrap();

        for (i, (a, r)) in amp_synced.iter().zip(ref_synced.iter()).enumerate() {
            assert_bits_eq(a, r, &format!("step {step} param slot {i}"));
        }
    }
}

#[test]
fn step_adamw_amp_scale_one_matches_non_amp_per_step_bit_exact() {
    let model = build_model();
    let (x_data, y_data) = fixed_batch();
    let config = AdamWConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };

    let init_tape_amp = fandhe_ai::tape();
    let mut amp_store = model.init_device_param_store(&init_tape_amp).unwrap();
    drop(init_tape_amp);
    let mut scaler = GradScaler::new(scale_one_config()).unwrap();

    let init_tape_ref = fandhe_ai::tape();
    let mut ref_store = model.init_device_param_store(&init_tape_ref).unwrap();
    drop(init_tape_ref);

    for step in 0..6 {
        let amp_tape = fandhe_ai::tape();
        let x = amp_tape.var(&x_data);
        let y = amp_tape.var(&y_data);
        let pred = model
            .forward_resident(&amp_tape, &x, &mut amp_store)
            .unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let scaled_loss = scaler.scale_loss(&loss).unwrap();
        let grads = amp_tape
            .backward_device_param_store(&scaled_loss, &amp_store)
            .unwrap();
        let skipped = amp_tape
            .step_device_param_store_adamw_amp(&mut amp_store, &grads, &config, &mut scaler)
            .unwrap();
        assert!(
            !skipped,
            "step {step}: scale == 1.0 で非有限になるはずがない"
        );
        let amp_synced = amp_tape
            .sync_device_param_store_to_host(&amp_store)
            .unwrap();

        let ref_tape = fandhe_ai::tape();
        let x2 = ref_tape.var(&x_data);
        let y2 = ref_tape.var(&y_data);
        let pred2 = model
            .forward_resident(&ref_tape, &x2, &mut ref_store)
            .unwrap();
        let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
        let grads2 = ref_tape
            .backward_device_param_store(&loss2, &ref_store)
            .unwrap();
        ref_tape
            .step_device_param_store_adamw(&mut ref_store, &grads2, &config)
            .unwrap();
        let ref_synced = ref_tape
            .sync_device_param_store_to_host(&ref_store)
            .unwrap();

        for (i, (a, r)) in amp_synced.iter().zip(ref_synced.iter()).enumerate() {
            assert_bits_eq(a, r, &format!("step {step} param slot {i}"));
        }
    }
}

/// AC3: 非有限勾配は skip され、パラメータ・`GradScaler` の scale/
/// growth_tracker が正しく backoff することを検証する。
#[test]
fn step_amp_skip_on_overflow_leaves_params_unchanged_and_backoffs() {
    let model = build_model();
    let (x_data, _y_data) = fixed_batch();
    // scale 後の勾配を非有限にするため target を極端な値にする。
    let huge_y = tensor(vec![1.0e35; BATCH * D_OUT], &[BATCH, D_OUT]);
    let config = SgdConfig::new(0.05);

    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();

    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&huge_y);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let scaled_loss = scaler.scale_loss(&loss).unwrap();
    let grads = tape
        .backward_device_param_store(&scaled_loss, &store)
        .unwrap();

    let before = tape.sync_device_param_store_to_host(&store).unwrap();
    let skipped = tape
        .step_device_param_store_amp(&mut store, &grads, &config, &mut scaler)
        .unwrap();
    assert!(skipped, "巨大な target は scale 後の勾配を非有限にするはず");
    let after = tape.sync_device_param_store_to_host(&store).unwrap();
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_bits_eq(a, b, &format!("skip 後パラメータが不変であるべき slot {i}"));
    }
    assert_eq!(
        scaler.scale(),
        GradScalerConfig::default().init_scale * GradScalerConfig::default().backoff_factor
    );
    assert_eq!(scaler.growth_tracker(), 0);

    // pending 消費の確認: 次の forward_resident が
    // `PendingForwardUnconsumed` にならず成功すること。
    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let next = model.forward_resident(&tape2, &x2, &mut store);
    assert!(
        next.is_ok(),
        "skip は pending を消費し次回 forward を妨げてはならない"
    );
}
