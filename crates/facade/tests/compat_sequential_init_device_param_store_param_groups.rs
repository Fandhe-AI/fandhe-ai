//! イシュー #3838（`Sequential::init_device_param_store_with_param_groups`。設計
//! `docs/autodiff-param-groups-decision.md` §19.13）の CPU 受け入れテスト。
//!
//! facade の公開入口が内部入口（#3837 `device_param_store_with_slot_groups`）への薄い委譲で
//! あること（同一の学習結果・検査順・fail-closed）を、`fandhe_ai::tape()`（CPU）で固定する。
//! 内部入口の optimizer ごとの網羅（ホスト `step_with_groups` との bit 一致ほか）は
//! `device_param_store_param_groups_train.rs` が持つ。実機（CUDA/Metal）は
//! `device_param_store_param_groups_backend_parity.rs`（`#[ignore]`。#3862・#3863）。
//!
//! ROCm（capability なし）と `distributed::step_*` の拒否は、facade から模擬バックエンドの
//! `Tape` を作る公開手段がないため #3837 の autodiff 単体テストに委ね、ここでは重複させない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{AdamConfig, AdamWConfig, GradScaler, GradScalerConfig, ParamGroup};
use fandhe_ai::{AutodiffError, BackendError, DeviceParamStore, Gradients, SgdConfig, Tensor};
use fandhe_ai_autodiff::optim::resident_param_groups::device_param_store_with_slot_groups;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const STEPS: usize = 6;

fn bits(ts: &[Tensor<f32>]) -> Vec<Vec<u32>> {
    ts.iter()
        .map(|t| {
            t.contiguous()
                .as_slice()
                .unwrap_or(&[])
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

fn model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap()
}

fn data() -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (
        Tensor::new(x, &[BATCH, D_IN]).unwrap(),
        Tensor::new(y, &[BATCH, D_OUT]).unwrap(),
    )
}

/// スロット: 0=W1, 1=b1, 2=W2, 3=b2。slot 0 は単独、slot 1・2 は同値で隣り合う、slot 3 は group 外。
fn groups() -> Vec<ParamGroup> {
    vec![
        ParamGroup::new(vec![0], 5e-4, 0.1),
        ParamGroup::new(vec![1, 2], 2e-3, 0.0),
    ]
}

fn facade_store(m: &Sequential, gs: &[ParamGroup]) -> Result<DeviceParamStore, BackendError> {
    m.init_device_param_store_with_param_groups(&fandhe_ai::tape(), gs)
}

fn internal_store(m: &Sequential, gs: &[ParamGroup]) -> Result<DeviceParamStore, BackendError> {
    let t = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
        fandhe_ai_backend_cpu::CpuBackendOps::new(),
    ));
    device_param_store_with_slot_groups(&t, &m.trainable_parameters(), gs)
}

fn fwd_bwd(m: &Sequential, tape: &fandhe_ai::Tape, store: &mut DeviceParamStore) -> Gradients {
    let (x, y) = data();
    let pred = m.forward_resident(tape, &tape.var(&x), store).unwrap();
    let loss = pred.mse_loss(&tape.var(&y)).unwrap();
    tape.backward_device_param_store(&loss, store).unwrap()
}

#[derive(Clone, Copy)]
enum Opt {
    Sgd,
    Adam,
    AdamW,
}

fn step(opt: Opt, tape: &fandhe_ai::Tape, store: &mut DeviceParamStore, g: &Gradients) {
    match opt {
        Opt::Sgd => tape
            .step_device_param_store(
                store,
                g,
                &SgdConfig::new(0.05).with_momentum(0.9).with_nesterov(true),
            )
            .unwrap(),
        Opt::Adam => tape
            .step_device_param_store_adam(store, g, &AdamConfig::default())
            .unwrap(),
        Opt::AdamW => tape
            .step_device_param_store_adamw(store, g, &AdamWConfig::default())
            .unwrap(),
    }
}

/// 2 つの store を同じ手順で `STEPS` 回学習し、毎 step の全パラメータが bit 一致すること。
fn assert_same_training(mut a: DeviceParamStore, mut b: DeviceParamStore, opt: Opt, ctx: &str) {
    let m = model();
    for s in 0..STEPS {
        let ta = fandhe_ai::tape();
        let ga = fwd_bwd(&m, &ta, &mut a);
        step(opt, &ta, &mut a, &ga);
        let tb = fandhe_ai::tape();
        let gb = fwd_bwd(&m, &tb, &mut b);
        step(opt, &tb, &mut b, &gb);
        assert_eq!(
            bits(&ta.sync_device_param_store_to_host(&a).unwrap()),
            bits(&tb.sync_device_param_store_to_host(&b).unwrap()),
            "{ctx}: step {s}"
        );
    }
}

#[test]
fn empty_groups_equal_init_device_param_store_bits() {
    let m = model();
    for (opt, name) in [(Opt::Sgd, "sgd"), (Opt::Adam, "adam")] {
        let plain = m.init_device_param_store(&fandhe_ai::tape()).unwrap();
        let grouped = facade_store(&m, &[]).unwrap();
        assert_same_training(plain, grouped, opt, name);
    }
}

#[test]
fn facade_entry_equals_internal_entry_bits() {
    let m = model();
    for (opt, name) in [
        (Opt::Sgd, "sgd"),
        (Opt::Adam, "adam"),
        (Opt::AdamW, "adamw"),
    ] {
        let via_facade = facade_store(&m, &groups()).unwrap();
        let via_internal = internal_store(&m, &groups()).unwrap();
        assert_same_training(via_facade, via_internal, opt, name);
    }
}

/// groups ありの store は groups なしの store と学習結果が変わる（すり替えの検出）。
#[test]
fn groups_actually_change_the_training_result() {
    let m = model();
    let grouped = facade_store(&m, &groups()).unwrap();
    let plain = m.init_device_param_store(&fandhe_ai::tape()).unwrap();
    let (mut a, mut b) = (grouped, plain);
    let (ta, tb) = (fandhe_ai::tape(), fandhe_ai::tape());
    let ga = fwd_bwd(&m, &ta, &mut a);
    step(Opt::Adam, &ta, &mut a, &ga);
    let gb = fwd_bwd(&m, &tb, &mut b);
    step(Opt::Adam, &tb, &mut b, &gb);
    assert_ne!(
        bits(&ta.sync_device_param_store_to_host(&a).unwrap()),
        bits(&tb.sync_device_param_store_to_host(&b).unwrap())
    );
}

#[test]
fn zero_lr_group_keeps_slots_unchanged_and_others_move() {
    let m = model();
    let gs = vec![ParamGroup::new(vec![0, 1], 0.0, 0.0)];
    let mut store = facade_store(&m, &gs).unwrap();
    let init = fandhe_ai::tape()
        .sync_device_param_store_to_host(&store)
        .unwrap();
    for _ in 0..3 {
        let tape = fandhe_ai::tape();
        let g = fwd_bwd(&m, &tape, &mut store);
        step(Opt::Adam, &tape, &mut store, &g);
    }
    let after = fandhe_ai::tape()
        .sync_device_param_store_to_host(&store)
        .unwrap();
    assert_eq!(bits(&after[..2]), bits(&init[..2]), "group 内は bit 不変");
    assert_ne!(bits(&after[2..]), bits(&init[2..]), "group 外は更新される");
}

#[test]
fn invalid_groups_are_rejected_with_invalid_argument() {
    let m = model();
    for bad in [
        vec![ParamGroup::new(vec![], 0.1, 0.0)],
        vec![ParamGroup::new(vec![9], 0.1, 0.0)],
        vec![ParamGroup::new(vec![0, 0], 0.1, 0.0)],
        vec![ParamGroup::new(vec![0], f32::NAN, 0.0)],
        vec![ParamGroup::new(vec![0], f32::INFINITY, 0.0)],
        vec![ParamGroup::new(vec![0], -1.0, 0.0)],
        vec![ParamGroup::new(vec![0], 0.1, f32::NAN)],
        vec![ParamGroup::new(vec![0], 0.1, -0.5)],
        vec![
            ParamGroup::new(vec![0], 0.1, 0.0),
            ParamGroup::new(vec![0], 0.2, 0.0),
        ],
    ] {
        let err = facade_store(&m, &bad).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    }
}

/// 検査順: モデル構成（非対応層）の `Unsupported` が group の検証より先に返る。
#[test]
fn unsupported_layer_is_rejected_before_group_validation() {
    let m = Sequential::new()
        .add_linear(4, 4, 1)
        .unwrap()
        .add_layer_norm(4, 1e-5)
        .unwrap();
    let bad = [ParamGroup::new(vec![99], f32::NAN, -1.0)];
    let err = facade_store(&m, &bad).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
}

#[test]
fn amp_steps_are_rejected_for_grouped_store_and_leave_params_unchanged() {
    let m = model();
    let mut store = facade_store(&m, &groups()).unwrap();
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let tape = fandhe_ai::tape();
    let before = tape.sync_device_param_store_to_host(&store).unwrap();
    let grads = fwd_bwd(&m, &tape, &mut store);
    let err = tape
        .step_device_param_store_amp(&mut store, &grads, &SgdConfig::new(0.1), &mut scaler)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert_eq!(
        bits(&tape.sync_device_param_store_to_host(&store).unwrap()),
        bits(&before)
    );
}

/// `train_step_resident` は groups ありでチェーンを使わず Tape 経路へ全体フォールバックし、
/// 手書きループと bit 一致する。
#[test]
fn train_step_resident_with_groups_matches_manual_loop() {
    let m = model();
    let cfg = SgdConfig::new(0.05);
    let mut a = facade_store(&m, &groups()).unwrap();
    let mut b = facade_store(&m, &groups()).unwrap();
    let (x, y) = data();
    let t = fandhe_ai::tape();
    let xa = t.upload_resident(&a, &x).unwrap();
    let ya = t.upload_resident(&a, &y).unwrap();
    for _ in 0..3 {
        let loss = m.train_step_resident(&mut a, &xa, &ya, &cfg).unwrap();
        assert!(loss.read(&mut a).unwrap().is_finite());
        let tb = fandhe_ai::tape();
        let g = fwd_bwd(&m, &tb, &mut b);
        tb.step_device_param_store(&mut b, &g, &cfg).unwrap();
    }
    assert_eq!(
        bits(&t.sync_device_param_store_to_host(&a).unwrap()),
        bits(&t.sync_device_param_store_to_host(&b).unwrap())
    );
}
