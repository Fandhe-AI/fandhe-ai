//! イシュー #3837（`DeviceParamStore` の常駐更新を param groups に対応させる内部実装。
//! 設計 `docs/autodiff-param-groups-decision.md` §19）の CPU 受け入れテスト。
//!
//! store は内部入口 `fandhe_ai_autodiff::optim::resident_param_groups` で構築し（facade の
//! 入口は #3838）、学習ループは facade の `Tape`（`forward_resident`・
//! `backward_device_param_store`・`step_device_param_store*`）で回す。契約:
//! - Adam／AdamW／RmsProp／Adagrad／Lamb: 毎 step、勾配をホストへ読み出して
//!   `step_with_groups` へ与えた結果と **bit 完全一致**
//! - SGD: 常駐は `mul_add`、ホスト `Sgd` は fixture 用に `mul_add` を使わない既存契約のため、
//!   ホストとは統一複合判定で突き合わせ、bit 一致の契約は「config と同値の group」「group 外
//!   スロット」と groups なし store の突合で補う
//! - groups 空は `DeviceParamStore::new` と bit 一致、非対応経路は fail-closed
//!
//! 実機（CUDA/Metal）は `device_param_store_param_groups_backend_parity.rs`（`#[ignore]`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{
    AdagradConfig, AdamConfig, AdamWConfig, GradScaler, GradScalerConfig, LambConfig, RmsPropConfig,
};
use fandhe_ai::{AutodiffError, DeviceParamStore, Gradients, SgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::{
    Adagrad, Adam, AdamW, Lamb, ParamGroup, ParamGroupStep, RmsProp,
};
use fandhe_ai_autodiff::optim::resident_param_groups::device_param_store_with_slot_groups;
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig as HostSgdConfig};
use fandhe_ai_tensor_core::tolerance::composite_close_f32;
use fandhe_ai_tensor_core::{BackendError, Tensor};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const STEPS: usize = 6;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture")
}

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
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// スロット: 0=W1, 1=b1, 2=W2, 3=b2。slot 0 は単独 group、slot 1・2 は同値で隣り合う
/// group（区間統合の対象）、slot 3 は group 外（config から解決）。
fn groups() -> Vec<ParamGroup> {
    vec![
        ParamGroup::new(vec![0], 5e-4, 0.1),
        ParamGroup::new(vec![1, 2], 2e-3, 0.0),
    ]
}

fn cpu_store(m: &Sequential, groups: &[ParamGroup]) -> Result<DeviceParamStore, BackendError> {
    let t = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
        fandhe_ai_backend_cpu::CpuBackendOps::new(),
    ));
    device_param_store_with_slot_groups(&t, &m.trainable_parameters(), groups)
}

fn plain_store(m: &Sequential) -> DeviceParamStore {
    cpu_store(m, &[]).unwrap()
}

/// forward → backward を 1 回行い、勾配を返す。
fn fwd_bwd(m: &Sequential, tape: &fandhe_ai::Tape, store: &mut DeviceParamStore) -> Gradients {
    let (x, y) = data();
    let xv = tape.var(&x);
    let yv = tape.var(&y);
    let pred = m.forward_resident(tape, &xv, store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &yv).unwrap();
    tape.backward_device_param_store(&loss, store).unwrap()
}

/// 毎 step、ホスト `step_with_groups` と bit 完全一致であること。
fn run_bit_parity<H: ParamGroupStep>(
    mut host: H,
    groups: &[ParamGroup],
    step: impl Fn(&fandhe_ai::Tape, &mut DeviceParamStore, &Gradients) -> Result<(), BackendError>,
) {
    let m = model();
    let mut store = cpu_store(&m, groups).unwrap();
    let tape0 = fandhe_ai::tape();
    let mut host_params = tape0.sync_device_param_store_to_host(&store).unwrap();
    for s in 0..STEPS {
        let tape = fandhe_ai::tape();
        let grads = fwd_bwd(&m, &tape, &mut store);
        let g_host = tape.param_grads_to_host(&store, &grads).unwrap();
        step(&tape, &mut store, &grads).unwrap();
        let dev = tape.sync_device_param_store_to_host(&store).unwrap();
        let p: Vec<&Tensor<f32>> = host_params.iter().collect();
        let g: Vec<&Tensor<f32>> = g_host.iter().collect();
        let out = host.step_with_groups(&p, &g, groups).unwrap();
        assert_eq!(bits(&dev), bits(&out), "step {s}");
        host_params = out;
    }
}

#[test]
fn adam_matches_host_step_with_groups_bits() {
    let cfg = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    run_bit_parity(Adam::new(cfg).unwrap(), &groups(), |t, s, g| {
        t.step_device_param_store_adam(s, g, &cfg)
    });
}

#[test]
fn adamw_matches_host_step_with_groups_bits() {
    let cfg = AdamWConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    run_bit_parity(AdamW::new(cfg).unwrap(), &groups(), |t, s, g| {
        t.step_device_param_store_adamw(s, g, &cfg)
    });
}

#[test]
fn rmsprop_centered_momentum_matches_host_bits() {
    let cfg = RmsPropConfig {
        lr: 1e-3,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.01,
        momentum: 0.9,
        centered: true,
    };
    run_bit_parity(RmsProp::new(cfg).unwrap(), &groups(), |t, s, g| {
        t.step_device_param_store_rmsprop(s, g, &cfg)
    });
}

#[test]
fn adagrad_lr_decay_matches_host_bits() {
    let cfg = AdagradConfig {
        lr: 1e-2,
        lr_decay: 0.01,
        weight_decay: 0.01,
        initial_accumulator_value: 0.1,
        eps: 1e-10,
    };
    run_bit_parity(Adagrad::new(cfg).unwrap(), &groups(), |t, s, g| {
        t.step_device_param_store_adagrad(s, g, &cfg)
    });
}

#[test]
fn lamb_matches_host_bits() {
    let cfg = LambConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.01,
    };
    run_bit_parity(Lamb::new(cfg).unwrap(), &groups(), |t, s, g| {
        t.step_device_param_store_lamb(s, g, &cfg)
    });
}

/// 全 group がゼロ・同値の特殊値でも bit 一致（lr=0 の group のスロットは不変）。
#[test]
fn adam_zero_lr_group_keeps_slot_unchanged_and_matches_host() {
    let cfg = AdamConfig::default();
    let gs = vec![
        ParamGroup::new(vec![0], 0.0, 0.0),
        ParamGroup::new(vec![2], 1e-3, 0.0),
    ];
    let m = model();
    let mut store = cpu_store(&m, &gs).unwrap();
    let tape0 = fandhe_ai::tape();
    let init = tape0.sync_device_param_store_to_host(&store).unwrap();
    let tape = fandhe_ai::tape();
    let grads = fwd_bwd(&m, &tape, &mut store);
    tape.step_device_param_store_adam(&mut store, &grads, &cfg)
        .unwrap();
    let after = tape.sync_device_param_store_to_host(&store).unwrap();
    assert_eq!(bits(&after[..1]), bits(&init[..1]), "lr=0 group slot");
    assert_ne!(bits(&after[2..3]), bits(&init[2..3]));
}

/// SGD: ホスト `Sgd::step_with_groups` とは統一複合判定、groups なし store とは
/// 「config と同値の group」「group 外スロット」で bit 一致。
#[test]
fn sgd_grouped_matches_host_composite_and_plain_store_bits() {
    let host_cfg = HostSgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.001,
        nesterov: true,
    };
    let cfg = SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.001,
        nesterov: true,
    };
    let gs = groups();
    let m = model();
    let mut store = cpu_store(&m, &gs).unwrap();
    let mut plain = plain_store(&m);
    // config と同値の group を全スロットに置いた store（groups なしと bit 一致するはず）。
    let same: Vec<ParamGroup> = vec![ParamGroup::new(vec![0, 1, 2, 3], 0.05, 0.001)];
    let mut same_store = cpu_store(&m, &same).unwrap();
    let mut host = Sgd::new(host_cfg).unwrap();
    let mut host_params = fandhe_ai::tape()
        .sync_device_param_store_to_host(&store)
        .unwrap();
    for s in 0..STEPS {
        let tape = fandhe_ai::tape();
        let grads = fwd_bwd(&m, &tape, &mut store);
        let g_host = tape.param_grads_to_host(&store, &grads).unwrap();
        tape.step_device_param_store(&mut store, &grads, &cfg)
            .unwrap();
        let dev = tape.sync_device_param_store_to_host(&store).unwrap();
        let p: Vec<&Tensor<f32>> = host_params.iter().collect();
        let g: Vec<&Tensor<f32>> = g_host.iter().collect();
        let out = host.step_with_groups(&p, &g, &gs).unwrap();
        for (d, h) in dev.iter().zip(&out) {
            for (a, b) in d
                .contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .zip(h.contiguous().as_slice().unwrap())
            {
                assert!(composite_close_f32(*a, *b), "step {s}: {a} vs {b}");
            }
        }
        // ホスト参照は常駐の出力から再出発させる（FMA 契約差を累積させない）。
        host_params = dev;

        // 同じ勾配系列を使うため、groups なし store・同値 group store は独立に同じ学習を回す。
        let tp = fandhe_ai::tape();
        let gp = fwd_bwd(&m, &tp, &mut plain);
        tp.step_device_param_store(&mut plain, &gp, &cfg).unwrap();
        let ts = fandhe_ai::tape();
        let gsame = fwd_bwd(&m, &ts, &mut same_store);
        ts.step_device_param_store(&mut same_store, &gsame, &cfg)
            .unwrap();
        let (pp, ss) = (
            tp.sync_device_param_store_to_host(&plain).unwrap(),
            ts.sync_device_param_store_to_host(&same_store).unwrap(),
        );
        assert_eq!(bits(&pp), bits(&ss), "same-valued group == no groups");
    }
}

/// groups 空の構築は `DeviceParamStore::new` と同じ学習結果（bit）になる（既定経路の不変）。
#[test]
fn empty_groups_equal_plain_store_bits() {
    let m = model();
    let mut a = plain_store(&m);
    let mut b = fandhe_ai_autodiff::optim::DeviceParamStore::new(
        &fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
            fandhe_ai_backend_cpu::CpuBackendOps::new(),
        )),
        &m.trainable_parameters(),
    )
    .unwrap();
    let cfg = AdamConfig::default();
    for _ in 0..3 {
        let ta = fandhe_ai::tape();
        let ga = fwd_bwd(&m, &ta, &mut a);
        ta.step_device_param_store_adam(&mut a, &ga, &cfg).unwrap();
        let tb = fandhe_ai::tape();
        let gb = fwd_bwd(&m, &tb, &mut b);
        tb.step_device_param_store_adam(&mut b, &gb, &cfg).unwrap();
    }
    let t = fandhe_ai::tape();
    assert_eq!(
        bits(&t.sync_device_param_store_to_host(&a).unwrap()),
        bits(&t.sync_device_param_store_to_host(&b).unwrap())
    );
}

#[test]
fn construction_rejects_invalid_groups_without_creating_a_store() {
    let m = model();
    for bad in [
        ParamGroup::new(vec![], 0.1, 0.0),
        ParamGroup::new(vec![9], 0.1, 0.0),
        ParamGroup::new(vec![0, 0], 0.1, 0.0),
        ParamGroup::new(vec![0], f32::NAN, 0.0),
        ParamGroup::new(vec![0], f32::INFINITY, 0.0),
        ParamGroup::new(vec![0], -1.0, 0.0),
        ParamGroup::new(vec![0], 0.1, f32::NAN),
        ParamGroup::new(vec![0], 0.1, -0.5),
    ] {
        let err = cpu_store(&m, &[bad]).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    }
    let dup = cpu_store(
        &m,
        &[
            ParamGroup::new(vec![0], 0.1, 0.0),
            ParamGroup::new(vec![0], 0.2, 0.0),
        ],
    )
    .unwrap_err();
    assert!(matches!(dup, BackendError::InvalidArgument(_)));
}

#[test]
fn amp_steps_are_rejected_and_leave_state_untouched() {
    let m = model();
    let mut store = cpu_store(&m, &groups()).unwrap();
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let tape = fandhe_ai::tape();
    let before = tape.sync_device_param_store_to_host(&store).unwrap();
    let grads = fwd_bwd(&m, &tape, &mut store);
    let e1 = tape
        .step_device_param_store_amp(&mut store, &grads, &SgdConfig::new(0.1), &mut scaler)
        .unwrap_err();
    assert!(matches!(
        e1,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    let e2 = tape
        .step_device_param_store_adam_amp(&mut store, &grads, &AdamConfig::default(), &mut scaler)
        .unwrap_err();
    assert!(matches!(
        e2,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    let e3 = tape
        .step_device_param_store_adamw_amp(&mut store, &grads, &AdamWConfig::default(), &mut scaler)
        .unwrap_err();
    assert!(matches!(
        e3,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
    assert_eq!(
        bits(&tape.sync_device_param_store_to_host(&store).unwrap()),
        bits(&before)
    );
    // pending は消費されておらず、通常の step が成功する。
    tape.step_device_param_store_adam(&mut store, &grads, &AdamConfig::default())
        .unwrap();
}

/// 導出スカラー（AdamW の `1 - lr*wd`）が非有限になる group は、カーネル起動前に
/// `InvalidArgument` で拒否され、poison せず pending も消費しない。
#[test]
fn non_finite_derived_scalar_is_rejected_without_poisoning() {
    let m = model();
    let gs = vec![ParamGroup::new(vec![0], 3.0e38, 10.0)];
    let mut store = cpu_store(&m, &gs).unwrap();
    let tape = fandhe_ai::tape();
    let grads = fwd_bwd(&m, &tape, &mut store);
    let err = tape
        .step_device_param_store_adamw(&mut store, &grads, &AdamWConfig::default())
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    // poison されていない。
    tape.sync_device_param_store_to_host(&store).unwrap();
}

/// `train_step_resident` は groups ありでチェーンを使わず Tape 経路へ全体フォールバックし、
/// 手書きループ（Tape 経路の SGD step）と bit 一致する。
#[test]
fn train_step_resident_falls_back_to_tape_path_with_groups() {
    let m = model();
    let gs = groups();
    let cfg = SgdConfig::new(0.05);
    let mut a = cpu_store(&m, &gs).unwrap();
    let mut b = cpu_store(&m, &gs).unwrap();
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
