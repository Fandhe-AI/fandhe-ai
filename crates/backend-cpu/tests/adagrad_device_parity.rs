//! `CpuBackendOps::adagrad_step_device`（in-place・デバイス常駐更新。
//! イシュー #2175）と `fandhe_ai_autodiff::nn::optim::adagrad::Adagrad`
//! （ホスト参照実装）の数値一致検証。
//!
//! `Adagrad::step` と `adagrad_step_device` は同一の演算列を使う設計で
//! あり、**`to_bits()` による bit 完全一致**で突合する
//! （`adam_device_parity.rs` と同じ方針）。`clr` はホスト `Adagrad::
//! step_with_slot_hparams` と同じ式でテスト側が事前計算してカーネルへ
//! 渡す（`AdagradStepConfig` doc コメント参照）。

use fandhe_ai_autodiff::nn::optim::{Adagrad, AdagradConfig};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{AdagradStepConfig, BackendError, BackendOps, Tensor};

fn assert_bits_eq(actual: f32, expected: f32, ctx: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{ctx}: actual={actual} ({:#010x}) expected={expected} ({:#010x})",
        actual.to_bits(),
        expected.to_bits()
    );
}

fn run_parity(config: AdagradConfig, steps: usize) {
    let ops = CpuBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut state_sum_buf = mem
        .upload(&Tensor::new(vec![config.initial_accumulator_value; 4], &[4]).unwrap())
        .unwrap();

    let mut host_adagrad = Adagrad::new(config).unwrap();
    let mut host_param = Tensor::new(init, &[4]).unwrap();

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data, &[4]).unwrap();
        let grad_buf = mem.upload(&grad_tensor).unwrap();

        // ホスト `Adagrad::step_with_slot_hparams` と同一式で `clr` を
        // 事前計算する（`AdagradStepConfig::clr` doc コメント参照）。
        let step_no = (step + 1) as f64;
        let clr = (config.lr as f64 / (1.0 + (step_no - 1.0) * config.lr_decay as f64)) as f32;
        let device_config = AdagradStepConfig {
            clr,
            eps: config.eps,
            weight_decay: config.weight_decay,
        };

        ops.adagrad_step_device(
            &mut param_buf,
            &grad_buf,
            &mut state_sum_buf,
            &device_config,
        )
        .unwrap();

        let host_out = host_adagrad.step(&[(&host_param, &grad_tensor)]).unwrap();
        host_param = host_out.into_iter().next().unwrap();
    }

    let device_result = mem.download(&param_buf).unwrap();
    for i in 0..4 {
        assert_bits_eq(
            device_result.get(&[i]).unwrap(),
            host_param.get(&[i]).unwrap(),
            &format!("Adagrad index {i} (config={config:?})"),
        );
    }
}

#[test]
fn default_config_matches_host_reference() {
    run_parity(AdagradConfig::default(), 6);
}

#[test]
fn lr_decay_matches_host_reference() {
    run_parity(
        AdagradConfig {
            lr_decay: 0.1,
            ..AdagradConfig::default()
        },
        6,
    );
}

#[test]
fn initial_accumulator_value_matches_host_reference() {
    run_parity(
        AdagradConfig {
            initial_accumulator_value: 0.5,
            ..AdagradConfig::default()
        },
        6,
    );
}

#[test]
fn weight_decay_matches_host_reference() {
    run_parity(
        AdagradConfig {
            weight_decay: 0.01,
            ..AdagradConfig::default()
        },
        6,
    );
}

#[test]
fn adagrad_step_device_rejects_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap())
        .unwrap();
    let mut state_sum_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = AdagradStepConfig {
        clr: 1e-2,
        eps: 1e-10,
        weight_decay: 0.0,
    };
    let err = ops
        .adagrad_step_device(&mut param_buf, &grad_buf, &mut state_sum_buf, &config)
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
