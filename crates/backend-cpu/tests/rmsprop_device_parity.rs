//! `CpuBackendOps::rmsprop_step_device`（in-place・デバイス常駐更新。
//! イシュー #2175）と `fandhe_ai_autodiff::nn::optim::rmsprop::RmsProp`
//! （ホスト参照実装）の数値一致検証。
//!
//! `RmsProp::step` と `rmsprop_step_device` は同一の演算列（`f32::
//! mul_add`・同一の項の並び）を使う設計であり、**`to_bits()` による
//! bit 完全一致**で突合する（`adam_device_parity.rs` と同じ方針）。

use fandhe_ai_autodiff::nn::optim::{RmsProp, RmsPropConfig};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, RmsPropOptionalBuffers, RmsPropStepConfig, Tensor,
};

fn assert_bits_eq(actual: f32, expected: f32, ctx: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{ctx}: actual={actual} ({:#010x}) expected={expected} ({:#010x})",
        actual.to_bits(),
        expected.to_bits()
    );
}

fn run_parity(config: RmsPropConfig, steps: usize) {
    let ops = CpuBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut square_avg_buf = mem.alloc_zeroed(&[4]).unwrap();
    let mut grad_avg_buf = mem.alloc_zeroed(&[4]).unwrap();
    let mut momentum_buf = mem.alloc_zeroed(&[4]).unwrap();

    let mut host_rmsprop = RmsProp::new(config).unwrap();
    let mut host_param = Tensor::new(init, &[4]).unwrap();

    let device_config = RmsPropStepConfig {
        lr: config.lr,
        alpha: config.alpha,
        eps: config.eps,
        weight_decay: config.weight_decay,
        momentum: config.momentum,
        centered: config.centered,
    };

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data, &[4]).unwrap();
        let grad_buf = mem.upload(&grad_tensor).unwrap();

        ops.rmsprop_step_device(
            &mut param_buf,
            &grad_buf,
            &mut square_avg_buf,
            RmsPropOptionalBuffers {
                grad_avg: if config.centered {
                    Some(&mut grad_avg_buf)
                } else {
                    None
                },
                momentum_buf: if config.momentum > 0.0 {
                    Some(&mut momentum_buf)
                } else {
                    None
                },
            },
            &device_config,
        )
        .unwrap();

        let host_out = host_rmsprop.step(&[(&host_param, &grad_tensor)]).unwrap();
        host_param = host_out.into_iter().next().unwrap();
    }

    let device_result = mem.download(&param_buf).unwrap();
    for i in 0..4 {
        assert_bits_eq(
            device_result.get(&[i]).unwrap(),
            host_param.get(&[i]).unwrap(),
            &format!("RmsProp index {i} (config={config:?})"),
        );
    }
}

#[test]
fn default_config_matches_host_reference() {
    run_parity(RmsPropConfig::default(), 6);
}

#[test]
fn centered_matches_host_reference() {
    run_parity(
        RmsPropConfig {
            centered: true,
            ..RmsPropConfig::default()
        },
        6,
    );
}

#[test]
fn momentum_matches_host_reference() {
    run_parity(
        RmsPropConfig {
            momentum: 0.9,
            ..RmsPropConfig::default()
        },
        6,
    );
}

#[test]
fn weight_decay_matches_host_reference() {
    run_parity(
        RmsPropConfig {
            weight_decay: 0.01,
            ..RmsPropConfig::default()
        },
        6,
    );
}

#[test]
fn centered_and_momentum_matches_host_reference() {
    run_parity(
        RmsPropConfig {
            centered: true,
            momentum: 0.9,
            ..RmsPropConfig::default()
        },
        6,
    );
}

#[test]
fn alpha_below_half_matches_host_reference() {
    // `alpha < 0.5` は centered lerp の終点基準分岐（`weight = 1-alpha
    // >= 0.5`）を通る（`rmsprop.rs` モジュールコメント参照）。
    run_parity(
        RmsPropConfig {
            alpha: 0.3,
            centered: true,
            ..RmsPropConfig::default()
        },
        6,
    );
}

#[test]
fn rmsprop_step_device_rejects_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap())
        .unwrap();
    let mut square_avg_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = RmsPropStepConfig {
        lr: 1e-2,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        momentum: 0.0,
        centered: false,
    };
    let err = ops
        .rmsprop_step_device(
            &mut param_buf,
            &grad_buf,
            &mut square_avg_buf,
            RmsPropOptionalBuffers {
                grad_avg: None,
                momentum_buf: None,
            },
            &config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn rmsprop_step_device_rejects_missing_grad_avg_when_centered() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let init = vec![1.0f32, -2.0];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![0.1, 0.2], &[2]).unwrap())
        .unwrap();
    let mut square_avg_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = RmsPropStepConfig {
        lr: 1e-2,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        momentum: 0.0,
        centered: true,
    };
    let err = ops
        .rmsprop_step_device(
            &mut param_buf,
            &grad_buf,
            &mut square_avg_buf,
            RmsPropOptionalBuffers {
                grad_avg: None,
                momentum_buf: None,
            },
            &config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)));
    // どの要素も更新前に返す契約（`param` が変更されていない）。
    let out = mem.download(&param_buf).unwrap();
    assert_eq!(out.get(&[0]).unwrap(), init[0]);
    assert_eq!(out.get(&[1]).unwrap(), init[1]);
}

#[test]
fn rmsprop_step_device_rejects_missing_momentum_buf() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let init = vec![1.0f32, -2.0];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![0.1, 0.2], &[2]).unwrap())
        .unwrap();
    let mut square_avg_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = RmsPropStepConfig {
        lr: 1e-2,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        momentum: 0.9,
        centered: false,
    };
    let err = ops
        .rmsprop_step_device(
            &mut param_buf,
            &grad_buf,
            &mut square_avg_buf,
            RmsPropOptionalBuffers {
                grad_avg: None,
                momentum_buf: None,
            },
            &config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)));
    let out = mem.download(&param_buf).unwrap();
    assert_eq!(out.get(&[0]).unwrap(), init[0]);
    assert_eq!(out.get(&[1]).unwrap(), init[1]);
}
