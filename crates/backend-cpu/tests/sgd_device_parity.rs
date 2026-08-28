//! `CpuBackendOps::sgd_step_device`（in-place・デバイス常駐更新。イシュー
//! #935）と `fandhe_ai_autodiff::optim::Sgd::step`（ホスト参照実装）の数値
//! 一致検証。
//!
//! `Sgd::step` は式順序（weight_decay → momentum〈初回 `b ← g` 分岐〉→
//! nesterov → 減算）の正本であり、`sgd_step_device` はこれと同一順序で
//! `f32::mul_add`（backend-cpu 側の FMA 契約統一方針。`.claude/rules/
//! coding-rust.md`）を用いる。両者は丸め手順が完全には同一でない
//! （`Sgd::step` は PyTorch fixture parity のため `mul_add` を使わない）
//! ため、突合は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5
//! 未満」（`.claude/rules/coding-rust.md`）で行う。

use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, SgdStepConfig, Tensor};

/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

/// `sgd_step_device` を N ステップ回し、同じハイパーパラメータで
/// `Sgd::step`（ホスト参照実装）を回した結果と突合する。
fn run_parity(momentum: f32, dampening: f32, weight_decay: f32, nesterov: bool, steps: usize) {
    let ops = CpuBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut velocity_buf = if momentum != 0.0 {
        Some(mem.alloc_zeroed(&[4]).unwrap())
    } else {
        None
    };

    let mut sgd_config = SgdConfig::new(0.1);
    if momentum != 0.0 {
        sgd_config = sgd_config.with_momentum(momentum);
    }
    if dampening != 0.0 {
        sgd_config = sgd_config.with_dampening(dampening);
    }
    if weight_decay != 0.0 {
        sgd_config = sgd_config.with_weight_decay(weight_decay);
    }
    if nesterov {
        sgd_config = sgd_config.with_nesterov(true);
    }
    let mut sgd = Sgd::new(sgd_config).unwrap();
    let mut host_param = Tensor::new(init, &[4]).unwrap();

    for step in 0..steps {
        // ステップごとに異なる勾配を使う（全ステップ同一だと momentum
        // バッファの経路が実質定数になり検証にならない）。
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data.clone(), &[4]).unwrap();

        // デバイス側: grad のみ毎ステップ upload、param/velocity は使い回す。
        let grad_buf = mem.upload(&grad_tensor).unwrap();
        let device_config = SgdStepConfig {
            lr: 0.1,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            is_first_step: step == 0,
        };
        ops.sgd_step_device(
            &mut param_buf,
            &grad_buf,
            velocity_buf.as_mut(),
            &device_config,
        )
        .unwrap();

        // ホスト側参照実装。
        let host_out = sgd.step(&[&host_param], &[&grad_tensor]).unwrap();
        host_param = host_out.into_iter().next().unwrap();
    }

    let device_result = mem.download(&param_buf).unwrap();
    for i in 0..4 {
        assert_close(
            device_result.get(&[i]).unwrap(),
            host_param.get(&[i]).unwrap(),
            &format!(
                "index {i} (momentum={momentum}, dampening={dampening}, weight_decay={weight_decay}, nesterov={nesterov})"
            ),
        );
    }
}

#[test]
fn vanilla_sgd_matches_host_reference() {
    run_parity(0.0, 0.0, 0.0, false, 5);
}

#[test]
fn momentum_sgd_matches_host_reference() {
    run_parity(0.9, 0.0, 0.0, false, 5);
}

#[test]
fn momentum_with_dampening_matches_host_reference() {
    run_parity(0.9, 0.2, 0.0, false, 5);
}

#[test]
fn weight_decay_matches_host_reference() {
    run_parity(0.0, 0.0, 0.01, false, 5);
}

#[test]
fn nesterov_momentum_matches_host_reference() {
    run_parity(0.9, 0.0, 0.0, true, 5);
}

#[test]
fn full_combo_matches_host_reference() {
    run_parity(0.9, 0.0, 0.01, true, 8);
}

#[test]
fn sgd_step_device_rejects_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap())
        .unwrap();
    let config = SgdStepConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    let err = ops
        .sgd_step_device(&mut param_buf, &grad_buf, None, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::ShapeMismatch(_)
    ));
}

#[test]
fn sgd_step_device_rejects_missing_velocity_when_momentum_enabled() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem.upload(&Tensor::new(vec![1.0], &[1]).unwrap()).unwrap();
    let grad_buf = mem.upload(&Tensor::new(vec![0.1], &[1]).unwrap()).unwrap();
    let config = SgdStepConfig {
        lr: 0.1,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    let err = ops
        .sgd_step_device(&mut param_buf, &grad_buf, None, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::Unsupported(_)
    ));
}

#[test]
fn empty_tensor_step_is_a_no_op() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(Vec::<f32>::new(), &[0]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(Vec::<f32>::new(), &[0]).unwrap())
        .unwrap();
    let config = SgdStepConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    ops.sgd_step_device(&mut param_buf, &grad_buf, None, &config)
        .unwrap();
    let out = mem.download(&param_buf).unwrap();
    assert_eq!(out.numel(), 0);
}
