//! `MetalBackendOps::sgd_step_device`（イシュー #935・in-place デバイス
//! 常駐 SGD 更新）の実機テスト。
//!
//! Metal 実機（Apple Silicon）は本 CI・実装検証環境で利用可能なため
//! （`.claude/rules/ci.md`「実機依存」節・実機検証環境ドキュメント）、
//! `backend-cuda` 側の `#[ignore]` 分離とは異なり通常テストとして実行する
//! （`crates/backend-metal/tests/` 配下の既存 parity テスト群と同じ方針。
//! macOS 以外の CI（Linux ubuntu-latest）では本クレート自体が
//! `cfg(target_os = "macos")` で空になるためコンパイル対象に入らない）。

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, SgdStepConfig, Tensor};

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn run_parity(momentum: f32, dampening: f32, weight_decay: f32, nesterov: bool, steps: usize) {
    let metal_ops = MetalBackendOps::new();
    let metal_mem = metal_ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut metal_param = metal_mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut metal_velocity = if momentum != 0.0 {
        Some(metal_mem.alloc_zeroed(&[4]).unwrap())
    } else {
        None
    };
    let mut cpu_param = cpu_mem.upload(&Tensor::new(init, &[4]).unwrap()).unwrap();
    let mut cpu_velocity = if momentum != 0.0 {
        Some(cpu_mem.alloc_zeroed(&[4]).unwrap())
    } else {
        None
    };

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data, &[4]).unwrap();
        let metal_grad = metal_mem.upload(&grad_tensor).unwrap();
        let cpu_grad = cpu_mem.upload(&grad_tensor).unwrap();

        let config = SgdStepConfig {
            lr: 0.1,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            is_first_step: step == 0,
        };

        metal_ops
            .sgd_step_device(
                &mut metal_param,
                &metal_grad,
                metal_velocity.as_mut(),
                &config,
            )
            .expect("metal sgd_step_device must succeed on real hardware");
        cpu_ops
            .sgd_step_device(&mut cpu_param, &cpu_grad, cpu_velocity.as_mut(), &config)
            .unwrap();
    }

    let metal_result = metal_mem.download(&metal_param).unwrap();
    let cpu_result = cpu_mem.download(&cpu_param).unwrap();
    for i in 0..4 {
        assert_close(
            metal_result.get(&[i]).unwrap(),
            cpu_result.get(&[i]).unwrap(),
            &format!(
                "index {i} (momentum={momentum}, dampening={dampening}, weight_decay={weight_decay}, nesterov={nesterov})"
            ),
        );
    }
}

#[test]
fn vanilla_sgd_matches_cpu_reference() {
    run_parity(0.0, 0.0, 0.0, false, 5);
}

#[test]
fn momentum_sgd_matches_cpu_reference() {
    run_parity(0.9, 0.0, 0.0, false, 5);
}

#[test]
fn full_combo_matches_cpu_reference() {
    run_parity(0.9, 0.0, 0.01, true, 8);
}

#[test]
fn sgd_step_device_rejects_shape_mismatch() {
    let ops = MetalBackendOps::new();
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
    let ops = MetalBackendOps::new();
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
    let ops = MetalBackendOps::new();
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
