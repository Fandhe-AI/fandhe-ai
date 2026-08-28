//! `MetalBackendOps::sgd_step_device`（イシュー #935・in-place デバイス
//! 常駐 SGD 更新）の実機テスト。
//!
//! `MetalBackendOps` は `crates/backend-metal/src/lib.rs` 側で
//! `cfg(target_os = "macos")` ゲートされている（`backend-metal` クレート
//! 自体は全 OS でビルド対象になる）ため、本テストファイルにも同じ
//! `cfg(target_os = "macos")` を付けて非 macOS の CI（GitHub ホステッド
//! ubuntu-latest）でコンパイル対象から除外する（付けないと
//! `unresolved import` でビルド失敗する）。`cfg(target_os = "macos")` は
//! コンパイル対象を絞るのみで実機の有無までは保証しないため
//! （`.claude/rules/ci.md`「実機依存」節・`.claude/rules/coding-rust.md`
//! 「テスト・ベンチ」節）、各 `#[test]` には理由付き `#[ignore]` を付け
//! `crates/backend-metal/tests/cpu_metal_parity.rs` と同じ方針で通常
//! CI（GitHub ホステッド ubuntu-latest。Metal 実機非搭載）から除外し、
//! macOS 実機で `--ignored` を明示指定したときのみ実行する:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

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
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn vanilla_sgd_matches_cpu_reference() {
    run_parity(0.0, 0.0, 0.0, false, 5);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn momentum_sgd_matches_cpu_reference() {
    run_parity(0.9, 0.0, 0.0, false, 5);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn full_combo_matches_cpu_reference() {
    run_parity(0.9, 0.0, 0.01, true, 8);
}

/// イシュー #936 §5.3（`docs/device-resident-update-design.md`）が要求する
/// 「100 step 程度の累積・最終値判定」を Metal 実機で保険的に検証する
/// （`crates/backend-cpu/tests/sgd_device_parity.rs` の同名ケースと対の
/// Metal 版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn full_combo_matches_cpu_reference_across_100_steps() {
    run_parity(0.9, 0.0, 0.01, true, 100);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
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
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
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
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
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
