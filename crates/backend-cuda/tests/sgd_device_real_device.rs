//! `CudaBackendOps::sgd_step_device`（イシュー #935・in-place デバイス
//! 常駐 SGD 更新）の実機必須テスト。`crates/backend-cuda/tests/
//! memory_real_device.rs` と同じ構成方針（`#[ignore]` 分離）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --test sgd_device_real_device -- --ignored --nocapture
//! ```
//!
//! `backend-cpu::ops::CpuBackendOps::sgd_step_device` との数値一致は
//! `crates/backend-cpu/tests/sgd_device_parity.rs` の CPU 対 CPU 参照実装
//! 突合とは別に、実機（CUDA vs CPU）横断の数値一致は本テストで検証する
//! （統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）。

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::{BackendOps, SgdStepConfig, Tensor};

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn momentum_sgd_matches_cpu_reference_across_multiple_steps() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut cuda_param = cuda_mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut cuda_velocity = cuda_mem.alloc_zeroed(&[4]).unwrap();
    let mut cpu_param = cpu_mem.upload(&Tensor::new(init, &[4]).unwrap()).unwrap();
    let mut cpu_velocity = cpu_mem.alloc_zeroed(&[4]).unwrap();

    for step in 0..5 {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data, &[4]).unwrap();
        let cuda_grad = cuda_mem.upload(&grad_tensor).unwrap();
        let cpu_grad = cpu_mem.upload(&grad_tensor).unwrap();

        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.01,
            nesterov: true,
            is_first_step: step == 0,
        };

        cuda_ops
            .sgd_step_device(
                &mut cuda_param,
                &cuda_grad,
                Some(&mut cuda_velocity),
                &config,
            )
            .expect("cuda sgd_step_device must succeed on real hardware");
        cpu_ops
            .sgd_step_device(&mut cpu_param, &cpu_grad, Some(&mut cpu_velocity), &config)
            .unwrap();
    }

    let cuda_result = cuda_mem.download(&cuda_param).unwrap();
    let cpu_result = cpu_mem.download(&cpu_param).unwrap();
    for i in 0..4 {
        assert_close(
            cuda_result.get(&[i]).unwrap(),
            cpu_result.get(&[i]).unwrap(),
            &format!("index {i}"),
        );
    }
}
