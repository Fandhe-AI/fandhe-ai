//! `CudaBackendOps::gemm_resident_rhs`／`gemm_resident_lhs`（イシュー
//! #1022・デバイス常駐パラメータのまま forward/backward する GEMM）の
//! 実機必須テスト。`crates/backend-cuda/tests/sgd_device_real_device.rs`
//! と同じ構成方針（`#[ignore]` 分離。CPU 参照実装との統一複合判定）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --test gemm_resident_real_device -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn assert_tensor_close(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    for (i, (av, ev)) in a
        .as_slice()
        .unwrap()
        .iter()
        .zip(e.as_slice().unwrap())
        .enumerate()
    {
        assert_close(*av, *ev, &format!("{ctx}: element {i}"));
    }
}

/// `gemm_resident_rhs`（CUDA）が CPU 参照実装（`gemm_bias_act`。ホスト
/// 常駐版）と統一複合判定内で一致することを、bias あり／なしの双方で
/// 検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_resident_rhs_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(m, k, n) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33)] {
        for has_bias in [false, true] {
            let a = tensor(
                (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect(),
                &[m, k],
            );
            let w = tensor(
                (0..k * n).map(|i| (i as f32) * 0.02 - 0.3).collect(),
                &[k, n],
            );
            let bias = has_bias.then(|| tensor((0..n).map(|i| i as f32 * 0.1).collect(), &[n]));

            let expected = cpu_ops
                .gemm_bias_act(
                    &a,
                    &w,
                    bias.as_ref(),
                    fandhe_ai_tensor_core::Activation::None,
                )
                .unwrap();

            let w_dev = cuda_mem.upload(&w).unwrap();
            let bias_dev = bias.as_ref().map(|b| cuda_mem.upload(b).unwrap());
            let actual = cuda_ops
                .gemm_resident_rhs(&a, &w_dev, bias_dev.as_ref())
                .unwrap();

            assert_tensor_close(
                &actual,
                &expected,
                &format!("gemm_resident_rhs m={m} k={k} n={n} has_bias={has_bias}"),
            );
        }
    }
}

/// `gemm_resident_lhs`（CUDA）が CPU 参照実装（`gemm`）と統一複合判定内で
/// 一致することを検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_resident_lhs_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(p, q, r) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33)] {
        let w = tensor(
            (0..p * q).map(|i| (i as f32) * 0.01 - 0.5).collect(),
            &[p, q],
        );
        let b = tensor(
            (0..q * r).map(|i| (i as f32) * 0.02 - 0.3).collect(),
            &[q, r],
        );

        let expected = cpu_ops.gemm(&w, &b).unwrap();

        let w_dev = cuda_mem.upload(&w).unwrap();
        let actual = cuda_ops.gemm_resident_lhs(&w_dev, &b).unwrap();

        assert_tensor_close(
            &actual,
            &expected,
            &format!("gemm_resident_lhs p={p} q={q} r={r}"),
        );
    }
}
