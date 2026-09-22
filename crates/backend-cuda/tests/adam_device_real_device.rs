//! `CudaBackendOps::adam_step_device`（イシュー #2069・in-place デバイス
//! 常駐 Adam・AdamW 更新）の実機必須テスト。`crates/backend-cuda/tests/
//! sgd_device_real_device.rs` と同じ構成方針（`#[ignore]` 分離）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test adam_device_real_device -- --ignored --nocapture
//! ```
//!
//! `backend-cpu::ops::CpuBackendOps::adam_step_device` との CPU 対 CPU
//! 参照実装突合は `crates/backend-cpu/tests/adam_device_parity.rs` が
//! bit 完全一致で検証済み。本テストは実機（CUDA vs CPU）横断の数値一致を
//! 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で検証し、
//! あわせて run-to-run 決定性（`to_bits()` 完全一致）を検査する
//! （事前登録判定規則。`docs/perf/logs/adam-device-step-cuda-2069/
//! README.md` 参照）。

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::{AdamStepConfig, AdamStepKind, BackendOps, Tensor};

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

/// `kind`・`weight_decay`・`numel` を指定して `steps` 回
/// `adam_step_device` を CUDA／CPU の双方で回し、最終パラメータ・
/// bit 一致要素数を返す（`steps` 回のうち何要素が GPU/CPU で bit
/// 完全一致したかは付帯記録のみに使う。事前登録判定規則 §5 item 4）。
fn run_adam_parity(
    kind: AdamStepKind,
    weight_decay: f32,
    numel: usize,
    steps: usize,
) -> (Vec<f32>, Vec<f32>, usize) {
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

    let init: Vec<f32> = (0..numel)
        .map(|i| 1.0 - 0.001 * i as f32 + if i % 2 == 0 { 0.5 } else { -0.25 })
        .collect();
    let mut cuda_param = cuda_mem
        .upload(&Tensor::new(init.clone(), &[numel]).unwrap())
        .unwrap();
    let mut cuda_m = cuda_mem.alloc_zeroed(&[numel]).unwrap();
    let mut cuda_v = cuda_mem.alloc_zeroed(&[numel]).unwrap();
    let mut cpu_param = cpu_mem
        .upload(&Tensor::new(init, &[numel]).unwrap())
        .unwrap();
    let mut cpu_m = cpu_mem.alloc_zeroed(&[numel]).unwrap();
    let mut cpu_v = cpu_mem.alloc_zeroed(&[numel]).unwrap();

    let lr = 0.001f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    // ホスト `Adam::step`／`AdamW::step` と同じ `f64` 逐次積で `beta^t`
    // を追跡する（`AdamStepConfig` doc コメントの契約。`beta1_pow_t`／
    // `beta2_pow_t` はカーネルへ渡す前に呼び出し元〈ここではテスト〉が
    // 導出する）。
    let mut beta1_pow_t: f64 = 1.0;
    let mut beta2_pow_t: f64 = 1.0;

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..numel)
            .map(|i| 0.05 * (step as f32 + 1.0) + 0.01 * i as f32 - 0.02)
            .collect();
        let grad_tensor = Tensor::new(grad_data, &[numel]).unwrap();
        let cuda_grad = cuda_mem.upload(&grad_tensor).unwrap();
        let cpu_grad = cpu_mem.upload(&grad_tensor).unwrap();

        beta1_pow_t *= beta1 as f64;
        beta2_pow_t *= beta2 as f64;
        let bias_correction1 = 1.0 - beta1_pow_t;
        let bias_correction2 = 1.0 - beta2_pow_t;
        let decay_factor = match kind {
            AdamStepKind::Coupled => 1.0,
            AdamStepKind::Decoupled => 1.0 - lr * weight_decay,
            _ => unreachable!("test only exercises Coupled/Decoupled"),
        };
        let config = AdamStepConfig {
            beta1,
            beta2,
            eps: 1e-8,
            weight_decay,
            decay_factor,
            step_size: (lr as f64 / bias_correction1) as f32,
            bias_correction2_sqrt: bias_correction2.sqrt() as f32,
            kind,
        };

        cuda_ops
            .adam_step_device(
                &mut cuda_param,
                &cuda_grad,
                &mut cuda_m,
                &mut cuda_v,
                &config,
            )
            .expect("cuda adam_step_device must succeed on real hardware");
        cpu_ops
            .adam_step_device(&mut cpu_param, &cpu_grad, &mut cpu_m, &mut cpu_v, &config)
            .unwrap();
    }

    let cuda_result = cuda_mem.download(&cuda_param).unwrap();
    let cpu_result = cpu_mem.download(&cpu_param).unwrap();
    let mut cuda_vec = Vec::with_capacity(numel);
    let mut cpu_vec = Vec::with_capacity(numel);
    let mut bit_exact_count = 0usize;
    for i in 0..numel {
        let a = cuda_result.get(&[i]).unwrap();
        let b = cpu_result.get(&[i]).unwrap();
        if a.to_bits() == b.to_bits() {
            bit_exact_count += 1;
        }
        cuda_vec.push(a);
        cpu_vec.push(b);
    }
    (cuda_vec, cpu_vec, bit_exact_count)
}

fn assert_parity(kind: AdamStepKind, weight_decay: f32, numel: usize, steps: usize, label: &str) {
    let (cuda_result, cpu_result, bit_exact_count) =
        run_adam_parity(kind, weight_decay, numel, steps);
    for i in 0..numel {
        assert_close(
            cuda_result[i],
            cpu_result[i],
            &format!(
                "{label} index {i} (kind={kind:?}, weight_decay={weight_decay}, numel={numel}, steps={steps})"
            ),
        );
    }
    eprintln!(
        "{label}: bit-exact {bit_exact_count}/{numel} elements (kind={kind:?}, weight_decay={weight_decay}, steps={steps})"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn coupled_wd_zero_matches_cpu_reference_across_100_steps() {
    assert_parity(AdamStepKind::Coupled, 0.0, 4, 100, "Coupled wd=0");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn coupled_wd_nonzero_matches_cpu_reference_across_100_steps() {
    assert_parity(AdamStepKind::Coupled, 0.01, 4, 100, "Coupled wd=0.01");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn decoupled_wd_zero_matches_cpu_reference_across_100_steps() {
    assert_parity(AdamStepKind::Decoupled, 0.0, 4, 100, "Decoupled wd=0");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn decoupled_wd_nonzero_matches_cpu_reference_across_100_steps() {
    assert_parity(AdamStepKind::Decoupled, 0.01, 4, 100, "Decoupled wd=0.01");
}

/// 複数ブロック＋末尾ブロック余剰スレッド（`numel % ADAM_BLOCK_DIM !=
/// 0`）の境界チェック（REQ-8）を実機で検証する奇数 numel ケース。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn odd_numel_multi_block_matches_cpu_reference_across_5_steps() {
    assert_parity(AdamStepKind::Coupled, 0.01, 65_549, 5, "odd numel");
}

/// 事前登録判定規則 §5 item 2: 同一入力・同一設定で 5 回実行し、
/// `to_bits()` 列が 5 回とも完全一致すること（run-to-run 決定性）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn run_to_run_bit_identical_across_5_runs() {
    let mut reference: Option<Vec<u32>> = None;
    for run in 0..5 {
        let device =
            CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
        let cuda_ops = CudaBackendOps::new(device.ordinal());
        let mem = cuda_ops
            .memory_ops()
            .expect("CudaBackendOps must implement MemoryOps");

        let numel = 4usize;
        let init: Vec<f32> = vec![1.0, -2.0, 0.5, 3.25];
        let mut param = mem.upload(&Tensor::new(init, &[numel]).unwrap()).unwrap();
        let mut m = mem.alloc_zeroed(&[numel]).unwrap();
        let mut v = mem.alloc_zeroed(&[numel]).unwrap();

        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let mut beta1_pow_t: f64 = 1.0;
        let mut beta2_pow_t: f64 = 1.0;

        for step in 0..20 {
            let grad_data: Vec<f32> = (0..numel)
                .map(|i| 0.05 * (step as f32 + 1.0) + 0.01 * i as f32)
                .collect();
            let grad = mem
                .upload(&Tensor::new(grad_data, &[numel]).unwrap())
                .unwrap();

            beta1_pow_t *= beta1 as f64;
            beta2_pow_t *= beta2 as f64;
            let config = AdamStepConfig {
                beta1,
                beta2,
                eps: 1e-8,
                weight_decay: 0.01,
                decay_factor: 1.0,
                step_size: (lr as f64 / (1.0 - beta1_pow_t)) as f32,
                bias_correction2_sqrt: (1.0 - beta2_pow_t).sqrt() as f32,
                kind: AdamStepKind::Coupled,
            };
            cuda_ops
                .adam_step_device(&mut param, &grad, &mut m, &mut v, &config)
                .unwrap();
        }

        let result = mem.download(&param).unwrap();
        let bits: Vec<u32> = (0..numel)
            .map(|i| result.get(&[i]).unwrap().to_bits())
            .collect();
        match &reference {
            None => reference = Some(bits),
            Some(expected) => {
                assert_eq!(
                    &bits, expected,
                    "run {run}: bit 列が前回実行と一致しない（run-to-run 決定性違反）"
                );
            }
        }
    }
}
