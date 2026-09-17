//! `CpuBackendOps::adam_step_device`（in-place・デバイス常駐更新。イシュー
//! #1959）と `fandhe_ai_autodiff::nn::optim::{adam::Adam, adamw::AdamW}`
//! （ホスト参照実装）の数値一致検証。
//!
//! `Adam::step`／`AdamW::step` と `adam_step_device` は同一の演算列
//! （`f32::mul_add`・同一の項の並び）を使う設計であり、`beta1_pow_t`／
//! `beta2_pow_t`（`f64` 逐次積）から導出する `step_size`／
//! `bias_correction2_sqrt`（Coupled のみ `decay_factor`）をテスト側も
//! ホスト実装と同じ式で計算してカーネルへ渡すため、**`to_bits()` による
//! bit 完全一致**で突合する（`docs/device-resident-update-design.md`・
//! `.claude/rules/coding-rust.md` の CPU 実装 bit 一致契約）。

use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig, AdamW, AdamWConfig};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{AdamStepConfig, AdamStepKind, BackendError, BackendOps, Tensor};

fn assert_bits_eq(actual: f32, expected: f32, ctx: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{ctx}: actual={actual} ({:#010x}) expected={expected} ({:#010x})",
        actual.to_bits(),
        expected.to_bits()
    );
}

/// `adam_step_device`（`AdamStepKind::Coupled`）を N ステップ回し、同じ
/// ハイパーパラメータで `Adam::step`（ホスト参照実装）を回した結果と
/// bit 完全一致で突合する。
fn run_adam_parity(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, steps: usize) {
    let ops = CpuBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[4]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[4]).unwrap();

    let mut host_adam = Adam::new(AdamConfig {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    })
    .unwrap();
    let mut host_param = Tensor::new(init, &[4]).unwrap();

    // ホスト側 `Adam::step` と同じ `f64` 逐次積で `beta^t` を追跡する
    // （`AdamStepConfig` doc コメントの契約どおり、カーネル呼び出し側が
    // 毎ステップ `step_size`／`bias_correction2_sqrt` を導出して渡す）。
    let mut beta1_pow_t: f64 = 1.0;
    let mut beta2_pow_t: f64 = 1.0;

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data.clone(), &[4]).unwrap();
        let grad_buf = mem.upload(&grad_tensor).unwrap();

        beta1_pow_t *= beta1 as f64;
        beta2_pow_t *= beta2 as f64;
        let bias_correction1 = 1.0 - beta1_pow_t;
        let bias_correction2 = 1.0 - beta2_pow_t;
        let device_config = AdamStepConfig {
            beta1,
            beta2,
            eps,
            weight_decay,
            decay_factor: 1.0,
            step_size: (lr as f64 / bias_correction1) as f32,
            bias_correction2_sqrt: bias_correction2.sqrt() as f32,
            kind: AdamStepKind::Coupled,
        };
        ops.adam_step_device(
            &mut param_buf,
            &grad_buf,
            &mut m_buf,
            &mut v_buf,
            &device_config,
        )
        .unwrap();

        let host_out = host_adam.step(&[(&host_param, &grad_tensor)]).unwrap();
        host_param = host_out.into_iter().next().unwrap();
    }

    let device_result = mem.download(&param_buf).unwrap();
    for i in 0..4 {
        assert_bits_eq(
            device_result.get(&[i]).unwrap(),
            host_param.get(&[i]).unwrap(),
            &format!(
                "Adam index {i} (lr={lr}, beta1={beta1}, beta2={beta2}, weight_decay={weight_decay})"
            ),
        );
    }
}

/// `adam_step_device`（`AdamStepKind::Decoupled`）を N ステップ回し、
/// `AdamW::step`（ホスト参照実装）と bit 完全一致で突合する。
fn run_adamw_parity(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, steps: usize) {
    let ops = CpuBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[4]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[4]).unwrap();

    let mut host_adamw = AdamW::new(AdamWConfig {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    })
    .unwrap();
    let mut host_param = Tensor::new(init, &[4]).unwrap();

    let mut beta1_pow_t: f64 = 1.0;
    let mut beta2_pow_t: f64 = 1.0;

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data.clone(), &[4]).unwrap();
        let grad_buf = mem.upload(&grad_tensor).unwrap();

        beta1_pow_t *= beta1 as f64;
        beta2_pow_t *= beta2 as f64;
        let bias_correction1 = 1.0 - beta1_pow_t;
        let bias_correction2 = 1.0 - beta2_pow_t;
        let device_config = AdamStepConfig {
            beta1,
            beta2,
            eps,
            weight_decay,
            decay_factor: 1.0 - lr * weight_decay,
            step_size: (lr as f64 / bias_correction1) as f32,
            bias_correction2_sqrt: bias_correction2.sqrt() as f32,
            kind: AdamStepKind::Decoupled,
        };
        ops.adam_step_device(
            &mut param_buf,
            &grad_buf,
            &mut m_buf,
            &mut v_buf,
            &device_config,
        )
        .unwrap();

        let host_out = host_adamw.step(&[(&host_param, &grad_tensor)]).unwrap();
        host_param = host_out.into_iter().next().unwrap();
    }

    let device_result = mem.download(&param_buf).unwrap();
    for i in 0..4 {
        assert_bits_eq(
            device_result.get(&[i]).unwrap(),
            host_param.get(&[i]).unwrap(),
            &format!(
                "AdamW index {i} (lr={lr}, beta1={beta1}, beta2={beta2}, weight_decay={weight_decay})"
            ),
        );
    }
}

#[test]
fn adam_wd_zero_matches_host_reference() {
    run_adam_parity(1e-3, 0.9, 0.999, 1e-8, 0.0, 10);
}

#[test]
fn adam_wd_nonzero_matches_host_reference() {
    run_adam_parity(1e-3, 0.9, 0.999, 1e-8, 0.01, 10);
}

#[test]
fn adamw_wd_zero_matches_host_reference() {
    run_adamw_parity(1e-3, 0.9, 0.999, 1e-8, 0.0, 10);
}

#[test]
fn adamw_wd_nonzero_matches_host_reference() {
    run_adamw_parity(1e-3, 0.9, 0.999, 1e-8, 0.01, 10);
}

/// `Adam(wd=0)` と `AdamW(wd=0)` はホスト実装同士で bit 一致する契約
/// （`adam.rs` モジュール doc「`weight_decay == 0` で完全に一致する」）。
/// これがデバイス側カーネルでも保たれることを確認する。
#[test]
fn adam_and_adamw_kernels_agree_when_wd_zero() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut adam_param = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();
    let mut adam_m = mem.alloc_zeroed(&[4]).unwrap();
    let mut adam_v = mem.alloc_zeroed(&[4]).unwrap();
    let mut adamw_param = mem.upload(&Tensor::new(init, &[4]).unwrap()).unwrap();
    let mut adamw_m = mem.alloc_zeroed(&[4]).unwrap();
    let mut adamw_v = mem.alloc_zeroed(&[4]).unwrap();

    let (lr, beta1, beta2, eps) = (1e-3f32, 0.9f32, 0.999f32, 1e-8f32);
    let mut beta1_pow_t: f64 = 1.0;
    let mut beta2_pow_t: f64 = 1.0;
    for step in 0..5 {
        let grad_data: Vec<f32> = (0..4)
            .map(|i| 0.1 * (step as f32 + 1.0) + 0.05 * i as f32)
            .collect();
        let grad_buf = mem.upload(&Tensor::new(grad_data, &[4]).unwrap()).unwrap();

        beta1_pow_t *= beta1 as f64;
        beta2_pow_t *= beta2 as f64;
        let step_size = (lr as f64 / (1.0 - beta1_pow_t)) as f32;
        let bias_correction2_sqrt = (1.0 - beta2_pow_t).sqrt() as f32;

        ops.adam_step_device(
            &mut adam_param,
            &grad_buf,
            &mut adam_m,
            &mut adam_v,
            &AdamStepConfig {
                beta1,
                beta2,
                eps,
                weight_decay: 0.0,
                decay_factor: 1.0,
                step_size,
                bias_correction2_sqrt,
                kind: AdamStepKind::Coupled,
            },
        )
        .unwrap();
        ops.adam_step_device(
            &mut adamw_param,
            &grad_buf,
            &mut adamw_m,
            &mut adamw_v,
            &AdamStepConfig {
                beta1,
                beta2,
                eps,
                weight_decay: 0.0,
                decay_factor: 1.0, // 1 - lr*0 == 1.0
                step_size,
                bias_correction2_sqrt,
                kind: AdamStepKind::Decoupled,
            },
        )
        .unwrap();
    }

    let adam_result = mem.download(&adam_param).unwrap();
    let adamw_result = mem.download(&adamw_param).unwrap();
    for i in 0..4 {
        assert_bits_eq(
            adam_result.get(&[i]).unwrap(),
            adamw_result.get(&[i]).unwrap(),
            &format!("index {i}"),
        );
    }
}

#[test]
fn adam_step_device_rejects_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[2]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        decay_factor: 1.0,
        step_size: 0.1,
        bias_correction2_sqrt: 1.0,
        kind: AdamStepKind::Coupled,
    };
    let err = ops
        .adam_step_device(&mut param_buf, &grad_buf, &mut m_buf, &mut v_buf, &config)
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn empty_tensor_adam_step_is_a_no_op() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(Vec::<f32>::new(), &[0]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(Vec::<f32>::new(), &[0]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[0]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[0]).unwrap();
    let config = AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        decay_factor: 1.0,
        step_size: 0.1,
        bias_correction2_sqrt: 1.0,
        kind: AdamStepKind::Coupled,
    };
    ops.adam_step_device(&mut param_buf, &grad_buf, &mut m_buf, &mut v_buf, &config)
        .unwrap();
    let out = mem.download(&param_buf).unwrap();
    assert_eq!(out.numel(), 0);
}
