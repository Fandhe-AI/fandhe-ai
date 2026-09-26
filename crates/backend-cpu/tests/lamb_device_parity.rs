//! `CpuBackendOps::lamb_step_device`（in-place・デバイス常駐更新。
//! イシュー #2175）と `fandhe_ai_autodiff::nn::optim::lamb::Lamb`
//! （ホスト参照実装）の数値一致検証。
//!
//! `Lamb::step` は 1 スロット（1 パラメータテンソル）ごとに独立に
//! trust ratio を計算する（layer-wise）。`lamb_step_device` は連結
//! バッファ上で `segment_numels` によりこの境界を表現するため、テスト
//! 側は複数 `(param, grad)` ペアを 1 本の連結バッファへ結合してホスト
//! `Lamb::step`（複数ペアを同時に渡す呼び出し）と突合する。
//! **`to_bits()` による bit 完全一致**で判定する
//! （`adam_device_parity.rs` と同じ方針）。

use fandhe_ai_autodiff::nn::optim::{Lamb, LambConfig};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendError, BackendOps, LambMoments, LambStepConfig, Tensor};

fn assert_bits_eq(actual: f32, expected: f32, ctx: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{ctx}: actual={actual} ({:#010x}) expected={expected} ({:#010x})",
        actual.to_bits(),
        expected.to_bits()
    );
}

/// 複数 segment（例 `[3, 5, 1]`）の初期値・勾配生成子を受け取り、連結
/// バッファ上の `lamb_step_device` とホスト `Lamb::step`（segment ごとの
/// `(param, grad)` ペア列）を突合する。
fn run_parity(config: LambConfig, segment_numels: &[usize], steps: usize) {
    let ops = CpuBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");
    let total: usize = segment_numels.iter().sum();

    // segment ごとに異なる初期値を割り当てる（全ゼロ segment を含めたい
    // 場合は呼び出し側で `init_offset` を調整する。ここでは単純に
    // index ベースの初期値にする）。
    let init: Vec<f32> = (0..total).map(|i| 0.5 + 0.1 * i as f32).collect();

    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[total]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[total]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[total]).unwrap();

    let mut host_lamb = Lamb::new(config).unwrap();
    let mut host_params: Vec<Tensor<f32>> = {
        let mut offset = 0;
        let mut out = Vec::with_capacity(segment_numels.len());
        for &n in segment_numels {
            out.push(Tensor::new(init[offset..offset + n].to_vec(), &[n]).unwrap());
            offset += n;
        }
        out
    };

    let mut beta1_pow_t: f64 = 1.0;
    let mut beta2_pow_t: f64 = 1.0;

    for step in 0..steps {
        let grad_data: Vec<f32> = (0..total)
            .map(|i| 0.01 * (step as f32 + 1.0) + 0.002 * i as f32)
            .collect();
        let grad_tensor = Tensor::new(grad_data.clone(), &[total]).unwrap();
        let grad_buf = mem.upload(&grad_tensor).unwrap();

        beta1_pow_t *= config.beta1 as f64;
        beta2_pow_t *= config.beta2 as f64;
        let bias_correction1 = 1.0 - beta1_pow_t;
        let bias_correction2 = 1.0 - beta2_pow_t;
        let device_config = LambStepConfig {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
            step_size: (config.lr as f64 / bias_correction1) as f32,
            bias_correction2_sqrt: bias_correction2.sqrt() as f32,
        };

        ops.lamb_step_device(
            &mut param_buf,
            &grad_buf,
            LambMoments {
                m: &mut m_buf,
                v: &mut v_buf,
            },
            segment_numels,
            &device_config,
        )
        .unwrap();

        let host_grads: Vec<Tensor<f32>> = {
            let mut offset = 0;
            let mut out = Vec::with_capacity(segment_numels.len());
            for &n in segment_numels {
                out.push(Tensor::new(grad_data[offset..offset + n].to_vec(), &[n]).unwrap());
                offset += n;
            }
            out
        };
        let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
            host_params.iter().zip(host_grads.iter()).collect();
        let host_out = host_lamb.step(&pairs).unwrap();
        host_params = host_out;
    }

    let device_result = mem.download(&param_buf).unwrap();
    let mut offset = 0;
    for (seg_idx, &n) in segment_numels.iter().enumerate() {
        for j in 0..n {
            assert_bits_eq(
                device_result.get(&[offset + j]).unwrap(),
                host_params[seg_idx].get(&[j]).unwrap(),
                &format!("Lamb segment {seg_idx} index {j} (config={config:?})"),
            );
        }
        offset += n;
    }
}

#[test]
fn default_config_multi_segment_matches_host_reference() {
    run_parity(LambConfig::default(), &[3, 5, 1], 6);
}

#[test]
fn weight_decay_matches_host_reference() {
    run_parity(
        LambConfig {
            weight_decay: 0.01,
            ..LambConfig::default()
        },
        &[3, 5, 1],
        6,
    );
}

#[test]
fn all_zero_param_falls_back_to_trust_ratio_one() {
    // `norm_x == 0`（全ゼロ param）は trust ratio を 1.0 へフォール
    // バックする分岐を通る（`lamb.rs` モジュール doc「実装形」節）。
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let config = LambConfig::default();

    let mut param_buf = mem
        .upload(&Tensor::new(vec![0.0f32; 3], &[3]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[3]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[3]).unwrap();
    let grad_tensor = Tensor::new(vec![0.1, 0.2, 0.3], &[3]).unwrap();
    let grad_buf = mem.upload(&grad_tensor).unwrap();

    let beta1_pow_t = config.beta1 as f64;
    let beta2_pow_t = config.beta2 as f64;
    let device_config = LambStepConfig {
        lr: config.lr,
        beta1: config.beta1,
        beta2: config.beta2,
        eps: config.eps,
        weight_decay: config.weight_decay,
        step_size: (config.lr as f64 / (1.0 - beta1_pow_t)) as f32,
        bias_correction2_sqrt: (1.0 - beta2_pow_t).sqrt() as f32,
    };
    ops.lamb_step_device(
        &mut param_buf,
        &grad_buf,
        LambMoments {
            m: &mut m_buf,
            v: &mut v_buf,
        },
        &[3],
        &device_config,
    )
    .unwrap();

    let mut host_lamb = Lamb::new(config).unwrap();
    let host_param = Tensor::new(vec![0.0f32; 3], &[3]).unwrap();
    let host_out = host_lamb.step(&[(&host_param, &grad_tensor)]).unwrap();

    let device_result = mem.download(&param_buf).unwrap();
    for i in 0..3 {
        assert_bits_eq(
            device_result.get(&[i]).unwrap(),
            host_out[0].get(&[i]).unwrap(),
            &format!("all-zero param index {i}"),
        );
    }
}

#[test]
fn non_finite_gradient_is_rejected_without_mutating_state() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let config = LambConfig::default();

    let init = vec![1.0f32, 2.0];
    let mut param_buf = mem
        .upload(&Tensor::new(init.clone(), &[2]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[2]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[2]).unwrap();
    // `g=1e30` は `g*g` が f32 の表現域を超えて overflow し `v` が非有限
    // になる（`lamb.rs` モジュール doc 「非有限 norm の扱い」節参照）。
    let grad_buf = mem
        .upload(&Tensor::new(vec![1e30f32, 0.1], &[2]).unwrap())
        .unwrap();

    let device_config = LambStepConfig {
        lr: config.lr,
        beta1: config.beta1,
        beta2: config.beta2,
        eps: config.eps,
        weight_decay: config.weight_decay,
        step_size: (config.lr as f64 / (1.0 - config.beta1 as f64)) as f32,
        bias_correction2_sqrt: (1.0 - config.beta2 as f64).sqrt() as f32,
    };
    let err = ops
        .lamb_step_device(
            &mut param_buf,
            &grad_buf,
            LambMoments {
                m: &mut m_buf,
                v: &mut v_buf,
            },
            &[2],
            &device_config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));

    // param/m/v はいずれも無変更（no-op 失敗契約）。
    let out = mem.download(&param_buf).unwrap();
    assert_eq!(out.get(&[0]).unwrap(), init[0]);
    assert_eq!(out.get(&[1]).unwrap(), init[1]);
    let m_out = mem.download(&m_buf).unwrap();
    assert_eq!(m_out.get(&[0]).unwrap(), 0.0);
    assert_eq!(m_out.get(&[1]).unwrap(), 0.0);
}

#[test]
fn lamb_step_device_rejects_segment_numels_sum_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![0.1, 0.2, 0.3], &[3]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[3]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[3]).unwrap();
    let config = LambStepConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.0,
        step_size: 1e-3,
        bias_correction2_sqrt: 1.0,
    };
    // 合計 2 != numel 3。
    let err = ops
        .lamb_step_device(
            &mut param_buf,
            &grad_buf,
            LambMoments {
                m: &mut m_buf,
                v: &mut v_buf,
            },
            &[1, 1],
            &config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
}

#[test]
fn lamb_step_device_rejects_segment_numels_overflow() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![0.1, 0.2], &[2]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[2]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = LambStepConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.0,
        step_size: 1e-3,
        bias_correction2_sqrt: 1.0,
    };
    let err = ops
        .lamb_step_device(
            &mut param_buf,
            &grad_buf,
            LambMoments {
                m: &mut m_buf,
                v: &mut v_buf,
            },
            &[usize::MAX, 1],
            &config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)));
}

#[test]
fn lamb_step_device_rejects_shape_mismatch() {
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
    let config = LambStepConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.0,
        step_size: 1e-3,
        bias_correction2_sqrt: 1.0,
    };
    let err = ops
        .lamb_step_device(
            &mut param_buf,
            &grad_buf,
            LambMoments {
                m: &mut m_buf,
                v: &mut v_buf,
            },
            &[2],
            &config,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
