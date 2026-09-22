//! `adam`（イシュー #2070）の GPU 非依存な純関数群（shape 検証・
//! `AdamStepKind` 分岐フラグ導出・カーネルのホストモデル）。
//! `crate::unique_model`・`crates/backend-cuda/src/adam.rs` と同じ
//! 「`objc2` 系 FFI に触れないため `cfg(target_os = "macos")` を付けず
//! Linux（本実装環境・CI）でも単体テストが回る」設計判断を踏襲する。
//!
//! # `Device::Metal` を扱わない理由（`gather_scatter_model`／
//! `constant_pad_model` 等の既存 `*_model.rs` と同じ制約）
//!
//! [`fandhe_ai_tensor_core::device::Device::Metal`] variant 自体が
//! `#[cfg(target_os = "macos")]` で条件コンパイルされているため
//! （`device.rs` 参照）、本モジュールは Linux でもコンパイルが通る
//! 必要上 `Device::Metal` に一切触れない。デバイス一致検査
//! （`param.device() != Device::Metal` 等）は macOS 限定の `crate::ops`
//! （`adam_step_device_impl`）側の責務とし、本モジュールは shape 検証・
//! `AdamStepKind` 分岐フラグ導出・カーネル逐語ホストモデルという
//! デバイスに依存しない部分のみを担う（`crate::sgd_step_device_impl`
//! が同様に device 検査を自身で行い、`gather_scatter_model` 等の
//! 検証ヘルパーは shape のみを検査する既存の役割分担と同じ）。
//!
//! # ホストモデルの目的
//!
//! [`adam_step_host_model`] は `shaders/adam.metal::adam_step_f32` の
//! 演算列を逐語再現する `f32` モデルである（`fma` = `f32::mul_add`、
//! 括弧の結合順序も同一）。Linux で `CpuBackendOps::adam_step_device`
//! と `to_bits()` 完全一致を検証することで、「カーネルに写像した演算列
//! が CPU 参照実装と等価」であることを実機なしでロックする（実機での
//! bit 一致主張はこのモデルと MSL ソースの対応 + 実機テストの組で担う。
//! `docs/perf/logs/adam-device-step-metal-2070/README.md` 参照）。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{AdamStepConfig, AdamStepKind, ShapeError};

/// `AdamStepKind` の分岐をホストで事前評価した bool 相当フラグ
/// （`crate::adam::AdamKernelParams::from_config` がここへ写す。
/// `backend-cuda::adam::CudaAdam::run` が呼び出し元でホストへ展開する
/// のと同じ設計）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdamKernelFlags {
    /// `AdamStepKind::Decoupled` なら真（`p_eff = p * decay_factor` を
    /// 無条件適用）。
    pub decoupled: bool,
    /// `AdamStepKind::Coupled && weight_decay != 0.0` なら真（`g_eff`
    /// へ coupled L2 decay を適用）。
    pub use_coupled_wd: bool,
}

/// `param`／`grad`／`m`／`v` の shape が一致することを検証する
/// （`gather_scatter_model::validate_gather_launch` と同じ「デバイス
/// には触れず shape のみを検査する」責務分担）。CPU 参照実装
/// （`backend-cpu::ops::CpuBackendOps::adam_step_device`）・CUDA 側
/// （`backend-cuda::ops::CudaBackendOps::adam_step_device`）と同じ検査
/// 順序（grad → m → v）で `ShapeError::ShapeMismatch` を返す。
pub fn validate_adam_step_shapes(
    param_shape: &[usize],
    grad_shape: &[usize],
    m_shape: &[usize],
    v_shape: &[usize],
) -> Result<(), ShapeError> {
    if param_shape != grad_shape {
        return Err(ShapeError::ShapeMismatch {
            lhs: param_shape.to_vec(),
            rhs: grad_shape.to_vec(),
        });
    }
    if param_shape != m_shape {
        return Err(ShapeError::ShapeMismatch {
            lhs: param_shape.to_vec(),
            rhs: m_shape.to_vec(),
        });
    }
    if param_shape != v_shape {
        return Err(ShapeError::ShapeMismatch {
            lhs: param_shape.to_vec(),
            rhs: v_shape.to_vec(),
        });
    }
    Ok(())
}

/// `AdamStepConfig::kind` を [`AdamKernelFlags`] へ導出する。
///
/// `#[non_exhaustive]` の `AdamStepKind` は将来 variant が増えうるため、
/// 未知の variant は fail-closed に拒否する（`.claude/rules/security.md`
/// A08。`backend-cpu::ops::adam_step_device`／`backend-cuda::ops::
/// adam_step_device` と同じ「どの要素も更新前」契約。呼び出し元
/// `crate::ops::MetalBackendOps::adam_step_device_impl` はこの検査を
/// device・shape 検証の後・実際のバッファ downcast より前に行う）。
pub fn adam_kernel_flags(
    kind: AdamStepKind,
    weight_decay: f32,
) -> Result<AdamKernelFlags, BackendError> {
    match kind {
        AdamStepKind::Coupled => Ok(AdamKernelFlags {
            decoupled: false,
            use_coupled_wd: weight_decay != 0.0,
        }),
        AdamStepKind::Decoupled => Ok(AdamKernelFlags {
            decoupled: true,
            use_coupled_wd: false,
        }),
        _ => Err(BackendError::Unsupported(
            "adam_step_device: unknown AdamStepKind variant".into(),
        )),
    }
}

/// `shaders/adam.metal::adam_step_f32` の演算列を逐語再現する `f32`
/// モデル（`crates/backend-cpu/src/ops.rs::adam_step_device` の
/// ループ本体と同一の演算列・結合順序。`fma` = `f32::mul_add`）。
///
/// `param`／`m`／`v` を in-place で更新する（カーネルと同じ「独立した
/// `out` バッファを持たない」契約）。`param`／`grad`／`m`／`v` は同じ
/// 長さを要求する（呼び出し元が [`validate_adam_step_shapes`] で検証
/// 済みの前提。本関数自体は境界検査を持たない——テスト専用のホスト
/// モデルであり、MSL カーネル側の `if (idx < numel)` 境界検査〈REQ-8〉
/// はこの関数の対象外）。
#[allow(clippy::too_many_arguments)]
pub fn adam_step_host_model(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    config: &AdamStepConfig,
    flags: AdamKernelFlags,
) {
    for j in 0..param.len() {
        let p = param[j];
        let g = grad[j];

        let g_eff = if flags.use_coupled_wd {
            f32::mul_add(config.weight_decay, p, g)
        } else {
            g
        };

        let p_eff = if flags.decoupled {
            p * config.decay_factor
        } else {
            p
        };

        let m_new = f32::mul_add(config.beta1, m[j], (1.0 - config.beta1) * g_eff);
        let v_new = f32::mul_add(config.beta2, v[j], ((1.0 - config.beta2) * g_eff) * g_eff);
        m[j] = m_new;
        v[j] = v_new;

        let denom = v_new.sqrt() / config.bias_correction2_sqrt + config.eps;
        param[j] = p_eff - config.step_size * m_new / denom;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_adam_step_shapes_accepts_matching_shapes() {
        assert!(validate_adam_step_shapes(&[4], &[4], &[4], &[4]).is_ok());
    }

    #[test]
    fn validate_adam_step_shapes_rejects_grad_mismatch() {
        let err = validate_adam_step_shapes(&[4], &[3], &[4], &[4]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn validate_adam_step_shapes_rejects_m_mismatch() {
        let err = validate_adam_step_shapes(&[4], &[4], &[3], &[4]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn validate_adam_step_shapes_rejects_v_mismatch() {
        let err = validate_adam_step_shapes(&[4], &[4], &[4], &[3]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn adam_kernel_flags_coupled_wd_zero_disables_coupled_wd() {
        let flags = adam_kernel_flags(AdamStepKind::Coupled, 0.0).unwrap();
        assert_eq!(
            flags,
            AdamKernelFlags {
                decoupled: false,
                use_coupled_wd: false,
            }
        );
    }

    #[test]
    fn adam_kernel_flags_coupled_wd_nonzero_enables_coupled_wd() {
        let flags = adam_kernel_flags(AdamStepKind::Coupled, 0.01).unwrap();
        assert_eq!(
            flags,
            AdamKernelFlags {
                decoupled: false,
                use_coupled_wd: true,
            }
        );
    }

    #[test]
    fn adam_kernel_flags_decoupled_ignores_weight_decay_value() {
        let flags_zero = adam_kernel_flags(AdamStepKind::Decoupled, 0.0).unwrap();
        let flags_nonzero = adam_kernel_flags(AdamStepKind::Decoupled, 0.01).unwrap();
        let expected = AdamKernelFlags {
            decoupled: true,
            use_coupled_wd: false,
        };
        assert_eq!(flags_zero, expected);
        assert_eq!(flags_nonzero, expected);
    }

    /// `adam_step_host_model`（本モジュール）と `CpuBackendOps::
    /// adam_step_device`（`backend-cpu`）が bit 完全一致することを
    /// 100 step・Coupled／Decoupled × weight_decay=0／≠0 の全組合せで
    /// 検証する（`beta^t` は `f64` 逐次積で追跡し、ホスト `Adam`／
    /// `AdamW::step` と同じ `f32` キャストで `step_size`／
    /// `bias_correction2_sqrt` を導出する）。カーネルへ写像した演算列が
    /// CPU 参照実装と等価であることを実機なしでロックする（本モジュール
    /// doc「ホストモデルの目的」参照）。
    #[test]
    fn adam_step_host_model_bit_matches_cpu_reference_across_100_steps() {
        use fandhe_ai_backend_cpu::CpuBackendOps;
        use fandhe_ai_tensor_core::{BackendOps, Tensor};

        let cpu = CpuBackendOps::new();
        let mem = cpu
            .memory_ops()
            .expect("CpuBackendOps must implement MemoryOps");

        for kind in [AdamStepKind::Coupled, AdamStepKind::Decoupled] {
            for weight_decay in [0.0f32, 0.01f32] {
                let init: Vec<f32> = (0..17).map(|i| (i as f32) * 0.1 - 0.8).collect();
                let grad_seed: Vec<f32> =
                    (0..17).map(|i| ((i as f32) * 0.37).sin() * 0.05).collect();

                let mut host_param = init.clone();
                let mut host_m = vec![0.0f32; init.len()];
                let mut host_v = vec![0.0f32; init.len()];

                let mut cpu_param = mem
                    .upload(&Tensor::new(init.clone(), &[init.len()]).unwrap())
                    .unwrap();
                let mut cpu_m = mem.alloc_zeroed(&[init.len()]).unwrap();
                let mut cpu_v = mem.alloc_zeroed(&[init.len()]).unwrap();

                let lr = 0.001f64;
                let beta1 = 0.9f64;
                let beta2 = 0.999f64;
                let mut beta1_pow_t = 1.0f64;
                let mut beta2_pow_t = 1.0f64;

                for step in 0..100u32 {
                    beta1_pow_t *= beta1;
                    beta2_pow_t *= beta2;
                    let step_size = (lr / (1.0 - beta1_pow_t)) as f32;
                    let bias_correction2_sqrt = ((1.0 - beta2_pow_t).sqrt()) as f32;
                    let decay_factor = (1.0 - lr * weight_decay as f64) as f32;

                    let grad: Vec<f32> = grad_seed
                        .iter()
                        .map(|g| g * (1.0 + step as f32 * 0.001))
                        .collect();

                    let config = AdamStepConfig {
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        weight_decay,
                        decay_factor,
                        step_size,
                        bias_correction2_sqrt,
                        kind,
                    };
                    let flags = adam_kernel_flags(kind, weight_decay).unwrap();

                    adam_step_host_model(
                        &mut host_param,
                        &grad,
                        &mut host_m,
                        &mut host_v,
                        &config,
                        flags,
                    );

                    let grad_tensor = Tensor::new(grad, &[init.len()]).unwrap();
                    let cpu_grad = mem.upload(&grad_tensor).unwrap();
                    cpu.adam_step_device(
                        &mut cpu_param,
                        &cpu_grad,
                        &mut cpu_m,
                        &mut cpu_v,
                        &config,
                    )
                    .unwrap();
                }

                let cpu_param_host = mem.download(&cpu_param).unwrap();
                let cpu_param_slice = cpu_param_host
                    .as_slice()
                    .expect("downloaded CPU tensor must be contiguous");
                for (h, c) in host_param.iter().zip(cpu_param_slice.iter()) {
                    assert_eq!(
                        h.to_bits(),
                        c.to_bits(),
                        "kind={kind:?} weight_decay={weight_decay}: host model と CPU 参照実装が bit 不一致"
                    );
                }
            }
        }
    }
}
