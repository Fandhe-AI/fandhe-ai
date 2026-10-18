//! デバイス常駐パラメータの in-place 更新（SGD・Adam/AdamW）の ROCm 実行入口
//! （`backend-cuda::{sgd,adam}` の移植。イシュー #3440・分解案 R9）。
//!
//! 役割: HIP/HIPRTC をロードして `kernels_optim` のソースをコンパイルし、**既にデバイス上にある**
//! `param`・`velocity`・`m`・`v` を直接書き換える。`RocmBackendOps` の `sgd_step_device`／
//! `adam_step_device`（`ops/optim.rs`）から呼ばれる。ハンドル型の判別・shape 検査・engine 分岐は
//! 呼び出し側の責務で、本ファイルは「型付きバッファとカーネル引数ビュー」だけを受ける。
//!
//! - 他の ROCm 実行型（`RocmElementwise` 等）と異なり、入出力が同一バッファの in-place 更新で、
//!   書き込み先は `&mut TypedDeviceBuffer<f32>` として型で表す。クレート外へは公開しない
//!   （常駐バッファの型 `TypedDeviceBuffer` が `pub(crate)` のため）。
//! - 起動は `KernelHost::run`（呼び出しごとに `synchronize`。Metal と同じ同期実行で、CUDA の非同期
//!   投入とは異なる。trait doc が許す差異）。CUDA Graph の記録計数は持ち込まない（graph capture は
//!   被覆計画 §5 で `Unsupported`）。
//! - momentum 無効時の `velocity` 引数は grad のポインタをダミーで渡す（`use_momentum == 0` の分岐では
//!   カーネルが読まず書かない。`Arg::Ptr` は生のポインタ値で借用の問題もない。CUDA と同じ手当て）。
//! - 新しい `unsafe` はない。監査済みの `KernelHost::run`（K1）の呼び出し元が 2 か所増えるだけ。
//!
//! f64 アキュムレータ契約は要素ごとの更新で縮約がないため該当しない（`kernels_optim.rs` 参照）。

use fandhe_ai_tensor_core::{AdamStepConfig, AdamStepKind, SgdStepConfig};

use crate::elementwise::fit_i32;
use crate::error::RocmError;
use crate::kernels_optim::{ADAM_STEP_SPEC, SGD_STEP_SPEC, render_optim_source};
use crate::launch::{Arg, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP_SGD: &str = "sgd_step";
const OP_ADAM: &str = "adam_step";

/// SGD カーネルへ渡すハイパーパラメータのビュー。
///
/// `use_momentum` は `config.momentum != 0.0` から決める（CPU 参照と同じ。CUDA の
/// `velocity.is_some()` 基準とは異なる。`kernels_optim.rs` 参照）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SgdKernelParams {
    pub(crate) lr: f32,
    pub(crate) momentum: f32,
    pub(crate) dampening: f32,
    pub(crate) weight_decay: f32,
    pub(crate) nesterov: bool,
    pub(crate) is_first_step: bool,
    pub(crate) use_momentum: bool,
}

impl SgdKernelParams {
    pub(crate) fn from_config(c: &SgdStepConfig) -> Self {
        Self {
            lr: c.lr,
            momentum: c.momentum,
            dampening: c.dampening,
            weight_decay: c.weight_decay,
            nesterov: c.nesterov,
            is_first_step: c.is_first_step,
            use_momentum: c.momentum != 0.0,
        }
    }
}

/// Adam カーネルへ渡すハイパーパラメータのビュー（`kind` はホストで 2 つのフラグへ事前評価する）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AdamKernelParams {
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    pub(crate) eps: f32,
    pub(crate) weight_decay: f32,
    pub(crate) decay_factor: f32,
    pub(crate) step_size: f32,
    pub(crate) bias_correction2_sqrt: f32,
    /// `AdamStepKind::Decoupled`（`p_eff = p * decay_factor` を無条件に適用）。
    pub(crate) decoupled: bool,
    /// `Coupled` かつ `weight_decay != 0.0`（`g_eff` へ coupled L2 decay を適用）。
    pub(crate) use_coupled_wd: bool,
}

impl AdamKernelParams {
    /// 未知の `AdamStepKind`（`#[non_exhaustive]`）は `None`。呼び出し側が `Unsupported` にする。
    pub(crate) fn from_config(c: &AdamStepConfig) -> Option<Self> {
        let (decoupled, use_coupled_wd) = match c.kind {
            AdamStepKind::Coupled => (false, c.weight_decay != 0.0),
            AdamStepKind::Decoupled => (true, false),
            _ => return None,
        };
        Some(Self {
            beta1: c.beta1,
            beta2: c.beta2,
            eps: c.eps,
            weight_decay: c.weight_decay,
            decay_factor: c.decay_factor,
            step_size: c.step_size,
            bias_correction2_sqrt: c.bias_correction2_sqrt,
            decoupled,
            use_coupled_wd,
        })
    }
}

fn flag(b: bool) -> i32 {
    i32::from(b)
}

fn check_len(
    op: &'static str,
    name: &str,
    expected: usize,
    actual: usize,
) -> Result<(), RocmError> {
    if expected != actual {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("{name} の長さ不一致: 期待 {expected}, 実際 {actual}"),
        });
    }
    Ok(())
}

/// コンパイル済みの SGD・Adam カーネルを保持し、常駐バッファを in-place で更新する。
pub(crate) struct RocmOptim {
    host: KernelHost,
}

impl RocmOptim {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub(crate) fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_optim_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// SGD 1 ステップを `param`（と momentum 有効時の `velocity`）へ in-place で適用する。
    ///
    /// 長さ 0 は呼び出し側が HIP に触れず処理する契約（ここでは `InvalidOpArgs`）。
    pub(crate) fn sgd_step_resident(
        &self,
        param: &mut TypedDeviceBuffer<f32>,
        grad: &TypedDeviceBuffer<f32>,
        velocity: Option<&mut TypedDeviceBuffer<f32>>,
        p: &SgdKernelParams,
    ) -> Result<(), RocmError> {
        let n = param.len();
        check_len(OP_SGD, "grad", n, grad.len())?;
        if p.use_momentum {
            let v = velocity
                .as_deref()
                .ok_or_else(|| RocmError::InvalidOpArgs {
                    op: OP_SGD,
                    detail: "momentum 有効なのに velocity がありません".into(),
                })?;
            check_len(OP_SGD, "velocity", n, v.len())?;
        }
        let n_i = fit_i32(OP_SGD, n, "numel")?;
        let _guard = self.host.select()?;
        // momentum 無効時はカーネルが velocity を読み書きしないので grad をダミーで渡す。
        let velocity_ptr = match (p.use_momentum, velocity.as_deref()) {
            (true, Some(v)) => v.as_device_ptr(),
            _ => grad.as_device_ptr(),
        };
        let args = [
            Arg::Ptr(param.as_device_ptr()),
            Arg::Ptr(grad.as_device_ptr()),
            Arg::Ptr(velocity_ptr),
            Arg::I32(n_i),
            Arg::F32(p.lr),
            Arg::F32(p.momentum),
            Arg::F32(p.dampening),
            Arg::F32(p.weight_decay),
            Arg::I32(flag(p.nesterov)),
            Arg::I32(flag(p.is_first_step)),
            Arg::I32(flag(p.use_momentum)),
        ];
        self.host.run(OP_SGD, &SGD_STEP_SPEC, n, &args)
    }

    /// Adam／AdamW 1 ステップを `param`・`m`・`v` へ in-place で適用する。
    pub(crate) fn adam_step_resident(
        &self,
        param: &mut TypedDeviceBuffer<f32>,
        grad: &TypedDeviceBuffer<f32>,
        m: &mut TypedDeviceBuffer<f32>,
        v: &mut TypedDeviceBuffer<f32>,
        p: &AdamKernelParams,
    ) -> Result<(), RocmError> {
        let n = param.len();
        check_len(OP_ADAM, "grad", n, grad.len())?;
        check_len(OP_ADAM, "m", n, m.len())?;
        check_len(OP_ADAM, "v", n, v.len())?;
        let n_i = fit_i32(OP_ADAM, n, "numel")?;
        let _guard = self.host.select()?;
        let args = [
            Arg::Ptr(param.as_device_ptr()),
            Arg::Ptr(grad.as_device_ptr()),
            Arg::Ptr(m.as_device_ptr()),
            Arg::Ptr(v.as_device_ptr()),
            Arg::I32(n_i),
            Arg::F32(p.beta1),
            Arg::F32(p.beta2),
            Arg::F32(p.eps),
            Arg::F32(p.weight_decay),
            Arg::F32(p.decay_factor),
            Arg::F32(p.step_size),
            Arg::F32(p.bias_correction2_sqrt),
            Arg::I32(flag(p.decoupled)),
            Arg::I32(flag(p.use_coupled_wd)),
        ];
        self.host.run(OP_ADAM, &ADAM_STEP_SPEC, n, &args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sgd_cfg(momentum: f32) -> SgdStepConfig {
        SgdStepConfig {
            lr: 0.1,
            momentum,
            dampening: 0.1,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        }
    }

    #[test]
    fn use_momentum_follows_config_not_velocity() {
        assert!(!SgdKernelParams::from_config(&sgd_cfg(0.0)).use_momentum);
        assert!(SgdKernelParams::from_config(&sgd_cfg(0.9)).use_momentum);
    }

    #[test]
    fn adam_flags_are_pre_evaluated() {
        let base = AdamStepConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            decay_factor: 0.99,
            step_size: 0.001,
            bias_correction2_sqrt: 0.5,
            kind: AdamStepKind::Coupled,
        };
        let c = AdamKernelParams::from_config(&base).unwrap();
        assert!(!c.decoupled && c.use_coupled_wd);
        let c0 = AdamKernelParams::from_config(&AdamStepConfig {
            weight_decay: 0.0,
            ..base
        })
        .unwrap();
        assert!(!c0.decoupled && !c0.use_coupled_wd);
        let d = AdamKernelParams::from_config(&AdamStepConfig {
            kind: AdamStepKind::Decoupled,
            ..base
        })
        .unwrap();
        assert!(d.decoupled && !d.use_coupled_wd);
    }

    #[test]
    fn new_without_rocm_is_typed_error_not_panic() {
        if crate::is_available() {
            return;
        }
        assert!(RocmOptim::new(GfxTarget::Gfx942).is_err());
    }
}
