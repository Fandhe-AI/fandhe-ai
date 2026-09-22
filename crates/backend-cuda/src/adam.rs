//! デバイス上パラメータ更新（Adam・AdamW in-place）の起動 API（NVRTC
//! コンパイル・保持・実行。イシュー #2069・`docs/
//! device-resident-update-design.md`「Adam／AdamW の常駐 step 結線」
//! 節）。
//!
//! `sgd.rs::CudaSgd` と同じ構成方針を踏襲する: [`CudaAdam::new`] が
//! `CudaDevice` から 1 カーネルを NVRTC コンパイルして保持し、以降は
//! [`CudaAdam::run`] へ配置非依存の `crate::memory::CudaArgMut`／
//! `CudaArg`（イシュー #1352）を渡すだけで GPU 実行できる。`param`／
//! `m`／`v` はホスト常駐往復なしでデバイス上に直接読み書きする（本
//! イシューの主目的）。
//!
//! `ops.rs::CudaBackendOps::adam_step_device` から
//! `BackendOps::adam_step_device` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_adam::{self, ADAM_BLOCK_DIM};
use crate::memory::{CudaArg, CudaArgMut};
use crate::nvrtc::compile_ptx;

/// Adam カーネル 1 個あたりのブロック次元（1 次元、`ADAM_BLOCK_DIM`
/// 幅）。
const ADAM_BLOCK: (u32, u32, u32) = (ADAM_BLOCK_DIM, 1, 1);

/// 長さが `i32::MAX` に収まることを検証する（`sgd.rs::
/// validate_sgd_len` と同じ理由。カーネル引数 `int numel` は C の
/// 32bit 符号付き整数のため）。
pub(crate) fn validate_adam_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "adam_step_device numel must fit in i32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

fn adam_launch_config(numel: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (numel.div_ceil(ADAM_BLOCK.0), 1, 1),
        block_dim: ADAM_BLOCK,
        shared_mem_bytes: 0,
    }
}

/// `adam_step_f32`（`kernels_adam.rs`）のコンパイル済みハンドルを保持
/// する。
pub struct CudaAdam {
    stream: Arc<CudaStream>,
    adam_step_f32: cudarc::driver::CudaFunction,
}

/// `AdamStepConfig`（`tensor-core::backend_ops`）と同一のハイパー
/// パラメータをカーネル起動用にまとめたもの（`CudaAdam::run` の引数を
/// 減らすための内部用ビュー）。`decoupled`／`use_coupled_wd` は
/// `AdamStepKind` の分岐をホストで事前評価した bool 相当フラグ
/// （`kernels_adam.rs::ADAM_STEP_F32` doc コメント参照）。
pub struct AdamKernelParams {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub decay_factor: f32,
    pub step_size: f32,
    pub bias_correction2_sqrt: f32,
    pub decoupled: bool,
    pub use_coupled_wd: bool,
}

impl CudaAdam {
    /// `device` 上で `adam_step_f32` カーネルを NVRTC コンパイルし保持
    /// するハンドルを構築する（`sgd.rs::CudaSgd::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(kernels_adam::ADAM_STEP_F32, arch)?;
        let adam_step_f32 = device
            .context()
            .load_module(ptx)?
            .load_function("adam_step_f32")?;
        Ok(Self {
            stream: device.stream().clone(),
            adam_step_f32,
        })
    }

    /// Adam・AdamW 1 ステップを in-place で実行する（H2D／D2H なし）。
    ///
    /// 非同期投入契約（イシュー #1013）: 本関数はカーネルをストリームへ
    /// 投入するのみで完了を待たない（`synchronize()` を呼ばない）。
    /// 呼び出し元が `param`／`m`／`v` の内容をホストで読む場合は、
    /// `MemoryOps::download`（`memory.rs::readback` 経由で同期する）を
    /// 別途呼ぶ責務を負う。
    ///
    /// `param`／`grad`／`m`／`v` は同じ `numel` を要求する（呼び出し元
    /// `ops.rs::CudaBackendOps::adam_step_device` が shape 検証済みの
    /// バッファを渡す契約）。
    ///
    /// `numel == 0` の場合はカーネル起動自体を回避する（`sgd.rs::
    /// CudaSgd::run` と同じ理由）。
    ///
    /// Adam は SGD と異なり 4 バッファすべてが常に実体（`velocity` の
    /// ような opt-in・ダミーエイリアス引数は不要）。
    pub(crate) fn run(
        &self,
        mut param: CudaArgMut<'_>,
        grad: CudaArg<'_>,
        mut m: CudaArgMut<'_>,
        mut v: CudaArgMut<'_>,
        params: &AdamKernelParams,
    ) -> Result<(), CudaError> {
        let numel = param.len();
        validate_adam_len(numel)?;
        if numel == 0 {
            return Ok(());
        }
        if grad.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "adam_step_device length mismatch: param={numel}, grad={}",
                    grad.len()
                ),
            });
        }
        if m.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "adam_step_device length mismatch: param={numel}, m={}",
                    m.len()
                ),
            });
        }
        if v.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "adam_step_device length mismatch: param={numel}, v={}",
                    v.len()
                ),
            });
        }

        let numel_i = numel as i32;
        let beta1 = params.beta1;
        let beta2 = params.beta2;
        let eps = params.eps;
        let weight_decay = params.weight_decay;
        let decay_factor = params.decay_factor;
        let step_size = params.step_size;
        let bias_correction2_sqrt = params.bias_correction2_sqrt;
        let kind_decoupled = if params.decoupled { 1i32 } else { 0i32 };
        let use_coupled_wd = if params.use_coupled_wd { 1i32 } else { 0i32 };

        // SAFETY: `param`／`grad`／`m`／`v` はいずれも呼び出し元がこの
        // `numel` に対応する長さで確保済みのデバイスバッファであり、
        // カーネル内の手動境界チェック（`if (idx < numel)`。
        // `kernels_adam.rs` 参照、REQ-8）と合わせて OOB 読み書きが起き
        // ない根拠とする。グリッド次元は `div_ceil` で numel を包含する
        // よう構築しており（`adam_launch_config`）、末尾ブロックの余剰
        // スレッドはカーネル内境界チェックで弾かれる。4 バッファは
        // `ops.rs::CudaBackendOps::adam_step_device` が別個の
        // `DeviceBuffer` から取得した独立ストレージであり、SGD の
        // `velocity` ダミーエイリアスのような借用の使い回しは発生し
        // ない。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.adam_step_f32);
            param.push(&mut builder);
            grad.push(&mut builder);
            m.push(&mut builder);
            v.push(&mut builder);
            builder
                .arg(&numel_i)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&weight_decay)
                .arg(&decay_factor)
                .arg(&step_size)
                .arg(&bias_correction2_sqrt)
                .arg(&kind_decoupled)
                .arg(&use_coupled_wd)
                .launch(adam_launch_config(numel as u32))?;
        }
        // ここでは `synchronize()` を呼ばない（イシュー #1013。`sgd.rs::
        // CudaSgd::run` と同じ非同期投入契約）。Adam は CUDA Graph
        // capture 対象外（イシュー #1959 設計節。`SGD_KERNEL_LAUNCH_COUNT`
        // 相当の計上も行わない）。
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_adam_len_accepts_boundary_and_rejects_overflow() {
        assert!(validate_adam_len(0).is_ok());
        assert!(validate_adam_len(i32::MAX as usize).is_ok());
        assert!(validate_adam_len(i32::MAX as usize + 1).is_err());
    }
}
