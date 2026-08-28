//! デバイス上パラメータ更新（SGD in-place）の起動 API（NVRTC コンパイル・
//! 保持・実行。イシュー #935・`docs/device-resident-update-design.md`
//! §3.2）。
//!
//! `elementwise.rs::CudaElementwise` と同じ構成方針を踏襲する:
//! `CudaSgd::new` が `CudaDevice` から 1 カーネルを NVRTC コンパイルして
//! 保持し、以降は [`CudaSgd::run`] へ `CudaSlice<f32>` を渡すだけで GPU
//! 実行できる。`elementwise.rs` と異なり、本モジュールはホスト常駐
//! `&[f32]` を受け取らず**デバイス上に既に存在する `CudaSlice<f32>`
//! を直接読み書きする**（H2D／D2H を挟まない。本イシューの主目的である
//! 「param のホスト往復排除」の実体）。
//!
//! `ops.rs::CudaBackendOps::sgd_step_device` から
//! `BackendOps::sgd_step_device` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_sgd::{self, SGD_BLOCK_DIM};
use crate::nvrtc::compile_ptx;

/// SGD カーネル 1 個あたりのブロック次元（1 次元、`SGD_BLOCK_DIM` 幅）。
const SGD_BLOCK: (u32, u32, u32) = (SGD_BLOCK_DIM, 1, 1);

/// 長さが `i32::MAX` に収まることを検証する（`elementwise.rs::
/// validate_elementwise_len` と同じ理由。カーネル引数 `int numel` は C の
/// 32bit 符号付き整数のため）。
pub(crate) fn validate_sgd_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "sgd_step_device numel must fit in i32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

fn sgd_launch_config(numel: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (numel.div_ceil(SGD_BLOCK.0), 1, 1),
        block_dim: SGD_BLOCK,
        shared_mem_bytes: 0,
    }
}

/// `sgd_step_f32`（`kernels_sgd.rs`）のコンパイル済みハンドルを保持する。
pub struct CudaSgd {
    stream: Arc<CudaStream>,
    sgd_step_f32: cudarc::driver::CudaFunction,
}

/// `SgdStepConfig`（`tensor-core::backend_ops`）と同一のハイパー
/// パラメータをカーネル起動用にまとめたもの（`CudaSgd::run` の引数を
/// 減らすための内部用ビュー）。
pub struct SgdKernelParams {
    pub lr: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
    pub is_first_step: bool,
}

impl CudaSgd {
    /// `device` 上で `sgd_step_f32` カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する（`elementwise.rs::CudaElementwise::new` と同一
    /// 手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(kernels_sgd::SGD_STEP_F32, arch)?;
        let sgd_step_f32 = device
            .context()
            .load_module(ptx)?
            .load_function("sgd_step_f32")?;
        Ok(Self {
            stream: device.stream().clone(),
            sgd_step_f32,
        })
    }

    /// SGD 1 ステップを in-place で実行する（H2D／D2H なし）。
    ///
    /// `param`／`grad` は同じ `numel` を要求する（呼び出し元
    /// `ops.rs::CudaBackendOps::sgd_step_device` が shape 検証済みの
    /// バッファを渡す契約）。`velocity` は `params.momentum != 0.0` の
    /// 場合のみ `Some` を要求する（`None` の場合は `use_momentum = 0` で
    /// 起動し、カーネル側は `velocity` を読み書きしないため、その場合は
    /// `param` 自身をダミーとして渡す。`kernels_sgd.rs` 冒頭コメント
    /// 「velocity 引数」参照）。
    ///
    /// `numel == 0` の場合はカーネル起動自体を回避する（`elementwise.rs::
    /// CudaElementwise::run_binary` と同じ理由）。
    pub fn run(
        &self,
        param: &mut CudaSlice<f32>,
        grad: &CudaSlice<f32>,
        velocity: Option<&mut CudaSlice<f32>>,
        params: &SgdKernelParams,
    ) -> Result<(), CudaError> {
        let numel = param.len();
        validate_sgd_len(numel)?;
        if numel == 0 {
            return Ok(());
        }
        if grad.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "sgd_step_device length mismatch: param={numel}, grad={}",
                    grad.len()
                ),
            });
        }

        let use_momentum = if velocity.is_some() { 1i32 } else { 0i32 };
        let numel_i = numel as i32;
        let lr = params.lr;
        let momentum = params.momentum;
        let dampening = params.dampening;
        let weight_decay = params.weight_decay;
        let nesterov = if params.nesterov { 1i32 } else { 0i32 };
        let is_first_step = if params.is_first_step { 1i32 } else { 0i32 };

        // SAFETY: `param`／`grad`（存在すれば `velocity`）はいずれも呼び
        // 出し元がこの `numel` に対応する長さで確保済みのデバイスバッファ
        // であり、カーネル内の手動境界チェック（`if (idx < numel)`。
        // `kernels_sgd.rs` 参照、REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。`use_momentum == 0` の場合は `velocity` 引数として
        // `param` 自身の別名エイリアスを渡すが、カーネル側は
        // `use_momentum` が真の場合しか `velocity` を読み書きしないため
        // エイリアシングによる未定義動作は発生しない。グリッド次元は
        // `div_ceil` で numel を包含するよう構築しており
        // （`sgd_launch_config`）、末尾ブロックの余剰スレッドはカーネル内
        // 境界チェックで弾かれる。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.sgd_step_f32);
            // `&mut *param` で毎回リボローする（`param` は `&mut
            // CudaSlice<f32>` であり `Copy` ではないため、`use_momentum ==
            // 0` の分岐で `velocity` 引数としても再度使う場合は
            // 明示的にリボローする必要がある。各 `arg()` 呼び出しは
            // 引数のデバイスポインタのみを取り出して即座に手放すため
            // 〈`elementwise.rs::run_binary` の `.arg(&mut out_dev)` と
            // 同様〉、逐次のリボローは重ならず借用検査を通る）。
            builder.arg(&mut *param);
            builder.arg(grad);
            match velocity {
                Some(v) => {
                    builder.arg(v);
                }
                None => {
                    // `param` は既に上で可変借用済みのため再借用できない
                    // （`cudarc::driver::LaunchArgs::arg` は起動〈`launch()`〉
                    // まで借用を保持する）。`grad`（共有参照。`Copy`）を
                    // 未使用ダミーとして再利用する（カーネル側は
                    // `use_momentum == 0` の場合この引数を読み書きしない。
                    // `kernels_sgd.rs` 冒頭コメント「velocity 引数」参照）。
                    builder.arg(grad);
                }
            }
            builder
                .arg(&numel_i)
                .arg(&lr)
                .arg(&momentum)
                .arg(&dampening)
                .arg(&weight_decay)
                .arg(&nesterov)
                .arg(&is_first_step)
                .arg(&use_momentum)
                .launch(sgd_launch_config(numel as u32))?;
        }
        self.stream.synchronize()?;
        Ok(())
    }
}
