//! 平均二乗誤差（MSE）融合カーネルの起動 API（NVRTC コンパイル・保持・
//! 実行。イシュー #1045・親イシュー #1043）。
//!
//! `elementwise.rs::CudaElementwise`・`sgd.rs::CudaSgd` と同じ構成方針を
//! 踏襲する: [`CudaMse::new`] が `CudaDevice` から 3 カーネル
//! （`kernels_mse.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaMse::run_mse_loss_f32`]／[`CudaMse::run_mse_backward_f32`] へ
//! ホスト側スライスを渡すだけで H2D → 起動 → 同期 → D2H を内部で完結
//! できる。`ops.rs::CudaBackendOps::mse_loss`／`mse_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_mse::{self, MSE_BLOCK_DIM, MSE_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `numel`（`pred`／`target`／`dpred` の要素数）が `i32::MAX` に収まる
/// ことを検証する（`elementwise.rs::validate_elementwise_len` と同じ
/// 理由。カーネル引数 `int numel` は C の 32bit 符号付き整数のため）。
pub(crate) fn validate_mse_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("mse_loss numel must fit in i32 (kernel argument type): numel={len}"),
        });
    }
    Ok(())
}

/// forward 1 段目（`mse_partial_f32`）の起動ブロック数を決定する。
/// `min(ceil_div(numel, MSE_BLOCK_DIM), MSE_MAX_BLOCKS)`（`kernels_mse.rs`
/// 冒頭コメント「forward の 2 段構成」の契約）。`numel > 0` を前提とする
/// （`numel == 0` は呼び出し元がカーネル起動自体を回避する）。
fn mse_num_blocks(numel: u32) -> u32 {
    numel.div_ceil(MSE_BLOCK_DIM).min(MSE_MAX_BLOCKS)
}

/// MSE 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みハンドルを保持する。
pub struct CudaMse {
    stream: Arc<CudaStream>,
    allocator: Arc<CudaAllocator>,
    partial_f32: CudaFunction,
    finalize_f32: CudaFunction,
    backward_f32: CudaFunction,
}

impl CudaMse {
    /// `device` 上で MSE 3 カーネルを NVRTC コンパイルし保持するハンドル
    /// を構築する（`elementwise.rs::CudaElementwise::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let partial_ptx = compile_ptx(kernels_mse::MSE_PARTIAL_F32, arch)?;
        let finalize_ptx = compile_ptx(kernels_mse::MSE_FINALIZE_F32, arch)?;
        let backward_ptx = compile_ptx(kernels_mse::MSE_BACKWARD_F32, arch)?;

        let partial_f32 = device
            .context()
            .load_module(partial_ptx)?
            .load_function("mse_partial_f32")?;
        let finalize_f32 = device
            .context()
            .load_module(finalize_ptx)?
            .load_function("mse_finalize_f32")?;
        let backward_f32 = device
            .context()
            .load_module(backward_ptx)?
            .load_function("mse_backward_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            allocator,
            partial_f32,
            finalize_f32,
            backward_f32,
        })
    }

    /// forward: `reduction(Σ(pred[i]−target[i])²)`。`pred.len() ==
    /// target.len()` は呼び出し元（`ops.rs`）が検証済みの契約。
    /// `numel == 0` はカーネル起動を回避し `0.0` を返す（`Mean`／`Sum`
    /// いずれも空和の契約。`backend-cpu::mse` と同じ）。
    pub fn run_mse_loss_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        factor: f32,
    ) -> Result<f32, CudaError> {
        assert_eq!(pred.len(), target.len(), "length mismatch (pred vs target)");
        validate_mse_len(pred.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(0.0);
        }

        let pred_dev = self.stream.clone_htod(pred)?;
        let target_dev = self.stream.clone_htod(target)?;

        let num_blocks = mse_num_blocks(numel as u32);
        // `partial_f32` は起動する `num_blocks` 個のブロックそれぞれが
        // `partial[blockIdx.x]` を必ず 1 回書く（`kernels_mse.rs` 冒頭
        // コメント参照）ため `alloc_uninit_f32` を使える（`pool.rs`
        // `alloc_uninit_f32` doc の適用条件）。
        let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

        let numel_i = numel as i32;
        let partial_cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (MSE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `pred_dev`／`target_dev` は `numel` 要素の H2D 済み
        // デバイスバッファ、`partial_dev` は `num_blocks` 要素確保済みで
        // カーネルが `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ書く
        // （`kernels_mse.rs::MSE_PARTIAL_F32` 参照）。カーネル内の
        // grid-stride ループは `idx < numel` を維持する（REQ-8）ため
        // OOB 読み出しは起きない。
        unsafe {
            self.stream
                .launch_builder(&self.partial_f32)
                .arg(&pred_dev)
                .arg(&target_dev)
                .arg(&mut partial_dev.as_view_mut())
                .arg(&numel_i)
                .launch(partial_cfg)?;
        }

        let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
        let num_partials_i = num_blocks as i32;
        let finalize_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (MSE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべてが書き
        // 込み済み、`out_dev` は `mse_finalize_f32` が単一ブロックの
        // lane 0 で `out[0]` を必ず 1 回書く（`kernels_mse.rs` 参照）ため
        // `alloc_uninit_f32(1)` を使える。`num_partials_i` は起動時に
        // 確保した `partial_dev` の長さと同一の値を渡す（本関数内の
        // 単一の `num_blocks` 由来。呼び出し元がずらせない）。
        unsafe {
            self.stream
                .launch_builder(&self.finalize_f32)
                .arg(&partial_dev.as_view())
                .arg(&mut out_dev.as_view_mut())
                .arg(&num_partials_i)
                .arg(&factor)
                .launch(finalize_cfg)?;
        }

        let host: Vec<f32> = readback(&self.stream, &out_dev.as_view())?;
        Ok(host.first().copied().unwrap_or(0.0))
    }

    /// backward: `dPred[i] = scale·(pred[i]−target[i])`。`dTarget` は
    /// 呼び出し元（`ops.rs`／`fandhe_ai_autodiff::grad::vjp`）がホスト側
    /// で符号反転して得る契約（`backend_ops.rs::BackendOps::
    /// mse_loss_backward` doc 参照）のため、本関数は `dPred` のみを
    /// 計算する。`numel == 0` は空 `Vec` を返す。
    pub fn run_mse_backward_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, CudaError> {
        assert_eq!(pred.len(), target.len(), "length mismatch (pred vs target)");
        validate_mse_len(pred.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let pred_dev = self.stream.clone_htod(pred)?;
        let target_dev = self.stream.clone_htod(target)?;
        // イシュー #1020: `mse_backward_f32` は `if (idx < numel)` ガード
        // 内で `dpred[idx]` を必ず埋める（`kernels_mse.rs` 参照）ため
        // `alloc_uninit_f32` を使う（`elementwise.rs::run_unary` と同じ
        // 適用条件）。
        let mut dpred_dev = self.allocator.alloc_uninit_f32(numel)?;

        let numel_i = numel as i32;
        let cfg = LaunchConfig {
            grid_dim: (numel.div_ceil(MSE_BLOCK_DIM as usize) as u32, 1, 1),
            block_dim: (MSE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `pred_dev`／`target_dev`／`dpred_dev` はいずれも
        // `numel` 要素のデバイスバッファであり、カーネル内の手動境界
        // チェック（`if (idx < numel)`。REQ-8）と合わせて OOB 読み書き
        // が起きない根拠とする。グリッド次元は `div_ceil` で numel を
        // 包含するよう構築しており、末尾ブロックの余剰スレッドは
        // カーネル内境界チェックで弾かれる。
        unsafe {
            self.stream
                .launch_builder(&self.backward_f32)
                .arg(&pred_dev)
                .arg(&target_dev)
                .arg(&mut dpred_dev.as_view_mut())
                .arg(&numel_i)
                .arg(&scale)
                .launch(cfg)?;
        }

        readback(&self.stream, &dpred_dev.as_view())
    }
}
