//! Huber／SmoothL1 損失融合カーネルの起動 API（NVRTC コンパイル・保持・
//! 実行。イシュー #1739）。`mse.rs`（イシュー #1045）と同じ構成方針を
//! 踏襲する: [`CudaHuber::new`] が `CudaDevice` から 3 カーネル
//! （`kernels_huber.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaHuber::run_huber_loss_f32`]／[`CudaHuber::run_huber_backward_f32`]
//! へホスト側スライスを渡すだけで H2D → 起動 → 同期 → D2H を内部で
//! 完結できる。`ops.rs::CudaBackendOps::huber_loss`／
//! `huber_loss_backward` から `BackendOps` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::HuberKind;

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_huber::{self, HUBER_BLOCK_DIM, HUBER_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `HuberKind` を CUDA カーネル引数（`int kind`。0=Huber, 1=SmoothL1）へ
/// 変換する（`kernels_huber.rs` の `huber_elem_loss`／`huber_elem_grad`
/// 契約と対応。`#[non_exhaustive]` な `HuberKind` の未知 variant は
/// `0`〈Huber〉へ安全側フォールバックする——`eval::huber_elem_loss` と
/// 同型の判断。カーネル側も `kind != 1` をすべて `Huber` として扱う
/// ため、この変換とカーネル側の分岐は整合する）。
fn huber_kind_to_i32(kind: HuberKind) -> i32 {
    match kind {
        HuberKind::SmoothL1 => 1,
        _ => 0,
    }
}

/// `numel`（`pred`／`target`／`dpred` の要素数）が `i32::MAX` に収まる
/// ことを検証する（`mse.rs::validate_mse_len` と同じ理由）。
pub(crate) fn validate_huber_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("huber_loss numel must fit in i32 (kernel argument type): numel={len}"),
        });
    }
    Ok(())
}

/// `pred_len`／`target_len` の一致と `i32::MAX` 上限の両方を検証する
/// （`mse.rs::validate_mse_binary_len` と同じ構成・同じ理由）。
pub(crate) fn validate_huber_binary_len(
    pred_len: usize,
    target_len: usize,
) -> Result<(), CudaError> {
    if pred_len != target_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("huber length mismatch: pred_len={pred_len}, target_len={target_len}"),
        });
    }
    validate_huber_len(pred_len)
}

/// forward 1 段目（`huber_partial_f32`）の起動ブロック数を決定する
/// （`mse.rs::mse_num_blocks` と同じ式・同じ理由）。
fn huber_num_blocks(numel: u32) -> u32 {
    numel.div_ceil(HUBER_BLOCK_DIM).min(HUBER_MAX_BLOCKS)
}

/// Huber 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みハンドルを保持する（`mse.rs::CudaMse` と同型）。
pub struct CudaHuber {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（`CudaMse::ordinal` と同じ役割。
    /// `Self::with_driver_call` が `context_cache::with_driver_call` を
    /// 呼ぶ際のキーとして使う）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    partial_f32: CudaFunction,
    finalize_f32: CudaFunction,
    backward_f32: CudaFunction,
}

impl CudaHuber {
    /// `device` 上で Huber 3 カーネルを NVRTC コンパイルし保持するハンドル
    /// を構築する（`mse.rs::CudaMse::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let partial_ptx = compile_ptx(kernels_huber::HUBER_PARTIAL_F32, arch)?;
        let finalize_ptx = compile_ptx(kernels_huber::HUBER_FINALIZE_F32, arch)?;
        let backward_ptx = compile_ptx(kernels_huber::HUBER_BACKWARD_F32, arch)?;

        let partial_f32 = device
            .context()
            .load_module(partial_ptx)?
            .load_function("huber_partial_f32")?;
        let finalize_f32 = device
            .context()
            .load_module(finalize_ptx)?
            .load_function("huber_finalize_f32")?;
        let backward_f32 = device
            .context()
            .load_module(backward_ptx)?
            .load_function("huber_backward_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            partial_f32,
            finalize_f32,
            backward_f32,
        })
    }

    /// `CudaHuber` の driver 呼び出しを CUDA Graph capture 排他へ参加
    /// させる共通ヘルパー（`mse.rs::CudaMse::with_driver_call` と同じ
    /// 設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// forward: `reduction(Σ l(pred[i]−target[i]))`（`l` は
    /// `kernels_huber.rs::huber_elem_loss`）。`pred.len() ==
    /// target.len()` は呼び出し元（`ops.rs`）が検証済みの契約。
    /// `numel == 0` はカーネル起動を回避し `0.0` を返す（`Mean`／`Sum`
    /// いずれも空和の契約。`backend-cpu::huber` と同じ）。
    pub fn run_huber_loss_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        kind: HuberKind,
        delta: f32,
        factor: f32,
    ) -> Result<f32, CudaError> {
        validate_huber_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(0.0);
        }
        let kind_i = huber_kind_to_i32(kind);

        // `mse.rs::run_mse_loss_f32` と同じ理由で `Self::with_driver_call`
        // で本体全体を capture 排他へ参加させる。
        self.with_driver_call(|| {
            let pred_dev = self.stream.clone_htod(pred)?;
            let target_dev = self.stream.clone_htod(target)?;

            let num_blocks = huber_num_blocks(numel as u32);
            // `partial_f32` は起動する `num_blocks` 個のブロックそれぞれが
            // `partial[blockIdx.x]` を必ず 1 回書く（`kernels_huber.rs`
            // 参照）ため `alloc_uninit_f32` を使える（`mse.rs` と同じ
            // 適用条件）。
            let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (HUBER_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `pred_dev`／`target_dev` は `numel` 要素の H2D 済み
            // デバイスバッファ、`partial_dev` は `num_blocks` 要素確保済みで
            // カーネルが `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ書く
            // （`kernels_huber.rs::HUBER_PARTIAL_F32` 参照）。カーネル内の
            // grid-stride ループは `idx < numel` を維持する（REQ-8）ため
            // OOB 読み出しは起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.partial_f32)
                    .arg(&pred_dev)
                    .arg(&target_dev)
                    .arg(&mut partial_dev.as_view_mut())
                    .arg(&numel_i)
                    .arg(&kind_i)
                    .arg(&delta)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (HUBER_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべてが書き
            // 込み済み、`out_dev` は `huber_finalize_f32` が単一ブロックの
            // lane 0 で `out[0]` を必ず 1 回書く（`kernels_huber.rs` 参照）
            // ため `alloc_uninit_f32(1)` を使える。`num_partials_i` は起動
            // 時に確保した `partial_dev` の長さと同一の値を渡す。
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
        })
    }

    /// backward: `dPred[i] = scale·grad_elem(pred[i]−target[i])`
    /// （`grad_elem` は `kernels_huber.rs::huber_elem_grad`）。`dTarget`
    /// は呼び出し元（`ops.rs`／`fandhe_ai_autodiff::grad::vjp`）がホスト
    /// 側で符号反転して得る契約（`backend_ops.rs::BackendOps::
    /// huber_loss_backward` doc 参照）のため、本関数は `dPred` のみを
    /// 計算する。`numel == 0` は空 `Vec` を返す。
    ///
    /// **ストリーム順序契約**: `mse.rs::run_mse_backward_f32`（イシュー
    /// #1692）と同じ理由で、明示 `synchronize()` を持たず末尾の
    /// `readback` 呼び出し 1 箇所へ完了待ちを集約する
    /// （`docs/backend-cuda-async-execution-design.md` §2.3・§16）。
    pub fn run_huber_backward_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        kind: HuberKind,
        delta: f32,
        scale: f32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_huber_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(Vec::new());
        }
        let kind_i = huber_kind_to_i32(kind);

        // `run_huber_loss_f32` と同じ理由で `Self::with_driver_call` へ
        // 参加させる。
        self.with_driver_call(|| {
            let pred_dev = self.stream.clone_htod(pred)?;
            let target_dev = self.stream.clone_htod(target)?;
            // `huber_backward_f32` は `if (idx < numel)` ガード内で
            // `dpred[idx]` を必ず埋める（`kernels_huber.rs` 参照）ため
            // `alloc_uninit_f32` を使う（`mse.rs` と同じ適用条件）。
            let mut dpred_dev = self.allocator.alloc_uninit_f32(numel)?;

            let numel_i = numel as i32;
            let cfg = LaunchConfig {
                grid_dim: (numel.div_ceil(HUBER_BLOCK_DIM as usize) as u32, 1, 1),
                block_dim: (HUBER_BLOCK_DIM, 1, 1),
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
                    .arg(&kind_i)
                    .arg(&delta)
                    .arg(&scale)
                    .launch(cfg)?;
            }

            readback(&self.stream, &dpred_dev.as_view())
        })
    }
}
