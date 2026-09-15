//! Kullback-Leibler ダイバージェンス損失（`KLDivLoss`）融合カーネルの
//! 起動 API（NVRTC コンパイル・保持・実行。イシュー #1738・親イシュー
//! #1609「損失関数の拡張」）。`bce.rs`（イシュー #1737）を雛形にした
//! 同型構成: [`CudaKlDiv::new`] が `CudaDevice` から 3 カーネル
//! （`kernels_kl_div.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaKlDiv::run_kl_div_loss_f32`]／
//! [`CudaKlDiv::run_kl_div_backward_f32`] へホスト側スライスを渡すだけ
//! で H2D → 起動 → 同期 → D2H を内部で完結できる。
//! `ops.rs::CudaBackendOps::kl_div_loss`／`kl_div_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::KlDivTarget;

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_kl_div::{self, KL_DIV_BLOCK_DIM, KL_DIV_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// [`KlDivTarget`] を CUDA カーネル引数の `int`（`0` =
/// `Probabilities`、`1` = `LogProbabilities`）へ変換する
/// （`bce.rs::bce_kind_to_i32` と同型。`KlDivTarget`〈`#[non_exhaustive]`〉
/// の未知 variant は `LogProbabilities`（`1`）へ安全側フォールバック
/// する）。
fn kl_div_kind_to_i32(kind: KlDivTarget) -> i32 {
    match kind {
        KlDivTarget::Probabilities => 0,
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "kl_div::kl_div_kind_to_i32: 未知の KlDivTarget variant へフォールバックした\
                 （契約違反）"
            );
            1
        }
    }
}

/// `numel`（`input`／`target`／`dinput` の要素数）が `i32::MAX` に収まる
/// ことを検証する（`bce.rs::validate_bce_len` と同じ理由）。
pub(crate) fn validate_kl_div_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "kl_div_loss numel must fit in i32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

/// `input_len`／`target_len` の一致と `i32::MAX` 上限の両方を検証する
/// （`bce.rs::validate_bce_binary_len` と同じ構成）。
pub(crate) fn validate_kl_div_binary_len(
    input_len: usize,
    target_len: usize,
) -> Result<(), CudaError> {
    if input_len != target_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "kl_div length mismatch: input_len={input_len}, target_len={target_len}"
            ),
        });
    }
    validate_kl_div_len(input_len)
}

/// forward 1 段目（`kl_div_partial_f32`）の起動ブロック数を決定する
/// （`bce.rs::bce_num_blocks` と同じ式）。
fn kl_div_num_blocks(numel: u32) -> u32 {
    numel.div_ceil(KL_DIV_BLOCK_DIM).min(KL_DIV_MAX_BLOCKS)
}

/// KLDiv 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みハンドルを保持する（`bce.rs::CudaBce` と同型構成）。
pub struct CudaKlDiv {
    stream: Arc<CudaStream>,
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    partial_f32: CudaFunction,
    finalize_f32: CudaFunction,
    backward_f32: CudaFunction,
}

impl CudaKlDiv {
    /// `device` 上で KLDiv 3 カーネルを NVRTC コンパイルし保持するハンドル
    /// を構築する（`bce.rs::CudaBce::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let partial_src = kernels_kl_div::kl_div_partial_f32_source();
        let backward_src = kernels_kl_div::kl_div_backward_f32_source();

        let partial_ptx = compile_ptx(&partial_src, arch)?;
        let finalize_ptx = compile_ptx(kernels_kl_div::KL_DIV_FINALIZE_F32, arch)?;
        let backward_ptx = compile_ptx(&backward_src, arch)?;

        let partial_f32 = device
            .context()
            .load_module(partial_ptx)?
            .load_function("kl_div_partial_f32")?;
        let finalize_f32 = device
            .context()
            .load_module(finalize_ptx)?
            .load_function("kl_div_finalize_f32")?;
        let backward_f32 = device
            .context()
            .load_module(backward_ptx)?
            .load_function("kl_div_backward_f32")?;

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

    /// `CudaKlDiv` の driver 呼び出しを CUDA Graph capture 排他へ参加
    /// させる共通ヘルパー（`bce.rs::CudaBce::with_driver_call` と同じ
    /// 設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// forward: `reduction(Σ kl_div_elem_loss(input[i], target[i],
    /// kind))`。`input.len() == target.len()` は呼び出し元（`ops.rs`）
    /// が検証済みの契約。`numel == 0` はカーネル起動を回避し `0.0` を
    /// 返す（`Mean`／`Sum` いずれも空和の契約。`backend-cpu::kl_div` と
    /// 同じ）。
    pub fn run_kl_div_loss_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: KlDivTarget,
        factor: f32,
    ) -> Result<f32, CudaError> {
        validate_kl_div_binary_len(input.len(), target.len())?;
        let numel = input.len();
        if numel == 0 {
            return Ok(0.0);
        }
        let kind_i = kl_div_kind_to_i32(kind);

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let target_dev = self.stream.clone_htod(target)?;

            let num_blocks = kl_div_num_blocks(numel as u32);
            // `partial_f32` は起動する `num_blocks` 個のブロックそれぞれが
            // `partial[blockIdx.x]` を必ず 1 回書く（`kernels_kl_div.rs`
            // 参照）ため `alloc_uninit_f32` を使える。
            let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (KL_DIV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev`／`target_dev` は `numel` 要素の H2D 済み
            // デバイスバッファ、`partial_dev` は `num_blocks` 要素確保済み
            // でカーネルが `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ
            // 書く（`kernels_kl_div.rs::kl_div_partial_f32_source` 参照）。
            // カーネル内の grid-stride ループは `idx < numel` を維持する
            // （REQ-8）ため OOB 読み出しは起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.partial_f32)
                    .arg(&input_dev)
                    .arg(&target_dev)
                    .arg(&mut partial_dev.as_view_mut())
                    .arg(&numel_i)
                    .arg(&kind_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (KL_DIV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべてが
            // 書き込み済み、`out_dev` は `kl_div_finalize_f32` が単一
            // ブロックの lane 0 で `out[0]` を必ず 1 回書く。
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

    /// backward: `dInput[i] = scale·kl_div_elem_grad_input(input[i],
    /// target[i], kind)`。`dTarget` は呼び出し元（`ops.rs`／
    /// `fandhe_ai_autodiff::grad::vjp`）がホスト側の逐次 map で計算する
    /// 契約（`backend_ops.rs::BackendOps::kl_div_loss_backward` doc
    /// 参照）のため、本関数は `dInput` のみを計算する。`numel == 0` は
    /// 空 `Vec` を返す。
    ///
    /// **ストリーム順序契約**: `bce.rs::run_bce_backward_f32` と同じく
    /// launch 直後の明示 `synchronize()` を持たず、末尾の `readback`
    /// 呼び出し 1 箇所（D2H＋同期）へ完了待ちを集約する
    /// （`docs/backend-cuda-async-execution-design.md` §2.3・§16）。
    pub fn run_kl_div_backward_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: KlDivTarget,
        scale: f32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_kl_div_binary_len(input.len(), target.len())?;
        let numel = input.len();
        if numel == 0 {
            return Ok(Vec::new());
        }
        let kind_i = kl_div_kind_to_i32(kind);

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let target_dev = self.stream.clone_htod(target)?;
            // `kl_div_backward_f32` は `if (idx < numel)` ガード内で
            // `dinput[idx]` を必ず埋める（`kernels_kl_div.rs` 参照）ため
            // `alloc_uninit_f32` を使う。
            let mut dinput_dev = self.allocator.alloc_uninit_f32(numel)?;

            let numel_i = numel as i32;
            let cfg = LaunchConfig {
                grid_dim: (numel.div_ceil(KL_DIV_BLOCK_DIM as usize) as u32, 1, 1),
                block_dim: (KL_DIV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev`／`target_dev`／`dinput_dev` はいずれも
            // `numel` 要素のデバイスバッファであり、カーネル内の手動
            // 境界チェック（`if (idx < numel)`。REQ-8）と合わせて OOB
            // 読み書きが起きない根拠とする。
            unsafe {
                self.stream
                    .launch_builder(&self.backward_f32)
                    .arg(&input_dev)
                    .arg(&target_dev)
                    .arg(&mut dinput_dev.as_view_mut())
                    .arg(&numel_i)
                    .arg(&kind_i)
                    .arg(&scale)
                    .launch(cfg)?;
            }

            readback(&self.stream, &dinput_dev.as_view())
        })
    }
}
