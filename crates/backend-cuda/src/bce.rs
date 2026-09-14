//! 二値交差エントロピー損失（BCE）融合カーネルの起動 API（NVRTC コン
//! パイル・保持・実行。イシュー #1737・親イシュー #1609）。`mse.rs` を
//! 雛形にした同型構成: [`CudaBce::new`] が `CudaDevice` から 3 カーネル
//! （`kernels_bce.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaBce::run_bce_loss_f32`]／[`CudaBce::run_bce_backward_f32`] へ
//! ホスト側スライスを渡すだけで H2D → 起動 → 同期 → D2H を内部で完結
//! できる。`ops.rs::CudaBackendOps::bce_loss`／`bce_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::BceKind;

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_bce::{self, BCE_BLOCK_DIM, BCE_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// [`BceKind`] を CUDA カーネル引数の `int`（`0` = `Probabilities`、
/// `1` = `Logits`）へ変換する（`kernels_bce.rs::BCE_DEVICE_FUNCS` の
/// `kind` 分岐と対応。`BceKind`〈`#[non_exhaustive]`〉の未知 variant は
/// `Logits`（`1`）へ安全側フォールバックする。`backend-cpu::bce` の
/// 未知 variant フォールバック規律と同型）。
fn bce_kind_to_i32(kind: BceKind) -> i32 {
    match kind {
        BceKind::Probabilities => 0,
        kind => {
            debug_assert!(
                matches!(kind, BceKind::Logits),
                "bce::bce_kind_to_i32: 未知の BceKind variant へフォールバックした（契約違反）"
            );
            1
        }
    }
}

/// `numel`（`input`／`target`／`dinput` の要素数）が `i32::MAX` に収まる
/// ことを検証する（`mse.rs::validate_mse_len` と同じ理由）。
pub(crate) fn validate_bce_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("bce_loss numel must fit in i32 (kernel argument type): numel={len}"),
        });
    }
    Ok(())
}

/// `input_len`／`target_len` の一致と `i32::MAX` 上限の両方を検証する
/// （`mse.rs::validate_mse_binary_len` と同じ構成）。
pub(crate) fn validate_bce_binary_len(
    input_len: usize,
    target_len: usize,
) -> Result<(), CudaError> {
    if input_len != target_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("bce length mismatch: input_len={input_len}, target_len={target_len}"),
        });
    }
    validate_bce_len(input_len)
}

/// forward 1 段目（`bce_partial_f32`）の起動ブロック数を決定する
/// （`mse.rs::mse_num_blocks` と同じ式）。
fn bce_num_blocks(numel: u32) -> u32 {
    numel.div_ceil(BCE_BLOCK_DIM).min(BCE_MAX_BLOCKS)
}

/// BCE 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みハンドルを保持する（`mse.rs::CudaMse` と同型構成）。
pub struct CudaBce {
    stream: Arc<CudaStream>,
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    partial_f32: CudaFunction,
    finalize_f32: CudaFunction,
    backward_f32: CudaFunction,
}

impl CudaBce {
    /// `device` 上で BCE 3 カーネルを NVRTC コンパイルし保持するハンドル
    /// を構築する（`mse.rs::CudaMse::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let partial_src = kernels_bce::bce_partial_f32_source();
        let backward_src = kernels_bce::bce_backward_f32_source();

        let partial_ptx = compile_ptx(&partial_src, arch)?;
        let finalize_ptx = compile_ptx(kernels_bce::BCE_FINALIZE_F32, arch)?;
        let backward_ptx = compile_ptx(&backward_src, arch)?;

        let partial_f32 = device
            .context()
            .load_module(partial_ptx)?
            .load_function("bce_partial_f32")?;
        let finalize_f32 = device
            .context()
            .load_module(finalize_ptx)?
            .load_function("bce_finalize_f32")?;
        let backward_f32 = device
            .context()
            .load_module(backward_ptx)?
            .load_function("bce_backward_f32")?;

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

    /// `CudaBce` の driver 呼び出しを CUDA Graph capture 排他へ参加させる
    /// 共通ヘルパー（`mse.rs::CudaMse::with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// forward: `reduction(Σ bce_elem_loss(input[i], target[i], kind))`。
    /// `input.len() == target.len()` は呼び出し元（`ops.rs`）が検証済み
    /// の契約。`numel == 0` はカーネル起動を回避し `0.0` を返す
    /// （`Mean`／`Sum` いずれも空和の契約。`backend-cpu::bce` と同じ）。
    pub fn run_bce_loss_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: BceKind,
        factor: f32,
    ) -> Result<f32, CudaError> {
        validate_bce_binary_len(input.len(), target.len())?;
        let numel = input.len();
        if numel == 0 {
            return Ok(0.0);
        }
        let kind_i = bce_kind_to_i32(kind);

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let target_dev = self.stream.clone_htod(target)?;

            let num_blocks = bce_num_blocks(numel as u32);
            // `partial_f32` は起動する `num_blocks` 個のブロックそれぞれが
            // `partial[blockIdx.x]` を必ず 1 回書く（`kernels_bce.rs`
            // 参照）ため `alloc_uninit_f32` を使える。
            let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (BCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev`／`target_dev` は `numel` 要素の H2D 済み
            // デバイスバッファ、`partial_dev` は `num_blocks` 要素確保済み
            // でカーネルが `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ
            // 書く（`kernels_bce.rs::bce_partial_f32_source` 参照）。
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
                block_dim: (BCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべてが
            // 書き込み済み、`out_dev` は `bce_finalize_f32` が単一
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

    /// backward: `dInput[i] = scale·bce_elem_grad_input(input[i],
    /// target[i], kind)`。`dTarget` は呼び出し元（`ops.rs`／
    /// `fandhe_ai_autodiff::grad::vjp`）がホスト側の逐次 map で計算する
    /// 契約（`backend_ops.rs::BackendOps::bce_loss_backward` doc 参照）
    /// のため、本関数は `dInput` のみを計算する。`numel == 0` は空 `Vec`
    /// を返す。
    ///
    /// **ストリーム順序契約**: `mse.rs::run_mse_backward_f32` と同じく
    /// launch 直後の明示 `synchronize()` を持たず、末尾の `readback`
    /// 呼び出し 1 箇所（D2H＋同期）へ完了待ちを集約する
    /// （`docs/backend-cuda-async-execution-design.md` §2.3・§16）。
    pub fn run_bce_backward_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: BceKind,
        scale: f32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_bce_binary_len(input.len(), target.len())?;
        let numel = input.len();
        if numel == 0 {
            return Ok(Vec::new());
        }
        let kind_i = bce_kind_to_i32(kind);

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let target_dev = self.stream.clone_htod(target)?;
            // `bce_backward_f32` は `if (idx < numel)` ガード内で
            // `dinput[idx]` を必ず埋める（`kernels_bce.rs` 参照）ため
            // `alloc_uninit_f32` を使う。
            let mut dinput_dev = self.allocator.alloc_uninit_f32(numel)?;

            let numel_i = numel as i32;
            let cfg = LaunchConfig {
                grid_dim: (numel.div_ceil(BCE_BLOCK_DIM as usize) as u32, 1, 1),
                block_dim: (BCE_BLOCK_DIM, 1, 1),
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
