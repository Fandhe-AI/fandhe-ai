//! 負対数尤度損失（`NLLLoss`）融合カーネルの起動 API（NVRTC コンパイル・
//! 保持・実行。イシュー #1738・親イシュー #1609「損失関数の拡張」）。
//! `mse.rs` を雛形にした同型構成: [`CudaNll::new`] が `CudaDevice` から
//! 3 カーネル（`kernels_nll.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaNll::run_nll_loss_f32`]／[`CudaNll::run_nll_backward_f32`] へ
//! ホスト側スライスを渡すだけで H2D → 起動 → 同期 → D2H を内部で完結
//! できる。`ops.rs::CudaBackendOps::nll_loss`／`nll_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_nll::{self, NLL_BLOCK_DIM, NLL_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `input`/`class_dim` から導出される `(outer, num_classes, inner)`
/// レイアウト（`backend-cpu::nll::NllLayout` と同型。呼び出し元
/// `ops.rs::CudaBackendOps::nll_loss`／`nll_loss_backward` が
/// `input.shape()`／`class_dim` から構築する）。
#[derive(Debug, Clone, Copy)]
pub struct NllLayout {
    pub outer: usize,
    pub num_classes: usize,
    pub inner: usize,
}

impl NllLayout {
    fn n_samples(&self) -> usize {
        self.outer * self.inner
    }

    fn numel(&self) -> usize {
        self.outer * self.num_classes * self.inner
    }
}

/// `n_samples`／`numel` が `i32::MAX` に収まることを検証する
/// （`mse.rs::validate_mse_len` と同じ理由）。
pub(crate) fn validate_nll_layout(layout: NllLayout) -> Result<(), CudaError> {
    if layout.n_samples() > i32::MAX as usize || layout.numel() > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "nll_loss dims must fit in i32 (kernel argument type): n_samples={}, numel={}",
                layout.n_samples(),
                layout.numel()
            ),
        });
    }
    Ok(())
}

/// forward 1 段目（`nll_partial_f32`）の起動ブロック数を決定する
/// （`mse.rs::mse_num_blocks` と同じ式。`n_samples` 基準）。
fn nll_num_blocks(n_samples: u32) -> u32 {
    n_samples.div_ceil(NLL_BLOCK_DIM).min(NLL_MAX_BLOCKS)
}

/// NLL 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みハンドルを保持する（`mse.rs::CudaMse` と同型構成）。
pub struct CudaNll {
    stream: Arc<CudaStream>,
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    partial_f32: CudaFunction,
    finalize_f32: CudaFunction,
    backward_f32: CudaFunction,
}

impl CudaNll {
    /// `device` 上で NLL 3 カーネルを NVRTC コンパイルし保持するハンドル
    /// を構築する（`mse.rs::CudaMse::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let partial_ptx = compile_ptx(kernels_nll::NLL_PARTIAL_F32, arch)?;
        let finalize_ptx = compile_ptx(kernels_nll::NLL_FINALIZE_F32, arch)?;
        let backward_ptx = compile_ptx(kernels_nll::NLL_BACKWARD_F32, arch)?;

        let partial_f32 = device
            .context()
            .load_module(partial_ptx)?
            .load_function("nll_partial_f32")?;
        let finalize_f32 = device
            .context()
            .load_module(finalize_ptx)?
            .load_function("nll_finalize_f32")?;
        let backward_f32 = device
            .context()
            .load_module(backward_ptx)?
            .load_function("nll_backward_f32")?;

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

    /// `CudaNll` の driver 呼び出しを CUDA Graph capture 排他へ参加させる
    /// 共通ヘルパー（`mse.rs::CudaMse::with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// forward: `reduction(Σ_s −input[(o·C+t_s)·inner+i])`（サンプル
    /// `s = o·inner+i`）。`input.len() == layout.numel()`／
    /// `targets.len() == layout.n_samples()` は呼び出し元（`ops.rs`）が
    /// 検証済みの契約。`n_samples == 0` はカーネル起動を回避し `0.0`
    /// を返す（`Mean`／`Sum` いずれも空和の契約。`backend-cpu::nll` と
    /// 同じ）。
    pub fn run_nll_loss_f32(
        &self,
        input: &[f32],
        targets: &[i32],
        layout: NllLayout,
        factor: f32,
    ) -> Result<f32, CudaError> {
        validate_nll_layout(layout)?;
        let n_samples = layout.n_samples();
        if n_samples == 0 {
            return Ok(0.0);
        }

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let targets_dev = self.stream.clone_htod(targets)?;

            let num_blocks = nll_num_blocks(n_samples as u32);
            // `partial_f32` は起動する `num_blocks` 個のブロックそれぞれが
            // `partial[blockIdx.x]` を必ず 1 回書く（`kernels_nll.rs`
            // 参照）ため `alloc_uninit_f32` を使える。
            let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

            let outer_i = layout.outer as i32;
            let num_classes_i = layout.num_classes as i32;
            let inner_i = layout.inner as i32;
            let n_samples_i = n_samples as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (NLL_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev` は `layout.numel()` 要素、`targets_dev`
            // は `n_samples` 要素の H2D 済みデバイスバッファ。`partial_dev`
            // は `num_blocks` 要素確保済みでカーネルが `blockIdx.x`
            // （`0..num_blocks`）ごとに 1 回だけ書く（`kernels_nll.rs`
            // 参照）。カーネル内の grid-stride ループは `idx < n_samples`
            // を維持し、`targets[idx]`（`0 <= t < num_classes` はホスト側
            // 事前検証済み）から導出する `input_idx` は `layout.numel()`
            // 未満（REQ-8）ため OOB 読み出しは起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.partial_f32)
                    .arg(&input_dev)
                    .arg(&targets_dev)
                    .arg(&mut partial_dev.as_view_mut())
                    .arg(&outer_i)
                    .arg(&num_classes_i)
                    .arg(&inner_i)
                    .arg(&n_samples_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (NLL_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべてが
            // 書き込み済み、`out_dev` は `nll_finalize_f32` が単一
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

    /// backward: `dInput[(o·C+t_s)·inner+i] = −scale`（ターゲット位置
    /// 以外は `0.0`）。`targets` は非追跡のため `dInput` のみを返す
    /// 契約（`backend_ops.rs::BackendOps::nll_loss_backward` doc
    /// 参照）。`n_samples == 0` は `layout.numel()` 要素すべてゼロの
    /// `Vec` を返す。
    ///
    /// **ストリーム順序契約**: `mse.rs::run_mse_backward_f32` と同じく
    /// launch 直後の明示 `synchronize()` を持たず、末尾の `readback`
    /// 呼び出し 1 箇所（D2H＋同期）へ完了待ちを集約する
    /// （`docs/backend-cuda-async-execution-design.md` §2.3・§16）。
    pub fn run_nll_backward_f32(
        &self,
        targets: &[i32],
        layout: NllLayout,
        scale: f32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_nll_layout(layout)?;
        let numel = layout.numel();
        let n_samples = layout.n_samples();

        self.with_driver_call(|| {
            // `layout.numel() == 0` の場合も `alloc_zeroed_f32(0)` は
            // 空バッファとして正しく振る舞う（下記カーネル起動は
            // `n_samples == 0` のとき起動しない）。
            let mut dinput_dev = self.allocator.alloc_zeroed_f32(numel)?;
            if n_samples == 0 {
                return readback(&self.stream, &dinput_dev.as_view());
            }

            let targets_dev = self.stream.clone_htod(targets)?;

            let outer_i = layout.outer as i32;
            let num_classes_i = layout.num_classes as i32;
            let inner_i = layout.inner as i32;
            let n_samples_i = n_samples as i32;
            let cfg = LaunchConfig {
                grid_dim: (n_samples.div_ceil(NLL_BLOCK_DIM as usize) as u32, 1, 1),
                block_dim: (NLL_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `dinput_dev` は `layout.numel()` 要素・ゼロ初期化
            // 済み。`targets_dev` は `n_samples` 要素。カーネル内の手動
            // 境界チェック（`if (idx < n_samples)`。REQ-8）と、各サンプル
            // が一意な `input_idx`（`kernels_nll.rs` 冒頭「backward の
            // 書き込み一意性」参照）へのみ書き込む設計により、異なる
            // スレッドが同一アドレスへ競合書き込みすることはない。
            unsafe {
                self.stream
                    .launch_builder(&self.backward_f32)
                    .arg(&targets_dev)
                    .arg(&mut dinput_dev.as_view_mut())
                    .arg(&outer_i)
                    .arg(&num_classes_i)
                    .arg(&inner_i)
                    .arg(&n_samples_i)
                    .arg(&scale)
                    .launch(cfg)?;
            }

            readback(&self.stream, &dinput_dev.as_view())
        })
    }
}
