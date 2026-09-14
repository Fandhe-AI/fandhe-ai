//! 最近傍リサンプリングの起動 API（NVRTC コンパイル・保持・実行。
//! イシュー #1757）。
//!
//! `gather_scatter.rs::CudaGatherScatter`／`constant_pad.rs::
//! CudaConstantPad`（#1756 相当テンプレート）と同じ構成方針を踏襲する:
//! [`CudaInterpolate::new`] が `CudaDevice` から `interpolate_nearest_f32`
//! カーネル（`kernels_interpolate.rs`）を NVRTC コンパイルして保持し、
//! 以降は [`CudaInterpolate::run_nearest_f32`] へホスト側スライスを
//! 渡すだけで GPU 実行できる。`ops.rs::CudaBackendOps::interpolate`
//! から `BackendOps` の実装として呼ばれる。
//!
//! **shape 検証の責務分担**（`gather_scatter.rs` モジュール doc・
//! `.claude/rules/security.md` A08 と同じ二重検査方針）: 呼び出し元
//! `ops.rs` が [`fandhe_ai_tensor_core::interpolate_out_shape`] で
//! `input.shape()`／`size` の rank 整合を再検査してから本モジュールへ
//! 委譲する契約のため、本モジュール自身は `rank`／`spatial_start` の
//! 整合を `debug_assert!` でのみ確認する（release ビルドでは無効化
//! される内部契約検証）。一方で `input` の**要素数**（`checked_numel`
//! で導出した期待長との一致）は release ビルドでも独立に検証する
//! （`constant_pad.rs` モジュール doc と同じ理由）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::bilinear_scale;

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gather_scatter::{row_major_strides_i32, shape_to_i32};
use crate::kernels_interpolate::{self, INTERPOLATE_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int`
/// は C の 32bit 符号付き整数のため。`constant_pad.rs::
/// validate_i32_bound` と同じ理由の複製）。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::InvalidInterpolateShape {
        detail: format!(
            "interpolate dimension must fit in i32 (kernel argument type): {name}={value}"
        ),
    })
}

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`constant_pad.rs::checked_numel` と同型。クレート内で
/// `pub(crate)` 共有できないため複製する）。
fn checked_numel(shape: &[usize]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CudaError::InvalidInterpolateShape {
            detail: "checked_numel: element count overflow".to_string(),
        })
}

/// interpolate カーネルのコンパイル済みハンドルを保持する。
pub struct CudaInterpolate {
    stream: Arc<CudaStream>,
    /// `gather_scatter.rs::CudaGatherScatter::ordinal` と同じ役割
    /// （`Self::with_driver_call` が `context_cache::with_driver_call`
    /// を呼ぶ際のキー）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    nearest_f32: CudaFunction,
    /// bilinear カーネル（イシュー #1762）。`nearest_f32` と同じ
    /// `CudaInterpolate::new` で同時にコンパイル・保持する。
    bilinear_f32: CudaFunction,
}

impl CudaInterpolate {
    /// `device` 上で interpolate カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する（`constant_pad.rs::CudaConstantPad::new` と
    /// 同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(kernels_interpolate::INTERPOLATE_NEAREST_F32, arch)?;
        let nearest_f32 = device
            .context()
            .load_module(ptx)?
            .load_function("interpolate_nearest_f32")?;

        // `gather_scatter.rs::CudaGatherScatter::new` と同じ方式で、
        // カーネルごとに独立した NVRTC コンパイル・モジュールロードを
        // 行う（イシュー #1762）。
        let bilinear_ptx = compile_ptx(kernels_interpolate::INTERPOLATE_BILINEAR_F32, arch)?;
        let bilinear_f32 = device
            .context()
            .load_module(bilinear_ptx)?
            .load_function("interpolate_bilinear_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            nearest_f32,
            bilinear_f32,
        })
    }

    /// `CudaInterpolate` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`gather_scatter.rs::CudaGatherScatter::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `torch.nn.functional.interpolate(mode='nearest')` 相当
    /// （`kernels_interpolate.rs` モジュール doc 参照）。`out_shape` は
    /// 呼び出し元（`fandhe_ai_tensor_core::interpolate_out_shape`）が
    /// 検査・確定済みの出力 shape。`spatial_start` は
    /// `out_shape.len() - size.len()`（空間軸の開始位置）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_nearest_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        spatial_start: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let rank = in_shape.len();
        debug_assert_eq!(
            out_shape.len(),
            rank,
            "run_nearest_f32: caller must pre-validate rank match via interpolate_out_shape"
        );
        debug_assert!(
            spatial_start <= rank,
            "run_nearest_f32: caller must pre-validate spatial_start <= rank"
        );

        let numel_out = checked_numel(out_shape)?;
        if numel_out == 0 {
            // 空出力早期リターン（`constant_pad.rs::run_pad_f32` と同じ
            // 理由）。
            return Ok(Vec::new());
        }

        let numel_in = checked_numel(in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidInterpolateShape {
                detail: format!(
                    "interpolate: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let out_shape_i32 = shape_to_i32(out_shape)?;
        let in_shape_i32 = shape_to_i32(in_shape)?;
        let in_strides_i32 = row_major_strides_i32(in_shape)?;
        let rank_i = validate_i32_bound(rank, "rank")?;
        let spatial_start_i = validate_i32_bound(spatial_start, "spatial_start")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let out_shape_dev = self.stream.clone_htod(&out_shape_i32)?;
            let in_shape_dev = self.stream.clone_htod(&in_shape_i32)?;
            let in_strides_dev = self.stream.clone_htod(&in_strides_i32)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(INTERPOLATE_BLOCK_DIM), 1, 1),
                block_dim: (INTERPOLATE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ
            // （直上の長さ検証でホストスライスと一致することを確認
            // 済み）。`out_dev` は `numel_out` 要素確保済みでカーネルは
            // `idx < numel`（REQ-8）を維持したまま各出力要素を 1 回だけ
            // 書く。`out_shape_dev`／`in_shape_dev`／`in_strides_dev`
            // はいずれも `rank` 要素の配列でカーネル内ループも `rank`
            // 回のみ走査するため範囲外読み出しはない
            // （`kernels_interpolate.rs::INTERPOLATE_NEAREST_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.nearest_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&out_shape_dev)
                    .arg(&in_shape_dev)
                    .arg(&in_strides_dev)
                    .arg(&rank_i)
                    .arg(&spatial_start_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }

    /// `torch.nn.functional.interpolate(mode='bilinear', align_corners=
    /// …)` 相当（`kernels_interpolate.rs::INTERPOLATE_BILINEAR_F32`
    /// モジュール doc 参照。イシュー #1762）。`out_shape` は呼び出し元
    /// （`fandhe_ai_tensor_core::interpolate_out_shape_for_mode`）が
    /// 検査・確定済みの出力 shape で、空間軸はちょうど 2 軸
    /// （末尾 `rank-2`／`rank-1`）。`scale_h`／`scale_w` はホスト側
    /// （`fandhe_ai_tensor_core::bilinear_scale`。forward のホスト
    /// 参照実装・CPU ネイティブ実装と共有する単一情報源）で 1 回だけ
    /// 計算しカーネル引数として渡す。
    #[allow(clippy::too_many_arguments)]
    pub fn run_bilinear_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        align_corners: bool,
    ) -> Result<Vec<f32>, CudaError> {
        let rank = in_shape.len();
        debug_assert_eq!(
            out_shape.len(),
            rank,
            "run_bilinear_f32: caller must pre-validate rank match via \
             interpolate_out_shape_for_mode"
        );
        debug_assert!(
            rank >= 2,
            "run_bilinear_f32: caller must pre-validate exactly 2 spatial axes \
             via interpolate_out_shape_for_mode"
        );

        let numel_out = checked_numel(out_shape)?;
        if numel_out == 0 {
            return Ok(Vec::new());
        }

        let numel_in = checked_numel(in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidInterpolateShape {
                detail: format!(
                    "interpolate(bilinear): input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let out_shape_i32 = shape_to_i32(out_shape)?;
        let in_shape_i32 = shape_to_i32(in_shape)?;
        let in_strides_i32 = row_major_strides_i32(in_shape)?;
        let rank_i = validate_i32_bound(rank, "rank")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;
        let align_corners_i: i32 = if align_corners { 1 } else { 0 };

        let scale_h = bilinear_scale(in_shape[rank - 2], out_shape[rank - 2], align_corners);
        let scale_w = bilinear_scale(in_shape[rank - 1], out_shape[rank - 1], align_corners);

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let out_shape_dev = self.stream.clone_htod(&out_shape_i32)?;
            let in_shape_dev = self.stream.clone_htod(&in_shape_i32)?;
            let in_strides_dev = self.stream.clone_htod(&in_strides_i32)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(INTERPOLATE_BLOCK_DIM), 1, 1),
                block_dim: (INTERPOLATE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ
            // （直上の長さ検証でホストスライスと一致することを確認
            // 済み）。`out_dev` は `numel_out` 要素確保済みでカーネルは
            // `idx >= numel` の早期 return（REQ-8）を維持したまま各
            // 出力要素を 1 回だけ書く。`out_shape_dev`／`in_shape_dev`／
            // `in_strides_dev` はいずれも `rank` 要素の配列でカーネル内
            // ループも `rank` 回のみ走査するため範囲外読み出しはない
            // （`kernels_interpolate.rs::INTERPOLATE_BILINEAR_F32`
            // 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.bilinear_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&out_shape_dev)
                    .arg(&in_shape_dev)
                    .arg(&in_strides_dev)
                    .arg(&rank_i)
                    .arg(&scale_h)
                    .arg(&scale_w)
                    .arg(&align_corners_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }
}
