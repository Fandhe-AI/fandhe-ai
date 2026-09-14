//! 定数パディングの起動 API（NVRTC コンパイル・保持・実行。イシュー
//! #1756）。
//!
//! `gather_scatter.rs::CudaGatherScatter` と同じ構成方針を踏襲する:
//! [`CudaConstantPad::new`] が `CudaDevice` から `constant_pad_f32`
//! カーネル（`kernels_constant_pad.rs`）を NVRTC コンパイルして保持し、
//! 以降は [`CudaConstantPad::run_pad_f32`] へホスト側スライスを渡すだけで
//! GPU 実行できる。`ops.rs::CudaBackendOps::pad` から `BackendOps` の
//! 実装として呼ばれる。
//!
//! **shape 検証の責務分担**（`gather_scatter.rs` モジュール doc・
//! `.claude/rules/security.md` A08 と同じ二重検査方針）: 呼び出し元
//! `ops.rs` が [`fandhe_ai_tensor_core::pad_out_shape`] で `input.shape()`／
//! `pads` の rank 整合を再検査してから本モジュールへ委譲する契約のため、
//! 本モジュール自身は `rank` の整合を `debug_assert!` でのみ確認する
//! （release ビルドでは無効化される内部契約検証）。一方で `input` の
//! **要素数**（`checked_numel` で導出した期待長との一致）は release
//! ビルドでも独立に検証する（`gather_scatter.rs` モジュール doc と同じ
//! 理由: ホストスライスが短いと GPU 側で未定義動作になりうるため、
//! H2D 転送前に必須の独立した安全策）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gather_scatter::{row_major_strides_i32, shape_to_i32};
use crate::kernels_constant_pad::{self, CONSTANT_PAD_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int` は
/// C の 32bit 符号付き整数のため。`gather_scatter.rs::validate_i32_bound`
/// と同じ理由の複製——モジュールごとにエラー型が異なるため専用に持つ
/// 既存方針を踏襲する）。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::InvalidConstantPadShape {
        detail: format!("pad dimension must fit in i32 (kernel argument type): {name}={value}"),
    })
}

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`gather_scatter.rs::checked_numel` と同型。クレート内で
/// `pub(crate)` 共有できないため複製する。空 shape チェックより前に
/// 無検査の `.iter().product()` を計算すると overflow-checks 有効時に
/// panic しうるため——CPU 側・gather/scatter 側と同じ理由）。
fn checked_numel(shape: &[usize]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CudaError::InvalidConstantPadShape {
            detail: "checked_numel: element count overflow".to_string(),
        })
}

/// pad カーネルのコンパイル済みハンドルを保持する。
pub struct CudaConstantPad {
    stream: Arc<CudaStream>,
    /// `gather_scatter.rs::CudaGatherScatter::ordinal` と同じ役割
    /// （`Self::with_driver_call` が `context_cache::with_driver_call`
    /// を呼ぶ際のキー）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    pad_f32: CudaFunction,
}

impl CudaConstantPad {
    /// `device` 上で pad カーネルを NVRTC コンパイルし保持するハンドルを
    /// 構築する（`gather_scatter.rs::CudaGatherScatter::new` と同一
    /// 手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(kernels_constant_pad::CONSTANT_PAD_F32, arch)?;
        let pad_f32 = device
            .context()
            .load_module(ptx)?
            .load_function("constant_pad_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            pad_f32,
        })
    }

    /// `CudaConstantPad` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`gather_scatter.rs::CudaGatherScatter::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `torch.nn.functional.pad(mode='constant')` 相当
    /// （`kernels_constant_pad.rs` モジュール doc 参照）。`out_shape` は
    /// 呼び出し元（`fandhe_ai_tensor_core::pad_out_shape`）が検査・確定
    /// 済みの出力 shape。`pads[a] = (before, after)` は `in_shape` と
    /// 同じ rank（呼び出し元の検査契約）。
    pub fn run_pad_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        pads: &[(usize, usize)],
        value: f32,
    ) -> Result<Vec<f32>, CudaError> {
        let rank = in_shape.len();
        debug_assert_eq!(
            out_shape.len(),
            rank,
            "run_pad_f32: caller must pre-validate rank match via pad_out_shape"
        );
        debug_assert_eq!(
            pads.len(),
            rank,
            "run_pad_f32: caller must pre-validate pads rank via pad_out_shape"
        );

        let numel_out = checked_numel(out_shape)?;
        if numel_out == 0 {
            // 空出力早期リターン（`gather_scatter.rs::run_gather_f32` の
            // 空出力早期リターンと同じ理由・同じ順序方針: `input` を
            // 読む必要が一切ないため、`in_shape` の要素数検査自体を
            // 行わずに空 `Vec` を返す）。
            return Ok(Vec::new());
        }

        // `numel_out` が非ゼロでも `input`（`in_shape`）が空の場合が
        // ありうる（pad は「入力が空でも出力は非空になりうる」演算。
        // `constant_pad.rs`〈CPU〉モジュール doc と同じ契約）。このとき
        // `input` は読まず全域 `value` で埋める。
        let in_is_empty = in_shape.contains(&0);

        let out_shape_i32 = shape_to_i32(out_shape)?;
        let rank_i = validate_i32_bound(rank, "rank")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        if in_is_empty {
            // `input` を H2D 転送せず、出力全域を `value` で埋める軽量
            // 経路（カーネル起動自体は行わずホスト側で埋める——`input`
            // が空である以上 GPU 計算は不要）。
            return Ok(vec![value; numel_out]);
        }

        let numel_in = checked_numel(in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidConstantPadShape {
                detail: format!(
                    "pad: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let in_shape_i32 = shape_to_i32(in_shape)?;
        let in_strides_i32 = row_major_strides_i32(in_shape)?;
        let before_i32: Vec<i32> = pads
            .iter()
            .map(|&(before, _after)| validate_i32_bound(before, "before"))
            .collect::<Result<_, _>>()?;

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let out_shape_dev = self.stream.clone_htod(&out_shape_i32)?;
            let in_shape_dev = self.stream.clone_htod(&in_shape_i32)?;
            let before_dev = self.stream.clone_htod(&before_i32)?;
            let in_strides_dev = self.stream.clone_htod(&in_strides_i32)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(CONSTANT_PAD_BLOCK_DIM), 1, 1),
                block_dim: (CONSTANT_PAD_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ
            // （直上の長さ検証でホストスライスと一致することを確認
            // 済み）。`out_dev` は `numel_out` 要素確保済みでカーネルは
            // `idx < numel`（REQ-8）を維持したまま各出力要素を 1 回だけ
            // 書く。`out_shape_dev`／`in_shape_dev`／`before_dev`／
            // `in_strides_dev` はいずれも `rank` 要素の配列でカーネル内
            // ループも `rank` 回のみ走査するため範囲外読み出しはない
            // （`kernels_constant_pad.rs::CONSTANT_PAD_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.pad_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&out_shape_dev)
                    .arg(&in_shape_dev)
                    .arg(&before_dev)
                    .arg(&in_strides_dev)
                    .arg(&rank_i)
                    .arg(&numel_i)
                    .arg(&value)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }
}
