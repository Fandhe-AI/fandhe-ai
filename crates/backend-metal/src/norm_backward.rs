//! RMSNorm／LayerNorm backward カーネルの起動 API（イシュー #1953・親
//! #1947。`layer_norm.rs`〈#1596〉・CUDA 側 `norm_backward.rs`〈#1950〉と
//! 同じ構成方針を踏襲する）。
//!
//! [`MetalNormBackward::new`] が `shaders/norm_backward.metal`（4 カーネル:
//! `rmsnorm_bwd_dx_f32`／`rmsnorm_bwd_dw_f32`／`layer_norm_bwd_dx_f32`／
//! `layer_norm_bwd_dwdb_f32`）を実行時コンパイルしてパイプラインを保持し、
//! [`MetalNormBackward::run_rmsnorm_backward_f32`]／[`MetalNormBackward::
//! run_layer_norm_backward_f32`] へホスト側スライスを渡すだけでバッファ
//! 確保・ディスパッチ・readback を内部で完結できる。
//!
//! `ops.rs::MetalBackendOps::rmsnorm_backward`／`layer_norm_backward`
//! （`BackendOps` の独立エントリ）から直接呼ばれる。forward カーネル
//! （`rmsnorm.metal`／`layer_norm.metal`）・tape 記録は一切変更しない
//! （recompute-in-backward 方式。`shaders/norm_backward.metal` 冒頭
//! コメント参照）。
//!
//! # 数値契約
//!
//! `dx`／`dw`／`db` いずれも CPU ホスト参照実装（`fandhe_ai_autodiff::
//! grad::rmsnorm_vjp_rows`／`layer_norm_vjp_rows`）と REQ-2 統一複合
//! 判定で一致させる（bit 完全一致は主張しない。`shaders/norm_backward.metal`
//! 冒頭コメント「数値契約」参照）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};
use crate::row_kernel::{self, RowKernelValidationError};

/// `shaders/norm_backward.metal` のソース（4 カーネル共通の単一
/// ライブラリ）。
const NORM_BACKWARD_MSL_SRC: &str = include_str!("shaders/norm_backward.metal");

/// dx カーネルの threadgroup 幅（32 スレッド = 1 simdgroup 固定。
/// `layer_norm.rs::LAYER_NORM_THREADGROUP_WIDTH` と同じ理由）。
const NORM_BACKWARD_DX_THREADGROUP_WIDTH: usize = 32;

/// dw／db カーネル（1 スレッド = 1 列）の threadgroup 幅。
/// `gemm.rs::encode_bias_grad_reduce_dispatch` の `REDUCE_TG` と同じ
/// 経験的な値（列方向 grid-stride ではなく `n=hidden` 個の出力列を
/// 単純に `div_ceil` した threadgroup 数で覆う設計）。
const NORM_BACKWARD_REDUCE_TG: usize = 64;

/// `hidden` の `(float)hidden` 厳密表現上限（`2^24`。`layer_norm.rs::
/// LAYER_NORM_MAX_HIDDEN_EXACT_F32` と同じ理由・同じ値。本ファイルの
/// カーネルも `nb_f64_widen(as_type<uint>((float)hidden))` で `hidden`
/// を `f32` 経由で widen するため同じ境界検査が必要）。
const NORM_BACKWARD_MAX_HIDDEN_EXACT_F32: usize = 1 << 24;

fn map_validation_error(err: RowKernelValidationError) -> MetalError {
    MetalError::InvalidRowKernelShape {
        detail: err.to_string(),
    }
}

fn validate_hidden_exact_f32(hidden: usize) -> Result<(), MetalError> {
    if hidden > NORM_BACKWARD_MAX_HIDDEN_EXACT_F32 {
        return Err(MetalError::InvalidRowKernelShape {
            detail: format!(
                "norm_backward hidden exceeds exact f32 integer range: hidden={hidden} > {NORM_BACKWARD_MAX_HIDDEN_EXACT_F32} (2^24)"
            ),
        });
    }
    Ok(())
}

/// `dy.len() == x.len()` を検査する（`row_kernel::validate_row_kernel_launch`
/// は `x` 自身の shape のみを検査するため、`dy` の長さ一致は本ファイル側で
/// 追加検査する。OWASP A03）。
fn validate_dy_len(x_len: usize, dy_len: usize) -> Result<(), MetalError> {
    if dy_len != x_len {
        return Err(MetalError::InvalidRowKernelShape {
            detail: format!("norm_backward dy length mismatch: x.len()={x_len}, dy.len()={dy_len}"),
        });
    }
    Ok(())
}

/// RMSNorm／LayerNorm backward の 4 カーネルのコンパイル済みパイプライン
/// を保持するハンドル。
pub struct MetalNormBackward {
    rmsnorm_dx: objc2::rc::Retained<MtlPipeline>,
    rmsnorm_dw: objc2::rc::Retained<MtlPipeline>,
    layer_norm_dx: objc2::rc::Retained<MtlPipeline>,
    layer_norm_dwdb: objc2::rc::Retained<MtlPipeline>,
}

impl MetalNormBackward {
    /// `ctx` のデバイス上で 4 カーネルを実行時コンパイルしパイプラインを
    /// 構築する。dx 系 2 カーネルの `threadExecutionWidth` が 32 と
    /// 一致することを検証する（fail-closed。`MetalLayerNorm::new` と
    /// 同じ理由。dw／db カーネルは simdgroup 幅に依存しないため検証しない）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(NORM_BACKWARD_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let rmsnorm_dx = pipeline::make_pipeline(ctx.device(), &library, "rmsnorm_bwd_dx_f32")?;
        let width = rmsnorm_dx.threadExecutionWidth();
        if width != NORM_BACKWARD_DX_THREADGROUP_WIDTH {
            return Err(MetalError::UnexpectedThreadExecutionWidth {
                expected: NORM_BACKWARD_DX_THREADGROUP_WIDTH,
                actual: width,
            });
        }
        let rmsnorm_dw = pipeline::make_pipeline(ctx.device(), &library, "rmsnorm_bwd_dw_f32")?;
        let layer_norm_dx =
            pipeline::make_pipeline(ctx.device(), &library, "layer_norm_bwd_dx_f32")?;
        let width2 = layer_norm_dx.threadExecutionWidth();
        if width2 != NORM_BACKWARD_DX_THREADGROUP_WIDTH {
            return Err(MetalError::UnexpectedThreadExecutionWidth {
                expected: NORM_BACKWARD_DX_THREADGROUP_WIDTH,
                actual: width2,
            });
        }
        let layer_norm_dwdb =
            pipeline::make_pipeline(ctx.device(), &library, "layer_norm_bwd_dwdb_f32")?;

        Ok(Self {
            rmsnorm_dx,
            rmsnorm_dw,
            layer_norm_dx,
            layer_norm_dwdb,
        })
    }

    /// RMSNorm backward（`dx`・`dw`）を実行する。`w` を渡す場合のみ
    /// `dw`（shape `[hidden]`）を返す。`rows == 0 || hidden == 0` は
    /// 空 `dx`・（`w` ありなら）ゼロ埋め `dw` を返す（`BackendOps::
    /// rmsnorm_backward` の 0 要素契約）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_rmsnorm_backward_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), MetalError> {
        row_kernel::validate_row_kernel_launch(
            rows,
            hidden,
            x.len(),
            w.map(|s| s.len()),
            Some(eps),
        )
        .map_err(map_validation_error)?;
        validate_dy_len(x.len(), dy.len())?;
        validate_hidden_exact_f32(hidden)?;

        if rows == 0 || hidden == 0 {
            let dw = w.map(|_| vec![0.0f32; hidden]);
            return Ok((Vec::new(), dw));
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let (w_buf, has_weight) = match w {
            Some(w_slice) => (MetalBuffer::new_with_data(ctx, w_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, hidden)?, 0i32),
        };
        let dy_buf = MetalBuffer::new_with_data(ctx, dy)?;
        let dx_buf = MetalBuffer::alloc_uninit_pooled(ctx, x.len())?;
        let rstd_buf = MetalBuffer::alloc_uninit_pooled(ctx, rows)?;

        let rows_u = rows as u32;
        let hidden_u = hidden as u32;
        let grid_size = ctx.occupancy_params().map_or_else(
            || row_kernel::derive_persistent_grid_fallback(rows_u),
            |p| {
                row_kernel::derive_persistent_grid(
                    p.gpu_core_count,
                    p.max_threadgroup_memory_bytes,
                    0,
                    rows_u,
                )
            },
        );

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.rmsnorm_dx);
            // SAFETY: FFI 境界（`setBuffer_offset_atIndex`）。`x_buf`／
            // `w_buf`／`dy_buf`／`dx_buf`／`rstd_buf` は本 `dispatch_sync`
            // 呼び出しが完了するまで生存する（`layer_norm.rs::
            // encode_layer_norm_dispatch` と同じ契約）。
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dy_buf.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(dx_buf.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(rstd_buf.raw()), 0, 4);
            }
            // SAFETY: FFI 境界（`setBytes_length_atIndex`。即時複製契約。
            // `layer_norm.rs::encode_layer_norm_dispatch` と同型）。
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&rows_u).cast(),
                    std::mem::size_of::<u32>(),
                    5,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&hidden_u).cast(),
                    std::mem::size_of::<u32>(),
                    6,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&eps).cast(),
                    std::mem::size_of::<f32>(),
                    7,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&has_weight).cast(),
                    std::mem::size_of::<i32>(),
                    8,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&grid_size).cast(),
                    std::mem::size_of::<u32>(),
                    9,
                );
            }
            let threads_per_tg = MTLSize {
                width: NORM_BACKWARD_DX_THREADGROUP_WIDTH,
                height: 1,
                depth: 1,
            };
            let threadgroups = MTLSize {
                width: grid_size as usize,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        let dw_buf = MetalBuffer::alloc_uninit_pooled(ctx, hidden)?;
        ctx.dispatch_sync(|encoder| {
            encode_reduce_dispatch(
                encoder,
                &self.rmsnorm_dw,
                &[&x_buf, &dy_buf, &rstd_buf],
                &dw_buf,
                rows_u,
                hidden_u,
            );
        })?;

        let dx = dx_buf.read_to_vec();
        let dw = w.map(|_| dw_buf.read_to_vec());
        Ok((dx, dw))
    }

    /// LayerNorm backward（`dx`・`dw`・`db`）を実行する。
    /// [`Self::run_rmsnorm_backward_f32`] と同じ 0 要素契約。
    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_norm_backward_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        w: Option<&[f32]>,
        has_bias: bool,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>), MetalError> {
        row_kernel::validate_row_kernel_launch(
            rows,
            hidden,
            x.len(),
            w.map(|s| s.len()),
            Some(eps),
        )
        .map_err(map_validation_error)?;
        validate_dy_len(x.len(), dy.len())?;
        validate_hidden_exact_f32(hidden)?;

        if rows == 0 || hidden == 0 {
            let dw = w.map(|_| vec![0.0f32; hidden]);
            let db = has_bias.then(|| vec![0.0f32; hidden]);
            return Ok((Vec::new(), dw, db));
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let (w_buf, has_weight) = match w {
            Some(w_slice) => (MetalBuffer::new_with_data(ctx, w_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, hidden)?, 0i32),
        };
        let dy_buf = MetalBuffer::new_with_data(ctx, dy)?;
        let dx_buf = MetalBuffer::alloc_uninit_pooled(ctx, x.len())?;
        // `mean`／`rstd` scratch は `f64` bit パターン（`ulong`。8 バイト
        // = `f32` 2 要素分）で書き出すため `rows * 2` 要素確保する
        // （`shaders/norm_backward.metal::layer_norm_bwd_dx_f32` の
        // `device ulong* mean_out`／`rstd_out` 宣言とバイト単位で一致）。
        let mean_buf = MetalBuffer::alloc_uninit_pooled(ctx, rows * 2)?;
        let rstd_buf = MetalBuffer::alloc_uninit_pooled(ctx, rows * 2)?;

        let rows_u = rows as u32;
        let hidden_u = hidden as u32;
        let grid_size = ctx.occupancy_params().map_or_else(
            || row_kernel::derive_persistent_grid_fallback(rows_u),
            |p| {
                row_kernel::derive_persistent_grid(
                    p.gpu_core_count,
                    p.max_threadgroup_memory_bytes,
                    0,
                    rows_u,
                )
            },
        );

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.layer_norm_dx);
            // SAFETY: FFI 境界（`setBuffer_offset_atIndex`）。各バッファは
            // 本 `dispatch_sync` 呼び出しが完了するまで生存する。
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(w_buf.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dy_buf.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(dx_buf.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(mean_buf.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(rstd_buf.raw()), 0, 5);
            }
            // SAFETY: FFI 境界（`setBytes_length_atIndex`。即時複製契約）。
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&rows_u).cast(),
                    std::mem::size_of::<u32>(),
                    6,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&hidden_u).cast(),
                    std::mem::size_of::<u32>(),
                    7,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&eps).cast(),
                    std::mem::size_of::<f32>(),
                    8,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&has_weight).cast(),
                    std::mem::size_of::<i32>(),
                    9,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&grid_size).cast(),
                    std::mem::size_of::<u32>(),
                    10,
                );
            }
            let threads_per_tg = MTLSize {
                width: NORM_BACKWARD_DX_THREADGROUP_WIDTH,
                height: 1,
                depth: 1,
            };
            let threadgroups = MTLSize {
                width: grid_size as usize,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        let dw_buf = MetalBuffer::alloc_uninit_pooled(ctx, hidden)?;
        let db_buf = MetalBuffer::alloc_uninit_pooled(ctx, hidden)?;
        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.layer_norm_dwdb);
            // SAFETY: FFI 境界（`setBuffer_offset_atIndex`）。各バッファは
            // 本 `dispatch_sync` 呼び出しが完了するまで生存する。
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dy_buf.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(mean_buf.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(rstd_buf.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(dw_buf.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(db_buf.raw()), 0, 5);
            }
            // SAFETY: FFI 境界（`setBytes_length_atIndex`。即時複製契約）。
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&rows_u).cast(),
                    std::mem::size_of::<u32>(),
                    6,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&hidden_u).cast(),
                    std::mem::size_of::<u32>(),
                    7,
                );
            }
            let threads_per_tg = MTLSize {
                width: NORM_BACKWARD_REDUCE_TG,
                height: 1,
                depth: 1,
            };
            let threadgroups = MTLSize {
                width: (hidden_u as usize).div_ceil(NORM_BACKWARD_REDUCE_TG),
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        let dx = dx_buf.read_to_vec();
        let dw = w.map(|_| dw_buf.read_to_vec());
        let db = has_bias.then(|| db_buf.read_to_vec());
        Ok((dx, dw, db))
    }
}

/// RMSNorm dw カーネル（`rmsnorm_bwd_dw_f32`）のパイプライン設定・
/// バッファ結線（index 0〜2 は `inputs`・index 3 は `out`）・スカラー
/// 引数（index 4〜5）・ディスパッチ（1 スレッド = 1 列。
/// `gemm.rs::encode_bias_grad_reduce_dispatch` の `REDUCE_TG` 方式と同型）。
fn encode_reduce_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    inputs: &[&MetalBuffer],
    out: &MetalBuffer,
    rows: u32,
    hidden: u32,
) {
    encoder.setComputePipelineState(pipeline);
    // SAFETY: FFI 境界（`setBuffer_offset_atIndex`）。`inputs`／`out` は
    // 呼び出し元の `dispatch_sync` クロージャが完了するまで生存する。
    unsafe {
        for (i, buf) in inputs.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf.raw()), 0, i);
        }
        encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, inputs.len());
    }
    // SAFETY: FFI 境界（`setBytes_length_atIndex`。即時複製契約）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rows).cast(),
            std::mem::size_of::<u32>(),
            inputs.len() + 1,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&hidden).cast(),
            std::mem::size_of::<u32>(),
            inputs.len() + 2,
        );
    }
    let threads_per_tg = MTLSize {
        width: NORM_BACKWARD_REDUCE_TG,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: (hidden as usize).div_ceil(NORM_BACKWARD_REDUCE_TG),
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_hidden_exact_f32_accepts_boundary() {
        assert!(validate_hidden_exact_f32(NORM_BACKWARD_MAX_HIDDEN_EXACT_F32).is_ok());
        assert!(validate_hidden_exact_f32(1).is_ok());
        assert!(validate_hidden_exact_f32(0).is_ok());
    }

    #[test]
    fn validate_hidden_exact_f32_rejects_above_boundary() {
        let err = validate_hidden_exact_f32(NORM_BACKWARD_MAX_HIDDEN_EXACT_F32 + 1)
            .expect_err("hidden = 2^24 + 1 must be rejected");
        assert!(matches!(err, MetalError::InvalidRowKernelShape { .. }));
    }

    #[test]
    fn validate_dy_len_rejects_mismatch() {
        let err = validate_dy_len(10, 9).expect_err("mismatched dy length must be rejected");
        assert!(matches!(err, MetalError::InvalidRowKernelShape { .. }));
        assert!(validate_dy_len(10, 10).is_ok());
    }
}
