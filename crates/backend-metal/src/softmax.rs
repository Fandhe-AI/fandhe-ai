//! online softmax カーネルの起動 API（イシュー #604）。
//!
//! [`MetalSoftmax::new`] が `shaders/softmax.metal`（1 パス／2 パスの
//! 2 エントリ）を実行時コンパイルしてパイプラインを保持し、
//! [`MetalSoftmax::run_softmax_f32`] へホスト側スライスを渡すだけで
//! 経路選択・persistent grid 導出・バッファ確保・ディスパッチ・readback
//! を内部で完結できる（`crate::rmsnorm::MetalRmsNorm` と同型の構成）。
//!
//! `ops.rs::MetalBackendOps::run_fused` から canonical softmax プラン
//! （`exp(x - max(x)) / sum(exp(x - max(x)))`）検出時に呼ばれる
//! （[`crate::row_kernel::match_softmax_plan`] 参照）。
//!
//! **CUDA との parity 状況（重要）**: CUDA 側の online softmax（#594・
//! G-7）は本イシュー時点で OPEN（未実装）のため、CUDA 直接の parity
//! 相手はまだ存在しない。両バックエンドとも CPU 参照実装（REQ-2 統一
//! 複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）に対する
//! 数値一致を共有することで推移的に parity を担保する設計判断
//! （`tests/softmax_parity.rs` が同じ CPU 参照〈素朴な
//! `exp(x - max(x)) / sum` 実装〉を使う。#594 実装後は CUDA 側テストが
//! 同一参照を採ることで整合させる）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};
use crate::row_kernel::{self, ONEPASS_SMEM_BYTES_PER_GROUP, RowKernelRoute};

/// `shaders/softmax.metal` のソース（1 パス／2 パスの 2 カーネルを含む）。
const SOFTMAX_MSL_SRC: &str = include_str!("shaders/softmax.metal");

/// カーネル起動時の threadgroup 幅（32 スレッド = 1 simdgroup 固定。
/// `shaders/softmax.metal` 冒頭コメント参照。`crate::rmsnorm` と同じ値）。
const SOFTMAX_THREADGROUP_WIDTH: usize = 32;

/// [`crate::row_kernel::RowKernelValidationError`] → [`MetalError`] の変換
/// （`crate::rmsnorm::map_validation_error` と同じ変換規則。独立実装する
/// 理由は同ファイルのコメント参照）。
fn map_validation_error(err: row_kernel::RowKernelValidationError) -> MetalError {
    MetalError::InvalidRowKernelShape {
        detail: err.to_string(),
    }
}

/// online softmax カーネル（1 パス／2 パスの 2 エントリ）のコンパイル済み
/// パイプラインを保持するハンドル。
pub struct MetalSoftmax {
    onepass: objc2::rc::Retained<MtlPipeline>,
    twopass: objc2::rc::Retained<MtlPipeline>,
}

impl MetalSoftmax {
    /// `ctx` のデバイス上で softmax 2 カーネルを実行時コンパイルし
    /// パイプラインを構築する。`threadExecutionWidth == 32` の起動前検証は
    /// [`crate::rmsnorm::MetalRmsNorm::new`] と同じ理由・同じ契約
    /// （fail-closed）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(SOFTMAX_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let onepass = pipeline::make_pipeline(ctx.device(), &library, "softmax_f32_onepass")?;
        let twopass = pipeline::make_pipeline(ctx.device(), &library, "softmax_f32_twopass")?;

        for pipeline in [&onepass, &twopass] {
            let width = pipeline.threadExecutionWidth();
            if width != SOFTMAX_THREADGROUP_WIDTH {
                return Err(MetalError::UnexpectedThreadExecutionWidth {
                    expected: SOFTMAX_THREADGROUP_WIDTH,
                    actual: width,
                });
            }
        }

        Ok(Self { onepass, twopass })
    }

    /// `out = softmax(x)`（行ごと。`x` は `[rows, hidden]` の行優先
    /// 1 次元化済みバッファ）を実行する。`rows == 0 || hidden == 0` は
    /// 空結果の早期 return（`crate::rmsnorm::MetalRmsNorm::run_rmsnorm_f32_raw`
    /// と同じ 0 要素契約）。
    pub fn run_softmax_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, MetalError> {
        row_kernel::validate_row_kernel_launch(rows, hidden, x.len(), None, None)
            .map_err(map_validation_error)?;

        if rows == 0 || hidden == 0 {
            return Ok(Vec::new());
        }

        let route = row_kernel::select_route(hidden);

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::new_zeroed(ctx, x.len())?;

        let rows_u = rows as u32;
        let hidden_u = hidden as u32;

        let (pipeline, grid_size) = match route {
            RowKernelRoute::OnePass => {
                let grid_size = ctx.occupancy_params().map_or_else(
                    || row_kernel::derive_persistent_grid_fallback(rows_u),
                    |p| {
                        row_kernel::derive_persistent_grid(
                            p.gpu_core_count,
                            p.max_threadgroup_memory_bytes,
                            ONEPASS_SMEM_BYTES_PER_GROUP,
                            rows_u,
                        )
                    },
                );
                (&self.onepass, grid_size)
            }
            RowKernelRoute::TwoPass => {
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
                (&self.twopass, grid_size)
            }
        };

        ctx.dispatch_sync(|encoder| {
            encode_softmax_dispatch(
                encoder, pipeline, &x_buf, &out_buf, rows_u, hidden_u, grid_size,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// パイプライン設定・バッファ結線（index 0〜1）・スカラー引数
/// （index 2〜4）の結線・ディスパッチを行う（`crate::rmsnorm::
/// encode_rmsnorm_dispatch` と同型の構成。buffer index が異なるのは
/// `shaders/softmax.metal` のカーネル引数宣言が `w`／`eps`／`inv_n`／
/// `has_weight` を持たないため）。
fn encode_softmax_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    rows: u32,
    hidden: u32,
    grid_size: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`crate::rmsnorm::encode_rmsnorm_dispatch` と
    // 同じ「生存中の参照を保持するのみ」契約。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: FFI 境界 2/2。`crate::rmsnorm::encode_rmsnorm_dispatch` と
    // 同じ「即時複製」契約。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rows).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&hidden).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&grid_size).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    let threads_per_tg = MTLSize {
        width: SOFTMAX_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: grid_size as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
