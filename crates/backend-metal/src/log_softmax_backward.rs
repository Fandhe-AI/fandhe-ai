//! `log_softmax` backward（`dx = g − exp(y)·Σ_dim(g)`。イシュー
//! #1952・親 #1947）の起動 API（実行時コンパイル・パイプライン保持・
//! 実行）。
//!
//! `crate::reduce::MetalReduce` と同じ構成方針を踏襲する:
//! [`MetalLogSoftmaxBackward::new`] が `shaders/log_softmax_backward.
//! metal`（`log_softmax_bwd_lane_sum`／`log_softmax_bwd_apply_f32`）を
//! 実行時コンパイルしてパイプラインを保持し、
//! [`MetalLogSoftmaxBackward::run_f32`] へホスト側スライスを渡すだけで
//! バッファ確保・ディスパッチ・readback を内部で完結できる。
//!
//! **数値契約**（`Σ_dim(g)` の縮約・`exp(y)` との乗算・`g` からの
//! 減算を binary64 ソフトウェアエミュレーションで計算し、`exp` の丸め
//! 差を除いてホスト参照実装 `fandhe_ai_autodiff::grad::
//! log_softmax_vjp_along` と bit 完全一致する契約）は `shaders/
//! log_softmax_backward.metal` 冒頭コメントおよび `crate::
//! log_softmax_backward_model` doc が正。
//!
//! **結線について**: 本モジュールは `MetalBackendOps::
//! log_softmax_backward`（`ops.rs`）から `context_cache::
//! cached_log_softmax_backward` 経由で呼ばれる（`Op::LogSoftmax` の
//! VJP が到達する）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::log_softmax_backward_model;
use crate::pipeline::{self, MtlPipeline};
use crate::reduce_model::ReducePrepareError;

/// `shaders/log_softmax_backward.metal` のソース。
const LOG_SOFTMAX_BACKWARD_MSL_SRC: &str = include_str!("shaders/log_softmax_backward.metal");

/// 1 スレッドグループあたりのスレッド数（`reduce.rs::
/// REDUCE_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const LOG_SOFTMAX_BACKWARD_THREADGROUP_WIDTH: usize = 256;

/// [`ReducePrepareError`] を [`MetalError::InvalidReduceShape`] へ写像
/// する（`crate::reduce::map_prepare_error` と同型。専用エラー型を
/// 新設せず `crate::reduce_model::ReducePrepareError` を再利用する
/// 契約は `log_softmax_backward_model::plan_log_softmax_backward` doc
/// 参照）。
fn map_prepare_error(err: ReducePrepareError) -> MetalError {
    MetalError::InvalidReduceShape {
        detail: err.to_string(),
    }
}

/// `log_softmax` backward 2 カーネル（lane 単位の `Σ_dim(g)`・要素
/// 単位の `dx` 計算）のコンパイル済みパイプラインを保持するハンドル。
pub struct MetalLogSoftmaxBackward {
    lane_sum: objc2::rc::Retained<MtlPipeline>,
    apply_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalLogSoftmaxBackward {
    /// `ctx` のデバイス上で 2 カーネルを実行時コンパイルしパイプライン
    /// を構築する（`reduce.rs::MetalReduce::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(LOG_SOFTMAX_BACKWARD_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let lane_sum = pipeline::make_pipeline(ctx.device(), &library, "log_softmax_bwd_lane_sum")?;
        let apply_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "log_softmax_bwd_apply_f32")?;

        Ok(Self {
            lane_sum,
            apply_f32,
        })
    }

    /// `dx = g − exp(y)·Σ_dim(g)`（`torch.log_softmax` backward 相当）を
    /// 計算する。`y`（forward 記録値 `log_softmax(x, dim)`）・`g`
    /// （上流勾配）は同一 shape の稠密（contiguous）スライス
    /// （呼び出し元 `ops.rs::MetalBackendOps::log_softmax_backward` が
    /// `contiguous()` を経て保証）で、`outer * axis_len * inner`
    /// 要素からなる（`dim` 軸限定契約は呼び出し元で検査済み）。
    ///
    /// **encode-only dispatch は使わない**: 戻り値を同期的に消費する
    /// ため `ctx.dispatch_sync` で足りる（`reduce.rs::run_sum_axis_f32`
    /// と同じ理由）。2 カーネルは単一の `dispatch_sync` クロージャ内で
    /// 同一エンコーダへ順にエンコードする（カーネル 1 の出力
    /// `lane_sum` をカーネル 2 が読むため、両者は同一コマンドバッファ
    /// 内でエンコード順どおりに実行される必要がある——`MTLComputeCommand
    /// Encoder` は同一エンコーダ内のディスパッチをエンコード順に
    /// シリアライズして実行する契約）。
    pub fn run_f32(
        &self,
        ctx: &MetalContext,
        y: &[f32],
        g: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let shape_for_plan: Vec<usize> = {
            // `plan_log_softmax_backward` は `shape`／`dim` からの分解を
            // 前提とするため、既に分解済みの `outer`／`axis_len`／
            // `inner` を `[outer, axis_len, inner]`・`dim=1` として渡す
            // （`crate::sort_model::line_layout` は各次元が 1 個の
            // 3 要素形状でも `outer`／`axis_len`／`inner` をそのまま
            // 再現する）。
            vec![outer, axis_len, inner]
        };
        let plan = log_softmax_backward_model::plan_log_softmax_backward(&shape_for_plan, 1)
            .map_err(map_prepare_error)?;
        let lanes = plan.lanes;
        let numel = lanes
            .checked_mul(axis_len)
            .ok_or_else(|| MetalError::InvalidReduceShape {
                detail: "run_f32: lanes * axis_len overflowed usize".to_string(),
            })?;
        if y.len() != numel || g.len() != numel {
            return Err(MetalError::InvalidReduceShape {
                detail: format!(
                    "run_f32: y.len()={}／g.len()={} does not match numel={numel}",
                    y.len(),
                    g.len()
                ),
            });
        }
        // `numel==0`（`lanes==0` または `axis_len==0` のいずれか）は
        // ディスパッチを回避し空 `Vec` を返す（`y.len()==g.len()==0`
        // の検証を上で通過済み。`MetalBuffer::new_with_data` は 0 要素
        // 確保を拒否するため——`reduce.rs::run_sum_axis_f32` の
        // `lanes==0` 早期リターンと同じ理由。`axis_len==0` かつ
        // `lanes>0` の場合も `numel==0` となり本分岐で処理される：
        // 空縮約〈`sum==0.0`〉により `dx=g` になるはずだが `numel==0`
        // なのでいずれにせよ空 `Vec` で正しい）。
        if numel == 0 {
            return Ok(Vec::new());
        }
        let lanes_u = u32::try_from(lanes).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!("run_f32: lanes={lanes} exceeds u32 range (kernel argument type)"),
        })?;
        let axis_len_u = u32::try_from(axis_len).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!(
                "run_f32: axis_len={axis_len} exceeds u32 range (kernel argument type)"
            ),
        })?;
        let inner_u = u32::try_from(inner).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!("run_f32: inner={inner} exceeds u32 range (kernel argument type)"),
        })?;
        let numel_u = u32::try_from(numel).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!("run_f32: numel={numel} exceeds u32 range (kernel argument type)"),
        })?;

        let y_buf = MetalBuffer::new_with_data(ctx, y)?;
        let g_buf = MetalBuffer::new_with_data(ctx, g)?;
        // `log_softmax_bwd_lane_sum` は `gid < lanes` の各スレッドで
        // 必ず `lane_sum[gid]` を書く（`shaders/log_softmax_backward.
        // metal` 参照）ため未初期化確保でよいが、`u64` 要素バッファは
        // `MetalIndexBuffer::new_zeroed_u64`（ゼロ初期化のみ）しか
        // 用意されていない（`reduce.rs::run_sum_all_f32` の `partial_buf`
        // と同じ流用）。
        let lane_sum_buf = MetalIndexBuffer::new_zeroed_u64(ctx, lanes)?;
        // `log_softmax_bwd_apply_f32` は `gid < numel` の各スレッドで
        // 必ず `out[gid]` を書くため未初期化確保でよい。
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_lane_sum_dispatch(
                encoder,
                &self.lane_sum,
                &g_buf,
                &lane_sum_buf,
                lanes_u,
                axis_len_u,
                inner_u,
            );
            encode_apply_dispatch(
                encoder,
                &self.apply_f32,
                &y_buf,
                &g_buf,
                &lane_sum_buf,
                &out_buf,
                numel_u,
                axis_len_u,
                inner_u,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// `log_softmax_bwd_lane_sum` のエンコード（バッファ index 0〜1・
/// スカラー index 2〜4・ディスパッチ）。`shaders/log_softmax_backward.
/// metal::log_softmax_bwd_lane_sum` のバッファ宣言と一致させる
/// （`reduce.rs::encode_sum_axis_dispatch` と同型）。
fn encode_lane_sum_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    g_buf: &MetalBuffer,
    lane_sum_buf: &MetalIndexBuffer,
    lanes: u32,
    axis_len: u32,
    inner: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2（`reduce.rs::encode_sum_axis_dispatch` と同じ
    // 契約）。`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への
    // 参照を保持するのみで即座に読み書きしない。各バッファは呼び出し元
    // `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(g_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(lane_sum_buf.raw()), 0, 1);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から指定バイト数を即座に複製する。各ローカル変数は本呼び出し中
    // 生存し、型・バイト数は `shaders/log_softmax_backward.metal::
    // log_softmax_bwd_lane_sum` の `constant uint&` 宣言（index 2〜4）
    // と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&lanes).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&axis_len).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inner).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    let threads_per_tg = MTLSize {
        width: LOG_SOFTMAX_BACKWARD_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (lanes as usize).div_ceil(LOG_SOFTMAX_BACKWARD_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `log_softmax_bwd_apply_f32` のエンコード（バッファ index 0〜3・
/// スカラー index 4〜6・ディスパッチ）。`shaders/log_softmax_backward.
/// metal::log_softmax_bwd_apply_f32` のバッファ宣言と一致させる。
#[allow(clippy::too_many_arguments)]
fn encode_apply_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    y_buf: &MetalBuffer,
    g_buf: &MetalBuffer,
    lane_sum_buf: &MetalIndexBuffer,
    out_buf: &MetalBuffer,
    numel: u32,
    axis_len: u32,
    inner: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_lane_sum_dispatch` と同じ契約。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(y_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(g_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(lane_sum_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 3);
    }

    // SAFETY: `encode_lane_sum_dispatch` と同じ契約。型・バイト数・
    // index は `shaders/log_softmax_backward.metal::
    // log_softmax_bwd_apply_f32` の `constant uint&` 宣言（index 4〜6）
    // と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&axis_len).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inner).cast(),
            std::mem::size_of::<u32>(),
            6,
        );
    }

    let threads_per_tg = MTLSize {
        width: LOG_SOFTMAX_BACKWARD_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(LOG_SOFTMAX_BACKWARD_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_backward_msl_source_declares_all_kernels() {
        assert!(LOG_SOFTMAX_BACKWARD_MSL_SRC.contains("kernel void log_softmax_bwd_lane_sum("));
        assert!(LOG_SOFTMAX_BACKWARD_MSL_SRC.contains("kernel void log_softmax_bwd_apply_f32("));
    }
}
