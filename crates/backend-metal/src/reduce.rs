//! f32 `sum` reduction（全要素・単一軸。`torch.sum` 相当。イシュー
//! #1895・親イシュー #1894）の起動 API（実行時コンパイル・パイプライン
//! 保持・実行）。
//!
//! `crate::scan::MetalScan` と同じ構成方針を踏襲する:
//! [`MetalReduce::new`] が `shaders/reduce.metal`
//! （`reduce_sum_all_chunk_f32`／`reduce_sum_all_finalize_f32`／
//! `reduce_sum_axis_f32`）を実行時コンパイルしてパイプラインを保持し、
//! [`MetalReduce::run_sum_all_f32`]／[`MetalReduce::run_sum_axis_f32`]
//! へホスト側スライスを渡すだけでバッファ確保・ディスパッチ・readback
//! を内部で完結できる。
//!
//! **数値契約**（`fandhe_ai_backend_cpu::reduction::sum` の 2 段構成
//! 〈全要素: チャンク内逐次 → チャンク間逐次〉／単一 lane 逐次〈単一
//! 軸〉を binary64 ソフトウェアエミュレーションで逐語再現し CPU 参照
//! 実装と bit 完全一致する契約）は `shaders/reduce.metal` 冒頭コメント
//! および `crate::reduce_model` doc が正。
//!
//! **結線について**: 本モジュールは `MetalBackendOps::sum` から呼ばれ
//! ない（`context_cache::cached_reduce` は未追加・`ops.rs` からの参照
//! なし）。結線・`Var::sum` 経由の到達確認は #1896 のスコープ。
//!
//! 呼び出し元を前提としないため（現時点で `ops.rs` からの呼び出しが
//! 存在しない）、本モジュール自身が事前検査する（`crate::scan::
//! MetalScan` の「呼び出し元の検査結果を信頼しない」方針を先取りする
//! 形。#1896 で `ops.rs` 側にも事前検査が追加される見込み）。
//! [`MetalReduce::run_sum_all_f32`] は `crate::reduce_model::
//! plan_reduce_all` を呼んで検査する一方、[`MetalReduce::
//! run_sum_axis_f32`] は既に `outer`／`axis_len`／`inner` へ分解済み
//! の引数を受け取る都合上 `crate::reduce_model::plan_reduce_axis`
//! （`shape`／`dim` 起点）は呼ばず、同等の検査（`checked_mul`・
//! `u32::try_from` による `REDUCE_KERNEL_ARG_LIMIT` 相当の上限確認）
//! をインラインで行う（`plan_reduce_axis` はテストからのみ参照）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};
use crate::reduce_model::{self, ReducePrepareError};

/// `shaders/reduce.metal` のソース。
const REDUCE_MSL_SRC: &str = include_str!("shaders/reduce.metal");

/// 1 スレッドグループあたりのスレッド数（`scan.rs::
/// SCAN_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const REDUCE_THREADGROUP_WIDTH: usize = 256;

/// [`ReducePrepareError`] を [`MetalError::InvalidReduceShape`] へ
/// 写像する（`crate::scan::MetalScan` が現時点では `plan_scan` の
/// 呼び出し元〈`ops.rs`〉に検証を委ねているのに対し、本モジュールは
/// 呼び出し元が存在しないため自ら検証しここで写像する）。
fn map_prepare_error(err: ReducePrepareError) -> MetalError {
    MetalError::InvalidReduceShape {
        detail: err.to_string(),
    }
}

/// f32 `sum` reduction 3 カーネル（全要素 2 段・単一軸 1 段）のコンパイル
/// 済みパイプラインを保持するハンドル。
pub struct MetalReduce {
    sum_all_chunk_f32: objc2::rc::Retained<MtlPipeline>,
    sum_all_finalize_f32: objc2::rc::Retained<MtlPipeline>,
    sum_axis_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalReduce {
    /// `ctx` のデバイス上で 3 カーネルを実行時コンパイルしパイプライン
    /// を構築する（`scan.rs::MetalScan::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(REDUCE_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let sum_all_chunk_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "reduce_sum_all_chunk_f32")?;
        let sum_all_finalize_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "reduce_sum_all_finalize_f32")?;
        let sum_axis_f32 = pipeline::make_pipeline(ctx.device(), &library, "reduce_sum_axis_f32")?;

        Ok(Self {
            sum_all_chunk_f32,
            sum_all_finalize_f32,
            sum_axis_f32,
        })
    }

    /// `torch.sum(x)`（全要素・単一スカラー出力）相当。`x` が空の場合
    /// はディスパッチを回避し `0.0` を返す（`fandhe_ai_backend_cpu::
    /// reduction::sum(a, None)` の空縮約契約と同じ。`shaders/
    /// reduce.metal` 冒頭コメント「数値方式」参照）。
    ///
    /// **encode-only dispatch は使わない**: `scan.rs::MetalScan::
    /// run_scan` と同じ理由（戻り値を同期的に消費するため
    /// `ctx.dispatch_sync` で足りる。`*_tracked`／`DispatchFailureCell`
    /// が必要になるのは呼び出し元へ制御を返す前に GPU 完了を待たない
    /// encode-only 経路のみ）。2 段のディスパッチ（チャンク内縮約 →
    /// チャンク間縮約）は単一の `dispatch_sync` クロージャ内で同一
    /// エンコーダへ順にエンコードする（`mse.rs` の 2 回 `ctx.encode`
    /// 方式とは異なり、本関数は完了を待つ必要があるディスパッチが 1
    /// 回のみで済むため `dispatch_sync` の単一クロージャで完結できる）。
    pub fn run_sum_all_f32(&self, ctx: &MetalContext, x: &[f32]) -> Result<f32, MetalError> {
        // `plan_reduce_all(x.len())` は渡した `x.len()` をそのまま
        // `ReduceAllPlan::numel` へ格納する（`crate::reduce_model::
        // plan_reduce_all` 参照）ため、直後の `x.len() != plan.numel`
        // という再比較は構造的に常に false であり冗長だった（is-dead
        // code。呼び出し引数と戻り値が同一フィールドを指すだけで
        // 実際の検証にはならない）。よってここでは削除し、`plan.numel`
        // をそのまま `x.len()` の検証済み値として扱う。
        let plan = reduce_model::plan_reduce_all(x.len()).map_err(map_prepare_error)?;
        if plan.numel == 0 {
            return Ok(0.0);
        }
        let numel_u = u32::try_from(plan.numel).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!(
                "run_sum_all_f32: numel={} exceeds u32 range (kernel argument type)",
                plan.numel
            ),
        })?;
        let num_chunks_u =
            u32::try_from(plan.num_chunks).map_err(|_| MetalError::InvalidReduceShape {
                detail: format!(
                    "run_sum_all_f32: num_chunks={} exceeds u32 range (kernel argument type)",
                    plan.num_chunks
                ),
            })?;

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        // `partial` は `reduce_sum_all_chunk_f32` が `gid < num_chunks`
        // の各スレッドで必ず 1 回書く（`shaders/reduce.metal` 参照）
        // ため未初期化確保でよいが、`u64` 要素バッファは
        // `MetalIndexBuffer::new_zeroed_u64`（ゼロ初期化のみ。`sort.rs`
        // が合成キー配列で使う型）しか用意されていないためそれを流用
        // する（`f64` bit 表現の中間値・GPU 上でのみ消費されホストへは
        // 読み戻さない。`crate::sort::MetalSort` の合成キー配列と同じ
        // 使い方）。
        let partial_buf = MetalIndexBuffer::new_zeroed_u64(ctx, plan.num_chunks)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, 1)?;

        ctx.dispatch_sync(|encoder| {
            encode_sum_all_chunk_dispatch(
                encoder,
                &self.sum_all_chunk_f32,
                &x_buf,
                &partial_buf,
                numel_u,
                num_chunks_u,
            );
            encode_sum_all_finalize_dispatch(
                encoder,
                &self.sum_all_finalize_f32,
                &partial_buf,
                &out_buf,
                num_chunks_u,
            );
        })?;

        Ok(out_buf.read_to_vec().first().copied().unwrap_or(0.0))
    }

    /// `torch.sum(x, dim=dim)` 相当。`x` は `outer * axis_len * inner`
    /// 要素の稠密（contiguous）スライス。`lanes == outer * inner == 0`
    /// （空出力）の場合はディスパッチを回避し空 `Vec` を返す。
    /// `axis_len == 0 && lanes > 0`（空縮約）の場合もディスパッチを
    /// 回避し `vec![0.0; lanes]` を返す（`fandhe_ai_backend_cpu::
    /// reduction::axis_reduce_sum` の `k in 0..0` が何も加算せず
    /// `0.0` を返す契約と同じ。`shaders/reduce.metal` を起動しても
    /// 同じ結果になるが、`numel == 0` は `MetalBuffer::new_with_data`
    /// が 0 バイト確保を拒否するため早期 return が必要——`scan.rs::
    /// run_scan` の `numel == 0` 早期リターンと同じ理由）。
    pub fn run_sum_axis_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let lanes = outer
            .checked_mul(inner)
            .ok_or_else(|| MetalError::InvalidReduceShape {
                detail: "run_sum_axis_f32: outer * inner overflowed usize".to_string(),
            })?;
        let numel = lanes
            .checked_mul(axis_len)
            .ok_or_else(|| MetalError::InvalidReduceShape {
                detail: "run_sum_axis_f32: lanes * axis_len overflowed usize".to_string(),
            })?;
        if x.len() != numel {
            return Err(MetalError::InvalidReduceShape {
                detail: format!(
                    "run_sum_axis_f32: x.len()={} does not match numel={numel}",
                    x.len()
                ),
            });
        }
        if lanes == 0 {
            return Ok(Vec::new());
        }
        let lanes_u = u32::try_from(lanes).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!(
                "run_sum_axis_f32: lanes={lanes} exceeds u32 range (kernel argument type)"
            ),
        })?;
        let axis_len_u = u32::try_from(axis_len).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!(
                "run_sum_axis_f32: axis_len={axis_len} exceeds u32 range (kernel argument type)"
            ),
        })?;
        let inner_u = u32::try_from(inner).map_err(|_| MetalError::InvalidReduceShape {
            detail: format!(
                "run_sum_axis_f32: inner={inner} exceeds u32 range (kernel argument type)"
            ),
        })?;

        if axis_len == 0 {
            return Ok(vec![0.0; lanes]);
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        // `reduce_sum_axis_f32` は `gid < lanes` の各スレッドで必ず
        // `out[gid]` を 1 回書く（`shaders/reduce.metal` 参照）ため
        // `alloc_uninit_pooled` を使える（`scan.rs::run_scan` と同じ
        // 適用条件）。
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, lanes)?;

        ctx.dispatch_sync(|encoder| {
            encode_sum_axis_dispatch(
                encoder,
                &self.sum_axis_f32,
                &x_buf,
                &out_buf,
                lanes_u,
                axis_len_u,
                inner_u,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// `reduce_sum_all_chunk_f32` のエンコード（バッファ index 0〜1・
/// スカラー index 2〜3・ディスパッチ）。`shaders/reduce.metal::
/// reduce_sum_all_chunk_f32` のバッファ宣言と一致させる。
fn encode_sum_all_chunk_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    partial_buf: &MetalIndexBuffer,
    numel: u32,
    num_chunks: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2（`scan.rs::encode_scan_dispatch` と同じ
    // 契約）。`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への
    // 参照を保持するのみで即座に読み書きしない。各バッファは呼び出し元
    // `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 1);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から指定バイト数を即座に複製する。各ローカル変数は本呼び出し中
    // 生存し、型・バイト数は `shaders/reduce.metal::
    // reduce_sum_all_chunk_f32` の `constant uint&` 宣言と一致させて
    // いる。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&num_chunks).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
    }

    let threads_per_tg = MTLSize {
        width: REDUCE_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (num_chunks as usize).div_ceil(REDUCE_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `reduce_sum_all_finalize_f32` のエンコード（バッファ index 0〜1・
/// `num_chunks` index 2・単一スレッド起動）。
fn encode_sum_all_finalize_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    partial_buf: &MetalIndexBuffer,
    out_buf: &MetalBuffer,
    num_chunks: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_sum_all_chunk_dispatch` と同じ根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&num_chunks).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
    }

    // 単一スレッド（`shaders/reduce.metal::reduce_sum_all_finalize_f32`
    // の契約。`gid != 0` は早期 return）。
    let threads_per_tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `reduce_sum_axis_f32` のエンコード（バッファ index 0〜1・スカラー
/// index 2〜4・ディスパッチ）。`shaders/reduce.metal::
/// reduce_sum_axis_f32` のバッファ宣言と一致させる（`scan.rs::
/// encode_scan_dispatch` と同型）。
fn encode_sum_axis_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    lanes: u32,
    axis_len: u32,
    inner: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_sum_all_chunk_dispatch` と同じ根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }
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
        width: REDUCE_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (lanes as usize).div_ceil(REDUCE_THREADGROUP_WIDTH);
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
    fn reduce_msl_source_declares_all_kernels() {
        assert!(REDUCE_MSL_SRC.contains("kernel void reduce_sum_all_chunk_f32("));
        assert!(REDUCE_MSL_SRC.contains("kernel void reduce_sum_all_finalize_f32("));
        assert!(REDUCE_MSL_SRC.contains("kernel void reduce_sum_axis_f32("));
    }
}
