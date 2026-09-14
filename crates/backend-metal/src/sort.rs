//! `sort`／`topk`（イシュー #1741）の起動 API（実行時コンパイル・
//! パイプライン保持・実行）。
//!
//! `unique.rs::MetalUnique` と同じ構成方針を踏襲する: [`MetalSort::
//! new`] が `shaders/sort.metal`（`sort_build_keys_u64`／
//! `bitonic_step_u64`／`sort_finalize_f32`）を実行時コンパイルして
//! パイプラインを保持し、[`MetalSort::run_sort_f32`] へホスト側
//! スライスを渡すだけでバッファ確保・ディスパッチ・readback を内部で
//! 完結できる。`ops.rs::MetalBackendOps::sort`／`topk` から呼ばれる。
//!
//! # アルゴリズム
//!
//! `crate::sort_model` モジュール doc（鍵設計・ライン分解・パディング）
//! と同一。GPU 側の違いは、ホストと同一エンコーダ（`ctx.dispatch_sync`
//! が生成する serial encoder）へ全ステップ（build_keys → 全ビットニック
//! ステップ → finalize）を連続してエンコードし 1 回の `dispatch_sync`
//! （＝1 回の同期）で完結させる点（`unique.rs::MetalUnique::
//! run_unique_f32` と同じ「複数ステップを 1 回の同期区間へまとめる」
//! 設計。CUDA 側〈`sort.rs::CudaSort::run_sort_f32`〉の同一ストリーム
//! 上への繰り返し `launch` に相当）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBarrierScope, MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};
use crate::sort_model::{SortPlan, SortPrepareError, plan_sort};

/// `shaders/sort.metal` のソース。
const SORT_MSL_SRC: &str = include_str!("shaders/sort.metal");

/// 1 スレッドグループあたりのスレッド数（`unique.rs::
/// UNIQUE_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const SORT_THREADGROUP_WIDTH: usize = 256;

/// `sort_build_keys_u64`／`bitonic_step_u64`／`sort_finalize_f32` の
/// コンパイル済みパイプラインを保持するハンドル。
pub struct MetalSort {
    build_keys: objc2::rc::Retained<MtlPipeline>,
    bitonic_step: objc2::rc::Retained<MtlPipeline>,
    finalize: objc2::rc::Retained<MtlPipeline>,
}

/// `usize` 値を `u32` カーネル引数へ変換する（`plan_sort` が事前に
/// `i32::MAX` 上限を検証済みのため通常は失敗しないが、`out_len`
/// （`ops.rs` から渡される。`plan_sort` の検証対象外）は独立に検証
/// する必要がある。失敗時は `MetalError::InvalidGatherScatterShape`
/// （`unique.rs` が `UniquePrepareError` の写像に流用しているのと同じ
/// 既存 variant。sort/topk 専用の新規 variant は追加しない）。
fn checked_u32(value: usize, name: &str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::InvalidGatherScatterShape {
        detail: format!("sort/topk: {name}={value} exceeds u32 range"),
    })
}

impl MetalSort {
    /// `ctx` のデバイス上で 3 カーネルを実行時コンパイルしパイプラインを
    /// 構築する（`unique.rs::MetalUnique::new` と同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(SORT_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let build_keys = pipeline::make_pipeline(ctx.device(), &library, "sort_build_keys_u64")?;
        let bitonic_step = pipeline::make_pipeline(ctx.device(), &library, "bitonic_step_u64")?;
        let finalize = pipeline::make_pipeline(ctx.device(), &library, "sort_finalize_f32")?;

        Ok(Self {
            build_keys,
            bitonic_step,
            finalize,
        })
    }

    /// `sort`（`out_len == dim_size`）／`topk`（`out_len == k`）共通の
    /// 実行本体（`crate::sort_model` モジュール doc 参照）。
    ///
    /// `input`（行優先 contiguous。`ops.rs` が事前に `.contiguous()`
    /// 済みのスライスを渡す契約）・`shape`・`dim`・`descending`
    /// （sort の降順指定／topk の largest を `descending` へ写像。
    /// `ops.rs` が変換する）・`out_len` を受け取り `(values, index)`
    /// を返す。
    ///
    /// 空出力（`out_len == 0`。`topk(k=0)` 等）は `MetalIndexBuffer` が
    /// 0 バイト確保を拒否するため、バッファ確保・`dispatch_sync` に
    /// 入る前に早期 return する（`unique.rs::MetalUnique::
    /// run_unique_f32` の `n == 0`／`n == 1` 早期リターンと同じ理由）。
    /// `plan_sort`（サイズ上限検証）もクロージャに入る前に完了させる
    /// （`dispatch_sync` のクロージャは `Result` を返せないため）。
    pub fn run_sort_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        shape: &[usize],
        dim: usize,
        descending: bool,
        out_len: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), MetalError> {
        let plan = plan_sort(shape, dim).map_err(|e: SortPrepareError| {
            MetalError::InvalidGatherScatterShape {
                detail: e.to_string(),
            }
        })?;
        let SortPlan {
            dim_size,
            inner,
            lines,
            padded,
            ..
        } = plan;

        if input.len() != lines * dim_size {
            return Err(MetalError::InvalidGatherScatterShape {
                detail: format!(
                    "sort/topk: input length {} does not match lines*dim_size {}",
                    input.len(),
                    lines * dim_size
                ),
            });
        }

        let total_keys = lines * padded;
        let total_out = lines * out_len;
        if total_out == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let lines_u = checked_u32(lines, "lines")?;
        let dim_size_u = checked_u32(dim_size, "dim_size")?;
        let inner_u = checked_u32(inner, "inner")?;
        let padded_u = checked_u32(padded, "padded")?;
        let out_len_u = checked_u32(out_len, "out_len")?;
        let numel_in_u = checked_u32(input.len(), "numel_in")?;
        let numel_out_u = checked_u32(total_out, "numel_out")?;
        let descending_u: u32 = if descending { 1 } else { 0 };

        // `MetalIndexBuffer::new_with_u32` は `u32` バッファとして確保
        // するが、シェーダ側は `device const float*` として同一バイト
        // 列を読む（`f32`／`u32` は同じ 4 バイト幅のビットパターンを
        // そのまま再解釈するだけであり、アップロード自体は
        // `newBufferWithBytes_length_options` による生バイトコピーの
        // ため型は無関係。`bytemuck_f32_as_u32` 参照）。
        let input_buf = MetalIndexBuffer::new_with_u32(ctx, bytemuck_f32_as_u32(input))?;
        let keys_buf = MetalIndexBuffer::new_zeroed_u64(ctx, total_keys)?;
        let values_buf = crate::buffer::MetalBuffer::new_zeroed(ctx, total_out)?;
        let index_buf = MetalIndexBuffer::new_zeroed_i32(ctx, total_out)?;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.build_keys);
            encode_build_keys(
                encoder,
                &input_buf,
                &keys_buf,
                lines_u,
                dim_size_u,
                inner_u,
                padded_u,
                descending_u,
                numel_in_u,
            );
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);

            encoder.setComputePipelineState(&self.bitonic_step);
            if padded > 1 {
                let mut k = 2usize;
                while k <= padded {
                    let mut j = k / 2;
                    while j >= 1 {
                        let j_u = checked_u32(j, "j").unwrap_or(0);
                        let k_u = checked_u32(k, "k").unwrap_or(0);
                        encode_bitonic_step(encoder, &keys_buf, j_u, k_u, padded_u, lines_u);
                        encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                        j /= 2;
                    }
                    k *= 2;
                }
            }

            encoder.setComputePipelineState(&self.finalize);
            encode_finalize(
                encoder,
                &input_buf,
                &keys_buf,
                &values_buf,
                &index_buf,
                lines_u,
                dim_size_u,
                inner_u,
                padded_u,
                out_len_u,
                numel_in_u,
                numel_out_u,
            );
        })?;

        let values = values_buf.read_to_vec();
        let index = index_buf.read_to_vec_i32();
        Ok((values, index))
    }
}

/// `&[f32]` を `&[u32]` として再解釈する（同一 4 バイト幅のビット
/// パターンをそのままアップロードするためのキャスト。値の変換は
/// 行わない。`MetalIndexBuffer::new_with_u32` へ渡すためのみに使う）。
fn bytemuck_f32_as_u32(data: &[f32]) -> &[u32] {
    // SAFETY: `f32`／`u32` はいずれも 4 バイト幅・4 バイトアラインの
    // POD 型であり、任意のビットパターンが両者とも有効値（`f32` は
    // NaN／非正規化数を含め全ビットパターンが有効）。長さ・アライン
    // メントを変えないポインタキャストのみで、読み出しも書き込みも
    // 行わない。
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) }
}

#[allow(clippy::too_many_arguments)]
fn encode_build_keys(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input_buf: &MetalIndexBuffer,
    keys_buf: &MetalIndexBuffer,
    lines: u32,
    dim_size: u32,
    inner: u32,
    padded: u32,
    descending: u32,
    numel_in: u32,
) {
    // SAFETY: FFI 境界（`unique.rs::encode_bitonic_step` と同じ契約）。
    // `setBuffer_offset_atIndex` は生存中の `MTLBuffer` への参照を保持
    // するのみで即座に読み書きしない。両バッファは `ctx.dispatch_sync`
    // が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(keys_buf.raw()), 0, 1);
    }
    set_scalar_args(
        encoder,
        &[lines, dim_size, inner, padded, descending, numel_in],
        2,
    );
    let total = (lines as usize) * (padded as usize);
    dispatch_1d(encoder, total);
}

fn encode_bitonic_step(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    keys_buf: &MetalIndexBuffer,
    j: u32,
    k: u32,
    padded: u32,
    lines: u32,
) {
    // SAFETY: `encode_build_keys` と同一の契約。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(keys_buf.raw()), 0, 0);
    }
    set_scalar_args(encoder, &[j, k, padded, lines], 1);
    let total = (lines as usize) * (padded as usize);
    dispatch_1d(encoder, total);
}

#[allow(clippy::too_many_arguments)]
fn encode_finalize(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input_buf: &MetalIndexBuffer,
    keys_buf: &MetalIndexBuffer,
    values_buf: &crate::buffer::MetalBuffer,
    index_buf: &MetalIndexBuffer,
    lines: u32,
    dim_size: u32,
    inner: u32,
    padded: u32,
    out_len: u32,
    numel_in: u32,
    numel_out: u32,
) {
    // SAFETY: `encode_build_keys` と同一の契約。`values_buf`
    // （`crate::buffer::MetalBuffer`）は `raw()` で生バッファ参照を
    // 返す既存の f32 専用型（`unique`／`sort` 双方の出力用途で共有）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(keys_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(values_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(index_buf.raw()), 0, 3);
    }
    set_scalar_args(
        encoder,
        &[lines, dim_size, inner, padded, out_len, numel_in, numel_out],
        4,
    );
    let total = (lines as usize) * (out_len as usize);
    dispatch_1d(encoder, total);
}

/// `values`（`constant uint&` 系スカラー引数）を `start_index` から
/// 連番のバッファインデックスへ順に `setBytes_length_atIndex` で
/// エンコードする（`unique.rs::encode_bitonic_step` の繰り返しパターン
/// を可変長引数向けに一般化）。
fn set_scalar_args(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    values: &[u32],
    start_index: usize,
) {
    for (offset, v) in values.iter().enumerate() {
        // SAFETY: `unique.rs::encode_bitonic_step` と同一の契約。
        // `setBytes_length_atIndex` は指定ポインタから指定バイト数を
        // 即座に複製する。`v` はこの呼び出し中生存し、バイト数は
        // `shaders/sort.metal` 側の `constant uint&` 宣言と一致する。
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(v).cast(),
                std::mem::size_of::<u32>(),
                start_index + offset,
            );
        }
    }
}

fn dispatch_1d(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, total: usize) {
    let threads_per_tg = MTLSize {
        width: SORT_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = total.div_ceil(SORT_THREADGROUP_WIDTH).max(1);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
