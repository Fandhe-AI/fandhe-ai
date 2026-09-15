//! MaxPool／AvgPool／AdaptiveAvgPool（1d／2d）の起動 API（イシュー
//! #1730・親 #1607・設計 `docs/pooling-ops-design.md`）。
//!
//! `batch_norm.rs`・`im2col.rs` と同じ構成方針: [`MetalPooling::new`]
//! が `shaders/pooling.metal` の 3 カーネルを実行時コンパイルして
//! パイプラインを保持し、[`MetalPooling::run_max_pool2d_f32`]／
//! [`MetalPooling::run_avg_pool2d_f32`]／[`MetalPooling::
//! run_adaptive_avg_pool2d_f32`] へホスト側スライスを渡すだけで
//! バッファ確保・ディスパッチ・readback を内部で完結できる。1d は
//! 呼び出し元が `[N, C, 1, L]` へ reshape 併合してから渡す契約
//! （本モジュールは常に rank 4 の `PoolDims` を扱う）。
//!
//! **`ops::MetalBackendOps::{max_pool2d, avg_pool2d,
//! adaptive_avg_pool2d}` への結線（Layer B）は本 PR 時点では未実施**
//! （兄弟イシュー #1728〈CPU〉が `Pool2dParams`／`BackendOps` の
//! シグネチャを確定させるまで、並列実行される本イシューが `ops.rs`
//! へ推測実装を加えると重複実装・rebase 衝突を招くため。
//! `docs/pooling-ops-design.md` §15「Metal 実装（イシュー #1730）」
//! 参照）。
//!
//! **shape 検証の責務分担**（`im2col.rs` モジュール doc・
//! `.claude/rules/security.md` A08 と同じ二重検査方針）: 本モジュール
//! 自身が [`crate::pooling_model::{derive_pool_dims,
//! derive_adaptive_dims}`] で形状を独立に再検証する（呼び出し元の
//! 検査結果を信頼しない）。ホストスライスの実長検査も独立に行う。
//!
//! **数値契約**（[`shaders/pooling.metal`](shaders/pooling.metal)
//! 冒頭コメント参照）: `max_pool2d_f32` は選択演算のみのため CPU
//! 参照実装と bit 完全一致（値・索引とも）。`avg_pool2d_f32`／
//! `adaptive_avg_pool2d_f32` は soft-f64 逐次加算・除算により CPU
//! `f64` 参照実装と bit 完全一致。

use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};
use crate::pooling_model::{self, PoolDims, PoolingPrepareError};

/// `shaders/pooling.metal` のソース（3 カーネルを含む単一ファイル）。
const POOLING_MSL_SRC: &str = include_str!("shaders/pooling.metal");

/// 1 スレッドグループあたりのスレッド数（`im2col.rs::
/// IM2COL_THREADGROUP_WIDTH` と同じ値・同じ判断根拠。grid-stride 不要
/// の単純 1 スレッド = 1 出力位置カーネル）。
const POOLING_THREADGROUP_WIDTH: usize = 256;

/// [`PoolDims`] のバイト数を検証する場所を 1 箇所に集約するための
/// `size_of` 定数（`im2col.rs::IM2COL_DIMS_SIZE` と同型）。
const POOL_DIMS_SIZE: usize = std::mem::size_of::<PoolDims>();

/// [`PoolingPrepareError`] を [`MetalError`] へ変換する（3 つの
/// `run_*` 共通。`SizeLimitExceeded` と `InvalidShape` の区別を保った
/// まま伝播させる——Layer B 実装時に `ops.rs::map_pooling_error` が
/// この区別を使って `Unsupported`（ホストフォールバック）／
/// `ShapeMismatch` を振り分ける想定。`im2col.rs::map_prepare_error`
/// と同型）。
fn map_prepare_error(err: PoolingPrepareError) -> MetalError {
    match err {
        PoolingPrepareError::SizeLimitExceeded { .. } => MetalError::PoolingSizeLimitExceeded {
            detail: err.to_string(),
        },
        PoolingPrepareError::InvalidShape { .. } => MetalError::InvalidPoolingShape {
            detail: err.to_string(),
        },
    }
}

/// ホストスライスの実長が `dims` の入力要素数と一致するかを検証する
/// （`im2col.rs::checked_numel`／実長検査と同じ多層防御。`pub` な
/// 起動 API を経由せず直接呼ばれても安全にする）。
fn check_input_len(x: &[f32], dims: &PoolDims) -> Result<(), MetalError> {
    pooling_model::validate_input_len(x.len(), dims).map_err(map_prepare_error)
}

/// MaxPool／AvgPool／AdaptiveAvgPool 3 カーネルのコンパイル済み
/// パイプラインを保持するハンドル。
pub struct MetalPooling {
    max_pool2d_f32: objc2::rc::Retained<MtlPipeline>,
    avg_pool2d_f32: objc2::rc::Retained<MtlPipeline>,
    adaptive_avg_pool2d_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalPooling {
    /// `ctx` のデバイス上で 3 カーネルを実行時コンパイルしパイプライン
    /// を構築する（`im2col.rs::MetalIm2col::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(POOLING_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let max_pool2d_f32 = pipeline::make_pipeline(ctx.device(), &library, "max_pool2d_f32")?;
        let avg_pool2d_f32 = pipeline::make_pipeline(ctx.device(), &library, "avg_pool2d_f32")?;
        let adaptive_avg_pool2d_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "adaptive_avg_pool2d_f32")?;

        Ok(Self {
            max_pool2d_f32,
            avg_pool2d_f32,
            adaptive_avg_pool2d_f32,
        })
    }

    /// MaxPool2d（`in_shape: [N, C, H, W]`）を実行する。
    /// `derive_pool_dims`（`docs/pooling-ops-design.md` §3／§4）で
    /// 独立に形状を再検証し、`N == 0 || C == 0` は device 非接触で
    /// 空 `Vec` を返す（`h_in`／`w_in` はゼロ長禁止のため到達しない）。
    /// 戻り値は `(値, 索引)` の対（索引は入力平面内 row-major 添字
    /// `h*w_in + w`）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_max_pool2d_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        in_shape: &[usize],
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<(Vec<f32>, Vec<i32>), MetalError> {
        let dims = pooling_model::derive_pool_dims(
            in_shape, kernel, stride, padding, dilation, false, true,
        )
        .map_err(map_prepare_error)?;
        check_input_len(x, &dims)?;

        if dims.numel_out == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, dims.numel_out as usize)?;
        let idx_buf = MetalIndexBuffer::new_zeroed_i32(ctx, dims.numel_out as usize)?;

        ctx.dispatch_sync(|encoder| {
            encode_max_pool2d_dispatch(
                encoder,
                &self.max_pool2d_f32,
                &x_buf,
                &out_buf,
                &idx_buf,
                &dims,
            );
        })?;

        Ok((out_buf.read_to_vec(), idx_buf.read_to_vec_i32()))
    }

    /// AvgPool2d（`in_shape: [N, C, H, W]`）を実行する。
    /// `count_include_pad` は padding 位置を divisor（`kh*kw`）に
    /// 含めるかどうか（`false` なら有効タップ数のみ）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_avg_pool2d_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        in_shape: &[usize],
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Vec<f32>, MetalError> {
        let dims = pooling_model::derive_pool_dims(
            in_shape,
            kernel,
            stride,
            padding,
            dilation,
            false,
            count_include_pad,
        )
        .map_err(map_prepare_error)?;
        check_input_len(x, &dims)?;

        if dims.numel_out == 0 {
            return Ok(Vec::new());
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, dims.numel_out as usize)?;

        ctx.dispatch_sync(|encoder| {
            encode_avg_pool2d_dispatch(encoder, &self.avg_pool2d_f32, &x_buf, &out_buf, &dims);
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// AdaptiveAvgPool2d（`in_shape: [N, C, H, W]`）を実行する。
    /// `output_size` は入力長以下であることを要求しない。
    pub fn run_adaptive_avg_pool2d_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        in_shape: &[usize],
        output_size: (usize, usize),
    ) -> Result<Vec<f32>, MetalError> {
        let dims = pooling_model::derive_adaptive_dims(in_shape, output_size)
            .map_err(map_prepare_error)?;
        check_input_len(x, &dims)?;

        if dims.numel_out == 0 {
            return Ok(Vec::new());
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, dims.numel_out as usize)?;

        ctx.dispatch_sync(|encoder| {
            encode_adaptive_avg_pool2d_dispatch(
                encoder,
                &self.adaptive_avg_pool2d_f32,
                &x_buf,
                &out_buf,
                &dims,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// 3 カーネル共通のディスパッチサイズ計算（`dims.numel_out` を
/// [`POOLING_THREADGROUP_WIDTH`] で `div_ceil` する。REQ-8: 末尾
/// ブロックの余剰スレッドはカーネル内境界チェック〈`gid >=
/// dims.numel_out`〉に委ねる契約）。
fn pooling_dispatch_sizes(numel_out: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: POOLING_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel_out as usize).div_ceil(POOLING_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups.max(1),
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// `max_pool2d_f32` カーネルのエンコード（バッファ index 0〜2・
/// `PoolDims` index 3。`shaders/pooling.metal::max_pool2d_f32` の
/// バッファ宣言と一致させる）。
fn encode_max_pool2d_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    idx_buf: &MetalIndexBuffer,
    dims: &PoolDims,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2（`im2col.rs::encode_im2col_dispatch` と同じ
    // 契約）。`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への
    // 参照を保持するのみで即座に読み書きしない。各バッファは呼び出し
    // 元 `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(idx_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から `size_of::<PoolDims>()` バイトを即座に複製する。`dims` は
    // 本呼び出し中生存するローカル参照で、長さ・レイアウトは
    // `shaders/pooling.metal::struct PoolDims`（18 × uint）と一致
    // させている（`pooling_model.rs::pool_dims_size_matches_msl_struct`
    // が Linux で機械検証）。
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(dims).cast(), POOL_DIMS_SIZE, 3);
    }

    let (threadgroups, threads_per_tg) = pooling_dispatch_sizes(dims.numel_out);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `avg_pool2d_f32` カーネルのエンコード（バッファ index 0〜1・
/// `PoolDims` index 2）。
fn encode_avg_pool2d_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    dims: &PoolDims,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_max_pool2d_dispatch` と同じ契約。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: `encode_max_pool2d_dispatch` と同じ契約。
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(dims).cast(), POOL_DIMS_SIZE, 2);
    }

    let (threadgroups, threads_per_tg) = pooling_dispatch_sizes(dims.numel_out);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `adaptive_avg_pool2d_f32` カーネルのエンコード（`encode_avg_pool2d_dispatch`
/// と同型。カーネルが異なるのみ）。
fn encode_adaptive_avg_pool2d_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    dims: &PoolDims,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_max_pool2d_dispatch` と同じ契約。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: `encode_max_pool2d_dispatch` と同じ契約。
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(dims).cast(), POOL_DIMS_SIZE, 2);
    }

    let (threadgroups, threads_per_tg) = pooling_dispatch_sizes(dims.numel_out);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_msl_source_declares_all_kernels() {
        assert!(POOLING_MSL_SRC.contains("kernel void max_pool2d_f32("));
        assert!(POOLING_MSL_SRC.contains("kernel void avg_pool2d_f32("));
        assert!(POOLING_MSL_SRC.contains("kernel void adaptive_avg_pool2d_f32("));
    }
}
