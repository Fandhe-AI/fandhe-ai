//! Conv2d の im2col／col2im 起動 API（実行時コンパイル・保持・実行。
//! イシュー #1768・親 #1644・設計 `docs/conv-ops-design.md`）。
//!
//! `constant_pad.rs::MetalConstantPad`・`scan.rs::MetalScan` と同じ
//! 構成方針を踏襲する: [`MetalIm2col::new`] が `shaders/im2col.metal`
//! （`im2col_f32`／`col2im_f32`）を実行時コンパイルしてパイプラインを
//! 保持し、[`MetalIm2col::run_im2col_f32`]／[`MetalIm2col::
//! run_col2im_f32`] へホスト側スライスを渡すだけでバッファ確保・
//! ディスパッチ・readback を内部で完結できる。`ops.rs::
//! MetalBackendOps::im2col`／`col2im` から呼ばれる。
//!
//! **shape 検証の責務分担**（`constant_pad.rs` モジュール doc・
//! `.claude/rules/security.md` A08 と同じ二重検査方針）: 呼び出し元
//! `ops.rs` が [`fandhe_ai_tensor_core::im2col_out_shape`] で
//! `input.shape()`／`params`（`im2col`）または `d_col.shape()`／
//! `input_shape`／`params`（`col2im`）を再検査してから本モジュールへ
//! 委譲する契約だが、`pub` な起動 API を `ops.rs` を経由せず直接
//! 呼び出す経路でも安全なよう、本モジュール自身も独立に
//! [`crate::im2col_model::derive_im2col_dims`] で形状を再検証する
//! （呼び出し元の検査結果を信頼しない。`gather_scatter.rs`／
//! `interpolate.rs` と同じ多層防御）。ホストスライスの実長検査も
//! 独立に行う（スライスが短いと GPU 側で範囲外読み出しになりうる
//! ため、バッファ確保前の必須の安全策）。
//!
//! **数値契約**（[`shaders/im2col.metal`](../shaders/im2col.metal)
//! 冒頭コメント参照）: `im2col_f32` は算術を含まない純粋コピーのため
//! CPU 参照実装と bit 完全一致。`col2im_f32` は binary64 ソフトウェア
//! エミュレーションアキュムレータへの逐次加算で CPU `f64` 逐次和と
//! bit 完全一致。

use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::Conv2dParams;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::im2col_model::{self, Im2colDims};
use crate::pipeline::{self, MtlPipeline};

/// `shaders/im2col.metal` のソース。
const IM2COL_MSL_SRC: &str = include_str!("shaders/im2col.metal");

/// 1 スレッドグループあたりのスレッド数（`scan.rs::
/// SCAN_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const IM2COL_THREADGROUP_WIDTH: usize = 256;

/// [`Im2colDims`] のバイト数を検証する場所を 1 箇所に集約するための
/// `size_of` 定数。
const IM2COL_DIMS_SIZE: usize = std::mem::size_of::<Im2colDims>();

/// [`crate::im2col_model::Im2colPrepareError`] を [`MetalError`] へ
/// 変換する（[`MetalIm2col::run_im2col_f32`]／[`MetalIm2col::
/// run_col2im_f32`] 共通。`SizeLimitExceeded` と `InvalidShape` の
/// 区別を保ったまま伝播させる——呼び出し元 `ops.rs::map_im2col_error`
/// がこの区別を使って `Unsupported`（ホストフォールバック）／
/// `ShapeMismatch` を振り分ける。`.claude/rules/security.md` A08）。
fn map_prepare_error(err: im2col_model::Im2colPrepareError) -> MetalError {
    match err {
        im2col_model::Im2colPrepareError::SizeLimitExceeded { .. } => {
            MetalError::Im2colSizeLimitExceeded {
                detail: err.to_string(),
            }
        }
        im2col_model::Im2colPrepareError::InvalidShape { .. } => MetalError::InvalidIm2colShape {
            detail: err.to_string(),
        },
    }
}

/// im2col／col2im 2 カーネルのコンパイル済みパイプラインを保持する
/// ハンドル。
pub struct MetalIm2col {
    im2col_f32: objc2::rc::Retained<MtlPipeline>,
    col2im_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalIm2col {
    /// `ctx` のデバイス上で 2 カーネルを実行時コンパイルしパイプライン
    /// を構築する（`scan.rs::MetalScan::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(IM2COL_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let im2col_f32 = pipeline::make_pipeline(ctx.device(), &library, "im2col_f32")?;
        let col2im_f32 = pipeline::make_pipeline(ctx.device(), &library, "col2im_f32")?;

        Ok(Self {
            im2col_f32,
            col2im_f32,
        })
    }

    /// Conv2d の im2col（`shaders/im2col.metal` 冒頭コメント参照）。
    /// `in_shape: [N, Cin, H, W]`・`out_shape: [N, G, K_g, P]`
    /// （呼び出し元 `ops.rs` が `im2col_out_shape` で検査・確定済み。
    /// モジュール doc「shape 検証の責務分担」参照）。
    ///
    /// `out_shape` の要素数が 0 の場合、それは `in_shape` のいずれかの
    /// 軸が 0（`N`／`Cin`。空間軸 `H`／`W` は `im2col_out_shape` が
    /// 事前に拒否する契約）であることに起因し `input` を読む必要が
    /// 一切ないため、空 `Vec` を早期に返す（`MetalBuffer` は 0 バイト
    /// 確保を拒否するため、確保・ディスパッチに入る前に返す。
    /// `scan.rs::run_scan` の空出力早期リターンと同じ理由）。
    pub fn run_im2col_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Vec<f32>, MetalError> {
        let numel_out: usize = out_shape.iter().product();
        if numel_out == 0 {
            return Ok(Vec::new());
        }
        let numel_in: usize = in_shape.iter().product();
        if input.len() != numel_in {
            return Err(MetalError::InvalidIm2colShape {
                detail: format!(
                    "im2col: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let dims = im2col_model::derive_im2col_dims(in_shape, out_shape, params, numel_out)
            .map_err(map_prepare_error)?;

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel_out)?;

        ctx.dispatch_sync(|encoder| {
            encode_im2col_dispatch(encoder, &self.im2col_f32, &input_buf, &out_buf, &dims);
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// Conv2d の col2im（[`Self::run_im2col_f32`] の随伴。
    /// `shaders/im2col.metal` 冒頭コメント参照）。`d_col: [N, G, K_g,
    /// P]`・`input_shape: [N, Cin, H, W]`（呼び出し元 `ops.rs` が
    /// `d_col.shape()` との完全一致を再検査済み）。
    ///
    /// `input_shape` の要素数が 0 の場合、空 `Vec` を早期に返す
    /// （`d_col` を読む必要が一切ない。[`Self::run_im2col_f32`] と同じ
    /// 早期リターン契約）。
    pub fn run_col2im_f32(
        &self,
        ctx: &MetalContext,
        d_col: &[f32],
        col_shape: &[usize],
        input_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Vec<f32>, MetalError> {
        let numel_out: usize = input_shape.iter().product();
        if numel_out == 0 {
            return Ok(Vec::new());
        }
        let numel_col: usize = col_shape.iter().product();
        if d_col.len() != numel_col {
            return Err(MetalError::InvalidIm2colShape {
                detail: format!(
                    "col2im: d_col.len()={} does not match col numel={numel_col}",
                    d_col.len()
                ),
            });
        }

        let dims = im2col_model::derive_im2col_dims(input_shape, col_shape, params, numel_out)
            .map_err(map_prepare_error)?;

        let d_col_buf = MetalBuffer::new_with_data(ctx, d_col)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel_out)?;

        ctx.dispatch_sync(|encoder| {
            encode_col2im_dispatch(encoder, &self.col2im_f32, &d_col_buf, &out_buf, &dims);
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// im2col／col2im カーネル共通のディスパッチサイズ計算（`dims.numel`
/// を [`IM2COL_THREADGROUP_WIDTH`] で `div_ceil` する。REQ-8: 末尾
/// ブロックの余剰スレッドはカーネル内境界チェック〈`gid >=
/// dims.numel`〉に委ねる契約）。
fn im2col_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: IM2COL_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(IM2COL_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// `im2col_f32` カーネルのエンコード（バッファ index 0〜1・
/// `Im2colDims` index 2・ディスパッチ）。`shaders/im2col.metal::
/// im2col_f32` のバッファ宣言と一致させる。
fn encode_im2col_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    dims: &Im2colDims,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2（`scan.rs::encode_scan_dispatch` と同じ
    // 契約）。`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への
    // 参照を保持するのみで即座に読み書きしない。各バッファは呼び出し
    // 元 `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から `size_of::<Im2colDims>()` バイトを即座に複製する。`dims`
    // は本呼び出し中生存するローカル参照で、長さ・レイアウトは
    // `shaders/im2col.metal::struct Im2colDims`（19 × uint）と
    // 一致させている（`im2col_model.rs::
    // im2col_dims_size_matches_msl_struct` が Linux で機械検証）。
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(dims).cast(), IM2COL_DIMS_SIZE, 2);
    }

    let (threadgroups, threads_per_tg) = im2col_dispatch_sizes(dims.numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `col2im_f32` カーネルのエンコード（[`encode_im2col_dispatch`] と
/// 同型。カーネル・入力バッファの役割名のみ異なる）。
fn encode_col2im_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    d_col_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    dims: &Im2colDims,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_im2col_dispatch` と同じ契約。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(d_col_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: `encode_im2col_dispatch` と同じ契約。
    unsafe {
        encoder.setBytes_length_atIndex(NonNull::from(dims).cast(), IM2COL_DIMS_SIZE, 2);
    }

    let (threadgroups, threads_per_tg) = im2col_dispatch_sizes(dims.numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn im2col_msl_source_declares_both_kernels() {
        assert!(IM2COL_MSL_SRC.contains("kernel void im2col_f32("));
        assert!(IM2COL_MSL_SRC.contains("kernel void col2im_f32("));
    }
}
