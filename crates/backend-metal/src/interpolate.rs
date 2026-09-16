//! 最近傍リサンプリングカーネルの起動 API（イシュー #1757）。
//!
//! [`MetalInterpolate::new`] が `shaders/interpolate.metal`
//! （`interpolate_nearest_f32`）を実行時コンパイルしてパイプラインを
//! 保持し、[`MetalInterpolate::run_nearest_f32`] へホスト側スライスを
//! 渡すだけでバッファ確保・ディスパッチ・readback を内部で完結できる
//! （`crate::gather_scatter::MetalGatherScatter` と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps::interpolate` から呼ばれる。呼び出し元
//! （`ops.rs`）は shape 検査（[`fandhe_ai_tensor_core::
//! interpolate_out_shape`]）・[`crate::interpolate_model::
//! validate_interpolate_launch`] を済ませてから本モジュールを呼ぶが、
//! [`MetalInterpolate::run_nearest_f32`] は `pub` であり `ops.rs` を
//! 経由せず直接呼び出せるため、呼び出し元の検査結果を信頼せず本モジュール
//! 自身でも独立に同じ検査（[`crate::interpolate_model::
//! validate_interpolate_launch`]）を行う（判定迂回経路を作らないための
//! 多層防御。`.claude/rules/security.md` A08。`gather_scatter.rs` と
//! 同じ二重検査方針）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::{ShapeError, bilinear_scale};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::interpolate_model::{validate_interpolate_bilinear_launch, validate_interpolate_launch};
use crate::pipeline::{self, MtlPipeline};

/// `shape` の行優先（row-major）ストライドを `u32` 配列として計算する
/// （`interpolate_model.rs::row_major_strides`〈`usize` 版〉と同じ
/// 定義。カーネル引数 `constant uint*` 用にホスト側で事前計算する。
/// 各 shape 要素は [`validate_interpolate_launch`] が事前に `u32`
/// 収容を検査済みのため `as u32` の切り詰めは発生しない）。
fn row_major_strides_u32(shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides.iter().map(|&s| s as u32).collect()
}

/// [`ShapeError`] を [`MetalError::InvalidInterpolateShape`] へ変換
/// する（本モジュールが独立に行う shape 検査の共通変換ヘルパー）。
fn shape_err_to_metal(err: ShapeError) -> MetalError {
    MetalError::InvalidInterpolateShape {
        detail: err.to_string(),
    }
}

/// `shaders/interpolate.metal` のソース。
const INTERPOLATE_MSL_SRC: &str = include_str!("shaders/interpolate.metal");

/// 1 スレッドグループあたりのスレッド数（`crate::gather_scatter::
/// GS_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const IP_THREADGROUP_WIDTH: usize = 256;

/// interpolate カーネルのコンパイル済みパイプラインを保持するハンドル。
pub struct MetalInterpolate {
    nearest_f32: objc2::rc::Retained<MtlPipeline>,
    /// bilinear パイプライン（イシュー #1762）。`nearest_f32` と同じ
    /// `MetalInterpolate::new` で同時にコンパイル・保持する。
    bilinear_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalInterpolate {
    /// `ctx` のデバイス上でカーネルを実行時コンパイルしパイプラインを
    /// 構築する（`crate::gather_scatter::MetalGatherScatter::new` と
    /// 同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(INTERPOLATE_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let nearest_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "interpolate_nearest_f32")?;
        let bilinear_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "interpolate_bilinear_f32")?;

        Ok(Self {
            nearest_f32,
            bilinear_f32,
        })
    }

    /// `interpolate_nearest_f32` カーネルを起動する（`torch.nn.
    /// functional.interpolate(mode='nearest')` 相当）。
    ///
    /// `ops.rs` を経由しない直接呼び出しでも安全なよう、本関数自身が
    /// 独立に shape を検査する（モジュール冒頭コメント参照。実体は
    /// [`crate::interpolate_model::validate_interpolate_launch`] へ
    /// 切り出し済み）。出力要素数が 0 の場合は空配列を返す
    /// （`MetalBuffer` は 0 バイト確保を拒否するため、デバイス確保前に
    /// この契約上自明な結果を返す。`gather_scatter.rs::
    /// run_gather_f32` の `numel == 0` 早期リターンと同じ理由）。
    pub fn run_nearest_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        in_shape: &[usize],
        size: &[usize],
        out_shape: &[usize],
    ) -> Result<Vec<f32>, MetalError> {
        let numel = validate_interpolate_launch(input, in_shape, size, out_shape)
            .map_err(shape_err_to_metal)?;
        if numel == 0 {
            return Ok(Vec::new());
        }

        let rank = out_shape.len();
        let spatial_start = rank - size.len();
        let in_strides = row_major_strides_u32(in_shape);

        let mut shapes: Vec<u32> = Vec::with_capacity(rank * 3);
        shapes.extend(out_shape.iter().map(|&d| d as u32));
        shapes.extend(in_shape.iter().map(|&d| d as u32));
        shapes.extend(in_strides);

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let shapes_buf = MetalIndexBuffer::new_with_u32(ctx, &shapes)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        let rank_u = rank as u32;
        let spatial_start_u = spatial_start as u32;
        let numel_u = numel as u32;

        ctx.dispatch_sync(|encoder| {
            encode_interpolate_dispatch(
                encoder,
                &self.nearest_f32,
                &input_buf,
                &out_buf,
                &shapes_buf,
                NearestScalars {
                    rank: rank_u,
                    spatial_start: spatial_start_u,
                    numel: numel_u,
                },
            );
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// `interpolate_bilinear_f32` カーネルを起動する（`torch.nn.
    /// functional.interpolate(mode='bilinear', align_corners=…)`
    /// 相当。イシュー #1762）。`out_shape` は `size.len() == 2` を
    /// 満たす契約（[`crate::interpolate_model::
    /// validate_interpolate_bilinear_launch`] が検査）。`scale_h`／
    /// `scale_w` はホスト側（`fandhe_ai_tensor_core::bilinear_scale`。
    /// forward の他バックエンド・ホスト参照実装と共有する単一情報源）
    /// で 1 回だけ計算しカーネル引数として渡す。
    pub fn run_bilinear_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        in_shape: &[usize],
        size: &[usize],
        out_shape: &[usize],
        align_corners: bool,
    ) -> Result<Vec<f32>, MetalError> {
        let numel = validate_interpolate_bilinear_launch(input, in_shape, size, out_shape)
            .map_err(shape_err_to_metal)?;
        if numel == 0 {
            return Ok(Vec::new());
        }

        let rank = out_shape.len();
        let in_strides = row_major_strides_u32(in_shape);

        let mut shapes: Vec<u32> = Vec::with_capacity(rank * 3);
        shapes.extend(out_shape.iter().map(|&d| d as u32));
        shapes.extend(in_shape.iter().map(|&d| d as u32));
        shapes.extend(in_strides);

        let scale_h = bilinear_scale(in_shape[rank - 2], out_shape[rank - 2], align_corners);
        let scale_w = bilinear_scale(in_shape[rank - 1], out_shape[rank - 1], align_corners);

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let shapes_buf = MetalIndexBuffer::new_with_u32(ctx, &shapes)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        let rank_u = rank as u32;
        let align_corners_u: u32 = if align_corners { 1 } else { 0 };
        let numel_u = numel as u32;

        ctx.dispatch_sync(|encoder| {
            encode_interpolate_bilinear_dispatch(
                encoder,
                &self.bilinear_f32,
                &input_buf,
                &out_buf,
                &shapes_buf,
                rank_u,
                align_corners_u,
                numel_u,
                scale_h,
                scale_w,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// `interpolate_nearest_f32` カーネルへ `setBytes` で渡すスカラー引数
/// （バッファ index 3〜5）。`encode_interpolate_dispatch` の引数を
/// 8 個から 6 個へ束ねる（clippy `too_many_arguments` 是正。`#[allow]`
/// で抑止しない方針 `.claude/rules/coding-rust.md`）。フィールド順は
/// バッファ index 順（3: `rank`・4: `spatial_start`・5: `numel`）。
#[derive(Clone, Copy)]
struct NearestScalars {
    rank: u32,
    spatial_start: u32,
    numel: u32,
}

/// `interpolate_nearest_f32` カーネルのエンコード（バッファ index
/// 0〜2・スカラー index 3〜5・ディスパッチ）。`shaders/
/// interpolate.metal::interpolate_nearest_f32` のバッファ宣言と
/// 一致させる。
fn encode_interpolate_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    shapes_buf: &MetalIndexBuffer,
    scalars: NearestScalars,
) {
    let NearestScalars {
        rank,
        spatial_start,
        numel,
    } = scalars;
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`crate::gather_scatter::encode_gather_dispatch` の同種コメント
    // 参照）。各バッファは呼び出し元 `ctx.dispatch_sync` が完了するまで
    // 生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(shapes_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。各ローカル変数は本呼び出し中生存し、
    // 型・バイト数は `shaders/interpolate.metal::interpolate_nearest_f32`
    // の `constant uint&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rank).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&spatial_start).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
    }

    let (threadgroups, threads_per_tg) = ip_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `interpolate_bilinear_f32` カーネルのエンコード（バッファ index
/// 0〜2・スカラー index 3〜5・`float` index 6〜7・ディスパッチ）。
/// `shaders/interpolate.metal::interpolate_bilinear_f32` のバッファ
/// 宣言と一致させる（イシュー #1762）。
#[allow(clippy::too_many_arguments)]
fn encode_interpolate_bilinear_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    shapes_buf: &MetalIndexBuffer,
    rank: u32,
    align_corners: u32,
    numel: u32,
    scale_h: f32,
    scale_w: f32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`encode_interpolate_dispatch` と同じ理由
    // （`setBuffer_offset_atIndex` は生存中の `MTLBuffer` への参照を
    // 保持するのみ・各バッファは `ctx.dispatch_sync` が完了するまで
    // 生存する）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(shapes_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。各ローカル変数は本呼び出し中生存し、
    // 型・バイト数は `shaders/interpolate.metal::interpolate_bilinear_f32`
    // の `constant uint&`／`constant float&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rank).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&align_corners).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&scale_h).cast(),
            std::mem::size_of::<f32>(),
            6,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&scale_w).cast(),
            std::mem::size_of::<f32>(),
            7,
        );
    }

    let (threadgroups, threads_per_tg) = ip_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `numel` に対する grid/threadgroup サイズを構築する（`crate::
/// gather_scatter::gs_dispatch_sizes` と同一構成。`div_ceil` による末尾
/// ブロックの余剰スレッドはカーネル内境界チェックに委ねる契約。REQ-8）。
fn ip_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: IP_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(IP_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}
