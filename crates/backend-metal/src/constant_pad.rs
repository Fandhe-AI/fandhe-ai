//! 定数パディングカーネルの起動 API（イシュー #1756）。
//!
//! [`MetalConstantPad::new`] が `shaders/constant_pad.metal`
//! （`constant_pad_f32`）を実行時コンパイルしてパイプラインを保持し、
//! [`MetalConstantPad::run_pad_f32`] へホスト側スライスを渡すだけで
//! バッファ確保・ディスパッチ・readback を内部で完結できる
//! （`crate::gather_scatter::MetalGatherScatter` と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps::pad` から呼ばれる。呼び出し元（`ops.rs`）は
//! shape 検査（[`fandhe_ai_tensor_core::pad_out_shape`]）・
//! [`crate::constant_pad_model::validate_pad_launch`] を済ませてから
//! 本モジュールを呼ぶが、[`MetalConstantPad::run_pad_f32`] は `pub` で
//! あり `ops.rs` を経由せず直接呼び出せるため、呼び出し元の検査結果を
//! 信頼せず本モジュール自身でも独立に同じ検査
//! （[`crate::constant_pad_model::validate_pad_launch`]）を行う
//! （判定迂回経路を作らないための多層防御。`.claude/rules/security.md`
//! A08。`gather_scatter.rs` と同じ二重検査方針）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::ShapeError;

use crate::buffer::MetalBuffer;
use crate::constant_pad_model::validate_pad_launch;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};

/// `shape` の行優先（row-major）ストライドを `u32` 配列として計算する
/// （`constant_pad_model.rs::row_major_strides`〈`usize` 版〉と同じ定義。
/// カーネル引数 `constant uint*` 用にホスト側で事前計算する——
/// `shaders/gather_scatter.metal` がカーネル内で毎回 `gs_ravel` を計算
/// するのと異なり、pad は `dim` 軸限定ではなく全軸が対象のため、
/// ストライドをホスト側で 1 回だけ計算して渡す設計とした。各 shape
/// 要素は [`validate_pad_launch`] が事前に `u32` 収容を検査済みのため
/// `as u32` の切り詰めは発生しない）。
fn row_major_strides_u32(shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides.iter().map(|&s| s as u32).collect()
}

/// [`ShapeError`] を [`MetalError::InvalidConstantPadShape`] へ変換する
/// （本モジュールが独立に行う shape 検査の共通変換ヘルパー）。
fn shape_err_to_metal(err: ShapeError) -> MetalError {
    MetalError::InvalidConstantPadShape {
        detail: err.to_string(),
    }
}

/// `shaders/constant_pad.metal` のソース。
const CONSTANT_PAD_MSL_SRC: &str = include_str!("shaders/constant_pad.metal");

/// 1 スレッドグループあたりのスレッド数（`crate::gather_scatter::
/// GS_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const CP_THREADGROUP_WIDTH: usize = 256;

/// pad カーネルのコンパイル済みパイプラインを保持するハンドル。
pub struct MetalConstantPad {
    pad_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalConstantPad {
    /// `ctx` のデバイス上でカーネルを実行時コンパイルしパイプラインを
    /// 構築する（`crate::gather_scatter::MetalGatherScatter::new` と
    /// 同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(CONSTANT_PAD_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let pad_f32 = pipeline::make_pipeline(ctx.device(), &library, "constant_pad_f32")?;

        Ok(Self { pad_f32 })
    }

    /// `constant_pad_f32` カーネルを起動する（`torch.nn.functional.pad
    /// (mode='constant')` 相当）。
    ///
    /// `ops.rs` を経由しない直接呼び出しでも安全なよう、本関数自身が
    /// 独立に shape を検査する（モジュール冒頭コメント参照。実体は
    /// [`crate::constant_pad_model::validate_pad_launch`] へ切り出し
    /// 済み）。出力要素数が 0 の場合は空配列を、`in_shape` が空
    /// （出力は非空）の場合は `input` を読まず全域 `value` で埋めた
    /// 配列を返す（`MetalBuffer` は 0 バイト確保を拒否するため、デバイス
    /// 確保前にこれらの契約上自明な結果を返す。`gather_scatter.rs::
    /// run_scatter_f32` の `idx_numel == 0` 早期リターンと同じ理由）。
    pub fn run_pad_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        in_shape: &[usize],
        pads: &[(usize, usize)],
        out_shape: &[usize],
        value: f32,
    ) -> Result<Vec<f32>, MetalError> {
        let numel =
            validate_pad_launch(input, in_shape, pads, out_shape).map_err(shape_err_to_metal)?;
        if numel == 0 {
            return Ok(Vec::new());
        }
        let in_is_empty = in_shape.contains(&0);
        if in_is_empty {
            return Ok(vec![value; numel]);
        }

        let rank = out_shape.len();
        if rank == 0 {
            // rank 0（スカラー）は `pads` も空のため恒等コピー
            // （`eval::pad`／CPU `constant_pad::pad` の同型早期分岐と
            // 対称）。`shapes` 配列がここで空（`rank * 4 == 0`）になり
            // `MetalIndexBuffer::new_with_u32` が空スライスに対して
            // `MetalError::ZeroLengthAllocation` を返してしまう
            // （バッファ確保前に検査済みの `numel == 1` を使い切り
            // カーネル起動自体を回避する。PR #1831 codex-review P2
            // 是正）。`validate_pad_launch` が `input.len() ==
            // checked_numel(in_shape) == 1` を既に保証しているため
            // `input.to_vec()` は必ず 1 要素の `Vec` になる。
            return Ok(input.to_vec());
        }
        let in_strides = row_major_strides_u32(in_shape);

        let mut shapes: Vec<u32> = Vec::with_capacity(rank * 4);
        shapes.extend(out_shape.iter().map(|&d| d as u32));
        shapes.extend(in_shape.iter().map(|&d| d as u32));
        shapes.extend(pads.iter().map(|&(before, _after)| before as u32));
        shapes.extend(in_strides);

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let shapes_buf = MetalIndexBuffer::new_with_u32(ctx, &shapes)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        let rank_u = rank as u32;
        let numel_u = numel as u32;

        ctx.dispatch_sync(|encoder| {
            encode_pad_dispatch(
                encoder,
                &self.pad_f32,
                &input_buf,
                &out_buf,
                &shapes_buf,
                rank_u,
                numel_u,
                value,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// `constant_pad_f32` カーネルのエンコード（バッファ index 0〜2・
/// スカラー index 3〜5・ディスパッチ）。`shaders/constant_pad.metal::
/// constant_pad_f32` のバッファ宣言と一致させる。
fn encode_pad_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    shapes_buf: &MetalIndexBuffer,
    rank: u32,
    numel: u32,
    value: f32,
) {
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
    // 型・バイト数は `shaders/constant_pad.metal::constant_pad_f32` の
    // `constant uint&`／`constant float&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rank).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&value).cast(),
            std::mem::size_of::<f32>(),
            5,
        );
    }

    let (threadgroups, threads_per_tg) = pad_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `numel` に対する grid/threadgroup サイズを構築する（`crate::
/// gather_scatter::gs_dispatch_sizes` と同一構成。`div_ceil` による末尾
/// ブロックの余剰スレッドはカーネル内境界チェックに委ねる契約。REQ-8）。
fn pad_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: CP_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(CP_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}
