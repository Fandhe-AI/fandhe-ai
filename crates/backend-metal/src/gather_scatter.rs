//! gather／scatter／scatter_add カーネルの起動 API（イシュー #1778）。
//!
//! [`MetalGatherScatter::new`] が `shaders/gather_scatter.metal`（3
//! カーネル: `gather_f32`／`scatter_overwrite_f32`／`scatter_add_f32`）を
//! 実行時コンパイルしてパイプラインを保持し、[`MetalGatherScatter::
//! run_gather_f32`]／[`MetalGatherScatter::run_scatter_f32`] へホスト側
//! スライスを渡すだけでバッファ確保・ディスパッチ・readback を内部で
//! 完結できる（`crate::elementwise::MetalElementwise` と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps::gather`／`scatter` から呼ばれる。呼び出し元
//! （`ops.rs`）が shape 検査（[`fandhe_ai_tensor_core::gather_out_shape`]／
//! [`fandhe_ai_tensor_core::scatter_out_shape`]）・
//! [`crate::gather_scatter_model::validate_shapes_fit_u32`]・
//! [`crate::gather_scatter_model::validate_index_range`] を済ませてから
//! 本モジュールを呼ぶ契約のため、shape 次元・index 範囲の検査は本
//! モジュールでは行わずディスパッチに専念する（カーネル側には防御的
//! ガードを残す。`shaders/gather_scatter.metal` 冒頭コメント参照）。
//! ただし `numel`（形状次元の積）が `u32` カーネル引数へ収まるかの
//! 検査（[`validate_gather_scatter_len`]）は上記呼び出し元検査ではまだ
//! 担保されないため本モジュール側で行う（`crate::elementwise::
//! validate_elementwise_len` と同じ理由）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::ScatterReduce;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/gather_scatter.metal` のソース（3 カーネルを含む）。
const GATHER_SCATTER_MSL_SRC: &str = include_str!("shaders/gather_scatter.metal");

/// 1 スレッドグループあたりのスレッド数（`crate::elementwise::
/// EW_THREADGROUP_WIDTH` と同じ値・同じ判断根拠: threadgroup 共有メモリを
/// 使わないためオキュパンシ最適化のみが関心事。チューニングは別イシューの
/// スコープ・`.claude/rules/out-of-scope-tracking.md` 対象）。
const GS_THREADGROUP_WIDTH: usize = 256;

/// gather／scatter 3 カーネルのコンパイル済みパイプラインを保持する
/// ハンドル。
pub struct MetalGatherScatter {
    gather_f32: objc2::rc::Retained<MtlPipeline>,
    scatter_overwrite_f32: objc2::rc::Retained<MtlPipeline>,
    scatter_add_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalGatherScatter {
    /// `ctx` のデバイス上で 3 カーネルを実行時コンパイルしパイプラインを
    /// 構築する（`crate::elementwise::MetalElementwise::new` と同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(GATHER_SCATTER_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let gather_f32 = pipeline::make_pipeline(ctx.device(), &library, "gather_f32")?;
        let scatter_overwrite_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "scatter_overwrite_f32")?;
        let scatter_add_f32 = pipeline::make_pipeline(ctx.device(), &library, "scatter_add_f32")?;

        Ok(Self {
            gather_f32,
            scatter_overwrite_f32,
            scatter_add_f32,
        })
    }

    /// `gather_f32` カーネルを起動する（`torch.gather` 相当）。
    ///
    /// 呼び出し前提（本モジュール冒頭コメント参照）: `input`／`index` は
    /// 既に shape・値検査済み。`out_numel == index_shape` の要素数積が
    /// 0 の場合は空配列を返す（`MetalBuffer` は 0 バイト確保を拒否する
    /// ため、デバイス確保前に早期リターンする）。
    pub fn run_gather_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        in_shape: &[usize],
        index: &[i32],
        index_shape: &[usize],
        dim: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let numel: usize = index_shape.iter().product();
        if numel == 0 {
            return Ok(Vec::new());
        }
        validate_gather_scatter_len(numel)?;

        let rank = in_shape.len();
        let mut shapes: Vec<u32> = Vec::with_capacity(rank * 2);
        shapes.extend(in_shape.iter().map(|&d| d as u32));
        shapes.extend(index_shape.iter().map(|&d| d as u32));

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let index_buf = MetalIndexBuffer::new_with_i32(ctx, index)?;
        let shapes_buf = MetalIndexBuffer::new_with_u32(ctx, &shapes)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        let rank_u = rank as u32;
        let dim_u = dim as u32;
        let numel_u = numel as u32;

        ctx.dispatch_sync(|encoder| {
            encode_gather_dispatch(
                encoder,
                &self.gather_f32,
                &input_buf,
                &index_buf,
                &out_buf,
                &shapes_buf,
                rank_u,
                dim_u,
                numel_u,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// `scatter_overwrite_f32`／`scatter_add_f32` カーネルを起動する
    /// （`torch.scatter`／`torch.scatter_add` 相当。`reduce` で選択）。
    ///
    /// 呼び出し前提は [`Self::run_gather_f32`] と同じ。`out_shape`
    /// （＝`input.shape()`）の要素数積が 0 の場合は空配列を返す。
    #[allow(clippy::too_many_arguments)]
    pub fn run_scatter_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        out_shape: &[usize],
        index: &[i32],
        index_shape: &[usize],
        src: &[f32],
        dim: usize,
        reduce: ScatterReduce,
    ) -> Result<Vec<f32>, MetalError> {
        let numel_out: usize = out_shape.iter().product();
        if numel_out == 0 {
            return Ok(Vec::new());
        }
        validate_gather_scatter_len(numel_out)?;

        let rank = out_shape.len();
        let mut shapes: Vec<u32> = Vec::with_capacity(rank * 2);
        shapes.extend(out_shape.iter().map(|&d| d as u32));
        shapes.extend(index_shape.iter().map(|&d| d as u32));

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let index_buf = MetalIndexBuffer::new_with_i32(ctx, index)?;
        let src_buf = MetalBuffer::new_with_data(ctx, src)?;
        let shapes_buf = MetalIndexBuffer::new_with_u32(ctx, &shapes)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel_out)?;

        let rank_u = rank as u32;
        let dim_u = dim as u32;
        let numel_out_u = numel_out as u32;

        // `Overwrite`、および `ScatterReduce`（`#[non_exhaustive]`）の
        // 未知 variant は同じ「上書き」意味論へフォールバックする
        // （CPU 参照実装・ホストモデルと同じ安全側の割り切り方針。
        // `crates/backend-cpu/src/gather_scatter.rs::scatter` の
        // `debug_assert!` 併記方針と同じくここでも明示する）。
        let pipeline = match reduce {
            ScatterReduce::Add => &self.scatter_add_f32,
            reduce => {
                debug_assert!(
                    matches!(reduce, ScatterReduce::Overwrite),
                    "scatter: 未知の ScatterReduce variant へフォールバックした（契約違反）"
                );
                &self.scatter_overwrite_f32
            }
        };

        ctx.dispatch_sync(|encoder| {
            encode_scatter_dispatch(
                encoder,
                pipeline,
                &input_buf,
                &index_buf,
                &src_buf,
                &out_buf,
                &shapes_buf,
                rank_u,
                dim_u,
                numel_out_u,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// gather カーネルのエンコード（バッファ index 0〜3・スカラー index
/// 4〜6・ディスパッチ）。`shaders/gather_scatter.metal::gather_f32` の
/// バッファ宣言と一致させる。
#[allow(clippy::too_many_arguments)]
fn encode_gather_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    index_buf: &MetalIndexBuffer,
    out_buf: &MetalBuffer,
    shapes_buf: &MetalIndexBuffer,
    rank: u32,
    dim: u32,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`crate::elementwise::encode_binary_dispatch` の同種コメント
    // 参照）。各バッファは呼び出し元 `ctx.dispatch_sync` が完了するまで
    // 生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(index_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(shapes_buf.raw()), 0, 3);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。各ローカル変数は本呼び出し中生存し、
    // 型・バイト数は `shaders/gather_scatter.metal::gather_f32` の
    // `constant uint&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rank).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dim).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            6,
        );
    }

    let (threadgroups, threads_per_tg) = gs_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// scatter カーネル（`Overwrite`／`Add` 共通）のエンコード（バッファ
/// index 0〜4・スカラー index 5〜7・ディスパッチ）。
/// `shaders/gather_scatter.metal::scatter_overwrite_f32`／
/// `scatter_add_f32` のバッファ宣言と一致させる。
#[allow(clippy::too_many_arguments)]
fn encode_scatter_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    index_buf: &MetalIndexBuffer,
    src_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    shapes_buf: &MetalIndexBuffer,
    rank: u32,
    dim: u32,
    numel_out: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_gather_dispatch` と同一の根拠（該当コメント参照）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(index_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(src_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(shapes_buf.raw()), 0, 4);
    }

    // SAFETY: `encode_gather_dispatch` と同一の根拠。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rank).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&dim).cast(),
            std::mem::size_of::<u32>(),
            6,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel_out).cast(),
            std::mem::size_of::<u32>(),
            7,
        );
    }

    let (threadgroups, threads_per_tg) = gs_dispatch_sizes(numel_out);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// gather／scatter カーネル引数 `constant uint& numel`（`shaders/
/// gather_scatter.metal`）は 32bit のため、`numel as u32` キャストを
/// 検証なしに行うと `numel`（`index_shape`／`out_shape` の要素数積）が
/// `u32::MAX` を超える形状で切り詰まり、`gs_dispatch_sizes` が実際の
/// 要素数より少ないスレッドグループしかディスパッチしなくなる。結果と
/// して `alloc_uninit_pooled` で確保した出力バッファの一部が未初期化
/// のまま `read_to_vec()` でホストへ返る（`crate::elementwise::
/// validate_elementwise_len` と同じ理由・同じ対策。呼び出し元の
/// `gather_scatter_model::validate_shapes_fit_u32` は各 shape 次元が
/// `u32::MAX` 以下であることのみを検査し、積である numel 自体は検査し
/// ないため本関数が必要。OWASP A03。`.claude/rules/security.md`）。
fn validate_gather_scatter_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "gather/scatter numel must fit in u32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

/// `numel` に対する grid/threadgroup サイズを構築する（`crate::
/// elementwise::ew_dispatch_sizes` と同一構成。`div_ceil` による末尾
/// ブロックの余剰スレッドはカーネル内境界チェックに委ねる契約。REQ-8）。
fn gs_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: GS_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(GS_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// `validate_gather_scatter_len` は純関数（Metal ランタイム非依存）
/// のため Linux 実行可能。`elementwise.rs::tests::
/// validate_elementwise_len_*` と同型の境界値検査（review 指摘の
/// 回帰防止。イシュー #1778 レビュー指摘）。
#[cfg(test)]
mod validate_len_tests {
    use super::*;

    #[test]
    fn accepts_len_at_u32_max() {
        assert!(validate_gather_scatter_len(u32::MAX as usize).is_ok());
    }

    #[test]
    fn rejects_len_exceeding_u32_max() {
        let err = validate_gather_scatter_len(u32::MAX as usize + 1)
            .expect_err("u32::MAX を超える numel は拒否されるべき");
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }
}
