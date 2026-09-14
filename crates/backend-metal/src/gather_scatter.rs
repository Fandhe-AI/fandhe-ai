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
//! （`ops.rs`）は shape 検査（[`fandhe_ai_tensor_core::gather_out_shape`]／
//! [`fandhe_ai_tensor_core::scatter_out_shape`]）・
//! [`crate::gather_scatter_model::validate_shapes_fit_u32`]・
//! [`crate::gather_scatter_model::validate_index_range`] を済ませてから
//! 本モジュールを呼ぶが、[`MetalGatherScatter::run_gather_f32`]／
//! [`MetalGatherScatter::run_scatter_f32`] は `pub` であり `ops.rs` を
//! 経由せず直接呼び出せるため、呼び出し元の検査結果を信頼せず本モジュール
//! 自身でも独立に同じ検査（[`crate::gather_scatter_model::
//! validate_gather_launch`]／[`crate::gather_scatter_model::
//! validate_scatter_launch`] に切り出し済み: [`fandhe_ai_tensor_core::
//! gather_out_shape`]／[`fandhe_ai_tensor_core::scatter_out_shape`] の
//! 再利用による rank 一致・`dim` が rank 範囲内・非 `dim` 軸の
//! `in_shape`／`out_shape` と `index_shape` の整合〈gather は完全一致・
//! scatter は `index_shape[axis] <= out_shape[axis]`〉・shape 各次元の
//! `u32` 収容・`numel`（形状次元の積）が `u32` カーネル引数へ収まる
//! こと・`index` の値域・**各スライス〈`input`／`index`／`src`〉の実長
//! が対応する shape の要素数積と一致すること**）を行う（判定迂回経路を
//! 作らないための多層防御。`.claude/rules/security.md` A08。`crates/
//! backend-cpu/src/gather_scatter.rs` と同じ二重検査方針。codex-review
//! 指摘）。非 `dim` 軸の整合検査を怠ると、rank・`dim` 自体は正しくても
//! （例: `in_shape=[2,3]`・`index_shape=[5,3]`・`dim=1`）カーネル
//! （`shaders/gather_scatter.metal`）の `gs_ravel(coords, in_shape,
//! rank)` が `index_shape` 由来の非 `dim` 軸座標から `in_shape` の
//! 実バッファ長を超えるオフセットを計算し GPU 側バッファ範囲外読み出し
//! になりうる（advisor 指摘）。スライス実長の検査を怠ると、`shape`
//! 引数（要素数積からカーネル引数 `numel` を導出）とスライス自体の
//! 実長が食い違う入力（`MetalBuffer`／`MetalIndexBuffer` は渡された
//! スライスの実長でバッファを確保する）で、カーネルが `shapes`／
//! `numel` から導出した添字でバッファ範囲外を読む GPU 側 OOB になりうる。
//! `validate_gather_launch`／`validate_scatter_launch` は `objc2` FFI に
//! 一切触れない純関数のため Linux（本実装環境・CI）でも回帰テストできる
//! （`crate::gather_scatter_model` の単体テスト。イシュー #1799
//! レビュー指摘: 従来は macOS 実機限定〈`#[ignore]`〉の
//! `tests/gather_scatter_parity.rs` でしか検証できておらず、Linux で
//! 走る CI では実質未検証だった）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::{ScatterReduce, ShapeError};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::gather_scatter_model::{ScatterLaunch, validate_gather_launch, validate_scatter_launch};
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};

/// [`ShapeError`] を [`MetalError::InvalidGatherScatterShape`] へ変換
/// する（本モジュールが独立に行う shape・rank・`dim`・`index` 検査
/// 〈モジュール冒頭コメント参照〉の共通変換ヘルパー）。
fn shape_err_to_metal(err: ShapeError) -> MetalError {
    MetalError::InvalidGatherScatterShape {
        detail: err.to_string(),
    }
}

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
    /// `ops.rs` を経由しない直接呼び出しでも安全なよう、本関数自身が
    /// 独立に shape・rank・`dim`・`index` 値域を検査する（モジュール
    /// 冒頭コメント参照。実体は [`crate::gather_scatter_model::
    /// validate_gather_launch`] へ切り出し済み——`objc2` FFI に触れない
    /// 純関数のため Linux（本実装環境・CI）でも回帰テストできる。
    /// イシュー #1799 レビュー指摘）。`index_shape` の要素数積
    /// （＝出力要素数）が 0 の場合は空配列を返す（`MetalBuffer` は
    /// 0 バイト確保を拒否するため、デバイス確保前に早期リターンする）。
    pub fn run_gather_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        in_shape: &[usize],
        index: &[i32],
        index_shape: &[usize],
        dim: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let numel = validate_gather_launch(input, in_shape, index, index_shape, dim)
            .map_err(shape_err_to_metal)?;
        if numel == 0 {
            return Ok(Vec::new());
        }
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
    /// 独立検査の方針は [`Self::run_gather_f32`] と同じ（実体は
    /// [`crate::gather_scatter_model::validate_scatter_launch`] へ
    /// 切り出し済み。イシュー #1799 レビュー指摘）。`out_shape`
    /// （＝`input.shape()`）の要素数積が 0 の場合は空配列を返す。
    /// `index_shape`（＝`src.shape()`）の要素数積が 0（`out_shape` は
    /// 非空）の場合は、`scatter_out_shape` が非 `dim` 軸で
    /// `index_shape[axis] <= out_shape[axis]` のみを課す契約上
    /// （`shaders/gather_scatter.metal` 冒頭コメント「scatter」節）
    /// どの出力位置も `index` から触れられずホストモデル
    /// （[`crate::gather_scatter_model::scatter_model`]）と同じく
    /// `input` の完全なパススルーになるため、`MetalIndexBuffer`／
    /// `MetalBuffer` が 0 バイト確保を拒否する前にこの契約上自明な
    /// 結果を返す（Cursor Bugbot・codex-review P2 指摘）。
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
        let ScatterLaunch {
            numel_out,
            idx_numel,
        } = validate_scatter_launch(input, out_shape, index, index_shape, src, dim)
            .map_err(shape_err_to_metal)?;
        if numel_out == 0 {
            return Ok(Vec::new());
        }
        if idx_numel == 0 {
            return Ok(input.to_vec());
        }
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
