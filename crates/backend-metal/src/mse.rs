//! 平均二乗誤差（MSE）融合カーネルの起動 API（イシュー #1045・親イシュー
//! #1043。CUDA 側 `backend-cuda::mse`〈同イシュー〉の Metal 対応版）。
//!
//! `elementwise.rs::MetalElementwise`・`sgd.rs::MetalSgd` と同じ構成
//! 方針を踏襲する: [`MetalMse::new`] が `shaders/mse.metal` を実行時
//! コンパイルして 3 パイプラインを保持し、[`MetalMse::run_mse_loss_f32`]／
//! [`MetalMse::run_mse_backward_f32`] へホスト側スライスを渡すだけで
//! バッファ確保・ディスパッチ・readback を内部で完結できる。
//! `ops.rs::MetalBackendOps::mse_loss`／`mse_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/mse.metal` のソース（3 カーネルを含む）。
const MSE_MSL_SRC: &str = include_str!("shaders/mse.metal");

/// 1 threadgroup あたりのスレッド数（1 次元、8 simdgroup 分）。
/// `elementwise.rs::EW_THREADGROUP_WIDTH`・`sgd.rs::SGD_THREADGROUP_WIDTH`
/// と同じ値・同じ理由。`shaders/mse.metal::MSE_SIMDGROUPS_PER_TG`（8）と
/// 対応させる（256/32=8）。
const MSE_THREADGROUP_WIDTH: usize = 256;

/// forward 2 段目（`mse_finalize_f32`）が単一 threadgroup で処理しきれる
/// `partial` の最大長（＝ forward 1 段目の起動 threadgroup 数の上限）。
/// CUDA 側 `kernels_mse::MSE_MAX_BLOCKS` と同じ値。
const MSE_MAX_THREADGROUPS: usize = 1024;

/// 長さが `u32::MAX` に収まることを検証する（`elementwise.rs::
/// validate_elementwise_len` と同じ理由）。
pub(crate) fn validate_mse_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!("mse_loss numel must fit in u32 (kernel argument type): numel={len}"),
        });
    }
    Ok(())
}

/// `pred_len`／`target_len` の一致と `u32::MAX` 上限の両方を検証する
/// （`elementwise.rs::validate_elementwise_binary_dims` と同じ構成）。
///
/// `run_mse_loss_f32`／`run_mse_backward_f32` は `MetalMse` の公開メソッド
/// であり `ops.rs` を経由しない外部呼び出しに対しても長さ不一致を
/// `panic!`（`assert_eq!`）ではなく型付きエラーとして返す契約とする
/// （AGENTS.md「本番経路の panic 禁止」）。
pub(crate) fn validate_mse_binary_len(
    pred_len: usize,
    target_len: usize,
) -> Result<(), MetalError> {
    if pred_len != target_len {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!("mse length mismatch: pred_len={pred_len}, target_len={target_len}"),
        });
    }
    validate_mse_len(pred_len)
}

/// forward 1 段目（`mse_partial_f32`）の起動 threadgroup 数を決定する。
/// `min(ceil_div(numel, MSE_THREADGROUP_WIDTH), MSE_MAX_THREADGROUPS)`
/// （`shaders/mse.metal` 冒頭コメント「forward の 2 段構成」の契約）。
/// `numel > 0` を前提とする（`numel == 0` は呼び出し元がディスパッチ
/// 自体を回避する）。
fn mse_num_threadgroups(numel: usize) -> usize {
    numel
        .div_ceil(MSE_THREADGROUP_WIDTH)
        .min(MSE_MAX_THREADGROUPS)
}

/// MSE 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みパイプラインを保持する。
pub struct MetalMse {
    partial_f32: objc2::rc::Retained<MtlPipeline>,
    finalize_f32: objc2::rc::Retained<MtlPipeline>,
    backward_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalMse {
    /// `ctx` のデバイス上で MSE 3 カーネルを実行時コンパイルしパイプ
    /// ラインを構築する（`elementwise.rs::MetalElementwise::new` と同一
    /// 手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(MSE_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let partial_f32 = pipeline::make_pipeline(ctx.device(), &library, "mse_partial_f32")?;
        let finalize_f32 = pipeline::make_pipeline(ctx.device(), &library, "mse_finalize_f32")?;
        let backward_f32 = pipeline::make_pipeline(ctx.device(), &library, "mse_backward_f32")?;
        Ok(Self {
            partial_f32,
            finalize_f32,
            backward_f32,
        })
    }

    /// forward: `reduction(Σ(pred[i]−target[i])²)`。`pred.len() ==
    /// target.len()` は呼び出し元（`ops.rs`）が検証済みの契約。
    /// `numel == 0` はディスパッチを回避し `0.0` を返す（`Mean`／`Sum`
    /// いずれも空和の契約。`backend-cpu::mse` と同じ）。
    pub fn run_mse_loss_f32(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        factor: f32,
    ) -> Result<f32, MetalError> {
        validate_mse_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(0.0);
        }

        let pred_buf = MetalBuffer::new_with_data(ctx, pred)?;
        let target_buf = MetalBuffer::new_with_data(ctx, target)?;
        let num_tg = mse_num_threadgroups(numel);
        // `mse_partial_f32` は起動する `num_tg` 個の threadgroup それぞれ
        // が `partial[tg_id]` を必ず 1 回書く（`shaders/mse.metal` 参照）
        // ため `alloc_uninit_pooled` を使える（`elementwise.rs::run_binary`
        // と同じ適用条件。イシュー #1021 設計文書 §6「A02」）。
        let partial_buf = MetalBuffer::alloc_uninit_pooled(ctx, num_tg)?;

        ctx.dispatch_sync(|encoder| {
            encode_partial_dispatch(
                encoder,
                &self.partial_f32,
                &pred_buf,
                &target_buf,
                &partial_buf,
                numel as u32,
                num_tg,
            );
        })?;

        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, 1)?;
        ctx.dispatch_sync(|encoder| {
            encode_finalize_dispatch(
                encoder,
                &self.finalize_f32,
                &partial_buf,
                &out_buf,
                num_tg as u32,
                factor,
            );
        })?;

        Ok(out_buf.read_to_vec().first().copied().unwrap_or(0.0))
    }

    /// backward: `dPred[i] = scale·(pred[i]−target[i])`。`dTarget` は
    /// 呼び出し元がホスト側で符号反転して得る契約（`backend_ops.rs::
    /// BackendOps::mse_loss_backward` doc 参照）のため、本関数は
    /// `dPred` のみを計算する。`numel == 0` は空 `Vec` を返す。
    pub fn run_mse_backward_f32(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, MetalError> {
        validate_mse_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let pred_buf = MetalBuffer::new_with_data(ctx, pred)?;
        let target_buf = MetalBuffer::new_with_data(ctx, target)?;
        // イシュー #1021: `mse_backward_f32` は `idx < numel` ガード内で
        // `dpred[idx]` を必ず埋める（`shaders/mse.metal` 参照）ため
        // `alloc_uninit_pooled` を使う（`elementwise.rs::run_unary` と
        // 同じ適用条件）。
        let dpred_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_backward_dispatch(
                encoder,
                &self.backward_f32,
                &pred_buf,
                &target_buf,
                &dpred_buf,
                numel as u32,
                scale,
            );
        })?;

        Ok(dpred_buf.read_to_vec())
    }
}

fn mse_dispatch_sizes(units: usize, threadgroup_width: usize) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: units,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// forward 1 段目のエンコード（バッファ結線 index 0〜2・`numel` index
/// 3・`num_tg` 個の threadgroup をディスパッチ）。[`MetalMse::
/// run_mse_loss_f32`] が [`MetalContext::dispatch_sync`] のクロージャ
/// から呼ぶ。
fn encode_partial_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    pred_buf: &MetalBuffer,
    target_buf: &MetalBuffer,
    partial_buf: &MetalBuffer,
    numel: u32,
    num_tg: usize,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`elementwise.rs::encode_binary_dispatch` と同種のコメント参照）。
    // 各バッファは呼び出し元 `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(pred_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(target_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。`numel` はローカル変数でありポインタ
    // は本呼び出し中生存し、長さは `constant uint&` 宣言の型と揃えている
    // （`shaders/mse.metal` 参照）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
    }

    let (threadgroups, threads_per_tg) = mse_dispatch_sizes(num_tg, MSE_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// forward 2 段目のエンコード（`partial`／`out` index 0〜1・
/// `num_partials`／`factor` index 2〜3・単一 threadgroup をディスパッチ）。
fn encode_finalize_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    partial_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    num_partials: u32,
    factor: f32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_partial_dispatch` と同じ根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&num_partials).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&factor).cast(),
            std::mem::size_of::<f32>(),
            3,
        );
    }

    // 単一 threadgroup（`shaders/mse.metal::mse_finalize_f32` の契約）。
    let (threadgroups, threads_per_tg) = mse_dispatch_sizes(1, MSE_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// backward のエンコード（`pred`／`target`／`dpred` index 0〜2・
/// `numel`／`scale` index 3〜4・1 スレッド 1 要素の 1 次元ディスパッチ）。
#[allow(clippy::too_many_arguments)]
fn encode_backward_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    pred_buf: &MetalBuffer,
    target_buf: &MetalBuffer,
    dpred_buf: &MetalBuffer,
    numel: u32,
    scale: f32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_partial_dispatch` と同じ根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(pred_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(target_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(dpred_buf.raw()), 0, 2);
    }
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&scale).cast(),
            std::mem::size_of::<f32>(),
            4,
        );
    }

    let groups = (numel as usize).div_ceil(MSE_THREADGROUP_WIDTH);
    let (threadgroups, threads_per_tg) = mse_dispatch_sizes(groups, MSE_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
