//! Huber／SmoothL1 損失融合カーネルの起動 API（イシュー #1739。CUDA 側
//! `backend-cuda::huber`〈同イシュー〉の Metal 対応版）。`mse.rs`
//! （イシュー #1045・#1690）と同じ構成方針を踏襲する: [`MetalHuber::new`]
//! が `shaders/huber.metal` を実行時コンパイルして 3 パイプラインを
//! 保持し、[`MetalHuber::run_huber_loss_f32`]／
//! [`MetalHuber::run_huber_backward_f32`] へホスト側スライスを渡すだけで
//! バッファ確保・ディスパッチ・readback を内部で完結できる。
//! `ops.rs::MetalBackendOps::huber_loss`／`huber_loss_backward` から
//! `BackendOps` の実装として呼ばれる。
//!
//! `mse.rs`（イシュー #1690）が確立した `ctx.encode` +
//! [`fandhe_ai_tensor_core::DispatchFailureCell`] 方式（forward 2 段を
//! 同一バッチへ積み 1 回だけ `ctx.synchronize()`・backward は呼び出し元
//! の即時ホストアクセス契約により 1 回の `synchronize()` を維持）を
//! 新規実装時点から採用する（`mse.rs` モジュール冒頭コメント参照）。

use fandhe_ai_tensor_core::{DispatchFailureCell, HuberKind};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/huber.metal` のソース（3 カーネルを含む）。
const HUBER_MSL_SRC: &str = include_str!("shaders/huber.metal");

/// 1 threadgroup あたりのスレッド数（`mse.rs::MSE_THREADGROUP_WIDTH`
/// と同じ値・同じ理由。`shaders/huber.metal::HUBER_SIMDGROUPS_PER_TG`
/// （8）と対応させる）。
const HUBER_THREADGROUP_WIDTH: usize = 256;

/// forward 2 段目（`huber_finalize_f32`）が単一 threadgroup で処理
/// しきれる `partial` の最大長（`mse.rs::MSE_MAX_THREADGROUPS` と同じ
/// 値）。
const HUBER_MAX_THREADGROUPS: usize = 1024;

/// `HuberKind` を Metal カーネル引数（`uint kind`。0=Huber,
/// 1=SmoothL1）へ変換する（`crate::huber` module の CUDA 側
/// `huber_kind_to_i32` と同じ役割・同じ安全側フォールバック）。
fn huber_kind_to_u32(kind: HuberKind) -> u32 {
    match kind {
        HuberKind::SmoothL1 => 1,
        _ => 0,
    }
}

/// 長さが `u32::MAX` に収まることを検証する（`mse.rs::
/// validate_mse_len` と同じ理由）。
pub(crate) fn validate_huber_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!("huber_loss numel must fit in u32 (kernel argument type): numel={len}"),
        });
    }
    Ok(())
}

/// `pred_len`／`target_len` の一致と `u32::MAX` 上限の両方を検証する
/// （`mse.rs::validate_mse_binary_len` と同じ構成）。
pub(crate) fn validate_huber_binary_len(
    pred_len: usize,
    target_len: usize,
) -> Result<(), MetalError> {
    if pred_len != target_len {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!("huber length mismatch: pred_len={pred_len}, target_len={target_len}"),
        });
    }
    validate_huber_len(pred_len)
}

/// forward 1 段目（`huber_partial_f32`）の起動 threadgroup 数を決定する
/// （`mse.rs::mse_num_threadgroups` と同じ式・同じ理由）。
fn huber_num_threadgroups(numel: usize) -> usize {
    numel
        .div_ceil(HUBER_THREADGROUP_WIDTH)
        .min(HUBER_MAX_THREADGROUPS)
}

/// Huber 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みパイプラインを保持する（`mse.rs::MetalMse` と同型）。
pub struct MetalHuber {
    partial_f32: objc2::rc::Retained<MtlPipeline>,
    finalize_f32: objc2::rc::Retained<MtlPipeline>,
    backward_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalHuber {
    /// `ctx` のデバイス上で Huber 3 カーネルを実行時コンパイルし
    /// パイプラインを構築する（`mse.rs::MetalMse::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(HUBER_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let partial_f32 = pipeline::make_pipeline(ctx.device(), &library, "huber_partial_f32")?;
        let finalize_f32 = pipeline::make_pipeline(ctx.device(), &library, "huber_finalize_f32")?;
        let backward_f32 = pipeline::make_pipeline(ctx.device(), &library, "huber_backward_f32")?;
        Ok(Self {
            partial_f32,
            finalize_f32,
            backward_f32,
        })
    }

    /// forward: `reduction(Σ l(pred[i]−target[i]))`（`l` は
    /// `shaders/huber.metal::huber_elem_loss`）。`pred.len() ==
    /// target.len()` は呼び出し元（`ops.rs`）が検証済みの契約。
    /// `numel == 0` はディスパッチを回避し `0.0` を返す（`Mean`／`Sum`
    /// いずれも空和の契約）。
    ///
    /// 内部でローカルの [`DispatchFailureCell`] を生成し
    /// [`Self::run_huber_loss_f32_tracked`] へ委譲する薄いラッパー
    /// （`mse.rs::run_mse_loss_f32` と同型）。
    pub fn run_huber_loss_f32(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        kind: HuberKind,
        delta: f32,
        factor: f32,
    ) -> Result<f32, MetalError> {
        let token = DispatchFailureCell::new();
        self.run_huber_loss_f32_tracked(ctx, pred, target, kind, delta, factor, &token)
    }

    /// [`Self::run_huber_loss_f32`] の本体。forward 1・2 段目とも
    /// `ctx.encode`（待たない）で同一バッチへ積み、最後に 1 回だけ
    /// `ctx.synchronize()` する（`mse.rs::run_mse_loss_f32_tracked` と
    /// 同じ設計・同じ `token` 競合対策）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_huber_loss_f32_tracked(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        kind: HuberKind,
        delta: f32,
        factor: f32,
        token: &DispatchFailureCell,
    ) -> Result<f32, MetalError> {
        validate_huber_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(0.0);
        }
        let kind_u = huber_kind_to_u32(kind);

        let pred_buf = MetalBuffer::new_with_data(ctx, pred)?;
        let target_buf = MetalBuffer::new_with_data(ctx, target)?;
        let num_tg = huber_num_threadgroups(numel);
        // `huber_partial_f32` は起動する `num_tg` 個の threadgroup
        // それぞれが `partial[tg_id]` を必ず 1 回書く
        // （`shaders/huber.metal` 参照）ため `alloc_uninit_pooled` を
        // 使える（`mse.rs` と同じ適用条件）。
        let partial_buf = MetalBuffer::alloc_uninit_pooled(ctx, num_tg)?;

        ctx.encode(
            "huber_partial_f32",
            &[pred_buf.raw(), target_buf.raw(), partial_buf.raw()],
            Some(token),
            |encoder| {
                encode_partial_dispatch(
                    encoder,
                    &self.partial_f32,
                    &pred_buf,
                    &target_buf,
                    &partial_buf,
                    numel as u32,
                    kind_u,
                    delta,
                    num_tg,
                );
            },
        )?;

        // `alloc_uninit_pooled` は `zero_on_reuse=false` のため、この
        // 2 段目の確保自体が 1 段目の GPU 完了を待つ同期境界にはならない
        // （`mse.rs` モジュール冒頭コメント参照）。
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, 1)?;
        ctx.encode(
            "huber_finalize_f32",
            &[partial_buf.raw(), out_buf.raw()],
            Some(token),
            |encoder| {
                encode_finalize_dispatch(
                    encoder,
                    &self.finalize_f32,
                    &partial_buf,
                    &out_buf,
                    num_tg as u32,
                    factor,
                );
            },
        )?;

        check_dispatch_token(ctx.synchronize(), token)?;

        Ok(out_buf.read_to_vec().first().copied().unwrap_or(0.0))
    }

    /// backward: `dPred[i] = scale·grad_elem(pred[i]−target[i])`
    /// （`grad_elem` は `shaders/huber.metal::huber_elem_grad`）。
    /// `dTarget` は呼び出し元がホスト側で符号反転して得る契約
    /// （`backend_ops.rs::BackendOps::huber_loss_backward` doc 参照）の
    /// ため、本関数は `dPred` のみを計算する。`numel == 0` は空 `Vec`
    /// を返す。
    ///
    /// 内部でローカルの [`DispatchFailureCell`] を生成し
    /// [`Self::run_huber_backward_f32_tracked`] へ委譲する薄いラッパー
    /// （`mse.rs::run_mse_backward_f32` と同型）。
    pub fn run_huber_backward_f32(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        kind: HuberKind,
        delta: f32,
        scale: f32,
    ) -> Result<Vec<f32>, MetalError> {
        let token = DispatchFailureCell::new();
        self.run_huber_backward_f32_tracked(ctx, pred, target, kind, delta, scale, &token)
    }

    /// [`Self::run_huber_backward_f32`] の本体。呼び出し元
    /// `crates/autodiff/src/grad.rs`（`Op::HuberLoss` の VJP）が戻り値
    /// `Tensor<f32>`（`dpred`）に対して直後に `dense_vec(&dpred)`（ホスト
    /// 即時アクセス）で `dtarget` を計算する契約のため、本関数は関数を
    /// 出る前に**必ず自ら** `ctx.synchronize()` する（`mse.rs::
    /// run_mse_backward_f32_tracked` と同じ理由・同じ `token` 検査）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_huber_backward_f32_tracked(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        kind: HuberKind,
        delta: f32,
        scale: f32,
        token: &DispatchFailureCell,
    ) -> Result<Vec<f32>, MetalError> {
        validate_huber_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(Vec::new());
        }
        let kind_u = huber_kind_to_u32(kind);

        let pred_buf = MetalBuffer::new_with_data(ctx, pred)?;
        let target_buf = MetalBuffer::new_with_data(ctx, target)?;
        // `huber_backward_f32` は `idx < numel` ガード内で `dpred[idx]`
        // を必ず埋める（`shaders/huber.metal` 参照）ため
        // `alloc_uninit_pooled` を使う（`mse.rs` と同じ適用条件）。
        let dpred_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.encode(
            "huber_backward_f32",
            &[pred_buf.raw(), target_buf.raw(), dpred_buf.raw()],
            Some(token),
            |encoder| {
                encode_backward_dispatch(
                    encoder,
                    &self.backward_f32,
                    &pred_buf,
                    &target_buf,
                    &dpred_buf,
                    numel as u32,
                    kind_u,
                    delta,
                    scale,
                );
            },
        )?;

        check_dispatch_token(ctx.synchronize(), token)?;

        Ok(dpred_buf.read_to_vec())
    }
}

/// `ctx.synchronize()` の戻り値を `token` の状態と突き合わせる共通
/// ヘルパ（`mse.rs::check_dispatch_token` と同一実装。crate 内で
/// 独立複製する理由も同一——モジュール独立性を優先する判断）。
fn check_dispatch_token(
    sync_result: Result<(), MetalError>,
    token: &DispatchFailureCell,
) -> Result<(), MetalError> {
    sync_result?;
    if token.is_set() {
        let message = token.take().map(|err| err.to_string()).unwrap_or_else(|| {
            "dispatch failure token was set by a concurrent synchronize() on the shared \
                 MetalContext before this call's own synchronize() observed the batch"
                .to_string()
        });
        return Err(MetalError::CommandBufferExecutionFailed { message });
    }
    Ok(())
}

fn huber_dispatch_sizes(units: usize, threadgroup_width: usize) -> (MTLSize, MTLSize) {
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
/// 3・`kind` index 4・`delta` index 5・`num_tg` 個の threadgroup を
/// ディスパッチ）。[`MetalHuber::run_huber_loss_f32_tracked`] が
/// [`MetalContext::encode`] のクロージャから呼ぶ。
#[allow(clippy::too_many_arguments)]
fn encode_partial_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    pred_buf: &MetalBuffer,
    target_buf: &MetalBuffer,
    partial_buf: &MetalBuffer,
    numel: u32,
    kind: u32,
    delta: f32,
    num_tg: usize,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`mse.rs::encode_partial_dispatch` と同種のコメント参照）。各
    // バッファは `MetalHuber::run_huber_loss_f32_tracked` が
    // `ctx.encode` の `resources` 引数として渡し `Batch::in_flight` へ
    // retain される、または本関数呼び出し中は関数スコープで生存する
    // ため、エンコード完了まで有効である。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(pred_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(target_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。`numel`／`kind`／`delta` はいずれも
    // ローカル変数でありポインタは本呼び出し中生存し、長さは
    // `constant uint&`／`constant float&` 宣言の型と揃えている
    // （`shaders/huber.metal` 参照）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&kind).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&delta).cast(),
            std::mem::size_of::<f32>(),
            5,
        );
    }

    let (threadgroups, threads_per_tg) = huber_dispatch_sizes(num_tg, HUBER_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// forward 2 段目のエンコード（`partial`／`out` index 0〜1・
/// `num_partials`／`factor` index 2〜3・単一 threadgroup をディスパッチ。
/// `mse.rs::encode_finalize_dispatch` と同一構成）。
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

    // 単一 threadgroup（`shaders/huber.metal::huber_finalize_f32` の契約）。
    let (threadgroups, threads_per_tg) = huber_dispatch_sizes(1, HUBER_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// backward のエンコード（`pred`／`target`／`dpred` index 0〜2・
/// `numel`／`kind`／`delta`／`scale` index 3〜6・1 スレッド 1 要素の
/// 1 次元ディスパッチ）。
#[allow(clippy::too_many_arguments)]
fn encode_backward_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    pred_buf: &MetalBuffer,
    target_buf: &MetalBuffer,
    dpred_buf: &MetalBuffer,
    numel: u32,
    kind: u32,
    delta: f32,
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
            std::ptr::NonNull::from(&kind).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&delta).cast(),
            std::mem::size_of::<f32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&scale).cast(),
            std::mem::size_of::<f32>(),
            6,
        );
    }

    let groups = (numel as usize).div_ceil(HUBER_THREADGROUP_WIDTH);
    let (threadgroups, threads_per_tg) = huber_dispatch_sizes(groups, HUBER_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
