//! Kullback-Leibler ダイバージェンス損失（`KLDivLoss`）融合カーネルの
//! 起動 API（イシュー #1738・親イシュー #1609「損失関数の拡張」。
//! CUDA 側 `backend-cuda::kl_div`〈同イシュー〉の Metal 対応版・
//! `bce.rs`〈イシュー #1737〉を雛形にした同型構成）。
//!
//! `mse.rs`（#1045）の 2 段構成は踏襲しつつ、ディスパッチ方式は
//! `bce.rs` と同じ `MetalContext::dispatch_sync`（1 encode = 1 即時
//! 同期の単純な方式）を使う（`bce.rs` モジュール冒頭コメントと同じ
//! 理由。encode-only 化は #1690／#1691 の性能最適化スコープで本 issue
//! の対象外）。
//!
//! `ops.rs::MetalBackendOps::kl_div_loss`／`kl_div_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::KlDivTarget;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/kl_div.metal` のソース（3 カーネルを含む）。
const KL_DIV_MSL_SRC: &str = include_str!("shaders/kl_div.metal");

/// 1 threadgroup あたりのスレッド数（`bce.rs::BCE_THREADGROUP_WIDTH` と
/// 同じ値・同じ理由）。
const KL_DIV_THREADGROUP_WIDTH: usize = 256;

/// forward 2 段目（`kl_div_finalize_f32`）が単一 threadgroup で処理
/// しきれる `partial` の最大長（`bce.rs::BCE_MAX_THREADGROUPS` と同じ
/// 値）。
const KL_DIV_MAX_THREADGROUPS: usize = 1024;

/// [`KlDivTarget`] を MSL カーネル引数の `uint`（`0` = `Probabilities`、
/// `1` = `LogProbabilities`）へ変換する（`bce.rs::bce_kind_to_u32` と
/// 対応する Metal 側実装。`KlDivTarget` の未知 variant は
/// `LogProbabilities`（`1`）へ安全側フォールバックする）。
fn kl_div_kind_to_u32(kind: KlDivTarget) -> u32 {
    match kind {
        KlDivTarget::Probabilities => 0,
        kind => {
            debug_assert!(
                matches!(kind, KlDivTarget::LogProbabilities),
                "kl_div::kl_div_kind_to_u32: 未知の KlDivTarget variant へフォールバックした\
                 （契約違反）"
            );
            1
        }
    }
}

/// 長さが `u32::MAX` に収まることを検証する（`bce.rs::validate_bce_len`
/// と同じ理由）。
pub(crate) fn validate_kl_div_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "kl_div_loss numel must fit in u32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

/// `input_len`／`target_len` の一致と `u32::MAX` 上限の両方を検証する
/// （`bce.rs::validate_bce_binary_len` と同じ構成）。
pub(crate) fn validate_kl_div_binary_len(
    input_len: usize,
    target_len: usize,
) -> Result<(), MetalError> {
    if input_len != target_len {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "kl_div length mismatch: input_len={input_len}, target_len={target_len}"
            ),
        });
    }
    validate_kl_div_len(input_len)
}

/// forward 1 段目（`kl_div_partial_f32`）の起動 threadgroup 数を決定
/// する（`bce.rs::bce_num_threadgroups` と同じ式）。
fn kl_div_num_threadgroups(numel: usize) -> usize {
    numel
        .div_ceil(KL_DIV_THREADGROUP_WIDTH)
        .min(KL_DIV_MAX_THREADGROUPS)
}

/// KLDiv 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みパイプラインを保持する。
pub struct MetalKlDiv {
    partial_f32: objc2::rc::Retained<MtlPipeline>,
    finalize_f32: objc2::rc::Retained<MtlPipeline>,
    backward_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalKlDiv {
    /// `ctx` のデバイス上で KLDiv 3 カーネルを実行時コンパイルしパイプ
    /// ラインを構築する（`bce.rs::MetalBce::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(KL_DIV_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let partial_f32 = pipeline::make_pipeline(ctx.device(), &library, "kl_div_partial_f32")?;
        let finalize_f32 = pipeline::make_pipeline(ctx.device(), &library, "kl_div_finalize_f32")?;
        let backward_f32 = pipeline::make_pipeline(ctx.device(), &library, "kl_div_backward_f32")?;
        Ok(Self {
            partial_f32,
            finalize_f32,
            backward_f32,
        })
    }

    /// forward: `reduction(Σ kl_div_elem_loss(input[i], target[i],
    /// kind))`。`input.len() == target.len()` は呼び出し元（`ops.rs`）
    /// が検証済みの契約。`numel == 0` はディスパッチを回避し `0.0` を
    /// 返す（`Mean`／`Sum` いずれも空和の契約。`backend-cpu::kl_div` と
    /// 同じ）。
    pub fn run_kl_div_loss_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        target: &[f32],
        kind: KlDivTarget,
        factor: f32,
    ) -> Result<f32, MetalError> {
        validate_kl_div_binary_len(input.len(), target.len())?;
        let numel = input.len();
        if numel == 0 {
            return Ok(0.0);
        }
        let kind_u = kl_div_kind_to_u32(kind);

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let target_buf = MetalBuffer::new_with_data(ctx, target)?;
        let num_tg = kl_div_num_threadgroups(numel);
        // `kl_div_partial_f32` は起動する `num_tg` 個の threadgroup
        // それぞれが `partial[tg_id]` を必ず 1 回書く（`shaders/
        // kl_div.metal` 参照）ため `alloc_uninit_pooled` を使える。
        let partial_buf = MetalBuffer::alloc_uninit_pooled(ctx, num_tg)?;

        ctx.dispatch_sync(|encoder| {
            encode_partial_dispatch(
                encoder,
                &self.partial_f32,
                &input_buf,
                &target_buf,
                &partial_buf,
                numel as u32,
                kind_u,
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

    /// backward: `dInput[i] = scale·kl_div_elem_grad_input(input[i],
    /// target[i], kind)`。`dTarget` は呼び出し元がホスト側の逐次 map で
    /// 計算する契約（`backend_ops.rs::BackendOps::
    /// kl_div_loss_backward` doc 参照）のため、本関数は `dInput` のみを
    /// 計算する。`numel == 0` は空 `Vec` を返す。
    pub fn run_kl_div_backward_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        target: &[f32],
        kind: KlDivTarget,
        scale: f32,
    ) -> Result<Vec<f32>, MetalError> {
        validate_kl_div_binary_len(input.len(), target.len())?;
        let numel = input.len();
        if numel == 0 {
            return Ok(Vec::new());
        }
        let kind_u = kl_div_kind_to_u32(kind);

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let target_buf = MetalBuffer::new_with_data(ctx, target)?;
        // `kl_div_backward_f32` は `idx < numel` ガード内で `dinput[idx]`
        // を必ず埋める（`shaders/kl_div.metal` 参照）ため
        // `alloc_uninit_pooled` を使う。
        let dinput_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_backward_dispatch(
                encoder,
                &self.backward_f32,
                &input_buf,
                &target_buf,
                &dinput_buf,
                numel as u32,
                kind_u,
                scale,
            );
        })?;

        Ok(dinput_buf.read_to_vec())
    }
}

fn kl_div_dispatch_sizes(units: usize, threadgroup_width: usize) -> (MTLSize, MTLSize) {
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

/// forward 1 段目のエンコード（バッファ結線 index 0〜2・`numel`／`kind`
/// index 3〜4・`num_tg` 個の threadgroup をディスパッチ）。
/// [`MetalKlDiv::run_kl_div_loss_f32`] が [`MetalContext::
/// dispatch_sync`] のクロージャから呼ぶ（`bce.rs::
/// encode_partial_dispatch` と同型）。
#[allow(clippy::too_many_arguments)]
fn encode_partial_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    target_buf: &MetalBuffer,
    partial_buf: &MetalBuffer,
    numel: u32,
    kind: u32,
    num_tg: usize,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`bce.rs::encode_partial_dispatch` と同種のコメント参照）。各
    // バッファは本関数呼び出し元 `MetalKlDiv::run_kl_div_loss_f32` の
    // `ctx.dispatch_sync` 呼び出し中は関数スコープで生存するため
    // エンコード完了まで有効である。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(target_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 2);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。`numel`／`kind` はローカル変数で
    // ポインタは本呼び出し中生存し、長さは `constant uint&` 宣言の型と
    // 揃えている（`shaders/kl_div.metal` 参照）。
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
    }

    let (threadgroups, threads_per_tg) = kl_div_dispatch_sizes(num_tg, KL_DIV_THREADGROUP_WIDTH);
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

    // 単一 threadgroup（`shaders/kl_div.metal::kl_div_finalize_f32` の
    // 契約）。
    let (threadgroups, threads_per_tg) = kl_div_dispatch_sizes(1, KL_DIV_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// backward のエンコード（`input`／`target`／`dinput` index 0〜2・
/// `numel`／`kind`／`scale` index 3〜5・1 スレッド 1 要素の 1 次元
/// ディスパッチ）。
#[allow(clippy::too_many_arguments)]
fn encode_backward_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    target_buf: &MetalBuffer,
    dinput_buf: &MetalBuffer,
    numel: u32,
    kind: u32,
    scale: f32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_partial_dispatch` と同じ根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(target_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(dinput_buf.raw()), 0, 2);
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
            std::ptr::NonNull::from(&scale).cast(),
            std::mem::size_of::<f32>(),
            5,
        );
    }

    let groups = (numel as usize).div_ceil(KL_DIV_THREADGROUP_WIDTH);
    let (threadgroups, threads_per_tg) = kl_div_dispatch_sizes(groups, KL_DIV_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
