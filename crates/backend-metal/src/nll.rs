//! 負対数尤度損失（`NLLLoss`）融合カーネルの起動 API（イシュー #1738・
//! 親イシュー #1609「損失関数の拡張」。CUDA 側 `backend-cuda::nll`
//! 〈同イシュー〉の Metal 対応版）。
//!
//! `mse.rs`（#1045）の 2 段構成（`nll_partial_f32` → `nll_finalize_f32`
//! forward・`nll_backward_f32` backward）は踏襲しつつ、ディスパッチ方式
//! は `rmsnorm.rs`／`softmax.rs`／`bce.rs`（#1737）が使う
//! `MetalContext::dispatch_sync`（1 encode = 1 即時同期の単純な方式）を
//! 使う（`bce.rs` モジュール冒頭コメントと同じ理由。encode-only 化は
//! #1690／#1691 の性能最適化スコープで本 issue の対象外）。
//!
//! `ops.rs::MetalBackendOps::nll_loss`／`nll_loss_backward` から
//! `BackendOps` の実装として呼ばれる。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/nll.metal` のソース（3 カーネルを含む）。
const NLL_MSL_SRC: &str = include_str!("shaders/nll.metal");

/// 1 threadgroup あたりのスレッド数（`mse.rs::MSE_THREADGROUP_WIDTH` と
/// 同じ値・同じ理由。`shaders/nll.metal::NLL_SIMDGROUPS_PER_TG`（8）と
/// 対応させる）。
const NLL_THREADGROUP_WIDTH: usize = 256;

/// forward 2 段目（`nll_finalize_f32`）が単一 threadgroup で処理しきれる
/// `partial` の最大長（`mse.rs::MSE_MAX_THREADGROUPS` と同じ値）。
const NLL_MAX_THREADGROUPS: usize = 1024;

/// `input`/`class_dim` から導出される `(outer, num_classes, inner)`
/// レイアウト（`backend-cpu::nll::NllLayout`／`backend-cuda::nll::
/// NllLayout` と同型。呼び出し元 `ops.rs::MetalBackendOps::nll_loss`／
/// `nll_loss_backward` が `input.shape()`／`class_dim` から構築する）。
#[derive(Debug, Clone, Copy)]
pub struct NllLayout {
    pub outer: usize,
    pub num_classes: usize,
    pub inner: usize,
}

impl NllLayout {
    fn n_samples(&self) -> usize {
        self.outer * self.inner
    }

    fn numel(&self) -> usize {
        self.outer * self.num_classes * self.inner
    }
}

/// 長さが `u32::MAX` に収まることを検証する（`mse.rs::validate_mse_len`
/// と同じ理由）。
pub(crate) fn validate_nll_layout(layout: NllLayout) -> Result<(), MetalError> {
    if layout.n_samples() > u32::MAX as usize || layout.numel() > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "nll_loss dims must fit in u32 (kernel argument type): n_samples={}, numel={}",
                layout.n_samples(),
                layout.numel()
            ),
        });
    }
    Ok(())
}

/// forward 1 段目（`nll_partial_f32`）の起動 threadgroup 数を決定する
/// （`mse.rs::mse_num_threadgroups` と同じ式。`n_samples` 基準）。
fn nll_num_threadgroups(n_samples: usize) -> usize {
    n_samples
        .div_ceil(NLL_THREADGROUP_WIDTH)
        .min(NLL_MAX_THREADGROUPS)
}

/// NLL 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みパイプラインを保持する。
pub struct MetalNll {
    partial_f32: objc2::rc::Retained<MtlPipeline>,
    finalize_f32: objc2::rc::Retained<MtlPipeline>,
    backward_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalNll {
    /// `ctx` のデバイス上で NLL 3 カーネルを実行時コンパイルしパイプ
    /// ラインを構築する（`mse.rs::MetalMse::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(NLL_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let partial_f32 = pipeline::make_pipeline(ctx.device(), &library, "nll_partial_f32")?;
        let finalize_f32 = pipeline::make_pipeline(ctx.device(), &library, "nll_finalize_f32")?;
        let backward_f32 = pipeline::make_pipeline(ctx.device(), &library, "nll_backward_f32")?;
        Ok(Self {
            partial_f32,
            finalize_f32,
            backward_f32,
        })
    }

    /// forward: `reduction(Σ_s −input[(o·C+t_s)·inner+i])`（サンプル
    /// `s = o·inner+i`）。`input.len() == layout.numel()`／
    /// `targets.len() == layout.n_samples()` は呼び出し元（`ops.rs`）が
    /// 検証済みの契約。`n_samples == 0` はディスパッチを回避し `0.0`
    /// を返す（`Mean`／`Sum` いずれも空和の契約。`backend-cpu::nll` と
    /// 同じ）。
    pub fn run_nll_loss_f32(
        &self,
        ctx: &MetalContext,
        input: &[f32],
        targets: &[i32],
        layout: NllLayout,
        factor: f32,
    ) -> Result<f32, MetalError> {
        validate_nll_layout(layout)?;
        let n_samples = layout.n_samples();
        if n_samples == 0 {
            return Ok(0.0);
        }

        let input_buf = MetalBuffer::new_with_data(ctx, input)?;
        let targets_buf = MetalIndexBuffer::new_with_i32(ctx, targets)?;
        let num_tg = nll_num_threadgroups(n_samples);
        // `nll_partial_f32` は起動する `num_tg` 個の threadgroup それぞれ
        // が `partial[tg_id]` を必ず 1 回書く（`shaders/nll.metal` 参照）
        // ため `alloc_uninit_pooled` を使える。
        let partial_buf = MetalBuffer::alloc_uninit_pooled(ctx, num_tg)?;

        ctx.dispatch_sync(|encoder| {
            encode_partial_dispatch(
                encoder,
                &self.partial_f32,
                &input_buf,
                &targets_buf,
                &partial_buf,
                layout,
                n_samples as u32,
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

    /// backward: `dInput[(o·C+t_s)·inner+i] = −scale`（ターゲット位置
    /// 以外は `0.0`）。`targets` は非追跡のため `dInput` のみを返す
    /// 契約（`backend_ops.rs::BackendOps::nll_loss_backward` doc
    /// 参照）。`n_samples == 0` は `layout.numel()` 要素すべてゼロの
    /// `Vec` を返す。
    pub fn run_nll_backward_f32(
        &self,
        ctx: &MetalContext,
        targets: &[i32],
        layout: NllLayout,
        scale: f32,
    ) -> Result<Vec<f32>, MetalError> {
        validate_nll_layout(layout)?;
        let numel = layout.numel();
        let n_samples = layout.n_samples();

        // `numel == 0` の場合も `MetalBuffer::alloc_zeroed_pooled` は
        // 0 長を拒否する（`ZeroLengthAllocation`）ため、その場合は
        // ディスパッチ自体を回避し空 `Vec` を返す（`numel == 0` ⟹
        // `n_samples == 0` のため下記カーネル起動も同時に回避される）。
        if numel == 0 {
            return Ok(Vec::new());
        }

        let dinput_buf = MetalBuffer::alloc_zeroed_pooled(ctx, numel)?;
        if n_samples == 0 {
            return Ok(dinput_buf.read_to_vec());
        }

        let targets_buf = MetalIndexBuffer::new_with_i32(ctx, targets)?;

        ctx.dispatch_sync(|encoder| {
            encode_backward_dispatch(
                encoder,
                &self.backward_f32,
                &targets_buf,
                &dinput_buf,
                layout,
                n_samples as u32,
                scale,
            );
        })?;

        Ok(dinput_buf.read_to_vec())
    }
}

fn nll_dispatch_sizes(units: usize, threadgroup_width: usize) -> (MTLSize, MTLSize) {
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

/// forward 1 段目のエンコード（バッファ結線 index 0〜2・`outer`／
/// `num_classes`／`inner`／`n_samples` index 3〜6・`num_tg` 個の
/// threadgroup をディスパッチ）。[`MetalNll::run_nll_loss_f32`] が
/// [`MetalContext::dispatch_sync`] のクロージャから呼ぶ（`bce.rs::
/// encode_partial_dispatch` と同型）。
#[allow(clippy::too_many_arguments)]
fn encode_partial_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    input_buf: &MetalBuffer,
    targets_buf: &MetalIndexBuffer,
    partial_buf: &MetalBuffer,
    layout: NllLayout,
    n_samples: u32,
    num_tg: usize,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`bce.rs::encode_partial_dispatch` と同種のコメント参照）。各
    // バッファは本関数呼び出し元 `MetalNll::run_nll_loss_f32` の
    // `ctx.dispatch_sync` 呼び出し中は関数スコープで生存するため
    // エンコード完了まで有効である。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(targets_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(partial_buf.raw()), 0, 2);
    }

    let outer = layout.outer as u32;
    let num_classes = layout.num_classes as u32;
    let inner = layout.inner as u32;
    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。各値はローカル変数でポインタは本
    // 呼び出し中生存し、長さは `constant uint&` 宣言の型と揃えている
    // （`shaders/nll.metal` 参照）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&outer).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&num_classes).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inner).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&n_samples).cast(),
            std::mem::size_of::<u32>(),
            6,
        );
    }

    let (threadgroups, threads_per_tg) = nll_dispatch_sizes(num_tg, NLL_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// forward 2 段目のエンコード（`partial`／`out` index 0〜1・
/// `num_partials`／`factor` index 2〜3・単一 threadgroup をディスパッチ。
/// `bce.rs::encode_finalize_dispatch` と同型）。
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

    // 単一 threadgroup（`shaders/nll.metal::nll_finalize_f32` の契約）。
    let (threadgroups, threads_per_tg) = nll_dispatch_sizes(1, NLL_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// backward のエンコード（`targets`／`dinput` index 0〜1・`outer`／
/// `num_classes`／`inner`／`n_samples`／`scale` index 2〜6・1 スレッド
/// 1 サンプルの 1 次元ディスパッチ）。
#[allow(clippy::too_many_arguments)]
fn encode_backward_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    targets_buf: &MetalIndexBuffer,
    dinput_buf: &MetalBuffer,
    layout: NllLayout,
    n_samples: u32,
    scale: f32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_partial_dispatch` と同じ根拠。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(targets_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dinput_buf.raw()), 0, 1);
    }
    let outer = layout.outer as u32;
    let num_classes = layout.num_classes as u32;
    let inner = layout.inner as u32;
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&outer).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&num_classes).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inner).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&n_samples).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&scale).cast(),
            std::mem::size_of::<f32>(),
            6,
        );
    }

    let groups = (n_samples as usize).div_ceil(NLL_THREADGROUP_WIDTH);
    let (threadgroups, threads_per_tg) = nll_dispatch_sizes(groups, NLL_THREADGROUP_WIDTH);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
