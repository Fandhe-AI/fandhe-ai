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
//!
//! イシュー #1690（親 #1582）で本モジュールに残っていた 3 箇所の
//! `ctx.dispatch_sync`（forward の `mse_partial_f32`／`mse_finalize_f32`
//! 2 段・backward の `mse_backward_f32` 1 段）を `ctx.encode` +
//! [`fandhe_ai_tensor_core::DispatchFailureCell`] 登録へ切り替えた
//! （`sgd.rs::MetalSgd::run` と同型のパターン。`docs/backend-metal-
//! command-batching-design.md` §3.7）。forward は 2 段目の間にある
//! `partial_buf`／`out_buf` の `alloc_uninit_pooled` 確保が
//! `zero_on_reuse=false`（＝同期を強制しない。`buffer.rs` 参照）ため、
//! 2 encode を同一バッチへ積んでから最後に 1 回だけ
//! `ctx.synchronize()` する形へ実際に集約できる（2 wait → 1 wait）。
//! backward は呼び出し元 `crates/autodiff/src/grad.rs`（`Op::MseLoss`
//! の VJP）が戻り値 `Tensor<f32>`（`dpred`）へ直後に `dense_vec(&dpred)`
//! （ホスト即時アクセス）で `dtarget` を計算する契約のため、本関数は
//! 関数を出る前に必ず自ら `ctx.synchronize()` する必要があり、wait 回数
//! は 1→1 のまま変わらない（詳細は `run_mse_backward_f32_tracked` の
//! doc コメント）。

use fandhe_ai_tensor_core::DispatchFailureCell;
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
    ///
    /// 内部でローカルの [`DispatchFailureCell`] を生成し
    /// [`Self::run_mse_loss_f32_tracked`] へ委譲する薄いラッパー
    /// （イシュー #1690）。既存の公開シグネチャ・戻り値契約（GPU 実行
    /// 完了まで待って `f32` を返す）は不変。
    pub fn run_mse_loss_f32(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        factor: f32,
    ) -> Result<f32, MetalError> {
        let token = DispatchFailureCell::new();
        self.run_mse_loss_f32_tracked(ctx, pred, target, factor, &token)
    }

    /// [`Self::run_mse_loss_f32`] の本体（イシュー #1690）。forward
    /// 1・2 段目とも `ctx.encode`（待たない）で同一バッチへ積み、最後に
    /// 1 回だけ `ctx.synchronize()` する（2 wait → 1 wait。モジュール
    /// 冒頭コメント参照）。`token` はバッチ全体で共有する
    /// `DispatchFailureCell` で、両 `encode` 呼び出しがそれぞれ登録する
    /// （first-writer-wins のため二重登録しても安全）。
    ///
    /// `ctx.synchronize()` が `Ok` を返した直後にも `token.is_set()` を
    /// 検査する: プロセスワイド singleton `MetalContext` を使う別スレッド
    /// が本関数の `synchronize()` より先に同じバッチを `synchronize()`
    /// し尽くしていた場合、本関数自身の `synchronize()` は「待つべき
    /// バッチが既に空」で無条件に `Ok(())` を返してしまい、本来検出す
    /// べき実行時エラーを見逃す（fail-open）おそれがある
    /// （`docs/backend-metal-command-batching-design.md` §3.7 (2) と
    /// 同種の競合）。`token` は `encode` と同一ロック区間で登録済みの
    /// ため、他スレッドの `synchronize()` がエラーを検出していれば
    /// 確実に `set()` 済みであり、`is_set()` の追加検査で fail-closed に
    /// 拾い上げる（`.claude/rules/security.md` A08）。
    pub fn run_mse_loss_f32_tracked(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        factor: f32,
        token: &DispatchFailureCell,
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

        ctx.encode(
            "mse_partial_f32",
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
                    num_tg,
                );
            },
        )?;

        // `alloc_uninit_pooled` は `zero_on_reuse=false`（未初期化確保）
        // のため、この 2 段目の確保自体が 1 段目の GPU 完了を待つ同期
        // 境界にはならない（モジュール冒頭コメント参照）。したがって
        // 1・2 段目とも待たずに同一バッチへ積める。
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, 1)?;
        ctx.encode(
            "mse_finalize_f32",
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

    /// backward: `dPred[i] = scale·(pred[i]−target[i])`。`dTarget` は
    /// 呼び出し元がホスト側で符号反転して得る契約（`backend_ops.rs::
    /// BackendOps::mse_loss_backward` doc 参照）のため、本関数は
    /// `dPred` のみを計算する。`numel == 0` は空 `Vec` を返す。
    ///
    /// 内部でローカルの [`DispatchFailureCell`] を生成し
    /// [`Self::run_mse_backward_f32_tracked`] へ委譲する薄いラッパー
    /// （イシュー #1690）。既存の公開シグネチャ・戻り値契約は不変。
    pub fn run_mse_backward_f32(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, MetalError> {
        let token = DispatchFailureCell::new();
        self.run_mse_backward_f32_tracked(ctx, pred, target, scale, &token)
    }

    /// [`Self::run_mse_backward_f32`] の本体（イシュー #1690）。ディス
    /// パッチ自体は `ctx.encode`（待たない）へ切り替えたが、呼び出し元
    /// `crates/autodiff/src/grad.rs`（`Op::MseLoss` の VJP）が戻り値
    /// `Tensor<f32>`（`dpred`）に対して直後に `dense_vec(&dpred)`（ホスト
    /// 即時アクセス）で `dtarget` を計算する契約のため、本関数は関数を
    /// 出る前に**必ず自ら** `ctx.synchronize()` する（無条件）。
    /// backward は 1 encode のみのため、この待ちは encode-only 化の
    /// 前後で 1→1 のまま変わらない（forward の 2→1 とは異なり待ち回数
    /// の削減効果はない。モジュール冒頭コメント参照。PR 本文にも明記）。
    ///
    /// `token.is_set()` の追加検査は
    /// [`Self::run_mse_loss_f32_tracked`] と同じ理由
    /// （別スレッドが先に同じバッチを drain してしまう競合への fail-
    /// closed な保険）。
    pub fn run_mse_backward_f32_tracked(
        &self,
        ctx: &MetalContext,
        pred: &[f32],
        target: &[f32],
        scale: f32,
        token: &DispatchFailureCell,
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

        ctx.encode(
            "mse_backward_f32",
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
                    scale,
                );
            },
        )?;

        check_dispatch_token(ctx.synchronize(), token)?;

        Ok(dpred_buf.read_to_vec())
    }
}

/// `ctx.synchronize()` の戻り値を `token` の状態と突き合わせる共通
/// ヘルパ（[`MetalMse::run_mse_loss_f32_tracked`]／
/// [`MetalMse::run_mse_backward_f32_tracked`] で共有。イシュー #1690）。
///
/// `sync_result` が `Err` ならそれをそのまま伝播する（`token` への
/// 二重報告はしない）。`Ok(())` の場合のみ `token.is_set()` を検査し、
/// 設定済みなら（別スレッドが先に同じバッチを `synchronize()` して
/// 実行時エラーを消費してしまった競合。両関数の doc コメント参照）
/// [`MetalError::CommandBufferExecutionFailed`] として fail-closed に
/// 返す。
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
/// 3・`num_tg` 個の threadgroup をディスパッチ）。
/// [`MetalMse::run_mse_loss_f32_tracked`] が [`MetalContext::encode`]
/// のクロージャから呼ぶ（イシュー #1690。旧 `ctx.dispatch_sync` から
/// 切替）。
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
    // 各バッファは `MetalMse::run_mse_loss_f32_tracked` が `ctx.encode`
    // の `resources` 引数として渡し `Batch::in_flight` へ retain
    // される、または本関数呼び出し中は関数スコープで生存するため
    // （イシュー #1690。`sgd.rs::encode_sgd_dispatch` と同種の根拠）、
    // エンコード完了まで有効である。
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
