//! デバイス上パラメータ更新（Adam・AdamW in-place）の起動 API（イシュー
//! #2070・`docs/device-resident-update-design.md`「Adam／AdamW の常駐
//! step 結線」節の追補。CUDA 側 `backend-cuda::adam`〈イシュー #2069〉の
//! Metal 対応版）。
//!
//! `crate::sgd::MetalSgd` と同じ構成方針を踏襲する: [`MetalAdam::new`]
//! が `shaders/adam.metal` を実行時コンパイルしてパイプラインを保持し、
//! [`MetalAdam::run`] へ `MetalBuffer` を渡すだけでディスパッチできる。
//! `param`／`m`／`v` はホスト常駐往復なしでデバイス上に直接読み書きする
//! （本イシューの主目的。`crate::sgd` モジュール doc 参照）。
//!
//! `ops.rs::MetalBackendOps::adam_step_device_impl` から
//! `BackendOps::adam_step_device`／`adam_step_device_tracked` の実装
//! として呼ばれる。

use fandhe_ai_tensor_core::DispatchFailureCell;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::adam_model::AdamKernelFlags;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/adam.metal` のソース。
const ADAM_MSL_SRC: &str = include_str!("shaders/adam.metal");

/// 1 スレッドグループあたりのスレッド数（1 次元）。`sgd.rs::
/// SGD_THREADGROUP_WIDTH` と同じ値・同じ理由（PoC 実測なしの保守的な
/// 固定値）。
const ADAM_THREADGROUP_WIDTH: usize = 256;

/// 長さが `u32::MAX` に収まることを検証する（`sgd.rs::validate_sgd_len`
/// と同じ理由。カーネル引数 `constant uint& numel` の上限）。
pub(crate) fn validate_adam_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "adam_step_device numel must fit in u32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

fn adam_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: ADAM_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(ADAM_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// `adam_step_f32`（`shaders/adam.metal`）のコンパイル済みパイプライン
/// を保持するハンドル。
pub struct MetalAdam {
    adam_step_f32: objc2::rc::Retained<MtlPipeline>,
}

/// `AdamStepConfig`（`tensor-core::backend_ops`）と同一のハイパー
/// パラメータをカーネル起動用にまとめたもの（`backend-cuda::adam::
/// AdamKernelParams` の Metal 対応）。`decoupled`／`use_coupled_wd` は
/// `AdamStepKind` の分岐をホストで事前評価した bool 相当フラグ
/// （`crate::adam_model::validate_adam_step_inputs` が導出する
/// [`AdamKernelFlags`] から写す）。
pub struct AdamKernelParams {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub decay_factor: f32,
    pub step_size: f32,
    pub bias_correction2_sqrt: f32,
    pub decoupled: bool,
    pub use_coupled_wd: bool,
}

impl AdamKernelParams {
    /// `AdamStepConfig`（呼び出し元が保持するハイパーパラメータ）と
    /// [`AdamKernelFlags`]（`adam_model::validate_adam_step_inputs` が
    /// 導出したフラグ）から組み立てる（`ops.rs` 側の重複記述を避ける）。
    pub(crate) fn from_config(
        config: &fandhe_ai_tensor_core::AdamStepConfig,
        flags: AdamKernelFlags,
    ) -> Self {
        Self {
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
            decay_factor: config.decay_factor,
            step_size: config.step_size,
            bias_correction2_sqrt: config.bias_correction2_sqrt,
            decoupled: flags.decoupled,
            use_coupled_wd: flags.use_coupled_wd,
        }
    }
}

impl MetalAdam {
    /// `ctx` のデバイス上で `adam_step_f32` カーネルを実行時コンパイル
    /// しパイプラインを構築する（`sgd.rs::MetalSgd::new` と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(ADAM_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let adam_step_f32 = pipeline::make_pipeline(ctx.device(), &library, "adam_step_f32")?;
        Ok(Self { adam_step_f32 })
    }

    /// Adam・AdamW 1 ステップを in-place で実行する（readback なし）。
    ///
    /// `param`／`grad`／`m`／`v` は同じ要素数を要求する（呼び出し元
    /// `ops.rs::MetalBackendOps::adam_step_device_impl` が shape 検証
    /// 済みのバッファを渡す契約）。`numel == 0` の場合はディスパッチ
    /// 自体を回避する（`sgd.rs::MetalSgd::run` と同じ理由）。
    ///
    /// 同期契約は `sgd.rs::MetalSgd::run` と同一（イシュー #1017）:
    /// `token` が `None`（`ops.rs::MetalBackendOps::adam_step_device`
    /// 経由）の場合は `encode` 直後に本メソッド内で `ctx.synchronize()`
    /// まで行い、復帰時点で GPU 実行の完了・成否（`Result`）を呼び出し
    /// 元へ返す。`token` が `Some`（`adam_step_device_tracked` 経由）の
    /// 場合はバッチ化のため待たず、実行時エラーは `ctx.encode` と同一
    /// ロック区間でバッチへ登録した `token` へ、後続の
    /// `ctx.synchronize()`（ホスト実体化時）が検出した際に書き込まれる
    /// （非同期契約）。
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        ctx: &MetalContext,
        param: &MetalBuffer,
        grad: &MetalBuffer,
        m: &MetalBuffer,
        v: &MetalBuffer,
        numel: usize,
        params: &AdamKernelParams,
        token: Option<&DispatchFailureCell>,
    ) -> Result<(), MetalError> {
        validate_adam_len(numel)?;
        if numel == 0 {
            return Ok(());
        }

        // SAFETY: `ctx.encode` が積むバッチの `synchronize()` 完了まで、
        // `param`／`grad`／`m`／`v` を `Batch::in_flight` へ retain させる
        // 必要がある（`sgd.rs::MetalSgd::run` と同一の理由。`context.rs::
        // Batch` フィールド doc「§2.5」参照）。Adam は SGD と異なり
        // 4 バッファすべてが常に実体（`velocity` のような opt-in・
        // ダミーエイリアス引数は不要。`backend-cuda::adam::CudaAdam::run`
        // と同じ設計）。
        ctx.encode(
            "adam_step_f32",
            &[param.raw(), grad.raw(), m.raw(), v.raw()],
            token,
            |encoder| {
                encode_adam_dispatch(
                    encoder,
                    &self.adam_step_f32,
                    param,
                    grad,
                    m,
                    v,
                    numel as u32,
                    params,
                );
            },
        )?;

        // `token` が `None` の場合のみここで待つ（上記ドキュメント参照）。
        // `Some` の場合はバッチ化された非同期契約のため待たずに返す。
        match token {
            None => ctx.synchronize(),
            Some(_) => Ok(()),
        }
    }
}

/// Adam カーネルのエンコード（バッファ結線 index 0〜3・スカラー
/// index 4〜13・ディスパッチ）。[`MetalAdam::run`] が `ctx.encode` の
/// クロージャから呼ぶ（`sgd.rs::encode_sgd_dispatch` と同型）。
#[allow(clippy::too_many_arguments)]
fn encode_adam_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    param_buf: &MetalBuffer,
    grad_buf: &MetalBuffer,
    m_buf: &MetalBuffer,
    v_buf: &MetalBuffer,
    numel: u32,
    params: &AdamKernelParams,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`sgd.rs::encode_sgd_dispatch` と同種のコメント参照）。
    // `param_buf`／`grad_buf`／`m_buf`／`v_buf` は `MetalAdam::run` が
    // `ctx.encode` の `resources` へ渡し、`Batch::in_flight` へ retain
    // されるため `ctx.synchronize()` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(param_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(grad_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(m_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(v_buf.raw()), 0, 3);
    }

    let kind_decoupled_i: i32 = if params.decoupled { 1 } else { 0 };
    let use_coupled_wd_i: i32 = if params.use_coupled_wd { 1 } else { 0 };

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から指定バイト数を即座に複製する。各ローカル変数はこの呼び出し
    // 中生存し、長さは対応する `constant` 宣言の型（`uint`／`float`／
    // `int`。`shaders/adam.metal` 参照）と揃えている（`sgd.rs::
    // encode_sgd_dispatch` と同種のコメント参照）。
    unsafe {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&numel).cast(), 4, 4);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.beta1).cast(), 4, 5);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.beta2).cast(), 4, 6);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.eps).cast(), 4, 7);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.weight_decay).cast(), 4, 8);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.decay_factor).cast(), 4, 9);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.step_size).cast(), 4, 10);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&params.bias_correction2_sqrt).cast(),
            4,
            11,
        );
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&kind_decoupled_i).cast(), 4, 12);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&use_coupled_wd_i).cast(), 4, 13);
    }

    let (threadgroups, threads_per_tg) = adam_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_adam_len_accepts_boundary_and_rejects_overflow() {
        assert!(validate_adam_len(0).is_ok());
        assert!(validate_adam_len(u32::MAX as usize).is_ok());
        assert!(validate_adam_len(u32::MAX as usize + 1).is_err());
    }
}
