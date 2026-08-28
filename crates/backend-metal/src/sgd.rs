//! デバイス上パラメータ更新（SGD in-place）の起動 API（イシュー #935・
//! `docs/device-resident-update-design.md` §3.2。CUDA 側
//! `backend-cuda::sgd`〈同イシュー〉の Metal 対応版）。
//!
//! `elementwise.rs::MetalElementwise` と同じ構成方針を踏襲する:
//! [`MetalSgd::new`] が `shaders/sgd.metal` を実行時コンパイルして
//! パイプラインを保持し、[`MetalSgd::run`] へ `MetalBuffer` を渡すだけで
//! ディスパッチできる。`elementwise.rs` と異なり、本モジュールはホスト
//! 常駐 `&[f32]` を受け取らず**デバイス上に既に存在する `MetalBuffer` を
//! 直接読み書きする**（`StorageModeShared` の UMA でもホスト側
//! `Vec<f32>` の再構築・再アップロードが発生する `MetalMemory::upload`
//! を毎ステップ避けるのが目的。本イシューの主目的）。
//!
//! `ops.rs::MetalBackendOps::sgd_step_device` から
//! `BackendOps::sgd_step_device` の実装として呼ばれる。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/sgd.metal` のソース。
const SGD_MSL_SRC: &str = include_str!("shaders/sgd.metal");

/// 1 スレッドグループあたりのスレッド数（1 次元）。`elementwise.rs::
/// EW_THREADGROUP_WIDTH` と同じ値・同じ理由。
const SGD_THREADGROUP_WIDTH: usize = 256;

/// 長さが `u32::MAX` に収まることを検証する（`elementwise.rs::
/// validate_elementwise_len` と同じ理由）。
pub(crate) fn validate_sgd_len(len: usize) -> Result<(), MetalError> {
    if len > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "sgd_step_device numel must fit in u32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

fn sgd_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: SGD_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(SGD_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// `sgd_step_f32`（`shaders/sgd.metal`）のコンパイル済みパイプラインを
/// 保持するハンドル。
pub struct MetalSgd {
    sgd_step_f32: objc2::rc::Retained<MtlPipeline>,
}

/// `SgdStepConfig`（`tensor-core::backend_ops`）と同一のハイパー
/// パラメータをカーネル起動用にまとめたもの（`backend-cuda::sgd::
/// SgdKernelParams` の Metal 対応）。
pub struct SgdKernelParams {
    pub lr: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
    pub is_first_step: bool,
}

impl MetalSgd {
    /// `ctx` のデバイス上で `sgd_step_f32` カーネルを実行時コンパイルし
    /// パイプラインを構築する（`elementwise.rs::MetalElementwise::new` と
    /// 同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(SGD_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;
        let sgd_step_f32 = pipeline::make_pipeline(ctx.device(), &library, "sgd_step_f32")?;
        Ok(Self { sgd_step_f32 })
    }

    /// SGD 1 ステップを in-place で実行する（readback なし）。
    ///
    /// `param`／`grad` は同じ要素数を要求する（呼び出し元
    /// `ops.rs::MetalBackendOps::sgd_step_device` が shape 検証済みの
    /// バッファを渡す契約）。`velocity` は momentum 有効時のみ `Some` を
    /// 要求する。`numel == 0` の場合はディスパッチ自体を回避する
    /// （`elementwise.rs::MetalElementwise::run_binary` と同じ理由）。
    pub fn run(
        &self,
        ctx: &MetalContext,
        param: &MetalBuffer,
        grad: &MetalBuffer,
        velocity: Option<&MetalBuffer>,
        numel: usize,
        params: &SgdKernelParams,
    ) -> Result<(), MetalError> {
        validate_sgd_len(numel)?;
        if numel == 0 {
            return Ok(());
        }

        let use_momentum = if velocity.is_some() { 1i32 } else { 0i32 };
        // `use_momentum == 0` の場合、カーネルは `velocity` 引数を一切
        // 読み書きしないため、未使用のダミーとして `grad` を渡す
        // （`param` は書き込み用バッファとして別 index に既に bind 済み
        // だが、Metal の `setBuffer` は CUDA の `LaunchArgs::arg` と異なり
        // 借用を起動まで保持する制約がないため、`param` を再度渡しても
        // 実装上は問題ない。`grad` を使うのは CUDA 側実装との対称性の
        // ため）。
        let velocity_buf = velocity.unwrap_or(grad);

        ctx.dispatch_sync(|encoder| {
            encode_sgd_dispatch(
                encoder,
                &self.sgd_step_f32,
                param,
                grad,
                velocity_buf,
                numel as u32,
                params,
                use_momentum,
            );
        })
    }
}

/// SGD カーネルのエンコード（バッファ結線 index 0〜2・スカラー index
/// 3〜10・ディスパッチ）。[`MetalSgd::run`] が
/// [`MetalContext::dispatch_sync`] のクロージャから呼ぶ
/// （`elementwise.rs::encode_binary_dispatch` と同型）。
#[allow(clippy::too_many_arguments)]
fn encode_sgd_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    param_buf: &MetalBuffer,
    grad_buf: &MetalBuffer,
    velocity_buf: &MetalBuffer,
    numel: u32,
    params: &SgdKernelParams,
    use_momentum: i32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`elementwise.rs::encode_binary_dispatch` と同種のコメント参照）。
    // `param_buf`／`grad_buf`／`velocity_buf` は呼び出し元
    // `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(param_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(grad_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(velocity_buf.raw()), 0, 2);
    }

    let nesterov_i = if params.nesterov { 1i32 } else { 0i32 };
    let is_first_step_i = if params.is_first_step { 1i32 } else { 0i32 };

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。各ローカル変数はこの呼び出し中生存
    // し、長さは対応する `constant` 宣言の型（`uint`／`float`／`int`。
    // `shaders/sgd.metal` 参照）と揃えている
    // （`elementwise.rs::encode_binary_dispatch` と同種のコメント参照）。
    unsafe {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&numel).cast(), 4, 3);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.lr).cast(), 4, 4);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.momentum).cast(), 4, 5);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.dampening).cast(), 4, 6);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&params.weight_decay).cast(), 4, 7);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&nesterov_i).cast(), 4, 8);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&is_first_step_i).cast(), 4, 9);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&use_momentum).cast(), 4, 10);
    }

    let (threadgroups, threads_per_tg) = sgd_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
