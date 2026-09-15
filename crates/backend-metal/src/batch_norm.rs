//! BatchNorm1d／2d 順伝播カーネルの起動 API（イシュー #1736・親
//! #1608。`layer_norm.rs`〈#1596〉と同じ構成方針を踏襲する Metal 対応
//! 版）。
//!
//! [`MetalBatchNorm::new`] が `shaders/batch_norm.metal` の 2 カーネル
//! （train／infer）を実行時コンパイルして両パイプラインを保持し、
//! [`MetalBatchNorm::run_batch_norm_train_f32`]／
//! [`MetalBatchNorm::run_batch_norm_infer_f32`] へホスト側スライスを
//! 渡すだけでバッファ確保・ディスパッチ・readback を内部で完結できる
//! （`crate::layer_norm::MetalLayerNorm` と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps::batch_norm_train`／`batch_norm_infer`
//! （`BackendOps::batch_norm_train`／`batch_norm_infer` の独立エントリ）
//! から直接呼ばれる。起動前 fail-closed 検証・添字写像・`M` の f64
//! ビット分割は `crate::batch_norm_model`（`cfg(target_os = "macos")`
//! なしの純関数層。Linux でも単体テストが回る）に集約する。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::batch_norm_model::{self, BatchNormPrepareError};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};
use crate::row_kernel;

/// `shaders/batch_norm.metal` のソース（train／infer 2 カーネルを
/// 含む単一ファイル）。
const BATCH_NORM_MSL_SRC: &str = include_str!("shaders/batch_norm.metal");

/// train カーネル起動時の threadgroup 幅（32 スレッド = 1 simdgroup
/// 固定。`shaders/batch_norm.metal` 冒頭コメント「1 threadgroup = 1
/// simdgroup 固定」と一致させる。`layer_norm.rs::
/// LAYER_NORM_THREADGROUP_WIDTH` と同じ役割）。
const BATCH_NORM_TRAIN_THREADGROUP_WIDTH: usize = 32;

/// infer カーネル起動時の threadgroup 幅（grid-stride 不要の単純
/// elementwise。`bce.rs::BCE_THREADGROUP_WIDTH` と同じ値）。
const BATCH_NORM_INFER_THREADGROUP_WIDTH: usize = 256;

/// [`BatchNormPrepareError`] → [`MetalError`] の変換
/// （`InvalidShape` → `InvalidBatchNormShape`・`SizeLimitExceeded` →
/// `BatchNormSizeLimitExceeded`。`ops.rs::map_batch_norm_error` が
/// 後者のみ `BackendError::Unsupported` へ写像する）。
fn map_prepare_error(err: BatchNormPrepareError) -> MetalError {
    match err {
        BatchNormPrepareError::InvalidShape { detail } => {
            MetalError::InvalidBatchNormShape { detail }
        }
        BatchNormPrepareError::SizeLimitExceeded { detail } => {
            MetalError::BatchNormSizeLimitExceeded { detail }
        }
    }
}

/// BatchNorm1d／2d 順伝播カーネル（train／infer）のコンパイル済み
/// パイプラインを保持するハンドル。
pub struct MetalBatchNorm {
    train_pipeline: objc2::rc::Retained<MtlPipeline>,
    infer_pipeline: objc2::rc::Retained<MtlPipeline>,
}

impl MetalBatchNorm {
    /// `ctx` のデバイス上で train／infer カーネルを実行時コンパイルし
    /// 両パイプラインを構築する。train 側の `threadExecutionWidth` が
    /// 32 と一致することも検証する（fail-closed。`MetalLayerNorm::new`
    /// と同じ理由）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(BATCH_NORM_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let train_pipeline =
            pipeline::make_pipeline(ctx.device(), &library, "batch_norm_train_f32")?;
        let width = train_pipeline.threadExecutionWidth();
        if width != BATCH_NORM_TRAIN_THREADGROUP_WIDTH {
            return Err(MetalError::UnexpectedThreadExecutionWidth {
                expected: BATCH_NORM_TRAIN_THREADGROUP_WIDTH,
                actual: width,
            });
        }
        let infer_pipeline =
            pipeline::make_pipeline(ctx.device(), &library, "batch_norm_infer_f32")?;

        Ok(Self {
            train_pipeline,
            infer_pipeline,
        })
    }

    /// BatchNorm1d／2d train モード（バッチ統計）を実行する。
    /// `x` は `[n, c, spatial]` 相当の行優先平坦化済みスライス
    /// （`docs/batch-norm-ops-design.md` §1 レイアウト契約）。
    /// `n == 0 || c == 0 || spatial == 0` は空出力・ゼロ統計を返す
    /// （`crate::batch_norm_model::batch_norm_train_host_model` と
    /// 同じ早期 return 契約）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_batch_norm_train_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<BatchNormTrainRaw, MetalError> {
        batch_norm_model::validate_batch_norm_launch(
            n,
            c,
            spatial,
            x.len(),
            w.map(|s| s.len()),
            b.map(|s| s.len()),
            None,
            None,
            eps,
        )
        .map_err(map_prepare_error)?;

        if n == 0 || c == 0 || spatial == 0 {
            return Ok(BatchNormTrainRaw {
                out: Vec::new(),
                mean: vec![0.0f32; c],
                var: vec![0.0f32; c],
            });
        }

        let m = n * spatial;

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        // `w`／`b` が `None` の場合もカーネル引数としてバッファは必要
        // （`layer_norm.rs::run_layer_norm_f32` の `w_buf` と同じ理由:
        // predicated load 対策。REQ-8）。
        let (w_buf, has_weight) = match w {
            Some(w_slice) => (MetalBuffer::new_with_data(ctx, w_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, c)?, 0i32),
        };
        let (b_buf, has_bias) = match b {
            Some(b_slice) => (MetalBuffer::new_with_data(ctx, b_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, c)?, 0i32),
        };
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, x.len())?;
        let mean_buf = MetalBuffer::alloc_uninit_pooled(ctx, c)?;
        let var_buf = MetalBuffer::alloc_uninit_pooled(ctx, c)?;

        let n_u = n as u32;
        let c_u = c as u32;
        let spatial_u = spatial as u32;
        let m_u = m as u32;
        let (m_f64_hi, m_f64_lo) = batch_norm_model::m_f64_bits(m);

        // persistent grid（threadgroup 数）を導出する（`row_kernel::
        // derive_persistent_grid` を「行」の代わりに「チャネル数 c」へ
        // 適用する——LayerNorm の `rows` と同じ役割。train カーネルは
        // threadgroup memory を使わないため `smem_bytes_per_group = 0`。
        // `layer_norm.rs::run_layer_norm_f32` と同じ経路）。
        let grid_size = ctx.occupancy_params().map_or_else(
            || row_kernel::derive_persistent_grid_fallback(c_u),
            |p| {
                row_kernel::derive_persistent_grid(
                    p.gpu_core_count,
                    p.max_threadgroup_memory_bytes,
                    0,
                    c_u,
                )
            },
        );

        ctx.dispatch_sync(|encoder| {
            encode_batch_norm_train_dispatch(
                encoder,
                &self.train_pipeline,
                &x_buf,
                &w_buf,
                &b_buf,
                &out_buf,
                &mean_buf,
                &var_buf,
                n_u,
                c_u,
                spatial_u,
                m_u,
                m_f64_hi,
                m_f64_lo,
                eps,
                has_weight,
                has_bias,
                grid_size,
            );
        })?;

        Ok(BatchNormTrainRaw {
            out: out_buf.read_to_vec(),
            mean: mean_buf.read_to_vec(),
            var: var_buf.read_to_vec(),
        })
    }

    /// BatchNorm1d／2d infer モード（固定統計）を実行する。`mean`／
    /// `var`（長さ `c`）はバッチから計算し直さずそのまま使う。
    #[allow(clippy::too_many_arguments)]
    pub fn run_batch_norm_infer_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        mean: &[f32],
        var: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<Vec<f32>, MetalError> {
        batch_norm_model::validate_batch_norm_launch(
            n,
            c,
            spatial,
            x.len(),
            w.map(|s| s.len()),
            b.map(|s| s.len()),
            Some(mean.len()),
            Some(var.len()),
            eps,
        )
        .map_err(map_prepare_error)?;

        if n == 0 || c == 0 || spatial == 0 {
            return Ok(Vec::new());
        }

        let numel = x.len();

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let mean_buf = MetalBuffer::new_with_data(ctx, mean)?;
        let var_buf = MetalBuffer::new_with_data(ctx, var)?;
        let (w_buf, has_weight) = match w {
            Some(w_slice) => (MetalBuffer::new_with_data(ctx, w_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, c)?, 0i32),
        };
        let (b_buf, has_bias) = match b {
            Some(b_slice) => (MetalBuffer::new_with_data(ctx, b_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, c)?, 0i32),
        };
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        let c_u = c as u32;
        let spatial_u = spatial as u32;
        let numel_u = numel as u32;

        ctx.dispatch_sync(|encoder| {
            encode_batch_norm_infer_dispatch(
                encoder,
                &self.infer_pipeline,
                &x_buf,
                &mean_buf,
                &var_buf,
                &w_buf,
                &b_buf,
                &out_buf,
                c_u,
                spatial_u,
                numel_u,
                eps,
                has_weight,
                has_bias,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// [`MetalBatchNorm::run_batch_norm_train_f32`] の戻り値（`crate::
/// batch_norm_model::BatchNormTrainHostModel` と同型のフィールド。
/// `out` は入力と同一 shape の平坦化データ、`mean`／`var`（biased
/// ÷M）は長さ `c`）。
#[derive(Debug)]
pub struct BatchNormTrainRaw {
    pub out: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

/// train カーネルのバッファ結線（index 0〜5）・スカラー引数
/// （index 6〜15）の起動処理（`layer_norm.rs::
/// encode_layer_norm_dispatch` と同型の構成）。
#[allow(clippy::too_many_arguments)]
fn encode_batch_norm_train_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    w_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    mean_buf: &MetalBuffer,
    var_buf: &MetalBuffer,
    n: u32,
    c: u32,
    spatial: u32,
    m: u32,
    m_f64_hi: u32,
    m_f64_lo: u32,
    eps: f32,
    has_weight: i32,
    has_bias: i32,
    grid_size: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`layer_norm.rs::encode_layer_norm_dispatch` の同種コメント
    // 参照）。全バッファは呼び出し元 `ctx.dispatch_sync` が完了する
    // まで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(w_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(mean_buf.raw()), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(var_buf.raw()), 0, 5);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から指定バイト数を即座に複製する（`layer_norm.rs` と同じ「即時
    // 複製」契約）。各ローカル変数は本呼び出し中生存しており、型・
    // バイト数はカーネル引数宣言（`shaders/batch_norm.metal` の
    // `constant uint&`/`constant float&`/`constant int&`）と一致させ
    // ている。
    unsafe {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&n).cast(), 4, 6);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&c).cast(), 4, 7);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&spatial).cast(), 4, 8);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&m).cast(), 4, 9);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&m_f64_hi).cast(), 4, 10);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&m_f64_lo).cast(), 4, 11);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&eps).cast(), 4, 12);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&has_weight).cast(), 4, 13);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&has_bias).cast(), 4, 14);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&grid_size).cast(), 4, 15);
    }

    let threads_per_tg = MTLSize {
        width: BATCH_NORM_TRAIN_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: grid_size as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// infer カーネルのバッファ結線（index 0〜5）・スカラー引数
/// （index 6〜11）の起動処理（grid-stride 不要の単純 elementwise。
/// `bce.rs` の dispatch サイズ算出と同型）。
#[allow(clippy::too_many_arguments)]
fn encode_batch_norm_infer_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    mean_buf: &MetalBuffer,
    var_buf: &MetalBuffer,
    w_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    c: u32,
    spatial: u32,
    numel: u32,
    eps: f32,
    has_weight: i32,
    has_bias: i32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2（train 側と同じ契約）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(mean_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(var_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(w_buf.raw()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 5);
    }

    // SAFETY: FFI 境界 2/2（train 側と同じ契約）。
    unsafe {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&c).cast(), 4, 6);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&spatial).cast(), 4, 7);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&numel).cast(), 4, 8);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&eps).cast(), 4, 9);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&has_weight).cast(), 4, 10);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&has_bias).cast(), 4, 11);
    }

    let threads_per_tg = MTLSize {
        width: BATCH_NORM_INFER_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(BATCH_NORM_INFER_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups.max(1),
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}
