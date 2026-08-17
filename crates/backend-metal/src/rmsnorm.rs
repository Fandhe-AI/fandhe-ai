//! 融合 RMSNorm 順伝播カーネルの起動 API（イシュー #604。CUDA 側
//! `backend-cuda::rmsnorm`〈#592・G-6〉の Metal 対応版）。
//!
//! [`MetalRmsNorm::new`] が `shaders/rmsnorm.metal`（1 パス／2 パスの
//! 2 エントリ）を実行時コンパイルしてパイプラインを保持し、
//! [`MetalRmsNorm::run_rmsnorm_f32_raw`] へホスト側スライスを渡すだけで
//! 経路選択（[`crate::row_kernel::select_route`]）・persistent grid 導出
//! （[`crate::row_kernel::derive_persistent_grid`]）・バッファ確保・
//! ディスパッチ・readback を内部で完結できる（`crate::gemm::MetalGemm`
//! と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps::run_fused` から canonical RMSNorm プラン
//! （`x * rsqrt(sum(x^2))`。`mean`／`eps`／`weight` を含まない）検出時に
//! 呼ばれる（[`crate::row_kernel::match_rmsnorm_plan`] 参照）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};
use crate::row_kernel::{
    self, ONEPASS_SMEM_BYTES_PER_GROUP, RowKernelRoute, RowKernelValidationError,
};

/// `shaders/rmsnorm.metal` のソース（1 パス／2 パスの 2 カーネルを含む）。
const RMSNORM_MSL_SRC: &str = include_str!("shaders/rmsnorm.metal");

/// カーネル起動時の threadgroup 幅（32 スレッド = 1 simdgroup 固定。
/// `shaders/rmsnorm.metal` 冒頭コメント「1 threadgroup = 1 simdgroup 固定」
/// と一致させる）。
const RMSNORM_THREADGROUP_WIDTH: usize = 32;

/// [`RowKernelValidationError`] → [`MetalError`] の変換（`softmax.rs` と
/// 共有できる形にせず独立に持つ理由: `MetalError` は `#[non_exhaustive]`
/// の列挙で新規バリアントを両カーネルが個別の意味を持って追加しうるため。
/// 現状は同一メッセージ形式で `MetalError::InvalidRowKernelShape` へ丸める）。
fn map_validation_error(err: RowKernelValidationError) -> MetalError {
    MetalError::InvalidRowKernelShape {
        detail: err.to_string(),
    }
}

/// 融合 RMSNorm 順伝播カーネル（1 パス／2 パスの 2 エントリ）のコンパイル
/// 済みパイプラインを保持するハンドル。
pub struct MetalRmsNorm {
    onepass: objc2::rc::Retained<MtlPipeline>,
    twopass: objc2::rc::Retained<MtlPipeline>,
}

impl MetalRmsNorm {
    /// `ctx` のデバイス上で RMSNorm 2 カーネルを実行時コンパイルし
    /// パイプラインを構築する。
    ///
    /// `threadExecutionWidth`（構築後のパイプラインが報告する実際の
    /// simdgroup 幅）が 32 と一致することも検証する（fail-closed。
    /// `shaders/rmsnorm.metal` は 1 threadgroup = 1 simdgroup = 32 スレッド
    /// 固定を前提にしており、これが崩れると reduction・persistent grid
    /// 導出の意味論が壊れるため、実行時に不一致を検出する）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(RMSNORM_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let onepass = pipeline::make_pipeline(ctx.device(), &library, "rmsnorm_f32_onepass")?;
        let twopass = pipeline::make_pipeline(ctx.device(), &library, "rmsnorm_f32_twopass")?;

        for pipeline in [&onepass, &twopass] {
            let width = pipeline.threadExecutionWidth();
            if width != RMSNORM_THREADGROUP_WIDTH {
                return Err(MetalError::UnexpectedThreadExecutionWidth {
                    expected: RMSNORM_THREADGROUP_WIDTH,
                    actual: width,
                });
            }
        }

        Ok(Self { onepass, twopass })
    }

    /// 標準 RMSNorm（mean 正規化あり）: `out = x * rsqrt(mean(x^2, axis=-1)
    /// + eps) * w`（`w` が `None` の場合は乗算をスキップ）を実行する。
    ///
    /// `inv_n = 1/hidden` を内部導出し [`Self::run_rmsnorm_f32_raw`] へ
    /// 委譲する（CUDA 側 `CudaRmsNorm::run_rmsnorm_f32` と同じ構成。
    /// `ops.rs::MetalBackendOps::run_fused` は canonical プランの意味論に
    /// 厳密一致させるため本メソッドを経由せず `run_rmsnorm_f32_raw` を
    /// `inv_n = 1.0` で直接呼ぶ）。
    pub fn run_rmsnorm_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, MetalError> {
        if rows == 0 || hidden == 0 {
            return self.run_rmsnorm_f32_raw(ctx, x, w, eps, 1.0, rows, hidden);
        }
        let inv_n = 1.0f32 / hidden as f32;
        self.run_rmsnorm_f32_raw(ctx, x, w, eps, inv_n, rows, hidden)
    }

    /// `out = x * rsqrt(sum(x^2, axis=-1) * inv_n + eps) * w`（`w` が
    /// `None` の場合は `has_weight = 0` で乗算をスキップ）を実行する内部
    /// エントリ（CUDA 側 `run_rmsnorm_f32_raw` と同じ二重入口構成。
    /// `ops.rs::MetalBackendOps::run_fused` は `inv_n = 1.0`・`eps = 0.0`
    /// で直接呼ぶ）。
    pub(crate) fn run_rmsnorm_f32_raw(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        inv_n: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, MetalError> {
        row_kernel::validate_row_kernel_launch(
            rows,
            hidden,
            x.len(),
            w.map(|s| s.len()),
            Some(eps),
        )
        .map_err(map_validation_error)?;

        if rows == 0 || hidden == 0 {
            return Ok(Vec::new());
        }

        let route = row_kernel::select_route(hidden);

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        // `w` が `None` の場合もカーネル引数としてバッファは必要。
        // `shaders/rmsnorm.metal` のスカラー正規化経路は
        // `(has_weight != 0) ? w[idx] : 1.0f` という条件式で `w` を参照する
        // が、Metal コンパイラがこれを分岐ではなく select（両辺を無条件に
        // 評価してから選択する命令）へ最適化する可能性があり、その場合
        // `w[idx]`（`idx` は最大 `hidden - 1`）が `has_weight == 0` でも
        // 実際にロードされうる。1 要素のダミーバッファでは `hidden > 1`
        // のとき範囲外読み出しになるため（REQ-8。CUDA 側は動的 SMEM の
        // 都合上バッファ長が別途カーネル内境界検査で守られているが、
        // Metal 側は select 化の可能性がある以上バッファ長自体で保証する）、
        // `hidden` 要素のゼロ初期化バッファを渡し、無条件ロードされても
        // 確保範囲内に収まるようにする（コンパイラの最適化戦略に依存しない
        // fail-closed な対策）。
        let (w_buf, has_weight) = match w {
            Some(w_slice) => (MetalBuffer::new_with_data(ctx, w_slice)?, 1i32),
            None => (MetalBuffer::new_zeroed(ctx, hidden)?, 0i32),
        };
        let out_buf = MetalBuffer::new_zeroed(ctx, x.len())?;

        let rows_u = rows as u32;
        let hidden_u = hidden as u32;

        let (pipeline, grid_size) = match route {
            RowKernelRoute::OnePass => {
                let grid_size = ctx.occupancy_params().map_or_else(
                    || row_kernel::derive_persistent_grid_fallback(rows_u),
                    |p| {
                        row_kernel::derive_persistent_grid(
                            p.gpu_core_count,
                            p.max_threadgroup_memory_bytes,
                            ONEPASS_SMEM_BYTES_PER_GROUP,
                            rows_u,
                        )
                    },
                );
                (&self.onepass, grid_size)
            }
            RowKernelRoute::TwoPass => {
                let grid_size = ctx.occupancy_params().map_or_else(
                    || row_kernel::derive_persistent_grid_fallback(rows_u),
                    |p| {
                        row_kernel::derive_persistent_grid(
                            p.gpu_core_count,
                            p.max_threadgroup_memory_bytes,
                            0,
                            rows_u,
                        )
                    },
                );
                (&self.twopass, grid_size)
            }
        };

        ctx.dispatch_sync(|encoder| {
            encode_rmsnorm_dispatch(
                encoder, pipeline, &x_buf, &w_buf, &out_buf, rows_u, hidden_u, eps, inv_n,
                has_weight, grid_size,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// パイプライン設定・バッファ結線（index 0〜2）・スカラー引数
/// （index 3〜8）の `setBuffer_offset_atIndex`／`setBytes_length_atIndex`・
/// `dispatchThreadgroups_threadsPerThreadgroup` を行う。
/// [`MetalRmsNorm::run_rmsnorm_f32_raw`] が [`MetalContext::dispatch_sync`]
/// のクロージャから呼ぶ（`crate::gemm::encode_dispatch` と同型の構成）。
#[allow(clippy::too_many_arguments)]
fn encode_rmsnorm_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    w_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    rows: u32,
    hidden: u32,
    eps: f32,
    inv_n: f32,
    has_weight: i32,
    grid_size: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY（FFI 境界 1/2）: `setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きはしない
    // （`crate::gemm::encode_dispatch` の同種コメント参照）。
    // `x_buf`/`w_buf`/`out_buf` は呼び出し元 `ctx.dispatch_sync` が
    // 完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(w_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 2);
    }

    // SAFETY（FFI 境界 2/2）: `setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する（`crate::gemm::encode_dispatch` と同じ
    // 「即時複製」契約）。各ローカル変数は本呼び出し中生存しており、
    // 型・バイト数はカーネル引数宣言（`shaders/rmsnorm.metal` の
    // `constant uint&`/`constant float&`/`constant int&`）と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rows).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&hidden).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&eps).cast(),
            std::mem::size_of::<f32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inv_n).cast(),
            std::mem::size_of::<f32>(),
            6,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&has_weight).cast(),
            std::mem::size_of::<i32>(),
            7,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&grid_size).cast(),
            std::mem::size_of::<u32>(),
            8,
        );
    }

    let threads_per_tg = MTLSize {
        width: RMSNORM_THREADGROUP_WIDTH,
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
