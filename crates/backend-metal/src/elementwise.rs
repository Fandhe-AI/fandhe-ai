//! elementwise（`add`／`mul`／`relu`／`exp`／`tanh`）の起動 API（イシュー
//! #605。CUDA 側 `backend-cuda::elementwise`〈#599〉の Metal 対応版）。
//!
//! [`MetalElementwise::new`] が `shaders/elementwise.metal` を実行時
//! コンパイルして 5 パイプラインを保持し、[`MetalElementwise::run_add_f32`]
//! 等へホスト側スライスを渡すだけでバッファ確保・ディスパッチ・readback を
//! 内部で完結できる（`crate::gemm::MetalGemm`・`crate::rmsnorm::MetalRmsNorm`
//! と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps` から `BackendOps::add`／`mul`／`relu`／`exp`／
//! `tanh` の実装として呼ばれる。ブロードキャスト対応（NumPy 互換）は
//! `ops.rs` 側が `Tensor::broadcast_with` → `contiguous()` で同一 shape の
//! 密なバッファへ実体化してから本モジュールへ渡す契約（本モジュール自体は
//! 同一長バッファの 1:1 演算のみを扱う。`shaders/elementwise.metal` 冒頭
//! コメント「ブロードキャスト」参照）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/elementwise.metal` のソース（5 カーネルを含む）。
const ELEMENTWISE_MSL_SRC: &str = include_str!("shaders/elementwise.metal");

/// 1 スレッドグループあたりのスレッド数（1 次元）。`crate::gemm` の
/// `THREADGROUP_SIDE`（16×16・2 次元）とは無関係の独立したパラメータ
/// （elementwise カーネルは threadgroup 共有メモリを使わないため、幅は
/// オキュパンシ最適化のみが関心事）。PoC 実測なしの保守的な固定値
/// （CUDA 側 `kernels_elementwise::EW_BLOCK_DIM` と同じ 256）とし、
/// チューニングは別イシューのスコープとする（out-of-scope-tracking.md 対象）。
const EW_THREADGROUP_WIDTH: usize = 256;

/// `a_len`／`b_len` が一致することを検証する（二項演算向け）。
///
/// `pub(crate)`: 実機非依存の単体テスト（本ファイル末尾 `#[cfg(test)]`）
/// から直接呼べるよう公開範囲をクレート内に限定する（CUDA 側
/// `elementwise.rs::validate_elementwise_binary_dims` と同じ設計）。
pub(crate) fn validate_elementwise_binary_dims(
    a_len: usize,
    b_len: usize,
) -> Result<(), MetalError> {
    if a_len != b_len {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!("elementwise length mismatch: a_len={a_len}, b_len={b_len}"),
        });
    }
    Ok(())
}

/// elementwise 5 カーネル（`add`／`mul`／`relu`／`exp`／`tanh`。いずれも
/// f32）のコンパイル済みパイプラインを保持するハンドル。
pub struct MetalElementwise {
    add_f32: objc2::rc::Retained<MtlPipeline>,
    mul_f32: objc2::rc::Retained<MtlPipeline>,
    relu_f32: objc2::rc::Retained<MtlPipeline>,
    exp_f32: objc2::rc::Retained<MtlPipeline>,
    tanh_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalElementwise {
    /// `ctx` のデバイス上で elementwise 5 カーネルを実行時コンパイルし
    /// パイプラインを構築する。
    ///
    /// `pipeline::compile_options()`（`MathMode::Safe` +
    /// `MathFloatingPointFunctions::Precise`）を使う点は
    /// `crate::pipeline::compile_gemm_library`・`crate::rmsnorm::MetalRmsNorm::new`
    /// と同一であり、両カーネル間で丸め・関数ディスパッチ先の精度契約が
    /// 揃うことを保証する。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(ELEMENTWISE_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let add_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_add_f32")?;
        let mul_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_mul_f32")?;
        let relu_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_relu_f32")?;
        let exp_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_exp_f32")?;
        let tanh_f32 = pipeline::make_pipeline(ctx.device(), &library, "ew_tanh_f32")?;

        Ok(Self {
            add_f32,
            mul_f32,
            relu_f32,
            exp_f32,
            tanh_f32,
        })
    }

    /// 二項演算共通の起動手続き（バッファ確保 → ディスパッチ → readback）。
    ///
    /// `a.len() == 0`（呼び出し元の shape が空要素）の場合はカーネル起動
    /// 自体を回避し空の結果を返す（`crate::gemm::MetalGemm::dispatch_variant`
    /// の `m == 0 || n == 0` 早期 return と同じ理由。0 バイトバッファ確保は
    /// `crate::buffer::MetalBuffer` が `ZeroLengthAllocation` として拒否する
    /// ため、その手前で回避する）。
    fn run_binary(
        &self,
        ctx: &MetalContext,
        pipeline: &MtlPipeline,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        validate_elementwise_binary_dims(a.len(), b.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let b_buf = MetalBuffer::new_with_data(ctx, b)?;
        let out_buf = MetalBuffer::new_zeroed(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_binary_dispatch(encoder, pipeline, &a_buf, &b_buf, &out_buf, numel as u32);
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// 単項演算共通の起動手続き。[`Self::run_binary`] と同一構造。
    fn run_unary(
        &self,
        ctx: &MetalContext,
        pipeline: &MtlPipeline,
        a: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let out_buf = MetalBuffer::new_zeroed(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_unary_dispatch(encoder, pipeline, &a_buf, &out_buf, numel as u32);
        })?;

        Ok(out_buf.read_to_vec())
    }

    /// `out[i] = a[i] + b[i]`（f32・同一長）。
    pub fn run_add_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        self.run_binary(ctx, &self.add_f32, a, b)
    }

    /// `out[i] = a[i] * b[i]`（f32・同一長）。
    pub fn run_mul_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, MetalError> {
        self.run_binary(ctx, &self.mul_f32, a, b)
    }

    /// `out[i] = max(a[i], 0)`（f32）。
    pub fn run_relu_f32(&self, ctx: &MetalContext, a: &[f32]) -> Result<Vec<f32>, MetalError> {
        self.run_unary(ctx, &self.relu_f32, a)
    }

    /// `out[i] = exp(a[i])`（f32、`metal::precise::exp`）。
    pub fn run_exp_f32(&self, ctx: &MetalContext, a: &[f32]) -> Result<Vec<f32>, MetalError> {
        self.run_unary(ctx, &self.exp_f32, a)
    }

    /// `out[i] = tanh(a[i])`（f32、`metal::precise::tanh`）。
    pub fn run_tanh_f32(&self, ctx: &MetalContext, a: &[f32]) -> Result<Vec<f32>, MetalError> {
        self.run_unary(ctx, &self.tanh_f32, a)
    }
}

/// `numel` に対する grid/threadgroup サイズを構築する（`div_ceil` による
/// 末尾ブロックの余剰スレッドはカーネル内境界チェックに委ねる契約。REQ-8。
/// `crate::gemm` の `THREADGROUP_SIDE`／grid 計算と同じ考え方の 1 次元版）。
fn ew_dispatch_sizes(numel: u32) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: EW_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(EW_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// 二項カーネル共通のエンコード（バッファ結線 index 0〜2・`numel`
/// index 3・ディスパッチ）。[`MetalElementwise::run_binary`] が
/// [`MetalContext::dispatch_sync`] のクロージャから呼ぶ。
fn encode_binary_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY（FFI 境界 1/2）: `setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`crate::gemm::encode_dispatch` の同種コメント参照）。`a_buf`／
    // `b_buf`／`out_buf` は呼び出し元 `ctx.dispatch_sync` が完了するまで
    // 生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 2);
    }

    // SAFETY（FFI 境界 2/2）: `setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する。`numel` はローカル変数でありポインタは
    // 本呼び出し中生存し、長さは `size_of::<u32>()` と一致する
    // （`shaders/elementwise.metal` の `constant uint& numel` 宣言と型を
    // 揃える）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
    }

    let (threadgroups, threads_per_tg) = ew_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// 単項カーネル共通のエンコード（バッファ結線 index 0〜1・`numel`
/// index 2・ディスパッチ）。[`encode_binary_dispatch`] と同一構造だが
/// バッファ引数が 1 つ少ないため index が 1 つずつ前へずれる
/// （`shaders/elementwise.metal` の `ew_relu_f32`／`ew_exp_f32`／
/// `ew_tanh_f32` のバッファ宣言と一致させる）。
fn encode_unary_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `encode_binary_dispatch` と同一の根拠（該当コメント参照）。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: `encode_binary_dispatch` と同一の根拠（該当コメント参照）。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
    }

    let (threadgroups, threads_per_tg) = ew_dispatch_sizes(numel);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_elementwise_binary_dims_accepts_matching_lengths() {
        assert!(validate_elementwise_binary_dims(4, 4).is_ok());
    }

    #[test]
    fn validate_elementwise_binary_dims_rejects_length_mismatch() {
        let err = validate_elementwise_binary_dims(4, 5).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn ew_dispatch_sizes_covers_all_elements_with_div_ceil() {
        let (threadgroups, threads_per_tg) = ew_dispatch_sizes(EW_THREADGROUP_WIDTH as u32 + 1);
        assert_eq!(threads_per_tg.width, EW_THREADGROUP_WIDTH);
        assert_eq!(threadgroups.width, 2);
    }
}
