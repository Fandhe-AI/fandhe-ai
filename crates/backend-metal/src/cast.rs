//! dtype 変換（`fandhe_ai_tensor_core::cast::CastOps`。イシュー #1751・
//! 親 #1613・依存 #1750）の起動 API（実行時コンパイル・パイプライン
//! 保持・実行）と `CastOps for MetalBackendOps` 実装。
//!
//! `unique.rs::MetalUnique`（CUDA 版 `unique.rs::CudaUnique` と対に
//! なる構成）と同じ方針を踏襲する: [`MetalCast::new`] が
//! `shaders/cast.metal` を実行時コンパイルしてパイプライン（6 個）を
//! 保持し、`run_*` へホスト側スライスを渡すだけでバッファ確保・
//! ディスパッチ・readback を内部で完結できる。`impl CastOps for
//! MetalBackendOps` は `crates/backend-cuda/src/cast.rs` と同じ配置
//! 規約（「impl は dtype ファイル・accessor は `ops.rs`」）で本
//! ファイルへ置く。
//!
//! # f64 2 方向は実装しない
//!
//! MSL は `double` 非対応のため `cast_f32_to_f64`／`cast_f64_to_f32`
//! は本ファイルにメソッド自体を実装しない（`CastOps` の既定
//! `Unsupported` のまま。`shaders/cast.metal` 冒頭コメント参照）。
//! `Var::cast::<f64>()`／`Tensor<f64>::to_f32()` を Metal バックエンド
//! で呼ぶと自動的にホスト参照実装へフォールバックする（`tensor-core::
//! cast` モジュール doc「フォールバック規則」）。
//!
//! # 数値契約
//!
//! 算術を含まない変換のため **bit 完全一致**（NaN のみクラス一致）。
//! カーネル記述規則の詳細は `shaders/cast.metal` 冒頭コメントを正とする。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{CastOps, Tensor};

use crate::buffer::MetalBuffer;
use crate::cast_buffer::MetalCastBuffer;
use crate::context::MetalContext;
use crate::context_cache;
use crate::error::MetalError;
use crate::ops::{MetalBackendOps, checked_shape_numel};
use crate::pipeline::{self, MtlPipeline};

/// `shaders/cast.metal` のソース。
const CAST_MSL_SRC: &str = include_str!("shaders/cast.metal");

/// 1 スレッドグループあたりのスレッド数（`gather_scatter.rs::
/// GS_THREADGROUP_WIDTH`・`unique.rs::UNIQUE_THREADGROUP_WIDTH` と
/// 同じ値・同じ判断根拠）。
const CAST_THREADGROUP_WIDTH: usize = 256;

/// `numel` がカーネル引数 `constant uint&` へ収まるかを検証する
/// （`checked_u32_numel`〈`impl CastOps for MetalBackendOps` の入口
/// 検査〉の `MetalError` 版）。
///
/// `MetalCast::run_*` は `pub` メソッドであり `impl CastOps for
/// MetalBackendOps`（`checked_u32_numel` を経由する）を経ずに直接
/// 呼び出せる。`u32::MAX` 超のスライスを直接渡すと、是正前は
/// `numel as u32` が下位ビットへ切り詰められ、`MetalCastBuffer::
/// new_zeroed`／`MetalBuffer::new_zeroed` が確保する出力バッファの
/// 要素数（切り詰め後の `numel_u`）とカーネルへ渡す境界検査用の値
/// （同じ切り詰め後の `numel_u`）は一致するため確保自体は成功するが、
/// 実際にホストから渡された入力の一部だけを変換した結果を
/// `Ok`（成功）として返してしまう（サイレントな部分変換。codex-review
/// P2 指摘・イシュー #1751）。各 `run_*` の確保直前でこの検証を行い、
/// 収まらない場合は確保前に拒否する（呼び出し元は形状理由の
/// `Unsupported` を経由してホストフォールバックへ委ねる設計のため、
/// ここでは `MetalError::ShapeMismatch` を返し `map_cast_error` に
/// 委ねる。`impl CastOps` の `checked_u32_numel` と多重検査になるが、
/// 公開入口〈`MetalCast::run_*` 自体〉での fail-closed 検査として
/// 独立して機能させる）。
fn checked_u32_numel_for_dispatch(numel: usize) -> Result<u32, MetalError> {
    u32::try_from(numel).map_err(|_| MetalError::ShapeMismatch {
        detail: format!("cast: numel={numel} exceeds u32::MAX (Metal kernel argument type)"),
    })
}

/// 6 方向の cast カーネルのコンパイル済みパイプラインを保持する
/// ハンドル。
pub struct MetalCast {
    cast_f32_to_i32: objc2::rc::Retained<MtlPipeline>,
    cast_f32_to_i64: objc2::rc::Retained<MtlPipeline>,
    cast_f32_to_bool: objc2::rc::Retained<MtlPipeline>,
    cast_i32_to_f32: objc2::rc::Retained<MtlPipeline>,
    cast_i64_to_f32: objc2::rc::Retained<MtlPipeline>,
    cast_bool_to_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalCast {
    /// `ctx` のデバイス上で 6 カーネルすべてを実行時コンパイルし
    /// パイプラインを構築する（`unique.rs::MetalUnique::new` と同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(CAST_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let cast_f32_to_i32 = pipeline::make_pipeline(ctx.device(), &library, "cast_f32_to_i32")?;
        let cast_f32_to_i64 = pipeline::make_pipeline(ctx.device(), &library, "cast_f32_to_i64")?;
        let cast_f32_to_bool = pipeline::make_pipeline(ctx.device(), &library, "cast_f32_to_bool")?;
        let cast_i32_to_f32 = pipeline::make_pipeline(ctx.device(), &library, "cast_i32_to_f32")?;
        let cast_i64_to_f32 = pipeline::make_pipeline(ctx.device(), &library, "cast_i64_to_f32")?;
        let cast_bool_to_f32 = pipeline::make_pipeline(ctx.device(), &library, "cast_bool_to_f32")?;

        Ok(Self {
            cast_f32_to_i32,
            cast_f32_to_i64,
            cast_f32_to_bool,
            cast_i32_to_f32,
            cast_i64_to_f32,
            cast_bool_to_f32,
        })
    }

    /// f32→i32（ゼロ方向切り捨て・範囲外は飽和・NaN→0）。
    pub fn run_f32_to_i32(&self, ctx: &MetalContext, x: &[f32]) -> Result<Vec<i32>, MetalError> {
        let numel = x.len();
        let numel_u = checked_u32_numel_for_dispatch(numel)?;
        let in_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalCastBuffer::<i32>::new_zeroed(ctx, numel)?;
        ctx.dispatch_sync(|encoder| {
            encode_cast_dispatch(
                encoder,
                &self.cast_f32_to_i32,
                in_buf.raw(),
                out_buf.raw(),
                numel_u,
            );
        })?;
        Ok(out_buf.read_to_vec())
    }

    /// f32→i64（同上）。
    pub fn run_f32_to_i64(&self, ctx: &MetalContext, x: &[f32]) -> Result<Vec<i64>, MetalError> {
        let numel = x.len();
        let numel_u = checked_u32_numel_for_dispatch(numel)?;
        let in_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalCastBuffer::<i64>::new_zeroed(ctx, numel)?;
        ctx.dispatch_sync(|encoder| {
            encode_cast_dispatch(
                encoder,
                &self.cast_f32_to_i64,
                in_buf.raw(),
                out_buf.raw(),
                numel_u,
            );
        })?;
        Ok(out_buf.read_to_vec())
    }

    /// f32→bool（`u8` 0／1 の生表現で読み出す。`!= 0` での `bool`
    /// 実体化は呼び出し元 `impl CastOps` の責務）。
    pub fn run_f32_to_bool_raw(
        &self,
        ctx: &MetalContext,
        x: &[f32],
    ) -> Result<Vec<u8>, MetalError> {
        let numel = x.len();
        let numel_u = checked_u32_numel_for_dispatch(numel)?;
        let in_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalCastBuffer::<u8>::new_zeroed(ctx, numel)?;
        ctx.dispatch_sync(|encoder| {
            encode_cast_dispatch(
                encoder,
                &self.cast_f32_to_bool,
                in_buf.raw(),
                out_buf.raw(),
                numel_u,
            );
        })?;
        Ok(out_buf.read_to_vec())
    }

    /// i32→f32（最近接偶数丸め。`|v| > 2^24` は非可逆）。
    pub fn run_i32_to_f32(&self, ctx: &MetalContext, x: &[i32]) -> Result<Vec<f32>, MetalError> {
        let numel = x.len();
        let numel_u = checked_u32_numel_for_dispatch(numel)?;
        let in_buf = MetalCastBuffer::<i32>::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::new_zeroed(ctx, numel)?;
        ctx.dispatch_sync(|encoder| {
            encode_cast_dispatch(
                encoder,
                &self.cast_i32_to_f32,
                in_buf.raw(),
                out_buf.raw(),
                numel_u,
            );
        })?;
        Ok(out_buf.read_to_vec())
    }

    /// i64→f32（同上）。
    pub fn run_i64_to_f32(&self, ctx: &MetalContext, x: &[i64]) -> Result<Vec<f32>, MetalError> {
        let numel = x.len();
        let numel_u = checked_u32_numel_for_dispatch(numel)?;
        let in_buf = MetalCastBuffer::<i64>::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::new_zeroed(ctx, numel)?;
        ctx.dispatch_sync(|encoder| {
            encode_cast_dispatch(
                encoder,
                &self.cast_i64_to_f32,
                in_buf.raw(),
                out_buf.raw(),
                numel_u,
            );
        })?;
        Ok(out_buf.read_to_vec())
    }

    /// bool→f32（`true→1.0`・`false→0.0`。呼び出し元 `impl CastOps` が
    /// `bool` を `u8`〈0／1〉へ変換してから渡す）。
    pub fn run_bool_to_f32_raw(
        &self,
        ctx: &MetalContext,
        x: &[u8],
    ) -> Result<Vec<f32>, MetalError> {
        let numel = x.len();
        let numel_u = checked_u32_numel_for_dispatch(numel)?;
        let in_buf = MetalCastBuffer::<u8>::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::new_zeroed(ctx, numel)?;
        ctx.dispatch_sync(|encoder| {
            encode_cast_dispatch(
                encoder,
                &self.cast_bool_to_f32,
                in_buf.raw(),
                out_buf.raw(),
                numel_u,
            );
        })?;
        Ok(out_buf.read_to_vec())
    }
}

/// cast カーネル共通のエンコード（バッファ index 0〜1・スカラー
/// index 2・ディスパッチ。`shaders/cast.metal` の各カーネルの
/// バッファ宣言と一致させる。`gather_scatter.rs::
/// encode_one_hot_dispatch` と同型）。
fn encode_cast_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    in_buf: &crate::buffer::MtlBuffer,
    out_buf: &crate::buffer::MtlBuffer,
    numel: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: `gather_scatter.rs::encode_gather_dispatch` と同一の根拠
    // （該当コメント参照）。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない。
    // `in_buf`／`out_buf` は呼び出し元 `ctx.dispatch_sync` が完了する
    // まで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
    }

    // SAFETY: `setBytes_length_atIndex` は指定ポインタから指定バイト数
    // を即座に複製する。`numel` はこの呼び出し中生存し、型・バイト数は
    // `shaders/cast.metal` の `constant uint&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&numel).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
    }

    let threads_per_tg = MTLSize {
        width: CAST_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (numel as usize).div_ceil(CAST_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

/// `x` を稠密化し `f32`（`ops::checked_shape_numel` に基づく共通前処理。
/// `crates/backend-cuda/src/cast.rs::checked_contiguous` と同じ設計。
/// 空テンソルの場合は `Ok(None)` を返す）。
///
/// 加えて `ops::checked_bytes_for::<T>` で要素型 `T` 換算のバイト
/// サイズが `Vec` の allocation 上限（`isize::MAX` バイト）に収まるか
/// も `.contiguous()` 呼び出し直前に検査する（codex-review P1 指摘・
/// Cursor Bugbot 同一箇所指摘・イシュー #1751。`crates/backend-cuda/
/// src/cast.rs::checked_contiguous` の是正と同じ理由・同じ判断）。
fn checked_contiguous<T: fandhe_ai_tensor_core::Element>(
    x: &Tensor<T>,
) -> Result<Option<Tensor<T>>, BackendError> {
    checked_shape_numel(x.shape()).map_err(BackendError::ShapeMismatch)?;
    if x.shape().iter().product::<usize>() == 0 {
        return Ok(None);
    }
    crate::ops::checked_bytes_for::<T>(x.shape()).map_err(BackendError::ShapeMismatch)?;
    Ok(Some(x.contiguous()))
}

/// `numel` がカーネル引数 `constant uint&` へ収まるかを検証する
/// （`crates/backend-cuda/src/cast.rs::checked_i32_numel` の `u32` 版。
/// `CudaError` 相当の専用エラー variant を新設せず直接
/// `BackendError::Unsupported` を返す点も同じ判断）。
fn checked_u32_numel(numel: usize) -> Result<u32, BackendError> {
    u32::try_from(numel).map_err(|_| {
        BackendError::Unsupported(format!(
            "cast: numel={numel} exceeds u32::MAX (Metal kernel argument type); \
             falling back to host cast"
        ))
    })
}

/// `MetalCast::run_*` のエラーを `BackendError` へ変換する
/// （`crates/backend-cuda/src/cast.rs::map_cast_error` と同じ設計。
/// `MetalError::ShapeMismatch` を `ShapeError::ElementCountOverflow`
/// へ、それ以外は `BackendError::KernelLaunchFailed` へ写像する）。
fn map_cast_error(err: MetalError) -> BackendError {
    match err {
        MetalError::ShapeMismatch { .. } => {
            BackendError::ShapeMismatch(fandhe_ai_tensor_core::ShapeError::ElementCountOverflow)
        }
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// `x`（cast 元。要素数はすでに `checked_u32_numel` で検証済み）から
/// `MetalCast` インスタンスを取得して `run_fn` を呼び、結果を
/// `Tensor<Out>` へ包む共通骨格（`crates/backend-cuda/src/
/// cast.rs::dispatch_cast` と同型）。
fn dispatch_cast<In, Out>(
    x: &Tensor<In>,
    run_fn: impl FnOnce(&MetalCast, &MetalContext, &[In]) -> Result<Vec<Out>, MetalError>,
) -> Result<Tensor<Out>, BackendError>
where
    In: fandhe_ai_tensor_core::Element,
    Out: fandhe_ai_tensor_core::Element,
{
    let shape = x.shape().to_vec();
    let Some(x_owned) = checked_contiguous(x)? else {
        return Tensor::new(Vec::new(), &shape).map_err(BackendError::ShapeMismatch);
    };
    let x_slice = x_owned
        .as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed("cast: input not contiguous".into()))?;
    checked_u32_numel(x_slice.len())?;

    let (cast, ctx) = cached_cast_and_context()?;
    let out = run_fn(&cast, &ctx, x_slice).map_err(map_cast_error)?;
    Tensor::new(out, &shape).map_err(BackendError::ShapeMismatch)
}

/// `MetalCast`・`MetalContext` のプロセス内キャッシュ済みインスタンス
/// を取得する（`dispatch_cast` と [`CastOps::cast_f32_to_bool`]／
/// [`CastOps::cast_bool_to_f32`]〈`u8` 中継のため `dispatch_cast` の
/// generic 境界〈`Element`〉に載せられない特別扱い〉が共有するヘルパー）。
fn cached_cast_and_context()
-> Result<(std::sync::Arc<MetalCast>, std::sync::Arc<MetalContext>), BackendError> {
    let ctx = context_cache::cached_context().map_err(map_cast_error)?;
    let cast = context_cache::cached_cast(&ctx).map_err(map_cast_error)?;
    Ok((cast, ctx))
}

impl CastOps for MetalBackendOps {
    // f32→f64／f64→f32 は実装しない（MSL `double` 非対応。既定
    // `Unsupported` のままホストフォールバックへ委ねる。モジュール
    // doc「f64 2 方向は実装しない」参照）。

    fn cast_f32_to_i32(&self, x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> {
        dispatch_cast(x, |cast, ctx, s| cast.run_f32_to_i32(ctx, s))
    }

    fn cast_f32_to_i64(&self, x: &Tensor<f32>) -> Result<Tensor<i64>, BackendError> {
        dispatch_cast(x, |cast, ctx, s| cast.run_f32_to_i64(ctx, s))
    }

    fn cast_f32_to_bool(&self, x: &Tensor<f32>) -> Result<Tensor<bool>, BackendError> {
        // `u8` は `fandhe_ai_tensor_core::Element` 未実装（sealed な
        // 対象集合が f32／f64／i32／i64／bool の 5 型に限定されている
        // ため）で `dispatch_cast<In, Out>` の generic 境界に載せられ
        // ない。`Tensor<u8>` を経由せず生 `Vec<u8>` のまま扱う（CPU 側
        // `backend-cpu::cast` は `tensor_core::cast::cast_from_f32` へ
        // 委譲するだけなのでこの制約に触れない）。
        let shape = x.shape().to_vec();
        let Some(x_owned) = checked_contiguous(x)? else {
            return Tensor::new(Vec::new(), &shape).map_err(BackendError::ShapeMismatch);
        };
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("cast: input not contiguous".into()))?;
        checked_u32_numel(x_slice.len())?;

        let (cast, ctx) = cached_cast_and_context()?;
        let raw = cast
            .run_f32_to_bool_raw(&ctx, x_slice)
            .map_err(map_cast_error)?;
        let data: Vec<bool> = raw.into_iter().map(|v| v != 0).collect();
        Tensor::new(data, &shape).map_err(BackendError::ShapeMismatch)
    }

    fn cast_i32_to_f32(&self, x: &Tensor<i32>) -> Result<Tensor<f32>, BackendError> {
        dispatch_cast(x, |cast, ctx, s| cast.run_i32_to_f32(ctx, s))
    }

    fn cast_i64_to_f32(&self, x: &Tensor<i64>) -> Result<Tensor<f32>, BackendError> {
        dispatch_cast(x, |cast, ctx, s| cast.run_i64_to_f32(ctx, s))
    }

    fn cast_bool_to_f32(&self, x: &Tensor<bool>) -> Result<Tensor<f32>, BackendError> {
        // `cast_f32_to_bool` と対称の理由で `dispatch_cast` を使わず
        // 生 `Vec<u8>` のまま扱う（`bool` 自体は `Element` だが、中継
        // する `u8` 側が `Element` 未実装のため）。
        let shape = x.shape().to_vec();
        let Some(x_owned) = checked_contiguous(x)? else {
            return Tensor::new(Vec::new(), &shape).map_err(BackendError::ShapeMismatch);
        };
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("cast: input not contiguous".into()))?;
        let raw_data: Vec<u8> = x_slice.iter().map(|&b| u8::from(b)).collect();
        checked_u32_numel(raw_data.len())?;

        let (cast, ctx) = cached_cast_and_context()?;
        let out = cast
            .run_bool_to_f32_raw(&ctx, &raw_data)
            .map_err(map_cast_error)?;
        Tensor::new(out, &shape).map_err(BackendError::ShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// codex-review P1 指摘・Cursor Bugbot 同一箇所指摘の回帰テスト
    /// （イシュー #1751）。`crates/backend-cuda/src/cast.rs::tests::
    /// checked_contiguous_rejects_huge_broadcast_view_input_without_panicking`
    /// と同型。`checked_contiguous` は GPU driver 呼び出しより前に
    /// 完結するため GPU 非依存の通常テストとして実行できる。
    #[test]
    fn checked_contiguous_rejects_huge_broadcast_view_input_without_panicking() {
        let base = Tensor::<f32>::new(vec![0.0f32], &[1usize]).unwrap();
        let huge_len = (isize::MAX as usize) / std::mem::size_of::<f32>() + 10;
        let huge = base.broadcast_to(&[huge_len]).unwrap();
        assert_eq!(huge.shape(), &[huge_len]);

        let err = checked_contiguous(&huge)
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(fandhe_ai_tensor_core::ShapeError::ElementCountOverflow)
        ));
    }
}
