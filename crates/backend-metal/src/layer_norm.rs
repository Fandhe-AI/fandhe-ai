//! LayerNorm 順伝播カーネルの起動 API（イシュー #1596。`rmsnorm.rs`
//! 〈#604〉の構成方針を踏襲する Metal 対応版）。
//!
//! [`MetalLayerNorm::new`] が `shaders/layer_norm.metal`（単一カーネル
//! `layer_norm_f32`。`rmsnorm.rs` の 1 パス／2 パス 2 エントリと異なり
//! 常に device メモリ再読の「2 パス」構成——冒頭コメント参照）を
//! 実行時コンパイルしてパイプラインを保持し、[`MetalLayerNorm::
//! run_layer_norm_f32`] へホスト側スライスを渡すだけでバッファ確保・
//! persistent grid 導出・ディスパッチ・readback を内部で完結できる
//! （`crate::rmsnorm::MetalRmsNorm` と同じ構成方針）。
//!
//! `ops.rs::MetalBackendOps::layer_norm`（`BackendOps::layer_norm` の
//! 独立エントリ）から直接呼ばれる。既存の `run_fused`（canonical 融合
//! プラン一致経路）に LayerNorm 一致経路は追加しない
//! （`docs/norm-ops-design.md`「LayerNorm は本エントリ経由でのみ到達
//! する」）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};
use crate::row_kernel::{self, RowKernelValidationError};

/// `shaders/layer_norm.metal` のソース（単一カーネル）。
const LAYER_NORM_MSL_SRC: &str = include_str!("shaders/layer_norm.metal");

/// カーネル起動時の threadgroup 幅（32 スレッド = 1 simdgroup 固定。
/// `shaders/layer_norm.metal` 冒頭コメント「1 threadgroup = 1 simdgroup
/// 固定」と一致させる）。
const LAYER_NORM_THREADGROUP_WIDTH: usize = 32;

/// [`RowKernelValidationError`] → [`MetalError`] の変換
/// （`rmsnorm.rs::map_validation_error` と同型。独立に持つ理由も同じ）。
fn map_validation_error(err: RowKernelValidationError) -> MetalError {
    MetalError::InvalidRowKernelShape {
        detail: err.to_string(),
    }
}

/// `bias` 長さの起動前 fail-closed 検証（`row_kernel::
/// validate_row_kernel_launch` は `w_len` のみを検査するため、`b_len`
/// は本ファイル側で追加検査する。OWASP A03・`.claude/rules/
/// security.md`）。
fn validate_bias_len(hidden: usize, b_len: Option<usize>) -> Result<(), MetalError> {
    if let Some(bl) = b_len
        && bl != hidden
    {
        return Err(MetalError::InvalidRowKernelShape {
            detail: format!("layer_norm bias length mismatch: hidden={hidden}, b.len()={bl}"),
        });
    }
    Ok(())
}

/// `(float)hidden` が丸め無しで表現できる上限（`2^24`。`f32` の仮数部
/// 23 bit + 暗黙の先頭 1 bit で表現できる最大の連続整数）。
const LAYER_NORM_MAX_HIDDEN_EXACT_F32: usize = 1 << 24;

/// `hidden` の起動前 fail-closed 検証（codex-review 指摘・PR #1671
/// スレッド 1 件目）: `shaders/layer_norm.metal` は平均計算で `hidden`
/// を `(float)hidden` へ直接変換し厳密除算する（`row_kernel::
/// validate_row_kernel_launch` は `hidden` を `i32::MAX` までしか
/// 検査しないため、この `f32` 表現の exact 境界〈`2^24`〉は本ファイル
/// 側で別途検査する必要がある）。`hidden > 2^24`（例 `16777217`）では
/// `(float)hidden` が最近接偶数丸めにより `16777216` へ丸められ、真の
/// 除数とのずれが `mean_lo` へ残存して出力へ伝播しうるため、この軸長
/// は「対応できない」として起動前に明示的に拒否する（対応する実装
/// 〈整数の正確な値を保持した除算〉は行わない。`.claude/rules/
/// coding-rust.md`「カーネル実装の境界検査」節）。
fn validate_hidden_exact_f32(hidden: usize) -> Result<(), MetalError> {
    if hidden > LAYER_NORM_MAX_HIDDEN_EXACT_F32 {
        return Err(MetalError::InvalidRowKernelShape {
            detail: format!(
                "layer_norm hidden exceeds exact f32 integer range: hidden={hidden} > {LAYER_NORM_MAX_HIDDEN_EXACT_F32} (2^24); \
                 Metal カーネルは平均計算で hidden を (float)hidden へ直接変換するため 2^24 超では丸め誤差が生じる"
            ),
        });
    }
    Ok(())
}

/// 融合 LayerNorm 順伝播カーネルのコンパイル済みパイプラインを保持する
/// ハンドル。
pub struct MetalLayerNorm {
    pipeline: objc2::rc::Retained<MtlPipeline>,
}

impl MetalLayerNorm {
    /// `ctx` のデバイス上で LayerNorm カーネルを実行時コンパイルし
    /// パイプラインを構築する。`threadExecutionWidth` が 32 と一致する
    /// ことも検証する（fail-closed。`MetalRmsNorm::new` と同じ理由）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(LAYER_NORM_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let pipeline = pipeline::make_pipeline(ctx.device(), &library, "layer_norm_f32")?;

        let width = pipeline.threadExecutionWidth();
        if width != LAYER_NORM_THREADGROUP_WIDTH {
            return Err(MetalError::UnexpectedThreadExecutionWidth {
                expected: LAYER_NORM_THREADGROUP_WIDTH,
                actual: width,
            });
        }

        Ok(Self { pipeline })
    }

    /// LayerNorm（`out = (x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
    /// `w`／`b` はそれぞれ `None` の場合は対応する演算をスキップ。分散は
    /// biased ÷N）を実行する。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: `MetalRmsNorm::
    /// run_rmsnorm_f32_raw` と同じ理由（各引数がカーネル起動に必須の
    /// 独立パラメータであり構造体集約が可読性を下げる）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_norm_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
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
        validate_bias_len(hidden, b.map(|s| s.len()))?;
        validate_hidden_exact_f32(hidden)?;

        if rows == 0 || hidden == 0 {
            return Ok(Vec::new());
        }

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        // `w`／`b` が `None` の場合もカーネル引数としてバッファは必要
        // （`rmsnorm.rs::run_rmsnorm_f32_raw` の `w_buf` と同じ理由:
        // コンパイラが `has_weight != 0 ? w[idx] : 1.0f` を select へ
        // 最適化し `w[idx]` を無条件ロードしうるため、`hidden` 要素の
        // ゼロ初期化バッファを渡して範囲外読み出しを避ける。REQ-8）。
        let (w_buf, has_weight) = match w {
            Some(w_slice) => (MetalBuffer::new_with_data(ctx, w_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, hidden)?, 0i32),
        };
        let (b_buf, has_bias) = match b {
            Some(b_slice) => (MetalBuffer::new_with_data(ctx, b_slice)?, 1i32),
            None => (MetalBuffer::alloc_zeroed_pooled(ctx, hidden)?, 0i32),
        };
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, x.len())?;

        let rows_u = rows as u32;
        let hidden_u = hidden as u32;
        let inv_n = 1.0f32 / hidden as f32;

        // 常に「2 パス」相当（threadgroup memory 不使用）のため、
        // `rmsnorm.rs` の `RowKernelRoute::TwoPass` 分岐と同じく
        // `smem_bytes_per_group = 0` で persistent grid を導出する。
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

        ctx.dispatch_sync(|encoder| {
            encode_layer_norm_dispatch(
                encoder,
                &self.pipeline,
                &x_buf,
                &w_buf,
                &b_buf,
                &out_buf,
                rows_u,
                hidden_u,
                eps,
                inv_n,
                has_weight,
                has_bias,
                grid_size,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// パイプライン設定・バッファ結線（index 0〜3）・スカラー引数
/// （index 4〜10）の起動処理（`rmsnorm.rs::encode_rmsnorm_dispatch` と
/// 同型の構成）。
#[allow(clippy::too_many_arguments)]
fn encode_layer_norm_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    w_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    rows: u32,
    hidden: u32,
    eps: f32,
    inv_n: f32,
    has_weight: i32,
    has_bias: i32,
    grid_size: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2。`setBuffer_offset_atIndex` は生存中の
    // `MTLBuffer` への参照を保持するのみで即座に読み書きしない
    // （`rmsnorm.rs::encode_rmsnorm_dispatch` の同種コメント参照）。
    // `x_buf`/`w_buf`/`b_buf`/`out_buf` は呼び出し元 `ctx.dispatch_sync`
    // が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(w_buf.raw()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 3);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタから
    // 指定バイト数を即座に複製する（`rmsnorm.rs` と同じ「即時複製」
    // 契約）。各ローカル変数は本呼び出し中生存しており、型・バイト数は
    // カーネル引数宣言（`shaders/layer_norm.metal` の
    // `constant uint&`/`constant float&`/`constant int&`）と一致させて
    // いる。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&rows).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&hidden).cast(),
            std::mem::size_of::<u32>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&eps).cast(),
            std::mem::size_of::<f32>(),
            6,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inv_n).cast(),
            std::mem::size_of::<f32>(),
            7,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&has_weight).cast(),
            std::mem::size_of::<i32>(),
            8,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&has_bias).cast(),
            std::mem::size_of::<i32>(),
            9,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&grid_size).cast(),
            std::mem::size_of::<u32>(),
            10,
        );
    }

    let threads_per_tg = MTLSize {
        width: LAYER_NORM_THREADGROUP_WIDTH,
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

/// `validate_hidden_exact_f32`（Metal デバイスに触れないホスト側純関数
/// 検証。codex-review 指摘・PR #1671 スレッド 1 件目）の自己検証。
/// `layer_norm` モジュール自体が `cfg(target_os = "macos")` のため本
/// テストも macOS 限定でのみコンパイルされるが、デバイス初期化は
/// 不要なため `#[ignore]` は付けない。
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_hidden_exact_f32_accepts_boundary() {
        assert!(validate_hidden_exact_f32(LAYER_NORM_MAX_HIDDEN_EXACT_F32).is_ok());
        assert!(validate_hidden_exact_f32(1).is_ok());
        assert!(validate_hidden_exact_f32(0).is_ok());
    }

    #[test]
    fn validate_hidden_exact_f32_rejects_above_boundary() {
        let err = validate_hidden_exact_f32(LAYER_NORM_MAX_HIDDEN_EXACT_F32 + 1)
            .expect_err("hidden = 2^24 + 1 must be rejected");
        assert!(matches!(err, MetalError::InvalidRowKernelShape { .. }));
    }
}
