//! 累積和／累積積（`torch.cumsum`／`torch.cumprod` 相当。イシュー
//! #1740・親イシュー #1731）の起動 API（実行時コンパイル・パイプライン
//! 保持・実行）。
//!
//! `gather_scatter.rs::MetalGatherScatter` と同じ構成方針を踏襲する:
//! [`MetalScan::new`] が `shaders/scan.metal`（`cumsum_f32`／
//! `cumprod_f32`）を実行時コンパイルしてパイプラインを保持し、
//! [`MetalScan::run_cumsum_f32`]／[`MetalScan::run_cumprod_f32`] へ
//! ホスト側スライスを渡すだけでバッファ確保・ディスパッチ・readback を
//! 内部で完結できる。`ops.rs::MetalBackendOps::cumsum`／`cumprod` から
//! 呼ばれる。
//!
//! **数値契約**（lane ごとの binary64 ソフトウェアエミュレーション
//! アキュムレータ逐次計算・CPU 参照実装と bit 完全一致）は
//! `shaders/scan.metal` 冒頭コメントおよび `fandhe_ai_tensor_core::
//! BackendOps::cumsum`／`cumprod` doc が正。
//!
//! 呼び出し元 `ops.rs` が `reduce_out_shape` で `dim` を検査してから
//! `.contiguous()` で稠密化した `outer`／`axis_len`／`inner` を渡す
//! 契約のため、本モジュールは `run_gather_f32` を直接呼び出す経路でも
//! 安全なよう、`lanes`／`numel` の `u32` 収容・スライス実長の整合を
//! 独立に検証する（`unique.rs::MetalUnique::run_unique_f32` と同じ
//! 「呼び出し元の検査結果を信頼しない」二重検査方針。既存の
//! [`crate::error::MetalError::InvalidGatherScatterShape`] を再利用する
//! ——`unique.rs` も同じ variant を shape 検証エラーの汎用表現として
//! 再利用しており、本モジュール専用の新 variant を追加しない最小差分
//! 方針。`.claude/rules/security.md` A08）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/scan.metal` のソース。
const SCAN_MSL_SRC: &str = include_str!("shaders/scan.metal");

/// 1 スレッドグループあたりのスレッド数（`gather_scatter.rs::
/// GS_THREADGROUP_WIDTH` と同じ値・同じ判断根拠）。
const SCAN_THREADGROUP_WIDTH: usize = 256;

/// scan（累積和／累積積）2 カーネルのコンパイル済みパイプラインを
/// 保持するハンドル。
pub struct MetalScan {
    cumsum_f32: objc2::rc::Retained<MtlPipeline>,
    cumprod_f32: objc2::rc::Retained<MtlPipeline>,
}

/// [`MetalScan::run_cumsum_f32`]／[`run_cumprod_f32`] の共通本体が
/// カーネルを選択するための内部列挙。
enum ScanKind {
    Sum,
    Prod,
}

impl MetalScan {
    /// `ctx` のデバイス上で 2 カーネルを実行時コンパイルしパイプライン
    /// を構築する（`gather_scatter.rs::MetalGatherScatter::new` と同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(SCAN_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let cumsum_f32 = pipeline::make_pipeline(ctx.device(), &library, "cumsum_f32")?;
        let cumprod_f32 = pipeline::make_pipeline(ctx.device(), &library, "cumprod_f32")?;

        Ok(Self {
            cumsum_f32,
            cumprod_f32,
        })
    }

    /// `torch.cumsum` 相当（`shaders/scan.metal` 冒頭コメント参照）。
    /// `x` は `outer * axis_len * inner` 要素の稠密（contiguous）
    /// スライス。
    pub fn run_cumsum_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, MetalError> {
        self.run_scan(ctx, ScanKind::Sum, x, outer, axis_len, inner)
    }

    /// `torch.cumprod` 相当。[`Self::run_cumsum_f32`] と同じ検査・
    /// ディスパッチ構造だが `cumprod_f32` カーネルを起動する。
    pub fn run_cumprod_f32(
        &self,
        ctx: &MetalContext,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, MetalError> {
        self.run_scan(ctx, ScanKind::Prod, x, outer, axis_len, inner)
    }

    /// [`Self::run_cumsum_f32`]／[`run_cumprod_f32`] の共通本体（カーネル
    /// のみ `kind` で分岐する）。`numel == 0` は `MetalBuffer` が 0
    /// バイト確保を拒否するため、バッファ確保・`dispatch_sync` に入る
    /// 前に早期 return する（`gather_scatter.rs::run_scatter_f32` の
    /// 空出力早期リターンと同じ理由）。
    ///
    /// **encode-only dispatch は新設していない**: 呼び出し元
    /// `ops.rs::MetalBackendOps::run_scan` は本メソッドの戻り値
    /// （readback 済みホスト `Vec<f32>`）を同期的に消費するため、
    /// `ctx.dispatch_sync`（即座に GPU 完了を待つ）を使う。`*_tracked`
    /// 版・`DispatchFailureCell` 登録が必要になるのは `ctx.encode` を
    /// 使う encode-only 経路（呼び出し元へ制御を返す前に GPU 完了を
    /// 待たない設計）のみであり、本モジュールはその契約の対象外。
    fn run_scan(
        &self,
        ctx: &MetalContext,
        kind: ScanKind,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let lanes =
            outer
                .checked_mul(inner)
                .ok_or_else(|| MetalError::InvalidGatherScatterShape {
                    detail: "run_scan: outer * inner overflowed usize".to_string(),
                })?;
        let numel =
            lanes
                .checked_mul(axis_len)
                .ok_or_else(|| MetalError::InvalidGatherScatterShape {
                    detail: "run_scan: lanes * axis_len overflowed usize".to_string(),
                })?;
        if x.len() != numel {
            return Err(MetalError::InvalidGatherScatterShape {
                detail: format!("run_scan: x.len()={} does not match numel={numel}", x.len()),
            });
        }
        if numel == 0 {
            return Ok(Vec::new());
        }
        let lanes_u = u32::try_from(lanes).map_err(|_| MetalError::InvalidGatherScatterShape {
            detail: format!("run_scan: lanes={lanes} exceeds u32 range (kernel argument type)"),
        })?;
        let axis_len_u =
            u32::try_from(axis_len).map_err(|_| MetalError::InvalidGatherScatterShape {
                detail: format!(
                    "run_scan: axis_len={axis_len} exceeds u32 range (kernel argument type)"
                ),
            })?;
        let inner_u = u32::try_from(inner).map_err(|_| MetalError::InvalidGatherScatterShape {
            detail: format!("run_scan: inner={inner} exceeds u32 range (kernel argument type)"),
        })?;

        let pipeline = match kind {
            ScanKind::Sum => &self.cumsum_f32,
            ScanKind::Prod => &self.cumprod_f32,
        };

        let x_buf = MetalBuffer::new_with_data(ctx, x)?;
        let out_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encode_scan_dispatch(
                encoder, pipeline, &x_buf, &out_buf, lanes_u, axis_len_u, inner_u,
            );
        })?;

        Ok(out_buf.read_to_vec())
    }
}

/// scan カーネル（`cumsum_f32`／`cumprod_f32` 共通）のエンコード
/// （バッファ index 0〜1・スカラー index 2〜4・ディスパッチ）。
/// `shaders/scan.metal::cumsum_f32`／`cumprod_f32` のバッファ宣言と
/// 一致させる。
fn encode_scan_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    x_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    lanes: u32,
    axis_len: u32,
    inner: u32,
) {
    encoder.setComputePipelineState(pipeline);

    // SAFETY: FFI 境界 1/2（`gather_scatter.rs::encode_gather_dispatch`
    // と同じ契約）。`setBuffer_offset_atIndex` は生存中の `MTLBuffer`
    // への参照を保持するのみで即座に読み書きしない。各バッファは
    // 呼び出し元 `ctx.dispatch_sync` が完了するまで生存する。
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x_buf.raw()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
    }

    // SAFETY: FFI 境界 2/2。`setBytes_length_atIndex` は指定ポインタ
    // から指定バイト数を即座に複製する。各ローカル変数は本呼び出し中
    // 生存し、型・バイト数は `shaders/scan.metal::cumsum_f32`／
    // `cumprod_f32` の `constant uint&` 宣言と一致させている。
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&lanes).cast(),
            std::mem::size_of::<u32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&axis_len).cast(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&inner).cast(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    let threads_per_tg = MTLSize {
        width: SCAN_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = (lanes as usize).div_ceil(SCAN_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_msl_source_declares_both_kernels() {
        assert!(SCAN_MSL_SRC.contains("kernel void cumsum_f32("));
        assert!(SCAN_MSL_SRC.contains("kernel void cumprod_f32("));
    }
}
