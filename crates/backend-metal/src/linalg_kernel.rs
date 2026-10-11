//! 線形代数 4 カーネル（`shaders/linalg.metal`）の MSL ソース保持と
//! パイプライン構築（イシュー #2676・設計記録 `docs/backend-gpu-linalg-decision.md`）。
//!
//! # 役割と承認状況
//!
//! [`MetalLinalg::new`] は `shaders/linalg.metal` を実行時コンパイルし、
//! `linalg_lu_f32`／`linalg_lu_solve_f32`／`linalg_cholesky_f32`／`linalg_qr_f32`
//! のパイプラインを保持する（safe API のみ。`unsafe` を含まない）。`ops.rs` の
//! 内部 opt-in ゲート（`crate::linalg::gpu_linalg_enabled`）の背後から使う想定で、
//! **カーネル起動（`setBuffer`／`setBytes` の新規 `unsafe`）は承認待ち**のため
//! 本モジュールには含めない（`crate::linalg` 冒頭・
//! `docs/perf/logs/metal-linalg-lu-chol-qr-2676/README.md` 参照）。
//!
//! `[[buffer(i)]]` 番号は MSL 側宣言と一致させる（`tests/linalg_source_evidence.rs`
//! が文字列で固定）。

use objc2::rc::Retained;

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// `shaders/linalg.metal` のソース。
pub const LINALG_MSL_SRC: &str = include_str!("shaders/linalg.metal");

/// 4 カーネルのコンパイル済みパイプライン。
pub struct MetalLinalg {
    lu: Retained<MtlPipeline>,
    lu_solve: Retained<MtlPipeline>,
    cholesky: Retained<MtlPipeline>,
    qr: Retained<MtlPipeline>,
}

// getter の呼び出し元（カーネル起動）は新規 unsafe の承認待ちで未実装のため、
// Stage B まで dead_code を許容する（理由つき・限定的）。
#[allow(dead_code)]
impl MetalLinalg {
    /// `ctx` のデバイス上で 4 カーネルを実行時コンパイルしパイプラインを構築する
    /// （`pipeline::compile_source`＋`make_pipeline`。`MetalScan::new` と同型）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let library = pipeline::compile_source(ctx.device(), LINALG_MSL_SRC)?;
        Ok(Self {
            lu: pipeline::make_pipeline(ctx.device(), &library, "linalg_lu_f32")?,
            lu_solve: pipeline::make_pipeline(ctx.device(), &library, "linalg_lu_solve_f32")?,
            cholesky: pipeline::make_pipeline(ctx.device(), &library, "linalg_cholesky_f32")?,
            qr: pipeline::make_pipeline(ctx.device(), &library, "linalg_qr_f32")?,
        })
    }

    /// LU 分解パイプライン。
    pub(crate) fn lu_pipeline(&self) -> &MtlPipeline {
        &self.lu
    }

    /// LU 代入パイプライン。
    pub(crate) fn lu_solve_pipeline(&self) -> &MtlPipeline {
        &self.lu_solve
    }

    /// cholesky パイプライン。
    pub(crate) fn cholesky_pipeline(&self) -> &MtlPipeline {
        &self.cholesky
    }

    /// QR パイプライン。
    pub(crate) fn qr_pipeline(&self) -> &MtlPipeline {
        &self.qr
    }
}
