//! 線形代数 4 カーネル（`shaders/linalg.metal`）の MSL ソース保持・パイプライン構築・
//! 起動（イシュー #2676 Stage A、#3137 Stage B・設計記録
//! `docs/backend-gpu-linalg-decision.md`）。
//!
//! # 役割と起動方式（Stage B。イシュー #3137）
//!
//! [`MetalLinalg::new`] は `shaders/linalg.metal` を実行時コンパイルし、
//! `linalg_lu_f32`／`linalg_lu_solve_f32`／`linalg_cholesky_f32`／`linalg_qr_f32`
//! のパイプラインを保持する。`run_*_f32` はバッファ確保・`ctx.dispatch_sync` による
//! 起動・readback を完結させ、`ops.rs` の内部 opt-in ゲート（`crate::linalg::
//! gpu_linalg_enabled`）の背後から呼ばれる。`setBuffer`／`setBytes` の `unsafe` は
//! `crate::encoder_args`（監査記録 `docs/gpu-linalg-kernel-launch-unsafe-audit.md`
//! §3 案 A）に集約済みで、本モジュールは `unsafe` を含まない。
//!
//! # 不変条件（監査 §6 の申し送り）
//!
//! - `dispatch_sync` に固定する。バッファは関数スコープで生存し、`synchronize` 完了後に
//!   読み戻すため MT-1 の生存契約を満たす。
//! - `status`／`applied` は 0 初期化バッファ（`new_zeroed_i32`）で起動する（カーネルは
//!   0 初期化を前提に 1 を立てるのみ）。
//! - 因数分解 3 カーネルは単一 threadgroup（`(1,1,1)`）で起動する。幅は
//!   `maxTotalThreadsPerThreadgroup` で clamp する（`crate::linalg::clamp_tg_width`）。
//! - solve は `k == 0` では encode しない（grid 0 起動・0 長確保の回避。監査 §6-2）。
//! - uint ストライドのラップは `crate::linalg::validate_launch_stride_headroom` で
//!   拒否する（監査 §6-3）。
//!
//! `[[buffer(i)]]` 番号は MSL 側宣言と一致させる（`tests/linalg_source_evidence.rs`
//! が文字列で固定）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBarrierScope, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::encoder_args::{set_buffer_arg, set_u32_args};
use crate::error::MetalError;
use crate::index_buffer::MetalIndexBuffer;
use crate::linalg::{clamp_tg_width, validate_launch_stride_headroom};
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

/// [`MetalLinalg::run_lu_solve_f32`] の生出力。
pub struct LuSolveRaw {
    /// `[特異フラグ, 行入替回数]`（`crate::linalg::decode_lu_status` で解釈する）。
    pub status: Vec<i32>,
    /// 解（`n*k`）。特異時はカーネルが書かず、`k == 0` では未起動のため `None`。
    pub x: Option<Vec<f32>>,
}

/// `(因数分解結果, status)` を返す関数の戻り値型（`clippy::type_complexity` 回避）。
pub type FactorWithStatus = (Vec<f32>, Vec<i32>);

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

    /// LU（＋代入）起動（`solve`／`inv`）。`a: n*n`・`b: n*k`（行優先）。特異なら `x` は
    /// `None`（カーネルは `x` を書かない）。`k == 0` は LU のみ起動する。
    pub fn run_lu_solve_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        n: usize,
        b: &[f32],
        k: usize,
    ) -> Result<LuSolveRaw, MetalError> {
        // pub API を直接呼ぶ経路でも要素積の u32 上限（MSL の uint 添字ラップ）を確保前に拒否する。
        check_lu_dims(n, k)?;
        let nn = checked_len("lu_solve", "n*n", n, n)?;
        let nk = checked_len("lu_solve", "n*k", n, k)?;
        require_len("lu_solve", "a", a.len(), nn)?;
        require_len("lu_solve", "b", b.len(), nk)?;
        if n == 0 {
            return Err(invalid("lu_solve: n == 0 は呼び出し側で短絡する契約"));
        }
        validate_launch_stride_headroom("linalg LU solve", nn.max(k))
            .map_err(|e| invalid(&e.to_string()))?;
        let n_u = to_u32("lu_solve", "n", n)?;
        let k_u = to_u32("lu_solve", "k", k)?;

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let perm_buf = MetalIndexBuffer::new_zeroed_i32(ctx, n)?;
        let status_buf = MetalIndexBuffer::new_zeroed_i32(ctx, 2)?;
        let solve_bufs = if k > 0 {
            Some((
                MetalBuffer::new_with_data(ctx, b)?,
                MetalBuffer::new_zeroed(ctx, nk)?,
            ))
        } else {
            None
        };
        let lu_width = clamp_tg_width(self.lu.maxTotalThreadsPerThreadgroup());
        let solve_width = clamp_tg_width(self.lu_solve.maxTotalThreadsPerThreadgroup());

        ctx.dispatch_sync(|encoder| {
            self.encode_lu(encoder, &a_buf, &perm_buf, &status_buf, n_u, lu_width);
            if let Some((b_buf, x_buf)) = &solve_bufs {
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                encoder.setComputePipelineState(&self.lu_solve);
                set_buffer_arg(encoder, a_buf.raw(), 0);
                set_buffer_arg(encoder, perm_buf.raw(), 1);
                set_buffer_arg(encoder, status_buf.raw(), 2);
                set_buffer_arg(encoder, b_buf.raw(), 3);
                set_buffer_arg(encoder, x_buf.raw(), 4);
                set_u32_args(encoder, &[n_u, k_u], 5);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    size(k.div_ceil(solve_width)),
                    size(solve_width),
                );
            }
        })?;

        Ok(LuSolveRaw {
            status: status_buf.read_to_vec_i32(),
            x: solve_bufs.map(|(_, x_buf)| x_buf.read_to_vec()),
        })
    }

    /// LU のみ起動（`det`）。戻り値は `(lu: n*n, status: [特異, 入替回数])`。
    pub fn run_lu_det_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        n: usize,
    ) -> Result<FactorWithStatus, MetalError> {
        check_lu_dims(n, 0)?;
        let nn = checked_len("lu_det", "n*n", n, n)?;
        require_len("lu_det", "a", a.len(), nn)?;
        if n == 0 {
            return Err(invalid("lu_det: n == 0 は呼び出し側で短絡する契約"));
        }
        validate_launch_stride_headroom("linalg LU det", nn)
            .map_err(|e| invalid(&e.to_string()))?;
        let n_u = to_u32("lu_det", "n", n)?;

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let perm_buf = MetalIndexBuffer::new_zeroed_i32(ctx, n)?;
        let status_buf = MetalIndexBuffer::new_zeroed_i32(ctx, 2)?;
        let width = clamp_tg_width(self.lu.maxTotalThreadsPerThreadgroup());
        ctx.dispatch_sync(|encoder| {
            self.encode_lu(encoder, &a_buf, &perm_buf, &status_buf, n_u, width);
        })?;
        Ok((a_buf.read_to_vec(), status_buf.read_to_vec_i32()))
    }

    /// cholesky 起動。戻り値は `(l: n*n, status: [非正定値フラグ])`。
    pub fn run_cholesky_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        n: usize,
    ) -> Result<FactorWithStatus, MetalError> {
        check_lu_dims(n, 0)?;
        let nn = checked_len("cholesky", "n*n", n, n)?;
        require_len("cholesky", "a", a.len(), nn)?;
        if n == 0 {
            return Err(invalid("cholesky: n == 0 は呼び出し側で短絡する契約"));
        }
        validate_launch_stride_headroom("linalg cholesky", nn)
            .map_err(|e| invalid(&e.to_string()))?;
        let n_u = to_u32("cholesky", "n", n)?;

        let a_buf = MetalBuffer::new_with_data(ctx, a)?;
        let l_buf = MetalBuffer::new_zeroed(ctx, nn)?;
        let status_buf = MetalIndexBuffer::new_zeroed_i32(ctx, 1)?;
        let width = clamp_tg_width(self.cholesky.maxTotalThreadsPerThreadgroup());
        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.cholesky);
            set_buffer_arg(encoder, a_buf.raw(), 0);
            set_buffer_arg(encoder, l_buf.raw(), 1);
            set_buffer_arg(encoder, status_buf.raw(), 2);
            set_u32_args(encoder, &[n_u], 3);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(width));
        })?;
        Ok((l_buf.read_to_vec(), status_buf.read_to_vec_i32()))
    }

    /// Householder QR 起動。戻り値は `(r_work: m*n, q_work: m*k)`（後段は
    /// `crate::linalg::qr_finalize`）。
    pub fn run_qr_f32(
        &self,
        ctx: &MetalContext,
        a: &[f32],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
        check_qr_dims(m, n)?;
        let k = m.min(n);
        let mn = checked_len("qr", "m*n", m, n)?;
        let mk = checked_len("qr", "m*k", m, k)?;
        require_len("qr", "a", a.len(), mn)?;
        if m == 0 || n == 0 {
            return Err(invalid("qr: 空入力は呼び出し側で短絡する契約"));
        }
        // QR の uint ストライド上限は m と n の大きい方（監査 §6-3）。
        validate_launch_stride_headroom("linalg QR", m.max(n))
            .map_err(|e| invalid(&e.to_string()))?;
        let m_u = to_u32("qr", "m", m)?;
        let n_u = to_u32("qr", "n", n)?;

        let r_buf = MetalBuffer::new_with_data(ctx, a)?;
        let q_buf = MetalBuffer::new_zeroed(ctx, mk)?;
        let v_buf = MetalBuffer::new_zeroed(ctx, mk)?;
        let applied_buf = MetalIndexBuffer::new_zeroed_i32(ctx, k)?;
        let width = clamp_tg_width(self.qr.maxTotalThreadsPerThreadgroup());
        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.qr);
            set_buffer_arg(encoder, r_buf.raw(), 0);
            set_buffer_arg(encoder, q_buf.raw(), 1);
            set_buffer_arg(encoder, v_buf.raw(), 2);
            set_buffer_arg(encoder, applied_buf.raw(), 3);
            set_u32_args(encoder, &[m_u, n_u], 4);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(width));
        })?;
        Ok((r_buf.read_to_vec(), q_buf.read_to_vec()))
    }

    /// LU カーネルの encode（`lu`・`lu_solve` 両経路で共有）。buffer(0..3) は
    /// a／perm／status／n（`linalg_source_evidence.rs` が固定）。
    fn encode_lu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a_buf: &MetalBuffer,
        perm_buf: &MetalIndexBuffer,
        status_buf: &MetalIndexBuffer,
        n_u: u32,
        width: usize,
    ) {
        encoder.setComputePipelineState(&self.lu);
        set_buffer_arg(encoder, a_buf.raw(), 0);
        set_buffer_arg(encoder, perm_buf.raw(), 1);
        set_buffer_arg(encoder, status_buf.raw(), 2);
        set_u32_args(encoder, &[n_u], 3);
        // 単一 threadgroup（MSL は `tg_pos != 0` で何もしない）。
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(width));
    }
}

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

/// LU 系・cholesky の次元検査（`validate_lu_dims`）を `MetalError` へ写す。デバイス非接触。
fn check_lu_dims(n: usize, k: usize) -> Result<(), MetalError> {
    crate::linalg::validate_lu_dims(n, k).map_err(|e| invalid(&e.to_string()))
}

/// QR の次元検査（`validate_qr_dims`）を `MetalError` へ写す。デバイス非接触。
fn check_qr_dims(m: usize, n: usize) -> Result<(), MetalError> {
    crate::linalg::validate_qr_dims(m, n).map_err(|e| invalid(&e.to_string()))
}

fn invalid(detail: &str) -> MetalError {
    MetalError::InvalidLinalgShape {
        detail: detail.to_string(),
    }
}

fn checked_len(op: &str, name: &str, a: usize, b: usize) -> Result<usize, MetalError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(&format!("{op}: {name} が usize を超える")))
}

fn require_len(op: &str, name: &str, actual: usize, expected: usize) -> Result<(), MetalError> {
    if actual != expected {
        return Err(invalid(&format!(
            "{op}: {name} の長さ {actual} が期待値 {expected} と一致しない"
        )));
    }
    Ok(())
}

/// スカラー引数を切り詰めずに `u32` へ変換する（失敗は型付きエラー）。
fn to_u32(op: &str, name: &str, v: usize) -> Result<u32, MetalError> {
    u32::try_from(v).map_err(|_| invalid(&format!("{op}: {name}={v} が u32 を超える")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_dim_checks_reject_uint_overflow_before_allocation() {
        // m=n=65536: m*n = 2^32 > u32::MAX（従来の usize/u32 変換検査だけでは通っていた）。
        assert!(check_qr_dims(65_536, 65_536).is_err());
        assert!(check_lu_dims(65_536, 1).is_err());
        assert!(check_lu_dims(8, 65_536 * 65_536).is_err());
        assert!(check_qr_dims(usize::MAX, 2).is_err());
        assert!(check_qr_dims(8, 3).is_ok());
        assert!(check_lu_dims(8, 3).is_ok());
    }
}
