//! CUDA 線形代数（LU 系: `solve`・`inv`・`det`）の内部 opt-in ゲート・
//! 入力検証・ホスト後処理・NVRTC ハンドル（イシュー #2674・親イシュー
//! #2672・設計判断記録 `docs/backend-gpu-linalg-decision.md` §3〜§7）。
//!
//! # 役割と呼び出し文脈
//!
//! - `ops.rs::CudaBackendOps::{linalg_inv, linalg_solve, linalg_det}` が
//!   [`gpu_linalg_enabled`] と検証関数を呼ぶ。ゲートが OFF（既定）なら従来
//!   どおり `BackendError::Unsupported` を返し、`Var` 側がホスト `f64` 参照
//!   実装へフォールバックする既存契約を保つ。
//! - [`CudaLinalg::new`] が `kernels_linalg.rs` の 2 カーネルを NVRTC
//!   コンパイル・ロードする（safe API のみ）。
//! - [`lu_diag_product`] は LU 対角積のホスト後処理（`backend-cpu/src/
//!   linalg.rs::lu_diag_product` と同一演算列の意図的複製。クレート間依存を
//!   作らない。設計記録 §3）。
//!
//! # 公開範囲
//!
//! 本モジュールは `backend-cuda` 内 `pub` だが **`facade` へは再公開しない**
//! （`crates/facade/tests/api_surface.rs` の保留ガードで機械固定。公開形の
//! 推奨は設計記録 §7、公開は別 issue でユーザー承認後）。
//!
//! # カーネル起動の承認状況
//!
//! cudarc の `launch` は `unsafe fn` であり、カーネル起動には新規 `unsafe`
//! が必要になる。新規 `unsafe` はユーザー承認事項（イシュー #2674 契約）で
//! あり、承認が確認できていないため本実装は起動を含まない。ゲート ON 時は
//! fail-closed の `Unsupported` を返す（`ops.rs`）。起動仕様は
//! `docs/perf/logs/cuda-linalg-lu-2674/README.md` に承認依頼として記載する。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{CudaFunction, CudaStream};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_linalg::{
    LINALG_LU_BLOCK_DIM, LINALG_LU_F64, LINALG_LU_SOLVE_F64, LINALG_SOLVE_BLOCK_DIM,
};
use crate::nvrtc::compile_ptx;

/// GPU 線形代数 opt-in ゲート本体。既定 `false`（`fused_elementwise.rs` の
/// `GPU_ELEMENTWISE_FUSION_ENABLED` と同形。設定変更は低頻度のため
/// `SeqCst`）。
static GPU_LINALG_ENABLED: AtomicBool = AtomicBool::new(false);

/// CUDA の LU 系線形代数（`linalg_inv`／`linalg_solve`／`linalg_det`）を
/// 有効化する（既定 OFF。プロセスワイド）。`backend-cuda` 内 `pub` だが
/// `facade` への再公開は行わない（モジュール冒頭「公開範囲」参照）。
pub fn set_gpu_linalg_enabled(enabled: bool) {
    GPU_LINALG_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在のゲート状態を返す（既定 `false`）。
pub fn gpu_linalg_enabled() -> bool {
    GPU_LINALG_ENABLED.load(Ordering::SeqCst)
}

/// `n`（行列次元）・右辺列数 `k` がカーネルの `int` 引数と `f64` バッファの
/// 確保上限に収まることを、デバイス初期化・確保より前に検証する
/// （OWASP A03。`backend-cpu::linalg::checked_numel_for` と同じ規律）。
///
/// 検査: `n*n`・`n*k` が `usize` で overflow せず `i32::MAX` 以内、`n`・`k`
/// 自体も `i32::MAX` 以内、`f64` 換算バイト数が `isize::MAX` 以内。
pub fn validate_lu_dims(n: usize, k: usize) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidElementwiseShape { detail };
    let limit = i32::MAX as usize;
    if n > limit || k > limit {
        return Err(invalid(format!(
            "linalg LU: n・k は i32 に収まる必要がある（カーネル引数型）: n={n}, k={k}"
        )));
    }
    for (name, dim_a, dim_b) in [("n*n", n, n), ("n*k", n, k)] {
        let numel = dim_a
            .checked_mul(dim_b)
            .ok_or_else(|| invalid(format!("linalg LU: {name} が usize を超える")))?;
        if numel > limit {
            return Err(invalid(format!(
                "linalg LU: {name}={numel} が i32::MAX を超える（カーネルの添字型）"
            )));
        }
        let bytes = numel
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| invalid(format!("linalg LU: {name} のバイト数が usize を超える")))?;
        if bytes > isize::MAX as usize {
            return Err(invalid(format!(
                "linalg LU: {name} のバイト数が確保上限（isize::MAX）を超える"
            )));
        }
    }
    Ok(())
}

/// `x = m · 2^e`（`0.5 <= |m| < 1`）へ分解する。`0` と非有限は素通し
/// （`e = 0`）。`backend-cpu/src/linalg.rs::frexp` と同一実装
/// （`to_bits`／`from_bits` のみ・`unsafe` なし）。
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exponent_field = (bits >> 52) & 0x7ff;
    if exponent_field == 0 {
        let (m, e) = frexp(x * 2f64.powi(64));
        return (m, e - 64);
    }
    let sign_bit = bits & 0x8000_0000_0000_0000;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;
    let m = f64::from_bits(sign_bit | (1022u64 << 52) | mantissa_bits);
    (m, exponent_field as i32 - 1022)
}

/// `m · 2^e`（[`frexp`] の逆演算）。
///
/// `2^e` を一度に作ると、積が有限でも `2^e` 単体が overflow／underflow する
/// （例: `m=0.9, e=1024`）。ゼロ・非有限の仮数は先に素通しし
/// （`0 · Inf = NaN` を避ける）、指数は ±1000 ずつ段階的に掛ける。
/// 最終合成が `f64` 範囲外のときのみ `Inf`／`0.0` になる。
fn ldexp(m: f64, e: i32) -> f64 {
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    const STEP: i32 = 1000;
    let mut result = m;
    let mut remaining = e;
    while remaining != 0 {
        let step = remaining.clamp(-STEP, STEP);
        result *= 2f64.powi(step);
        remaining -= step;
        if result == 0.0 || !result.is_finite() {
            break;
        }
    }
    result
}

/// LU 対角（`diag`）の総積（置換符号 `sign` 込み）を仮数・指数分離で
/// 計算する。`diag([1e30; 11], [1e-30; 11])` のような大小スケール混在でも
/// 中間積が overflow しない（`backend-cpu/src/linalg.rs::lu_diag_product`
/// と同一演算列。codex-review 指摘 PRRT_kwDOTuUCJc6hxiys の是正を踏襲）。
/// 最終合成が `f64` 範囲外のときのみ `Inf`／`0.0`。
pub fn lu_diag_product(diag: &[f64], sign: f64) -> f64 {
    let mut mantissa = 1.0f64;
    let mut exponent: i64 = 0;
    for &d in diag {
        let (m, e) = frexp(d);
        mantissa *= m;
        exponent += i64::from(e);
        if mantissa != 0.0 && mantissa.is_finite() {
            let (m2, e2) = frexp(mantissa);
            mantissa = m2;
            exponent += i64::from(e2);
        }
    }
    mantissa *= sign;
    let exponent = exponent.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    ldexp(mantissa, exponent)
}

/// 置換回数 `swap_count` の偶奇から置換符号（`+1.0`／`-1.0`）を返す。
pub fn permutation_sign(swap_count: u32) -> f64 {
    if swap_count.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// LU カーネルの block 次元（grid は `(1,1,1)`）。
pub const LU_BLOCK_DIM: u32 = LINALG_LU_BLOCK_DIM;

/// 代入カーネルの block 次元（grid は `ceil(k / 本値)`）。
pub const LU_SOLVE_BLOCK_DIM: u32 = LINALG_SOLVE_BLOCK_DIM;

/// LU 分解・代入の 2 カーネル（`double`）のコンパイル済みハンドル。
pub struct CudaLinalg {
    stream: Arc<CudaStream>,
    ordinal: usize,
    lu: CudaFunction,
    lu_solve: CudaFunction,
}

impl CudaLinalg {
    /// `device` 上で 2 カーネルを NVRTC コンパイル・ロードする
    /// （`mse.rs::CudaMse::new` と同一手順。safe API のみ）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let lu_ptx = compile_ptx(LINALG_LU_F64, arch)?;
        let solve_ptx = compile_ptx(LINALG_LU_SOLVE_F64, arch)?;
        let lu = device
            .context()
            .load_module(lu_ptx)?
            .load_function("linalg_lu_f64")?;
        let lu_solve = device
            .context()
            .load_module(solve_ptx)?
            .load_function("linalg_lu_solve_f64")?;
        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            lu,
            lu_solve,
        })
    }

    /// 構築元デバイスの ordinal。
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// 構築元デバイスのストリーム。
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// LU 分解カーネル（`linalg_lu_f64`）。
    pub fn lu_function(&self) -> &CudaFunction {
        &self.lu
    }

    /// 前進・後退代入カーネル（`linalg_lu_solve_f64`）。
    pub fn lu_solve_function(&self) -> &CudaFunction {
        &self.lu_solve
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ゲートの既定は OFF（本クレート単体テストはゲートを書き換えないため
    /// 並列実行でも競合しない。書き換えは統合テスト側で直列化する）。
    #[test]
    fn gate_defaults_to_off() {
        assert!(!gpu_linalg_enabled());
    }

    #[test]
    fn validate_dims_accepts_small_and_rejects_oversize() {
        assert!(validate_lu_dims(0, 0).is_ok());
        assert!(validate_lu_dims(64, 64).is_ok());
        assert!(validate_lu_dims(46_340, 1).is_ok()); // 46340^2 < i32::MAX
        assert!(validate_lu_dims(46_341, 1).is_err()); // n*n > i32::MAX
        assert!(validate_lu_dims(2, i32::MAX as usize).is_err()); // n*k overflow i32
        assert!(validate_lu_dims(usize::MAX, 1).is_err());
        assert!(validate_lu_dims(1, usize::MAX).is_err());
    }

    #[test]
    fn diag_product_survives_mixed_scale_diagonal() {
        let mut diag = vec![1e30f64; 11];
        diag.extend(vec![1e-30f64; 11]);
        let det = lu_diag_product(&diag, 1.0);
        assert!(det.is_finite());
        assert!((det - 1.0).abs() < 1e-9, "det={det}");
        assert!((lu_diag_product(&[2.0, 3.0], -1.0) + 6.0).abs() < 1e-12);
        assert_eq!(lu_diag_product(&[], 1.0), 1.0);
        assert_eq!(lu_diag_product(&[2.0, 0.0], 1.0), 0.0);
    }

    #[test]
    fn diag_product_recomposition_avoids_intermediate_overflow() {
        // 積が有限でも 2^1024 単体は overflow する（PRRT_kwDOTuUCJc6rJBKF）。
        assert_eq!(lu_diag_product(&[1e308], 1.0), 1e308);
        assert_eq!(lu_diag_product(&[-1e308], 1.0), -1e308);
        // 大値の後に 0 があっても 0 * Inf = NaN にならない。
        assert_eq!(lu_diag_product(&[1e308, 1e308, 0.0], 1.0), 0.0);
        assert_eq!(lu_diag_product(&[1e308, 0.0], 1.0), 0.0);
        // 範囲外のときのみ Inf／0.0。
        assert_eq!(lu_diag_product(&[1e308, 10.0], 1.0), f64::INFINITY);
        assert_eq!(lu_diag_product(&[1e308, 10.0], -1.0), f64::NEG_INFINITY);
        assert_eq!(lu_diag_product(&[1e-300, 1e-300], 1.0), 0.0);
        assert!(lu_diag_product(&[f64::NAN, 2.0], 1.0).is_nan());
    }

    #[test]
    fn permutation_sign_follows_swap_parity() {
        assert_eq!(permutation_sign(0), 1.0);
        assert_eq!(permutation_sign(1), -1.0);
        assert_eq!(permutation_sign(4), 1.0);
    }
}
