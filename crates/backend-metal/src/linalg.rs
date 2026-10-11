//! Metal 線形代数（LU 系: `solve`・`inv`・`det`、および `cholesky`・`qr`）の
//! 内部 opt-in ゲート・入力検証・ホスト後処理・scale/ssq ノルム（イシュー
//! #2676・親イシュー #2672・設計判断記録 `docs/backend-gpu-linalg-decision.md`
//! §3〜§7）。
//!
//! # 役割と呼び出し文脈
//!
//! - `ops.rs::MetalBackendOps::{linalg_inv, linalg_solve, linalg_det,
//!   linalg_cholesky, linalg_qr}` が [`gpu_linalg_enabled`] と検証関数を
//!   呼ぶ。ゲートが OFF（既定）なら従来どおり `BackendError::Unsupported`
//!   を返し、`Var` 側がホスト `f64` 参照実装へフォールバックする既存契約を
//!   保つ。
//! - 本モジュールは `objc2` 系 FFI に触れないため `cfg(target_os = "macos")`
//!   を付けず、Linux（CI）でも単体テストが回る（`split_k_runtime` と同じ判断）。
//! - [`lu_diag_product`]・[`qr_finalize`] 等は `backend-cpu/src/linalg.rs` の
//!   同名処理と同一演算列の意図的複製（`backend-cpu` は dev-dependency のみ
//!   でクレート間依存を作らない。設計記録 §3）。
//! - [`scale_ssq_norm_f32`] は QR の Householder ノルム用（LAPACK SLASSQ 系。
//!   要素の二乗を直接作らず overflow を避ける。設計記録 §3）で、
//!   `shaders/linalg.metal` の同名関数と演算列を一致させる。
//!
//! # 公開範囲
//!
//! 本モジュールは `backend-metal` 内 `pub` だが **`facade` へは再公開しない**
//! （`crates/facade/tests/api_surface.rs` の保留ガードで機械固定。公開形の
//! 推奨は設計記録 §7、公開は別 issue でユーザー承認後）。
//!
//! # カーネル起動の承認状況
//!
//! objc2-metal の `setBuffer_offset_atIndex`／`setBytes_length_atIndex` は
//! `unsafe fn` であり、カーネル起動には新規 `unsafe` が必要になる。新規
//! `unsafe` はユーザー承認事項であり、承認が確認できていないため本実装は
//! 起動を含まない（MSL ソース・パイプライン構築・ホスト逐語モデルまで。
//! Stage A）。ゲート ON 時は fail-closed の `Unsupported` を返す（`ops.rs`）。
//! 起動仕様と承認依頼は `docs/perf/logs/metal-linalg-lu-chol-qr-2676/README.md`。

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::MetalError;

/// カーネルが使う threadgroup 幅（`shaders/linalg.metal` はストライドに
/// `threads_per_threadgroup` を使うため正しさは幅に依存しない。Stage B では
/// パイプラインの `maxTotalThreadsPerThreadgroup` で clamp する）。
pub const LINALG_TG_WIDTH: usize = 256;

/// GPU 線形代数 opt-in ゲート本体。既定 `false`（`split_k_runtime` と同形。
/// 設定変更は低頻度のため `SeqCst`）。
static GPU_LINALG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Metal の線形代数 5 演算（`linalg_inv`／`solve`／`det`／`cholesky`／`qr`）を
/// 有効化する（既定 OFF。プロセスワイド）。`backend-metal` 内 `pub` だが
/// `facade` への再公開は行わない（モジュール冒頭「公開範囲」参照）。
pub fn set_gpu_linalg_enabled(enabled: bool) {
    GPU_LINALG_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在のゲート状態を返す（既定 `false`）。
pub fn gpu_linalg_enabled() -> bool {
    GPU_LINALG_ENABLED.load(Ordering::SeqCst)
}

/// ゲートを操作するテスト間の直列化ロック（`cfg(test)` 限定。
/// `split_k_runtime::test_support` と同型）。
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    pub(crate) fn gate_lock() -> &'static Mutex<()> {
        static LOCK: Mutex<()> = Mutex::new(());
        &LOCK
    }
}

fn invalid(detail: String) -> MetalError {
    MetalError::InvalidElementwiseShape { detail }
}

/// `a*b` 要素の f32 バッファがカーネルの `uint` 添字と確保上限に収まる
/// ことを検証する共通部。
fn check_numel(op: &str, name: &str, a: usize, b: usize) -> Result<usize, MetalError> {
    let numel = a
        .checked_mul(b)
        .ok_or_else(|| invalid(format!("{op}: {name} が usize を超える")))?;
    if numel > u32::MAX as usize {
        return Err(invalid(format!(
            "{op}: {name}={numel} が u32::MAX を超える（カーネルの添字型）"
        )));
    }
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| invalid(format!("{op}: {name} のバイト数が usize を超える")))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(format!(
            "{op}: {name} のバイト数が確保上限（isize::MAX）を超える"
        )));
    }
    Ok(numel)
}

/// LU 系（`n`: 行列次元・`k`: 右辺列数）のサイズを、デバイス初期化・確保より
/// 前に検証する（OWASP A03。`backend-cpu::linalg::checked_numel_for` と同じ規律）。
pub fn validate_lu_dims(n: usize, k: usize) -> Result<(), MetalError> {
    let limit = u32::MAX as usize;
    if n > limit || k > limit {
        return Err(invalid(format!(
            "linalg LU: n・k は u32 に収まる必要がある（カーネル引数型）: n={n}, k={k}"
        )));
    }
    check_numel("linalg LU", "n*n", n, n)?;
    check_numel("linalg LU", "n*k", n, k)?;
    Ok(())
}

/// 正方行列分解（cholesky）のサイズ検証。
pub fn validate_square_factor_dims(op: &str, n: usize) -> Result<(), MetalError> {
    if n > u32::MAX as usize {
        return Err(invalid(format!("{op}: n={n} が u32 を超える")));
    }
    check_numel(op, "n*n", n, n)?;
    Ok(())
}

/// QR のサイズ検証（`m*n`・反射ベクトル保存領域 `m*k`、`k = min(m,n)`）。
pub fn validate_qr_dims(m: usize, n: usize) -> Result<(), MetalError> {
    if m > u32::MAX as usize || n > u32::MAX as usize {
        return Err(invalid(format!(
            "linalg QR: m・n は u32 に収まる必要がある: m={m}, n={n}"
        )));
    }
    let k = m.min(n);
    check_numel("linalg QR", "m*n", m, n)?;
    check_numel("linalg QR", "m*k", m, k)?;
    Ok(())
}

/// `x = m · 2^e`（`0.5 <= |m| < 1`）へ分解する。`0` と非有限は素通し。
/// `backend-cpu/src/linalg.rs::frexp` と同一実装。
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

/// `m · 2^e`（[`frexp`] の逆演算）。指数は ±1000 ずつ段階的に掛ける。
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

/// LU 対角（`diag`）の総積（置換符号 `sign` 込み）を仮数・指数分離で計算する
/// （`backend-cpu/src/linalg.rs::lu_diag_product` と同一演算列）。
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

/// QR カーネルの出力（`r_work`: `m*n`、`q_work`: `m*k`）から `R`（`k*n`・下三角 0）
/// と `Q`（`m*k`）を作り、対角が負の行は `R` の行と `Q` の列を反転する
/// （`backend-cpu/src/linalg.rs::qr` の後段と同一）。戻り値は `(R, Q)`。
pub fn qr_finalize(
    r_work: &[f32],
    q_work: &[f32],
    m: usize,
    n: usize,
) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
    let k = m.min(n);
    let overflow = |name: &str| invalid(format!("linalg QR 後処理: {name} が usize を超える"));
    let mn = m.checked_mul(n).ok_or_else(|| overflow("m*n"))?;
    let mk = m.checked_mul(k).ok_or_else(|| overflow("m*k"))?;
    let kn = k.checked_mul(n).ok_or_else(|| overflow("k*n"))?;
    if r_work.len() < mn || q_work.len() < mk {
        return Err(invalid(format!(
            "linalg QR 後処理: 入力長が不足（r_work={} < m*n={mn} または q_work={} < m*k={mk}）",
            r_work.len(),
            q_work.len()
        )));
    }
    let mut r = vec![0.0f32; kn];
    for row in 0..k {
        for c in 0..n {
            r[row * n + c] = if c < row { 0.0 } else { r_work[row * n + c] };
        }
    }
    let mut q = q_work[..mk].to_vec();
    for i in 0..k {
        if r[i * n + i] < 0.0 {
            for c in 0..n {
                r[i * n + c] = -r[i * n + c];
            }
            for row in 0..m {
                q[row * k + i] = -q[row * k + i];
            }
        }
    }
    Ok((r, q))
}

/// LAPACK SLASSQ 系の overflow-safe な 2-ノルム（f32）。`scale=0・ssq=1`
/// から添字昇順に処理する。NaN を含めば NaN、NaN なしで inf を含めば
/// `+inf` を明示的に返す。`shaders/linalg.metal::scale_ssq_norm` と演算列が同一。
pub fn scale_ssq_norm_f32(xs: &[f32]) -> f32 {
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;
    let mut has_inf = false;
    for &x in xs {
        if x.is_nan() {
            return f32::NAN;
        }
        if x.is_infinite() {
            has_inf = true;
            continue;
        }
        let a = x.abs();
        if a != 0.0 {
            if scale < a {
                let r = scale / a;
                ssq = 1.0f32 + ssq * (r * r);
                scale = a;
            } else {
                let r = a / scale;
                ssq += r * r;
            }
        }
    }
    if has_inf {
        return f32::INFINITY;
    }
    scale * ssq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_defaults_off_and_toggles() {
        let _g = test_support::gate_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let orig = gpu_linalg_enabled();
        set_gpu_linalg_enabled(false);
        assert!(!gpu_linalg_enabled());
        set_gpu_linalg_enabled(true);
        assert!(gpu_linalg_enabled());
        set_gpu_linalg_enabled(orig);
    }

    #[test]
    fn validate_dims_accept_small_and_reject_overflow() {
        assert!(validate_lu_dims(8, 3).is_ok());
        assert!(validate_lu_dims(65_536, 1).is_err()); // n*n = 2^32 > u32::MAX
        assert!(validate_lu_dims(usize::MAX, 1).is_err());
        assert!(validate_square_factor_dims("x", 100).is_ok());
        assert!(validate_square_factor_dims("x", 1 << 20).is_err());
        assert!(validate_qr_dims(1000, 3).is_ok());
        assert!(validate_qr_dims(usize::MAX, 2).is_err());
    }

    #[test]
    fn lu_diag_product_mixed_scale_does_not_overflow() {
        let mut diag = vec![1e30; 11];
        diag.extend(vec![1e-30; 11]);
        let p = lu_diag_product(&diag, 1.0);
        assert!((p - 1.0).abs() < 1e-9, "{p}");
        assert_eq!(lu_diag_product(&[2.0, 3.0], -1.0), -6.0);
        assert_eq!(permutation_sign(3), -1.0);
        assert_eq!(permutation_sign(4), 1.0);
    }

    #[test]
    fn qr_finalize_flips_negative_diagonal() {
        let r_work = [-2.0f32, 1.0, 0.5, 3.0];
        let q_work = [1.0f32, 0.0, 0.0, 1.0];
        let (r, q) = qr_finalize(&r_work, &q_work, 2, 2).unwrap();
        assert_eq!(r, vec![2.0, -1.0, 0.0, 3.0]);
        assert_eq!(q, vec![-1.0, 0.0, 0.0, 1.0]);
        assert!(qr_finalize(&[0.0], &[0.0], 2, 2).is_err());
    }

    #[test]
    fn scale_ssq_handles_extremes() {
        assert_eq!(scale_ssq_norm_f32(&[3.0, 4.0]), 5.0);
        assert_eq!(scale_ssq_norm_f32(&[]), 0.0);
        let big = scale_ssq_norm_f32(&[2e20, 2e20]);
        assert!(
            big.is_finite() && (big / 2.828_427e20 - 1.0).abs() < 1e-6,
            "{big}"
        );
        let small = scale_ssq_norm_f32(&[1e-30, 1e-30]);
        assert!(
            small > 0.0 && (small / 1.414_213_6e-30 - 1.0).abs() < 1e-5,
            "{small}"
        );
        assert!(scale_ssq_norm_f32(&[1.0, f32::NAN]).is_nan());
        assert_eq!(scale_ssq_norm_f32(&[1.0, f32::INFINITY]), f32::INFINITY);
        assert!(scale_ssq_norm_f32(&[f32::INFINITY, f32::NAN]).is_nan());
    }
}
