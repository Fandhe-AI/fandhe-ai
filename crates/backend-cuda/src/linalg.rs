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
//! - 分解系 3 演算（`linalg_cholesky`／`linalg_qr`／`linalg_eigh`。イシュー
//!   #2675）は同じゲートの背後にあり、[`validate_square_factor_dims`]・
//!   [`validate_qr_dims`]（サイズ検証）、[`normalize_max_abs_sign`]・
//!   [`eigh_sort_and_normalize`]・[`qr_finalize`]（ホスト後処理。
//!   `backend-cpu/src/linalg.rs` の同名処理と同一演算列の意図的複製）、
//!   [`CudaLinalgFactor::new`]（`kernels_linalg_factor.rs` の 3 カーネルの
//!   NVRTC コンパイル・ロード）を提供する。
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
use crate::kernels_linalg_factor::{
    LINALG_CHOLESKY_F64, LINALG_EIGH_F64, LINALG_FACTOR_BLOCK_DIM, LINALG_QR_F64,
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

/// 分解系 3 カーネルの block 次元（grid は `(1,1,1)`）。
pub const FACTOR_BLOCK_DIM: u32 = LINALG_FACTOR_BLOCK_DIM;

/// 要素数 `numel` が `i32::MAX` 以内かつ `f64` バイト数が `isize::MAX` 以内に
/// 収まることを検証する（`validate_lu_dims` と同じ規律）。
fn check_numel(op: &str, name: &str, numel: usize) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidElementwiseShape { detail };
    if numel > i32::MAX as usize {
        return Err(invalid(format!(
            "{op}: {name}={numel} が i32::MAX を超える（カーネルの添字型）"
        )));
    }
    let bytes = numel
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| invalid(format!("{op}: {name} のバイト数が usize を超える")))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(format!(
            "{op}: {name} のバイト数が確保上限（isize::MAX）を超える"
        )));
    }
    Ok(())
}

/// 正方行列 `[n,n]` を扱う分解（cholesky・eigh）のサイズ検証。デバイス初期化・
/// 確保より前に呼ぶ（OWASP A03）。`n`・`n*n` が `i32` に収まり、`n*n` 要素の
/// `f64` バッファ（eigh は作業行列・固有ベクトルの 2 本）が確保上限以内。
pub fn validate_square_factor_dims(op: &str, n: usize) -> Result<(), CudaError> {
    if n > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("{op}: n は i32 に収まる必要がある（カーネル引数型）: n={n}"),
        });
    }
    let numel = n
        .checked_mul(n)
        .ok_or_else(|| CudaError::InvalidElementwiseShape {
            detail: format!("{op}: n*n が usize を超える"),
        })?;
    check_numel(op, "n*n", numel)
}

/// reduced QR（`A: [m,n]`、`k = min(m,n)`）のサイズ検証。作業行列 `m*n`・
/// `Q` の `m*k`・`R` 出力の `k*n`・反射ベクトル格納の `k*m` の各バッファが
/// `i32`／確保上限に収まること、`m`・`n` 自体が `i32` に収まることを確認する。
pub fn validate_qr_dims(m: usize, n: usize) -> Result<(), CudaError> {
    let limit = i32::MAX as usize;
    if m > limit || n > limit {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("linalg QR: m・n は i32 に収まる必要がある: m={m}, n={n}"),
        });
    }
    let k = m.min(n);
    for (name, a, b) in [("m*n", m, n), ("m*k", m, k), ("k*n", k, n), ("k*m", k, m)] {
        let numel = a
            .checked_mul(b)
            .ok_or_else(|| CudaError::InvalidElementwiseShape {
                detail: format!("linalg QR: {name} が usize を超える"),
            })?;
        check_numel("linalg QR", name, numel)?;
    }
    Ok(())
}

/// ベクトルの符号を「最大絶対値成分（厳密 `>`・同値は最小添字）が正」に
/// 正規化する（`backend-cpu/src/linalg.rs::normalize_max_abs_sign` と同一。
/// eigh の固有ベクトル列に適用する。設計記録 §3 のホスト後処理の複製）。
pub fn normalize_max_abs_sign(v: &mut [f64]) {
    let mut max_abs = 0.0;
    let mut max_idx = 0usize;
    for (idx, &x) in v.iter().enumerate() {
        if x.abs() > max_abs {
            max_abs = x.abs();
            max_idx = idx;
        }
    }
    if v.get(max_idx).is_some_and(|&x| x < 0.0) {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// eigh カーネルの未ソート出力（`evals`: `n`、`vecs_row_major`: `n*n`・列が
/// 固有ベクトル）を、固有値の昇順（`total_cmp` の安定ソート）に並べ替え、
/// 各列へ [`normalize_max_abs_sign`] を適用して `(固有値, 行優先固有ベクトル)`
/// で返す（`backend-cpu/src/linalg.rs::eigh` の後段と同一）。`f32` 化は呼び出し側。
pub fn eigh_sort_and_normalize(
    evals: &[f64],
    vecs_row_major: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), CudaError> {
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| CudaError::InvalidElementwiseShape {
            detail: "linalg eigh 後処理: n*n が usize を超える".to_string(),
        })?;
    if evals.len() < n || vecs_row_major.len() < nn {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "linalg eigh 後処理: 入力長が不足（evals={} < n={n} または vecs={} < n*n={nn}）",
                evals.len(),
                vecs_row_major.len()
            ),
        });
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let sorted_evals: Vec<f64> = order.iter().map(|&i| evals[i]).collect();
    let mut out = vec![0.0f64; nn];
    let mut col_buf = vec![0.0f64; n];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        for (row, slot) in col_buf.iter_mut().enumerate() {
            *slot = vecs_row_major[row * n + old_idx];
        }
        normalize_max_abs_sign(&mut col_buf);
        for (row, &x) in col_buf.iter().enumerate() {
            out[row * n + new_idx] = x;
        }
    }
    Ok((sorted_evals, out))
}

/// QR カーネルの出力（`r_work`: `m*n` の上三角化済み作業行列、`q_work`: `m*k`）
/// から `R`（`k*n`・下三角を 0）と `Q`（`m*k`）を作り、対角が負の行は `R` の行と
/// `Q` の列を反転して `Q R = A` を保つ（`backend-cpu/src/linalg.rs::qr` の後段と
/// 同一。反転は丸めと可換なので bit 一致に影響しない）。戻り値は `(R, Q)`。
pub fn qr_finalize(
    r_work: &[f64],
    q_work: &[f64],
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), CudaError> {
    let k = m.min(n);
    let overflow = |name: &str| CudaError::InvalidElementwiseShape {
        detail: format!("linalg QR 後処理: {name} が usize を超える"),
    };
    let mn = m.checked_mul(n).ok_or_else(|| overflow("m*n"))?;
    let mk = m.checked_mul(k).ok_or_else(|| overflow("m*k"))?;
    let kn = k.checked_mul(n).ok_or_else(|| overflow("k*n"))?;
    if r_work.len() < mn || q_work.len() < mk {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "linalg QR 後処理: 入力長が不足（r_work={} < m*n={mn} または q_work={} < m*k={mk}）",
                r_work.len(),
                q_work.len()
            ),
        });
    }
    let mut r = vec![0.0f64; kn];
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

/// cholesky・QR・eigh の 3 カーネル（`double`）のコンパイル済みハンドル。
/// LU 側の [`CudaLinalg`] とは別型にして、LU の経路で余分なカーネルを
/// コンパイルしない。
pub struct CudaLinalgFactor {
    stream: Arc<CudaStream>,
    ordinal: usize,
    cholesky: CudaFunction,
    qr: CudaFunction,
    eigh: CudaFunction,
}

impl CudaLinalgFactor {
    /// `device` 上で 3 カーネルを NVRTC コンパイル・ロードする
    /// （[`CudaLinalg::new`] と同一手順。safe API のみ）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let load = |src: &str, name: &str| -> Result<CudaFunction, CudaError> {
            let ptx = compile_ptx(src, arch)?;
            Ok(device.context().load_module(ptx)?.load_function(name)?)
        };
        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            cholesky: load(LINALG_CHOLESKY_F64, "linalg_cholesky_f64")?,
            qr: load(LINALG_QR_F64, "linalg_qr_f64")?,
            eigh: load(LINALG_EIGH_F64, "linalg_eigh_f64")?,
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

    /// Cholesky カーネル（`linalg_cholesky_f64`）。
    pub fn cholesky_function(&self) -> &CudaFunction {
        &self.cholesky
    }

    /// QR カーネル（`linalg_qr_f64`）。
    pub fn qr_function(&self) -> &CudaFunction {
        &self.qr
    }

    /// eigh カーネル（`linalg_eigh_f64`）。
    pub fn eigh_function(&self) -> &CudaFunction {
        &self.eigh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_factor_dims_boundaries() {
        assert!(validate_square_factor_dims("t", 0).is_ok());
        assert!(validate_square_factor_dims("t", 46_340).is_ok());
        assert!(validate_square_factor_dims("t", 46_341).is_err());
        assert!(validate_square_factor_dims("t", usize::MAX).is_err());
    }

    #[test]
    fn qr_dims_boundaries() {
        assert!(validate_qr_dims(0, 0).is_ok());
        assert!(validate_qr_dims(64, 8).is_ok());
        assert!(validate_qr_dims(8, 64).is_ok());
        assert!(validate_qr_dims(i32::MAX as usize, 1).is_ok());
        assert!(validate_qr_dims(i32::MAX as usize, 2).is_err());
        assert!(validate_qr_dims(usize::MAX, 1).is_err());
        assert!(validate_qr_dims(1, usize::MAX).is_err());
        assert!(validate_qr_dims(46_341, 46_341).is_err());
    }

    #[test]
    fn normalize_sign_rules() {
        let mut v = [1.0, -3.0, 2.0];
        normalize_max_abs_sign(&mut v);
        assert_eq!(v, [-1.0, 3.0, -2.0]);
        // 同値は最小添字が採用される（厳密 >）。
        let mut v = [-2.0, 2.0];
        normalize_max_abs_sign(&mut v);
        assert_eq!(v, [2.0, -2.0]);
        let mut z = [0.0, 0.0];
        normalize_max_abs_sign(&mut z);
        assert_eq!(z, [0.0, 0.0]);
        let mut e: [f64; 0] = [];
        normalize_max_abs_sign(&mut e);
    }

    #[test]
    fn eigh_sort_is_ascending_stable_with_total_cmp() {
        // 固有値 [2, 0.0, -0.0]。total_cmp では -0.0 < 0.0。列 j は e_j。
        let evals = [2.0, 0.0, -0.0];
        let vecs = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (e, v) = eigh_sort_and_normalize(&evals, &vecs, 3).unwrap();
        assert!(e[0].is_sign_negative() && e[0] == 0.0);
        assert!(e[1].is_sign_positive() && e[1] == 0.0);
        assert_eq!(e[2], 2.0);
        // 新しい列 0 は旧列 2（e_2）、列 1 は旧列 1、列 2 は旧列 0。
        assert_eq!(v[2 * 3], 1.0);
        assert_eq!(v[3 + 1], 1.0);
        assert_eq!(v[2], 1.0);
        // 同値固有値は元の順序を保つ（安定）。
        let (e2, v2) = eigh_sort_and_normalize(&[1.0, 1.0], &[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(e2, vec![1.0, 1.0]);
        assert_eq!(v2, vec![1.0, 0.0, 0.0, 1.0]);
        // 列の符号は最大絶対値成分が正になる。
        let (_, v3) = eigh_sort_and_normalize(&[1.0], &[-1.0], 1).unwrap();
        assert_eq!(v3, vec![1.0]);
    }

    #[test]
    fn qr_finalize_zeroes_lower_and_flips_negative_diagonal() {
        // m=3, n=2, k=2。R 作業行列 (0,0) が負、(1,0) に残差。
        let r_work = [-2.0, 1.0, 1e-17, 3.0, 0.0, 0.0];
        let q_work = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3x2
        let (r, q) = qr_finalize(&r_work, &q_work, 3, 2).unwrap();
        assert_eq!(r.len(), 4);
        assert_eq!(&r[..2], &[2.0, -1.0]);
        assert_eq!(&r[2..], &[0.0, 3.0]);
        assert_eq!(q[0], -1.0);
        assert_eq!(q[1], 0.0);
        assert_eq!(q[3], 1.0);
    }

    #[test]
    fn post_processing_rejects_short_buffers_and_overflow() {
        assert!(eigh_sort_and_normalize(&[], &[], 1).is_err());
        assert!(eigh_sort_and_normalize(&[1.0], &[], 1).is_err());
        assert!(eigh_sort_and_normalize(&[], &[], usize::MAX).is_err());
        assert!(eigh_sort_and_normalize(&[], &[], 0).is_ok());
        assert!(qr_finalize(&[], &[], 1, 1).is_err());
        assert!(qr_finalize(&[1.0], &[], 1, 1).is_err());
        assert!(qr_finalize(&[], &[], usize::MAX, 2).is_err());
        assert!(qr_finalize(&[], &[], 0, 0).is_ok());
    }

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
