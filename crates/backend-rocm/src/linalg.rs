//! ROCm 線形代数（`inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`）の内部 opt-in ゲート・入力検証・
//! ホスト後処理・実行器（イシュー #3444・親 #3150・設計判断記録 `docs/backend-gpu-linalg-decision.md`
//! §3〜§7・§10。`backend-cuda::linalg` の移植）。
//!
//! # 役割と呼び出し文脈
//!
//! - ゲート [`gpu_linalg_enabled`]（既定 OFF）は、将来 `BackendOps` へ結線する層（R0 #3431 の
//!   `RocmBackendOps`。本 Issue 時点では未マージ）が `Unsupported`／実行の分岐に使う。OFF のとき
//!   `Var` 側がホスト `f64` 参照実装へフォールバックする既存契約を保つ。CUDA 側ゲートとは別の
//!   `static`。
//! - [`RocmLinalgLu::new`]・[`RocmLinalgFactor::new`] が `kernels_linalg*.rs` のソースを HIPRTC で
//!   コンパイル・ロードし、`run_*_f32` が「検証 → `f32`→`f64` 変換 → H2D → 起動 → 同期 → D2H →
//!   ホスト後処理 → 1 回だけ `as f32`」を行う（設計記録 §4。内部は全経路 `f64`）。
//! - ホスト後処理（`lu_diag_product`・`qr_finalize` 等）は `backend-cpu/src/linalg.rs` の同名処理と
//!   同一演算列の**意図的複製**（`backend-cpu` を本クレートの本体依存にしない。設計記録 §3）。
//!
//! # 公開範囲
//!
//! 本モジュールは crate 内 `pub` だが `facade` は `backend-rocm` を参照しないため再公開されない
//! （保留ガード不要。Device への ROCm 追加は #3157）。
//!
//! # 初期化と fail-closed（A04・A08）
//!
//! `hipMalloc` は 0 初期化しないため、状態・出力を表す全バッファを起動前に `DeviceBuffer::from_slice`
//! で書き込む。カーネルが書く `status`・`refl_valid` は `{0,1}` 以外の sentinel を先に書いて読み戻し、
//! 契約外の値（未起動を含む）は成功扱いにしない。unsafe の監査記録は
//! `docs/backend-rocm-unsafe-audit.md` §3.5（行 ID L1〜L5）。

use crate::error::RocmError;
use crate::hip::{self, HipRuntime};
use crate::hiprtc::{HiprtcRuntime, compile_code_object};
use crate::kernels_linalg::{
    LINALG_LU_ARG_COUNT, LINALG_LU_BLOCK_DIM, LINALG_LU_KERNEL_NAME, LINALG_SOLVE_ARG_COUNT,
    LINALG_SOLVE_BLOCK_DIM, LINALG_SOLVE_KERNEL_NAME, LinalgSource, render_linalg_lu_source,
};
use crate::kernels_linalg_factor::{
    LINALG_CHOLESKY_ARG_COUNT, LINALG_CHOLESKY_KERNEL_NAME, LINALG_EIGH_ARG_COUNT,
    LINALG_EIGH_KERNEL_NAME, LINALG_FACTOR_BLOCK_DIM, LINALG_QR_ARG_COUNT, LINALG_QR_KERNEL_NAME,
    render_linalg_factor_source,
};
use crate::module::{DeviceBuffer, Module};
use crate::target::GfxTarget;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// GPU 線形代数 opt-in ゲート本体。既定 `false`（設定変更は低頻度のため `SeqCst`）。
static GPU_LINALG_ENABLED: AtomicBool = AtomicBool::new(false);

/// ROCm の線形代数 6 演算を有効化する（既定 OFF。プロセスワイド。CUDA 側ゲートとは独立）。
pub fn set_gpu_linalg_enabled(enabled: bool) {
    GPU_LINALG_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在のゲート状態を返す（既定 `false`）。
pub fn gpu_linalg_enabled() -> bool {
    GPU_LINALG_ENABLED.load(Ordering::SeqCst)
}

fn invalid_dims(detail: String) -> RocmError {
    RocmError::InvalidLinalgDims { detail }
}

/// `n`（行列次元）・右辺列数 `k` がカーネルの `int` 引数と `f64` バッファの確保上限に収まることを、
/// デバイス初期化・確保より前に検証する（OWASP A03）。
///
/// 検査: `n*n`・`n*k` が `usize` で overflow せず `i32::MAX` 以内、`n`・`k` 自体も `i32::MAX` 以内、
/// `f64` 換算バイト数が `isize::MAX` 以内。
pub(crate) fn validate_lu_dims(n: usize, k: usize) -> Result<(), RocmError> {
    let limit = i32::MAX as usize;
    if n > limit || k > limit {
        return Err(invalid_dims(format!(
            "linalg LU: n・k は i32 に収まる必要がある（カーネル引数型）: n={n}, k={k}"
        )));
    }
    for (name, a, b) in [("n*n", n, n), ("n*k", n, k)] {
        let numel = a
            .checked_mul(b)
            .ok_or_else(|| invalid_dims(format!("linalg LU: {name} が usize を超える")))?;
        check_numel("linalg LU", name, numel)?;
    }
    Ok(())
}

/// 要素数 `numel` が `i32::MAX` 以内かつ `f64` バイト数が `isize::MAX` 以内に収まることを検証する。
pub(crate) fn check_numel(op: &str, name: &str, numel: usize) -> Result<(), RocmError> {
    if numel > i32::MAX as usize {
        return Err(invalid_dims(format!(
            "{op}: {name}={numel} が i32::MAX を超える（カーネルの添字型）"
        )));
    }
    let bytes = numel
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| invalid_dims(format!("{op}: {name} のバイト数が usize を超える")))?;
    if bytes > isize::MAX as usize {
        return Err(invalid_dims(format!(
            "{op}: {name} のバイト数が確保上限（isize::MAX）を超える"
        )));
    }
    Ok(())
}

/// 正方行列 `[n,n]` を扱う分解（cholesky・eigh）のサイズ検証。デバイス初期化・確保より前に呼ぶ。
pub(crate) fn validate_square_factor_dims(op: &str, n: usize) -> Result<(), RocmError> {
    if n > i32::MAX as usize {
        return Err(invalid_dims(format!(
            "{op}: n は i32 に収まる必要がある（カーネル引数型）: n={n}"
        )));
    }
    let numel = n
        .checked_mul(n)
        .ok_or_else(|| invalid_dims(format!("{op}: n*n が usize を超える")))?;
    check_numel(op, "n*n", numel)
}

/// reduced QR（`A: [m,n]`、`k = min(m,n)`）のサイズ検証。作業行列 `m*n`・`Q` の `m*k`・`R` 出力の
/// `k*n`・反射ベクトル格納の `k*m` が `i32`／確保上限に収まること、`m`・`n` 自体が `i32` に収まること。
pub(crate) fn validate_qr_dims(m: usize, n: usize) -> Result<(), RocmError> {
    let limit = i32::MAX as usize;
    if m > limit || n > limit {
        return Err(invalid_dims(format!(
            "linalg QR: m・n は i32 に収まる必要がある: m={m}, n={n}"
        )));
    }
    let k = m.min(n);
    for (name, a, b) in [("m*n", m, n), ("m*k", m, k), ("k*n", k, n), ("k*m", k, m)] {
        let numel = a
            .checked_mul(b)
            .ok_or_else(|| invalid_dims(format!("linalg QR: {name} が usize を超える")))?;
        check_numel("linalg QR", name, numel)?;
    }
    Ok(())
}

/// 入力スライス長が `a == n*n`・`b == n*k` と一致することを検査する（H2D 長とカーネルの添字範囲の一致）。
pub(crate) fn check_lu_lens(
    a_len: usize,
    b_len: usize,
    n: usize,
    k: usize,
) -> Result<(), RocmError> {
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| invalid_dims("linalg LU: n*n が usize を超える".into()))?;
    let nk = n
        .checked_mul(k)
        .ok_or_else(|| invalid_dims("linalg LU: n*k が usize を超える".into()))?;
    if a_len != nn || b_len != nk {
        return Err(invalid_dims(format!(
            "linalg LU: 入力長が不一致（a={a_len} != n*n={nn} または b={b_len} != n*k={nk}）"
        )));
    }
    Ok(())
}

/// 入力長が期待要素数と一致することを検証する。
pub(crate) fn check_len(op: &str, name: &str, got: usize, want: usize) -> Result<(), RocmError> {
    if got != want {
        return Err(invalid_dims(format!(
            "{op}: {name} の長さ {got} が期待 {want} と一致しない"
        )));
    }
    Ok(())
}

/// `usize` を `i32`（カーネル引数型）へ切り詰めなしで変換する（`as` は使わない）。
pub(crate) fn to_i32(op: &str, name: &str, v: usize) -> Result<i32, RocmError> {
    i32::try_from(v).map_err(|_| {
        invalid_dims(format!(
            "{op}: {name}={v} が i32 に収まらない（カーネル引数型）"
        ))
    })
}

/// H2D 前の f32→f64 変換（1e±20 の二乗が f32 では overflow／underflow するため）。
pub(crate) fn widen(a: &[f32]) -> Vec<f64> {
    a.iter().map(|&x| f64::from(x)).collect()
}

/// D2H・ホスト後処理後の `as f32`（出力ごとに 1 回だけ）。
pub(crate) fn narrow(a: &[f64]) -> Vec<f32> {
    a.iter().map(|&x| x as f32).collect()
}

/// `x = m · 2^e`（`0.5 <= |m| < 1`）へ分解する。`0` と非有限は素通し（`e = 0`）。
/// `backend-cpu/src/linalg.rs::frexp` と同一実装（`to_bits`／`from_bits` のみ・`unsafe` なし）。
pub(crate) fn frexp(x: f64) -> (f64, i32) {
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

/// `m · 2^e`（[`frexp`] の逆演算）。`2^e` を一度に作ると積が有限でも単体が overflow／underflow する
/// ため、ゼロ・非有限の仮数は先に素通しし、指数は ±1000 ずつ段階的に掛ける。
pub(crate) fn ldexp(m: f64, e: i32) -> f64 {
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
/// （`backend-cpu/src/linalg.rs::lu_diag_product` と同一演算列。中間積が overflow しない）。
pub(crate) fn lu_diag_product(diag: &[f64], sign: f64) -> f64 {
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
pub(crate) fn permutation_sign(swap_count: u32) -> f64 {
    if swap_count.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// LU 系（solve／inv）の GPU 実行結果。特異（ピボットが厳密 `0`）はエラーにせず専用 variant で表す
/// （呼び出し側が `BackendError::InvalidArgument` へ写す。設計記録 §4）。
#[derive(Debug, Clone, PartialEq)]
pub enum LuSolveOutcome {
    /// 行優先 `n*k` の解（`f32`。`k == 0` のときは空）。
    Solved(Vec<f32>),
    /// 係数行列が特異。
    Singular,
}

/// `status` をホストが書き込む「未実行」sentinel（カーネルは `{0,1}`・非負の swap 数しか書かないため、
/// 読み戻して本値のままなら起動が走っていない。A08 fail-closed）。
const STATUS_SENTINEL: i32 = -1;

/// LU カーネルの `status`（`[singular_flag, swap_count]`）の解釈結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LuStatus {
    pub(crate) singular: bool,
    pub(crate) swap_count: u32,
}

/// `status` を解釈する。長さ不足・`flag` が `{0,1}` 以外（sentinel のまま・想定外）・負の
/// `swap_count` は型付きエラー（成功扱いにしない。A08）。
pub(crate) fn interpret_lu_status(status: &[i32]) -> Result<LuStatus, RocmError> {
    let bad = |detail: String| RocmError::LinalgKernelStatusInvalid { detail };
    let [flag, swaps, ..] = status else {
        return Err(bad(format!(
            "linalg LU: status の長さが不足（{} < 2）",
            status.len()
        )));
    };
    if *flag != 0 && *flag != 1 {
        return Err(bad(format!(
            "linalg LU: singular_flag={flag} は契約外（sentinel のまま、または想定外）"
        )));
    }
    let swap_count =
        u32::try_from(*swaps).map_err(|_| bad(format!("linalg LU: swap_count が負: {swaps}")))?;
    Ok(LuStatus {
        singular: *flag == 1,
        swap_count,
    })
}

/// LU 済み行列（行優先 `n*n`）と `status` から行列式を組み立てる。特異は `lu` を参照せず `0.0`。
/// それ以外は対角と置換符号を `lu_diag_product` へ渡す。swap 数は `n` 以下（置換回数の上限）。
pub(crate) fn det_from_lu(
    status: &[i32],
    lu_row_major: &[f64],
    n: usize,
) -> Result<f64, RocmError> {
    let st = interpret_lu_status(status)?;
    if st.singular {
        return Ok(0.0);
    }
    if st.swap_count as usize > n {
        return Err(RocmError::LinalgKernelStatusInvalid {
            detail: format!("linalg LU: swap_count={} が n={n} を超える", st.swap_count),
        });
    }
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| invalid_dims("linalg LU 後処理: n*n が usize を超える".into()))?;
    if lu_row_major.len() < nn {
        return Err(invalid_dims(format!(
            "linalg LU 後処理: LU の長さが不足（{} < n*n={nn}）",
            lu_row_major.len()
        )));
    }
    let diag: Vec<f64> = (0..n).map(|i| lu_row_major[i * n + i]).collect();
    Ok(lu_diag_product(&diag, permutation_sign(st.swap_count)))
}

/// solve カーネルの `solve_err` を判定する純関数。`0` のみ成功で、`1`（`perm` の値域違反）・その他は
/// [`RocmError::LinalgKernelStatusInvalid`]（黙って成功扱いにしない。REQ-8・A08）。
pub(crate) fn solve_err_to_result(v: i32) -> Result<(), RocmError> {
    match v {
        0 => Ok(()),
        other => Err(RocmError::LinalgKernelStatusInvalid {
            detail: format!("linalg_lu_solve: perm の値域違反を検出（solve_err={other}）"),
        }),
    }
}

/// ベクトルの符号を「最大絶対値成分（厳密 `>`・同値は最小添字）が正」に正規化する
/// （`backend-cpu/src/linalg.rs::normalize_max_abs_sign` と同一。eigh の固有ベクトル列に適用）。
pub(crate) fn normalize_max_abs_sign(v: &mut [f64]) {
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

/// eigh カーネルの未ソート出力（`evals`: `n`、`vecs_row_major`: `n*n`・列が固有ベクトル）を、固有値の
/// 昇順（`total_cmp` の安定ソート）に並べ替え、各列へ [`normalize_max_abs_sign`] を適用して
/// `(固有値, 行優先固有ベクトル)` で返す（`backend-cpu/src/linalg.rs::eigh` の後段と同一）。
pub(crate) fn eigh_sort_and_normalize(
    evals: &[f64],
    vecs_row_major: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), RocmError> {
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| invalid_dims("linalg eigh 後処理: n*n が usize を超える".into()))?;
    if evals.len() < n || vecs_row_major.len() < nn {
        return Err(invalid_dims(format!(
            "linalg eigh 後処理: 入力長が不足（evals={} < n={n} または vecs={} < n*n={nn}）",
            evals.len(),
            vecs_row_major.len()
        )));
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

/// QR カーネルの出力（`r_work`: `m*n` の上三角化済み作業行列、`q_work`: `m*k`）から `R`（`k*n`・下三角を
/// 0）と `Q`（`m*k`）を作り、対角が負の行は `R` の行と `Q` の列を反転して `Q R = A` を保つ
/// （`backend-cpu/src/linalg.rs::qr` の後段と同一）。戻り値は `(R, Q)`。
pub(crate) fn qr_finalize(
    r_work: &[f64],
    q_work: &[f64],
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), RocmError> {
    let k = m.min(n);
    let overflow = |name: &str| invalid_dims(format!("linalg QR 後処理: {name} が usize を超える"));
    let mn = m.checked_mul(n).ok_or_else(|| overflow("m*n"))?;
    let mk = m.checked_mul(k).ok_or_else(|| overflow("m*k"))?;
    let kn = k.checked_mul(n).ok_or_else(|| overflow("k*n"))?;
    if r_work.len() < mn || q_work.len() < mk {
        return Err(invalid_dims(format!(
            "linalg QR 後処理: 入力長が不足（r_work={} < m*n={mn} または q_work={} < m*k={mk}）",
            r_work.len(),
            q_work.len()
        )));
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

/// 分解系カーネルの演算種別（`status == 1` の意味を分ける）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FactorOp {
    Cholesky,
    Eigh,
}

/// `status[0]` を判定する純関数。`0` は成功、`1` は入力起因の失敗
/// （[`RocmError::LinalgInvalidArgument`]。CPU 参照実装と同じ `InvalidArgument` 分類）、それ以外
/// （sentinel のまま・想定外の値）は [`RocmError::LinalgKernelStatusInvalid`]（成功扱いにしない）。
pub(crate) fn factor_status_to_result(op: FactorOp, status: i32) -> Result<(), RocmError> {
    match status {
        0 => Ok(()),
        1 => Err(RocmError::LinalgInvalidArgument {
            detail: match op {
                FactorOp::Cholesky => {
                    "linalg_cholesky: 行列が対称正定値でない（対角が非有限または非正）".to_string()
                }
                FactorOp::Eigh => {
                    "linalg_eigh: 巡回 Jacobi 法が反復上限（60 スイープ）内に収束しなかった"
                        .to_string()
                }
            },
        }),
        other => Err(RocmError::LinalgKernelStatusInvalid {
            detail: format!("{op:?}: status={other} は契約外（sentinel のまま、または想定外）"),
        }),
    }
}

/// QR の `refl_valid` が全要素 `{0,1}` であることを確認する（sentinel のまま残った要素は起動が走って
/// いない証拠。`qr_finalize` は値を使わないが、読み戻して判定することで fail-closed にする）。
pub(crate) fn check_refl_valid(refl_valid: &[i32], k: usize) -> Result<(), RocmError> {
    if refl_valid.len() != k || refl_valid.iter().any(|&v| v != 0 && v != 1) {
        return Err(RocmError::LinalgKernelStatusInvalid {
            detail: format!(
                "linalg_qr: refl_valid が契約外（len={} / 期待 {k}、全要素 0/1 が必要）",
                refl_valid.len()
            ),
        });
    }
    Ok(())
}

/// grid／block の 3 次元寸法。
type Dim3 = (u32, u32, u32);

/// 代入カーネルの起動形状 `(grid, block)`（grid は `ceil(k / block)`）。`k == 0` は `None`
/// （grid 0 の起動をしない）。`div_ceil` なので `u32` で overflow しない。
pub(crate) fn solve_launch_geometry(k: u32) -> Option<(Dim3, Dim3)> {
    if k == 0 {
        return None;
    }
    Some((
        (k.div_ceil(LINALG_SOLVE_BLOCK_DIM), 1, 1),
        (LINALG_SOLVE_BLOCK_DIM, 1, 1),
    ))
}

/// 1 block 起動（LU・分解系）の形状。
fn single_block(block_dim: u32) -> (Dim3, Dim3) {
    ((1, 1, 1), (block_dim, 1, 1))
}

/// HIP/HIPRTC をロードして `source` を `target` 向けにコンパイルし module を作る。
/// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
fn load_module(
    target: GfxTarget,
    source: &LinalgSource,
) -> Result<(Arc<HipRuntime>, Module, i32), RocmError> {
    let rt = Arc::new(hip::load()?);
    let hiprtc = HiprtcRuntime::load()?;
    if rt.device_count()? < 1 {
        return Err(RocmError::NoDevice);
    }
    let device = 0;
    let module = {
        // `new` 中も呼び出し元の選択を壊さないようガードで復元する。
        let _guard = rt.select_device(device)?;
        let code = compile_code_object(&hiprtc, source, target)?;
        Module::load(Arc::clone(&rt), code, device)?
    };
    Ok((rt, module, device))
}

/// LU 分解・代入の 2 カーネル（`double`）を保持し、`solve`・`inv`・`det` を実行する。
pub struct RocmLinalgLu {
    // Drop 順: module（アンロード）→ rt。module が `Arc` を持つため順序に依らず安全。
    module: Module,
    rt: Arc<HipRuntime>,
    /// module・バッファ・同期の対象デバイス ID（`new` が選択したもの）。
    device: i32,
}

impl RocmLinalgLu {
    /// 2 カーネルを `target` 向けにコンパイル・ロードする。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let (rt, module, device) = load_module(target, &render_linalg_lu_source())?;
        Ok(Self { module, rt, device })
    }

    /// LU 分解カーネルを単一 block で起動する（監査行 ID L1。`a`＝`n*n`・`perm`＝`n`・`status`＝2 要素）。
    fn launch_lu(
        &self,
        a: &DeviceBuffer<f64>,
        perm: &DeviceBuffer<i32>,
        status: &DeviceBuffer<i32>,
        n: i32,
    ) -> Result<(), RocmError> {
        let func = self.module.function(LINALG_LU_KERNEL_NAME)?;
        let (mut pa, mut pp, mut ps) = (
            a.as_device_ptr(),
            perm.as_device_ptr(),
            status.as_device_ptr(),
        );
        let mut n = n;
        let mut params: [*mut c_void; LINALG_LU_ARG_COUNT] = [
            (&mut pa as *mut *mut c_void).cast(),
            (&mut pp as *mut *mut c_void).cast(),
            (&mut ps as *mut *mut c_void).cast(),
            (&mut n as *mut i32).cast(),
        ];
        let (grid, block) = single_block(LINALG_LU_BLOCK_DIM);
        // SAFETY: (U5・L1) `params` の個数・型・順序はカーネル宣言 `(double* a, int* perm,
        // int* status, int n)` と一致し（`LINALG_LU_ARG_COUNT` とソース署名をテストで固定）、指す
        // 変数は下の `synchronize` 完了まで本スコープで生存する。`a` は呼び出し元が `n*n`・`perm` は
        // `n`・`status` は 2 要素で確保・初期化済み（`validate_lu_dims` で `i32::MAX`・確保上限を検証
        // 済み）。grid は `(1,1,1)`・block はソースの `LU_BLOCK` と同値（`LINALG_LU_BLOCK_DIM`）。
        // カーネル内の全ループは `i < n`・`c < n` の手動境界チェック（REQ-8）を維持する。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()
    }

    /// `A X = B`（`a`: `n*n`・`b`: `n*k`、いずれも行優先 `f32`）を GPU で解く。`n == 0` はデバイスに
    /// 触れず `Solved(vec![])`。ピボットが厳密 `0` の特異行列は [`LuSolveOutcome::Singular`]。`n > 0`
    /// かつ `k == 0` は LU だけを起動して特異判定のみ行い、代入は起動しない（CPU の `solve` と同じ判定）。
    pub fn run_lu_solve_f32(
        &self,
        a: &[f32],
        b: &[f32],
        n: usize,
        k: usize,
    ) -> Result<LuSolveOutcome, RocmError> {
        validate_lu_dims(n, k)?;
        check_lu_lens(a.len(), b.len(), n, k)?;
        if n == 0 {
            return Ok(LuSolveOutcome::Solved(Vec::new()));
        }
        let n_i = to_i32("linalg LU", "n", n)?;
        let k_i = to_i32("linalg LU", "k", k)?;
        let k_u = u32::try_from(k)
            .map_err(|_| invalid_dims(format!("linalg LU: k={k} が u32 に収まらない")))?;

        // module の所有デバイスを選び、確保・起動・同期を同一デバイスで行う。ガードを先に宣言する
        // ため、バッファの Drop（hipFree）後に呼び出し元の選択へ復元される。
        let _guard = self.rt.select_device(self.device)?;
        let a_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &widen(a))?;
        let perm_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0i32; n])?;
        let status_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &[STATUS_SENTINEL; 2])?;
        self.launch_lu(&a_dev, &perm_dev, &status_dev, n_i)?;

        let mut status = [0i32; 2];
        status_dev.download(&mut status)?;
        if interpret_lu_status(&status)?.singular {
            return Ok(LuSolveOutcome::Singular);
        }
        let Some((grid, block)) = solve_launch_geometry(k_u) else {
            return Ok(LuSolveOutcome::Solved(Vec::new()));
        };

        let b_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &widen(b))?;
        let x_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; n * k])?;
        // perm 値域違反の通知先（カーネルは 1 を書くのみ。0 を upload）。
        let err_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &[0i32])?;
        let func = self.module.function(LINALG_SOLVE_KERNEL_NAME)?;
        let (mut plu, mut pp, mut ps, mut pb, mut px, mut pe) = (
            a_dev.as_device_ptr(),
            perm_dev.as_device_ptr(),
            status_dev.as_device_ptr(),
            b_dev.as_device_ptr(),
            x_dev.as_device_ptr(),
            err_dev.as_device_ptr(),
        );
        let (mut n_arg, mut k_arg) = (n_i, k_i);
        let mut params: [*mut c_void; LINALG_SOLVE_ARG_COUNT] = [
            (&mut plu as *mut *mut c_void).cast(),
            (&mut pp as *mut *mut c_void).cast(),
            (&mut ps as *mut *mut c_void).cast(),
            (&mut pb as *mut *mut c_void).cast(),
            (&mut px as *mut *mut c_void).cast(),
            (&mut pe as *mut *mut c_void).cast(),
            (&mut n_arg as *mut i32).cast(),
            (&mut k_arg as *mut i32).cast(),
        ];
        // SAFETY: (U5・L2) `params` の個数・型・順序はカーネル宣言 `(const double* lu, const int*
        // perm, const int* status, const double* b, double* x, int* solve_err, int n, int k)` と
        // 一致し（`LINALG_SOLVE_ARG_COUNT` とソース署名をテストで固定）、指す変数は下の
        // `synchronize` 完了まで本スコープで生存する。`a_dev`（LU 済み・`n*n`）・`perm_dev`（`n`）・
        // `status_dev`（2）は L1 と同一デバイス上で同期済み、`b_dev`・`x_dev` は `n*k` 要素
        // （`check_lu_lens`・`validate_lu_dims` で検証済み）、`err_dev` は 1 要素で全て初期化済み。
        // `perm` の値域（`[0, n)`）はカーネルが `b` を読む前に検査し、違反なら `solve_err` へ 1 を
        // 書いて何も書かず return する（REQ-8）。ホストは下で必ず読み戻して型付きエラーにする。
        // grid は `k.div_ceil(block)`、カーネル内の `col >= k` 手動境界チェックで余剰スレッドは
        // 何も書かない。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()?;

        let mut solve_err = [STATUS_SENTINEL];
        err_dev.download(&mut solve_err)?;
        solve_err_to_result(solve_err[0])?;
        let mut x = vec![0.0f64; n * k];
        x_dev.download(&mut x)?;
        Ok(LuSolveOutcome::Solved(narrow(&x)))
    }

    /// `inv(A)`（`A X = I` の `X`）。`B` はホストで作る単位行列で、[`Self::run_lu_solve_f32`] へ委譲する。
    pub fn run_lu_inv_f32(&self, a: &[f32], n: usize) -> Result<LuSolveOutcome, RocmError> {
        validate_lu_dims(n, n)?;
        // 単位行列（n*n）を確保する前に入力長を検証する（空 a・巨大 n での過大確保を防ぐ）。
        check_lu_lens(a.len(), n * n, n, n)?;
        let mut ident = vec![0.0f32; n * n];
        for i in 0..n {
            ident[i * n + i] = 1.0;
        }
        self.run_lu_solve_f32(a, &ident, n, n)
    }

    /// `det(A)`（LU 対角積 × 置換符号。CPU と同じ `lu_diag_product`）。`n == 0` は空積 `1.0`、特異は
    /// `0.0`（エラーにしない。CPU 契約）。
    pub fn run_lu_det_f32(&self, a: &[f32], n: usize) -> Result<f32, RocmError> {
        validate_lu_dims(n, 0)?;
        check_lu_lens(a.len(), 0, n, 0)?;
        if n == 0 {
            return Ok(1.0);
        }
        let n_i = to_i32("linalg LU", "n", n)?;

        let _guard = self.rt.select_device(self.device)?;
        let a_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &widen(a))?;
        let perm_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0i32; n])?;
        let status_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &[STATUS_SENTINEL; 2])?;
        self.launch_lu(&a_dev, &perm_dev, &status_dev, n_i)?;

        let mut status = [0i32; 2];
        status_dev.download(&mut status)?;
        if interpret_lu_status(&status)?.singular {
            // 特異ならカーネルが途中で return し `a` は分解途中のため読まない。
            return Ok(0.0f32);
        }
        let mut lu = vec![0.0f64; n * n];
        a_dev.download(&mut lu)?;
        Ok(det_from_lu(&status, &lu, n)? as f32)
    }
}

/// cholesky・QR・eigh の 3 カーネル（`double`）を保持して実行する。LU 側の [`RocmLinalgLu`] とは別型にして、
/// LU の経路で余分なカーネルをコンパイルしない。
pub struct RocmLinalgFactor {
    module: Module,
    rt: Arc<HipRuntime>,
    device: i32,
}

impl RocmLinalgFactor {
    /// 3 カーネルを `target` 向けにコンパイル・ロードする。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let (rt, module, device) = load_module(target, &render_linalg_factor_source())?;
        Ok(Self { module, rt, device })
    }

    /// Cholesky 分解 `A = L Lᵀ`（`a`: 行優先 `n*n` の `f32`）を GPU で実行し `L`（`n*n`・`f32`）を返す。
    ///
    /// 呼び出し側が `n == 0` を短絡する契約（空入力は型付きエラー）。入口で検証し直す（多層防御）。
    pub fn run_cholesky_f32(&self, a: &[f32], n: usize) -> Result<Vec<f32>, RocmError> {
        const OP: &str = "linalg_cholesky";
        validate_square_factor_dims(OP, n)?;
        if n == 0 {
            return Err(invalid_dims(format!(
                "{OP}: n=0 はカーネルを起動しない（呼び出し側で短絡する）"
            )));
        }
        check_len(OP, "a", a.len(), n * n)?;
        let n_i = to_i32(OP, "n", n)?;

        let _guard = self.rt.select_device(self.device)?;
        let a_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &widen(a))?;
        let l_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; n * n])?;
        let status_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &[STATUS_SENTINEL])?;
        let func = self.module.function(LINALG_CHOLESKY_KERNEL_NAME)?;
        let (mut pa, mut pl, mut ps) = (
            a_dev.as_device_ptr(),
            l_dev.as_device_ptr(),
            status_dev.as_device_ptr(),
        );
        let mut n_arg = n_i;
        let mut params: [*mut c_void; LINALG_CHOLESKY_ARG_COUNT] = [
            (&mut pa as *mut *mut c_void).cast(),
            (&mut pl as *mut *mut c_void).cast(),
            (&mut ps as *mut *mut c_void).cast(),
            (&mut n_arg as *mut i32).cast(),
        ];
        let (grid, block) = single_block(LINALG_FACTOR_BLOCK_DIM);
        // SAFETY: (U5・L3) `params` の個数・型・順序はカーネル宣言 `(const double* a, double* l,
        // int* status, int n)` と一致し（`LINALG_CHOLESKY_ARG_COUNT` とソース署名をテストで固定）、
        // 指す変数は下の `synchronize` 完了まで本スコープで生存する。`a_dev`・`l_dev` は検証済みの
        // `n*n`（`<= i32::MAX`）要素、`status_dev` は 1 要素で、全て初期化済み。block 次元は
        // `LINALG_FACTOR_BLOCK_DIM`（ソースの `CHOL_BLOCK` と同値）、grid は 1 block。カーネル内は
        // `idx < total`・`i < n` の手動境界チェック（REQ-8）を維持し、ストライド変数は `long long`。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()?;

        let mut status = [STATUS_SENTINEL];
        status_dev.download(&mut status)?;
        factor_status_to_result(FactorOp::Cholesky, status[0])?;
        let mut l = vec![0.0f64; n * n];
        l_dev.download(&mut l)?;
        Ok(narrow(&l))
    }

    /// reduced Householder QR（`a`: 行優先 `m*n`）を GPU で実行し `(R: k*n, Q: m*k)`（`k = min(m,n)`・
    /// `f32`）を返す。呼び出し文脈・変換規約は [`Self::run_cholesky_f32`] と同じ。
    pub fn run_qr_f32(
        &self,
        a: &[f32],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), RocmError> {
        const OP: &str = "linalg_qr";
        validate_qr_dims(m, n)?;
        if m == 0 || n == 0 {
            return Err(invalid_dims(format!(
                "{OP}: 空入力はカーネルを起動しない（呼び出し側で短絡する）"
            )));
        }
        let k = m.min(n);
        check_len(OP, "a", a.len(), m * n)?;
        let m_i = to_i32(OP, "m", m)?;
        let n_i = to_i32(OP, "n", n)?;

        let _guard = self.rt.select_device(self.device)?;
        let r_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &widen(a))?;
        let q_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; m * k])?;
        let refl_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; k * m])?;
        let valid_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![STATUS_SENTINEL; k])?;
        let func = self.module.function(LINALG_QR_KERNEL_NAME)?;
        let (mut pr, mut pq, mut pf, mut pv) = (
            r_dev.as_device_ptr(),
            q_dev.as_device_ptr(),
            refl_dev.as_device_ptr(),
            valid_dev.as_device_ptr(),
        );
        let (mut m_arg, mut n_arg) = (m_i, n_i);
        let mut params: [*mut c_void; LINALG_QR_ARG_COUNT] = [
            (&mut pr as *mut *mut c_void).cast(),
            (&mut pq as *mut *mut c_void).cast(),
            (&mut pf as *mut *mut c_void).cast(),
            (&mut pv as *mut *mut c_void).cast(),
            (&mut m_arg as *mut i32).cast(),
            (&mut n_arg as *mut i32).cast(),
        ];
        let (grid, block) = single_block(LINALG_FACTOR_BLOCK_DIM);
        // SAFETY: (U5・L4) `params` の個数・型・順序はカーネル宣言 `(double* r, double* q,
        // double* refl, int* refl_valid, int m, int n)` と一致し（`LINALG_QR_ARG_COUNT` と
        // ソース署名をテストで固定）、指す変数は下の `synchronize` 完了まで本スコープで生存する。
        // `r_dev` は `m*n`、`q_dev` は `m*k`、`refl_dev` は `k*m`、`valid_dev` は `k` 要素で全て
        // 初期化済み（`validate_qr_dims` が `i32::MAX`・確保上限以内を保証）。block 次元は
        // `LINALG_FACTOR_BLOCK_DIM`（ソースの `QR_BLOCK` と同値）、grid は 1 block。カーネル内は
        // `i < m - col`・`c < n`・`c < k` の手動境界チェック（REQ-8）を維持し、ストライド変数は
        // `long long`。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()?;

        let mut valid = vec![STATUS_SENTINEL; k];
        valid_dev.download(&mut valid)?;
        check_refl_valid(&valid, k)?;
        let mut r_work = vec![0.0f64; m * n];
        r_dev.download(&mut r_work)?;
        let mut q_work = vec![0.0f64; m * k];
        q_dev.download(&mut q_work)?;
        let (r, q) = qr_finalize(&r_work, &q_work, m, n)?;
        Ok((narrow(&r), narrow(&q)))
    }

    /// 対称固有値分解（巡回 Jacobi・下三角を読む。`a`: 行優先 `n*n`）を GPU で実行し
    /// `(固有値昇順 n, 固有ベクトル n*n・列が固有ベクトル)` を `f32` で返す。呼び出し文脈・変換規約は
    /// [`Self::run_cholesky_f32`] と同じ。
    pub fn run_eigh_f32(&self, a: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>), RocmError> {
        const OP: &str = "linalg_eigh";
        validate_square_factor_dims(OP, n)?;
        if n == 0 {
            return Err(invalid_dims(format!(
                "{OP}: n=0 はカーネルを起動しない（呼び出し側で短絡する）"
            )));
        }
        check_len(OP, "a", a.len(), n * n)?;
        let n_i = to_i32(OP, "n", n)?;

        let _guard = self.rt.select_device(self.device)?;
        let a_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &widen(a))?;
        let sym_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; n * n])?;
        let v_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; n * n])?;
        let evals_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &vec![0.0f64; n])?;
        let status_dev = DeviceBuffer::from_slice(Arc::clone(&self.rt), &[STATUS_SENTINEL])?;
        let func = self.module.function(LINALG_EIGH_KERNEL_NAME)?;
        let (mut pa, mut psym, mut pv, mut pe, mut ps) = (
            a_dev.as_device_ptr(),
            sym_dev.as_device_ptr(),
            v_dev.as_device_ptr(),
            evals_dev.as_device_ptr(),
            status_dev.as_device_ptr(),
        );
        let mut n_arg = n_i;
        let mut params: [*mut c_void; LINALG_EIGH_ARG_COUNT] = [
            (&mut pa as *mut *mut c_void).cast(),
            (&mut psym as *mut *mut c_void).cast(),
            (&mut pv as *mut *mut c_void).cast(),
            (&mut pe as *mut *mut c_void).cast(),
            (&mut ps as *mut *mut c_void).cast(),
            (&mut n_arg as *mut i32).cast(),
        ];
        let (grid, block) = single_block(LINALG_FACTOR_BLOCK_DIM);
        // SAFETY: (U5・L5) `params` の個数・型・順序はカーネル宣言 `(const double* a, double* sym,
        // double* v, double* evals, int* status, int n)` と一致し（`LINALG_EIGH_ARG_COUNT` と
        // ソース署名をテストで固定）、指す変数は下の `synchronize` 完了まで本スコープで生存する。
        // `a_dev`・`sym_dev`・`v_dev` は `n*n`、`evals_dev` は `n`、`status_dev` は 1 要素で全て
        // 初期化済み（`n*n <= i32::MAX`）。block 次元は `LINALG_FACTOR_BLOCK_DIM`（ソースの
        // `EIGH_BLOCK` と同値）、grid は 1 block。カーネル内は `idx < total`・
        // `k < n && k != p && k != q` 等の手動境界チェック（REQ-8）を維持する。回転更新の
        // `sym`／`v` への書き込みはスレッド間で重ならず、tid 0 の対角書き込みは直前の
        // `__syncthreads()` の後。早期 return は同期後の共有値 `s_conv` で全スレッド一様。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()?;

        let mut status = [STATUS_SENTINEL];
        status_dev.download(&mut status)?;
        factor_status_to_result(FactorOp::Eigh, status[0])?;
        let mut evals = vec![0.0f64; n];
        evals_dev.download(&mut evals)?;
        let mut vecs = vec![0.0f64; n * n];
        v_dev.download(&mut vecs)?;
        let (ev, vv) = eigh_sort_and_normalize(&evals, &vecs, n)?;
        Ok((narrow(&ev), narrow(&vv)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = i32::MAX as usize;

    #[test]
    fn gate_default_is_off() {
        // 単体テストのバイナリ内ではゲートを ON にしない（グローバル状態の競合回避）。
        assert!(!gpu_linalg_enabled());
    }

    #[test]
    fn lu_dims_boundaries() {
        assert!(validate_lu_dims(0, 0).is_ok());
        assert!(validate_lu_dims(46340, 1).is_ok()); // 46340^2 < i32::MAX
        assert!(validate_lu_dims(46341, 1).is_err()); // n*n が i32::MAX 超
        assert!(validate_lu_dims(MAX + 1, 0).is_err());
        assert!(validate_lu_dims(1, MAX + 1).is_err());
        assert!(validate_lu_dims(usize::MAX, usize::MAX).is_err());
        assert!(matches!(
            validate_lu_dims(MAX + 1, 0),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
    }

    #[test]
    fn square_and_qr_dims_boundaries() {
        assert!(validate_square_factor_dims("x", 0).is_ok());
        assert!(validate_square_factor_dims("x", 46340).is_ok());
        assert!(validate_square_factor_dims("x", 46341).is_err());
        assert!(validate_square_factor_dims("x", usize::MAX).is_err());
        assert!(validate_qr_dims(0, 0).is_ok());
        assert!(validate_qr_dims(MAX, 1).is_ok()); // m*n = i32::MAX
        assert!(validate_qr_dims(MAX, 2).is_err());
        assert!(validate_qr_dims(MAX + 1, 1).is_err());
        assert!(validate_qr_dims(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn length_checks() {
        assert!(check_lu_lens(4, 2, 2, 1).is_ok());
        assert!(check_lu_lens(3, 2, 2, 1).is_err());
        assert!(check_lu_lens(4, 3, 2, 1).is_err());
        assert!(check_len("x", "a", 3, 3).is_ok());
        assert!(check_len("x", "a", 3, 4).is_err());
    }

    #[test]
    fn frexp_ldexp_round_trip() {
        for x in [1.0, -3.5, 1e-300, 1e300, 5e-324, 0.1] {
            let (m, e) = frexp(x);
            assert!((0.5..1.0).contains(&m.abs()), "{x}");
            assert_eq!(ldexp(m, e), x, "{x}");
        }
        assert_eq!(frexp(0.0), (0.0, 0));
        assert!(frexp(f64::NAN).0.is_nan());
        assert_eq!(ldexp(0.9, 1024), 0.9 * 2f64.powi(1000) * 2f64.powi(24));
    }

    #[test]
    fn diag_product_handles_extreme_scales() {
        let diag: Vec<f64> = (0..22)
            .map(|i| if i % 2 == 0 { 1e30 } else { 1e-30 })
            .collect();
        let p = lu_diag_product(&diag, 1.0);
        assert!((p - 1.0).abs() < 1e-9, "{p}");
        assert_eq!(lu_diag_product(&[2.0, 3.0], -1.0), -6.0);
        assert_eq!(lu_diag_product(&[], 1.0), 1.0);
        assert_eq!(lu_diag_product(&[1e200, 1e200], 1.0), f64::INFINITY);
        assert_eq!(lu_diag_product(&[1e-200, 1e-200], 1.0), 0.0);
    }

    #[test]
    fn permutation_sign_parity() {
        assert_eq!(permutation_sign(0), 1.0);
        assert_eq!(permutation_sign(1), -1.0);
        assert_eq!(permutation_sign(2), 1.0);
    }

    #[test]
    fn lu_status_interpretation_is_fail_closed() {
        assert_eq!(
            interpret_lu_status(&[0, 3]).unwrap(),
            LuStatus {
                singular: false,
                swap_count: 3
            }
        );
        assert!(interpret_lu_status(&[1, 0]).unwrap().singular);
        for bad in [&[][..], &[0][..], &[2, 0][..], &[-1, -1][..], &[0, -1][..]] {
            assert!(
                matches!(
                    interpret_lu_status(bad),
                    Err(RocmError::LinalgKernelStatusInvalid { .. })
                ),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn det_from_lu_cases() {
        // 特異は LU を参照せず 0.0。
        assert_eq!(det_from_lu(&[1, 0], &[], 2).unwrap(), 0.0);
        // diag(2,3) の LU・swap 1 回 → -6。
        assert_eq!(
            det_from_lu(&[0, 1], &[2.0, 0.0, 0.0, 3.0], 2).unwrap(),
            -6.0
        );
        assert!(det_from_lu(&[0, 0], &[1.0], 2).is_err());
        assert!(det_from_lu(&[0, 5], &[1.0; 4], 2).is_err());
        assert!(det_from_lu(&[-1, -1], &[1.0; 4], 2).is_err());
    }

    #[test]
    fn status_mappings() {
        assert!(solve_err_to_result(0).is_ok());
        for v in [1, -1, 7] {
            assert!(matches!(
                solve_err_to_result(v),
                Err(RocmError::LinalgKernelStatusInvalid { .. })
            ));
        }
        assert!(factor_status_to_result(FactorOp::Cholesky, 0).is_ok());
        assert!(matches!(
            factor_status_to_result(FactorOp::Cholesky, 1),
            Err(RocmError::LinalgInvalidArgument { .. })
        ));
        assert!(matches!(
            factor_status_to_result(FactorOp::Eigh, 1),
            Err(RocmError::LinalgInvalidArgument { .. })
        ));
        for v in [STATUS_SENTINEL, 2] {
            assert!(matches!(
                factor_status_to_result(FactorOp::Eigh, v),
                Err(RocmError::LinalgKernelStatusInvalid { .. })
            ));
        }
        assert!(check_refl_valid(&[0, 1, 1], 3).is_ok());
        assert!(check_refl_valid(&[0, STATUS_SENTINEL], 2).is_err());
        assert!(check_refl_valid(&[0], 2).is_err());
    }

    #[test]
    fn sign_normalization_and_stable_sort() {
        let mut v = [-3.0, 3.0, 1.0];
        normalize_max_abs_sign(&mut v);
        assert_eq!(v, [3.0, -3.0, -1.0]); // 同値は最小添字（-3）が基準
        let mut z = [0.0, 0.0];
        normalize_max_abs_sign(&mut z);
        assert_eq!(z, [0.0, 0.0]);
        // 同値固有値は安定順（元の列順）。
        let (ev, vv) = eigh_sort_and_normalize(
            &[2.0, 1.0, 1.0],
            &[
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            3,
        )
        .unwrap();
        assert_eq!(ev, [1.0, 1.0, 2.0]);
        assert_eq!(vv, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(eigh_sort_and_normalize(&[1.0], &[1.0], 2).is_err());
    }

    #[test]
    fn qr_finalize_flips_negative_diagonal() {
        // m=n=2: r_work 対角 -2・3 の行を反転して対角非負にし、Q の列も反転する。
        let r_work = [-2.0, 1.0, 0.5, 3.0]; // 下三角 0.5 は 0 に潰される
        let q_work = [1.0, 0.0, 0.0, 1.0];
        let (r, q) = qr_finalize(&r_work, &q_work, 2, 2).unwrap();
        assert_eq!(r, [2.0, -1.0, 0.0, 3.0]);
        assert_eq!(q, [-1.0, 0.0, -0.0, 1.0]);
        assert!(qr_finalize(&[0.0; 3], &[0.0; 4], 2, 2).is_err());
    }

    #[test]
    fn solve_geometry() {
        assert_eq!(solve_launch_geometry(0), None);
        assert_eq!(solve_launch_geometry(1), Some(((1, 1, 1), (128, 1, 1))));
        assert_eq!(solve_launch_geometry(128), Some(((1, 1, 1), (128, 1, 1))));
        assert_eq!(solve_launch_geometry(129), Some(((2, 1, 1), (128, 1, 1))));
        assert_eq!(
            solve_launch_geometry(u32::MAX),
            Some(((u32::MAX.div_ceil(128), 1, 1), (128, 1, 1)))
        );
    }

    #[test]
    fn widen_narrow_round_trip() {
        let v = [1.5f32, -2.0, 1e20, 1e-20];
        assert_eq!(narrow(&widen(&v)), v);
    }
}
