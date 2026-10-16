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
//! # カーネル起動（Stage B。イシュー #3135・親イシュー #3133）
//!
//! cudarc の `launch` は `unsafe fn` のため、起動 2 箇所だけが本モジュールの
//! 起動ブロックになる（`docs/gpu-linalg-kernel-launch-unsafe-audit.md` の
//! 行 ID CU-1＝`launch_lu`・CU-2＝`run_lu_solve_f32`。共通不変条件 (a)〜(h)
//! は同記録 §3）。承認状態は同記録 §1 のとおり「親 #3133 本文に承認済みの
//! 記載あり（承認コメントの URL は未確認）」であり、security-auditor の
//! 監査結果は同記録 §8 に記す。
//!
//! - 実行 API: `CudaLinalg::run_lu_solve_f32`・`CudaLinalg::run_lu_inv_f32`・
//!   `CudaLinalg::run_lu_det_f32`（`ops.rs` の LU 系 3 演算が
//!   `context_cache::cached_linalg` 経由で呼ぶ）。H2D の前にホストで
//!   `f32`→`f64` へ変換し、D2H の後に 1 回だけ `as f32` する（設計記録 §4）。
//!   `status` は必ず読み戻し、特異は `LuSolveOutcome`（solve／inv）・`0.0`
//!   （det）へ写す。
//! - 実機（GB10）での NVRTC コンパイル・parity・特異写像の実測は #3138。
//!   `docs/perf/logs/cuda-linalg-lu-2674/README.md` に状態を記す。
//!
//! 分解系 3 演算（cholesky・QR・eigh）は D-C 承認（親 #3133）と監査記録（CU-3・CU-4・CU-5）に基づき
//! イシュー #3136 で起動を実装済みで、[`CudaLinalgFactor::run_cholesky_f32`]・
//! [`CudaLinalgFactor::run_qr_f32`]・[`CudaLinalgFactor::run_eigh_f32`] が `unsafe`（`launch`）を持つ。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_linalg::{
    LINALG_LU_BLOCK_DIM, LINALG_LU_F64, LINALG_LU_SOLVE_F64, LINALG_SOLVE_BLOCK_DIM,
};
use crate::kernels_linalg_factor::{
    LINALG_CHOLESKY_F64, LINALG_EIGH_F64, LINALG_FACTOR_BLOCK_DIM, LINALG_QR_F64,
};
use crate::memory::readback;
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

    /// H2D・確保・起動・readback を CUDA Graph capture 排他へ参加させる
    /// 共通ヘルパー（`mse.rs::CudaMse::with_driver_call` と同型）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// LU 分解カーネル（`linalg_lu_f64`）を単一 block で起動する。監査記録の
    /// 行 ID CU-1（本モジュールで LU 分解を起動する唯一の箇所）。
    ///
    /// 呼び出し元（`run_lu_solve_f32`・`run_lu_det_f32`）が `a_dev`＝`n*n`・
    /// `perm_dev`＝`n`・`status_dev`＝2 要素を確保済みで、`n_i` は検証済みの
    /// `n` の `i32` 表現であることを前提とする。
    fn launch_lu(
        &self,
        a_dev: &mut CudaSlice<f64>,
        perm_dev: &mut CudaSlice<i32>,
        status_dev: &mut CudaSlice<i32>,
        n_i: i32,
    ) -> Result<(), CudaError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (LU_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: CU-1（`docs/gpu-linalg-kernel-launch-unsafe-audit.md`・共通
        // 不変条件 (b)(c)(e)(f)(g)(h)）。引数は `(double* a, int* perm, int*
        // status, int n)` の順・型でカーネルのシグネチャと一致する。`a_dev`
        // は呼び出し元が `n*n` 要素で確保・H2D 済み、`perm_dev` は `n` 要素、
        // `status_dev` は 2 要素（`validate_lu_dims` と `i32::try_from` で
        // `n`・`n*n` の範囲を検証済み）。grid は `(1,1,1)` の単一 block で
        // block 次元はソースの `LU_BLOCK` と同値（`LU_BLOCK_DIM`）。カーネル
        // 内の全ループは `i < n`・`c < n` の手動境界チェックを維持する
        // （REQ-8）ため OOB 読み書きは起きない。バッファは呼び出し元の
        // クロージャ内で readback 完了まで生存する。
        unsafe {
            self.stream
                .launch_builder(&self.lu)
                .arg(a_dev)
                .arg(perm_dev)
                .arg(status_dev)
                .arg(&n_i)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `A X = B`（`a`: `n*n`・`b`: `n*k`、いずれも行優先 `f32`）を GPU で解く。
    /// `n == 0` はデバイスに触れず `Solved(vec![])`。ピボットが厳密 `0` の
    /// 特異行列は [`LuSolveOutcome::Singular`]（`ops.rs` が `InvalidArgument`
    /// へ写す）。`n > 0` かつ `k == 0` は LU だけを起動して特異判定のみ行い、
    /// 代入は起動しない（CPU の `solve` と同じ判定。監査記録 §6 の #2）。
    pub(crate) fn run_lu_solve_f32(
        &self,
        a: &[f32],
        b: &[f32],
        n: usize,
        k: usize,
    ) -> Result<LuSolveOutcome, CudaError> {
        validate_lu_dims(n, k)?;
        check_lu_lens(a.len(), b.len(), n, k)?;
        if n == 0 {
            return Ok(LuSolveOutcome::Solved(Vec::new()));
        }
        let n_i = checked_i32("n", n)?;
        let k_i = checked_i32("k", k)?;
        // `k` は `i32` に収まる検証済みのため `u32` へも収まる。
        let k_u = u32::try_from(k).map_err(|_| CudaError::InvalidElementwiseShape {
            detail: format!("linalg LU: k={k} が u32 に収まらない"),
        })?;

        self.with_driver_call(|| {
            let a64: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
            let mut a_dev = self.stream.clone_htod(&a64)?;
            let mut perm_dev = self.stream.alloc_zeros::<i32>(n)?;
            let mut status_dev = self.stream.alloc_zeros::<i32>(2)?;
            self.launch_lu(&mut a_dev, &mut perm_dev, &mut status_dev, n_i)?;

            // status の読み戻しが同期点。特異なら代入を起動せず返す。
            let status: Vec<i32> = readback(&self.stream, &status_dev)?;
            if interpret_lu_status(&status)?.singular {
                return Ok(LuSolveOutcome::Singular);
            }
            let Some(cfg) = solve_launch_config(k_u) else {
                return Ok(LuSolveOutcome::Solved(Vec::new()));
            };

            let b64: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
            let b_dev = self.stream.clone_htod(&b64)?;
            let mut x_dev = self.stream.alloc_zeros::<f64>(n * k)?;
            // perm 値域違反の通知先（カーネルは 1 を書くのみ。0 初期化）。
            let mut solve_err_dev = self.stream.alloc_zeros::<i32>(1)?;
            // SAFETY: CU-2（`docs/gpu-linalg-kernel-launch-unsafe-audit.md`・
            // 共通不変条件 (b)(c)(e)(f)(g)(h)）。引数は `(const double* lu,
            // const int* perm, const int* status, const double* b, double* x,
            // int* solve_err, int n, int k)` の順・型でカーネルと一致する。`a_dev`（LU 済み・
            // `n*n`）・`perm_dev`（`n`）・`status_dev`（2）は CU-1 と同一 stream
            // 上で先に実行済みで、`b_dev`・`x_dev` は `n*k` 要素
            // （`check_lu_lens` と `validate_lu_dims` で検証済み）。`solve_err_dev` は
            // 1 要素。`perm` の値域（`[0, n)`）はカーネル側が `b` を読む前に検査し、
            // 違反なら `solve_err` へ 1 を書いて何も書かず return する（REQ-8）。
            // ホストは下で必ず読み戻して型付きエラーにする。grid は
            // `k.div_ceil(block)`、カーネル内の `col >= k` 手動境界チェック
            // （REQ-8）で余剰スレッドは何も書かない。バッファは readback 完了
            // まで本クロージャ内で生存する。
            unsafe {
                self.stream
                    .launch_builder(&self.lu_solve)
                    .arg(&a_dev)
                    .arg(&perm_dev)
                    .arg(&status_dev)
                    .arg(&b_dev)
                    .arg(&mut x_dev)
                    .arg(&mut solve_err_dev)
                    .arg(&n_i)
                    .arg(&k_i)
                    .launch(cfg)?;
            }
            // 読み戻しが同期点。perm 値域違反は成功扱いにせず型付きエラーで返す。
            let solve_err: Vec<i32> = readback(&self.stream, &solve_err_dev)?;
            solve_err_to_result(solve_err.first().copied().unwrap_or(-1))?;
            let x: Vec<f64> = readback(&self.stream, &x_dev)?;
            Ok(LuSolveOutcome::Solved(
                x.into_iter().map(|v| v as f32).collect(),
            ))
        })
    }

    /// `inv(A)`（`A X = I` の `X`）。`B` はホストで作る単位行列（`f32` の
    /// `1.0`／`0.0` は `f64` へ厳密に変換される）で、[`Self::run_lu_solve_f32`]
    /// へ委譲する。
    pub(crate) fn run_lu_inv_f32(&self, a: &[f32], n: usize) -> Result<LuSolveOutcome, CudaError> {
        validate_lu_dims(n, n)?;
        let mut ident = vec![0.0f32; n * n];
        for i in 0..n {
            ident[i * n + i] = 1.0;
        }
        self.run_lu_solve_f32(a, &ident, n, n)
    }

    /// `det(A)`（LU 対角積 × 置換符号。CPU と同じ [`lu_diag_product`]）。
    /// `n == 0` は空積 `1.0`、特異は `0.0`（エラーにしない。CPU 契約）。
    pub(crate) fn run_lu_det_f32(&self, a: &[f32], n: usize) -> Result<f32, CudaError> {
        validate_lu_dims(n, 0)?;
        check_lu_lens(a.len(), 0, n, 0)?;
        if n == 0 {
            return Ok(1.0);
        }
        let n_i = checked_i32("n", n)?;

        self.with_driver_call(|| {
            let a64: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
            let mut a_dev = self.stream.clone_htod(&a64)?;
            let mut perm_dev = self.stream.alloc_zeros::<i32>(n)?;
            let mut status_dev = self.stream.alloc_zeros::<i32>(2)?;
            self.launch_lu(&mut a_dev, &mut perm_dev, &mut status_dev, n_i)?;

            let status: Vec<i32> = readback(&self.stream, &status_dev)?;
            if interpret_lu_status(&status)?.singular {
                // 特異ならカーネルが途中で return し `a` は分解途中のため読まない。
                return Ok(0.0f32);
            }
            let lu: Vec<f64> = readback(&self.stream, &a_dev)?;
            Ok(det_from_lu(&status, &lu, n)? as f32)
        })
    }
}

/// LU 系（solve／inv）の GPU 実行結果。特異（ピボットが厳密 `0`）は
/// `CudaError` を増やさず専用 variant で表す（`map_cuda_error` の汎用腕で
/// `KernelLaunchFailed` に化けるのと sticky エラー観測へ乗るのを避ける。
/// 設計記録 §4）。`ops.rs` が `BackendError::InvalidArgument` へ写す。
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LuSolveOutcome {
    /// 行優先 `n*k` の解（`f32`。`k == 0` のときは空）。
    Solved(Vec<f32>),
    /// 係数行列が特異。
    Singular,
}

/// LU カーネルの `status`（`[singular_flag, swap_count]`）の解釈結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LuStatus {
    pub(crate) singular: bool,
    pub(crate) swap_count: u32,
}

/// `status` を解釈する。長さ不足・負の `swap_count` は型付きエラー
/// （カーネルが書く値の整合性検査。OWASP A08）。
pub(crate) fn interpret_lu_status(status: &[i32]) -> Result<LuStatus, CudaError> {
    let invalid = |detail: String| CudaError::InvalidElementwiseShape { detail };
    let (flag, swaps) = match status {
        [flag, swaps, ..] => (*flag, *swaps),
        _ => {
            return Err(invalid(format!(
                "linalg LU: status の長さが不足（{} < 2）",
                status.len()
            )));
        }
    };
    let swap_count = u32::try_from(swaps)
        .map_err(|_| invalid(format!("linalg LU: swap_count が負: {swaps}")))?;
    Ok(LuStatus {
        singular: flag != 0,
        swap_count,
    })
}

/// LU 済み行列（行優先 `n*n`）と `status` から行列式を組み立てる。特異は
/// `lu` を参照せず `0.0`。それ以外は対角と置換符号を [`lu_diag_product`] へ渡す。
pub(crate) fn det_from_lu(
    status: &[i32],
    lu_row_major: &[f64],
    n: usize,
) -> Result<f64, CudaError> {
    let st = interpret_lu_status(status)?;
    if st.singular {
        return Ok(0.0);
    }
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| CudaError::InvalidElementwiseShape {
            detail: "linalg LU 後処理: n*n が usize を超える".to_string(),
        })?;
    if lu_row_major.len() < nn {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "linalg LU 後処理: LU の長さが不足（{} < n*n={nn}）",
                lu_row_major.len()
            ),
        });
    }
    let diag: Vec<f64> = (0..n).map(|i| lu_row_major[i * n + i]).collect();
    Ok(lu_diag_product(&diag, permutation_sign(st.swap_count)))
}

/// 代入カーネルの起動形状（grid は `ceil(k / block)`）。`k == 0` は `None`
/// （grid 0 の起動をしない）。`div_ceil` なので `u32` で overflow しない。
pub(crate) fn solve_launch_config(k: u32) -> Option<LaunchConfig> {
    if k == 0 {
        return None;
    }
    Some(LaunchConfig {
        grid_dim: (k.div_ceil(LU_SOLVE_BLOCK_DIM), 1, 1),
        block_dim: (LU_SOLVE_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// `usize` を `i32`（カーネル引数型）へ検証つきで変換する（`as` は使わない）。
fn checked_i32(name: &str, v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError::InvalidElementwiseShape {
        detail: format!("linalg LU: {name}={v} が i32 に収まらない（カーネル引数型）"),
    })
}

/// 入力スライス長が `a == n*n`・`b == n*k` と一致することを検査する
/// （H2D バッファ長とカーネルの添字範囲の一致を保証する。不変条件 (b)）。
fn check_lu_lens(a_len: usize, b_len: usize, n: usize, k: usize) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidElementwiseShape { detail };
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| invalid("linalg LU: n*n が usize を超える".to_string()))?;
    let nk = n
        .checked_mul(k)
        .ok_or_else(|| invalid("linalg LU: n*k が usize を超える".to_string()))?;
    if a_len != nn || b_len != nk {
        return Err(invalid(format!(
            "linalg LU: 入力長が不一致（a={a_len} != n*n={nn} または b={b_len} != n*k={nk}）"
        )));
    }
    Ok(())
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

/// `status`／`refl_valid` をホストが書き込む「未実行」sentinel（カーネルは
/// `{0,1}` しか書かないため、読み戻して本値のままなら起動が走っていない。
/// 監査記録 不変条件 (f)・A08 fail-closed）。
const FACTOR_STATUS_SENTINEL: i32 = -1;

/// 分解系カーネルの演算種別（`status == 1` の意味を分ける）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FactorOp {
    Cholesky,
    Eigh,
}

/// `status[0]` を判定する純関数。`0` は成功、`1` は入力起因の失敗
/// （[`CudaError::LinalgInvalidArgument`]。CPU 参照実装と同じ `InvalidArgument`
/// 分類）、それ以外（sentinel のまま・想定外の値）は
/// [`CudaError::LinalgKernelStatusInvalid`]（成功扱いにしない）。
///
/// 入力起因の失敗はドライバ障害ではないため `context_cache::observe_cuda_result`
/// は ordinal を poison しない（`CudaError::Driver` のときだけ poison する）。
fn factor_status_to_result(op: FactorOp, status: i32) -> Result<(), CudaError> {
    match status {
        0 => Ok(()),
        1 => Err(CudaError::LinalgInvalidArgument {
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
        other => Err(CudaError::LinalgKernelStatusInvalid {
            detail: format!("{op:?}: status={other} は契約外（sentinel のまま、または想定外）"),
        }),
    }
}

/// solve カーネルの `solve_err` を判定する純関数。`0` のみ成功で、`1`
/// （`perm` の値域違反）・その他は [`CudaError::LinalgKernelStatusInvalid`]
/// （黙って成功扱いにしない。REQ-8・A08）。
fn solve_err_to_result(v: i32) -> Result<(), CudaError> {
    match v {
        0 => Ok(()),
        other => Err(CudaError::LinalgKernelStatusInvalid {
            detail: format!("linalg_lu_solve: perm の値域違反を検出（solve_err={other}）"),
        }),
    }
}

/// QR の `refl_valid` が全要素 `{0,1}` であることを確認する（sentinel のまま
/// 残った要素は起動が走っていない証拠。`qr_finalize` は値を使わないが、
/// 読み戻して判定する不変条件 (f) を意味あるものにする）。
fn check_refl_valid(refl_valid: &[i32], k: usize) -> Result<(), CudaError> {
    if refl_valid.len() != k || refl_valid.iter().any(|&v| v != 0 && v != 1) {
        return Err(CudaError::LinalgKernelStatusInvalid {
            detail: format!(
                "linalg_qr: refl_valid が契約外（len={} / 期待 {k}、全要素 0/1 が必要）",
                refl_valid.len()
            ),
        });
    }
    Ok(())
}

/// `usize` の次元を `i32` へ切り詰めなしで変換する（不変条件 (c)）。
fn to_i32(op: &str, name: &str, v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError::InvalidElementwiseShape {
        detail: format!("{op}: {name}={v} が i32 に収まらない（カーネル引数型）"),
    })
}

/// 入力長が期待要素数と一致することを検証する。
fn check_len(op: &str, name: &str, got: usize, want: usize) -> Result<(), CudaError> {
    if got != want {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("{op}: {name} の長さ {got} が期待 {want} と一致しない"),
        });
    }
    Ok(())
}

/// H2D 前の f32→f64 変換（1e±20 の二乗が f32 では overflow／underflow するため）。
fn widen(a: &[f32]) -> Vec<f64> {
    a.iter().map(|&x| f64::from(x)).collect()
}

/// D2H・ホスト後処理後の `as f32`（出力ごとに 1 回だけ）。
fn narrow(a: &[f64]) -> Vec<f32> {
    a.iter().map(|&x| x as f32).collect()
}

/// 分解系 3 カーネル共通の起動形状（grid は 1 block・block は
/// `FACTOR_BLOCK_DIM`＝カーネルの `*_BLOCK` と厳密一致）。
fn factor_launch_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (FACTOR_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
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

    /// Cholesky 分解 `A = L Lᵀ`（`a`: 行優先 `n*n` の `f32`）を GPU で実行し
    /// `L`（`n*n`・`f32`）を返す（イシュー #3136・監査記録 CU-3）。
    ///
    /// `ops.rs::linalg_cholesky` から `with_driver_call` 内で呼ばれる。`pub` で
    /// 直接呼べるため入口で検証し直す（多層防御）。f32→f64 変換は H2D の前、
    /// `as f32` は D2H とホスト後処理の後に 1 回だけ。
    pub fn run_cholesky_f32(&self, a: &[f32], n: usize) -> Result<Vec<f32>, CudaError> {
        const OP: &str = "linalg_cholesky";
        validate_square_factor_dims(OP, n)?;
        if n == 0 {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!("{OP}: n=0 はカーネルを起動しない（呼び出し側で短絡する）"),
            });
        }
        check_len(OP, "a", a.len(), n * n)?;
        let n_i = to_i32(OP, "n", n)?;
        let a64 = widen(a);
        context_cache::with_driver_call(self.ordinal, || {
            let a_dev = self.stream.clone_htod(&a64)?;
            let mut l_dev = self.stream.alloc_zeros::<f64>(n * n)?;
            let mut status_dev = self.stream.clone_htod(&[FACTOR_STATUS_SENTINEL])?;
            let cfg = factor_launch_config();
            // SAFETY: CU-3（docs/gpu-linalg-kernel-launch-unsafe-audit.md §2）。
            // (a) ゲート・形状・サイズ検証は `ops.rs` と本関数入口の
            // `validate_square_factor_dims`・長さ検査で済み、デバイス確保より前。
            // (b) `a_dev`・`l_dev` は検証済みの `n*n` 要素の f64 バッファ、
            // `status_dev` は 1 要素（いずれも `n*n <= i32::MAX`）。f32→f64 変換は
            // H2D 前に完了。(c) `n_i` は `i32::try_from` で切り詰めなし。
            // (d) 本クロージャは `context_cache::with_driver_call` の内側で、
            // 構築側も `ops.rs` の `with_driver_call` 内。(e) 引数順・型は
            // `linalg_cholesky_f64(const double* a, double* l, int* status, int n)`
            // と一致。(f) `status_dev` は下で必ず読み戻して判定する。
            // (g) 3 バッファは下の readback（同期点）の完了まで本スコープで生存。
            // block 次元は `FACTOR_BLOCK_DIM`（カーネルの CHOL_BLOCK=256 と厳密
            // 一致）・grid は 1 block で block 間同期は不要。カーネル内は
            // `idx < total`・`i < n` の手動境界チェックを維持（REQ-8）。
            // int 添字は `n*n <= i32::MAX` で overflow しない。
            unsafe {
                self.stream
                    .launch_builder(&self.cholesky)
                    .arg(&a_dev)
                    .arg(&mut l_dev)
                    .arg(&mut status_dev)
                    .arg(&n_i)
                    .launch(cfg)?;
            }
            let status: Vec<i32> = readback(&self.stream, &status_dev)?;
            let status0 = status.first().copied().unwrap_or(FACTOR_STATUS_SENTINEL);
            factor_status_to_result(FactorOp::Cholesky, status0)?;
            let l64: Vec<f64> = readback(&self.stream, &l_dev)?;
            check_len(OP, "l(readback)", l64.len(), n * n)?;
            Ok(narrow(&l64))
        })
    }

    /// reduced Householder QR（`a`: 行優先 `m*n`）を GPU で実行し
    /// `(R: k*n, Q: m*k)`（`k = min(m,n)`・`f32`）を返す（イシュー #3136・
    /// 監査記録 CU-4）。呼び出し文脈・変換規約は [`Self::run_cholesky_f32`] と同じ。
    pub fn run_qr_f32(
        &self,
        a: &[f32],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        const OP: &str = "linalg_qr";
        validate_qr_dims(m, n)?;
        if m == 0 || n == 0 {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!("{OP}: 空入力はカーネルを起動しない（呼び出し側で短絡する）"),
            });
        }
        let k = m.min(n);
        check_len(OP, "a", a.len(), m * n)?;
        let m_i = to_i32(OP, "m", m)?;
        let n_i = to_i32(OP, "n", n)?;
        let a64 = widen(a);
        context_cache::with_driver_call(self.ordinal, || {
            let mut r_dev = self.stream.clone_htod(&a64)?;
            let mut q_dev = self.stream.alloc_zeros::<f64>(m * k)?;
            let mut refl_dev = self.stream.alloc_zeros::<f64>(k * m)?;
            let mut valid_dev = self.stream.clone_htod(&vec![FACTOR_STATUS_SENTINEL; k])?;
            let cfg = factor_launch_config();
            // SAFETY: CU-4（docs/gpu-linalg-kernel-launch-unsafe-audit.md §2）。
            // (a) 検証は `ops.rs` と入口の `validate_qr_dims`・長さ検査で済み、
            // デバイス確保より前。(b) `r_dev` は `m*n`（入力の f64 コピーを H2D）、
            // `q_dev` は `m*k`、`refl_dev` は `k*m`、`valid_dev` は `k` 要素で、
            // いずれも `validate_qr_dims` が `i32::MAX`・確保上限以内を保証。
            // (c) `m_i`・`n_i` は `i32::try_from` で切り詰めなし。
            // (d) `context_cache::with_driver_call` の内側（構築側も `ops.rs` の
            // `with_driver_call` 内）。(e) 引数順・型は
            // `linalg_qr_f64(double* r, double* q, double* refl, int* refl_valid,
            // int m, int n)` と一致。(f) `valid_dev` は下で必ず読み戻し全要素が
            // {0,1} であることを確認する。(g) 4 バッファは readback の完了まで
            // 本スコープで生存。block 次元は `FACTOR_BLOCK_DIM`（QR_BLOCK=256 と
            // 厳密一致）・grid は 1 block で block 間同期は不要。カーネル内は
            // `i < m - col`・`c < n`・`c < k` の手動境界チェックを維持（REQ-8）。
            // ストライド変数は `long long`（`idx += 256` の int overflow を回避）。
            unsafe {
                self.stream
                    .launch_builder(&self.qr)
                    .arg(&mut r_dev)
                    .arg(&mut q_dev)
                    .arg(&mut refl_dev)
                    .arg(&mut valid_dev)
                    .arg(&m_i)
                    .arg(&n_i)
                    .launch(cfg)?;
            }
            let valid: Vec<i32> = readback(&self.stream, &valid_dev)?;
            check_refl_valid(&valid, k)?;
            let r_work: Vec<f64> = readback(&self.stream, &r_dev)?;
            let q_work: Vec<f64> = readback(&self.stream, &q_dev)?;
            let (r, q) = qr_finalize(&r_work, &q_work, m, n)?;
            Ok((narrow(&r), narrow(&q)))
        })
    }

    /// 対称固有値分解（巡回 Jacobi・下三角を読む。`a`: 行優先 `n*n`）を GPU で
    /// 実行し `(固有値昇順 n, 固有ベクトル n*n・列が固有ベクトル)` を `f32` で
    /// 返す（イシュー #3136・監査記録 CU-5）。呼び出し文脈・変換規約は
    /// [`Self::run_cholesky_f32`] と同じ。
    pub fn run_eigh_f32(&self, a: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        const OP: &str = "linalg_eigh";
        validate_square_factor_dims(OP, n)?;
        if n == 0 {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!("{OP}: n=0 はカーネルを起動しない（呼び出し側で短絡する）"),
            });
        }
        check_len(OP, "a", a.len(), n * n)?;
        let n_i = to_i32(OP, "n", n)?;
        let a64 = widen(a);
        context_cache::with_driver_call(self.ordinal, || {
            let a_dev = self.stream.clone_htod(&a64)?;
            let mut sym_dev = self.stream.alloc_zeros::<f64>(n * n)?;
            let mut v_dev = self.stream.alloc_zeros::<f64>(n * n)?;
            let mut evals_dev = self.stream.alloc_zeros::<f64>(n)?;
            let mut status_dev = self.stream.clone_htod(&[FACTOR_STATUS_SENTINEL])?;
            let cfg = factor_launch_config();
            // SAFETY: CU-5（docs/gpu-linalg-kernel-launch-unsafe-audit.md §2・§5）。
            // (a) 検証は `ops.rs` と入口の `validate_square_factor_dims`・長さ検査
            // で済み、デバイス確保より前。(b) `a_dev`・`sym_dev`・`v_dev` は
            // `n*n`、`evals_dev` は `n`、`status_dev` は 1 要素（`n*n <= i32::MAX`）。
            // (c) `n_i` は `i32::try_from` で切り詰めなし。(d) `context_cache::
            // with_driver_call` の内側（構築側も `ops.rs` の `with_driver_call` 内）。
            // (e) 引数順・型は `linalg_eigh_f64(const double* a, double* sym,
            // double* v, double* evals, int* status, int n)` と一致。
            // (f) `status_dev` は下で必ず読み戻して判定する。(g) 5 バッファは
            // readback の完了まで本スコープで生存。block 次元は `FACTOR_BLOCK_DIM`
            // （EIGH_BLOCK=256 と厳密一致）・grid は 1 block で block 間同期は不要。
            // カーネル内は `idx < total`・`k < n && k != p && k != q` 等の手動
            // 境界チェックを維持（REQ-8）。回転更新の `sym`／`v` 書き込みはバリアが
            // ないが、書き込み先配列が別で、スレッド k が書く `sym[k*n+p|q]`・
            // `sym[p|q*n+k]`（k≠p,q）・`v[k*n+p|q]` は他スレッドと重ならず、
            // tid 0 の対角書き込みは直前の `__syncthreads()` の後（§5。最終判定は
            // 監査記録 §8 の #3136 節）。早期 return は同期後の共有値 `s_conv` で
            // 一様。
            unsafe {
                self.stream
                    .launch_builder(&self.eigh)
                    .arg(&a_dev)
                    .arg(&mut sym_dev)
                    .arg(&mut v_dev)
                    .arg(&mut evals_dev)
                    .arg(&mut status_dev)
                    .arg(&n_i)
                    .launch(cfg)?;
            }
            let status: Vec<i32> = readback(&self.stream, &status_dev)?;
            let status0 = status.first().copied().unwrap_or(FACTOR_STATUS_SENTINEL);
            factor_status_to_result(FactorOp::Eigh, status0)?;
            let evals: Vec<f64> = readback(&self.stream, &evals_dev)?;
            let vecs: Vec<f64> = readback(&self.stream, &v_dev)?;
            let (ev, vv) = eigh_sort_and_normalize(&evals, &vecs, n)?;
            Ok((narrow(&ev), narrow(&vv)))
        })
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
    fn factor_status_mapping() {
        assert!(factor_status_to_result(FactorOp::Cholesky, 0).is_ok());
        assert!(factor_status_to_result(FactorOp::Eigh, 0).is_ok());
        for op in [FactorOp::Cholesky, FactorOp::Eigh] {
            assert!(matches!(
                factor_status_to_result(op, 1),
                Err(CudaError::LinalgInvalidArgument { .. })
            ));
            for bad in [FACTOR_STATUS_SENTINEL, 2, i32::MIN] {
                assert!(matches!(
                    factor_status_to_result(op, bad),
                    Err(CudaError::LinalgKernelStatusInvalid { .. })
                ));
            }
        }
    }

    #[test]
    fn solve_err_mapping() {
        assert!(solve_err_to_result(0).is_ok());
        for bad in [1, -1, i32::MIN] {
            assert!(matches!(
                solve_err_to_result(bad),
                Err(CudaError::LinalgKernelStatusInvalid { .. })
            ));
        }
    }

    #[test]
    fn refl_valid_check() {
        assert!(check_refl_valid(&[0, 1, 1], 3).is_ok());
        assert!(check_refl_valid(&[0, FACTOR_STATUS_SENTINEL], 2).is_err());
        assert!(check_refl_valid(&[2], 1).is_err());
        assert!(check_refl_valid(&[1], 2).is_err());
    }

    /// 1e±20 の f32 を f64 へ昇格してから二乗しても有限・非ゼロ（f32 のままなら
    /// overflow／underflow する。H2D 前の変換が必要な根拠）。
    #[test]
    fn widen_keeps_extreme_squares_finite() {
        for x in widen(&[1e20f32, 1e-20f32]) {
            let sq = x * x;
            assert!(sq.is_finite() && sq != 0.0, "x={x}");
        }
        assert!((1e20f32 * 1e20f32).is_infinite());
    }

    #[test]
    fn run_input_helpers_reject_bad_values() {
        assert!(check_len("t", "a", 3, 4).is_err());
        assert!(check_len("t", "a", 4, 4).is_ok());
        assert!(to_i32("t", "n", i32::MAX as usize).is_ok());
        assert!(to_i32("t", "n", i32::MAX as usize + 1).is_err());
    }

    #[test]
    fn permutation_sign_follows_swap_parity() {
        assert_eq!(permutation_sign(0), 1.0);
        assert_eq!(permutation_sign(1), -1.0);
        assert_eq!(permutation_sign(4), 1.0);
    }

    #[test]
    fn solve_launch_config_grid_and_zero() {
        assert!(solve_launch_config(0).is_none());
        let g = |k| solve_launch_config(k).expect("cfg").grid_dim.0;
        assert_eq!(g(1), 1);
        assert_eq!(g(LU_SOLVE_BLOCK_DIM), 1);
        assert_eq!(g(LU_SOLVE_BLOCK_DIM + 1), 2);
        assert_eq!(
            g(i32::MAX as u32),
            (i32::MAX as u32).div_ceil(LU_SOLVE_BLOCK_DIM)
        );
        assert_eq!(g(u32::MAX), u32::MAX.div_ceil(LU_SOLVE_BLOCK_DIM));
        let cfg = solve_launch_config(5).expect("cfg");
        assert_eq!(cfg.block_dim, (LU_SOLVE_BLOCK_DIM, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn interpret_status_cases() {
        assert_eq!(
            interpret_lu_status(&[0, 3]).unwrap(),
            LuStatus {
                singular: false,
                swap_count: 3
            }
        );
        assert!(interpret_lu_status(&[1, 0]).unwrap().singular);
        assert!(interpret_lu_status(&[0]).is_err());
        assert!(interpret_lu_status(&[]).is_err());
        assert!(interpret_lu_status(&[0, -1]).is_err());
    }

    #[test]
    fn det_from_lu_sign_and_singular() {
        let lu = [2.0, 9.0, 9.0, 3.0]; // 対角 2・3（非対角は無関係）
        assert_eq!(det_from_lu(&[0, 0], &lu, 2).unwrap(), 6.0);
        assert_eq!(det_from_lu(&[0, 1], &lu, 2).unwrap(), -6.0);
        assert_eq!(det_from_lu(&[0, 2], &lu, 2).unwrap(), 6.0);
        // 特異は LU を参照せず 0.0（途中状態でも長さ検査に落ちない）。
        assert_eq!(det_from_lu(&[1, 0], &[], 2).unwrap(), 0.0);
        assert!(det_from_lu(&[0, 0], &[1.0], 2).is_err());
        assert!(det_from_lu(&[0], &lu, 2).is_err());
        // CPU と同じ演算列（lu_diag_product 直接）と一致。
        assert_eq!(
            det_from_lu(&[0, 1], &lu, 2).unwrap(),
            lu_diag_product(&[2.0, 3.0], -1.0)
        );
    }

    #[test]
    fn lens_and_i32_checks() {
        assert!(check_lu_lens(4, 2, 2, 1).is_ok());
        assert!(check_lu_lens(4, 0, 2, 0).is_ok());
        assert!(check_lu_lens(3, 2, 2, 1).is_err());
        assert!(check_lu_lens(4, 3, 2, 1).is_err());
        assert!(check_lu_lens(0, 0, usize::MAX, 2).is_err());
        assert_eq!(checked_i32("n", 5).unwrap(), 5);
        assert!(checked_i32("n", i32::MAX as usize + 1).is_err());
    }
}
