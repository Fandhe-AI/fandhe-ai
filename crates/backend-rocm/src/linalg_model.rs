//! `kernels_linalg*.rs` の 5 カーネルを CPU 上で逐語に再現するテスト専用モデルと、CPU 参照実装との照合。
//!
//! 役割: AMD 実機がない CI で、各カーネルの添字計算・同期区切りのフェーズ分け・共有メモリの木構造縮約・
//! 判定値の一様分岐・status／補助フラグの書き込みを検証する。`gemm_model.rs`・Metal の
//! `linalg_model.rs` と同じ方式。`__syncthreads` で区切られたフェーズを `for tid in 0..BLOCK` で回し、
//! 範囲外添字は `Vec` の添字アクセスの panic で検出する。デバイス出力バッファは `NaN`／sentinel で初期化し
//! （`hipMalloc` が未初期化である前提の再現）、読まれる要素が全て先に書かれていることを数値の NaN 伝播で
//! 検出する。実機での HIPRTC コンパイル結果の検証ではない（実機 parity は #3159 段 B・
//! `tests/linalg_real_device.rs`）。
//!
//! モデルの出力は本番と同じホスト後処理関数（`linalg.rs`）へ通し、`CpuBackendOps::linalg_*` と
//! 統一複合判定（`assert_parity`）で照合する。さらに、Rust の `f64` 演算は縮約されず正しく丸められる
//! ため CPU 逐次実装と bit 一致するはずで、不一致はモデル・移植の演算順序の誤りとして扱う
//! （tolerance は緩めない）。

use crate::error::RocmError;
use crate::kernels_linalg::{LINALG_LU_BLOCK_DIM, LINALG_SOLVE_BLOCK_DIM};
use crate::kernels_linalg_factor::{LINALG_EIGH_MAX_SWEEPS, LINALG_FACTOR_BLOCK_DIM};
use crate::linalg::{
    FactorOp, LuSolveOutcome, check_len, check_lu_lens, check_refl_valid, det_from_lu,
    eigh_sort_and_normalize, factor_status_to_result, interpret_lu_status, narrow, qr_finalize,
    solve_err_to_result, to_i32, validate_lu_dims, validate_qr_dims, validate_square_factor_dims,
    widen,
};

const LU_B: usize = LINALG_LU_BLOCK_DIM as usize;
const SOLVE_B: usize = LINALG_SOLVE_BLOCK_DIM as usize;
const FACT_B: usize = LINALG_FACTOR_BLOCK_DIM as usize;
/// `status` 等の未書き込み sentinel（`linalg.rs::STATUS_SENTINEL` と同値）。
const SENTINEL: i32 = -1;

/// `for (i = start + tid; i < bound; i += block)` を全 tid ぶん順に回す。
fn strided(start: usize, bound: usize, block: usize, mut f: impl FnMut(usize)) {
    for tid in 0..block {
        let mut i = start + tid;
        while i < bound {
            f(i);
            i += block;
        }
    }
}

/// `linalg_lu_f64`。`a` を in-place で LU 化し `(perm, status)` を返す。
pub(crate) fn lu(a: &mut [f64], n: usize) -> (Vec<i32>, [i32; 2]) {
    let mut perm = vec![i32::MIN; n];
    let mut status = [SENTINEL; 2];
    strided(0, n, LU_B, |i| perm[i] = i as i32);
    status[0] = 0;
    status[1] = 0;
    for k in 0..n {
        // ピボット探索（thread ごとの strided 走査 → 共有メモリの木構造縮約）。
        let mut s_val = vec![0.0f64; LU_B];
        let mut s_idx = vec![0usize; LU_B];
        for tid in 0..LU_B {
            let mut best_v = -1.0f64;
            let mut best_i = n;
            let mut i = k + 1 + tid;
            while i < n {
                let v = a[i * n + k].abs();
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
                i += LU_B;
            }
            s_val[tid] = best_v;
            s_idx[tid] = best_i;
        }
        let mut stride = LU_B / 2;
        while stride > 0 {
            for tid in 0..stride {
                let (ov, oi) = (s_val[tid + stride], s_idx[tid + stride]);
                if ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid]) {
                    s_val[tid] = ov;
                    s_idx[tid] = oi;
                }
            }
            stride >>= 1;
        }
        let v0 = a[k * n + k].abs();
        let mut p = k;
        if s_idx[0] < n && s_val[0] > v0 {
            p = s_idx[0];
        }
        if p != k {
            strided(0, n, LU_B, |c| {
                a.swap(k * n + c, p * n + c);
            });
            perm.swap(k, p);
            status[1] += 1;
        }
        let pivot = a[k * n + k];
        if pivot == 0.0 {
            status[0] = 1;
            return (perm, status);
        }
        strided(k + 1, n, LU_B, |i| a[i * n + k] /= pivot);
        let m = n - k - 1;
        strided(0, m * m, LU_B, |e| {
            let i = k + 1 + e / m;
            let c = k + 1 + e % m;
            if i < n && c < n {
                let f = a[i * n + k];
                if f != 0.0 {
                    a[i * n + c] -= f * a[k * n + c];
                }
            }
        });
    }
    (perm, status)
}

/// `linalg_lu_solve_f64`。`(x, solve_err)` を返す（`x` は未書き込み要素が NaN）。
pub(crate) fn lu_solve(
    lu: &[f64],
    perm: &[i32],
    status: &[i32],
    b: &[f64],
    n: usize,
    k: usize,
) -> (Vec<f64>, i32) {
    let mut x = vec![f64::NAN; n * k];
    let mut solve_err = 0;
    for blk in 0..k.div_ceil(SOLVE_B) {
        for t in 0..SOLVE_B {
            let col = blk * SOLVE_B + t;
            if col >= k || status[0] != 0 {
                continue;
            }
            if perm.iter().take(n).any(|&pi| pi < 0 || pi as usize >= n) {
                solve_err = 1;
                continue;
            }
            for i in 0..n {
                let mut sum = b[perm[i] as usize * k + col];
                for j in 0..i {
                    sum -= lu[i * n + j] * x[j * k + col];
                }
                x[i * k + col] = sum;
            }
            for i in (0..n).rev() {
                let mut sum = x[i * k + col];
                for j in i + 1..n {
                    sum -= lu[i * n + j] * x[j * k + col];
                }
                x[i * k + col] = sum / lu[i * n + i];
            }
        }
    }
    (x, solve_err)
}

/// `linalg_cholesky_f64`。`(l, status)` を返す。
pub(crate) fn cholesky(a: &[f64], n: usize) -> (Vec<f64>, i32) {
    let mut l = vec![f64::NAN; n * n];
    strided(0, n * n, FACT_B, |idx| l[idx] = 0.0);
    for j in 0..n {
        let mut sum = a[j * n + j];
        for k in 0..j {
            let ljk = l[j * n + k];
            sum -= ljk * ljk;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return (l, 1);
        }
        let d = sum.sqrt();
        l[j * n + j] = d;
        strided(j + 1, n, FACT_B, |i| {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = sum / d;
        });
    }
    (l, 0)
}

/// `linalg_qr_f64`。`(r_work, q, refl_valid)` を返す。
pub(crate) fn qr(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let k = m.min(n);
    let mut r = a.to_vec();
    let mut q = vec![f64::NAN; m * k];
    let mut refl = vec![f64::NAN; k * m];
    let mut valid = vec![SENTINEL; k];
    for col in 0..k {
        // thread 0: 反射ベクトルの構築（逐次縮約）。
        let mut norm_x = 0.0f64;
        for i in col..m {
            let x = r[i * n + col];
            norm_x += x * x;
        }
        norm_x = norm_x.sqrt();
        let mut v = 0;
        let mut norm_v = 0.0f64;
        if norm_x != 0.0 {
            let x0 = r[col * n + col];
            let alpha = if x0 >= 0.0 { -norm_x } else { norm_x };
            refl[col * m] = x0 - alpha;
            for i in 1..m - col {
                refl[col * m + i] = r[(col + i) * n + col];
            }
            let mut acc = 0.0f64;
            for i in 0..m - col {
                let e = refl[col * m + i];
                acc += e * e;
            }
            norm_v = acc.sqrt();
            if norm_v != 0.0 {
                v = 1;
            }
        }
        valid[col] = v;
        if v != 0 {
            strided(0, m - col, FACT_B, |i| refl[col * m + i] /= norm_v);
            strided(col, n, FACT_B, |c| {
                let mut dot = 0.0f64;
                for i in 0..m - col {
                    dot += refl[col * m + i] * r[(col + i) * n + c];
                }
                if dot != 0.0 {
                    for i in 0..m - col {
                        let vi = refl[col * m + i];
                        let cur = r[(col + i) * n + c];
                        r[(col + i) * n + c] = cur - (2.0 * vi) * dot;
                    }
                }
            });
        }
    }
    strided(0, m * k, FACT_B, |idx| q[idx] = 0.0);
    strided(0, k, FACT_B, |i| q[i * k + i] = 1.0);
    for col in (0..k).rev() {
        if valid[col] != 0 {
            strided(0, k, FACT_B, |c| {
                let mut dot = 0.0f64;
                for i in 0..m - col {
                    dot += refl[col * m + i] * q[(col + i) * k + c];
                }
                if dot != 0.0 {
                    for i in 0..m - col {
                        let vi = refl[col * m + i];
                        let cur = q[(col + i) * k + c];
                        q[(col + i) * k + c] = cur - (2.0 * vi) * dot;
                    }
                }
            });
        }
    }
    (r, q, valid)
}

/// `linalg_eigh_f64`。`(evals, v, status)` を返す（`evals` は未ソート）。
pub(crate) fn eigh_kernel(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, i32) {
    let mut sym = vec![f64::NAN; n * n];
    let mut v = vec![f64::NAN; n * n];
    let mut evals = vec![f64::NAN; n];
    strided(0, n * n, FACT_B, |idx| {
        let (i, j) = (idx / n, idx % n);
        sym[idx] = if i >= j { a[i * n + j] } else { a[j * n + i] };
        v[idx] = if i == j { 1.0 } else { 0.0 };
    });
    let mut acc = 0.0f64;
    for &x in &sym {
        acc += x * x;
    }
    let frob = acc.sqrt();
    for _sweep in 0..LINALG_EIGH_MAX_SWEEPS {
        let mut off_sq = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let x = sym[p * n + q];
                off_sq += x * x;
            }
        }
        if (2.0 * off_sq).sqrt() <= 1e-14 * frob {
            strided(0, n, FACT_B, |i| evals[i] = sym[i * n + i]);
            return (evals, v, 0);
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = sym[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = sym[p * n + p];
                let aqq = sym[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let mut t = 1.0f64;
                if theta != 0.0 {
                    let denom = theta.abs() + (1.0 + theta * theta).sqrt();
                    t = 1.0f64.copysign(theta) / denom;
                }
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                // 非対角の回転更新（k ごとに独立。k != p, q）。
                strided(0, n, FACT_B, |k| {
                    if k != p && k != q {
                        let akp = sym[k * n + p];
                        let akq = sym[k * n + q];
                        let new_kp = c * akp - s * akq;
                        let new_kq = s * akp + c * akq;
                        sym[k * n + p] = new_kp;
                        sym[p * n + k] = new_kp;
                        sym[k * n + q] = new_kq;
                        sym[q * n + k] = new_kq;
                    }
                });
                // thread 0: 対角と (p,q) 成分（直前の同期の後）。
                let cc = c * c;
                let ss = s * s;
                let cross = (2.0 * s) * c;
                let app_new = (cc * app - cross * apq) + ss * aqq;
                let aqq_new = (ss * app + cross * apq) + cc * aqq;
                sym[p * n + p] = app_new;
                sym[q * n + q] = aqq_new;
                sym[p * n + q] = 0.0;
                sym[q * n + p] = 0.0;
                strided(0, n, FACT_B, |k| {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                });
            }
        }
    }
    (evals, v, 1)
}

// ---- 本番 `run_*_f32` と同じ前段・後段を通すモデル版 ----

/// `RocmLinalgLu::run_lu_solve_f32` のモデル版（デバイス部分だけをモデルへ置換）。
pub(crate) fn run_solve(
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
    to_i32("linalg LU", "n", n)?;
    to_i32("linalg LU", "k", k)?;
    let mut a64 = widen(a);
    let (perm, status) = lu(&mut a64, n);
    if interpret_lu_status(&status)?.singular {
        return Ok(LuSolveOutcome::Singular);
    }
    if k == 0 {
        return Ok(LuSolveOutcome::Solved(Vec::new()));
    }
    let (x, err) = lu_solve(&a64, &perm, &status, &widen(b), n, k);
    solve_err_to_result(err)?;
    Ok(LuSolveOutcome::Solved(narrow(&x)))
}

pub(crate) fn run_inv(a: &[f32], n: usize) -> Result<LuSolveOutcome, RocmError> {
    validate_lu_dims(n, n)?;
    let mut ident = vec![0.0f32; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    run_solve(a, &ident, n, n)
}

pub(crate) fn run_det(a: &[f32], n: usize) -> Result<f32, RocmError> {
    validate_lu_dims(n, 0)?;
    check_lu_lens(a.len(), 0, n, 0)?;
    if n == 0 {
        return Ok(1.0);
    }
    let mut a64 = widen(a);
    let (_perm, status) = lu(&mut a64, n);
    if interpret_lu_status(&status)?.singular {
        return Ok(0.0);
    }
    Ok(det_from_lu(&status, &a64, n)? as f32)
}

pub(crate) fn run_cholesky(a: &[f32], n: usize) -> Result<Vec<f32>, RocmError> {
    validate_square_factor_dims("linalg_cholesky", n)?;
    check_len("linalg_cholesky", "a", a.len(), n * n)?;
    let (l, status) = cholesky(&widen(a), n);
    factor_status_to_result(FactorOp::Cholesky, status)?;
    Ok(narrow(&l))
}

pub(crate) fn run_qr(a: &[f32], m: usize, n: usize) -> Result<(Vec<f32>, Vec<f32>), RocmError> {
    validate_qr_dims(m, n)?;
    check_len("linalg_qr", "a", a.len(), m * n)?;
    let (r_work, q_work, valid) = qr(&widen(a), m, n);
    check_refl_valid(&valid, m.min(n))?;
    let (r, q) = qr_finalize(&r_work, &q_work, m, n)?;
    Ok((narrow(&r), narrow(&q)))
}

pub(crate) fn run_eigh(a: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>), RocmError> {
    validate_square_factor_dims("linalg_eigh", n)?;
    check_len("linalg_eigh", "a", a.len(), n * n)?;
    let (evals, vecs, status) = eigh_kernel(&widen(a), n);
    factor_status_to_result(FactorOp::Eigh, status)?;
    let (ev, vv) = eigh_sort_and_normalize(&evals, &vecs, n)?;
    Ok((narrow(&ev), narrow(&vv)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    /// 決定的な線形合同法（`[-1, 1)` の f32）。
    fn rand_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// 対角優位行列（条件数が小さい）。
    fn diag_dominant(n: usize, seed: u64) -> Vec<f32> {
        let mut a = rand_vec(seed, n * n);
        for i in 0..n {
            a[i * n + i] += n as f32 + 1.0;
        }
        a
    }

    /// 対称正定値（`M Mᵀ/n + I`。κ は数十程度）。下三角から対称に作る。
    fn spd(n: usize, seed: u64) -> Vec<f32> {
        let m = rand_vec(seed, n * n);
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut acc = 0.0f64;
                for t in 0..n {
                    acc += f64::from(m[i * n + t]) * f64::from(m[j * n + t]);
                }
                let v = (acc / n as f64 + if i == j { 1.0 } else { 0.0 }) as f32;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    fn sym(n: usize, seed: u64) -> Vec<f32> {
        let m = rand_vec(seed, n * n);
        let mut a = m.clone();
        for i in 0..n {
            for j in 0..i {
                a[j * n + i] = a[i * n + j];
            }
        }
        a
    }

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    fn v(x: &Tensor<f32>) -> Vec<f32> {
        x.as_slice().unwrap().to_vec()
    }

    /// 統一複合判定と bit 一致の両方を assert する。
    fn check(ctx: &str, actual: &[f32], expected: &[f32]) {
        assert_parity(ctx, actual, expected);
        let diff = actual
            .iter()
            .zip(expected)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diff, 0, "{ctx}: CPU 逐次実装と bit 不一致 {diff} 要素");
    }

    const LU_SIZES: [usize; 8] = [1, 2, 3, 17, 64, 65, 129, 257];

    #[test]
    fn solve_matches_cpu_across_block_boundaries() {
        let cpu = CpuBackendOps::new();
        for &n in &LU_SIZES {
            for k in [1usize, 3] {
                let a = diag_dominant(n, 10 + n as u64);
                let b = rand_vec(20 + n as u64, n * k);
                let got = run_solve(&a, &b, n, k).unwrap();
                let want = cpu.linalg_solve(&t(&a, &[n, n]), &t(&b, &[n, k])).unwrap();
                let LuSolveOutcome::Solved(x) = got else {
                    panic!("n={n} singular")
                };
                check(&format!("solve n={n} k={k}"), &x, &v(&want));
            }
        }
        // k が代入カーネルの block（128）を跨ぐ。
        let (n, k) = (5, 300);
        let a = diag_dominant(n, 7);
        let b = rand_vec(8, n * k);
        let LuSolveOutcome::Solved(x) = run_solve(&a, &b, n, k).unwrap() else {
            panic!()
        };
        let want = cpu.linalg_solve(&t(&a, &[n, n]), &t(&b, &[n, k])).unwrap();
        check("solve k>block", &x, &v(&want));
    }

    #[test]
    fn solve_needs_row_pivoting() {
        // 先頭ピボットが 0 に近い行列で置換が起きる（swap 経路・perm を通す）。
        let cpu = CpuBackendOps::new();
        for n in [2usize, 17, 65] {
            let mut a = rand_vec(3 + n as u64, n * n);
            a[0] = 1e-6;
            let b = rand_vec(4, n * 2);
            let LuSolveOutcome::Solved(x) = run_solve(&a, &b, n, 2).unwrap() else {
                panic!()
            };
            let want = cpu.linalg_solve(&t(&a, &[n, n]), &t(&b, &[n, 2])).unwrap();
            check(&format!("pivot solve n={n}"), &x, &v(&want));
        }
    }

    #[test]
    fn inv_and_det_match_cpu() {
        let cpu = CpuBackendOps::new();
        for &n in &[1usize, 2, 3, 17, 64, 65] {
            let a = diag_dominant(n, 40 + n as u64);
            let LuSolveOutcome::Solved(x) = run_inv(&a, n).unwrap() else {
                panic!()
            };
            check(
                &format!("inv n={n}"),
                &x,
                &v(&cpu.linalg_inv(&t(&a, &[n, n])).unwrap()),
            );
        }
        for &n in &LU_SIZES {
            // det は積が f32 範囲に収まるよう単位に近い行列（対角 1 + 小さな摂動）にする。
            let mut a: Vec<f32> = rand_vec(50 + n as u64, n * n)
                .iter()
                .map(|x| x * 0.01)
                .collect();
            for i in 0..n {
                a[i * n + i] += 1.0;
            }
            let got = run_det(&a, n).unwrap();
            let want = cpu.linalg_det(&t(&a, &[n, n])).unwrap();
            check(&format!("det n={n}"), &[got], &v(&want));
        }
        assert_eq!(run_det(&[], 0).unwrap(), 1.0);
    }

    #[test]
    fn scaled_inputs_match_cpu() {
        let cpu = CpuBackendOps::new();
        for scale in [1e20f32, 1e-20] {
            let n = 17;
            let a: Vec<f32> = diag_dominant(n, 5).iter().map(|x| x * scale).collect();
            let b = rand_vec(6, n);
            let LuSolveOutcome::Solved(x) = run_solve(&a, &b, n, 1).unwrap() else {
                panic!()
            };
            let want = cpu.linalg_solve(&t(&a, &[n, n]), &t(&b, &[n, 1])).unwrap();
            check(&format!("scaled solve {scale:e}"), &x, &v(&want));
            let spd_s: Vec<f32> = spd(n, 9).iter().map(|x| x * scale).collect();
            let l = run_cholesky(&spd_s, n).unwrap();
            check(
                &format!("scaled cholesky {scale:e}"),
                &l,
                &v(&cpu.linalg_cholesky(&t(&spd_s, &[n, n])).unwrap()),
            );
        }
        // 1e20 × I2 は f32 の二乗が overflow するが内部 f64 で正しく計算される。
        let a = [1e20f32, 0.0, 0.0, 1e20];
        let l = run_cholesky(&a, 2).unwrap();
        check(
            "1e20*I2 cholesky",
            &l,
            &v(&cpu.linalg_cholesky(&t(&a, &[2, 2])).unwrap()),
        );
    }

    #[test]
    fn singular_maps_to_singular_and_zero_det() {
        let cpu = CpuBackendOps::new();
        for a in [vec![1.0f32, 2.0, 2.0, 4.0], vec![0.0f32; 9]] {
            let n = if a.len() == 4 { 2 } else { 3 };
            assert_eq!(
                run_solve(&a, &vec![1.0; n], n, 1).unwrap(),
                LuSolveOutcome::Singular
            );
            assert_eq!(run_inv(&a, n).unwrap(), LuSolveOutcome::Singular);
            assert_eq!(run_det(&a, n).unwrap(), 0.0);
            assert!(cpu.linalg_inv(&t(&a, &[n, n])).is_err());
        }
        // k == 0 は特異判定のみ（Singular／空の Solved）。
        assert_eq!(
            run_solve(&[1.0, 2.0, 2.0, 4.0], &[], 2, 0).unwrap(),
            LuSolveOutcome::Singular
        );
        assert_eq!(
            run_solve(&[2.0, 0.0, 0.0, 2.0], &[], 2, 0).unwrap(),
            LuSolveOutcome::Solved(vec![])
        );
        assert_eq!(
            run_solve(&[], &[], 0, 3).unwrap(),
            LuSolveOutcome::Solved(vec![])
        );
    }

    #[test]
    fn invalid_dims_are_rejected_before_device() {
        assert!(matches!(
            run_solve(&[1.0; 3], &[1.0; 2], 2, 1),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        assert!(matches!(
            run_cholesky(&[1.0; 3], 2),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
        assert!(matches!(
            run_qr(&[1.0; 5], 2, 3),
            Err(RocmError::InvalidLinalgDims { .. })
        ));
    }

    #[test]
    fn solve_perm_range_violation_sets_solve_err() {
        // perm を壊す（REQ-8: カーネル側の値域検査）。x へは書かれず solve_err が 1 になる。
        let lu_m = [2.0, 0.0, 0.0, 2.0];
        for bad in [[0, 2], [-1, 1]] {
            let (x, err) = lu_solve(&lu_m, &bad, &[0, 0], &[1.0, 1.0], 2, 1);
            assert_eq!(err, 1);
            assert!(x.iter().all(|v| v.is_nan()), "違反時は x へ書かない");
            assert!(matches!(
                solve_err_to_result(err),
                Err(RocmError::LinalgKernelStatusInvalid { .. })
            ));
        }
        // 特異（status[0] != 0）は何も書かず solve_err も 0。
        let (x, err) = lu_solve(&lu_m, &[0, 1], &[1, 0], &[1.0, 1.0], 2, 1);
        assert_eq!(err, 0);
        assert!(x.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn lu_initial_sentinel_is_overwritten() {
        let mut a = vec![4.0, 1.0, 1.0, 3.0];
        let (perm, status) = lu(&mut a, 2);
        assert_eq!(perm, [0, 1]);
        assert_eq!(status, [0, 0]); // sentinel(-1) のまま残らない
    }

    #[test]
    fn cholesky_matches_cpu_and_rejects_non_spd() {
        let cpu = CpuBackendOps::new();
        for &n in &LU_SIZES {
            let a = spd(n, 60 + n as u64);
            let l = run_cholesky(&a, n).unwrap();
            check(
                &format!("cholesky n={n}"),
                &l,
                &v(&cpu.linalg_cholesky(&t(&a, &[n, n])).unwrap()),
            );
        }
        for a in [
            vec![1.0f32, 2.0, 2.0, 1.0],   // 非正定値
            vec![0.0f32; 4],               // 対角 0
            vec![f32::NAN, 0.0, 0.0, 1.0], // 非有限
        ] {
            assert!(matches!(
                run_cholesky(&a, 2),
                Err(RocmError::LinalgInvalidArgument { .. })
            ));
            assert!(cpu.linalg_cholesky(&t(&a, &[2, 2])).is_err());
        }
    }

    #[test]
    fn qr_matches_cpu_for_all_aspect_ratios() {
        let cpu = CpuBackendOps::new();
        for (m, n) in [
            (1, 1),
            (5, 3),
            (3, 5),
            (17, 17),
            (64, 64),
            (65, 17),
            (17, 65),
            (257, 9),
            (9, 257),
        ] {
            let a = rand_vec(70 + (m * 31 + n) as u64, m * n);
            let (r, q) = run_qr(&a, m, n).unwrap();
            let want = cpu.linalg_qr(&t(&a, &[m, n])).unwrap();
            check(&format!("qr R {m}x{n}"), &r, &v(&want.r));
            check(&format!("qr Q {m}x{n}"), &q, &v(&want.q));
        }
        // 零列を含む（反射が無効になる経路。refl_valid=0）。
        let mut a = rand_vec(99, 6 * 4);
        for i in 0..6 {
            a[i * 4 + 1] = 0.0;
        }
        let (r, q) = run_qr(&a, 6, 4).unwrap();
        let want = cpu.linalg_qr(&t(&a, &[6, 4])).unwrap();
        check("qr zero col R", &r, &v(&want.r));
        check("qr zero col Q", &q, &v(&want.q));
        let (_, _, valid) = qr(&widen(&a), 6, 4);
        assert!(valid.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    fn eigh_matches_cpu() {
        let cpu = CpuBackendOps::new();
        for &n in &[1usize, 2, 3, 17, 64, 65] {
            let a = sym(n, 80 + n as u64);
            let (ev, vv) = run_eigh(&a, n).unwrap();
            let want = cpu.linalg_eigh(&t(&a, &[n, n])).unwrap();
            check(&format!("eigh evals n={n}"), &ev, &v(&want.eigenvalues));
            check(&format!("eigh vecs n={n}"), &vv, &v(&want.eigenvectors));
        }
        // 既に対角（回転なし・スキップ経路）と重複固有値。
        let a = [3.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (ev, vv) = run_eigh(&a, 3).unwrap();
        let want = cpu.linalg_eigh(&t(&a, &[3, 3])).unwrap();
        check("eigh diag evals", &ev, &v(&want.eigenvalues));
        check("eigh diag vecs", &vv, &v(&want.eigenvectors));
    }

    #[test]
    fn eigh_non_convergence_is_invalid_argument() {
        let a = [f32::NAN, 1.0, 1.0, 2.0];
        assert!(matches!(
            run_eigh(&a, 2),
            Err(RocmError::LinalgInvalidArgument { .. })
        ));
    }

    #[test]
    fn model_outputs_are_fully_written() {
        // 未初期化相当（NaN）のまま残る要素がない。
        let n = 65;
        let (l, s) = cholesky(&widen(&spd(n, 1)), n);
        assert_eq!(s, 0);
        assert!(l.iter().all(|x| x.is_finite()));
        let (_, v, s) = eigh_kernel(&widen(&sym(n, 2)), n);
        assert_eq!(s, 0);
        assert!(v.iter().all(|x| x.is_finite()));
        let (_, q, _) = qr(&widen(&rand_vec(3, 9 * 5)), 9, 5);
        assert!(q.iter().all(|x| x.is_finite()));
    }
}
