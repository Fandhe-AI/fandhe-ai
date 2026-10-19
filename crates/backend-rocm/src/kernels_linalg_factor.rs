//! 線形代数の分解系 3 演算（Cholesky・reduced Householder QR・対称固有値分解の巡回 Jacobi 法）の
//! HIP カーネルソース生成（`backend-cuda::kernels_linalg_factor` の移植。イシュー #3444・
//! 親 #3150・設計判断記録 `docs/backend-gpu-linalg-decision.md` §3〜§6・§10）。
//!
//! 役割: `linalg.rs::RocmLinalgFactor::new` が [`render_linalg_factor_source`] を呼び、得た
//! [`LinalgSource`](crate::kernels_linalg::LinalgSource) を HIPRTC へ渡す（3 カーネルを 1 module に
//! まとめる）。LU 系は `kernels_linalg.rs`。
//!
//! # 意味論の正
//!
//! `crates/backend-cpu/src/linalg.rs::{cholesky, qr, eigh_jacobi, eigh}`（`f64` 逐次実装）。
//! **各出力要素の算術順序を CPU の逐次ループと同一に保つ**。縮約（二乗和・内積・収束判定の
//! `off_sq`）は並列リダクションにせず thread 0 または要素担当スレッドが添字昇順で逐次加算する
//! （`f64` アキュムレータ契約は全経路 `double` のため追加の読み替え不要。
//! `.claude/rules/coding-rust.md`）。
//!
//! # FMA 契約
//!
//! `kernels_linalg.rs` と同じ: 先頭の `#pragma clang fp contract(off)` と素の演算子・`sqrt`
//! （double）。`fma(`・`__builtin_fma*` は使わない。CUDA 版との意図的な差分は `__d*_rn` を使わない
//! 点と、cholesky・eigh の `total = n*n` を走るストライドループ変数を `long long` にした点（QR が
//! 既にそうしている理由と同じ。`idx += BLOCK` が `INT_MAX` を超える符号付きオーバーフロー〈UB〉を避ける。
//! 数値には影響しない）。
//!
//! # 構成と REQ-8・wave64
//!
//! 1 行列 = 1 thread block（grid `(1,1,1)`）で block 間同期を要しない。分岐の判定値（失敗・収束・
//! スキップ・反射の有無）は共有メモリでブロードキャストし、`__syncthreads` の後に全スレッドが
//! 同じ値を読んで一様に `return`／分岐する（発散した同期を作らない）。全ループ・スレッド添字に
//! 手動境界チェックを維持する。warp 幅依存の命令は使わず block 次元 256 は 64 の倍数のため
//! wave64 で正しい。出力の符号正規化・昇順ソート・`f32` 化はホスト側（`linalg.rs`）。

use crate::kernels_linalg::LinalgSource;

/// 3 カーネル共通の block 次元（2 のべき乗かつ 64 の倍数）。ソース内の `*_BLOCK` マクロの値。
pub(crate) const LINALG_FACTOR_BLOCK_DIM: u32 = 256;

/// eigh の巡回 Jacobi スイープ上限（CPU 参照実装と同値）。
pub(crate) const LINALG_EIGH_MAX_SWEEPS: u32 = 60;

const _: () = {
    assert!(LINALG_FACTOR_BLOCK_DIM.is_power_of_two());
    assert!(LINALG_FACTOR_BLOCK_DIM.is_multiple_of(64));
};

/// Cholesky カーネルのエントリ名と引数個数（`a, l, status, n`）。
pub(crate) const LINALG_CHOLESKY_KERNEL_NAME: &str = "linalg_cholesky_f64";
pub(crate) const LINALG_CHOLESKY_ARG_COUNT: usize = 4;
/// QR カーネルのエントリ名と引数個数（`r, q, refl, refl_valid, m, n`）。
pub(crate) const LINALG_QR_KERNEL_NAME: &str = "linalg_qr_f64";
pub(crate) const LINALG_QR_ARG_COUNT: usize = 6;
/// eigh カーネルのエントリ名と引数個数（`a, sym, v, evals, status, n`）。
pub(crate) const LINALG_EIGH_KERNEL_NAME: &str = "linalg_eigh_f64";
pub(crate) const LINALG_EIGH_ARG_COUNT: usize = 6;

/// 分解系 3 カーネルを含む HIP ソースを生成する。
pub(crate) fn render_linalg_factor_source() -> LinalgSource {
    LinalgSource::from_template(
        "fandhe_linalg_factor_f64.hip",
        &[
            ("CHOL_BLOCK", LINALG_FACTOR_BLOCK_DIM),
            ("QR_BLOCK", LINALG_FACTOR_BLOCK_DIM),
            ("EIGH_BLOCK", LINALG_FACTOR_BLOCK_DIM),
            ("EIGH_MAX_SWEEPS", LINALG_EIGH_MAX_SWEEPS),
        ],
        LINALG_FACTOR_BODY,
    )
}

/// 3 カーネルのテンプレート本体。
///
/// - Cholesky `A = L Lᵀ`（下三角のみ読む）: `a`・`l`: `n*n`（`l` は上三角も 0 を明示書き込み）、
///   `status[0]`: 非正定値で 1（カーネルが先頭で 0 を書く）。
/// - reduced Householder QR: `r`: 入力 `A` の作業コピー `m*n`（in-place で上三角化）、`q`: `m*k`
///   （`k = min(m,n)`）、`refl`: `k*m`（列 `col` の反射ベクトルを `refl[col*m ..]` に先頭詰め）、
///   `refl_valid`: `k`（各要素 0/1 をカーネルが書く）。
/// - eigh（巡回 Jacobi・下三角を読む）: `sym`・`v`: `n*n`、`evals`: `n`（未ソート）、`status[0]`:
///   60 スイープで未収束なら 1。
const LINALG_FACTOR_BODY: &str = r#"
extern "C" __global__ void linalg_cholesky_f64(
    const double* a, double* l, int* status, int n) {
    __shared__ int s_fail;
    __shared__ double s_diag;

    int tid = threadIdx.x;
    long long total = (long long)n * (long long)n;
    if (tid == 0) {
        status[0] = 0;
        s_fail = 0;
        s_diag = 0.0;
    }
    for (long long idx = tid; idx < total; idx += CHOL_BLOCK) {
        if (idx < total) {
            l[idx] = 0.0;
        }
    }
    __syncthreads();

    for (int j = 0; j < n; ++j) {
        if (tid == 0) {
            double sum = a[j * n + j];
            for (int k = 0; k < j; ++k) {
                double ljk = l[j * n + k];
                sum = sum - ljk * ljk;
            }
            if (!isfinite(sum) || sum <= 0.0) {
                s_fail = 1;
                status[0] = 1;
            } else {
                double d = sqrt(sum);
                l[j * n + j] = d;
                s_diag = d;
            }
        }
        __syncthreads();
        if (s_fail != 0) {
            return;
        }
        double diag = s_diag;
        for (int i = j + 1 + tid; i < n; i += CHOL_BLOCK) {
            if (i < n) {
                double sum = a[i * n + j];
                for (int k = 0; k < j; ++k) {
                    sum = sum - l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = sum / diag;
            }
        }
        __syncthreads();
    }
}

// ストライドループ変数は long long とする。int だと最終要素の後の `+= QR_BLOCK` が
// INT_MAX を超えて符号付きオーバーフロー（UB）になる（REQ-8 境界保護）。
extern "C" __global__ void linalg_qr_f64(
    double* r, double* q, double* refl, int* refl_valid, int m, int n) {
    __shared__ int s_valid;
    __shared__ double s_norm_v;

    int tid = threadIdx.x;
    int k = m < n ? m : n;

    for (int col = 0; col < k; ++col) {
        if (tid == 0) {
            double norm_x = 0.0;
            for (int i = col; i < m; ++i) {
                double x = r[i * n + col];
                norm_x = norm_x + x * x;
            }
            norm_x = sqrt(norm_x);
            int valid = 0;
            double norm_v = 0.0;
            if (norm_x != 0.0) {
                double x0 = r[col * n + col];
                double alpha = x0 >= 0.0 ? -norm_x : norm_x;
                refl[col * m] = x0 - alpha;
                for (int i = 1; i < m - col; ++i) {
                    refl[col * m + i] = r[(col + i) * n + col];
                }
                double acc = 0.0;
                for (int i = 0; i < m - col; ++i) {
                    double e = refl[col * m + i];
                    acc = acc + e * e;
                }
                norm_v = sqrt(acc);
                if (norm_v != 0.0) {
                    valid = 1;
                }
            }
            refl_valid[col] = valid;
            s_valid = valid;
            s_norm_v = norm_v;
        }
        __syncthreads();
        if (s_valid != 0) {
            double norm_v = s_norm_v;
            for (long long i = tid; i < m - col; i += QR_BLOCK) {
                if (i < m - col) {
                    refl[col * m + i] = refl[col * m + i] / norm_v;
                }
            }
            __syncthreads();
            for (long long c = col + tid; c < n; c += QR_BLOCK) {
                if (c < n) {
                    double dot = 0.0;
                    for (int i = 0; i < m - col; ++i) {
                        dot = dot + refl[col * m + i] * r[(col + i) * n + c];
                    }
                    if (dot != 0.0) {
                        for (int i = 0; i < m - col; ++i) {
                            double vi = refl[col * m + i];
                            double cur = r[(col + i) * n + c];
                            r[(col + i) * n + c] = cur - (2.0 * vi) * dot;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Q = E_k（単位行列の先頭 k 列）から出発し、有効な反射を逆順に適用する。
    int qtotal = m * k;
    for (long long idx = tid; idx < qtotal; idx += QR_BLOCK) {
        if (idx < qtotal) {
            q[idx] = 0.0;
        }
    }
    __syncthreads();
    for (long long i = tid; i < k; i += QR_BLOCK) {
        if (i < k) {
            q[i * k + i] = 1.0;
        }
    }
    __syncthreads();
    for (int col = k - 1; col >= 0; --col) {
        if (refl_valid[col] != 0) {
            for (long long c = tid; c < k; c += QR_BLOCK) {
                if (c < k) {
                    double dot = 0.0;
                    for (int i = 0; i < m - col; ++i) {
                        dot = dot + refl[col * m + i] * q[(col + i) * k + c];
                    }
                    if (dot != 0.0) {
                        for (int i = 0; i < m - col; ++i) {
                            double vi = refl[col * m + i];
                            double cur = q[(col + i) * k + c];
                            q[(col + i) * k + c] = cur - (2.0 * vi) * dot;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void linalg_eigh_f64(
    const double* a, double* sym, double* v, double* evals, int* status, int n) {
    __shared__ double s_frob;
    __shared__ int s_conv;
    __shared__ int s_skip;
    __shared__ double s_c;
    __shared__ double s_s;
    __shared__ double s_app;
    __shared__ double s_aqq;
    __shared__ double s_apq;

    int tid = threadIdx.x;
    long long total = (long long)n * (long long)n;
    if (tid == 0) {
        status[0] = 0;
    }
    for (long long idx = tid; idx < total; idx += EIGH_BLOCK) {
        if (idx < total) {
            int i = (int)(idx / n);
            int j = (int)(idx % n);
            sym[idx] = (i >= j) ? a[i * n + j] : a[j * n + i];
            v[idx] = (i == j) ? 1.0 : 0.0;
        }
    }
    __syncthreads();
    if (tid == 0) {
        double acc = 0.0;
        for (long long idx = 0; idx < total; ++idx) {
            double x = sym[idx];
            acc = acc + x * x;
        }
        s_frob = sqrt(acc);
    }
    __syncthreads();

    for (int sweep = 0; sweep < EIGH_MAX_SWEEPS; ++sweep) {
        if (tid == 0) {
            double off_sq = 0.0;
            for (int p = 0; p < n; ++p) {
                for (int q = p + 1; q < n; ++q) {
                    double x = sym[p * n + q];
                    off_sq = off_sq + x * x;
                }
            }
            s_conv = (sqrt(2.0 * off_sq) <= 1e-14 * s_frob) ? 1 : 0;
        }
        __syncthreads();
        if (s_conv != 0) {
            for (int i = tid; i < n; i += EIGH_BLOCK) {
                if (i < n) {
                    evals[i] = sym[i * n + i];
                }
            }
            return;
        }
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (tid == 0) {
                    double apq = sym[p * n + q];
                    if (apq == 0.0) {
                        s_skip = 1;
                    } else {
                        s_skip = 0;
                        double app = sym[p * n + p];
                        double aqq = sym[q * n + q];
                        double theta = (aqq - app) / (2.0 * apq);
                        double t = 1.0;
                        if (theta != 0.0) {
                            double denom = fabs(theta) + sqrt(1.0 + theta * theta);
                            t = copysign(1.0, theta) / denom;
                        }
                        double c = 1.0 / sqrt(1.0 + t * t);
                        s_c = c;
                        s_s = c * t;
                        s_app = app;
                        s_aqq = aqq;
                        s_apq = apq;
                    }
                }
                __syncthreads();
                if (s_skip == 0) {
                    double c = s_c;
                    double s = s_s;
                    for (int k = tid; k < n; k += EIGH_BLOCK) {
                        if (k < n && k != p && k != q) {
                            double akp = sym[k * n + p];
                            double akq = sym[k * n + q];
                            double new_kp = c * akp - s * akq;
                            double new_kq = s * akp + c * akq;
                            sym[k * n + p] = new_kp;
                            sym[p * n + k] = new_kp;
                            sym[k * n + q] = new_kq;
                            sym[q * n + k] = new_kq;
                        }
                    }
                    __syncthreads();
                    if (tid == 0) {
                        double app = s_app;
                        double aqq = s_aqq;
                        double apq = s_apq;
                        double cc = c * c;
                        double ss = s * s;
                        double cross = (2.0 * s) * c;
                        double app_new = (cc * app - cross * apq) + ss * aqq;
                        double aqq_new = (ss * app + cross * apq) + cc * aqq;
                        sym[p * n + p] = app_new;
                        sym[q * n + q] = aqq_new;
                        sym[p * n + q] = 0.0;
                        sym[q * n + p] = 0.0;
                    }
                    for (int k = tid; k < n; k += EIGH_BLOCK) {
                        if (k < n) {
                            double vkp = v[k * n + p];
                            double vkq = v[k * n + q];
                            v[k * n + p] = c * vkp - s * vkq;
                            v[k * n + q] = s * vkp + c * vkq;
                        }
                    }
                    __syncthreads();
                }
                __syncthreads();
            }
        }
    }
    if (tid == 0) {
        status[0] = 1;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn src() -> String {
        render_linalg_factor_source().as_str().to_string()
    }

    /// 先頭の pragma・FMA 非縮約の形・ヘッダ非依存の後退検出（実 GPU 不要。CI で常時実行）。
    #[test]
    fn source_keeps_contract_off() {
        let s = src();
        assert!(s.starts_with("#pragma clang fp contract(off)\n"));
        assert!(!s.contains("fma("));
        assert!(!s.contains("__builtin_fma"));
        assert!(!s.contains("_rn("));
        assert!(!s.contains("#include"));
        assert!(s.contains("__syncthreads()"));
        assert!(s.contains("sqrt("));
    }

    /// 手動境界チェック（REQ-8）が維持されていること。
    #[test]
    fn source_keeps_manual_bounds_checks() {
        let s = src();
        for needle in [
            "if (idx < total)",
            "if (i < n)",
            "if (i < m - col)",
            "if (c < n)",
            "if (c < k)",
            "if (k < n && k != p && k != q)",
            "if (idx < qtotal)",
        ] {
            assert!(s.contains(needle), "{needle}");
        }
    }

    /// `total = n*n` を走るストライドループ変数は `long long`（int の符号付きオーバーフロー回避）。
    #[test]
    fn strided_loops_over_n_squared_use_long_long() {
        let s = src();
        assert!(s.contains("for (long long idx = tid; idx < total; idx += CHOL_BLOCK)"));
        assert!(s.contains("for (long long idx = tid; idx < total; idx += EIGH_BLOCK)"));
        assert!(!s.contains("for (int idx = tid"));
    }

    #[test]
    fn block_constants_match_source() {
        assert_eq!(LINALG_FACTOR_BLOCK_DIM, 256);
        let r = render_linalg_factor_source();
        let defines: Vec<&str> = r
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(
            defines,
            [
                "#define CHOL_BLOCK 256",
                "#define QR_BLOCK 256",
                "#define EIGH_BLOCK 256",
                "#define EIGH_MAX_SWEEPS 60"
            ]
        );
        assert_eq!(r.program_name(), "fandhe_linalg_factor_f64.hip");
    }

    #[test]
    fn entry_names_and_arg_counts_match_source() {
        let s = src();
        for (name, argc) in [
            (LINALG_CHOLESKY_KERNEL_NAME, LINALG_CHOLESKY_ARG_COUNT),
            (LINALG_QR_KERNEL_NAME, LINALG_QR_ARG_COUNT),
            (LINALG_EIGH_KERNEL_NAME, LINALG_EIGH_ARG_COUNT),
        ] {
            let head = format!("extern \"C\" __global__ void {name}(");
            let at = s.find(&head).unwrap_or_else(|| panic!("{name}"));
            let sig = &s[at + head.len()..];
            let sig = &sig[..sig.find(") {").unwrap()];
            assert_eq!(sig.split(',').count(), argc, "{name}");
        }
    }

    /// 収束判定の縮約を並列リダクションにしない（停止判定が CPU とずれない）。
    #[test]
    fn eigh_convergence_check_is_single_thread() {
        let s = src();
        let conv = s.find("s_conv = ").expect("s_conv 代入");
        let guard = s[..conv].rfind("if (tid == 0)").expect("tid==0 ガード");
        assert!(conv - guard < 600, "収束判定は thread 0 の逐次ループ内");
    }
}
