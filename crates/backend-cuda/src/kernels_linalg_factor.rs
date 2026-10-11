//! 線形代数の分解系 3 演算（Cholesky・reduced Householder QR・対称固有値分解
//! の巡回 Jacobi 法）の CUDA C カーネルソース（NVRTC 実行時コンパイル用の
//! 静的文字列。イシュー #2675・親イシュー #2672・設計判断記録
//! `docs/backend-gpu-linalg-decision.md` §3〜§6）。
//!
//! `linalg.rs::CudaLinalgFactor::new`（呼び出し元）が本モジュールの 3 定数を
//! `nvrtc::compile_ptx` に渡し `CudaFunction` を得る。他の `kernels_*.rs` と
//! 同じ理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む
//! （ビルド時に CUDA ヘッダを要求しない。`.claude/rules/deps-policy.md`）。
//!
//! # 意味論の正
//!
//! `crates/backend-cpu/src/linalg.rs::{cholesky, qr, eigh_jacobi, eigh}`
//! （`f64` 逐次実装）。**各出力要素の算術順序を CPU の逐次ループと同一に
//! 保つ**（設計記録 §5 が「baseline 方式不要・統一複合判定を厳密適用」と
//! する前提）。縮約（二乗和・内積・収束判定の `off_sq`）は並列リダクション
//! にせず thread 0 または要素担当スレッドが添字昇順で逐次加算する。
//!
//! # FMA 契約（設計記録 §4）
//!
//! CPU の線形代数参照実装は `mul_add` を使わない。NVRTC は `--fmad` 既定 ON
//! のため、全ての `double` 四則・平方根を非縮約 intrinsic（`__dmul_rn`／
//! `__dadd_rn`／`__dsub_rn`／`__ddiv_rn`／`__dsqrt_rn`）で書き暗黙の FMA
//! 縮約を避ける（先例 `kernels_linalg.rs`・`kernels_adam.rs`）。
//!
//! # 構成と REQ-8
//!
//! 1 行列 = 1 thread block（grid `(1,1,1)`）で block 間同期を要しない。
//! 分岐の判定値（失敗・収束・スキップ・反射の有無）は shared memory で
//! ブロードキャストし、`__syncthreads` の後に全スレッドが同じ値を読んで
//! 一様に `return`／分岐する（発散した同期を作らない）。全ループ・スレッド
//! 添字に手動境界チェックを維持する（`.claude/rules/coding-rust.md`
//! カーネル実装の境界検査）。
//!
//! 出力の符号正規化・昇順ソート・`f32` 化はホスト側（`linalg.rs`）で行う。

/// 3 カーネル共通の block 次元（2 のべき乗）。ソース内の `*_BLOCK` と一致させる。
pub(crate) const LINALG_FACTOR_BLOCK_DIM: u32 = 256;

/// Cholesky 分解 `A = L Lᵀ`（下三角のみ読む）。`a`: `n*n`、`l`: `n*n`
/// （出力。上三角も 0 を明示書き込み）、`status[0]`: 非正定値で 1。
pub(crate) const LINALG_CHOLESKY_F64: &str = r#"
#define CHOL_BLOCK 256

extern "C" __global__ void linalg_cholesky_f64(
    const double* a, double* l, int* status, int n) {
    __shared__ int s_fail;
    __shared__ double s_diag;

    int tid = threadIdx.x;
    int total = n * n;
    if (tid == 0) {
        status[0] = 0;
        s_fail = 0;
        s_diag = 0.0;
    }
    for (int idx = tid; idx < total; idx += CHOL_BLOCK) {
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
                sum = __dsub_rn(sum, __dmul_rn(ljk, ljk));
            }
            if (!isfinite(sum) || sum <= 0.0) {
                s_fail = 1;
                status[0] = 1;
            } else {
                double d = __dsqrt_rn(sum);
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
                    sum = __dsub_rn(sum, __dmul_rn(l[i * n + k], l[j * n + k]));
                }
                l[i * n + j] = __ddiv_rn(sum, diag);
            }
        }
        __syncthreads();
    }
}
"#;

/// reduced Householder QR。`r`: 入力 `A` の作業コピー `m*n`（in-place で
/// 上三角化）、`q`: `m*k`（出力。`k = min(m,n)`）、`refl`: `k*m`（列 `col` の
/// 反射ベクトルを `refl[col*m ..]` に先頭詰めで格納）、`refl_valid`: `k`。
pub(crate) const LINALG_QR_F64: &str = r#"
#define QR_BLOCK 256

// ストライドループ変数は long long とする。int だと最終要素の後の `+= QR_BLOCK` が
// INT_MAX を超えて符号付きオーバーフロー（UB）になる（validate_qr_dims が
// (i32::MAX, 1) 等を許可するため。REQ-8 境界保護）。
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
                norm_x = __dadd_rn(norm_x, __dmul_rn(x, x));
            }
            norm_x = __dsqrt_rn(norm_x);
            int valid = 0;
            double norm_v = 0.0;
            if (norm_x != 0.0) {
                double x0 = r[col * n + col];
                double alpha = x0 >= 0.0 ? -norm_x : norm_x;
                refl[col * m] = __dsub_rn(x0, alpha);
                for (int i = 1; i < m - col; ++i) {
                    refl[col * m + i] = r[(col + i) * n + col];
                }
                double acc = 0.0;
                for (int i = 0; i < m - col; ++i) {
                    double e = refl[col * m + i];
                    acc = __dadd_rn(acc, __dmul_rn(e, e));
                }
                norm_v = __dsqrt_rn(acc);
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
                    refl[col * m + i] = __ddiv_rn(refl[col * m + i], norm_v);
                }
            }
            __syncthreads();
            for (long long c = col + tid; c < n; c += QR_BLOCK) {
                if (c < n) {
                    double dot = 0.0;
                    for (int i = 0; i < m - col; ++i) {
                        dot = __dadd_rn(dot, __dmul_rn(refl[col * m + i], r[(col + i) * n + c]));
                    }
                    if (dot != 0.0) {
                        for (int i = 0; i < m - col; ++i) {
                            double vi = refl[col * m + i];
                            double cur = r[(col + i) * n + c];
                            r[(col + i) * n + c] =
                                __dsub_rn(cur, __dmul_rn(__dmul_rn(2.0, vi), dot));
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
                        dot = __dadd_rn(dot, __dmul_rn(refl[col * m + i], q[(col + i) * k + c]));
                    }
                    if (dot != 0.0) {
                        for (int i = 0; i < m - col; ++i) {
                            double vi = refl[col * m + i];
                            double cur = q[(col + i) * k + c];
                            q[(col + i) * k + c] =
                                __dsub_rn(cur, __dmul_rn(__dmul_rn(2.0, vi), dot));
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}
"#;

/// 対称固有値分解（巡回 Jacobi 法・下三角を読む）。`a`: `n*n`、`sym`: `n*n`
/// 作業行列、`v`: `n*n` 固有ベクトル（列）、`evals`: `n`（未ソート）、
/// `status[0]`: 60 スイープで未収束なら 1。
pub(crate) const LINALG_EIGH_F64: &str = r#"
#define EIGH_BLOCK 256
#define EIGH_MAX_SWEEPS 60

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
    int total = n * n;
    if (tid == 0) {
        status[0] = 0;
    }
    for (int idx = tid; idx < total; idx += EIGH_BLOCK) {
        if (idx < total) {
            int i = idx / n;
            int j = idx % n;
            sym[idx] = (i >= j) ? a[i * n + j] : a[j * n + i];
            v[idx] = (i == j) ? 1.0 : 0.0;
        }
    }
    __syncthreads();
    if (tid == 0) {
        double acc = 0.0;
        for (int idx = 0; idx < total; ++idx) {
            double x = sym[idx];
            acc = __dadd_rn(acc, __dmul_rn(x, x));
        }
        s_frob = __dsqrt_rn(acc);
    }
    __syncthreads();

    for (int sweep = 0; sweep < EIGH_MAX_SWEEPS; ++sweep) {
        if (tid == 0) {
            double off_sq = 0.0;
            for (int p = 0; p < n; ++p) {
                for (int q = p + 1; q < n; ++q) {
                    double x = sym[p * n + q];
                    off_sq = __dadd_rn(off_sq, __dmul_rn(x, x));
                }
            }
            s_conv = (__dsqrt_rn(__dmul_rn(2.0, off_sq)) <= __dmul_rn(1e-14, s_frob)) ? 1 : 0;
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
                        double theta = __ddiv_rn(__dsub_rn(aqq, app), __dmul_rn(2.0, apq));
                        double t = 1.0;
                        if (theta != 0.0) {
                            double denom = __dadd_rn(
                                fabs(theta),
                                __dsqrt_rn(__dadd_rn(1.0, __dmul_rn(theta, theta))));
                            t = __ddiv_rn(copysign(1.0, theta), denom);
                        }
                        double c = __ddiv_rn(1.0, __dsqrt_rn(__dadd_rn(1.0, __dmul_rn(t, t))));
                        s_c = c;
                        s_s = __dmul_rn(c, t);
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
                            double new_kp = __dsub_rn(__dmul_rn(c, akp), __dmul_rn(s, akq));
                            double new_kq = __dadd_rn(__dmul_rn(s, akp), __dmul_rn(c, akq));
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
                        double cc = __dmul_rn(c, c);
                        double ss = __dmul_rn(s, s);
                        double cross = __dmul_rn(__dmul_rn(2.0, s), c);
                        double app_new = __dadd_rn(
                            __dsub_rn(__dmul_rn(cc, app), __dmul_rn(cross, apq)),
                            __dmul_rn(ss, aqq));
                        double aqq_new = __dadd_rn(
                            __dadd_rn(__dmul_rn(ss, app), __dmul_rn(cross, apq)),
                            __dmul_rn(cc, aqq));
                        sym[p * n + p] = app_new;
                        sym[q * n + q] = aqq_new;
                        sym[p * n + q] = 0.0;
                        sym[q * n + p] = 0.0;
                    }
                    for (int k = tid; k < n; k += EIGH_BLOCK) {
                        if (k < n) {
                            double vkp = v[k * n + p];
                            double vkq = v[k * n + q];
                            v[k * n + p] = __dsub_rn(__dmul_rn(c, vkp), __dmul_rn(s, vkq));
                            v[k * n + q] = __dadd_rn(__dmul_rn(s, vkp), __dmul_rn(c, vkq));
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

    const SOURCES: [(&str, &str); 3] = [
        ("cholesky", LINALG_CHOLESKY_F64),
        ("qr", LINALG_QR_F64),
        ("eigh", LINALG_EIGH_F64),
    ];

    /// 丸め契約・ヘッダ非依存の後退検出（実 GPU 不要。CI で常時実行）。
    #[test]
    fn sources_use_non_fused_intrinsics() {
        for (name, src) in SOURCES {
            assert!(src.contains("__dmul_rn("), "{name}: __dmul_rn を使うはず");
            assert!(
                src.contains("__dadd_rn(") || src.contains("__dsub_rn("),
                "{name}: 非縮約の加減算を使うはず"
            );
            assert!(!src.contains("fma("), "{name}: fma は使わないはず");
            assert!(!src.contains("#include"), "{name}: ヘッダ非依存のはず");
            assert!(src.contains("__syncthreads()"), "{name}: block 内同期");
            assert!(src.contains("__ddiv_rn("), "{name}: __ddiv_rn を使うはず");
        }
        assert!(LINALG_CHOLESKY_F64.contains("__dsqrt_rn("));
        assert!(LINALG_QR_F64.contains("__dsqrt_rn("));
        assert!(LINALG_EIGH_F64.contains("__dsqrt_rn("));
    }

    /// 手動境界チェック（REQ-8）が維持されていること。
    #[test]
    fn sources_keep_manual_bounds_checks() {
        assert!(LINALG_CHOLESKY_F64.contains("if (idx < total)"));
        assert!(LINALG_CHOLESKY_F64.contains("if (i < n)"));
        assert!(LINALG_QR_F64.contains("if (i < m - col)"));
        assert!(LINALG_QR_F64.contains("if (c < n)"));
        assert!(LINALG_QR_F64.contains("if (c < k)"));
        assert!(LINALG_EIGH_F64.contains("if (idx < total)"));
        assert!(LINALG_EIGH_F64.contains("if (k < n && k != p && k != q)"));
    }

    #[test]
    fn block_constants_match_source() {
        assert_eq!(LINALG_FACTOR_BLOCK_DIM, 256);
        assert!(LINALG_FACTOR_BLOCK_DIM.is_power_of_two());
        assert!(LINALG_CHOLESKY_F64.contains("#define CHOL_BLOCK 256"));
        assert!(LINALG_QR_F64.contains("#define QR_BLOCK 256"));
        assert!(LINALG_EIGH_F64.contains("#define EIGH_BLOCK 256"));
        assert!(LINALG_EIGH_F64.contains("#define EIGH_MAX_SWEEPS 60"));
    }

    /// 収束判定の縮約を並列リダクションにしない（停止判定が CPU とずれない）。
    #[test]
    fn eigh_convergence_check_is_single_thread() {
        let src = LINALG_EIGH_F64;
        let conv = src.find("s_conv = ").expect("s_conv 代入");
        let guard = src[..conv].rfind("if (tid == 0)").expect("tid==0 ガード");
        assert!(conv - guard < 600, "収束判定は thread 0 の逐次ループ内");
    }
}
