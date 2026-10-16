//! 線形代数（部分ピボット LU 分解・前進／後退代入）の CUDA C カーネル
//! ソース（NVRTC 実行時コンパイル用の静的文字列。イシュー #2674・親
//! イシュー #2672・設計判断記録 `docs/backend-gpu-linalg-decision.md`
//! §3〜§6）。
//!
//! `linalg.rs::CudaLinalg::new`（呼び出し元）が本モジュールの 2 定数を
//! `nvrtc::compile_ptx` に渡し `CudaFunction` を得る。他の `kernels_*.rs`
//! と同じ理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む
//! （ビルド時に CUDA ヘッダを要求しない。`.claude/rules/deps-policy.md`）。
//!
//! # 意味論の正
//!
//! `crates/backend-cpu/src/linalg.rs::{lu_decompose, forward_substitute_unit,
//! backward_substitute}`（`f64` 逐次実装）。ピボット選択は「列 k の行 k..n
//! で `|a|` 厳密最大・同値は最小添字・NaN は選ばれない」、ピボットが厳密
//! `0.0` なら特異、`factor == 0.0` の行は更新を飛ばす（`x - 0*inf` が NaN
//! になるのを避けて CPU と一致させるため。性能目的ではない）。
//!
//! # FMA 契約（設計記録 §4）
//!
//! CPU の線形代数参照実装は `mul_add` を使わず乗算と減算を分ける。NVRTC は
//! `--fmad` 既定 ON のため、全ての `double` 四則演算を非縮約 intrinsic
//! （`__dmul_rn`／`__dsub_rn`／`__ddiv_rn`）で書き暗黙の FMA 縮約を避ける
//! （先例 `kernels_adam.rs`）。目標は CPU との bit 一致、受け入れ判定は
//! REQ-2 統一複合判定。
//!
//! # 構成と REQ-8
//!
//! LU は 1 行列 = 1 thread block（`__syncthreads` 以外の block 間同期を
//! 要しない）。代入は 1 スレッド = 右辺 1 列。全ループ・スレッド添字に手動
//! 境界チェックを維持する（`.claude/rules/coding-rust.md` カーネル実装の
//! 境界検査）。

/// LU カーネルの block 次元（2 のべき乗であること。shared memory の
/// 木構造縮約が前提とする）。ソース内の `LU_BLOCK` と一致させる。
pub(crate) const LINALG_LU_BLOCK_DIM: u32 = 256;

/// 部分ピボット LU 分解（in-place・単一 block）。`a`: `n*n` 行優先、
/// `perm`: `n`、`status`: `[singular_flag, swap_count]`。
pub(crate) const LINALG_LU_F64: &str = r#"
#define LU_BLOCK 256

extern "C" __global__ void linalg_lu_f64(double* a, int* perm, int* status, int n) {
    __shared__ double s_val[LU_BLOCK];
    __shared__ int s_idx[LU_BLOCK];
    __shared__ int s_pivot_row;

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += LU_BLOCK) {
        if (i < n) {
            perm[i] = i;
        }
    }
    if (tid == 0) {
        status[0] = 0;
        status[1] = 0;
    }
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        // ピボット探索: 各スレッドは行 k+1.. を strided に走査し、厳密 > の
        // 最初の最大（スレッド内で最小添字）を保持する。NaN は選ばれない。
        double best_v = -1.0;
        int best_i = n;
        for (int i = k + 1 + tid; i < n; i += LU_BLOCK) {
            if (i < n) {
                double v = fabs(a[i * n + k]);
                if (v > best_v) {
                    best_v = v;
                    best_i = i;
                }
            }
        }
        s_val[tid] = best_v;
        s_idx[tid] = best_i;
        __syncthreads();
        for (int stride = LU_BLOCK / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                double ov = s_val[tid + stride];
                int oi = s_idx[tid + stride];
                if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                    s_val[tid] = ov;
                    s_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            // CPU: 初期値は行 k の |a[k][k]|、厳密に大きい行のみ置換。
            // |a[k][k]| が NaN なら比較は常に偽で行 k のまま。
            double v0 = fabs(a[k * n + k]);
            int p = k;
            if (s_idx[0] < n && s_val[0] > v0) {
                p = s_idx[0];
            }
            s_pivot_row = p;
        }
        __syncthreads();

        int p = s_pivot_row;
        if (p != k) {
            for (int c = tid; c < n; c += LU_BLOCK) {
                if (c < n) {
                    double t = a[k * n + c];
                    a[k * n + c] = a[p * n + c];
                    a[p * n + c] = t;
                }
            }
            if (tid == 0) {
                int tp = perm[k];
                perm[k] = perm[p];
                perm[p] = tp;
                status[1] = status[1] + 1;
            }
        }
        __syncthreads();

        double pivot = a[k * n + k];
        if (pivot == 0.0) {
            if (tid == 0) {
                status[0] = 1;
            }
            return;
        }

        for (int i = k + 1 + tid; i < n; i += LU_BLOCK) {
            if (i < n) {
                a[i * n + k] = __ddiv_rn(a[i * n + k], pivot);
            }
        }
        __syncthreads();

        int m = n - k - 1;
        long long total = (long long)m * (long long)m;
        for (long long e = tid; e < total; e += LU_BLOCK) {
            int i = k + 1 + (int)(e / m);
            int c = k + 1 + (int)(e % m);
            if (i < n && c < n) {
                double f = a[i * n + k];
                if (f != 0.0) {
                    a[i * n + c] = __dsub_rn(a[i * n + c], __dmul_rn(f, a[k * n + c]));
                }
            }
        }
        __syncthreads();
    }
}
"#;

/// 前進・後退代入の block 次元（1 スレッド = 右辺 1 列）。
pub(crate) const LINALG_SOLVE_BLOCK_DIM: u32 = 128;

/// LU 済み `lu` と `perm` で `A X = B` を解く。`b`: `n*k`（行優先）、
/// `x`: `n*k`（出力・前進代入の中間値 y も保持）。`status[0] != 0`
/// （特異）なら何も書かない。`perm[i]` が `[0, n)` の外なら（REQ-8: 値域を
/// カーネル側でも検査する）`b` を読む前に `atomicExch(solve_err, 1)`（複数列スレッドが同一要素へ書くため atomic。#3401 P2）で通知して何も書かず
/// return し、ホストが読み戻して型付きエラーにする（黙って成功扱いにしない）。
pub(crate) const LINALG_LU_SOLVE_F64: &str = r#"
extern "C" __global__ void linalg_lu_solve_f64(
    const double* lu, const int* perm, const int* status,
    const double* b, double* x, int* solve_err, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= k) {
        return;
    }
    if (status[0] != 0) {
        return;
    }
    // 置換添字の値域検査（x・b への書き込み・読み出しより前に全 i を検査する）。
    for (int i = 0; i < n; ++i) {
        int pi = perm[i];
        if (pi < 0 || pi >= n) {
            atomicExch(solve_err, 1);
            return;
        }
    }
    // 前進代入（単位下三角）: y[i] = b[perm[i]] - sum_{j<i} L[i][j] * y[j]
    for (int i = 0; i < n; ++i) {
        double sum = b[perm[i] * k + col];
        for (int j = 0; j < i; ++j) {
            sum = __dsub_rn(sum, __dmul_rn(lu[i * n + j], x[j * k + col]));
        }
        x[i * k + col] = sum;
    }
    // 後退代入（上三角）: x[i] = (y[i] - sum_{j>i} U[i][j] * x[j]) / U[i][i]
    for (int i = n - 1; i >= 0; --i) {
        double sum = x[i * k + col];
        for (int j = i + 1; j < n; ++j) {
            sum = __dsub_rn(sum, __dmul_rn(lu[i * n + j], x[j * k + col]));
        }
        x[i * k + col] = __ddiv_rn(sum, lu[i * n + i]);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// 丸め契約・境界検査・ヘッダ非依存の後退検出（実 GPU 不要。CI で常時実行）。
    #[test]
    fn sources_use_non_fused_intrinsics_and_keep_bounds_checks() {
        for (name, src) in [("lu", LINALG_LU_F64), ("solve", LINALG_LU_SOLVE_F64)] {
            assert!(src.contains("__dmul_rn("), "{name}: __dmul_rn を使うはず");
            assert!(src.contains("__dsub_rn("), "{name}: __dsub_rn を使うはず");
            assert!(src.contains("__ddiv_rn("), "{name}: __ddiv_rn を使うはず");
            assert!(!src.contains("fma("), "{name}: fma は使わないはず");
            assert!(!src.contains("#include"), "{name}: ヘッダ非依存のはず");
        }
        assert!(LINALG_LU_F64.contains("if (i < n && c < n)"));
        assert!(LINALG_LU_F64.contains("if (c < n)"));
        assert!(LINALG_LU_SOLVE_F64.contains("if (col >= k)"));
        assert!(LINALG_LU_SOLVE_F64.contains("if (pi < 0 || pi >= n)"));
        assert!(LINALG_LU_SOLVE_F64.contains("atomicExch(solve_err, 1);"));
        // perm の値域検査は b の読み出し（b[perm[i] * k）より前にある。
        let chk = LINALG_LU_SOLVE_F64.find("pi >= n").unwrap();
        let rd = LINALG_LU_SOLVE_F64.find("b[perm[i] * k").unwrap();
        assert!(chk < rd);
    }

    #[test]
    fn lu_block_constant_matches_source() {
        assert_eq!(LINALG_LU_BLOCK_DIM, 256);
        assert!(LINALG_LU_F64.contains("#define LU_BLOCK 256"));
        assert!(LINALG_LU_BLOCK_DIM.is_power_of_two());
        const { assert!(LINALG_SOLVE_BLOCK_DIM > 0) };
    }
}
