//! 線形代数（部分ピボット LU 分解・前進／後退代入）の HIP カーネルソース生成
//! （`backend-cuda::kernels_linalg` の移植。イシュー #3444・親 #3150・設計判断記録
//! `docs/backend-gpu-linalg-decision.md` §4・§10）。
//!
//! 役割: HIPRTC へ渡すソース文字列の唯一の生成元（LU 系）。静的テンプレートに、`const u32`
//! の block 次元だけを `#define` として前置する（文字列入力は受けない。A03）。
//! `linalg.rs::RocmLinalgLu::new` が [`render_linalg_lu_source`] を呼び、`hiprtc.rs` が
//! [`LinalgSource`]（本モジュールが定義する検証済み newtype）だけを受け取る。
//! 分解系 3 カーネルは `kernels_linalg_factor.rs`（同じ [`LinalgSource`] を使う）。
//!
//! # 意味論の正
//!
//! `crates/backend-cpu/src/linalg.rs::{lu_decompose, forward_substitute_unit,
//! backward_substitute}`（`f64` 逐次実装）。ピボットは「列 k の行 k..n で `|a|` 厳密最大・
//! 同値は最小添字・NaN は選ばれない」、ピボットが厳密 `0.0` なら特異、`factor == 0.0` の行は
//! 更新を飛ばす（`x - 0*inf` の NaN を避けて CPU と一致させるため）。
//!
//! # FMA 契約（CUDA 版との差分）
//!
//! CPU の線形代数参照実装は `mul_add` を使わない。CUDA 版は `--fmad` 既定 ON のため
//! `__dmul_rn` 等の非縮約 intrinsic で書いたが、HIP-Clang の contraction 既定は異なり、
//! `__d*_rn` が HIPRTC の暗黙ヘッダ内の通常演算子として定義されていると、ソース先頭の pragma は
//! その内部の演算に効かず縮約の保証が取れない。そこで本ソースは安全側の形（Metal の linalg と
//! 同じ先例）を採る: 先頭に `#pragma clang fp contract(off)` を置き、演算は素の `*`・`+`・`-`・`/`
//! と正しく丸める `sqrt`（double）で書く。`fma(`・`__builtin_fma*` は使わない
//! （matmul 系の FMA 契約〈`__builtin_fmaf`〉は線形代数には適用しない別の軸。
//! `.claude/rules/coding-rust.md`）。実命令（`v_fma_f64` の不在）の確認は実機・オフライン
//! コンパイル診断（#3159 段 B）へ申し送る。
//!
//! # 構成・REQ-8・wave64
//!
//! LU は 1 行列 = 1 thread block（`__syncthreads` 以外の block 間同期を要しない）。代入は
//! 1 スレッド = 右辺 1 列。全ループ・スレッド添字に手動境界チェックを維持する
//! （`.claude/rules/coding-rust.md`）。warp 幅依存の命令は使わず（共有メモリの木構造縮約と
//! `__syncthreads` のみ）、block 次元は 64 の倍数かつ 2 のべき乗（const assert で固定）のため
//! wave64（MI300X）でそのまま正しい。

/// LU カーネルの block 次元（2 のべき乗かつ 64 の倍数。共有メモリの木構造縮約が前提）。
pub(crate) const LINALG_LU_BLOCK_DIM: u32 = 256;

/// 前進・後退代入の block 次元（1 スレッド = 右辺 1 列。64 の倍数）。
pub(crate) const LINALG_SOLVE_BLOCK_DIM: u32 = 128;

const _: () = {
    assert!(LINALG_LU_BLOCK_DIM.is_power_of_two());
    assert!(LINALG_LU_BLOCK_DIM.is_multiple_of(64));
    assert!(LINALG_SOLVE_BLOCK_DIM.is_multiple_of(64));
};

/// LU カーネルのエントリ名。
pub(crate) const LINALG_LU_KERNEL_NAME: &str = "linalg_lu_f64";
/// LU カーネルの引数個数（`a, perm, status, n`）。
pub(crate) const LINALG_LU_ARG_COUNT: usize = 4;
/// 代入カーネルのエントリ名。
pub(crate) const LINALG_SOLVE_KERNEL_NAME: &str = "linalg_lu_solve_f64";
/// 代入カーネルの引数個数（`lu, perm, status, b, x, solve_err, n, k`）。
pub(crate) const LINALG_SOLVE_ARG_COUNT: usize = 8;

/// HIPRTC へ渡せる線形代数カーネルソース（検証済み newtype）。
///
/// 構築できるのは [`render_linalg_lu_source`]・`kernels_linalg_factor::render_linalg_factor_source`
/// （[`LinalgSource::from_template`] 経由）だけで、いずれも静的テンプレートと `const u32` の数値だけから
/// 作る。crate 外からは構築できない（A03）。
pub(crate) struct LinalgSource {
    text: String,
    program: &'static str,
}

impl LinalgSource {
    /// 静的テンプレートの本体 `body` に、数値だけの `#define` 群と FMA 非縮約 pragma を前置する。
    ///
    /// `defines` は `(マクロ名, 値)` の列で、マクロ名は `&'static str` の固定リテラル・値は `u32`
    /// のみのため、文字列がソースへ流れ込む経路はない。
    pub(crate) fn from_template(
        program: &'static str,
        defines: &[(&'static str, u32)],
        body: &'static str,
    ) -> Self {
        let mut text = String::from("#pragma clang fp contract(off)\n");
        for (name, value) in defines {
            text.push_str(&format!("#define {name} {value}\n"));
        }
        text.push_str(body);
        Self { text, program }
    }

    /// ソース文字列。
    pub(crate) fn as_str(&self) -> &str {
        &self.text
    }

    /// `hiprtcCreateProgram` へ渡す固定のプログラム名。
    pub(crate) fn program_name(&self) -> &'static str {
        self.program
    }
}

/// LU 系 2 カーネル（`linalg_lu_f64`・`linalg_lu_solve_f64`）を含む HIP ソースを生成する。
pub(crate) fn render_linalg_lu_source() -> LinalgSource {
    LinalgSource::from_template(
        "fandhe_linalg_lu_f64.hip",
        &[("LU_BLOCK", LINALG_LU_BLOCK_DIM)],
        LINALG_LU_BODY,
    )
}

/// 2 カーネルのテンプレート本体。
///
/// LU: 部分ピボット LU 分解（in-place・単一 block）。`a`: `n*n` 行優先、`perm`: `n`、
/// `status`: `[singular_flag, swap_count]`（カーネルが先頭で 0 を書く）。
///
/// 代入: LU 済み `lu` と `perm` で `A X = B` を解く。`b`・`x`: `n*k`（`x` は前進代入の中間値 y も
/// 保持）。`status[0] != 0`（特異）なら何も書かない。`perm[i]` が `[0, n)` の外なら（REQ-8:
/// 値域をカーネル側でも検査する）`b` を読む前に `atomicExch(solve_err, 1)`（複数列スレッドが
/// 同一要素へ書くため atomic）で通知して何も書かず return し、ホストが読み戻して型付きエラーに
/// する（黙って成功扱いにしない）。
const LINALG_LU_BODY: &str = r#"
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
                a[i * n + k] = a[i * n + k] / pivot;
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
                    a[i * n + c] = a[i * n + c] - f * a[k * n + c];
                }
            }
        }
        __syncthreads();
    }
}

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
            sum = sum - lu[i * n + j] * x[j * k + col];
        }
        x[i * k + col] = sum;
    }
    // 後退代入（上三角）: x[i] = (y[i] - sum_{j>i} U[i][j] * x[j]) / U[i][i]
    for (int i = n - 1; i >= 0; --i) {
        double sum = x[i * k + col];
        for (int j = i + 1; j < n; ++j) {
            sum = sum - lu[i * n + j] * x[j * k + col];
        }
        x[i * k + col] = sum / lu[i * n + i];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// 先頭の pragma・FMA 非縮約の形・ヘッダ非依存・境界検査の後退検出（実 GPU 不要）。
    #[test]
    fn source_keeps_contract_off_and_bounds_checks() {
        let s = render_linalg_lu_source();
        let src = s.as_str();
        assert!(src.starts_with("#pragma clang fp contract(off)\n"));
        assert!(!src.contains("fma("), "fma は使わないはず");
        assert!(!src.contains("__builtin_fma"), "fma は使わないはず");
        assert!(!src.contains("_rn("), "素の演算子で書く");
        assert!(!src.contains("#include"), "ヘッダ非依存のはず");
        assert!(src.contains("if (i < n && c < n)"));
        assert!(src.contains("if (c < n)"));
        assert!(src.contains("if (i < n)"));
        assert!(src.contains("if (col >= k)"));
        assert!(src.contains("if (pi < 0 || pi >= n)"));
        assert!(src.contains("atomicExch(solve_err, 1);"));
        // perm の値域検査は b の読み出し（b[perm[i] * k）より前にある。
        let chk = src.find("pi >= n").unwrap();
        let rd = src.find("b[perm[i] * k").unwrap();
        assert!(chk < rd);
    }

    #[test]
    fn defines_are_numeric_and_deterministic() {
        let a = render_linalg_lu_source();
        let b = render_linalg_lu_source();
        assert_eq!(a.as_str(), b.as_str());
        let defines: Vec<&str> = a
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define LU_BLOCK 256"]);
        for l in defines {
            let v = l.split_whitespace().nth(2).unwrap();
            assert!(v.chars().all(|c| c.is_ascii_digit()), "{l}");
        }
        assert_eq!(a.program_name(), "fandhe_linalg_lu_f64.hip");
    }

    #[test]
    fn entry_names_and_arg_counts_match_source() {
        let s = render_linalg_lu_source();
        let src = s.as_str();
        for (name, argc) in [
            (LINALG_LU_KERNEL_NAME, LINALG_LU_ARG_COUNT),
            (LINALG_SOLVE_KERNEL_NAME, LINALG_SOLVE_ARG_COUNT),
        ] {
            let head = format!("extern \"C\" __global__ void {name}(");
            let at = src.find(&head).unwrap_or_else(|| panic!("{name}"));
            let sig = &src[at + head.len()..];
            let sig = &sig[..sig.find(") {").unwrap()];
            assert_eq!(sig.split(',').count(), argc, "{name}");
        }
    }

    #[test]
    fn block_constants_match_source() {
        assert_eq!(LINALG_LU_BLOCK_DIM, 256);
        assert_eq!(LINALG_SOLVE_BLOCK_DIM, 128);
        // 共有メモリ配列とストライドは LU_BLOCK マクロ経由（値は前置 #define と一致）。
        assert!(
            render_linalg_lu_source()
                .as_str()
                .contains("s_val[LU_BLOCK]")
        );
    }
}
