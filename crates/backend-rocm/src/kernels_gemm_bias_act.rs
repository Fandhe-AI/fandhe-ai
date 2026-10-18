//! GEMM + bias + activation 融合カーネルの HIP ソース（`backend-cuda::kernels::TILED_BIAS_ACT_F32` の
//! 移植。イシュー #3442・分解案 R11）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_gemm_bias_act_source`]）。呼び出し元は
//! `gemm_bias_act.rs` の `RocmGemmBiasAct` で、`ops/gemm_derived.rs` が `gemm_resident_rhs`・
//! `gemm_resident_lhs`・`gemm_fp32_strict_into`・`linear_forward_device` の 4 メソッドへ結線する。
//! 4 メソッドは同じカーネル 1 本を `has_bias`／`act`／オフセットの違いだけで使い分ける。
//!
//! # 設計
//! - 主ループ（guarded load・単一アキュムレータ・k 昇順・`__builtin_fmaf`）は R0 の
//!   `kernels_gemm::render_gemm_f32_source` と**バイト単位で同一**（テストで固定する）。CUDA 版の
//!   `acc += a*b` は持ち込まない（FMA 契約は CPU 参照の `f32::mul_add` と揃える）。
//! - オペランドは「base ポインタ + `long long` の要素オフセット」で受ける。`TypedDeviceBuffer` は
//!   base ポインタしか持たず、ホスト側で生ポインタを足さないことで、起動の `Arg::Ptr` が
//!   `as_device_ptr` の値だけという K1 監査の前提を崩さない。
//! - epilogue は書き込みガード（`row < m` → `col < n`）の内側でだけ `bias[col]` を読み、ReLU を
//!   適用して 1 回だけ書く（REQ-8）。bias なしのときは 1 要素のダミーを渡し `has_bias = 0` にする。
//! - A03: 静的本体と検証済みタイル定数の `#define` だけで組み立てる。shape・offset・k は
//!   カーネル引数で渡し、ソースへは入れない。
//! - `#pragma clang fp contract(off)` は epilogue の加算が暗黙に縮約されるのを防ぐ予防策
//!   （明示した `__builtin_fmaf` には影響しない）。wave 幅依存の命令は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::kernels_gemm::GemmF32Tile;
use crate::launch::{ArgKind, KernelSpec};

/// カーネルのエントリ名。
pub(crate) const GEMM_BIAS_ACT_KERNEL_NAME: &str = "gemm_tiled_bias_act_f32";

/// `(a, b, bias, c, a_off, b_off, bias_off, c_off, m, n, k, has_bias, act)`。
pub(crate) const GEMM_BIAS_ACT_SPEC: KernelSpec = KernelSpec {
    name: GEMM_BIAS_ACT_KERNEL_NAME,
    args: &[
        ArgKind::Ptr,
        ArgKind::Ptr,
        ArgKind::Ptr,
        ArgKind::Ptr,
        ArgKind::I64,
        ArgKind::I64,
        ArgKind::I64,
        ArgKind::I64,
        ArgKind::I32,
        ArgKind::I32,
        ArgKind::I32,
        ArgKind::I32,
        ArgKind::I32,
    ],
};

/// 静的テンプレート本体。`BM` 等のマクロは前置する `#define` で与える。
pub(crate) const GEMM_BIAS_ACT_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ void gemm_tiled_bias_act_f32(
    const float* __restrict__ a_base,
    const float* __restrict__ b_base,
    const float* __restrict__ bias_base,
    float* __restrict__ c_base,
    long long a_off, long long b_off, long long bias_off, long long c_off,
    int m, int n, int k, int has_bias, int act)
{
    // オフセット適用後のポインタ。w と bias は同じ連結バッファを指しうるが、どちらも読み取り専用。
    // 書き込み先 c は新規確保か out で、a・b とは重ならない（呼び出し側の契約）。
    const float* a = a_base + a_off;
    const float* b = b_base + b_off;
    const float* bias = bias_base + bias_off;
    float* c = c_base + c_off;

    __shared__ float as_tile[BK][BM + PAD];
    __shared__ float bs_tile[BK][BN + PAD];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // 桁溢れしない num_tiles 計算（k > 0 のとき (k-1)/BK + 1）。
    int num_tiles = (k > 0) ? (k - 1) / BK + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * BK;

        // REQ-8: A タイルの guarded load（転置格納）。範囲外は 0 埋め。
#pragma unroll
        for (int i = 0; i < (BM * BK) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int mm = idx % BM;
            int kk = idx / BM;
            int row = block_row + mm;
            int col = k_base + kk;
            as_tile[kk][mm] = (row < m && col < k) ? a[row * k + col] : 0.0f;
        }

        // REQ-8: B タイルの guarded load。
#pragma unroll
        for (int i = 0; i < (BN * BK) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int nn = idx % BN;
            int kk = idx / BN;
            int b_row = k_base + kk;
            int b_col = block_col + nn;
            bs_tile[kk][nn] = (b_row < k && b_col < n) ? b[b_row * n + b_col] : 0.0f;
        }
        __syncthreads();

        // 単一アキュムレータ・k 昇順・明示 FMA（CPU 参照 f32::mul_add と契約を揃える）。
#pragma unroll
        for (int p = 0; p < BK; ++p) {
            float a_frag[TM];
            float b_frag[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = as_tile[p][threadIdx.y * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = bs_tile[p][threadIdx.x * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // REQ-8: C への書き込み時の手動境界チェック（row/col を個別判定）。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + threadIdx.y * TM + i;
        if (row < m) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = block_col + threadIdx.x * TN + j;
                if (col < n) {
                    float v = acc[i][j];
                    if (has_bias != 0) {
                        v = v + bias[col];
                    }
                    if (act == 1) {
                        v = v > 0.0f ? v : 0.0f;
                    }
                    c[row * n + col] = v;
                }
            }
        }
    }
}
"#;

/// 検証済みタイル設定の数値だけを `#define` として前置したソースを生成する。
pub(crate) fn render_gemm_bias_act_source(tile: &GemmF32Tile) -> Result<KernelSource, RocmError> {
    KernelSource::render(
        &[
            ("BM", tile.bm()),
            ("BN", tile.bn()),
            ("BK", tile.bk()),
            ("TM", tile.tm()),
            ("TN", tile.tn()),
            ("PAD", tile.pad()),
            ("THREADS", tile.threads()),
        ],
        GEMM_BIAS_ACT_BODY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};
    use crate::kernels_gemm::render_gemm_f32_source;

    fn src() -> KernelSource {
        render_gemm_bias_act_source(&GemmF32Tile::DEFAULT).unwrap()
    }

    #[test]
    fn spec_matches_source_signature() {
        assert_eq!(
            kernel_param_kinds(GEMM_BIAS_ACT_BODY, GEMM_BIAS_ACT_KERNEL_NAME),
            GEMM_BIAS_ACT_SPEC.args
        );
    }

    #[test]
    fn defines_are_seven_numeric_only() {
        let s = src();
        let lines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(lines.len(), 7);
        for line in lines {
            let v = line.split_whitespace().nth(2).unwrap();
            assert!(v.chars().all(|c| c.is_ascii_digit()), "{line}");
        }
        assert_eq!(s, src());
    }

    /// `__shared__` 宣言から C 書き込みコメント直前まで（主ループ全体）を切り出す。
    fn main_loop(s: &str) -> &str {
        let start = s.find("    __shared__ float as_tile").unwrap();
        let end = s.find("    // REQ-8: C への書き込み").unwrap();
        &s[start..end]
    }

    #[test]
    fn main_loop_is_byte_identical_to_r0_gemm() {
        let r0 = render_gemm_f32_source(&GemmF32Tile::DEFAULT);
        assert_eq!(main_loop(src().as_str()), main_loop(r0.as_str()));
    }

    #[test]
    fn req8_guards_and_fma_present() {
        let s = src();
        let s = s.as_str();
        assert!(s.contains("(row < m && col < k) ? a[row * k + col] : 0.0f"));
        assert!(s.contains("(b_row < k && b_col < n) ? b[b_row * n + b_col] : 0.0f"));
        assert!(s.contains("__builtin_fmaf(a_frag[i], b_frag[j], acc[i][j])"));
        assert!(!s.contains("+= a_frag"));
        // bias[col] は col < n ガードの内側でだけ読む。
        let guard = s.find("if (col < n)").unwrap();
        let read = s.find("bias[col]").unwrap();
        assert!(read > guard);
        assert_eq!(s.matches("bias[col]").count(), 1);
        // 出力への書き込みは 1 か所で、ガードの内側。
        assert_eq!(s.matches("c[row * n + col]").count(), 1);
        assert!(s.find("c[row * n + col]").unwrap() > guard);
    }

    #[test]
    fn contract_off_and_no_warp_dependence() {
        assert!(GEMM_BIAS_ACT_BODY.starts_with("#pragma clang fp contract(off)\n"));
        for w in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!src().as_str().contains(w), "{w}");
        }
    }
}
