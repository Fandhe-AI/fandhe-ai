//! naive GEMM の CUDA C カーネルソース（NVRTC 実行時コンパイル用の静的文字列）。
//!
//! `gemm.rs`（呼び出し元）は本モジュールの定数を `nvrtc::compile_ptx` に渡し
//! `CudaFunction` を得る。ソースを `nvcc` で事前コンパイルせず文字列のまま
//! 埋め込むのは、ビルド時に nvcc/CUDA ヘッダを一切要求しないためであり、
//! これが TASK-1.7（#31）の維持条件である「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」ことの要になる
//! （`nvrtc.rs` の A03 対応コメント・`.claude/rules/deps-policy.md` 参照）。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/kernels.rs`
//! の `NAIVE_F32`／`NAIVE_F16` のみ（tiled 版は #34 のスコープであり本ファイル
//! には含めない）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 両カーネルとも `if (row < m && col < n)` の手動境界チェックを、性能上の
//! 理由で省略していない。naive カーネルは 1 スレッド = C の 1 要素で
//! グリッドを `div_ceil` により切り上げ生成するため、末尾ブロックでは
//! `row`／`col` が `m`／`n` を超えるスレッドが必ず発生する（`gemm.rs` の
//! グリッド計算参照）。この境界チェックを外すと OOB 書き込みが発生するため、
//! `.claude/rules/coding-rust.md` の REQ-8 規約に従い必須要件として維持する。

/// naive GEMM（f32）。1 スレッド = C の 1 要素。
///
/// キャッシュ・共有メモリを一切使わない素朴実装（PoC-v2-3 の段階 1 相当）。
/// tiled 版（#34）との性能比較の基準点として残す。
pub const NAIVE_F32: &str = r#"
extern "C" __global__ void gemm_naive_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; ++p) {
            acc += a[row * k + p] * b[p * n + col];
        }
        c[row * n + col] = acc;
    }
}
"#;

/// naive GEMM（f16 入出力・f32 アキュムレート）。
///
/// f16 は仮数部が 10bit しかなく、K が大きい GEMM を f16 のまま
/// アキュムレートすると桁落ちが急速に蓄積するため、内部アキュムレータは
/// f32 に固定する（PyTorch の `torch.matmul`（f16）が cuBLAS 内部で
/// FP32 アキュムレートするのと同じ方針に揃え、数値比較〈PoC-v2-5〉の
/// 前提を合わせる。`.claude/rules/coding-rust.md` の FMA 契約統一節）。
pub const NAIVE_F16: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void gemm_naive_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; ++p) {
            acc += __half2float(a[row * k + p]) * __half2float(b[p * n + col]);
        }
        c[row * n + col] = __float2half(acc);
    }
}
"#;
