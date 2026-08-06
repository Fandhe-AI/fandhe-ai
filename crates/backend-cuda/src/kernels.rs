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
//! の `NAIVE_F32`／`NAIVE_F16`（#33）・`TILED_F32`／`TILED_F16`（本ファイル。#34）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! naive カーネルは `if (row < m && col < n)` の手動境界チェックを、性能上の
//! 理由で省略していない。1 スレッド = C の 1 要素でグリッドを `div_ceil`
//! により切り上げ生成するため、末尾ブロックでは `row`／`col` が `m`／`n` を
//! 超えるスレッドが必ず発生する（`gemm.rs` のグリッド計算参照）。この境界
//! チェックを外すと OOB 書き込みが発生するため、
//! `.claude/rules/coding-rust.md` の REQ-8 規約に従い必須要件として維持する。
//!
//! tiled カーネルも同様に、共有メモリタイルへのロード時（`(row < m &&
//! a_col < k) ? ... : 0` 相当の三項ガード）と C への書き込み時（`if (row < m
//! && col < n)`）の両方で手動境界チェックを維持する。`#pragma unroll` に
//! よるタイル内積和ループの展開は演算命令数を削減する最適化であり、
//! 境界チェックそのものを無効化しないため REQ-8 の許容範囲内である
//! （「境界検査を無効化する最適化を適用する場合は境界チェックを維持した
//! うえで行う」の実例）。

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

/// 共有メモリタイルの一辺（正方行列）。
///
/// `gemm.rs` の `run_tiled_f32`／`run_tiled_f16` はブロック次元を
/// `(TILE, TILE, 1)` に固定して起動する必要がある（カーネル内
/// `__shared__` 配列サイズはコンパイル時定数の `TILE`（カーネルソース側の
/// `#define TILE 32`）と一致していなければならず、ホスト側のこの定数が
/// 唯一の真実源である。PoC-v2-3（`cuda/kernels.rs:16`）と同じ値 32 を踏襲）。
pub const TILE: u32 = 32;

/// tiled GEMM（f32）。共有メモリタイリング（`TILE` x `TILE`）版。
///
/// A・B の各タイルを一度だけグローバルメモリから読み、共有メモリ上で
/// ブロック内の全スレッドが再利用することで、naive 版に対しグローバル
/// メモリアクセス回数を概ね `1/TILE` に削減する（受け入れ条件「tiled GEMM
/// が naive 比で PoC-v2-3 相当の性能改善を示す」の対象。PoC-v2-3 実測では
/// f32 で naive 比 約 1.19〜1.46 倍。
/// `docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md` 計測結果節）。
///
/// **productize での PoC からの差分**: PoC の `int num_tiles = (k + TILE -
/// 1) / TILE;` は `k` が `i32::MAX - (TILE - 1)` 超のとき `k + TILE - 1` が
/// C の `int`（32bit 符号付き）算術でオーバーフローし未定義動作となる
/// （#240 の Cursor Bugbot 指摘と同系統の 32bit int 算術問題）。本カーネルは
/// 桁溢れしない `(k > 0) ? (k - 1) / TILE + 1 : 0` へ書き換えている。ホスト側
/// （`gemm.rs::validate_gemm_dims` に加えタイル専用の追加ガード）でも
/// `k` の上限を別途検証しており、二重に防御する。
pub const TILED_F32: &str = r#"
#define TILE 32

extern "C" __global__ void gemm_tiled_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    __shared__ float as_tile[TILE][TILE];
    __shared__ float bs_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // 桁溢れしない num_tiles 計算（上記ドキュメンテーションコメント参照）。
    int num_tiles = (k > 0) ? (k - 1) / TILE + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // REQ-8: タイルロード時の手動境界チェック（三項ガード）。末尾
        // タイルでは a_col/b_row が k を超えうるため、範囲外読み出しの
        // 代わりに 0 を共有メモリへ書く（0 を掛けても acc に寄与しないため
        // 数値的にも安全）。
        as_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        bs_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += as_tile[threadIdx.y][p] * bs_tile[p][threadIdx.x];
        }
        __syncthreads();
    }

    // REQ-8: C への書き込み時の手動境界チェック。末尾ブロックでは
    // row/col が m/n を超えるスレッドが必ず発生する（naive 版と同じ理由）。
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}
"#;

/// tiled GEMM（f16 入出力・f32 アキュムレート）。
///
/// 共有メモリタイルは f16 で確保する（`TILED_F32` 比でタイルのメモリ
/// フットプリントを半減させ、同一 `TILE` サイズでも占有率（occupancy）が
/// 上がりやすいことを見込む。PoC-v2-3 と同じ設計判断）。アキュムレータは
/// `NAIVE_F16` と同じ理由（桁落ち対策）で f32 に固定する。`num_tiles` の
/// 桁溢れ対策は `TILED_F32` と同一。
pub const TILED_F16: &str = r#"
#include <cuda_fp16.h>

#define TILE 32

extern "C" __global__ void gemm_tiled_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m, int n, int k)
{
    __shared__ __half as_tile[TILE][TILE];
    __shared__ __half bs_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    int num_tiles = (k > 0) ? (k - 1) / TILE + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // REQ-8: TILED_F32 と同じ手動境界チェック（三項ガード）。
        as_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : __float2half(0.0f);
        bs_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : __float2half(0.0f);
        __syncthreads();

#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += __half2float(as_tile[threadIdx.y][p]) * __half2float(bs_tile[p][threadIdx.x]);
        }
        __syncthreads();
    }

    // REQ-8: C への書き込み時の手動境界チェック。
    if (row < m && col < n) {
        c[row * n + col] = __float2half(acc);
    }
}
"#;
