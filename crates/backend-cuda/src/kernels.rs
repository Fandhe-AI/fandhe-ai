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

/// GEMM epilogue（bias 加算・activation）を融合した tiled GEMM（f32）。
/// イシュー #599（TASK-12.1f `gemm_bias_act` の実融合化）。
///
/// [`TILED_F32`] のアキュムレーション部分（共有メモリタイリング・
/// `#pragma unroll` による積和ループ）はそのまま維持し、C への書き込み
/// 直前の epilogue で `has_bias` が真なら `acc + bias[col]`、`act == 1`
/// なら続けて `max(v, 0)` を適用してから 1 回だけ HBM へ書く（`gemm` →
/// `add` → `relu` の非融合合成のように中間結果を HBM へ書いて読み直す
/// ことをしない。CPU 側 `gemm_blis_bias_act_parallel`
/// （`crates/backend-cpu/src/gemm_blis/mod.rs`）と同じ「epilogue をカーネル
/// 内で完結させる」設計思想を CUDA へ適用したもの。
/// `docs/kernel-fusion.md` §2.2）。
///
/// **数値契約**: アキュムレーション自体は [`TILED_F32`] と完全に同一の
/// 演算順序（同じ shared memory タイリング・同じ `#pragma unroll` ループ）
/// のため、`gemm`→`add`→`relu` の非融合合成（同じ [`TILED_F32`] を経由し
/// た後に別カーネルで bias 加算・relu を適用する経路）と bit 完全一致に
/// なる（epilogue の加算・比較は要素独立で演算順序に依存しないため。
/// `.claude/rules/coding-rust.md` の FMA 契約統一節・
/// `docs/kernel-fusion.md` §2.2「bit 完全一致」と同じ論拠）。
///
/// **`bias` が `None`（`has_bias == 0`）の場合**: ホスト側
/// （`gemm.rs::CudaGemm::run_tiled_bias_act_f32`）は null ポインタではなく
/// ダミーの 1 要素デバイスバッファを `bias` へ渡す契約とする（`has_bias`
/// ガードにより当該バッファは実際には参照されないが、CUDA カーネル引数に
/// null を渡す経路を作らないため）。
///
/// # REQ-8（カーネル境界検査規約）
///
/// タイルロード時の三項ガード・C への書き込み時の `if (row < m && col <
/// n)` ガードは [`TILED_F32`] と同一（該当コメント参照）。epilogue の
/// `bias[col]` 参照は書き込みガード（`row < m && col < n`、したがって
/// `col < n`）の内側でのみ行うため、`bias`（`n` 要素想定）への範囲外
/// 読み出しは発生しない。
pub const TILED_BIAS_ACT_F32: &str = r#"
#define TILE 32

extern "C" __global__ void gemm_tiled_bias_act_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    int m, int n, int k,
    int has_bias, int act)
{
    __shared__ float as_tile[TILE][TILE];
    __shared__ float bs_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // 桁溢れしない num_tiles 計算（TILED_F32 と同一。上記ドキュメンテー
    // ションコメント参照）。
    int num_tiles = (k > 0) ? (k - 1) / TILE + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // REQ-8: TILED_F32 と同じ手動境界チェック（三項ガード）。
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

    // REQ-8: C への書き込み時の手動境界チェック（TILED_F32 と同一）。
    // epilogue（bias 加算・activation）はこのガードの内側でのみ適用し、
    // 中間結果を別カーネルへ渡さず 1 回の書き込みで完結させる。
    if (row < m && col < n) {
        float v = acc;
        if (has_bias) {
            v += bias[col];
        }
        if (act == 1) {
            v = v > 0.0f ? v : 0.0f;
        }
        c[row * n + col] = v;
    }
}
"#;

/// WMMA TF32 GEMM のブロックタイル一辺（M・N とも 32。2x2 warp グリッド）。
///
/// `gemm.rs` の `run_wmma_tf32` はグリッド計算（`div_ceil(m/n, WMMA_TF32_BLOCK_*)`）
/// に、カーネル起動 API（本モジュール外）はブロック次元（後述 `WMMA_TF32_THREADS`）に
/// この値を用いる。設計メモ（`docs/cuda-tensor-core-design.md` 4.2 節）の
/// ブロックタイル 128×128・warp タイル 64×64 は #61〜#63 の実測により確定する
/// 候補値であり、本イシュー（#62、#61 未マージのため WMMA 共通基盤を最小実装する
/// 安全側判断。イシュー #62 実装計画 2 節）では「共有メモリ配置・タイル形状の
/// 基本実装」を優先し、正しさを検証しやすい 32×32（2x2 warp グリッド、各 warp が
/// `m16n16k8` fragment 1 個を直接担当）から開始する。タイル拡大・warp あたり複数
/// fragment 化は #63 のスコープ。
pub const WMMA_TF32_BLOCK_M: u32 = 32;
pub const WMMA_TF32_BLOCK_N: u32 = 32;

/// WMMA TF32 GEMM の K タイル幅。TF32 fragment の K 次元（`m16n16k8`）と一致させ、
/// 共有メモリへの 1 回のロードがそのまま 1 回の `mma_sync` 入力になるようにする
/// （設計メモ 4.2 節「k タイル TF32: 8」の候補値をそのまま採用）。
pub const WMMA_TF32_K_TILE: u32 = 8;

/// WMMA fragment の M・N 一辺（`m16n16k8` の 16）。ブロックタイル 32 を
/// 16 で割った 2×2 が block 内の warp グリッド次元になる
/// （`WMMA_TF32_BLOCK_M / WMMA_TF32_FRAG == 2` 等が `gemm.rs` 側の暗黙契約）。
///
/// Rust 側での実利用は `gemm.rs::CudaGemm::new` 内の
/// `const _: () = assert!(...)`（ブロックタイル・warp グリッドの倍数関係
/// をコンパイル時検査する）のみで、通常の実行時コードパスからは参照され
/// ない（`#[cfg(test)]` の `wmma_tf32_constants_match_kernel_source_defines`
/// テストからは参照されるが、`cargo clippy --lib`〈非 test 版〉には効かな
/// い）。rustc 1.88 系の dead-code 解析はネストした無名 `const _` 内から
/// のみ参照される `pub const` を誤って未使用と判定する（1.92 以降では
/// 解消済み。`cargo +1.88.0 clippy` と `cargo +1.92.0 clippy` の実測差分で
/// 確認済み。#149 PR CI 指摘対応）。実行時 `debug_assert` への置換は
/// 「CUDA 非搭載の通常 CI では `new` 自体が実行されず検査が効かない」
/// というレビュー指摘 #62 の踏襲事項に反するため行わない。
#[allow(dead_code)]
pub const WMMA_TF32_FRAG: u32 = 16;

/// WMMA TF32 GEMM 1 ブロックあたりのスレッド数（4 warp = 128 スレッド。
/// `(WMMA_TF32_BLOCK_M / WMMA_TF32_FRAG) * (WMMA_TF32_BLOCK_N / WMMA_TF32_FRAG)`
/// = 2×2 warp を 1 次元ブロックとして起動する。`gemm.rs::run_wmma_tf32` の
/// ブロック次元はこの値を x 成分に用いる）。
pub const WMMA_TF32_THREADS: u32 = 128;

/// WMMA（Tensor Core）を用いた TF32 GEMM。入出力は f32、Tensor Core への投入時に
/// `wmma::__float_to_tf32` で明示的に丸める（REQ-11・TASK-11.1c・#62）。
///
/// **設計根拠**: `docs/cuda-tensor-core-design.md`（#60。3.3 節で方式 A =
/// WMMA C++ API `<mma.h>` を採用、4.1 節で fragment `m16n16k8`・TF32 精度・
/// f32 累算を選定）。本カーネルは同メモリの「共有メモリ配置・タイル形状の
/// 基本実装」（#62 イシュー追記のスコープ）にあたり、レジスタブロッキング・
/// ダブルバッファリング・ベクトル化ロード等の本格最適化は #63 に委ねる。
///
/// **構成**: ブロックタイル `WMMA_TF32_BLOCK_M` x `WMMA_TF32_BLOCK_N`（32×32）を
/// 4 warp（2×2 グリッド）で分担し、各 warp が `m16n16k8` fragment を 1 個直接
/// 担当する。K 方向は `WMMA_TF32_K_TILE`（8。fragment の K と一致）単位で
/// 共有メモリへロード → 全 warp が `load_matrix_sync` → 丸め → `mma_sync` を
/// 繰り返し、ブロック内 warp 間でグローバルメモリアクセスを共有する
/// （naive・tiled 版と同じ「共有メモリで再利用する」思想を Tensor Core 経路にも
/// 適用する）。
///
/// # REQ-8（カーネル境界検査規約）
///
/// - **入力タイルの guarded load**: `as_tile`／`bs_tile` への共有メモリロードは
///   `(global_row < m/k && global_col < k/n)` の三項ガードを通し、範囲外要素は
///   ゼロ充填する（`kernels::TILED_F32` と同じ方式。設計メモ 5 節 1）。これにより
///   `wmma::load_matrix_sync` は常にゼロ充填済み共有メモリのみを読み、グローバル
///   メモリへの範囲外アクセスは発生しない。
/// - **エピローグ store のガード条件**: `wmma::store_matrix_sync` は fragment
///   （16×16 全体）を無条件で書き込む API であり、境界を跨ぐ warp の店頭書き込み
///   をそのままグローバル C へ向けると OOB write になりうる。本カーネルは
///   いったん共有メモリ `c_tile` へ store し、`__syncthreads()` 後に
///   要素単位で `(global_row < m && global_col < n)` を判定してから
///   グローバル C へコピーする（設計メモ 5 節 2 の「範囲外の fragment 要素は
///   書き戻さない」を、store 後のガード付きコピーという形で満たす）。
/// - K 端（k が `WMMA_TF32_K_TILE` の倍数でない）は `num_k_tiles` を
///   `TILED_F32` と同じ桁溢れしない式（`(k > 0) ? (k - 1) / K_TILE + 1 : 0`）で
///   計算し、最終タイルの余剰要素は上記 guarded load のゼロ充填で処理する。
///
/// # 数値契約
///
/// TF32 は f32 の仮数部 23bit を 10bit に丸めて Tensor Core へ投入する
/// （`wmma::__float_to_tf32` による明示変換。NVIDIA 公式 `cudaTensorCoreGemm`
/// サンプルと同じ「load 後に fragment の各要素を変換してから mma_sync に渡す」
/// 手順を踏襲）。これにより `f32::mul_add`（CPU 参照実装、REQ-2 の FMA 契約統一）
/// との比較では tiled f32 版より誤差が大きくなりうるが、統一複合判定
/// （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は既に TF32 前提の複合指標に
/// 改定済みである（`.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」・
/// 設計メモ 6 節）。この閾値自体は本カーネルでは変更しない
/// （変更はユーザー承認必須。#186 のスコープ）。
pub const WMMA_TF32_F32: &str = r#"
#include <mma.h>

using namespace nvcuda;

#define WMMA_TF32_BLOCK_M 32
#define WMMA_TF32_BLOCK_N 32
#define WMMA_TF32_K_TILE 8
#define WMMA_TF32_FRAG 16

extern "C" __global__ void gemm_wmma_tf32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // WMMA の load_matrix_sync / store_matrix_sync は 256-bit（32 byte）
    // アライメントされたポインタを要求する（kernels_wmma.rs の f16 経路と同じ
    // 制約）。2x2 warp グリッド構成では warp_col=1 がタイル内 64 byte オフセット
    // を取るため、__align__(32) を明示しないと nvcc 既定の配置ではアライメント
    // が保証されず、デバイス上での起動失敗や不正な計算結果（サイレント）を
    // 招きうる。
    __shared__ __align__(32) float as_tile[WMMA_TF32_BLOCK_M][WMMA_TF32_K_TILE];
    __shared__ __align__(32) float bs_tile[WMMA_TF32_K_TILE][WMMA_TF32_BLOCK_N];
    __shared__ __align__(32) float c_tile[WMMA_TF32_BLOCK_M][WMMA_TF32_BLOCK_N];

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int block_row_base = blockIdx.y * WMMA_TF32_BLOCK_M;
    const int block_col_base = blockIdx.x * WMMA_TF32_BLOCK_N;

    wmma::fragment<wmma::matrix_a, WMMA_TF32_FRAG, WMMA_TF32_FRAG, WMMA_TF32_K_TILE,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TF32_FRAG, WMMA_TF32_FRAG, WMMA_TF32_K_TILE,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TF32_FRAG, WMMA_TF32_FRAG, WMMA_TF32_K_TILE, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 桁溢れしない num_k_tiles 計算（TILED_F32 と同じ方式。上記ドキュメンテーション
    // コメント参照）。
    int num_k_tiles = (k > 0) ? (k - 1) / WMMA_TF32_K_TILE + 1 : 0;

    for (int t = 0; t < num_k_tiles; ++t) {
        int k_base = t * WMMA_TF32_K_TILE;

        // REQ-8: A タイルの guarded load。範囲外要素はゼロ充填共有メモリへ
        // 書く（0 を掛けても acc に寄与しないため数値的にも安全）。
        for (int idx = tid; idx < WMMA_TF32_BLOCK_M * WMMA_TF32_K_TILE; idx += num_threads) {
            int local_row = idx / WMMA_TF32_K_TILE;
            int local_col = idx % WMMA_TF32_K_TILE;
            int global_row = block_row_base + local_row;
            int global_col = k_base + local_col;
            as_tile[local_row][local_col] =
                (global_row < m && global_col < k) ? a[global_row * k + global_col] : 0.0f;
        }

        // REQ-8: B タイルの guarded load。A と同じ根拠。
        for (int idx = tid; idx < WMMA_TF32_K_TILE * WMMA_TF32_BLOCK_N; idx += num_threads) {
            int local_row = idx / WMMA_TF32_BLOCK_N;
            int local_col = idx % WMMA_TF32_BLOCK_N;
            int global_row = k_base + local_row;
            int global_col = block_col_base + local_col;
            bs_tile[local_row][local_col] =
                (global_row < k && global_col < n) ? b[global_row * n + global_col] : 0.0f;
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, &as_tile[warp_row * WMMA_TF32_FRAG][0], WMMA_TF32_K_TILE);
        wmma::load_matrix_sync(b_frag, &bs_tile[0][warp_col * WMMA_TF32_FRAG], WMMA_TF32_BLOCK_N);

        // TF32 丸め特性（f32 仮数 23bit → 10bit）: fragment の各要素を明示的に
        // wmma::__float_to_tf32 で変換してから mma_sync へ渡す（NVIDIA
        // cudaTensorCoreGemm サンプルと同じ手順。上記ドキュメンテーション
        // コメント「数値契約」参照）。
        for (int i = 0; i < a_frag.num_elements; ++i) {
            a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
        }
        for (int i = 0; i < b_frag.num_elements; ++i) {
            b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);
        }

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(
        &c_tile[warp_row * WMMA_TF32_FRAG][warp_col * WMMA_TF32_FRAG],
        c_frag, WMMA_TF32_BLOCK_N, wmma::mem_row_major);
    __syncthreads();

    // REQ-8: エピローグ store のガード条件。store_matrix_sync は fragment
    // 全体（16x16）を無条件で書くため、共有メモリへ一旦 store したうえで
    // 要素単位のガード付きコピーによりグローバル C への範囲外書き込みを防ぐ
    // （上記ドキュメンテーションコメント参照）。
    for (int idx = tid; idx < WMMA_TF32_BLOCK_M * WMMA_TF32_BLOCK_N; idx += num_threads) {
        int local_row = idx / WMMA_TF32_BLOCK_N;
        int local_col = idx % WMMA_TF32_BLOCK_N;
        int global_row = block_row_base + local_row;
        int global_col = block_col_base + local_col;
        if (global_row < m && global_col < n) {
            c[global_row * n + global_col] = c_tile[local_row][local_col];
        }
    }
}
"#;

/// [`TILED_F32`] のアンカー 2 行（`row`/`col` のグローバル添字計算）を、
/// `swizzle.rs::swizzled_block_idx` と同一の整数式（グループ幅
/// `group_width` の M 方向グルーピング remap）へ差し替えた変種ソースを
/// 生成する（イシュー #1034。f16 `mma.sync` 経路の #499・TF32 opt-staged
/// 経路の #741 と同一設計。`kernels_mma.rs::mma_f16_source_with_swizzle`
/// 参照）。
///
/// **`TILED_F32`（本番既定 f32 カーネル。`docs/perf/
/// cuda-gemm-kernel-improvement-policy.md` §1）自体は変更しない**
/// （`replacen` で新規 `String` を都度構築するのみ）。
///
/// # 呼び出し元
///
/// `gemm.rs::CudaGemm::new_with_tiled_f32_swizzle`（`internal-diagnostics`
/// feature 限定・診断用 opt-in 入口）専用。**本番既定コンストラクタ
/// （`CudaGemm::new`）・`run_tiled_f32`／`launch_tiled_f32` はこの関数を
/// 呼ばない**（実装計画 §2「本番結線を本ランで行わない判断」。ncu による
/// L2 ヒット率実測・N=4096 改善値・サイズ条件付き適用へのユーザー承認を
/// 実機セッションで経てから `new` への昇格を検討する。`swizzle.rs`
/// 冒頭コメントが記録する f16 経路の先例〈#740 差し戻し→#775/#782 実機
/// ゲート後に `new` へ昇格〉と同じ安全側の順序を踏む）。
///
/// # remap の整数式（`swizzle.rs::swizzled_block_idx` と単一の設計を共有）
///
/// `gemm.rs::launch_config` は `TILED_BLOCK_DIM`（`kernels::TILE` 一辺）
/// 版のグリッドを `gridDim.x = num_n_blocks`・`gridDim.y = num_m_blocks`
/// （`blockIdx.x` が N 方向・`blockIdx.y` が M 方向）で構築する
/// （`kernels_mma.rs::mma_f16_source_with_swizzle` の grid レイアウトと
/// 同一）。したがって remap 式自体は `mma_f16_source_with_swizzle` と
/// 同一構造（`BM`/`BN` の代わりに `TILE` を単位とする）をそのまま適用
/// できる。線形 index・ブロック数・積は `long long`（64 bit）で計算し、
/// 最終座標を明示的に範囲検査してから `int` へ縮小する（PR #667
/// codex-review P0 是正を踏襲。REQ-8「境界検査の省略禁止」）。
///
/// remap 後も、タイルロード時の三項ガード・C 書き込み時の `if (row < m &&
/// col < n)`（`TILED_F32` 本体の既存手動境界チェック）は一切変更しない
/// （swizzle はブロックがどの `(m_block, n_block)` を担当するかの割り当て
/// のみを変え、各出力要素の積和順序・境界チェックの要否は変えないため）。
///
/// # エラー契約
///
/// `group_width < 2` は `CudaError::InvalidShape` で拒否する
/// （`mma_f16_source_with_swizzle` と同一の理由。`group_width == 1` は
/// 恒等写像に等しく L2 再利用効果を持たない）。アンカー未検出・複数検出
/// （`TILED_F32` 改変で本関数の前提が崩れた場合）も `CudaError::
/// InvalidShape` で拒否し、panic しない（本番経路から呼ばれる
/// `mma_f16_source_with_swizzle` と同じ fail-closed 契約。本関数自体は
/// `internal-diagnostics` feature 限定の opt-in 入口のみから呼ばれるが、
/// 契約を弱めない）。
pub fn tiled_f32_source_with_swizzle(group_width: u32) -> Result<String, crate::error::CudaError> {
    if group_width < 2 {
        return Err(crate::error::CudaError::InvalidShape {
            detail: format!(
                "tiled_f32_source_with_swizzle requires group_width >= 2 (got {group_width}); \
                 group_width == 1 degenerates to the identity block mapping and offers no \
                 L2 reuse benefit"
            ),
        });
    }

    const ANCHOR: &str = "    int row = blockIdx.y * TILE + threadIdx.y;\n    \
                           int col = blockIdx.x * TILE + threadIdx.x;\n";
    let source = TILED_F32;
    let occurrences = source.matches(ANCHOR).count();
    // `unwrap()`/`expect()`・panic 系マクロを本番経路で使わない方針
    // （coding-rust.md）に合わせ、型付きエラーで返す
    // （`mma_f16_source_with_swizzle` と同じ理由）。
    if occurrences != 1 {
        return Err(crate::error::CudaError::InvalidShape {
            detail: format!(
                "TILED_F32 中のグローバル添字アンカー（row/col）の出現数が 1 では \
                 ありません（{occurrences} 件検出。tiled_f32_source_with_swizzle \
                 の前提が崩れています）"
            ),
        });
    }

    let remap = format!(
        "    // イシュー #1034: L2 再利用のためのタイル→SM 割り当てスウィズル\n\
         \x20   // remap（swizzle.rs::swizzled_block_idx と同一式。\n\
         \x20   // kernels_mma.rs::mma_f16_source_with_swizzle と同型で\n\
         \x20   // `BM`/`BN` の代わりに `TILE` を単位とする。本ファイル\n\
         \x20   // tiled_f32_source_with_swizzle ドキュメンテーションコメント\n\
         \x20   // 参照）。PR #667 codex-review P0 是正を踏襲し、線形 index・\n\
         \x20   // ブロック数・積は `long long`（64 bit）で計算し、最終座標を\n\
         \x20   // 明示的に範囲検査してから `int` へ縮小する（REQ-8「境界検査\n\
         \x20   // の省略禁止」）。\n\
         \x20   #define SWIZZLE_GROUP {group_width}\n\
         \x20   long long num_m_blocks = gridDim.y;\n\
         \x20   long long num_n_blocks = gridDim.x;\n\
         \x20   long long linear_idx = (long long)blockIdx.y * gridDim.x + blockIdx.x;\n\
         \x20   long long full_groups = num_m_blocks / SWIZZLE_GROUP;\n\
         \x20   long long remainder = num_m_blocks % SWIZZLE_GROUP;\n\
         \x20   long long full_group_blocks = (long long)SWIZZLE_GROUP * num_n_blocks;\n\
         \x20   long long full_groups_total_blocks = full_groups * full_group_blocks;\n\
         \x20   long long m_block, n_block;\n\
         \x20   if (linear_idx < full_groups_total_blocks) {{\n\
         \x20       long long group_idx = linear_idx / full_group_blocks;\n\
         \x20       long long idx_in_group = linear_idx % full_group_blocks;\n\
         \x20       m_block = group_idx * SWIZZLE_GROUP + (idx_in_group % SWIZZLE_GROUP);\n\
         \x20       n_block = idx_in_group / SWIZZLE_GROUP;\n\
         \x20   }} else {{\n\
         \x20       long long idx_in_group = linear_idx - full_groups_total_blocks;\n\
         \x20       m_block = full_groups * SWIZZLE_GROUP + (idx_in_group % remainder);\n\
         \x20       n_block = idx_in_group / remainder;\n\
         \x20   }}\n\
         \x20   if (m_block < 0 || m_block >= num_m_blocks || n_block < 0 ||\n\
         \x20       n_block >= num_n_blocks) {{\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   int row = (int)(m_block * TILE) + threadIdx.y;\n\
         \x20   int col = (int)(n_block * TILE) + threadIdx.x;\n"
    );

    Ok(source.replacen(ANCHOR, &remap, 1))
}
#[cfg(test)]
mod tests {
    use super::*;

    /// `TILE`（Rust 側の「唯一の真実源」宣言。本ファイル上記ドキュメント
    /// コメント参照）が `TILED_F32`／`TILED_F16` の CUDA ソース文字列内の
    /// `#define TILE` と食い違わないことを検査する。
    ///
    /// 両者をリンクする仕組み（コード生成等）が存在しないため、`TILE` の値を
    /// 変更した際に片方だけ更新し忘れると `TILED_BLOCK_DIM` とカーネル内
    /// `__shared__` 配列サイズ・ループ範囲がずれ、コンパイルエラーにもならず
    /// 誤った積和結果を静かに生成しうる（レビュー指摘 #34）。この test は
    /// その不整合を CI 上（`cargo test -p fandhe-ai-backend-cuda`。実機不要・文字列
    /// 突合のみ）で機械検出する。
    #[test]
    fn tile_constant_matches_kernel_source_define() {
        let expected = format!("#define TILE {TILE}");
        assert!(
            TILED_F32.contains(&expected),
            "TILED_F32 の `#define TILE` が Rust 側の TILE 定数（{TILE}）と一致しません"
        );
        assert!(
            TILED_F16.contains(&expected),
            "TILED_F16 の `#define TILE` が Rust 側の TILE 定数（{TILE}）と一致しません"
        );
        assert!(
            TILED_BIAS_ACT_F32.contains(&expected),
            "TILED_BIAS_ACT_F32 の `#define TILE` が Rust 側の TILE 定数（{TILE}）と一致しません"
        );
    }

    /// `WMMA_TF32_BLOCK_M`／`WMMA_TF32_BLOCK_N`／`WMMA_TF32_K_TILE`（Rust 側の
    /// 「唯一の真実源」）が `WMMA_TF32_F32` カーネルソース内の `#define` と
    /// 食い違わないことを検査する（`tile_constant_matches_kernel_source_define`
    /// と同じ理由・同じ検査方式。`gemm.rs::run_wmma_tf32` のグリッド計算は
    /// この Rust 側定数を使うため、両者が一致しないとホスト側の期待するタイル
    /// 境界とカーネル実体の共有メモリ配列サイズがずれる）。
    #[test]
    fn wmma_tf32_constants_match_kernel_source_defines() {
        let checks = [
            ("WMMA_TF32_BLOCK_M", WMMA_TF32_BLOCK_M),
            ("WMMA_TF32_BLOCK_N", WMMA_TF32_BLOCK_N),
            ("WMMA_TF32_K_TILE", WMMA_TF32_K_TILE),
            ("WMMA_TF32_FRAG", WMMA_TF32_FRAG),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                WMMA_TF32_F32.contains(&expected),
                "WMMA_TF32_F32 の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
        }
    }

    /// イシュー #1034 受け入れ基準: `group_width < 2` を拒否する
    /// （`tiled_f32_source_with_swizzle` ドキュメンテーションコメント
    /// 「エラー契約」参照。`mma_f16_source_with_swizzle` と同型の検査）。
    #[test]
    fn tiled_f32_source_with_swizzle_rejects_group_width_below_two() {
        let err = tiled_f32_source_with_swizzle(1).expect_err("group_width=1 must be rejected");
        assert!(matches!(err, crate::error::CudaError::InvalidShape { .. }));
        let err = tiled_f32_source_with_swizzle(0).expect_err("group_width=0 must be rejected");
        assert!(matches!(err, crate::error::CudaError::InvalidShape { .. }));
    }

    /// イシュー #1034 受け入れ基準: `group_width >= 2` では生成ソースに
    /// `#define SWIZZLE_GROUP <group_width>` と remap 断片が含まれ、かつ
    /// 元のアンカー（`blockIdx.y`/`blockIdx.x` 直書き）は除去されている
    /// ことを検査する（`mma_f16_source_with_swizzle_contains_group_
    /// define_and_remap_fragment` と同型）。
    #[test]
    fn tiled_f32_source_with_swizzle_contains_group_define_and_remap_fragment() {
        for group_width in [2u32, 8, 16] {
            let src = tiled_f32_source_with_swizzle(group_width)
                .unwrap_or_else(|err| panic!("group_width={group_width}: {err}"));

            let expected_define = format!("#define SWIZZLE_GROUP {group_width}");
            assert!(
                src.contains(&expected_define),
                "group_width={group_width}: 生成ソースに `{expected_define}` が \
                 見つかりません"
            );
            for needle in [
                "long long linear_idx = (long long)blockIdx.y * gridDim.x + blockIdx.x;",
                "long long full_groups = num_m_blocks / SWIZZLE_GROUP;",
                "long long remainder = num_m_blocks % SWIZZLE_GROUP;",
                "int row = (int)(m_block * TILE) + threadIdx.y;",
                "int col = (int)(n_block * TILE) + threadIdx.x;",
            ] {
                assert!(
                    src.contains(needle),
                    "group_width={group_width}: 生成ソースに remap 断片 `{needle}` \
                     が見つかりません"
                );
            }
            assert!(
                !src.contains("int row = blockIdx.y * TILE + threadIdx.y;"),
                "group_width={group_width}: 元のアンカー（blockIdx.y 直書き）が \
                 remap 後も残っています"
            );
        }
    }

    /// `tiled_f32_source_with_swizzle` はアンカー置換のみを行い、
    /// `TILED_F32`（本番既定 f32 カーネル）自体は不変であることを
    /// ロックする（`mma_f16_source_with_swizzle_does_not_mutate_
    /// mma_f16_source` と同型の回帰防止。実装計画 2 節の安全側判断）。
    #[test]
    fn tiled_f32_source_with_swizzle_does_not_mutate_tiled_f32_source() {
        let before = TILED_F32;
        let _ = tiled_f32_source_with_swizzle(8).expect("group_width=8 must be accepted");
        assert_eq!(
            TILED_F32, before,
            "tiled_f32_source_with_swizzle 呼び出し後に TILED_F32 が変化しています"
        );
        assert!(
            TILED_F32.contains("int row = blockIdx.y * TILE + threadIdx.y;"),
            "TILED_F32 の元のアンカー行が失われています（本番カーネルは無変更のはず）"
        );
    }
}
