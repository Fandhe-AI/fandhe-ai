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
/// **イシュー #1032 以降は [`TILED_F16`] 専用**（f32 tiled 系はレジスタ
/// ブロッキング刷新に伴い [`TILED_F32_BM`] 等へ分離した。`TILED_F16` は
/// 本イシューのスコープ外〈実装計画 §2 非スコープ〉のため `TILE`＝32 の
/// まま無変更）。`gemm.rs` の `run_tiled_f16` はブロック次元を
/// `(TILE, TILE, 1)` に固定して起動する必要がある（カーネル内
/// `__shared__` 配列サイズはコンパイル時定数の `TILE`（カーネルソース側の
/// `#define TILE 32`）と一致していなければならず、ホスト側のこの定数が
/// 唯一の真実源である。PoC-v2-3（`cuda/kernels.rs:16`）と同じ値 32 を踏襲）。
pub const TILE: u32 = 32;

/// tiled f32 GEMM（[`TILED_F32`]／[`TILED_BIAS_ACT_F32`]）のブロックタイル
/// M 一辺。イシュー #1032（レジスタブロッキング拡大・smem バンク
/// コンフリクト対策）で `TILE`（32、1 スレッド 1 出力）から刷新し、
/// Tensor Core 経路のレジスタブロッキング先例（#493 の `kernels_mma.rs`
/// 2x2 warp タイル）を FP32 SIMT 経路へ水平展開したもの。1 スレッドが
/// `TILED_F32_TM` x `TILED_F32_TN`（4x4=16）出力を担当することで、
/// 共有メモリからの 1 ロードあたり再利用される演算数を旧実装比 4 倍に
/// 増やす（`docs/perf/cuda-gemm-simt-register-blocking.md` 設計根拠節）。
pub const TILED_F32_BM: u32 = 64;
/// ブロックタイル N 一辺。[`TILED_F32_BM`] と同一の設計根拠。
pub const TILED_F32_BN: u32 = 64;
/// ブロックタイル K 一辺（1 反復で共有メモリへロードする K 方向幅）。
pub const TILED_F32_BK: u32 = 16;
/// 1 スレッドが担当する出力タイルの M 方向要素数（レジスタブロッキング）。
///
/// Rust 側での実利用は `kernels.rs::tests` の定数突合テストのみで、通常の
/// 実行時コードパス（`gemm.rs`）からは値そのものではなくカーネル
/// ソース文字列中の `#define TM 4` 経由でのみ参照される（`WMMA_TF32_FRAG`
/// と同じ理由・同じ `#[allow(dead_code)]` 適用。当該定数のドキュメント
/// コメント参照）。
#[allow(dead_code)]
pub const TILED_F32_TM: u32 = 4;
/// 1 スレッドが担当する出力タイルの N 方向要素数（レジスタブロッキング）。
/// [`TILED_F32_TM`] と同じ理由で `#[allow(dead_code)]`。
#[allow(dead_code)]
pub const TILED_F32_TN: u32 = 4;
/// 共有メモリタイル（`as_tile`／`bs_tile`）のパディング要素数（float 4 個
/// = 16 byte）。`kernels_wmma_opt.rs` の非 2 冪パディング先例（#498）と
/// 同方針で、タイル一辺をそのまま配列サイズにすると生じうるバンク
/// コンフリクトを緩和する（XOR swizzle は本イシューでは導入せず、ncu
/// 実測でコンフリクト残存が確認された場合のみ後続で検討する。
/// `docs/perf/cuda-gemm-simt-register-blocking.md` 参照）。
/// [`TILED_F32_TM`] と同じ理由で `#[allow(dead_code)]`。
#[allow(dead_code)]
pub const TILED_F32_PAD: u32 = 4;
/// スレッドブロックの x 方向スレッド数（`TILED_F32_BN / TILED_F32_TN`）。
/// `gemm.rs::TILED_F32_BLOCK_DIM` の唯一の真実源はこの定数群であり、
/// ブロック次元は `(TILED_F32_THREADS_X, TILED_F32_THREADS_Y, 1)`。
pub const TILED_F32_THREADS_X: u32 = 16;
/// スレッドブロックの y 方向スレッド数（`TILED_F32_BM / TILED_F32_TM`）。
pub const TILED_F32_THREADS_Y: u32 = 16;

/// tiled GEMM（f32）。レジスタブロッキング（スレッドあたり
/// `TILED_F32_TM` x `TILED_F32_TN` 出力）+ 転置 A タイル + 共有メモリ
/// パディングによる 2D ブロックタイリング版（イシュー #1032）。
///
/// **旧実装（1 スレッド 1 出力・32x32 タイル）からの刷新理由**: 旧実装は
/// 共有メモリからの 1 ロードにつき 1 回の積和しか行わず、スレッドあたり
/// 算術強度が低いままメモリ律速になっていた（`docs/perf/
/// cuda-gemm-optimization-baseline.md`・#928 ベースライン。N=2048/4096 で
/// candle/Burn 比 約 0.56〜0.87 倍）。本実装は `as_tile`／`bs_tile` から
/// 読んだ 1 要素を `TILED_F32_TM`／`TILED_F32_TN` 通りの積和で再利用する
/// ことで、同じ共有メモリ帯域あたりの FLOPs を増やす（siboehm 系 2D
/// register-blocked SGEMM と同型の構成）。
///
/// **A タイルは転置格納**する（`as_tile[kk][mm]`。K 方向を先頭添字にする
/// ことで、積和ループ内の `as_tile[p][threadIdx.y * TM + i]` アクセスが
/// 固定の `p` 行内で完結し、M 方向へ連続アクセスできる）。ロード段階は
/// `TILED_F32_BM * TILED_F32_BK`（1024）要素を 256 スレッドで 4 要素ずつ
/// 分担する（`BM/BN` が `THREADS_X * THREADS_Y`（256）の整数倍であること
/// はホスト側静的テスト `tiled_f32_constants_satisfy_thread_and_tile_
/// invariants` で検査する）。
///
/// **num_tiles の桁溢れ対策**: [`TILED_F32_BK`] 基準で `(k > 0) ? (k - 1) /
/// TILED_F32_BK + 1 : 0` を用いる（`TILE` 版と同じ理由・同じ式形。#240 の
/// Cursor Bugbot 指摘と同系統の 32bit int 算術保護）。
///
/// # REQ-8（カーネル境界検査規約）
///
/// - タイルロード時: `as_tile`／`bs_tile` への書き込みはいずれも
///   `(グローバル行 < 上限 && グローバル列 < 上限)` の三項ガードを通し、
///   範囲外要素はゼロ充填する（旧実装と同じ方式）。
/// - C への書き込み時: `row < m` と `col < n` を個別にガードしてから
///   書き込む（レジスタタイルの各要素ごとに判定するため、旧実装の単一
///   ガードより判定回数は増えるが OOB 書き込みを防ぐ契約は同一）。
/// - `#pragma unroll` によるループ展開は演算命令数を削減する最適化で
///   あり境界チェック自体は無効化しないため REQ-8 の許容範囲内
///   （`TILED_F32`〈旧実装〉ドキュメントコメントと同じ論拠）。
pub const TILED_F32: &str = r#"
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define PAD 4
#define THREADS 256

extern "C" __global__ void gemm_tiled_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
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

    // 桁溢れしない num_tiles 計算（上記ドキュメンテーションコメント参照）。
    int num_tiles = (k > 0) ? (k - 1) / BK + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * BK;

        // REQ-8: A タイルの guarded load（転置格納）。BM*BK 要素を
        // THREADS スレッドで (BM*BK)/THREADS 要素ずつ分担する。範囲外は
        // 0 埋め（0 を掛けても acc に寄与しないため数値的にも安全）。
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

        // レジスタブロッキング積和: 共有メモリから読んだ 1 要素を
        // TM/TN 通りの積和で再利用する（旧実装比 TM*TN=16 倍の算術強度）。
        // アキュムレーションは単一アキュムレータ・k 昇順・逐次 FMA を
        // 維持し、CPU 参照実装（`f32::mul_add`）との FMA 契約
        // （`.claude/rules/coding-rust.md`）を崩さない。
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
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // REQ-8: C への書き込み時の手動境界チェック（要素ごとに row/col を
    // 個別判定する。旧実装の単一ガードとは判定回数が異なるが、末尾
    // ブロックで m/n を超える要素を書き込まない契約は同一）。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + threadIdx.y * TM + i;
        if (row < m) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = block_col + threadIdx.x * TN + j;
                if (col < n) {
                    c[row * n + col] = acc[i][j];
                }
            }
        }
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
/// イシュー #599（TASK-12.1f `gemm_bias_act` の実融合化）。イシュー #1032
/// で [`TILED_F32`] のレジスタブロッキング刷新に合わせて同時に書き換えた
/// （下記「数値契約」参照。片側だけの変更は bit 完全一致契約を崩すため
/// 禁止。実装計画 §3.3）。
///
/// [`TILED_F32`] のアキュムレーション部分（共有メモリタイリング・
/// レジスタブロッキング・`#pragma unroll` による積和ループ）はそのまま
/// 同一構造で維持し、C への書き込み直前の epilogue で `has_bias` が真なら
/// `acc + bias[col]`、`act == 1` なら続けて `max(v, 0)` を適用してから
/// 1 回だけ HBM へ書く（`gemm` → `add` → `relu` の非融合合成のように
/// 中間結果を HBM へ書いて読み直すことをしない。CPU 側
/// `gemm_blis_bias_act_parallel`
/// （`crates/backend-cpu/src/gemm_blis/mod.rs`）と同じ「epilogue をカーネル
/// 内で完結させる」設計思想を CUDA へ適用したもの。
/// `docs/kernel-fusion.md` §2.2）。
///
/// **数値契約**: アキュムレーション自体は [`TILED_F32`] と完全に同一の
/// 演算順序（同じ共有メモリタイリング・同じレジスタブロッキング・同じ
/// `#pragma unroll` ループ）のため、`gemm`→`add`→`relu` の非融合合成
/// （同じ [`TILED_F32`] を経由した後に別カーネルで bias 加算・relu を
/// 適用する経路）と bit 完全一致になる（epilogue の加算・比較は要素
/// 独立で演算順序に依存しないため。`.claude/rules/coding-rust.md` の
/// FMA 契約統一節・`docs/kernel-fusion.md` §2.2「bit 完全一致」と同じ
/// 論拠）。
///
/// **`bias` が `None`（`has_bias == 0`）の場合**: ホスト側
/// （`gemm.rs::CudaGemm::run_tiled_bias_act_f32`）は null ポインタではなく
/// ダミーの 1 要素デバイスバッファを `bias` へ渡す契約とする（`has_bias`
/// ガードにより当該バッファは実際には参照されないが、CUDA カーネル引数に
/// null を渡す経路を作らないため）。
///
/// # REQ-8（カーネル境界検査規約）
///
/// タイルロード時の三項ガード・C への書き込み時の要素単位ガードは
/// [`TILED_F32`] と同一（該当コメント参照）。epilogue の `bias[col]`
/// 参照は書き込みガード（`col < n`）の内側でのみ行うため、`bias`
/// （`n` 要素想定）への範囲外読み出しは発生しない。
pub const TILED_BIAS_ACT_F32: &str = r#"
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define PAD 4
#define THREADS 256

extern "C" __global__ void gemm_tiled_bias_act_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ c,
    int m, int n, int k,
    int has_bias, int act)
{
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

    // 桁溢れしない num_tiles 計算（TILED_F32 と同一。該当ドキュメンテー
    // ションコメント参照）。
    int num_tiles = (k > 0) ? (k - 1) / BK + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * BK;

        // REQ-8: TILED_F32 と同じ guarded load（転置格納）。
#pragma unroll
        for (int i = 0; i < (BM * BK) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int mm = idx % BM;
            int kk = idx / BM;
            int row = block_row + mm;
            int col = k_base + kk;
            as_tile[kk][mm] = (row < m && col < k) ? a[row * k + col] : 0.0f;
        }
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

        // TILED_F32 と同一構造のレジスタブロッキング積和（bit 完全一致
        // 契約。該当ドキュメンテーションコメント参照）。
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
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // REQ-8: C への書き込み時の手動境界チェック（TILED_F32 と同一）。
    // epilogue（bias 加算・activation）はこのガードの内側でのみ適用し、
    // 中間結果を別カーネルへ渡さず 1 回の書き込みで完結させる。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + threadIdx.y * TM + i;
        if (row < m) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = block_col + threadIdx.x * TN + j;
                if (col < n) {
                    float v = acc[i][j];
                    if (has_bias) {
                        v += bias[col];
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `TILE`（Rust 側の「唯一の真実源」宣言。本ファイル上記ドキュメント
    /// コメント参照）が `TILED_F16` の CUDA ソース文字列内の `#define
    /// TILE` と食い違わないことを検査する。
    ///
    /// イシュー #1032 で `TILED_F32`／`TILED_BIAS_ACT_F32` は `TILE` を
    /// 使わなくなった（[`TILED_F32_BM`] 等へ分離。下記
    /// `tiled_f32_constants_match_kernel_source_defines` が検査する）ため、
    /// 本 test の対象は `TILED_F16` のみへ縮小した。
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
            TILED_F16.contains(&expected),
            "TILED_F16 の `#define TILE` が Rust 側の TILE 定数（{TILE}）と一致しません"
        );
    }

    /// `TILED_F32_BM`／`BN`／`BK`／`TM`／`TN`／`PAD`（Rust 側の「唯一の
    /// 真実源」）が `TILED_F32`／`TILED_BIAS_ACT_F32` の CUDA ソース文字列内
    /// の `#define` と食い違わないことを検査する
    /// （`tile_constant_matches_kernel_source_define` と同じ理由・同じ
    /// 検査方式。`gemm.rs::tiled_f32_launch_config` のグリッド・ブロック
    /// 次元計算はこれら Rust 側定数を使うため、両者が一致しないとホスト側の
    /// 期待するタイル境界とカーネル実体の共有メモリ配列サイズがずれる。
    /// イシュー #1032）。
    #[test]
    fn tiled_f32_constants_match_kernel_source_defines() {
        let checks = [
            ("BM", TILED_F32_BM),
            ("BN", TILED_F32_BN),
            ("BK", TILED_F32_BK),
            ("TM", TILED_F32_TM),
            ("TN", TILED_F32_TN),
            ("PAD", TILED_F32_PAD),
            // カーネルソース内のロード段階スレッド分担（`(BM*BK)/THREADS`
            // 等）が生の `256` を重複保持せず `THREADS` マクロ経由に
            // なったため、その `#define THREADS` も Rust 側の唯一の
            // 真実源（`TILED_F32_THREADS_X * TILED_F32_THREADS_Y`）との
            // 突合対象に加える（レビュー指摘。イシュー #1032）。
            ("THREADS", TILED_F32_THREADS_X * TILED_F32_THREADS_Y),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                TILED_F32.contains(&expected),
                "TILED_F32 の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
            assert!(
                TILED_BIAS_ACT_F32.contains(&expected),
                "TILED_BIAS_ACT_F32 の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
        }
    }

    /// tiled f32 レジスタブロッキング構成の整合条件（実装計画 §3.1）を
    /// 検査する: スレッドブロック次元がタイル/レジスタタイルの比と一致
    /// し、ロード段階のスレッド分担（`(BM*BK)/256`・`(BN*BK)/256`）が
    /// 割り切れること。この不変条件が崩れるとカーネル内 `#pragma unroll`
    /// ループの反復回数と実スレッド数がずれ、一部要素が未ロード・未計算
    /// のまま静かに誤った結果を生成しうるため、定数変更時に機械検出する。
    #[test]
    fn tiled_f32_constants_satisfy_thread_and_tile_invariants() {
        assert_eq!(
            TILED_F32_BM % TILED_F32_TM,
            0,
            "BM は TM で割り切れる必要がある"
        );
        assert_eq!(
            TILED_F32_BN % TILED_F32_TN,
            0,
            "BN は TN で割り切れる必要がある"
        );
        assert_eq!(
            TILED_F32_THREADS_X,
            TILED_F32_BN / TILED_F32_TN,
            "THREADS_X は BN/TN と一致する必要がある"
        );
        assert_eq!(
            TILED_F32_THREADS_Y,
            TILED_F32_BM / TILED_F32_TM,
            "THREADS_Y は BM/TM と一致する必要がある"
        );
        let threads = TILED_F32_THREADS_X * TILED_F32_THREADS_Y;
        assert_eq!(
            threads, 256,
            "スレッドブロックは 256 スレッド（16x16）を前提とする"
        );
        assert_eq!(
            (TILED_F32_BM * TILED_F32_BK) % threads,
            0,
            "A タイルのロード要素数はスレッド数で割り切れる必要がある"
        );
        assert_eq!(
            (TILED_F32_BN * TILED_F32_BK) % threads,
            0,
            "B タイルのロード要素数はスレッド数で割り切れる必要がある"
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
}
