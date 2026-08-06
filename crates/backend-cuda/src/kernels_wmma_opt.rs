//! WMMA（Tensor Core）GEMM の共有メモリ・タイル最適化版カーネルソース
//! （TASK-11.1d・#63。NVRTC 実行時コンパイル用の静的文字列）。
//!
//! `kernels.rs::WMMA_TF32_F32`（#62）・`kernels_wmma.rs::WMMA_F16`（#61）は
//! いずれもブロックタイル = warp タイル = fragment タイルという「warp あたり
//! fragment 1 個のみ」の安全側最小構成であり、Tensor Core の計算密度を
//! 活かせていない（両ファイル冒頭ドキュメントコメント「タイル拡大・
//! warp あたり複数 fragment 化は #63 のスコープ」参照）。本モジュールは
//! `docs/cuda-tensor-core-design.md` 4.2 節が #63 に引き継いだ最適化
//! （タイル拡大・レジスタブロッキング・バンクコンフリクト回避パディング・
//! ダブルバッファリング）を実装する。
//!
//! `gemm.rs`／`gemm_wmma.rs`（呼び出し元）は本モジュールの定数・カーネルを
//! `nvrtc::compile_ptx` に渡して `CudaFunction` を得る。既存カーネルと同じく
//! ソースを `nvcc` で事前コンパイルせず文字列のまま埋め込み、ビルド時に
//! nvcc／CUDA ヘッダを一切要求しない契約（TASK-1.7・
//! `.claude/rules/deps-policy.md`）を維持する。
//!
//! # 公開 API への影響（フォールバック方針）
//!
//! REQ-11 は明示切替 API を提供しない方針（設計メモ 10 節）のため、公開 API
//! （`CudaGemm::run_wmma_tf32`／`CudaWmmaGemm::run_f16`）のシグネチャは変更
//! しない。本モジュールのカーネルは `gemm.rs`／`gemm_wmma.rs` 側で
//! 「opt カーネルが `CudaGemm::new`／`CudaWmmaGemm::new` 時点でコンパイル・
//! ロードに成功していればそちらを優先し、失敗していれば #61/#62 の基本
//! WMMA カーネルへ自動フォールバックする」という `Option` パターン
//! （`kernels.rs::WMMA_TF32_F32` の `wmma_tf32`／`wmma_tf32_error` と同方式）
//! で結線される。opt カーネルのコンパイル失敗は naive／tiled／基本 WMMA の
//! 可用性を道連れにしない。
//!
//! # サンドボックス制約と安全側設計
//!
//! このモジュールの CUDA C++ ソースは、CUDA toolkit／実機が存在しない
//! サンドボックス環境で作成されており、NVRTC による実コンパイル検証が
//! できない（#61 冒頭ドキュメントコメントと同じ制約）。緩和策:
//!
//! 1. opt カーネルは基本 WMMA カーネルと独立にコンパイル・ロードし、
//!    失敗しても `Option::None` として扱い基本カーネルへフォールバックする
//!    （上記「公開 API への影響」参照）。
//! 2. タイル・パディング定数は `#define` パラメータ化し、Rust 側の定数を
//!    「唯一の真実源」として文字列突合テストでロックする（`kernels.rs::
//!    tile_constant_matches_kernel_source_define` と同じ方式。実機での
//!    チューニング（#64）を小差分化する）。
//! 3. 不変条件（ブロックタイルが fragment 辺の倍数・スレッド数が warp 数
//!    ×32 に一致等）は `gemm.rs`／`gemm_wmma.rs` 側の `const` アサーション
//!    で機械検査する（`kernels.rs::WMMA_TF32_BLOCK_M` 系と同じ方式）。
//!
//! # タイル構成
//!
//! TF32 opt・f16 opt とも共通してブロックタイル 64×64・warp タイル 32×32
//! （2×2 warp グリッド、4 warp = 128 スレッド）・warp あたり `m16n16k*`
//! fragment 2×2 個（レジスタブロッキング）を採用する。K タイル幅・共有
//! メモリのパディング幅は各カーネルのドキュメンテーションコメント参照。
//!
//! # ダブルバッファリング
//!
//! A／B の共有メモリタイルを 2 面確保し、現在の K タイル（`cur`）の
//! `mma_sync` 計算と次の K タイル（`nxt`）のグローバル→共有メモリ
//! プリフェッチを同一ループ本体内で発行する。`cur`／`nxt` は互いに独立した
//! 配列要素（`tile[cur]`／`tile[nxt]`）を指すため、プリフェッチ書き込みと
//! 計算読み出しの間にレースは生じない。ループ末尾の 1 回の `__syncthreads()`
//! が「今回の `cur` 読み出し（計算）」と「今回の `nxt` 書き込み
//! （プリフェッチ）」の両方の完了を保証したうえで `cur`/`nxt` を入れ替える
//! （標準的な 2 段パイプラインの契約。`cp.async` 等の非同期コピー命令は
//! 使わず `__syncthreads()` ベースに限定する。#187 のスコープ外事項）。
//!
//! # 境界検査（REQ-8。省略禁止）
//!
//! 1. **guarded load**: グローバル→共有メモリのロードは全て
//!    `(gr < 境界) ? ... : 0` の三項ガード＋ゼロ充填を維持する
//!    （`kernels.rs::TILED_F32`・`kernels_wmma.rs::WMMA_F16` と同方式）。
//! 2. **エピローグ**: `store_matrix_sync` は fragment 全体を無条件に書く
//!    ため、いったん warp 専有の共有メモリ `c_tile`／`cs_tile` へ store し、
//!    `__syncthreads()` 後に要素単位のガード付きコピーでグローバル C へ
//!    書き戻す（`kernels.rs::WMMA_TF32_F32` エピローグと同方式）。
//! 3. K 端（`k` が K タイル幅の倍数でない）は `num_k_tiles` を桁溢れしない
//!    式 `(k > 0) ? (k - 1) / K_TILE + 1 : 0`（`kernels.rs::TILED_F32` と
//!    同一）で計算し、末尾タイルの余剰要素は guarded load のゼロ充填で
//!    処理する。
//!
//! # アライメント
//!
//! `load_matrix_sync`／`store_matrix_sync` が要求するのは要素サイズへの
//! 自然アライメント（f32: 4 byte、half: 2 byte。16 byte 境界に揃うと
//! 追加の高速パスが選択されるが必須要件ではない）。本モジュールの共有
//! メモリタイルはバンクコンフリクト回避のためパディング幅を `K_TILE`／
//! `BLOCK_N` の非 2 冪数（+4／+8）に取るため、warp オフセット先頭ポインタは
//! 32 byte 境界には揃わない。これは意図した設計判断であり（パディングと
//! 32 byte アライメントは両立不可能なため、バンクコンフリクト回避を優先
//! した）、要素サイズへの自然アライメントは配列宣言の `__align__(32)`
//! （配列全体の先頭アライメント）と要素サイズの倍数であるパディング幅
//! （f32: 4 の倍数、half: 8 の倍数）により常に満たされる。

/// TF32 opt GEMM のブロックタイル一辺（M・N とも 64。2×2 warp グリッド、
/// warp あたり 32×32 = `m16n16k8` fragment 2×2 個を担当する）。
pub const WMMA_TF32_OPT_BLOCK_M: u32 = 64;
pub const WMMA_TF32_OPT_BLOCK_N: u32 = 64;

/// TF32 opt GEMM の共有メモリ K タイル幅。fragment の K 次元
/// （`WMMA_TF32_OPT_FRAG_K` = 8）の 2 倍を 1 回のロードでまとめて取得し、
/// ロード回数を半減させる（設計メモ 4.2 節「k タイル TF32: 16」候補）。
/// `mma_sync` 自体は fragment の K=8 単位で 1 K タイルあたり 2 回発行する。
pub const WMMA_TF32_OPT_K_TILE: u32 = 16;

/// TF32 opt GEMM の fragment M・N 一辺（`m16n16k8` の 16）。
pub const WMMA_TF32_OPT_FRAG: u32 = 16;

/// TF32 opt GEMM の fragment K 一辺（`m16n16k8` の 8）。
/// `WMMA_TF32_OPT_K_TILE` は必ずこの倍数でなければならない
/// （`gemm.rs` の const アサーションで検査）。
pub const WMMA_TF32_OPT_FRAG_K: u32 = 8;

/// TF32 opt GEMM の warp タイル一辺（32。fragment 辺 16 の 2 倍 =
/// レジスタブロッキング 2×2）。
pub const WMMA_TF32_OPT_WARP_TILE: u32 = 32;

/// TF32 opt GEMM 1 ブロックあたりのスレッド数（4 warp = 128 スレッド。
/// `(WMMA_TF32_OPT_BLOCK_M / WMMA_TF32_OPT_WARP_TILE) *
/// (WMMA_TF32_OPT_BLOCK_N / WMMA_TF32_OPT_WARP_TILE) * 32` = 2×2 warp を
/// 1 次元ブロックとして起動する。`kernels.rs::WMMA_TF32_THREADS` と同じ
/// 「ホスト側ブロック次元とカーネル内 warp グリッドの 1:1 対応」契約）。
pub const WMMA_TF32_OPT_THREADS: u32 = 128;

/// A タイル（`as_tile[2][BLOCK_M][A_PAD]`）の行幅（パディング後）。
/// `K_TILE`（16）に 4 要素加算し、f32 の `ldm` 制約（4 の倍数）を保ちながら
/// バンクコンフリクトを避ける（設計メモ 4.2 節・本ファイル冒頭
/// ドキュメンテーションコメント「アライメント」参照）。
pub const WMMA_TF32_OPT_A_PAD: u32 = WMMA_TF32_OPT_K_TILE + 4;

/// B タイル（`bs_tile[2][K_TILE][B_PAD]`）の行幅（パディング後）。
/// `BLOCK_N`（64）に 4 要素加算する。A パディングと同じ根拠。
pub const WMMA_TF32_OPT_B_PAD: u32 = WMMA_TF32_OPT_BLOCK_N + 4;

/// WMMA（Tensor Core）を用いた TF32 GEMM の共有メモリ・タイル最適化版
/// （TASK-11.1d・#63）。`kernels::WMMA_TF32_F32`（#62。warp あたり fragment
/// 1 個のみ）に対し、ブロックタイル 64×64・warp あたり fragment 2×2 個
/// （レジスタブロッキング）・バンクコンフリクト回避パディング・
/// ダブルバッファリングを適用する（本ファイル冒頭ドキュメントコメント
/// 参照）。数値契約（TF32 丸め・f32 累算）は `kernels::WMMA_TF32_F32` と
/// 同一（`wmma::__float_to_tf32` による明示変換。統一複合判定の閾値は
/// 変更しない）。
///
/// 受け入れ条件（#63）: tiled 実装（1.832 TFLOPS、PoC-v2-3、M=N=K=4096
/// の f32）を上回る実測（5 回中央値）。実測確定は #64（実機チューニング）
/// に引き継ぐ（本カーネルはサンドボックス環境でコンパイル未検証。上記
/// 「サンドボックス制約と安全側設計」参照）。
pub const WMMA_TF32_F32_OPT: &str = r#"
#include <mma.h>

using namespace nvcuda;

#define WMMA_TF32_OPT_BLOCK_M 64
#define WMMA_TF32_OPT_BLOCK_N 64
#define WMMA_TF32_OPT_K_TILE 16
#define WMMA_TF32_OPT_FRAG 16
#define WMMA_TF32_OPT_FRAG_K 8
#define WMMA_TF32_OPT_WARP_TILE 32
#define WMMA_TF32_OPT_THREADS 128
#define WMMA_TF32_OPT_A_PAD 20
#define WMMA_TF32_OPT_B_PAD 68
#define WMMA_TF32_OPT_FRAG_ROWS 2
#define WMMA_TF32_OPT_FRAG_COLS 2
#define WMMA_TF32_OPT_K_SUBSTEPS 2

extern "C" __global__ void gemm_wmma_tf32_opt(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // ダブルバッファ（cur/nxt）の A/B タイル。バンクコンフリクト回避の
    // ため行幅を K_TILE/BLOCK_N ちょうどではなく +4 要素パディングする
    // （本ファイル冒頭ドキュメントコメント「アライメント」参照）。
    __shared__ __align__(32) float as_tile[2][WMMA_TF32_OPT_BLOCK_M][WMMA_TF32_OPT_A_PAD];
    __shared__ __align__(32) float bs_tile[2][WMMA_TF32_OPT_K_TILE][WMMA_TF32_OPT_B_PAD];

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int block_row_base = blockIdx.y * WMMA_TF32_OPT_BLOCK_M;
    const int block_col_base = blockIdx.x * WMMA_TF32_OPT_BLOCK_N;
    const int warp_row_base = warp_row * WMMA_TF32_OPT_WARP_TILE;
    const int warp_col_base = warp_col * WMMA_TF32_OPT_WARP_TILE;

    // レジスタブロッキング: warp あたり 2x2 = 4 個の accumulator fragment
    // を直接レジスタに保持し、K タイル反復間で使い回す。
    wmma::fragment<wmma::accumulator, WMMA_TF32_OPT_FRAG, WMMA_TF32_OPT_FRAG,
                   WMMA_TF32_OPT_FRAG_K, float> c_frag[WMMA_TF32_OPT_FRAG_ROWS][WMMA_TF32_OPT_FRAG_COLS];
#pragma unroll
    for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
            wmma::fill_fragment(c_frag[fi][fj], 0.0f);
        }
    }

    // 桁溢れしない num_k_tiles 計算（kernels.rs::TILED_F32 と同じ方式）。
    int num_k_tiles = (k > 0) ? (k - 1) / WMMA_TF32_OPT_K_TILE + 1 : 0;

    int cur = 0;
    // 初回タイル（t=0）のプリフェッチ。ループ内では「今回の cur を計算
    // しつつ次の nxt をプリフェッチする」構造にするため、t=0 分だけは
    // ループに入る前に用意する。
    if (num_k_tiles > 0) {
        for (int idx = tid; idx < WMMA_TF32_OPT_BLOCK_M * WMMA_TF32_OPT_K_TILE; idx += num_threads) {
            int lr = idx / WMMA_TF32_OPT_K_TILE;
            int lc = idx % WMMA_TF32_OPT_K_TILE;
            int gr = block_row_base + lr;
            int gc = lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            as_tile[cur][lr][lc] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0f;
        }
        for (int idx = tid; idx < WMMA_TF32_OPT_K_TILE * WMMA_TF32_OPT_BLOCK_N; idx += num_threads) {
            int lr = idx / WMMA_TF32_OPT_BLOCK_N;
            int lc = idx % WMMA_TF32_OPT_BLOCK_N;
            int gr = lr;
            int gc = block_col_base + lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            bs_tile[cur][lr][lc] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0f;
        }
    }
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        int nxt = cur ^ 1;

        // 次タイルのプリフェッチ（nxt バッファへ書く。cur バッファの
        // 計算読み出しとは独立したメモリ領域のためレースしない。本ファイル
        // 冒頭ドキュメントコメント「ダブルバッファリング」参照）。
        if (t + 1 < num_k_tiles) {
            int k_base_next = (t + 1) * WMMA_TF32_OPT_K_TILE;
            for (int idx = tid; idx < WMMA_TF32_OPT_BLOCK_M * WMMA_TF32_OPT_K_TILE; idx += num_threads) {
                int lr = idx / WMMA_TF32_OPT_K_TILE;
                int lc = idx % WMMA_TF32_OPT_K_TILE;
                int gr = block_row_base + lr;
                int gc = k_base_next + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                as_tile[nxt][lr][lc] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0f;
            }
            for (int idx = tid; idx < WMMA_TF32_OPT_K_TILE * WMMA_TF32_OPT_BLOCK_N; idx += num_threads) {
                int lr = idx / WMMA_TF32_OPT_BLOCK_N;
                int lc = idx % WMMA_TF32_OPT_BLOCK_N;
                int gr = k_base_next + lr;
                int gc = block_col_base + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                bs_tile[nxt][lr][lc] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0f;
            }
        }

        // cur バッファを用いた計算: K_TILE(16) を fragment K(8) 単位の
        // 2 サブステップに分け、各サブステップで 2x2 fragment（レジスタ
        // ブロッキング）を mma_sync する。
#pragma unroll
        for (int ks = 0; ks < WMMA_TF32_OPT_K_SUBSTEPS; ++ks) {
            int k_off = ks * WMMA_TF32_OPT_FRAG_K;

            wmma::fragment<wmma::matrix_a, WMMA_TF32_OPT_FRAG, WMMA_TF32_OPT_FRAG,
                           WMMA_TF32_OPT_FRAG_K, wmma::precision::tf32, wmma::row_major> a_frag[WMMA_TF32_OPT_FRAG_ROWS];
            wmma::fragment<wmma::matrix_b, WMMA_TF32_OPT_FRAG, WMMA_TF32_OPT_FRAG,
                           WMMA_TF32_OPT_FRAG_K, wmma::precision::tf32, wmma::row_major> b_frag[WMMA_TF32_OPT_FRAG_COLS];

#pragma unroll
            for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
                wmma::load_matrix_sync(
                    a_frag[fi],
                    &as_tile[cur][warp_row_base + fi * WMMA_TF32_OPT_FRAG][k_off],
                    WMMA_TF32_OPT_A_PAD);
#pragma unroll
                for (int e = 0; e < a_frag[fi].num_elements; ++e) {
                    a_frag[fi].x[e] = wmma::__float_to_tf32(a_frag[fi].x[e]);
                }
            }
#pragma unroll
            for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
                wmma::load_matrix_sync(
                    b_frag[fj],
                    &bs_tile[cur][k_off][warp_col_base + fj * WMMA_TF32_OPT_FRAG],
                    WMMA_TF32_OPT_B_PAD);
#pragma unroll
                for (int e = 0; e < b_frag[fj].num_elements; ++e) {
                    b_frag[fj].x[e] = wmma::__float_to_tf32(b_frag[fj].x[e]);
                }
            }

#pragma unroll
            for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
                for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
                    wmma::mma_sync(c_frag[fi][fj], a_frag[fi], b_frag[fj], c_frag[fi][fj]);
                }
            }
        }

        // 今回の cur 読み出し（計算）と今回の nxt 書き込み（プリフェッチ）
        // の両方の完了を待ってから cur/nxt を入れ替える（本ファイル冒頭
        // ドキュメントコメント「ダブルバッファリング」参照）。
        __syncthreads();
        cur = nxt;
    }

    // REQ-8: エピローグ store のガード条件。store_matrix_sync は fragment
    // 全体（16x16）を無条件で書くため、共有メモリへ一旦 store したうえで
    // 要素単位のガード付きコピーによりグローバル C への範囲外書き込みを防ぐ
    // （kernels.rs::WMMA_TF32_F32 エピローグと同方式）。
    __shared__ __align__(32) float c_tile[WMMA_TF32_OPT_BLOCK_M][WMMA_TF32_OPT_BLOCK_N];
#pragma unroll
    for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
            wmma::store_matrix_sync(
                &c_tile[warp_row_base + fi * WMMA_TF32_OPT_FRAG][warp_col_base + fj * WMMA_TF32_OPT_FRAG],
                c_frag[fi][fj], WMMA_TF32_OPT_BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < WMMA_TF32_OPT_BLOCK_M * WMMA_TF32_OPT_BLOCK_N; idx += num_threads) {
        int lr = idx / WMMA_TF32_OPT_BLOCK_N;
        int lc = idx % WMMA_TF32_OPT_BLOCK_N;
        int gr = block_row_base + lr;
        int gc = block_col_base + lc;
        if (gr < m && gc < n) {
            c[gr * n + gc] = c_tile[lr][lc];
        }
    }
}
"#;

/// f16 opt GEMM のブロックタイル一辺（M・N とも 64。TF32 opt と同じ 2×2
/// warp グリッド・warp あたり fragment 2×2 個構成）。
pub const WMMA_F16_OPT_BLOCK_M: u32 = 64;
pub const WMMA_F16_OPT_BLOCK_N: u32 = 64;

/// f16 opt GEMM の fragment M・N・K 一辺（`m16n16k16` の 16）。f16 fragment
/// は K=16 のため、TF32 opt と異なり共有メモリ K タイル幅とサブステップ
/// 分割が不要（1 ロード = 1 `mma_sync` 入力。`kernels_wmma.rs::WMMA_TILE`
/// と同じ値）。
pub const WMMA_F16_OPT_FRAG: u32 = 16;

/// f16 opt GEMM の共有メモリ K タイル幅（fragment K と同じ 16）。
pub const WMMA_F16_OPT_K_TILE: u32 = WMMA_F16_OPT_FRAG;

/// f16 opt GEMM の warp タイル一辺（32。TF32 opt と同じレジスタブロッキング
/// 2×2）。
pub const WMMA_F16_OPT_WARP_TILE: u32 = 32;

/// f16 opt GEMM 1 ブロックあたりのスレッド数（4 warp = 128 スレッド。
/// TF32 opt と同じ 2×2 warp グリッド構成）。
pub const WMMA_F16_OPT_THREADS: u32 = 128;

/// A タイル（`as_tile[2][BLOCK_M][A_PAD]`）の行幅（パディング後）。
/// `K_TILE`（16）に 8 要素加算し、half の `ldm` 制約（8 の倍数）を保ちながら
/// バンクコンフリクトを避ける（`kernels_wmma.rs` 冒頭ドキュメントコメント
/// 「ldm 制約」参照）。
pub const WMMA_F16_OPT_A_PAD: u32 = WMMA_F16_OPT_K_TILE + 8;

/// B タイル（`bs_tile[2][K_TILE][B_PAD]`）の行幅（パディング後）。
/// `BLOCK_N`（64）に 8 要素加算する。A パディングと同じ根拠。
pub const WMMA_F16_OPT_B_PAD: u32 = WMMA_F16_OPT_BLOCK_N + 8;

/// f16 WMMA GEMM の共有メモリ・タイル最適化版（TASK-11.1d・#63）。
/// `kernels_wmma::WMMA_F16`（#61。1 ブロック = 1 warp = fragment 1 個のみ）
/// に対し、ブロックタイル 64×64・warp あたり fragment 2×2 個（レジスタ
/// ブロッキング）・バンクコンフリクト回避パディング・ダブルバッファ
/// リングを適用する。数値契約（f16 入出力・f32 累算）は
/// `kernels_wmma::WMMA_F16` と同一。
pub const WMMA_F16_OPT: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_F16_OPT_BLOCK_M 64
#define WMMA_F16_OPT_BLOCK_N 64
#define WMMA_F16_OPT_K_TILE 16
#define WMMA_F16_OPT_FRAG 16
#define WMMA_F16_OPT_WARP_TILE 32
#define WMMA_F16_OPT_THREADS 128
#define WMMA_F16_OPT_A_PAD 24
#define WMMA_F16_OPT_B_PAD 72
#define WMMA_F16_OPT_FRAG_ROWS 2
#define WMMA_F16_OPT_FRAG_COLS 2

extern "C" __global__ void gemm_wmma_f16_opt(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m, int n, int k)
{
    __shared__ __align__(32) __half as_tile[2][WMMA_F16_OPT_BLOCK_M][WMMA_F16_OPT_A_PAD];
    __shared__ __align__(32) __half bs_tile[2][WMMA_F16_OPT_K_TILE][WMMA_F16_OPT_B_PAD];

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int block_row_base = blockIdx.y * WMMA_F16_OPT_BLOCK_M;
    const int block_col_base = blockIdx.x * WMMA_F16_OPT_BLOCK_N;
    const int warp_row_base = warp_row * WMMA_F16_OPT_WARP_TILE;
    const int warp_col_base = warp_col * WMMA_F16_OPT_WARP_TILE;

    wmma::fragment<wmma::accumulator, WMMA_F16_OPT_FRAG, WMMA_F16_OPT_FRAG,
                   WMMA_F16_OPT_FRAG, float> c_frag[WMMA_F16_OPT_FRAG_ROWS][WMMA_F16_OPT_FRAG_COLS];
#pragma unroll
    for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
            wmma::fill_fragment(c_frag[fi][fj], 0.0f);
        }
    }

    // 桁溢れしない num_k_tiles 計算（kernels.rs::TILED_F32 と同じ方式）。
    int num_k_tiles = (k > 0) ? (k - 1) / WMMA_F16_OPT_K_TILE + 1 : 0;

    int cur = 0;
    if (num_k_tiles > 0) {
        for (int idx = tid; idx < WMMA_F16_OPT_BLOCK_M * WMMA_F16_OPT_K_TILE; idx += num_threads) {
            int lr = idx / WMMA_F16_OPT_K_TILE;
            int lc = idx % WMMA_F16_OPT_K_TILE;
            int gr = block_row_base + lr;
            int gc = lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            as_tile[cur][lr][lc] = (gr < m && gc < k) ? a[gr * k + gc] : __float2half(0.0f);
        }
        for (int idx = tid; idx < WMMA_F16_OPT_K_TILE * WMMA_F16_OPT_BLOCK_N; idx += num_threads) {
            int lr = idx / WMMA_F16_OPT_BLOCK_N;
            int lc = idx % WMMA_F16_OPT_BLOCK_N;
            int gr = lr;
            int gc = block_col_base + lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            bs_tile[cur][lr][lc] = (gr < k && gc < n) ? b[gr * n + gc] : __float2half(0.0f);
        }
    }
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        int nxt = cur ^ 1;

        // 次タイルのプリフェッチ。kernels_wmma_opt.rs::WMMA_TF32_F32_OPT
        // と同じダブルバッファ契約（本ファイルの Rust 側ドキュメンテーション
        // コメント「ダブルバッファリング」参照）。
        if (t + 1 < num_k_tiles) {
            int k_base_next = (t + 1) * WMMA_F16_OPT_K_TILE;
            for (int idx = tid; idx < WMMA_F16_OPT_BLOCK_M * WMMA_F16_OPT_K_TILE; idx += num_threads) {
                int lr = idx / WMMA_F16_OPT_K_TILE;
                int lc = idx % WMMA_F16_OPT_K_TILE;
                int gr = block_row_base + lr;
                int gc = k_base_next + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                as_tile[nxt][lr][lc] = (gr < m && gc < k) ? a[gr * k + gc] : __float2half(0.0f);
            }
            for (int idx = tid; idx < WMMA_F16_OPT_K_TILE * WMMA_F16_OPT_BLOCK_N; idx += num_threads) {
                int lr = idx / WMMA_F16_OPT_BLOCK_N;
                int lc = idx % WMMA_F16_OPT_BLOCK_N;
                int gr = k_base_next + lr;
                int gc = block_col_base + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                bs_tile[nxt][lr][lc] = (gr < k && gc < n) ? b[gr * n + gc] : __float2half(0.0f);
            }
        }

        wmma::fragment<wmma::matrix_a, WMMA_F16_OPT_FRAG, WMMA_F16_OPT_FRAG,
                       WMMA_F16_OPT_FRAG, __half, wmma::row_major> a_frag[WMMA_F16_OPT_FRAG_ROWS];
        wmma::fragment<wmma::matrix_b, WMMA_F16_OPT_FRAG, WMMA_F16_OPT_FRAG,
                       WMMA_F16_OPT_FRAG, __half, wmma::row_major> b_frag[WMMA_F16_OPT_FRAG_COLS];

#pragma unroll
        for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
            wmma::load_matrix_sync(
                a_frag[fi],
                &as_tile[cur][warp_row_base + fi * WMMA_F16_OPT_FRAG][0],
                WMMA_F16_OPT_A_PAD);
        }
#pragma unroll
        for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
            wmma::load_matrix_sync(
                b_frag[fj],
                &bs_tile[cur][0][warp_col_base + fj * WMMA_F16_OPT_FRAG],
                WMMA_F16_OPT_B_PAD);
        }

#pragma unroll
        for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
            for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
                wmma::mma_sync(c_frag[fi][fj], a_frag[fi], b_frag[fj], c_frag[fi][fj]);
            }
        }

        // 今回の cur 読み出し（計算）と今回の nxt 書き込み（プリフェッチ）
        // の両方の完了を待ってから cur/nxt を入れ替える。
        __syncthreads();
        cur = nxt;
    }

    // REQ-8: エピローグ store のガード条件（kernels_wmma.rs::WMMA_F16
    // エピローグと同方式）。
    __shared__ __align__(32) float cs_tile[WMMA_F16_OPT_BLOCK_M][WMMA_F16_OPT_BLOCK_N];
#pragma unroll
    for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
            wmma::store_matrix_sync(
                &cs_tile[warp_row_base + fi * WMMA_F16_OPT_FRAG][warp_col_base + fj * WMMA_F16_OPT_FRAG],
                c_frag[fi][fj], WMMA_F16_OPT_BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < WMMA_F16_OPT_BLOCK_M * WMMA_F16_OPT_BLOCK_N; idx += num_threads) {
        int lr = idx / WMMA_F16_OPT_BLOCK_N;
        int lc = idx % WMMA_F16_OPT_BLOCK_N;
        int gr = block_row_base + lr;
        int gc = block_col_base + lc;
        if (gr < m && gc < n) {
            c[gr * n + gc] = __float2half(cs_tile[lr][lc]);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// `WMMA_TF32_OPT_*`（Rust 側の「唯一の真実源」）が `WMMA_TF32_F32_OPT`
    /// カーネルソース内の `#define` と食い違わないことを検査する
    /// （`kernels.rs::wmma_tf32_constants_match_kernel_source_defines` と
    /// 同じ方式）。
    #[test]
    fn wmma_tf32_opt_constants_match_kernel_source_defines() {
        let checks = [
            ("WMMA_TF32_OPT_BLOCK_M", WMMA_TF32_OPT_BLOCK_M),
            ("WMMA_TF32_OPT_BLOCK_N", WMMA_TF32_OPT_BLOCK_N),
            ("WMMA_TF32_OPT_K_TILE", WMMA_TF32_OPT_K_TILE),
            ("WMMA_TF32_OPT_FRAG", WMMA_TF32_OPT_FRAG),
            ("WMMA_TF32_OPT_FRAG_K", WMMA_TF32_OPT_FRAG_K),
            ("WMMA_TF32_OPT_WARP_TILE", WMMA_TF32_OPT_WARP_TILE),
            ("WMMA_TF32_OPT_THREADS", WMMA_TF32_OPT_THREADS),
            ("WMMA_TF32_OPT_A_PAD", WMMA_TF32_OPT_A_PAD),
            ("WMMA_TF32_OPT_B_PAD", WMMA_TF32_OPT_B_PAD),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                WMMA_TF32_F32_OPT.contains(&expected),
                "WMMA_TF32_F32_OPT の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
        }
    }

    /// `WMMA_F16_OPT_*`（Rust 側の「唯一の真実源」）が `WMMA_F16_OPT`
    /// カーネルソース内の `#define` と食い違わないことを検査する。
    #[test]
    fn wmma_f16_opt_constants_match_kernel_source_defines() {
        let checks = [
            ("WMMA_F16_OPT_BLOCK_M", WMMA_F16_OPT_BLOCK_M),
            ("WMMA_F16_OPT_BLOCK_N", WMMA_F16_OPT_BLOCK_N),
            ("WMMA_F16_OPT_K_TILE", WMMA_F16_OPT_K_TILE),
            ("WMMA_F16_OPT_FRAG", WMMA_F16_OPT_FRAG),
            ("WMMA_F16_OPT_WARP_TILE", WMMA_F16_OPT_WARP_TILE),
            ("WMMA_F16_OPT_THREADS", WMMA_F16_OPT_THREADS),
            ("WMMA_F16_OPT_A_PAD", WMMA_F16_OPT_A_PAD),
            ("WMMA_F16_OPT_B_PAD", WMMA_F16_OPT_B_PAD),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                WMMA_F16_OPT.contains(&expected),
                "WMMA_F16_OPT の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
        }
    }

    /// TASK-11.3（tensor core 命令使用の証跡）を兼ねる。
    /// `kernels_wmma.rs::wmma_f16_source_uses_wmma_instructions` と同方式。
    #[test]
    fn wmma_tf32_opt_source_uses_wmma_instructions() {
        for needle in [
            "#include <mma.h>",
            "wmma::fragment",
            "wmma::load_matrix_sync",
            "wmma::mma_sync",
            "wmma::store_matrix_sync",
            "wmma::fill_fragment",
            "wmma::__float_to_tf32",
        ] {
            assert!(
                WMMA_TF32_F32_OPT.contains(needle),
                "WMMA_TF32_F32_OPT に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    #[test]
    fn wmma_f16_opt_source_uses_wmma_instructions() {
        for needle in [
            "#include <mma.h>",
            "wmma::fragment",
            "wmma::load_matrix_sync",
            "wmma::mma_sync",
            "wmma::store_matrix_sync",
            "wmma::fill_fragment",
        ] {
            assert!(
                WMMA_F16_OPT.contains(needle),
                "WMMA_F16_OPT に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// REQ-8: guarded load／guarded store がソースから除去されていない
    /// ことをロックする（`kernels_wmma.rs::wmma_f16_source_retains_req8_boundary_guards`
    /// と同方式）。
    #[test]
    fn wmma_tf32_opt_source_retains_req8_boundary_guards() {
        for needle in ["gr < m && gc < k", "gr < k && gc < n", "gr < m && gc < n"] {
            assert!(
                WMMA_TF32_F32_OPT.contains(needle),
                "WMMA_TF32_F32_OPT に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    #[test]
    fn wmma_f16_opt_source_retains_req8_boundary_guards() {
        for needle in ["gr < m && gc < k", "gr < k && gc < n", "gr < m && gc < n"] {
            assert!(
                WMMA_F16_OPT.contains(needle),
                "WMMA_F16_OPT に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    /// ダブルバッファリング（`__syncthreads()` ベースの 2 段パイプライン）
    /// がソースから除去されていないことをロックする。プリフェッチ分岐
    /// （`t + 1 < num_k_tiles`）と `cur`/`nxt` の入れ替えは #63 の受け入れ
    /// 条件（tiled 実装超過）を支える主要最適化のため、実装が回帰的に
    /// 単純化（例: ダブルバッファ除去）された場合に検出する。
    #[test]
    fn wmma_opt_sources_retain_double_buffering() {
        for src in [WMMA_TF32_F32_OPT, WMMA_F16_OPT] {
            assert!(
                src.contains("cur ^ 1"),
                "ダブルバッファの cur/nxt 切替が見つかりません"
            );
            assert!(
                src.contains("t + 1 < num_k_tiles"),
                "ダブルバッファのプリフェッチ分岐が見つかりません"
            );
        }
    }
}
