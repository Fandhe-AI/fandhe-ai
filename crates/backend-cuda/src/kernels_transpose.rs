//! smem パディング + スウィズルによる転置カーネルの CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列。イシュー #601・親 #582 の
//! G-11）。
//!
//! `transpose.rs`（呼び出し元）は本モジュールの定数・関数を
//! `nvrtc::compile_ptx` に渡し `CudaFunction` を得る。他のカーネルモジュール
//! （`kernels.rs`・`kernels_mma.rs`）と同じく、ソースを `nvcc` で事前
//! コンパイルせず文字列のまま埋め込むことで「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約（`.claude/rules/deps-policy.md`）
//! を維持する。
//!
//! # アルゴリズムの出自
//!
//! 親イシュー #582（Phase G）が TileLang の転置カーネル群から抽出した
//! アルゴリズム（TileLang 自体には依存しない。整数式のみを NVRTC カーネル
//! へ手動転写した独立実装）。要点は 2 つ:
//!
//! 1. **smem パディング**: `tile[TILE][TILE + PAD]` の形で共有メモリタイルの
//!    行幅をタイル一辺より広く確保し、行間のバンク位相をずらしてバンク
//!    コンフリクトを回避する（`kernels_mma.rs::MMA_A_PAD`/`MMA_B_PAD` と
//!    同じ設計思想。本ファイルの [`SMEM_PAD_F32`]/[`SMEM_PAD_F16`] 参照）。
//! 2. **dtype 依存スウィズル**: 行ごとに列インデックスを XOR で並べ替える
//!    ことで、smem タイル自体の物理配置を変えずにバンク衝突を分散する
//!    （[`transpose::swizzled_smem_col`](crate::transpose::swizzled_smem_col)
//!    がホスト側の唯一の参照実装。カーネル側は同一整数式を独立した文字列
//!    として保持し、`swizzle.rs` と同じく needle テストで不一致を検出する）。
//!
//! # 検証状態（実装計画 2 節「実行環境制約」）
//!
//! 本セッションのホストは RTX 3060（sm_86）・NVRTC 非搭載のため、NVRTC
//! コンパイル・実機実行・nsight-compute 計測は到達不能。#498・#499・#688
//! と同一の先例に従い、GPU 非依存の単体テスト（needle・smem サイズ
//! assert）までを本 PR の完了条件とし、実機 A/B・採否確定は実機セッション
//! （#408 系／G-12 #602）へ引き継ぐ（`docs/perf/cuda-gemm-transpose-ab.md`
//! 参照）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! naive カーネルは `if (row < m && col < n)` の手動境界チェックを性能上の
//! 理由で省略していない（`kernels.rs::NAIVE_F32` と同じ理由: 末尾ブロックで
//! `row`/`col` が `m`/`n` を超えるスレッドが必ず発生するため）。smem 版も
//! ロード側（`src` 範囲内のみ smem へ書く）・ストア側（`dst` 範囲内のみ書く）
//! の両方で手動境界チェックを維持する（`kernels.rs::TILED_F32` の三項ガード
//! と同じ方式）。

use crate::kernels;

/// smem 転置タイルの一辺（32×32。`kernels::TILE` と同値だが、転置カーネルは
/// GEMM のアキュムレーションと無関係な独立モジュールのため、値の変更経路を
/// 分離する意図で専用定数として持つ）。
pub const TRANSPOSE_TILE: u32 = 32;

/// f32 版 smem タイルのパディング要素数。
///
/// 32×32 の f32 タイル（4 バイト要素）は無パディングだと行幅 `32*4=128`
/// バイト = ちょうど 32 バンク（1 バンク 4 バイト）分となり、列方向アクセス
/// （転置ストア時の `tile[threadIdx.x][k]`）で全行が同一バンク位相を踏み
/// 32-way バンクコンフリクトを起こす。行幅を 1 要素（4 バイト = 1 バンク）
/// 広げると `33*4=132` バイトとなり 128 の倍数から外れるため、行ごとに
/// バンク位相が 1 ずつずれて衝突が解消される（CUDA の定番パディング技法。
/// `kernels_mma.rs::MMA_A_PAD` と同じ「128 バイトの倍数を外す」判断軸）。
pub const SMEM_PAD_F32: u32 = 1;

/// f16 版 smem タイルのパディング要素数。
///
/// f16（2 バイト要素）は 1 要素のパディングでは位相シフトが半バンク
/// （2 バイト）に留まり、4 バイト境界のバンク単位では不十分（同一バンクの
/// 異なる半分を踏む 2 要素は依然として同一トランザクションで処理されうる）。
/// 1 バンク分（4 バイト）を確実にシフトするには `ceil(4 / 要素バイト数)`
/// 要素のパディングが必要であり、f16 では `4 / 2 = 2` 要素となる
/// （f32 の `SMEM_PAD_F32 = ceil(4/4) = 1` と同一の一般式）。
pub const SMEM_PAD_F16: u32 = 2;

/// dtype 依存スウィズルの周期（要素数）。
///
/// TileLang 由来の一般形「周期 = 8 / dtype バイト数」をそのまま採用する
/// （実装計画 3.1 節）。f32（4 バイト）は周期 2、f16（2 バイト）は周期 4。
/// 周期はいずれも 2 のべき乗であり、[`transpose::swizzled_smem_col`
/// ](crate::transpose::swizzled_smem_col) の XOR ベース実装（下限ビットの
/// みを反転する）が全単射になるための前提を満たす（同関数のドキュメント
/// コメント・単体テスト `swizzle_is_bijective_per_row` 参照）。
pub const SWIZZLE_PERIOD_F32: u32 = 2;
pub const SWIZZLE_PERIOD_F16: u32 = 4;

/// 素朴転置（f32）。1 スレッド = 1 要素。smem 版との A/B 計測の基準点
/// （実装計画 3.1 節・3.4 節）。
pub const TRANSPOSE_NAIVE_F32: &str = r#"
extern "C" __global__ void transpose_naive_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        dst[col * m + row] = src[row * n + col];
    }
}
"#;

/// 素朴転置（f16）。`TRANSPOSE_NAIVE_F32` の f16 版。転置は演算を伴わない
/// 純置換のため f32 版と異なりアキュムレータ精度の考慮は不要（`__half` を
/// そのままコピーする）。
pub const TRANSPOSE_NAIVE_F16: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void transpose_naive_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        dst[col * m + row] = src[row * n + col];
    }
}
"#;

/// smem タイル転置（f32）のソースを生成する。
///
/// `swizzle == false`: パディングのみ変種。`swizzle == true`:
/// パディング + dtype 依存スウィズル変種。両方を同一関数から生成すること
/// で、パディング単体の効果とスウィズル追加の効果を実機 A/B で分離計測
/// できるようにする（実装計画 3.1 節・3.4 節。`kernels_mma.rs::
/// mma_f16_source_with_swizzle` と同様「ソース生成関数で変種を作る」方式）。
///
/// 手順: 32×32 タイルへ coalesced に行ロード（`__syncthreads()`）→
/// 転置先（`dst`）へ coalesced に行ストア。ロード側の smem 列インデックスに
/// スウィズル式（有効時のみ）を適用することで、ストア側の読み出し
/// （`tile[threadIdx.x][...]`。列方向アクセス）でのバンク分散を狙う。
///
/// # REQ-8
///
/// ロード側（`(g_row < m && g_col < n)`）・ストア側（`(out_row < n &&
/// out_col < m)`）の両方で手動境界チェックを維持する（タイル非倍数形状で
/// 必須。モジュール冒頭コメント参照）。
pub fn transpose_smem_source_f32(swizzle: bool) -> String {
    let swizzle_define = if swizzle {
        format!(
            "#define SWIZZLE_PERIOD {SWIZZLE_PERIOD_F32}\n\
             #define SWIZZLE_MASK (SWIZZLE_PERIOD - 1)\n\
             #define SMEM_COL(col, row) \\\n\
             \x20   (((col) & ~SWIZZLE_MASK) | (((col) & SWIZZLE_MASK) ^ ((row) & SWIZZLE_MASK)))\n"
        )
    } else {
        "#define SMEM_COL(col, row) (col)\n".to_string()
    };

    format!(
        r#"
#define TILE {TRANSPOSE_TILE}
#define PAD {SMEM_PAD_F32}
{swizzle_define}
extern "C" __global__ void transpose_smem_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int m, int n)
{{
    __shared__ float tile[TILE][TILE + PAD];

    int g_row = blockIdx.y * TILE + threadIdx.y;
    int g_col = blockIdx.x * TILE + threadIdx.x;

    // REQ-8: ロード時の手動境界チェック。範囲外は smem へ書かない
    // （後段の転置ストアでも同じ threadIdx から読むため未初期化値は
    // 読み出されない。境界チェックは読み書き両方に必要）。
    if (g_row < m && g_col < n) {{
        int sc = SMEM_COL(threadIdx.x, threadIdx.y);
        tile[threadIdx.y][sc] = src[g_row * n + g_col];
    }}
    __syncthreads();

    // 転置ストア: ブロック内で行・列を入れ替えた宛先座標へ書く
    // （`dst` は n x m 行優先。`kernels.rs::TILED_F32` と同じ「タイル境界を
    // div_ceil で切り上げグリッド生成する」方針のため、末尾ブロックでは
    // out_row/out_col が n/m を超えるスレッドが必ず発生する）。
    int out_row = blockIdx.x * TILE + threadIdx.y;
    int out_col = blockIdx.y * TILE + threadIdx.x;

    // REQ-8: ストア時の手動境界チェック。
    if (out_row < n && out_col < m) {{
        int sc = SMEM_COL(threadIdx.y, threadIdx.x);
        dst[out_row * m + out_col] = tile[threadIdx.x][sc];
    }}
}}
"#,
    )
}

/// smem タイル転置（f16）のソースを生成する。[`transpose_smem_source_f32`]
/// の f16 版（`SMEM_PAD_F16`/`SWIZZLE_PERIOD_F16` を使う点のみ異なる）。
pub fn transpose_smem_source_f16(swizzle: bool) -> String {
    let swizzle_define = if swizzle {
        format!(
            "#define SWIZZLE_PERIOD {SWIZZLE_PERIOD_F16}\n\
             #define SWIZZLE_MASK (SWIZZLE_PERIOD - 1)\n\
             #define SMEM_COL(col, row) \\\n\
             \x20   (((col) & ~SWIZZLE_MASK) | (((col) & SWIZZLE_MASK) ^ ((row) & SWIZZLE_MASK)))\n"
        )
    } else {
        "#define SMEM_COL(col, row) (col)\n".to_string()
    };

    format!(
        r#"
#include <cuda_fp16.h>

#define TILE {TRANSPOSE_TILE}
#define PAD {SMEM_PAD_F16}
{swizzle_define}
extern "C" __global__ void transpose_smem_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int m, int n)
{{
    __shared__ __half tile[TILE][TILE + PAD];

    int g_row = blockIdx.y * TILE + threadIdx.y;
    int g_col = blockIdx.x * TILE + threadIdx.x;

    // REQ-8: transpose_smem_source_f32 と同じ理由。
    if (g_row < m && g_col < n) {{
        int sc = SMEM_COL(threadIdx.x, threadIdx.y);
        tile[threadIdx.y][sc] = src[g_row * n + g_col];
    }}
    __syncthreads();

    int out_row = blockIdx.x * TILE + threadIdx.y;
    int out_col = blockIdx.y * TILE + threadIdx.x;

    // REQ-8: transpose_smem_source_f32 と同じ理由。
    if (out_row < n && out_col < m) {{
        int sc = SMEM_COL(threadIdx.y, threadIdx.x);
        dst[out_row * m + out_col] = tile[threadIdx.x][sc];
    }}
}}
"#,
    )
}

/// GEMM epilogue 融合転置（f32）の smem ステージングタイルパディング。
///
/// [`SMEM_PAD_F32`] と同一の根拠（f32・32×32 タイル）だが、融合カーネル
/// （[`TILED_TRANSPOSED_F32`]）は `kernels::TILED_F32` のアキュムレーション
/// 用タイル（`as_tile`/`bs_tile`。無パディング）とは別に epilogue 専用の
/// ステージングタイルを持つため、独立した定数として分離する。
pub const EPILOGUE_TRANSPOSE_PAD_F32: u32 = 1;

/// 静的共有メモリ使用量（バイト）。`kernels::TILED_F32` のアキュムレーション
/// 用タイル（`as_tile`/`bs_tile`。`TILE*TILE*4` バイトずつ）+ epilogue
/// ステージングタイル（`TILE*(TILE+PAD)*4` バイト）の合計。
/// `kernels_mma.rs::MMA_SHARED_MEM_BYTES` と同じくコンパイル時 assert で
/// Hopper 以前の 48KiB 静的共有メモリ上限（cc に依らず保証される下限）を
/// 下回ることを機械検査する（`gemm.rs`/`CudaTranspose::new` 側の
/// `const _: () = assert!(...)` から参照）。
pub const TILED_TRANSPOSED_SHARED_MEM_BYTES: u32 = (kernels::TILE * kernels::TILE * 2
    + kernels::TILE * (kernels::TILE + EPILOGUE_TRANSPOSE_PAD_F32))
    * 4;

/// [`TRANSPOSE_TILE`]（本ファイルの転置カーネル群が使うタイル一辺）と
/// `kernels::TILE`（`TILED_TRANSPOSED_F32` のアキュムレーション部が実質
/// 依拠する GEMM 側タイル一辺）が食い違わないことをコンパイル時に検査
/// する。[`TILED_TRANSPOSED_SHARED_MEM_BYTES`] は `kernels::TILE` から
/// 計算する一方、`transpose.rs::tiled_launch_config`／
/// `validate_tiled_transposed_gemm_dims` は [`TRANSPOSE_TILE`] を使う
/// ため、両者が独立に変更されるとブロック次元と共有メモリ確保サイズの
/// 対応が静かに崩れる（advisor 指摘: 「TRANSPOSE_TILE と kernels::TILE
/// という 1 つの数値に対する 3 つの情報源」の結合を明示的に固定する）。
const _: () = assert!(TRANSPOSE_TILE == kernels::TILE);

/// GEMM epilogue で C タイルを smem 経由で転置ストアする融合 tiled GEMM
/// （f32）。C^T（n×m 行優先）を直接書き、中間バッファ C を HBM へ書かない
/// （イシュー #601 §3.3「GEMM epilogue 転置」）。
///
/// **積和ループ・アキュムレート順序は `kernels::TILED_F32` と完全同一**
/// （同じ shared memory タイリング・同じ `#pragma unroll`）。変更点は
/// epilogue の store 経路のみであり、`run_tiled_f32` の出力をホスト側で
/// 転置した結果と bit 完全一致することが期待される（FMA 契約の継承。
/// `.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」・
/// `kernels.rs::TILED_BIAS_ACT_F32` の「数値契約」節と同じ論拠）。
///
/// # REQ-8
///
/// アキュムレーション部のタイルロード時ガード・書き込み判定
/// （`row < m && col < n`）は `TILED_F32` と同一。epilogue の smem
/// ステージング（`c_tile[threadIdx.y][...]`）はこの書き込み判定の内側
/// でのみ行うため、判定を満たさないスレッドは smem へ書かず（他ブロックの
/// 転置ストア読み出しに影響しない・各ブロックが自身の smem タイルのみを
/// 参照するため無関係）、転置ストア側も `(out_row < n && out_col < m)`
/// の手動境界チェックを独立に維持する。
pub const TILED_TRANSPOSED_F32: &str = r#"
#define TILE 32
#define PAD 1

extern "C" __global__ void gemm_tiled_transposed_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c_t,
    int m, int n, int k)
{
    __shared__ float as_tile[TILE][TILE];
    __shared__ float bs_tile[TILE][TILE];
    __shared__ float c_tile[TILE][TILE + PAD];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // 桁溢れしない num_tiles 計算（kernels::TILED_F32 と同一）。
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

    // epilogue: acc を smem へステージし、ブロック内で行・列を入れ替えた
    // 宛先へ転置ストアする（c_t は n x m 行優先）。REQ-8: 書き込み判定の
    // 内側でのみ smem へ書く。
    if (row < m && col < n) {
        c_tile[threadIdx.y][threadIdx.x] = acc;
    }
    __syncthreads();

    int out_row = blockIdx.x * TILE + threadIdx.y;
    int out_col = blockIdx.y * TILE + threadIdx.x;

    // REQ-8: 転置ストア側の手動境界チェック（独立判定。上記コメント参照）。
    if (out_row < n && out_col < m) {
        c_t[out_row * m + out_col] = c_tile[threadIdx.x][threadIdx.y];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// [`TRANSPOSE_TILE`]（Rust 側の唯一の真実源）が smem カーネルソース内の
    /// `#define TILE` と食い違わないことを検査する（`kernels.rs::
    /// tile_constant_matches_kernel_source_define` と同じ理由・同じ検査
    /// 方式）。
    #[test]
    fn tile_constant_matches_kernel_source_defines() {
        let expected = format!("#define TILE {TRANSPOSE_TILE}");
        for swizzle in [false, true] {
            assert!(
                transpose_smem_source_f32(swizzle).contains(&expected),
                "transpose_smem_source_f32(swizzle={swizzle}) の `#define TILE` が \
                 TRANSPOSE_TILE（{TRANSPOSE_TILE}）と一致しません"
            );
            assert!(
                transpose_smem_source_f16(swizzle).contains(&expected),
                "transpose_smem_source_f16(swizzle={swizzle}) の `#define TILE` が \
                 TRANSPOSE_TILE（{TRANSPOSE_TILE}）と一致しません"
            );
        }
        assert!(TILED_TRANSPOSED_F32.contains(&expected));
    }

    /// [`SMEM_PAD_F32`]/[`SMEM_PAD_F16`] が対応するソース内の `#define PAD`
    /// と食い違わないことを検査する。
    #[test]
    fn pad_constants_match_kernel_source_defines() {
        let expected_f32 = format!("#define PAD {SMEM_PAD_F32}");
        let expected_f16 = format!("#define PAD {SMEM_PAD_F16}");
        for swizzle in [false, true] {
            assert!(
                transpose_smem_source_f32(swizzle).contains(&expected_f32),
                "transpose_smem_source_f32(swizzle={swizzle}) の `#define PAD` が \
                 SMEM_PAD_F32（{SMEM_PAD_F32}）と一致しません"
            );
            assert!(
                transpose_smem_source_f16(swizzle).contains(&expected_f16),
                "transpose_smem_source_f16(swizzle={swizzle}) の `#define PAD` が \
                 SMEM_PAD_F16（{SMEM_PAD_F16}）と一致しません"
            );
        }

        // advisor 指摘: EPILOGUE_TRANSPOSE_PAD_F32 は
        // TILED_TRANSPOSED_SHARED_MEM_BYTES（コンパイル時 assert が検査する
        // 共有メモリ確保サイズ）の唯一の入力だが、needle テストが無いと
        // Rust 側の値を変更してもカーネル文字列側の `#define PAD` が
        // ずれたまま気づけない（`tile_constant_matches_kernel_source_define`
        // と同じ理由でここに追加する）。
        let expected_epilogue_pad = format!("#define PAD {EPILOGUE_TRANSPOSE_PAD_F32}");
        assert!(
            TILED_TRANSPOSED_F32.contains(&expected_epilogue_pad),
            "TILED_TRANSPOSED_F32 の `#define PAD` が EPILOGUE_TRANSPOSE_PAD_F32（\
             {EPILOGUE_TRANSPOSE_PAD_F32}）と一致しません"
        );
    }

    /// スウィズル有効時のみ `#define SWIZZLE_PERIOD` とスウィズル整数式が
    /// 含まれ、無効時は含まれないことを検査する（needle テスト。
    /// `swizzle.rs`/`kernels_mma.rs` と同じ「ホスト側参照実装との不一致を
    /// 機械検出する」方針をカーネル文字列レベルで担保する）。
    #[test]
    fn swizzle_define_presence_matches_flag() {
        let period_define_f32 = format!("#define SWIZZLE_PERIOD {SWIZZLE_PERIOD_F32}");
        let period_define_f16 = format!("#define SWIZZLE_PERIOD {SWIZZLE_PERIOD_F16}");

        assert!(transpose_smem_source_f32(true).contains(&period_define_f32));
        assert!(!transpose_smem_source_f32(false).contains("SWIZZLE_PERIOD"));
        assert!(transpose_smem_source_f16(true).contains(&period_define_f16));
        assert!(!transpose_smem_source_f16(false).contains("SWIZZLE_PERIOD"));

        for swizzle in [false, true] {
            assert!(transpose_smem_source_f32(swizzle).contains("SMEM_COL"));
            assert!(transpose_smem_source_f16(swizzle).contains("SMEM_COL"));
        }
    }

    /// naive カーネルが REQ-8 の手動境界チェック（`if (row < m && col <
    /// n)`）を含むことを検査する（`kernels.rs` の同種テストと同じ「省略
    /// されていないことをソース文字列突合で機械検出する」方針）。
    #[test]
    fn naive_kernels_retain_boundary_check() {
        assert!(TRANSPOSE_NAIVE_F32.contains("if (row < m && col < n)"));
        assert!(TRANSPOSE_NAIVE_F16.contains("if (row < m && col < n)"));
    }

    /// smem カーネルがロード側・ストア側双方で手動境界チェックを維持する
    /// ことを検査する。
    #[test]
    fn smem_kernels_retain_boundary_checks() {
        for swizzle in [false, true] {
            let f32_src = transpose_smem_source_f32(swizzle);
            assert!(f32_src.contains("if (g_row < m && g_col < n)"));
            assert!(f32_src.contains("if (out_row < n && out_col < m)"));
            let f16_src = transpose_smem_source_f16(swizzle);
            assert!(f16_src.contains("if (g_row < m && g_col < n)"));
            assert!(f16_src.contains("if (out_row < n && out_col < m)"));
        }
        assert!(TILED_TRANSPOSED_F32.contains("if (row < m && col < n)"));
        assert!(TILED_TRANSPOSED_F32.contains("if (out_row < n && out_col < m)"));
    }

    /// [`TILED_TRANSPOSED_SHARED_MEM_BYTES`] が 48KiB（cc に依らず保証される
    /// 静的共有メモリ下限）を下回ることをコンパイル時 assert として検査する
    /// （`kernels_mma.rs::MMA_SHARED_MEM_BYTES` の const assert と同型。
    /// ここでは `#[cfg(test)]` 内で assert! として実行することで単体テスト
    /// からも直接検証できるようにする）。
    #[test]
    fn tiled_transposed_shared_mem_within_48kib_limit() {
        // clippy `assertions_on_constants` 対策: 両辺が const だと lint に
        // 引っかかるため、`std::hint::black_box` で非 const 値として扱う
        // （`transpose.rs::CudaTranspose::new` 側の `const _: () =
        // assert!(...)` が実際のコンパイル時検査を担い、本テストはその
        // 主張を単体テストからも直接確認する）。
        let limit = std::hint::black_box(48 * 1024u32);
        assert!(
            TILED_TRANSPOSED_SHARED_MEM_BYTES < limit,
            "TILED_TRANSPOSED_SHARED_MEM_BYTES ({TILED_TRANSPOSED_SHARED_MEM_BYTES}) が \
             48KiB 上限を超えています"
        );
    }
}
