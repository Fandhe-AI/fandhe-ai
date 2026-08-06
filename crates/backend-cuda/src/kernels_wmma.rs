//! f16 WMMA GEMM の CUDA C++ カーネルソース（NVRTC 実行時コンパイル用の静的文字列）。
//!
//! TASK-11.1b（#61。親 #59 の再分解サブタスク）。設計は
//! `docs/cuda-tensor-core-design.md`（#60・origin/main 4199651 でマージ済み）
//! で確定済みであり、本モジュールはその「方式 A（WMMA C++ API）・f16
//! 入力 `m16n16k16` fragment・f32 アキュムレート」を初めて実装するもの
//! である。`kernels.rs`（naive／tiled）とは意図的に別ファイルへ分離した
//! （PR #244「tiled」が `kernels.rs`／`gemm.rs`／`lib.rs` を並行編集中の
//! ため。設計メモに対応する実装計画 3.4 節参照）。
//!
//! `gemm_wmma.rs`（呼び出し元）は本モジュールの [`WMMA_F16`] を
//! `nvrtc::compile_ptx` に渡して `CudaFunction` を得る。既存 naive／tiled
//! カーネルと同じく、ソースを `nvcc` で事前コンパイルせず文字列のまま
//! 埋め込むことで、ビルド時に nvcc／CUDA ヘッダを一切要求しない契約
//! （TASK-1.7・`.claude/rules/deps-policy.md`「CUDA toolkit 非搭載環境でも
//! ビルド成立」）を維持する。`<mma.h>` の実行時 include パス解決自体は
//! 実機でのみ確認可能であり、未検証事項として設計メモ 8 節・本 PR の
//! 引き渡し事項に記録する（#65 へ）。
//!
//! # 命令選定（設計メモ 3.3 節）
//!
//! sm_121（DGX Spark GB10）は Ampere 系譜の WMMA／`mma.sync` プログラミング
//! モデルを維持し `tcgen05`／`wgmma` を要求しない。方式 A（`#include <mma.h>`、
//! `nvcuda::wmma::fragment`／`load_matrix_sync`／`mma_sync`／
//! `store_matrix_sync`）を採用する。これらの識別子はソース内に実在する
//! ことをテスト（本ファイル末尾の `mod tests`）で固定し、TASK-11.3
//! （tensor core 命令使用の証跡）の一部を兼ねる。
//!
//! # タイル構成（設計メモ 4.2 節からの意図的な縮小・逸脱）
//!
//! 設計メモの候補値（ブロックタイル 128×128・warp タイル 64×64・2×2 warp・
//! warp あたり 4×4 fragment）は「実測により確定・調整する」候補であり、
//! #63（共有メモリ・タイル基本最適化）のスコープで拡張する前提のもの
//! （設計メモ 4.2 節「#63 との境界」）。本実装（初回 WMMA 化）はこの
//! サンドボックス環境に CUDA toolkit／実機が存在せず、生成した CUDA C++
//! を一切コンパイル検証できない制約下にあるため、**索引演算の複雑度を
//! 最小化する安全側の判断として、ブロックタイル = warp タイル =
//! fragment タイル = `m16n16k16`（1 ブロック = 1 warp = 32 スレッド、
//! fragment 1 個のみ）に縮小する**。この逸脱理由（実機未接続・
//! コンパイル未検証によるリスク最小化）と、レジスタブロッキング・
//! ダブルバッファリング・ベクトル化ロード・2×2 warp 化等による拡張は
//! #63 へ引き継ぐ（実装計画 8 節「リスク」）。
//!
//! # 境界検査（REQ-8。省略禁止）
//!
//! 1. **A／B タイルの guarded load**: グローバル→共有メモリのロード時、
//!    `(gr < m && gc < k)`／`(gr < k && gc < n)` を満たさない要素は
//!    ゼロ充填する（範囲外のグローバルメモリ読み出しは発生させない）。
//! 2. **エピローグの guarded store**: `store_matrix_sync` で
//!    `wmma::accumulator` を共有メモリ `cs_tile` へ書き戻した後、
//!    `(gr < m && gc < n)` を満たす要素のみをグローバル `c` へ
//!    書き戻す（範囲外のグローバルメモリ書き込みは発生させない）。
//! 3. ホスト側 `gemm_wmma.rs::CudaWmmaGemm::run_f16` は起動前に
//!    `gemm::validate_gemm_dims`（`crates/backend-cuda/src/gemm.rs`）を
//!    必ず先行させる（i32 インデックス積ガード含む）。
//!
//! # `ldm`（leading dimension）制約
//!
//! WMMA API は `load_matrix_sync`（half 入力）で ldm が 8 要素の倍数、
//! `store_matrix_sync`（f32 アキュムレータ）で ldm が 4 要素の倍数である
//! ことを要求する（設計メモ 4.2 節）。本実装は共有メモリタイル行幅を
//! フラグメント次元と同じ 16 要素に固定しており、16 は 8 の倍数かつ 4 の
//! 倍数を同時に満たすため、行幅パディング（設計メモが挙げる 24 要素幅
//! 候補）を導入せずに両制約を満たす。バンクコンフリクト低減目的の
//! パディング適用は #63 のスコープとする（設計メモ 8 節 未検証事項 5）。
//!
//! # 数値契約
//!
//! f16 入出力・f32 内部アキュムレートは `kernels::NAIVE_F16`／
//! `kernels::TILED_F16` と同じ方針（PyTorch の f16 GEMM が cuBLAS 内部で
//! FP32 アキュムレートするのと精度前提を揃える。`.claude/rules/coding-rust.md`
//! FMA 契約統一節）。

/// WMMA fragment 次元（`m16n16k16`。設計メモ 4.1 節）。
///
/// ホスト側（`gemm_wmma.rs`）のブロック次元・グリッド次元計算はこの値を
/// 唯一の真実源として参照する。カーネルソース側の `#define WMMA_M/N/K` と
/// 値が一致することを本ファイル末尾の `mod tests` で検証する
/// （`kernels.rs::tile_constant_matches_kernel_source_define` と同じ方針）。
pub const WMMA_TILE: u32 = 16;

/// f16 WMMA GEMM（f16 入出力・f32 アキュムレート）。
///
/// 1 ブロック = 1 warp（32 スレッド）= C の `WMMA_TILE x WMMA_TILE`
/// 部分行列を 1 個の `m16n16k16` fragment で計算する（本ファイル冒頭
/// ドキュメントコメント「タイル構成」参照。設計メモ 4.2 節候補値からの
/// 意図的な縮小）。
pub const WMMA_F16: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void gemm_wmma_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m, int n, int k)
{
    // ブロック = 1 warp（32 スレッド）が C の WMMA_M x WMMA_N タイル 1 個を
    // 担当する（本ファイル冒頭ドキュメントコメント「タイル構成」参照）。
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    int row0 = tile_row * WMMA_M;
    int col0 = tile_col * WMMA_N;
    int lane = threadIdx.x; // 0..31（1 ブロック = 1 warp）

    // A/B の K タイルを一時保持する共有メモリ。行幅を fragment 次元と
    // 同じ 16 要素に固定するため、load_matrix_sync（half, ldm 8 の倍数
    // 要件）と store_matrix_sync（f32, ldm 4 の倍数要件）を追加パディング
    // なしで同時に満たす（本ファイル冒頭ドキュメントコメント「ldm 制約」）。
    __shared__ __half as_tile[WMMA_M][WMMA_K];
    __shared__ __half bs_tile[WMMA_K][WMMA_N];
    __shared__ float cs_tile[WMMA_M][WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // 桁溢れしない num_k_tiles 計算（kernels.rs::TILED_F32 と同じ方針。
    // k==0 の場合はループを回さずアキュムレータ 0 のまま以降へ進む）。
    int num_k_tiles = (k > 0) ? (k - 1) / WMMA_K + 1 : 0;
    for (int t = 0; t < num_k_tiles; ++t) {
        int k0 = t * WMMA_K;

        // REQ-8: A タイルの guarded load（WMMA_M x WMMA_K = 256 要素を
        // 32 スレッドで分担、1 スレッドあたり 8 要素）。範囲外は 0 充填。
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += 32) {
            int r = idx / WMMA_K;
            int c_ = idx % WMMA_K;
            int gr = row0 + r;
            int gc = k0 + c_;
            as_tile[r][c_] = (gr < m && gc < k) ? a[gr * k + gc] : __float2half(0.0f);
        }
        // REQ-8: B タイルの guarded load（WMMA_K x WMMA_N = 256 要素）。
        for (int idx = lane; idx < WMMA_K * WMMA_N; idx += 32) {
            int r = idx / WMMA_N;
            int c_ = idx % WMMA_N;
            int gr = k0 + r;
            int gc = col0 + c_;
            bs_tile[r][c_] = (gr < k && gc < n) ? b[gr * n + gc] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &as_tile[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &bs_tile[0][0], WMMA_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(&cs_tile[0][0], acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    // REQ-8: エピローグの guarded store。末尾タイルでは row0+r/col0+c_ が
    // m/n を超える要素が発生しうるため、範囲内の要素のみグローバル C へ
    // 書き戻す（範囲外書き込みを発生させない）。
    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
        int r = idx / WMMA_N;
        int c_ = idx % WMMA_N;
        int gr = row0 + r;
        int gc = col0 + c_;
        if (gr < m && gc < n) {
            c[gr * n + gc] = __float2half(cs_tile[r][c_]);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// `WMMA_TILE`（Rust 側の「唯一の真実源」宣言）が `WMMA_F16` の CUDA
    /// ソース文字列内の `#define WMMA_M/N/K` と食い違わないことを検査する
    /// （`kernels.rs::tile_constant_matches_kernel_source_define` と同じ
    /// 方針。値の不一致はコンパイルエラーにならず誤った積和結果を静かに
    /// 生成しうるため CI 上で機械検出する）。
    #[test]
    fn wmma_tile_constant_matches_kernel_source_defines() {
        for dim in ["M", "N", "K"] {
            let expected = format!("#define WMMA_{dim} {WMMA_TILE}");
            assert!(
                WMMA_F16.contains(&expected),
                "WMMA_F16 の `#define WMMA_{dim}` が Rust 側の WMMA_TILE 定数（{WMMA_TILE}）と一致しません"
            );
        }
    }

    /// TASK-11.3（tensor core 命令使用の証跡）を兼ねる: WMMA C++ API の
    /// 主要識別子（fragment 宣言・ロード／積和／ストア関数）がソース文字列
    /// 内に実在することをロックする。これにより将来の書き換えで tensor
    /// core 命令が誤って除去された場合に検出できる。
    #[test]
    fn wmma_f16_source_uses_wmma_instructions() {
        for needle in [
            "#include <mma.h>",
            "wmma::fragment",
            "wmma::load_matrix_sync",
            "wmma::mma_sync",
            "wmma::store_matrix_sync",
            "wmma::fill_fragment",
        ] {
            assert!(
                WMMA_F16.contains(needle),
                "WMMA_F16 に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// REQ-8: A/B タイルロード・エピローグ store の手動境界チェック
    /// （guarded load／guarded store）がソースから除去されていないことを
    /// ロックする（`kernels.rs` の REQ-8 テスト方針と同様、性能最適化を
    /// 理由に境界検査が省略される回帰を防ぐ）。
    #[test]
    fn wmma_f16_source_retains_req8_boundary_guards() {
        for needle in ["gr < m && gc < k", "gr < k && gc < n", "gr < m && gc < n"] {
            assert!(
                WMMA_F16.contains(needle),
                "WMMA_F16 に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }
}
