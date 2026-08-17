//! 融合 RMSNorm 順伝播カーネル（NVRTC 実行時コンパイル用の静的文字列。
//! イシュー #592）。
//!
//! `rmsnorm.rs`（呼び出し元）は本モジュールの 2 定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る。`kernels_elementwise.rs` と同じ理由で
//! ソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に
//! nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # 設計（TileKernels engram gate カーネルの構造イディオムを転用）
//!
//! - **1 CTA = 1 warp（`blockDim.x == 32`）**。`__syncthreads()` は一切
//!   使わず、共有メモリの可視性が必要な箇所のみ `__syncwarp()`（warp 内
//!   バリア）で確立する。ブロック全体の同期プリミティブを要求しない
//!   （設計上の主張。実装計画 §4.1）。
//! - **persistent block 方式**: 各 CTA は `for (row = blockIdx.x; row <
//!   rows; row += gridDim.x)` で複数行を serial に処理する。grid 次元
//!   （persistent block 数）は `rmsnorm.rs::derive_persistent_grid_*` が
//!   デバイスの SMEM/SM 予算から実行時に導出する（Hopper 固定値を流用
//!   しない。`docs/perf/sm121-device-attributes.md` C-8 注記と同方針）。
//! - **意味論**: `out = x * rsqrt(sum(x^2) * inv_n + eps) * w`（`w` は
//!   `has_weight == 0` なら乗算をスキップする）。`sum(x^2)` の縮約 1 本が
//!   数学的に必要な唯一の縮約だが、「単一ロードパス内で複数アキュムレータを
//!   同時縮約する」という参照実装の構造イディオムは、逆伝播（#596。dx/dw
//!   算出に追加の縮約が要る）拡張を見込んだ構造として維持する（実装計画
//!   §1「3 本同時」の位置づけ）。
//!
//! # ベクトル化と REQ-8（カーネル境界検査規約）
//!
//! `float4` ベクトル化ロード／ストアは **`hidden % 4 == 0` の場合のみ**
//! 適用する（`vec_hidden = hidden`）。それ以外（`hidden % 4 != 0`）では
//! 一切ベクトル化せず全要素をスカラー経路で処理する（`vec_hidden = 0`）。
//!
//! この判定を「行の先頭が 16 バイト境界に整列していることを保証できる
//! かどうか」で行う理由: `x_row = x + row * hidden` の行頭オフセットは
//! `hidden` の倍数で進むため、`hidden % 4 != 0` だと行ごとにオフセットの
//! 4 要素剰余（アライメント位相）が変化し、`row >= 1` では
//! `reinterpret_cast<const float4*>` が 16 バイト非整列ポインタを指しうる
//! （未定義動作・実機での fault リスク）。`hidden % 4 == 0` なら
//! `x`（`cudarc` の `alloc_zeros`/`clone_htod` が返すデバイスバッファは
//! ページ単位で確保され十分整列している）からの全行オフセットが 4 要素の
//! 倍数になり安全に整列する。この判定は「行内の端要素のみスカラー化」する
//! という実装計画の記述を、行間アライメント位相という実装上の制約に対し
//! 安全側に具体化したものである（値としては何ら緩和していない: 手動境界
//! チェック `if (base + 3 < hidden)` はベクトル化適用時も維持する）。
//!
//! # ループ添字のオーバーフロー安全性（REQ-8 境界検査規約）
//!
//! `validate_rmsnorm_launch`（`rmsnorm.rs`）は `hidden`／`rows` を
//! `i32::MAX` 以下であれば受理する。カーネル引数の型は `int`（kernel
//! ABI 契約。行頭オフセット計算のみ `long long` へ昇格済み）だが、grid
//! -stride ループの添字（`base`／`i`／`row`）を `int` のまま
//! `+= 32*4`／`+= 32`／`+= gridDim.x` で進めると、`hidden`（または
//! `rows`）が `i32::MAX` 近傍の場合に添字が符号あり `int` の範囲を
//! signed overflow（C++ の未定義動作）でラップし負値化しうる。ラップ後
//! `i < hidden` 等のループ条件が負値相手に再度真となり、`x_row[i]`／
//! `out_row[i]` へ負インデックスで範囲外アクセスする（手動境界チェック
//! `if (base + 3 < hidden)` を実質迂回する。codex-review 指摘・PR #706
//! レビュー r3793473231 相当）。よって `base`／`i`／`row` は
//! `long long`（`vec_hidden`／`hidden`／`rows` との比較は暗黙昇格）で
//! 宣言し、`i32::MAX` 近傍でもオーバーフローしない終了条件を保証する
//! （`hidden`／`rows` 自体の型は kernel ABI 契約のため `int` のまま
//! 据え置く）。
//!
//! # 数値契約（FMA・丸め）
//!
//! 二乗和の蓄積・正規化の積和は `fmaf` を明示使用し、CPU 参照実装
//! （`f32::mul_add`）と丸め方針を揃える（`.claude/rules/coding-rust.md`）。
//! `rstd = rsqrtf(fmaf(acc, inv_n, eps))`。`run_fused` 経由（`ops.rs`）で
//! 呼ばれる場合は `inv_n = 1.0`・`eps = 0.0`・`has_weight = 0` を渡し、
//! プランの意味論 `x * rsqrt(sum(x^2))` と厳密一致させる（`ops.rs`
//! ドキュメンテーションコメント参照）。
//!
//! # 意味論の正
//!
//! `tests/rmsnorm_parity.rs` 内のテスト専用 CPU 参照実装（`f32::mul_add`
//! 使用）が意味論の正である。`backend-cpu` 側 RMSNorm 実装は別スコープ
//! （#607）のため本カーネルはこれに依存しない。

/// 1 スレッドブロックあたりのスレッド数（1 warp = 32 固定）。
///
/// GEMM／elementwise の `TILE`／`EW_BLOCK_DIM` とは無関係の独立した
/// パラメータ。`__shfl_xor_sync`／`__syncwarp` による warp 内 reduction を
/// 前提とするため、32 以外の値は本カーネル設計と両立しない
/// （`rmsnorm.rs::derive_persistent_grid_*` もこの前提で SM あたり
/// block 数を導出する）。
pub const RMSNORM_BLOCK_DIM: u32 = 32;

/// 1 パス経路（動的 SMEM 常駐）: `x` を 1 回だけロードしつつ Σx² を
/// 同時蓄積し、warp 内 butterfly reduction → 正規化 → 書き出しまでを
/// 中間テンソルの HBM 書き出しなしで完結する。
///
/// 起動時に `extern __shared__ float smem[]` へ `hidden * 4` バイトの
/// 動的共有メモリを割り当てる契約（`rmsnorm.rs::run_rmsnorm_f32` が
/// `LaunchConfig::shared_mem_bytes` で指定する）。適用条件（`hidden * 4`
/// が SMEM 予算に収まること）はホスト側 `rmsnorm_route` が判定し、
/// 収まらない場合は [`RMSNORM_F32_TWOPASS`] へルーティングする。
pub const RMSNORM_F32_ONEPASS: &str = r#"
extern "C" __global__ void rmsnorm_f32_onepass(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int rows,
    int hidden,
    float eps,
    float inv_n,
    int has_weight)
{
    extern __shared__ float smem[];
    int lane = threadIdx.x;
    // hidden % 4 != 0 の場合は行間アライメント位相がずれるため
    // ベクトル化を一切適用しない（本ファイル冒頭コメント「ベクトル化と
    // REQ-8」参照）。
    int vec_hidden = (hidden % 4 == 0) ? hidden : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)hidden;
        float* out_row = out + row * (long long)hidden;

        float acc = 0.0f;

        // ベクトル化ロード（float4）+ SMEM 格納 + 二乗和蓄積。
        // `base + 3 < hidden` は vec_hidden の定義上常に成立するが、
        // 最適化を理由に手動境界チェックを省略しない（REQ-8）。`base` は
        // `long long`（本ファイル冒頭コメント「ループ添字のオーバーフロー
        // 安全性」参照。`hidden` が `i32::MAX` 近傍でも `int` 添字の
        // signed overflow による境界チェック迂回を防ぐ）。
        for (long long base = lane * 4; base < vec_hidden; base += 32 * 4) {
            if (base + 3 < hidden) {
                float4 v = *reinterpret_cast<const float4*>(x_row + base);
                smem[base + 0] = v.x;
                smem[base + 1] = v.y;
                smem[base + 2] = v.z;
                smem[base + 3] = v.w;
                acc = fmaf(v.x, v.x, acc);
                acc = fmaf(v.y, v.y, acc);
                acc = fmaf(v.z, v.z, acc);
                acc = fmaf(v.w, v.w, acc);
            }
        }
        // スカラー経路: hidden % 4 != 0 なら全要素、それ以外は端要素なし
        // （vec_hidden == hidden のため本ループは実行されない）。`i` は
        // `long long`（同上）。
        for (long long i = (long long)vec_hidden + lane; i < hidden; i += 32) {
            float v = x_row[i];
            smem[i] = v;
            acc = fmaf(v, v, acc);
        }

        // 上記ロードループが書いた smem を他レーンが読めるようにする
        // warp 内バリア（`__syncthreads()` は使わない。ブロック全体
        // 同期を要求しない設計。本ファイル冒頭コメント「設計」参照）。
        __syncwarp(0xffffffffu);

        // warp 内 butterfly reduction（5 段、全レーンが総和を保持する）。
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, offset);
        }

        float rstd = rsqrtf(fmaf(acc, inv_n, eps));

        for (long long i = lane; i < hidden; i += 32) {
            float normed = smem[i] * rstd;
            if (has_weight) {
                normed = normed * w[i];
            }
            out_row[i] = normed;
        }

        // 次の行（grid-stride ループの次反復）が smem を上書きする前に、
        // 全レーンの上記読み出しが完了していることを保証する。
        __syncwarp(0xffffffffu);
    }
}
"#;

/// 2 パス経路（global 再読）: 1 パス経路の SMEM 予算を超える行長
/// （`rmsnorm_route` が `TwoPass` を選んだ場合）で使う。Pass 1 で `x` を
/// global から読み Σx² を蓄積、Pass 2 で同一カーネル・同一行ループ内で
/// `x` を再度 global から読み正規化・書き出す。共有メモリを一切使わない
/// ため warp 内バリアも不要（レーン間の共有状態がない。各レーンは自分の
/// 担当インデックスのみを読み書きする）。
pub const RMSNORM_F32_TWOPASS: &str = r#"
extern "C" __global__ void rmsnorm_f32_twopass(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int rows,
    int hidden,
    float eps,
    float inv_n,
    int has_weight)
{
    int lane = threadIdx.x;
    int vec_hidden = (hidden % 4 == 0) ? hidden : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)hidden;
        float* out_row = out + row * (long long)hidden;

        // Pass 1: Σx² を計算する（smem 非使用・global 直読）。`base`／`i`
        // は `long long`（本ファイル冒頭コメント「ループ添字の
        // オーバーフロー安全性」参照。`hidden` が `i32::MAX` 近傍でも
        // `int` 添字の signed overflow による境界チェック迂回を防ぐ）。
        float acc = 0.0f;
        for (long long base = lane * 4; base < vec_hidden; base += 32 * 4) {
            if (base + 3 < hidden) {
                float4 v = *reinterpret_cast<const float4*>(x_row + base);
                acc = fmaf(v.x, v.x, acc);
                acc = fmaf(v.y, v.y, acc);
                acc = fmaf(v.z, v.z, acc);
                acc = fmaf(v.w, v.w, acc);
            }
        }
        for (long long i = (long long)vec_hidden + lane; i < hidden; i += 32) {
            float v = x_row[i];
            acc = fmaf(v, v, acc);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, offset);
        }

        float rstd = rsqrtf(fmaf(acc, inv_n, eps));

        // Pass 2: x を再度 global から読み正規化して書き出す（同一カーネル・
        // 同一行ループ内で完結。中間テンソルは書き出さない）。`base`／`i`
        // は `long long`（同上）。
        for (long long base = lane * 4; base < vec_hidden; base += 32 * 4) {
            if (base + 3 < hidden) {
                float4 v = *reinterpret_cast<const float4*>(x_row + base);
                float4 o;
                o.x = v.x * rstd;
                o.y = v.y * rstd;
                o.z = v.z * rstd;
                o.w = v.w * rstd;
                if (has_weight) {
                    o.x = o.x * w[base + 0];
                    o.y = o.y * w[base + 1];
                    o.z = o.z * w[base + 2];
                    o.w = o.w * w[base + 3];
                }
                *reinterpret_cast<float4*>(out_row + base) = o;
            }
        }
        for (long long i = (long long)vec_hidden + lane; i < hidden; i += 32) {
            float v = x_row[i] * rstd;
            if (has_weight) {
                v = v * w[i];
            }
            out_row[i] = v;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// grid-stride ループの添字（`row`／`base`／`i`）が `long long` で
    /// 宣言されていることをソース文字列に対して検査する（本ファイル
    /// 冒頭コメント「ループ添字のオーバーフロー安全性」参照）。
    ///
    /// `hidden`／`rows` は `i32::MAX` まで許容され（`rmsnorm.rs::
    /// validate_rmsnorm_launch`）、実機で `hidden == i32::MAX` を実行する
    /// には行あたり約 8 GiB のバッファが要るため決定的な実行検証は
    /// 非現実的（実機依存の `#[ignore]` テストの対象にもしない）。
    /// 代わりに、オーバーフロー安全性の根拠となる「添字が `int` へ縮退
    /// していない」ことをソース文字列上で回帰検出する。1 パス・2 パス
    /// 双方のカーネルで 3 種の添字（`row`／`base`／`i`）を検査する
    /// （codex-review 指摘・PR #706 レビュー r3793473231 相当）。
    #[test]
    fn onepass_and_twopass_loop_indices_are_declared_long_long() {
        for src in [RMSNORM_F32_ONEPASS, RMSNORM_F32_TWOPASS] {
            assert!(
                src.contains("for (long long row = blockIdx.x; row < rows; row += gridDim.x)"),
                "row ループ添字が long long で宣言されていない"
            );
            assert!(
                src.contains("for (long long base = lane * 4; base < vec_hidden; base += 32 * 4)"),
                "base ループ添字が long long で宣言されていない"
            );
            assert!(
                src.contains(
                    "for (long long i = (long long)vec_hidden + lane; i < hidden; i += 32)"
                ),
                "i ループ添字が long long で宣言されていない"
            );
            // ループ添字が `int` のまま宣言される回帰（今回是正した
            // signed overflow の再発）を検出する。
            assert!(
                !src.contains("for (int row = blockIdx.x")
                    && !src.contains("for (int base = lane * 4")
                    && !src.contains("for (int i = vec_hidden + lane")
                    && !src.contains("for (int i = (int)vec_hidden + lane"),
                "grid-stride ループ添字が int へ縮退している"
            );
        }
    }
}
