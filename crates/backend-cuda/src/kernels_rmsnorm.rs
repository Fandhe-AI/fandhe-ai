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

/// 逆伝播カーネル（[`RMSNORM_BWD_DX_F32`]／[`RMSNORM_BWD_DW_F32`]）専用の
/// ブロックあたりスレッド数（イシュー #596）。
///
/// 順伝播 [`RMSNORM_BLOCK_DIM`]（32・1 CTA = 1 warp）とは**独立**の
/// パラメータである。逆伝播は保存 `rstd` から再計算する 2 パス構成
/// （Σ(dy·w·x) の縮約 → dx 書き出し）のみで SMEM 常駐を必要とせず、warp
/// 内 reduction 単体では収まらない広い `hidden` でもスループットを確保
/// するため 8 warp（256 スレッド）のクロス warp reduction 構成を採る
/// （参照実装〈TileKernels engram gate カーネル〉の 2 段 reduction
/// イディオムを転用。実装計画 §3.2）。`derive_persistent_grid_two_pass`
/// （grid = SM 数 × 16 を `rows` でクランプ）はブロック幅に依存しない
/// 行方向の persistent grid 導出のため、`RMSNORM_BLOCK_DIM` 前提の
/// ドキュメンテーションコメント（32 以外は「本カーネル設計」= 順伝播
/// warp-only reduction と両立しないという記述）と矛盾しない: 逆伝播は
/// 独自の `RMSNORM_BWD_BLOCK_DIM` の下で `derive_persistent_grid_two_pass`
/// を再利用し（`rmsnorm.rs` 参照）、block_dim は起動時に呼び出し元が
/// 明示的に差し替える。
pub const RMSNORM_BWD_BLOCK_DIM: u32 = 256;

#[cfg(test)]
mod bwd_block_dim_tests {
    use super::RMSNORM_BWD_BLOCK_DIM;

    /// `RMSNORM_BWD_DX_F32` 内の `__shared__ float smem_dot[8]`
    /// （warp あたり 1 要素の静的配列）が `RMSNORM_BWD_BLOCK_DIM / 32`
    /// と一致することを回帰検出する（不一致は未初期化 warp スロットの
    /// 読み出しに繋がる）。
    #[test]
    fn smem_dot_array_size_matches_warps_per_block() {
        assert_eq!(RMSNORM_BWD_BLOCK_DIM / 32, 8);
        assert!(super::RMSNORM_BWD_DX_F32.contains("__shared__ float smem_dot[8]"));
    }
}

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
    float* __restrict__ rstd_out,
    int rows,
    int hidden,
    float eps,
    float inv_n,
    int has_weight,
    int save_rstd)
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

        // 学習経路（`save_rstd != 0`）でのみ行あたり 1 スカラーを書く
        // （イシュー #596: 逆伝播は正規化済みテンソルを保存せず、この
        // `rstd` と生の `x` から再計算する）。推論経路
        // （`save_rstd == 0`）では `rstd_out` を一切デリファレンスしない
        // （受け入れ基準「保存はスカラーのみ」を推論時ゼロ保存で満たす）。
        if (save_rstd && lane == 0) {
            rstd_out[row] = rstd;
        }

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
    float* __restrict__ rstd_out,
    int rows,
    int hidden,
    float eps,
    float inv_n,
    int has_weight,
    int save_rstd)
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

        // 学習経路のみ行あたり 1 スカラーを保存する（1 パス経路と同じ
        // 契約。本ファイル冒頭コメント「意味論」・イシュー #596 参照）。
        if (save_rstd && lane == 0) {
            rstd_out[row] = rstd;
        }

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

/// 逆伝播 dx カーネル（recompute-in-backward。イシュー #596）:
/// 正規化済みテンソルを保存せず、生の `x` と保存スカラー `rstd`（行あたり
/// 1 要素。[`RMSNORM_F32_ONEPASS`]／[`RMSNORM_F32_TWOPASS`] が
/// `save_rstd != 0` 時に書く）から dx をその場で再計算する。
///
/// 数式（`out = x * rstd * w`・`rstd = rsqrt(sum(x^2) * inv_n + eps)`）:
/// `dx_i = rstd·dy_i·w_i − rstd³·inv_n·x_i·Σ_j(dy_j·w_j·x_j)`。
///
/// [`RMSNORM_BWD_BLOCK_DIM`]（256・8 warp）の 2 段 reduction（warp 内
/// butterfly → 静的 smem 経由クロス warp reduction）で `Σ_j(dy_j·w_j·x_j)`
/// を求める。1 行の処理につき `__syncthreads()` を 3 回使う: (1)
/// 各 warp が部分和を `smem_dot` へ書いた後、(2) warp 0 が縮約結果を
/// `dot_broadcast` へ書いた後、(3) persistent grid-stride ループの次反復が
/// `smem_dot`/`dot_broadcast` を上書きする前（速い warp が遅い warp の
/// 読み出し前に上書きするデータ競合を防ぐ。順伝播 1 パスカーネルの末尾
/// `__syncwarp` と同じ目的をブロック全体スコープで担う）。
pub const RMSNORM_BWD_DX_F32: &str = r#"
extern "C" __global__ void rmsnorm_bwd_dx_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    const float* __restrict__ rstd,
    float* __restrict__ dx,
    int rows,
    int hidden,
    float inv_n,
    int has_weight)
{
    __shared__ float smem_dot[8];
    __shared__ float dot_broadcast;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int vec_hidden = (hidden % 4 == 0) ? hidden : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)hidden;
        const float* dy_row = dy + row * (long long)hidden;
        float* dx_row = dx + row * (long long)hidden;
        float r = rstd[row];

        // Pass 1: acc = Σ(dy_j * w_j * x_j)（`base`/`i` は `long long`。
        // `kernels_rmsnorm.rs` 冒頭コメント「ループ添字のオーバーフロー
        // 安全性」と同じ理由で signed overflow による境界チェック迂回を
        // 防ぐ）。
        float acc = 0.0f;
        for (long long base = tid * 4; base < vec_hidden; base += 256 * 4) {
            if (base + 3 < hidden) {
                float4 xv = *reinterpret_cast<const float4*>(x_row + base);
                float4 dyv = *reinterpret_cast<const float4*>(dy_row + base);
                if (has_weight) {
                    float4 wv = *reinterpret_cast<const float4*>(w + base);
                    acc = fmaf(dyv.x * wv.x, xv.x, acc);
                    acc = fmaf(dyv.y * wv.y, xv.y, acc);
                    acc = fmaf(dyv.z * wv.z, xv.z, acc);
                    acc = fmaf(dyv.w * wv.w, xv.w, acc);
                } else {
                    acc = fmaf(dyv.x, xv.x, acc);
                    acc = fmaf(dyv.y, xv.y, acc);
                    acc = fmaf(dyv.z, xv.z, acc);
                    acc = fmaf(dyv.w, xv.w, acc);
                }
            }
        }
        for (long long i = (long long)vec_hidden + tid; i < hidden; i += 256) {
            float xv = x_row[i];
            float dyv = dy_row[i];
            float wv = has_weight ? w[i] : 1.0f;
            acc = fmaf(dyv * wv, xv, acc);
        }

        // warp 内 butterfly reduction（5 段、全レーンが warp 内総和を保持）。
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) {
            smem_dot[warp_id] = acc;
        }
        __syncthreads();

        float dot = 0.0f;
        if (warp_id == 0) {
            float v = (lane < 8) ? smem_dot[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            if (lane == 0) {
                dot_broadcast = v;
            }
        }
        __syncthreads();
        dot = dot_broadcast;

        // Pass 2: x/dy を再度 global から読み dx を書き出す
        // （中間テンソルは一切保存・参照しない。イシュー #596 の主旨）。
        float coef = -(r * r * r * inv_n * dot);

        for (long long base = tid * 4; base < vec_hidden; base += 256 * 4) {
            if (base + 3 < hidden) {
                float4 xv = *reinterpret_cast<const float4*>(x_row + base);
                float4 dyv = *reinterpret_cast<const float4*>(dy_row + base);
                float4 o;
                if (has_weight) {
                    float4 wv = *reinterpret_cast<const float4*>(w + base);
                    o.x = fmaf(coef, xv.x, r * dyv.x * wv.x);
                    o.y = fmaf(coef, xv.y, r * dyv.y * wv.y);
                    o.z = fmaf(coef, xv.z, r * dyv.z * wv.z);
                    o.w = fmaf(coef, xv.w, r * dyv.w * wv.w);
                } else {
                    o.x = fmaf(coef, xv.x, r * dyv.x);
                    o.y = fmaf(coef, xv.y, r * dyv.y);
                    o.z = fmaf(coef, xv.z, r * dyv.z);
                    o.w = fmaf(coef, xv.w, r * dyv.w);
                }
                *reinterpret_cast<float4*>(dx_row + base) = o;
            }
        }
        for (long long i = (long long)vec_hidden + tid; i < hidden; i += 256) {
            float wv = has_weight ? w[i] : 1.0f;
            dx_row[i] = fmaf(coef, x_row[i], r * dy_row[i] * wv);
        }

        // 次の persistent grid-stride 反復が smem_dot/dot_broadcast を
        // 上書きする前に、全スレッドの Pass2 読み出しが完了していることを
        // 保証するバリア（本カーネルコメント冒頭「__syncthreads() を
        // 3 回使う」参照）。
        __syncthreads();
    }
}
"#;

/// 逆伝播 dw カーネル（`has_weight` 時のみホスト側が起動する。イシュー
/// #596）: `dw_i = Σ_r dy[r,i]·x[r,i]·rstd[r]` を列（`hidden`）方向
/// grid-stride で導出する。1 スレッドが 1 列を担当し `rows` を serial に
/// 蓄積する（atomics 不使用のため決定的。`fmaf` で FMA 契約統一）。
/// dx カーネルとはグリッド構成の軸が異なる（dx は行方向 persistent grid・
/// dw は列方向 grid-stride）ため、grid 導出は共有しない
/// （`rmsnorm.rs::derive_persistent_grid_dw` 参照）。
///
/// **行数が大きい形状では並列度 1（列方向のみ）に頭打ちになる**性能限界を
/// 持つ（1 スレッドが `rows` 全体を serial 蓄積するため）。この限界を
/// 解消する split-K 二段構成（[`RMSNORM_BWD_DW_PARTIAL_F32`]／
/// [`RMSNORM_BWD_DW_REDUCE_F32`]。イシュー #597）を新設したが、本カーネルは
/// 小規模形状（`rmsnorm.rs::derive_dw_split` が `num_blocks <= 1` を返す
/// 場合）のフォールバック経路として削除せず維持する（余分なカーネル起動・
/// 部分和バッファ確保を避けるため）。
pub const RMSNORM_BWD_DW_F32: &str = r#"
extern "C" __global__ void rmsnorm_bwd_dw_f32(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ rstd,
    float* __restrict__ dw,
    int rows,
    int hidden)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += (long long)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (long long row = 0; row < rows; row += 1) {
            long long idx = row * (long long)hidden + i;
            float xv = x[idx];
            float dyv = dy[idx];
            float r = rstd[row];
            acc = fmaf(dyv * r, xv, acc);
        }
        dw[i] = acc;
    }
}
"#;

/// [`RMSNORM_BWD_DW_PARTIAL_F32`] のブロックあたりスレッド数（列方向
/// grid-stride の 1 スレッド = 1 列。[`RMSNORM_BWD_DW_F32`] の実行時
/// ブロック幅（`rmsnorm.rs::RMSNORM_BWD_DW_BLOCK_DIM` = 256）と同じ値だが、
/// 意味的に独立したパラメータ（列方向 grid-stride の幅）として split-K
/// 経路専用に持つ（イシュー #597 実装計画 §3.2）。
pub const RMSNORM_BWD_DW_PARTIAL_BLOCK_DIM: u32 = 256;

/// [`RMSNORM_BWD_DW_REDUCE_F32`] のブロックあたりスレッド数（1 スレッド =
/// 1 列。静的 smem `smem[2][RMSNORM_DW_REDUCE_BATCH][RMSNORM_DW_REDUCE_
/// BLOCK_DIM]` の第 3 次元と一致する契約——起動時ブロック幅がこの値と
/// 異なると `threadIdx.x` が smem 配列境界を超えうる。ホスト側
/// （`rmsnorm.rs`）は必ずこの定数を `block_dim` に渡す）。
pub const RMSNORM_DW_REDUCE_BLOCK_DIM: u32 = 256;

/// [`RMSNORM_BWD_DW_REDUCE_F32`] の 2 段パイプラインが 1 イテレーションで
/// 処理する block（`num_blocks` 次元）のバッチ数。smem double buffer の
/// 静的サイズは「2 かける RMSNORM_DW_REDUCE_BATCH かける
/// RMSNORM_DW_REDUCE_BLOCK_DIM かける 4」バイト（`BATCH=4`・
/// `BLOCK_DIM=256` で 8 KiB。静的 smem 予算に常に収まる小さな固定値）。
///
/// `#[allow(dead_code)]` について: 通常ビルドではカーネル文字列内の
/// リテラル（`smem[2][4][256]`／`(num_blocks + 3) / 4`）を直接埋め込む
/// ため本定数はホスト側実行経路から参照されないが、
/// `mod tests::split_k_dw_reduce_smem_size_matches_batch_and_block_dim_consts`
/// がこの定数を単一の真実源として `format!` でリテラルを再構成し
/// 一致検証する（advisor 指摘: 定数変更時のソース側乖離を防ぐ回帰検出。
/// `module_cache.rs`／`swizzle.rs` の同アノテーションと同じ理由）。
#[allow(dead_code)]
pub const RMSNORM_DW_REDUCE_BATCH: u32 = 4;

/// weight gradient の split-K 二段リダクション（イシュー #597）:
/// 第 1 カーネル。[`RMSNORM_BWD_DW_F32`]（列 1 スレッドが `rows` 全体を
/// serial 蓄積）の行方向並列度 1 という限界を、行（`rows`）方向を
/// `num_blocks`（`blockIdx.y`）個の CTA へ分割することで解消する。
///
/// # 参照実装（TileKernels engram gate カーネル）との対応付け
///
/// 参照実装は `[num_blocks, ...]` 形状の部分和バッファを書き出し、第 2
/// カーネルで縮約する 2 段構成を取る。本カーネルはその第 1 段に相当し、
/// 各 `(col_tile, b)` の CTA が担当する行範囲
/// `[b*rows_per_block, min((b+1)*rows_per_block, rows))`
/// （`rows_per_block = ceil(rows / num_blocks)`）を **レジスタ `acc` で
/// 蓄積し、最後に 1 回だけ** `dw_partial[b*hidden + i] = acc` を
/// 書く（atomics 不使用。各 `(b, i)` の書き手 CTA は一意のため決定的。
/// [`RMSNORM_BWD_DW_F32`] と同じ `fmaf` 蓄積で FMA 契約を統一する）。
///
/// # §3.1 連鎖則対応付け（設計判断）
///
/// 参照実装は重みが 2 因子（`wh`・`we`）で縮約 epilogue に連鎖則の乗算が
/// 残るが、本リポジトリの RMSNorm は単一重み `w`
/// （`out = x·rstd ⊙ w`）であり、連鎖則係数 `rstd[row]` は**行ごとの
/// スカラー**である。よって連鎖則は列方向の後置演算にはならず、本カーネル
/// （行方向蓄積）のレジスタ加算へ直接融合するしかない
/// （`acc = fmaf(dy·rstd, x, acc)`）。中間の正規化済みテンソルは HBM へ
/// 一切書かない（イシュー #596 の recompute-in-backward 契約を split-K化後も
/// 維持する）。
///
/// # 末尾要素ブロックの扱い（REQ-8・決定的性）
///
/// `b*rows_per_block >= rows` となる末尾 block（`num_blocks` が `rows` を
/// 割り切らない場合に生じる）は行範囲が空になるが、`acc = 0.0f` の
/// まま**無条件に** `dw_partial` へ書く（早期 return や条件付き書き出しに
/// しない）。これにより `dw_partial` の全要素が必ず書かれることを保証し、
/// `alloc_zeros` のゼロ初期化に依存しない（ホスト側の未初期化読み出しを
/// 防ぐ fail-closed な設計）。
///
/// # ループ添字のオーバーフロー安全性
///
/// `row_start`／`row_end`／`idx` は `long long`（本ファイル冒頭コメント
/// 「ループ添字のオーバーフロー安全性」と同じ理由。`rows`／`num_blocks`
/// は `int` の乗算前に `long long` へ昇格する）。
pub const RMSNORM_BWD_DW_PARTIAL_F32: &str = r#"
extern "C" __global__ void rmsnorm_bwd_dw_partial_f32(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ rstd,
    float* __restrict__ dw_partial,
    int rows,
    int hidden,
    int num_blocks)
{
    int b = blockIdx.y;
    long long rows_per_block = ((long long)rows + num_blocks - 1) / num_blocks;
    long long row_start = (long long)b * rows_per_block;
    long long row_end = row_start + rows_per_block;
    if (row_end > rows) {
        row_end = rows;
    }

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += (long long)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        // `row_start >= row_end`（末尾の空 block）ではこのループが 0 回
        // 実行され `acc` は `0.0f` のまま。本ファイル冒頭コメント
        // 「末尾要素ブロックの扱い」参照。
        for (long long row = row_start; row < row_end; row += 1) {
            long long idx = row * (long long)hidden + i;
            float xv = x[idx];
            float dyv = dy[idx];
            float r = rstd[row];
            acc = fmaf(dyv * r, xv, acc);
        }
        // 無条件書き出し（条件分岐で省略しない。REQ-8 と同じ
        // 「最適化を理由に手動保証を省略しない」精神を空 block の扱いにも
        // 適用する）。
        dw_partial[(long long)b * (long long)hidden + i] = acc;
    }
}
"#;

/// weight gradient の split-K 二段リダクション（イシュー #597）:
/// 第 2 カーネル。[`RMSNORM_BWD_DW_PARTIAL_F32`] が書いた
/// `[num_blocks, hidden]` 形状の部分和バッファを `num_blocks` 次元方向に
/// 縮約し、最終 `dw` を **1 回だけ** HBM へ書く（縮約結果を HBM へ書いて
/// 読み戻す第 3 パスを作らない。§3.1 の epilogue 融合方針）。
///
/// # 2 段パイプライン（受け入れ基準 3）
///
/// `num_blocks` を [`RMSNORM_DW_REDUCE_BATCH`] 個ずつのバッチに分け、
/// 静的 smem double buffer `smem[2][RMSNORM_DW_REDUCE_BATCH][
/// RMSNORM_DW_REDUCE_BLOCK_DIM]`（8 KiB）で「次バッチの global ロードを
/// レジスタ経由で発行してから今バッチの加算を行う」順に進める（プロローグで
/// バッチ 0 をロード → ループ内で「次バッチロード発行 → 今バッチ加算 →
/// バッファ入替」を繰り返す）。範囲外バッチ要素（`b >= num_blocks`）は
/// `0.0f` 充填で手動ガードする（REQ-8）。
///
/// # smem の役割（1 スレッド = 1 列という本カーネル構成に固有の注記）
///
/// 1 スレッドが 1 列（`col`）を担当し、各スレッドは自分の `col` に対応する
/// `smem[*][*][threadIdx.x]` スロットのみを読み書きする。**スレッド間で
/// smem を共有しない**（参照実装〈TileKernels engram gate カーネル〉は
/// `b` 次元をスレッド次元へ割る変種でスレッド間共有が生じるが、本カーネル
/// の列並列構成ではその前提が成立しない）。したがって本カーネルの
/// `__syncthreads()` はスレッド間データ競合の防止としては pass-through
/// （各スレッドが自分のスロットのみを触るため理論上は不要）であり、
/// 実際のロード/加算オーバーラップは「次バッチの global ロード命令を
/// 今バッチの加算命令より先に発行する」というプログラム順序（コンパイラ・
/// スケジューラによる命令レベル並列性）に由来する。受け入れ基準 3
/// （2 段パイプライン・smem double buffer）を文字通り満たすため smem
/// 構成自体は維持するが、この非自明な前提を記録する（`code-comment-
/// style.md`: 非自明な前提を書く）。
///
/// # ループ添字のオーバーフロー安全性
///
/// `col` は `long long`（本ファイル冒頭コメントと同じ理由）。
pub const RMSNORM_BWD_DW_REDUCE_F32: &str = r#"
extern "C" __global__ void rmsnorm_bwd_dw_reduce_f32(
    const float* __restrict__ dw_partial,
    float* __restrict__ dw,
    int hidden,
    int num_blocks)
{
    __shared__ float smem[2][4][256];
    int tid = threadIdx.x;
    int num_batches = (num_blocks + 3) / 4;

    // 外側ループの継続条件は `base`（= blockIdx.x * blockDim.x を起点に
    // ブロック単位で進める block-stride 添字）でのみ判定する。`base` は
    // ブロック内の全スレッドで共通の値であり、`threadIdx.x`（`tid`）に
    // 依存しない。これにより「同一ブロック内の全スレッドが
    // `__syncthreads()` へ到達する回数を必ず揃える」という CUDA の
    // ブロック同期契約を満たす（codex-review P0 指摘・PR #716:
    // 旧実装は `col < hidden` という tid 依存の条件でループを継続して
    // いたため、`hidden` がブロック幅未満の部分ブロック（例:
    // `hidden=32` かつ `blockDim.x=256`）で `col >= hidden` の
    // スレッドがループ本体へ一度も入らず `__syncthreads()` を踏まず、
    // 残りのスレッドだけがバリアへ到達して不一致・ハングを起こしていた）。
    // 個々のスレッドの有効性（`col < hidden`）は `active` フラグで
    // マスクし、範囲外スレッドは smem ロード・加算・`dw` 書き出しを
    // 素通り（0.0f 充填・書き出しスキップ）しつつバリアには全員で
    // 到達する。
    for (long long base = (long long)blockIdx.x * blockDim.x; base < hidden;
         base += (long long)blockDim.x * gridDim.x) {
        long long col = base + tid;
        bool active = col < hidden;

        // プロローグ: バッチ 0 を smem[0] へロードする（範囲外
        // `b >= num_blocks` または `!active`（列自体が範囲外）は
        // 0.0f 充填。REQ-8 手動境界チェック）。
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int b = j;
            float v =
                (active && b < num_blocks) ? dw_partial[(long long)b * (long long)hidden + col] : 0.0f;
            smem[0][j][tid] = v;
        }
        __syncthreads();

        float acc = 0.0f;
        int buf = 0;
        for (int batch = 0; batch < num_batches; batch++) {
            int next_buf = buf ^ 1;
            // (1) 次バッチの global ロードを発行（今バッチの加算命令より
            // 先にコンパイラへ提示することでレイテンシを重ねる）。
            if (batch + 1 < num_batches) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int b = (batch + 1) * 4 + j;
                    float v = (active && b < num_blocks)
                                   ? dw_partial[(long long)b * (long long)hidden + col]
                                   : 0.0f;
                    smem[next_buf][j][tid] = v;
                }
            }
            // (2) 今バッチを加算する。
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc += smem[buf][j][tid];
            }
            __syncthreads();
            buf = next_buf;
        }

        // epilogue: 最終 dw を 1 回だけ書く（縮約結果を HBM へ往復させる
        // 第 3 パスを作らない。本ファイル冒頭コメント参照）。範囲外
        // スレッド（`!active`）は書き出さない。
        if (active) {
            dw[col] = acc;
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

    /// 逆伝播カーネル（[`RMSNORM_BWD_BLOCK_DIM`] = 256 前提のため
    /// ストライド定数が順伝播〈32〉と異なる）のループ添字が `long long`
    /// 宣言であることを検査する。`onepass_and_twopass_loop_indices_are_
    /// declared_long_long` の対象配列には含めず、256 スレッド構成専用の
    /// 文字列で別途回帰検出する（本ファイル冒頭コメント「ループ添字の
    /// オーバーフロー安全性」と同じ根拠。イシュー #596）。
    #[test]
    fn bwd_dx_loop_indices_are_declared_long_long() {
        let src = RMSNORM_BWD_DX_F32;
        assert!(src.contains("for (long long row = blockIdx.x; row < rows; row += gridDim.x)"));
        assert!(src.contains("for (long long base = tid * 4; base < vec_hidden; base += 256 * 4)"));
        assert!(
            src.contains("for (long long i = (long long)vec_hidden + tid; i < hidden; i += 256)")
        );
        assert!(
            !src.contains("for (int row =")
                && !src.contains("for (int base =")
                && !src.contains("for (int i =")
        );
    }

    #[test]
    fn bwd_dw_loop_indices_are_declared_long_long() {
        let src = RMSNORM_BWD_DW_F32;
        assert!(src.contains(
            "for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < hidden;"
        ));
        assert!(src.contains("for (long long row = 0; row < rows; row += 1)"));
        assert!(!src.contains("for (int i =") && !src.contains("for (int row ="));
    }

    /// 順伝播カーネルの [`RMSNORM_F32_ONEPASS`]／[`RMSNORM_F32_TWOPASS`]
    /// が `save_rstd`/`rstd_out` の学習経路パラメータを持つことを回帰
    /// 検出する（イシュー #596 の受け入れ基準「保存はスカラーのみ」の
    /// 入口）。
    #[test]
    fn forward_kernels_have_save_rstd_train_params() {
        for src in [RMSNORM_F32_ONEPASS, RMSNORM_F32_TWOPASS] {
            assert!(src.contains("float* __restrict__ rstd_out"));
            assert!(src.contains("int save_rstd"));
            assert!(src.contains("if (save_rstd && lane == 0)"));
        }
    }

    // --- split-K dw（イシュー #597） ---

    /// 受け入れ基準 1「atomicAdd 等を一切使わない」の機械検査。
    #[test]
    fn split_k_dw_kernels_do_not_use_atomics() {
        for src in [RMSNORM_BWD_DW_PARTIAL_F32, RMSNORM_BWD_DW_REDUCE_F32] {
            assert!(
                !src.contains("atomicAdd"),
                "atomics 不使用の受け入れ基準に反する"
            );
        }
    }

    /// ループ添字（`row_start`／`row_end`／`idx`・`col`）が `long long`
    /// 宣言であることを検査する（本ファイル冒頭コメント「ループ添字の
    /// オーバーフロー安全性」と同じ根拠）。
    #[test]
    fn split_k_dw_partial_loop_indices_are_declared_long_long() {
        let src = RMSNORM_BWD_DW_PARTIAL_F32;
        assert!(src.contains("long long rows_per_block ="));
        assert!(src.contains("long long row_start ="));
        assert!(src.contains("long long row_end ="));
        assert!(src.contains(
            "for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < hidden;"
        ));
        assert!(src.contains("for (long long row = row_start; row < row_end; row += 1)"));
        assert!(src.contains("long long idx ="));
    }

    #[test]
    fn split_k_dw_reduce_loop_index_is_declared_long_long() {
        let src = RMSNORM_BWD_DW_REDUCE_F32;
        // 外側ループは `base`（ブロック内全スレッド共通）で継続判定する
        // （codex-review P0 修正・PR #716。本ファイル冒頭「divergent
        // barrier」コメント参照）。`col` は `base + tid` から `long long`
        // で導出される。
        assert!(
            src.contains(
                "for (long long base = (long long)blockIdx.x * blockDim.x; base < hidden;"
            )
        );
        assert!(src.contains("long long col = base + tid;"));
    }

    /// `__syncthreads()` を含む外側ループの継続条件が `tid`（スレッド
    /// 固有の値）に依存しないことを検査する（codex-review P0 指摘・
    /// PR #716「部分ブロックが `__syncthreads()` に到達せずカーネルが
    /// 停止する」の回帰防止）。修正前は `col < hidden` という tid 依存の
    /// 条件でループ本体（バリアを含む）への進入可否が決まっており、
    /// `hidden` がブロック幅未満の部分ブロックでスレッド間の到達回数が
    /// 不一致になっていた。
    #[test]
    fn split_k_dw_reduce_outer_loop_condition_is_block_uniform() {
        let src = RMSNORM_BWD_DW_REDUCE_F32;
        assert!(
            !src.contains("col < hidden;\n         col +="),
            "外側ループの継続条件に tid 依存の `col` を使ってはならない（divergent barrier 回帰）"
        );
        assert!(src.contains("bool active = col < hidden;"));
        assert!(src.contains("if (active) {\n            dw[col] = acc;\n        }"));
    }

    /// 部分和は末尾の空 block でも無条件に書かれる（受け入れ基準 1・
    /// REQ-8「末尾要素ブロックの扱い」参照。条件分岐に包まれた回帰
    /// （`if (row_start < row_end) { dw_partial[...] = acc; }` 等）を
    /// 検出する）。
    #[test]
    fn split_k_dw_partial_writes_unconditionally() {
        let src = RMSNORM_BWD_DW_PARTIAL_F32;
        assert!(src.contains("dw_partial[(long long)b * (long long)hidden + i] = acc;"));
    }

    /// ソース中の `needle[` の各出現について、対応する `]` の直後
    /// （空白を読み飛ばした先）が代入演算子 `=`（`==` 等の比較演算子は
    /// 除く）であるかどうかを走査し、代入箇所（`needle[<式>] = ...`）の
    /// 個数を返す。添字式中に `[`／`]` を含む可能性（本カーネルには
    /// ないが将来の変更に備える）を考慮し、括弧の対応を数えて閉じ括弧
    /// を特定する（Bugbot 指摘・PR #716: 旧実装は
    /// `dw_partial[...] = `（`...` はリテラルの 3 文字）という実カーネル
    /// コードに存在しない文字列をそのまま `contains` していたため、
    /// 実際に `dw_partial[<式>] = <式>;` という書き戻しが混入しても
    /// 常に真になり回帰を検知できない vacuous check になっていた）。
    fn count_bracket_assignments(src: &str, needle: &str) -> usize {
        let bytes = src.as_bytes();
        let mut count = 0;
        let mut search_from = 0;
        while let Some(rel) = src[search_from..].find(needle) {
            let open = search_from + rel + needle.len() - 1;
            debug_assert_eq!(bytes[open], b'[');
            let mut depth = 1i32;
            let mut idx = open + 1;
            let mut close = None;
            while idx < bytes.len() {
                match bytes[idx] {
                    b'[' => depth += 1,
                    b']' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(idx);
                            break;
                        }
                    }
                    _ => {}
                }
                idx += 1;
            }
            let Some(close) = close else { break };
            let mut after = close + 1;
            while after < bytes.len() && (bytes[after] as char).is_whitespace() {
                after += 1;
            }
            if bytes.get(after) == Some(&b'=') && bytes.get(after + 1) != Some(&b'=') {
                count += 1;
            }
            search_from = close + 1;
        }
        count
    }

    /// 縮約カーネルの epilogue が `dw` へ 1 回だけ書き、中間の縮約結果を
    /// HBM へ書き戻す第 3 パスを作らないことを検査する（§3.1・受け入れ
    /// 基準 2）。
    #[test]
    fn split_k_dw_reduce_writes_dw_exactly_once_in_epilogue() {
        let src = RMSNORM_BWD_DW_REDUCE_F32;
        let occurrences = src.matches("dw[col] = acc;").count();
        assert_eq!(
            occurrences, 1,
            "epilogue の dw 書き出しは 1 回のみである契約"
        );

        // `dw_partial` は読み出し専用（`const float* __restrict__` 引数）
        // であることを確認する。`dw_partial[<式>] = ` という代入パターン
        // （縮約結果を HBM へ書いて読み戻す第 3 パスに相当）が実際に
        // 存在しないことを `count_bracket_assignments` で検査し、読み出し
        // 自体（`dw_partial[...]` が式の右辺に現れる形）はテストの前提
        // として最低 1 回存在することを確認する。
        assert!(
            src.contains("const float* __restrict__ dw_partial"),
            "dw_partial は const（読み出し専用）引数である契約"
        );
        assert!(
            src.matches("dw_partial[").count() >= 1,
            "dw_partial の読み出しが見つからない（テスト自体の前提崩れ）"
        );
        assert_eq!(
            count_bracket_assignments(src, "dw_partial["),
            0,
            "縮約カーネルは dw_partial へ書き出さない契約"
        );
    }

    /// smem double buffer の静的サイズが [`RMSNORM_DW_REDUCE_BATCH`]／
    /// [`RMSNORM_DW_REDUCE_BLOCK_DIM`] と一致することを回帰検出する
    /// （advisor 指摘: 定数変更時にソース側の宣言が黙って乖離するのを
    /// 防ぐため、期待値をハードコードせず定数から `format!` で組み立てる）。
    #[test]
    fn split_k_dw_reduce_smem_size_matches_batch_and_block_dim_consts() {
        let expected = format!(
            "__shared__ float smem[2][{}][{}];",
            RMSNORM_DW_REDUCE_BATCH, RMSNORM_DW_REDUCE_BLOCK_DIM
        );
        assert!(
            RMSNORM_BWD_DW_REDUCE_F32.contains(&expected),
            "smem 宣言が RMSNORM_DW_REDUCE_BATCH/RMSNORM_DW_REDUCE_BLOCK_DIM と乖離している: \
             expected `{expected}`"
        );
        // バッチ内アンロールのループ境界（`j < RMSNORM_DW_REDUCE_BATCH`）も
        // 定数と揃っていることを確認する。
        let expected_unroll_bound = format!("j < {}", RMSNORM_DW_REDUCE_BATCH);
        assert!(RMSNORM_BWD_DW_REDUCE_F32.contains(&expected_unroll_bound));
        let expected_batch_advance = format!("(batch + 1) * {}", RMSNORM_DW_REDUCE_BATCH);
        assert!(RMSNORM_BWD_DW_REDUCE_F32.contains(&expected_batch_advance));
    }

    /// [`RMSNORM_DW_REDUCE_BATCH`]（4）が num_batches 導出の除数（`+3`／
    /// `/4`）と一致することを検査する（`(num_blocks + BATCH - 1) / BATCH`
    /// の意図。定数変更時の乖離を防ぐ）。
    #[test]
    fn split_k_dw_reduce_num_batches_uses_batch_const() {
        let expected = format!(
            "int num_batches = (num_blocks + {}) / {};",
            RMSNORM_DW_REDUCE_BATCH - 1,
            RMSNORM_DW_REDUCE_BATCH
        );
        assert!(RMSNORM_BWD_DW_REDUCE_F32.contains(&expected));
    }
}
