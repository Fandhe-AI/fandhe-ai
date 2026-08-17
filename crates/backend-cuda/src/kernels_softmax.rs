//! online softmax（FlashAttention-2 型）順伝播カーネル（NVRTC 実行時
//! コンパイル用の静的文字列。イシュー #594）。
//!
//! `softmax.rs`（呼び出し元）は本モジュールの 2 定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る。構造イディオム（1 CTA = 1 warp・
//! persistent block・1 パス／2 パスの 2 経路・`long long` ループ添字・
//! `float4` ベクトル化条件）は `kernels_rmsnorm.rs`（#592）をそのまま踏襲
//! する（`rmsnorm.rs`／`kernels_rmsnorm.rs` 冒頭コメント参照）。本コメントは
//! RMSNorm と共通の設計判断は繰り返さず、softmax 固有の判断のみを記す。
//!
//! # 数値契約（online softmax 固有）
//!
//! - **`log2(e)` 事前スケール + `exp2f` のみ使用**: `scale` 引数（ホスト側
//!   `softmax.rs::CudaSoftmax::run_softmax_f32` が `std::f32::consts::LOG2_E`
//!   を渡す。将来 attention 融合時の `log2(e)/sqrt(d)` 合成を見込んだ引数化）
//!   を入力へ乗算してから `exp2f` を適用することで、`e^x = 2^(x*log2(e))`
//!   の恒等式によりカーネル内は **`expf` を一切使わず `exp2f` のみ**で
//!   完結する（GPU のハードウェア `exp2` 命令に直結させる定石）。
//! - **オンライン最大値更新・補正係数スキップ**: 各 lane は担当要素を
//!   走査しながら `m`（走査済み最大値。exp2 ドメイン）・`l`（走査済み
//!   分母）を同時更新する。`m_new = max(m, v)` が **更新されたときのみ**
//!   `l *= exp2f(m - m_new)` を実行し（更新されない場合は `correction = 1`
//!   の乗算を省く）、続けて `l += exp2f(v - m_new)` する。warp 内の
//!   32 lane 間の結合（5 段 `__shfl_xor_sync` butterfly。offset
//!   16→8→4→2→1）でも同じ補正係数スキップを適用する。
//! - **2 段シャッフルではなく 5 段 butterfly を採る理由**: 参照実装
//!   （metal-flash-attention）は幅 1・8 の 2 段シャッフルで済ませるが、
//!   これは Metal simdgroup の Morton 順（zigzag）レーンレイアウト
//!   固有の最適化であり（`docs/backend-metal-morton-mapping-decision.md`
//!   の「標準 API 下では Morton 順マッピングが適用できない」という判断と
//!   根が同じ）、CUDA の warp は線形（lane 0..31 が連番）レーンレイアウト
//!   のため 2 段では全 32 lane を結合しきれない。よって
//!   `kernels_rmsnorm.rs` の Σx² butterfly reduction と同じ 5 段
//!   （offset 16→1）を採用し、CUDA の線形レーンレイアウトで全 lane の
//!   `(m, l)` ペアを正しく結合する。
//! - **境界マスク定数（有限値。`-INFINITY` 不使用）**: lane の初期値は
//!   `m = SOFTMAX_MASK_E2`（`-0.875f * __FLT_MAX__`。exp2 ドメインの
//!   有限な大きな負値）・`l = 0`。`__FLT_MAX__` は Clang/NVRTC の
//!   プリプロセッサ組み込みマクロ（`<cfloat>`/`<float.h>` の include を
//!   要求しない。`kernels_rmsnorm.rs` 同様「ビルド時に nvrtc/CUDA
//!   ヘッダを一切要求しない」契約を維持する）。`-INFINITY` を直接使うと
//!   「担当要素ゼロの lane 同士」が結合する際に `(-INF) - (-INF) = NaN`
//!   が発生しうる（`m_t - m` の差分計算が両者とも `-INF` になるケース）。
//!   有限マージン値であれば `mask - mask == 0.0`（有限）となり NaN を
//!   生まない。マージン係数 0.875（`f32::MAX` に対して掛ける）・値の
//!   妥当性は `softmax.rs` の数値検証テスト（`mask_value_e2_f32` 系）で
//!   個別に確認する（参照実装の値を無検証で採用しない。実装計画 §3.3）。
//!   ホスト側の `softmax.rs::SOFTMAX_MASK_E2_F32` 定数と同じ IEEE 754
//!   単精度乗算（`-0.875 * f32::MAX`）であるため、ホスト・デバイス間で
//!   ビット同一の値になる。
//! - **意味論の正**: `tests/softmax_parity.rs` 内のテスト専用 CPU 参照
//!   実装（`f32::mul_add` 使用）が意味論の正である。`onnx-interop` 側の
//!   素朴実装（3 パス）とも parity を取るが、本カーネルの数値的な正は
//!   `backend_cpu::parity::assert_parity`（REQ-2 複合判定）で判定する。
//!
//! # ベクトル化・ループ添字・REQ-8 境界検査
//!
//! `kernels_rmsnorm.rs` 冒頭コメント「ベクトル化と REQ-8」「ループ添字の
//! オーバーフロー安全性」と同一方針（`hidden` を `cols` に読み替える）。
//! `float4` ロードは `cols % 4 == 0` の場合のみ適用し、手動境界チェック
//! `if (base + 3 < cols)` はベクトル化適用時も維持する。grid-stride
//! ループの添字（`row`／`base`／`i`）は `long long` で宣言し、`cols`／
//! `rows` が `i32::MAX` 近傍でも `int` 添字の signed overflow による境界
//! チェック迂回を防ぐ（同ファイルの `#[cfg(test)]` ソース文字列回帰
//! テストで検査する）。

/// 1 スレッドブロックあたりのスレッド数（1 warp = 32 固定）。
/// `kernels_rmsnorm.rs::RMSNORM_BLOCK_DIM` と同じ理由（`__shfl_xor_sync`／
/// `__syncwarp` による warp 内演算を前提とするため 32 固定）。
pub const SOFTMAX_BLOCK_DIM: u32 = 32;

/// 1 パス経路（動的 SMEM 常駐）: `x` を 1 回だけロードしつつ online
/// softmax（`(m, l)` 同時更新）を行い、warp 内 butterfly 結合 →
/// 正規化 → 書き出しまでを中間テンソルの HBM 書き出しなしで完結する。
///
/// 起動時に `extern __shared__ float smem[]` へ `cols * 4` バイトの動的
/// 共有メモリを割り当てる契約（`softmax.rs::CudaSoftmax::run_softmax_f32_raw`
/// が `LaunchConfig::shared_mem_bytes` で指定する）。適用条件（`cols * 4`
/// が SMEM 予算に収まること）はホスト側 `softmax_route`
/// （`rmsnorm.rs::rmsnorm_route` を共有再利用）が判定し、収まらない場合は
/// [`SOFTMAX_F32_TWOPASS`] へルーティングする。
pub const SOFTMAX_F32_ONEPASS: &str = r#"
#define SOFTMAX_MASK_E2 (-0.875f * __FLT_MAX__)

extern "C" __global__ void softmax_f32_onepass(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols,
    float scale)
{
    extern __shared__ float smem[];
    int lane = threadIdx.x;
    // cols % 4 != 0 の場合は行間アライメント位相がずれるため
    // ベクトル化を一切適用しない（`kernels_rmsnorm.rs` 冒頭コメント
    // 「ベクトル化と REQ-8」と同じ理由）。
    int vec_cols = (cols % 4 == 0) ? cols : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)cols;
        float* out_row = out + row * (long long)cols;

        // online softmax の走査済み最大値（exp2 ドメイン）・走査済み分母。
        // `m` の初期値は有限マージン値（本ファイル冒頭コメント「境界
        // マスク定数」参照）。
        float m = SOFTMAX_MASK_E2;
        float l = 0.0f;

        // ベクトル化ロード（float4）+ SMEM への raw 値格納 + online 更新。
        // `base` は `long long`（本ファイル冒頭コメント「ベクトル化・
        // ループ添字・REQ-8 境界検査」参照）。
        for (long long base = lane * 4; base < vec_cols; base += 32 * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                smem[base + 0] = v4.x;
                smem[base + 1] = v4.y;
                smem[base + 2] = v4.z;
                smem[base + 3] = v4.w;
                float vals[4] = {
                    v4.x * scale, v4.y * scale, v4.z * scale, v4.w * scale
                };
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float v = vals[k];
                    float m_new = fmaxf(m, v);
                    // 補正係数スキップ: 最大値が更新されないときは
                    // `correction = 1` の乗算を省く（本ファイル冒頭
                    // コメント「オンライン最大値更新・補正係数スキップ」
                    // 参照）。
                    if (m_new > m) {
                        l *= exp2f(m - m_new);
                    }
                    l += exp2f(v - m_new);
                    m = m_new;
                }
            }
        }
        // スカラー経路: cols % 4 != 0 なら全要素、それ以外は端要素なし。
        for (long long i = (long long)vec_cols + lane; i < cols; i += 32) {
            float raw = x_row[i];
            smem[i] = raw;
            float v = raw * scale;
            float m_new = fmaxf(m, v);
            if (m_new > m) {
                l *= exp2f(m - m_new);
            }
            l += exp2f(v - m_new);
            m = m_new;
        }

        // 上記ロードループが書いた smem を他レーンが読めるようにする
        // warp 内バリア（`__syncthreads()` は使わない。`kernels_rmsnorm.rs`
        // と同じ設計方針）。
        __syncwarp(0xffffffffu);

        // warp 内 (m, l) ペアの butterfly 結合（5 段。offset 16→1。本
        // ファイル冒頭コメント「2 段シャッフルではなく 5 段 butterfly を
        // 採る理由」参照）。等値側（`m_t == m` または `m_t == m_o`）は
        // 補正係数スキップを適用する。
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m_o = __shfl_xor_sync(0xffffffffu, m, offset);
            float l_o = __shfl_xor_sync(0xffffffffu, l, offset);
            float m_t = fmaxf(m, m_o);
            float l_self = (m_t > m) ? (l * exp2f(m - m_t)) : l;
            float l_peer = (m_t > m_o) ? (l_o * exp2f(m_o - m_t)) : l_o;
            l = l_self + l_peer;
            m = m_t;
        }

        float inv_l = 1.0f / l;

        for (long long i = lane; i < cols; i += 32) {
            float v = smem[i] * scale;
            out_row[i] = exp2f(v - m) * inv_l;
        }

        // 次の行（grid-stride ループの次反復）が smem を上書きする前に、
        // 全レーンの上記読み出しが完了していることを保証する。
        __syncwarp(0xffffffffu);
    }
}
"#;

/// 2 パス経路（global 再読）: 1 パス経路の SMEM 予算を超える行長
/// （`softmax_route` が `TwoPass` を選んだ場合）で使う。Pass 1 で `x` を
/// global から読み online softmax の `(m, l)` を確定、Pass 2 で同一
/// カーネル・同一行ループ内で `x` を再度 global から読み
/// `exp2f(v - m) * inv_l` を書き出す（Pass 1 とビット同一の
/// `v = x[i] * scale` を再計算する。決定性維持。`kernels_rmsnorm.rs`
/// の 2 パス経路と同じ「中間テンソルを書き出さない」構造）。
pub const SOFTMAX_F32_TWOPASS: &str = r#"
#define SOFTMAX_MASK_E2 (-0.875f * __FLT_MAX__)

extern "C" __global__ void softmax_f32_twopass(
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows,
    int cols,
    float scale)
{
    int lane = threadIdx.x;
    int vec_cols = (cols % 4 == 0) ? cols : 0;

    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* x_row = x + row * (long long)cols;
        float* out_row = out + row * (long long)cols;

        // Pass 1: online (m, l) を計算する（smem 非使用・global 直読）。
        float m = SOFTMAX_MASK_E2;
        float l = 0.0f;
        for (long long base = lane * 4; base < vec_cols; base += 32 * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float vals[4] = {
                    v4.x * scale, v4.y * scale, v4.z * scale, v4.w * scale
                };
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float v = vals[k];
                    float m_new = fmaxf(m, v);
                    if (m_new > m) {
                        l *= exp2f(m - m_new);
                    }
                    l += exp2f(v - m_new);
                    m = m_new;
                }
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += 32) {
            float v = x_row[i] * scale;
            float m_new = fmaxf(m, v);
            if (m_new > m) {
                l *= exp2f(m - m_new);
            }
            l += exp2f(v - m_new);
            m = m_new;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m_o = __shfl_xor_sync(0xffffffffu, m, offset);
            float l_o = __shfl_xor_sync(0xffffffffu, l, offset);
            float m_t = fmaxf(m, m_o);
            float l_self = (m_t > m) ? (l * exp2f(m - m_t)) : l;
            float l_peer = (m_t > m_o) ? (l_o * exp2f(m_o - m_t)) : l_o;
            l = l_self + l_peer;
            m = m_t;
        }

        float inv_l = 1.0f / l;

        // Pass 2: x を再度 global から読み、Pass 1 とビット同一の
        // `v = x[i] * scale` を再計算して書き出す（同一カーネル・同一
        // 行ループ内で完結。中間テンソルは書き出さない）。
        for (long long base = lane * 4; base < vec_cols; base += 32 * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float4 o;
                o.x = exp2f(v4.x * scale - m) * inv_l;
                o.y = exp2f(v4.y * scale - m) * inv_l;
                o.z = exp2f(v4.z * scale - m) * inv_l;
                o.w = exp2f(v4.w * scale - m) * inv_l;
                *reinterpret_cast<float4*>(out_row + base) = o;
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += 32) {
            out_row[i] = exp2f(x_row[i] * scale - m) * inv_l;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// grid-stride ループの添字（`row`／`base`／`i`）が `long long` で
    /// 宣言されていることをソース文字列に対して検査する
    /// （`kernels_rmsnorm.rs::tests::onepass_and_twopass_loop_indices_are_declared_long_long`
    /// と同じ理由・同じ検査パターン）。
    #[test]
    fn onepass_and_twopass_loop_indices_are_declared_long_long() {
        for src in [SOFTMAX_F32_ONEPASS, SOFTMAX_F32_TWOPASS] {
            assert!(
                src.contains("for (long long row = blockIdx.x; row < rows; row += gridDim.x)"),
                "row ループ添字が long long で宣言されていない"
            );
            assert!(
                src.contains("for (long long base = lane * 4; base < vec_cols; base += 32 * 4)"),
                "base ループ添字が long long で宣言されていない"
            );
            assert!(
                src.contains("for (long long i = (long long)vec_cols + lane; i < cols; i += 32)"),
                "i ループ添字が long long で宣言されていない"
            );
            assert!(
                !src.contains("for (int row = blockIdx.x")
                    && !src.contains("for (int base = lane * 4")
                    && !src.contains("for (int i = vec_cols + lane")
                    && !src.contains("for (int i = (int)vec_cols + lane"),
                "grid-stride ループ添字が int へ縮退している"
            );
        }
    }

    /// 意味論契約（本ファイル冒頭コメント「log2(e) 事前スケール +
    /// exp2f のみ使用」）: カーネルは `expf` を一切呼ばず `exp2f` のみを
    /// 使う。`expf` の部分文字列一致は `exp2f` にも `"exp2f".contains("expf")`
    /// が偽（`exp2f` は `e`,`x`,`p`,`2`,`f` の並びであり `expf` の並びを
    /// 部分文字列として含まない）ため誤検出しない。
    #[test]
    fn onepass_and_twopass_use_exp2f_only_not_expf() {
        for src in [SOFTMAX_F32_ONEPASS, SOFTMAX_F32_TWOPASS] {
            assert!(src.contains("exp2f("), "exp2f が使われていない");
            assert!(
                !src.contains("expf("),
                "expf が使われている（exp2f のみ使用する契約）"
            );
        }
    }

    /// 手動境界チェック（REQ-8）: ベクトル化ロード／ストアが
    /// `if (base + 3 < cols)` の境界チェックを維持していることを検査する。
    #[test]
    fn onepass_and_twopass_keep_manual_bounds_check() {
        for src in [SOFTMAX_F32_ONEPASS, SOFTMAX_F32_TWOPASS] {
            assert!(
                src.contains("if (base + 3 < cols)"),
                "float4 ベクトル化の手動境界チェックが見当たらない"
            );
        }
    }

    /// 境界マスク定数（本ファイル冒頭コメント「境界マスク定数」参照）:
    /// `-INFINITY` を直接使わず、有限の大きな負値マクロを使うことを検査
    /// する。
    #[test]
    fn onepass_and_twopass_use_finite_mask_not_infinity() {
        for src in [SOFTMAX_F32_ONEPASS, SOFTMAX_F32_TWOPASS] {
            assert!(
                src.contains("#define SOFTMAX_MASK_E2 (-0.875f * __FLT_MAX__)"),
                "有限マージンの境界マスク定数が定義されていない"
            );
            assert!(
                !src.contains("-INFINITY") && !src.contains("-INFINITY)"),
                "境界マスクに -INFINITY を直接使用している"
            );
        }
    }
}
