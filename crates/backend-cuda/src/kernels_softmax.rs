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
//! - **`log2(e)` 事前スケールは「最大値との差分を取った後」に適用し
//!   `exp2f` のみを使用**: `scale` 引数（ホスト側
//!   `softmax.rs::CudaSoftmax::run_softmax_f32` が `std::f32::consts::LOG2_E`
//!   を渡す。将来 attention 融合時の `log2(e)/sqrt(d)` 合成を見込んだ引数化）
//!   は、オンライン最大値 `m`（**入力と同じ生ドメイン。スケール未適用**）
//!   との差分 `(raw - m) * scale` を計算する `exp2f` 呼び出し直前でのみ
//!   乗算する。`e^x = 2^(x*log2(e))` の恒等式によりカーネル内は
//!   **`expf` を一切使わず `exp2f` のみ**で完結する（GPU のハードウェア
//!   `exp2` 命令に直結させる定石）という性質は保ったまま、`raw * scale`
//!   を先に計算しない（イシュー #594 PR #712 codex-review 指摘・P1 修正:
//!   旧実装は `v = raw * scale` を最大値減算より先に行っていたため、
//!   `raw = f32::MAX` のような有限な極値でも `v = +Inf` へオーバーフロー
//!   し、続く `m_new = max(m, v) = +Inf` と合わせて `exp2f(v - m_new) =
//!   exp2f(Inf - Inf) = NaN` が発生していた。CPU 参照実装
//!   〈`tests/softmax_parity.rs`〉は max-sub 後に指数を取るため同じ
//!   `±f32::MAX` 付近の入力でも有限な結果になり、バックエンド間の数値
//!   契約〈`.claude/rules/coding-rust.md` の FMA 契約統一・複合誤差判定と
//!   同種の一致要件〉が破られていた。差分 `raw - m` は `m` が走査済み
//!   最大値であるため常に `<= 0` であり、スケール未適用の生ドメインで
//!   引き算することで同種のオーバーフローを避ける）。
//! - **オンライン最大値更新・補正係数スキップ**: 各 lane は担当要素を
//!   走査しながら `m`（走査済み最大値。**生ドメイン**。上記参照）・`l`
//!   （走査済み分母）を同時更新する。`m_new = max(m, raw)` が **更新
//!   されたときのみ** `l *= exp2f((m - m_new) * scale)` を実行し
//!   （更新されない場合は `correction = 1` の乗算を省く）、続けて
//!   `l += exp2f((raw - m_new) * scale)` する。warp 内の 32 lane 間の
//!   結合（5 段 `__shfl_xor_sync` butterfly。offset 16→8→4→2→1）でも
//!   同じ補正係数スキップを適用する。
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
//! - **境界マスク定数（有限値。`-INFINITY` 不使用。イシュー #594 PR #712
//!   codex-review 指摘・Cursor Bugbot 指摘の P1 修正で `-0.875f *
//!   __FLT_MAX__` マージン方式から本方式へ変更）**: lane の初期値は
//!   `m = SOFTMAX_MASK_E2`（自前リテラル `-3.402823466e+38f`。`f32` が
//!   表現できる**最小の有限値**であり `f32::MAX` へビット正確に丸まる。
//!   **生ドメイン**〈上記「`log2(e)` 事前スケール」参照。`m` は
//!   スケール未適用のため、このマスク値も生ドメインの値として機能する〉・
//!   `l = 0`。当初は Clang/NVRTC のプリプロセッサ組み込みマクロ
//!   `__FLT_MAX__`（`<cfloat>`/`<float.h>` の include を要求しない）を
//!   使っていたが、compute_121（DGX Spark GB10）向け NVRTC コンパイルで
//!   `__FLT_MAX__` が未定義になり softmax parity テストが失敗する実測が
//!   あったため（イシュー #1101）、ターゲット・NVRTC バージョンに依存
//!   しない自前の定数リテラルへ置換した（`kernels_rmsnorm.rs` 同様
//!   「ビルド時に nvrtc/CUDA ヘッダを一切要求しない」契約は維持）。
//!   旧実装（`-0.875f * __FLT_MAX__` のマージン値）は「行の全要素がこの
//!   マスク値未満の正規の有限入力」（例: 全要素 `-f32::MAX`）で `m` が
//!   一度も更新されず、`exp2f((raw - m) * scale)` が 0 へアンダーフロー
//!   して `l == 0` のまま `inv_l == Inf`・最終出力 `0 * Inf == NaN` に
//!   なる欠陥があった。**マスク値は `f32` の有効な値域の下限その
//!   ものであるため、いかなる正規の有限入力 `raw` についても常に
//!   `raw >= m_init` が成立する**。よって `fmaxf` によるオンライン最大値
//!   `m` は非空の行では必ずその行の真の最大値へ収束する（`raw` が
//!   マスク値そのもの、すなわち `raw == m` の場合は `m_new == m` となり
//!   `m` は「更新されない」が、これは既に真の値と一致しているため無害
//!   であり、旧実装の「マスク値未満の入力に対して `m` が真の値へ到達
//!   できない」欠陥とは異なる）。最大値を保持する要素は常に
//!   `exp2f((raw - m) * scale) == exp2f(0) == 1` を `l` へ加算するため、
//!   非空の行では常に `l >= 1` となり `inv_l = 1.0f / l` が有限になる
//!   （旧実装の `l == 0 → inv_l == Inf → 0 * Inf == NaN` という失敗
//!   経路がそもそも成立しなくなる）。加えて「担当要素ゼロの lane 同士」
//!   が結合する場合も `m_t == m == m_o`（マスク値どうしで等しい）となり、
//!   本ファイルの補正係数スキップ条件（`m_new > m` のときのみ `exp2f`
//!   の差分計算を行う。この条件は **狭義**の `>` であり `m_new == m` の
//!   等値ケースは分岐に入らない）により減算自体が発生しないため NaN を
//!   生まない（`-INFINITY` を直接使わない理由と同じ設計。
//!   `(-INF) - (-INF) = NaN` を避けるのと同様、等値の場合は分岐で減算を
//!   スキップする構造そのものが NaN を防ぐ。この性質は「実データを持つ
//!   lane の `m` がたまたまマスク値と一致する」ケース——例えば行の最大値
//!   自体が `-f32::MAX` であるケース——にも同様に適用され、空 lane との
//!   結合に限らない）。値の妥当性は `softmax.rs` の数値検証テスト
//!   （`mask_value_e2_f32` 系）で個別に確認する（参照実装の値を無検証で
//!   採用しない。実装計画 §3.3）。ホスト側の
//!   `softmax.rs::SOFTMAX_MASK_E2_F32` 定数（`-f32::MAX`）と同一の値で
//!   あるため、ホスト・デバイス間でビット同一の値になる。
//!   **なお全要素が `-INFINITY` である行は本方式でも `l == 0` のまま
//!   （`exp2f(-inf) == 0` が全要素で加算されるため）であり出力は NaN に
//!   なるが、これは数学的に softmax(全要素 `-inf`) が `0/0` で未定義な
//!   ケースであり、CPU 参照実装（`tests/softmax_parity.rs`）も
//!   `(-inf) - (-inf) = NaN` で同じく NaN を返すため、バックエンド間の
//!   数値契約は破られない（本欠陥修正の対象は有限入力のみ）。
//! - **意味論の正**: `tests/softmax_parity.rs` 内のテスト専用 CPU 参照
//!   実装（`f32::mul_add` 使用）が意味論の正である。`onnx-interop` 側の
//!   素朴実装（3 パス）とも parity を取るが、本カーネルの数値的な正は
//!   `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 複合判定）で判定する。
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
#define SOFTMAX_MASK_E2 (-3.402823466e+38f)

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

        // online softmax の走査済み最大値（**生ドメイン**。スケール未適用。
        // 本ファイル冒頭コメント「`log2(e)` 事前スケールは『最大値との
        // 差分を取った後』に適用」参照）・走査済み分母。`m` の初期値は
        // `f32` の値域下限（本ファイル冒頭コメント「境界マスク定数」参照）。
        float m = SOFTMAX_MASK_E2;
        float l = 0.0f;

        // ベクトル化ロード（float4）+ SMEM への raw 値格納 + online 更新。
        // `base` は `long long`（本ファイル冒頭コメント「ベクトル化・
        // ループ添字・REQ-8 境界検査」参照）。`scale` は最大値との差分
        // 計算後の `exp2f` 直前でのみ乗算する（イシュー #594 PR #712
        // codex-review 指摘・P1 修正: `raw * scale` を先に計算すると
        // `raw = f32::MAX` 等の有限な極値で `+Inf` へオーバーフローし
        // `exp2f(Inf - Inf) = NaN` になるため）。
        for (long long base = lane * 4; base < vec_cols; base += 32 * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                smem[base + 0] = v4.x;
                smem[base + 1] = v4.y;
                smem[base + 2] = v4.z;
                smem[base + 3] = v4.w;
                float vals[4] = { v4.x, v4.y, v4.z, v4.w };
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float raw = vals[k];
                    float m_new = fmaxf(m, raw);
                    // 補正係数スキップ: 最大値が更新されないときは
                    // `correction = 1` の乗算を省く（本ファイル冒頭
                    // コメント「オンライン最大値更新・補正係数スキップ」
                    // 参照）。
                    if (m_new > m) {
                        l *= exp2f((m - m_new) * scale);
                    }
                    l += exp2f((raw - m_new) * scale);
                    m = m_new;
                }
            }
        }
        // スカラー経路: cols % 4 != 0 なら全要素、それ以外は端要素なし。
        for (long long i = (long long)vec_cols + lane; i < cols; i += 32) {
            float raw = x_row[i];
            smem[i] = raw;
            float m_new = fmaxf(m, raw);
            if (m_new > m) {
                l *= exp2f((m - m_new) * scale);
            }
            l += exp2f((raw - m_new) * scale);
            m = m_new;
        }

        // 上記ロードループが書いた smem を他レーンが読めるようにする
        // warp 内バリア（`__syncthreads()` は使わない。`kernels_rmsnorm.rs`
        // と同じ設計方針）。
        __syncwarp(0xffffffffu);

        // warp 内 (m, l) ペアの butterfly 結合（5 段。offset 16→1。本
        // ファイル冒頭コメント「2 段シャッフルではなく 5 段 butterfly を
        // 採る理由」参照）。等値側（`m_t == m` または `m_t == m_o`）は
        // 補正係数スキップを適用する。`m`／`m_o` は生ドメインのため差分
        // 計算後に `scale` を乗算する（上記と同じ理由）。
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m_o = __shfl_xor_sync(0xffffffffu, m, offset);
            float l_o = __shfl_xor_sync(0xffffffffu, l, offset);
            float m_t = fmaxf(m, m_o);
            float l_self = (m_t > m) ? (l * exp2f((m - m_t) * scale)) : l;
            float l_peer = (m_t > m_o) ? (l_o * exp2f((m_o - m_t) * scale)) : l_o;
            l = l_self + l_peer;
            m = m_t;
        }

        float inv_l = 1.0f / l;

        for (long long i = lane; i < cols; i += 32) {
            out_row[i] = exp2f((smem[i] - m) * scale) * inv_l;
        }

        // 次の行（grid-stride ループの次反復）が smem を上書きする前に、
        // 全レーンの上記読み出しが完了していることを保証する。
        __syncwarp(0xffffffffu);
    }
}
"#;

/// 2 パス経路（global 再読）: 1 パス経路の SMEM 予算を超える行長
/// （`softmax_route` が `TwoPass` を選んだ場合）で使う。Pass 1 で `x` を
/// global から読み online softmax の `(m, l)`（`m` は生ドメイン。本
/// ファイル冒頭コメント「`log2(e)` 事前スケール」参照）を確定、Pass 2 で
/// 同一カーネル・同一行ループ内で `x` を再度 global から読み
/// `exp2f((raw - m) * scale) * inv_l` を書き出す（Pass 1 とビット同一の
/// `raw = x[i]` を再計算する。決定性維持。`kernels_rmsnorm.rs` の
/// 2 パス経路と同じ「中間テンソルを書き出さない」構造）。
pub const SOFTMAX_F32_TWOPASS: &str = r#"
#define SOFTMAX_MASK_E2 (-3.402823466e+38f)

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
        // `m` は生ドメイン（スケール未適用）。`scale` は差分計算後の
        // `exp2f` 直前でのみ乗算する（イシュー #594 PR #712 codex-review
        // 指摘・P1 修正。本ファイル冒頭コメント参照）。
        float m = SOFTMAX_MASK_E2;
        float l = 0.0f;
        for (long long base = lane * 4; base < vec_cols; base += 32 * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float vals[4] = { v4.x, v4.y, v4.z, v4.w };
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float raw = vals[k];
                    float m_new = fmaxf(m, raw);
                    if (m_new > m) {
                        l *= exp2f((m - m_new) * scale);
                    }
                    l += exp2f((raw - m_new) * scale);
                    m = m_new;
                }
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += 32) {
            float raw = x_row[i];
            float m_new = fmaxf(m, raw);
            if (m_new > m) {
                l *= exp2f((m - m_new) * scale);
            }
            l += exp2f((raw - m_new) * scale);
            m = m_new;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float m_o = __shfl_xor_sync(0xffffffffu, m, offset);
            float l_o = __shfl_xor_sync(0xffffffffu, l, offset);
            float m_t = fmaxf(m, m_o);
            float l_self = (m_t > m) ? (l * exp2f((m - m_t) * scale)) : l;
            float l_peer = (m_t > m_o) ? (l_o * exp2f((m_o - m_t) * scale)) : l_o;
            l = l_self + l_peer;
            m = m_t;
        }

        float inv_l = 1.0f / l;

        // Pass 2: x を再度 global から読み、Pass 1 とビット同一の
        // `raw = x[i]` を再計算し `exp2f((raw - m) * scale)` を書き出す
        // （同一カーネル・同一行ループ内で完結。中間テンソルは書き出さ
        // ない）。
        for (long long base = lane * 4; base < vec_cols; base += 32 * 4) {
            if (base + 3 < cols) {
                float4 v4 = *reinterpret_cast<const float4*>(x_row + base);
                float4 o;
                o.x = exp2f((v4.x - m) * scale) * inv_l;
                o.y = exp2f((v4.y - m) * scale) * inv_l;
                o.z = exp2f((v4.z - m) * scale) * inv_l;
                o.w = exp2f((v4.w - m) * scale) * inv_l;
                *reinterpret_cast<float4*>(out_row + base) = o;
            }
        }
        for (long long i = (long long)vec_cols + lane; i < cols; i += 32) {
            out_row[i] = exp2f((x_row[i] - m) * scale) * inv_l;
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

    /// `log2(e)` 事前スケール順序（本ファイル冒頭コメント「`log2(e)`
    /// 事前スケール」参照。イシュー #594 PR #712 codex-review 指摘・P1・
    /// threadId PRRT_kwDOTuUCJc6ZtHwL の直接的な回帰テスト）: ソース文字列
    /// に対し「最大値との差分を取った後に `scale` を乗算する」形
    /// （`(raw - m_new) * scale` 等）が存在し、「差分より先に `scale` を
    /// 乗算する」形（`raw * scale` 等。旧実装は `float v = raw * scale;`
    /// のように最大値減算より先にスケールしていた）が存在しないことを
    /// 検査する。実機必須の数値回帰（`f32::MAX` 入力）は
    /// `softmax_parity.rs::softmax_numerically_stable_for_extreme_inputs`
    /// （`#[ignore]`）で別途検証するため、本テストは実機非依存で
    /// ソースコード構造そのものの後退を防ぐ役割を持つ。
    #[test]
    fn onepass_and_twopass_scale_multiply_happens_after_max_subtraction() {
        for src in [SOFTMAX_F32_ONEPASS, SOFTMAX_F32_TWOPASS] {
            assert!(
                src.contains("exp2f((raw - m_new) * scale)"),
                "online 更新の exp2f が『差分後に scale 乗算』の形になっていない"
            );
            assert!(
                src.contains("exp2f((m - m_new) * scale)"),
                "補正係数の exp2f が『差分後に scale 乗算』の形になっていない"
            );
            assert!(
                !src.contains("= raw * scale"),
                "raw に scale を先乗算する旧実装のパターンが残っている（最大値減算前のオーバーフローを招く）"
            );
        }
        // 正規化書き出しの exp2f も同じ順序であることを個別に検査する
        // （1 パス: smem 読み出し／2 パス: x_row・v4 直読の 2 形態）。
        assert!(
            SOFTMAX_F32_ONEPASS.contains("exp2f((smem[i] - m) * scale)"),
            "1 パス経路の正規化書き出しが『差分後に scale 乗算』の形になっていない"
        );
        assert!(
            SOFTMAX_F32_TWOPASS.contains("exp2f((x_row[i] - m) * scale)")
                && SOFTMAX_F32_TWOPASS.contains("exp2f((v4.x - m) * scale)"),
            "2 パス経路の正規化書き出しが『差分後に scale 乗算』の形になっていない"
        );
    }

    /// 境界マスク定数（本ファイル冒頭コメント「境界マスク定数」参照）:
    /// `-INFINITY` を直接使わず、`f32` の値域下限（自前リテラル
    /// `-3.402823466e+38f`）を使うことを検査する。加えて `__FLT_MAX__`
    /// （NVRTC 組み込みマクロ。compute_121 で未定義になり NVRTC
    /// コンパイルが失敗する実測があった。イシュー #1101）へ逆戻りして
    /// いないことも fail-closed に検査する。
    #[test]
    fn onepass_and_twopass_use_finite_mask_not_infinity() {
        for src in [SOFTMAX_F32_ONEPASS, SOFTMAX_F32_TWOPASS] {
            assert!(
                src.contains("#define SOFTMAX_MASK_E2 (-3.402823466e+38f)"),
                "境界マスク定数が f32 値域下限（-3.402823466e+38f）で定義されていない"
            );
            assert!(
                !src.contains("-INFINITY") && !src.contains("-INFINITY)"),
                "境界マスクに -INFINITY を直接使用している"
            );
            assert!(
                !src.contains("__FLT_MAX__"),
                "境界マスクが NVRTC 組み込みマクロ __FLT_MAX__ に逆戻りしている（イシュー #1101）"
            );
        }
    }

    /// 自前リテラル `3.402823466e+38` が `f32::MAX` へビット正確に丸まる
    /// ことを検証する（イシュー #1101）。この性質が崩れると
    /// `softmax.rs::SOFTMAX_MASK_E2_F32`（`-f32::MAX`）とのホスト・
    /// デバイス間ビット同一契約（本ファイル冒頭コメント「境界マスク
    /// 定数」参照）が破られる。
    #[test]
    fn mask_literal_parses_to_bit_exact_f32_max() {
        let parsed: f32 = "3.402823466e+38".parse().unwrap();
        assert_eq!(
            parsed.to_bits(),
            f32::MAX.to_bits(),
            "自前リテラルが f32::MAX へビット正確に丸まらない"
        );
    }
}
