//! MaxPool／AvgPool／AdaptiveAvgPool（2d。イシュー #1729・親 #1607・
//! 設計 `docs/pooling-ops-design.md`）の CUDA C カーネルソース（NVRTC
//! 実行時コンパイル用の静的文字列）。
//!
//! `pooling.rs`（呼び出し元）は本モジュールの定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る。`kernels_im2col.rs`／
//! `kernels_constant_pad.rs` と同じ理由でソースを `nvcc` 事前
//! コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA ヘッダを
//! 一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! **本モジュールの位置づけ**: イシュー #1729 実装時点（2026-09-15）
//! では兄弟イシュー #1728（CPU 実装。設計 doc §9 が指す共有基盤
//! `fandhe_ai_tensor_core::backend_ops::BackendOps::max_pool2d`／
//! `avg_pool2d`／`adaptive_avg_pool2d`・`Pool2dParams`・出力 shape
//! 関数）が `main` に未マージだったため、本モジュール・`pooling.rs`
//! はクレート内に閉じた forward カーネル実装のみを提供していた
//! （`docs/pooling-ops-design.md` §15「実装記録（#1729）」に経緯を
//! 記録）。`ops.rs::CudaBackendOps` への override 配線は #1728
//! マージ後の追従イシューで完了済み（`pooling.rs` モジュール doc
//! 参照）。
//!
//! 1d は `[N, C, 1, L]` へ reshape して 2d カーネルへ併合する
//! （設計 doc §2）ため本モジュールに 1d 専用カーネルは無い。
//!
//! # 数値契約（`docs/pooling-ops-design.md` §5／§7・
//! `.claude/rules/coding-rust.md`「正規化統計・勾配の長軸縮約」節）
//!
//! - [`MAX_POOL2D_F32`]: 窓内を row-major（`kh` 外側・`kw` 内側）で
//!   走査し `v > best || (isnan(v) && !isnan(best))` の条件でのみ更新
//!   する先勝ち決定的タイ規則・NaN 伝播（走査順で最初に現れた NaN の
//!   索引を保持）。`best`／`best_idx` は `-INFINITY` 番兵ではなく最初の
//!   有効タップで初期化する（`found` フラグ。設計 doc §3 の空窓拒否
//!   検査が「すべての窓が少なくとも 1 つの有効タップを含む」ことを
//!   前提として成立させる）。算術を一切含まない純粋な選択演算のため
//!   `double` を使わず、ホスト参照実装（`pooling_model.rs`）と値・
//!   索引とも **bit 完全一致**する。索引は (n, c) 平面内の flat 添字
//!   `h * w_in + w`（padding 位置は勝者になり得ないため常に
//!   `[0, h_in*w_in)` の範囲に収まる）。
//! - [`AVG_POOL2D_F32`]／[`ADAPTIVE_AVG_POOL2D_F32`]: 窓内を row-major
//!   固定順で `double` へ逐次加算し、`double` で除算してから最後に
//!   1 回だけ `float` へ downcast する。ホスト参照実装の `f64` 逐次和・
//!   1 回 downcast と **bit 完全一致**する。`AvgPool` の `dilation` は
//!   設計 doc §3 のとおり常に `1` 固定のためカーネル引数に持たない。
//!
//! padding 位置（範囲外座標）は 3 カーネルともタップを完全に無視する
//! （`MaxPool` の勝者になり得ない・`Avg` 系の合計へ加算しない。設計 doc
//! §5／§7）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `if (idx < numel)` を維持する。添字演算はすべて `long long`
//! （`kernels_im2col.rs` と同じ理由。`i32::MAX` 近傍でのオーバー
//! フロー回避）。`in` への読み出しは常に `0 <= h < h_in && 0 <= w <
//! w_in` を手動検査してから行う（padding 領域を範囲外読み出しせず
//! スキップする）。境界チェックを維持したまま性能最適化を適用する
//! 規約（`.claude/rules/coding-rust.md`）どおり、本カーネルは境界
//! チェック省略による高速化を一切行わない。

/// 1 スレッドブロックあたりのスレッド数（`kernels_im2col::
/// IM2COL_BLOCK_DIM` と同じ値・同じ理由）。
pub const POOLING_BLOCK_DIM: u32 = 256;

/// MaxPool2d（`torch.nn.functional.max_pool2d(..., return_indices=True)`
/// 相当）。1 出力要素（`out`／`idx_out` の 1 flat 添字。`[N,C,Hout,Wout]`
/// の row-major）= 1 スレッド。
///
/// 引数はすべて呼び出し元（`pooling.rs::CudaPooling::
/// run_max_pool2d_f32`）が `i32` 範囲検査済みのスカラー: `n_batch, c,
/// h_in, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw, numel`
/// （`numel` は出力 `[N,C,Hout,Wout]` の総要素数＝起動グリッドの範囲）。
pub const MAX_POOL2D_F32: &str = r#"
extern "C" __global__ void max_pool2d_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int* __restrict__ idx_out,
    int n_batch, int c, int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long ow = rem % w_out;
        rem /= w_out;
        long long oh = rem % h_out;
        rem /= h_out;
        long long cc = rem % c;
        long long n = rem / c;

        float best = 0.0f;
        int best_idx = 0;
        int found = 0;

        for (int kh_ = 0; kh_ < kh; kh_++) {
            long long h = oh * (long long)sh + (long long)kh_ * dh - (long long)ph;
            if (h < 0 || h >= h_in) continue;
            for (int kw_ = 0; kw_ < kw; kw_++) {
                long long w = ow * (long long)sw + (long long)kw_ * dw - (long long)pw;
                if (w < 0 || w >= w_in) continue;
                long long in_idx = ((n * (long long)c + cc) * h_in + h) * w_in + w;
                float v = in[in_idx];
                int cur_idx = (int)(h * w_in + w);
                if (!found) {
                    best = v;
                    best_idx = cur_idx;
                    found = 1;
                } else if (v > best || (isnan(v) && !isnan(best))) {
                    best = v;
                    best_idx = cur_idx;
                }
            }
        }
        out[idx] = best;
        idx_out[idx] = best_idx;
    }
}
"#;

/// AvgPool2d（`torch.nn.functional.avg_pool2d` 相当。`dilation` は
/// 設計 doc §3 のとおり常に `1` 固定）。1 出力要素 = 1 スレッド。
///
/// 引数は [`MAX_POOL2D_F32`] から `dh`／`dw` を除き、`count_include_pad`
/// （`0`／`1`）を加えたもの。`count_include_pad=1` の divisor は
/// `kh*kw`（padding 込み）・`0` の divisor は有効タップ数。
pub const AVG_POOL2D_F32: &str = r#"
extern "C" __global__ void avg_pool2d_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_batch, int c, int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw, int ph, int pw,
    int count_include_pad, int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long ow = rem % w_out;
        rem /= w_out;
        long long oh = rem % h_out;
        rem /= h_out;
        long long cc = rem % c;
        long long n = rem / c;

        double acc = 0.0;
        long long valid_count = 0;
        for (int kh_ = 0; kh_ < kh; kh_++) {
            long long h = oh * (long long)sh + (long long)kh_ - (long long)ph;
            if (h < 0 || h >= h_in) continue;
            for (int kw_ = 0; kw_ < kw; kw_++) {
                long long w = ow * (long long)sw + (long long)kw_ - (long long)pw;
                if (w < 0 || w >= w_in) continue;
                long long in_idx = ((n * (long long)c + cc) * h_in + h) * w_in + w;
                acc += (double)in[in_idx];
                valid_count++;
            }
        }
        long long divisor = count_include_pad ? ((long long)kh * (long long)kw) : valid_count;
        out[idx] = (float)(acc / (double)divisor);
    }
}
"#;

/// AdaptiveAvgPool2d（`torch.nn.functional.adaptive_avg_pool2d` 相当）。
/// 1 出力要素 = 1 スレッド。窓は `start = floor(o*in/out)`・
/// `end = ceil((o+1)*in/out)`（設計 doc §4）で決まり、`padding` は
/// 存在しない（暗黙のゼロパディングも行わない）。
pub const ADAPTIVE_AVG_POOL2D_F32: &str = r#"
extern "C" __global__ void adaptive_avg_pool2d_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_batch, int c, int h_in, int w_in, int h_out, int w_out,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long ow = rem % w_out;
        rem /= w_out;
        long long oh = rem % h_out;
        rem /= h_out;
        long long cc = rem % c;
        long long n = rem / c;

        long long h_start = (oh * (long long)h_in) / h_out;
        long long h_end = ((oh + 1) * (long long)h_in + h_out - 1) / h_out;
        long long w_start = (ow * (long long)w_in) / w_out;
        long long w_end = ((ow + 1) * (long long)w_in + w_out - 1) / w_out;

        double acc = 0.0;
        for (long long h = h_start; h < h_end; h++) {
            for (long long w = w_start; w < w_end; w++) {
                long long in_idx = ((n * (long long)c + cc) * h_in + h) * w_in + w;
                acc += (double)in[in_idx];
            }
        }
        long long divisor = (h_end - h_start) * (w_end - w_start);
        out[idx] = (float)(acc / (double)divisor);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 の境界検査（`idx < numel`）を維持していることの機械検証
    /// （`kernels_im2col.rs` と同型の文字列テスト）。
    #[test]
    fn kernels_include_bounds_check() {
        assert!(MAX_POOL2D_F32.contains("if (idx < numel)"));
        assert!(AVG_POOL2D_F32.contains("if (idx < numel)"));
        assert!(ADAPTIVE_AVG_POOL2D_F32.contains("if (idx < numel)"));
    }

    /// 添字演算は `long long`（`i32::MAX` 近傍でのオーバーフロー回避）。
    #[test]
    fn kernels_use_long_long_for_flat_index_arithmetic() {
        assert!(MAX_POOL2D_F32.contains("long long idx"));
        assert!(AVG_POOL2D_F32.contains("long long idx"));
        assert!(ADAPTIVE_AVG_POOL2D_F32.contains("long long idx"));
    }

    /// [`MAX_POOL2D_F32`] は算術を含まない純粋な選択演算のため `double`
    /// を一切使わない（モジュール doc の数値契約・bit 完全一致）。
    #[test]
    fn max_pool_kernel_contains_no_double_arithmetic() {
        assert!(!MAX_POOL2D_F32.contains("double"));
    }

    /// [`MAX_POOL2D_F32`] のタイ規則（先勝ち・NaN 伝播）の更新条件が
    /// ソースへ実装されていること（設計 doc §5 の式と同一文字列）。
    #[test]
    fn max_pool_kernel_uses_first_match_nan_propagating_update_rule() {
        assert!(MAX_POOL2D_F32.contains("v > best || (isnan(v) && !isnan(best))"));
        // 番兵初期化ではなく最初の有効タップで初期化する（`found` フラグ）。
        assert!(MAX_POOL2D_F32.contains("if (!found) {"));
        assert!(!MAX_POOL2D_F32.contains("-INFINITY"));
    }

    /// [`AVG_POOL2D_F32`]／[`ADAPTIVE_AVG_POOL2D_F32`] は `double`
    /// アキュムレータへ逐次加算し最後に 1 回だけ `float` へ downcast
    /// する（モジュール doc の数値契約）。
    #[test]
    fn avg_kernels_use_double_accumulator_with_single_downcast() {
        assert!(AVG_POOL2D_F32.contains("double acc = 0.0;"));
        assert!(AVG_POOL2D_F32.contains("acc += (double)in[in_idx];"));
        assert_eq!(
            AVG_POOL2D_F32
                .matches("(float)(acc / (double)divisor)")
                .count(),
            1
        );

        assert!(ADAPTIVE_AVG_POOL2D_F32.contains("double acc = 0.0;"));
        assert!(ADAPTIVE_AVG_POOL2D_F32.contains("acc += (double)in[in_idx];"));
        assert_eq!(
            ADAPTIVE_AVG_POOL2D_F32
                .matches("(float)(acc / (double)divisor)")
                .count(),
            1
        );
    }

    /// `AvgPool2d` の `dilation` はカーネル引数に存在しない（設計 doc
    /// §3「Avg は `dilation=1` 固定」の直接的な機械検証）。
    #[test]
    fn avg_pool_kernel_has_no_dilation_argument() {
        assert!(!AVG_POOL2D_F32.contains("int dh"));
        assert!(!AVG_POOL2D_F32.contains("int dw"));
    }
}
