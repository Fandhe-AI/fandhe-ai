//! `argmax`／`argmin`（`torch.argmax`／`torch.argmin` 相当。イシュー
//! #1948・親イシュー #1947）の CUDA C カーネルソース（NVRTC 実行時
//! コンパイル用の静的文字列）。
//!
//! `kernels_reduce.rs`（`sum`／`max`／`min`）と同じ「ビルド時に
//! nvcc/CUDA ヘッダを一切要求しない」方針でソースを文字列のまま埋め
//! 込む（`.claude/rules/deps-policy.md`。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約）。
//!
//! # 走査契約（`fandhe_ai_tensor_core::BackendOps::argmax`／`argmin` doc
//! が正。`backend-cpu::reduction::axis_arg_reduce`／`arg_slice` と同一）
//!
//! 1. タイ（`==`。`+0.0` と `-0.0` も同値）は縮約軸上の**最初の**添字。
//!    argmax は `v > best` の厳密比較でのみ更新し、argmin は `v < best`。
//! 2. NaN は無視する。`best` が未確定なら最初の非 NaN 要素で確定する。
//!    全要素 NaN の場合は添字 0。
//! 3. 決定的な走査順序（軸縮約は `0..axis_len` 昇順・全軸縮約は
//!    チャンク昇順＋チャンク内昇順）であり run-to-run bit 同一。
//!
//! # 添字選択は結合順序に依存しない（並列分割の正当化）
//!
//! 「値がより良い・同値なら小さい添字を残す・NaN は無視する」という
//! 更新規則は浮動小数点演算（加算・乗算）を一切含まない純粋な順序選択
//! （`>`／`<` の厳密比較のみ）であり、区間を分割してから結合しても
//! 全体を逐次走査した場合と厳密に同じ添字を得る（本ファイル `tests`
//! モジュールおよび `arg_reduce.rs` のホストモデルテストがこの結合律を
//! 実データで検証する）。よって `sum`／正規化統計の `f64` アキュムレータ
//! 契約（`.claude/rules/coding-rust.md`）とは異なる系統の演算であり、
//! 本カーネルは丸め・結合順序に関する数値契約を一切必要としない。
//!
//! # 全軸縮約の 2 段構成（チャンク分割・`kernels_reduce.rs` の
//! sum/max/min とは異なる粒度）
//!
//! `reduce_arg{max,min}_all_partial_f32` は 1 スレッドが**連続チャンク**
//! `[t*chunk_len, min((t+1)*chunk_len, numel))` を昇順走査し、`pval[t]`・
//! `pidx[t]`（グローバル添字。非 NaN 要素が 1 つも無ければ `-1`）を書く。
//! `reduce_arg{max,min}_all_finalize_f32` は単一スレッド（`grid=1,
//! block=1`）が `pidx[c] < 0` のチャンクを skip しつつ `0..num_chunks`
//! 昇順・厳密比較のみで結合する。チャンクが連続・昇順かつ finalize が
//! 昇順・厳密比較のため、タイは必ず最小添字を含むチャンクが勝つ
//! （`arg_reduce.rs::arg_all_plan` がチャンク分割自体を決定する）。
//! `atomicAdd`／`atomicMax` は使わない（`kernels_reduce.rs` と同じ
//! 非決定性回避の理由）。
//!
//! # NaN 判定（bit パターン直接検査。イシュー #1893 と同じ理由）
//!
//! `isnan()`（NVRTC で `<math.h>` 暗黙 include が無いため未定義になり
//! うる）ではなく、`kernels_cast.rs` 等で実績のある `__float_as_uint`
//! による bit パターン検査 `(__float_as_uint(v) & 0x7fffffffu) >
//! 0x7f800000u`（指数部全 1・仮数部非ゼロ）を使う。単位元（`±inf` 相当）
//! も `INFINITY` マクロ（NVRTC 未定義。イシュー #1893）を使わず、
//! 「未確定」状態を明示フラグ（`has_best`／`pidx[t] = -1`）で表現する
//! ため単位元の bit パターン自体は不要。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルで手動境界チェック（`idx < total`／`t < num_chunks`／
//! `blockIdx.x == 0 && threadIdx.x == 0`）を維持する。

/// `argmax` 単一軸縮約（`arg_reduce.rs::CudaArgReduce::run_argmax_axis_f32`
/// が起動する。`outer`／`axis_len`／`inner` は呼び出し元
/// `reduce.rs::validate_axis_layout` が `i32::MAX` 収容を検証済み）。
/// 1 スレッド 1 出力要素（`idx = o * inner + i`）が縮約軸を
/// `0..axis_len` の昇順で厳密比較のみ（`v > best`）で走査する
/// （`backend-cpu::reduction::axis_arg_reduce` と同一の決定的順序）。
pub const REDUCE_ARGMAX_AXIS_F32: &str = r#"
extern "C" __global__ void reduce_argmax_axis_f32(
    const float* __restrict__ in,
    int* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            float v = in[src];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v > best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)a;
            }
        }
        out[idx] = best_idx;
    }
}
"#;

/// `argmin` 単一軸縮約。[`REDUCE_ARGMAX_AXIS_F32`] の逐語ミラー（`v >
/// best` を `v < best` へ反転する以外は完全同一）。
pub const REDUCE_ARGMIN_AXIS_F32: &str = r#"
extern "C" __global__ void reduce_argmin_axis_f32(
    const float* __restrict__ in,
    int* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            float v = in[src];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v < best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)a;
            }
        }
        out[idx] = best_idx;
    }
}
"#;

/// `argmax` 全軸縮約 1 段目: スレッド `t` が連続チャンク `[t*chunk_len,
/// min((t+1)*chunk_len, numel))` を昇順走査し `pval[t]`／`pidx[t]`
/// （非 NaN 要素が無ければ `pidx[t] = -1`）を書く（本ファイル冒頭
/// コメント「全軸縮約の 2 段構成」参照）。`chunk_len`／`num_chunks` は
/// `arg_reduce.rs::arg_all_plan` が起動前に決定する。
pub const REDUCE_ARGMAX_ALL_PARTIAL_F32: &str = r#"
extern "C" __global__ void reduce_argmax_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ pval,
    int* __restrict__ pidx,
    int numel,
    int chunk_len,
    int num_chunks)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_chunks) {
        long long start = (long long)t * (long long)chunk_len;
        long long end = start + (long long)chunk_len;
        if (end > numel) {
            end = numel;
        }
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = -1;
        for (long long i = start; i < end; i++) {
            float v = in[i];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v > best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)i;
            }
        }
        pval[t] = best_val;
        pidx[t] = best_idx;
    }
}
"#;

/// `argmin` 全軸縮約 1 段目。[`REDUCE_ARGMAX_ALL_PARTIAL_F32`] の逐語
/// ミラー（`v > best` を `v < best` へ反転する以外は完全同一）。
pub const REDUCE_ARGMIN_ALL_PARTIAL_F32: &str = r#"
extern "C" __global__ void reduce_argmin_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ pval,
    int* __restrict__ pidx,
    int numel,
    int chunk_len,
    int num_chunks)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_chunks) {
        long long start = (long long)t * (long long)chunk_len;
        long long end = start + (long long)chunk_len;
        if (end > numel) {
            end = numel;
        }
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = -1;
        for (long long i = start; i < end; i++) {
            float v = in[i];
            if ((__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u) {
                continue;
            }
            if (!has_best || v < best_val) {
                has_best = 1;
                best_val = v;
                best_idx = (int)i;
            }
        }
        pval[t] = best_val;
        pidx[t] = best_idx;
    }
}
"#;

/// `argmax` 全軸縮約 2 段目: 単一スレッド（`grid=1, block=1`）が
/// `pidx[c] < 0`（対応チャンクに非 NaN 要素が無かった）のチャンクを
/// skip しつつ `0..num_chunks` 昇順・厳密比較のみで結合する
/// （チャンク自体が連続・昇順のためタイは必ず最小添字が勝つ。本ファイル
/// 冒頭コメント「全軸縮約の 2 段構成」参照）。全チャンクが非 NaN 要素
/// を持たない（全要素 NaN）場合は添字 0 を書く。
pub const REDUCE_ARGMAX_ALL_FINALIZE_F32: &str = r#"
extern "C" __global__ void reduce_argmax_all_finalize_f32(
    const float* __restrict__ pval,
    const int* __restrict__ pidx,
    int* __restrict__ out,
    int num_chunks)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (int c = 0; c < num_chunks; c++) {
            int idx = pidx[c];
            if (idx < 0) {
                continue;
            }
            float v = pval[c];
            if (!has_best || v > best_val) {
                has_best = 1;
                best_val = v;
                best_idx = idx;
            }
        }
        out[0] = best_idx;
    }
}
"#;

/// `argmin` 全軸縮約 2 段目。[`REDUCE_ARGMAX_ALL_FINALIZE_F32`] の逐語
/// ミラー（`v > best` を `v < best` へ反転する以外は完全同一）。
pub const REDUCE_ARGMIN_ALL_FINALIZE_F32: &str = r#"
extern "C" __global__ void reduce_argmin_all_finalize_f32(
    const float* __restrict__ pval,
    const int* __restrict__ pidx,
    int* __restrict__ out,
    int num_chunks)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int has_best = 0;
        float best_val = 0.0f;
        int best_idx = 0;
        for (int c = 0; c < num_chunks; c++) {
            int idx = pidx[c];
            if (idx < 0) {
                continue;
            }
            float v = pval[c];
            if (!has_best || v < best_val) {
                has_best = 1;
                best_val = v;
                best_idx = idx;
            }
        }
        out[0] = best_idx;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// 単一軸縮約 2 定数が境界検査・`long long` 添字・NaN bit パターン
    /// 判定を含み、argmax/argmin で厳密比較演算子（`>`／`<`）を取り違え
    /// ていないことを固定する。
    #[test]
    fn axis_kernels_have_bound_check_nan_guard_and_correct_comparator() {
        for src in [REDUCE_ARGMAX_AXIS_F32, REDUCE_ARGMIN_AXIS_F32] {
            assert!(src.contains("idx < total"));
            assert!(src.contains("long long total = (long long)outer * (long long)inner;"));
            assert!(src.contains("(__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
        assert!(REDUCE_ARGMAX_AXIS_F32.contains("v > best_val"));
        assert!(!REDUCE_ARGMAX_AXIS_F32.contains("v < best_val"));
        assert!(REDUCE_ARGMIN_AXIS_F32.contains("v < best_val"));
        assert!(!REDUCE_ARGMIN_AXIS_F32.contains("v > best_val"));
    }

    /// 全軸縮約 1 段目（partial）2 定数が境界検査・チャンク境界の
    /// クランプ・`pidx = -1` の未確定表現を含むことを固定する。
    #[test]
    fn all_partial_kernels_have_bound_check_and_unconfirmed_sentinel() {
        for src in [REDUCE_ARGMAX_ALL_PARTIAL_F32, REDUCE_ARGMIN_ALL_PARTIAL_F32] {
            assert!(src.contains("t < num_chunks"));
            assert!(src.contains("if (end > numel)"));
            assert!(src.contains("int best_idx = -1;"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
        assert!(REDUCE_ARGMAX_ALL_PARTIAL_F32.contains("v > best_val"));
        assert!(REDUCE_ARGMIN_ALL_PARTIAL_F32.contains("v < best_val"));
    }

    /// 全軸縮約 2 段目（finalize）2 定数が単一スレッドガード・`pidx < 0`
    /// skip・全チャンク未確定時の添字 0 既定を含むことを固定する。
    #[test]
    fn all_finalize_kernels_have_single_thread_guard_and_skip_sentinel() {
        for src in [
            REDUCE_ARGMAX_ALL_FINALIZE_F32,
            REDUCE_ARGMIN_ALL_FINALIZE_F32,
        ] {
            assert!(src.contains("blockIdx.x == 0 && threadIdx.x == 0"));
            assert!(src.contains("if (idx < 0)"));
            assert!(src.contains("int best_idx = 0;"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
        assert!(REDUCE_ARGMAX_ALL_FINALIZE_F32.contains("v > best_val"));
        assert!(REDUCE_ARGMIN_ALL_FINALIZE_F32.contains("v < best_val"));
    }

    /// argmax／argmin の 6 定数すべてが `isnan`／`INFINITY`／`#include`／
    /// `__FLT_MAX__`（NVRTC で未定義・イシュー #1893／#1101）・`double`
    /// （本カーネルは丸めを伴わない添字選択のため `f64` アキュムレータ
    /// 契約の対象外）を参照しないことを fail-closed に固定する。
    #[test]
    fn arg_reduce_kernels_avoid_nvrtc_undefined_symbols_and_double() {
        for src in [
            REDUCE_ARGMAX_AXIS_F32,
            REDUCE_ARGMIN_AXIS_F32,
            REDUCE_ARGMAX_ALL_PARTIAL_F32,
            REDUCE_ARGMIN_ALL_PARTIAL_F32,
            REDUCE_ARGMAX_ALL_FINALIZE_F32,
            REDUCE_ARGMIN_ALL_FINALIZE_F32,
        ] {
            assert!(!src.contains("isnan"), "isnan は NVRTC で未定義: {src}");
            assert!(
                !src.contains("INFINITY"),
                "INFINITY は NVRTC で未定義: {src}"
            );
            assert!(!src.contains("#include"), "ヘッダ依存を追加しない: {src}");
            assert!(
                !src.contains("__FLT_MAX__"),
                "__FLT_MAX__ は compute_121 で未定義（#1101）: {src}"
            );
            assert!(!src.contains("double"), "添字選択に double は不要: {src}");
        }
    }
}
