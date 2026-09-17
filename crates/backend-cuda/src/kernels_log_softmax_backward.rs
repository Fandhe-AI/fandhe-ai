//! `log_softmax` backward（VJP）カーネル（NVRTC 実行時コンパイル用の
//! 静的文字列。イシュー #1949・親 #1947「GPU ホストフォールバック残存
//! 演算の専用カーネル化」）。
//!
//! `kernels_layer_norm.rs`（#1596）と同じ理由でソースを `nvcc` 事前
//! コンパイルせず文字列のまま埋め込む（CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する契約を維持する。`.claude/rules/
//! deps-policy.md`）。
//!
//! # 設計
//!
//! `kernels_layer_norm.rs` と同型: **1 CTA = 1 warp（32 レーン）が行を
//! 1 行担当し、grid 次元 = `rows`**（persistent block ではない単純な
//! 1 対 1 マッピング）。行方向の縮約は `Σ_dim(g)` の 1 回のみ（LayerNorm
//! の 2 回縮約より単純）で、`__shared__` メモリは使わない。
//!
//! 意味論: `dx = g − exp(y)·Σ_dim(g)`（`y` = forward 記録値
//! `log_softmax(x, dim)`・`g` = upstream 勾配。`grad::log_softmax_vjp_along`
//! と同じ式）。
//!
//! # 縮約精度契約（`.claude/rules/coding-rust.md`）
//!
//! `Σ_dim(g)` は要素積を伴わない単純和のため、要素を**直接** `double`
//! へ昇格してから加算する（二乗和方式の正規化統計とは異なる区分。
//! `kernels_layer_norm.rs` の平均縮約〈パス 1〉と同じ扱い）。縮約は
//! `__shfl_xor_sync` による warp butterfly reduction（`double` を直接
//! シャッフルできるため Metal の soft-f64 代替は不要）。
//!
//! **bit 完全一致は主張しない**（butterfly 縮約はホスト参照実装の
//! `dim` 添字昇順逐次和と結合順序が異なるため）。REQ-2 統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で検証する
//! （`crates/tensor-core/src/backend_ops.rs::BackendOps::
//! log_softmax_backward` doc 参照）。
//!
//! # REQ-8 境界検査
//!
//! `if (row >= rows) return;`（grid-stride しない単純な 1 対 1 マッピング
//! のため CTA 単位のガードのみ）・`for (i = lane; i < cols; i += 32)`
//! （warp 内ループの手動ガード）。ループ添字は `long long`（`row_base`）
//! で `rows * cols` の乗算オーバーフローを避ける（`kernels_layer_norm.rs`
//! と同じ対策）。`INFINITY`／`__FLT_MAX__`／`#include` は使わない
//! （イシュー #1893／#1101 の NVRTC 未定義マクロ教訓を踏襲。本カーネルは
//! そもそも境界マスク定数を必要としない）。

/// `log_softmax` backward カーネル（単一カーネル。冒頭コメント参照）。
///
/// 引数: `y`（forward 記録値。`[rows, cols]` 行優先）・`g`（upstream
/// 勾配。同一 shape）・`dx`（出力。同一 shape）・`rows`・`cols`。
pub const LOG_SOFTMAX_BACKWARD_F32: &str = r#"
extern "C" __global__ void log_softmax_backward_f32(
    const float* __restrict__ y,
    const float* __restrict__ g,
    float* __restrict__ dx,
    int rows,
    int cols)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)cols;
    const float* y_row = y + row_base;
    const float* g_row = g + row_base;
    float* dx_row = dx + row_base;

    // Σ_dim(g) を double アキュムレータで蓄積する（要素積を伴わない
    // 単純和のため要素を直接 double へ昇格。`.claude/rules/
    // coding-rust.md`「勾配の長軸縮約」）。
    double sum = 0.0;
    for (long long i = lane; i < cols; i += 32) {
        sum += (double)g_row[i];
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }

    // dx = g − exp(y)·Σ_dim(g)。`exp(y)` は f32 で確定してから double へ
    // 昇格し（ホスト参照実装 `grad::log_softmax_vjp_along` の
    // `y[idx].exp()` と同じ丸め位置）、乗算・減算は double のまま行い、
    // 最終書き出し直前の 1 回だけ float へ downcast する（有限 f32 入力
    // での中間 overflow を避ける契約。`kernels_layer_norm.rs` の affine
    // 書き出しと同じ「最後の 1 回だけ丸める」方針）。
    for (long long i = lane; i < cols; i += 32) {
        double exp_y = (double)expf(y_row[i]);
        dx_row[i] = (float)((double)g_row[i] - exp_y * sum);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// カーネルソースの静的検査（`kernels_reduce.rs` の
    /// `reduce_kernels_do_not_reference_nvrtc_undefined_infinity_macro`
    /// と同型。イシュー #1893／#1101 の NVRTC 未定義マクロ教訓を
    /// fail-closed に固定する逆戻り防止テスト）: `INFINITY`／
    /// `__FLT_MAX__`／`#include` を参照しないこと。
    #[test]
    fn kernel_source_does_not_reference_nvrtc_undefined_macros() {
        assert!(
            !LOG_SOFTMAX_BACKWARD_F32.contains("INFINITY"),
            "NVRTC 未定義の INFINITY マクロが残存: {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            !LOG_SOFTMAX_BACKWARD_F32.contains("__FLT_MAX__"),
            "__FLT_MAX__ は compute_121 で未定義（#1101）: {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            !LOG_SOFTMAX_BACKWARD_F32.contains("#include"),
            "ヘッダ依存を追加しない: {LOG_SOFTMAX_BACKWARD_F32}"
        );
    }

    /// エントリ名・REQ-8 境界検査（`if (row >= rows) return;`）・
    /// 縮約アキュムレータ型（`double`）・オーバーフロー対策
    /// （`long long`）が実装どおりに存在することを固定する。
    #[test]
    fn kernel_source_has_entry_point_and_boundary_guard() {
        assert!(
            LOG_SOFTMAX_BACKWARD_F32.contains("void log_softmax_backward_f32("),
            "エントリ名が一致しない: {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            LOG_SOFTMAX_BACKWARD_F32.contains("if (row >= rows) {"),
            "REQ-8 境界検査（CTA ガード）が欠落: {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            LOG_SOFTMAX_BACKWARD_F32.contains("double sum = 0.0;"),
            "縮約アキュムレータが double でない: {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            LOG_SOFTMAX_BACKWARD_F32.contains("long long row_base"),
            "row_base の乗算オーバーフロー対策（long long）が欠落: {LOG_SOFTMAX_BACKWARD_F32}"
        );
    }
}
