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
//! の 2 回縮約より単純）。
//!
//! 意味論: `dx = g − exp(y)·Σ_dim(g)`（`y` = forward 記録値
//! `log_softmax(x, dim)`・`g` = upstream 勾配。`grad::log_softmax_vjp_along`
//! と同じ式）。
//!
//! # 縮約精度契約（`.claude/rules/coding-rust.md`）
//!
//! `Σ_dim(g)` は要素積を伴わない単純和のため、要素を**直接** `double`
//! へ昇格してから加算する（二乗和方式の正規化統計とは異なる区分。
//! `kernels_layer_norm.rs` の平均縮約〈パス 1〉と同じ扱い）。
//!
//! **縮約順序はホスト参照実装（`grad::log_softmax_vjp_along` の `dim`
//! 添字昇順逐次和）と一致させる**（PR #1994 codex-review 指摘。当初
//! 採用していた `__shfl_xor_sync` による warp butterfly reduction は
//! 加算順序がホストと異なるため、大きさの近い符号違いの値が相殺する
//! 入力（例: `logits=[0,0,0,0]`・`g=[1e20,-1e20,1,0]`）で桁落ちの位置
//! がホストとずれ、`dx` の一部要素が REQ-2 統一複合判定〈相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満〉を外れるケースが実在すると判明
//! したため撤回した）。行あたり 1 レーン（`lane == 0`）のみが `i = 0,
//! 1, …, cols-1` の昇順で逐次和を計算し（ホストと完全同順）、
//! `__shared__` 変数へ書いてから `__syncthreads()` で残り 31 レーンへ
//! 公開する。warp 内並列縮約は行わないため縮約自体の速度は host と
//! 同じ O(cols) だが、行間（grid 次元 `rows`）・列方向の後続
//! elementwise 書き出しは従来どおり 32 レーン並列のまま。
//!
//! **bit 完全一致は主張しない**（`expf` の丸めが `f32::exp`〈ホスト側〉
//! と厳密に一致する保証はない）。REQ-2 統一複合判定（相対誤差 1e-3
//! 未満 または 絶対誤差 1e-5 未満）で検証する（`crates/tensor-core/
//! src/backend_ops.rs::BackendOps::log_softmax_backward` doc 参照）。
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
    // coding-rust.md`「勾配の長軸縮約」）。ホスト参照実装と完全同順
    // （i 昇順の逐次和）にするため lane 0 のみが計算する（冒頭コメント
    // 「縮約精度契約」参照。PR #1994 codex-review 指摘）。
    __shared__ double row_sum;
    if (lane == 0) {
        double sum = 0.0;
        for (long long i = 0; i < cols; ++i) {
            sum += (double)g_row[i];
        }
        row_sum = sum;
    }
    __syncthreads();
    double sum = row_sum;

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

    /// PR #1994 codex-review 指摘の逆戻り防止テスト: `Σ_dim(g)` の
    /// warp butterfly reduction（`__shfl_xor_sync`）を再導入しないこと
    /// を固定する（`__shfl_sync` broadcast 系〈`row_sum` の
    /// `__syncthreads()` 経由公開〉のみ許容）。butterfly reduction は
    /// 加算順序がホスト参照実装〈`grad::log_softmax_vjp_along` の添字
    /// 昇順逐次和〉と異なり、大きさの近い符号違いの値が相殺する入力で
    /// 桁落ちの位置がずれる（冒頭コメント「縮約精度契約」参照）。
    #[test]
    fn kernel_source_does_not_use_butterfly_reduction_for_sum() {
        assert!(
            !LOG_SOFTMAX_BACKWARD_F32.contains("__shfl_xor_sync"),
            "Σ_dim(g) の warp butterfly reduction（加算順序がホストと \
             異なり相殺入力で桁落ち位置がずれる）を再導入しないこと: \
             {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            LOG_SOFTMAX_BACKWARD_F32.contains("for (long long i = 0; i < cols; ++i) {"),
            "Σ_dim(g) はホストと同じ i 昇順の逐次和であること: \
             {LOG_SOFTMAX_BACKWARD_F32}"
        );
        assert!(
            LOG_SOFTMAX_BACKWARD_F32.contains("if (lane == 0) {"),
            "Σ_dim(g) の逐次和は lane 0 のみが計算すること: \
             {LOG_SOFTMAX_BACKWARD_F32}"
        );
    }

    /// カーネルの `Σ_dim(g)` 計算（`if (lane == 0) { double sum = 0.0;
    /// for (i = 0; i < cols; ++i) sum += (double)g[i]; }`）を Rust で
    /// 逐語再現したホストモデル。GPU 非依存の単体テストで数値契約
    /// （ホストと完全同順の逐次和）を検証するために使う
    /// （`crate::pooling_model` 等と同型の「ホスト逐語モデル」方針）。
    fn host_model_row_sum(g_row: &[f32]) -> f64 {
        let mut sum = 0.0f64;
        for &v in g_row {
            sum += v as f64;
        }
        sum
    }

    /// カーネル全体（`Σ_dim(g)` の逐次和 → `dx = g − exp(y)·Σ_dim(g)`
    /// の要素ごと計算）を Rust で逐語再現したホストモデル。
    fn host_model_dx(y_row: &[f32], g_row: &[f32]) -> Vec<f32> {
        let sum = host_model_row_sum(g_row);
        y_row
            .iter()
            .zip(g_row.iter())
            .map(|(&y, &g)| {
                let exp_y = (y.exp()) as f64;
                ((g as f64) - exp_y * sum) as f32
            })
            .collect()
    }

    /// PR #1994 codex-review 指摘の具体例（`logits=[0,0,0,0]`・
    /// `g=[1e20,-1e20,1,0]`）で、ホストと完全同順の逐次和（butterfly
    /// reduction 是正後の本カーネル設計）が桁落ちなく `Σ_dim(g)=1`
    /// を再現し、指摘が挙げた `dx[2]≈0.75`〈CUDA 側は是正前 butterfly
    /// reduction では 1 になっていた〉と一致することを確認する。
    /// `logits=[0,0,0,0]` の `log_softmax` は全要素 `ln(0.25)` で
    /// `y.exp()=0.25` （`Var::log_softmax` の forward と同じ式。
    /// 本テストは `Op::LogSoftmax` の forward を経由せず `y` を
    /// 直接与える単体テストのため `autodiff` クレートへの依存は
    /// 発生しない）。
    #[test]
    fn host_model_matches_sequential_sum_for_cancelling_upstream_grad() {
        let y_row = [0.25f32.ln(); 4];
        let g_row = [1e20f32, -1e20f32, 1.0f32, 0.0f32];

        let sum = host_model_row_sum(&g_row);
        assert_eq!(sum, 1.0, "ホスト同順の逐次和は相殺後に厳密 1.0 となるべき");

        let dx = host_model_dx(&y_row, &g_row);
        // dx[2] = g[2] - exp(y[2])*sum = 1.0 - 0.25*1.0 = 0.75
        assert!(
            (dx[2] - 0.75).abs() < 1e-6,
            "dx[2] は 0.75 に一致すべき（butterfly reduction 是正前は \
             1.0 になっていた）: dx={dx:?}"
        );
        // dx[3] = 0.0 - 0.25*1.0 = -0.25
        assert!(
            (dx[3] - (-0.25)).abs() < 1e-6,
            "dx[3] は -0.25 に一致すべき: dx={dx:?}"
        );
    }
}
