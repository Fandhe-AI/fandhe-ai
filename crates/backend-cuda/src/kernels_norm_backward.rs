//! RMSNorm／LayerNorm backward の CUDA カーネル（NVRTC 実行時コンパイル用
//! の静的文字列。イシュー #1950）。
//!
//! `kernels_rmsnorm.rs`／`kernels_layer_norm.rs`（#592・#1596）と同じ理由で
//! ソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む（CUDA toolkit
//! 非搭載環境でも `cargo build --workspace` が成立する契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # forward を一切変更しない設計（recompute-in-backward）
//!
//! `Op::RmsNorm`／`Op::LayerNorm` の forward カーネル（`kernels_rmsnorm.rs`・
//! `kernels_layer_norm.rs`）・tape 記録は本イシューで一切変更しない
//! （`rstd`／`mean` の保存を forward へ追加しない）。本ファイルの backward
//! カーネルは `x`（と `weight`）から行内統計をカーネル内で再計算する
//! （`fandhe_ai_autodiff::grad::rmsnorm_vjp_rows`／`layer_norm_vjp_rows`
//! と同じ「`input` を実体化し直して統計を再計算する」方針の CUDA 版）。
//!
//! # 既存 #596 `RMSNORM_BWD_DX_F32`／`RMSNORM_BWD_DW_F32` との関係
//!
//! `rmsnorm.rs::CudaRmsNorm::run_rmsnorm_bwd_f32`（保存済み `rstd` を
//! 受け取り学習ループの resident 経路から呼ばれる既存 API）は無変更のまま
//! 残す。本ファイルのカーネルはそれとは独立した新設エントリで、
//! `BackendOps::rmsnorm_backward`／`layer_norm_backward`
//! （`fandhe_ai_autodiff::grad::vjp` からの一般的な backward 呼び出し）
//! 専用に、ホスト参照実装（`grad::rmsnorm_vjp_rows`／
//! `layer_norm_vjp_rows`）と同じ縮約精度契約（dw／db の長軸縮約は
//! `.claude/rules/coding-rust.md`「要素積を f32 で確定 → f64 昇格蓄積 → 1
//! 回だけ f32 downcast」）を満たすよう新規に設計する。
//!
//! # カーネル構成（`norm_backward.rs` から起動される 4 カーネル）
//!
//! - `rmsnorm_bwd_dx_new_f32`・`layer_norm_bwd_dx_f32`:
//!   1 CTA = 1 warp（32 レーン）が 1 行を担当（`grid_dim = rows`。
//!   `kernels_layer_norm.rs::LAYER_NORM_F32` と同じ単純な 1 対 1
//!   マッピング・SMEM 不使用）。行内統計（RMSNorm は二乗和・LayerNorm は
//!   平均＋分散）を `double` アキュムレータ・`__shfl_xor_sync` butterfly
//!   （offset 16→1）で計算し `dx` を書き出すと同時に、行ごとの統計スカラー
//!   （RMSNorm は `rstd`・LayerNorm は `mean`／`rstd`）を `double` スクラッチ
//!   バッファ（`rows` 要素）へ lane 0 が書き出す（dw／db カーネルが同じ
//!   統計を再計算せず再利用するため）。
//!
//!   行内の縮約順序（butterfly。offset 16→1）は、ホスト参照実装
//!   （`rmsnorm_vjp_rows` の `dot_acc`・`layer_norm_vjp_rows` の
//!   `sum_dxhat`／`dot_acc`）側が `eval::warp_reduce_f64`（レーン
//!   ストライド `idx = lane; idx += 32` で分担してから同じ offset
//!   16→1 の butterfly で合流する、GPU 側とビット単位で同一の縮約
//!   関数）を経由するよう揃えてある（イシュー #1950・PR #1995
//!   codex-review P1 是正）。相殺を含む符号付き入力（例:
//!   `dy = [1e20, 1, -1e20, 0, ...]`）では縮約順序の違いだけで `dx` が
//!   O(1) 規模で乖離しうるため、単純逐次和のままでは REQ-2 統一複合
//!   判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる
//!   ケースがあった（縮約自体が異なると判定の余地がないため、`dx` は
//!   引き続き bit 一致は主張しない。`.claude/rules/coding-rust.md`
//!   「結合順序が単一の連続 K ループと異なるカーネルの parity 判定方式」
//!   の一般原則とは別に、縮約順序そのものをホスト側で GPU に揃える
//!   ことで REQ-2 判定を成立させる対処である）。二乗和（RMSNorm の
//!   `acc`）は符号なし項のみで相殺が生じないため対象外のまま（`eval::
//!   row_rms_stats` は単純逐次和を維持）。
//!
//! - `rmsnorm_bwd_dw_new_f32`・`layer_norm_bwd_dwdb_f32`:
//!   列（`hidden`）方向 grid-stride（1 スレッド = 1 列）で `rows` を
//!   `r = 0..rows` の順に逐次走査し、`double` アキュムレータへ
//!   `acc = (double)term + acc`（`term` は `f32` で確定済みの積）の順で
//!   蓄積する。これはホスト参照実装の `for r in 0..rows { dw_acc[i] +=
//!   term as f64; }` と同じ走査順序（列 `i` を固定して行 `r` を順に処理）
//!   であり、IEEE 754 の単一加算が可換であることから、行ごとの `term`
//!   自体が host と数値的に一致する範囲では `dw`／`db` の縮約順序自体は
//!   host と揃う（`term` の値そのものは dx カーネルの butterfly 経由の
//!   統計を使うため host と厳密には一致しないが、縮約契約
//!   〈`.claude/rules/coding-rust.md`〉が要求する「要素積を f32 で確定
//!   してから f64 へ昇格して蓄積する」という走査順序・精度契約は満たす）。
//!
//! # REQ-8 境界検査
//!
//! dx カーネルは `if (row >= rows) return;`・`i < hidden` の
//! grid-stride ループ手動ガード。dw／db カーネルは `if (col >= hidden)
//! return;`。ループ添字はいずれも `long long`（`row_base`／`idx`）で
//! `rows * hidden` の乗算オーバーフローを避ける（`kernels_rmsnorm.rs`・
//! `kernels_layer_norm.rs` と同じ対策）。
//!
//! `rsqrtf`／`INFINITY`／`#include` は使わない（イシュー #1105／#1893 の
//! 教訓。`sqrt`〈`double` 版〉と `fma` は CUDA 組み込みの暗黙宣言のため
//! `#include` 不要）。

/// RMSNorm backward の dx カーネル（冒頭コメント参照）。
///
/// 引数: `x`（`[rows, hidden]` 行優先）・`w`（`has_weight == 0` なら
/// 未参照。呼び出し元は必ず `hidden` 要素のダミーバッファを渡す。
/// `rmsnorm.rs::run_rmsnorm_f32_inner` と同じ predicated-load 対策）・
/// `dy`・`dx`（出力）・`rstd_out`（出力。`rows` 要素の `double` スクラッチ。
/// `f32` 精度で確定した `rstd` を widen して書く——ホスト `rmsnorm_vjp_rows`
/// の `rstd` は `row_rms_stats` が返す `f32` そのものであり、`f64` へ
/// widen しても情報は失われない）・`rows`・`hidden`・`eps`・`has_weight`。
pub const RMSNORM_BWD_DX_NEW_F32: &str = r#"
extern "C" __global__ void rmsnorm_bwd_dx_new_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    double* __restrict__ rstd_out,
    int rows,
    int hidden,
    float eps,
    int has_weight)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    const float* dy_row = dy + row_base;
    float* dx_row = dx + row_base;

    double acc = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        double v = (double)x_row[i];
        acc = fma(v, v, acc);
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    double inv_n = 1.0 / (double)hidden;
    float rstd = (float)(1.0 / sqrt(acc * inv_n + (double)eps));
    if (lane == 0) {
        rstd_out[row] = (double)rstd;
    }

    double dot = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        float xhat = x_row[i] * rstd;
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        float term = dxhat * xhat;
        dot += (double)term;
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xffffffffu, dot, offset);
    }
    double mean_dot = dot * inv_n;

    for (long long i = lane; i < hidden; i += 32) {
        float xhat = x_row[i] * rstd;
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        double d = (double)rstd * ((double)dxhat - (double)xhat * mean_dot);
        dx_row[i] = (float)d;
    }
}
"#;

/// RMSNorm backward の dw カーネル（冒頭コメント参照。`weight.is_some()`
/// のときのみ起動元が呼ぶ）。
///
/// 引数: `x`・`dy`・`rstd_in`（dx カーネルが書いた `rows` 要素の `double`
/// スクラッチ）・`dw`（出力。`hidden` 要素）・`rows`・`hidden`。
pub const RMSNORM_BWD_DW_NEW_F32: &str = r#"
extern "C" __global__ void rmsnorm_bwd_dw_new_f32(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const double* __restrict__ rstd_in,
    float* __restrict__ dw,
    int rows,
    int hidden)
{
    long long col = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (col >= hidden) {
        return;
    }
    double acc_w = 0.0;
    for (long long r = 0; r < rows; r++) {
        long long idx = r * (long long)hidden + col;
        float rstd = (float)rstd_in[r];
        float xhat = x[idx] * rstd;
        float dyv = dy[idx];
        float term = dyv * xhat;
        acc_w = (double)term + acc_w;
    }
    dw[col] = (float)acc_w;
}
"#;

/// LayerNorm backward の dx カーネル（冒頭コメント参照）。統計計算
/// （`mean`／分散）は `kernels_layer_norm.rs::LAYER_NORM_F32` と同じ
/// 縮約順序（lane-stride sum → butterfly → 直接除算 `sum/hidden`。
/// 二パス分散）を再現する。
///
/// 引数: `x`・`w`（`has_weight == 0` なら未参照。ダミーは `hidden` 要素）・
/// `dy`・`dx`（出力）・`mean_out`／`rstd_out`（出力。`rows` 要素の
/// `double` スクラッチ）・`rows`・`hidden`・`eps`・`has_weight`。
pub const LAYER_NORM_BWD_DX_F32: &str = r#"
extern "C" __global__ void layer_norm_bwd_dx_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    double* __restrict__ mean_out,
    double* __restrict__ rstd_out,
    int rows,
    int hidden,
    float eps,
    int has_weight)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    const float* dy_row = dy + row_base;
    float* dx_row = dx + row_base;

    double sum = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        sum += (double)x_row[i];
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    double mean = sum / (double)hidden;

    double sq_acc = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        double d = (double)x_row[i] - mean;
        sq_acc = fma(d, d, sq_acc);
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq_acc += __shfl_xor_sync(0xffffffffu, sq_acc, offset);
    }
    double var = sq_acc / (double)hidden;
    double rstd = 1.0 / sqrt(var + (double)eps);
    if (lane == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    double sum_dxhat = 0.0;
    double dot = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        float xhat = (float)(((double)x_row[i] - mean) * rstd);
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        sum_dxhat += (double)dxhat;
        float term = dxhat * xhat;
        dot += (double)term;
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_dxhat += __shfl_xor_sync(0xffffffffu, sum_dxhat, offset);
        dot += __shfl_xor_sync(0xffffffffu, dot, offset);
    }
    double mean_dxhat = sum_dxhat / (double)hidden;
    double mean_dot = dot / (double)hidden;

    for (long long i = lane; i < hidden; i += 32) {
        float xhat = (float)(((double)x_row[i] - mean) * rstd);
        float dyv = dy_row[i];
        float dxhat = (has_weight != 0) ? dyv * w[i] : dyv;
        double d = rstd * ((double)dxhat - mean_dxhat - (double)xhat * mean_dot);
        dx_row[i] = (float)d;
    }
}
"#;

/// LayerNorm backward の dw／db カーネル（冒頭コメント参照。1 カーネルで
/// 両方を計算する——列走査が共通のため。`has_weight`／`has_bias` に応じて
/// 該当する出力への書き込みのみ行う）。
///
/// 引数: `x`・`dy`・`mean_in`／`rstd_in`（dx カーネルが書いた `rows` 要素の
/// `double` スクラッチ）・`dw`／`db`（出力。`has_weight`／`has_bias` が
/// 偽のときは未参照——ダミーは `hidden` 要素）・`rows`・`hidden`・
/// `has_weight`・`has_bias`。
pub const LAYER_NORM_BWD_DWDB_F32: &str = r#"
extern "C" __global__ void layer_norm_bwd_dwdb_f32(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const double* __restrict__ mean_in,
    const double* __restrict__ rstd_in,
    float* __restrict__ dw,
    float* __restrict__ db,
    int rows,
    int hidden,
    int has_weight,
    int has_bias)
{
    long long col = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (col >= hidden) {
        return;
    }
    double acc_w = 0.0;
    double acc_b = 0.0;
    for (long long r = 0; r < rows; r++) {
        long long idx = r * (long long)hidden + col;
        double mean = mean_in[r];
        double rstd = rstd_in[r];
        float dyv = dy[idx];
        if (has_weight != 0) {
            float xhat = (float)(((double)x[idx] - mean) * rstd);
            float term = dyv * xhat;
            acc_w = (double)term + acc_w;
        }
        if (has_bias != 0) {
            acc_b = (double)dyv + acc_b;
        }
    }
    if (has_weight != 0) {
        dw[col] = (float)acc_w;
    }
    if (has_bias != 0) {
        db[col] = (float)acc_b;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8: 手動境界チェックが残っていること（イシュー #1105／#1893 の
    /// 教訓を踏まえた fail-closed 静的検査）。
    #[test]
    fn kernels_have_manual_boundary_guards() {
        assert!(RMSNORM_BWD_DX_NEW_F32.contains("if (row >= rows)"));
        assert!(RMSNORM_BWD_DW_NEW_F32.contains("if (col >= hidden)"));
        assert!(LAYER_NORM_BWD_DX_F32.contains("if (row >= rows)"));
        assert!(LAYER_NORM_BWD_DWDB_F32.contains("if (col >= hidden)"));
    }

    /// `long long` 添字でオーバーフローを避けていること。
    #[test]
    fn kernels_use_long_long_indexing() {
        for src in [
            RMSNORM_BWD_DX_NEW_F32,
            RMSNORM_BWD_DW_NEW_F32,
            LAYER_NORM_BWD_DX_F32,
            LAYER_NORM_BWD_DWDB_F32,
        ] {
            assert!(src.contains("long long"));
        }
    }

    /// dw／db の長軸縮約が `.claude/rules/coding-rust.md` の契約
    /// （`float term` を確定してから `(double)term` を蓄積する）を
    /// 満たしていること。
    #[test]
    fn dw_kernels_accumulate_f32_term_into_f64() {
        assert!(RMSNORM_BWD_DW_NEW_F32.contains("float term = dyv * xhat;"));
        assert!(RMSNORM_BWD_DW_NEW_F32.contains("acc_w = (double)term + acc_w;"));
        assert!(LAYER_NORM_BWD_DWDB_F32.contains("float term = dyv * xhat;"));
        assert!(LAYER_NORM_BWD_DWDB_F32.contains("acc_w = (double)term + acc_w;"));
        assert!(LAYER_NORM_BWD_DWDB_F32.contains("acc_b = (double)dyv + acc_b;"));
    }

    /// イシュー #1105／#1893 の教訓（`rsqrtf`／`INFINITY`／`#include` を
    /// 使わない）を全カーネルで固定する fail-closed 検査。
    #[test]
    fn kernels_do_not_reference_forbidden_symbols() {
        for src in [
            RMSNORM_BWD_DX_NEW_F32,
            RMSNORM_BWD_DW_NEW_F32,
            LAYER_NORM_BWD_DX_F32,
            LAYER_NORM_BWD_DWDB_F32,
        ] {
            assert!(!src.contains("rsqrtf"));
            assert!(!src.contains("INFINITY"));
            assert!(!src.contains("#include"));
        }
    }

    /// dx カーネルが行ごとの統計スクラッチ（RMSNorm は `rstd`・LayerNorm
    /// は `mean`／`rstd`）を書き出していること（dw／db カーネルが再利用
    /// する契約の存在確認）。
    #[test]
    fn dx_kernels_write_stat_scratch_buffers() {
        assert!(RMSNORM_BWD_DX_NEW_F32.contains("rstd_out[row] = (double)rstd;"));
        assert!(LAYER_NORM_BWD_DX_F32.contains("mean_out[row] = mean;"));
        assert!(LAYER_NORM_BWD_DX_F32.contains("rstd_out[row] = rstd;"));
    }
}
