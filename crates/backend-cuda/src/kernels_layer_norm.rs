//! LayerNorm 順伝播カーネル（NVRTC 実行時コンパイル用の静的文字列。
//! イシュー #1596）。
//!
//! `rmsnorm.rs`／`kernels_rmsnorm.rs`（#592）と同じ理由でソースを
//! `nvcc` 事前コンパイルせず文字列のまま埋め込む（CUDA toolkit
//! 非搭載環境でも `cargo build --workspace` が成立する契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # 設計（`kernels_rmsnorm.rs` より簡素化。イシュー #1596 実装計画
//! §4.4「最も単純で安全なカーネル設計」）
//!
//! `kernels_rmsnorm.rs` の persistent block・SMEM 常駐・`float4`
//! ベクトル化はいずれも採用しない。**1 CTA = 1 warp（32 レーン）が
//! 1 行を担当し、grid 次元 = `rows`（persistent block ではない単純な
//! 1 対 1 マッピング）**、device メモリを 2 回読む「二パス」構成
//! （平均 → 分散 → 書き出しの 3 段走査。`__shared__` メモリを一切
//! 使わない）とする。理由:
//!
//! - LayerNorm は平均・分散という 2 回の縮約を要し、RMSNorm の
//!   「1 縮約 + SMEM 常駐再利用」ほど単純な 1 パス化ができない
//! - 本イシューの受け入れ条件は「既存カーネルの接続パターンに倣った
//!   正しい新設」であり、persistent grid・occupancy 予算に基づく
//!   ブロック数最適化（`derive_persistent_grid_*`。`rmsnorm.rs`）は
//!   後続の性能課題として `docs/norm-ops-design.md` に切り出す
//!
//! `rows` は `i32::MAX` 以下（`validate_layer_norm_launch` がホスト側で
//! 検証する。CUDA の grid.x 上限は `2^31-1` のため単純な 1 対 1
//! マッピングでも起動できる）。
//!
//! # 縮約精度契約（`.claude/rules/coding-rust.md`）
//!
//! 平均・分散とも `double` アキュムレータで蓄積する（`kernels_rmsnorm.rs`
//! と同じ `__shfl_xor_sync` の `double` 直接対応を利用した warp
//! butterfly reduction。CUDA は `double` 型を持つため Metal の
//! Neumaier/scale-ssq 代替は不要）。二乗和ではなく「二パス分散」
//! （`Σ(x−μ)²/N`。`E[x²]−μ²` は使わない）を採用する
//! （`docs/norm-ops-design.md`）。
//!
//! # REQ-8 境界検査
//!
//! ベクトル化を行わないため境界検査は単純: `i < hidden`（grid-stride
//! ループの手動ガード）。ループ添字は `long long`（`row_base`）で
//! `rows * hidden` の乗算オーバーフローを避ける（`kernels_rmsnorm.rs`
//! と同じ対策。CUDA 側 PR #706 是正と同等）。

/// LayerNorm 順伝播カーネル（単一カーネル。冒頭コメント参照）。
///
/// 引数: `x`（`[rows, hidden]` 行優先）・`w`（`has_weight == 0` なら
/// 未参照——ただし呼び出し元は必ず `hidden` 要素のダミーバッファを渡す。
/// Metal 側 `rmsnorm.rs` と同じ理由でコンパイラの条件式最適化に対する
/// fail-closed な境界確保）・`b`（同様）・`out`・`rows`・`hidden`・
/// `eps`・`inv_n`（`= 1/hidden`）・`has_weight`・`has_bias`。
pub const LAYER_NORM_F32: &str = r#"
extern "C" __global__ void layer_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int rows,
    int hidden,
    float eps,
    float inv_n,
    int has_weight,
    int has_bias)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long row_base = (long long)row * (long long)hidden;
    const float* x_row = x + row_base;
    float* out_row = out + row_base;

    // パス 1: 平均（double アキュムレータ）。
    double sum = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        sum += (double)x_row[i];
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    double mean = sum * (double)inv_n;

    // パス 2: 分散（二パス。`(x-mean)^2` を double で蓄積）。
    double sq_acc = 0.0;
    for (long long i = lane; i < hidden; i += 32) {
        double d = (double)x_row[i] - mean;
        sq_acc = fma(d, d, sq_acc);
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq_acc += __shfl_xor_sync(0xffffffffu, sq_acc, offset);
    }
    double var = sq_acc * (double)inv_n;
    float rstd = (float)(1.0 / sqrt(var + (double)eps));
    float mean_f = (float)mean;

    // パス 3: 書き出し（device メモリを再読）。
    for (long long i = lane; i < hidden; i += 32) {
        float xhat = (x_row[i] - mean_f) * rstd;
        float wv = (has_weight != 0) ? w[i] : 1.0f;
        float bv = (has_bias != 0) ? b[i] : 0.0f;
        out_row[i] = xhat * wv + bv;
    }
}
"#;
