//! BatchNorm1d／2d 順伝播カーネル（NVRTC 実行時コンパイル用の静的
//! 文字列。イシュー #1735・親 #1608）。
//!
//! `kernels_layer_norm.rs`（#1596）と同じ理由でソースを `nvcc`
//! 事前コンパイルせず文字列のまま埋め込む（CUDA toolkit 非搭載環境
//! でも `cargo build --workspace` が成立する契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # 設計（`docs/batch-norm-ops-design.md` §3.1「CUDA（1 warp = 1
//! channel）が再現すべき契約」の逐語実装）
//!
//! LayerNorm（`kernels_layer_norm.rs`）と同じ「persistent block・SMEM
//! 常駐・`float4` ベクトル化は採用しない」単純方針だが、縮約軸が
//! `hidden`（行方向）ではなく `M = n*spatial`（チャネル方向・ストライド
//! アクセス）である点が異なる。
//!
//! - **`batch_norm_train_f32`**: **1 CTA = 1 warp（32 レーン）= 1
//!   チャネル**、`grid_dim = c`（1 対 1 マッピング。persistent block
//!   ではない）。パス 1（平均）→ パス 2（二パス分散）→ パス 3（書き
//!   出し）の 3 段走査で `__shared__` メモリを一切使わない
//!   （`kernels_layer_norm.rs` と同型）。チャネル `ch` の局所添字
//!   `i in 0..M` は `batch = i/spatial`・`sp = i%spatial` へ分解し
//!   実データ添字 `batch*(c*spatial) + ch*spatial + sp` へ写像する
//!   （`backend-cpu::batch_norm::channel_index` の GPU 側複製）。
//!   `mean`／`var`（biased ÷M）は lane 0 のみが書き出す
//!   （`BatchNormTrainOutput::batch_mean`／`batch_var` 用）。
//! - **`batch_norm_infer_f32`**: 統計を再計算しないため縮約が不要
//!   （`grid-stride` の単純 elementwise。`block_dim = 256`）。要素
//!   添字 `idx` からチャネル `ch = (idx/spatial) % c` を求め、`mean`／
//!   `var`（呼び出し元が保持する running stats）をそのまま使う。
//!
//! `n`／`c`／`spatial`／`m`／`numel` はいずれも `i32::MAX` 以下
//! （`batch_norm.rs::validate_batch_norm_launch` がホスト側で検証する。
//! `kernels_layer_norm.rs` と同じ CUDA grid.x 上限の理由）。
//!
//! # 縮約精度契約（`.claude/rules/coding-rust.md`）
//!
//! 平均・分散とも `double` アキュムレータで蓄積する（`kernels_layer_
//! norm.rs` と同じ `__shfl_xor_sync` の `double` 直接対応を利用した
//! warp butterfly reduction。offset 16→8→4→2→1。CPU 側
//! `backend-cpu::batch_norm::warp_reduce_f64` と同一の加算順序）。
//! 二パス分散（`Σ(x−μ)²/M`。`E[x²]−μ²` は使わない）を採用する。
//! `mean`／`rstd` を `double` のまま保持し、`x̂` を確定する直前の
//! 1 回だけ `float` へ丸める（`kernels_layer_norm.rs` と同じ理由。
//! `mean` の早期丸めが `x̂` を歪める問題の再発防止）。affine は `fmaf`
//! で明示的に融合する（FMA 契約統一）。
//!
//! # REQ-8 境界検査
//!
//! `batch_norm_train_f32` は `if (ch >= c) return;`（grid 次元が `c`
//! と厳密一致するため通常到達しないが `kernels_layer_norm.rs` と同じ
//! fail-closed 方針で維持する）・ループ添字 `i < m` を手動ガードする。
//! `batch_norm_infer_f32` は grid-stride ループの `idx < numel` を
//! 手動ガードする。添字演算は `long long`（`n*c*spatial` の乗算
//! オーバーフロー回避。`kernels_layer_norm.rs::row_base` と同じ対策）。
//! `batch_norm_train_f32` の 3 本の縮約ループ自体のカウンタ `i` も
//! `long long` とする（`m` は `i32::MAX` まで許容されるホスト側検証の
//! ため、`int` のまま `i += 32` を続けると終端付近で符号付き
//! オーバーフローし負の添字による範囲外アクセスへつながる。ループ
//! 内部の添字計算〈`BN_IDX` マクロ〉はもともと `long long` キャスト
//! 済みだったが、カウンタ自身の更新はそれとは独立の別脆弱性だった
//! ため、カウンタの型ごと修正する）。

/// BatchNorm1d／2d train モードカーネル（冒頭コメント参照）。
///
/// 引数: `x`（`[n, c, spatial]` 行優先平坦化済み）・`w`（`has_weight
/// == 0` なら未参照——ただし呼び出し元は必ず `c` 要素のダミーバッファ
/// を渡す。`kernels_layer_norm.rs` と同じ理由でコンパイラの predicated
/// load に対する fail-closed な境界確保）・`b`（同様）・`out`・
/// `mean_out`／`var_out`（`BatchNormTrainOutput::batch_mean`／
/// `batch_var` 用。長さ `c`）・`n`・`c`・`spatial`・`m`（`n*spatial`。
/// ホスト側で事前計算した縮約要素数）・`eps`・`has_weight`・
/// `has_bias`。
pub const BATCH_NORM_TRAIN_F32: &str = r#"
extern "C" __global__ void batch_norm_train_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int n,
    int c,
    int spatial,
    int m,
    float eps,
    int has_weight,
    int has_bias)
{
    int lane = threadIdx.x;
    int ch = blockIdx.x;
    if (ch >= c) {
        return;
    }

    // チャネル `ch` の局所添字 `i` を実データ添字へ写像する
    // （`backend-cpu::batch_norm::channel_index` の GPU 側複製。
    // モジュール doc comment「設計」参照）。
    #define BN_IDX(i) ((long long)((i) / spatial) * (long long)c * (long long)spatial \
                       + (long long)ch * (long long)spatial + (long long)((i) % spatial))

    // パス 1: 平均（double アキュムレータ・warp butterfly reduction）。
    double sum = 0.0;
    for (long long i = lane; i < m; i += 32) {
        sum += (double)x[BN_IDX(i)];
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    double mean = sum / (double)m;

    // パス 2: 分散（二パス。`(x-mean)^2` を double で蓄積）。
    double sq_acc = 0.0;
    for (long long i = lane; i < m; i += 32) {
        double d = (double)x[BN_IDX(i)] - mean;
        sq_acc = fma(d, d, sq_acc);
    }
    __syncwarp(0xffffffffu);
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq_acc += __shfl_xor_sync(0xffffffffu, sq_acc, offset);
    }
    double var = sq_acc / (double)m;
    double rstd = 1.0 / sqrt(var + (double)eps);

    if (lane == 0) {
        mean_out[ch] = (float)mean;
        var_out[ch] = (float)var;
    }

    // パス 3: 書き出し（device メモリを再読）。`mean`／`rstd` を double
    // のまま偏差計算に使い、`x̂` を確定する直前の 1 回だけ `float` へ
    // 丸める（モジュール doc comment「縮約精度契約」参照）。
    float wv = (has_weight != 0) ? w[ch] : 1.0f;
    float bv = (has_bias != 0) ? b[ch] : 0.0f;
    for (long long i = lane; i < m; i += 32) {
        long long idx = BN_IDX(i);
        float xhat = (float)(((double)x[idx] - mean) * rstd);
        out[idx] = fmaf(xhat, wv, bv);
    }

    #undef BN_IDX
}
"#;

/// BatchNorm1d／2d eval モードカーネル（冒頭コメント参照）。統計を
/// 再計算しないため grid-stride の単純 elementwise とする。
///
/// 引数: `x`（`[n, c, spatial]` 行優先平坦化済み）・`mean`／`var`
/// （呼び出し元が保持する running stats。長さ `c`）・`w`／`b`
/// （`batch_norm_train_f32` と同じダミーバッファ契約）・`out`・`c`・
/// `spatial`・`numel`（`n*c*spatial`。`n` 自体はカーネル内で使わない
/// ため引数に含めない）・`eps`・`has_weight`・`has_bias`。
pub const BATCH_NORM_INFER_F32: &str = r#"
extern "C" __global__ void batch_norm_infer_f32(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int c,
    int spatial,
    long long numel,
    float eps,
    int has_weight,
    int has_bias)
{
    long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
         idx < numel;
         idx += stride) {
        int ch = (int)((idx / (long long)spatial) % (long long)c);
        double rstd = 1.0 / sqrt((double)var[ch] + (double)eps);
        float xhat = (float)(((double)x[idx] - (double)mean[ch]) * rstd);
        float wv = (has_weight != 0) ? w[ch] : 1.0f;
        float bv = (has_bias != 0) ? b[ch] : 0.0f;
        out[idx] = fmaf(xhat, wv, bv);
    }
}
"#;
