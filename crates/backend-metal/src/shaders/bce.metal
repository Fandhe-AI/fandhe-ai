// 二値交差エントロピー損失（BCE）融合カーネルの MSL ソース（イシュー
// #1737・親イシュー #1609「損失関数の拡張」。CUDA 側
// `backend-cuda::kernels_bce`〈同イシュー〉の Metal 対応版・`mse.metal`
// を雛形にした同型構成）。
//
// `crate::bce` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/mse.metal` と同一設定）で実行時コンパイルする。
//
// # forward の 2 段構成（`bce_partial_f32` → `bce_finalize_f32`）
//
// `mse.metal` と同じ 2 段 reduction 方式だが、`Σ(pred−target)²`
// （単純な積和）の代わりに `Σ bce_elem_loss(input, target, kind)`
// （`log`／`exp` を含む合成式）を縮約する。`kind`（`uint`。`0` =
// `Probabilities`、`1` = `Logits`）はカーネル引数として渡す。
//
// # simdgroup 内総和 + threadgroup 間結合
//
// `mse.metal` と同じ設計（`simd_sum` → `threadgroup_barrier` →
// simdgroup 0 の lane 0 が固定順序で逐次結合）。決定的（bit 再現可能）。
//
// # log1p の自作（MSL に `log1p` 組み込みが無いため）
//
// MSL（Metal Shading Language）には C99 の `log1p`／`log1pf` に相当する
// 組み込み関数が無い（`metal_stdlib` に未収録）。`bce_log1p_f32` は
// `fandhe_ai_autodiff::eval` の Rust 側 `f32::ln_1p`
// （`x` が小さいとき `log(1.0 + x)` の素朴計算が桁落ちする問題を
// 回避する標準アルゴリズム）と同じ「`u = 1.0 + x`; `u == 1.0` なら
// `x` をそのまま返す（`x` が丸め誤差の範囲で無視できるほど小さい）；
// それ以外は `log(u) * (x / (u - 1.0))`（`u - 1.0` は `x` の丸め誤差を
// 打ち消す補正項）」という式を実装する。`x = -|input|` は常に `<= 0`
// のため `u > 0` が保証され `log(u)` は有限。
//
// # REQ-8（カーネル境界検査規約）
//
// `bce_partial_f32`・`bce_backward_f32` は `idx < numel` の手動境界
// チェックを維持する。`bce_finalize_f32` も `idx < num_partials` を
// 維持する。
//
// # 意味論の正
//
// `backend-cpu::bce`（`crates/backend-cpu/src/bce.rs`）・
// `fandhe_ai_autodiff::eval::bce_elem_loss`／`bce_elem_grad_input` が
// 意味論の正。累積順序は異なるため、バックエンド間の数値突合は統一
// 複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で検証する。

#include <metal_stdlib>
using namespace metal;

// simdgroup 数（`mse.metal::MSE_SIMDGROUPS_PER_TG` と同じ値・同じ理由）。
constant uint BCE_SIMDGROUPS_PER_TG = 8u;

/// `log(1 + x)`（MSL に組み込みが無いための自作。冒頭コメント参照）。
inline float bce_log1p_f32(float x) {
    float u = 1.0f + x;
    if (u == 1.0f) {
        return x;
    }
    return precise::log(u) * (x / (u - 1.0f));
}

/// `docs/compat-api-scope.md` §1.2 の要素式（`kind`: `0` =
/// Probabilities、`1` = Logits）。`fandhe_ai_autodiff::eval::
/// bce_elem_loss` と同じ数式を MSL として複製する。
inline float bce_elem_loss(float input, float target, uint kind) {
    if (kind == 0u) {
        float log_p = max(precise::log(input), -100.0f);
        float log_1mp = max(precise::log(1.0f - input), -100.0f);
        return -(target * log_p + (1.0f - target) * log_1mp);
    }
    return max(input, 0.0f) - input * target + bce_log1p_f32(precise::exp(-fabs(input)));
}

/// `bce_elem_loss` の `dInput`（`scale` を乗じる前。`fandhe_ai_autodiff::
/// eval::bce_elem_grad_input` と同じ数式）。
inline float bce_elem_grad_input(float input, float target, uint kind) {
    if (kind == 0u) {
        float denom = max(input * (1.0f - input), 1e-12f);
        return (input - target) / denom;
    }
    float sigmoid;
    if (input >= 0.0f) {
        sigmoid = 1.0f / (1.0f + precise::exp(-input));
    } else {
        float e = precise::exp(input);
        sigmoid = e / (1.0f + e);
    }
    return sigmoid - target;
}

/// forward 1 段目: 各 threadgroup が担当区間の
/// `Σ bce_elem_loss(input[i], target[i], kind)` を計算し
/// `partial[tg_id]` へ書く。`partial` の長さは呼び出し元が起動
/// threadgroup 数と一致させて確保する契約（`mse.metal` と同じ）。
kernel void bce_partial_f32(
    device const float* input [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& kind [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[BCE_SIMDGROUPS_PER_TG];

    float acc = 0.0f;
    // `mse.metal::mse_partial_f32` と同じ理由（`numel` 近傍での unsigned
    // wraparound 回避）で `ulong` 添字を使う。
    ulong stride = (ulong)grid_size * (ulong)tg_size;
    for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride) {
        acc += bce_elem_loss(input[idx], target[idx], kind);
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < BCE_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        partial[tg_id] = block_sum;
    }
}

/// forward 2 段目: `partial`（`num_partials` 要素。1 threadgroup のみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く（`mse.metal::
/// mse_finalize_f32` と同一構造。`kind` 引数は不要 ── `partial` は既に
/// `bce_elem_loss` 適用済みの部分和のため）。
kernel void bce_finalize_f32(
    device const float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[BCE_SIMDGROUPS_PER_TG];

    float acc = 0.0f;
    for (uint idx = tid; idx < num_partials; idx += tg_size) {
        acc += partial[idx];
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < BCE_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        out[0] = block_sum * factor;
    }
}

/// backward: `dInput[i] = scale·bce_elem_grad_input(input[i], target[i],
/// kind)`（1 スレッド 1 要素。`mse.metal::mse_backward_f32` と同型）。
/// `dTarget` はホスト側が計算する契約（`backend_ops.rs::BackendOps::
/// bce_loss_backward` doc 参照）のため、本カーネルは `dInput` のみを
/// 出力する。
kernel void bce_backward_f32(
    device const float* input [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* dinput [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& kind [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < numel) {
        dinput[idx] = scale * bce_elem_grad_input(input[idx], target[idx], kind);
    }
}
