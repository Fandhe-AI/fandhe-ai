// Kullback-Leibler ダイバージェンス損失（KLDivLoss）融合カーネルの MSL
// ソース（イシュー #1738・親イシュー #1609「損失関数の拡張」。CUDA 側
// `backend-cuda::kernels_kl_div`〈同イシュー〉の Metal 対応版・
// `bce.metal`（イシュー #1737）を雛形にした同型構成）。
//
// `crate::kl_div` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/bce.metal` と同一設定）で実行時コンパイルする。
//
// # forward の 2 段構成（`kl_div_partial_f32` → `kl_div_finalize_f32`）
//
// `bce.metal` と同じ 2 段 reduction 方式だが、`Σ bce_elem_loss(..)` の
// 代わりに `Σ kl_div_elem_loss(input, target, kind)`（`log`／`exp` を
// 含む合成式）を縮約する。`kind`（`uint`。`0` = `Probabilities`、
// `1` = `LogProbabilities`）はカーネル引数として渡す。
//
// # simdgroup 内総和 + threadgroup 間結合
//
// `bce.metal` と同じ設計・同じ理由。
//
// # REQ-8（カーネル境界検査規約）
//
// `kl_div_partial_f32`・`kl_div_backward_f32` は `idx < numel` の手動
// 境界チェックを維持する。`kl_div_finalize_f32` も `idx < num_partials`
// を維持する。
//
// # 意味論の正
//
// `backend-cpu::kl_div`（`crates/backend-cpu/src/kl_div.rs`）・
// `fandhe_ai_autodiff::eval::kl_div_elem_loss`／
// `kl_div_elem_grad_input` が意味論の正。累積順序は異なるため、
// バックエンド間の数値突合は統一複合判定「相対誤差 1e-3 未満 または
// 絶対誤差 1e-5 未満」で検証する。

#include <metal_stdlib>
using namespace metal;

// simdgroup 数（`bce.metal::BCE_SIMDGROUPS_PER_TG` と同じ値・同じ理由）。
constant uint KL_DIV_SIMDGROUPS_PER_TG = 8u;

/// `docs/compat-api-scope.md` §1.2 の要素式（`kind`: `0` =
/// Probabilities、`1` = LogProbabilities）。`fandhe_ai_autodiff::eval::
/// kl_div_elem_loss` と同じ数式を MSL として複製する。
inline float kl_div_elem_loss(float input, float target, uint kind) {
    if (kind == 0u) {
        if (target == 0.0f) {
            return 0.0f;
        }
        return target * (precise::log(target) - input);
    }
    return precise::exp(target) * (target - input);
}

/// `kl_div_elem_loss` の `dInput`（`scale` を乗じる前。
/// `fandhe_ai_autodiff::eval::kl_div_elem_grad_input` と同じ数式）。
inline float kl_div_elem_grad_input(float input, float target, uint kind) {
    (void)input;
    if (kind == 0u) {
        return -target;
    }
    return -precise::exp(target);
}

/// forward 1 段目: 各 threadgroup が担当区間の
/// `Σ kl_div_elem_loss(input[i], target[i], kind)` を計算し
/// `partial[tg_id]` へ書く（`bce.metal::bce_partial_f32` と同型構成）。
kernel void kl_div_partial_f32(
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
    threadgroup float simd_sums[KL_DIV_SIMDGROUPS_PER_TG];

    float acc = 0.0f;
    ulong stride = (ulong)grid_size * (ulong)tg_size;
    for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride) {
        acc += kl_div_elem_loss(input[idx], target[idx], kind);
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < KL_DIV_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        partial[tg_id] = block_sum;
    }
}

/// forward 2 段目: `partial`（`num_partials` 要素。1 threadgroup のみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く（`bce.metal::
/// bce_finalize_f32` と同一構造）。
kernel void kl_div_finalize_f32(
    device const float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[KL_DIV_SIMDGROUPS_PER_TG];

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
        for (uint i = 0; i < KL_DIV_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        out[0] = block_sum * factor;
    }
}

/// backward: `dInput[i] = scale·kl_div_elem_grad_input(input[i],
/// target[i], kind)`（1 スレッド 1 要素。`bce.metal::bce_backward_f32`
/// と同型）。`dTarget` はホスト側が計算する契約
/// （`backend_ops.rs::BackendOps::kl_div_loss_backward` doc 参照）。
kernel void kl_div_backward_f32(
    device const float* input [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* dinput [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& kind [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < numel) {
        dinput[idx] = scale * kl_div_elem_grad_input(input[idx], target[idx], kind);
    }
}
