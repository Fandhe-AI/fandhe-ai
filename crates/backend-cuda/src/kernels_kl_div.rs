//! Kullback-Leibler ダイバージェンス損失（`KLDivLoss`）融合カーネルの
//! CUDA C カーネルソース（NVRTC 実行時コンパイル用の静的文字列。
//! イシュー #1738・親イシュー #1609「損失関数の拡張」）。
//! `kernels_bce.rs`（イシュー #1737）を雛形にした同型構成。
//!
//! `kl_div.rs`（呼び出し元）は本モジュールの 3 定数を
//! `nvrtc::compile_ptx` に渡し `CudaFunction` を得る。
//!
//! # forward の 2 段構成（`kl_div_partial_f32` → `kl_div_finalize_f32`）
//!
//! `kernels_bce.rs` と同じ 2 段 reduction 方式だが、`Σ bce_elem_loss(..)`
//! の代わりに `Σ kl_div_elem_loss(input, target, kind)`（`ln`／`expf` を
//! 含む合成式）を縮約する。`kind`（`int`。`0` =
//! [`fandhe_ai_tensor_core::KlDivTarget::Probabilities`]、`1` =
//! [`fandhe_ai_tensor_core::KlDivTarget::LogProbabilities`]）はカーネル
//! 引数として渡す（`kernels_bce.rs` の `kind` と同型）。
//!
//! # 決定性（float atomicAdd を使わない理由）
//!
//! `kernels_bce.rs` と同じ理由・同じ設計。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `kl_div_partial_f32`・`kl_div_backward_f32` は `idx < numel`
//! （grid-stride ループの場合は `for (...; idx < numel; ...)`）の手動
//! 境界チェックを維持する。`kl_div_finalize_f32` も `idx <
//! num_partials` を維持する。
//!
//! # 意味論の正
//!
//! `backend-cpu::kl_div`（`crates/backend-cpu/src/kl_div.rs`）・
//! `fandhe_ai_autodiff::eval::kl_div_elem_loss`／
//! `kl_div_elem_grad_input` が意味論の正。累積順序は異なるため、
//! バックエンド間の数値突合は統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」で検証する（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_bce::BCE_BLOCK_DIM`
/// と同値・同じ理由）。
pub const KL_DIV_BLOCK_DIM: u32 = 256;

/// forward 2 段目（`kl_div_finalize_f32`）が単一ブロックで処理しきれる
/// `partial` の最大長（`kernels_bce::BCE_MAX_BLOCKS` と同値・同じ理由）。
pub const KL_DIV_MAX_BLOCKS: u32 = 1024;

/// `kl_div_elem_loss`／`kl_div_elem_grad_input` device 関数を各カーネル
/// へ前置する共通ソース断片（`kind`: `0` = Probabilities、`1` =
/// LogProbabilities。`fandhe_ai_autodiff::eval::kl_div_elem_loss`／
/// `kl_div_elem_grad_input` と同じ数式を CUDA C として複製する）。
const KL_DIV_DEVICE_FUNCS: &str = r#"
__device__ __forceinline__ float kl_div_elem_loss(float input, float target, int kind) {
    if (kind == 0) {
        if (target == 0.0f) {
            return 0.0f;
        }
        return target * (logf(target) - input);
    }
    return expf(target) * (target - input);
}

__device__ __forceinline__ float kl_div_elem_grad_input(float input, float target, int kind) {
    (void)input;
    if (kind == 0) {
        return -target;
    }
    return -expf(target);
}
"#;

/// forward 1 段目: 各ブロックが担当区間の `Σ kl_div_elem_loss(..)` を
/// 計算し `partial[blockIdx.x]` へ書く（`kernels_bce::
/// bce_partial_f32_source` と同型構成）。
pub fn kl_div_partial_f32_source() -> String {
    format!(
        r#"
{KL_DIV_DEVICE_FUNCS}
extern "C" __global__ void kl_div_partial_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel,
    int kind)
{{
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {{
        acc += kl_div_elem_loss(input[idx], target[idx], kind);
    }}

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {{
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }}
    if (lane == 0) {{
        warp_sums[warp_id] = acc;
    }}
    __syncthreads();

    if (warp_id == 0) {{
        float block_sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {{
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, offset);
        }}
        if (lane == 0) {{
            partial[blockIdx.x] = block_sum;
        }}
    }}
}}
"#
    )
}

/// forward 2 段目: `partial`（`num_partials` 要素。1 ブロックのみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く（`kernels_bce::
/// BCE_FINALIZE_F32` と同一構造）。
pub const KL_DIV_FINALIZE_F32: &str = r#"
extern "C" __global__ void kl_div_finalize_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int num_partials,
    float factor)
{
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = 0.0f;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc += partial[idx];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            out[0] = block_sum * factor;
        }
    }
}
"#;

/// backward: `dInput[i] = scale·kl_div_elem_grad_input(input[i],
/// target[i], kind)`（1 スレッド 1 要素。`kernels_bce::
/// bce_backward_f32_source` と同型）。`dTarget` はホスト側（`grad.rs`）
/// が別途計算する契約（`backend_ops.rs::BackendOps::
/// kl_div_loss_backward` doc 参照）。
pub fn kl_div_backward_f32_source() -> String {
    format!(
        r#"
{KL_DIV_DEVICE_FUNCS}
extern "C" __global__ void kl_div_backward_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ dinput,
    int numel,
    int kind,
    float scale)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        dinput[idx] = scale * kl_div_elem_grad_input(input[idx], target[idx], kind);
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_div_partial_f32_has_grid_stride_bound_check() {
        let src = kl_div_partial_f32_source();
        assert!(src.contains("idx < numel"));
        assert!(!src.contains("atomicAdd"));
        assert!(src.contains("long long stride = (long long)gridDim.x * blockDim.x;"));
        assert!(src.contains(
            "for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride)"
        ));
    }

    #[test]
    fn kl_div_finalize_f32_has_bound_check() {
        assert!(KL_DIV_FINALIZE_F32.contains("idx < num_partials"));
        assert!(!KL_DIV_FINALIZE_F32.contains("atomicAdd"));
    }

    #[test]
    fn kl_div_backward_f32_has_bound_check() {
        let src = kl_div_backward_f32_source();
        assert!(src.contains("if (idx < numel)"));
        assert!(!src.contains("atomicAdd"));
    }
}
