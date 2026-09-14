//! 二値交差エントロピー損失（BCE）融合カーネルの CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列。イシュー #1737・親イシュー
//! #1609「損失関数の拡張」）。`kernels_mse.rs` を雛形にした同型構成。
//!
//! `bce.rs`（呼び出し元）は本モジュールの 3 定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る。`kernels_mse.rs` と同じ理由でソースを
//! `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA
//! ヘッダを一切要求しない。`.claude/rules/deps-policy.md`）。
//!
//! # forward の 2 段構成（`bce_partial_f32` → `bce_finalize_f32`）
//!
//! `kernels_mse.rs` と同じ 2 段 reduction 方式だが、`Σ(pred−target)²`
//! （単純な積和）の代わりに `Σ bce_elem_loss(input, target, kind)`
//! （`ln`／`log1pf`／`expf` を含む合成式）を縮約する。`kind`（`int`。
//! `0` = [`fandhe_ai_tensor_core::BceKind::Probabilities`]、`1` =
//! [`fandhe_ai_tensor_core::BceKind::Logits`]）はカーネル引数として渡し
//! （function constant ではなく通常の引数。分岐コストは 2 択のみで
//! 軽微）、`bce_elem_loss` device 関数が分岐する。
//!
//! # 決定性（float atomicAdd を使わない理由）
//!
//! `kernels_mse.rs` と同じ理由・同じ設計（ブロック間は `blockIdx.x`
//! 昇順の逐次結合、ブロック内は `__shfl_xor_sync` の固定 offset 列）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `bce_partial_f32`・`bce_backward_f32` は `idx < numel`
//! （grid-stride ループの場合は `for (...; idx < numel; ...)`）の手動
//! 境界チェックを維持する。`bce_finalize_f32` も `idx < num_partials`
//! を維持する。
//!
//! # 意味論の正
//!
//! `backend-cpu::bce`（`crates/backend-cpu/src/bce.rs`）・
//! `fandhe_ai_autodiff::eval::bce_elem_loss`／`bce_elem_grad_input` が
//! 意味論の正。累積順序（CPU: 固定チャンク／CUDA: grid-stride → warp
//! butterfly → ブロック間逐次結合）は異なるため、バックエンド間の数値
//! 突合は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で
//! 検証する（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_mse::MSE_BLOCK_DIM`
/// と同値・同じ理由）。
pub const BCE_BLOCK_DIM: u32 = 256;

/// forward 2 段目（`bce_finalize_f32`）が単一ブロックで処理しきれる
/// `partial` の最大長（`kernels_mse::MSE_MAX_BLOCKS` と同値・同じ理由）。
pub const BCE_MAX_BLOCKS: u32 = 1024;

/// `bce_elem_loss`／`bce_elem_grad_input` device 関数を各カーネルへ
/// 前置する共通ソース断片（`kind`: `0` = Probabilities、`1` = Logits。
/// `fandhe_ai_autodiff::eval::bce_elem_loss`／`bce_elem_grad_input` と
/// 同じ数式を CUDA C として複製する。`ln(input)` は `logf`、`ln(1+x)`
/// は `log1pf`、`|x|` は `fabsf` を使う）。
const BCE_DEVICE_FUNCS: &str = r#"
__device__ __forceinline__ float bce_elem_loss(float input, float target, int kind) {
    if (kind == 0) {
        float log_p = fmaxf(logf(input), -100.0f);
        float log_1mp = fmaxf(logf(1.0f - input), -100.0f);
        return -(target * log_p + (1.0f - target) * log_1mp);
    }
    return fmaxf(input, 0.0f) - input * target + log1pf(expf(-fabsf(input)));
}

__device__ __forceinline__ float bce_elem_grad_input(float input, float target, int kind) {
    if (kind == 0) {
        float denom = fmaxf(input * (1.0f - input), 1e-12f);
        return (input - target) / denom;
    }
    float sigmoid;
    if (input >= 0.0f) {
        sigmoid = 1.0f / (1.0f + expf(-input));
    } else {
        float e = expf(input);
        sigmoid = e / (1.0f + e);
    }
    return sigmoid - target;
}
"#;

/// forward 1 段目: 各ブロックが担当区間の `Σ bce_elem_loss(..)` を計算
/// し `partial[blockIdx.x]` へ書く（`kernels_mse::MSE_PARTIAL_F32` の
/// `diff*diff` を `bce_elem_loss` 呼び出しへ差し替えた同型構成）。
pub fn bce_partial_f32_source() -> String {
    format!(
        r#"
{BCE_DEVICE_FUNCS}
extern "C" __global__ void bce_partial_f32(
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
        acc += bce_elem_loss(input[idx], target[idx], kind);
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
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く
/// （`kernels_mse::MSE_FINALIZE_F32` と同一構造・`kind` 引数は不要
/// ── `partial` は既に `bce_elem_loss` 適用済みの部分和のため）。
pub const BCE_FINALIZE_F32: &str = r#"
extern "C" __global__ void bce_finalize_f32(
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

/// backward: `dInput[i] = scale·bce_elem_grad_input(input[i], target[i],
/// kind)`（1 スレッド 1 要素。`kernels_mse::MSE_BACKWARD_F32` と同型）。
/// `dTarget` はホスト側（`grad.rs`）が別途計算する契約
/// （`backend_ops.rs::BackendOps::bce_loss_backward` doc 参照）。
pub fn bce_backward_f32_source() -> String {
    format!(
        r#"
{BCE_DEVICE_FUNCS}
extern "C" __global__ void bce_backward_f32(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ dinput,
    int numel,
    int kind,
    float scale)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        dinput[idx] = scale * bce_elem_grad_input(input[idx], target[idx], kind);
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 境界検査・非 atomic 決定性・`long long` grid-stride 添字の
    /// 証跡（`kernels_mse.rs::tests` と同型の文字列検査）。
    #[test]
    fn bce_partial_f32_has_grid_stride_bound_check() {
        let src = bce_partial_f32_source();
        assert!(src.contains("idx < numel"));
        assert!(!src.contains("atomicAdd"));
        assert!(src.contains("long long stride = (long long)gridDim.x * blockDim.x;"));
        assert!(src.contains(
            "for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride)"
        ));
    }

    #[test]
    fn bce_finalize_f32_has_bound_check() {
        assert!(BCE_FINALIZE_F32.contains("idx < num_partials"));
        assert!(!BCE_FINALIZE_F32.contains("atomicAdd"));
    }

    #[test]
    fn bce_backward_f32_has_bound_check() {
        let src = bce_backward_f32_source();
        assert!(src.contains("if (idx < numel)"));
        assert!(!src.contains("atomicAdd"));
    }

    /// `log1pf` 使用（`docs/compat-api-scope.md` §1.2 の数値安定形。
    /// `ln(sigmoid(x))` の素朴計算による桁落ちを避けるための必須要件）
    /// を固定する。
    #[test]
    fn bce_device_funcs_use_log1pf() {
        assert!(BCE_DEVICE_FUNCS.contains("log1pf"));
    }
}
