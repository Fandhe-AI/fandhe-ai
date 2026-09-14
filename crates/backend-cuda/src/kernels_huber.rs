//! Huber／SmoothL1 損失融合カーネルの CUDA C カーネルソース（NVRTC
//! 実行時コンパイル用の静的文字列。イシュー #1739）。
//!
//! `kernels_mse.rs`（MSE。イシュー #1045）と同じ 2 段 reduction 構成を
//! 踏襲する: [`HUBER_PARTIAL_F32`]（forward 1 段目・grid-stride ループで
//! 各ブロックが担当区間の `Σ l(pred−target)` を計算）→
//! [`HUBER_FINALIZE_F32`]（forward 2 段目・`partial` を単一ブロックで
//! 再度総和し `factor` を乗じる）。`kernels_mse.rs::MSE_FINALIZE_F32` を
//! 再利用せず本モジュール専用に複製する理由: 両者は数式的に同一（総和
//! とスカラー乗算）だが、モジュール独立性（`huber.rs` が `kernels_mse`
//! へ依存しない）を優先する——将来 MSE 側の `finalize` 実装が変わっても
//! Huber 側が意図せず影響を受けない（`docs/kernel-fusion.md` の各融合
//! カーネルの独立複製方針と同じ判断）。
//!
//! `kind`（`int`。0=Huber, 1=SmoothL1）・`delta`（`float`）はカーネル
//! 引数として渡す（`autodiff::eval::huber_elem_loss`／`huber_elem_grad`
//! の意味論を GPU 側で複製する。分岐は呼び出し単位で全スレッド一様
//! ——ダイバージェンスなし）。`huber_elem_loss`／`huber_elem_grad`
//! （`__device__` inline）は forward・backward 両カーネルソースへ
//! `concat!` でコンパイル時に埋め込む（ソースが実行時コンパイル用の
//! 静的文字列のため、`kernels_mse.rs` と同じく `pub const &'static str`
//! のまま公開する）。
//!
//! # 決定性（float atomicAdd を使わない理由）・warp butterfly
//!
//! `kernels_mse.rs` 冒頭コメントと同一の設計判断（`atomicAdd` 不使用・
//! `blockIdx.x` 昇順結合・`HUBER_BLOCK_DIM = 256`〈8 warp〉の 5 段
//! butterfly）をそのまま踏襲する。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `huber_partial_f32`・`huber_backward_f32` は `idx < numel`（grid-stride
//! ループ）、`huber_finalize_f32` は `idx < num_partials` の手動境界
//! チェックを維持する。
//!
//! # sign(d) の統一（`copysignf`）
//!
//! `sign(d)`（`d = pred − target`）は 3 バックエンドとも `copysign` で
//! 統一する（`.claude/rules/coding-rust.md` の丸め方針統一と同じ理由で
//! `±0`／符号の扱いをバックエンド間で一致させる。実装計画 §2.1）。CUDA
//! は `copysignf` を使う。
//!
//! # 意味論の正
//!
//! `fandhe_ai_autodiff::eval::huber_elem_loss`／`huber_elem_grad` が
//! 意味論の正。累積順序（grid-stride → warp butterfly → ブロック間
//! 逐次結合）は CPU 側と異なるため、バックエンド間の数値突合は統一
//! 複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で検証する
//! （`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_mse::MSE_BLOCK_DIM`
/// と同値・同じ理由）。
pub const HUBER_BLOCK_DIM: u32 = 256;

/// forward 2 段目（`huber_finalize_f32`）が単一ブロックで処理しきれる
/// `partial` の最大長（`kernels_mse::MSE_MAX_BLOCKS` と同値・同じ理由）。
pub const HUBER_MAX_BLOCKS: u32 = 1024;

/// forward 1 段目: 各ブロックが担当区間の `Σ l(pred[i]−target[i])` を
/// 計算し `partial[blockIdx.x]` へ書く（`kernels_mse::MSE_PARTIAL_F32`
/// と同じ 2 段 reduction 構成・同じ grid-stride ループ・`long long`
/// 添字契約。積和ではない区分損失のため `fmaf` によるチャンク内 FMA は
/// 使わない——`.claude/rules/coding-rust.md` の FMA 契約は積和演算
/// 〈GEMM〉限定）。
pub const HUBER_PARTIAL_F32: &str = concat!(
    r#"
"#,
    r#"__device__ __forceinline__ float huber_elem_loss(float d, int kind, float delta) {
    float abs_d = fabsf(d);
    if (kind == 1) {
        if (abs_d < delta) {
            // d*d を先に計算すると delta・d が巨大な有限値のとき
            // 中間積が overflow しうる。abs_d < delta 分岐内では
            // |d/delta| < 1 が保証されるため、先に delta で割って
            // から d を掛けることで中間値を |d| 以下に抑える
            // （backend-cpu::huber::elem_loss・autodiff::eval::
            // huber_elem_loss・Metal shaders/huber.metal と同じ
            // 演算順序で揃える）。
            return 0.5f * (d / delta) * d;
        }
        return abs_d - 0.5f * delta;
    }
    if (abs_d < delta) {
        return 0.5f * d * d;
    }
    return delta * (abs_d - 0.5f * delta);
}
"#,
    r#"
extern "C" __global__ void huber_partial_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel,
    int kind,
    float delta)
{
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += huber_elem_loss(pred[idx] - target[idx], kind, delta);
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
            partial[blockIdx.x] = block_sum;
        }
    }
}
"#
);

/// forward 2 段目: `partial`（`num_partials` 要素。1 ブロックのみで起動）
/// を総和し `factor` を乗じて `out[0]` へ書く（`kernels_mse::
/// MSE_FINALIZE_F32` と数式的に同一だが本モジュール専用に複製。冒頭
/// コメント参照）。
pub const HUBER_FINALIZE_F32: &str = r#"
extern "C" __global__ void huber_finalize_f32(
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

/// backward: `dPred[i] = scale·grad_elem(pred[i]−target[i])`（1 スレッド
/// 1 要素。`kernels_mse::MSE_BACKWARD_F32` と同じ 1 次元 grid・境界
/// 検査）。`dTarget = −dPred` はホスト側（`huber.rs`）が計算する契約
/// （`backend_ops.rs::BackendOps::huber_loss_backward` doc 参照）。
pub const HUBER_BACKWARD_F32: &str = concat!(
    r#"__device__ __forceinline__ float huber_elem_grad(float d, int kind, float delta) {
    float abs_d = fabsf(d);
    if (kind == 1) {
        if (abs_d < delta) {
            return d / delta;
        }
        return copysignf(1.0f, d);
    }
    if (abs_d < delta) {
        return d;
    }
    return copysignf(delta, d);
}
"#,
    r#"
extern "C" __global__ void huber_backward_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ dpred,
    int numel,
    int kind,
    float delta,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dpred[idx] = scale * huber_elem_grad(pred[idx] - target[idx], kind, delta);
    }
}
"#
);

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 境界検査・非 atomic 決定性・`copysignf` 使用の証跡
    /// （`kernels_mse.rs::tests` と対になる文字列検査）。
    #[test]
    fn huber_partial_f32_has_grid_stride_bound_check() {
        assert!(HUBER_PARTIAL_F32.contains("idx < numel"));
        assert!(!HUBER_PARTIAL_F32.contains("atomicAdd"));
    }

    #[test]
    fn huber_partial_f32_grid_stride_loop_index_is_declared_long_long() {
        assert!(
            HUBER_PARTIAL_F32.contains("long long stride = (long long)gridDim.x * blockDim.x;"),
            "stride が long long で宣言されていない"
        );
        assert!(
            HUBER_PARTIAL_F32.contains(
                "for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride)"
            ),
            "idx ループ添字が long long で宣言されていない"
        );
    }

    #[test]
    fn huber_finalize_f32_has_bound_check() {
        assert!(HUBER_FINALIZE_F32.contains("idx < num_partials"));
        assert!(!HUBER_FINALIZE_F32.contains("atomicAdd"));
    }

    #[test]
    fn huber_backward_f32_has_bound_check() {
        assert!(HUBER_BACKWARD_F32.contains("if (idx < numel)"));
        assert!(!HUBER_BACKWARD_F32.contains("atomicAdd"));
    }

    #[test]
    fn huber_kernels_use_copysignf() {
        assert!(HUBER_BACKWARD_F32.contains("copysignf"));
    }

    #[test]
    fn huber_partial_f32_does_not_use_fma_contract() {
        // 区分損失（積和ではない）のため `fmaf` を使わない
        // （`kernels_mse.rs::mse_kernels_use_fma_contract` の対照）。
        assert!(!HUBER_PARTIAL_F32.contains("fmaf"));
    }
}
