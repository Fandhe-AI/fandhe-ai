// 平均二乗誤差（MSE）融合カーネルの MSL ソース（イシュー #1045・親イシュー
// #1043「カーネル融合・autodiff 実行モデルの強化」。CUDA 側
// `backend-cuda::kernels_mse`〈同イシュー〉の Metal 対応版）。
//
// `crate::mse` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/elementwise.metal` と同一設定）で実行時コンパイルする。
//
// # forward の 2 段構成（`mse_partial_f32` → `mse_finalize_f32`）
//
// `docs/kernel-fusion.md` 限界表が「reduction 融合はバックエンド実行
// レベルで未実装」と記録していた対象のうち、MSE の `Σ(pred−target)²`
// 全要素縮約を古典的な 2 段 reduction で実装する（CUDA 側
// `kernels_mse.rs` と同じ 2 段構成。冒頭コメント参照）:
//
//   1. `mse_partial_f32`: grid-stride ループで各 threadgroup が担当区間の
//      `Σ(pred−target)²` を計算し、threadgroup ごとの部分和 1 個を
//      `partial[tg_id]` へ書く（起動 threadgroup 数は `mse.rs` が
//      `min(ceil_div(numel, MSE_THREADGROUP_WIDTH), MSE_MAX_THREADGROUPS)`
//      で決定し、`partial` バッファの長さと必ず一致させる契約）。
//   2. `mse_finalize_f32`: 1 threadgroup のみを起動し、`partial`
//      （高々 `MSE_MAX_THREADGROUPS` 要素）を再度総和したのち `factor`
//      （`Mean` は `1/n`、`Sum` は `1.0`。呼び出し元がホスト側で決定）を
//      乗じて `out[0]` へ書く。
//
// # simdgroup 内総和（`simd_sum`）+ threadgroup 間結合
//
// `MSE_THREADGROUP_WIDTH = 256`（8 simdgroup）に対し、各 simdgroup は
// `simd_sum`（Metal 組み込みの simdgroup 内総和関数）で 32 レーン分を
// 1 個の値へ縮約し、simdgroup 代表値を `threadgroup float
// simd_sums[8]` へ格納する。`threadgroup_barrier(mem_flags::
// mem_threadgroup)` で同期したのち、simdgroup 0 の lane 0 が
// `simd_sums[0..8]` を固定順序（添字昇順）で逐次結合して
// `partial[tg_id]` へ書く。ブロック内の結合順序が
// `simdgroup_index_in_threadgroup` 昇順で固定されているため、CUDA 側の
// warp butterfly と同様に **決定的**（bit 再現可能）である
// （`backend-cpu::mse::mse_sum_sq_f32` の固定チャンク決定性契約と同種の
// 設計判断）。
//
// # REQ-8（カーネル境界検査規約）
//
// `mse_partial_f32`・`mse_backward_f32` は `idx < numel` の手動境界
// チェックを維持する。`mse_finalize_f32` も `idx < num_partials` を維持
// する。ベクトル化ロード等の最適化は本イシューでは適用しない。
//
// # 意味論の正
//
// `backend-cpu::mse`（`crates/backend-cpu/src/mse.rs`）が意味論の正。
// `diff*diff` の累積に `fma`（単精度 FMA）を用いる点は CPU 側
// `f32::mul_add`・CUDA 側 `fmaf` と同じ FMA 契約統一方針
// （`.claude/rules/coding-rust.md`）だが、累積順序は異なるため、
// バックエンド間の数値突合は統一複合判定「相対誤差 1e-3 未満 または
// 絶対誤差 1e-5 未満」で検証する。

#include <metal_stdlib>
using namespace metal;

// simdgroup 数（`MSE_THREADGROUP_WIDTH / 32`。Rust 側 `mse.rs::
// MSE_THREADGROUP_WIDTH` と同期させる固定値）。
constant uint MSE_SIMDGROUPS_PER_TG = 8u;

/// forward 1 段目: 各 threadgroup が担当区間の
/// `Σ(pred[i]−target[i])²` を計算し `partial[tg_id]` へ書く。
///
/// `partial` の長さは呼び出し元が起動 threadgroup 数と一致させて確保
/// する契約（本ファイル冒頭コメント参照）。`numel`（`uint`）は `pred`／
/// `target` の要素数。
kernel void mse_partial_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[MSE_SIMDGROUPS_PER_TG];

    float acc = 0.0f;
    uint stride = grid_size * tg_size;
    for (uint idx = tg_id * tg_size + tid; idx < numel; idx += stride) {
        float diff = pred[idx] - target[idx];
        acc = fma(diff, diff, acc);
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < MSE_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        partial[tg_id] = block_sum;
    }
}

/// forward 2 段目: `partial`（`num_partials` 要素。1 threadgroup のみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く。`factor` は `Mean`
/// なら `1.0 / n`、`Sum` なら `1.0`（ホスト側 `mse.rs` が決定する）。
kernel void mse_finalize_f32(
    device const float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[MSE_SIMDGROUPS_PER_TG];

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
        for (uint i = 0; i < MSE_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        out[0] = block_sum * factor;
    }
}

/// backward: `dPred[i] = scale·(pred[i]−target[i])`（1 スレッド 1 要素。
/// `elementwise.metal` の単項カーネルと同じ 1 次元グリッド・境界検査）。
/// `dTarget = −dPred` はホスト側が計算する契約
/// （`backend_ops.rs::BackendOps::mse_loss_backward` doc 参照。本
/// カーネルは `dPred` のみを出力し、追加のディスパッチ・バッファ確保・
/// readback を発生させない）。
kernel void mse_backward_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* dpred [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < numel) {
        dpred[idx] = scale * (pred[idx] - target[idx]);
    }
}
