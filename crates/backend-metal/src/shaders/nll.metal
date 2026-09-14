// 負対数尤度損失（NLLLoss）融合カーネルの MSL ソース（イシュー #1738・
// 親イシュー #1609「損失関数の拡張」。CUDA 側 `backend-cuda::
// kernels_nll`〈同イシュー〉の Metal 対応版・`mse.metal` を雛形にした
// 同型構成）。
//
// `crate::nll` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/mse.metal` と同一設定）で実行時コンパイルする。
//
// # forward の 2 段構成（`nll_partial_f32` → `nll_finalize_f32`）
//
// `mse.metal` と同じ 2 段 reduction 方式だが、縮約対象は要素
// （`numel`）ではなく**サンプル**（`n_samples = outer*inner`。1 サンプル
// = 1 スレッドの粒度）で、各サンプル `s` は正解クラス添字
// `targets[s]`（`device const int*`）を読み
// `−input[(o·C+t)·inner+i]` を加算する（`o = s/inner`, `i = s%inner`）。
//
// # simdgroup 内総和 + threadgroup 間結合
//
// `mse.metal` と同じ設計（`simd_sum` → `threadgroup_barrier` →
// simdgroup 0 の lane 0 が固定順序で逐次結合）。決定的（bit 再現可能）。
//
// # backward（`nll_backward_f32`）の書き込み一意性
//
// `kernels_nll.rs` 冒頭コメントと同じ根拠（`(o, i)` の組ごとに書き込み
// 添字 `idx = o·C·inner + t·inner + i` が一意）で競合の恐れがなく
// 排他制御用命令は不要（呼び出し元 `nll.rs::run_nll_backward_f32` が `dinput` を
// `alloc_zeroed_pooled` でゼロ初期化してから本カーネルを起動する契約。
// ターゲット位置以外は `0.0` のまま残る）。
//
// # REQ-8（カーネル境界検査規約）
//
// `nll_partial_f32`・`nll_backward_f32` は `idx < n_samples` の手動
// 境界チェックを維持する。`nll_finalize_f32` も `idx < num_partials`
// を維持する。`t`（`targets[idx]`）の範囲外検査（`0 <= t < C`）は
// ホスト側が実体化前に検証済みの契約のためカーネル内では行わない
// （`backend_ops.rs::BackendOps::nll_loss` doc 参照）。
//
// # 意味論の正
//
// `backend-cpu::nll`（`crates/backend-cpu/src/nll.rs`）・
// `fandhe_ai_autodiff::eval::nll_loss` が意味論の正。累積順序は異なる
// ため、バックエンド間の数値突合は統一複合判定「相対誤差 1e-3 未満
// または 絶対誤差 1e-5 未満」で検証する。

#include <metal_stdlib>
using namespace metal;

// simdgroup 数（`mse.metal::MSE_SIMDGROUPS_PER_TG` と同じ値・同じ理由）。
constant uint NLL_SIMDGROUPS_PER_TG = 8u;

/// forward 1 段目: 各 threadgroup が担当区間の
/// `Σ −input[(o·C+t)·inner+i]` を計算し `partial[tg_id]` へ書く
/// （`mse.metal::mse_partial_f32` のサンプル版）。
kernel void nll_partial_f32(
    device const float* input [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant uint& outer [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant uint& n_samples [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[NLL_SIMDGROUPS_PER_TG];
    (void)outer;

    float acc = 0.0f;
    // REQ-8: `mse.metal::mse_partial_f32` と同じ理由（`n_samples` 近傍
    // での unsigned wraparound 回避）で `ulong` 添字を使う。
    ulong stride = (ulong)grid_size * (ulong)tg_size;
    for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < n_samples; idx += stride) {
        uint o = (uint)(idx / inner);
        uint i = (uint)(idx % inner);
        int t = targets[idx];
        ulong input_idx = ((ulong)o * (ulong)num_classes + (ulong)t) * (ulong)inner + (ulong)i;
        acc -= input[input_idx];
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < NLL_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        partial[tg_id] = block_sum;
    }
}

/// forward 2 段目: `partial`（`num_partials` 要素。1 threadgroup のみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く（`mse.metal::
/// mse_finalize_f32` と同一構造）。
kernel void nll_finalize_f32(
    device const float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[NLL_SIMDGROUPS_PER_TG];

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
        for (uint i = 0; i < NLL_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        out[0] = block_sum * factor;
    }
}

/// backward: `dInput[(o·C+t)·inner+i] = −scale`（1 スレッド 1 サンプル。
/// `dinput` は呼び出し元が `alloc_zeroed_pooled` で事前ゼロ初期化済み。
/// ファイル冒頭「backward の書き込み一意性」参照）。
kernel void nll_backward_f32(
    device const int* targets [[buffer(0)]],
    device float* dinput [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& n_samples [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    (void)outer;
    if (idx < n_samples) {
        uint o = idx / inner;
        uint i = idx % inner;
        int t = targets[idx];
        ulong input_idx = ((ulong)o * (ulong)num_classes + (ulong)t) * (ulong)inner + (ulong)i;
        dinput[input_idx] = -scale;
    }
}
