// online softmax カーネル（イシュー #604）。CUDA 側の G-7（#594・OPEN）に
// 先行する Metal 実装であり、CUDA 直接の parity 相手はまだ存在しない
// （`softmax.rs` ドキュメンテーションコメント「#594 の gap」参照。両
// バックエンドとも CPU 参照実装〈REQ-2 統一複合判定〉を経由した推移的な
// 数値担保に留まる）。
//
// 意味論: softmax(x)_i = e^(x_i - max(x)) / sum_j e^(x_j - max(x))
// （行ごと）。実装は MFA の softmax 設計（`log2(e)` 事前スケール +
// `exp2`・オンライン最大値更新・補正係数スキップ・有限負値境界マスク）を
// 踏襲する:
//
//   y = x * log2(e)                          （`exp` ではなく `exp2` のみ使用）
//   m_new = max(m, chunk_max(y))             （オンライン最大値更新）
//   correction = (m_new > m) ? exp2(m - m_new) : 1.0
//   l = l * correction + sum(exp2(y - m_new))
//
// 範囲外レーンの寄与は無限大の負値を直接使わず有限負値
// `SOFTMAX_MASK_Y = -(0.875 * FLT_MAX)`（y ドメイン。MFA の
// `(0.875 / log2(e)) * -FLT_MAX` を事前スケール後ドメインへ写像した値。
// マージン係数 0.875 の数値検証は `row_kernel.rs::softmax_mask_value`
// テスト参照）で表現し、`exp2(mask - m)` が有限入力から NaN/-inf を
// 生成しないことを保証する。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）固定。persistent threadgroup
// 方式・`grid_size` 引数・reduction 5 段 butterfly は `rmsnorm.metal` と
// 同一構造（同ファイル冒頭コメント参照）。
//
// 1 パス経路（`softmax_f32_onepass`）は最初の走査で threadgroup memory に
// `y`（事前スケール済み値）を常駐しつつ online max/sum を蓄積し、確定した
// `m`・`l` で正規化して書き出す。2 パス経路（`softmax_f32_twopass`）は
// device メモリを再読する（`row_kernel::select_route` がホスト側で判定。
// `RMSNORM_ONEPASS_MAX_HIDDEN` と同じ固定長 `SOFTMAX_ONEPASS_MAX_HIDDEN`
// を使う）。
//
// REQ-8: ループ添字は `row_base` を `ulong` で宣言し乗算オーバーフローを
// 避ける。境界外レーン（`idx >= hidden`）はマスク値を使い分岐で除外する
// （範囲外メモリアクセス自体は行わない）。

#include <metal_stdlib>
using namespace metal;

constant uint SOFTMAX_ONEPASS_MAX_HIDDEN = 4096u;
constant uint SOFTMAX_SIMD_WIDTH = 32u;
constant float SOFTMAX_LOG2E = 1.4426950408889634f;
// `row_kernel::softmax_mask_value`（Rust 側）と同じ値
// （`-(0.875 * f32::MAX)`）を MSL リテラルとして再現する。
constant float SOFTMAX_MASK_Y = -(0.875f * 3.402823466e+38f);

// 32 レーン全体の最大値を 5 段 butterfly で reduction する。
inline float softmax_reduce_max(float v) {
    v = max(v, simd_shuffle_xor(v, 16u));
    v = max(v, simd_shuffle_xor(v, 8u));
    v = max(v, simd_shuffle_xor(v, 4u));
    v = max(v, simd_shuffle_xor(v, 2u));
    v = max(v, simd_shuffle_xor(v, 1u));
    return v;
}

// 32 レーン全体の総和を 5 段 butterfly で reduction する。
inline float softmax_reduce_sum(float v) {
    v += simd_shuffle_xor(v, 16u);
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

kernel void softmax_f32_onepass(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& grid_size [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float smem[SOFTMAX_ONEPASS_MAX_HIDDEN];

    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;

        float m = SOFTMAX_MASK_Y;
        float l = 0.0f;

        // --- パス 1: online max/sum を蓄積しつつ `y` を smem へ常駐 ---
        for (uint chunk_start = 0u; chunk_start < hidden; chunk_start += SOFTMAX_SIMD_WIDTH) {
            uint idx = chunk_start + lane;
            bool valid = idx < hidden;
            float y = valid ? (x[row_base + idx] * SOFTMAX_LOG2E) : SOFTMAX_MASK_Y;
            if (valid) {
                smem[idx] = y;
            }

            float chunk_max = softmax_reduce_max(y);
            float m_new = max(m, chunk_max);
            float correction = (m_new > m) ? exp2(m - m_new) : 1.0f;
            float p = exp2(y - m_new);
            float chunk_sum = softmax_reduce_sum(p);
            l = l * correction + chunk_sum;
            m = m_new;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // --- パス 2: 確定した m・l で正規化して書き出す（smem 再利用） ---
        for (uint idx = lane; idx < hidden; idx += SOFTMAX_SIMD_WIDTH) {
            float y = smem[idx];
            float p = exp2(y - m);
            out[row_base + idx] = p / l;
        }
    }
}

kernel void softmax_f32_twopass(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& grid_size [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;

        float m = SOFTMAX_MASK_Y;
        float l = 0.0f;

        // --- パス 1: online max/sum を蓄積（smem 不使用） ---
        for (uint chunk_start = 0u; chunk_start < hidden; chunk_start += SOFTMAX_SIMD_WIDTH) {
            uint idx = chunk_start + lane;
            bool valid = idx < hidden;
            float y = valid ? (x[row_base + idx] * SOFTMAX_LOG2E) : SOFTMAX_MASK_Y;

            float chunk_max = softmax_reduce_max(y);
            float m_new = max(m, chunk_max);
            float correction = (m_new > m) ? exp2(m - m_new) : 1.0f;
            float p = exp2(y - m_new);
            float chunk_sum = softmax_reduce_sum(p);
            l = l * correction + chunk_sum;
            m = m_new;
        }

        // --- パス 2: device メモリを再読して正規化して書き出す ---
        for (uint idx = lane; idx < hidden; idx += SOFTMAX_SIMD_WIDTH) {
            float y = x[row_base + idx] * SOFTMAX_LOG2E;
            float p = exp2(y - m);
            out[row_base + idx] = p / l;
        }
    }
}
