// 融合 RMSNorm 順伝播カーネル（イシュー #604・CUDA 側 G-6・#592 の Metal 対応版）。
//
// 意味論: out = x * rsqrt(sum(x^2, axis=-1) * inv_n + eps) * w
// （has_weight == 0 の場合は w への乗算をスキップ）。
// `ops.rs::MetalBackendOps::run_fused` から canonical プラン（mean 化なし・
// eps=0・has_weight=0）検出時に inv_n=1.0 で呼ばれる（`rmsnorm.rs` 参照）。
//
// FMA 契約: 累算は `fma()` を明示使用し CPU 参照実装（`f32::mul_add`）・
// CUDA 側 `fmaf` と揃える（REQ-2・`.claude/rules/coding-rust.md`）。
// コンパイルオプションは `pipeline::compile_options()`
// （`mathMode=Safe`・`mathFloatingPointFunctions=Precise`）を適用する。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）固定。threadgroup 全体を跨ぐ
// バリア（barrier 系関数の threadgroup 版）は使わず、threadgroup memory の
// 可視性が必要な箇所のみ
// `simdgroup_barrier(mem_flags::mem_threadgroup)`（CUDA `__syncwarp` 対応）
// を使う。ホスト側（`rmsnorm.rs::MetalRmsNorm::new`）が
// `threadExecutionWidth == 32` を起動前に検証する（fail-closed）。
//
// reduction は 5 段 butterfly（`simd_shuffle_xor` 幅 16/8/4/2/1）。MFA の
// 2 段シャッフル（幅 1・8）は `simdgroup_matrix` の Morton 順レイアウト
// 前提であり、本カーネルは行方向の線形レイアウトのため適用しない
// （`docs/backend-metal-morton-mapping-decision.md` の判断と整合）。
//
// persistent threadgroup 方式: 各 threadgroup が
// `for (row = tg_id; row < rows; row += grid_size)` で複数行を処理する。
// `grid_size`（実際に dispatch した threadgroup 数）はホスト側が
// `row_kernel::derive_persistent_grid` で導出しカーネル引数として渡す
// （MSL の `[[threadgroups_per_grid]]` 属性に依存せず、ホスト側の
// dispatch 値と一致させる単一の真実源にするため）。
//
// 1 パス経路（`rmsnorm_f32_onepass`）は threadgroup memory に行を常駐し
// x の HBM 読みを 1 回のみにする。固定長配列
// `RMSNORM_ONEPASS_MAX_HIDDEN`（`row_kernel::ONEPASS_MAX_HIDDEN` と
// 一致させる単一の真実源。`tests/rmsnorm_softmax_source_evidence.rs` が
// 数値の一致をロックする）を超える行長は 2 パス経路
// （`rmsnorm_f32_twopass`。device メモリを再読、threadgroup memory 不使用）
// へ回す（`row_kernel::select_route` がホスト側で判定）。
//
// REQ-8 境界検査: ベクトル化（`float4`）ロードは `hidden % 4 == 0` の
// 場合のみ適用し、適用時も手動境界チェック（`base + 3 < hidden`）を
// 維持する。ループ添字は `ulong`（`row_base` 等）で宣言し、大きな
// `rows * hidden` に対する乗算オーバーフローを避ける
// （CUDA 側 PR #706 是正と同等の対策）。

#include <metal_stdlib>
using namespace metal;

constant uint RMSNORM_ONEPASS_MAX_HIDDEN = 4096u;
constant uint RMSNORM_SIMD_WIDTH = 32u;

// 32 レーン全体の二乗和を 5 段 butterfly で reduction する（全レーンが
// 同じ合計値を持つ状態で戻る）。
inline float rmsnorm_reduce_sum(float v) {
    v += simd_shuffle_xor(v, 16u);
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

kernel void rmsnorm_f32_onepass(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant float& inv_n [[buffer(6)]],
    constant int& has_weight [[buffer(7)]],
    constant uint& grid_size [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float smem[RMSNORM_ONEPASS_MAX_HIDDEN];

    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;
        float acc = 0.0f;

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                smem[base] = v.x;
                smem[base + 1u] = v.y;
                smem[base + 2u] = v.z;
                smem[base + 3u] = v.w;
                acc = fma(v.x, v.x, acc);
                acc = fma(v.y, v.y, acc);
                acc = fma(v.z, v.z, acc);
                acc = fma(v.w, v.w, acc);
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                smem[idx] = v;
                acc = fma(v, v, acc);
            }
        }

        // 各レーンが書いた `smem` は正規化フェーズで同じレーン自身のみが
        // 読み戻す（ストライドパターンが往復で一致するため）が、
        // コンパイラ最適化・命令並び替えに対する防御として明示的に
        // 可視性バリアを置く（CUDA `__syncwarp` 相当）。
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sum_sq = rmsnorm_reduce_sum(acc);
        float rstd = rsqrt(fma(sum_sq, inv_n, eps));

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(smem[base], smem[base + 1u], smem[base + 2u], smem[base + 3u]);
                float4 wv = float4(1.0f);
                if (has_weight != 0) {
                    wv = float4(w[base], w[base + 1u], w[base + 2u], w[base + 3u]);
                }
                float4 o = v * rstd * wv;
                out[row_base + base] = o.x;
                out[row_base + base + 1u] = o.y;
                out[row_base + base + 2u] = o.z;
                out[row_base + base + 3u] = o.w;
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = smem[idx];
                float wv = (has_weight != 0) ? w[idx] : 1.0f;
                out[row_base + idx] = v * rstd * wv;
            }
        }
    }
}

kernel void rmsnorm_f32_twopass(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant float& inv_n [[buffer(6)]],
    constant int& has_weight [[buffer(7)]],
    constant uint& grid_size [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;
        float acc = 0.0f;

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                acc = fma(v.x, v.x, acc);
                acc = fma(v.y, v.y, acc);
                acc = fma(v.z, v.z, acc);
                acc = fma(v.w, v.w, acc);
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                acc = fma(v, v, acc);
            }
        }

        float sum_sq = rmsnorm_reduce_sum(acc);
        float rstd = rsqrt(fma(sum_sq, inv_n, eps));

        // device メモリを再読（threadgroup memory 不使用の 2 パス経路）。
        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                float4 wv = float4(1.0f);
                if (has_weight != 0) {
                    wv = float4(w[base], w[base + 1u], w[base + 2u], w[base + 3u]);
                }
                float4 o = v * rstd * wv;
                out[row_base + base] = o.x;
                out[row_base + base + 1u] = o.y;
                out[row_base + base + 2u] = o.z;
                out[row_base + base + 3u] = o.w;
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                float wv = (has_weight != 0) ? w[idx] : 1.0f;
                out[row_base + idx] = v * rstd * wv;
            }
        }
    }
}
