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
// 縮約精度契約（正規化統計の f64 アキュムレータ統一。イシュー #1102。
// ユーザー承認 2026-09-01）: CUDA・CPU バックエンドは二乗和（`rstd`
// 導出）の蓄積を `double`／`f64` アキュムレータで行うが、Apple GPU の
// Metal Shading Language は `double` 型を持たない（Apple GPU family は
// 倍精度浮動小数点演算をサポートしない）。そのため本カーネルは
// **Neumaier 改良版 Kahan 補償和**（`f32` のまま、丸め誤差の補償項
// `comp` を別に保持し毎回加算前に打ち消す）を「`f64` アキュムレータ
// 相当」の実装形として適用する。二乗和の**レーン内蓄積**（各レーンが
// 担当する `hidden/32` 要素程度の逐次和。CUDA 側の「レーン内部分和を
// double 化」に対応）と、**32 レーン間の butterfly 縮約**（CUDA 側の
// warp shuffle を double で行うのに対応）の両方に Kahan 補償を適用する
// （`rmsnorm_reduce_sum_kahan` 参照）。
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

// Neumaier 改良版 Kahan 補償和の 1 ステップ: `sum` に `value` を加算し、
// 打ち消された丸め誤差を `comp`（補償項）へ蓄積する。`|sum| >= |value|`
// で分岐するのが Neumaier 版の要点（元祖 Kahan は `sum` が常に大きい側
// という前提を置くが、本カーネルではレーン間縮約で大小が入れ替わりうる
// ため Neumaier 版が必須）。
inline void rmsnorm_kahan_add(thread float& sum, thread float& comp, float value) {
    float t = sum + value;
    if (fabs(sum) >= fabs(value)) {
        comp += (sum - t) + value;
    } else {
        comp += (value - t) + sum;
    }
    sum = t;
}

// 32 レーン全体の二乗和を 5 段 butterfly で reduction する（`f64`
// アキュムレータ相当の Kahan 補償和。本ファイル冒頭コメント「縮約精度
// 契約」参照）。各レーンの Kahan 補償和ペア `(sum, comp)` を
// `simd_shuffle_xor` でレーン間交換しながら Neumaier 方式で合成する
// （2 つの既に補償済みの部分和を単純に足すと補償情報が失われるため、
// レーン内蓄積と同じ Neumaier ステップをレーン間結合にも適用する）。
// 全レーンが同じ合計値（`sum + comp`）を持つ状態で戻る。
inline float rmsnorm_reduce_sum_kahan(thread float& sum, thread float& comp) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_sum = simd_shuffle_xor(sum, offset);
        float other_comp = simd_shuffle_xor(comp, offset);
        float t = sum + other_sum;
        float c;
        if (fabs(sum) >= fabs(other_sum)) {
            c = (sum - t) + other_sum;
        } else {
            c = (other_sum - t) + sum;
        }
        sum = t;
        comp = comp + other_comp + c;
    }
    return sum + comp;
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
        // `acc`（Kahan 補償和の主項）・`acc_c`（補償項）でレーン内の
        // 二乗和を蓄積する（縮約精度契約。本ファイル冒頭コメント参照）。
        float acc = 0.0f;
        float acc_c = 0.0f;

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                smem[base] = v.x;
                smem[base + 1u] = v.y;
                smem[base + 2u] = v.z;
                smem[base + 3u] = v.w;
                rmsnorm_kahan_add(acc, acc_c, v.x * v.x);
                rmsnorm_kahan_add(acc, acc_c, v.y * v.y);
                rmsnorm_kahan_add(acc, acc_c, v.z * v.z);
                rmsnorm_kahan_add(acc, acc_c, v.w * v.w);
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                smem[idx] = v;
                rmsnorm_kahan_add(acc, acc_c, v * v);
            }
        }

        // 各レーンが書いた `smem` は正規化フェーズで同じレーン自身のみが
        // 読み戻す（ストライドパターンが往復で一致するため）が、
        // コンパイラ最適化・命令並び替えに対する防御として明示的に
        // 可視性バリアを置く（CUDA `__syncwarp` 相当）。
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sum_sq = rmsnorm_reduce_sum_kahan(acc, acc_c);
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
        // `acc`（Kahan 補償和の主項）・`acc_c`（補償項）でレーン内の
        // 二乗和を蓄積する（縮約精度契約。本ファイル冒頭コメント参照）。
        float acc = 0.0f;
        float acc_c = 0.0f;

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                rmsnorm_kahan_add(acc, acc_c, v.x * v.x);
                rmsnorm_kahan_add(acc, acc_c, v.y * v.y);
                rmsnorm_kahan_add(acc, acc_c, v.z * v.z);
                rmsnorm_kahan_add(acc, acc_c, v.w * v.w);
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                rmsnorm_kahan_add(acc, acc_c, v * v);
            }
        }

        float sum_sq = rmsnorm_reduce_sum_kahan(acc, acc_c);
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
