// LayerNorm 順伝播カーネル（イシュー #1596。`rmsnorm.metal`〈#604〉と
// 同じモジュール構成・縮約精度契約を踏襲する Metal 対応版）。
//
// 意味論: out = (x - mean(x, axis=-1)) * rsqrt(var(x, axis=-1) + eps) * w + b
// （has_weight == 0 の場合は w への乗算を、has_bias == 0 の場合は b への
// 加算をそれぞれスキップ）。分散は biased（÷N。`E[x^2]-mean^2` ではなく
// 二パス `Sigma(x-mean)^2/N` で計算する。`docs/norm-ops-design.md`）。
//
// FMA 契約: 正規化統計の縮約（平均・分散）自体は下記の縮約精度契約が
// 優先し `fma()` を使わない単純な加減算のみだが、**affine（`x̂·w+b`）は
// CUDA カーネル（`fmaf`）・CPU/ホスト参照実装（`f32::mul_add`）と揃える
// ため `fma()` で明示的に融合する**（`.claude/rules/coding-rust.md` の
// FMA 契約統一。codex-review 指摘）。コンパイルオプションは
// `pipeline::compile_options()` を適用する。
//
// 縮約精度契約（正規化統計の f64 アキュムレータ統一。イシュー #1102）:
// Apple GPU の MSL は `double` を持たないため、`rmsnorm.metal` と同じ
// Neumaier 改良版 Kahan 補償和 + scale/ssq 方式（分散の二乗和）を適用
// する。**平均は Welford オンライン平均**（`ln_welford_merge`／
// `ln_reduce_mean`。単純合計〈Neumaier 補償和〉では `f32` の表現範囲を
// 超える有限入力〈例 `[2e38, 2e38]`〉で合計自体が overflow して `mean`
// が `NaN` 化するため、各更新値が常に入力値域に収まり中間 overflow が
// 起きない Welford 方式へ変更した。分散側の scale/ssq は `(x-mean)` の
// 二乗和 overflow のみを救済し平均側の overflow は救済できない。
// codex-review 指摘）。
// `rmsnorm.metal` の同名ヘルパーとの重複実装だが、Metal ソースは
// `newLibraryWithSource` で個別ファイル単位にコンパイルされ翻訳単位を
// 共有できないため、`ln_` 接頭辞を付けた本ファイル内で独立に定義する
// （意図的な重複。`rmsnorm.metal` のアルゴリズム自体は変更しない）。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）固定・persistent threadgroup
// 方式（`for (row = tg_id; row < rows; row += grid_size)`）・reduction は
// 5 段 butterfly（`simd_shuffle_xor` 幅 16/8/4/2/1）。いずれも
// `rmsnorm.metal` と同じ設計（`docs/backend-metal-morton-mapping-decision.md`
// と整合）。
//
// **rmsnorm.metal との差分**: 常に「2 パス」（device メモリを再読。
// threadgroup memory 不使用）とし、`rmsnorm_f32_onepass` に相当する
// threadgroup memory キャッシュ経路は持たない（LayerNorm は平均・分散の
// 2 回の縮約〈mean → var〉が必要で、x を 2 回読む構造が RMSNorm より
// 複雑になるため、本イシュー時点では単純な 3 パス〈mean → var → 書き
// 出し〉構成のみを実装する。性能上の onepass 化は後続課題として
// `docs/norm-ops-design.md` に記録する）。
//
// REQ-8 境界検査: ベクトル化ロードは行わず（`rmsnorm.metal` の
// `float4` 経路に相当する最適化は後続課題）、ループ添字は `ulong`
// （`row_base`）で宣言し `rows * hidden` の乗算オーバーフローを避ける。

#include <metal_stdlib>
using namespace metal;

constant uint LAYER_NORM_SIMD_WIDTH = 32u;

// Neumaier 改良版 Kahan 補償和の 1 ステップ（`rmsnorm.metal::
// rmsnorm_kahan_add` と同一アルゴリズム。本ファイル内で独立定義する
// 理由は冒頭コメント参照）。
inline void ln_kahan_add(thread float& sum, thread float& comp, float value) {
    float t = sum + value;
    if (fabs(sum) >= fabs(value)) {
        comp += (sum - t) + value;
    } else {
        comp += (value - t) + sum;
    }
    sum = t;
}

// 平均を Welford オンライン平均（`mean_{k} = mean_{k-1} + (x_k -
// mean_{k-1})/k`）で計算する（overflow-safe。codex-review 指摘:
// 単純合計〈旧 `ln_kahan_add`／`ln_reduce_sum` の Neumaier 補償和〉は
// `f32` の表現範囲を超える有限入力〈例 `[2e38, 2e38]`〉で合計自体が
// `inf` になり `mean` が `NaN` 化する。Welford は各更新値が常に入力
// 値域に収まるため中間 overflow が起きない）。分散の scale/ssq 方式
// とは独立の対策——分散側は `(x-mean)` の二乗和 overflow を救済する
// のみで、平均そのものの overflow は救済できない）。
inline void ln_welford_merge(thread float& meanA, thread float& countA,
                              float meanB, float countB) {
    float countAB = countA + countB;
    if (countAB > 0.0f) {
        float delta = meanB - meanA;
        meanA = meanA + delta * (countB / countAB);
    }
    countA = countAB;
}

// 32 レーンの `(mean, count)`（Welford 状態）を 5 段 butterfly で
// all-reduce する（Chan の並列合成公式。`ln_ssq_combine` と同じ
// butterfly パターン。加算の結合則・交換則と異なり Welford の合成は
// 非可換だが、`ln_welford_merge` は対称〈`countA`／`countB` を対等に
// 扱う〉ため任意の合成順序で正しい全体平均に収束する）。
inline void ln_reduce_mean(thread float& mean, thread float& count) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_mean = simd_shuffle_xor(mean, offset);
        float other_count = simd_shuffle_xor(count, offset);
        ln_welford_merge(mean, count, other_mean, other_count);
    }
}

// scale/ssq 方式（LAPACK SLASSQ 系）による overflow-safe な二乗和蓄積の
// 1 ステップ（`rmsnorm.metal::rmsnorm_ssq_add` と同一アルゴリズム・同一
// NaN／inf 伝播契約。本ファイル内で独立定義する理由は冒頭コメント参照）。
inline void ln_ssq_add(thread float& scale, thread float& ssq, thread float& comp, float a) {
    if (isnan(a) || isnan(ssq) || isnan(scale)) {
        scale = 1.0f;
        ssq = NAN;
        comp = 0.0f;
        return;
    }
    if (isinf(scale) && isinf(a)) {
        ln_kahan_add(ssq, comp, 1.0f);
        return;
    }
    if (a > scale) {
        if (scale > 0.0f) {
            float ratio = scale / a;
            float r2 = ratio * ratio;
            ssq *= r2;
            comp *= r2;
        }
        scale = a;
        ln_kahan_add(ssq, comp, 1.0f);
    } else if (scale > 0.0f) {
        float ratio = a / scale;
        ln_kahan_add(ssq, comp, ratio * ratio);
    }
}

// 2 つの scale/ssq 状態を結合する（`rmsnorm.metal::rmsnorm_ssq_combine`
// と同一アルゴリズム）。
inline void ln_ssq_combine(thread float& scale, thread float& ssq, thread float& comp,
                            float other_scale, float other_ssq, float other_comp) {
    if (isnan(ssq) || isnan(other_ssq)) {
        scale = 1.0f;
        ssq = NAN;
        comp = 0.0f;
        return;
    }
    if (other_scale == 0.0f) {
        return;
    }
    if (scale == 0.0f) {
        scale = other_scale;
        ssq = other_ssq;
        comp = other_comp;
        return;
    }
    if (isinf(scale) && isinf(other_scale)) {
        ssq = 1.0f;
        comp = 0.0f;
        return;
    }
    if (scale >= other_scale) {
        float ratio = other_scale / scale;
        float r2 = ratio * ratio;
        ln_kahan_add(ssq, comp, other_ssq * r2);
        ln_kahan_add(ssq, comp, other_comp * r2);
    } else {
        float ratio = scale / other_scale;
        float r2 = ratio * ratio;
        float new_ssq = other_ssq;
        float new_comp = other_comp;
        ln_kahan_add(new_ssq, new_comp, ssq * r2);
        ln_kahan_add(new_ssq, new_comp, comp * r2);
        scale = other_scale;
        ssq = new_ssq;
        comp = new_comp;
    }
}

// 32 レーン全体の scale/ssq 状態を 5 段 butterfly で reduction する
// （`rmsnorm.metal::rmsnorm_reduce_ssq` と同一）。
inline void ln_reduce_ssq(thread float& scale, thread float& ssq, thread float& comp) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_scale = simd_shuffle_xor(scale, offset);
        float other_ssq = simd_shuffle_xor(ssq, offset);
        float other_comp = simd_shuffle_xor(comp, offset);
        ln_ssq_combine(scale, ssq, comp, other_scale, other_ssq, other_comp);
    }
}

// `scale`／`ssq`／`comp`（`ln_reduce_ssq` 適用後。`eps` の疑似要素折り
// 込み前）と `eps`・`inv_n`（`= 1/hidden`）から
// `rstd = 1/sqrt(var + eps)`（`var = sum((x-mean)^2)/hidden`）を
// overflow-safe に導出する（`rmsnorm.metal::rmsnorm_finalize_rstd` と
// 同一の疑似要素トリック: `sqrt(eps*n) = sqrt(eps)*sqrt(n)` で中間
// overflow を回避）。
inline float ln_finalize_rstd(float scale, float ssq, float comp, float eps, float inv_n) {
    float n = 1.0f / inv_n;
    float eps_elem = sqrt(eps) * sqrt(n);
    ln_ssq_add(scale, ssq, comp, eps_elem);
    return 1.0f / (scale * sqrt((ssq + comp) * inv_n));
}

kernel void layer_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant float& inv_n [[buffer(7)]],
    constant int& has_weight [[buffer(8)]],
    constant int& has_bias [[buffer(9)]],
    constant uint& grid_size [[buffer(10)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;

        // パス 1: 平均（Welford オンライン平均。overflow-safe。上記
        // `ln_welford_merge`／`ln_reduce_mean` 冒頭コメント参照）。
        float lane_mean = 0.0f;
        float lane_count = 0.0f;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            lane_count += 1.0f;
            float delta = x[row_base + idx] - lane_mean;
            lane_mean += delta / lane_count;
        }
        ln_reduce_mean(lane_mean, lane_count);
        float mean = lane_mean;

        // パス 2: 分散（`(x-mean)` に対する scale/ssq 方式二乗和）。
        float scale = 0.0f;
        float ssq = 0.0f;
        float ssq_c = 0.0f;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            float d = x[row_base + idx] - mean;
            ln_ssq_add(scale, ssq, ssq_c, fabs(d));
        }
        ln_reduce_ssq(scale, ssq, ssq_c);
        float rstd = ln_finalize_rstd(scale, ssq, ssq_c, eps, inv_n);

        // パス 3: 書き出し（device メモリを再読。threadgroup memory
        // 不使用の 2 パス経路——`rmsnorm_f32_twopass` と同じ構成）。
        // affine は CUDA カーネルの既定 FMA contraction・CPU/ホスト参照
        // 実装の `f32::mul_add` と揃えるため `fma()` で明示的に融合する
        // （`.claude/rules/coding-rust.md` の FMA 契約統一。codex-review
        // 指摘）。
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            float v = x[row_base + idx];
            float xhat = (v - mean) * rstd;
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float bv = (has_bias != 0) ? b[idx] : 0.0f;
            out[row_base + idx] = fma(xhat, wv, bv);
        }
    }
}
