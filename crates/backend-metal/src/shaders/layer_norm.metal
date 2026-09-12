// LayerNorm 順伝播カーネル（イシュー #1596。`rmsnorm.metal`〈#604〉と
// 同じモジュール構成・縮約精度契約を踏襲する Metal 対応版）。
//
// 意味論: out = (x - mean(x, axis=-1)) * rsqrt(var(x, axis=-1) + eps) * w + b
// （has_weight == 0 の場合は w への乗算を、has_bias == 0 の場合は b への
// 加算をそれぞれスキップ）。分散は biased（÷N。`E[x^2]-mean^2` ではなく
// 二パス `Sigma(x-mean)^2/N` で計算する。`docs/norm-ops-design.md`）。
//
// FMA 契約: `rmsnorm.metal` と同じく `fma()` を明示使用しない単純な
// 加減乗算のみ（正規化統計の縮約精度契約が優先するため。`.claude/rules/
// coding-rust.md`）。コンパイルオプションは `pipeline::compile_options()`
// を適用する。
//
// 縮約精度契約（正規化統計の f64 アキュムレータ統一。イシュー #1102）:
// Apple GPU の MSL は `double` を持たないため、`rmsnorm.metal` と同じ
// Neumaier 改良版 Kahan 補償和 + scale/ssq 方式（分散の二乗和のみ。平均
// は単純合計のため素の Neumaier 補償和で十分——`f32` の表現範囲を超える
// ほどの入力規模でも合計自体は有限に収まる実用域を想定する）を適用する。
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

// 32 レーンの `(sum, comp)`（Neumaier 補償和の状態）を 5 段 butterfly で
// all-reduce する。単純合計のため `rmsnorm_ssq_combine` のような
// scale 併用は不要——`simd_shuffle_xor` で得た相手レーンの `sum`／`comp`
// を素の Neumaier 加算で取り込むだけで全 32 レーンに正しい合計が伝播する
// （加算の結合則・交換則により、この butterfly パターンは二重カウント
// なしに全要素をちょうど 1 回ずつ合成する）。
inline void ln_reduce_sum(thread float& sum, thread float& comp) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_sum = simd_shuffle_xor(sum, offset);
        float other_comp = simd_shuffle_xor(comp, offset);
        ln_kahan_add(sum, comp, other_sum);
        ln_kahan_add(sum, comp, other_comp);
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

        // パス 1: 平均（単純な Neumaier 補償和。冒頭コメント参照）。
        float sum = 0.0f;
        float sum_c = 0.0f;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            ln_kahan_add(sum, sum_c, x[row_base + idx]);
        }
        ln_reduce_sum(sum, sum_c);
        float mean = (sum + sum_c) * inv_n;

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
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            float v = x[row_base + idx];
            float xhat = (v - mean) * rstd;
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float bv = (has_bias != 0) ? b[idx] : 0.0f;
            out[row_base + idx] = xhat * wv + bv;
        }
    }
}
