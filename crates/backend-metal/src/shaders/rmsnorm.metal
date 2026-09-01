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
// **Neumaier 改良版 Kahan 補償和 + scale/ssq 方式（LAPACK SLASSQ 系の
// overflow-safe な二乗和アルゴリズム）**を「`f64` アキュムレータ相当」
// の実装形として適用する。
//
// **scale/ssq 方式が必須な理由（codex-review 指摘・PR #1120。二重の
// P1 是正）**: 当初は単純な Kahan 補償和のみを適用していたが、
// `v.x * v.x` を `f32` のまま先に計算する実装だったため、有限入力
// （例: `2e20f`）でも二乗が `f32` の表現範囲（最大約 3.4e38）を超えて
// `inf` になり、`inf - inf` の Kahan 補償計算で `NaN` が発生していた。
// CUDA・CPU は要素を `f64` へ昇格してから二乗するため有限値を保つ
// （`kernels_rmsnorm.rs` 冒頭コメント「精度契約」・`crates/backend-cpu/
// src/rmsnorm.rs` 冒頭コメント「縮約精度契約」参照）ため、この Metal 側
// の単純 Kahan 実装は意味論が他バックエンドと片側で割れていた。
// scale/ssq 方式は各要素の絶対値のうち最大値を `scale` として括り出し、
// 残りを `scale` に対する比の二乗（`(a/scale)^2`。常に `[0, 1]` に収まる）
// として `ssq` へ蓄積するため、`v.x * v.x` を直接計算せずに二乗和を
// `f32` の表現範囲内で安全に求められる（`rmsnorm_ssq_add`）。二乗和の
// **レーン内蓄積**（各レーンが担当する `hidden/32` 要素程度の逐次和。
// CUDA 側の「レーン内部分和を double 化」に対応）でこの方式を使い、
// **32 レーン間の butterfly 縮約**（CUDA 側の warp shuffle を double で
// 行うのに対応）では 2 つの `(scale, ssq)` 状態を結合する
// `rmsnorm_ssq_combine` を使う。いずれの蓄積・結合ステップも `ssq` 自体
// の加算には Neumaier 改良版 Kahan 補償和（`rmsnorm_kahan_add`）を併用
// し、precision を維持する。最終的な `rstd` の導出も `scale` の二乗を
// 明示的に計算しない形（`rstd = 1 / (scale * sqrt(ssq * inv_n))`）へ
// 整理し、`scale^2` 自体のオーバーフロー（`scale` が `2e20` 級の場合
// `scale^2` は `f32` 表現範囲外）を避ける。`eps` は最終段で別途加算する
// のではなく、`sqrt(eps * n)`（`n = 1/inv_n`）を追加の「疑似要素」として
// 同じ `rmsnorm_ssq_add` へ通すことで、実要素の二乗和と同じ overflow-safe
// な経路に統一する（`scale == 0`〈実要素が全て 0〉かつ `eps > 0` の
// ケースでも、この疑似要素が `scale` を `sqrt(eps * n)` へ更新するため、
// 特別分岐なしに従来どおり `rstd = 1/sqrt(eps)` へ帰着する）。
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

// scale/ssq 方式（LAPACK SLASSQ 系）による overflow-safe な二乗和蓄積の
// 1 ステップ。非負の値 `a`（呼び出し元が `fabs(v)` を渡す）を
// `(scale, ssq, comp)` へ取り込む: `a` が現在の `scale` を超えたら
// `scale` を更新し、既存の `ssq`（+ 補償項 `comp`）を新旧スケール比の
// 二乗でリスケールしてから `1.0` を Neumaier 加算する。それ以外は
// `(a/scale)^2` を Neumaier 加算する（`scale == 0` の初期状態は `a` を
// そのまま新しい `scale` として採用し `ssq = 1` になる。`a == 0` は
// 実質的に無視される——`a > scale` が偽かつ `scale > 0` が偽なら分岐に
// 入らず何も変化しない）。`sum(a_i^2) = scale^2 * (ssq + comp)` が不変量。
//
// **NaN 伝播（codex-review 指摘・PR #1120 2 件目）**: `a` が `NaN` の
// 場合、IEEE 754 の比較規則により `a > scale` は常に偽になる。`scale`
// が未だ `0.0f`（このレーンで最初の要素が `NaN` だった場合）だと
// `scale > 0.0f` も偽になり、いずれの分岐にも入らず `NaN` 入力が黙って
// 捨てられてしまう（CPU/CUDA の `f64` 逐次和は `NaN` を含む行全体が
// `NaN` になる意味論のため、これは意味論の不一致だった）。`isnan(a)`
// （または既に `ssq`／`scale` が汚染済み）を検出したら `ssq` を `NaN`
// へ確定し、`scale` を**有限の正値**（`1.0f`）へ強制する（`scale` を
// `0.0f` のままにすると `rmsnorm_ssq_combine` の `other_scale == 0.0f`
// 早期 return で汚染情報が握り潰されるため、`scale > 0.0f` を満たす値に
// 固定して以降のすべての結合・蓄積へ確実に伝播させる）。
inline void rmsnorm_ssq_add(thread float& scale, thread float& ssq, thread float& comp, float a) {
    if (isnan(a) || isnan(ssq) || isnan(scale)) {
        scale = 1.0f;
        ssq = NAN;
        comp = 0.0f;
        return;
    }
    if (isinf(scale) && isinf(a)) {
        // `scale`・`a` とも +inf（2 個目以降の inf 要素。codex-review・
        // Bugbot 指摘・PR #1120 の同根バグ）。通常のリスケール分岐は
        // `a > scale` が偽（inf > inf は偽）になり、else 分岐の
        // `ratio = a / scale = inf/inf` が `NaN` になって誤って `ssq` を
        // 汚染してしまう（呼び出し元は `fabs(v)` を渡すため `a` は
        // 常に非負であり `isinf(a)` は必ず `+inf` を意味する）。
        // 二乗和は inf 要素が何個あっても inf のままという不変量を保つ
        // ため、`scale` は `+inf` のまま維持し `ssq` のみ Neumaier 加算で
        // 更新する（リスケール計算自体は行わない）。
        rmsnorm_kahan_add(ssq, comp, 1.0f);
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
        rmsnorm_kahan_add(ssq, comp, 1.0f);
    } else if (scale > 0.0f) {
        float ratio = a / scale;
        rmsnorm_kahan_add(ssq, comp, ratio * ratio);
    }
}

// 2 つの scale/ssq 状態（`sum(a_i^2) = scale^2*(ssq+comp)` を満たす組）を
// 結合する（butterfly reduction のレーン間結合専用）。`scale` の大きい
// 側を採用し、小さい側の `(ssq+comp)` をスケール比の二乗で縮めてから
// Neumaier 加算で取り込む（`rmsnorm_ssq_add` の「リスケール」ステップと
// 同じ考え方をレーン間結合へ拡張したもの）。相手側 `scale` が 0（寄与
// なし）なら何もしない。
//
// **NaN 伝播**: どちらかの `ssq` が既に `NaN`（`rmsnorm_ssq_add` が検出
// 済み）なら結合結果も `NaN` にする（`rmsnorm_ssq_add` が `scale` を
// `1.0f` へ固定しているため、この分岐は `other_scale == 0.0f` の早期
// return より先に評価する必要がある）。
//
// **inf 同士の結合**（`scale` が両者とも `+inf`）: IEEE 754 の
// `inf + inf = inf` に倣い、結合結果も `+inf` のまま維持する。通常の
// リスケール分岐は `ratio = other_scale / scale` を計算するため
// `inf/inf = NaN` になり誤って `NaN` へ汚染してしまう特殊ケースであり、
// 明示的に分岐する。
inline void rmsnorm_ssq_combine(thread float& scale, thread float& ssq, thread float& comp,
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
        rmsnorm_kahan_add(ssq, comp, other_ssq * r2);
        rmsnorm_kahan_add(ssq, comp, other_comp * r2);
    } else {
        float ratio = scale / other_scale;
        float r2 = ratio * ratio;
        float new_ssq = other_ssq;
        float new_comp = other_comp;
        rmsnorm_kahan_add(new_ssq, new_comp, ssq * r2);
        rmsnorm_kahan_add(new_ssq, new_comp, comp * r2);
        scale = other_scale;
        ssq = new_ssq;
        comp = new_comp;
    }
}

// 32 レーン全体の scale/ssq 状態を 5 段 butterfly で reduction する
// （`f64` アキュムレータ相当の overflow-safe 二乗和。本ファイル冒頭
// コメント「縮約精度契約」参照）。各レーンの `(scale, ssq, comp)` を
// `simd_shuffle_xor` でレーン間交換しながら `rmsnorm_ssq_combine` で
// 合成する。全レーンが同じ状態を持つ状態で戻る（呼び出し元は `scale`・
// `ssq`・`comp` の最終値から `rstd` を導出する）。
inline void rmsnorm_reduce_ssq(thread float& scale, thread float& ssq, thread float& comp) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_scale = simd_shuffle_xor(scale, offset);
        float other_ssq = simd_shuffle_xor(ssq, offset);
        float other_comp = simd_shuffle_xor(comp, offset);
        rmsnorm_ssq_combine(scale, ssq, comp, other_scale, other_ssq, other_comp);
    }
}

// `scale`／`ssq`／`comp`（`rmsnorm_reduce_ssq` 適用後。`eps` の疑似要素
// 折り込み前）と `eps`・`inv_n`（`= 1/hidden`。`n = 1/inv_n`）から
// `rstd = 1/sqrt(sum(x^2)/n + eps)` を overflow-safe に導出する。`eps`
// を `sqrt(eps) * sqrt(n)` という追加の疑似要素として `rmsnorm_ssq_add`
// へ通してから、`rstd = 1/(scale * sqrt(ssq * inv_n))`（`scale` を二乗
// せず最後に 1 回だけ掛ける形）で計算する（本ファイル冒頭コメント「縮約
// 精度契約」参照。`scale^2` を明示的に計算しないため `scale` が極端に
// 大きい／小さい場合でも `f32` の表現範囲内で完結する）。
//
// **`eps * n` の中間 overflow 回避（codex-review 指摘・PR #1120 1 件目）**:
// 疑似要素は本来 `sqrt(eps * n)` だが、`eps` が `f32::MAX` 級・`n`
// （`hidden`）がある程度大きい場合、平方根を取る前の `eps * n` 自体が
// `f32` の表現範囲を超えて `inf` になりうる（最終的な平方根の結果自体は
// `f32` で表現可能な範囲内であっても）。`sqrt(eps * n) == sqrt(eps) *
// sqrt(n)`（数学的に同値な変形。両辺とも非負）という恒等式を使い、`eps`・
// `n` それぞれを個別に平方根を取ってから掛け合わせることで、この中間
// overflow を避ける（`eps`・`n` はいずれも個別には `f32` の表現範囲内の
// 有限値であるため、それぞれの平方根も表現範囲内に収まる）。
inline float rmsnorm_finalize_rstd(float scale, float ssq, float comp, float eps, float inv_n) {
    float n = 1.0f / inv_n;
    float eps_elem = sqrt(eps) * sqrt(n);
    rmsnorm_ssq_add(scale, ssq, comp, eps_elem);
    return 1.0f / (scale * sqrt((ssq + comp) * inv_n));
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
        // `scale`／`ssq`／`ssq_c` で overflow-safe な二乗和（scale/ssq
        // 方式 + Neumaier 補償和）をレーン内蓄積する（縮約精度契約。
        // 本ファイル冒頭コメント参照）。
        float scale = 0.0f;
        float ssq = 0.0f;
        float ssq_c = 0.0f;

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                smem[base] = v.x;
                smem[base + 1u] = v.y;
                smem[base + 2u] = v.z;
                smem[base + 3u] = v.w;
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.x));
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.y));
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.z));
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.w));
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                smem[idx] = v;
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v));
            }
        }

        // 各レーンが書いた `smem` は正規化フェーズで同じレーン自身のみが
        // 読み戻す（ストライドパターンが往復で一致するため）が、
        // コンパイラ最適化・命令並び替えに対する防御として明示的に
        // 可視性バリアを置く（CUDA `__syncwarp` 相当）。
        simdgroup_barrier(mem_flags::mem_threadgroup);

        rmsnorm_reduce_ssq(scale, ssq, ssq_c);
        float rstd = rmsnorm_finalize_rstd(scale, ssq, ssq_c, eps, inv_n);

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
        // `scale`／`ssq`／`ssq_c` で overflow-safe な二乗和（scale/ssq
        // 方式 + Neumaier 補償和）をレーン内蓄積する（縮約精度契約。
        // 本ファイル冒頭コメント参照）。
        float scale = 0.0f;
        float ssq = 0.0f;
        float ssq_c = 0.0f;

        if (hidden % 4u == 0u) {
            for (uint base = lane * 4u; base + 3u < hidden; base += RMSNORM_SIMD_WIDTH * 4u) {
                float4 v = float4(x[row_base + base], x[row_base + base + 1u],
                                   x[row_base + base + 2u], x[row_base + base + 3u]);
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.x));
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.y));
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.z));
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v.w));
            }
        } else {
            for (uint idx = lane; idx < hidden; idx += RMSNORM_SIMD_WIDTH) {
                float v = x[row_base + idx];
                rmsnorm_ssq_add(scale, ssq, ssq_c, fabs(v));
            }
        }

        rmsnorm_reduce_ssq(scale, ssq, ssq_c);
        float rstd = rmsnorm_finalize_rstd(scale, ssq, ssq_c, eps, inv_n);

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
