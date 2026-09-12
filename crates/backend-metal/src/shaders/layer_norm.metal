// LayerNorm 順伝播カーネル（イシュー #1596。`rmsnorm.metal`〈#604〉と
// 同じモジュール構成を踏襲する Metal 対応版）。
//
// 意味論: out = (x - mean(x, axis=-1)) * rsqrt(var(x, axis=-1) + eps) * w + b
// （has_weight == 0 の場合は w への乗算を、has_bias == 0 の場合は b への
// 加算をそれぞれスキップ）。分散は biased（÷N。`E[x^2]-mean^2` ではなく
// 二パス `Sigma(x-mean)^2/N` で計算する。`docs/norm-ops-design.md`）。
//
// FMA 契約: 正規化統計の縮約（平均・分散）自体は下記の縮約精度契約が
// 優先し `fma()` を使わない単純な加減算のみだが、affine（`x̂·w+b`）は
// 最終段を `fma()` で明示的に融合する（`.claude/rules/coding-rust.md`
// の FMA 契約統一。codex-review 指摘）。**PR #1671 スレッド 2 件目の
// 是正（パス 4 コメント参照）により、`weight` を正規化係数の除算より
// 先に乗じる分解（`fma(dev*wv/scale, norm, bv)`）へ変更済み**——`eps`
// が `x` を極端に上回る行で中間値が subnormal に潰れ Apple GPU が
// flush-to-zero する問題への対応。CUDA カーネル（`fmaf`）・CPU/ホスト
// 参照実装（`xhat.mul_add(w,b)`）とは融合の分解点が異なるため bit
// 一致は保証しない（LayerNorm の CPU-Metal 数値一致は REQ-2 統一複合
// 判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉であり bit-exact
// 契約ではない）。コンパイルオプションは `pipeline::compile_options()`
// を適用する。
//
// 縮約精度契約（正規化統計の f64 アキュムレータ統一。イシュー #1102・
// #1596）: `.claude/rules/coding-rust.md`「正規化統計の二乗和」節が定める
// とおり Apple GPU の MSL は `double` を持たないため、Metal 実装形は
// Neumaier 改良版 Kahan 補償和 + scale/ssq 方式（分散側。`rmsnorm.metal`
// と同じ LAPACK SLASSQ 系 overflow-safe 二乗和）を正とする（同節の対象は
// 「勾配の長軸縮約」〈#1566・soft_f64 の bit 完全一致方式〉とは別軸であり、
// 正規化統計側はこの節が定める補償和方式のまま）。
//
// **平均計算（codex-review 指摘。当初の Welford オンライン平均を破棄した
// 経緯）**: Welford（`mean_k = mean_{k-1} + (x_k-mean_{k-1})/k`）は各更新値
// が入力値域に収まるため overflow-safe だが、butterfly merge の
// `meanA + delta*(countB/countAB)` が毎ステップ `f32` へ丸められるため
// `[16777216, 16777218]`（真の平均 `16777217` が `f32` で表現不能）で
// 丸め誤差が生じ、CPU/CUDA（`f64` のまま偏差計算まで保持）と符号が食い違う
// 結果（期待 `[-1,1]` に対し `[0, 1.4142]`）を生んだ。さらに Welford の
// `meanB - meanA` 自体、入力 `[2^38, -2^38]` のような遠い有限値の差が
// `f32` の表現範囲（`|x| <= f32::MAX ≈ 3.4e38`）を超え `±inf` へ overflow
// する問題もあった。
//
// **採用した設計（行内 2 の冪スケーリング + 2 段補償和）**: 行の
// `maxabs = max(|x_i|)` を求め、`s = 2^floor(log2(maxabs))`（`maxabs` 以下
// の最大の 2 の冪。ビット直接構成——`as_type<uint>`/`as_type<float>` で
// 指数フィールドを直接読み書きする。`crate::soft_f64` の bit 演算方針と
// 同じスタイル）を求める。以降のすべての縮約は `x_i/s`（2 の冪除算は
// 丸め無しの厳密演算）という**比スケール領域**（`|x_i/s| < 2`）で行う。
// 比スケール領域では総和が `O(hidden)` に収まり overflow の心配がなく
// （元の `x` がどれほど巨大でも比は高々 `[-2,2)`）、Neumaier 改良版 Kahan
// 補償和（`ln_kahan_add`／`ln_reduce_kahan`）だけで `f64` 相当の精度が
// 得られる（`f32` へ丸めるのは各段の加算内部のみで、桁落ち成分
// `comp` に残差を保持し続けるため）。
//
// 平均は比スケール総和 `(sum, comp)` を `hidden` で厳密除算してさらに
// doubled-float（FMA によるロスレス除算誤差抽出。Dekker の手法）へ
// 拡張した `(mean_hi, mean_lo)` として保持し、**偏差計算まで `f32`
// 単一値へ丸めず 2 語のまま保持する**（codex-review 指摘: 平均を
// 偏差計算前に丸めると丸め誤差がそのまま偏差へ伝播する）。除算は
// ホストから渡される `inv_n`〈事前丸め済み `1/hidden`〉への乗算では
// なく `hidden` 自体（`(float)hidden` は `hidden <= 2^24` の範囲で
// exact。この上限は暗黙の前提ではなく、ホスト側
// `layer_norm.rs::validate_hidden_exact_f32` が起動前に fail-closed で
// 検査する——codex-review 指摘・PR #1671 スレッド 1 件目: `hidden >
// 2^24`〈例 `16777217`〉では `(float)hidden` が最近接偶数丸めで
// `16777216` へ丸められ、真の除数とのずれが `mean_lo` へ残存し出力へ
// 伝播しうるため、この軸長は起動前に明示的に拒否する）への直接除算で
// 行う（`inv_n` 経由だと `mean_hi+mean_lo` が
// 表す値が真の平均 `sum/hidden` ではなく `sum*inv_n_rounded` になり、
// `inv_n` 自身の丸め誤差が `mean_lo` へ残存する。全要素が同一値の行
// では本来 0 になるべき偏差にこの残存誤差が現れ、後段で `eps` 由来の
// 極小 `scale` により増幅されていた——codex-review 指摘・パス 2 の
// 実装コメント参照）。偏差 `dev = (x_i/s - mean_hi) -
// mean_lo` も比スケール領域内（`O(1)`）に収まるため overflow しない
// （`[2^38, -2^38] × 999` のような偏差自体が `f32::MAX` を超える入力でも、
// 比スケール領域では `dev` が有界に保たれる——真の偏差を **一度も
// 元スケールへ戻さない**のが要点。元スケールへ戻すと真値
// `dev_actual ≈ 3.996e38` は `f32` で表現不能なため、比スケール領域内で
// 完結させる必要がある）。
//
// 分散は比スケール偏差 `dev` に対する `rmsnorm.metal` と同型の scale/ssq
// 方式（LAPACK SLASSQ 系）で求める。`eps` は比スケール領域の等価量
// `eps/s^2` として `eps_elem = sqrt(eps)*sqrt(n)/s`（分散側の scale/ssq が
// 使う「疑似要素」トリックをそのまま流用し、`s^2` で先に割らず `sqrt(eps)`
// と `s` をそれぞれ独立に扱うことで `eps/s^2` 自体の overflow/underflow を
// 避ける）という擬似要素として同じ scale/ssq 蓄積へ折り込む。
//
// **`row_scale` の eps 対応拡張・weight 先乗算（codex-review 指摘・
// PR #1671 スレッド 2 件目の是正。2 段構え）**: `row_scale` を `x` の
// `maxabs` のみから決めると、`eps` が `x` に比べて極端に大きい行
// （例 `x=[1e-20,-1e-20], eps=1e38`）で `eps_elem =
// sqrt(eps)*sqrt(n)/row_scale` 自体が `f32` の表現範囲（約 `3.4e38`）を
// 超えて `+inf` になり、`ln_ssq_add` が `scale` を文字通り `+inf` へ
// 設定してしまう。以降のすべての `dev/scale`（`x` 由来の実要素）が
// `有限値/inf = 0` へ潰れ、期待される微小だが非ゼロな出力（本例では
// `weight=1e38` と合わせて `out≈[0.1,-0.1]`）が得られず `[0, 0]` に
// なっていた（`eps_elem` の真の数学的値〈約 `2.08e39`〉自体が `f32` で
// 表現不能なため、計算順序の工夫では解決できない）。
//
// **1 段目（`row_scale` 側）**: `row_scale` を「`x` の `maxabs` 由来の
// 2 の冪」と「`eps` 側疑似要素 `sqrt(eps)*sqrt(hidden_f)` を安全マージン
// だけ右シフトした 2 の冪（`LN_EPS_ELEM_SAFE_SHIFT`。`eps_elem` を
// `[1,2)` へ収める「正準」な選び方ではなく、overflow を避けるのに
// **必要最小限**の引き上げに留める）」の**大きい方**に選び直す（下記
// パス 1 参照）。「正準」な選び方（`sqrt(eps)` の 2 の冪をそのまま
// 採用）を最初に試したところ、`row_scale` が過大になり `x` 自身の比
// `x_i/row_scale` が `f32` の正規化下限（約 `1.18e-38`）を割り込む
// subnormal になり、Apple GPU 実機がこれを flush-to-zero して依然
// `[0, 0]` を返すことを実機実測で確認した（必要最小限の引き上げに
// 変更後は解消）。`eps` 側も `ln_pow2_scale_from_maxabs` を経由した
// 2 の冪の `ldexp`（2 の冪同士の乗除のため丸め無し）であるため、
// `row_scale` は常に厳密な 2 の冪のまま保たれる（除算が丸め無しの
// 厳密演算という前提は崩れない）。`x` が優越する通常ケース（`eps` が
// `x` に比べ十分小さい）では `eps_pow2_scale <= x_pow2_scale` となり
// `row_scale` は従来どおり `x_pow2_scale` のまま変化しない（既存の大
// `maxabs` 系テスト〈`layer_norm_deviation_overflow_case_stays_finite`
// 等〉の挙動は不変）。
//
// **2 段目（パス 4・weight 先乗算）**: 1 段目だけでは、`eps` が `x` を
// 極端に上回る行で `dev/scale` 自体が真に subnormal（本例で
// `dev/scale≈7e-40`。数学的に正しい中間値であり計算順序の誤りではない）
// になる場合が残る。これは `weight` によって最終的に正常範囲へ戻る
// べき値であるため、`weight` を `scale` 除算より**先に**乗じることで
// （`pre = dev*wv` を先に求めてから `pre/scale`）中間値が正常範囲に
// 留まるようにする（詳細はパス 4 のコメント参照）。
//
// **最終正規化係数は 1 つの逆数として合成しない**（codex-review が示唆
// した通り、退化ケース——全要素が同一の巨大値の行〈例 `[2e38, 2e38]`〉
// など——では実分散が 0 で `eps` 疑似要素だけが ssq の `scale` を極端に
// 小さい値にしうるため、`1/(scale*sqrt(...))` を単独の中間値として計算
// すると `f32::MAX` を超えて overflow しうる）。代わりに要素ごとに
// `xhat = (dev_i / scale) * (1/sqrt((ssq+comp)*inv_n))` の順で計算する
// （`dev_i` が 0 の退化ケースでは `0/scale = 0` が exact に成立し
// overflow しない。`scale`・`ssq`・`comp` は比スケール領域の
// scale/ssq 蓄積結果であり、比スケールの `s` は分子・分母で相殺して
// 最終式には現れない——`docs/norm-ops-design.md` 参照）。
//
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
// **rmsnorm.metal との差分**: 常に「4 パス」（device メモリを再読。
// threadgroup memory 不使用）とし、`rmsnorm_f32_onepass` に相当する
// threadgroup memory キャッシュ経路は持たない（LayerNorm は行の
// `maxabs`・平均・分散・書き出しの計 4 回 `x` を読む構成のみを実装する。
// 性能上の onepass 化は後続課題として `docs/norm-ops-design.md` に
// 記録する）。
//
// REQ-8 境界検査: ベクトル化ロードは行わず（`rmsnorm.metal` の
// `float4` 経路に相当する最適化は後続課題）、ループ添字は `ulong`
// （`row_base`）で宣言し `rows * hidden` の乗算オーバーフローを避ける。

#include <metal_stdlib>
using namespace metal;

constant uint LAYER_NORM_SIMD_WIDTH = 32u;

// `row_scale` の eps 対応拡張（パス 1）で `eps` 側疑似要素スケールを
// 右シフトする量。`f32` の表現上限（指数 `127`）から十分な余裕
// （`eps_elem` を `[2^100, 2^101)` 程度に収め、後続の scale/ssq 蓄積・
// `sqrt` 等の演算でも overflow しない安全マージン）を残しつつ、`x` 側の
// 比 `x_i/row_scale` を可能な限り正規化範囲内に保つ（`row_scale` の
// 引き上げ幅を最小限にする）よう選んだ値。冒頭コメント「`row_scale` の
// eps 対応拡張」参照。
constant int LN_EPS_ELEM_SAFE_SHIFT = 100;

// Neumaier 改良版 Kahan 補償和の 1 ステップ（`rmsnorm.metal::
// rmsnorm_kahan_add` と同一アルゴリズム。本ファイル内で独立定義する
// 理由は冒頭コメント参照）。NaN 入力は通常の IEEE754 加減算を通じて
// 自然に `sum`／`comp` へ伝播する（`isnan` 分岐を要さない。`ln_ssq_add`
// が明示 `isnan` 分岐を必要とするのは scale/ssq の比較ベースの
// リスケール判定〈`a > scale`〉が NaN を静かに無視しうるためで、本関数
// は純粋な加減算のみのため同じ問題を持たない）。
inline void ln_kahan_add(thread float& sum, thread float& comp, float value) {
    float t = sum + value;
    if (fabs(sum) >= fabs(value)) {
        comp += (sum - t) + value;
    } else {
        comp += (value - t) + sum;
    }
    sum = t;
}

// 2 レーン分の Neumaier 補償和状態 `(sum, comp)` を 1 つに統合する
// （他方の `sum`・`comp` を順に取り込むだけの単純な合成。加算の結合則・
// 交換則に依存するが `ln_kahan_add` 自体が誤差を追跡し続けるため合成
// 順序に依らず妥当な結果へ収束する）。
inline void ln_kahan_merge(thread float& sum, thread float& comp, float other_sum, float other_comp) {
    ln_kahan_add(sum, comp, other_sum);
    ln_kahan_add(sum, comp, other_comp);
}

// 32 レーンの `(sum, comp)` を 5 段 butterfly で all-reduce する。
inline void ln_reduce_kahan(thread float& sum, thread float& comp) {
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_sum = simd_shuffle_xor(sum, offset);
        float other_comp = simd_shuffle_xor(comp, offset);
        ln_kahan_merge(sum, comp, other_sum, other_comp);
    }
}

// 行の `maxabs = max(|x_i|)` から「`maxabs` 以下の最大の 2 の冪」`s` を
// 直接ビット構成する（`crate::soft_f64` と同じ bit 直接操作スタイル。
// 2 の冪除算は丸め無しの厳密演算のため、後続の比スケール縮約が
// `f32` の丸め誤差を追加で持ち込まない）。`maxabs <= 0.0`（全要素 0）は
// 任意の正の値でよいため `0.5` を返す（全要素比が `0` になり後続の
// 偏差・分散も自然に `0` へ収束する）。`maxabs` が subnormal
// （指数フィールド 0）の場合は最小正規化値 `2^-126` へフォールバックする
// （この極small領域は本 PR のテスト対象外。有限かつ正であることのみ
// 保証すれば十分）。`fmax` ベースの `maxabs` 縮約自体は NaN を無視する
// （IEEE754 `fmax` の仕様）ため、行に NaN が含まれる場合 `maxabs` は
// 残りの非 NaN 要素から決まるが、NaN 要素は `x_i/s` の除算で NaN の
// まま残り、後続の `ln_kahan_add`（純粋な加減算）を通じて総和全体・
// ひいては行全体の出力へ自然に伝播する（`ln_ssq_add` も独立に `isnan`
// を検査するため二重に安全）。
inline float ln_pow2_scale_from_maxabs(float maxabs) {
    if (maxabs <= 0.0f) {
        return 0.5f;
    }
    uint bits = as_type<uint>(maxabs);
    uint exp_field = (bits >> 23u) & 0xFFu;
    if (exp_field == 0u) {
        return as_type<float>(1u << 23); // 2^-126（最小正規化値）
    }
    return as_type<float>(exp_field << 23u); // 2^(exp_field-127) <= maxabs
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
        // `hidden` による厳密除算（`hidden <= 2^24` の範囲では
        // `(float)hidden` は exact 表現。ホスト側 `layer_norm.rs::
        // validate_hidden_exact_f32` が起動前に検査する）を平均計算
        // （パス 2）より前に確定させる。パス 1 の `eps` 側スケール
        // 計算（`sqrt(eps)*sqrt(hidden_f)`）でも使うため、ここで
        // 1 回だけ計算する。
        float hidden_f = (float)hidden;

        // パス 1: 行の 2 の冪スケール `row_scale`（`maxabs = max(|x_i|)`
        // から直接ビット構成した `x` 側スケールと、`sqrt(eps)` から同じ
        // 手法で構成した `eps` 側スケールの大きい方を採用する。冒頭
        // コメント「`row_scale` の eps 対応拡張」参照）。
        float lane_maxabs = 0.0f;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            lane_maxabs = fmax(lane_maxabs, fabs(x[row_base + idx]));
        }
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_maxabs = simd_shuffle_xor(lane_maxabs, offset);
            lane_maxabs = fmax(lane_maxabs, other_maxabs);
        }
        float x_pow2_scale = ln_pow2_scale_from_maxabs(lane_maxabs);
        // `eps` 側スケールは `sqrt(eps)*sqrt(hidden_f)` 自体（= パス 3
        // の `eps_elem` 分子。以下 `eps_pseudo_elem_scale`）の 2 の冪
        // ではなく、それを `LN_EPS_ELEM_SAFE_SHIFT`（`2^100`）だけ
        // 右シフトした**最小限の**値を採用する（`ldexp` は 2 の冪との
        // 厳密乗除のため丸め無し）。`eps_pseudo_elem_scale` の 2 の冪を
        // そのまま `row_scale` に採用する（`eps_elem` を `[1,2)` に
        // 収める「正準」な選び方）と、`eps` が `x` に比べて極端に
        // 大きい行（例 `x=[1e-20,-1e-20], eps=1e38`）で `row_scale` が
        // 過大になり、`x` 自身の比 `x_i/row_scale` が `f32` の
        // 正規化下限（約 `1.18e-38`）を割り込む subnormal になって
        // しまう。Apple GPU は subnormal な中間値を flush-to-zero
        // しうるため（実機実測で確認。codex-review 指摘・PR #1671
        // スレッド 2 件目「共通スケール抽出や指数分離を用い、中間値・
        // affine 適用まで有効精度を保つ」の対応）、`x` の比を正規化
        // 範囲内に保てる**必要最小限**の `row_scale` 引き上げに留める。
        // `eps_elem` は `row_scale` にこの引き上げを適用すると
        // `[2^100, 2^101)` 相当（`f32` の表現上限 `2^128` 付近から
        // 十分な余裕を持たせた値。以降の scale/ssq 蓄積・
        // `ln_ssq_combine` はいずれも `scale` 同士の比〈`[0,1]` に
        // 収まる〉のみを扱うため `scale` 自体がこの大きさでも
        // overflow しない）に収まり overflow しない。`x` 側が優越する
        // 通常ケース（`eps` が `x` に比べ十分小さい）では
        // `eps_pow2_scale <= x_pow2_scale` となり `row_scale` は従来
        // どおり `x_pow2_scale` のまま変化しない。
        float eps_pseudo_elem_scale = (eps > 0.0f) ? sqrt(eps) * sqrt(hidden_f) : 0.0f;
        float eps_pow2_scale = (eps_pseudo_elem_scale > 0.0f)
            ? ldexp(ln_pow2_scale_from_maxabs(eps_pseudo_elem_scale), -LN_EPS_ELEM_SAFE_SHIFT)
            : 0.0f;
        float row_scale = fmax(x_pow2_scale, eps_pow2_scale);

        // パス 2: 平均（比スケール領域 `x_i/row_scale` の Neumaier 補償
        // 総和 → `inv_n` 倍を Dekker の手法で doubled-float
        // `(mean_hi, mean_lo)` へ拡張。冒頭コメント参照）。
        float lane_sum = 0.0f;
        float lane_comp = 0.0f;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            float ratio = x[row_base + idx] / row_scale;
            ln_kahan_add(lane_sum, lane_comp, ratio);
        }
        ln_reduce_kahan(lane_sum, lane_comp);
        // `hidden` による厳密除算（`hidden <= 2^24` の実用範囲では
        // `(float)hidden` は exact 表現）を Dekker 型 div で
        // doubled-float `(mean_hi, mean_lo)` へ拡張する（codex-review
        // 指摘: ホストから渡される `inv_n`〈`1.0f32/hidden as f32`。
        // 既に 1 回丸め済み〉をそのまま乗算して Dekker 分割すると、
        // `mean_hi+mean_lo` が表す値は真の平均 `sum/hidden` ではなく
        // `sum*inv_n_rounded` になる。全要素が同一値の行では本来
        // 偏差が厳密に 0 になるべきだが、この差分〈`inv_n` 自身の
        // 丸め誤差由来〉が `mean_lo` に残存し、後段で `eps` 由来の
        // 極小 `scale` により 1000 倍規模へ増幅されていた（実測:
        // `x=[1024,1024,1024], eps=1e-5` で本来 0 のところ約
        // `-0.00965`）。`hidden` 自体による厳密除算に切り替えることで
        // `mean_hi+mean_lo` が `sum/hidden`〈厳密値〉の doubled-float
        // 表現になり、この増幅経路を根本から断つ）。`hidden_f` は
        // パス 1 冒頭で確定済み（`eps` 側スケール計算でも使うため）。
        float mean_hi = lane_sum / hidden_f;
        float mean_div_r = fma(-mean_hi, hidden_f, lane_sum);
        float mean_lo = mean_div_r / hidden_f + lane_comp / hidden_f;

        // パス 3: 分散（比スケール偏差 `dev = (x_i/row_scale - mean_hi) -
        // mean_lo` に対する scale/ssq 方式二乗和。`eps` は比スケール
        // 等価量の疑似要素として同じ蓄積へ折り込む。冒頭コメント参照）。
        float scale = 0.0f;
        float ssq = 0.0f;
        float ssq_c = 0.0f;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            float ratio = x[row_base + idx] / row_scale;
            float dev = (ratio - mean_hi) - mean_lo;
            ln_ssq_add(scale, ssq, ssq_c, fabs(dev));
        }
        ln_reduce_ssq(scale, ssq, ssq_c);
        // `eps_pseudo_elem_scale`（パス 1 で確定済み。`sqrt(eps)*
        // sqrt(hidden_f)`）を `row_scale` で割った比スケール疑似要素を
        // 二乗和へ折り込む。`row_scale` はパス 1 で `eps_pow2_scale`
        // （`eps_pseudo_elem_scale` を `LN_EPS_ELEM_SAFE_SHIFT` だけ
        // 右シフトした最小限のスケール）を考慮済みのため、`eps` が
        // `x` に比べ極端に大きい行でも `eps_elem` は `[2^100, 2^101)`
        // 程度に収まり overflow しない（冒頭コメント「`row_scale` の
        // eps 対応拡張」参照）。
        float eps_elem = eps_pseudo_elem_scale / row_scale;
        ln_ssq_add(scale, ssq, ssq_c, eps_elem);
        // `1/sqrt((ssq+ssq_c)*inv_n)` は通常オーダーの値（`eps` 疑似要素
        // により `ssq+ssq_c` が 0 になることはない）。`scale` 側の逆数は
        // 単独形成せず要素ごとに `dev_i / scale` として計算する
        // （冒頭コメント「最終正規化係数は 1 つの逆数として合成しない」）。
        float norm = 1.0f / sqrt((ssq + ssq_c) * inv_n);

        // パス 4: 書き出し（device メモリを再読）。affine の最終段
        // （正規化係数の乗算 + bias 加算）を `fma()` で明示的に融合
        // する（`.claude/rules/coding-rust.md` の FMA 契約統一。
        // codex-review 指摘）。`weight` の乗算をこの融合より前段
        // （`scale` 除算前）に置く理由は下記ループ内コメント参照
        // （冒頭コメント「FMA 契約」参照）。
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            float ratio = x[row_base + idx] / row_scale;
            float dev = (ratio - mean_hi) - mean_lo;
            // `dev == 0.0 && eps > 0.0` の場合のみ明示的に 0 を返す
            // （codex-review 指摘: 当初 `dev == 0.0` だけで分岐すると
            // `eps == 0` かつ真の分散も 0 の退化ケース〈例 `x=[1,1],
            // eps=0`〉で `scale` が数学的に厳密 0〈FTZ ではなく実際に
            // ゼロの mathematical scale〉になり、CPU/CUDA・ホスト参照
            // 実装が `0 * inf = NaN`（`rstd = 1/sqrt(var+eps) = 1/0 =
            // inf`）として返す NaN 伝播契約と食い違い Metal だけ `[0,
            // 0]` を返していた。`eps > 0.0` を条件に加えることで、この
            // ゼロ返却は「`scale` が `eps` 由来の極小疑似要素のみに
            // 由来し FTZ で潰れうる」退化ケース〈行の全要素が同一の
            // 巨大値・`eps > 0`。例 `[2e38, 2e38]`〉に限定される。
            // `eps == 0` かつ真の分散も 0 の場合は自然な `dev/scale =
            // 0/0 = NaN` 分岐へフォールバックし、参照実装と同じ NaN
            // 伝播契約を保つ。`eps > 0` かつ真の分散も 0 の通常ケース
            // （`scale` が subnormal 域でない。例 `x=[1024,1024,1024],
            // eps=1e-5`）では `dev` が厳密に 0 であるため、この分岐が
            // なくとも `0 / scale(非ゼロ) = 0` が exact に成立する
            // （分子が厳密 0 のため FTZ の影響を受けない）。
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float bv = (has_bias != 0) ? b[idx] : 0.0f;
            // **`weight` を `scale` 除算より先に乗じる（codex-review
            // 指摘・PR #1671 スレッド 2 件目「affine 適用まで有効精度を
            // 保つ」の対応）**: `eps` が `x` に比べて極端に大きい行では
            // `row_scale` の eps 対応拡張（パス 1）を適用しても
            // `dev/scale` 自体が真に `subnormal`（例 `x=[1e-20,
            // -1e-20], eps=1e38` で `dev/scale≈7e-40`）になりうる
            // （これは数学的に正しい中間値であり計算順序の誤りではない
            // ——真の `xhat` 自体が subnormal 級）。GPU が subnormal
            // 中間値を flush-to-zero する場合、`weight` が桁を戻す
            // 大きな値（例 `1e38`）であっても「先に `xhat`（重み適用前）
            // を subnormal として確定させてから乗じる」順序では `xhat`
            // 自体が 0 へ潰れて情報が失われる。`weight` を `dev` に
            // 先に乗じてから `scale` で割ることで（`pre = dev*wv` は
            // 通常範囲に収まる——本例では `pre≈1.37e29`）、最終除算
            // `pre/scale` も通常範囲（本例で `≈0.0707`）に収まり
            // subnormal を経由しない。`has_weight==0`（`wv=1.0`）の
            // 場合は `pre=dev` で従来と同値になり、通常ケースの挙動は
            // 不変。
            //
            // `weight` を先に乗じる都合上、正規化係数 `norm` の乗算と
            // `bias` の加算を `fma(pre/scale, norm, bv)` として融合する
            // （`xhat·w+b` を単一の `fma` にしていた是正前とは異なる
            // 分解だが、最終段を 1 回の融合演算にする「affine は
            // `fma()` で明示的に融合する」契約〈冒頭コメント「FMA
            // 契約」・`.claude/rules/coding-rust.md`〉は維持する。
            // CPU/CUDA 参照実装〈`xhat.mul_add(w,b)`〉とは分解が異なる
            // ため bit 一致は保証しないが、LayerNorm の CPU-Metal 数値
            // 一致は REQ-2 統一複合判定〈相対誤差 1e-3 未満 または
            // 絶対誤差 1e-5 未満〉であり bit-exact 契約ではない）。
            //
            // FTZ 対策のゼロ返却（既存分岐）は `scale` 自体が `eps`
            // 由来の極小疑似要素の flush-to-zero で厳密 0 になりうる
            // ケース向けであり、`pre/scale` の**除算を実行せず**
            // `bv` を直接返す短絡評価を維持する（除算してから `fma` に
            // 渡すと `scale` が flush 済みで `0.0f` の場合
            // `fma(0.0f/0.0f, norm, bv) = fma(NaN, norm, bv) = NaN` に
            // なってしまい、この分岐が意味を失う）。
            float pre = dev * wv;
            out[row_base + idx] =
                (eps > 0.0f && dev == 0.0f) ? bv : fma(pre / scale, norm, bv);
        }
    }
}
