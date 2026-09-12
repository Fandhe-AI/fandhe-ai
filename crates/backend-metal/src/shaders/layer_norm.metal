// LayerNorm 順伝播カーネル（イシュー #1596。`rmsnorm.metal`〈#604〉と
// 同じモジュール構成を踏襲する Metal 対応版）。
//
// 意味論: out = (x - mean(x, axis=-1)) * rsqrt(var(x, axis=-1) + eps) * w + b
// （has_weight == 0 の場合は w への乗算を、has_bias == 0 の場合は b への
// 加算をそれぞれスキップ）。分散は biased（÷N。`E[x^2]-mean^2` ではなく
// 二パス `Sigma(x-mean)^2/N` で計算する。`docs/norm-ops-design.md`）。
//
// FMA 契約: `affine`（`x̂·w+b`）は CPU 参照実装（`crates/backend-cpu/src/
// layer_norm.rs::layer_norm_row` の `xhat.mul_add(wv, bv)`）・CUDA
// （`kernels_layer_norm.rs` の nvcc 既定 FMA contraction）と同じ
// **「`f32` の `xhat`・`weight`・`bias` に対する単一丸めの FMA」**
// （`.claude/rules/coding-rust.md` の FMA 契約統一）を実現する。
//
// **平坦な `float` の `fma()` 命令は使わない**（GPU 実機が入力側の
// subnormal を flush-to-zero しうるため。下記パス 3 コメント「affine
// を round-to-odd で計算する理由」参照）。かといって「soft-f64（`f64`
// 53bit 仮数）へ widen して `mul`＋`add` してから `narrow` で `f32` へ
// 戻す」素朴な二段階丸めも**単一丸めの FMA と一致しない**（PR #1671
// codex-review 指摘・イシュー #1596: `xhat=31/16`・
// `weight=f32::from_bits(0x7f042108)`・`bias=-1` のような、積の指数と
// 加数の指数が大きく乖離する入力で、`f64` への丸め〈1 回目〉が
// `f32` の桁の決定に必要な情報を握り潰し、続く `narrow`〈2 回目〉が
// 単一丸めの結果〈本例では有限の `f32::MAX`〉と異なる値〈`+inf`〉を
// 返す「二重丸め」が発生する）。
//
// **本ファイルの affine 実装（round-to-odd 経由）**: `ln_f64_mul` で
// 積を厳密に求めた（`f32` 同士の積は仮数 48bit 以内に収まり `f64` の
// 53bit 仮数へ丸め無しで厳密表現できる）後、通常の最近接偶数丸め加算
// `ln_f64_add` の代わりに **round-to-odd 丸め加算 `ln_f64_add_ro`**
// （下記定義。`crates/backend-metal/src/soft_f64.rs::
// add_f64_bits_round_to_odd` の逐語移植）で `bias` を加え、最後に
// 通常の `ln_f64_narrow`（最近接偶数丸め）で 1 回だけ `f32` へ丸める。
// Boldo–Melquiond（2008）の round-to-odd 二重丸め定理（中間精度 `p2` が
// 目的精度 `p1` に対し `p2 >= p1+2` を満たせば
// `RN_p1(RO_p2(x)) == RN_p1(x)`。ここでは `p2=53`〈`f64`〉・`p1=24`
// 〈`f32`〉で `53 >= 26` を満たす）により、この 2 段階（`ln_f64_add_ro`
// → `ln_f64_narrow`）は厳密値 `xhat*w+b` を `f32` へ直接単一丸めした
// 結果と一致する（`ln_f64_add_ro` の doc comment に数学的根拠を記載。
// `crates/backend-metal/src/soft_f64.rs::fma_f32_bits` がホスト側
// 逐語モデルで、上記反例を含むランダム 200 万組・`f32::mul_add` との
// bit 完全一致をユニットテストで検証する）。`xhat` 自体は本ファイルの
// soft-f64（下記）経由で導出するため CPU/CUDA と bit 一致はしない
// （REQ-2 統一複合判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉
// の範囲で一致させる。bit-exact 契約ではない）。
//
// **数値方式（soft-f64 経由の全面再設計。PR #1671 codex-review 指摘への
// 是正・イシュー #1596）**: 当初実装（行の 2 の冪スケール `row_scale`
// によるリスケール総和 + Neumaier 補償和 + scale/ssq 分散）は、次の
// 2 系統の反例で数値契約を満たせないことが判明した:
//
// 1. **行スケール除算での微小値消失**: `row_scale` は行の `maxabs`
//    （または `eps` 側疑似要素）の大きい方から選ぶ 2 の冪だが、同じ行に
//    「巨大な値」と「小さいが非ゼロな寄与を持つ値」が混在する場合
//    （例 `x=[1e38,-1e38,1e-4,-1e-4]`）、小さい要素の比
//    `x_i/row_scale` が `f32` の正規化下限を割り込む subnormal になり、
//    Apple GPU 実機の flush-to-zero（FTZ）で消える。`weight` の乗算
//    順序を適応的に選ぶ工夫（是正前の実装）は `xhat` 自体を計算した
//    **後**の話であり、この「`ratio` 計算の時点」での消失には無力
//    だった。
// 2. **正規化係数の丸め誤差が affine の相殺で増幅される**: 分散の
//    scale/ssq 状態 `(scale, ssq, comp)` は `f64` 相当の精度を保持
//    していたが、`norm = 1/sqrt((ssq+comp)*inv_n)` の 1 行で単一
//    `f32` へ丸めてから `sqrt`／逆数を取っていたため、この時点で
//    `f64` 相当の精度が失われていた。`weight` が極端に大きく `bias`
//    がほぼ相殺する行（例 `weight=1e8, bias≈-weight*sqrt(1.5)`）では、
//    この 1 ULP 未満の差が `weight` 倍に増幅され CPU（`f64` で `mean`・
//    `var`・`rstd` を保持してから 1 回だけ `f32` へ丸める）との差が
//    数値契約を超えた。
//
// 両方とも根本原因は同じ: **`f32` の限られた指数範囲・仮数精度に収まる
// よう値をリスケールする**という当初のアプローチ自体が、リスケール後の
// 値がさらに `f32` の範囲外（微小値の消失）や精度不足（丸め誤差の
// 増幅）を引き起こす二次被害を生む。この問題は「もっと工夫したリ
// スケール」では解消せず、**`f32` の範囲・精度そのものに依存しない
// 計算方式**が必要だった。
//
// **採用した設計**: MSL は `double` 型非対応だが、64bit 整数
// （`ulong`／`long`）による **IEEE 754 binary64 のソフトウェア
// エミュレーション**（`gemm.metal::bias_f64_*`〈イシュー #1566・
// PR #1659〉と同じ手法を本ファイル独自に拡張）で `widen`（`f32→f64`）・
// `add`／`sub`・`mul`・Newton-Raphson 法による `recip`／`rsqrt`
// （除算命令を使わず `mul`／`add` のみで構成）・`narrow`（`f64→f32`）を
// 実装し、**CPU 参照実装と同じアルゴリズム構造**（`mean`／`var`／
// `rstd` を `f64` 相当の精度で保持し、`xhat = (x-mean)*rstd` を確定
// した後に 1 回だけ `f32` へ丸める）を Metal 上で再現する。`f64` は
// `f32` の全域（正規化数・subnormal を問わず）を正規化数として表現
// できる指数範囲を持つため（`f32` の subnormal 最小値 `2^-149` の平方
// でも `2^-298` は `f64` の表現範囲〈最小 subnormal `2^-1074`〉に
// 楽々収まる）、`row_scale` のようなリスケールが一切不要になり、
// 上記 2 系統の反例はどちらも構造的に解消する（`x` の値がどれほど
// 巨大・微小でも、二乗しても `f64` の指数範囲内に収まるため）。
//
// **Newton-Raphson の精度契約**: `recip`／`rsqrt` は「最近接偶数丸め」
// で正確に丸める代わりに、`f32` 精度（約 24bit）の種から出発し
// 4 回の反復（1 回ごとに正しい桁数がほぼ倍加: 24→48→52…）で `f64` の
// 52bit 精度へ収束させる（厳密な correctly-rounded ではないが、収束後
// の相対誤差は `f64` の 1 ULP のごく僅かな定数倍に収まり、最終的に
// `f32` へ丸める際の桁の決定には十分な余裕〈約 28bit 分〉がある。
// `crates/backend-metal/src/soft_f64.rs` の同名関数がホスト側の
// 逐語モデルであり、収束精度をユニットテストで検証する）。種の抽出
// （仮数を `f32` 精度で近似し指数は厳密に分離合成する）も同モジュールの
// `extract_reduced_mantissa_and_exp`／`scale_pow2_f64_bits` と 1 対 1
// 対応する。
//
// **ホスト側の逐語モデル**: `crates/backend-metal/src/soft_f64.rs` の
// `widen_f32_bits`／`neg_f64_bits`／`sub_f64_bits`／`add_f64_bits`／
// `add_f64_bits_round_to_odd`／`mul_f64_bits`／`div_f64_bits`／
// `narrow_f64_bits`／`recip_newton_f64_bits`／`rsqrt_newton_f64_bits`
// （加えて affine 全体のホスト側逐語モデルは `fma_f32_bits`）が本ファイルの
// `ln_f64_*` 系関数と 1 対 1 に対応し、`f64` 実演算に対する bit 完全
// 一致（`widen`／`add`／`mul`／`div`／`narrow`）または収束精度
// （`recip`／`rsqrt`）をユニットテスト（Linux 実行可能）で網羅検証
// する。本ファイルを変更した場合は同モジュールも追従させること。
// `mean`／`var` は `div_f64_bits`／`ln_f64_div`（正しく丸めた除算）で
// 求める（codex-review 指摘の再設計。PR #1671・イシュー #1596。
// Newton 近似逆数との積 `sum * recip(hidden)` は一様行等の割り切れる
// ケースで 1 ULP 誤差が悪化するため用いない）。
//
// **縮約精度契約との関係**（`.claude/rules/coding-rust.md`「正規化統計の
// 二乗和」節）: 同節は Metal の `f64` 相当実装形として Neumaier 補償和 +
// scale/ssq 方式を挙げているが、本ファイルは上記の理由からより精度の
// 高い soft-f64 方式を採用する（同節が禁止しているわけではなく、
// 「`f64` 相当の精度を保つ」という契約自体は本方式でも満たす。むしろ
// 補償和方式では満たせなかった反例が本方式で解消する）。
//
// **総和の順序について**: 本カーネルは 32 レーン SIMD 並列 + butterfly
// reduction で総和する（下記）。CPU 参照実装は行内を逐次（index 順）に
// 加算するため、加算順序は一致しない。`f64` 相当（52bit 精度）の
// 加算は非結合性の影響が極めて小さく（各加算ステップの丸め誤差は
// 少なくとも `2^-52` 相対）、REQ-2 統一複合判定（相対誤差 1e-3 未満
// または絶対誤差 1e-5 未満）の範囲内では順序差は無視できる
// （CUDA 実装〈`kernels_layer_norm.rs`〉も `double` を使うが warp
// butterfly reduction で加算順序が CPU と異なり、同じ前提で運用
// 済み）。
//
// コンパイルオプションは `pipeline::compile_options()` を適用する。
//
// **平均計算で Welford を採用しない理由**: Welford オンライン平均
// （`mean_k = mean_{k-1} + (x_k-mean_{k-1})/k`）は各更新値を毎回
// `f32` へ丸めるため、`f64` 相当の soft-f64 総和よりも精度が劣る
// （当初この問題に対処するため Welford を試したが `f32` 丸め誤差が
// 残存する経緯があった。soft-f64 化によりこの制約自体が解消したため
// 単純な総和で十分）。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）固定・persistent threadgroup
// 方式（`for (row = tg_id; row < rows; row += grid_size)`）・reduction は
// 5 段 butterfly（`simd_shuffle_xor` 幅 16/8/4/2/1）。いずれも
// `rmsnorm.metal` と同じ設計（`docs/backend-metal-morton-mapping-decision.md`
// と整合）。
//
// **rmsnorm.metal との差分**: 常に「3 パス」（device メモリを再読。
// threadgroup memory 不使用。旧実装の `row_scale` 算出パスは不要に
// なったため 4 パス→3 パスへ削減）とし、`rmsnorm_f32_onepass` に相当
// する threadgroup memory キャッシュ経路は持たない。性能上の onepass
// 化は後続課題として `docs/norm-ops-design.md` に記録する。
//
// REQ-8 境界検査: ベクトル化ロードは行わず（`rmsnorm.metal` の
// `float4` 経路に相当する最適化は後続課題）、ループ添字は `ulong`
// （`row_base`）で宣言し `rows * hidden` の乗算オーバーフローを避ける。

#include <metal_stdlib>
using namespace metal;

constant uint LAYER_NORM_SIMD_WIDTH = 32u;

// ---- IEEE 754 binary64 のソフトウェアエミュレーション ----
// `crates/backend-metal/src/soft_f64.rs` の逐語移植（`u64`→`ulong`・
// `u32`→`uint`・`leading_zeros()`→`clz()`。冒頭コメント「ホスト側の
// 逐語モデル」参照）。定数は同モジュールの `F64_*`／`F32_*` と同値。
// `gemm.metal::bias_f64_*` と機能重複するが、MSL は `newLibraryWithSource`
// で個別ファイル単位にコンパイルされ翻訳単位を共有できないため、
// `ln_f64_` 接頭辞を付けた本ファイル内で独立に定義する（意図的な
// 重複。`gemm.metal` 側は `mul`／`recip`／`rsqrt` を持たないため一部
// 機能はこちらが上位互換）。

#define LN_F64_SIGN      0x8000000000000000ul
#define LN_F64_EXP_MASK  0x7FFul
#define LN_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define LN_F64_QNAN      0x7FF8000000000000ul
#define LN_F64_INF       0x7FF0000000000000ul
#define LN_F32_QNAN      0x7FC00000u
#define LN_F32_INF       0x7F800000u

// 128bit 値（`hi:lo`）を表現する小さな構造体（MSL に `u128`／タプルが
// ないため。[`ln_f64_mul64_wide`]／[`ln_f64_shr128`]／
// [`ln_f64_low_bits128`] の戻り値に使う）。
struct LnU128 {
    ulong hi;
    ulong lo;
};

// 正規化済み仮数 `m`（隠れ 1 を bit52 に立てた 53bit 値）と、
// `value = m * 2^(exp_u - 52)` を満たす unbiased 指数 `exp_u` の対
// （[`ln_f64_normalize_mantissa`] の戻り値）。
struct LnNormMantissa {
    ulong m;
    long exp_u;
};

// 64bit leading zero count を 32bit `clz` 2 回で構成する（`clz(0u) == 32`
// は MSL 仕様で定義済み。`soft_f64::clz64` と同一構造）。
inline uint ln_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`。
inline ulong ln_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? LN_F64_QNAN : (sign | LN_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        // f32 subnormal（`frac × 2^-149`）は f64 では正規化数。
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & LN_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul); // 減算を先にすると exp < 127 で下溢れ。
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64` の符号反転。NaN も含め符号 bit を無条件に反転する。
inline ulong ln_f64_neg(ulong a) {
    return a ^ LN_F64_SIGN;
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` と同一手順（特殊値 → ガード 3 bit 付き桁合わせ
// → 加減算 → 正規化 → 丸め）。
inline ulong ln_f64_add(ulong a, ulong b) {
    ulong sa = a & LN_F64_SIGN;
    ulong sb = b & LN_F64_SIGN;
    ulong ea = (a >> 52) & LN_F64_EXP_MASK;
    ulong eb = (b >> 52) & LN_F64_EXP_MASK;
    ulong fa = a & LN_F64_FRAC_MASK;
    ulong fb = b & LN_F64_FRAC_MASK;

    if (ea == LN_F64_EXP_MASK || eb == LN_F64_EXP_MASK) {
        bool a_nan = (ea == LN_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == LN_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return LN_F64_QNAN;
        }
        if (ea == LN_F64_EXP_MASK && eb == LN_F64_EXP_MASK) {
            return (sa == sb) ? a : LN_F64_QNAN;
        }
        return (ea == LN_F64_EXP_MASK) ? a : b;
    }
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if (a_zero && b_zero) {
        return sa & sb;
    }
    if (a_zero) {
        return b;
    }
    if (b_zero) {
        return a;
    }

    ulong ma = (ea == 0ul) ? fa : (fa | (1ul << 52));
    ulong ea_eff = (ea == 0ul) ? 1ul : ea;
    ulong mb = (eb == 0ul) ? fb : (fb | (1ul << 52));
    ulong eb_eff = (eb == 0ul) ? 1ul : eb;
    if (ea_eff < eb_eff || (ea_eff == eb_eff && ma < mb)) {
        ulong t;
        t = ma; ma = mb; mb = t;
        t = ea_eff; ea_eff = eb_eff; eb_eff = t;
        t = sa; sa = sb; sb = t;
    }
    ma <<= 3;
    mb <<= 3;
    ulong d = ea_eff - eb_eff;
    if (d >= 64ul) {
        mb = (mb != 0ul) ? 1ul : 0ul;
    } else if (d > 0ul) {
        ulong lost = mb & ((1ul << d) - 1ul);
        mb = (mb >> d) | ((lost != 0ul) ? 1ul : 0ul);
    }

    ulong e = ea_eff;
    ulong m;
    if (sa == sb) {
        m = ma + mb;
        if (m >= (1ul << 56)) {
            ulong lost = m & 1ul;
            m = (m >> 1) | lost;
            e += 1ul;
        }
    } else {
        m = ma - mb;
        if (m == 0ul) {
            return 0ul;
        }
        ulong sh = (ulong)ln_f64_clz64(m);
        sh = (sh >= 8ul) ? (sh - 8ul) : 0ul;
        if (sh > e - 1ul) {
            sh = e - 1ul;
        }
        m <<= sh;
        e -= sh;
    }

    ulong r = m & 7ul;
    m >>= 3;
    if (r > 4ul || (r == 4ul && (m & 1ul) == 1ul)) {
        m += 1ul;
    }
    if (m >= (1ul << 53)) {
        m >>= 1;
        e += 1ul;
    }
    ulong exp_field = (m >= (1ul << 52)) ? e : 0ul;
    if (exp_field >= LN_F64_EXP_MASK) {
        return sa | LN_F64_INF;
    }
    return sa | (exp_field << 52) | (m & LN_F64_FRAC_MASK);
}

// `f64 + f64` を **round-to-odd**（RO）で丸めた版（PR #1671 codex-review
// 指摘・イシュー #1596 是正: affine `x̂·w+b` の二重丸め回避。冒頭コメント
// 「FMA 契約」参照）。[`ln_f64_add`] とは丸め規則のみが異なる姉妹関数
// （逐語複製。共通化すると分岐が増え可読性が落ちるため意図的に複製する。
// `soft_f64::add_f64_bits_round_to_odd` の逐語移植——数学的根拠
// 〈Boldo–Melquiond の round-to-odd 二重丸め定理〉は同関数の doc comment
// を正としここでは繰り返さない）。
//
// [`ln_f64_add`] との差分は最終段のみ: 「ガード 3bit `r` から `r>4` は
// 切り上げ・`r==4` は偶数丸め」ではなく「`r != 0`（丸め落ちする情報が
// 何かあれば）なら結果の最下位 bit を強制的に 1 にする」。round-to-odd
// は算術的な繰り上がり（`m += 1`）を一切行わない（ビット単位の OR のみ）
// ため、[`ln_f64_add`] が丸め後に持つ「`m` が `2^53` へ繰り上がる場合の
// 指数調整」分岐は発生しえず、本関数には存在しない。
inline ulong ln_f64_add_ro(ulong a, ulong b) {
    ulong sa = a & LN_F64_SIGN;
    ulong sb = b & LN_F64_SIGN;
    ulong ea = (a >> 52) & LN_F64_EXP_MASK;
    ulong eb = (b >> 52) & LN_F64_EXP_MASK;
    ulong fa = a & LN_F64_FRAC_MASK;
    ulong fb = b & LN_F64_FRAC_MASK;

    if (ea == LN_F64_EXP_MASK || eb == LN_F64_EXP_MASK) {
        bool a_nan = (ea == LN_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == LN_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return LN_F64_QNAN;
        }
        if (ea == LN_F64_EXP_MASK && eb == LN_F64_EXP_MASK) {
            return (sa == sb) ? a : LN_F64_QNAN;
        }
        return (ea == LN_F64_EXP_MASK) ? a : b;
    }
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if (a_zero && b_zero) {
        return sa & sb;
    }
    if (a_zero) {
        return b;
    }
    if (b_zero) {
        return a;
    }

    ulong ma = (ea == 0ul) ? fa : (fa | (1ul << 52));
    ulong ea_eff = (ea == 0ul) ? 1ul : ea;
    ulong mb = (eb == 0ul) ? fb : (fb | (1ul << 52));
    ulong eb_eff = (eb == 0ul) ? 1ul : eb;
    if (ea_eff < eb_eff || (ea_eff == eb_eff && ma < mb)) {
        ulong t;
        t = ma; ma = mb; mb = t;
        t = ea_eff; ea_eff = eb_eff; eb_eff = t;
        t = sa; sa = sb; sb = t;
    }
    ma <<= 3;
    mb <<= 3;
    ulong d = ea_eff - eb_eff;
    if (d >= 64ul) {
        mb = (mb != 0ul) ? 1ul : 0ul;
    } else if (d > 0ul) {
        ulong lost = mb & ((1ul << d) - 1ul);
        mb = (mb >> d) | ((lost != 0ul) ? 1ul : 0ul);
    }

    ulong e = ea_eff;
    ulong m;
    if (sa == sb) {
        m = ma + mb;
        if (m >= (1ul << 56)) {
            ulong lost = m & 1ul;
            m = (m >> 1) | lost;
            e += 1ul;
        }
    } else {
        m = ma - mb;
        if (m == 0ul) {
            return 0ul;
        }
        ulong sh = (ulong)ln_f64_clz64(m);
        sh = (sh >= 8ul) ? (sh - 8ul) : 0ul;
        if (sh > e - 1ul) {
            sh = e - 1ul;
        }
        m <<= sh;
        e -= sh;
    }

    // round-to-odd: 丸め落ちする 3 bit（ガード/丸め/sticky）のいずれかが
    // 立っていれば、結果の最下位 bit を強制的に 1 にする（算術繰り上がり
    // は行わないため `m` が `2^53` へ達することはない）。
    ulong r = m & 7ul;
    m >>= 3;
    if (r != 0ul) {
        m |= 1ul;
    }
    ulong exp_field = (m >= (1ul << 52)) ? e : 0ul;
    if (exp_field >= LN_F64_EXP_MASK) {
        return sa | LN_F64_INF;
    }
    return sa | (exp_field << 52) | (m & LN_F64_FRAC_MASK);
}

// `a - b` = [`ln_f64_add`]`(a, `[`ln_f64_neg`]`(b))`。
inline ulong ln_f64_sub(ulong a, ulong b) {
    return ln_f64_add(a, ln_f64_neg(b));
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・underflow は f32
// subnormal／`±0`。NaN は quiet NaN へ正規化）。`soft_f64::narrow_f64_bits`。
inline uint ln_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & LN_F64_EXP_MASK;
    ulong f = bits & LN_F64_FRAC_MASK;
    if (e == LN_F64_EXP_MASK) {
        return (f != 0ul) ? LN_F32_QNAN : (sign | LN_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | LN_F32_INF;
    }
    long extra = (ef <= 0l) ? (1l - ef) : 0l;
    long shift_l = 29l + extra;
    if (shift_l >= 54l) {
        return sign;
    }
    uint shift = (uint)shift_l;
    ulong q0 = m >> shift;
    ulong rem = m & ((1ul << shift) - 1ul);
    ulong half_bit = 1ul << (shift - 1u);
    ulong q = q0;
    if (rem > half_bit || (rem == half_bit && (q0 & 1ul) == 1ul)) {
        q += 1ul;
    }
    uint exp_field = (ef <= 0l) ? 0u : (uint)ef;
    if (ef <= 0l) {
        if (q >= (1ul << 23)) {
            exp_field = 1u;
            q -= 1ul << 23;
        }
    } else {
        if (q >= (1ul << 24)) {
            q >>= 1;
            exp_field += 1u;
        }
        q -= 1ul << 23;
    }
    if (exp_field >= 255u) {
        return sign | LN_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// `u64 x u64` の厳密な 128bit 積（32bit 分割のスクールブック乗算。
// `soft_f64::mul64_wide` の逐語移植）。
inline LnU128 ln_f64_mul64_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFul) + (lo_hi & 0xFFFFFFFFul);
    LnU128 result;
    result.lo = (lo_lo & 0xFFFFFFFFul) | (mid << 32);
    result.hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    return result;
}

// `(hi,lo)` を右シフト `s`（`0..=128`）した値（`soft_f64::shr128`）。
inline LnU128 ln_f64_shr128(ulong hi, ulong lo, uint s) {
    LnU128 r;
    if (s == 0u) {
        r.hi = hi; r.lo = lo;
    } else if (s >= 128u) {
        r.hi = 0ul; r.lo = 0ul;
    } else if (s < 64u) {
        r.hi = hi >> s;
        r.lo = (lo >> s) | (hi << (64u - s));
    } else if (s == 64u) {
        r.hi = 0ul; r.lo = hi;
    } else {
        r.hi = 0ul; r.lo = hi >> (s - 64u);
    }
    return r;
}

// `(hi,lo)` の下位 `n` bit（`n <= 128`）（`soft_f64::low_bits128`）。
inline LnU128 ln_f64_low_bits128(ulong hi, ulong lo, uint n) {
    LnU128 r;
    if (n == 0u) {
        r.hi = 0ul; r.lo = 0ul;
    } else if (n >= 128u) {
        r.hi = hi; r.lo = lo;
    } else if (n <= 64u) {
        ulong mask = (n == 64u) ? (~0ul) : ((1ul << n) - 1ul);
        r.hi = 0ul; r.lo = lo & mask;
    } else {
        uint n2 = n - 64u;
        ulong mask = (n2 == 64u) ? (~0ul) : ((1ul << n2) - 1ul);
        r.hi = hi & mask; r.lo = lo;
    }
    return r;
}

// `(ahi,alo)` と `(bhi,blo)` の数値比較（`-1`／`0`／`1`）。
inline int ln_f64_cmp128(ulong ahi, ulong alo, ulong bhi, ulong blo) {
    if (ahi != bhi) {
        return (ahi < bhi) ? -1 : 1;
    }
    if (alo != blo) {
        return (alo < blo) ? -1 : 1;
    }
    return 0;
}

// `f64` の指数・仮数フィールド（`e`：バイアス済み・`f`：フラクション。
// `e==0 && f==0` の完全ゼロは呼び出し側で排除済みの前提）から
// 「隠れ 1 を bit52 に立てた 53bit 仮数」と `value = m * 2^(exp_u-52)`
// の unbiased 指数の対を求める（`soft_f64::normalize_f64_mantissa`）。
inline LnNormMantissa ln_f64_normalize_mantissa(ulong e, ulong f) {
    LnNormMantissa r;
    if (e == 0ul) {
        uint lead = 63u - ln_f64_clz64(f);
        uint shift = 52u - lead;
        r.m = f << shift;
        r.exp_u = -1022l - (long)shift;
    } else {
        r.m = f | (1ul << 52);
        r.exp_u = (long)e - 1023l;
    }
    return r;
}

// `f64 * f64`（最近接偶数丸め）の bit 表現版（NaN は quiet NaN へ
// 正規化）。仮数同士の厳密 106bit 積を [`ln_f64_mul64_wide`] で構成し、
// **1 回だけ**丸める（中間で `f32` はもとより暫定 `f64` へも丸めない。
// 二重丸め回避）。`soft_f64::mul_f64_bits` の逐語移植。
inline ulong ln_f64_mul(ulong a, ulong b) {
    ulong sa = a & LN_F64_SIGN;
    ulong sb = b & LN_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & LN_F64_EXP_MASK;
    ulong eb = (b >> 52) & LN_F64_EXP_MASK;
    ulong fa = a & LN_F64_FRAC_MASK;
    ulong fb = b & LN_F64_FRAC_MASK;

    bool a_nan = (ea == LN_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == LN_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return LN_F64_QNAN;
    }
    bool a_inf = (ea == LN_F64_EXP_MASK);
    bool b_inf = (eb == LN_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_zero && b_inf) || (a_inf && b_zero)) {
        return LN_F64_QNAN;
    }
    if (a_inf || b_inf) {
        return sign | LN_F64_INF;
    }
    if (a_zero || b_zero) {
        return sign;
    }

    LnNormMantissa na = ln_f64_normalize_mantissa(ea, fa);
    LnNormMantissa nb = ln_f64_normalize_mantissa(eb, fb);
    LnU128 p = ln_f64_mul64_wide(na.m, nb.m);
    // `na.m, nb.m ∈ [2^52, 2^53)` のため積は常に `[2^104, 2^106)`。
    // よって `p.hi` は常に非ゼロ（bit104 以上は `hi` 側〈bit64 以降〉に
    // 属する）。
    uint leadpos = 64u + (63u - ln_f64_clz64(p.hi));
    long exp_u = na.exp_u + nb.exp_u + ((long)leadpos - 104l);
    long shift_normal = (long)leadpos - 52l; // 52 か 53（常に < 64）。

    long biased_before_round = exp_u + 1023l;
    long final_shift_l;
    bool is_subnormal_target;
    if (biased_before_round >= 1l) {
        final_shift_l = shift_normal;
        is_subnormal_target = false;
    } else {
        final_shift_l = shift_normal + (1l - biased_before_round);
        is_subnormal_target = true;
    }

    if (final_shift_l < 0l || final_shift_l >= 128l) {
        // 到達性: 本カーネルの実用値域（`f32` 由来の `x`／`eps`／
        // `weight` から生じる soft-f64 中間値）では発生しない極端な
        // underflow。安全側として `±0` へ丸める。
        return sign;
    }
    uint final_shift = (uint)final_shift_l;
    LnU128 q = ln_f64_shr128(p.hi, p.lo, final_shift);
    ulong m = q.lo; // `q.hi` は常に 0（呼び出し前提の値域より）。
    LnU128 rem = ln_f64_low_bits128(p.hi, p.lo, final_shift);
    ulong half_hi = 0ul;
    ulong half_lo = 0ul;
    if (final_shift > 0u) {
        if (final_shift - 1u < 64u) {
            half_lo = 1ul << (final_shift - 1u);
        } else {
            half_hi = 1ul << (final_shift - 1u - 64u);
        }
    }
    int cmp = ln_f64_cmp128(rem.hi, rem.lo, half_hi, half_lo);
    bool round_up = (cmp > 0) || (cmp == 0 && (m & 1ul) == 1ul);
    if (round_up) {
        m += 1ul;
    }

    if (!is_subnormal_target) {
        long exp_final = exp_u;
        if (m >= (1ul << 53)) {
            m >>= 1;
            exp_final += 1l;
        }
        long biased_final = exp_final + 1023l;
        if (biased_final >= (long)LN_F64_EXP_MASK) {
            return sign | LN_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & LN_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

// `(hi,lo)`（128bit・呼び出し前提: 商が 64bit に収まる）を 64bit の
// 非ゼロ除数 `d` で割る筆算除算（2 進 shift-subtract 方式。
// `soft_f64::div64_wide` の逐語移植）。剰余 `rem` は各ステップで `d`
// 未満に保たれるため `d` が 64bit に収まる限り overflow しない。
// [`ln_f64_div`] 専用ヘルパー。
inline void ln_f64_div64_wide(ulong hi, ulong lo, ulong d, thread ulong &quotient_out, thread ulong &remainder_out) {
    ulong rem = 0ul;
    ulong q = 0ul;
    for (int i = 127; i >= 0; i--) {
        ulong bit = (i >= 64) ? ((hi >> (uint)(i - 64)) & 1ul) : ((lo >> (uint)i) & 1ul);
        rem = (rem << 1) | bit;
        if (rem >= d) {
            rem -= d;
            q = (q << 1) | 1ul;
        } else {
            q = q << 1;
        }
    }
    quotient_out = q;
    remainder_out = rem;
}

// `f64 / f64`（最近接偶数丸め）の bit 表現版（NaN は quiet NaN へ
// 正規化）。`soft_f64::div_f64_bits` の逐語移植——`mean`／`var` を
// `sum * recip(hidden)`（Newton 近似逆数との積）ではなく本関数の
// **正しく丸めた除算**で求めることで、一様行（例 `x=[1e30f32;49]`）
// のような「割り切れる」ケースで Newton 近似特有の 1 ULP 誤差が
// 悪化するのを防ぐ（PR #1671 codex-review・Cursor Bugbot 指摘。
// イシュー #1596）。アルゴリズムのコメントは `soft_f64::div_f64_bits`
// を参照（本関数は逐語移植のため二重に説明しない）。
inline ulong ln_f64_div(ulong a, ulong b) {
    ulong sa = a & LN_F64_SIGN;
    ulong sb = b & LN_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & LN_F64_EXP_MASK;
    ulong eb = (b >> 52) & LN_F64_EXP_MASK;
    ulong fa = a & LN_F64_FRAC_MASK;
    ulong fb = b & LN_F64_FRAC_MASK;

    bool a_nan = (ea == LN_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == LN_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return LN_F64_QNAN;
    }
    bool a_inf = (ea == LN_F64_EXP_MASK);
    bool b_inf = (eb == LN_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_inf && b_inf) || (a_zero && b_zero)) {
        return LN_F64_QNAN;
    }
    if (a_inf) {
        return sign | LN_F64_INF;
    }
    if (b_inf) {
        return sign;
    }
    if (b_zero) {
        return sign | LN_F64_INF;
    }
    if (a_zero) {
        return sign;
    }

    LnNormMantissa na = ln_f64_normalize_mantissa(ea, fa);
    LnNormMantissa nb = ln_f64_normalize_mantissa(eb, fb);
    // `S = 55`: `ma/mb ∈ (0.5,2)` のため商は `[2^54,2^56)` に収まり、
    // 53bit 仮数 + 2bit（guard/round）の精度が確保できる最小の追加
    // シフト量（`soft_f64::div_f64_bits` と同じ定数）。
    const uint S = 55u;
    LnU128 num = ln_f64_mul64_wide(na.m, 1ul << S);
    ulong raw_q;
    ulong rem;
    ln_f64_div64_wide(num.hi, num.lo, nb.m, raw_q, rem);
    ulong q = raw_q | (ulong)(rem != 0ul ? 1ul : 0ul);
    // `q` は非ゼロ（呼び出し前提より `na.m`／`nb.m` はいずれも非ゼロ）。
    uint leadpos = 63u - ln_f64_clz64(q);
    long exp_u = na.exp_u - nb.exp_u + ((long)leadpos - (long)S);
    long shift_normal = (long)leadpos - 52l; // 2 か 3（`leadpos` が 54 か 55）。

    long biased_before_round = exp_u + 1023l;
    long final_shift_l;
    bool is_subnormal_target;
    if (biased_before_round >= 1l) {
        final_shift_l = shift_normal;
        is_subnormal_target = false;
    } else {
        final_shift_l = shift_normal + (1l - biased_before_round);
        is_subnormal_target = true;
    }

    if (final_shift_l < 0l || final_shift_l >= 64l) {
        // 到達性: 本カーネルの実用値域（`f32` 由来の `sum`／`hidden`）
        // では発生しない極端な underflow。安全側として `±0` へ丸める。
        return sign;
    }
    uint final_shift = (uint)final_shift_l;
    ulong m = (final_shift == 0u) ? q : (q >> final_shift);
    ulong low_mask = (final_shift >= 64u) ? ~0ul : ((1ul << final_shift) - 1ul);
    ulong rem_low = (final_shift == 0u) ? 0ul : (q & low_mask);
    // `half`（MSL の予約型 `half` と衝突するため `half_bit` と命名）。
    ulong half_bit = (final_shift == 0u) ? 0ul : (1ul << (final_shift - 1u));
    bool round_up = (rem_low > half_bit) || (rem_low == half_bit && (m & 1ul) == 1ul);
    if (round_up) {
        m += 1ul;
    }

    if (!is_subnormal_target) {
        long exp_final = exp_u;
        if (m >= (1ul << 53)) {
            m >>= 1;
            exp_final += 1l;
        }
        long biased_final = exp_final + 1023l;
        if (biased_final >= (long)LN_F64_EXP_MASK) {
            return sign | LN_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & LN_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

// `x`（正規化数・subnormal 双方に対応。特殊値は呼び出し側で除外済みの
// 前提）を `2^k` 倍する（指数フィールドを直接加算するだけの厳密演算。
// Newton 反復の「種」専用——範囲を超える場合は `±inf`／`±0` へ丸め
// なしで潰す。`soft_f64::scale_pow2_f64_bits`）。
inline ulong ln_f64_scale_pow2(ulong x_bits, long k) {
    ulong sign = x_bits & LN_F64_SIGN;
    long e = (long)((x_bits >> 52) & LN_F64_EXP_MASK);
    ulong f = x_bits & LN_F64_FRAC_MASK;
    long new_e = e + k;
    if (new_e >= (long)LN_F64_EXP_MASK) {
        return sign | LN_F64_INF;
    }
    if (new_e <= 0l) {
        return sign;
    }
    return sign | (((ulong)new_e) << 52) | f;
}

// 仮数・指数を分離した種抽出（`soft_f64::extract_reduced_mantissa_and_exp`）。
// `want_sqrt_range == false`: `reduced ∈ [1,2)`・`exp_out = exp_u`
// （[`ln_f64_recip_newton`] 用）。`want_sqrt_range == true`: `reduced ∈
// [1,4)`（指数の偶奇に応じて範囲を揃える）・`exp_out = floor(exp_u/2)`
// 相当（[`ln_f64_rsqrt_newton`] 用）。
struct LnReducedSeed {
    uint reduced_bits;
    long exp_out;
};

inline LnReducedSeed ln_f64_extract_reduced_and_exp(ulong x_bits, bool want_sqrt_range) {
    ulong e = (x_bits >> 52) & LN_F64_EXP_MASK;
    ulong f = x_bits & LN_F64_FRAC_MASK;
    LnNormMantissa n = ln_f64_normalize_mantissa(e, f);
    LnReducedSeed r;
    if (!want_sqrt_range) {
        ulong reduced_bits = (1023ul << 52) | (n.m & LN_F64_FRAC_MASK);
        r.reduced_bits = ln_f64_narrow(reduced_bits);
        r.exp_out = n.exp_u;
    } else if ((n.exp_u & 1l) == 0l) {
        // `exp_u` が偶数（負値の剰余は `& 1` で符号に依らず 0/1 が出る）。
        ulong reduced_bits = (1023ul << 52) | (n.m & LN_F64_FRAC_MASK);
        r.reduced_bits = ln_f64_narrow(reduced_bits);
        r.exp_out = n.exp_u >> 1; // 偶数の算術右シフトは厳密な /2。
    } else {
        ulong reduced_bits = (1024ul << 52) | (n.m & LN_F64_FRAC_MASK);
        r.reduced_bits = ln_f64_narrow(reduced_bits);
        r.exp_out = (n.exp_u - 1l) >> 1;
    }
    return r;
}

// `f64` の逆数 `1/x` を Newton-Raphson（`y_{n+1} = y_n*(2 - x*y_n)`）で
// 求める（`x` は本カーネルの用途上〈`hidden` 由来〉常に有限・正の値。
// `soft_f64::recip_newton_f64_bits` の逐語移植）。**`mean`／`var` の
// 計算では現在使わない**（PR #1671 是正: Newton 近似逆数との積は
// 一様行で 1 ULP 誤差が悪化するため `ln_f64_div`〈正しく丸めた除算〉へ
// 置き換えた。イシュー #1596）。ホスト側 `recip_newton_f64_bits` と
// 1 対 1 対応する soft-f64 プリミティブとして、収束精度の回帰テスト
// （`soft_f64::tests::recip_newton_converges_for_hidden_range`）と
// ともに残置する（他ファイルの未使用だが残置されている診断・将来用
// 関数群と同じ方針。`context.rs::BatchGpuTimestamps` 等）。
inline ulong ln_f64_recip_newton(ulong x) {
    LnReducedSeed seed = ln_f64_extract_reduced_and_exp(x, false);
    float seed_reduced = 1.0f / as_type<float>(seed.reduced_bits);
    ulong y = ln_f64_scale_pow2(ln_f64_widen(as_type<uint>(seed_reduced)), -seed.exp_out);
    const ulong TWO = 0x4000000000000000ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong xy = ln_f64_mul(x, y);
        ulong two_minus_xy = ln_f64_sub(TWO, xy);
        y = ln_f64_mul(y, two_minus_xy);
    }
    return y;
}

// `f64` の逆数平方根 `1/sqrt(x)` を Newton-Raphson（`y_{n+1} =
// y_n*(1.5 - 0.5*x*y_n^2)`）で求める。特殊値は明示的に扱う（`x` は
// 分散 `+ eps`〈ともに非負〉由来で数学的に非負のはずだが、防御的に
// 負値も NaN として扱う）: `NaN -> NaN`・`±0 -> ±inf`・負（非ゼロ）
// `-> NaN`・`+inf -> +0`。`soft_f64::rsqrt_newton_f64_bits` の逐語移植。
inline ulong ln_f64_rsqrt_newton(ulong x) {
    ulong e = (x >> 52) & LN_F64_EXP_MASK;
    ulong f = x & LN_F64_FRAC_MASK;
    ulong sign = x & LN_F64_SIGN;
    if (e == LN_F64_EXP_MASK && f != 0ul) {
        return LN_F64_QNAN;
    }
    if (e == 0ul && f == 0ul) {
        return sign | LN_F64_INF;
    }
    if (sign != 0ul) {
        return LN_F64_QNAN;
    }
    if (e == LN_F64_EXP_MASK) {
        return 0ul; // +inf -> +0
    }

    LnReducedSeed seed = ln_f64_extract_reduced_and_exp(x, true);
    float seed_reduced = 1.0f / sqrt(as_type<float>(seed.reduced_bits));
    ulong y = ln_f64_scale_pow2(ln_f64_widen(as_type<uint>(seed_reduced)), -seed.exp_out);
    const ulong ONE_HALF = 0x3FE0000000000000ul;
    const ulong THREE_HALF = 0x3FF8000000000000ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong y2 = ln_f64_mul(y, y);
        ulong xy2 = ln_f64_mul(x, y2);
        ulong half_xy2 = ln_f64_mul(ONE_HALF, xy2);
        ulong inner = ln_f64_sub(THREE_HALF, half_xy2);
        y = ln_f64_mul(y, inner);
    }
    return y;
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
    // `inv_n`（ホストから渡される事前丸め済み `1/hidden`）は使わない
    // （`hidden` 自体を分母として soft-f64 の正しく丸めた除算
    // （`ln_f64_div`）へ渡し、二重丸め・Newton 近似逆数特有の
    // 1 ULP 誤差を避ける。バッファレイアウト互換のため引数自体は残す）。
    (void)inv_n;

    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;

        // `hidden` の f64 表現（行内で不変のため 1 回だけ widen する）。
        ulong hidden_f64 = ln_f64_widen(as_type<uint>((float)hidden));

        // パス 1: 平均（soft-f64 総和。冒頭コメント「総和の順序」参照）。
        ulong lane_sum = 0ul; // +0.0（f64）。
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            ulong xv = ln_f64_widen(as_type<uint>(x[row_base + idx]));
            lane_sum = ln_f64_add(lane_sum, xv);
        }
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            ulong other_hi = simd_shuffle_xor((uint)(lane_sum >> 32), offset);
            ulong other_lo = simd_shuffle_xor((uint)lane_sum, offset);
            ulong other_sum = (other_hi << 32) | other_lo;
            lane_sum = ln_f64_add(lane_sum, other_sum);
        }
        ulong mean = ln_f64_div(lane_sum, hidden_f64);

        // パス 2: 分散（二パス。`(x-mean)^2` を soft-f64 で蓄積する）。
        ulong lane_sq = 0ul;
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            ulong xv = ln_f64_widen(as_type<uint>(x[row_base + idx]));
            ulong dev = ln_f64_sub(xv, mean);
            ulong devsq = ln_f64_mul(dev, dev);
            lane_sq = ln_f64_add(lane_sq, devsq);
        }
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            ulong other_hi = simd_shuffle_xor((uint)(lane_sq >> 32), offset);
            ulong other_lo = simd_shuffle_xor((uint)lane_sq, offset);
            ulong other_sq = (other_hi << 32) | other_lo;
            lane_sq = ln_f64_add(lane_sq, other_sq);
        }
        ulong var = ln_f64_div(lane_sq, hidden_f64);
        ulong eps_f64 = ln_f64_widen(as_type<uint>(eps));
        ulong var_plus_eps = ln_f64_add(var, eps_f64);
        ulong rstd = ln_f64_rsqrt_newton(var_plus_eps);

        // パス 3: 書き出し（device メモリを再読）。`xhat` を soft-f64 で
        // 確定した後 1 回だけ `f32` へ丸める（`ln_f64_narrow`。CPU 参照
        // 実装 `((x-mean)*rstd) as f32` と同じ丸め位置）。
        //
        // **affine を round-to-odd で計算する理由**: affine
        // （`x̂·w+b`）は CPU/CUDA と同じ「`f32` の `xhat`・`weight`・
        // `bias` に対する単一丸めの FMA」でなければならない
        // （冒頭コメント「FMA 契約」参照）が、これを満たす経路には
        // 2 つの罠がある。
        //
        // 罠 1（平坦な `float` の `fma()` を使わない理由。GPU の
        // subnormal flush-to-zero 対策。codex-review 指摘・PR #1671
        // スレッド 1 件目の反例で実機実測により発覚）: `xhat`
        // （`f32` へ丸めた直後の値）自体が `f32` の subnormal に
        // なりうる（`dev`・`rstd` の値域次第。例
        // `x=[1e38,-1e38,1e-4,-1e-4]` の小さい要素）。この `xhat` を
        // 平坦な `float` の `fma(xhat, wv, bv)` へそのまま渡すと、
        // Apple GPU 実機がハードウェア命令の**入力側**で subnormal を
        // ゼロへ flush しうることを実機実測で確認した
        // （`layer_norm_tiny_x_huge_eps_stays_finite_and_nonzero` の
        // 反例で出力が `[0.0, 0.0]` になる形で顕在化）。
        //
        // 罠 2（soft-f64 の素朴な二段階丸め——`mul`＋`ln_f64_add`（最近接
        // 偶数丸め）で `f64` へ丸めてから `ln_f64_narrow` で `f32` へ
        // 丸める——では単一丸めの FMA と一致しない。codex-review 指摘・
        // PR #1671 スレッド 2 件目の反例。イシュー #1596）: 罠 1 対策
        // として `xhat`・`wv`・`bv` を soft-f64 へ widen し直し `mul`＋
        // `add` で affine を計算する方針自体は正しいが、当初の実装は
        // `add` に通常の最近接偶数丸め `ln_f64_add` を使っていた。
        // `xhat=31/16`・`weight=f32::from_bits(0x7f042108)`・
        // `bias=-1` のように積と加数の指数が大きく乖離する入力では、
        // `ln_f64_add` の `f64`（53bit）への丸めが `f32`（24bit）の
        // 桁の決定に必要な情報を握り潰し、続く `ln_f64_narrow` が
        // ハードウェア FMA（`f32::mul_add`。本例では有限の
        // `f32::MAX`）と異なる値（`+inf`）を返す「二重丸め」が
        // 発生する（REQ-2 統一複合判定の許容誤差を大幅に超える差であり
        // 「稀だが誤差の範囲内」とは言えない）。
        //
        // 対策（両方の罠を同時に回避）: `xhat`・`wv`・`bv` を soft-f64
        // へ widen し直し（subnormal を経由しても flush されない 64bit
        // 整数演算のみで構成——罠 1 の対策は不変）、積は `ln_f64_mul`
        // （`f32` 同士の積は仮数 48bit 以内に収まり `f64` の 53bit 仮数へ
        // 丸め無しで厳密表現できるため、この段は丸め無しの厳密演算）、
        // 和は通常の `ln_f64_add` ではなく**round-to-odd 丸め加算
        // `ln_f64_add_ro`**（上記定義。`crates/backend-metal/src/soft_f64.rs::
        // add_f64_bits_round_to_odd` の逐語移植）を使い、最後に通常の
        // `ln_f64_narrow`（最近接偶数丸め）で 1 回だけ `f32` へ丸める。
        // Boldo–Melquiond（2008）の round-to-odd 二重丸め定理（中間精度
        // `p2` が目的精度 `p1` に対し `p2 >= p1+2` を満たせば
        // `RN_p1(RO_p2(x)) == RN_p1(x)`。ここでは `p2=53`〈`f64`〉・
        // `p1=24`〈`f32`〉で `53 >= 26` を満たす）により、この経路は
        // 厳密値 `xhat*w+b` を `f32` へ直接単一丸めした結果——すなわち
        // CPU 参照実装のハードウェア `fma`（`f32::mul_add`。x86/ARM
        // CPU は subnormal 入力を flush しない）と数学的に同値の
        // 結果——と一致する（`ln_f64_add_ro` の doc comment に数学的
        // 根拠を記載。`crates/backend-metal/src/soft_f64.rs::fma_f32_bits` がホスト側逐語
        // モデルで、上記反例を含むランダム 200 万組・`f32::mul_add`
        // との bit 完全一致をユニットテストで検証する）。
        for (uint idx = lane; idx < hidden; idx += LAYER_NORM_SIMD_WIDTH) {
            ulong xv = ln_f64_widen(as_type<uint>(x[row_base + idx]));
            ulong dev = ln_f64_sub(xv, mean);
            ulong xhat64 = ln_f64_mul(dev, rstd);
            uint xhat_bits = ln_f64_narrow(xhat64);

            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float bv = (has_bias != 0) ? b[idx] : 0.0f;
            ulong wv64 = ln_f64_widen(as_type<uint>(wv));
            ulong bv64 = ln_f64_widen(as_type<uint>(bv));
            ulong affine64 = ln_f64_add_ro(ln_f64_mul(ln_f64_widen(xhat_bits), wv64), bv64);
            out[row_base + idx] = as_type<float>(ln_f64_narrow(affine64));
        }
    }
}
