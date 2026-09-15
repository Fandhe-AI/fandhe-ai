// BatchNorm1d／2d 順伝播カーネル（イシュー #1736・親 #1608。
// `layer_norm.metal`〈#1596〉と同じモジュール構成方針を踏襲する
// BatchNorm 対応版）。
//
// 意味論: train モードは `out = (x - batch_mean(x, axis=channel)) *
// rsqrt(batch_var(x, axis=channel) + eps) * w + b`（`has_weight == 0`
// の場合は `w` への乗算を、`has_bias == 0` の場合は `b` への加算を
// それぞれスキップ）。分散は biased（÷M。二パス `Σ(x-mean)^2/M` で
// 計算する。`docs/batch-norm-ops-design.md` §3.1〜§3.3）。infer
// モードは `mean`／`var`（呼び出し元が保持する running stats）を
// バッチから計算し直さずそのまま使う。チャネル軸は常に dim 1
// （NCHW／NCL 固定）で、チャネル `ch` に属する `M = n*spatial` 要素の
// 局所添字 `i` は `batch = i/spatial`・`sp = i%spatial` へ分解し実
// データ添字 `batch*(c*spatial) + ch*spatial + sp` へ写像する
// （`crates/backend-cpu/src/batch_norm.rs::channel_index` の GPU 側
// 複製）。
//
// **数値方式（採用: soft-f64。イシュー題名は「Neumaier＋scale/ssq」
// だが不採用——理由は以下）**: `.claude/rules/coding-rust.md`
// 「正規化統計の二乗和」節は Metal の `f64` 相当実装形として
// Neumaier 補償和 + scale/ssq 方式を挙げているが、本ファイルは
// `layer_norm.metal`（#1596）と同じ理由からより精度の高い soft-f64
// 方式（IEEE 754 binary64 の 64bit 整数ソフトウェアエミュレーション）
// を採用する（同節はこの方式を禁止しているわけではなく、「`f64`
// 相当の精度を保つ」という契約自体はむしろ本方式のほうがより強く
// 満たす）。経緯: `layer_norm.metal` は当初 Neumaier 補償和 +
// scale/ssq 分散を実装していたが、PR #1671 codex-review 指摘により
// 「行スケール除算での微小値消失」「正規化係数の丸め誤差が affine の
// 相殺で増幅される」という 2 系統の反例で REQ-2 統一複合判定を満たせ
// ないことが判明し、soft-f64 方式（`ln_f64_*` 系関数）へ全面置換
// された（`layer_norm.metal` 冒頭コメント「数値方式」参照）。
// BatchNorm の統計計算は LayerNorm と同型（`mean`／`var`／`rstd` を
// 高精度アキュムレータで保持し `xhat` を確定した後に 1 回だけ `f32`
// へ丸める。affine も同じ二重丸め問題を抱える）ため、同じ反例が
// 再発しうると判断し、実装時点から soft-f64 方式を採用する
// （`docs/backend-metal-command-batching-design.md` §10.14 の bias
// 縮約カーネルも同じ理由で soft-f64 方式へ最終的に収束した前例が
// ある）。
//
// **soft-f64 プリミティブ（`bn_f64_*`）**: `crates/backend-metal/src/
// soft_f64.rs` の逐語移植（`u64`→`ulong`・`u32`→`uint`・
// `leading_zeros()`→`clz()`）。`layer_norm.metal::ln_f64_*` と機能
// 重複するが、MSL は `newLibraryWithSource` で個別ファイル単位に
// コンパイルされ翻訳単位を共有できないため、`bn_f64_` 接頭辞を付けた
// 本ファイル内で独立に定義する（`gemm.metal::bias_f64_*`・
// `layer_norm.metal::ln_f64_*` に続く 3 つ目の意図的な複製）。本ファイル
// の `bn_f64_*` 関数本体は `layer_norm.metal` の `ln_f64_*` 関数本体と
// 接頭辞以外は逐語一致する（`tests/batch_norm_source_evidence.rs` の
// ドリフトガードが固定する）。
//
// **FMA 契約**: affine（`x̂·w+b`）は CPU 参照実装（`backend-cpu::
// batch_norm::apply_affine` の `xhat.mul_add(w, b)`）・CUDA
// （`kernels_batch_norm.rs` の `fmaf`）と同じ「`f32` の `xhat`・
// `weight`・`bias` に対する単一丸めの FMA」を実現するため、
// `layer_norm.metal` と同じ round-to-odd 経由（`bn_f64_mul` で積を
// 厳密に求めた後 `bn_f64_add_ro` で `bias` を加え `bn_f64_narrow` で
// 1 回だけ `f32` へ丸める。数学的根拠は `layer_norm.metal` 冒頭
// コメント「本ファイルの affine 実装」節・Boldo–Melquiond（2008）の
// round-to-odd 二重丸め定理を参照し本ファイルでは繰り返さない）で
// 計算する。
//
// **総和の順序**: `crates/backend-cpu/src/batch_norm.rs::
// warp_reduce_f64`（32 レーンのストライドアクセス + offset
// 16→8→4→2→1 の butterfly）と同一の加算順序をチャネルごとの `M` 要素
// へ適用する（`docs/batch-norm-ops-design.md` §3.1。CUDA
// `kernels_batch_norm.rs` も同じ契約で `__shfl_xor_sync` の `double`
// 直接対応を使う）。soft-f64（52bit 精度）の加算は非結合性の影響が
// 極めて小さいため、CPU の逐次加算との順序差は REQ-2 統一複合判定
// （相対誤差 1e-3 未満または絶対誤差 1e-5 未満）の範囲内で無視できる
// （`layer_norm.metal` と同じ判断）。
//
// **`M` の f64 表現**: `layer_norm.metal` は `hidden`（行長）を
// `(float)hidden` へ直接変換し `2^24` 超を起動前に拒否する
// （`layer_norm.rs::validate_hidden_exact_f32`）が、本ファイルは
// この方針を踏襲しない——BatchNorm2d の `M = n*spatial` は実用形状
// （例 `n=256`・`spatial=512*512`）で容易に `2^24` を超えるため、
// 拒否するとホストフォールバックへ落ちてしまい実用性を欠く
// （`docs/batch-norm-ops-design.md`「`M` の f64 表現」設計判断）。
// 代わりにホスト側（`batch_norm.rs::run_batch_norm_train_f32`）が
// `(m as f64).to_bits()` を計算し、上位 32bit／下位 32bit の 2 引数
// （`m_f64_hi`／`m_f64_lo`。`constant uint&` の `setBytes` 契約に
// 合わせた分割——`constant ulong&` の前例が本クレートに無いための
// 保守的な選択）としてカーネルへ渡す。カーネル側は
// `((ulong)m_f64_hi << 32) | (ulong)m_f64_lo` で厳密に復元するため、
// `M` の大小に関わらず丸め誤差が一切生じない。
//
// **REQ-2 判定契約**（CPU との bit 一致は主張しない）: `batch_mean`
// は CPU（`warp_reduce_f64` 逐次蓄積 → butterfly → `Σ/M`）と同じ
// 縮約順序・同じ正しく丸めた除算のため bit 一致する見込みだが、CPU の
// 分散は `d.mul_add(d, acc)`（`f64` FMA・単一丸め）で `rstd` は
// ハードウェア `sqrt` であるのに対し、soft-f64 は `mul`+`add`（二重
// 丸め）・Newton-Raphson `rsqrt` のため `var`／`rstd`／出力は bit
// 一致しない。よって本カーネル全体は REQ-2 統一複合判定
// （`fandhe_ai_backend_cpu::parity::assert_parity`）で CPU 参照実装と
// 検証する（`docs/batch-norm-ops-design.md`「Metal 実装記録」参照）。
//
// **1 threadgroup = 1 simdgroup（32 スレッド）固定・persistent
// threadgroup 方式**（`for (ch = tg_id; ch < c; ch += grid_size)`）・
// reduction は 5 段 butterfly（`simd_shuffle_xor` 幅 16/8/4/2/1）。
// いずれも `layer_norm.metal`／`rmsnorm.metal` と同じ設計。infer
// カーネルは統計を再計算しないため grid-stride 不要の単純 elementwise
// （`if (gid >= numel) return;` の手動境界検査。REQ-8）とする。
//
// REQ-8 境界検査: train カーネルの添字計算は `ulong`（`BN_IDX` マクロ）
// で `n*c*spatial` の乗算オーバーフローを避ける。infer カーネルは
// `gid >= numel` を明示ガードする。ベクトル化ロードは行わない
// （`layer_norm.metal` と同じく後続課題）。
//
// コンパイルオプションは `pipeline::compile_options()` を適用する。

#include <metal_stdlib>
using namespace metal;

constant uint BATCH_NORM_SIMD_WIDTH = 32u;

// ---- IEEE 754 binary64 のソフトウェアエミュレーション ----
// `crates/backend-metal/src/soft_f64.rs` の逐語移植（`u64`→`ulong`・
// `u32`→`uint`・`leading_zeros()`→`clz()`。冒頭コメント「ホスト側の
// 逐語モデル」参照）。定数は同モジュールの `F64_*`／`F32_*` と同値。
// `gemm.metal::bias_f64_*` と機能重複するが、MSL は `newLibraryWithSource`
// で個別ファイル単位にコンパイルされ翻訳単位を共有できないため、
// `bn_f64_` 接頭辞を付けた本ファイル内で独立に定義する（意図的な
// 重複。`gemm.metal` 側は `mul`／`recip`／`rsqrt` を持たないため一部
// 機能はこちらが上位互換）。

#define BN_F64_SIGN      0x8000000000000000ul
#define BN_F64_EXP_MASK  0x7FFul
#define BN_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define BN_F64_QNAN      0x7FF8000000000000ul
#define BN_F64_INF       0x7FF0000000000000ul
#define BN_F32_QNAN      0x7FC00000u
#define BN_F32_INF       0x7F800000u

// 128bit 値（`hi:lo`）を表現する小さな構造体（MSL に `u128`／タプルが
// ないため。[`bn_f64_mul64_wide`]／[`bn_f64_shr128`]／
// [`bn_f64_low_bits128`] の戻り値に使う）。
struct BnU128 {
    ulong hi;
    ulong lo;
};

// 正規化済み仮数 `m`（隠れ 1 を bit52 に立てた 53bit 値）と、
// `value = m * 2^(exp_u - 52)` を満たす unbiased 指数 `exp_u` の対
// （[`bn_f64_normalize_mantissa`] の戻り値）。
struct BnNormMantissa {
    ulong m;
    long exp_u;
};

// 64bit leading zero count を 32bit `clz` 2 回で構成する（`clz(0u) == 32`
// は MSL 仕様で定義済み。`soft_f64::clz64` と同一構造）。
inline uint bn_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`。
inline ulong bn_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? BN_F64_QNAN : (sign | BN_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        // f32 subnormal（`frac × 2^-149`）は f64 では正規化数。
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & BN_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul); // 減算を先にすると exp < 127 で下溢れ。
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64` の符号反転。NaN も含め符号 bit を無条件に反転する。
inline ulong bn_f64_neg(ulong a) {
    return a ^ BN_F64_SIGN;
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` と同一手順（特殊値 → ガード 3 bit 付き桁合わせ
// → 加減算 → 正規化 → 丸め）。
inline ulong bn_f64_add(ulong a, ulong b) {
    ulong sa = a & BN_F64_SIGN;
    ulong sb = b & BN_F64_SIGN;
    ulong ea = (a >> 52) & BN_F64_EXP_MASK;
    ulong eb = (b >> 52) & BN_F64_EXP_MASK;
    ulong fa = a & BN_F64_FRAC_MASK;
    ulong fb = b & BN_F64_FRAC_MASK;

    if (ea == BN_F64_EXP_MASK || eb == BN_F64_EXP_MASK) {
        bool a_nan = (ea == BN_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == BN_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return BN_F64_QNAN;
        }
        if (ea == BN_F64_EXP_MASK && eb == BN_F64_EXP_MASK) {
            return (sa == sb) ? a : BN_F64_QNAN;
        }
        return (ea == BN_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)bn_f64_clz64(m);
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
    if (exp_field >= BN_F64_EXP_MASK) {
        return sa | BN_F64_INF;
    }
    return sa | (exp_field << 52) | (m & BN_F64_FRAC_MASK);
}

// `f64 + f64` を **round-to-odd**（RO）で丸めた版（PR #1671 codex-review
// 指摘・イシュー #1596 是正: affine `x̂·w+b` の二重丸め回避。冒頭コメント
// 「FMA 契約」参照）。[`bn_f64_add`] とは丸め規則のみが異なる姉妹関数
// （逐語複製。共通化すると分岐が増え可読性が落ちるため意図的に複製する。
// `soft_f64::add_f64_bits_round_to_odd` の逐語移植——数学的根拠
// 〈Boldo–Melquiond の round-to-odd 二重丸め定理〉は同関数の doc comment
// を正としここでは繰り返さない）。
//
// [`bn_f64_add`] との差分は最終段のみ: 「ガード 3bit `r` から `r>4` は
// 切り上げ・`r==4` は偶数丸め」ではなく「`r != 0`（丸め落ちする情報が
// 何かあれば）なら結果の最下位 bit を強制的に 1 にする」。round-to-odd
// は算術的な繰り上がり（`m += 1`）を一切行わない（ビット単位の OR のみ）
// ため、[`bn_f64_add`] が丸め後に持つ「`m` が `2^53` へ繰り上がる場合の
// 指数調整」分岐は発生しえず、本関数には存在しない。
inline ulong bn_f64_add_ro(ulong a, ulong b) {
    ulong sa = a & BN_F64_SIGN;
    ulong sb = b & BN_F64_SIGN;
    ulong ea = (a >> 52) & BN_F64_EXP_MASK;
    ulong eb = (b >> 52) & BN_F64_EXP_MASK;
    ulong fa = a & BN_F64_FRAC_MASK;
    ulong fb = b & BN_F64_FRAC_MASK;

    if (ea == BN_F64_EXP_MASK || eb == BN_F64_EXP_MASK) {
        bool a_nan = (ea == BN_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == BN_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return BN_F64_QNAN;
        }
        if (ea == BN_F64_EXP_MASK && eb == BN_F64_EXP_MASK) {
            return (sa == sb) ? a : BN_F64_QNAN;
        }
        return (ea == BN_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)bn_f64_clz64(m);
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
    if (exp_field >= BN_F64_EXP_MASK) {
        return sa | BN_F64_INF;
    }
    return sa | (exp_field << 52) | (m & BN_F64_FRAC_MASK);
}

// `a - b` = [`bn_f64_add`]`(a, `[`bn_f64_neg`]`(b))`。
inline ulong bn_f64_sub(ulong a, ulong b) {
    return bn_f64_add(a, bn_f64_neg(b));
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・underflow は f32
// subnormal／`±0`。NaN は quiet NaN へ正規化）。`soft_f64::narrow_f64_bits`。
inline uint bn_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & BN_F64_EXP_MASK;
    ulong f = bits & BN_F64_FRAC_MASK;
    if (e == BN_F64_EXP_MASK) {
        return (f != 0ul) ? BN_F32_QNAN : (sign | BN_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | BN_F32_INF;
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
        return sign | BN_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// `u64 x u64` の厳密な 128bit 積（32bit 分割のスクールブック乗算。
// `soft_f64::mul64_wide` の逐語移植）。
inline BnU128 bn_f64_mul64_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFul) + (lo_hi & 0xFFFFFFFFul);
    BnU128 result;
    result.lo = (lo_lo & 0xFFFFFFFFul) | (mid << 32);
    result.hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    return result;
}

// `(hi,lo)` を右シフト `s`（`0..=128`）した値（`soft_f64::shr128`）。
inline BnU128 bn_f64_shr128(ulong hi, ulong lo, uint s) {
    BnU128 r;
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
inline BnU128 bn_f64_low_bits128(ulong hi, ulong lo, uint n) {
    BnU128 r;
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
inline int bn_f64_cmp128(ulong ahi, ulong alo, ulong bhi, ulong blo) {
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
inline BnNormMantissa bn_f64_normalize_mantissa(ulong e, ulong f) {
    BnNormMantissa r;
    if (e == 0ul) {
        uint lead = 63u - bn_f64_clz64(f);
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
// 正規化）。仮数同士の厳密 106bit 積を [`bn_f64_mul64_wide`] で構成し、
// **1 回だけ**丸める（中間で `f32` はもとより暫定 `f64` へも丸めない。
// 二重丸め回避）。`soft_f64::mul_f64_bits` の逐語移植。
inline ulong bn_f64_mul(ulong a, ulong b) {
    ulong sa = a & BN_F64_SIGN;
    ulong sb = b & BN_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & BN_F64_EXP_MASK;
    ulong eb = (b >> 52) & BN_F64_EXP_MASK;
    ulong fa = a & BN_F64_FRAC_MASK;
    ulong fb = b & BN_F64_FRAC_MASK;

    bool a_nan = (ea == BN_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == BN_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return BN_F64_QNAN;
    }
    bool a_inf = (ea == BN_F64_EXP_MASK);
    bool b_inf = (eb == BN_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_zero && b_inf) || (a_inf && b_zero)) {
        return BN_F64_QNAN;
    }
    if (a_inf || b_inf) {
        return sign | BN_F64_INF;
    }
    if (a_zero || b_zero) {
        return sign;
    }

    BnNormMantissa na = bn_f64_normalize_mantissa(ea, fa);
    BnNormMantissa nb = bn_f64_normalize_mantissa(eb, fb);
    BnU128 p = bn_f64_mul64_wide(na.m, nb.m);
    // `na.m, nb.m ∈ [2^52, 2^53)` のため積は常に `[2^104, 2^106)`。
    // よって `p.hi` は常に非ゼロ（bit104 以上は `hi` 側〈bit64 以降〉に
    // 属する）。
    uint leadpos = 64u + (63u - bn_f64_clz64(p.hi));
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
    BnU128 q = bn_f64_shr128(p.hi, p.lo, final_shift);
    ulong m = q.lo; // `q.hi` は常に 0（呼び出し前提の値域より）。
    BnU128 rem = bn_f64_low_bits128(p.hi, p.lo, final_shift);
    ulong half_hi = 0ul;
    ulong half_lo = 0ul;
    if (final_shift > 0u) {
        if (final_shift - 1u < 64u) {
            half_lo = 1ul << (final_shift - 1u);
        } else {
            half_hi = 1ul << (final_shift - 1u - 64u);
        }
    }
    int cmp = bn_f64_cmp128(rem.hi, rem.lo, half_hi, half_lo);
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
        if (biased_final >= (long)BN_F64_EXP_MASK) {
            return sign | BN_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & BN_F64_FRAC_MASK);
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
// [`bn_f64_div`] 専用ヘルパー。
inline void bn_f64_div64_wide(ulong hi, ulong lo, ulong d, thread ulong &quotient_out, thread ulong &remainder_out) {
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
inline ulong bn_f64_div(ulong a, ulong b) {
    ulong sa = a & BN_F64_SIGN;
    ulong sb = b & BN_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & BN_F64_EXP_MASK;
    ulong eb = (b >> 52) & BN_F64_EXP_MASK;
    ulong fa = a & BN_F64_FRAC_MASK;
    ulong fb = b & BN_F64_FRAC_MASK;

    bool a_nan = (ea == BN_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == BN_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return BN_F64_QNAN;
    }
    bool a_inf = (ea == BN_F64_EXP_MASK);
    bool b_inf = (eb == BN_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_inf && b_inf) || (a_zero && b_zero)) {
        return BN_F64_QNAN;
    }
    if (a_inf) {
        return sign | BN_F64_INF;
    }
    if (b_inf) {
        return sign;
    }
    if (b_zero) {
        return sign | BN_F64_INF;
    }
    if (a_zero) {
        return sign;
    }

    BnNormMantissa na = bn_f64_normalize_mantissa(ea, fa);
    BnNormMantissa nb = bn_f64_normalize_mantissa(eb, fb);
    // `S = 55`: `ma/mb ∈ (0.5,2)` のため商は `[2^54,2^56)` に収まり、
    // 53bit 仮数 + 2bit（guard/round）の精度が確保できる最小の追加
    // シフト量（`soft_f64::div_f64_bits` と同じ定数）。
    const uint S = 55u;
    BnU128 num = bn_f64_mul64_wide(na.m, 1ul << S);
    ulong raw_q;
    ulong rem;
    bn_f64_div64_wide(num.hi, num.lo, nb.m, raw_q, rem);
    ulong q = raw_q | (ulong)(rem != 0ul ? 1ul : 0ul);
    // `q` は非ゼロ（呼び出し前提より `na.m`／`nb.m` はいずれも非ゼロ）。
    uint leadpos = 63u - bn_f64_clz64(q);
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
        if (biased_final >= (long)BN_F64_EXP_MASK) {
            return sign | BN_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & BN_F64_FRAC_MASK);
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
inline ulong bn_f64_scale_pow2(ulong x_bits, long k) {
    ulong sign = x_bits & BN_F64_SIGN;
    long e = (long)((x_bits >> 52) & BN_F64_EXP_MASK);
    ulong f = x_bits & BN_F64_FRAC_MASK;
    long new_e = e + k;
    if (new_e >= (long)BN_F64_EXP_MASK) {
        return sign | BN_F64_INF;
    }
    if (new_e <= 0l) {
        return sign;
    }
    return sign | (((ulong)new_e) << 52) | f;
}

// 仮数・指数を分離した種抽出（`soft_f64::extract_reduced_mantissa_and_exp`）。
// `want_sqrt_range == false`: `reduced ∈ [1,2)`・`exp_out = exp_u`
// （[`bn_f64_recip_newton`] 用）。`want_sqrt_range == true`: `reduced ∈
// [1,4)`（指数の偶奇に応じて範囲を揃える）・`exp_out = floor(exp_u/2)`
// 相当（[`bn_f64_rsqrt_newton`] 用）。
struct BnReducedSeed {
    uint reduced_bits;
    long exp_out;
};

inline BnReducedSeed bn_f64_extract_reduced_and_exp(ulong x_bits, bool want_sqrt_range) {
    ulong e = (x_bits >> 52) & BN_F64_EXP_MASK;
    ulong f = x_bits & BN_F64_FRAC_MASK;
    BnNormMantissa n = bn_f64_normalize_mantissa(e, f);
    BnReducedSeed r;
    if (!want_sqrt_range) {
        ulong reduced_bits = (1023ul << 52) | (n.m & BN_F64_FRAC_MASK);
        r.reduced_bits = bn_f64_narrow(reduced_bits);
        r.exp_out = n.exp_u;
    } else if ((n.exp_u & 1l) == 0l) {
        // `exp_u` が偶数（負値の剰余は `& 1` で符号に依らず 0/1 が出る）。
        ulong reduced_bits = (1023ul << 52) | (n.m & BN_F64_FRAC_MASK);
        r.reduced_bits = bn_f64_narrow(reduced_bits);
        r.exp_out = n.exp_u >> 1; // 偶数の算術右シフトは厳密な /2。
    } else {
        ulong reduced_bits = (1024ul << 52) | (n.m & BN_F64_FRAC_MASK);
        r.reduced_bits = bn_f64_narrow(reduced_bits);
        r.exp_out = (n.exp_u - 1l) >> 1;
    }
    return r;
}

// `f64` の逆数 `1/x` を Newton-Raphson（`y_{n+1} = y_n*(2 - x*y_n)`）で
// 求める（`x` は本カーネルの用途上〈`hidden` 由来〉常に有限・正の値。
// `soft_f64::recip_newton_f64_bits` の逐語移植）。**`mean`／`var` の
// 計算では現在使わない**（PR #1671 是正: Newton 近似逆数との積は
// 一様行で 1 ULP 誤差が悪化するため `bn_f64_div`〈正しく丸めた除算〉へ
// 置き換えた。イシュー #1596）。ホスト側 `recip_newton_f64_bits` と
// 1 対 1 対応する soft-f64 プリミティブとして、収束精度の回帰テスト
// （`soft_f64::tests::recip_newton_converges_for_hidden_range`）と
// ともに残置する（他ファイルの未使用だが残置されている診断・将来用
// 関数群と同じ方針。`context.rs::BatchGpuTimestamps` 等）。
inline ulong bn_f64_recip_newton(ulong x) {
    BnReducedSeed seed = bn_f64_extract_reduced_and_exp(x, false);
    float seed_reduced = 1.0f / as_type<float>(seed.reduced_bits);
    ulong y = bn_f64_scale_pow2(bn_f64_widen(as_type<uint>(seed_reduced)), -seed.exp_out);
    const ulong TWO = 0x4000000000000000ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong xy = bn_f64_mul(x, y);
        ulong two_minus_xy = bn_f64_sub(TWO, xy);
        y = bn_f64_mul(y, two_minus_xy);
    }
    return y;
}

// `f64` の逆数平方根 `1/sqrt(x)` を Newton-Raphson（`y_{n+1} =
// y_n*(1.5 - 0.5*x*y_n^2)`）で求める。特殊値は明示的に扱う（`x` は
// 分散 `+ eps`〈ともに非負〉由来で数学的に非負のはずだが、防御的に
// 負値も NaN として扱う）: `NaN -> NaN`・`±0 -> ±inf`・負（非ゼロ）
// `-> NaN`・`+inf -> +0`。`soft_f64::rsqrt_newton_f64_bits` の逐語移植。
inline ulong bn_f64_rsqrt_newton(ulong x) {
    ulong e = (x >> 52) & BN_F64_EXP_MASK;
    ulong f = x & BN_F64_FRAC_MASK;
    ulong sign = x & BN_F64_SIGN;
    if (e == BN_F64_EXP_MASK && f != 0ul) {
        return BN_F64_QNAN;
    }
    if (e == 0ul && f == 0ul) {
        return sign | BN_F64_INF;
    }
    if (sign != 0ul) {
        return BN_F64_QNAN;
    }
    if (e == BN_F64_EXP_MASK) {
        return 0ul; // +inf -> +0
    }

    BnReducedSeed seed = bn_f64_extract_reduced_and_exp(x, true);
    float seed_reduced = 1.0f / sqrt(as_type<float>(seed.reduced_bits));
    ulong y = bn_f64_scale_pow2(bn_f64_widen(as_type<uint>(seed_reduced)), -seed.exp_out);
    const ulong ONE_HALF = 0x3FE0000000000000ul;
    const ulong THREE_HALF = 0x3FF8000000000000ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong y2 = bn_f64_mul(y, y);
        ulong xy2 = bn_f64_mul(x, y2);
        ulong half_xy2 = bn_f64_mul(ONE_HALF, xy2);
        ulong inner = bn_f64_sub(THREE_HALF, half_xy2);
        y = bn_f64_mul(y, inner);
    }
    return y;
}

// BatchNorm1d／2d train モードカーネル（冒頭コメント参照）。
//
// 引数: `x`（`[n, c, spatial]` 行優先平坦化済み）・`w`／`b`
// （`has_weight`／`has_bias == 0` の場合も `c` 要素のダミーバッファを
// 呼び出し元が必ず渡す契約。`layer_norm_f32` と同じ predicated load
// 対策）・`out`・`mean_out`／`var_out`（`BatchNormTrainOutput::
// batch_mean`／`batch_var` 用。長さ `c`）・`n`・`c`・`spatial`・`m`
// （`n*spatial`。ループ終端）・`m_f64_hi`／`m_f64_lo`（`M` の f64
// 表現の厳密なビット渡し。冒頭コメント「`M` の f64 表現」参照）・
// `eps`・`has_weight`・`has_bias`・`grid_size`（persistent grid の
// threadgroup 数。ホスト側 `row_kernel::derive_persistent_grid` が
// 導出）。
kernel void batch_norm_train_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* mean_out [[buffer(4)]],
    device float* var_out [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& c [[buffer(7)]],
    constant uint& spatial [[buffer(8)]],
    constant uint& m [[buffer(9)]],
    constant uint& m_f64_hi [[buffer(10)]],
    constant uint& m_f64_lo [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    constant int& has_weight [[buffer(13)]],
    constant int& has_bias [[buffer(14)]],
    constant uint& grid_size [[buffer(15)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    // `n` はカーネル本体では使わない（`BN_IDX` は `spatial`／`c` のみで
    // 添字を導出できる。バッファレイアウト互換のため引数自体は残す。
    // `kernels_batch_norm.rs::batch_norm_train_f32` と同じ設計）。
    (void)n;

    // チャネル `ch` の局所添字 `i` を実データ添字へ写像する
    // （`backend-cpu::batch_norm::channel_index` の GPU 側複製。
    // `ulong` 演算で `n*c*spatial` の乗算オーバーフローを避ける。
    // REQ-8）。
    #define BN_IDX(i) ((ulong)((i) / spatial) * (ulong)c * (ulong)spatial \
                       + (ulong)ch * (ulong)spatial + (ulong)((i) % spatial))

    ulong m_f64 = (((ulong)m_f64_hi) << 32) | (ulong)m_f64_lo;

    for (ulong ch = (ulong)tg_id; ch < (ulong)c; ch += (ulong)grid_size) {
        // パス 1: 平均（soft-f64 総和。冒頭コメント「総和の順序」参照）。
        ulong lane_sum = 0ul; // +0.0（f64）。
        for (ulong i = (ulong)lane; i < (ulong)m; i += (ulong)BATCH_NORM_SIMD_WIDTH) {
            ulong xv = bn_f64_widen(as_type<uint>(x[BN_IDX(i)]));
            lane_sum = bn_f64_add(lane_sum, xv);
        }
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            ulong other_hi = simd_shuffle_xor((uint)(lane_sum >> 32), offset);
            ulong other_lo = simd_shuffle_xor((uint)lane_sum, offset);
            ulong other_sum = (other_hi << 32) | other_lo;
            lane_sum = bn_f64_add(lane_sum, other_sum);
        }
        ulong mean = bn_f64_div(lane_sum, m_f64);

        // パス 2: 分散（二パス。`(x-mean)^2` を soft-f64 で蓄積する）。
        ulong lane_sq = 0ul;
        for (ulong i = (ulong)lane; i < (ulong)m; i += (ulong)BATCH_NORM_SIMD_WIDTH) {
            ulong xv = bn_f64_widen(as_type<uint>(x[BN_IDX(i)]));
            ulong dev = bn_f64_sub(xv, mean);
            ulong devsq = bn_f64_mul(dev, dev);
            lane_sq = bn_f64_add(lane_sq, devsq);
        }
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            ulong other_hi = simd_shuffle_xor((uint)(lane_sq >> 32), offset);
            ulong other_lo = simd_shuffle_xor((uint)lane_sq, offset);
            ulong other_sq = (other_hi << 32) | other_lo;
            lane_sq = bn_f64_add(lane_sq, other_sq);
        }
        ulong var = bn_f64_div(lane_sq, m_f64);
        ulong eps_f64 = bn_f64_widen(as_type<uint>(eps));
        ulong var_plus_eps = bn_f64_add(var, eps_f64);
        ulong rstd = bn_f64_rsqrt_newton(var_plus_eps);

        if (lane == 0u) {
            mean_out[ch] = as_type<float>(bn_f64_narrow(mean));
            var_out[ch] = as_type<float>(bn_f64_narrow(var));
        }

        // パス 3: 書き出し（device メモリを再読）。`xhat` を soft-f64 で
        // 確定した後 1 回だけ `f32` へ丸め、affine は round-to-odd
        // 経由で単一丸めの FMA と同値の結果を得る（冒頭コメント
        // 「FMA 契約」参照）。
        float wv = (has_weight != 0) ? w[ch] : 1.0f;
        float bv = (has_bias != 0) ? b[ch] : 0.0f;
        ulong wv64 = bn_f64_widen(as_type<uint>(wv));
        ulong bv64 = bn_f64_widen(as_type<uint>(bv));
        for (ulong i = (ulong)lane; i < (ulong)m; i += (ulong)BATCH_NORM_SIMD_WIDTH) {
            ulong idx = BN_IDX(i);
            ulong xv = bn_f64_widen(as_type<uint>(x[idx]));
            ulong dev = bn_f64_sub(xv, mean);
            ulong xhat64 = bn_f64_mul(dev, rstd);
            uint xhat_bits = bn_f64_narrow(xhat64);
            ulong affine64 = bn_f64_add_ro(bn_f64_mul(bn_f64_widen(xhat_bits), wv64), bv64);
            out[idx] = as_type<float>(bn_f64_narrow(affine64));
        }
    }

    #undef BN_IDX
}

// BatchNorm1d／2d infer モードカーネル（冒頭コメント参照）。統計を
// 再計算しないため縮約が不要（grid-stride 不要の単純 elementwise。
// `if (gid >= numel) return;` の手動境界検査。REQ-8）。
//
// 引数: `x`（`[n, c, spatial]` 行優先平坦化済み）・`mean`／`var`
// （呼び出し元が保持する running stats。長さ `c`）・`w`／`b`
// （`batch_norm_train_f32` と同じダミーバッファ契約）・`out`・`c`・
// `spatial`・`numel`（`c*spatial*n`。`n` 自体はカーネル内で使わない
// ため引数に含めない。`kernels_batch_norm.rs::batch_norm_infer_f32`
// と同じ設計）・`eps`・`has_weight`・`has_bias`。
kernel void batch_norm_infer_f32(
    device const float* x [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* var [[buffer(2)]],
    device const float* w [[buffer(3)]],
    device const float* b [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& c [[buffer(6)]],
    constant uint& spatial [[buffer(7)]],
    constant uint& numel [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant int& has_weight [[buffer(10)]],
    constant int& has_bias [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) {
        return;
    }
    ulong idx = (ulong)gid;
    uint ch = (uint)((idx / (ulong)spatial) % (ulong)c);

    ulong mean64 = bn_f64_widen(as_type<uint>(mean[ch]));
    ulong var64 = bn_f64_widen(as_type<uint>(var[ch]));
    ulong eps64 = bn_f64_widen(as_type<uint>(eps));
    ulong rstd = bn_f64_rsqrt_newton(bn_f64_add(var64, eps64));

    ulong xv = bn_f64_widen(as_type<uint>(x[idx]));
    ulong dev = bn_f64_sub(xv, mean64);
    ulong xhat64 = bn_f64_mul(dev, rstd);
    uint xhat_bits = bn_f64_narrow(xhat64);

    float wv = (has_weight != 0) ? w[ch] : 1.0f;
    float bv = (has_bias != 0) ? b[ch] : 0.0f;
    ulong wv64 = bn_f64_widen(as_type<uint>(wv));
    ulong bv64 = bn_f64_widen(as_type<uint>(bv));
    ulong affine64 = bn_f64_add_ro(bn_f64_mul(bn_f64_widen(xhat_bits), wv64), bv64);
    out[idx] = as_type<float>(bn_f64_narrow(affine64));
}
