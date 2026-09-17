// `log_softmax` backward（`dx = g − exp(y)·Σ_dim(g)`。イシュー #1952・
// 親 #1947「GPU ホストフォールバック残存演算の専用カーネル化」）。
//
// `crate::log_softmax_backward::MetalLogSoftmaxBackward`（`log_softmax_
// backward.rs`）から実行時コンパイルされる。`ops.rs::MetalBackendOps::
// log_softmax_backward` は `context_cache::cached_log_softmax_backward`
// 経由で本カーネルへ到達し（`Op::LogSoftmax` の VJP から
// `BackendOps::log_softmax_backward` 優先で呼ばれる）、`Unsupported`
// のときのみホスト参照実装（`grad::log_softmax_vjp_along`）へ
// フォールバックする。ホストモデルは `crate::log_softmax_backward_model`。
//
// ---- 数値方式（binary64 ソフトウェアエミュレーション）----
//
// MSL は `double` 型非対応のため、`.claude/rules/coding-rust.md`
// 「正規化統計・勾配の長軸縮約は `f64` アキュムレータで統一する」節・
// `crate::soft_f64` と同じ binary64 逐次演算の 64bit 整数ソフトウェア
// エミュレーションを適用する（`lsb_f64_*`。`layer_norm.metal::
// ln_f64_*`・`reduce.metal::red_f64_*` の逐語移植——MSL は
// `newLibraryWithSource` でファイル単位にコンパイルされ翻訳単位を
// 共有できないため、`lsb_f64_` 接頭辞を付けた本ファイル内で独立に
// 定義する〈意図的な重複〉。ホスト側の逐語モデルは `crate::soft_f64`／
// `crate::log_softmax_backward_model` を参照）。
//
// ホスト参照実装（`fandhe_ai_autodiff::grad::log_softmax_vjp_along`）は
// 次の演算列で計算する:
//
//   1. `Σ_dim(g)`: `dim` 軸を `0.0f64` から index 昇順に
//      「`f32→f64` ウィデン → `f64` 加算」で逐次和する。
//   2. 各要素について `e = y[idx].exp()`（**`f32` 精度**で計算してから
//      `f64` へウィデン）・`term = e_f64 * sum_f64`（`f64` 乗算・1 回の
//      丸め）・`d = g[idx]_f64 - term`（`f64` 減算・1 回の丸め）を計算し
//      最後に 1 回だけ `f32` へ downcast する（乗算と減算は Rust の
//      既定動作どおり FMA 縮約されない 2 段）。
//
// 本ファイルはこの演算列を 2 カーネルへ写像する:
//   1. `log_softmax_bwd_lane_sum`: 1 thread = 1 lane（`gid` は出力要素
//      〈`outer×inner` 個〉の平坦添字。`reduce.metal::
//      reduce_sum_axis_f32` と同じ添字規約）。`widen(0.0)` から昇順に
//      `lsb_f64_add` で逐次加算し、`narrow` せず `f64` bit（`ulong`）
//      のまま `lane_sum[gid]` へ書く（中間値を `f32` へ丸めてから
//      次段へ渡すと二重丸めで契約が崩れるため。`reduce.metal::
//      reduce_sum_all_chunk_f32` の `partial` と同じ設計判断）。
//   2. `log_softmax_bwd_apply_f32`: 1 thread = 1 要素（`gid` は
//      `y`／`g`／`out` の平坦添字。`numel = outer*axis_len*inner`）。
//      `e = precise::exp(y[gid])`（`f32` 精度）→ `widen` → `lsb_f64_mul`
//      （`lane_sum` との積）→ `lsb_f64_sub`（`widen(g[gid])` から減算）
//      → `lsb_f64_narrow` で 1 回だけ `f32` へ downcast する。
//
// **bit 一致を主張する範囲**は (1) の縮約と (2) の `f64` 連鎖
// （ウィデン → 乗算 → 減算 → narrow）のみであり、**`exp(y)` 自体の
// 丸めは bit 一致を主張しない**（Metal `precise::exp` とホスト
// `f32::exp` の丸めは規格上一致が保証されない）。よって実機の最終
// 出力（`exp` の丸め差を含む）は REQ-2 統一複合判定（相対誤差 1e-3
// 未満 または 絶対誤差 1e-5 未満）で検証する契約（`crate::
// log_softmax_backward_model` モジュール doc・
// `docs/backend-metal-reduce-sum-design.md` 追補節参照）。
// `mathMode=Safe`（`pipeline::compile_options`）だけでは超越関数は
// precise にならないため、`shaders/bce.metal` と同様に `precise::exp`
// を明示的に使う。
//
// ---- 既知の制約（並列度。`.claude/rules/out-of-scope-tracking.md`
// 対象）----
//
// `reduce.metal` モジュール doc と同じ理由: `dim` 方向の対数段
// reduction は結合順序が変わり bit 一致契約が崩れるため採らない。
// したがって `log_softmax_bwd_lane_sum` は `outer*inner` スレッドのみ
// （`axis_len` 方向は当該スレッド内で逐次）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// 各カーネルは `gid`（グローバルスレッド id）が起動対象範囲
// （`lanes`／`numel`）未満かどうかを検査してから処理する。添字計算は
// `ulong`（64bit）で行う。

#include <metal_stdlib>
using namespace metal;

// ---- IEEE 754 binary64 のソフトウェアエミュレーション ----
// `crates/backend-metal/src/soft_f64.rs`（`widen_f32_bits`／
// `add_f64_bits`／`sub_f64_bits`／`mul_f64_bits`／`narrow_f64_bits`）の
// 逐語移植（本ファイル冒頭コメント参照。`layer_norm.metal::ln_f64_*`
// と同一アルゴリズムの複製）。

#define LSB_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define LSB_F64_EXP_MASK  0x7FFul
#define LSB_F64_SIGN      0x8000000000000000ul
#define LSB_F64_QNAN      0x7FF8000000000000ul
#define LSB_F64_INF       0x7FF0000000000000ul
#define LSB_F32_QNAN      0x7FC00000u
#define LSB_F32_INF       0x7F800000u

// 128bit 値（`hi:lo`）を表現する小さな構造体（MSL に `u128`／タプルが
// ないため。`ln_f64_mul64_wide` 系と同型。[`lsb_f64_mul`] 専用）。
struct LsbU128 {
    ulong hi;
    ulong lo;
};

// 正規化済み仮数 `m`（隠れ 1 を bit52 に立てた 53bit 値）と、
// `value = m * 2^(exp_u - 52)` を満たす unbiased 指数 `exp_u` の対
// （[`lsb_f64_normalize_mantissa`] の戻り値）。
struct LsbNormMantissa {
    ulong m;
    long exp_u;
};

// 64bit leading zero count（`clz(0u) == 32` は MSL 仕様で定義済み。
// `soft_f64::clz64` と同一構造）。
inline uint lsb_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`
// の逐語移植。
inline ulong lsb_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? LSB_F64_QNAN : (sign | LSB_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        // f32 subnormal（`frac × 2^-149`）は f64 では正規化数。
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & LSB_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul); // 減算を先にすると exp < 127 で下溢れ。
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64` の符号反転。NaN も含め符号 bit を無条件に反転する
// （`soft_f64::neg_f64_bits`）。
inline ulong lsb_f64_neg(ulong a) {
    return a ^ LSB_F64_SIGN;
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` の逐語移植。
inline ulong lsb_f64_add(ulong a, ulong b) {
    ulong sa = a & LSB_F64_SIGN;
    ulong sb = b & LSB_F64_SIGN;
    ulong ea = (a >> 52) & LSB_F64_EXP_MASK;
    ulong eb = (b >> 52) & LSB_F64_EXP_MASK;
    ulong fa = a & LSB_F64_FRAC_MASK;
    ulong fb = b & LSB_F64_FRAC_MASK;

    if (ea == LSB_F64_EXP_MASK || eb == LSB_F64_EXP_MASK) {
        bool a_nan = (ea == LSB_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == LSB_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return LSB_F64_QNAN;
        }
        if (ea == LSB_F64_EXP_MASK && eb == LSB_F64_EXP_MASK) {
            return (sa == sb) ? a : LSB_F64_QNAN;
        }
        return (ea == LSB_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)lsb_f64_clz64(m);
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
    if (exp_field >= LSB_F64_EXP_MASK) {
        return sa | LSB_F64_INF;
    }
    return sa | (exp_field << 52) | (m & LSB_F64_FRAC_MASK);
}

// `a - b` = `lsb_f64_add(a, lsb_f64_neg(b))`（`soft_f64::sub_f64_bits`）。
inline ulong lsb_f64_sub(ulong a, ulong b) {
    return lsb_f64_add(a, lsb_f64_neg(b));
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・NaN は quiet NaN
// へ正規化）。`soft_f64::narrow_f64_bits` の逐語移植。
inline uint lsb_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & LSB_F64_EXP_MASK;
    ulong f = bits & LSB_F64_FRAC_MASK;
    if (e == LSB_F64_EXP_MASK) {
        return (f != 0ul) ? LSB_F32_QNAN : (sign | LSB_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | LSB_F32_INF;
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
        return sign | LSB_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// `u64 x u64` の厳密な 128bit 積（32bit 分割のスクールブック乗算。
// `soft_f64::mul64_wide`／`ln_f64_mul64_wide` の逐語移植）。
inline LsbU128 lsb_f64_mul64_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFul) + (lo_hi & 0xFFFFFFFFul);
    LsbU128 result;
    result.lo = (lo_lo & 0xFFFFFFFFul) | (mid << 32);
    result.hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    return result;
}

// `(hi,lo)` を右シフト `s`（`0..=128`）した値（`soft_f64::shr128`）。
inline LsbU128 lsb_f64_shr128(ulong hi, ulong lo, uint s) {
    LsbU128 r;
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
inline LsbU128 lsb_f64_low_bits128(ulong hi, ulong lo, uint n) {
    LsbU128 r;
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
inline int lsb_f64_cmp128(ulong ahi, ulong alo, ulong bhi, ulong blo) {
    if (ahi != bhi) {
        return (ahi < bhi) ? -1 : 1;
    }
    if (alo != blo) {
        return (alo < blo) ? -1 : 1;
    }
    return 0;
}

// `f64` の指数・仮数フィールドから「隠れ 1 を bit52 に立てた 53bit
// 仮数」と `value = m * 2^(exp_u-52)` の unbiased 指数の対を求める
// （`e==0 && f==0` の完全ゼロは呼び出し側で排除済みの前提。
// `soft_f64::normalize_f64_mantissa`）。
inline LsbNormMantissa lsb_f64_normalize_mantissa(ulong e, ulong f) {
    LsbNormMantissa r;
    if (e == 0ul) {
        uint lead = 63u - lsb_f64_clz64(f);
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
// 正規化）。仮数同士の厳密 106bit 積を `lsb_f64_mul64_wide` で構成し、
// **1 回だけ**丸める（二重丸め回避）。`soft_f64::mul_f64_bits` の
// 逐語移植（`layer_norm.metal::ln_f64_mul` と同一アルゴリズム）。
inline ulong lsb_f64_mul(ulong a, ulong b) {
    ulong sa = a & LSB_F64_SIGN;
    ulong sb = b & LSB_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & LSB_F64_EXP_MASK;
    ulong eb = (b >> 52) & LSB_F64_EXP_MASK;
    ulong fa = a & LSB_F64_FRAC_MASK;
    ulong fb = b & LSB_F64_FRAC_MASK;

    bool a_nan = (ea == LSB_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == LSB_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return LSB_F64_QNAN;
    }
    bool a_inf = (ea == LSB_F64_EXP_MASK);
    bool b_inf = (eb == LSB_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_zero && b_inf) || (a_inf && b_zero)) {
        return LSB_F64_QNAN;
    }
    if (a_inf || b_inf) {
        return sign | LSB_F64_INF;
    }
    if (a_zero || b_zero) {
        return sign;
    }

    LsbNormMantissa na = lsb_f64_normalize_mantissa(ea, fa);
    LsbNormMantissa nb = lsb_f64_normalize_mantissa(eb, fb);
    LsbU128 p = lsb_f64_mul64_wide(na.m, nb.m);
    // `na.m, nb.m ∈ [2^52, 2^53)` のため積は常に `[2^104, 2^106)`。
    // よって `p.hi` は常に非ゼロ（bit104 以上は `hi` 側〈bit64 以降〉に
    // 属する）。
    uint leadpos = 64u + (63u - lsb_f64_clz64(p.hi));
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
    LsbU128 q = lsb_f64_shr128(p.hi, p.lo, final_shift);
    ulong m = q.lo; // `q.hi` は常に 0（呼び出し前提の値域より）。
    LsbU128 rem = lsb_f64_low_bits128(p.hi, p.lo, final_shift);
    ulong half_hi = 0ul;
    ulong half_lo = 0ul;
    if (final_shift > 0u) {
        if (final_shift - 1u < 64u) {
            half_lo = 1ul << (final_shift - 1u);
        } else {
            half_hi = 1ul << (final_shift - 1u - 64u);
        }
    }
    int cmp = lsb_f64_cmp128(rem.hi, rem.lo, half_hi, half_lo);
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
        if (biased_final >= (long)LSB_F64_EXP_MASK) {
            return sign | LSB_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & LSB_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

// ---- カーネル 1: lane 単位の Σ_dim(g)（narrow しない f64 bit）----
//
// `g`: `outer * axis_len * inner` 要素の稠密（contiguous）スライス
// （呼び出し元 `log_softmax_backward.rs::MetalLogSoftmaxBackward::
// run_f32` が保証）。`lane_sum`: `lanes`（`outer * inner`）要素の
// `ulong`（`f64` bit 表現。narrow しない中間値）。`inner` は `gid` を
// `o = gid / inner`・`i = gid % inner` へ分解するための除数
// （`reduce.metal::reduce_sum_axis_f32` と同じ添字規約）。

kernel void log_softmax_bwd_lane_sum(
    device const float* g [[buffer(0)]],
    device ulong* lane_sum [[buffer(1)]],
    constant uint& lanes [[buffer(2)]],
    constant uint& axis_len [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= lanes) {
        return;
    }
    ulong o = (ulong)gid / (ulong)inner;
    ulong i = (ulong)gid % (ulong)inner;
    ulong acc = lsb_f64_widen(as_type<uint>(0.0f));
    for (uint a = 0u; a < axis_len; a++) {
        ulong idx = (o * (ulong)axis_len + (ulong)a) * (ulong)inner + i;
        acc = lsb_f64_add(acc, lsb_f64_widen(as_type<uint>(g[idx])));
    }
    lane_sum[gid] = acc;
}

// ---- カーネル 2: 要素単位の dx = g − exp(y)·sum ----
//
// `y`／`g`／`out`: `numel`（`outer * axis_len * inner`）要素。`gid` は
// `y`／`g`／`out` の平坦添字そのもの（縮約軸の添字を含む）。`lane_sum`:
// カーネル 1 の出力（`lanes` 要素の `ulong`）。`axis_len`／`inner` から
// `gid` の属する lane（`o*inner+i`）を逆算する（`o = gid /
// (axis_len*inner)`・`i = gid % inner`。`(o*axis_len+a)*inner+i` の
// 添字構造より `gid / (axis_len*inner) == o`・`gid % inner == i` が
// 成り立つ——`a` の値に依存しない）。

kernel void log_softmax_bwd_apply_f32(
    device const float* y [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device const ulong* lane_sum [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    constant uint& axis_len [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) {
        return;
    }
    ulong axis_inner = (ulong)axis_len * (ulong)inner;
    ulong o = (ulong)gid / axis_inner;
    ulong i = (ulong)gid % (ulong)inner;
    ulong lane = o * (ulong)inner + i;

    float e = precise::exp(y[gid]);
    ulong e64 = lsb_f64_widen(as_type<uint>(e));
    ulong sum64 = lane_sum[lane];
    ulong term = lsb_f64_mul(e64, sum64);
    ulong g64 = lsb_f64_widen(as_type<uint>(g[gid]));
    ulong d = lsb_f64_sub(g64, term);
    out[gid] = as_type<float>(lsb_f64_narrow(d));
}
