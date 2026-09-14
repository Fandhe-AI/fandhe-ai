// 累積和／累積積（`torch.cumsum`／`torch.cumprod` 相当。イシュー
// #1740・親イシュー #1731。`crates/backend-cuda/src/kernels_scan.rs`
// の Metal 対応版。ホストモデルは `crate::scan_model`）。
//
// `crate::scan::MetalScan`（`scan.rs`）から実行時コンパイルされ、
// `ops.rs::MetalBackendOps::cumsum`／`cumprod` から呼ばれる。
//
// ---- 数値方式（binary64 ソフトウェアエミュレーション）----
//
// MSL は `double` 型非対応のため、`.claude/rules/coding-rust.md`
// 「勾配の長軸縮約」節・`crate::soft_f64` と同じ binary64 逐次演算の
// 64bit 整数ソフトウェアエミュレーションを forward の scan へ適用する
// （`scan_f64_widen`／`scan_f64_add`／`scan_f64_mul`／`scan_f64_narrow`。
// `layer_norm.metal::ln_f64_*` の逐語移植——MSL は
// `newLibraryWithSource` でファイル単位にコンパイルされ翻訳単位を
// 共有できないため、`scan_f64_` 接頭辞を付けた本ファイル内で独立に
// 定義する。意図的な重複。ホスト側の逐語モデルは `crate::soft_f64`／
// `crate::scan_model` を参照）。各カーネルは `f64` アキュムレータ相当
// （64bit 整数表現の `ulong acc`）を `widen(0.0)`（cumsum）／
// `widen(1.0)`（cumprod）から開始し、`axis_len` 回のループで
// `acc = scan_f64_add(acc, scan_f64_widen(x[idx]))`（cumprod は
// `scan_f64_mul`）を計算したうえで各ステップ即座に
// `out[idx] = scan_f64_narrow(acc)` を書く（次ステップは `out[idx]`
// を読み戻さない）。この演算列はホスト `f64` 逐次参照実装
// （`backend-cpu::scan::cumsum`／`cumprod`）と bit 完全一致する契約
// （NaN のみ payload がハードウェア依存のためクラス一致。`crate::
// scan_model` の単体テストが Linux で機械的に裏付ける）。
//
// ---- lane 逐次 scan（ブロック内並列化不可の理由）----
//
// `kernels_scan.rs` モジュール doc と同じ理由: ブロック内 scan
// アルゴリズム（対数段の結合）は `dim` 添字昇順の逐次加算・乗算列と
// 異なる結合順序になり bit 一致しなくなるため採らない。本カーネルは
// 「1 スレッド = 1 lane・`axis_len` 方向は当該スレッド内で逐次」と
// いう構造のみを取る（lane 数が少ない形状では並列度が出ない既知の
// 制約。`.claude/rules/out-of-scope-tracking.md` 対象）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// `gid`（グローバルスレッド id＝lane 番号）が `lanes` 未満かどうかを
// 検査してから処理する。添字計算は `ulong`（64bit）で行う。

#include <metal_stdlib>
using namespace metal;

// ---- IEEE 754 binary64 のソフトウェアエミュレーション ----
// `crates/backend-metal/src/soft_f64.rs` の逐語移植（本ファイル冒頭
// コメント参照）。

#define SCAN_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define SCAN_F64_EXP_MASK  0x7FFul
#define SCAN_F64_SIGN      0x8000000000000000ul
#define SCAN_F64_QNAN      0x7FF8000000000000ul
#define SCAN_F64_INF       0x7FF0000000000000ul
#define SCAN_F32_QNAN      0x7FC00000u
#define SCAN_F32_INF       0x7F800000u

struct ScanU128 {
    ulong hi;
    ulong lo;
};

struct ScanNormMantissa {
    ulong m;
    long exp_u;
};

// 64bit leading zero count（`clz(0u) == 32` は MSL 仕様で定義済み。
// `soft_f64::clz64` と同一構造）。
inline uint scan_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`。
inline ulong scan_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? SCAN_F64_QNAN : (sign | SCAN_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & SCAN_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul);
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` の逐語移植。
inline ulong scan_f64_add(ulong a, ulong b) {
    ulong sa = a & SCAN_F64_SIGN;
    ulong sb = b & SCAN_F64_SIGN;
    ulong ea = (a >> 52) & SCAN_F64_EXP_MASK;
    ulong eb = (b >> 52) & SCAN_F64_EXP_MASK;
    ulong fa = a & SCAN_F64_FRAC_MASK;
    ulong fb = b & SCAN_F64_FRAC_MASK;

    if (ea == SCAN_F64_EXP_MASK || eb == SCAN_F64_EXP_MASK) {
        bool a_nan = (ea == SCAN_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == SCAN_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return SCAN_F64_QNAN;
        }
        if (ea == SCAN_F64_EXP_MASK && eb == SCAN_F64_EXP_MASK) {
            return (sa == sb) ? a : SCAN_F64_QNAN;
        }
        return (ea == SCAN_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)scan_f64_clz64(m);
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
    if (exp_field >= SCAN_F64_EXP_MASK) {
        return sa | SCAN_F64_INF;
    }
    return sa | (exp_field << 52) | (m & SCAN_F64_FRAC_MASK);
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・NaN は quiet NaN
// へ正規化）。`soft_f64::narrow_f64_bits` の逐語移植。
inline uint scan_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & SCAN_F64_EXP_MASK;
    ulong f = bits & SCAN_F64_FRAC_MASK;
    if (e == SCAN_F64_EXP_MASK) {
        return (f != 0ul) ? SCAN_F32_QNAN : (sign | SCAN_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | SCAN_F32_INF;
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
        return sign | SCAN_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// `u64 x u64` の厳密な 128bit 積（`soft_f64::mul64_wide` の逐語移植）。
inline ScanU128 scan_f64_mul64_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFul) + (lo_hi & 0xFFFFFFFFul);
    ScanU128 result;
    result.lo = (lo_lo & 0xFFFFFFFFul) | (mid << 32);
    result.hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    return result;
}

// `(hi,lo)` を右シフト `s`（`0..=128`）した値（`soft_f64::shr128`）。
inline ScanU128 scan_f64_shr128(ulong hi, ulong lo, uint s) {
    ScanU128 r;
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
inline ScanU128 scan_f64_low_bits128(ulong hi, ulong lo, uint n) {
    ScanU128 r;
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
inline int scan_f64_cmp128(ulong ahi, ulong alo, ulong bhi, ulong blo) {
    if (ahi != bhi) {
        return (ahi < bhi) ? -1 : 1;
    }
    if (alo != blo) {
        return (alo < blo) ? -1 : 1;
    }
    return 0;
}

// `f64` の指数・仮数フィールドから「隠れ 1 を bit52 に立てた 53bit
// 仮数」と unbiased 指数の対を求める（`soft_f64::normalize_f64_mantissa`）。
inline ScanNormMantissa scan_f64_normalize_mantissa(ulong e, ulong f) {
    ScanNormMantissa r;
    if (e == 0ul) {
        uint lead = 63u - scan_f64_clz64(f);
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
// 正規化）。`soft_f64::mul_f64_bits` の逐語移植。
inline ulong scan_f64_mul(ulong a, ulong b) {
    ulong sa = a & SCAN_F64_SIGN;
    ulong sb = b & SCAN_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & SCAN_F64_EXP_MASK;
    ulong eb = (b >> 52) & SCAN_F64_EXP_MASK;
    ulong fa = a & SCAN_F64_FRAC_MASK;
    ulong fb = b & SCAN_F64_FRAC_MASK;

    bool a_nan = (ea == SCAN_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == SCAN_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return SCAN_F64_QNAN;
    }
    bool a_inf = (ea == SCAN_F64_EXP_MASK);
    bool b_inf = (eb == SCAN_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_zero && b_inf) || (a_inf && b_zero)) {
        return SCAN_F64_QNAN;
    }
    if (a_inf || b_inf) {
        return sign | SCAN_F64_INF;
    }
    if (a_zero || b_zero) {
        return sign;
    }

    ScanNormMantissa na = scan_f64_normalize_mantissa(ea, fa);
    ScanNormMantissa nb = scan_f64_normalize_mantissa(eb, fb);
    ScanU128 p = scan_f64_mul64_wide(na.m, nb.m);
    uint leadpos = 64u + (63u - scan_f64_clz64(p.hi));
    long exp_u = na.exp_u + nb.exp_u + ((long)leadpos - 104l);
    long shift_normal = (long)leadpos - 52l;

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
        // 到達性: 本カーネルの実用値域（`f32` 由来の入力）では発生
        // しない極端な underflow。安全側として `±0` へ丸める
        // （`layer_norm.metal::ln_f64_mul` と同じ方針）。
        return sign;
    }
    uint final_shift = (uint)final_shift_l;
    ScanU128 q = scan_f64_shr128(p.hi, p.lo, final_shift);
    ulong m = q.lo;
    ScanU128 rem = scan_f64_low_bits128(p.hi, p.lo, final_shift);
    ulong half_hi = 0ul;
    ulong half_lo = 0ul;
    if (final_shift > 0u) {
        if (final_shift - 1u < 64u) {
            half_lo = 1ul << (final_shift - 1u);
        } else {
            half_hi = 1ul << (final_shift - 1u - 64u);
        }
    }
    int cmp = scan_f64_cmp128(rem.hi, rem.lo, half_hi, half_lo);
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
        if (biased_final >= (long)SCAN_F64_EXP_MASK) {
            return sign | SCAN_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & SCAN_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

// ---- 累積和／累積積カーネル ----
//
// `x`／`out`: `outer * axis_len * inner` 要素（呼び出し元
// `scan.rs::MetalScan::run_cumsum_f32`／`run_cumprod_f32` が
// `contiguous()` で稠密化した入力）。`lanes = outer * inner`（`gid`
// の境界検査対象）。`inner` は `gid` を `o = gid / inner`・
// `i = gid % inner` へ分解するための除数（本ファイル冒頭コメント
// 「REQ-8」節参照）。

kernel void cumsum_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
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
    ulong acc = scan_f64_widen(as_type<uint>(0.0f));
    for (uint a = 0; a < axis_len; a++) {
        ulong idx = (o * (ulong)axis_len + (ulong)a) * (ulong)inner + i;
        acc = scan_f64_add(acc, scan_f64_widen(as_type<uint>(x[idx])));
        out[idx] = as_type<float>(scan_f64_narrow(acc));
    }
}

kernel void cumprod_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
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
    ulong acc = scan_f64_widen(as_type<uint>(1.0f));
    for (uint a = 0; a < axis_len; a++) {
        ulong idx = (o * (ulong)axis_len + (ulong)a) * (ulong)inner + i;
        acc = scan_f64_mul(acc, scan_f64_widen(as_type<uint>(x[idx])));
        out[idx] = as_type<float>(scan_f64_narrow(acc));
    }
}
