// MaxPool／AvgPool／AdaptiveAvgPool（1d／2d。イシュー #1730・親
// #1607。1d は呼び出し元〈`crate::pooling::MetalPooling`〉が
// `[N,C,1,L]` へ reshape 併合してから本カーネルへ渡す契約——本
// ファイルは常に rank 4 を扱う）。ホストモデルは `crate::
// pooling_model`。設計は `docs/pooling-ops-design.md`。
//
// ---- レイアウト ----
//
// `in: [N, C, H, W]`（NCHW 固定）を row-major で走査する。窓は
// `ih = oh*sh + kh*dh - ph`／`iw = ow*sw + kw*dw - pw`（`kh`/`kw` は
// カーネル内オフセット添字。窓外〈padding〉は寄与しない）。
//
// ---- 数値方式 ----
//
// `max_pool2d_f32` は選択演算のみ（算術を含まない）のため CPU 参照
// 実装と bit 完全一致。タイは先勝ち（`v > best` の厳密比較のみで
// 更新——同値では更新しない）・最初の有効タップで初期化し、NaN が
// 現れたら以後更新されず「最初に出現した NaN」の索引を保持したまま
// 確定する（`v.is_nan() && !best.is_nan()` の 1 回だけの遷移）。
//
// `avg_pool2d_f32`／`adaptive_avg_pool2d_f32` は窓内の有効タップを
// row-major に soft-f64（binary64 の 64bit 整数ソフトウェアエミュ
// レーション）アキュムレータへ逐次加算し、`divisor`（`u32` から
// 厳密変換した binary64 値）で soft-f64 除算してから 1 回だけ `f32`
// へ narrow する（`.claude/rules/coding-rust.md`「勾配の長軸縮約」
// 節と同型の binary64 ソフトウェアエミュレーション契約。加算・除算
// とも正しく丸められた binary64 のためハードウェア `f64` と一致し、
// CPU 参照実装〈ホスト `f64` 逐次和 → 除算 → 1 回 `f32` へ丸め〉と
// **bit 完全一致**する）。
//
// **soft-f64 プリミティブ（`pool_f64_*`）**: `crates/backend-metal/
// src/soft_f64.rs`／`shaders/batch_norm.metal::bn_f64_*` の逐語移植
// （`u64`→`ulong`・`u32`→`uint`・`leading_zeros()`→`clz()`）。MSL は
// `newLibraryWithSource` でファイル単位にコンパイルされ翻訳単位を
// 共有できないため、`pool_f64_` 接頭辞を付けた本ファイル内で独立に
// 定義する（`gemm.metal::bias_f64_*`・`layer_norm.metal::ln_f64_*`・
// `batch_norm.metal::bn_f64_*`・`im2col.metal::im2col_f64_*` に続く
// 意図的な複製）。本ファイルの `pool_f64_*` は本演算（Avg 系の
// 逐次加算・除算のみ）に必要な最小集合（`clz64`／`widen`／`add`／
// `narrow`／`mul64_wide`／`normalize_mantissa`／`div64_wide`／
// `div`）に限定し、`batch_norm.metal::bn_f64_*` の対応関数本体と
// 接頭辞以外で逐語一致する（`tests/pooling_source_evidence.rs` の
// ドリフトガードが固定する）。
//
// `pool_f64_from_uint(uint v)` は `u32 -> binary64` の厳密変換
// （`u32` は常に `f64` で正確に表現できるため丸めは発生しない）で、
// Rust 側 `(v as f64).to_bits()` と bit 完全一致することを
// `pooling_model.rs` の単体テストが固定する。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// `if (gid >= dims.numel_out) return;` による手動境界チェックを
// 全カーネルで維持する（`.claude/rules/coding-rust.md`。性能を理由
// に省略しない）。座標演算は `long`（窓外を負値として自然に表現
// するため）で行う。

struct PoolDims {
    uint n;
    uint c;
    uint h_in;
    uint w_in;
    uint h_out;
    uint w_out;
    uint kh;
    uint kw;
    uint sh;
    uint sw;
    uint ph;
    uint pw;
    uint dh;
    uint dw;
    uint count_include_pad;
    uint numel_out;
    uint plane_in;
    uint plane_out;
};

// ---- soft-f64 プリミティブ（`batch_norm.metal::bn_f64_*` の逐語
// 移植。冒頭コメント参照）----

struct PoolU128 {
    ulong hi;
    ulong lo;
};

struct PoolNormMantissa {
    ulong m;
    long exp_u;
};

#define POOL_F64_SIGN 0x8000000000000000ul
#define POOL_F64_EXP_MASK 0x7FFul
#define POOL_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define POOL_F64_QNAN 0x7FF8000000000000ul
#define POOL_F64_INF 0x7FF0000000000000ul
#define POOL_F32_QNAN 0x7FC00000u
#define POOL_F32_INF 0x7F800000u

// 64bit leading zero count を 32bit `clz` 2 回で構成する
// （`soft_f64::clz64` と同一構造）。
inline uint pool_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`。
inline ulong pool_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? POOL_F64_QNAN : (sign | POOL_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        // f32 subnormal（`frac × 2^-149`）は f64 では正規化数。
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & POOL_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul); // 減算を先にすると exp < 127 で下溢れ。
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` と同一手順。
inline ulong pool_f64_add(ulong a, ulong b) {
    ulong sa = a & POOL_F64_SIGN;
    ulong sb = b & POOL_F64_SIGN;
    ulong ea = (a >> 52) & POOL_F64_EXP_MASK;
    ulong eb = (b >> 52) & POOL_F64_EXP_MASK;
    ulong fa = a & POOL_F64_FRAC_MASK;
    ulong fb = b & POOL_F64_FRAC_MASK;

    if (ea == POOL_F64_EXP_MASK || eb == POOL_F64_EXP_MASK) {
        bool a_nan = (ea == POOL_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == POOL_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return POOL_F64_QNAN;
        }
        if (ea == POOL_F64_EXP_MASK && eb == POOL_F64_EXP_MASK) {
            return (sa == sb) ? a : POOL_F64_QNAN;
        }
        return (ea == POOL_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)pool_f64_clz64(m);
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
    if (exp_field >= POOL_F64_EXP_MASK) {
        return sa | POOL_F64_INF;
    }
    return sa | (exp_field << 52) | (m & POOL_F64_FRAC_MASK);
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・underflow は f32
// subnormal／`±0`。NaN は quiet NaN へ正規化）。`soft_f64::narrow_f64_bits`。
inline uint pool_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & POOL_F64_EXP_MASK;
    ulong f = bits & POOL_F64_FRAC_MASK;
    if (e == POOL_F64_EXP_MASK) {
        return (f != 0ul) ? POOL_F32_QNAN : (sign | POOL_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | POOL_F32_INF;
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
        return sign | POOL_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// `u64 x u64` の厳密な 128bit 積（32bit 分割のスクールブック乗算。
// `soft_f64::mul64_wide` の逐語移植）。
inline PoolU128 pool_f64_mul64_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFul) + (lo_hi & 0xFFFFFFFFul);
    PoolU128 result;
    result.lo = (lo_lo & 0xFFFFFFFFul) | (mid << 32);
    result.hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    return result;
}

// `f64` の指数・仮数フィールドから「隠れ 1 を bit52 に立てた 53bit
// 仮数」と `value = m * 2^(exp_u-52)` の unbiased 指数の対を求める
// （`soft_f64::normalize_f64_mantissa`）。
inline PoolNormMantissa pool_f64_normalize_mantissa(ulong e, ulong f) {
    PoolNormMantissa r;
    if (e == 0ul) {
        uint lead = 63u - pool_f64_clz64(f);
        uint shift = 52u - lead;
        r.m = f << shift;
        r.exp_u = -1022l - (long)shift;
    } else {
        r.m = f | (1ul << 52);
        r.exp_u = (long)e - 1023l;
    }
    return r;
}

// `(hi,lo)`（128bit・呼び出し前提: 商が 64bit に収まる）を 64bit の
// 非ゼロ除数 `d` で割る筆算除算（2 進 shift-subtract 方式。
// `soft_f64::div64_wide` の逐語移植）。
inline void pool_f64_div64_wide(ulong hi, ulong lo, ulong d, thread ulong &quotient_out, thread ulong &remainder_out) {
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
// 正規化）。`soft_f64::div_f64_bits` の逐語移植。
inline ulong pool_f64_div(ulong a, ulong b) {
    ulong sa = a & POOL_F64_SIGN;
    ulong sb = b & POOL_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & POOL_F64_EXP_MASK;
    ulong eb = (b >> 52) & POOL_F64_EXP_MASK;
    ulong fa = a & POOL_F64_FRAC_MASK;
    ulong fb = b & POOL_F64_FRAC_MASK;

    bool a_nan = (ea == POOL_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == POOL_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return POOL_F64_QNAN;
    }
    bool a_inf = (ea == POOL_F64_EXP_MASK);
    bool b_inf = (eb == POOL_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_inf && b_inf) || (a_zero && b_zero)) {
        return POOL_F64_QNAN;
    }
    if (a_inf) {
        return sign | POOL_F64_INF;
    }
    if (b_inf) {
        return sign;
    }
    if (b_zero) {
        return sign | POOL_F64_INF;
    }
    if (a_zero) {
        return sign;
    }

    PoolNormMantissa na = pool_f64_normalize_mantissa(ea, fa);
    PoolNormMantissa nb = pool_f64_normalize_mantissa(eb, fb);
    // `S = 55`: `ma/mb ∈ (0.5,2)` のため商は `[2^54,2^56)` に収まり、
    // 53bit 仮数 + 2bit（guard/round）の精度が確保できる最小の追加
    // シフト量（`soft_f64::div_f64_bits` と同じ定数）。
    const uint S = 55u;
    PoolU128 num = pool_f64_mul64_wide(na.m, 1ul << S);
    ulong raw_q;
    ulong rem;
    pool_f64_div64_wide(num.hi, num.lo, nb.m, raw_q, rem);
    ulong q = raw_q | (ulong)(rem != 0ul ? 1ul : 0ul);
    // `q` は非ゼロ（呼び出し前提より `na.m`／`nb.m` はいずれも非ゼロ）。
    uint leadpos = 63u - pool_f64_clz64(q);
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
        if (biased_final >= (long)POOL_F64_EXP_MASK) {
            return sign | POOL_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & POOL_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

// `u32 -> binary64` の厳密変換（`u32` は常に `f64` で正確に表現
// できるため丸めは発生しない。ホスト側 `(v as f64).to_bits()` と
// bit 完全一致——`crate::pooling_model` の単体テストが固定する）。
// Avg 系カーネルの `divisor`（`count_include_pad` 有効時の `kh*kw`
// 固定値／無効時の有効タップ数）を soft-f64 除算へ渡す前に使う。
inline ulong pool_f64_from_uint(uint v) {
    if (v == 0u) {
        return 0ul;
    }
    uint lead = 31u - clz(v);
    ulong exp64 = (ulong)((int)lead + 1023);
    ulong frac = (((ulong)v) << (52u - lead)) & POOL_F64_FRAC_MASK;
    return (exp64 << 52) | frac;
}

// ---- カーネル本体 ----

kernel void max_pool2d_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device int* idx [[buffer(2)]],
    constant PoolDims& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.numel_out) {
        return;
    }
    long rem = (long)gid;
    long ow = rem % (long)dims.w_out;
    rem /= (long)dims.w_out;
    long oh = rem % (long)dims.h_out;
    rem /= (long)dims.h_out;
    long ch = rem % (long)dims.c;
    long nb = rem / (long)dims.c;

    long h_in = (long)dims.h_in;
    long w_in = (long)dims.w_in;

    float best = 0.0f;
    int best_idx = 0;
    bool first = true;

    for (uint kh_i = 0; kh_i < dims.kh; kh_i++) {
        long ih = oh * (long)dims.sh + (long)kh_i * (long)dims.dh - (long)dims.ph;
        if (ih < 0 || ih >= h_in) {
            continue;
        }
        for (uint kw_i = 0; kw_i < dims.kw; kw_i++) {
            long iw = ow * (long)dims.sw + (long)kw_i * (long)dims.dw - (long)dims.pw;
            if (iw < 0 || iw >= w_in) {
                continue;
            }
            long in_idx = ((nb * (long)dims.c + ch) * h_in + ih) * w_in + iw;
            float v = x[in_idx];
            int this_idx = (int)(ih * w_in + iw);
            if (first) {
                best = v;
                best_idx = this_idx;
                first = false;
            } else if (v > best || (isnan(v) && !isnan(best))) {
                best = v;
                best_idx = this_idx;
            }
        }
    }
    out[gid] = best;
    idx[gid] = best_idx;
}

kernel void avg_pool2d_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant PoolDims& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.numel_out) {
        return;
    }
    long rem = (long)gid;
    long ow = rem % (long)dims.w_out;
    rem /= (long)dims.w_out;
    long oh = rem % (long)dims.h_out;
    rem /= (long)dims.h_out;
    long ch = rem % (long)dims.c;
    long nb = rem / (long)dims.c;

    long h_in = (long)dims.h_in;
    long w_in = (long)dims.w_in;

    ulong acc = 0ul; // pool_f64_widen(as_type<uint>(0.0f)) == 0ul。
    uint valid = 0u;

    for (uint kh_i = 0; kh_i < dims.kh; kh_i++) {
        long ih = oh * (long)dims.sh + (long)kh_i * (long)dims.dh - (long)dims.ph;
        if (ih < 0 || ih >= h_in) {
            continue;
        }
        for (uint kw_i = 0; kw_i < dims.kw; kw_i++) {
            long iw = ow * (long)dims.sw + (long)kw_i * (long)dims.dw - (long)dims.pw;
            if (iw < 0 || iw >= w_in) {
                continue;
            }
            long in_idx = ((nb * (long)dims.c + ch) * h_in + ih) * w_in + iw;
            float v = x[in_idx];
            acc = pool_f64_add(acc, pool_f64_widen(as_type<uint>(v)));
            valid += 1u;
        }
    }
    uint divisor = (dims.count_include_pad != 0u) ? (dims.kh * dims.kw) : valid;
    out[gid] = as_type<float>(pool_f64_narrow(pool_f64_div(acc, pool_f64_from_uint(divisor))));
}

kernel void adaptive_avg_pool2d_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant PoolDims& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.numel_out) {
        return;
    }
    long rem = (long)gid;
    long ow = rem % (long)dims.w_out;
    rem /= (long)dims.w_out;
    long oh = rem % (long)dims.h_out;
    rem /= (long)dims.h_out;
    long ch = rem % (long)dims.c;
    long nb = rem / (long)dims.c;

    long h_in = (long)dims.h_in;
    long w_in = (long)dims.w_in;
    long h_out = (long)dims.h_out;
    long w_out = (long)dims.w_out;

    long start_h = (oh * h_in) / h_out;
    long end_h = ((oh + 1) * h_in + h_out - 1) / h_out;
    long start_w = (ow * w_in) / w_out;
    long end_w = ((ow + 1) * w_in + w_out - 1) / w_out;

    ulong acc = 0ul;
    for (long ih = start_h; ih < end_h; ih++) {
        for (long iw = start_w; iw < end_w; iw++) {
            long in_idx = ((nb * (long)dims.c + ch) * h_in + ih) * w_in + iw;
            float v = x[in_idx];
            acc = pool_f64_add(acc, pool_f64_widen(as_type<uint>(v)));
        }
    }
    uint divisor = (uint)((end_h - start_h) * (end_w - start_w));
    out[gid] = as_type<float>(pool_f64_narrow(pool_f64_div(acc, pool_f64_from_uint(divisor))));
}
