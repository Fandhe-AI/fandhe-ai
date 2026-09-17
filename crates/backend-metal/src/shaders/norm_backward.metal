// LayerNorm／RMSNorm backward カーネル（イシュー #1953・親 #1947。CUDA
// 側 `kernels_norm_backward.rs`〈イシュー #1950〉と同じ設計方針の Metal
// 対応版）。
//
// forward カーネル（`rmsnorm.metal`／`layer_norm.metal`）・tape 記録は
// 一切変更しない（`rstd`／`mean` の保存を forward へ追加しない）。本
// ファイルのカーネルは `x`（と `weight`）から行内統計をカーネル内で
// 再計算する recompute-in-backward 方式（`fandhe_ai_autodiff::grad::
// rmsnorm_vjp_rows`／`layer_norm_vjp_rows` と同じ「`input` を実体化し
// 直して統計を再計算する」方針の Metal 版。CUDA #1950 と同一設計）。
//
// **数値契約（REQ-2。bit 完全一致ではない）**: `dx`／`dw`／`db` いずれも
// CPU ホスト参照実装と `fandhe_ai_backend_cpu::parity::assert_parity`
// （統一複合判定。相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）で検証
// する。理由は `layer_norm.metal` 冒頭コメント「xhat 自体は本ファイルの
// soft-f64 経由で導出するため CPU/CUDA と bit 一致はしない」と同じ
// （行内統計の縮約順序が GPU レーンストライド + butterfly であり、
// CPU の逐次走査と異なるため）。CUDA #1950 も同じ判定方式を採用済み
// （`crates/backend-cuda/tests/norm_backward_parity.rs` 冒頭コメント
// 参照）。
//
// **soft-f64 実装（`nb_f64_*` 系関数）**: `layer_norm.metal` の
// `ln_f64_*` 系関数の逐語複製（MSL は `newLibraryWithSource` でファイル
// 単位にコンパイルされ翻訳単位を共有できないため、`gemm.metal`・
// `layer_norm.metal` と同様に独立実装を持つ。`crates/backend-metal/src/
// soft_f64.rs`・`crates/backend-metal/src/norm_backward_model.rs` が
// ホスト側の逐語モデル）。backward は affine（round-to-odd 加算・FMA
// 契約）を持たないため `nb_f64_add_ro`／`nb_f64_recip_newton` は移植
// しない（未使用）。
//
// **カーネル構成（4 カーネル。`norm_backward.rs` から起動される）**:
// - `rmsnorm_bwd_dx_f32`・`layer_norm_bwd_dx_f32`: 1 threadgroup = 1
//   simdgroup（32 レーン）が 1 行を担当する persistent grid 方式
//   （`layer_norm_f32` と同じ `for (row = tg_id; row < rows; row +=
//   grid_size)`）。行内統計（RMSNorm は二乗和・LayerNorm は平均＋分散）
//   をレーンストライド + 5 段 butterfly（offset 16→1）で計算し、`dx` を
//   書き出すと同時に行ごとの統計スカラー（RMSNorm は `rstd`（`f32`。
//   CPU `row_rms_stats` が返す型と揃える）・LayerNorm は `mean`／`rstd`
//   （`f64` bit パターンのまま。CPU `row_ln_stats` が `f64` のまま
//   返す型と揃える）を lane 0 が scratch バッファ（`rows` 要素）へ
//   書き出す（dw／db カーネルが同じ統計を再計算せず再利用するため）。
// - `rmsnorm_bwd_dw_f32`・`layer_norm_bwd_dwdb_f32`: 1 スレッド = 1 列
//   （`hidden`）で `rows` を `r = 0..rows` の順に逐次走査し、`nb_f64_add`
//   （`f64` 相当の soft-f64 アキュムレータ）へ `term`（`f32` で確定済みの
//   積）を蓄積する（`.claude/rules/coding-rust.md`「要素積を `f32` で
//   確定してから `f64` へ昇格して蓄積する」契約に従う）。`dw`／`db` は
//   `weight`／`bias` の値そのものに依存しない（`dw[i] = Σ_r dy[r,i]·
//   x̂[r,i]`・`db[i] = Σ_r dy[r,i]`）ため、`has_weight`／`has_bias` は
//   ホスト側（`norm_backward.rs`）が戻り値を `Some`/`None` へ振り分ける
//   判断にのみ使い、カーネル自体は無条件に両方計算する。
//
// **REQ-8 境界検査**: dx カーネルは `row >= rows` の persistent ループ
// ガード・`idx < hidden` のレーンストライドループガード。dw／db カーネルは
// `if (col >= hidden) return;`。ループ添字は `ulong`（`row_base`）で
// `rows * hidden` の乗算オーバーフローを避ける（`layer_norm.metal` と
// 同じ対策）。
//
// コンパイルオプションは `pipeline::compile_options()` を適用する。

#include <metal_stdlib>
using namespace metal;

constant uint NB_SIMD_WIDTH = 32u;

// ---- IEEE 754 binary64 のソフトウェアエミュレーション（`nb_f64_*`）----
// `crates/backend-metal/src/soft_f64.rs`／`layer_norm.metal::ln_f64_*` の
// 逐語移植（冒頭コメント参照）。

#define NB_F64_SIGN      0x8000000000000000ul
#define NB_F64_EXP_MASK  0x7FFul
#define NB_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define NB_F64_QNAN      0x7FF8000000000000ul
#define NB_F64_INF       0x7FF0000000000000ul
#define NB_F32_QNAN      0x7FC00000u
#define NB_F32_INF       0x7F800000u

struct NbU128 {
    ulong hi;
    ulong lo;
};

struct NbNormMantissa {
    ulong m;
    long exp_u;
};

inline uint nb_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

inline ulong nb_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? NB_F64_QNAN : (sign | NB_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & NB_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul);
    return sign | (exp64 << 52) | (frac << 29);
}

inline ulong nb_f64_neg(ulong a) {
    return a ^ NB_F64_SIGN;
}

inline ulong nb_f64_add(ulong a, ulong b) {
    ulong sa = a & NB_F64_SIGN;
    ulong sb = b & NB_F64_SIGN;
    ulong ea = (a >> 52) & NB_F64_EXP_MASK;
    ulong eb = (b >> 52) & NB_F64_EXP_MASK;
    ulong fa = a & NB_F64_FRAC_MASK;
    ulong fb = b & NB_F64_FRAC_MASK;

    if (ea == NB_F64_EXP_MASK || eb == NB_F64_EXP_MASK) {
        bool a_nan = (ea == NB_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == NB_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return NB_F64_QNAN;
        }
        if (ea == NB_F64_EXP_MASK && eb == NB_F64_EXP_MASK) {
            return (sa == sb) ? a : NB_F64_QNAN;
        }
        return (ea == NB_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)nb_f64_clz64(m);
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
    if (exp_field >= NB_F64_EXP_MASK) {
        return sa | NB_F64_INF;
    }
    return sa | (exp_field << 52) | (m & NB_F64_FRAC_MASK);
}

inline ulong nb_f64_sub(ulong a, ulong b) {
    return nb_f64_add(a, nb_f64_neg(b));
}

inline uint nb_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & NB_F64_EXP_MASK;
    ulong f = bits & NB_F64_FRAC_MASK;
    if (e == NB_F64_EXP_MASK) {
        return (f != 0ul) ? NB_F32_QNAN : (sign | NB_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | NB_F32_INF;
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
        return sign | NB_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

inline NbU128 nb_f64_mul64_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong mid = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFul) + (lo_hi & 0xFFFFFFFFul);
    NbU128 result;
    result.lo = (lo_lo & 0xFFFFFFFFul) | (mid << 32);
    result.hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    return result;
}

inline NbU128 nb_f64_shr128(ulong hi, ulong lo, uint s) {
    NbU128 r;
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

inline NbU128 nb_f64_low_bits128(ulong hi, ulong lo, uint n) {
    NbU128 r;
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

inline int nb_f64_cmp128(ulong ahi, ulong alo, ulong bhi, ulong blo) {
    if (ahi != bhi) {
        return (ahi < bhi) ? -1 : 1;
    }
    if (alo != blo) {
        return (alo < blo) ? -1 : 1;
    }
    return 0;
}

inline NbNormMantissa nb_f64_normalize_mantissa(ulong e, ulong f) {
    NbNormMantissa r;
    if (e == 0ul) {
        uint lead = 63u - nb_f64_clz64(f);
        uint shift = 52u - lead;
        r.m = f << shift;
        r.exp_u = -1022l - (long)shift;
    } else {
        r.m = f | (1ul << 52);
        r.exp_u = (long)e - 1023l;
    }
    return r;
}

inline ulong nb_f64_mul(ulong a, ulong b) {
    ulong sa = a & NB_F64_SIGN;
    ulong sb = b & NB_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & NB_F64_EXP_MASK;
    ulong eb = (b >> 52) & NB_F64_EXP_MASK;
    ulong fa = a & NB_F64_FRAC_MASK;
    ulong fb = b & NB_F64_FRAC_MASK;

    bool a_nan = (ea == NB_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == NB_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return NB_F64_QNAN;
    }
    bool a_inf = (ea == NB_F64_EXP_MASK);
    bool b_inf = (eb == NB_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_zero && b_inf) || (a_inf && b_zero)) {
        return NB_F64_QNAN;
    }
    if (a_inf || b_inf) {
        return sign | NB_F64_INF;
    }
    if (a_zero || b_zero) {
        return sign;
    }

    NbNormMantissa na = nb_f64_normalize_mantissa(ea, fa);
    NbNormMantissa nb = nb_f64_normalize_mantissa(eb, fb);
    NbU128 p = nb_f64_mul64_wide(na.m, nb.m);
    uint leadpos = 64u + (63u - nb_f64_clz64(p.hi));
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
        return sign;
    }
    uint final_shift = (uint)final_shift_l;
    NbU128 q = nb_f64_shr128(p.hi, p.lo, final_shift);
    ulong m = q.lo;
    NbU128 rem = nb_f64_low_bits128(p.hi, p.lo, final_shift);
    ulong half_hi = 0ul;
    ulong half_lo = 0ul;
    if (final_shift > 0u) {
        if (final_shift - 1u < 64u) {
            half_lo = 1ul << (final_shift - 1u);
        } else {
            half_hi = 1ul << (final_shift - 1u - 64u);
        }
    }
    int cmp = nb_f64_cmp128(rem.hi, rem.lo, half_hi, half_lo);
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
        if (biased_final >= (long)NB_F64_EXP_MASK) {
            return sign | NB_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & NB_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

inline void nb_f64_div64_wide(ulong hi, ulong lo, ulong d, thread ulong &quotient_out, thread ulong &remainder_out) {
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

inline ulong nb_f64_div(ulong a, ulong b) {
    ulong sa = a & NB_F64_SIGN;
    ulong sb = b & NB_F64_SIGN;
    ulong sign = sa ^ sb;
    ulong ea = (a >> 52) & NB_F64_EXP_MASK;
    ulong eb = (b >> 52) & NB_F64_EXP_MASK;
    ulong fa = a & NB_F64_FRAC_MASK;
    ulong fb = b & NB_F64_FRAC_MASK;

    bool a_nan = (ea == NB_F64_EXP_MASK) && (fa != 0ul);
    bool b_nan = (eb == NB_F64_EXP_MASK) && (fb != 0ul);
    if (a_nan || b_nan) {
        return NB_F64_QNAN;
    }
    bool a_inf = (ea == NB_F64_EXP_MASK);
    bool b_inf = (eb == NB_F64_EXP_MASK);
    bool a_zero = (ea == 0ul) && (fa == 0ul);
    bool b_zero = (eb == 0ul) && (fb == 0ul);
    if ((a_inf && b_inf) || (a_zero && b_zero)) {
        return NB_F64_QNAN;
    }
    if (a_inf) {
        return sign | NB_F64_INF;
    }
    if (b_inf) {
        return sign;
    }
    if (b_zero) {
        return sign | NB_F64_INF;
    }
    if (a_zero) {
        return sign;
    }

    NbNormMantissa na = nb_f64_normalize_mantissa(ea, fa);
    NbNormMantissa nb = nb_f64_normalize_mantissa(eb, fb);
    const uint S = 55u;
    NbU128 num = nb_f64_mul64_wide(na.m, 1ul << S);
    ulong raw_q;
    ulong rem;
    nb_f64_div64_wide(num.hi, num.lo, nb.m, raw_q, rem);
    ulong q = raw_q | (ulong)(rem != 0ul ? 1ul : 0ul);
    uint leadpos = 63u - nb_f64_clz64(q);
    long exp_u = na.exp_u - nb.exp_u + ((long)leadpos - (long)S);
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

    if (final_shift_l < 0l || final_shift_l >= 64l) {
        return sign;
    }
    uint final_shift = (uint)final_shift_l;
    ulong m = (final_shift == 0u) ? q : (q >> final_shift);
    ulong low_mask = (final_shift >= 64u) ? ~0ul : ((1ul << final_shift) - 1ul);
    ulong rem_low = (final_shift == 0u) ? 0ul : (q & low_mask);
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
        if (biased_final >= (long)NB_F64_EXP_MASK) {
            return sign | NB_F64_INF;
        }
        return sign | (((ulong)biased_final) << 52) | (m & NB_F64_FRAC_MASK);
    } else {
        if (m >= (1ul << 52)) {
            return sign | (1ul << 52);
        }
        return sign | m;
    }
}

inline ulong nb_f64_scale_pow2(ulong x_bits, long k) {
    ulong sign = x_bits & NB_F64_SIGN;
    long e = (long)((x_bits >> 52) & NB_F64_EXP_MASK);
    ulong f = x_bits & NB_F64_FRAC_MASK;
    long new_e = e + k;
    if (new_e >= (long)NB_F64_EXP_MASK) {
        return sign | NB_F64_INF;
    }
    if (new_e <= 0l) {
        return sign;
    }
    return sign | (((ulong)new_e) << 52) | f;
}

struct NbReducedSeed {
    uint reduced_bits;
    long exp_out;
};

inline NbReducedSeed nb_f64_extract_reduced_and_exp(ulong x_bits, bool want_sqrt_range) {
    ulong e = (x_bits >> 52) & NB_F64_EXP_MASK;
    ulong f = x_bits & NB_F64_FRAC_MASK;
    NbNormMantissa n = nb_f64_normalize_mantissa(e, f);
    NbReducedSeed r;
    if (!want_sqrt_range) {
        ulong reduced_bits = (1023ul << 52) | (n.m & NB_F64_FRAC_MASK);
        r.reduced_bits = nb_f64_narrow(reduced_bits);
        r.exp_out = n.exp_u;
    } else if ((n.exp_u & 1l) == 0l) {
        ulong reduced_bits = (1023ul << 52) | (n.m & NB_F64_FRAC_MASK);
        r.reduced_bits = nb_f64_narrow(reduced_bits);
        r.exp_out = n.exp_u >> 1;
    } else {
        ulong reduced_bits = (1024ul << 52) | (n.m & NB_F64_FRAC_MASK);
        r.reduced_bits = nb_f64_narrow(reduced_bits);
        r.exp_out = (n.exp_u - 1l) >> 1;
    }
    return r;
}

// `f64` の逆数平方根（`soft_f64::rsqrt_newton_f64_bits`／
// `ln_f64_rsqrt_newton` の逐語移植）。
inline ulong nb_f64_rsqrt_newton(ulong x) {
    ulong e = (x >> 52) & NB_F64_EXP_MASK;
    ulong f = x & NB_F64_FRAC_MASK;
    ulong sign = x & NB_F64_SIGN;
    if (e == NB_F64_EXP_MASK && f != 0ul) {
        return NB_F64_QNAN;
    }
    if (e == 0ul && f == 0ul) {
        return sign | NB_F64_INF;
    }
    if (sign != 0ul) {
        return NB_F64_QNAN;
    }
    if (e == NB_F64_EXP_MASK) {
        return 0ul;
    }

    NbReducedSeed seed = nb_f64_extract_reduced_and_exp(x, true);
    float seed_reduced = 1.0f / sqrt(as_type<float>(seed.reduced_bits));
    ulong y = nb_f64_scale_pow2(nb_f64_widen(as_type<uint>(seed_reduced)), -seed.exp_out);
    const ulong ONE_HALF = 0x3FE0000000000000ul;
    const ulong THREE_HALF = 0x3FF8000000000000ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong y2 = nb_f64_mul(y, y);
        ulong xy2 = nb_f64_mul(x, y2);
        ulong half_xy2 = nb_f64_mul(ONE_HALF, xy2);
        ulong inner = nb_f64_sub(THREE_HALF, half_xy2);
        y = nb_f64_mul(y, inner);
    }
    return y;
}

// レーンストライド + 5 段 butterfly 縮約用の 1 ステップ（`ulong` を
// `simd_shuffle_xor` は直接扱えない環境差を避けるため hi/lo 32bit へ
// 分割してシャッフルする。`layer_norm_f32` の同型パターン）。
inline ulong nb_shuffle_xor_u64(ulong v, ushort offset) {
    ulong hi = simd_shuffle_xor((uint)(v >> 32), offset);
    ulong lo = simd_shuffle_xor((uint)v, offset);
    return (hi << 32) | lo;
}

// ---- RMSNorm backward ----

// dx カーネル（冒頭コメント参照）。引数: `x`（`[rows,hidden]` 行優先）・
// `w`（`has_weight==0` なら未参照。呼び出し元は必ず `hidden` 要素の
// ダミーバッファを渡す。REQ-8 の predicated-load 対策）・`dy`・`dx`
// （出力）・`rstd_out`（出力。`rows` 要素の `float` scratch。CPU
// `row_rms_stats` が `f32` を返す型に揃える）・`rows`・`hidden`・
// `eps`・`has_weight`・`grid_size`。
kernel void rmsnorm_bwd_dx_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dx [[buffer(3)]],
    device float* rstd_out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant int& has_weight [[buffer(8)]],
    constant uint& grid_size [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;
        ulong hidden_f64 = nb_f64_widen(as_type<uint>((float)hidden));

        // パス 1: 二乗和（符号なし項のみで相殺なし。単純逐次和のまま
        // 維持する CPU 参照実装〈`row_rms_stats`〉と異なり、GPU 側は
        // レーンストライド + butterfly で総和する。REQ-2 範囲での
        // 一致に留まる）。
        ulong lane_sq = 0ul;
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            ulong xv = nb_f64_widen(as_type<uint>(x[row_base + idx]));
            lane_sq = nb_f64_add(lane_sq, nb_f64_mul(xv, xv));
        }
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            lane_sq = nb_f64_add(lane_sq, nb_shuffle_xor_u64(lane_sq, offset));
        }
        ulong mean_sq = nb_f64_div(lane_sq, hidden_f64);
        ulong ve = nb_f64_add(mean_sq, nb_f64_widen(as_type<uint>(eps)));
        ulong rstd64 = nb_f64_rsqrt_newton(ve);
        uint rstd_bits = nb_f64_narrow(rstd64);
        float rstd = as_type<float>(rstd_bits);
        if (lane == 0u) {
            rstd_out[row] = rstd;
        }
        ulong rstd_w64 = nb_f64_widen(rstd_bits);

        // パス 2: `dot = Σ dxhat_i * xhat_i`（符号付き項。`eval::
        // warp_reduce_f64` と同一のレーンストライド + butterfly 順序）。
        ulong lane_dot = 0ul;
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            float xv = x[row_base + idx];
            float xhat = xv * rstd;
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float dxhat = dy[row_base + idx] * wv;
            float term = dxhat * xhat;
            lane_dot = nb_f64_add(lane_dot, nb_f64_widen(as_type<uint>(term)));
        }
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            lane_dot = nb_f64_add(lane_dot, nb_shuffle_xor_u64(lane_dot, offset));
        }
        // CPU 参照実装（`grad::rmsnorm_vjp_rows`）・CUDA
        // （`kernels_norm_backward.rs::rmsnorm_bwd_dx_new_f32`）と同じ
        // 演算列 `dot * (1.0 / hidden)`（逆数を丸めてから乗算）へ揃える
        // （PR #2001 codex-review P1 是正）。`dot / hidden` は数学的に
        // 同値でも丸め誤差が異なり、相殺する `dot` と組み合わさると
        // REQ-2 の統一複合判定を外れうる。
        ulong inv_hidden64 = nb_f64_div(0x3FF0000000000000ul, hidden_f64);
        ulong mean_dot = nb_f64_mul(lane_dot, inv_hidden64);

        // パス 3: `dx = rstd * (dxhat - xhat * mean_dot)`。
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            float xv = x[row_base + idx];
            float xhat = xv * rstd;
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float dxhat = dy[row_base + idx] * wv;
            ulong term2 = nb_f64_mul(nb_f64_widen(as_type<uint>(xhat)), mean_dot);
            ulong inner = nb_f64_sub(nb_f64_widen(as_type<uint>(dxhat)), term2);
            ulong d64 = nb_f64_mul(rstd_w64, inner);
            dx[row_base + idx] = as_type<float>(nb_f64_narrow(d64));
        }
    }
}

// dw カーネル（冒頭コメント参照）。1 スレッド = 1 列。`rstd_in` は
// `rmsnorm_bwd_dx_f32` が書き出した scratch（再計算しない）。
kernel void rmsnorm_bwd_dw_f32(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device const float* rstd_in [[buffer(2)]],
    device float* dw [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= hidden) {
        return;
    }
    ulong acc = 0ul;
    for (uint r = 0u; r < rows; r++) {
        ulong row_base = (ulong)r * (ulong)hidden;
        float rstd = rstd_in[r];
        float xhat = x[row_base + col] * rstd;
        float term = dy[row_base + col] * xhat;
        acc = nb_f64_add(acc, nb_f64_widen(as_type<uint>(term)));
    }
    dw[col] = as_type<float>(nb_f64_narrow(acc));
}

// ---- LayerNorm backward ----

// dx カーネル。`mean_out`／`rstd_out` は `f64` bit パターン（`ulong`）の
// まま scratch へ書き出す（CPU `row_ln_stats` が `f64` のまま `mean`／
// `rstd` を返す型に揃える）。
kernel void layer_norm_bwd_dx_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dx [[buffer(3)]],
    device ulong* mean_out [[buffer(4)]],
    device ulong* rstd_out [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant int& has_weight [[buffer(9)]],
    constant uint& grid_size [[buffer(10)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;
        ulong hidden_f64 = nb_f64_widen(as_type<uint>((float)hidden));

        // パス 1: 平均。
        ulong lane_sum = 0ul;
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            ulong xv = nb_f64_widen(as_type<uint>(x[row_base + idx]));
            lane_sum = nb_f64_add(lane_sum, xv);
        }
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            lane_sum = nb_f64_add(lane_sum, nb_shuffle_xor_u64(lane_sum, offset));
        }
        ulong mean = nb_f64_div(lane_sum, hidden_f64);

        // パス 2: 分散（二パス。`(x-mean)^2` を soft-f64 で蓄積）。
        ulong lane_sq = 0ul;
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            ulong xv = nb_f64_widen(as_type<uint>(x[row_base + idx]));
            ulong dev = nb_f64_sub(xv, mean);
            lane_sq = nb_f64_add(lane_sq, nb_f64_mul(dev, dev));
        }
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            lane_sq = nb_f64_add(lane_sq, nb_shuffle_xor_u64(lane_sq, offset));
        }
        ulong var = nb_f64_div(lane_sq, hidden_f64);
        ulong ve = nb_f64_add(var, nb_f64_widen(as_type<uint>(eps)));
        ulong rstd = nb_f64_rsqrt_newton(ve);
        if (lane == 0u) {
            mean_out[row] = mean;
            rstd_out[row] = rstd;
        }

        // パス 3: `sum_dxhat`・`dot`（ともに符号付き項の行内総和）。
        ulong lane_sum_dxhat = 0ul;
        ulong lane_dot = 0ul;
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            ulong xv = nb_f64_widen(as_type<uint>(x[row_base + idx]));
            ulong dev = nb_f64_sub(xv, mean);
            ulong xhat64 = nb_f64_mul(dev, rstd);
            uint xhat_bits = nb_f64_narrow(xhat64);
            float xhat = as_type<float>(xhat_bits);
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float dxhat = dy[row_base + idx] * wv;
            lane_sum_dxhat = nb_f64_add(lane_sum_dxhat, nb_f64_widen(as_type<uint>(dxhat)));
            float term = dxhat * xhat;
            lane_dot = nb_f64_add(lane_dot, nb_f64_widen(as_type<uint>(term)));
        }
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            lane_sum_dxhat = nb_f64_add(lane_sum_dxhat, nb_shuffle_xor_u64(lane_sum_dxhat, offset));
            lane_dot = nb_f64_add(lane_dot, nb_shuffle_xor_u64(lane_dot, offset));
        }
        ulong mean_dxhat = nb_f64_div(lane_sum_dxhat, hidden_f64);
        ulong mean_dot = nb_f64_div(lane_dot, hidden_f64);

        // パス 4: `dx = rstd * (dxhat - mean_dxhat - xhat * mean_dot)`。
        for (uint idx = lane; idx < hidden; idx += NB_SIMD_WIDTH) {
            ulong xv = nb_f64_widen(as_type<uint>(x[row_base + idx]));
            ulong dev = nb_f64_sub(xv, mean);
            ulong xhat64 = nb_f64_mul(dev, rstd);
            uint xhat_bits = nb_f64_narrow(xhat64);
            float xhat = as_type<float>(xhat_bits);
            float wv = (has_weight != 0) ? w[idx] : 1.0f;
            float dxhat = dy[row_base + idx] * wv;
            ulong term2 = nb_f64_mul(nb_f64_widen(as_type<uint>(xhat)), mean_dot);
            ulong inner = nb_f64_sub(nb_f64_sub(nb_f64_widen(as_type<uint>(dxhat)), mean_dxhat), term2);
            ulong d64 = nb_f64_mul(rstd, inner);
            dx[row_base + idx] = as_type<float>(nb_f64_narrow(d64));
        }
    }
}

// dw／db カーネル。1 スレッド = 1 列。`mean_in`／`rstd_in` は
// `layer_norm_bwd_dx_f32` が書き出した scratch（再計算しない）。
kernel void layer_norm_bwd_dwdb_f32(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device const ulong* mean_in [[buffer(2)]],
    device const ulong* rstd_in [[buffer(3)]],
    device float* dw [[buffer(4)]],
    device float* db [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= hidden) {
        return;
    }
    ulong dw_acc = 0ul;
    ulong db_acc = 0ul;
    for (uint r = 0u; r < rows; r++) {
        ulong row_base = (ulong)r * (ulong)hidden;
        ulong mean = mean_in[r];
        ulong rstd = rstd_in[r];
        ulong xv = nb_f64_widen(as_type<uint>(x[row_base + col]));
        ulong dev = nb_f64_sub(xv, mean);
        ulong xhat64 = nb_f64_mul(dev, rstd);
        float xhat = as_type<float>(nb_f64_narrow(xhat64));
        float dyv = dy[row_base + col];
        float term = dyv * xhat;
        dw_acc = nb_f64_add(dw_acc, nb_f64_widen(as_type<uint>(term)));
        db_acc = nb_f64_add(db_acc, nb_f64_widen(as_type<uint>(dyv)));
    }
    dw[col] = as_type<float>(nb_f64_narrow(dw_acc));
    db[col] = as_type<float>(nb_f64_narrow(db_acc));
}
