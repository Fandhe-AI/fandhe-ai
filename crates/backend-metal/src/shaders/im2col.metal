// Conv2d の im2col／col2im（イシュー #1768・親 #1644。
// `crates/backend-cuda/src/kernels_im2col.rs` の Metal 対応版。ホスト
// モデルは `crate::im2col_model`）。
//
// `crate::im2col::MetalIm2col`（`im2col.rs`）から実行時コンパイルされ、
// `ops.rs::MetalBackendOps::im2col`／`col2im` から呼ばれる。
//
// ---- レイアウト（`fandhe_ai_tensor_core::im2col_out_shape` と同一）----
//
// `im2col_f32`: `in: [N, Cin, H, W]` を `out: [N, G, K_g, P]`
// （`K_g` 軸は `(c_in_g, kh, kw)` の row-major・`P` 軸は `(oh, ow)` の
// row-major）へ展開する。`col2im_f32` はその随伴（転置畳み込み）で
// `d_col: [N, G, K_g, P]` を `out: [N, Cin, H, W]` へ畳み戻す。
//
// ---- 数値方式 ----
//
// `im2col_f32` は「`in` 内部位置ならそのままコピー・padding 位置なら
// `0.0`」の 2 分岐のみで決まる純粋なコピー演算（算術を含まない）のため
// CPU 参照実装（`backend-cpu::im2col::im2col`）と **bit 完全一致**
// （`.claude/rules/coding-rust.md` 数値契約節）。
//
// `col2im_f32` は重なり窓（`stride < dilation·(kernel−1)+1`）の加算順を
// `(kh, kw)` row-major に固定する。MSL は `double` 型非対応のため、
// `.claude/rules/coding-rust.md`「勾配の長軸縮約」節・`crate::soft_f64`
// と同じ binary64 逐次演算の 64bit 整数ソフトウェアエミュレーションを
// 適用する（`im2col_f64_widen`／`im2col_f64_add`／`im2col_f64_narrow`。
// `scan.metal::scan_f64_widen`／`scan_f64_add`／`scan_f64_narrow` の
// 逐語移植——MSL は `newLibraryWithSource` でファイル単位にコンパイル
// され翻訳単位を共有できないため、`im2col_f64_` 接頭辞を付けた本
// ファイル内で独立に定義する。意図的な重複。ホスト側の逐語モデルは
// `crate::soft_f64`／`crate::im2col_model` を参照）。アキュムレータ
// （64bit 整数表現の `ulong acc`）を `widen(0.0)` から開始し、寄与する
// 全 `(kh, kw)` を row-major に走査して `acc = im2col_f64_add(acc,
// im2col_f64_widen(d_col[col_idx]))` を計算したうえで最後に 1 回
// `out[idx] = im2col_f64_narrow(acc)` を書く。この演算列はホスト
// `f64` 逐次参照実装（`backend-cpu::im2col::col2im`）と bit 完全一致
// する契約（NaN のみ payload がハードウェア依存のためクラス一致。
// `crate::im2col_model` の単体テストが Linux で機械的に裏付ける）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// `gid`（グローバルスレッド id）が `dims.numel` 未満かどうかを両
// カーネルとも検査してから処理する。添字演算は `long`（符号付き
// 64bit）で行う（`ulong` ではなく `long` を使う理由: im2col の座標
// 計算 `h = oh*sh + kh*dh - ph` は `ph` が `oh*sh + kh*dh` を上回る
// 場合に負値になりうり、`ulong` では即座にラップアラウンドしてしまう
// ため。CUDA 版が `long long` を使うのと同じ理由）。

#include <metal_stdlib>
using namespace metal;

// im2col／col2im の形状引数一式（`crate::im2col_model::Im2colDims` と
// バイト単位で同一レイアウト。19 × uint = 76 バイト）。
// `crate::im2col::MetalIm2col` が `setBytes_length_atIndex`（buffer
// index 2）で構造体丸ごと 1 回の呼び出しで渡す（`gemm.metal::Dims` と
// 同じ「個別 setBytes を避ける」方針）。
struct Im2colDims {
    uint n_batch;
    uint cin;
    uint groups;
    uint cin_g;
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
    uint k_g;
    uint p;
    uint numel;
};

// ---- IEEE 754 binary64 のソフトウェアエミュレーション（col2im 用）----
// `crates/backend-metal/src/soft_f64.rs`／`scan.metal::scan_f64_*` の
// 逐語移植（本ファイル冒頭コメント参照）。`mul` は col2im が使わない
// ため定義しない。

#define IM2COL_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define IM2COL_F64_EXP_MASK  0x7FFul
#define IM2COL_F64_SIGN      0x8000000000000000ul
#define IM2COL_F64_QNAN      0x7FF8000000000000ul
#define IM2COL_F64_INF       0x7FF0000000000000ul
#define IM2COL_F32_QNAN      0x7FC00000u
#define IM2COL_F32_INF       0x7F800000u

// 64bit leading zero count（`clz(0u) == 32` は MSL 仕様で定義済み。
// `soft_f64::clz64` と同一構造）。
inline uint im2col_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`。
inline ulong im2col_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? IM2COL_F64_QNAN : (sign | IM2COL_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & IM2COL_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul);
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` の逐語移植。
inline ulong im2col_f64_add(ulong a, ulong b) {
    ulong sa = a & IM2COL_F64_SIGN;
    ulong sb = b & IM2COL_F64_SIGN;
    ulong ea = (a >> 52) & IM2COL_F64_EXP_MASK;
    ulong eb = (b >> 52) & IM2COL_F64_EXP_MASK;
    ulong fa = a & IM2COL_F64_FRAC_MASK;
    ulong fb = b & IM2COL_F64_FRAC_MASK;

    if (ea == IM2COL_F64_EXP_MASK || eb == IM2COL_F64_EXP_MASK) {
        bool a_nan = (ea == IM2COL_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == IM2COL_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return IM2COL_F64_QNAN;
        }
        if (ea == IM2COL_F64_EXP_MASK && eb == IM2COL_F64_EXP_MASK) {
            return (sa == sb) ? a : IM2COL_F64_QNAN;
        }
        return (ea == IM2COL_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)im2col_f64_clz64(m);
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
    if (exp_field >= IM2COL_F64_EXP_MASK) {
        return sa | IM2COL_F64_INF;
    }
    return sa | (exp_field << 52) | (m & IM2COL_F64_FRAC_MASK);
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・NaN は quiet NaN
// へ正規化）。`soft_f64::narrow_f64_bits` の逐語移植。
inline uint im2col_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & IM2COL_F64_EXP_MASK;
    ulong f = bits & IM2COL_F64_FRAC_MASK;
    if (e == IM2COL_F64_EXP_MASK) {
        return (f != 0ul) ? IM2COL_F32_QNAN : (sign | IM2COL_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | IM2COL_F32_INF;
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
        return sign | IM2COL_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// ---- im2col／col2im カーネル ----

kernel void im2col_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant Im2colDims& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.numel) {
        return;
    }
    long rem = (long)gid;
    long p_idx = rem % (long)dims.p;
    rem /= (long)dims.p;
    long k_idx = rem % (long)dims.k_g;
    rem /= (long)dims.k_g;
    long g = rem % (long)dims.groups;
    rem /= (long)dims.groups;
    long n = rem;

    long kw_ = k_idx % (long)dims.kw;
    long rest = k_idx / (long)dims.kw;
    long kh_ = rest % (long)dims.kh;
    long c_g = rest / (long)dims.kh;
    long c = g * (long)dims.cin_g + c_g;

    long ow = p_idx % (long)dims.w_out;
    long oh = p_idx / (long)dims.w_out;

    long h = oh * (long)dims.sh + kh_ * (long)dims.dh - (long)dims.ph;
    long w = ow * (long)dims.sw + kw_ * (long)dims.dw - (long)dims.pw;

    float value = 0.0f;
    if (h >= 0 && h < (long)dims.h_in && w >= 0 && w < (long)dims.w_in) {
        long in_idx = ((n * (long)dims.cin + c) * (long)dims.h_in + h) * (long)dims.w_in + w;
        value = in[in_idx];
    }
    out[gid] = value;
}

kernel void col2im_f32(
    device const float* d_col [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant Im2colDims& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.numel) {
        return;
    }
    long rem = (long)gid;
    long w = rem % (long)dims.w_in;
    rem /= (long)dims.w_in;
    long h = rem % (long)dims.h_in;
    rem /= (long)dims.h_in;
    long c = rem % (long)dims.cin;
    rem /= (long)dims.cin;
    long n = rem;

    long g = c / (long)dims.cin_g;
    long c_g = c % (long)dims.cin_g;

    ulong acc = im2col_f64_widen(as_type<uint>(0.0f));
    for (uint kh_ = 0; kh_ < dims.kh; kh_++) {
        long num_h = h + (long)dims.ph - (long)kh_ * (long)dims.dh;
        if (num_h < 0) continue;
        if (num_h % (long)dims.sh != 0) continue;
        long oh = num_h / (long)dims.sh;
        if (oh >= (long)dims.h_out) continue;
        for (uint kw_ = 0; kw_ < dims.kw; kw_++) {
            long num_w = w + (long)dims.pw - (long)kw_ * (long)dims.dw;
            if (num_w < 0) continue;
            if (num_w % (long)dims.sw != 0) continue;
            long ow = num_w / (long)dims.sw;
            if (ow >= (long)dims.w_out) continue;

            long k_idx = (c_g * (long)dims.kh + (long)kh_) * (long)dims.kw + (long)kw_;
            long p_idx = oh * (long)dims.w_out + ow;
            long col_idx = ((n * (long)dims.groups + g) * (long)dims.k_g + k_idx) * (long)dims.p + p_idx;
            acc = im2col_f64_add(acc, im2col_f64_widen(as_type<uint>(d_col[col_idx])));
        }
    }
    out[gid] = as_type<float>(im2col_f64_narrow(acc));
}
