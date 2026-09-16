// f32 `sum` reduction（全要素・単一軸。`torch.sum` 相当。イシュー
// #1895・親イシュー #1894）。
//
// `crate::reduce::MetalReduce`（`reduce.rs`）から実行時コンパイルされる。
// `ops.rs::MetalBackendOps::sum` は `context_cache::cached_reduce` 経由で
// `MetalReduce` を呼び本カーネルへ到達する（イシュー #1896）。ホスト
// モデルは `crate::reduce_model`。
//
// ---- 数値方式（binary64 ソフトウェアエミュレーション）----
//
// MSL は `double` 型非対応のため、`.claude/rules/coding-rust.md`
// 「正規化統計・勾配の長軸縮約は `f64` アキュムレータで統一する」節・
// `crate::soft_f64` と同じ binary64 逐次演算の 64bit 整数ソフトウェア
// エミュレーションを `sum` へ適用する（`red_f64_widen`／`red_f64_add`／
// `red_f64_narrow`。`scan.metal::scan_f64_*` の逐語移植——MSL は
// `newLibraryWithSource` でファイル単位にコンパイルされ翻訳単位を
// 共有できないため、`red_f64_` 接頭辞を付けた本ファイル内で独立に
// 定義する〈意図的な重複〉。ホスト側の逐語モデルは `crate::soft_f64`／
// `crate::reduce_model` を参照）。
//
// `fandhe_ai_backend_cpu::reduction::sum` は 2 つの異なる演算順序を
// 持つ（`crates/backend-cpu/src/reduction.rs` モジュール doc「決定性
// 契約」参照）ため、本ファイルもその両方を逐語再現する。
//
// **全要素（`dim=None`）**: CPU `sum_slice_f64` は `data` を
// `CHUNK`（4096）単位に分割し、各チャンク内を `0.0` から index 順に
// 逐次加算したうえで、チャンク部分和をチャンク番号順に `0.0` から
// 逐次加算し、最後に 1 回だけ `f32` へ downcast する。本ファイルは
// この 2 段構成を 2 カーネルへ写像する:
//   1. `reduce_sum_all_chunk_f32`: 1 スレッド = 1 チャンク（`REDUCE_
//      SUM_CHUNK=4096` 要素）。チャンク内を `widen(0.0)` から昇順に
//      `red_f64_add` で逐次加算し、`narrow` せず `f64` bit
//      （`ulong`）のまま `partial[gid]` へ書く。
//   2. `reduce_sum_all_finalize_f32`: 単一スレッド（`gid == 0` のみ
//      実処理。他は早期 return）。`partial[0..num_chunks]` を
//      `widen(0.0)` から昇順に `red_f64_add` で逐次加算し、最後に
//      1 回だけ `red_f64_narrow` して `out[0]` へ書く。
//
// **単一軸（`dim=Some(axis)`）**: CPU `axis_reduce_sum` は出力要素
// （`outer × inner` 個）ごとに独立して `0.0` から縮約軸を昇順に逐次
// 加算し `f32` へ downcast する。`reduce_sum_axis_f32` は 1 スレッド =
// 1 lane（`gid` は出力要素の平坦添字）で同じ演算列を実行する。
//
// この演算列はホスト `f64` 逐次参照実装（`backend-cpu::reduction::
// sum`）と bit 完全一致する契約（NaN のみ payload がハードウェア依存
// のためクラス一致。`crate::reduce_model` の単体テストが Linux で
// 機械的に裏付ける）。
//
// ---- 既知の制約（並列度。`.claude/rules/out-of-scope-tracking.md`
// 対象）----
//
// `scan.metal` モジュール doc と同じ理由: ブロック内の対数段 reduction
// アルゴリズムは `f64` 逐次加算列と異なる結合順序になり bit 一致しなく
// なるため採らない。したがって全要素縮約は `numel/REDUCE_SUM_CHUNK`
// スレッドのみ・単一軸縮約は `outer*inner` スレッドのみしか並列度が
// 出ない（`axis_len` 方向は当該スレッド内で逐次）。結合順序を変えると
// bit 一致契約が崩れるため、この制約は性能改善では解消できない
// （後続イシューで並列度を改善する場合は結合順序の維持を要件とする）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// 各カーネルは `gid`（グローバルスレッド id）が起動対象範囲
// （`num_chunks`／`1`／`lanes`）未満かどうかを検査してから処理する。
// 添字計算は `ulong`（64bit）で行う。

#include <metal_stdlib>
using namespace metal;

// ---- IEEE 754 binary64 のソフトウェアエミュレーション ----
// `crates/backend-metal/src/soft_f64.rs`（`widen_f32_bits`／
// `add_f64_bits`／`narrow_f64_bits`）の逐語移植（本ファイル冒頭
// コメント参照）。`sum` は加算のみ使うため乗算・除算は含めない。

#define RED_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define RED_F64_EXP_MASK  0x7FFul
#define RED_F64_SIGN      0x8000000000000000ul
#define RED_F64_QNAN      0x7FF8000000000000ul
#define RED_F64_INF       0x7FF0000000000000ul
#define RED_F32_QNAN      0x7FC00000u
#define RED_F32_INF       0x7F800000u

// 全要素縮約（`dim=None`）のチャンク分割サイズ。`crate::reduce_model::
// REDUCE_SUM_CHUNK`（Rust 側）と同値であることを
// `tests/reduce_source_evidence.rs` が機械的に検証する。
#define REDUCE_SUM_CHUNK 4096u

// 64bit leading zero count（`clz(0u) == 32` は MSL 仕様で定義済み。
// `soft_f64::clz64`／`scan_f64_clz64` と同一構造）。
inline uint red_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`
// の逐語移植。
inline ulong red_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? RED_F64_QNAN : (sign | RED_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & RED_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul);
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` の逐語移植。
inline ulong red_f64_add(ulong a, ulong b) {
    ulong sa = a & RED_F64_SIGN;
    ulong sb = b & RED_F64_SIGN;
    ulong ea = (a >> 52) & RED_F64_EXP_MASK;
    ulong eb = (b >> 52) & RED_F64_EXP_MASK;
    ulong fa = a & RED_F64_FRAC_MASK;
    ulong fb = b & RED_F64_FRAC_MASK;

    if (ea == RED_F64_EXP_MASK || eb == RED_F64_EXP_MASK) {
        bool a_nan = (ea == RED_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == RED_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return RED_F64_QNAN;
        }
        if (ea == RED_F64_EXP_MASK && eb == RED_F64_EXP_MASK) {
            return (sa == sb) ? a : RED_F64_QNAN;
        }
        return (ea == RED_F64_EXP_MASK) ? a : b;
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
        ulong sh = (ulong)red_f64_clz64(m);
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
    if (exp_field >= RED_F64_EXP_MASK) {
        return sa | RED_F64_INF;
    }
    return sa | (exp_field << 52) | (m & RED_F64_FRAC_MASK);
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・NaN は quiet NaN
// へ正規化）。`soft_f64::narrow_f64_bits` の逐語移植。
inline uint red_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & RED_F64_EXP_MASK;
    ulong f = bits & RED_F64_FRAC_MASK;
    if (e == RED_F64_EXP_MASK) {
        return (f != 0ul) ? RED_F32_QNAN : (sign | RED_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | RED_F32_INF;
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
        return sign | RED_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// ---- 全要素 sum reduction（2 段構成）----
//
// `x`: `numel` 要素の稠密（contiguous）スライス（呼び出し元
// `reduce.rs::MetalReduce::run_sum_all_f32` が保証）。
// `partial`: `num_chunks` 要素の `ulong`（`f64` bit 表現。narrow
// しない中間値）バッファ。`out`: 1 要素（最終 `f32` 結果）。

kernel void reduce_sum_all_chunk_f32(
    device const float* x [[buffer(0)]],
    device ulong* partial [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    constant uint& num_chunks [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_chunks) {
        return;
    }
    ulong begin = (ulong)gid * (ulong)REDUCE_SUM_CHUNK;
    ulong end = begin + (ulong)REDUCE_SUM_CHUNK;
    if (end > (ulong)numel) {
        end = (ulong)numel;
    }
    ulong acc = red_f64_widen(as_type<uint>(0.0f));
    for (ulong idx = begin; idx < end; idx++) {
        acc = red_f64_add(acc, red_f64_widen(as_type<uint>(x[idx])));
    }
    partial[gid] = acc;
}

kernel void reduce_sum_all_finalize_f32(
    device const ulong* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_chunks [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0u) {
        return;
    }
    ulong acc = red_f64_widen(as_type<uint>(0.0f));
    for (uint c = 0u; c < num_chunks; c++) {
        acc = red_f64_add(acc, partial[c]);
    }
    out[0] = as_type<float>(red_f64_narrow(acc));
}

// ---- 単一軸 sum reduction ----
//
// `x`／`out`: `x` は `outer * axis_len * inner` 要素、`out` は
// `outer * inner` 要素（呼び出し元 `reduce.rs::MetalReduce::
// run_sum_axis_f32` が `contiguous()` で稠密化した入力を渡す契約）。
// `lanes = outer * inner`（`gid` の境界検査対象）。`inner` は `gid` を
// `o = gid / inner`・`i = gid % inner` へ分解するための除数
// （`scan.metal::cumsum_f32` と同じ添字規約）。

kernel void reduce_sum_axis_f32(
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
    ulong acc = red_f64_widen(as_type<uint>(0.0f));
    for (uint a = 0u; a < axis_len; a++) {
        ulong idx = (o * (ulong)axis_len + (ulong)a) * (ulong)inner + i;
        acc = red_f64_add(acc, red_f64_widen(as_type<uint>(x[idx])));
    }
    out[gid] = as_type<float>(red_f64_narrow(acc));
}
