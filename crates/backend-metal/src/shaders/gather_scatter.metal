// gather／scatter／scatter_add／one_hot カーネル（`torch.gather`／
// `torch.scatter`／`torch.scatter_add`／`torch.nn.functional.one_hot`
// 相当。gather／scatter／scatter_add はイシュー #1778、one_hot は
// イシュー #1755。CPU 参照実装 `crates/backend-cpu/src/gather_scatter.rs`
// の Metal 対応版）。
//
// `crate::gather_scatter::MetalGatherScatter`（`gather_scatter.rs`）から
// 実行時コンパイルされ、`ops.rs::MetalBackendOps::gather`／`scatter`／
// `one_hot` から呼ばれる。
//
// ---- one_hot（`one_hot_f32`。**非微分演算**）----
//
// 座標展開・ストライドが不要な単純な整数除算・剰余のみで読み書き位置が
// 決まる（`kernels_gather_scatter.rs::ONE_HOT_F32`〈CUDA 版〉と同型の
// 設計）: `row = gid / num_classes`・`c = gid % num_classes`、
// `out[gid] = (index[row] == c) ? 1.0 : 0.0`。
//
// ---- gather（`gather_f32`）----
//
// 1 スレッド = 1 出力位置（`index_shape` 上の row-major 添字 `gid`）。
// `gid` を `index_shape` 上で row-major に unravel し、`dim` 軸だけ
// `index[gid]` へ差し替えた `input` の row-major オフセットから読む。
// 出力位置間に書き込み衝突がないため決定的集約順序の契約は不要で、
// 純粋なコピーのため NaN payload を含め CPU 参照実装（`crates/backend-cpu/
// src/gather_scatter.rs::gather`）と bit 完全一致する。
//
// ---- scatter（`scatter_overwrite_f32`／`scatter_add_f32`）----
//
// **出力定常（output-stationary）方式**を採る: 1 スレッド = 1 出力位置
// `pos`（`out_shape` 上で row-major に unravel した多次元添字。
// `out_shape == input.shape()`）。`pos` の非 `dim` 座標が `index_shape`
// の対応軸の範囲内であれば（`scatter_out_shape` は非 `dim` 軸で
// `index_shape[axis] <= out_shape[axis]` を課すのみで、範囲外の `pos` は
// どの `index` 要素からも触れられない＝`input` のパススルー）、
// `index[…, j, …]`（`j` は `dim` 軸の座標）を `j = 0 .. index_shape[dim]`
// の**昇順**に走査し、`index == pos` の `dim` 座標と一致する要素だけを
// 取り込む。
//
// **CPU 参照実装との順序等価性（根拠）**: CPU 実装は `index_shape` の
// 全要素を row-major（行優先。最終軸が最速で変化）で走査し、`dim` 軸を
// 差し替えた出力位置へ書き込む。ある固定の出力位置 `pos` に着目すると、
// その位置へ書き込む `index` 要素は「`dim` 軸以外の座標がすべて `pos`
// と一致し、`dim` 軸の座標だけが異なる」もの全体であり、row-major
// 走査で他の座標をすべて固定したまま 1 軸だけを動かした部分列は、
// その軸がネストのどこに位置していても常に昇順で現れる（他の座標が
// すべて固定されているため）。したがって「出力位置ごとに `dim` 軸を
// 昇順走査する」出力定常カーネルは、CPU の全走査を出力位置ごとに
// 部分列として見た場合と同じ順序で要素を処理し、`Add`（逐次加算）・
// `Overwrite`（最後の書き手）いずれの意味論も CPU 参照実装と一致する。
//
// **数値方式（`scatter_add_f32` のみ）**: `.claude/rules/coding-rust.md`
// の「勾配の長軸縮約は `f64` アキュムレータで統一する」規約と同じ精度
// 規律を `ScatterReduce::Add`（`tensor-core::ScatterReduce` doc の
// 決定的集約契約）に適用する。MSL は `double` 型非対応のため、
// `gemm.metal::bias_f64_*`（イシュー #1566・#1659）と同じ IEEE 754
// binary64 逐次加算の 64bit 整数ソフトウェアエミュレーション
// （`gs_f64_widen`／`gs_f64_add`／`gs_f64_narrow`。本ファイルは意図的な
// 逐語複製——MSL は `newLibraryWithSource` でファイル単位にコンパイル
// され翻訳単位を共有できないため、`gemm.metal`／`layer_norm.metal` は
// 変更しない）を用いる。出力位置ごとのアキュムレータは
// `acc = gs_f64_widen(input[pos])` から開始し、取り込んだ `src` 値を
// `j` 昇順に `gs_f64_add` で逐次加算し、走査完了後に 1 回だけ
// `gs_f64_narrow` で `f32` へ downcast する。これはホスト参照実装
// （`crates/backend-cpu/src/gather_scatter.rs::scatter` の
// `ScatterReduce::Add` 分岐: `acc: f64` を `input as f64` から開始し
// `index`／`src` を row-major 走査順で `+= src as f64`、最後に 1 回
// `as f32`）と bit 完全一致する（NaN のみ payload がハードウェア依存の
// ため quiet NaN へ正規化しクラス一致で比較する）。
//
// `scatter_overwrite_f32` は `Add` を経由しない単純代入（見つかった
// 最後の値、なければ `input[pos]` のパススルー）のため `f64` 昇格は
// 不要で丸めを伴わない。
//
// **ホスト側の逐語モデル**: `crates/backend-metal/src/
// gather_scatter_model.rs`（`gather_model`／`scatter_model`）が本
// ファイルのアルゴリズムのホスト側逐語再現であり、CPU 参照実装
// （`fandhe_ai_backend_cpu::CpuBackendOps`）との bit 一致をユニット
// テスト（Linux 実行可能）で網羅検証する。本ファイルを変更した場合は
// 同モジュールも追従させること。実機でのカーネル出力 bit 一致は
// `tests/gather_scatter_parity.rs`（`#[ignore]`）で確認する。
//
// **添字計算・境界検査（REQ-8）**: shape 配列（`in_shape`／`index_shape`／
// `out_shape`）はすべて `constant uint*` で渡し、添字計算は `ulong`
// （64bit）で行う（`numel` が `u32` を超えないことはホスト側で検証済み
// だが、中間の row-major ストライド積が `u32` を超えうるため）。
// `gid >= numel` の早期 return（REQ-8「手動境界チェックを省略しない」）
// を必ず持つ。`index` の値は呼び出し元（`gather_scatter.rs`）がホスト
// 側で `[0, dim_size)` を事前検査するが、カーネル側にも防御的な範囲外
// ガードを残す（多層防御。到達時は gather=書き込みスキップ〈出力
// そのまま=呼び出し前に確保したバッファの初期値。呼び出し元がゼロ
// 初期化する〉・scatter=当該要素スキップ）。

#include <metal_stdlib>
using namespace metal;

// `MetalGatherScatter::new`（`crate::gather_scatter`）が本ファイルを
// そのまま `newLibraryWithSource_options_error` へ渡して実行時
// コンパイルするため、`clz`／`as_type` 等の標準ライブラリ関数を
// 名前空間修飾なしで使うには上記 2 行が必須（`gemm.metal` と同じ
// 構成。欠落すると全経路が `LibraryCompilation` エラーになる。
// codex-review 指摘。イシュー #1799）。

// ---- IEEE 754 binary64 逐次加算のソフトウェアエミュレーション ----
// `crates/backend-metal/src/soft_f64.rs` の逐語移植（`gemm.metal::
// bias_f64_*` と同一構造。`u64`→`ulong`・`u32`→`uint`・
// `leading_zeros()`→`clz()`。本ファイル冒頭コメント「数値方式」参照）。
// 定数は同モジュールの `F64_*`／`F32_*` と同値。

#define GS_F64_SIGN      0x8000000000000000ul
#define GS_F64_EXP_MASK  0x7FFul
#define GS_F64_FRAC_MASK 0x000FFFFFFFFFFFFFul
#define GS_F64_QNAN      0x7FF8000000000000ul
#define GS_F64_INF       0x7FF0000000000000ul
#define GS_F32_QNAN      0x7FC00000u
#define GS_F32_INF       0x7F800000u

// 64bit leading zero count を 32bit `clz` 2 回で構成する（`clz(0u) == 32`
// は MSL 仕様で定義済み。`soft_f64::clz64`／`bias_clz64` と同一構造）。
inline uint gs_f64_clz64(ulong x) {
    uint hi = (uint)(x >> 32);
    uint lo = (uint)x;
    return (hi != 0u) ? clz(hi) : (32u + clz(lo));
}

// `f64::from(f32)`（NaN は quiet NaN へ正規化）。`soft_f64::widen_f32_bits`。
inline ulong gs_f64_widen(uint bits) {
    ulong sign = ((ulong)(bits >> 31)) << 63;
    uint exp = (bits >> 23) & 0xFFu;
    ulong frac = (ulong)(bits & 0x7FFFFFu);
    if (exp == 0xFFu) {
        return (frac != 0ul) ? GS_F64_QNAN : (sign | GS_F64_INF);
    }
    if (exp == 0u) {
        if (frac == 0ul) {
            return sign;
        }
        // f32 subnormal（`frac × 2^-149`）は f64 では正規化数。
        uint p = 31u - clz((uint)frac);
        ulong exp64 = (ulong)((int)p - 149 + 1023);
        ulong frac64 = (frac << (52u - p)) & GS_F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    ulong exp64 = (ulong)exp + (1023ul - 127ul); // 減算を先にすると exp < 127 で下溢れ。
    return sign | (exp64 << 52) | (frac << 29);
}

// `f64 + f64`（最近接偶数丸め。NaN は quiet NaN へ正規化）。
// `soft_f64::add_f64_bits` と同一手順。
inline ulong gs_f64_add(ulong a, ulong b) {
    ulong sa = a & GS_F64_SIGN;
    ulong sb = b & GS_F64_SIGN;
    ulong ea = (a >> 52) & GS_F64_EXP_MASK;
    ulong eb = (b >> 52) & GS_F64_EXP_MASK;
    ulong fa = a & GS_F64_FRAC_MASK;
    ulong fb = b & GS_F64_FRAC_MASK;

    if (ea == GS_F64_EXP_MASK || eb == GS_F64_EXP_MASK) {
        bool a_nan = (ea == GS_F64_EXP_MASK) && (fa != 0ul);
        bool b_nan = (eb == GS_F64_EXP_MASK) && (fb != 0ul);
        if (a_nan || b_nan) {
            return GS_F64_QNAN;
        }
        if (ea == GS_F64_EXP_MASK && eb == GS_F64_EXP_MASK) {
            return (sa == sb) ? a : GS_F64_QNAN;
        }
        return (ea == GS_F64_EXP_MASK) ? a : b;
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
    // `|a| >= |b|` に揃える（指数・仮数の辞書順比較）。
    if (ea_eff < eb_eff || (ea_eff == eb_eff && ma < mb)) {
        ulong t;
        t = ma; ma = mb; mb = t;
        t = ea_eff; ea_eff = eb_eff; eb_eff = t;
        t = sa; sa = sb; sb = t;
    }
    ma <<= 3;
    mb <<= 3;
    // 桁合わせ。64 以上のシフトは UB のため sticky のみへ縮退させる。
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
            // 完全相殺は最近接丸めでは `+0`。
            return 0ul;
        }
        ulong sh = (ulong)gs_f64_clz64(m);
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
    if (exp_field >= GS_F64_EXP_MASK) {
        return sa | GS_F64_INF;
    }
    return sa | (exp_field << 52) | (m & GS_F64_FRAC_MASK);
}

// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・underflow は f32
// subnormal／`±0`。NaN は quiet NaN へ正規化）。`soft_f64::narrow_f64_bits`。
inline uint gs_f64_narrow(ulong bits) {
    uint sign = ((uint)(bits >> 63)) << 31;
    ulong e = (bits >> 52) & GS_F64_EXP_MASK;
    ulong f = bits & GS_F64_FRAC_MASK;
    if (e == GS_F64_EXP_MASK) {
        return (f != 0ul) ? GS_F32_QNAN : (sign | GS_F32_INF);
    }
    if (e == 0ul && f == 0ul) {
        return sign;
    }
    ulong m = (e == 0ul) ? f : (f | (1ul << 52));
    long ee = (e == 0ul) ? -1022l : ((long)e - 1023l);
    long ef = ee + 127l;
    if (ef >= 255l) {
        return sign | GS_F32_INF;
    }
    long extra = (ef <= 0l) ? (1l - ef) : 0l;
    long shift_l = 29l + extra;
    if (shift_l >= 54l) {
        return sign;
    }
    uint shift = (uint)shift_l;
    ulong q0 = m >> shift;
    ulong rem = m & ((1ul << shift) - 1ul);
    ulong halfway = 1ul << (shift - 1u);
    ulong q = q0;
    if (rem > halfway || (rem == halfway && (q0 & 1ul) == 1ul)) {
        q += 1ul;
    }
    uint exp_field = (ef <= 0l) ? 0u : (uint)ef;
    if (ef <= 0l) {
        if (q >= (1ul << 23)) {
            exp_field = 1u;
            q -= (1ul << 23);
        }
    } else {
        if (q >= (1ul << 24)) {
            q >>= 1;
            exp_field += 1u;
        }
        q -= (1ul << 23);
    }
    if (exp_field >= 255u) {
        return sign | GS_F32_INF;
    }
    return sign | (exp_field << 23) | (uint)q;
}

// rank 上限（呼び出し元がホスト側で検証する。`gather_scatter.rs::
// MAX_GATHER_SCATTER_RANK` と一致させる）。`shapes` バッファの
// スタック配列サイズとして使う。
#define GS_MAX_RANK 8u

// 線形添字（row-major）を `rank` 次元の shape 上で多次元添字へ展開し
// `coords` へ書き込む（末尾軸から順に `rem % shape[a]` で分解）。
inline void gs_unravel(ulong flat, constant uint* shape, uint rank, thread ulong* coords) {
    ulong rem = flat;
    for (uint a = rank; a > 0u; a--) {
        uint axis = a - 1u;
        ulong d = (ulong)shape[axis];
        if (d == 0ul) {
            coords[axis] = 0ul;
            continue;
        }
        coords[axis] = rem % d;
        rem /= d;
    }
}

// 多次元添字（row-major）を線形オフセットへ畳み込む。
inline ulong gs_ravel(thread const ulong* coords, constant uint* shape, uint rank) {
    ulong offset = 0ul;
    for (uint axis = 0u; axis < rank; axis++) {
        offset = offset * (ulong)shape[axis] + coords[axis];
    }
    return offset;
}

// gather: `input(0)`／`index(1, int)`／`out(2)`／
// `shapes(3, uint: in_shape[rank] ++ index_shape[rank])`／
// `constant uint& rank(4)`／`dim(5)`／`numel(6)`（出力要素数
// ＝`index_shape` の要素数積）。
kernel void gather_f32(
    device const float* input [[buffer(0)]],
    device const int* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shapes [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant uint& numel [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // REQ-8: grid は `ceil(numel/W)` threadgroup のため端で `numel` を
    // はみ出しうる（手動境界チェックを省略しない）。
    if (gid >= numel) {
        return;
    }
    constant uint* in_shape = shapes;
    constant uint* index_shape = shapes + rank;

    thread ulong coords[GS_MAX_RANK];
    gs_unravel((ulong)gid, index_shape, rank, coords);

    int dim_idx_raw = index[gid];
    uint dim_size = in_shape[dim];
    // 防御的ガード（多層防御。ホスト側 `gather_scatter.rs` が事前検査
    // 済みのため通常到達しない。到達時は出力バッファの初期値のまま
    // スキップする——呼び出し元がゼロ初期化して渡す契約）。
    if (dim_idx_raw < 0 || (uint)dim_idx_raw >= dim_size) {
        return;
    }
    coords[dim] = (ulong)dim_idx_raw;
    ulong src_off = gs_ravel(coords, in_shape, rank);
    out[gid] = input[src_off];
}

// scatter（`Overwrite`）: `input(0)`／`index(1)`／`src(2)`／`out(3)`／
// `shapes(4, uint: out_shape[rank] ++ index_shape[rank])`／
// `constant uint& rank(5)`／`dim(6)`／`numel_out(7)`（出力要素数
// ＝`out_shape`＝`input.shape()` の要素数積）。
kernel void scatter_overwrite_f32(
    device const float* input [[buffer(0)]],
    device const int* index [[buffer(1)]],
    device const float* src [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* shapes [[buffer(4)]],
    constant uint& rank [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant uint& numel_out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel_out) {
        return;
    }
    constant uint* out_shape = shapes;
    constant uint* index_shape = shapes + rank;

    thread ulong pos[GS_MAX_RANK];
    gs_unravel((ulong)gid, out_shape, rank, pos);

    // 非 `dim` 軸のいずれかで `pos` が `index_shape` の範囲外なら、
    // どの `index` 要素からも触れられない（`input` のパススルー）。
    bool in_range = true;
    for (uint axis = 0u; axis < rank; axis++) {
        if (axis == dim) {
            continue;
        }
        if (pos[axis] >= (ulong)index_shape[axis]) {
            in_range = false;
            break;
        }
    }

    float result = input[gid];
    uint dim_size_index = index_shape[dim];
    if (in_range) {
        thread ulong idx_coords[GS_MAX_RANK];
        for (uint axis = 0u; axis < rank; axis++) {
            idx_coords[axis] = pos[axis];
        }
        // `j` 昇順走査（本ファイル冒頭コメント「CPU 参照実装との順序
        // 等価性」）。最後に一致した要素が残る（`Overwrite` 意味論）。
        for (uint j = 0u; j < dim_size_index; j++) {
            idx_coords[dim] = (ulong)j;
            ulong idx_off = gs_ravel(idx_coords, index_shape, rank);
            int dim_idx_raw = index[idx_off];
            uint dim_size_out = out_shape[dim];
            // 防御的ガード（多層防御。gather と同じ理由）。
            if (dim_idx_raw < 0 || (uint)dim_idx_raw >= dim_size_out) {
                continue;
            }
            if ((ulong)dim_idx_raw == pos[dim]) {
                result = src[idx_off];
            }
        }
    }
    out[gid] = result;
}

// scatter（`Add`）: バッファ配置は `scatter_overwrite_f32` と同一。
// `ScatterReduce::Add` の決定的集約契約（本ファイル冒頭コメント
// 「数値方式」）に従い、出力位置ごとの soft-f64 アキュムレータへ
// `j` 昇順に逐次加算する。
kernel void scatter_add_f32(
    device const float* input [[buffer(0)]],
    device const int* index [[buffer(1)]],
    device const float* src [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* shapes [[buffer(4)]],
    constant uint& rank [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant uint& numel_out [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel_out) {
        return;
    }
    constant uint* out_shape = shapes;
    constant uint* index_shape = shapes + rank;

    thread ulong pos[GS_MAX_RANK];
    gs_unravel((ulong)gid, out_shape, rank, pos);

    bool in_range = true;
    for (uint axis = 0u; axis < rank; axis++) {
        if (axis == dim) {
            continue;
        }
        if (pos[axis] >= (ulong)index_shape[axis]) {
            in_range = false;
            break;
        }
    }

    // `acc = input[pos]`（f64 へ昇格）から開始し、一致した `src` 値を
    // `j` 昇順に逐次加算する（本ファイル冒頭コメント「数値方式」）。
    ulong acc = gs_f64_widen(as_type<uint>(input[gid]));
    if (in_range) {
        thread ulong idx_coords[GS_MAX_RANK];
        for (uint axis = 0u; axis < rank; axis++) {
            idx_coords[axis] = pos[axis];
        }
        uint dim_size_index = index_shape[dim];
        for (uint j = 0u; j < dim_size_index; j++) {
            idx_coords[dim] = (ulong)j;
            ulong idx_off = gs_ravel(idx_coords, index_shape, rank);
            int dim_idx_raw = index[idx_off];
            uint dim_size_out = out_shape[dim];
            if (dim_idx_raw < 0 || (uint)dim_idx_raw >= dim_size_out) {
                continue;
            }
            if ((ulong)dim_idx_raw == pos[dim]) {
                acc = gs_f64_add(acc, gs_f64_widen(as_type<uint>(src[idx_off])));
            }
        }
    }
    out[gid] = as_type<float>(gs_f64_narrow(acc));
}

// one_hot（**非微分演算**）: `index(0)`／`out(1)`／`num_classes(2)`／
// `numel(3)`（出力要素数＝`index.numel() * num_classes`）。
kernel void one_hot_f32(
    device const int* index [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_classes [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // REQ-8: grid は `ceil(numel/W)` threadgroup のため端で `numel` を
    // はみ出しうる（手動境界チェックを省略しない）。
    if (gid >= numel) {
        return;
    }
    uint row = gid / num_classes;
    uint c = gid % num_classes;
    int index_val = index[row];
    // 範囲外添字（呼び出し元 `ops.rs` が起動前にホスト側で検査済みの
    // ため通常到達しない。REQ-8 の縦深防御として境界外読み出しを回避し
    // 安全側の値を書く。CUDA 版 `ONE_HOT_F32` と同じ方針）。
    out[gid] = (index_val >= 0 && (uint)index_val < num_classes && (uint)index_val == c)
        ? 1.0f
        : 0.0f;
}
