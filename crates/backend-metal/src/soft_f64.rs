//! IEEE 754 binary64（`f64`）逐次加算のソフトウェアエミュレーション
//! （64bit 整数演算のみ）。`crates/backend-metal/src/shaders/gemm.metal::
//! gemm_bias_grad_reduce_f32` が bias 勾配の行方向縮約に使う演算列の
//! **ホスト側の逐語モデル**であり、MSL 側の `bias_f64_*` 関数群と 1 対 1 に
//! 対応する（イシュー #1566・PR #1659）。
//!
//! # なぜ必要か
//!
//! `.claude/rules/coding-rust.md` は「勾配の長軸縮約は `f64` アキュムレータで
//! 統一する」と定めるが、MSL（Metal Shading Language）は `double` 型非対応
//! である。PR #1659 の当初実装（f32 のみの Neumaier 補償和 + 2 の冪 scale）は
//! `[2^48, 2^24, 1, -2^48, -2^24]` のような相殺列でホスト `f64` 逐次和と
//! 一致せず（codex-review P1 指摘の再現）、判定契約側を緩める（Tier A/B）
//! 案はレビューで受け入れられなかった。本モジュールは `f64` 加算そのものを
//! 64bit 整数で忠実に再現することで、ホスト参照実装
//! （`crate::layout::reduce_bias_grad_rows_host`／`autodiff::eval::
//! reduce_bias_grad_rows`。`f64` 逐次和 → 最後に 1 回 `f32` へ downcast）と
//! **bit 完全一致**させる。tolerance・baseline は一切変更しない。
//!
//! # 契約
//!
//! - [`widen_f32_bits`]・[`add_f64_bits`]・[`narrow_f64_bits`] はそれぞれ
//!   Rust の `f64::from(f32)`・`f64 + f64`（丸めモードは最近接偶数）・
//!   `f64 as f32` と、**NaN を除き bit 完全一致**する（本モジュールの
//!   ユニットテストが `to_bits` で網羅的・ランダムに検証する）
//! - NaN は正規化された quiet NaN（`0x7FF8_0000_0000_0000`／`0x7FC0_0000`）
//!   へ畳む。ハードウェアの NaN payload 伝播は実装依存（MSL 側も同様）の
//!   ため、NaN は「クラス一致」（`is_nan` 同士）で比較する
//! - 中間値の overflow は `f64` の表現範囲に従う（有限入力の和が `f64`
//!   範囲を超えることは f32 入力では起きない）。最終 downcast の overflow は
//!   `f64 as f32` と同じく `±inf` になる
//!
//! # MSL 側との対応
//!
//! `gemm.metal` の `bias_f64_widen`／`bias_f64_add`／`bias_f64_narrow`／
//! `bias_clz64` は本モジュールの同名関数の逐語移植であり、`u64`→`ulong`・
//! `u32`→`uint`・`leading_zeros()`→`clz()`（`clz(0u) == 32` は MSL 仕様で
//! 定義済み）の置換のみで対応する。シフト量が 64 以上になる経路は両言語で
//! 未定義動作（Rust は debug panic・MSL は UB）のため、必ず分岐で除外して
//! から shift する（`add_f64_bits` の桁合わせ・`narrow_f64_bits` の
//! subnormal 経路）。`gemm.metal` を変更した場合は本ファイルも追従させる。
//!
//! # `layer_norm.metal` との対応（イシュー #1596・PR #1671）
//!
//! 本モジュールは `layer_norm.metal` の `ln_f64_*` 系関数（`widen`・
//! `add`・`add_ro`・`sub`・`mul`・`div`・`narrow`・`recip_newton`・
//! `rsqrt_newton`）のホスト側逐語モデルでもある（`gemm.metal` 用途と
//! 機能重複するが、MSL は `newLibraryWithSource` でファイル単位に
//! コンパイルされ翻訳単位を共有できないため、両ファイルは意図的に
//! 独立実装を持つ。逐語対応の詳細は `layer_norm.metal` 冒頭コメント
//! 「ホスト側の逐語モデル」を参照）。
//!
//! [`add_f64_bits_round_to_odd`] は [`add_f64_bits`] とは異なり
//! **round-to-odd**（RO）丸めであり、Rust の `f64 +` 演算子に対応する
//! ものではない（RO 自体は IEEE 754 の標準丸めモードではなく、二重
//! 丸め回避のための補助的な丸め）。そのため [`fma_f32_bits`]
//! （`a*b+c` を単一丸め FMA として計算する合成関数。LayerNorm の affine
//! `x̂·w+b` のホスト側逐語モデル）経由でハードウェア `f32::mul_add` と
//! bit 完全一致することをユニットテストで検証する（`add_f64_bits_
//! round_to_odd` 自体の直接の比較対象は存在しない）。

/// 64bit 値の leading zero count。MSL の `clz(ulong)` の可用性に依存せず
/// 32bit `clz` 2 回で構成する（MSL 側 `bias_clz64` と同一構造）。
#[inline]
pub fn clz64(x: u64) -> u32 {
    let hi = (x >> 32) as u32;
    let lo = x as u32;
    if hi != 0 {
        hi.leading_zeros()
    } else {
        32 + lo.leading_zeros()
    }
}

const F64_SIGN: u64 = 1u64 << 63;
const F64_EXP_MASK: u64 = 0x7FF;
const F64_FRAC_MASK: u64 = (1u64 << 52) - 1;
const F64_QNAN: u64 = 0x7FF8_0000_0000_0000;
const F64_INF: u64 = 0x7FF0_0000_0000_0000;
const F32_QNAN: u32 = 0x7FC0_0000;
const F32_INF: u32 = 0x7F80_0000;

/// `f64::from(f32)` の bit 表現版（NaN は quiet NaN へ正規化）。
#[inline]
pub fn widen_f32_bits(bits: u32) -> u64 {
    let sign = ((bits >> 31) as u64) << 63;
    let exp = (bits >> 23) & 0xFF;
    let frac = (bits & 0x7F_FFFF) as u64;
    if exp == 0xFF {
        return if frac != 0 { F64_QNAN } else { sign | F64_INF };
    }
    if exp == 0 {
        if frac == 0 {
            return sign;
        }
        // f32 subnormal（`frac × 2^-149`）は f64 では正規化数。先頭 1 の
        // 位置 `p`（0..=22）から指数を決め、仮数を bit 52 へ揃える。
        let p = 31 - (frac as u32).leading_zeros();
        let exp64 = (p as i32 - 149 + 1023) as u64;
        let frac64 = (frac << (52 - p)) & F64_FRAC_MASK;
        return sign | (exp64 << 52) | frac64;
    }
    // `exp + (1023 - 127)`（減算を先にすると `exp < 127` で unsigned 下溢れ）。
    let exp64 = (exp as u64) + (1023 - 127);
    sign | (exp64 << 52) | (frac << 29)
}

/// `f64 + f64`（最近接偶数丸め）の bit 表現版（NaN は quiet NaN へ正規化）。
///
/// 手順は標準的な softfloat と同じ: 特殊値の分岐 → 仮数を 3 bit のガード
/// （guard／round／sticky）付きで桁合わせ → 加減算 → 正規化 → 丸め。
/// `|a| >= |b|` になるよう入れ替えてから減算するため、仮数の減算で負に
/// なることはない。
pub fn add_f64_bits(a: u64, b: u64) -> u64 {
    let sa = a & F64_SIGN;
    let sb = b & F64_SIGN;
    let ea = (a >> 52) & F64_EXP_MASK;
    let eb = (b >> 52) & F64_EXP_MASK;
    let fa = a & F64_FRAC_MASK;
    let fb = b & F64_FRAC_MASK;

    // NaN／inf。
    if ea == F64_EXP_MASK || eb == F64_EXP_MASK {
        let a_nan = ea == F64_EXP_MASK && fa != 0;
        let b_nan = eb == F64_EXP_MASK && fb != 0;
        if a_nan || b_nan {
            return F64_QNAN;
        }
        if ea == F64_EXP_MASK && eb == F64_EXP_MASK {
            // inf + inf（同符号）は inf、inf + (-inf) は NaN。
            return if sa == sb { a } else { F64_QNAN };
        }
        return if ea == F64_EXP_MASK { a } else { b };
    }
    // ゼロ。`(+0) + (-0) == +0`・`(-0) + (-0) == -0`（最近接丸め）。
    let a_zero = ea == 0 && fa == 0;
    let b_zero = eb == 0 && fb == 0;
    if a_zero && b_zero {
        return sa & sb;
    }
    if a_zero {
        return b;
    }
    if b_zero {
        return a;
    }

    // 仮数に隠れ 1 を立て（subnormal は実効指数 1・隠れ 1 なし）、
    // ガード 3 bit 分だけ左へ寄せる（bit 55 が隠れ 1 の位置）。
    let (mut ma, ea_eff) = if ea == 0 {
        (fa, 1u64)
    } else {
        (fa | (1u64 << 52), ea)
    };
    let (mut mb, eb_eff) = if eb == 0 {
        (fb, 1u64)
    } else {
        (fb | (1u64 << 52), eb)
    };
    let (mut sa, mut sb) = (sa, sb);
    // `|a| >= |b|` に揃える（指数・仮数の辞書順比較）。
    let (mut ea_eff, mut eb_eff) = (ea_eff, eb_eff);
    if (ea_eff, ma) < (eb_eff, mb) {
        core::mem::swap(&mut ma, &mut mb);
        core::mem::swap(&mut ea_eff, &mut eb_eff);
        core::mem::swap(&mut sa, &mut sb);
    }
    ma <<= 3;
    mb <<= 3;
    // 桁合わせ。差が 64 以上のシフトは未定義動作のため、`mb` を sticky
    // のみ（`mb != 0` なら 1）へ縮退させる（56 bit 幅の仮数に対し差 ≥ 57 で
    // 既に整数部が消えるので結果は同じ）。
    let d = ea_eff - eb_eff;
    if d >= 64 {
        mb = u64::from(mb != 0);
    } else if d > 0 {
        let lost = mb & ((1u64 << d) - 1);
        mb = (mb >> d) | u64::from(lost != 0);
    }

    let mut e = ea_eff;
    let mut m;
    if sa == sb {
        m = ma + mb;
        if m >= (1u64 << 56) {
            // 繰り上がり。落ちる最下位 bit は sticky へ畳む。
            let lost = m & 1;
            m = (m >> 1) | lost;
            e += 1;
        }
    } else {
        m = ma - mb;
        if m == 0 {
            // 完全相殺は最近接丸めでは `+0`。
            return 0;
        }
        // 正規化: 先頭 1 を bit 55 へ。ただし実効指数は 1 未満にできない
        // （それ以下は subnormal として bit 55 未満のまま残す）。
        let mut sh = clz64(m) as u64;
        sh = sh.saturating_sub(8);
        if sh > e - 1 {
            sh = e - 1;
        }
        m <<= sh;
        e -= sh;
    }

    // 丸め（ガード 3 bit）: `r > 4` は切り上げ、`r == 4` は偶数へ。
    let r = m & 7;
    m >>= 3;
    if r > 4 || (r == 4 && (m & 1) == 1) {
        m += 1;
    }
    if m >= (1u64 << 53) {
        m >>= 1;
        e += 1;
    }
    // bit 52（隠れ 1）が立っていれば正規化数、立っていなければ subnormal
    // （このとき `e == 1`）。指数が 0x7FF 以上なら overflow → inf。
    let exp_field = if m >= (1u64 << 52) { e } else { 0 };
    if exp_field >= F64_EXP_MASK {
        return sa | F64_INF;
    }
    sa | (exp_field << 52) | (m & F64_FRAC_MASK)
}

/// `f64 + f64` を**round-to-odd**（RO）で丸めた bit 表現版（PR #1671
/// codex-review 指摘・イシュー #1596 是正: LayerNorm affine `x̂·w+b` の
/// 二重丸め回避）。[`add_f64_bits`] とは丸め規則のみが異なる姉妹関数
/// （逐語複製。共通化すると分岐が増え可読性が落ちるため意図的に複製する。
/// `.claude/rules/code-comment-style.md`）。
///
/// # なぜ round-to-odd で二重丸めを避けられるか
///
/// LayerNorm の affine は「`xhat`（`f32`）・`weight`（`f32`）の積 `p = xhat*w`
/// を `f64` へ厳密表現（`p` は仮数 48bit 以内に収まるため [`mul_f64_bits`]
/// は無丸めで厳密値を返す）した後、`bias`（`f32`）を加えて 1 回だけ `f32`
/// へ丸める」という**単一丸めの FMA**（CPU 参照実装 `f32::mul_add`・CUDA
/// の `fmaf` 相当）を再現する必要がある。素朴に「`add_f64_bits` で `f64`
/// （53bit）へ丸めてから [`narrow_f64_bits`] で `f32`（24bit）へ丸める」
/// と**二重丸め**になり、`f32` の正しい単一丸め結果と食い違う実例が
/// 存在する（`xhat=31/16`・`weight=f32::from_bits(0x7f042108)`・
/// `bias=-1` で、真の和は `f32::MAX` と `+inf` の閾値からちょうど整数 `1`
/// だけ下にある 128bit 精度が必要な整数値になり、`f64` への丸めが先に
/// この差を吸収してしまうため `f32::mul_add` は `f32::MAX` を返す一方
/// 素朴な二段階丸めは `+inf` を返す。PR #1671 codex-review 実測）。
///
/// Boldo–Melquiond（2008）の round-to-odd 二重丸め定理: 中間精度 `p2` が
/// 目的精度 `p1` に対し `p2 >= p1 + 2` を満たせば、`RN_p1(RO_p2(x)) ==
/// RN_p1(x)`（すべての実数 `x` に対し）。本ケースは `p2=53`（`f64`）・
/// `p1=24`（`f32`。隠れ 1 込み）で `53 >= 24+2=26` を満たす。round-to-odd
/// は「丸め後の値が厳密値と異なるなら、結果の最下位 bit を強制的に 1
/// （奇数）にする」丸め（通常の最近接偶数丸めとは丸め先の bit パターンが
/// 異なるだけで、桁合わせ・ガード/sticky bit の抽出手順は
/// [`add_f64_bits`] と同一）。これにより「`f32` の丸め判定に影響しうる
/// 情報（厳密値からの正確な距離の符号）」が `f64` の丸めで握り潰されず
/// 保存され、続く [`narrow_f64_bits`]（通常の最近接偶数丸め）が厳密値
/// から直接 `f32` へ丸めた場合と同じ結果を返す。
///
/// # 丸め手順の差分（[`add_f64_bits`] 比）
///
/// 桁合わせ・仮数の加減算・正規化まで完全に同一（本関数はそれらを逐語
/// 複製する）。最終段のみ、「ガード 3bit `r` から `r>4` は切り上げ・
/// `r==4` は偶数丸め」ではなく「`r != 0`（丸め落ちする情報が何かあれば）
/// なら結果の最下位 bit を強制的に 1 にする」へ差し替える。round-to-odd
/// は算術的な繰り上がり（`m += 1`）を一切行わない（ビット単位の OR の
/// み）ため、[`add_f64_bits`] が丸め後に持つ「`m` が `2^53` へ繰り上がる
/// 場合の指数調整」分岐は発生しえず、本関数には存在しない。
pub fn add_f64_bits_round_to_odd(a: u64, b: u64) -> u64 {
    let sa = a & F64_SIGN;
    let sb = b & F64_SIGN;
    let ea = (a >> 52) & F64_EXP_MASK;
    let eb = (b >> 52) & F64_EXP_MASK;
    let fa = a & F64_FRAC_MASK;
    let fb = b & F64_FRAC_MASK;

    // NaN／inf（[`add_f64_bits`] と同一）。
    if ea == F64_EXP_MASK || eb == F64_EXP_MASK {
        let a_nan = ea == F64_EXP_MASK && fa != 0;
        let b_nan = eb == F64_EXP_MASK && fb != 0;
        if a_nan || b_nan {
            return F64_QNAN;
        }
        if ea == F64_EXP_MASK && eb == F64_EXP_MASK {
            return if sa == sb { a } else { F64_QNAN };
        }
        return if ea == F64_EXP_MASK { a } else { b };
    }
    // ゼロ（同一）。
    let a_zero = ea == 0 && fa == 0;
    let b_zero = eb == 0 && fb == 0;
    if a_zero && b_zero {
        return sa & sb;
    }
    if a_zero {
        return b;
    }
    if b_zero {
        return a;
    }

    let (mut ma, ea_eff) = if ea == 0 {
        (fa, 1u64)
    } else {
        (fa | (1u64 << 52), ea)
    };
    let (mut mb, eb_eff) = if eb == 0 {
        (fb, 1u64)
    } else {
        (fb | (1u64 << 52), eb)
    };
    let (mut sa, mut sb) = (sa, sb);
    let (mut ea_eff, mut eb_eff) = (ea_eff, eb_eff);
    if (ea_eff, ma) < (eb_eff, mb) {
        core::mem::swap(&mut ma, &mut mb);
        core::mem::swap(&mut ea_eff, &mut eb_eff);
        core::mem::swap(&mut sa, &mut sb);
    }
    ma <<= 3;
    mb <<= 3;
    let d = ea_eff - eb_eff;
    if d >= 64 {
        mb = u64::from(mb != 0);
    } else if d > 0 {
        let lost = mb & ((1u64 << d) - 1);
        mb = (mb >> d) | u64::from(lost != 0);
    }

    let mut e = ea_eff;
    let mut m;
    if sa == sb {
        m = ma + mb;
        if m >= (1u64 << 56) {
            let lost = m & 1;
            m = (m >> 1) | lost;
            e += 1;
        }
    } else {
        m = ma - mb;
        if m == 0 {
            return 0;
        }
        let mut sh = clz64(m) as u64;
        sh = sh.saturating_sub(8);
        if sh > e - 1 {
            sh = e - 1;
        }
        m <<= sh;
        e -= sh;
    }

    // round-to-odd: 丸め落ちする 3 bit（ガード/丸め/sticky）のいずれかが
    // 立っていれば、結果の最下位 bit を強制的に 1 にする（算術繰り上がり
    // は行わないため `m` が `2^53` へ達することはない）。
    let r = m & 7;
    m >>= 3;
    if r != 0 {
        m |= 1;
    }
    let exp_field = if m >= (1u64 << 52) { e } else { 0 };
    if exp_field >= F64_EXP_MASK {
        return sa | F64_INF;
    }
    sa | (exp_field << 52) | (m & F64_FRAC_MASK)
}

/// `f64 as f32`（最近接偶数丸め・overflow は `±inf`・underflow は f32
/// subnormal／`±0`）の bit 表現版（NaN は quiet NaN へ正規化）。
pub fn narrow_f64_bits(bits: u64) -> u32 {
    let sign = ((bits >> 63) as u32) << 31;
    let e = (bits >> 52) & F64_EXP_MASK;
    let f = bits & F64_FRAC_MASK;
    if e == F64_EXP_MASK {
        return if f != 0 { F32_QNAN } else { sign | F32_INF };
    }
    if e == 0 && f == 0 {
        return sign;
    }
    // 実効値 `m × 2^(ee - 52)`（`m` は隠れ 1 込みの 53 bit。subnormal は
    // 隠れ 1 なし・実効指数 -1022）。
    let (m, ee) = if e == 0 {
        (f, -1022i64)
    } else {
        (f | (1u64 << 52), e as i64 - 1023)
    };
    // f32 の指数フィールド候補。`ee > 127` は丸め前から overflow。
    let ef = ee + 127;
    if ef >= 255 {
        return sign | F32_INF;
    }
    // 仮数を 24 bit（隠れ 1 が bit 23）へ落とすシフト量 29 に、subnormal 化
    // （`ef <= 0`）で追加のシフト `1 - ef` を足す。`m` は 53 bit 幅なので、
    // シフト量が 54 以上なら半分未満（丸め bit が 0）で必ず `±0` へ落ちる。
    let extra = if ef <= 0 { 1 - ef } else { 0 };
    let shift = 29 + extra;
    if shift >= 54 {
        return sign;
    }
    let shift = shift as u32;
    let q0 = m >> shift;
    let rem = m & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let mut q = q0;
    if rem > half || (rem == half && (q0 & 1) == 1) {
        q += 1;
    }
    let mut exp_field = if ef <= 0 { 0u32 } else { ef as u32 };
    if ef <= 0 {
        // subnormal: `q ∈ [0, 2^23]`。`2^23` へ繰り上がれば最小正規化数。
        if q >= (1u64 << 23) {
            exp_field = 1;
            q -= 1u64 << 23;
        }
    } else {
        // 正規化数: `q ∈ [2^23, 2^24]`。`2^24` へ繰り上がれば指数 +1。
        if q >= (1u64 << 24) {
            q >>= 1;
            exp_field += 1;
        }
        q -= 1u64 << 23;
    }
    if exp_field >= 255 {
        return sign | F32_INF;
    }
    sign | (exp_field << 23) | (q as u32)
}

/// `f64` の符号反転（`-x`）の bit 表現版。NaN も含め符号 bit を無条件に
/// 反転する（IEEE754 の `negate` 演算・Rust の単項 `-` と同一）。
#[inline]
pub fn neg_f64_bits(a: u64) -> u64 {
    a ^ F64_SIGN
}

/// `a - b` の bit 表現版（[`add_f64_bits`]`(a, `[`neg_f64_bits`]`(b))`。
#[inline]
pub fn sub_f64_bits(a: u64, b: u64) -> u64 {
    add_f64_bits(a, neg_f64_bits(b))
}

/// `u64 x u64` の厳密な 128bit 積を `(hi, lo)`（`value = hi*2^64+lo`）で
/// 返す（32bit 分割のスクールブック乗算。[`mul_f64_bits`] の仮数積算出に
/// 使う。MSL 側 `ln_f64_mul64_wide` の逐語移植元——MSL は `ulong` のみで
/// 同じ構造を再現するため、Rust 側もここでは `u128` を使わず同じ
/// アルゴリズムを採る）。
#[inline]
fn mul64_wide(a: u64, b: u64) -> (u64, u64) {
    let a_lo = a & 0xFFFF_FFFF;
    let a_hi = a >> 32;
    let b_lo = b & 0xFFFF_FFFF;
    let b_hi = b >> 32;

    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_hi = a_hi * b_hi;

    let mid = (lo_lo >> 32) + (hi_lo & 0xFFFF_FFFF) + (lo_hi & 0xFFFF_FFFF);
    let lo = (lo_lo & 0xFFFF_FFFF) | (mid << 32);
    let hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
    (hi, lo)
}

/// `(hi,lo)`（128bit 値。両方 0 なら `None`）の先頭 1 の bit 位置
/// （0-indexed・LSB 起点）。
#[inline]
fn leading_bit_pos128(hi: u64, lo: u64) -> Option<u32> {
    if hi != 0 {
        Some(64 + (63 - hi.leading_zeros()))
    } else if lo != 0 {
        Some(63 - lo.leading_zeros())
    } else {
        None
    }
}

/// `(hi,lo)` の下位 `n` bit（`n <= 128`）を `(hi,lo)` 形式のまま取り出す。
#[inline]
fn low_bits128(hi: u64, lo: u64, n: u32) -> (u64, u64) {
    if n == 0 {
        (0, 0)
    } else if n >= 128 {
        (hi, lo)
    } else if n <= 64 {
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        (0, lo & mask)
    } else {
        let n2 = n - 64;
        let mask = if n2 == 64 { u64::MAX } else { (1u64 << n2) - 1 };
        (hi & mask, lo)
    }
}

/// `(hi,lo)` を右シフト `s`（`0..=128`）した値（sticky は持たない・
/// 単純な切り捨てシフト。丸め判定は呼び出し側が [`low_bits128`] の
/// 余りと別に行う）。
#[inline]
fn shr128(hi: u64, lo: u64, s: u32) -> (u64, u64) {
    if s == 0 {
        (hi, lo)
    } else if s >= 128 {
        (0, 0)
    } else if s < 64 {
        ((hi >> s), (lo >> s) | (hi << (64 - s)))
    } else if s == 64 {
        (0, hi)
    } else {
        (0, hi >> (s - 64))
    }
}

/// `(hi,lo)` の辞書式（＝数値としての）比較。
#[inline]
fn cmp128(a: (u64, u64), b: (u64, u64)) -> core::cmp::Ordering {
    a.cmp(&b)
}

/// `f64` の指数・仮数フィールド（`e`：バイアス済み・`f`：フラクション。
/// `e==0 && f==0` の完全ゼロは呼び出し側で排除済みの前提）から
/// 「隠れ 1 を bit52 に立てた 53bit 仮数 `m`」と、`value = m * 2^(exp_u
/// - 52)` を満たす unbiased 指数 `exp_u` を求める（正規化数はそのまま、
/// subnormal は仮数の先頭 1 の位置から逆算してシフトする）。
/// [`mul_f64_bits`] 専用ヘルパー。
#[inline]
fn normalize_f64_mantissa(e: u64, f: u64) -> (u64, i64) {
    if e == 0 {
        // subnormal（`f != 0` が呼び出し前提）: 先頭 1 の位置 `lead`
        // （0..=51）から bit52 へ寄せるシフト量を求める。
        let lead = 63 - f.leading_zeros() as i64;
        let shift = 52 - lead;
        let m = f << shift;
        let exp_u = -1022 - shift;
        (m, exp_u)
    } else {
        (f | (1u64 << 52), e as i64 - 1023)
    }
}

/// `f64 * f64`（最近接偶数丸め）の bit 表現版（NaN は quiet NaN へ
/// 正規化）。仮数同士の厳密 106bit 積を `mul64_wide` で構成し、
/// **1 回だけ**丸める（中間で `f32` はもとより暫定 `f64` へも丸めない。
/// 二重丸め回避——特に underflow して subnormal 化する経路で、正規化数
/// として一度丸めてから再度 subnormal へシフトし直すと誤った丸め結果に
/// なりうるため、最終シフト量を先に確定してから 1 回で丸める）。
pub fn mul_f64_bits(a: u64, b: u64) -> u64 {
    let sa = a & F64_SIGN;
    let sb = b & F64_SIGN;
    let sign = sa ^ sb;
    let ea = (a >> 52) & F64_EXP_MASK;
    let eb = (b >> 52) & F64_EXP_MASK;
    let fa = a & F64_FRAC_MASK;
    let fb = b & F64_FRAC_MASK;

    let a_nan = ea == F64_EXP_MASK && fa != 0;
    let b_nan = eb == F64_EXP_MASK && fb != 0;
    if a_nan || b_nan {
        return F64_QNAN;
    }
    let a_inf = ea == F64_EXP_MASK; // fa==0 はここまでに NaN 判定済み。
    let b_inf = eb == F64_EXP_MASK;
    let a_zero = ea == 0 && fa == 0;
    let b_zero = eb == 0 && fb == 0;
    // `0 * inf` は無効演算（NaN）。一般の inf／zero 判定より先に見る。
    if (a_zero && b_inf) || (a_inf && b_zero) {
        return F64_QNAN;
    }
    if a_inf || b_inf {
        return sign | F64_INF;
    }
    if a_zero || b_zero {
        return sign;
    }

    // 仮数を「隠れ 1 を bit52 に立てた 53bit 値」へ正規化する（subnormal
    // 入力も含め常に `m ∈ [2^52, 2^53)`。ゆえに積 `P = ma*mb` は常に
    // `[2^104, 2^106)`）。
    let (ma, ea_u) = normalize_f64_mantissa(ea, fa);
    let (mb, eb_u) = normalize_f64_mantissa(eb, fb);
    let (hi, lo) = mul64_wide(ma, mb);
    // `ma`／`mb` はいずれも `normalize_f64_mantissa` により `[2^52, 2^53)`
    // へ正規化済み（呼び出し前提: `a_zero`／`b_zero` は上で早期 return 済み
    // で、ここに到達する時点で両者非ゼロ）なので積 `hi:lo` が完全ゼロに
    // なることは数学的にありえない。`.expect()`（本番経路 panic）は
    // `.claude/rules/coding-rust.md` の「本番経路で unwrap/expect を
    // 使わない」規約に抵触するため、`debug_assert!` でこの不変条件を
    // テスト・デバッグビルドで検査したうえで、万一（バグにより）破れても
    // release ビルドではフォールバック値 104（`P ∈ [2^104, 2^106)` の下限
    // 指数）を採って安全側で処理を継続する（panic しない）。
    debug_assert!(
        hi != 0 || lo != 0,
        "非ゼロ仮数同士の積は非ゼロ（呼び出し前提より）"
    );
    let leadpos = leading_bit_pos128(hi, lo).unwrap_or(104);
    // `exp_u`: 丸め前の暫定 unbiased 指数（`value = m0 * 2^(exp_u-52)`、
    // `m0` は `leadpos-52` bit 右シフトした 53bit 候補仮数）。
    let exp_u = ea_u + eb_u + (leadpos as i64 - 104);
    let shift_normal = leadpos as i64 - 52; // 52 か 53（常に < 64）。

    let biased_before_round = exp_u + 1023;
    let (final_shift, is_subnormal_target) = if biased_before_round >= 1 {
        (shift_normal, false)
    } else {
        // 単一丸めで subnormal 化するため、追加シフトを事前に織り込む。
        (shift_normal + (1 - biased_before_round), true)
    };

    if !(0..128).contains(&final_shift) {
        // 到達性: `f32` 由来の `x`／`weight` から生じる本カーネルの実用
        // 値域では起きない極端な underflow（`final_shift` が 128 以上に
        // なるのは指数が f64 の subnormal 下限をはるかに超えて離れる
        // 場合のみ）。安全側として `±0` へ丸める（sticky 相当の判定を
        // 省略しても、実際に到達しうる限り常に半分未満で `±0` が正しい）。
        return sign;
    }
    let final_shift = final_shift as u32;
    let (qhi, mut m) = shr128(hi, lo, final_shift);
    debug_assert_eq!(qhi, 0, "mul_f64_bits: 商が 64bit を超えた（想定外）");
    let (rhi, rlo) = low_bits128(hi, lo, final_shift);
    let half = if final_shift == 0 {
        (0u64, 0u64)
    } else if final_shift - 1 < 64 {
        (0u64, 1u64 << (final_shift - 1))
    } else {
        (1u64 << (final_shift - 1 - 64), 0u64)
    };
    let cmp = cmp128((rhi, rlo), half);
    let round_up =
        cmp == core::cmp::Ordering::Greater || (cmp == core::cmp::Ordering::Equal && (m & 1) == 1);
    if round_up {
        m += 1;
    }

    if !is_subnormal_target {
        let mut exp_final = exp_u;
        if m >= (1u64 << 53) {
            m >>= 1;
            exp_final += 1;
        }
        let biased_final = exp_final + 1023;
        if biased_final >= (F64_EXP_MASK as i64) {
            return sign | F64_INF;
        }
        sign | ((biased_final as u64) << 52) | (m & F64_FRAC_MASK)
    } else {
        // `m` は `[0, 2^52]`（丸め上げで最大 `2^52` に達しうる＝最小
        // 正規化数への繰り上がり）。
        if m >= (1u64 << 52) {
            sign | (1u64 << 52) // biased_exp=1, frac=0
        } else {
            sign | m // exponent field 0（subnormal）
        }
    }
}

/// `(hi,lo)`（128bit・呼び出し前提: 全体の値が `d * 2^64` 未満、すなわち
/// 商が `u64` に収まる）を 64bit の非ゼロ除数 `d` で割る筆算除算
/// （2 進 shift-subtract 方式。128 回のシフトで 1 bit ずつ商を確定する。
/// 剰余 `rem` は各ステップで `d` 未満に保たれる〈標準的な復元法除算の
/// 不変条件〉ため `d` が `u64` に収まる限り `u64` の加算・シフトで
/// overflow しない）。[`div_f64_bits`] 専用ヘルパー。
#[inline]
fn div64_wide(hi: u64, lo: u64, d: u64) -> (u64, u64) {
    debug_assert!(d != 0, "0 除算は呼び出し側で除外済みの前提");
    let mut rem: u64 = 0;
    let mut quotient: u64 = 0;
    for i in (0..128).rev() {
        let bit = if i >= 64 {
            (hi >> (i - 64)) & 1
        } else {
            (lo >> i) & 1
        };
        rem = (rem << 1) | bit;
        if rem >= d {
            rem -= d;
            quotient = (quotient << 1) | 1;
        } else {
            quotient <<= 1;
        }
    }
    (quotient, rem)
}

/// `f64 / f64`（最近接偶数丸め）の bit 表現版（NaN は quiet NaN へ
/// 正規化）。[`mul_f64_bits`] と対になる soft-f64 演算——`layer_norm`
/// の `mean`／`var` を `sum * recip(hidden)`（Newton 近似逆数との積）
/// ではなく本関数の**正しく丸めた除算**で求めることで、一様行
/// （例 `x=[1e30f32;49]`）のような「割り切れる」ケースで Newton
/// 近似特有の 1 ULP 誤差が悪化するのを防ぐ（PR #1671 codex-review・
/// Cursor Bugbot 指摘。イシュー #1596）。
///
/// アルゴリズム: 仮数 `ma`／`mb` を `[2^52,2^53)` へ正規化し、
/// `numerator = ma << 55`（`mul64_wide` で厳密に 128bit 構成）を
/// `div64_wide` で `mb` 除算する。`ma/mb ∈ (0.5,2)` のため商は
/// `[2^54,2^56)` に収まり、53bit 仮数 + guard/round bit（丸め判定用の
/// 余剰 2bit）の精度を確保する。`final_shift ∈ {2,3}` は必ず `>= 2`
/// になるため、剰余が非ゼロなら sticky を商の最下位ビットへ OR
/// しても厳密な「半分ちょうど」判定（`half` の bit0 は常に 0）を
/// 壊さない（`half = 1 << (final_shift-1)` の最下位ビットは
/// `final_shift >= 2` のとき常に 0）。丸め処理自体は [`mul_f64_bits`]
/// の丸めテール（`(hi,lo)` を `(0,q)` として再利用）と同型。
pub fn div_f64_bits(a: u64, b: u64) -> u64 {
    let sa = a & F64_SIGN;
    let sb = b & F64_SIGN;
    let sign = sa ^ sb;
    let ea = (a >> 52) & F64_EXP_MASK;
    let eb = (b >> 52) & F64_EXP_MASK;
    let fa = a & F64_FRAC_MASK;
    let fb = b & F64_FRAC_MASK;

    let a_nan = ea == F64_EXP_MASK && fa != 0;
    let b_nan = eb == F64_EXP_MASK && fb != 0;
    if a_nan || b_nan {
        return F64_QNAN;
    }
    let a_inf = ea == F64_EXP_MASK;
    let b_inf = eb == F64_EXP_MASK;
    let a_zero = ea == 0 && fa == 0;
    let b_zero = eb == 0 && fb == 0;
    // `inf/inf`・`0/0` は無効演算（NaN）。
    if (a_inf && b_inf) || (a_zero && b_zero) {
        return F64_QNAN;
    }
    if a_inf {
        return sign | F64_INF; // 有限 `/` 有限inf 以外（0/0 は上で除外済み）。
    }
    if b_inf {
        return sign; // 有限 / inf = ±0（a_inf は上で除外済み）。
    }
    if b_zero {
        return sign | F64_INF; // 非ゼロ有限 / 0 = ±inf（a_zero は上で除外不要: a も 0 なら 0/0 で上に該当済み）。
    }
    if a_zero {
        return sign; // 0 / 非ゼロ有限 = ±0。
    }

    // 仮数を「隠れ 1 を bit52 に立てた 53bit 値」へ正規化する
    // （`mul_f64_bits` と同じ前処理）。
    let (ma, ea_u) = normalize_f64_mantissa(ea, fa);
    let (mb, eb_u) = normalize_f64_mantissa(eb, fb);
    // `S = 55`: `ma/mb ∈ (0.5,2)` のため商は `[2^54,2^56)` に収まり、
    // 53bit 仮数 + 2bit（guard/round）の精度が確保できる最小の追加
    // シフト量（`final_shift` を必ず `>= 2` にして下記の sticky OR が
    // 安全になる）。
    const S: u32 = 55;
    let (num_hi, num_lo) = mul64_wide(ma, 1u64 << S);
    let (raw_q, rem) = div64_wide(num_hi, num_lo, mb);
    // `raw_q` に剰余の非ゼロ性を最下位ビットへ折り込む（sticky bit）。
    // `final_shift >= 2` により `half` の bit0 が常に 0 のため、この
    // OR は「厳密な半分」と「半分よりわずかに大きい」の判別を保つ。
    let q = raw_q | (rem != 0) as u64;
    debug_assert!(q != 0, "非ゼロ被除数・除数の商は非ゼロ（呼び出し前提より）");
    let leadpos = leading_bit_pos128(0, q).unwrap_or(S - 1);
    // `numerator = ma * 2^S`・`value = numerator/mb * 2^(ea_u-eb_u-S)`
    // なので `raw_q` 自体の指数寄与は `leadpos - S`（`mul_f64_bits` の
    // `-104` に相当する定数を `S` に置き換えたもの）。
    let exp_u = ea_u - eb_u + (leadpos as i64 - S as i64);
    let shift_normal = leadpos as i64 - 52; // 2 か 3（`leadpos` が 54 か 55 のため）。

    let biased_before_round = exp_u + 1023;
    let (final_shift, is_subnormal_target) = if biased_before_round >= 1 {
        (shift_normal, false)
    } else {
        (shift_normal + (1 - biased_before_round), true)
    };

    if !(0..64).contains(&final_shift) {
        // 到達性: 本カーネルの実用値域（`f32` 由来の `sum`／`hidden`）
        // では発生しない極端な underflow。安全側として `±0` へ丸める。
        return sign;
    }
    let final_shift = final_shift as u32;
    // `q` は `u64` 1 語に収まる（`mul_f64_bits` と異なり (hi,lo)=(0,q)
    // として同型の丸めテールを適用する）。
    let m_shifted = if final_shift == 0 {
        q
    } else {
        q >> final_shift
    };
    let low_mask = if final_shift >= 64 {
        u64::MAX
    } else {
        (1u64 << final_shift) - 1
    };
    let rem_low = if final_shift == 0 { 0 } else { q & low_mask };
    let mut m = m_shifted;
    let half = if final_shift == 0 {
        0
    } else {
        1u64 << (final_shift - 1)
    };
    let round_up = rem_low > half || (rem_low == half && (m & 1) == 1);
    if round_up {
        m += 1;
    }

    if !is_subnormal_target {
        let mut exp_final = exp_u;
        if m >= (1u64 << 53) {
            m >>= 1;
            exp_final += 1;
        }
        let biased_final = exp_final + 1023;
        if biased_final >= (F64_EXP_MASK as i64) {
            return sign | F64_INF;
        }
        sign | ((biased_final as u64) << 52) | (m & F64_FRAC_MASK)
    } else {
        if m >= (1u64 << 52) {
            sign | (1u64 << 52)
        } else {
            sign | m
        }
    }
}

/// `x`（正規化数・subnormal 双方に対応。特殊値は呼び出し側で除外済みの
/// 前提）を `2^k` 倍する（`k` は符号付き整数）。指数フィールドを直接
/// 加算するだけの**厳密**演算（仮数は変えない）。結果が正規化数の指数
/// 範囲を超える場合は `±inf`（overflow）・`±0`（極端な underflow）へ
/// 丸めなしで潰す——本関数は Newton 反復の「種」（近似値）専用であり、
/// 種が多少劣化しても反復で収束するため、真の 2 の冪乗算のような
/// 厳密な subnormal 対応までは持たない（`mul_f64_bits` が本演算の
/// 汎用・正確版に相当する）。
#[inline]
fn scale_pow2_f64_bits(x_bits: u64, k: i64) -> u64 {
    let sign = x_bits & F64_SIGN;
    let e = ((x_bits >> 52) & F64_EXP_MASK) as i64;
    let f = x_bits & F64_FRAC_MASK;
    let new_e = e + k;
    if new_e >= F64_EXP_MASK as i64 {
        return sign | F64_INF;
    }
    if new_e <= 0 {
        return sign;
    }
    sign | ((new_e as u64) << 52) | f
}

/// [`recip_newton_f64_bits`]／[`rsqrt_newton_f64_bits`] 共通の種抽出:
/// `x`（正・有限・非ゼロが呼び出し前提）の指数部と仮数部を分離し、
/// 仮数側のみ `f32` 精度（ハードウェア除算・`sqrt`）で近似した後、
/// 指数側は [`scale_pow2_f64_bits`] で厳密に合成し直す。`narrow_f64_bits`
/// を `x` 全体へ直接適用しないため、`x` が `f32` の表現範囲外
/// （`|x| < 2^-149` や `|x| > f32::MAX` 相当）でも種が potentially 0 や
/// `inf` へ潰れず Newton 反復が退化しない（`layer_norm` の分散は
/// `f32` 由来要素の二乗和を `hidden` で割った値のため、理論上は `f32`
/// の表現範囲を超えて underflow/overflow しうる）。
/// `reduce_shift`（`recip` は `0`、`rsqrt` は仮数を `[1,4)` へ正規化する
/// ため指数の偶奇に応じて `0`／`1`）だけ挙動を切り替える。
#[inline]
fn extract_reduced_mantissa_and_exp(x_bits: u64, want_sqrt_range: bool) -> (u32, i64) {
    let e = (x_bits >> 52) & F64_EXP_MASK;
    let f = x_bits & F64_FRAC_MASK;
    let (m, exp_u) = normalize_f64_mantissa(e, f); // value = m * 2^(exp_u-52), m∈[2^52,2^53)
    if !want_sqrt_range {
        // reduced = value / 2^exp_u ∈ [1,2)。
        let reduced_bits = (1023u64 << 52) | (m & F64_FRAC_MASK);
        (narrow_f64_bits(reduced_bits), exp_u)
    } else if exp_u.rem_euclid(2) == 0 {
        // reduced = value / 2^exp_u ∈ [1,2)、k = exp_u/2。
        let reduced_bits = (1023u64 << 52) | (m & F64_FRAC_MASK);
        (narrow_f64_bits(reduced_bits), exp_u / 2)
    } else {
        // reduced = value / 2^(exp_u-1) ∈ [2,4)、k = (exp_u-1)/2
        // （`exp_u-1` は `exp_u` が奇数のとき常に偶数＝厳密に割り切れる）。
        let reduced_bits = (1024u64 << 52) | (m & F64_FRAC_MASK);
        (narrow_f64_bits(reduced_bits), (exp_u - 1) / 2)
    }
}

/// `f64` の逆数 `1/x` を Newton-Raphson（`y_{n+1} = y_n*(2 - x*y_n)`。
/// 除算命令を使わず [`mul_f64_bits`]／[`add_f64_bits`] のみで構成）で
/// 求める bit 表現版。`x` は本カーネルの用途上（`hidden` 由来）常に有限・
/// 正の値のため特殊値分岐は持たない（一般用途には非対応）。種は仮数を
/// `f32` 精度（ハードウェア除算 1 回。約 24bit 精度）で近似し指数を
/// 厳密合成した近似逆数（`extract_reduced_mantissa_and_exp` 参照）とし、
/// 4 回の反復で各段階精度がほぼ倍加し `f64` の 52bit 精度に収束する
/// （24→48→52…）。`shaders/layer_norm.metal` の `ln_f64_recip`（実体は
/// 文字列リソースのためモジュールパスではなくファイル参照）と同じ
/// 反復回数・アルゴリズム。
pub fn recip_newton_f64_bits(x: u64) -> u64 {
    let (reduced_bits, exp_u) = extract_reduced_mantissa_and_exp(x, false);
    let seed_reduced = 1.0f32 / f32::from_bits(reduced_bits);
    let mut y = scale_pow2_f64_bits(widen_f32_bits(seed_reduced.to_bits()), -exp_u);
    const TWO: u64 = 0x4000_0000_0000_0000;
    for _ in 0..4 {
        let xy = mul_f64_bits(x, y);
        let two_minus_xy = sub_f64_bits(TWO, xy);
        y = mul_f64_bits(y, two_minus_xy);
    }
    y
}

/// `f64` の逆数平方根 `1/sqrt(x)` を Newton-Raphson（`y_{n+1} =
/// y_n*(1.5 - 0.5*x*y_n^2)`。除算命令不要）で求める bit 表現版。
/// `x` の特殊値は明示的に扱う（`x` は分散 `+ eps`〈ともに非負〉に由来し
/// 数学的に非負のはずだが、防御的に負値も NaN として扱う）:
/// `NaN -> NaN`・`±0 -> ±inf`（符号付きゼロの逆数平方根の IEEE754 規約）・
/// 負（非ゼロ）`-> NaN`・`+inf -> +0`。これらは Newton 反復内で `0*inf`
/// （不定形・NaN で汚染される）を発生させないための事前分岐であり、
/// 反復本体には入らない。
pub fn rsqrt_newton_f64_bits(x: u64) -> u64 {
    let e = (x >> 52) & F64_EXP_MASK;
    let f = x & F64_FRAC_MASK;
    let sign = x & F64_SIGN;
    if e == F64_EXP_MASK && f != 0 {
        return F64_QNAN;
    }
    if e == 0 && f == 0 {
        return sign | F64_INF;
    }
    if sign != 0 {
        return F64_QNAN;
    }
    if e == F64_EXP_MASK {
        return 0; // +inf -> +0（f==0・sign==0 はここまでに確定）。
    }

    let (reduced_bits, k) = extract_reduced_mantissa_and_exp(x, true);
    let seed_reduced = 1.0f32 / f32::from_bits(reduced_bits).sqrt();
    let mut y = scale_pow2_f64_bits(widen_f32_bits(seed_reduced.to_bits()), -k);
    const ONE_HALF: u64 = 0x3FE0_0000_0000_0000;
    const THREE_HALF: u64 = 0x3FF8_0000_0000_0000;
    for _ in 0..4 {
        let y2 = mul_f64_bits(y, y);
        let xy2 = mul_f64_bits(x, y2);
        let half_xy2 = mul_f64_bits(ONE_HALF, xy2);
        let inner = sub_f64_bits(THREE_HALF, half_xy2);
        y = mul_f64_bits(y, inner);
    }
    y
}

/// `gemm_bias_grad_reduce_f32` の `m >= 2` 経路の逐語モデル: `+0.0`（`f64`）
/// から始めて `xs` を index 順に [`add_f64_bits`] で蓄積し、最後に 1 回
/// [`narrow_f64_bits`] で `f32` へ落とす。`autodiff::eval::reduce_bias_grad_
/// rows` の `acc: f64 = 0.0; acc += f64::from(x); acc as f32` と NaN を除き
/// bit 一致する。
pub fn sequential_sum_f32_bits<I: IntoIterator<Item = u32>>(xs: I) -> u32 {
    let mut acc = 0u64;
    for x in xs {
        acc = add_f64_bits(acc, widen_f32_bits(x));
    }
    narrow_f64_bits(acc)
}

/// [`sequential_sum_f32_bits`] の `f32` 引数版。
pub fn sequential_sum_f32(xs: &[f32]) -> f32 {
    f32::from_bits(sequential_sum_f32_bits(xs.iter().map(|x| x.to_bits())))
}

/// `a*b+c`（`f32`）を**単一丸めの FMA**（`f32::mul_add`・CUDA `fmaf` 相当）
/// として計算する `layer_norm.metal` パス 3（affine）のホスト側逐語モデル
/// （PR #1671 codex-review 指摘・イシュー #1596）。`ln_f64_widen`・
/// `ln_f64_mul`・`ln_f64_add_ro`・`ln_f64_narrow`（本モジュールの
/// [`widen_f32_bits`]・[`mul_f64_bits`]・[`add_f64_bits_round_to_odd`]・
/// [`narrow_f64_bits`]）と 1 対 1 対応する。
///
/// GPU ハードウェアの平坦な `float` 版 `fma()` を使わない理由（subnormal
/// 入力の flush-to-zero）は `shaders/layer_norm.metal` 冒頭コメント「FMA
/// 契約」を参照。[`add_f64_bits_round_to_odd`] の doc comment に二重丸め
/// 回避の数学的根拠（round-to-odd 二重丸め定理）を記載する。
pub fn fma_f32_bits(a_bits: u32, b_bits: u32, c_bits: u32) -> u32 {
    let product = mul_f64_bits(widen_f32_bits(a_bits), widen_f32_bits(b_bits));
    let sum = add_f64_bits_round_to_odd(product, widen_f32_bits(c_bits));
    narrow_f64_bits(sum)
}

/// [`fma_f32_bits`] の `f32` 引数版。
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    f32::from_bits(fma_f32_bits(a.to_bits(), b.to_bits(), c.to_bits()))
}

/// NaN をクラス一致・それ以外を bit 一致で比較する（NaN payload は
/// ハードウェア依存のため）。
pub fn f32_bits_match(actual: f32, expected: f32) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else {
        actual.to_bits() == expected.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// 指数を全域から一様に引いた f64 bit（NaN／inf を含む）。
        fn f64_bits(&mut self) -> u64 {
            let r = self.next();
            match r % 16 {
                0 => 0,
                1 => F64_SIGN,
                2 => r & (F64_FRAC_MASK | F64_SIGN), // subnormal
                3 => (r & F64_SIGN) | F64_INF,
                4 => self.next() | 0x7FF0_0000_0000_0001, // NaN
                _ => self.next(),
            }
        }
        fn f32_bits(&mut self) -> u32 {
            let r = self.next() as u32;
            match r % 16 {
                0 => 0,
                1 => 0x8000_0000,
                2 => r & 0x807F_FFFF, // subnormal
                3 => (r & 0x8000_0000) | F32_INF,
                4 => (self.next() as u32) | 0x7F80_0001, // NaN
                _ => self.next() as u32,
            }
        }
    }

    fn assert_f64_eq(actual: u64, expected: f64, ctx: &str) {
        if expected.is_nan() {
            assert!(
                f64::from_bits(actual).is_nan(),
                "{ctx}: NaN 期待に対し {actual:#x}"
            );
        } else {
            assert_eq!(
                actual,
                expected.to_bits(),
                "{ctx}: actual={:e} expected={:e}",
                f64::from_bits(actual),
                expected
            );
        }
    }

    fn assert_f32_eq(actual: u32, expected: f32, ctx: &str) {
        if expected.is_nan() {
            assert!(
                f32::from_bits(actual).is_nan(),
                "{ctx}: NaN 期待に対し {actual:#x}"
            );
        } else {
            assert_eq!(
                actual,
                expected.to_bits(),
                "{ctx}: actual={:e} expected={:e}",
                f32::from_bits(actual),
                expected
            );
        }
    }

    #[test]
    fn widen_matches_hardware_for_random_and_special_inputs() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for i in 0..2_000_000u32 {
            let b = rng.f32_bits();
            assert_f64_eq(
                widen_f32_bits(b),
                f64::from(f32::from_bits(b)),
                &format!("widen #{i} {b:#x}"),
            );
        }
        // 全 subnormal f32（2^23 個）を網羅。
        for b in 1u32..(1u32 << 23) {
            assert_f64_eq(
                widen_f32_bits(b),
                f64::from(f32::from_bits(b)),
                "widen subnormal",
            );
            let nb = b | 0x8000_0000;
            assert_f64_eq(
                widen_f32_bits(nb),
                f64::from(f32::from_bits(nb)),
                "widen -subnormal",
            );
        }
    }

    #[test]
    fn add_matches_hardware_for_random_pairs() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for i in 0..4_000_000u32 {
            let a = rng.f64_bits();
            let b = rng.f64_bits();
            let expected = f64::from_bits(a) + f64::from_bits(b);
            assert_f64_eq(
                add_f64_bits(a, b),
                expected,
                &format!("add #{i} {a:#x} + {b:#x}"),
            );
        }
    }

    #[test]
    fn add_matches_hardware_for_targeted_exponent_gaps() {
        // 指数差 0..=70・符号の組合せ・仮数の端（0・全 1・半端）を総当たり。
        let fracs = [
            0u64,
            1,
            F64_FRAC_MASK,
            F64_FRAC_MASK - 1,
            1u64 << 51,
            (1u64 << 51) + 1,
            0x0555_5555_5555_5555,
        ];
        for gap in 0..=70u64 {
            for &fa in &fracs {
                for &fb in &fracs {
                    for sa in [0u64, F64_SIGN] {
                        for sb in [0u64, F64_SIGN] {
                            for ea in [1u64, 2, 50, 1023, 1023 + 60, 2045, 2046] {
                                if gap > ea - 1 {
                                    continue;
                                }
                                let a = sa | (ea << 52) | fa;
                                let b = sb | ((ea - gap) << 52) | fb;
                                let ctx = format!("gap={gap} a={a:#x} b={b:#x}");
                                assert_f64_eq(
                                    add_f64_bits(a, b),
                                    f64::from_bits(a) + f64::from_bits(b),
                                    &ctx,
                                );
                                assert_f64_eq(
                                    add_f64_bits(b, a),
                                    f64::from_bits(b) + f64::from_bits(a),
                                    &ctx,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn add_handles_subnormal_and_boundary_cases() {
        let cases: [(f64, f64); 12] = [
            (f64::MIN_POSITIVE, -f64::from_bits(1)),
            (f64::from_bits(1), f64::from_bits(1)),
            (f64::from_bits(F64_FRAC_MASK), f64::from_bits(1)),
            (f64::MAX, f64::MAX),
            (f64::MAX, f64::from_bits(0x7FE0_0000_0000_0000)),
            (-f64::MAX, -f64::MAX),
            (1.0, -1.0),
            (0.0, -0.0),
            (-0.0, -0.0),
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::INFINITY, -f64::MAX),
            (f64::NAN, 1.0),
        ];
        for (a, b) in cases {
            assert_f64_eq(
                add_f64_bits(a.to_bits(), b.to_bits()),
                a + b,
                &format!("{a:e} + {b:e}"),
            );
            assert_f64_eq(
                add_f64_bits(b.to_bits(), a.to_bits()),
                b + a,
                &format!("{b:e} + {a:e}"),
            );
        }
    }

    #[test]
    fn narrow_matches_hardware_for_random_and_boundary_inputs() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for i in 0..4_000_000u32 {
            let b = rng.f64_bits();
            assert_f32_eq(
                narrow_f64_bits(b),
                f64::from_bits(b) as f32,
                &format!("narrow #{i} {b:#x}"),
            );
        }
        // f32 の指数範囲境界（overflow・subnormal・underflow）を、f64 指数
        // -160..=130 の全域 × 仮数パターンで総当たり。
        let fracs = [
            0u64,
            1,
            F64_FRAC_MASK,
            1u64 << 28, // f32 への丸め位置ちょうど半分
            (1u64 << 28) - 1,
            (1u64 << 28) + 1,
            (1u64 << 29) | (1u64 << 28), // 偶数丸めで繰り上がる
            0x000F_FFFF_FFF0_0000,
        ];
        for ee in -160i64..=130 {
            for &f in &fracs {
                for sign in [0u64, F64_SIGN] {
                    let b = sign | (((ee + 1023) as u64) << 52) | f;
                    assert_f32_eq(
                        narrow_f64_bits(b),
                        f64::from_bits(b) as f32,
                        &format!("narrow ee={ee} f={f:#x}"),
                    );
                }
            }
        }
        // f32 subnormal 域での偶数丸め: 各 subnormal f32 の中点を f64 で作る。
        for k in 0u64..2000 {
            let lo = f32::from_bits(k as u32);
            let hi = f32::from_bits(k as u32 + 1);
            let mid = (f64::from(lo) + f64::from(hi)) / 2.0;
            assert_f32_eq(
                narrow_f64_bits(mid.to_bits()),
                mid as f32,
                &format!("subnormal midpoint k={k}"),
            );
            let below = f64::from_bits(mid.to_bits() - 1);
            assert_f32_eq(
                narrow_f64_bits(below.to_bits()),
                below as f32,
                "below midpoint",
            );
        }
    }

    #[test]
    fn sequential_sum_matches_host_f64_reduction() {
        fn host(xs: &[f32]) -> f32 {
            let mut acc = 0f64;
            for &x in xs {
                acc += f64::from(x);
            }
            acc as f32
        }
        // codex-review が挙げた相殺・overflow・subnormal の名指しケース。
        let p = |e: i32| 2f32.powi(e);
        let named: Vec<Vec<f32>> = vec![
            vec![p(48), p(24), 1.0, -p(48), -p(24)],
            vec![p(50), p(25), 1.0, -p(50), -p(25)],
            vec![1e8, 1.0, -99999992.0],
            vec![100000000.0, -100000008.0, 8.0],
            vec![f32::from_bits(1), 1.0],
            vec![f32::MAX, f32::MAX, -f32::MAX, -f32::MAX],
            vec![f32::MAX, f32::MAX],
            vec![f32::MAX, f32::MAX, -f32::MAX],
            vec![f32::INFINITY, f32::NEG_INFINITY],
            vec![f32::INFINITY, -f32::MAX, 1.0],
            vec![-0.0, -0.0],
            vec![0.0, -0.0],
            vec![f32::from_bits(1), f32::from_bits(1), -f32::from_bits(1)],
            vec![p(127), p(-149), -p(127)],
            vec![1.0, p(-30), p(-30), p(-30), -1.0],
        ];
        for xs in &named {
            let got = sequential_sum_f32(xs);
            assert!(
                f32_bits_match(got, host(xs)),
                "{xs:?}: got={got:e} expected={:e}",
                host(xs)
            );
        }
        // ランダム列（長さ 1..=64・全指数域）。
        let mut rng = Rng(0xF00D_BABE_1234_5678);
        for i in 0..200_000u32 {
            let len = 1 + (rng.next() % 64) as usize;
            let xs: Vec<f32> = (0..len).map(|_| f32::from_bits(rng.f32_bits())).collect();
            let got = sequential_sum_f32(&xs);
            assert!(
                f32_bits_match(got, host(&xs)),
                "random #{i} {xs:?}: got={got:e} expected={:e}",
                host(&xs)
            );
        }
        // 相殺を狙ったランダム列（x と -x を離れた位置に置き、間に小さい値）。
        for i in 0..100_000u32 {
            let big = f32::from_bits(rng.f32_bits() & 0x7FFF_FFFF);
            let mid: Vec<f32> = (0..(rng.next() % 8))
                .map(|_| f32::from_bits(rng.f32_bits()))
                .collect();
            let mut xs = vec![big];
            xs.extend(mid);
            xs.push(-big);
            let got = sequential_sum_f32(&xs);
            assert!(
                f32_bits_match(got, host(&xs)),
                "cancel #{i} {xs:?}: got={got:e} expected={:e}",
                host(&xs)
            );
        }
    }

    #[test]
    fn mul_matches_hardware_for_random_pairs() {
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        for i in 0..4_000_000u32 {
            let a = rng.f64_bits();
            let b = rng.f64_bits();
            let expected = f64::from_bits(a) * f64::from_bits(b);
            assert_f64_eq(
                mul_f64_bits(a, b),
                expected,
                &format!("mul #{i} {a:#x} * {b:#x}"),
            );
        }
    }

    #[test]
    fn mul_matches_hardware_for_boundary_cases() {
        let cases: [(f64, f64); 20] = [
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, -0.0),
            (0.0, 1.0),
            (0.0, f64::INFINITY),
            (0.0, f64::NEG_INFINITY),
            (f64::INFINITY, f64::INFINITY),
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::NAN, 1.0),
            (f64::NAN, f64::INFINITY),
            (f64::NAN, 0.0),
            (f64::MAX, f64::MAX),
            (f64::MAX, 2.0),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (f64::from_bits(1), f64::from_bits(1)),
            (f64::from_bits(1), 2.0),
            (f64::from_bits(F64_FRAC_MASK), f64::from_bits(1)),
            (-1.0, f64::MAX),
            (f64::from_bits(1), f64::MAX),
            (1.5f64, 1.5f64),
        ];
        for (a, b) in cases {
            assert_f64_eq(
                mul_f64_bits(a.to_bits(), b.to_bits()),
                a * b,
                &format!("{a:e} * {b:e}"),
            );
            assert_f64_eq(
                mul_f64_bits(b.to_bits(), a.to_bits()),
                b * a,
                &format!("{b:e} * {a:e}"),
            );
        }
    }

    #[test]
    fn mul_matches_hardware_for_targeted_exponent_products() {
        // 指数の組合せを広く総当たりし、正規化・subnormal 化・overflow の
        // 各経路を機械的に網羅する（仮数は端値のみ）。
        let fracs = [0u64, 1, F64_FRAC_MASK, F64_FRAC_MASK - 1, 1u64 << 51];
        let exps = [0u64, 1, 2, 500, 1000, 1023, 1500, 2000, 2044, 2045, 2046];
        for &ea in &exps {
            for &eb in &exps {
                for &fa in &fracs {
                    for &fb in &fracs {
                        for sa in [0u64, F64_SIGN] {
                            for sb in [0u64, F64_SIGN] {
                                let a = sa | (ea << 52) | fa;
                                let b = sb | (eb << 52) | fb;
                                let ctx = format!("ea={ea} eb={eb} a={a:#x} b={b:#x}");
                                assert_f64_eq(
                                    mul_f64_bits(a, b),
                                    f64::from_bits(a) * f64::from_bits(b),
                                    &ctx,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn div_matches_hardware_for_random_pairs() {
        let mut rng = Rng(0x0BAD_F00D_DEAD_BEEF);
        for i in 0..4_000_000u32 {
            let a = rng.f64_bits();
            let b = rng.f64_bits();
            let expected = f64::from_bits(a) / f64::from_bits(b);
            assert_f64_eq(
                div_f64_bits(a, b),
                expected,
                &format!("div #{i} {a:#x} / {b:#x}"),
            );
        }
    }

    #[test]
    fn div_matches_hardware_for_boundary_cases() {
        let cases: [(f64, f64); 24] = [
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, -0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, f64::INFINITY),
            (0.0, f64::NEG_INFINITY),
            (f64::INFINITY, 0.0),
            (f64::INFINITY, f64::INFINITY),
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::NAN, 1.0),
            (f64::NAN, f64::INFINITY),
            (f64::NAN, 0.0),
            (f64::MAX, f64::MAX),
            (f64::MAX, 2.0),
            (2.0, f64::MAX),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (f64::MIN_POSITIVE, 3.0),
            (1.0, 3.0),
            (f64::from_bits(1), f64::from_bits(1)),
            (f64::from_bits(1), 2.0),
            (-1.0, f64::MAX),
            (1.5f64, 1.5f64),
        ];
        for (a, b) in cases {
            assert_f64_eq(
                div_f64_bits(a.to_bits(), b.to_bits()),
                a / b,
                &format!("{a:e} / {b:e}"),
            );
        }
    }

    #[test]
    fn div_matches_hardware_for_targeted_exponent_quotients() {
        // `mul_matches_hardware_for_targeted_exponent_products` の除算版:
        // 指数の組合せを広く総当たりし、正規化・subnormal 化・
        // overflow・underflow の各経路を機械的に網羅する。
        let fracs = [0u64, 1, F64_FRAC_MASK, F64_FRAC_MASK - 1, 1u64 << 51];
        let exps = [0u64, 1, 2, 500, 1000, 1023, 1500, 2000, 2044, 2045, 2046];
        for &ea in &exps {
            for &eb in &exps {
                for &fa in &fracs {
                    for &fb in &fracs {
                        for sa in [0u64, F64_SIGN] {
                            for sb in [0u64, F64_SIGN] {
                                let a = sa | (ea << 52) | fa;
                                let b = sb | (eb << 52) | fb;
                                let ctx = format!("ea={ea} eb={eb} a={a:#x} b={b:#x}");
                                assert_f64_eq(
                                    div_f64_bits(a, b),
                                    f64::from_bits(a) / f64::from_bits(b),
                                    &ctx,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// PR #1671 codex-review・Cursor Bugbot 指摘の再現ケース: 一様行
    /// （`layer_norm` の `mean` 相当。要素が全て同じ値の行で、和を要素数
    /// で割ると元の値へ厳密に戻るべきケース）で、Newton 近似逆数との
    /// 積（旧実装）ではなく正しく丸めた除算であることを検証する。
    /// `sum = n * v`（`f64` 演算で厳密に表現できる値を選ぶ）のとき
    /// `div_f64_bits(sum, n) == v` が bit 完全一致するはず。
    #[test]
    fn div_exact_for_uniform_row_mean_repro() {
        // `v` は `f32` 由来（`layer_norm` の入力 `x` は常に `f32` widen 経由
        // のため仮数は 24bit しか使わない）。`hidden`（6bit 未満）との積は
        // 52bit 仮数に楽々収まり、逐次 `f64` 加算（`layer_norm_row` の
        // `sum += v as f64`）は丸めなしで `hidden as f64 * v` と厳密一致
        // する（`crates/backend-cpu/src/layer_norm.rs::layer_norm_row`
        // 冒頭コメント参照）。よって「一様行の mean は入力値そのものへ
        // 厳密に戻るべき」という契約を、実際の呼び出し文脈（`f32` 由来の
        // 値・逐次和）に忠実な形で検証する。
        for &hidden in &[1u64, 2, 3, 7, 16, 49, 128, 4096] {
            for &val_f32 in &[1.0e30f32, 1.0, -1.0, 1.0e-30, 3.5, -1.0e20, 1.0e-38] {
                let v = f64::from_bits(widen_f32_bits(val_f32.to_bits()));
                let n = hidden as f64;
                let mut sum = 0.0f64;
                for _ in 0..hidden {
                    sum += v;
                }
                let ctx = format!("hidden={hidden} val={val_f32:e}");
                assert_f64_eq(div_f64_bits(sum.to_bits(), n.to_bits()), v, &ctx);
            }
        }
    }

    #[test]
    fn sub_matches_hardware_for_random_pairs() {
        let mut rng = Rng(0xABCD_EF01_2345_6789);
        for i in 0..500_000u32 {
            let a = rng.f64_bits();
            let b = rng.f64_bits();
            let expected = f64::from_bits(a) - f64::from_bits(b);
            assert_f64_eq(
                sub_f64_bits(a, b),
                expected,
                &format!("sub #{i} {a:#x} - {b:#x}"),
            );
        }
    }

    /// [`recip_newton_f64_bits`] が本カーネルの実用域（`hidden ∈
    /// [1, 2^24]`。有限・正の整数）で真の `f64` 逆数へ十分収束することを
    /// 確認する（Newton 反復は必ずしも「最近接丸め」と bit 完全一致では
    /// ないため、相対誤差ベースで検証する——`layer_norm.metal` 側の
    /// 用途では `narrow_f64_bits` で `f32` へ最終的に丸めるため、この
    /// 精度〈2^-50 未満〉があれば `f32` の丸め結果は真の値と一致する）。
    #[test]
    fn recip_newton_converges_for_hidden_range() {
        for hidden in [1u64, 2, 3, 7, 97, 1024, 1 << 20, 1 << 24] {
            let x = widen_f32_bits((hidden as f32).to_bits());
            let got = f64::from_bits(recip_newton_f64_bits(x));
            let expected = 1.0f64 / (hidden as f64);
            let rel_err = ((got - expected) / expected).abs();
            assert!(
                rel_err < 1e-14,
                "hidden={hidden}: got={got:e} expected={expected:e} rel_err={rel_err:e}"
            );
        }
        // ランダムな正の有限値でも収束することを確認する。
        let mut rng = Rng(0x2222_3333_4444_5555);
        for _ in 0..10_000u32 {
            let mut bits = rng.f64_bits();
            bits &= !F64_SIGN; // 正へ強制。
            let v = f64::from_bits(bits);
            if !v.is_finite() || v == 0.0 {
                continue;
            }
            let got = f64::from_bits(recip_newton_f64_bits(bits));
            let expected = 1.0 / v;
            // `recip_newton_f64_bits` は `hidden`（常に正規化数域の正の
            // 整数）専用のため、逆数が subnormal 域まで潰れる極端な
            // 入力（本関数の実用域外）は対象外とする（種の生成
            // `scale_pow2_f64_bits` は Newton の種としての用途に限定した
            // 簡略版で、そこまでの範囲は保証しない）。
            if !expected.is_finite() || expected.abs() < f64::MIN_POSITIVE {
                continue;
            }
            let rel_err = ((got - expected) / expected).abs();
            assert!(rel_err < 1e-13, "v={v:e} got={got:e} expected={expected:e}");
        }
    }

    /// [`rsqrt_newton_f64_bits`] の特殊値契約（NaN／±0/負/+inf）と、
    /// 通常域での収束精度を確認する。
    #[test]
    fn rsqrt_newton_handles_special_values_and_converges() {
        assert!(f64::from_bits(rsqrt_newton_f64_bits(f64::NAN.to_bits())).is_nan());
        assert_eq!(
            rsqrt_newton_f64_bits(0.0f64.to_bits()),
            f64::INFINITY.to_bits()
        );
        assert_eq!(
            rsqrt_newton_f64_bits((-0.0f64).to_bits()),
            f64::NEG_INFINITY.to_bits()
        );
        assert!(f64::from_bits(rsqrt_newton_f64_bits((-1.0f64).to_bits())).is_nan());
        assert_eq!(
            rsqrt_newton_f64_bits(f64::INFINITY.to_bits()),
            0.0f64.to_bits()
        );

        for v in [
            1.0f64,
            2.0,
            0.5,
            1e-300,
            1e300,
            1.5e75 * 1.5e75, // 巨大な分散相当値。
            f64::MIN_POSITIVE,
        ] {
            let got = f64::from_bits(rsqrt_newton_f64_bits(v.to_bits()));
            let expected = 1.0 / v.sqrt();
            let rel_err = ((got - expected) / expected).abs();
            assert!(
                rel_err < 1e-14,
                "v={v:e} got={got:e} expected={expected:e} rel_err={rel_err:e}"
            );
        }
    }

    /// PR #1671 codex-review 指摘の反例（イシュー #1596）: `xhat=31/16`・
    /// `weight=f32::from_bits(0x7f042108)`・`bias=-1` で、素朴な「`f64`
    /// へ丸めてから `f32` へ narrow する」二段階丸めは `+inf` を返すが、
    /// ハードウェア FMA（`f32::mul_add`）は `f32::MAX`（有限）を返す。
    /// [`fma_f32_bits`]（round-to-odd 経由の単一丸め）が `f32::mul_add`
    /// と bit 完全一致することを確認する（本関数が本 PR の是正そのもの
    /// を検証する回帰テスト）。
    #[test]
    fn fma_matches_hardware_mul_add_for_codex_review_overflow_boundary_case() {
        let xhat = 31.0f32 / 16.0;
        let w = f32::from_bits(0x7f042108);
        let b = -1.0f32;

        let expected = xhat.mul_add(w, b);
        assert_eq!(
            expected.to_bits(),
            f32::MAX.to_bits(),
            "反例の前提（ハードウェア fma が f32::MAX を返す）が崩れている: {expected:e}"
        );

        let got_bits = fma_f32_bits(xhat.to_bits(), w.to_bits(), b.to_bits());
        assert_eq!(
            got_bits,
            expected.to_bits(),
            "fma_f32_bits が f32::mul_add と bit 不一致: got={:e} expected={:e}",
            f32::from_bits(got_bits),
            expected
        );

        // 素朴な二段階丸め（`add_f64_bits` で `f64` へ丸めてから
        // `narrow_f64_bits` で `f32` へ丸める）が実際に `+inf` へ壊れる
        // ことも合わせて確認し、round-to-odd（`add_f64_bits_round_to_odd`）
        // が必須である根拠を残す（このテストが green のまま
        // `add_f64_bits_round_to_odd` を `add_f64_bits` へ差し戻すと
        // この assert が fail するため回帰を検知できる）。
        let product = mul_f64_bits(widen_f32_bits(xhat.to_bits()), widen_f32_bits(w.to_bits()));
        let naive_two_step = narrow_f64_bits(add_f64_bits(product, widen_f32_bits(b.to_bits())));
        assert!(
            f32::from_bits(naive_two_step).is_infinite(),
            "二段階丸めが有限値を返すようになった場合、この反例は本テストの \
             回帰検知として機能しなくなっている（前提の再確認が必要）: {:e}",
            f32::from_bits(naive_two_step)
        );
    }

    /// [`fma_f32_bits`] がランダムな `f32` 3 つ組（`NaN`／`inf`／subnormal
    /// を含む）に対し、ハードウェア `f32::mul_add`（本ホスト参照実装が
    /// `f32::mul_add` を使う唯一の理由はテスト時の比較対象としてであり、
    /// 本番の Metal 側ではこの関数〈のホスト逐語モデル〉を使う。
    /// `.claude/rules/coding-rust.md` の FMA 契約）と bit 完全一致
    /// （NaN はクラス一致）することを網羅的に検証する。
    #[test]
    fn fma_matches_hardware_mul_add_for_random_triples() {
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        for i in 0..2_000_000u32 {
            let a = f32::from_bits(rng.f32_bits());
            let b = f32::from_bits(rng.f32_bits());
            let c = f32::from_bits(rng.f32_bits());
            let expected = a.mul_add(b, c);
            let got = fma_f32(a, b, c);
            assert_f32_eq(
                got.to_bits(),
                expected,
                &format!("fma #{i} a={a:e} b={b:e} c={c:e}"),
            );
        }
    }
}
