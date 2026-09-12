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
}
