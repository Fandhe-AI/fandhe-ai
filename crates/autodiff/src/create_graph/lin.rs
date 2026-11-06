//! 丸めの一次伝播を追う数値型 [`Lin`] と、構造の演算列を再現する数値型の共通 trait [`Num`]
//! （イシュー #4075・設計記録 `docs/autodiff-higher-order-grad-decision.md` §77・§79・§80）。
//!
//! 役割: `backward_create_graph` の総括判定（二階の構造を積む前に、f32 の丸め・相殺・underflow・overflow で
//! REQ-2 の統一複合判定を外れうる入力を拒否する検査）で、f32 の構造と f64 の参照を同じ式で評価するための型。
//! `Lin` は丸めの事象ごとに誤差の一次式を持ち、同じ中間値を共有する式の誤差の打ち消しを上界へ畳む。
//!
//! 呼び出し元: 現状は `norm_reparam.rs` の WeightNorm の総括判定（`wn_group_judge`）と、同ファイルの
//! `T: Num` 総称関数群。SpectralNorm・LRN（#4073）や損失のクラス軸系（#4071）が流用する予定のため、
//! norm 専用ではない共通モジュールとして切り出した（挙動不変の移動）。norm 専用の `Rf`・`Dm` は
//! `norm_reparam.rs` に残す。
//!
//! 契約: IEEE 754 の正しい丸めを仮定する（GPU の実機は未実測）。拒否時の文面は `judge.rs` と同じく
//! `reject: fn(&str) -> AutodiffError`（[`super::judge::Reject`]）で呼び出し元へ委ね、tolerance 定数
//! （`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）は変更しない。

use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

use super::judge::Reject;
use crate::error::AutodiffError;

/// 構造の演算列を再現する数値型。`f32` は構造そのもの（`V_str`）、`norm_reparam.rs` の `Rf` は f64 の参照（`V`・`M`）。
pub(super) trait Num: Copy {
    /// 定数（ホストが f64 で作り、構造は `as f32` で載せる値）。
    fn k(x: f64) -> Self;
    /// 入力 `x` の値（`norm_reparam.rs` の `Dm` だけが依存の大きさとして区別する。ほかは [`Num::k`] と同じ）。
    fn input(x: f64) -> Self {
        Self::k(x)
    }
    fn plus(self, o: Self) -> Self;
    fn minus(self, o: Self) -> Self;
    fn times(self, o: Self) -> Self;
    fn over(self, o: Self) -> Self;
    fn negate(self) -> Self;
    fn root(self) -> Self;
    /// 多項の和（縮約・勾配の蓄積）。f32 の再現は先頭から逐次に足す（`0 + x₀` は厳密なので `x₀.plus(x₁)…`
    /// と同じ丸め）。実行側のカーネルは和の順序が異なりうるため、[`Lin`] だけは順序に依らない上界で上書きする。
    fn total(terms: &[Self]) -> Self {
        terms.iter().fold(Self::k(0.0), |a, &x| a.plus(x))
    }
    /// `PowScalar` の forward `x^e`（構造は `e as f32` を使う）。
    fn powe(self, e: f64) -> Self;
    /// `PowScalar` の VJP 係数 `e·x^(e−1)`（`e == 0` は 0。`eval/scalar.rs::unary_grad_factor`）。
    fn pow_grad(self, e: f64) -> Self;
    /// `Sqrt` の VJP 係数 `0.5 / y`（`self` は forward の出力 `y`）。
    fn root_grad(self) -> Self;
    /// ±inf を含まない（NaN は入力の NaN の伝播として通す。`norm_reparam.rs` の `no_inf` 参照）。
    fn no_inf(self) -> bool;
    /// 値が NaN（`norm_reparam.rs` の `Rf` は参照値 `v`）。
    fn nan(self) -> bool;
    /// 演算で生まれた NaN（入力の NaN に依存しない NaN）。f32 は区別できないので常に `false`（f32 だけの
    /// NaN は `norm_reparam.rs` の `same_nan` が参照との位置の不一致として拒否する）、`norm_reparam.rs` の `Rf` は taint の無い NaN。
    fn generated_nan(self) -> bool;
}

impl Num for f32 {
    fn k(x: f64) -> Self {
        x as f32
    }
    fn plus(self, o: Self) -> Self {
        self + o
    }
    fn minus(self, o: Self) -> Self {
        self - o
    }
    fn times(self, o: Self) -> Self {
        self * o
    }
    fn over(self, o: Self) -> Self {
        self / o
    }
    fn negate(self) -> Self {
        -self
    }
    fn root(self) -> Self {
        f32::sqrt(self)
    }
    fn powe(self, e: f64) -> Self {
        self.powf(e as f32)
    }
    fn pow_grad(self, e: f64) -> Self {
        let e = e as f32;
        if e == 0.0 {
            0.0
        } else {
            e * self.powf(e - 1.0)
        }
    }
    fn root_grad(self) -> Self {
        0.5 / self
    }
    fn no_inf(self) -> bool {
        !self.is_infinite()
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
    fn generated_nan(self) -> bool {
        false
    }
}

/// 丸め誤差の一次伝播を丸めの事象ごとに追う数値型（イシュー #3955・決定記録 §77）。値は f64 の参照値 `v`
/// （厳密な定数・入力から組んだ値）と、f32 の実行がそこから外れる誤差の一次式 `Σ cₖ·δₖ`（`δₖ` は k 番目の
/// 丸めの事象の相対誤差。`|δₖ| ≤ μₖ`）で表す。誤差の上界は `Σ|cₖ|·μₖ`（[`Lin::bound`]）。
/// `norm_reparam.rs` の `Rf` の「全演算を最終の大きさ `M` に一律に課す」係数 `coeff·M` や、演算ごとに独立な誤差として足す
/// running error bound と違い、**同じ中間値を何度も使う式**（`n_s` を `v̂`・`g/n_s`・VJP 係数で共有する
/// WeightNorm 等）で、共有した丸めの誤差が項の間で打ち消し合う分も一次式の係数に畳まれる。これが、大きな項
/// が相殺して小さい値になる Hessian 対角（g=1.5 の値約 0.01 など）を、実際の誤差の桁で見積もれる理由。
/// - 演算の丸め: 結果 `r` ごとに新しい事象 `μ = u·|r| + 2^-150`（`u = 2^-24`。除算・平方根を含め IEEE 754 の
///   正しい丸めを仮定する。CPU・CUDA の既定と、Metal の `MathMode::Safe` + `Precise`〈`pipeline::compile_options`〉
///   の想定。GPU の実機は未実測）。絶対項は非正規数の間隔の半分。
/// - 定数: f32 へ丸めた誤差を厳密に計算した事象（f32 で厳密な値は事象を持たない）。
/// - 多項の和 [`Num::total`]: 実カーネルの和の順序に依らないよう、各項の一次式を足したうえで、順序によらない
///   丸めの上界 `γ_{n−1}·Σ|vᵢ|` を 1 つの独立な事象にする。
/// - 一次近似の二次項: 事象の総数 `N` に対し `1 + 16·N·u` を掛け、`16·N·u ≥ 0.01` の入力は拒否する
///   （[`entry_check_lin`]）。
///
/// 係数の列は thread-local の領域（[`LIN_ARENA`]）に置き、`Lin` 自体は `Copy` にする。事象の数に比例する
/// 長さの列を値ごとに持つので、グループの幅が [`LIN_REST_MAX`] 以下のときだけ使う（`norm_reparam.rs` の `wn_group_judge`）。
/// `PowScalar`（`powe`・`pow_grad`）は WeightNorm が使わないので、上界を持たない NaN の値にする（常に拒否）。
#[derive(Clone, Copy, Debug)]
pub(super) struct Lin {
    v: f64,
    id: usize,
    t: bool,
}

/// 一次式の項（事象の番号, 係数）の列。事象の番号は昇順。
type LinTerms = Vec<(u32, f64)>;

#[derive(Default)]
struct LinArena {
    terms: Vec<LinTerms>,
    mus: Vec<f64>,
}

thread_local! {
    static LIN_ARENA: std::cell::RefCell<LinArena> = std::cell::RefCell::new(LinArena::default());
}

/// f32 の丸め単位 `u = 2^-24`。
const LIN_U: f64 = 1.0 / 16_777_216.0;
/// 非正規数の領域の絶対丸め誤差の上界 `2^-150`。
const LIN_TINY: f64 = 7.006_492_321_624_085e-46;
/// [`Lin`] を使うグループの幅の上限（事象の数が幅に比例して増え、係数の列が長くなるため）。
pub(super) const LIN_REST_MAX: usize = 8;
/// [`Lin`] で判定するグループ 1 つの走査量の見積り（幅 8 で約 4e5 反復相当、幅の 2 乗に比例とした見積り）。
/// 計測は `tests::lin_judge_cost_profile`（`#[ignore]`）、実測との照合は決定記録 §80.2（イシュー #4076）。
/// 値は据え置き（見直しは #4082）。
pub(super) fn lin_group_cost(rest: usize) -> usize {
    6000 * rest * rest
}
/// [`Lin`] を使う判定の走査量の上限。超える（グループ数が多い）テンソルは全グループを従来の `norm_reparam.rs` の `Rf` で判定する
/// （グループの位置で判定方式が変わらないよう、テンソル単位で選ぶ）。
pub(super) const LIN_BUDGET: usize = 1 << 26;

// 計測用の計数器（`#[cfg(test)]` 限定。非テストビルドには存在せず、生成コードを変えない）。
// イシュー #4076 の計測テスト（`tests::lin_judge_cost_profile`）が判定時間の内訳を読む。#4077（確保・clone の削減）・
// #4078（疎化）が同じ計数で改善量を測るため残す。バイト数は要素数 × `size_of::<(u32, f64)>()` で換算する
// （`Vec` のヘッダ・容量の余りは含まない）。添字は `PROF_*` 定数。
#[cfg(test)]
thread_local! {
    static LIN_PROF: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
}
/// 係数列の新規確保の回数。
#[cfg(test)]
const PROF_ALLOCS: usize = 0;
/// 新規確保した係数列の要素数（予約量）。
#[cfg(test)]
const PROF_ALLOC_ENTRIES: usize = 1;
/// 係数列の clone の回数。
#[cfg(test)]
const PROF_CLONES: usize = 2;
/// clone した係数列の要素数。
#[cfg(test)]
const PROF_CLONE_ENTRIES: usize = 3;
/// [`Lin::make`]（領域へ積まれる値）の回数。
#[cfg(test)]
const PROF_MAKES: usize = 4;

#[cfg(test)]
fn prof_add(idx: usize, n: usize) {
    LIN_PROF.with(|c| {
        let mut a = c.get();
        a[idx] += n as u64;
        c.set(a);
    });
}

/// 領域を空にする（グループの判定の先頭で呼ぶ）。
pub(super) fn lin_reset() {
    LIN_ARENA.with(|a| *a.borrow_mut() = LinArena::default());
}

/// 領域の現在の大きさ（`(値の数, 事象の数)`）。seed ごとに [`lin_truncate`] で戻すための印。
pub(super) fn lin_mark() -> (usize, usize) {
    LIN_ARENA.with(|a| {
        let a = a.borrow();
        (a.terms.len(), a.mus.len())
    })
}

pub(super) fn lin_truncate(mark: (usize, usize)) {
    LIN_ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.terms.truncate(mark.0);
        a.mus.truncate(mark.1);
    });
}

/// 事象の総数（二次項の余裕係数の入力）。
fn lin_events() -> usize {
    LIN_ARENA.with(|a| a.borrow().mus.len())
}

/// 2 つの一次式の線形結合 `ka·a + kb·b`（事象の番号の昇順の併合）。
fn lin_comb(a: &[(u32, f64)], ka: f64, b: &[(u32, f64)], kb: f64) -> LinTerms {
    let mut out = Vec::with_capacity(a.len() + b.len());
    #[cfg(test)]
    {
        prof_add(PROF_ALLOCS, 1);
        prof_add(PROF_ALLOC_ENTRIES, a.len() + b.len());
    }
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i].0 < b[j].0) {
            out.push((a[i].0, ka * a[i].1));
            i += 1;
        } else if i == a.len() || b[j].0 < a[i].0 {
            out.push((b[j].0, kb * b[j].1));
            j += 1;
        } else {
            out.push((a[i].0, ka * a[i].1 + kb * b[j].1));
            i += 1;
            j += 1;
        }
    }
    out
}

impl Lin {
    fn terms_of(self) -> LinTerms {
        let t = LIN_ARENA.with(|a| a.borrow().terms[self.id].clone());
        #[cfg(test)]
        {
            prof_add(PROF_CLONES, 1);
            prof_add(PROF_CLONE_ENTRIES, t.len());
        }
        t
    }

    fn make(v: f64, terms: LinTerms, t: bool) -> Self {
        #[cfg(test)]
        prof_add(PROF_MAKES, 1);
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.terms.push(terms);
            a.terms.len() - 1
        });
        Lin { v, id, t }
    }

    /// 結果 `v` の丸めの事象を 1 つ足す。
    fn rounded(v: f64, mut terms: LinTerms, t: bool) -> Self {
        let mu = LIN_U * v.abs() + LIN_TINY;
        let ev = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.mus.push(mu);
            (a.mus.len() - 1) as u32
        });
        terms.push((ev, 1.0));
        Lin::make(v, terms, t)
    }

    /// 誤差の絶対値の一次の上界 `Σ|cₖ|·μₖ`。
    fn bound(self) -> f64 {
        LIN_ARENA.with(|a| {
            let a = a.borrow();
            a.terms[self.id]
                .iter()
                .map(|&(k, c)| c.abs() * a.mus[k as usize])
                .sum()
        })
    }

    /// 入力の NaN が届いた NaN（taint 付きの NaN）。
    fn input_nan(self) -> bool {
        self.v.is_nan() && self.t
    }

    fn lin2(self, ka: f64, o: Self, kb: f64) -> LinTerms {
        LIN_ARENA.with(|a| {
            let a = a.borrow();
            lin_comb(&a.terms[self.id], ka, &a.terms[o.id], kb)
        })
    }
}

impl Num for Lin {
    fn k(x: f64) -> Self {
        // 定数は f32 へ丸めて載る。丸めの誤差は厳密に計算できる（f32 で厳密な値は事象を持たない）。
        let mut terms = LinTerms::new();
        let err = if x.is_finite() {
            (f64::from(x as f32) - x).abs()
        } else {
            0.0
        };
        if err > 0.0 {
            let ev = LIN_ARENA.with(|a| {
                let mut a = a.borrow_mut();
                a.mus.push(err);
                (a.mus.len() - 1) as u32
            });
            terms.push((ev, 1.0));
        }
        Lin::make(x, terms, x.is_nan())
    }
    fn plus(self, o: Self) -> Self {
        Lin::rounded(self.v + o.v, self.lin2(1.0, o, 1.0), self.t || o.t)
    }
    fn minus(self, o: Self) -> Self {
        Lin::rounded(self.v - o.v, self.lin2(1.0, o, -1.0), self.t || o.t)
    }
    fn times(self, o: Self) -> Self {
        Lin::rounded(self.v * o.v, self.lin2(o.v, o, self.v), self.t || o.t)
    }
    fn over(self, o: Self) -> Self {
        let q = self.v / o.v;
        Lin::rounded(q, self.lin2(1.0 / o.v, o, -q / o.v), self.t || o.t)
    }
    fn negate(self) -> Self {
        let terms = self.terms_of().into_iter().map(|(k, c)| (k, -c)).collect();
        Lin::make(-self.v, terms, self.t)
    }
    fn root(self) -> Self {
        let r = self.v.sqrt();
        let terms = self
            .terms_of()
            .into_iter()
            .map(|(k, c)| (k, c / (2.0 * r)))
            .collect();
        Lin::rounded(r, terms, self.t)
    }
    fn powe(self, e: f64) -> Self {
        Lin::make(self.v.powf(e), vec![], self.t).unbounded()
    }
    fn pow_grad(self, e: f64) -> Self {
        if e == 0.0 {
            return Lin::make(0.0, vec![], false);
        }
        Lin::make(e * self.v.powf(e - 1.0), vec![], self.t).unbounded()
    }
    fn root_grad(self) -> Self {
        Lin::k(0.5).over(self)
    }
    fn total(terms: &[Self]) -> Self {
        let v: f64 = terms.iter().map(|x| x.v).sum();
        let t = terms.iter().any(|x| x.t);
        let n = terms.len();
        let mut acc = LinTerms::new();
        for x in terms {
            acc = LIN_ARENA.with(|a| lin_comb(&acc, 1.0, &a.borrow().terms[x.id], 1.0));
        }
        let abs: f64 = terms.iter().map(|x| x.v.abs()).sum();
        let k = n.saturating_sub(1) as f64 * LIN_U;
        let mu = k / (1.0 - k) * abs + n as f64 * LIN_TINY;
        let ev = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.mus.push(mu);
            (a.mus.len() - 1) as u32
        });
        acc.push((ev, 1.0));
        Lin::make(v, acc, t)
    }
    fn no_inf(self) -> bool {
        !self.v.is_infinite()
    }
    fn nan(self) -> bool {
        self.v.is_nan()
    }
    fn generated_nan(self) -> bool {
        self.v.is_nan() && !self.t
    }
}

impl Lin {
    /// 上界を持たない値（常に拒否される）。係数の NaN で表す。
    fn unbounded(self) -> Self {
        let ev = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.mus.push(f64::NAN);
            (a.mus.len() - 1) as u32
        });
        Lin::make(self.v, vec![(ev, 1.0)], self.t)
    }
}

/// `norm_reparam.rs` の `entry_check` の [`Lin`] 版。`|V_str − V| + 一次の上界·(1 + 16·N·u)` が絶対 1e-5 未満か相対の半分以内で
/// なければ拒否する（tolerance 定数は変えない）。NaN・±inf の扱いは `norm_reparam.rs` の `entry_check` と同じ。
pub(super) fn entry_check_lin(
    r: Lin,
    s: f32,
    what: &str,
    reject: Reject,
) -> Result<(), AutodiffError> {
    if s.is_nan() && r.input_nan() {
        return Ok(());
    }
    let slack = 16.0 * lin_events() as f64 * LIN_U;
    let err = (f64::from(s) - r.v).abs() + r.bound() * (1.0 + slack);
    if slack < 0.01
        && r.v.is_finite()
        && err.is_finite()
        && (err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * r.v.abs())
    {
        return Ok(());
    }
    Err(reject(&format!(
        "{what}: 二階の要素が f32 構造の丸め・相殺・underflow・overflow で REQ-2 の判定を外れうるため拒否"
    )))
}

#[cfg(test)]
mod tests {
    use super::super::norm_reparam::{WnSeed, pow2, wn_coeff, wn_fwd, wn_group_judge, wn_rev};
    use super::*;
    use crate::create_graph_norm::pow2_ceil_exponent;

    /// 和の順序だけが [`f32`] と違う再現（多項の和を末尾から足す）。実カーネルの和の順序が違っても
    /// [`Lin`] の上界が覆うこと（[`Num::total`] の順序非依存）を確かめるために使う。
    #[derive(Clone, Copy)]
    struct F32Rev(f32);

    impl Num for F32Rev {
        fn k(x: f64) -> Self {
            F32Rev(x as f32)
        }
        fn plus(self, o: Self) -> Self {
            F32Rev(self.0 + o.0)
        }
        fn minus(self, o: Self) -> Self {
            F32Rev(self.0 - o.0)
        }
        fn times(self, o: Self) -> Self {
            F32Rev(self.0 * o.0)
        }
        fn over(self, o: Self) -> Self {
            F32Rev(self.0 / o.0)
        }
        fn negate(self) -> Self {
            F32Rev(-self.0)
        }
        fn root(self) -> Self {
            F32Rev(self.0.sqrt())
        }
        fn total(terms: &[Self]) -> Self {
            terms.iter().rev().fold(F32Rev(0.0), |a, &x| a.plus(x))
        }
        fn powe(self, e: f64) -> Self {
            F32Rev(self.0.powf(e as f32))
        }
        fn pow_grad(self, e: f64) -> Self {
            F32Rev(self.0.pow_grad(e))
        }
        fn root_grad(self) -> Self {
            F32Rev(0.5 / self.0)
        }
        fn no_inf(self) -> bool {
            !self.0.is_infinite()
        }
        fn nan(self) -> bool {
            self.0.is_nan()
        }
        fn generated_nan(self) -> bool {
            false
        }
    }

    /// 総括判定の soundness（#3955 の最重要テスト）: 乱数・スケールを振った WeightNorm のグループで、[`Lin`] の
    /// 一次の上界が、f32 の再現と和の順序を変えた再現の実測の誤差 `|V_str − V|` を seed 全部・全要素で
    /// 上から押さえる（二階の `v`・`up`・`g` への要素）。
    #[test]
    fn lin_bound_covers_f32_deviation_for_weight_norm_sweep() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f64) / f64::from(1u32 << 31)
        };
        let mut checked = 0usize;
        let mut tight = 0usize;
        for case in 0..300 {
            let rest = 2 + case % 7;
            let amp = [1e-3, 1.0, 1e3][case % 3];
            let v: Vec<f32> = (0..rest)
                .map(|_| ((next() * 2.0 - 1.0) * 2.0 * amp) as f32)
                .collect();
            if v.iter().all(|&x| x == 0.0) {
                continue;
            }
            let up: Vec<f32> = (0..rest)
                .map(|_| ((next() * 2.0 - 1.0) * 3.0) as f32)
                .collect();
            let g = (0.05 + next() * 2.0) as f32;
            let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            lin_reset();
            let (Some(fl), Some(fs), Some(fa)) = (
                wn_fwd::<Lin>(&v, &up, g, s),
                wn_fwd::<f32>(&v, &up, g, s),
                wn_fwd::<F32Rev>(&v, &up, g, s),
            ) else {
                continue;
            };
            let mark = lin_mark();
            for seed in (0..rest).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
                lin_truncate(mark);
                let (mut hv_l, mut hu_l) = (vec![Lin::k(0.0); rest], vec![Lin::k(0.0); rest]);
                let (mut hv_s, mut hu_s) = (vec![0f32; rest], vec![0f32; rest]);
                let (mut hv_a, mut hu_a) = (vec![F32Rev(0.0); rest], vec![F32Rev(0.0); rest]);
                let gl = wn_rev(&fl, seed, &mut hv_l, &mut hu_l);
                let gs = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
                let ga = wn_rev(&fa, seed, &mut hv_a, &mut hu_a);
                let slack = 1.0 + 16.0 * lin_events() as f64 * LIN_U;
                let mut pairs: Vec<(Lin, f32, f32)> = (0..rest)
                    .flat_map(|i| [(hv_l[i], hv_s[i], hv_a[i].0), (hu_l[i], hu_s[i], hu_a[i].0)])
                    .collect();
                if let (Some(gl), Some(gs), Some(ga)) = (gl, gs, ga) {
                    pairs.push((gl, gs, ga.0));
                }
                for (l, f, a) in pairs {
                    if !l.v.is_finite() || !f.is_finite() || !a.is_finite() {
                        continue;
                    }
                    let bound = l.bound() * slack;
                    for actual in [f, a] {
                        let dev = (f64::from(actual) - l.v).abs();
                        assert!(
                            dev <= bound,
                            "case {case} rest {rest}: 実測の誤差 {dev:e} が上界 {bound:e} を超えた（V = {:e}）",
                            l.v
                        );
                        checked += 1;
                        if dev > 0.1 * bound {
                            tight += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 1000, "検査した要素数 {checked}");
        // 上界が緩すぎて自明に成り立っているだけでないこと（実測の誤差が上界の 1 割を超える要素がある）。
        assert!(tight > 0, "上界が実測より桁違いに大きい");
    }

    /// 判定方式（[`Lin`]／従来の `Rf`。後者は幅の広いグループ・グループ数の多いテンソルの経路）のどちらでも、
    /// 相殺のない普通のグループは受理される。
    #[test]
    fn wn_group_judge_modes_agree_on_easy_group() {
        let (v, up) = ([0.9f32, 0.1, 0.6, 0.3], [0.7f32, -0.2, 0.4, 0.1]);
        for use_lin in [true, false] {
            assert!(wn_group_judge(&v, &up, 0.5, 1.0, wn_coeff(4), use_lin).is_ok());
        }
    }
    // ---- 判定時間の計測（イシュー #4076。`#[ignore]`・手動実行専用。記録は決定記録 §80.2） ----

    use super::super::norm_reparam::wn_judge_cost;
    use std::hint::black_box;
    use std::time::Instant;

    /// 計測の入力 `(v, up, g, s)`。
    type BenchInput = (Vec<f32>, Vec<f32>, f32, f64);

    /// 決定的な LCG（[0, 1)）。入力生成用。
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f64) / f64::from(1u32 << 31)
        }
    }

    /// 計測用の入力。`v` は振幅 1・全ゼロにならない。`s` は `v` の最大絶対値以上の 2 のべき乗。
    fn gen_input(rng: &mut Lcg, rest: usize) -> BenchInput {
        loop {
            let v: Vec<f32> = (0..rest)
                .map(|_| ((rng.next() * 2.0 - 1.0) * 2.0) as f32)
                .collect();
            let up: Vec<f32> = (0..rest)
                .map(|_| ((rng.next() * 2.0 - 1.0) * 3.0) as f32)
                .collect();
            let g = (0.05 + rng.next() * 2.0) as f32;
            if v.iter().all(|&x| x == 0.0) {
                continue;
            }
            let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            return (v, up, g, f64::from(pow2(pow2_ceil_exponent(mx))));
        }
    }

    fn bench_reject(_: &str) -> AutodiffError {
        AutodiffError::Backward(String::new())
    }

    /// 全走査の結果（`n_dv`／`n_dg` は seed 種別ごとの検査時点の事象数 `N` の最大）。
    #[derive(Default, Clone, Copy)]
    struct Scan {
        n_dv: usize,
        n_dg: usize,
        accepted: usize,
        rejected: usize,
    }

    /// `wn_group_judge_with::<Lin>` の判定ループを、最初の拒否で打ち切らず全要素を検査する版。本番は拒否で
    /// 短絡するため、拒否される入力でも時間が途中で切れないようにする計測専用の再実装。forward の NaN 位置の
    /// 照合（`same_nan`）は走査の主要コストでないため省く。
    fn scan_lin(v: &[f32], up: &[f32], g: f32, s: f64) -> Scan {
        let rest = v.len();
        lin_reset();
        let (Some(fl), Some(fs)) = (wn_fwd::<Lin>(v, up, g, s), wn_fwd::<f32>(v, up, g, s)) else {
            return Scan::default();
        };
        let mark = lin_mark();
        let mut out = Scan::default();
        let (mut hv_s, mut hu_s) = (vec![0f32; rest], vec![0f32; rest]);
        for seed in (0..rest).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
            lin_truncate(mark);
            let (mut hv_l, mut hu_l) = (vec![Lin::k(0.0); rest], vec![Lin::k(0.0); rest]);
            let gs = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
            let gl = wn_rev(&fl, seed, &mut hv_l, &mut hu_l);
            let n = lin_events();
            match seed {
                WnSeed::Dg => out.n_dg = out.n_dg.max(n),
                WnSeed::Dv(_) => out.n_dv = out.n_dv.max(n),
            }
            let mut tally = |r: Result<(), AutodiffError>| {
                if black_box(r).is_ok() {
                    out.accepted += 1;
                } else {
                    out.rejected += 1;
                }
            };
            if let (Some(gs), Some(gl)) = (gs, gl) {
                tally(entry_check_lin(gl, gs, "p", bench_reject));
            }
            for i in 0..rest {
                tally(entry_check_lin(hv_l[i], hv_s[i], "p", bench_reject));
                tally(entry_check_lin(hu_l[i], hu_s[i], "p", bench_reject));
            }
        }
        out
    }

    fn middle_value(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(|a, b| a.total_cmp(b));
        xs[xs.len() / 2]
    }

    /// 入力を全走査で判定する時間。1 グループ当たりの ns を返す。
    fn time_scan_lin(inputs: &[BenchInput]) -> f64 {
        let t0 = Instant::now();
        for (v, up, g, s) in inputs {
            black_box(scan_lin(v, up, *g, *s));
        }
        t0.elapsed().as_nanos() as f64 / inputs.len() as f64
    }

    /// 5 回計測の中央値（ウォームアップ 1 回つき）と各回の値。1 回が重いとき（>6 s）は回数を 3 回へ落とす。
    fn measure_lin(inputs: &[BenchInput]) -> (f64, Vec<f64>) {
        let first = time_scan_lin(inputs);
        let reps = if first * inputs.len() as f64 > 6e9 {
            3
        } else {
            5
        };
        let xs: Vec<f64> = (0..reps).map(|_| time_scan_lin(inputs)).collect();
        (middle_value(xs.clone()), xs)
    }

    /// 事象数 `N` の最大を、検査を省いて 3 つの seed（`Dg`・先頭と末尾の `Dv`）の逆伝播だけで数える
    /// （上限幅の探索用。全走査 [`scan_lin`] は幅の 3 乗超で重いため。`lin_judge_cost_profile` が小さい幅で
    /// 全走査の `N` と一致することを確かめる）。
    fn n_probe(v: &[f32], up: &[f32], g: f32, s: f64) -> usize {
        let rest = v.len();
        lin_reset();
        let Some(fl) = wn_fwd::<Lin>(v, up, g, s) else {
            return 0;
        };
        let mark = lin_mark();
        let mut n = 0;
        for seed in [WnSeed::Dg, WnSeed::Dv(0), WnSeed::Dv(rest - 1)] {
            lin_truncate(mark);
            let (mut hv, mut hu) = (vec![Lin::k(0.0); rest], vec![Lin::k(0.0); rest]);
            let _ = wn_rev(&fl, seed, &mut hv, &mut hu);
            n = n.max(lin_events());
        }
        n
    }

    fn n_max_of(rest: usize, rng: &mut Lcg) -> usize {
        let (v, up, g, s) = gen_input(rng, rest);
        n_probe(&v, &up, g, s)
    }

    /// `16·N·u < 0.01`（[`entry_check_lin`] の slack 条件）。
    fn slack_ok(n: usize) -> bool {
        16.0 * n as f64 * LIN_U < 0.01
    }

    /// `Lin` の判定時間の計測（イシュー #4076）。幅 2〜64 の 1 グループ当たりの時間（release・5 回中央値）、
    /// グループ数の影響、確保・clone の内訳、従来の `Rf` 経路との換算（反復相当）、`16·N·u < 0.01` で決まる
    /// 幅の上限を stderr に `LINPROF` 行で出す。判定・定数は変えない（値の変更は #4082）。
    ///
    /// 実行: `cargo test -p fandhe-ai-autodiff --release --lib -- --ignored --nocapture --test-threads=1 lin_judge_cost_profile`
    /// 記録先: `docs/autodiff-higher-order-grad-decision.md` §80.2。
    #[test]
    #[ignore = "手動実行専用の計測（stderr 出力のみ。#4076・§80.2）"]
    fn lin_judge_cost_profile() {
        let bytes = std::mem::size_of::<(u32, f64)>();
        let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
        let widths = [2usize, 4, 8, 16, 32, 64];
        let mut prev: Option<(usize, f64)> = None;
        // 1. 幅の軸と 3. 内訳（本番判定は `wn_group_judge(use_lin = true)` の結果）。
        for &rest in &widths {
            let groups = match rest {
                0..=8 => 32,
                9..=16 => 8,
                17..=32 => 2,
                _ => 1,
            };
            let inputs: Vec<_> = (0..groups).map(|_| gen_input(&mut rng, rest)).collect();
            let (med, raw) = measure_lin(&inputs);
            let exp = prev.map_or(f64::NAN, |(pr, pt)| {
                (med / pt).ln() / (rest as f64 / pr as f64).ln()
            });
            prev = Some((rest, med));
            let (v, up, g, s) = &inputs[0];
            LIN_PROF.with(|c| c.set([0; 5]));
            let sc = scan_lin(v, up, *g, *s);
            let p = LIN_PROF.with(|c| c.get());
            let n = sc.n_dv.max(sc.n_dg);
            let probe_matches = n_probe(v, up, *g, *s) == n;
            let verdict = wn_group_judge(v, up, *g, *s, wn_coeff(rest), true).is_ok();
            eprintln!(
                "LINPROF kind=width rest={rest} groups={groups} reps={} median_ns_per_group={med:.0} exp_vs_prev={exp:.2} \
                 N_dv={} N_dg={} slack={:.5} allocs={} alloc_entries={} alloc_bytes={} clones={} clone_entries={} clone_bytes={} \
                 makes={} entries_accepted={} entries_rejected={} probe_matches_scan={} prod_verdict={} raw_ns={:?}",
                raw.len(),
                sc.n_dv,
                sc.n_dg,
                16.0 * n as f64 * LIN_U,
                p[PROF_ALLOCS],
                p[PROF_ALLOC_ENTRIES],
                p[PROF_ALLOC_ENTRIES] as usize * bytes,
                p[PROF_CLONES],
                p[PROF_CLONE_ENTRIES],
                p[PROF_CLONE_ENTRIES] as usize * bytes,
                p[PROF_MAKES],
                sc.accepted,
                sc.rejected,
                probe_matches,
                if verdict { "accept" } else { "reject" },
                raw.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            );
        }
        // 2. グループ数の軸（グループごとに領域を戻すので、1 グループ当たりの時間は数に依らないはず）。
        for &rest in &[8usize, 32] {
            for &groups in &[1usize, 16, 128] {
                if rest * groups > 2048 {
                    eprintln!("LINPROF kind=groups rest={rest} groups={groups} skipped=too-long");
                    continue;
                }
                let inputs: Vec<_> = (0..groups).map(|_| gen_input(&mut rng, rest)).collect();
                let (med, _) = measure_lin(&inputs);
                eprintln!(
                    "LINPROF kind=groups rest={rest} groups={groups} median_ns_per_group={med:.0}"
                );
            }
        }
        // 4. 反復相当: `Rf` 経路（受理される入力でのみ採用）の 1 反復当たり時間で `Lin` の時間を割る。
        for &rest in &widths {
            let mut tries = 0usize;
            let mut found = None;
            while tries < 2000 {
                tries += 1;
                let inp = gen_input(&mut rng, rest);
                if wn_group_judge(&inp.0, &inp.1, inp.2, inp.3, wn_coeff(rest), false).is_ok() {
                    found = Some(inp);
                    break;
                }
            }
            let Some((v, up, g, s)) = found else {
                eprintln!(
                    "LINPROF kind=equiv rest={rest} rf_unmeasured=no-accepted-input tries={tries}"
                );
                continue;
            };
            let reps_inner = (2_000_000 / wn_judge_cost(1, rest)).max(1);
            let one = |lin: bool| -> f64 {
                let t0 = Instant::now();
                for _ in 0..reps_inner {
                    black_box(wn_group_judge(&v, &up, g, s, wn_coeff(rest), lin).is_ok());
                }
                t0.elapsed().as_nanos() as f64 / reps_inner as f64
            };
            let _ = one(false);
            let rf_ns = middle_value((0..5).map(|_| one(false)).collect());
            let ns_per_iter = rf_ns / wn_judge_cost(1, rest) as f64;
            let lin_ns = middle_value((0..5).map(|_| one(true)).collect());
            eprintln!(
                "LINPROF kind=equiv rest={rest} tries={tries} rf_ns_per_group={rf_ns:.0} ns_per_iter={ns_per_iter:.3} \
                 lin_prod_ns_per_group={lin_ns:.0} lin_equiv_iters={:.0} lin_group_cost={}",
                lin_ns / ns_per_iter,
                lin_group_cost(rest),
            );
        }
        // 5. 幅の上限: slack 条件を満たす最大の幅。倍々で探索してから二分探索（N は幅に対し単調と仮定）。
        let mut lo = 2usize;
        let mut hi = 4usize;
        while hi <= 512 && slack_ok(n_max_of(hi, &mut rng)) {
            lo = hi;
            hi *= 2;
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if slack_ok(n_max_of(mid, &mut rng)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (n_lo, n_hi) = (n_max_of(lo, &mut rng), n_max_of(hi, &mut rng));
        let same_n =
            (0..5).all(|_| n_max_of(lo, &mut rng) == n_lo && n_max_of(hi, &mut rng) == n_hi);
        let (v, up, g, s) = gen_input(&mut rng, hi);
        let hi_rejected = wn_group_judge(&v, &up, g, s, wn_coeff(hi), true).is_err();
        eprintln!(
            "LINPROF kind=limit rest_max={lo} N_at_max={n_lo} slack_at_max={:.5} rest_max_plus1={hi} N_at_plus1={n_hi} \
             slack_at_plus1={:.5} n_input_independent={same_n} plus1_rejected_by_prod={hi_rejected}",
            16.0 * n_lo as f64 * LIN_U,
            16.0 * n_hi as f64 * LIN_U,
        );
        assert!(slack_ok(n_lo) && !slack_ok(n_hi) && hi_rejected);
    }
}
