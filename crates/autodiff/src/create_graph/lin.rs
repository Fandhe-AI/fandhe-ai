//! 丸めの一次伝播を追う数値型 [`Lin`] と、構造の演算列を再現する数値型の共通 trait [`Num`]
//! （イシュー #4075・設計記録 `docs/autodiff-higher-order-grad-decision.md` §77・§79・§80）。
//!
//! 役割: `backward_create_graph` の総括判定（二階の構造を積む前に、f32 の丸め・相殺・underflow・overflow で
//! REQ-2 の統一複合判定を外れうる入力を拒否する検査）で、f32 の構造と f64 の参照を同じ式で評価するための型。
//! `Lin` は丸めの事象ごとに誤差の一次式を持ち、同じ中間値を共有する式の誤差の打ち消しを上界へ畳む。
//!
//! 呼び出し元: `norm_reparam.rs` の WeightNorm の総括判定（`wn_group_judge`）と SpectralNorm の総括判定
//! （`sn_judge`。#4080。全要素に依存する値の係数列は [`Num::settle`] で束ねる）、同ファイルの
//! `T: Num` 総称関数群。LRN（#4073）や損失のクラス軸系（#4071）が流用する予定のため、
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
    /// 係数列を束ねる（[`Lin`] だけが実装する。他の型は恒等）。`anchors` の係数列に現れる事象（共有するスカラー値の
    /// 誤差）は厳密に残し、それ以外の項を `Σ|cₖ|·μₖ` を大きさとする独立な事象 1 つへまとめる。三角不等式で上界は
    /// 広がる側にしか動かないので健全で、係数列の長さが全要素に依存して O(N) になるのを抑える（#4080）。
    fn settle(self, _anchors: &[Self]) -> Self {
        self
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
/// 長さの列を値ごとに持つので、WeightNorm はグループの幅が [`LIN_REST_MAX`] 以下のときだけ、SpectralNorm は `sn_use_lin` の条件内（係数列を束ねて走査量を `O(N²)` に抑える）のときだけ使う。
/// `PowScalar`（`powe`・`pow_grad`）は WeightNorm が使わないので、上界を持たない NaN の値にする（常に拒否）。
#[derive(Clone, Copy, Debug)]
pub(super) struct Lin {
    pub(super) v: f64,
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
pub(super) const LIN_U: f64 = 1.0 / 16_777_216.0;
/// 非正規数の領域の絶対丸め誤差の上界 `2^-150`。
const LIN_TINY: f64 = 7.006_492_321_624_085e-46;
/// [`Lin`] を使うグループの幅の上限（事象の数が幅に比例して増え、係数の列が長くなるため）。
pub(super) const LIN_REST_MAX: usize = 8;
/// [`Lin`] で判定するグループ 1 つの走査量の見積り（幅 8 で約 4e5 反復相当の実測。幅の 2 乗に比例）。
pub(super) fn lin_group_cost(rest: usize) -> usize {
    6000 * rest * rest
}
/// [`Lin`] を使う判定の走査量の上限。超える（グループ数が多い）テンソルは全グループを従来の `norm_reparam.rs` の `Rf` で判定する
/// （グループの位置で判定方式が変わらないよう、テンソル単位で選ぶ）。
pub(super) const LIN_BUDGET: usize = 1 << 26;

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
pub(super) fn lin_events() -> usize {
    LIN_ARENA.with(|a| a.borrow().mus.len())
}

/// 2 つの一次式の線形結合 `ka·a + kb·b`（事象の番号の昇順の併合）。
fn lin_comb(a: &[(u32, f64)], ka: f64, b: &[(u32, f64)], kb: f64) -> LinTerms {
    let mut out = Vec::with_capacity(a.len() + b.len());
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
        LIN_ARENA.with(|a| a.borrow().terms[self.id].clone())
    }

    fn make(v: f64, terms: LinTerms, t: bool) -> Self {
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
    pub(super) fn bound(self) -> f64 {
        LIN_ARENA.with(|a| {
            let a = a.borrow();
            a.terms[self.id]
                .iter()
                .map(|&(k, c)| c.abs() * a.mus[k as usize])
                .sum()
        })
    }

    /// 入力の NaN が届いた NaN（taint 付きの NaN）。
    pub(super) fn input_nan(self) -> bool {
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
        // 各項の一次式を足す。全項の（事象, 係数）を事象の番号で安定に並べて同じ番号を項の順に足す
        // （逐次の併合と同じ加算順。項数に対し O(L log L)）。
        let mut all: Vec<(u32, f64)> = LIN_ARENA.with(|a| {
            let a = a.borrow();
            terms
                .iter()
                .flat_map(|x| a.terms[x.id].iter().copied())
                .collect()
        });
        all.sort_by_key(|&(k, _)| k);
        let mut acc = LinTerms::with_capacity(all.len());
        for (k, c) in all {
            match acc.last_mut() {
                Some((lk, lc)) if *lk == k => *lc += c,
                _ => acc.push((k, c)),
            }
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
    fn settle(self, anchors: &[Self]) -> Self {
        let (kept, mu) = LIN_ARENA.with(|a| {
            let a = a.borrow();
            let mut keep: Vec<u32> = anchors
                .iter()
                .flat_map(|x| a.terms[x.id].iter().map(|&(k, _)| k))
                .collect();
            keep.sort_unstable();
            keep.dedup();
            let mut kept = LinTerms::new();
            let mut mu = 0.0f64;
            for &(k, c) in &a.terms[self.id] {
                if keep.binary_search(&k).is_ok() {
                    kept.push((k, c));
                } else {
                    mu += c.abs() * a.mus[k as usize];
                }
            }
            (kept, mu)
        });
        if mu == 0.0 {
            return Lin::make(self.v, kept, self.t);
        }
        let ev = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.mus.push(mu);
            (a.mus.len() - 1) as u32
        });
        let mut kept = kept;
        kept.push((ev, 1.0));
        Lin::make(self.v, kept, self.t)
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
    use super::super::norm_reparam::{
        WnSeed, pow2, sn_fwd, sn_rev, wn_coeff, wn_fwd, wn_group_judge, wn_rev,
    };
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

    /// SpectralNorm 版の soundness（#4080）: 乱数・スケール・`σ_s` の相殺の強さ（`W` を `P` 方向へ寄せる α）を振り、
    /// [`Lin`]（`σ_s`・`sp`・`σ_s` への cotangent の係数列を [`Num::settle`] で束ねる）の一次の上界が、f32 の再現と
    /// 縮約の順序を変えた再現の実測の誤差 `|V_str − V|` を seed 全部・全要素で上から押さえる（二階の `W`・`up`
    /// への要素）。
    #[test]
    fn lin_bound_covers_f32_deviation_for_spectral_norm_sweep() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f64) / f64::from(1u32 << 31)
        };
        let (mut checked, mut tight) = (0usize, 0usize);
        for case in 0..240 {
            let (rows, cols) = [(2, 3), (3, 3), (4, 4), (2, 8), (5, 6), (6, 6), (9, 9)][case % 7];
            let n = rows * cols;
            let amp = [1e-3, 1.0, 1e3][case % 3];
            let alpha = [0.0, 2.0][(case / 3) % 2];
            let up_scale = [1.0, 1e3][(case / 6) % 2];
            let unit = |len: usize, next: &mut dyn FnMut() -> f64| -> Vec<f32> {
                let raw: Vec<f64> = (0..len).map(|_| next() * 2.0 - 1.0).collect();
                let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
                raw.iter().map(|x| (x / norm) as f32).collect()
            };
            let (u, v) = (unit(rows, &mut next), unit(cols, &mut next));
            let p: Vec<f64> = (0..n)
                .map(|k| f64::from(u[k / cols]) * f64::from(v[k % cols]))
                .collect();
            let w: Vec<f32> = (0..n)
                .map(|k| (amp * ((next() * 2.0 - 1.0) + alpha * p[k] * n as f64 * 0.5)) as f32)
                .collect();
            let up: Vec<f32> = (0..n)
                .map(|_| (up_scale * (next() * 2.0 - 1.0) * 3.0) as f32)
                .collect();
            let mx = w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            if mx == 0.0 {
                continue;
            }
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            lin_reset();
            let (Some(fl), Some(fs), Some(fa)) = (
                sn_fwd::<Lin>(&w, &p, &up, s),
                sn_fwd::<f32>(&w, &p, &up, s),
                sn_fwd::<F32Rev>(&w, &p, &up, s),
            ) else {
                continue;
            };
            let mark = lin_mark();
            for j in 0..n {
                lin_truncate(mark);
                let (mut hw_l, mut hu_l) = (vec![Lin::k(0.0); n], vec![Lin::k(0.0); n]);
                let (mut hw_s, mut hu_s) = (vec![0f32; n], vec![0f32; n]);
                let (mut hw_a, mut hu_a) = (vec![F32Rev(0.0); n], vec![F32Rev(0.0); n]);
                sn_rev(&fl, j, &mut hw_l, &mut hu_l);
                sn_rev(&fs, j, &mut hw_s, &mut hu_s);
                sn_rev(&fa, j, &mut hw_a, &mut hu_a);
                let slack = 1.0 + 16.0 * lin_events() as f64 * LIN_U;
                for i in 0..n {
                    for (l, f, a) in [(hw_l[i], hw_s[i], hw_a[i].0), (hu_l[i], hu_s[i], hu_a[i].0)]
                    {
                        if !l.v.is_finite() || !f.is_finite() || !a.is_finite() {
                            continue;
                        }
                        let bound = l.bound() * slack;
                        for actual in [f, a] {
                            let dev = (f64::from(actual) - l.v).abs();
                            assert!(
                                dev <= bound,
                                "case {case} n {n}: 実測の誤差 {dev:e} が上界 {bound:e} を超えた（V = {:e}）",
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
        }
        assert!(checked > 1000, "検査した要素数 {checked}");
        assert!(tight > 0, "上界が実測より桁違いに大きい");
    }

    /// [`Num::settle`] の性質: 束ねた上界は束ねない上界以上（三角不等式）で、値は変わらず、anchors の係数は
    /// 残る。anchors に含まれない事象は 1 つにまとまる。
    #[test]
    fn settle_never_shrinks_bound_and_keeps_anchor_events() {
        lin_reset();
        let xs: Vec<Lin> = (0..12)
            .map(|i| Lin::k(0.1 * f64::from(i + 1)).times(Lin::k(1.0 / 3.0)))
            .collect();
        let anchor = Lin::k(0.7).over(Lin::k(3.0));
        let total = Lin::total(&xs).times(anchor);
        let before = total.bound();
        let events_before = lin_events();
        let settled = total.settle(&[anchor]);
        assert_eq!(settled.v, total.v);
        assert!(
            settled.bound() >= before * (1.0 - 1e-12),
            "束ねた上界 {} が束ねない上界 {before} を下回った",
            settled.bound()
        );
        assert_eq!(lin_events(), events_before + 1);
        let n_terms = settled.terms_of().len();
        let anchor_events = anchor.terms_of().len();
        assert!(n_terms <= anchor_events + 1, "項数 {n_terms}");
        // anchors が空なら全部が 1 つの事象になる。
        assert_eq!(total.settle(&[]).terms_of().len(), 1);
        lin_reset();
    }

    /// 縮約 [`Num::total`] の書き換え（事象の番号で安定に並べて足す）が、逐次の併合と同じ係数を作る。
    #[test]
    fn total_matches_sequential_merge() {
        lin_reset();
        let xs: Vec<Lin> = (0..9)
            .map(|i| Lin::k(0.3 + f64::from(i)).over(Lin::k(7.0)))
            .collect();
        let got = Lin::total(&xs).terms_of();
        let mut acc = LinTerms::new();
        for x in &xs {
            acc = lin_comb(&acc, 1.0, &x.terms_of(), 1.0);
        }
        // 末尾は `total` が足す順序によらない丸めの事象。
        assert_eq!(&got[..got.len() - 1], &acc[..]);
        lin_reset();
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
}
