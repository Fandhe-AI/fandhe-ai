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

/// 一次式の項（事象の番号, 係数）。値ごとの列の中で事象の番号は昇順。
type Term = (u32, f64);

/// [`Lin`] の thread-local の領域。全値の係数列を `coefs` 1 本へ連結して持ち（値 `id` の列は
/// `coefs[start..start + len]`、`(start, len)` は `spans[id]`）、演算の結果は常に `coefs` の末尾へ追記する。
///
/// 旧実装（#4075・§80.2）は値ごとに `Vec` を確保し、`lin_comb` の出力・`negate`／`root` の clone・`total` の
/// 逐次併合の中間列を毎回新しく確保していた（イシュー #4077）。本構造では `coefs` の容量を seed 間・演算間で
/// 使い回し、`lin_truncate` は長さを戻すだけで容量を残す。
/// 不変条件: 値の列は作成時に末尾へ連続して書かれ、後から書き換えない。mark 後に作った値の列は mark 時点の
/// `coefs.len()` より後ろにあり、mark 前の値は mark 後の列を参照しない（切り詰めが安全な理由）。
#[derive(Default)]
struct LinArena {
    coefs: Vec<Term>,
    spans: Vec<(usize, usize)>,
    mus: Vec<f64>,
    /// [`Num::total`] の作業バッファ 2 本（逐次併合の途中の和と併合先を交互に使う。値としては残らない）。
    scratch: Vec<Term>,
    scratch2: Vec<Term>,
}

/// [`lin_mark`] が返す領域の印（値の数・係数の長さ・事象の数）。
#[derive(Clone, Copy)]
pub(super) struct LinMark {
    values: usize,
    coefs: usize,
    mus: usize,
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
/// 計測は `tests::lin_judge_cost_profile`（`#[ignore]`）、実測との照合は決定記録 §80.2（イシュー #4076）・§80.3（#4077）。
/// 値は据え置き（見直しは #4082）。
pub(super) fn lin_group_cost(rest: usize) -> usize {
    6000 * rest * rest
}
/// [`Lin`] を使う判定の走査量の上限。超える（グループ数が多い）テンソルは全グループを従来の `norm_reparam.rs` の `Rf` で判定する
/// （グループの位置で判定方式が変わらないよう、テンソル単位で選ぶ）。
pub(super) const LIN_BUDGET: usize = 1 << 26;

// 計測用の計数器（`#[cfg(test)]` 限定。非テストビルドには存在せず、生成コードを変えない）。
// イシュー #4076 の計測テスト（`tests::lin_judge_cost_profile`）が判定時間の内訳を読む。#4078（疎化）が
// 同じ計数で改善量を測るため残す。バイト数は要素数 × `size_of::<(u32, f64)>()` で換算する。添字は `PROF_*` 定数。
// 意味は #4077 で定義し直した（§80.3）: 旧実装の「係数列ごとの確保」は、領域（`coefs`・`scratch`）の容量が
// 増えた回数・要素数になる。
#[cfg(test)]
thread_local! {
    static LIN_PROF: std::cell::Cell<[u64; 6]> = const { std::cell::Cell::new([0; 6]) };
}
/// 係数列の格納域（`coefs`・`scratch`）の確保・再確保の回数（容量が増えた回数）。
#[cfg(test)]
const PROF_ALLOCS: usize = 0;
/// 確保・再確保で増えた容量の要素数。
#[cfg(test)]
const PROF_ALLOC_ENTRIES: usize = 1;
/// 既存の係数列を新しい確保へ複製した回数（本実装では常に 0）。
#[cfg(test)]
const PROF_CLONES: usize = 2;
/// 複製した係数列の要素数（本実装では常に 0）。
#[cfg(test)]
const PROF_CLONE_ENTRIES: usize = 3;
/// [`LinArena::finish`]（領域へ積まれる値）の回数。
#[cfg(test)]
const PROF_MAKES: usize = 4;
/// 係数列へ書いた要素数（`coefs` の出力と、`total` の作業バッファへの併合結果の合計）。
#[cfg(test)]
const PROF_WRITE_ENTRIES: usize = 5;

#[cfg(test)]
fn prof_add(idx: usize, n: usize) {
    LIN_PROF.with(|c| {
        let mut a = c.get();
        a[idx] += n as u64;
        c.set(a);
    });
}

#[cfg(test)]
fn prof_grow(before: usize, after: usize) {
    if after > before {
        prof_add(PROF_ALLOCS, 1);
        prof_add(PROF_ALLOC_ENTRIES, after - before);
    }
}

/// 領域を空にして解放する（グループの判定の先頭で呼ぶ。判定後に大きな容量を thread-local に残さない）。
pub(super) fn lin_reset() {
    LIN_ARENA.with(|a| *a.borrow_mut() = LinArena::default());
}

/// 領域の現在の大きさ。seed ごとに [`lin_truncate`] で戻すための印（容量は戻さず次の seed で使い回す）。
pub(super) fn lin_mark() -> LinMark {
    LIN_ARENA.with(|a| {
        let a = a.borrow();
        LinMark {
            values: a.spans.len(),
            coefs: a.coefs.len(),
            mus: a.mus.len(),
        }
    })
}

pub(super) fn lin_truncate(mark: LinMark) {
    LIN_ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.spans.truncate(mark.values);
        a.coefs.truncate(mark.coefs);
        a.mus.truncate(mark.mus);
    });
}

/// 事象の総数（二次項の余裕係数の入力）。
fn lin_events() -> usize {
    LIN_ARENA.with(|a| a.borrow().mus.len())
}

/// 事象の番号の昇順に並んだ 2 つの列の線形結合 `ka·a + kb·b` を `out` の先頭から書き、書いた項数を返す
/// （`out` は `a.len() + b.len()` 以上）。a だけの項は `ka·c`、b だけは `kb·c`、両方は `ka·ca + kb·cb`。
/// 係数 0 の項も落とさず、事象の重複除去もしない（旧実装の `lin_comb` と同じ演算式・順序）。
/// `LinArena::comb`（二項演算）と `Num::total`（`ka = kb = 1.0` の逐次併合）が共有する。
fn merge_lin(xa: &[Term], ka: f64, xb: &[Term], kb: f64, out: &mut [Term]) -> usize {
    let (al, bl) = (xa.len(), xb.len());
    let (mut i, mut j, mut o) = (0, 0, 0);
    while i < al || j < bl {
        if j == bl || (i < al && xa[i].0 < xb[j].0) {
            out[o] = (xa[i].0, ka * xa[i].1);
            i += 1;
        } else if i == al || xb[j].0 < xa[i].0 {
            out[o] = (xb[j].0, kb * xb[j].1);
            j += 1;
        } else {
            out[o] = (xa[i].0, ka * xa[i].1 + kb * xb[j].1);
            i += 1;
            j += 1;
        }
        o += 1;
    }
    o
}

impl LinArena {
    /// `coefs` の容量を `extra` 要素分確保する（以降の追記が再確保を起こさないようにする）。
    fn reserve_coefs(&mut self, extra: usize) {
        #[cfg(test)]
        let before = self.coefs.capacity();
        self.coefs.reserve(extra);
        #[cfg(test)]
        prof_grow(before, self.coefs.capacity());
    }

    /// `coefs[start..]` を新しい値の列として確定し、その `id` を返す。
    fn finish(&mut self, start: usize) -> usize {
        let len = self.coefs.len() - start;
        #[cfg(test)]
        {
            prof_add(PROF_MAKES, 1);
            prof_add(PROF_WRITE_ENTRIES, len);
        }
        self.spans.push((start, len));
        self.spans.len() - 1
    }

    /// 事象を 1 つ足してその番号を返す。
    fn push_event(&mut self, mu: f64) -> u32 {
        self.mus.push(mu);
        (self.mus.len() - 1) as u32
    }

    /// 2 つの値の一次式の線形結合 `ka·a + kb·b`（事象の番号の昇順の併合）を `coefs` の末尾へ追記し、開始位置を返す。
    /// 演算式は旧実装の `lin_comb` と一字一句同じ（a だけの項は `ka·c`、b だけは `kb·c`、両方は `ka·ca + kb·cb`。
    /// 係数 0 の項も落とさない）。
    fn comb(&mut self, a: usize, ka: f64, b: usize, kb: f64) -> usize {
        let (a0, al) = self.spans[a];
        let (b0, bl) = self.spans[b];
        let start = self.coefs.len();
        self.reserve_coefs(al + bl + 1);
        // 入力（`start` より前）と出力（`start` 以降）を別々のスライスに分け、書き込みのたびに
        // `Vec` の長さ・ポインタを読み直さないようにする（ホットループの速度のため）。
        self.coefs.resize(start + al + bl, (0, 0.0));
        let (inp, out) = self.coefs.split_at_mut(start);
        let o = merge_lin(&inp[a0..a0 + al], ka, &inp[b0..b0 + bl], kb, out);
        self.coefs.truncate(start + o);
        start
    }

    /// 値 `id` の列の係数を `f` で写して `coefs` の末尾へ追記し、開始位置を返す（clone の `Vec` を作らない）。
    /// 事象の項を後から 1 つ足せるよう 1 要素分の余裕を取る。
    fn map_copy(&mut self, id: usize, f: impl Fn(f64) -> f64) -> usize {
        let (s, l) = self.spans[id];
        let start = self.coefs.len();
        self.reserve_coefs(l + 1);
        self.coefs.extend_from_within(s..s + l);
        for e in &mut self.coefs[start..] {
            e.1 = f(e.1);
        }
        start
    }
}

impl Lin {
    /// 係数列 0 個または 1 個の値を積む。
    fn make(v: f64, term: Option<Term>, t: bool) -> Self {
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let start = a.coefs.len();
            a.reserve_coefs(1);
            if let Some(term) = term {
                a.coefs.push(term);
            }
            a.finish(start)
        });
        Lin { v, id, t }
    }

    /// 二項演算の結果 `v`（係数 `ka`・`kb` の線形結合に、結果の丸めの事象を 1 つ足す）。読み取りと追記を
    /// `borrow_mut` 1 回の中で行い、途中に別の値を作らない。
    fn binary(self, o: Self, v: f64, ka: f64, kb: f64) -> Self {
        let mu = LIN_U * v.abs() + LIN_TINY;
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let start = a.comb(self.id, ka, o.id, kb);
            let ev = a.push_event(mu);
            a.coefs.push((ev, 1.0));
            a.finish(start)
        });
        Lin {
            v,
            id,
            t: self.t || o.t,
        }
    }

    /// 誤差の絶対値の一次の上界 `Σ|cₖ|·μₖ`。
    fn bound(self) -> f64 {
        LIN_ARENA.with(|a| {
            let a = a.borrow();
            let (s, l) = a.spans[self.id];
            a.coefs[s..s + l]
                .iter()
                .map(|&(k, c)| c.abs() * a.mus[k as usize])
                .sum()
        })
    }

    /// 入力の NaN が届いた NaN（taint 付きの NaN）。
    fn input_nan(self) -> bool {
        self.v.is_nan() && self.t
    }
}

impl Num for Lin {
    fn k(x: f64) -> Self {
        // 定数は f32 へ丸めて載る。丸めの誤差は厳密に計算できる（f32 で厳密な値は事象を持たない）。
        let err = if x.is_finite() {
            (f64::from(x as f32) - x).abs()
        } else {
            0.0
        };
        let term = (err > 0.0).then(|| {
            LIN_ARENA.with(|a| {
                let ev = a.borrow_mut().push_event(err);
                (ev, 1.0)
            })
        });
        Lin::make(x, term, x.is_nan())
    }
    fn plus(self, o: Self) -> Self {
        self.binary(o, self.v + o.v, 1.0, 1.0)
    }
    fn minus(self, o: Self) -> Self {
        self.binary(o, self.v - o.v, 1.0, -1.0)
    }
    fn times(self, o: Self) -> Self {
        self.binary(o, self.v * o.v, o.v, self.v)
    }
    fn over(self, o: Self) -> Self {
        let q = self.v / o.v;
        self.binary(o, q, 1.0 / o.v, -q / o.v)
    }
    fn negate(self) -> Self {
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let start = a.map_copy(self.id, |c| -c);
            a.finish(start)
        });
        Lin {
            v: -self.v,
            id,
            t: self.t,
        }
    }
    fn root(self) -> Self {
        let r = self.v.sqrt();
        let mu = LIN_U * r.abs() + LIN_TINY;
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let start = a.map_copy(self.id, |c| c / (2.0 * r));
            let ev = a.push_event(mu);
            a.coefs.push((ev, 1.0));
            a.finish(start)
        });
        Lin {
            v: r,
            id,
            t: self.t,
        }
    }
    fn powe(self, e: f64) -> Self {
        Lin::unbounded(self.v.powf(e), self.t)
    }
    fn pow_grad(self, e: f64) -> Self {
        if e == 0.0 {
            return Lin::make(0.0, None, false);
        }
        Lin::unbounded(e * self.v.powf(e - 1.0), self.t)
    }
    fn root_grad(self) -> Self {
        Lin::k(0.5).over(self)
    }
    /// 入力の列を順に併合する（旧実装の `acc = 1·acc + 1·x` と同じ演算式・順序）。中間の和は領域の作業バッファ 2 本を
    /// 交互に使うので、項ごとの確保をしない。最後に結果を `coefs` の末尾へ 1 回だけ書き、自身の事象を付ける。
    /// 連結して安定ソートする案は、実測で旧実装より遅く（比較コストが併合より高い）、採らなかった（§80.3）。
    fn total(terms: &[Self]) -> Self {
        let v: f64 = terms.iter().map(|x| x.v).sum();
        let t = terms.iter().any(|x| x.t);
        let n = terms.len();
        let abs: f64 = terms.iter().map(|x| x.v.abs()).sum();
        let k = n.saturating_sub(1) as f64 * LIN_U;
        let mu = k / (1.0 - k) * abs + n as f64 * LIN_TINY;
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let a = &mut *a;
            #[cfg(test)]
            let before = (a.scratch.capacity(), a.scratch2.capacity());
            a.scratch.clear();
            for (n, x) in terms.iter().enumerate() {
                let (s, l) = a.spans[x.id];
                let src = &a.coefs[s..s + l];
                if n == 0 {
                    // 空の列との併合は各項を `1.0·c`（厳密）にするだけなので、そのまま写す。
                    a.scratch.extend_from_slice(src);
                } else {
                    a.scratch2.clear();
                    a.scratch2.resize(a.scratch.len() + l, (0, 0.0));
                    let o = merge_lin(&a.scratch, 1.0, src, 1.0, &mut a.scratch2);
                    a.scratch2.truncate(o);
                    std::mem::swap(&mut a.scratch, &mut a.scratch2);
                }
                #[cfg(test)]
                prof_add(PROF_WRITE_ENTRIES, a.scratch.len());
            }
            #[cfg(test)]
            {
                prof_grow(before.0, a.scratch.capacity());
                prof_grow(before.1, a.scratch2.capacity());
            }
            let start = a.coefs.len();
            a.reserve_coefs(a.scratch.len() + 1);
            a.coefs.extend_from_slice(&a.scratch);
            let ev = a.push_event(mu);
            a.coefs.push((ev, 1.0));
            a.finish(start)
        });
        Lin { v, id, t }
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
    fn unbounded(v: f64, t: bool) -> Self {
        let ev = LIN_ARENA.with(|a| a.borrow_mut().push_event(f64::NAN));
        Lin::make(v, Some((ev, 1.0)), t)
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

    // ---- 前後比較用の参照型（イシュー #4077）----
    // 旧実装（#4075・§80.2）の `Lin` を逐語で写した型。値ごとの `Vec`・`lin_comb` の新規確保・`terms_of` の clone・
    // `total` の逐次併合をそのまま持ち、新しい平坦な領域の `Lin` と bit 同一であることを確かめる基準にする。
    // 実装を変えずに残すこと（後続の #4078 などが `Lin` を変えるときの回帰基準にも使える）。

    #[derive(Clone, Copy, Debug)]
    struct LinRef {
        v: f64,
        id: usize,
        t: bool,
    }

    type RefTerms = Vec<(u32, f64)>;

    #[derive(Default)]
    struct RefArena {
        terms: Vec<RefTerms>,
        mus: Vec<f64>,
    }

    thread_local! {
        static LIN_REF_ARENA: std::cell::RefCell<RefArena> = std::cell::RefCell::new(RefArena::default());
    }

    fn ref_reset() {
        LIN_REF_ARENA.with(|a| *a.borrow_mut() = RefArena::default());
    }

    fn ref_mark() -> (usize, usize) {
        LIN_REF_ARENA.with(|a| {
            let a = a.borrow();
            (a.terms.len(), a.mus.len())
        })
    }

    fn ref_truncate(mark: (usize, usize)) {
        LIN_REF_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            a.terms.truncate(mark.0);
            a.mus.truncate(mark.1);
        });
    }

    fn ref_events() -> usize {
        LIN_REF_ARENA.with(|a| a.borrow().mus.len())
    }

    fn ref_comb(a: &[(u32, f64)], ka: f64, b: &[(u32, f64)], kb: f64) -> RefTerms {
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

    impl LinRef {
        fn terms_of(self) -> RefTerms {
            LIN_REF_ARENA.with(|a| a.borrow().terms[self.id].clone())
        }

        fn make(v: f64, terms: RefTerms, t: bool) -> Self {
            let id = LIN_REF_ARENA.with(|a| {
                let mut a = a.borrow_mut();
                a.terms.push(terms);
                a.terms.len() - 1
            });
            LinRef { v, id, t }
        }

        fn rounded(v: f64, mut terms: RefTerms, t: bool) -> Self {
            let mu = LIN_U * v.abs() + LIN_TINY;
            let ev = LIN_REF_ARENA.with(|a| {
                let mut a = a.borrow_mut();
                a.mus.push(mu);
                (a.mus.len() - 1) as u32
            });
            terms.push((ev, 1.0));
            LinRef::make(v, terms, t)
        }

        fn bound(self) -> f64 {
            LIN_REF_ARENA.with(|a| {
                let a = a.borrow();
                a.terms[self.id]
                    .iter()
                    .map(|&(k, c)| c.abs() * a.mus[k as usize])
                    .sum()
            })
        }

        fn input_nan(self) -> bool {
            self.v.is_nan() && self.t
        }

        fn lin2(self, ka: f64, o: Self, kb: f64) -> RefTerms {
            LIN_REF_ARENA.with(|a| {
                let a = a.borrow();
                ref_comb(&a.terms[self.id], ka, &a.terms[o.id], kb)
            })
        }
    }

    impl Num for LinRef {
        fn k(x: f64) -> Self {
            // 定数は f32 へ丸めて載る。丸めの誤差は厳密に計算できる（f32 で厳密な値は事象を持たない）。
            let mut terms = RefTerms::new();
            let err = if x.is_finite() {
                (f64::from(x as f32) - x).abs()
            } else {
                0.0
            };
            if err > 0.0 {
                let ev = LIN_REF_ARENA.with(|a| {
                    let mut a = a.borrow_mut();
                    a.mus.push(err);
                    (a.mus.len() - 1) as u32
                });
                terms.push((ev, 1.0));
            }
            LinRef::make(x, terms, x.is_nan())
        }
        fn plus(self, o: Self) -> Self {
            LinRef::rounded(self.v + o.v, self.lin2(1.0, o, 1.0), self.t || o.t)
        }
        fn minus(self, o: Self) -> Self {
            LinRef::rounded(self.v - o.v, self.lin2(1.0, o, -1.0), self.t || o.t)
        }
        fn times(self, o: Self) -> Self {
            LinRef::rounded(self.v * o.v, self.lin2(o.v, o, self.v), self.t || o.t)
        }
        fn over(self, o: Self) -> Self {
            let q = self.v / o.v;
            LinRef::rounded(q, self.lin2(1.0 / o.v, o, -q / o.v), self.t || o.t)
        }
        fn negate(self) -> Self {
            let terms = self.terms_of().into_iter().map(|(k, c)| (k, -c)).collect();
            LinRef::make(-self.v, terms, self.t)
        }
        fn root(self) -> Self {
            let r = self.v.sqrt();
            let terms = self
                .terms_of()
                .into_iter()
                .map(|(k, c)| (k, c / (2.0 * r)))
                .collect();
            LinRef::rounded(r, terms, self.t)
        }
        fn powe(self, e: f64) -> Self {
            LinRef::make(self.v.powf(e), vec![], self.t).unbounded()
        }
        fn pow_grad(self, e: f64) -> Self {
            if e == 0.0 {
                return LinRef::make(0.0, vec![], false);
            }
            LinRef::make(e * self.v.powf(e - 1.0), vec![], self.t).unbounded()
        }
        fn root_grad(self) -> Self {
            LinRef::k(0.5).over(self)
        }
        fn total(terms: &[Self]) -> Self {
            let v: f64 = terms.iter().map(|x| x.v).sum();
            let t = terms.iter().any(|x| x.t);
            let n = terms.len();
            let mut acc = RefTerms::new();
            for x in terms {
                acc = LIN_REF_ARENA.with(|a| ref_comb(&acc, 1.0, &a.borrow().terms[x.id], 1.0));
            }
            let abs: f64 = terms.iter().map(|x| x.v.abs()).sum();
            let k = n.saturating_sub(1) as f64 * LIN_U;
            let mu = k / (1.0 - k) * abs + n as f64 * LIN_TINY;
            let ev = LIN_REF_ARENA.with(|a| {
                let mut a = a.borrow_mut();
                a.mus.push(mu);
                (a.mus.len() - 1) as u32
            });
            acc.push((ev, 1.0));
            LinRef::make(v, acc, t)
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

    impl LinRef {
        fn unbounded(self) -> Self {
            let ev = LIN_REF_ARENA.with(|a| {
                let mut a = a.borrow_mut();
                a.mus.push(f64::NAN);
                (a.mus.len() - 1) as u32
            });
            LinRef::make(self.v, vec![(ev, 1.0)], self.t)
        }
    }

    /// NaN を 1 つの bit 列へ正規化した bit 表現。2 つの NaN の加算で返る NaN の符号・payload は被演算子の順に
    /// 依存し、コンパイラが可換演算の順を入れ替えうる（新旧で NaN の符号だけ違う例を実測）。NaN 以外は厳密に bit 同一で比べる。
    fn canon_bits(x: f64) -> u64 {
        if x.is_nan() {
            f64::NAN.to_bits()
        } else {
            x.to_bits()
        }
    }

    /// 前後比較用の 1 値の写し（イシュー #4077）。値・taint・係数列・上界を bit で比べる（NaN は [`canon_bits`]）。
    #[derive(Debug, PartialEq, Eq)]
    struct Snap {
        v: u64,
        t: bool,
        terms: Vec<(u32, u64)>,
        bound: u64,
    }

    fn snap_new(x: Lin) -> Snap {
        let terms = LIN_ARENA.with(|a| {
            let a = a.borrow();
            let (s, l) = a.spans[x.id];
            a.coefs[s..s + l]
                .iter()
                .map(|&(k, c)| (k, canon_bits(c)))
                .collect()
        });
        Snap {
            v: canon_bits(x.v),
            t: x.t,
            terms,
            bound: canon_bits(x.bound()),
        }
    }

    fn snap_ref(x: LinRef) -> Snap {
        let terms = LIN_REF_ARENA.with(|a| {
            a.borrow().terms[x.id]
                .iter()
                .map(|&(k, c)| (k, canon_bits(c)))
                .collect()
        });
        Snap {
            v: canon_bits(x.v),
            t: x.t,
            terms,
            bound: canon_bits(x.bound()),
        }
    }

    /// `entry_check_lin` の参照型版（旧実装の式そのまま）。受理なら true。
    fn entry_ok_ref(r: LinRef, s: f32) -> bool {
        if s.is_nan() && r.input_nan() {
            return true;
        }
        let slack = 16.0 * ref_events() as f64 * LIN_U;
        let err = (f64::from(s) - r.v).abs() + r.bound() * (1.0 + slack);
        slack < 0.01
            && r.v.is_finite()
            && err.is_finite()
            && (err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * r.v.abs())
    }

    /// 前後比較 1: WeightNorm の掃引（soundness テストと同じ乱数・振幅）で、新旧の型が全 seed・全出力要素で
    /// bit 同一（値・taint・係数列・上界・事象の数・`entry_check_lin` の受理／拒否）になる。
    #[test]
    fn lin_matches_reference_for_weight_norm_sweep() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f64) / f64::from(1u32 << 31)
        };
        let mut compared = 0usize;
        for case in 0..200 {
            let rest = match case {
                c if c % 100 == 99 => 32,
                c if c % 40 == 39 => 16,
                c => 2 + c % 7,
            };
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
            ref_reset();
            let fn_ = wn_fwd::<Lin>(&v, &up, g, s);
            let fr = wn_fwd::<LinRef>(&v, &up, g, s);
            let fs = wn_fwd::<f32>(&v, &up, g, s);
            let (Some(fn_), Some(fr), Some(fs)) = (fn_, fr, fs) else {
                continue;
            };
            assert_eq!(
                lin_events(),
                ref_events(),
                "case {case}: forward の事象の数"
            );
            let (mark_n, mark_r) = (lin_mark(), ref_mark());
            for seed in (0..rest).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
                lin_truncate(mark_n);
                ref_truncate(mark_r);
                let (mut hv_n, mut hu_n) = (vec![Lin::k(0.0); rest], vec![Lin::k(0.0); rest]);
                let (mut hv_r, mut hu_r) = (vec![LinRef::k(0.0); rest], vec![LinRef::k(0.0); rest]);
                let (mut hv_s, mut hu_s) = (vec![0f32; rest], vec![0f32; rest]);
                let gn = wn_rev(&fn_, seed, &mut hv_n, &mut hu_n);
                let gr = wn_rev(&fr, seed, &mut hv_r, &mut hu_r);
                let gs = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
                assert_eq!(lin_events(), ref_events(), "case {case}: seed の事象の数");
                let mut pairs: Vec<(Lin, LinRef, f32)> = (0..rest)
                    .flat_map(|i| [(hv_n[i], hv_r[i], hv_s[i]), (hu_n[i], hu_r[i], hu_s[i])])
                    .collect();
                assert_eq!(gn.is_some(), gr.is_some());
                if let (Some(gn), Some(gr), Some(gs)) = (gn, gr, gs) {
                    pairs.push((gn, gr, gs));
                }
                for (n, r, f) in pairs {
                    assert_eq!(snap_new(n), snap_ref(r), "case {case} rest {rest}");
                    let ok_n = entry_check_lin(n, f, "p", bench_reject).is_ok();
                    assert_eq!(ok_n, entry_ok_ref(r, f), "case {case}: 受理／拒否");
                    compared += 1;
                }
            }
        }
        assert!(compared > 1000, "比較した要素数 {compared}");
    }

    /// 前後比較 2: 演算単位。決定的な乱数で `k`（f32 で厳密／不厳密・NaN・±inf・極端な値）と全演算（`total` は
    /// 0 項・1 項・重複・多項）を組み合わせた演算列を新旧の型で同じ順に実行し、作った値すべてと事象の数が
    /// bit 同一であることを確かめる。途中の `lin_mark`／`lin_truncate` 後に作る値も含める。
    #[test]
    fn lin_matches_reference_for_random_op_programs() {
        const KS: [f64; 14] = [
            0.0,
            -0.0,
            0.5,
            1.0,
            -2.75,
            0.1,
            3.0e-3,
            1.0e300,
            1.0e-320,
            16_777_217.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -7.3,
        ];
        fn run<T: Num, M: Copy>(
            prog: &[(u8, usize, usize, usize)],
            mark: fn() -> M,
            trunc: fn(M),
            snap: fn(T) -> Snap,
            events: fn() -> usize,
        ) -> Vec<(Snap, usize)> {
            let mut pool: Vec<T> = KS.iter().map(|&x| T::k(x)).collect();
            let mut log: Vec<(Snap, usize)> = pool.iter().map(|&x| (snap(x), events())).collect();
            let mut marks: Vec<(M, usize)> = Vec::new();
            for &(op, a, b, c) in prog {
                let (x, y) = (pool[a % pool.len()], pool[b % pool.len()]);
                let made = match op {
                    0 => T::k(KS[a % KS.len()] * (1.0 + (b % 5) as f64 * 0.37)),
                    1 => x.plus(y),
                    2 => x.minus(y),
                    3 => x.times(y),
                    4 => x.over(y),
                    5 => x.negate(),
                    6 => x.root(),
                    7 => x.root_grad(),
                    8 => {
                        let n = c % 7;
                        let idx: Vec<T> = (0..n)
                            .map(|i| pool[(a + i * (b + 1)) % pool.len()])
                            .collect();
                        T::total(&idx)
                    }
                    9 => x.powe(0.5 + (b % 3) as f64),
                    10 => x.pow_grad([0.0, 2.0, 1.5][b % 3]),
                    11 => {
                        marks.push((mark(), pool.len()));
                        continue;
                    }
                    _ => {
                        if let Some((m, len)) = marks.pop() {
                            trunc(m);
                            pool.truncate(len);
                        }
                        continue;
                    }
                };
                log.push((snap(made), events()));
                pool.push(made);
            }
            log
        }
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        let mut total_values = 0usize;
        for _ in 0..40 {
            let prog: Vec<(u8, usize, usize, usize)> = (0..300)
                .map(|_| {
                    let op = (rng.next() * 13.0) as u8;
                    let pick = |r: &mut Lcg| (r.next() * 1e6) as usize;
                    (op, pick(&mut rng), pick(&mut rng), pick(&mut rng))
                })
                .collect();
            lin_reset();
            ref_reset();
            let new = run::<Lin, LinMark>(&prog, lin_mark, lin_truncate, snap_new, lin_events);
            let old =
                run::<LinRef, (usize, usize)>(&prog, ref_mark, ref_truncate, snap_ref, ref_events);
            assert_eq!(new.len(), old.len());
            for (i, (n, o)) in new.iter().zip(&old).enumerate() {
                assert_eq!(n, o, "演算列の {i} 番目の値");
            }
            total_values += new.len();
        }
        assert!(total_values > 5000, "比較した値の数 {total_values}");
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
            LIN_PROF.with(|c| c.set([0; 6]));
            let sc = scan_lin(v, up, *g, *s);
            let p = LIN_PROF.with(|c| c.get());
            let n = sc.n_dv.max(sc.n_dg);
            let probe_matches = n_probe(v, up, *g, *s) == n;
            let verdict = wn_group_judge(v, up, *g, *s, wn_coeff(rest), true).is_ok();
            eprintln!(
                "LINPROF kind=width rest={rest} groups={groups} reps={} median_ns_per_group={med:.0} exp_vs_prev={exp:.2} \
                 N_dv={} N_dg={} slack={:.5} allocs={} alloc_entries={} alloc_bytes={} clones={} clone_entries={} clone_bytes={} \
                 makes={} write_entries={} entries_accepted={} entries_rejected={} probe_matches_scan={} prod_verdict={} raw_ns={:?}",
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
                p[PROF_WRITE_ENTRIES],
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
