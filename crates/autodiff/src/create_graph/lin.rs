//! 丸めの一次伝播を追う数値型 [`Lin`] と、構造の演算列を再現する数値型の共通 trait [`Num`]
//! （イシュー #4075・設計記録 `docs/autodiff-higher-order-grad-decision.md` §77・§79・§80）。
//!
//! 役割: `backward_create_graph` の総括判定（二階の構造を積む前に、f32 の丸め・相殺・underflow・overflow で
//! REQ-2 の統一複合判定を外れうる入力を拒否する検査）で、f32 の構造と f64 の参照を同じ式で評価するための型。
//! `Lin` は丸めの事象ごとに誤差の一次式を持ち、同じ中間値を共有する式の誤差の打ち消しを上界へ畳む。
//!
//! 呼び出し元: `norm_reparam.rs` の WeightNorm の総括判定（`wn_group_judge`）と SpectralNorm の総括判定
//! （`sn_judge`。#4080。全要素に依存する値の係数列は [`Num::settle`] で束ねる）、同ファイルの
//! `T: Num` 総称関数群、`loss_class.rs` の Cosine の相殺判定（#4090 で forward を `f32`・`f64` の総称関数にし、
//! #4091 で従来の見積りが拒否した要素を `Lin` の上界で救う判定へ配線。係数列は [`Num::settle`] で束ねる）。
//! LRN（#4073）や損失のクラス軸系（#4071）が流用する予定のため、
//! norm 専用ではない共通モジュールとして切り出した（挙動不変の移動）。norm 専用の `Rf`・`Dm` は
//! `norm_reparam.rs` に残す。
//!
//! 契約: IEEE 754 の正しい丸めを仮定する（GPU の実機は未実測）。ただし [`Num::powf_bounded`]・[`Num::exp_bounded`] だけは libm の `powf`・`exp` が
//! 正しい丸めではないため、所有者決定 #4083 の仮定（[`LIBM_F32_MAX_ULP`]・正規化数の範囲で 2 ulp 以内。設計値で実測値ではない。
//! GPU・aarch64 Linux・macOS は未実測で、この判定は GPU の値を保証しない）を追加で使う（#4084・#4092・設計記録 §81.10・§82.9）。拒否時の文面は `judge.rs` と同じく
//! `reject: fn(&str) -> AutodiffError`（[`super::judge::Reject`]）で呼び出し元へ委ね、tolerance 定数
//! （`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）は変更しない。

use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

use super::judge::Reject;
use crate::error::AutodiffError;

/// 構造の演算列を再現する数値型。`f32` は構造そのもの（`V_str`）、`norm_reparam.rs` の `Rf` と素の `f64`（`loss_class.rs` の Cosine。#4090）は f64 の参照（`V`・`M`）。
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
    /// 上界つきの `powf`（イシュー #4084・設計記録 §81.10）。`e` は参照の指数、`e_str` は構造の f32 が実際に使う指数
    /// （Triplet の `1.0 / p32`・`p32 - 1.0` のように `e as f32` と一致しない場合がある）。既定は [`Num::powe`] へ委ねる
    /// （`f64`・`norm_reparam.rs` の `Rf`・`Dm` は参照値・依存の追跡で `e_str` を使わない。`Dm` は `powe` 内で
    /// `e as f32 == 0` を見るため、`e_str != e as f32` の呼び出しでは食い違いうる。配線側〈#4088〉で確認すること）。
    /// `f32` は `powf(e_str)`、[`Lin`] だけが libm の精度の仮定（[`LIBM_F32_MAX_ULP`]）を使った上界を持つ。
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "#4088・#4095 が配線するまで本番から呼ばれない")
    )]
    fn powf_bounded(self, e: f64, _e_str: f32) -> Self {
        self.powe(e)
    }
    /// 上界つきの `PowScalar` の VJP 係数 `e·x^(e−1)`（[`Num::pow_grad`] の上界つき版。`e == 0` は 0）。
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "#4088・#4095 が配線するまで本番から呼ばれない")
    )]
    fn pow_grad_bounded(self, e: f64) -> Self {
        self.pow_grad(e)
    }
    /// 上界つきの `exp`（イシュー #4092・設計記録 §82.9）。`f32`・`f64` は `exp()`、[`Lin`] だけが libm の精度の仮定
    /// （[`LIBM_F32_MAX_ULP`]）を使った上界を持つ。`norm_reparam.rs` の `Rf` は上界なし（常に拒否）、`Dm` は依存を保つ。
    /// VJP の係数は出力 `y` そのものなので専用の勾配メソッドはない。
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "#4093・#4094 が配線するまで本番から呼ばれない")
    )]
    fn exp_bounded(self) -> Self;
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
    fn powf_bounded(self, _e: f64, e_str: f32) -> Self {
        self.powf(e_str)
    }
    fn exp_bounded(self) -> Self {
        self.exp()
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

/// f64 の素の演算（参照値側）。`loss_class.rs::cos_fwd` が f64 の参照値と f32 の構造の再現を同じ総称関数で
/// 組むための実装で、各演算は f64 の演算そのもの（丸めの追跡はしない）。`total` は従来の参照実装が使っていた
/// `Iterator::sum::<f64>()` と同じ演算列にするため、既定の逐次和ではなく `sum` で上書きする（イシュー #4090）。
impl Num for f64 {
    fn k(x: f64) -> Self {
        x
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
        f64::sqrt(self)
    }
    fn total(terms: &[Self]) -> Self {
        terms.iter().copied().sum::<f64>()
    }
    fn powe(self, e: f64) -> Self {
        self.powf(e)
    }
    fn pow_grad(self, e: f64) -> Self {
        if e == 0.0 {
            0.0
        } else {
            e * self.powf(e - 1.0)
        }
    }
    fn exp_bounded(self) -> Self {
        self.exp()
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
/// 上界が要る呼び出しは `powf_bounded`・`pow_grad_bounded` を使う（#4084。`powe`・`pow_grad` は常に拒否のまま）。
/// `exp` は [`Num::exp_bounded`] が libm の精度の仮定つきで上界を持つ（#4092・設計記録 §82.9）。
///
/// **疎モード**（[`lin_reset_sparse`]。WeightNorm の総括判定だけが使う。イシュー #4078・決定記録 §80.4）:
/// グループの幅が広いと `n_s` などのハブ値が全要素の事象を持ち、下流の係数列がほぼ密になって併合の書き込み量が幅の
/// 3 乗に近く伸びる。疎モードは演算の結果の列のうち上界に比べて小さい項を、係数 1 の独立な残余事象 1 つへまとめる
/// （[`LinArena::pool_small_terms`]）。三角不等式により上界は広がる側にしか動かず、判定は緩まない。既定の厳密モード
/// （[`lin_reset`]。SpectralNorm など）は従来どおり一切まとめない。
#[derive(Clone, Copy, Debug)]
pub(super) struct Lin {
    pub(super) v: f64,
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
    /// 事象ごとの「残余事象か」の印（`mus` と同じ長さ。[`LinArena::pool_small_terms`] が足した事象だけ `true`）。
    residual: Vec<bool>,
    /// 疎モード（[`lin_reset_sparse`]）か。`false`（既定・[`lin_reset`]）なら演算の結果を一切まとめない。
    sparse: bool,
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

/// CPU の `f32::powf`（libm）が、結果が正規化数の範囲で `|f32 − 厳密値| ≤ 2 ulp` に収まるという仮定の `ulp` 数
/// （所有者決定 #4083・2026-11-07。**設計値であって実測値ではない**。実測が超えたら値を上げず所有者へ報告する。
/// GPU・aarch64 Linux・macOS は未実測）。`exp` の規則（#4092）も共有できるよう `powf` 専用の名前にしない。
pub(super) const LIBM_F32_MAX_ULP: f64 = 2.0;
/// 非正規化数の間隔 `2^-149`（libm の絶対誤差の項は [`LIBM_F32_MAX_ULP`] 倍）。
pub(super) const LIN_ULP_SUBNORMAL: f64 = f32::from_bits(1) as f64;
/// [`Num::powf_bounded`] が受理する指数の増幅 `Z` の上限（本計画の設計値 `2^-10`。#4083 に値の記載はなく、
/// 拒否する側にだけ働く。二次の剰余を `Z` の二乗で覆うための一次近似の妥当域）。
pub(super) const LIN_POWF_Z_MAX: f64 = 1.0 / 1024.0;
/// [`Num::exp_bounded`] が受理する入力の誤差の絶対上界 `b` の上限（本計画の設計値 `1/2`。#4083 に値の記載はなく、
/// 拒否する側にだけ働く。`|δ| ≤ 1/2` なら `|e^δ − 1 − δ| ≤ δ²` に余裕があり、一次の上界を余裕 `(1+16Nu)` の分だけ
/// 超える真の `δ` も覆える。設計記録 §82.9）。
pub(super) const LIN_EXP_DELTA_MAX: f64 = 0.5;
/// f32 の丸め単位 `u = 2^-24`。
pub(super) const LIN_U: f64 = 1.0 / 16_777_216.0;
/// 非正規数の領域の絶対丸め誤差の上界 `2^-150`。
const LIN_TINY: f64 = 7.006_492_321_624_085e-46;
/// 疎モード（[`lin_reset_sparse`]）で小さな項を残余事象へ落とす閾値の比 `θ`（項 `|c|·μ` が `θ·B/L` 未満なら落とす。
/// `B` は列の上界、`L` は列の長さ）。落とす量の合計は `θ·B` 以下。値の選定（§79.2 で受理に戻した入力の余裕との
/// 兼ね合い）は決定記録 §80.4。
const LIN_FOLD_FRAC: f64 = 0.25;

#[cfg(test)]
thread_local! {
    /// テストが `θ` を差し替える（選定の掃引用）。`None` なら [`LIN_FOLD_FRAC`]。
    static FOLD_FRAC_OVERRIDE: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn fold_frac() -> f64 {
    FOLD_FRAC_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(LIN_FOLD_FRAC)
}

#[cfg(not(test))]
const fn fold_frac() -> f64 {
    LIN_FOLD_FRAC
}

/// [`Lin`] を使うグループの幅の上限。事象の数 `N` が幅に比例して増え、`entry_check_lin` の余裕係数の条件
/// （`16·N·u < 0.01`。`norm_reparam.rs::wn_use_lin`）と判定時間で決まる。#4082 で 8 から 128 へ広げた
/// （疎モードの `N` は幅 128 で約 4400、slack 0.0042。測った最大幅は 256 だが、`lin_group_cost` の余裕が
/// 薄くなるため 128 で止める。実測は決定記録 §80.5）。
pub(super) const LIN_REST_MAX: usize = 128;
/// [`Lin`] で判定するグループ 1 つの走査量の見積り（`Rf` の 1 反復相当。`150·rest² + 4000·rest`）。
/// 疎モードの本番判定を幅 2〜256 で実測した反復相当（`tests::lin_wide_apply_profile`。同じ実行内の `Rf` の
/// 1 反復時間で換算）の全点を覆い、見積り／実測は幅 2〜256 で 1.2 倍以上。測っていない幅へは外挿しない
/// （[`LIN_REST_MAX`] が上限）。実測と各点の比は決定記録 §80.5（イシュー #4082）。
pub(super) fn lin_group_cost(rest: usize) -> usize {
    150 * rest * rest + 4000 * rest
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
    static LIN_PROF: std::cell::Cell<[u64; 8]> = const { std::cell::Cell::new([0; 8]) };
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

/// 疎モードで小さな項をまとめた回数（#4078）。
#[cfg(test)]
const PROF_FOLDS: usize = 6;
/// まとめて落とした項の数の合計（#4078）。
#[cfg(test)]
const PROF_FOLD_DROPS: usize = 7;

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

/// [`lin_reset`] と同じく領域を空にしたうえで、疎モードへ入る（イシュー #4078・決定記録 §80.4）。疎モードでは、
/// 演算の結果の係数列のうち「上界 `B = Σ|c|·μ` に比べて小さい項」を 1 つの独立な残余事象へまとめる
/// （[`LinArena::pool_small_terms`]）。呼び出し元は WeightNorm の総括判定（`norm_reparam.rs::wn_group_judge`）だけで、
/// SpectralNorm（`sn_judge`）と他の呼び出しは従来どおりの厳密モード（`lin_reset`）を使う。モードは領域に属し、
/// `lin_mark`／`lin_truncate` では変わらず、次の reset で毎回設定し直される。
pub(super) fn lin_reset_sparse() {
    LIN_ARENA.with(|a| {
        *a.borrow_mut() = LinArena {
            sparse: true,
            ..LinArena::default()
        }
    });
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
        a.residual.truncate(mark.mus);
    });
}

/// 事象の総数（二次項の余裕係数の入力）。
pub(super) fn lin_events() -> usize {
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
        self.residual.push(false);
        (self.mus.len() - 1) as u32
    }

    /// 疎モードのとき、`coefs[start..]`（いま書いたばかりの値の列）の小さな項と既存の残余事象を、1 つの残余事象へまとめる。
    ///
    /// 呼び出し元は `binary`・`root`・`total`（結果の列が伸びる演算）で、`finish` の直前に呼ぶ。
    /// - 落とす項: `|c|·μ < θ·B / L`（`B` は列の上界、`L` は列の長さ、`θ` は [`fold_frac`]）の項と、既存の残余事象の項。
    ///   落とす量の合計は高々 `θ·B`。残余事象は係数 1・`μ = Σ_落とす |c|·μ`（+ 総和の丸めの余裕）の独立な事象で、
    ///   三角不等式により上界は広がる側にしか動かない（相殺を失うだけ）。
    /// - `B` が有限でない列（NaN／inf の事象〈`unbounded`〉や `0·NaN`）は触らない。NaN の上界を有限の残余へ畳んで
    ///   「常に拒否」を受理に変えないため。`B` が有限なら `c = 0` の項の `μ` も有限で、落としても寄与は厳密に 0。
    /// - 落とす項（`c ≠ 0`）が 2 個未満なら列を変えない（1 個を落として 1 個足しても短くならない）。
    /// - 残余事象の番号は最大なので、列の事象の番号の昇順は保たれる。確保は増やさず、`coefs` をその場で詰める。
    ///
    /// 残余の `μ` には、`B` の和の順序の差（まとめる前は index 順の一括和、後は「残す項 + 残余」）を覆う
    /// `2·L·ε·B` を足し、まとめた後の [`Lin::bound`] がまとめる前以上であることを丸めの順序まで含めて保つ。
    fn pool_small_terms(&mut self, start: usize) {
        if !self.sparse {
            return;
        }
        let len = self.coefs.len() - start;
        let mus = &self.mus;
        let bound: f64 = self.coefs[start..]
            .iter()
            .map(|&(k, c)| c.abs() * mus[k as usize])
            .sum();
        if !bound.is_finite() || bound == 0.0 || len < 3 {
            return;
        }
        let thr = fold_frac() * bound / len as f64;
        let (mut drops, mut dropped) = (0usize, 0.0f64);
        for &(k, c) in &self.coefs[start..] {
            let w = c.abs() * mus[k as usize];
            if self.residual[k as usize] || w < thr {
                if c != 0.0 {
                    drops += 1;
                }
                dropped += w;
            }
        }
        if drops < 2 {
            return;
        }
        let mut w = start;
        for i in start..start + len {
            let (k, c) = self.coefs[i];
            let x = c.abs() * self.mus[k as usize];
            if !(self.residual[k as usize] || x < thr) {
                self.coefs[w] = (k, c);
                w += 1;
            }
        }
        #[cfg(test)]
        {
            prof_add(PROF_FOLDS, 1);
            prof_add(PROF_FOLD_DROPS, len - (w - start));
        }
        self.coefs.truncate(w);
        let ev = self.push_event(dropped + 2.0 * len as f64 * f64::EPSILON * bound);
        self.residual[ev as usize] = true;
        self.coefs.push((ev, 1.0));
        // まとめた時点の性質(受け入れ条件 1)を、疎モードで走る全テストの全まとめ箇所で検査する。
        #[cfg(test)]
        {
            let after: f64 = self.coefs[start..]
                .iter()
                .map(|&(k, c)| c.abs() * self.mus[k as usize])
                .sum();
            assert!(
                after >= bound,
                "まとめた後の上界 {after:e} がまとめる前 {bound:e} を下回った"
            );
        }
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

#[cfg(test)]
impl Lin {
    /// 係数列の複製（テスト用）。
    fn terms_of(self) -> Vec<Term> {
        LIN_ARENA.with(|a| {
            let a = a.borrow();
            let (s, l) = a.spans[self.id];
            a.coefs[s..s + l].to_vec()
        })
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
            a.pool_small_terms(start);
            a.finish(start)
        });
        Lin {
            v,
            id,
            t: self.t || o.t,
        }
    }

    /// 誤差の絶対値の一次の上界 `Σ|cₖ|·μₖ`。
    pub(super) fn bound(self) -> f64 {
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
    pub(super) fn input_nan(self) -> bool {
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
            a.pool_small_terms(start);
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
    /// `x^e` の上界つき版（設計記録 §81.10）。底の誤差 `|δ| ≤ ρ = 2·bound/x`、指数の差 `Δe = e_str − e` を、
    /// 底の事象への係数 `e·y/x`（共有した丸めの打ち消しを保つ）と、二次の剰余・libm の誤差・指数の差を覆う
    /// 独立な事象 1 つへ畳む。底が 0 以下・非有限、`ρ > 1/2`、`Z` が [`LIN_POWF_Z_MAX`] 超、結果が f32 の overflow に
    /// 近い入力は上界なし（拒否）。libm の精度は [`LIBM_F32_MAX_ULP`] の仮定による。
    fn powf_bounded(self, e: f64, e_str: f32) -> Self {
        let (x, de) = (self.v, f64::from(e_str) - e);
        let y = x.powf(e);
        let reject = || Lin::unbounded(y, self.t);
        if !(0.0 < x && le(x, f64::from(f32::MAX)) && e.is_finite() && e_str.is_finite()) {
            return reject();
        }
        let rho = 2.0 * self.bound() / x;
        if !le(rho, 0.5) {
            return reject();
        }
        let (ae, ade, l, ya) = (e.abs(), de.abs(), x.ln().abs(), y.abs());
        let z = 2.0 * (ae + ade) * rho + ade * l;
        let top = ya * (1.0 + 2.0 * z) * (1.0 + 4.0 * LIN_U);
        if !le(z, LIN_POWF_Z_MAX) || !le(top, f64::from(f32::MAX)) || top == f64::from(f32::MAX) {
            return reject();
        }
        let k = e * y / x;
        let mu = LIBM_F32_MAX_ULP * 2.0 * LIN_U * ya * (1.0 + 2.0 * z)
            + LIBM_F32_MAX_ULP * LIN_ULP_SUBNORMAL
            + ya * (ade * l + z * z + ae * rho * rho + 2.0 * ade * rho);
        if !mu.is_finite() {
            return reject();
        }
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let start = a.map_copy(self.id, |c| c * k);
            let ev = a.push_event(mu);
            a.coefs.push((ev, 1.0));
            a.pool_small_terms(start);
            a.finish(start)
        });
        Lin {
            v: y,
            id,
            t: self.t,
        }
    }
    /// `exp(x)` の上界つき版（設計記録 §82.9）。入力の誤差の絶対上界 `b` に対し、`x̂ = x + δ`（`|δ| ≤ b`）なら
    /// `exp(x̂) = y·e^δ`。一次の項 `y·δ` は x の既存事象への係数 `y`（共有した丸めの打ち消しを保つ）、二次の剰余 `y·b²`・
    /// libm の誤差（[`LIBM_F32_MAX_ULP`] ulp）・非正規化数の絶対項は独立な事象 1 つへ畳む。x が非有限、`b` が
    /// [`LIN_EXP_DELTA_MAX`] 超、結果が f32 の overflow に近い入力は上界なし（拒否。`-inf` も安全側で拒否）。
    fn exp_bounded(self) -> Self {
        let y = self.v.exp();
        let reject = || Lin::unbounded(y, self.t);
        if !self.v.is_finite() {
            return reject();
        }
        let b = self.bound();
        let ya = y.abs();
        let top = ya * (1.0 + 2.0 * b) * (1.0 + 4.0 * LIN_U);
        if !le(b, LIN_EXP_DELTA_MAX) || !le(top, f64::from(f32::MAX)) || top == f64::from(f32::MAX)
        {
            return reject();
        }
        let mu = LIBM_F32_MAX_ULP * 2.0 * LIN_U * ya * (1.0 + 2.0 * b)
            + LIBM_F32_MAX_ULP * LIN_ULP_SUBNORMAL
            + ya * b * b;
        if !mu.is_finite() {
            return reject();
        }
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let start = a.map_copy(self.id, |c| c * y);
            let ev = a.push_event(mu);
            a.coefs.push((ev, 1.0));
            a.pool_small_terms(start);
            a.finish(start)
        });
        Lin {
            v: y,
            id,
            t: self.t,
        }
    }
    /// `e·x^(e−1)` の上界つき版。f32 の `e * x.powf(e - 1.0)` と同じ演算列（定数 `e` を `Lin::k`、冪を [`Lin::powf_bounded`]）。
    fn pow_grad_bounded(self, e: f64) -> Self {
        if e == 0.0 {
            return Lin::make(0.0, None, false);
        }
        let es = e as f32;
        if es == 0.0 {
            return Lin::unbounded(e * self.v.powf(e - 1.0), self.t);
        }
        Lin::k(e).times(self.powf_bounded(e - 1.0, es - 1.0))
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
            a.pool_small_terms(start);
            a.finish(start)
        });
        Lin { v, id, t }
    }
    fn settle(self, anchors: &[Self]) -> Self {
        let id = LIN_ARENA.with(|a| {
            let mut a = a.borrow_mut();
            let a = &mut *a;
            // 残す事象（anchors の係数列に現れるもの）。
            let mut keep: Vec<u32> = anchors
                .iter()
                .flat_map(|x| {
                    let (s, l) = a.spans[x.id];
                    a.coefs[s..s + l].iter().map(|&(k, _)| k)
                })
                .collect();
            keep.sort_unstable();
            keep.dedup();
            let (s, l) = a.spans[self.id];
            let start = a.coefs.len();
            a.reserve_coefs(l + 1);
            let mut mu = 0.0f64;
            for i in s..s + l {
                let (k, c) = a.coefs[i];
                if keep.binary_search(&k).is_ok() {
                    a.coefs.push((k, c));
                } else {
                    mu += c.abs() * a.mus[k as usize];
                }
            }
            if mu != 0.0 {
                let ev = a.push_event(mu);
                a.coefs.push((ev, 1.0));
            }
            a.finish(start)
        });
        Lin {
            v: self.v,
            id,
            t: self.t,
        }
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

/// `a <= b`（NaN は偽）。拒否条件を否定形で書くための補助（NaN を必ず拒否側へ倒す）。
fn le(a: f64, b: f64) -> bool {
    a <= b
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
#[path = "lin_libm_tests.rs"]
mod libm_tests;

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
    pub(super) struct F32Rev(pub(super) f32);

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
        fn powf_bounded(self, _e: f64, e_str: f32) -> Self {
            F32Rev(self.0.powf(e_str))
        }
        fn exp_bounded(self) -> Self {
            F32Rev(self.0.exp())
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
        fn exp_bounded(self) -> Self {
            LinRef::make(self.v.exp(), vec![], self.t).unbounded()
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
        let mut acc: Vec<Term> = Vec::new();
        for x in &xs {
            let xt = x.terms_of();
            let mut out = vec![(0, 0.0); acc.len() + xt.len()];
            let o = merge_lin(&acc, 1.0, &xt, 1.0, &mut out);
            out.truncate(o);
            acc = out;
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
    // ---- 疎モード（イシュー #4078・決定記録 §80.4） ----

    /// 1 つの出力要素の記録（値・taint・上界・受理／拒否）。
    #[derive(Clone, Copy)]
    struct Out {
        v: u64,
        t: bool,
        bound: f64,
        ok: bool,
    }

    /// WeightNorm の 1 グループを指定モードで走らせ、全 seed・全出力要素の記録と seed ごとの事象数を返す。
    fn wn_outputs(
        v: &[f32],
        up: &[f32],
        g: f32,
        s: f64,
        sparse: bool,
    ) -> Option<(Vec<Out>, Vec<usize>)> {
        let rest = v.len();
        if sparse {
            lin_reset_sparse();
        } else {
            lin_reset();
        }
        let (fl, fs) = (wn_fwd::<Lin>(v, up, g, s)?, wn_fwd::<f32>(v, up, g, s)?);
        let mark = lin_mark();
        let (mut outs, mut events) = (Vec::new(), Vec::new());
        for seed in (0..rest).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
            lin_truncate(mark);
            let (mut hv_l, mut hu_l) = (vec![Lin::k(0.0); rest], vec![Lin::k(0.0); rest]);
            let (mut hv_s, mut hu_s) = (vec![0f32; rest], vec![0f32; rest]);
            let gl = wn_rev(&fl, seed, &mut hv_l, &mut hu_l);
            let gs = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
            events.push(lin_events());
            let mut pairs: Vec<(Lin, f32)> = (0..rest)
                .flat_map(|i| [(hv_l[i], hv_s[i]), (hu_l[i], hu_s[i])])
                .collect();
            if let (Some(gl), Some(gs)) = (gl, gs) {
                pairs.push((gl, gs));
            }
            for (l, f) in pairs {
                outs.push(Out {
                    v: canon_bits(l.v),
                    t: l.t,
                    bound: l.bound(),
                    ok: entry_check_lin(l, f, "p", bench_reject).is_ok(),
                });
            }
        }
        Some((outs, events))
    }

    /// 幅・振幅・乱数を振った WeightNorm の入力（幅 2〜8 を中心に 12・16・24・32 と、#4082 で広げた範囲の
    /// 代表幅 48・64 を混ぜる）。
    fn wn_sweep_input(case: usize, next: &mut impl FnMut() -> f64) -> BenchInput {
        let rest = match case {
            c if c % 100 == 99 => 64,
            c if c % 100 == 74 => 48,
            c if c % 50 == 49 => 32,
            c if c % 25 == 24 => 24,
            c if c % 12 == 11 => 16,
            c if c % 6 == 5 => 12,
            c => 2 + c % 7,
        };
        let amp = [1e-3, 1.0, 1e3][case % 3];
        let v: Vec<f32> = (0..rest)
            .map(|_| ((next() * 2.0 - 1.0) * 2.0 * amp) as f32)
            .collect();
        let up: Vec<f32> = (0..rest)
            .map(|_| ((next() * 2.0 - 1.0) * 3.0) as f32)
            .collect();
        let g = (0.05 + next() * 2.0) as f32;
        let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = f64::from(pow2(pow2_ceil_exponent(mx)));
        (v, up, g, s)
    }

    fn sweep_rng() -> impl FnMut() -> f64 {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f64) / f64::from(1u32 << 31)
        }
    }

    /// 受け入れ条件 1（下流まで含む）: 同じ WeightNorm の掃引を厳密モードと疎モードで走らせ、全 seed・全出力要素で
    /// 値と taint が bit 同一、疎モードの上界が厳密モード以上、事象数が減らず、疎モードで受理なら厳密モードでも
    /// 受理である（判定を緩めない向きの含意）。まとめが実際に起きることも確かめる。
    #[test]
    fn sparse_lin_never_looser_than_exact_for_weight_norm_sweep() {
        let mut next = sweep_rng();
        let (mut compared, mut folded) = (0usize, 0u64);
        for case in 0..150 {
            let (v, up, g, s) = wn_sweep_input(case, &mut next);
            if v.iter().all(|&x| x == 0.0) {
                continue;
            }
            let Some((exact, ev_exact)) = wn_outputs(&v, &up, g, s, false) else {
                continue;
            };
            LIN_PROF.with(|c| c.set([0; 8]));
            let (sparse, ev_sparse) = wn_outputs(&v, &up, g, s, true).expect("同じ入力");
            folded += LIN_PROF.with(|c| c.get())[PROF_FOLDS];
            assert_eq!(exact.len(), sparse.len());
            for (e, p) in ev_exact.iter().zip(&ev_sparse) {
                assert!(p >= e, "case {case}: 事象数が減った");
            }
            for (e, p) in exact.iter().zip(&sparse) {
                assert_eq!(e.v, p.v, "case {case}: 値は不変");
                assert_eq!(e.t, p.t, "case {case}: taint は不変");
                assert!(
                    p.bound.is_nan() == e.bound.is_nan()
                        && (e.bound.is_nan() || p.bound >= e.bound * (1.0 - 1e-12)),
                    "case {case}: 疎の上界 {:e} が厳密 {:e} を下回った",
                    p.bound,
                    e.bound
                );
                assert!(!p.ok || e.ok, "case {case}: 疎で受理なのに厳密で拒否");
                compared += 1;
            }
        }
        lin_reset();
        assert!(compared > 1000, "比較した要素数 {compared}");
        assert!(folded > 0, "まとめが一度も起きていない");
    }

    /// 受け入れ条件 2: 幅 2〜64（#4082 で 32 から拡大）の soundness（疎モード）。f32 の再現と和の順序を変えた再現の実測の誤差が、疎モードの
    /// 上界 × 余裕を超えない。`Lin::bound` を 0 にすると失敗する（変異確認は §80.4）。
    /// 幅 65〜128 は [`lin_bound_covers_f32_deviation_for_weight_norm_sweep_65_to_128`]（§80.6）。
    #[test]
    fn lin_bound_covers_f32_deviation_for_weight_norm_wide_sweep() {
        let mut next = sweep_rng();
        let (mut checked, mut tight) = (0usize, 0usize);
        for case in 0..120 {
            let (v, up, g, s) = wn_sweep_input(case, &mut next);
            let rest = v.len();
            if v.iter().all(|&x| x == 0.0) {
                continue;
            }
            lin_reset_sparse();
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
        lin_reset();
        assert!(checked > 1000, "検査した要素数 {checked}");
        assert!(tight > 0, "上界が実測より桁違いに大きい");
    }

    /// 幅 65〜128（#4082 で `Lin` の適用範囲に加えた範囲。§80.6）の掃引の幅の表。96・128・奇数幅を含み、
    /// 上限 `LIN_REST_MAX` ちょうどの 128 を必ず含める。
    const WIDE_RESTS: [usize; 8] = [65, 72, 81, 96, 100, 113, 127, 128];

    /// 幅 65〜128 の掃引で、幅ごとの「実測の誤差 / 上界×slack」の最大比に課す下限。目的は上界が緩む回帰の
    /// 検出（soundness の検査ではない）。根拠は §80.8: §80.6 の実測の最大比は 0.0629〜0.0857 で、0.01 は
    /// その最小値の約 6 分の 1。テスト専用で、`Lin` の判定には使わない。
    const WN_WIDE_MAX_RATIO_FLOOR: f64 = 0.01;

    /// 幅 65〜128 用の入力生成。`variant` 0〜5 は 振幅 {1e-3, 1, 1e3} × 符号パターン
    /// {符号付き一様, 正のみ `[0.2, 1.0)·amp`}。`up`・`g`・`s` の作り方は [`wn_sweep_input`] と同じ。
    fn wn_sweep_input_at(
        rest: usize,
        variant: usize,
        next: &mut impl FnMut() -> f64,
    ) -> BenchInput {
        let amp = [1e-3, 1.0, 1e3][variant % 3];
        let signed = variant < 3;
        let v: Vec<f32> = (0..rest)
            .map(|_| {
                if signed {
                    ((next() * 2.0 - 1.0) * 2.0 * amp) as f32
                } else {
                    ((0.2 + 0.8 * next()) * amp) as f32
                }
            })
            .collect();
        let up: Vec<f32> = (0..rest)
            .map(|_| ((next() * 2.0 - 1.0) * 3.0) as f32)
            .collect();
        let g = (0.05 + next() * 2.0) as f32;
        let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = f64::from(pow2(pow2_ceil_exponent(mx)));
        (v, up, g, s)
    }

    /// 受け入れ条件 2（#4117）: 幅 65〜128（96・128 を含む 8 幅 × 6 ケース）の soundness（疎モード）。
    /// 照合の本体は [`lin_bound_covers_f32_deviation_for_weight_norm_wide_sweep`] と同じ（幅 2〜64 の入力列と
    /// 乱数の消費順を変えないため別テストにし、本体は複製した）。`Lin::bound` を 0 にすると失敗する（変異確認は §80.6）。
    /// 上界の緩さ（tight > 0）は幅 65〜128 では成立しないため合否にしない（§80.6）。
    /// 緩さの回帰は幅ごとの最大比の下限 [`WN_WIDE_MAX_RATIO_FLOOR`] で検査する（`Lin::bound` を 100 倍にすると失敗する。§80.8）。
    /// 幅ごとの被覆も fail-closed に検査する（`wn_fwd` が `None` で幅が黙って消えるのを防ぐ）。
    #[test]
    fn lin_bound_covers_f32_deviation_for_weight_norm_sweep_65_to_128() {
        let mut next = sweep_rng();
        let (mut checked, mut tight) = (0usize, 0usize);
        for &rest in &WIDE_RESTS {
            let (mut w_checked, mut w_tight, mut w_cases) = (0usize, 0usize, 0usize);
            let mut max_ratio = 0.0f64;
            for variant in 0..6 {
                let (v, up, g, s) = wn_sweep_input_at(rest, variant, &mut next);
                if v.iter().all(|&x| x == 0.0) {
                    continue;
                }
                lin_reset_sparse();
                let (Some(fl), Some(fs), Some(fa)) = (
                    wn_fwd::<Lin>(&v, &up, g, s),
                    wn_fwd::<f32>(&v, &up, g, s),
                    wn_fwd::<F32Rev>(&v, &up, g, s),
                ) else {
                    continue;
                };
                w_cases += 1;
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
                        .flat_map(|i| {
                            [(hv_l[i], hv_s[i], hv_a[i].0), (hu_l[i], hu_s[i], hu_a[i].0)]
                        })
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
                                "rest {rest} variant {variant}: 実測の誤差 {dev:e} が上界 {bound:e} を超えた（V = {:e}）",
                                l.v
                            );
                            w_checked += 1;
                            if dev > 0.1 * bound {
                                w_tight += 1;
                            }
                            if bound > 0.0 {
                                max_ratio = max_ratio.max(dev / bound);
                            }
                        }
                    }
                }
            }
            eprintln!(
                "[4117] rest {rest}: cases {w_cases} checked {w_checked} tight {w_tight} max(dev/bound) {max_ratio:e}"
            );
            assert!(w_cases >= 2, "rest {rest}: 照合できたケースが {w_cases}");
            assert!(w_checked > 0, "rest {rest}: 検査した要素が無い");
            assert!(
                max_ratio >= WN_WIDE_MAX_RATIO_FLOOR,
                "rest {rest}: 実測の誤差 / 上界の最大比 {max_ratio:e} が下限 {WN_WIDE_MAX_RATIO_FLOOR:e} を下回った（上界が緩む回帰。§80.8）"
            );
            checked += w_checked;
            tight += w_tight;
        }
        lin_reset();
        assert!(checked > 1000, "検査した要素数 {checked}");
        // 幅 65〜128 では tight（dev > 0.1·bound）が 0 になる（上界が緩い側。§80.6）。0.1 の係数は緩めず、
        // tight はログ出力のままにし、soundness（dev <= bound）と幅ごとの最大比の下限（§80.8）を合否にする。
        eprintln!("[4117] total checked {checked} tight {tight}");
    }

    /// 乱数の演算列を疎モードまたは厳密モードで走らせ、作った値すべての `(v, t, 上界)` を返す
    /// （疎モードではまとめ箇所ごとの上界の非後退が [`LinArena::pool_small_terms`] 内で検査される）。
    fn run_op_program(seed: u64, sparse: bool) -> Vec<(u64, bool, f64)> {
        const KS: [f64; 8] = [0.3, -0.7, 1.0, 2.5, 0.1, -1.9, 3.3, 0.6];
        let mut state = seed;
        let mut next = move |m: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize) % m
        };
        if sparse {
            lin_reset_sparse();
        } else {
            lin_reset();
        }
        let mut pool: Vec<Lin> = KS.iter().map(|&x| Lin::k(x)).collect();
        for _ in 0..400 {
            let a = pool[next(pool.len())];
            let b = pool[next(pool.len())];
            let r = match next(8) {
                0 => a.plus(b),
                1 => a.minus(b),
                2 | 3 => a.times(b),
                4 => a.times(Lin::k(KS[next(KS.len())])),
                5 => a.negate(),
                6 => {
                    let n = 2 + next(6);
                    let xs: Vec<Lin> = (0..n).map(|_| pool[next(pool.len())]).collect();
                    Lin::total(&xs)
                }
                _ => {
                    let sq = a.times(a).plus(Lin::k(1.0));
                    sq.root().over(sq)
                }
            };
            pool.push(r);
        }
        let out: Vec<_> = pool
            .iter()
            .map(|x| (canon_bits(x.v), x.t, x.bound()))
            .collect();
        lin_reset();
        out
    }

    /// まとめた時点の性質のプロパティ（決定的な乱数の演算列）: 各まとめ箇所で上界が非後退（`fold` 内の assert）、
    /// 値と taint は不変、下流でも疎モードの上界が厳密モード以上。
    #[test]
    fn sparse_fold_never_shrinks_bound_for_random_op_programs() {
        LIN_PROF.with(|c| c.set([0; 8]));
        for seed in 1..=40u64 {
            let exact = run_op_program(seed, false);
            let sparse = run_op_program(seed, true);
            for (i, (e, p)) in exact.iter().zip(&sparse).enumerate() {
                assert_eq!(
                    (e.0, e.1),
                    (p.0, p.1),
                    "seed {seed} 値 {i}: 値・taint は不変"
                );
                assert!(
                    (e.2.is_nan() && p.2.is_nan()) || p.2 >= e.2 * (1.0 - 1e-12),
                    "seed {seed} 値 {i}: 疎の上界 {:e} が厳密 {:e} を下回った",
                    p.2,
                    e.2
                );
            }
        }
        let p = LIN_PROF.with(|c| c.get());
        assert!(
            p[PROF_FOLDS] > 0 && p[PROF_FOLD_DROPS] > 0,
            "まとめが一度も起きていない"
        );
    }

    /// 上界が NaN／inf の列はまとめない（NaN の上界を有限の残余へ畳んで常に拒否を受理に変えない）。
    #[test]
    fn sparse_fold_leaves_non_finite_bounds_alone() {
        lin_reset_sparse();
        let n = Lin::k(2.0).powe(0.5);
        assert!(n.bound().is_nan());
        let mut acc = n.plus(Lin::k(1.0));
        for i in 0..20 {
            acc = acc.plus(Lin::k(0.1 * f64::from(i + 1)).times(Lin::k(1.0 / 3.0)));
        }
        assert!(acc.bound().is_nan(), "NaN の上界は NaN のまま");
        assert!(entry_check_lin(acc, acc.v as f32, "p", bench_reject).is_err());
        // 0 を掛けても NaN の事象の項は残る（0·NaN = NaN）。
        let z = n.times(Lin::k(0.0));
        assert!(z.bound().is_nan());
        lin_reset();
    }

    /// 本番経路（`wn_group_judge(use_lin = true)`）は疎モードで走り、`lin_reset` は厳密モードへ戻す。
    /// `lin_mark`／`lin_truncate` はモードを変えない。
    #[test]
    fn wn_judge_uses_sparse_mode_and_reset_restores_exact() {
        let sparse_flag = || LIN_ARENA.with(|a| a.borrow().sparse);
        let (v, up) = ([0.9f32, 0.1, 0.6, 0.3], [0.7f32, -0.2, 0.4, 0.1]);
        lin_reset();
        assert!(!sparse_flag());
        assert!(wn_group_judge(&v, &up, 0.5, 1.0, wn_coeff(4), true).is_ok());
        assert!(sparse_flag(), "WeightNorm の判定は疎モード");
        lin_reset();
        assert!(!sparse_flag(), "lin_reset は厳密モードへ戻す");
        lin_reset_sparse();
        let m = lin_mark();
        lin_truncate(m);
        assert!(sparse_flag(), "mark／truncate はモードを変えない");
        lin_reset();
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

    thread_local! {
        /// 計測の走査（[`scan_lin`]・[`n_probe`]）を疎モードで走らせるか（#4078。既定は厳密モード＝#4076・#4077 の計測と同じ）。
        static PROFILE_SPARSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn profile_reset() {
        if PROFILE_SPARSE.with(|c| c.get()) {
            lin_reset_sparse();
        } else {
            lin_reset();
        }
    }

    /// `wn_group_judge_with::<Lin>` の判定ループを、最初の拒否で打ち切らず全要素を検査する版。本番は拒否で
    /// 短絡するため、拒否される入力でも時間が途中で切れないようにする計測専用の再実装。forward の NaN 位置の
    /// 照合（`same_nan`）は走査の主要コストでないため省く。
    fn scan_lin(v: &[f32], up: &[f32], g: f32, s: f64) -> Scan {
        let rest = v.len();
        profile_reset();
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
        profile_reset();
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
            LIN_PROF.with(|c| c.set([0; 8]));
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
    /// 疎モードの効果の計測（イシュー #4078）。幅ごとに、厳密モードと疎モード（`θ` を差し替えて）の判定時間
    /// （release・5 回中央値）・係数列へ書いた要素数・事象数 `N`・まとめた回数・落とした項数・受理／拒否の件数を
    /// `FOLDPROF` 行で出す。判定・定数は変えない。
    ///
    /// 実行: `cargo test -p fandhe-ai-autodiff --release --lib -- --ignored --nocapture --test-threads=1 lin_sparse_cost_profile`
    /// 記録先: `docs/autodiff-higher-order-grad-decision.md` §80.4。
    #[test]
    #[ignore = "手動実行専用の計測（stderr 出力のみ。#4078・§80.4）"]
    fn lin_sparse_cost_profile() {
        let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
        let configs: [(&str, bool, f64); 4] = [
            ("exact", false, 0.0),
            ("sparse", true, 0.5),
            ("sparse", true, 0.25),
            ("sparse", true, 0.125),
        ];
        for &rest in &[8usize, 16, 32, 64] {
            let groups = match rest {
                0..=8 => 32,
                9..=16 => 8,
                17..=32 => 2,
                _ => 1,
            };
            let inputs: Vec<_> = (0..groups).map(|_| gen_input(&mut rng, rest)).collect();
            for &(mode, sparse, theta) in &configs {
                PROFILE_SPARSE.with(|c| c.set(sparse));
                FOLD_FRAC_OVERRIDE.with(|c| c.set(sparse.then_some(theta)));
                let (med, _) = measure_lin(&inputs);
                let (v, up, g, s) = &inputs[0];
                LIN_PROF.with(|c| c.set([0; 8]));
                let sc = scan_lin(v, up, *g, *s);
                let p = LIN_PROF.with(|c| c.get());
                eprintln!(
                    "FOLDPROF rest={rest} mode={mode} theta={theta} median_ns_per_group={med:.0} N_dv={} N_dg={} \
                     write_entries={} folds={} fold_drops={} entries_accepted={} entries_rejected={}",
                    sc.n_dv,
                    sc.n_dg,
                    p[PROF_WRITE_ENTRIES],
                    p[PROF_FOLDS],
                    p[PROF_FOLD_DROPS],
                    sc.accepted,
                    sc.rejected,
                );
            }
        }
        PROFILE_SPARSE.with(|c| c.set(false));
        FOLD_FRAC_OVERRIDE.with(|c| c.set(None));
    }

    /// 幅 8 超への適用範囲拡大のための計測（イシュー #4082）。疎モード（本番と同じ）で幅ごとに 1 グループの
    /// 全走査時間（5 回中央値）・事象数 `N`（乱数入力と振幅違いの最大）・slack を出し、同じ実行内で測った
    /// `Rf` 経路の 1 反復当たり時間で反復相当へ換算する。判定・定数は変えない。
    ///
    /// 実行: `cargo test -p fandhe-ai-autodiff --release --lib -- --ignored --nocapture --test-threads=1 lin_wide_apply_profile`
    /// 記録先: `docs/autodiff-higher-order-grad-decision.md` §80.5。
    #[test]
    #[ignore = "手動実行専用の計測（stderr 出力のみ。#4082・§80.5）"]
    fn lin_wide_apply_profile() {
        PROFILE_SPARSE.with(|c| c.set(true));
        let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
        // 同じ実行内の Rf 1 反復当たり時間（幅 8 の受理入力で測る）。
        let ns_per_iter = {
            let rest = 8usize;
            let inp = (0..5000)
                .map(|_| gen_input(&mut rng, rest))
                .find(|i| wn_group_judge(&i.0, &i.1, i.2, i.3, wn_coeff(rest), false).is_ok())
                .expect("幅 8 の Rf 受理入力");
            let reps = (2_000_000 / wn_judge_cost(1, rest)).max(1);
            let one = || -> f64 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    black_box(
                        wn_group_judge(&inp.0, &inp.1, inp.2, inp.3, wn_coeff(rest), false).is_ok(),
                    );
                }
                t0.elapsed().as_nanos() as f64 / reps as f64
            };
            let _ = one();
            middle_value((0..5).map(|_| one()).collect()) / wn_judge_cost(1, rest) as f64
        };
        eprintln!("WIDEPROF rf_ns_per_iter={ns_per_iter:.3}");
        for &rest in &[2usize, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
            let groups = match rest {
                0..=8 => 32,
                9..=16 => 8,
                17..=32 => 2,
                _ => 1,
            };
            let inputs: Vec<_> = (0..groups).map(|_| gen_input(&mut rng, rest)).collect();
            let (med, _) = measure_lin(&inputs);
            let mut n_max = 0usize;
            for amp in [1e-3f32, 1.0, 1e3] {
                for _ in 0..4 {
                    let (mut v, up, g, _) = gen_input(&mut rng, rest);
                    v.iter_mut().for_each(|x| *x *= amp);
                    let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
                    let s = f64::from(pow2(pow2_ceil_exponent(mx)));
                    n_max = n_max.max(n_probe(&v, &up, g, s));
                }
            }
            let (v, up, g, s) = &inputs[0];
            let sc = scan_lin(v, up, *g, *s);
            let equiv = med / ns_per_iter;
            eprintln!(
                "WIDEPROF rest={rest} groups={groups} median_ns_per_group={med:.0} equiv_iters={equiv:.0} \
                 lin_group_cost_old={} ratio_old_over_meas={:.2} N_probe_max={n_max} N_scan_dv={} N_scan_dg={} \
                 slack={:.5} 18r+17={}",
                lin_group_cost(rest),
                lin_group_cost(rest) as f64 / equiv,
                sc.n_dv,
                sc.n_dg,
                16.0 * n_max as f64 * LIN_U,
                18 * rest + 17,
            );
        }
        PROFILE_SPARSE.with(|c| c.set(false));
    }
}
