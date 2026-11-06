//! 正規化・再パラメータ 3 Op（`LocalResponseNorm`・`WeightNorm`・`SpectralNorm`）の `create_graph` 対応
//! （イシュー #3794・親 #3727・設計 `docs/autodiff-higher-order-grad-decision.md` §36、実装記録 §57）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `validate_ancestors`（入口検査
//! [`validate_norm_reparam`]）・`replay_op`（forward 再生。公開 `Var::local_response_norm`／
//! `weight_norm` と、状態を進めない `weight_reparam_ops::spectral_norm_from_snapshot` への委譲なので
//! 親側に置く）・`build_cgrads`（1 階 VJP の子テープ記録 [`lrn_cgrad`]・[`weight_norm_cgrads`]・
//! [`spectral_norm_cgrad`]）から呼ばれる。
//!
//! # 方式（値は 1 階 `grad.rs` の VJP と bit 一致。契約は REQ-2 のまま）
//!
//! 3 Op とも **V3**: 値（`cg.grad` の実体）は 1 階と同じ f64 ホスト関数
//! （`local_response_norm_vjp_host`・`weight_norm_vjp_host`・`spectral_norm_vjp_host`）を
//! [`push_v3`] で載せ、構造（二階の微分が流れる `Var` 演算グラフ）は f32 の合成で組む。構造は
//! 2 のべき乗スケール（**S1**）で値域を `|x_s| ≤ 1` に収めてから正規化し、最後に戻す。
//!
//! - `LocalResponseNorm`: `[N, C, inner]` へ畳み、`(n, inner)` の列ごとに `s = pow2ceil(max(max|x|,
//!   √(|k|·size/|α|)))`。`x_s = x/s`、`d_s = k/s² + (α/size)·Σ_W x_s²`、
//!   `dx = s^{-2β}·[g·d_s^{-β} − (2αβ/size)·x_s·Σ_{W'} g·x_s·d_s^{-β-1}]`。窓和は `Pad`（値 0）＋
//!   ずらした `Narrow` の `Add`。前方窓 `W` と反転窓 `W'` は別々に作る（偶数 `size` で非対称）。
//!   `s^{-2β}` は ±2^100 以下の 2 のべき乗チャンクへ分けて順に `Mul` する（forward の値に f32 の非正規数を
//!   作らないため）。二階の逆伝播の cotangent はチャンクの積 `s^{-2β}` そのもの（真の偏微分）になるので、
//!   それが f32 で表せない列（例 `s = 2^-80`・`β = −1` で `2^-160`）は第 2 軸で拒否する。
//!   `α = 0` は `dx = g·k^{-β}` に縮退するので別分岐にする。
//! - `WeightNorm`: `dim` 軸を先頭へ寄せた `[groups, rest]` に正準化し、グループごとに
//!   `v_s = v/s`、`n_s = ‖v_s‖`、`v̂ = v_s/n_s`、`dg = Σ(g_up⊙v̂)`、
//!   `dv = (g/(n_s·s))·(g_up − dg·v̂)`。`g`（パラメータ）は子テープ上の `Var` のまま使い、
//!   v と g の混合二階項を保つ。`dv` と `dg` の 2 出力を `v`・`g` の両方へ蓄積する。
//! - `SpectralNorm`: `u`・`v` は forward 時点の非追跡スナップショットで、**power iteration は定数扱い**
//!   （再生も `create_graph` も状態を進めない。#3767 の承認）。ホスト定数 `P[i,j] = u_i·v_j`（W と
//!   同じ並び）と `s = pow2ceil(max|W|)` から `W_s = W/s`、`σ_s = Σ(W_s⊙P)`、`Ŵ = W_s/σ_s`、
//!   `dW = (g_up − Σ(g_up⊙Ŵ)·P)/(σ_s·s)`。`σ` の符号は問わない。
//!
//! 子テープへ積む Op は `Reshape`・`Permute`・`Contiguous`・`Narrow`・`Pad`・`BroadcastTo`・`Sum`・
//! `Mul`・`Add`・`Sub`・`Div`・`Sqrt`・`PowScalar`・定数 `Leaf`（と `Op::SpectralNorm`／`WeightNorm`／
//! `LocalResponseNorm` 自身）のみで、すべて `supports_create_graph() == true`（閉包性。孫テープでも閉じる）。
//! 0/1 定数を掛ける方式は使わない（`inf·0 = NaN` を作らない。§21.1(3)）。
//!
//! # 拒否条件（fail-closed。保証範囲は §27.3 と同じ書き方）
//!
//! - **入口**（[`validate_norm_reparam`]。子テープへ書く前）: 入力に `±inf` がある（NaN は伝播させる。
//!   #3767 の承認）。LRN は窓内の `D = k + (α/size)·Σx²` が 0 以下（`k = 0` で窓が全ゼロ・`α < 0` で負）、
//!   WeightNorm はゼロノルムのグループ、SpectralNorm は `σ = 0` または非有限。
//! - **総括判定・第 1 軸（overflow。[`stage`] 系）**: `g` が手に入った時点で構造を積む前に、構造の forward の
//!   各段と二階の逆伝播（seed の各要素の絶対値 ≤ 1）の「cotangent × 微分係数」の上界を f64 で積み上げ、
//!   どれか 1 段でも `2^125` 以上か非有限なら拒否する。非有限の上流 `g` も拒否する。
//! - **総括判定・第 2 軸（相殺・f32 の underflow／overflow。本体は [`entry_check`]）**: 構造を積む前に、
//!   seed が 1 要素（値 1）のときの二階の各要素（LRN は `x`・`up` へ、WeightNorm は `dv` と `dg` の各 seed から
//!   `v`・`up`・`g` へ、SpectralNorm は `W`・`up` へ）を、構造の演算列を f64 の厳密な定数で評価した参照 `V` と
//!   各項の絶対値の和 `M`（[`Rf`]）、構造が子テープで実際に行う f32 の演算列（forward の中間値・
//!   `eval/scalar.rs` の VJP 係数・`cot·係数` の積・チャンクの `Mul` を逆順にたどる cotangent）を f32 で
//!   再現した `V_str` で求め、`(|V_str − V| + coeff·M)` が絶対 1e-5 未満か相対の半分以内でない要素が 1 つでも
//!   あれば拒否する（`loss_class.rs::struct_check` と同じ形。tolerance 定数は変えない）。f32 の forward 値・
//!   VJP 係数に `±inf` が 1 つでもあれば（逆伝播は全要素に `cot·係数` を作るので `0·inf = NaN`）拒否する。
//! - **NaN の到達範囲**: NaN は伝播させるが、NaN の届かない有限の要素は検査なしで通さない。到達範囲は
//!   f32 の再現と参照の両方に NaN を伝播させて構造の演算そのもので決め、判定が外すのは両方が NaN の要素
//!   だけ（片側だけ NaN は拒否。forward の NaN の位置の一致は [`same_nan`]）。LRN は NaN を含む列でも
//!   NaN の届かない窓を、WeightNorm は g が NaN のグループでも g に依存しない要素を判定する。段検査と判定を
//!   丸ごと外すのは、1 階と二階の全要素が入力の NaN で NaN になると [`Dm`] の依存の走査で確かめた単位
//!   （LRN の列・`v` に NaN がある WeightNorm のグループ・`W` に NaN がある SpectralNorm）だけで、段検査より
//!   前に外す。走査は判定と同じ規模なので走査量に数える。NaN を含むのに有限の出力が残る単位は外さず、
//!   段の上界を定められない形（LRN の有限の `d_s` が無い列、WN・SN の全要素 NaN でない単位）は拒否する
//!   （§57.2）。参照の NaN は入力の NaN への依存（[`Rf`] の taint）を持ち、演算で生まれた NaN は外さない。
//! - **ホストの 1 階**（[`host_nonfinite_check`]。全分岐・構造を積む前）: 非有限は、入力の NaN が届く
//!   位置の NaN だけを伝播として通し、有限の入力から演算で生まれた非有限（f32 の積の overflow・`0·inf`）は
//!   拒否する。
//!   走査量（f32 の再現 1 系統が回す内側ループの反復回数。LRN は seed ごとの反転窓 × 前方窓の二重走査を
//!   含む。[`lrn_judge_cost`] 等）が `2^27` を超える入力は計算量の fail-closed として拒否する。構造を積んだ
//!   後の 1 階の値の照合（[`shadow_check`]）も残す。
//! - 拒否は `AutodiffError::Backward`。`backward_create_graph` のロールバックで `child` は空へ戻り、
//!   安全な入力で再試行できる。
//!
//! 保証は判定を通った入力に限る。判定を通った入力では 1 階・`cg.grad`・二階（seed の各要素の絶対値 ≤ 1）・
//! 孫テープ再生が有限で、`cg.grad` は 1 階と bit 一致する（V3）。二階の各要素（seed が 1 要素のとき）は
//! 第 2 軸の見積りの範囲で f64 の参照と REQ-2 で一致する（見積りは一次の評価で、厳密な証明ではない）。
//! 対象外: 1 要素でない seed の縮約での相殺、`g` を作る上流の Op の逆伝播、絶対値 1 を超える seed、
//! 三階以上の微分、真の値そのものが f32 を超える場合、CPU 以外の演算順・非正規数の扱い（GPU は未実測）。

//! GPU: `BackendOps::{lrn_forward, weight_norm_forward, spectral_norm_forward}` を実装するのは
//! `backend-cpu` だけで、CUDA／Metal は forward も 1 階もホストへ落ちる経路区分 `H`。GPU カーネルは
//! 新設しない。実機 parity は未実測（Phase 10 の #3856・#3858）。

use fandhe_ai_tensor_core::lrn::{self, LrnParams};
use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};
use fandhe_ai_tensor_core::weight_reparam::{self as wr};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

use super::{child_value, judge, push_v3};
use crate::create_graph_norm::pow2_ceil_exponent;
use crate::error::AutodiffError;
use crate::eval::dense_vec;
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {msg}（正規化・再パラメータ LocalResponseNorm／WeightNorm／SpectralNorm の構造が\
         安全に扱えない入力。docs/autodiff-higher-order-grad-decision.md §36・§57 参照）"
    ))
}

/// 構造の 1 段の上界 `v`（f64）を判定する。非有限または `2^125` 以上なら型付きエラー。
fn stage(v: f64, what: &str) -> Result<(), AutodiffError> {
    if judge::stage_strict_ok(v) {
        Ok(())
    } else {
        Err(invalid(&format!(
            "構造段の上界判定: {what} の上界 {v:e} が 2^125 以上で f32 の安全域を外れる"
        )))
    }
}

fn check_finite(data: &[f32], what: &str) -> Result<(), AutodiffError> {
    if data.iter().any(|v| !v.is_finite()) {
        return Err(invalid(&format!("{what} に非有限要素がある")));
    }
    Ok(())
}

fn pow2(e: i32) -> f32 {
    2f64.powi(e) as f32
}

fn scalar_const(child: &Tape, v: f32) -> Result<Var<'_>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::full(&[1], v)?))
}

fn tensor_const<'c>(
    child: &'c Tape,
    vals: Vec<f32>,
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::new(vals, shape)?))
}

// ---------------------------------------------------------------------
// 相殺・underflow 判定（loss_class.rs::struct_check と同じ形。そちらとは共有しない）
// ---------------------------------------------------------------------

/// 丸め誤差係数 `(2·terms + 4)·2^-23`。`terms` は構造の最長の縮約長。
fn rounding_coeff(terms: usize) -> f64 {
    judge::rounding_coeff(terms)
}

/// 1 要素の構造値 `v_str`（f32）が参照値 `v` に REQ-2 の統一複合判定で一致するかの保守的検査。
/// `m` は同じ式を各項の絶対値の和で評価した大きさ。NaN の参照値（ホストの 1 階が NaN の要素）は伝播の
/// 契約に従い素通しにする。参照値が有限なら `m` が NaN でも拒否する（有限の要素を検査なしで通さない）。
fn struct_check(v: f64, v_str: f64, m: f64, coeff: f64, what: &str) -> Result<(), AutodiffError> {
    judge::struct_check_unscaled(v, v_str, m, coeff, what, invalid)
}

/// 子テープ上の構造 `composed` の実値を 1 階のホスト値 `host` と要素ごとに突き合わせる（構造を積んだ後の
/// 1 階の値の照合。二階の判定の本体は構造を積む前の [`entry_check`]）。
fn shadow_check(
    child: &Tape,
    composed: &Var<'_>,
    host: &[f32],
    m: &[f64],
    coeff: f64,
    what: &str,
) -> Result<(), AutodiffError> {
    let value = dense_vec(&child_value(child, composed)?);
    if value.len() != host.len() || m.len() != host.len() {
        return Err(invalid(&format!("{what}: 構造とホスト値の要素数が不一致")));
    }
    for ((&v_str, &v), &mi) in value.iter().zip(host).zip(m) {
        struct_check(f64::from(v), f64::from(v_str), mi, coeff, what)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 総括判定・第 2 軸の本体: 二階の各要素の f32 再現（PR #3950 の codex 指摘の類型を 1 つに寄せる）
// ---------------------------------------------------------------------
//
// 3 Op の構造は、二階（子テープの `backward`）で forward の中間値と VJP 係数・cotangent を f32 で
// 計算する。f32 の中間値の overflow・underflow・相殺・定数の丸めは、どれも「二階の要素が f64 の参照から
// 外れる」形で現れる（例: LRN の `s^{-2β}` を 2 のべき乗チャンクで戻すとき cotangent の積が 2^-160 で 0 に
// なる、WeightNorm の `g/n_s` が 0 になる、`1 − v̂²` や `I − ŴP` が丸めで消える、Div／PowScalar の係数が
// f32 で inf になり `0·inf = NaN` を作る）。経路ごとに条件を足さず、構造を積む前に、seed が 1 要素
// （値 1）のときの二階の各要素を次の 2 通りで計算して [`entry_check`] に掛ける。
//
// - `V`・`M`（[`Rf`]）: 同じ演算列を f64 の厳密な定数で評価した参照値と、各項の絶対値の和（相殺の大きさ）。
// - `V_str`（`f32`）: 構造が子テープで実際に行う f32 の演算列（forward の中間値・`eval/scalar.rs` の VJP
//   係数〈Div は `1/b`・`-(a/b)/b`、Sqrt は `0.5/y`、PowScalar は `e·x^(e−1)`（`e == 0` は 0）〉・
//   `grad.rs` の `cot·係数` の積・チャンクの `Mul` を逆順にたどる cotangent）を同じ順に f32 で再現した値。
//
// 構造は f32 の値から離散量（符号・マスク・分岐）を決めない（スケール指数・窓・グループ分けはホストの
// V2 定数で、`V_str` もそれを共有して正しい）。要素数 0 でない cotangent に掛かる係数だけでなく、
// 逆伝播は全要素に `cot·係数` を作るので、f32 の forward 値・VJP 係数が 1 つでも `±inf` なら `V_str` は
// NaN になる（`0·inf`）。これは forward の再現で `±inf` を見つけた時点で拒否する。入力の NaN は f32 と
// 参照の両方に同じ演算で伝播させ（forward で位置が一致することを [`same_nan`] で確かめる）、両方が NaN の
// 要素だけを判定から外す（[`entry_check`]）。

/// 総括判定の走査量の上限。走査量は f32 の再現（`V_str`）1 系統が回す内側ループの反復回数で数え
/// （[`lrn_judge_cost`]・[`wn_judge_cost`]・[`sn_judge_cost`]。参照の [`Rf`] も同じ回数を回す）、
/// 超える入力は計算量の fail-closed として拒否する（保守的。上限内の入力だけに保証を限る）。
const JUDGE_BUDGET: usize = 1 << 27;

/// LRN の総括判定の走査量（[`lrn_col_fwd`]・[`lrn_col_rev`]・[`lrn_col_judge`] の反復回数の上界）。
/// 列（`ncols` 本・長さ `c`）ごとに、forward が窓和 `d`・`sz` の `c·(nf + nr)`（[`win_sum`] は範囲外も
/// 0 として回るので `nf = bf + ff + 1`・`nr = br + fr + 1` は打ち切らない。最大 `2c − 1`）と長さ `c` の走査
/// `33 + 2·nchunks` 回（中間値 10 本・`dx`・±inf の検査 11 本・チャンクの `Mul` と検査・作業領域の確保
/// 4 本・NaN の位置の照合〈[`same_nan`]〉8 本。定数個のスカラーは数えない）、seed（`c` 個）ごとに作業領域の初期化・`x_s` から `x` への戻し・`x` への判定が各 `sx`、反転窓の
/// 走査と `up` への判定が各 `wr`、反転窓の各 `m` から前方窓への蓄積（二重走査）が `wr·wf`。seed 側の
/// `wf`・`wr`・`sx` は前方窓・反転窓・`x` への要素の幅で、どれも `c` に打ち切る。f64 の参照（`d`・`m` の窓和）は
/// forward の窓和と同じ大きさ。
fn lrn_judge_cost(ncols: usize, c: usize, w: LrnWin, nchunks: usize) -> usize {
    let wf = (w.bf + w.ff + 1).min(c);
    let wr = (w.br + w.fr + 1).min(c);
    let sx = (wr + w.bf + w.ff).min(c);
    let (nf, nr) = (w.bf + w.ff + 1, w.br + w.fr + 1);
    let fwd = c
        .saturating_mul(nf + nr)
        .saturating_add(c.saturating_mul(nchunks.saturating_mul(2).saturating_add(33)));
    let per_seed = sx
        .saturating_mul(3)
        .saturating_add(wr.saturating_mul(2))
        .saturating_add(wr.saturating_mul(wf));
    ncols.saturating_mul(fwd.saturating_add(c.saturating_mul(per_seed)))
}

/// WeightNorm の総括判定の走査量（[`wn_fwd`]・[`wn_rev`]・[`wn_group_judge`] の反復回数）。グループ
/// （`groups` 個・長さ `rest`）ごとに、forward が長さ `rest` の走査 18 回（`v_s`・`up`・`Σv_s²`・`v̂`・
/// Div 係数・`dg`・`t`・`dv` と ±inf の検査 5 本・NaN の位置の照合 5 本）、seed（`rest + 1` 個）ごとに逆伝播が 5 回（作業領域の
/// 確保・`up` への要素の初期化・`dg`・`v̂`・`v_s` の逆伝播）と判定が `2·rest + 1`。
fn wn_judge_cost(groups: usize, rest: usize) -> usize {
    let per_seed = rest.saturating_mul(7).saturating_add(1);
    let per_group = rest
        .saturating_mul(18)
        .saturating_add((rest + 1).saturating_mul(per_seed));
    groups.saturating_mul(per_group)
}

/// SpectralNorm の総括判定の走査量（[`sn_fwd`]・[`sn_rev`]・[`sn_judge`] の反復回数）。forward が長さ
/// `n` の走査 23 回（`W_s`・`P`・`up`・`σ_s`・`Ŵ`・Div 係数 2 本・`Σup⊙Ŵ`・`t`・`dW` と ±inf の検査 7 本・
/// NaN の位置の照合 6 本）、
/// seed（`n` 個）ごとに逆伝播が 2 回（`Ŵ`・`up` と `σ_s` の逆伝播）と判定が `2·n`。
fn sn_judge_cost(n: usize) -> usize {
    n.saturating_mul(23)
        .saturating_add(n.saturating_mul(n.saturating_mul(4)))
}

fn judge_budget(entries: usize, what: &str) -> Result<(), AutodiffError> {
    if entries <= JUDGE_BUDGET {
        Ok(())
    } else {
        Err(invalid(&format!(
            "{what}: 二階の総括判定の走査量 {entries} が上限 2^27 を超えるため拒否（保守的な fail-closed）"
        )))
    }
}

/// 構造の演算列を再現する数値型。`f32` は構造そのもの（`V_str`）、[`Rf`] は f64 の参照（`V`・`M`）。
trait Num: Copy {
    /// 定数（ホストが f64 で作り、構造は `as f32` で載せる値）。
    fn k(x: f64) -> Self;
    /// 入力 `x` の値（[`Dm`] だけが依存の大きさとして区別する。ほかは [`Num::k`] と同じ）。
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
    /// と同じ丸め）。実行側のカーネルは和の順序が異なりうるため、[`Eb`] だけは順序に依らない上界で上書きする。
    fn total(terms: &[Self]) -> Self {
        terms.iter().fold(Self::k(0.0), |a, &x| a.plus(x))
    }
    /// `PowScalar` の forward `x^e`（構造は `e as f32` を使う）。
    fn powe(self, e: f64) -> Self;
    /// `PowScalar` の VJP 係数 `e·x^(e−1)`（`e == 0` は 0。`eval/scalar.rs::unary_grad_factor`）。
    fn pow_grad(self, e: f64) -> Self;
    /// `Sqrt` の VJP 係数 `0.5 / y`（`self` は forward の出力 `y`）。
    fn root_grad(self) -> Self;
    /// ±inf を含まない（NaN は入力の NaN の伝播として通す。[`no_inf`] 参照）。
    fn no_inf(self) -> bool;
    /// 値が NaN（[`Rf`] は参照値 `v`）。
    fn nan(self) -> bool;
    /// 演算で生まれた NaN（入力の NaN に依存しない NaN）。f32 は区別できないので常に `false`（f32 だけの
    /// NaN は [`same_nan`] が参照との位置の不一致として拒否する）、[`Rf`] は taint の無い NaN。
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

/// f64 の参照値 `v` と、同じ式を各項の絶対値で評価した大きさ `m`（`m ≥ |v|`。相殺が無ければ `|v|`）。
/// 和・差は `m` を足し、積は `m` を掛ける。商・平方根・累乗は分母・底に**値** `|v|` を使い、分母・底の
/// 相殺は増幅率 `κ = m/|v|` として掛ける（`m` を分母に使うと相殺で判定が緩むため）。累乗 `x^e` が底の
/// 丸め誤差を `|e|` 倍する分は、`loss_class.rs` の Triplet と同じく係数側（[`judge_coeff`] の `amp`）に足す。
/// `t` は入力の NaN に依存すること（taint）。定数が NaN（入力の NaN を載せた値）なら立ち、演算は被演算子の
/// どれかが立っていれば立てる（定数を返す `pow_grad` の `e == 0` は f32 と同じく立てない）。`v` が NaN で
/// `t` が立っていなければ、演算（f64 の `0/0`・`0·inf` 等）で生まれた NaN なので判定で拒否する。
#[derive(Clone, Copy, Debug)]
struct Rf {
    v: f64,
    m: f64,
    t: bool,
}

impl Rf {
    fn kappa(self) -> f64 {
        if self.v == 0.0 {
            1.0
        } else {
            (self.m / self.v.abs()).max(1.0)
        }
    }

    /// 入力の NaN が届いた NaN（taint 付きの NaN）。
    fn input_nan(self) -> bool {
        self.v.is_nan() && self.t
    }

    /// 値と大きさから作る（taint は `t`）。
    fn of(v: f64, m: f64, t: bool) -> Self {
        Rf { v, m, t }
    }
}

impl Num for Rf {
    fn k(x: f64) -> Self {
        Rf::of(x, x.abs(), x.is_nan())
    }
    fn plus(self, o: Self) -> Self {
        Rf::of(self.v + o.v, self.m + o.m, self.t || o.t)
    }
    fn minus(self, o: Self) -> Self {
        Rf::of(self.v - o.v, self.m + o.m, self.t || o.t)
    }
    fn times(self, o: Self) -> Self {
        Rf::of(self.v * o.v, self.m * o.m, self.t || o.t)
    }
    fn over(self, o: Self) -> Self {
        Rf::of(self.v / o.v, self.m / o.v.abs() * o.kappa(), self.t || o.t)
    }
    fn negate(self) -> Self {
        Rf::of(-self.v, self.m, self.t)
    }
    fn root(self) -> Self {
        let v = self.v.sqrt();
        Rf::of(v, v.abs() * self.kappa(), self.t)
    }
    fn powe(self, e: f64) -> Self {
        let v = self.v.powf(e);
        Rf::of(v, v.abs() * self.kappa(), self.t)
    }
    fn pow_grad(self, e: f64) -> Self {
        if e == 0.0 {
            return Rf::of(0.0, 0.0, false);
        }
        let v = e * self.v.powf(e - 1.0);
        Rf::of(v, v.abs() * self.kappa(), self.t)
    }
    fn root_grad(self) -> Self {
        let v = 0.5 / self.v;
        Rf::of(v, v.abs() * self.kappa(), self.t)
    }
    fn no_inf(self) -> bool {
        !self.v.is_infinite() && !self.m.is_infinite()
    }
    fn nan(self) -> bool {
        self.v.is_nan()
    }
    fn generated_nan(self) -> bool {
        self.v.is_nan() && !self.t
    }
}

/// 丸め誤差の一次伝播を丸めの事象ごとに追う数値型（イシュー #3955・決定記録 §77）。値は f64 の参照値 `v`
/// （厳密な定数・入力から組んだ値）と、f32 の実行がそこから外れる誤差の一次式 `Σ cₖ·δₖ`（`δₖ` は k 番目の
/// 丸めの事象の相対誤差。`|δₖ| ≤ μₖ`）で表す。誤差の上界は `Σ|cₖ|·μₖ`（[`Lin::bound`]）。
/// [`Rf`] の「全演算を最終の大きさ `M` に一律に課す」係数 `coeff·M` や、演算ごとに独立な誤差として足す
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
/// 長さの列を値ごとに持つので、グループの幅が [`LIN_REST_MAX`] 以下のときだけ使う（[`wn_group_judge`]）。
/// `PowScalar`（`powe`・`pow_grad`）は WeightNorm が使わないので、上界を持たない NaN の値にする（常に拒否）。
#[derive(Clone, Copy, Debug)]
struct Lin {
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
const LIN_REST_MAX: usize = 8;
/// [`Lin`] で判定するグループ 1 つの走査量の見積り（幅 8 で約 4e5 反復相当の実測。幅の 2 乗に比例）。
fn lin_group_cost(rest: usize) -> usize {
    6000 * rest * rest
}
/// [`Lin`] を使う判定の走査量の上限。超える（グループ数が多い）テンソルは全グループを従来の [`Rf`] で判定する
/// （グループの位置で判定方式が変わらないよう、テンソル単位で選ぶ）。
const LIN_BUDGET: usize = 1 << 26;

/// 領域を空にする（グループの判定の先頭で呼ぶ）。
fn lin_reset() {
    LIN_ARENA.with(|a| *a.borrow_mut() = LinArena::default());
}

/// 領域の現在の大きさ（`(値の数, 事象の数)`）。seed ごとに [`lin_truncate`] で戻すための印。
fn lin_mark() -> (usize, usize) {
    LIN_ARENA.with(|a| {
        let a = a.borrow();
        (a.terms.len(), a.mus.len())
    })
}

fn lin_truncate(mark: (usize, usize)) {
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

/// [`entry_check`] の [`Lin`] 版。`|V_str − V| + 一次の上界·(1 + 16·N·u)` が絶対 1e-5 未満か相対の半分以内で
/// なければ拒否する（tolerance 定数は変えない）。NaN・±inf の扱いは [`entry_check`] と同じ。
fn entry_check_lin(r: Lin, s: f32, what: &str) -> Result<(), AutodiffError> {
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
    Err(invalid(&format!(
        "{what}: 二階の要素が f32 構造の丸め・相殺・underflow・overflow で REQ-2 の判定を外れうるため拒否"
    )))
}
/// 入力の NaN の依存を追う数値型（[`lrn_col_scan`] 等の走査用）。`nan` は値が入力の NaN で NaN になること、
/// `mx` はその値が依存する入力 `x`（[`Num::input`] で載せた値）の最大絶対値。NaN の伝わり方は構造の f32 の
/// 演算と同じにする: 四則・`√` は被演算子のどれかが NaN なら NaN（`0·NaN = NaN` なので定数 0 も特別扱い
/// しない）、NaN を消すのは冪だけで、`powe` の指数 0（`NaN^0 = 1`）と `pow_grad` の指数 0（定数 0）・指数 1
/// （`e·x^0 = e`）の定数になる場合。条件は f32 側と同じく `e as f32` で判定する。値の大きさ（overflow）は
/// 見ない（演算で生まれた非有限は f32 の再現と参照の判定が拒否する）。
#[derive(Clone, Copy, Debug)]
struct Dm {
    nan: bool,
    mx: f64,
}

impl Dm {
    fn join(self, o: Self) -> Self {
        Dm {
            nan: self.nan || o.nan,
            mx: self.mx.max(o.mx),
        }
    }

    const CONST: Dm = Dm {
        nan: false,
        mx: 0.0,
    };
}

impl Num for Dm {
    fn k(x: f64) -> Self {
        Dm {
            nan: x.is_nan(),
            mx: 0.0,
        }
    }
    fn input(x: f64) -> Self {
        Dm {
            nan: x.is_nan(),
            mx: if x.is_nan() { 0.0 } else { x.abs() },
        }
    }
    fn plus(self, o: Self) -> Self {
        self.join(o)
    }
    fn minus(self, o: Self) -> Self {
        self.join(o)
    }
    fn times(self, o: Self) -> Self {
        self.join(o)
    }
    fn over(self, o: Self) -> Self {
        self.join(o)
    }
    fn negate(self) -> Self {
        self
    }
    fn root(self) -> Self {
        self
    }
    fn powe(self, e: f64) -> Self {
        if (e as f32) == 0.0 { Dm::CONST } else { self }
    }
    fn pow_grad(self, e: f64) -> Self {
        let e = e as f32;
        if e == 0.0 || e - 1.0 == 0.0 {
            Dm::CONST
        } else {
            self
        }
    }
    fn root_grad(self) -> Self {
        self
    }
    fn no_inf(self) -> bool {
        true
    }
    fn nan(self) -> bool {
        self.nan
    }
    fn generated_nan(self) -> bool {
        false
    }
}

/// 総括判定（第 2 軸の本体）の丸め誤差係数 `(len + depth + amp)·u`（`u = 2^-24`）。二階の 1 項が通る
/// 丸めの回数を一次で数えたもの: `len` は縮約（forward の和・逆伝播の蓄積）の長さと、和から作った値
/// （ノルム・σ_s 等）が同じ項に何度も現れる分の合計、`depth` はそれ以外の演算の段数（定数の丸めを含む）、
/// `amp` は累乗が底の丸め誤差を `|e|` 倍する分（LRN だけ）。`loss_class.rs` の `(2L + 4)·2^-23` と同じ水準
/// の見積りで、分母・底の相殺の増幅は [`Rf`] の `m` 側（`κ`）に入る。
fn judge_coeff(len: usize, depth: usize, amp: f64) -> f64 {
    (len as f64 + depth as f64 + amp) * (-24f64).exp2()
}

/// LRN の係数。窓和 `sw`・`sz` と逆伝播の窓の蓄積（各 `nt`）、段数は forward 約 12・逆伝播約 12 と
/// チャンクの `Mul`（積と定数の丸め）。`a = d^{-β}`・`b = d^{-β-1}` とその係数は `d` までの丸め
/// （窓和と 2 乗・ρ 倍・kk の加算・定数の丸めで `nt_f + 4`）を `|β| + 1`・`|β| + 2` 倍し、1 項に `b` と
/// その係数が並ぶので `(2|β| + 4)·(nt_f + 4)`。
fn lrn_coeff(nt_f: usize, nt_r: usize, nchunks: usize, beta: f64) -> f64 {
    judge_coeff(
        2 * (nt_f + nt_r),
        24 + 2 * nchunks,
        (2.0 * beta.abs() + 4.0) * (nt_f as f64 + 4.0),
    )
}

/// WeightNorm の係数。`n_s = √Σv_s²`（丸め `rest/2 + 1`）が 1 項に最大 5 回（`g/n_s`・`v̂`・`-(v_s/n_s)/n_s`・
/// `0.5/n_s`）現れ、`Σup⊙v̂` と逆伝播の `n_s` への蓄積（各 `rest`）が加わるので `len = 5·rest`。
fn wn_coeff(rest: usize) -> f64 {
    judge_coeff(5 * rest, 24, 0.0)
}

/// SpectralNorm の係数。`σ_s = ΣW_s⊙P`（丸め `N`）が 1 項に最大 4 回（`1/σ_s`・`Ŵ`・2 つの Div 係数）
/// 現れ、`Σup⊙Ŵ` と逆伝播の `σ_s` への蓄積（各 `N`）が加わるので `len = 6·N`。σ_s の相殺は `κ` で入る。
fn sn_coeff(n: usize) -> f64 {
    judge_coeff(6 * n, 24, 0.0)
}

/// forward の再現の値・VJP 係数に ±inf が無いこと。±inf は `0·inf`・`inf − inf` で入力に無い NaN を作る
/// ので拒否する。NaN は入力の NaN の伝播として通し、[`same_nan`] で f32 と参照の NaN の位置が一致する
/// ことを確かめる（[`entry_check`] は両側が NaN の要素だけを外す）。
fn no_inf<T: Num>(vals: &[T]) -> bool {
    vals.iter().all(|v| v.no_inf())
}

/// f32 の再現と参照（[`Rf`]）の forward の値・VJP 係数で、NaN の位置が一致し、どの NaN も入力の NaN が
/// 届いたもの（taint 付き）であること。f32 だけの NaN（負の底の `powf`・`0/0` 等）や参照の演算で生まれた
/// NaN は、逆伝播の `0·NaN` で依存しない二階の要素まで NaN にするので、forward の時点で拒否する。
fn same_nan<A: Num, B: Num>(a: &[A], b: &[B]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.nan() == y.nan() && !x.generated_nan() && !y.generated_nan())
}

/// ホストの 1 階（V3 で `cg.grad` に載る値）の非有限の検査。3 Op の全分岐（LRN の `α = 0` を含む）で、
/// 構造を積む前に呼ぶ。非有限の要素は、NaN で、かつ入力の NaN が届く位置（`reach`。Op ごとのホスト VJP の
/// 依存範囲）にあるものだけを伝播として通す。それ以外（f32 の積の overflow で生まれた inf、`0·inf` で
/// 生まれた NaN 等。有限の入力から演算で生まれた非有限）は型付きで拒否する。
fn host_nonfinite_check(host: &[f32], reach: &[bool], what: &str) -> Result<(), AutodiffError> {
    if host.len() != reach.len() {
        return Err(invalid(&format!(
            "{what}: ホストの 1 階と到達範囲の要素数が不一致"
        )));
    }
    for (&h, &r) in host.iter().zip(reach) {
        if !h.is_finite() && !(h.is_nan() && r) {
            return Err(invalid(&format!(
                "{what}: ホストの 1 階に、入力の NaN が届かない位置で演算で生まれた非有限（f32 の積の \
                 overflow・0·inf 等）がある"
            )));
        }
    }
    Ok(())
}

/// LRN の NaN の位置（列ごとに軸 1 の NaN の個数の累積和で求める）。`reach` はホストの 1 階 `dx_m` に入力の
/// NaN が届く要素（窓 2 段の和 `[m − (size−1), m + (size−1)]` に NaN がある。forward 窓の分け方によらない
/// 上界。[`host_nonfinite_check`] が使う）、`col_nan` は NaN を含む列（[`lrn_col_scan`] で二階の要素ごとの
/// 依存を走査する列）。
fn lrn_nan_masks(
    x: &[f32],
    (n, c, inner): (usize, usize, usize),
    size: usize,
) -> (Vec<bool>, Vec<bool>) {
    let mut reach = vec![false; x.len()];
    let mut col_nan = vec![false; n * inner];
    let mut pre = vec![0usize; c + 1];
    for ni in 0..n {
        for s in 0..inner {
            for ci in 0..c {
                pre[ci + 1] = pre[ci] + usize::from(x[(ni * c + ci) * inner + s].is_nan());
            }
            if pre[c] == 0 {
                continue;
            }
            col_nan[ni * inner + s] = true;
            let has = |lo: usize, hi: usize| pre[hi.min(c - 1) + 1] > pre[lo];
            for ci in 0..c {
                reach[(ni * c + ci) * inner + s] =
                    has(ci.saturating_sub(size - 1), ci.saturating_add(size - 1));
            }
        }
    }
    (reach, col_nan)
}

fn nan_mismatch(what: &str) -> AutodiffError {
    invalid(&format!(
        "{what}: 構造の f32 の forward 値または VJP 係数に、入力の NaN に依存しない NaN がある"
    ))
}

/// 二階の 1 要素の判定（`loss_class.rs::struct_check` と同じ形。scale は 1）。`r` は参照値 `V` と大きさ
/// `M`、`s` は構造の f32 の再現値 `V_str`。`(|V_str − V| + coeff·M)` が絶対 1e-5 未満か相対の半分以内で
/// なければ拒否する（tolerance 定数は変えない）。外すのは f32 の再現と参照の**両方が NaN** で、参照の NaN が
/// 入力の NaN の届いたもの（taint 付き）の要素だけ（NaN は伝播させる契約）。片側だけ NaN・演算で生まれた
/// NaN・どちらかが ±inf は拒否する。
fn entry_check(r: Rf, s: f32, coeff: f64, what: &str) -> Result<(), AutodiffError> {
    if s.is_nan() && r.input_nan() {
        return Ok(());
    }
    let err = (f64::from(s) - r.v).abs() + coeff * r.m;
    if r.v.is_finite()
        && r.m.is_finite()
        && err.is_finite()
        && (err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * r.v.abs())
    {
        return Ok(());
    }
    Err(invalid(&format!(
        "{what}: 二階の要素が f32 構造の丸め・相殺・underflow・overflow で REQ-2 の判定を外れうるため拒否"
    )))
}

fn nonfinite_struct(what: &str) -> AutodiffError {
    invalid(&format!(
        "{what}: 構造の f32 の forward 値または VJP 係数が非有限（二階で 0·inf = NaN を作る）"
    ))
}

// ---------------------------------------------------------------------
// 入口検査（validate_ancestors から呼ぶ。子テープへ書く前に拒否する）
// ---------------------------------------------------------------------

/// `validate_ancestors` から `requires_grad == true` の非葉ノードごとに呼ばれる入口検査。
/// 3 Op 以外では何もしない。
pub(super) fn validate_norm_reparam(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    op: &Op,
) -> Result<(), AutodiffError> {
    match op {
        Op::LocalResponseNorm { input, params } => {
            validate_lrn(parent_nodes, parent_ops, *input, params)
        }
        Op::WeightNorm { v, g, dim } => {
            validate_weight_norm(parent_nodes, parent_ops, *v, *g, *dim)
        }
        Op::SpectralNorm { weight, u, v, dim } => {
            validate_spectral(parent_nodes, parent_ops, *weight, u, v, *dim)
        }
        _ => Ok(()),
    }
}

/// `[N, C, *S]` の `(n, c, inner)`。
fn lrn_dims(shape: &[usize]) -> Result<(usize, usize, usize), AutodiffError> {
    if shape.len() < 3 {
        return Err(invalid("LocalResponseNorm の入力が rank 3 未満"));
    }
    Ok((shape[0], shape[1], shape[2..].iter().product()))
}

fn validate_lrn(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    input: NodeId,
    params: &LrnParams,
) -> Result<(), AutodiffError> {
    let x = materialize_fallible(parent_nodes, parent_ops, input)?;
    if x.shape().contains(&0) {
        return Ok(());
    }
    let (n, c, inner) = lrn_dims(x.shape())?;
    let data = dense_vec(x);
    if data.iter().any(|v| v.is_infinite()) {
        return Err(invalid(
            "LocalResponseNorm の入力に ±inf がある（NaN は伝播するが ±inf は拒否する）",
        ));
    }
    let size = params.size();
    let rho = f64::from(params.alpha()) / size as f64;
    let k = f64::from(params.k());
    for ni in 0..n {
        for s in 0..inner {
            for ci in 0..c {
                let lo = ci.saturating_sub(size / 2);
                let hi = (ci + (size - 1) / 2).min(c - 1);
                let sum: f64 = (lo..=hi)
                    .map(|cj| {
                        let v = f64::from(data[(ni * c + cj) * inner + s]);
                        v * v
                    })
                    .sum();
                let d = k + rho * sum;
                if d.is_nan() {
                    continue;
                }
                if !d.is_finite() || d <= 0.0 {
                    return Err(invalid(
                        "LocalResponseNorm の窓内統計 D = k + (α/size)·Σx² が 0 以下（または非有限）",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_weight_norm(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    v: NodeId,
    g: NodeId,
    dim: Option<usize>,
) -> Result<(), AutodiffError> {
    let v_val = materialize_fallible(parent_nodes, parent_ops, v)?;
    let g_val = materialize_fallible(parent_nodes, parent_ops, g)?;
    if v_val.shape().contains(&0) {
        return Ok(());
    }
    let gm = GroupMap::new(v_val.shape(), dim)?;
    let v_data = dense_vec(v_val);
    if v_data.iter().any(|x| x.is_infinite()) || dense_vec(g_val).iter().any(|x| x.is_infinite()) {
        return Err(invalid(
            "WeightNorm の v／g に ±inf がある（NaN は伝播するが ±inf は拒否する）",
        ));
    }
    let v_gm = gm.to_group_major(&v_data);
    for grp in v_gm.chunks(gm.rest) {
        if grp.iter().any(|x| x.is_nan()) {
            continue;
        }
        if grp.iter().all(|&x| x == 0.0) {
            return Err(invalid("WeightNorm にゼロノルムのグループがある"));
        }
    }
    Ok(())
}

fn validate_spectral(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    weight: NodeId,
    u: &Tensor<f32>,
    v: &Tensor<f32>,
    dim: usize,
) -> Result<(), AutodiffError> {
    let w = materialize_fallible(parent_nodes, parent_ops, weight)?;
    let layout = wr::spectral_norm_layout(w.shape(), dim).map_err(AutodiffError::Shape)?;
    let data = dense_vec(w);
    if data.iter().any(|x| x.is_infinite()) {
        return Err(invalid(
            "SpectralNorm の重みに ±inf がある（NaN は伝播するが ±inf は拒否する）",
        ));
    }
    if data.iter().any(|x| x.is_nan()) {
        return Ok(());
    }
    let sigma = wr::spectral_sigma_host(&data, &layout, &dense_vec(u), &dense_vec(v))
        .map_err(AutodiffError::Shape)?;
    if sigma == 0.0 || !sigma.is_finite() {
        return Err(invalid("SpectralNorm の σ が 0 または非有限"));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// LocalResponseNorm
// ---------------------------------------------------------------------

/// `window_sum(v, c, back, fwd)` = 軸 1 に沿った `Σ_{o=-back}^{fwd} v[c+o]`（範囲外は 0）。
/// 窓のオフセット範囲は `[-(C-1), C-1]` との交差に打ち切る（`size > C` でも最大 `2C-1` 回）。
fn window_sum<'c>(
    v: &Var<'c>,
    c: usize,
    back: usize,
    fwd: usize,
) -> Result<Var<'c>, AutodiffError> {
    let back = back.min(c - 1);
    let fwd = fwd.min(c - 1);
    let padded = if back + fwd == 0 {
        *v
    } else {
        v.pad(&[(0, 0), (back, fwd), (0, 0)], 0.0)?
    };
    let mut acc = padded.narrow(1, 0, c)?;
    for start in 1..=(back + fwd) {
        acc = acc.add(&padded.narrow(1, start, c)?)?;
    }
    Ok(acc)
}

/// 窓幅 `size`・チャネル数 `c` のときの窓内ずらし回数。
fn window_terms(c: usize, back: usize, fwd: usize) -> usize {
    back.min(c - 1) + fwd.min(c - 1) + 1
}

/// `2^{L_lane}` を ±2^100 以下のチャンク列へ分ける（先頭から順に `Mul` する）。値は f64 の
/// `2^part`（構造は `as f32` で載せ、総括判定の参照 [`Rf`] は f64 のまま使う）。
fn chunk_factors(l: &[f64]) -> Vec<Vec<f64>> {
    let mut rem = l.to_vec();
    let mut out = Vec::new();
    for _ in 0..8 {
        if rem.iter().all(|r| r.abs() < 1e-12) {
            break;
        }
        let part: Vec<f64> = rem.iter().map(|r| r.clamp(-100.0, 100.0)).collect();
        out.push(part.iter().map(|p| 2f64.powf(*p)).collect());
        for (r, p) in rem.iter_mut().zip(&part) {
            *r -= p;
        }
    }
    out
}

/// LRN の窓（`window_sum` と同じく `C − 1` に打ち切ったずらし幅）。`f` は前方窓 `W`、`r` は反転窓 `W'`。
#[derive(Clone, Copy)]
struct LrnWin {
    bf: usize,
    ff: usize,
    br: usize,
    fr: usize,
}

/// 構造の窓和 `Σ_{o=-back}^{fwd} v[i+o]`（範囲外は 0）を `window_sum` と同じ加算順で再現する。
fn win_sum<T: Num>(v: &[T], i: usize, back: usize, fwd: usize) -> T {
    let at = |o: usize| -> T {
        let idx = i + o;
        if idx >= back && idx - back < v.len() {
            v[idx - back]
        } else {
            T::k(0.0)
        }
    };
    let mut acc = at(0);
    for o in 1..=(back + fwd) {
        acc = acc.plus(at(o));
    }
    acc
}

/// LRN の 1 列（`(n, inner)`）の構造の forward（二階の逆伝播が使う中間値と VJP 係数）。
struct LrnColFwd<T> {
    xs: Vec<T>,
    up: Vec<T>,
    a: Vec<T>,
    fa: Vec<T>,
    fb: Vec<T>,
    b: Vec<T>,
    ux: Vec<T>,
    sz: Vec<T>,
    inv_s: T,
    rho: T,
    c2: T,
    /// seed 1 をチャンクの `Mul` の逆順にたどった `term1 − term2` への cotangent（`s^{-2β}` の復元）。
    ci: T,
}

/// LRN の列の定数（ホストの f64 値）。
struct LrnColConst<'a> {
    s: f64,
    kk: f64,
    rho: f64,
    beta: f64,
    chunks: &'a [f64],
}

/// `lrn_cgrad` が積む構造（`x_s = x/s` 〜 `dx = (term1 − term2)·chunk…`）の forward を再現する。
/// f32 の forward 値・VJP 係数のどれかが ±inf なら `None`（NaN は伝播として通す。[`no_inf`]）。
fn lrn_col_fwd<T: Num>(
    x: &[f32],
    up: &[f32],
    k: &LrnColConst<'_>,
    w: LrnWin,
) -> Option<LrnColFwd<T>> {
    let s = T::k(k.s);
    let inv_s = T::k(1.0).over(s);
    let xs: Vec<T> = x.iter().map(|&v| T::input(f64::from(v)).over(s)).collect();
    let up: Vec<T> = up.iter().map(|&v| T::k(f64::from(v))).collect();
    let sq: Vec<T> = xs.iter().map(|&v| v.times(v)).collect();
    let rho = T::k(k.rho);
    let kk = T::k(k.kk);
    let c = x.len();
    let d: Vec<T> = (0..c)
        .map(|i| win_sum(&sq, i, w.bf, w.ff).times(rho).plus(kk))
        .collect();
    let a: Vec<T> = d.iter().map(|v| v.powe(-k.beta)).collect();
    let b: Vec<T> = d.iter().map(|v| v.powe(-k.beta - 1.0)).collect();
    let fa: Vec<T> = d.iter().map(|v| v.pow_grad(-k.beta)).collect();
    let fb: Vec<T> = d.iter().map(|v| v.pow_grad(-k.beta - 1.0)).collect();
    let ux: Vec<T> = up.iter().zip(&xs).map(|(&u, &v)| u.times(v)).collect();
    let z: Vec<T> = ux.iter().zip(&b).map(|(&u, &v)| u.times(v)).collect();
    let sz: Vec<T> = (0..c).map(|i| win_sum(&z, i, w.br, w.fr)).collect();
    let c2 = T::k(2.0 * k.rho * k.beta);
    let mut dx: Vec<T> = (0..c)
        .map(|i| {
            let t1 = up[i].times(a[i]);
            let t2 = xs[i].times(sz[i]).times(c2);
            t1.minus(t2)
        })
        .collect();
    let mut fwd_ok = no_inf(&dx);
    for &ch in k.chunks {
        for v in &mut dx {
            *v = v.times(T::k(ch));
        }
        fwd_ok &= no_inf(&dx);
    }
    let mut ci = T::k(1.0);
    for &ch in k.chunks.iter().rev() {
        ci = ci.times(T::k(ch));
    }
    // `sq = x_s²` は ±inf の検査に入れない。`sq` の値は逆伝播で係数として使われず（`Mul` の VJP は `x_s` を
    // 使う）、`0·inf` を作らない。消費先の `d` は検査するので、有限の `d` へ届く `sq` の inf は `d` の inf と
    // して拒否される。`d` が NaN の窓にしか入らない要素（有限の出力が依存しない値。スケールで守られない）の
    // `sq` の inf は NaN に吸収され、二階へ届かない。
    let ok = fwd_ok
        && inv_s.no_inf()
        && ci.no_inf()
        && [&xs, &d, &a, &b, &fa, &fb, &ux, &z, &sz]
            .iter()
            .all(|v| no_inf(v));
    ok.then_some(LrnColFwd {
        xs,
        up,
        a,
        fa,
        fb,
        b,
        ux,
        sz,
        inv_s,
        rho,
        c2,
        ci,
    })
}

/// LRN の逆伝播の作業領域（列長。seed ごとに触る範囲だけ 0 へ戻す）。
struct LrnScratch<T> {
    xs: Vec<T>,
    d: Vec<T>,
    sq: Vec<T>,
    up: Vec<T>,
}

impl<T: Num> LrnScratch<T> {
    fn new(c: usize) -> Self {
        Self {
            xs: vec![T::k(0.0); c],
            d: vec![T::k(0.0); c],
            sq: vec![T::k(0.0); c],
            up: vec![T::k(0.0); c],
        }
    }
}

/// seed `e_j`（`dx_j` に 1）の二階の逆伝播を構造の演算順で再現する。`x` への要素は `[xlo, xhi]`、
/// `up` への要素は `[ulo, uhi]` に入り（それ以外は 0）、`sc.xs`（`x` への値に変換済み）・`sc.up` に残る。
fn lrn_col_rev<T: Num>(f: &LrnColFwd<T>, w: LrnWin, j: usize, sc: &mut LrnScratch<T>) {
    let c = f.xs.len();
    let (ulo, uhi) = (j.saturating_sub(w.br), (j + w.fr).min(c - 1));
    let (xlo, xhi) = (ulo.saturating_sub(w.bf), (uhi + w.ff).min(c - 1));
    for l in xlo..=xhi {
        sc.xs[l] = T::k(0.0);
        sc.d[l] = T::k(0.0);
        sc.sq[l] = T::k(0.0);
        sc.up[l] = T::k(0.0);
    }
    // dx = term1 − term2（Sub の係数 1・−1 は厳密）。term2 = (x_s·sz)·c2。
    let cot_t1 = f.ci;
    let cot_t2a = f.ci.negate().times(f.c2);
    sc.xs[j] = sc.xs[j].plus(cot_t2a.times(f.sz[j]));
    let cot_sz = cot_t2a.times(f.xs[j]);
    // sz_j = Σ_{m ∈ W'(j)} z_m、z = (up·x_s)·b。
    for m in ulo..=uhi {
        let cot_ux = cot_sz.times(f.b[m]);
        let cot_b = cot_sz.times(f.ux[m]);
        sc.up[m] = sc.up[m].plus(cot_ux.times(f.xs[m]));
        sc.xs[m] = sc.xs[m].plus(cot_ux.times(f.up[m]));
        sc.d[m] = sc.d[m].plus(cot_b.times(f.fb[m]));
    }
    // term1 = up·a、a = d^{-β}。
    sc.up[j] = sc.up[j].plus(cot_t1.times(f.a[j]));
    let cot_a = cot_t1.times(f.up[j]);
    sc.d[j] = sc.d[j].plus(cot_a.times(f.fa[j]));
    // d = sw·ρ + kk、sw_m = Σ_{l ∈ W(m)} x_s,l²。
    for m in ulo..=uhi {
        let cot_sw = sc.d[m].times(f.rho);
        let lo = m.saturating_sub(w.bf);
        let hi = (m + w.ff).min(c - 1);
        for l in lo..=hi {
            sc.sq[l] = sc.sq[l].plus(cot_sw);
        }
    }
    // sq = x_s·x_s（同じ入力へ 2 回）、x_s = x/s。
    for l in xlo..=xhi {
        let g = sc.sq[l].times(f.xs[l]);
        sc.xs[l] = sc.xs[l].plus(g).plus(g).times(f.inv_s);
    }
}

/// LRN の 1 列の総括判定。seed が 1 要素のときの二階（`x` と `up` への要素）を [`entry_check`] に掛ける。
fn lrn_col_judge(
    x: &[f32],
    up: &[f32],
    k: &LrnColConst<'_>,
    w: LrnWin,
    coeff: f64,
) -> Result<(), AutodiffError> {
    const WHAT: &str = "LocalResponseNorm の二階";
    let (Some(fs), Some(fr)) = (
        lrn_col_fwd::<f32>(x, up, k, w),
        lrn_col_fwd::<Rf>(x, up, k, w),
    ) else {
        return Err(nonfinite_struct(WHAT));
    };
    let nan_ok = [
        (&fs.xs, &fr.xs),
        (&fs.up, &fr.up),
        (&fs.a, &fr.a),
        (&fs.fa, &fr.fa),
        (&fs.fb, &fr.fb),
        (&fs.b, &fr.b),
        (&fs.ux, &fr.ux),
        (&fs.sz, &fr.sz),
    ]
    .iter()
    .all(|(a, b)| same_nan(a, b))
        && same_nan(
            &[fs.inv_s, fs.rho, fs.c2, fs.ci],
            &[fr.inv_s, fr.rho, fr.c2, fr.ci],
        );
    if !nan_ok {
        return Err(nan_mismatch(WHAT));
    }
    let c = x.len();
    let (mut ss, mut sr) = (LrnScratch::<f32>::new(c), LrnScratch::<Rf>::new(c));
    for j in 0..c {
        lrn_col_rev(&fs, w, j, &mut ss);
        lrn_col_rev(&fr, w, j, &mut sr);
        let (ulo, uhi) = (j.saturating_sub(w.br), (j + w.fr).min(c - 1));
        let (xlo, xhi) = (ulo.saturating_sub(w.bf), (uhi + w.ff).min(c - 1));
        for l in xlo..=xhi {
            entry_check(sr.xs[l], ss.xs[l], coeff, WHAT)?;
        }
        for m in ulo..=uhi {
            entry_check(sr.up[m], ss.up[m], coeff, WHAT)?;
        }
    }
    Ok(())
}

/// NaN を含む LRN の列の依存の走査（[`Dm`]。構造の forward と、seed ごとの二階の逆伝播を [`lrn_col_fwd`]・
/// [`lrn_col_rev`] で回す）。1 階 `dx` と二階の各要素（帯の内側。帯の外は子テープで 0 か NaN）のうち、入力の
/// NaN で NaN にならない要素（有限の出力）があるかと、それらが依存する入力 `x` の最大絶対値を返す。NaN の
/// 伝わり方は値に依らないので、スケール等の定数は 1 で回す。
fn lrn_col_scan(x: &[f32], up: &[f32], w: LrnWin, beta: f64) -> (bool, f64) {
    let k = LrnColConst {
        s: 1.0,
        kk: 1.0,
        rho: 1.0,
        beta,
        chunks: &[],
    };
    let mut fin = false;
    let mut mx = 0.0f64;
    let mut note = |d: Dm| {
        if !d.nan {
            fin = true;
            mx = mx.max(d.mx);
        }
    };
    let Some(f) = lrn_col_fwd::<Dm>(x, up, &k, w) else {
        // `Dm` は ±inf を持たないので到達しない。安全側として有限の出力ありに倒す。
        return (true, f64::INFINITY);
    };
    let c = x.len();
    for i in 0..c {
        note(f.up[i].times(f.a[i]).minus(f.xs[i].times(f.sz[i])));
    }
    let mut sc = LrnScratch::<Dm>::new(c);
    for j in 0..c {
        lrn_col_rev(&f, w, j, &mut sc);
        let (ulo, uhi) = (j.saturating_sub(w.br), (j + w.fr).min(c - 1));
        let (xlo, xhi) = (ulo.saturating_sub(w.bf), (uhi + w.ff).min(c - 1));
        for l in xlo..=xhi {
            note(sc.xs[l]);
        }
        for m in ulo..=uhi {
            note(sc.up[m]);
        }
    }
    (fin, mx)
}

/// `LocalResponseNorm` ノード（入力 `input`）の d_input を子テープへ記録する。
pub(super) fn lrn_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    x_m: &Var<'c>,
    input: NodeId,
    params: &LrnParams,
    up: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let x_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let shape = x_val.shape().to_vec();
    if up.shape() != shape || x_m.shape() != shape {
        return Err(invalid("LocalResponseNorm の shape 不整合（内部契約違反）"));
    }
    if shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&shape)?));
    }
    let layout = lrn::lrn_layout(&shape).map_err(AutodiffError::Shape)?;
    let (n, c, inner) = lrn_dims(&shape)?;
    let up_data = dense_vec(&child_value(child, up)?);
    check_finite(&up_data, "上流 g")?;
    let x_data = dense_vec(x_val);
    let host_vec = lrn::local_response_norm_vjp_host(&x_data, &up_data, &layout, params)
        .map_err(AutodiffError::Shape)?;
    let host = Tensor::new(host_vec.clone(), &shape)?;

    let size = params.size();
    let beta = f64::from(params.beta());
    let alpha = f64::from(params.alpha());
    let k = f64::from(params.k());
    let u_max = up_data
        .iter()
        .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    // ホストの 1 階の非有限は、入力の NaN が届く位置の NaN だけを通す（全分岐の前。`α = 0` でもホストは
    // 一般式で `up·x` の f32 積を作るので、有限の入力から inf・`0·inf = NaN` が生まれうる）。
    let (reach, col_nan) = lrn_nan_masks(&x_data, (n, c, inner), size);
    host_nonfinite_check(&host_vec, &reach, "LocalResponseNorm")?;

    if alpha == 0.0 {
        // d = k なので dx = g·k^{-β}（x に依存しない）。
        let f = k.powf(-beta);
        stage(f, "LRN(α=0) の k^{-β}")?;
        stage(f * u_max, "LRN(α=0) の g·k^{-β}")?;
        // 総括判定（第 2 軸）: 二階は up への対角 `k^{-β}` だけで、構造は f32 定数として載せる。
        entry_check(
            Rf::k(f),
            f as f32,
            rounding_coeff(4),
            "LocalResponseNorm(α=0) の二階",
        )?;
        let composed = up.mul(&scalar_const(child, f as f32)?)?;
        // 構造の 1 階をホスト値と照合する（ほかの分岐と同じ。ホストが NaN の要素は上で到達範囲を確認済み）。
        let m: Vec<f64> = up_data.iter().map(|&u| f64::from(u).abs() * f).collect();
        shadow_check(
            child,
            &composed,
            &host_vec,
            &m,
            rounding_coeff(4),
            "LocalResponseNorm(α=0)",
        )?;
        return push_v3(child, &composed, host);
    }

    let rho = alpha / size as f64;
    let floor = (k.abs() * size as f64 / alpha.abs()).sqrt();
    let (back_f, fwd_f) = (size / 2, (size - 1) / 2);
    let (back_r, fwd_r) = ((size - 1) / 2, size / 2);
    let (nt_f, nt_r) = (
        window_terms(c, back_f, fwd_f),
        window_terms(c, back_r, fwd_r),
    );
    let at = |ni: usize, ci: usize, s: usize| (ni * c + ci) * inner + s;

    let win = LrnWin {
        bf: back_f.min(c - 1),
        ff: fwd_f.min(c - 1),
        br: back_r.min(c - 1),
        fr: fwd_r.min(c - 1),
    };
    let ncols = n * inner;
    let col_vals = |col: usize, data: &[f32]| -> Vec<f32> {
        let (ni, s) = (col / inner, col % inner);
        (0..c).map(|ci| data[at(ni, ci, s)]).collect()
    };
    // NaN を含む列は、二階の要素ごとの依存を走査する（[`lrn_col_scan`]）。除外の単位は列や窓ではなく
    // 二階の出力要素で、入力の NaN で NaN になると構造上決まる要素だけを外す（§57.2）。走査は判定と同じ
    // 大きさなので、走査の前に走査分の走査量を検査する。
    let n_nan = col_nan.iter().filter(|&&b| b).count();
    let scan_cost = lrn_judge_cost(n_nan, c, win, 0);
    judge_budget(scan_cost, "LocalResponseNorm")?;
    // 列ごとの（有限の出力があるか, 有限の出力が依存する x の最大絶対値）。NaN の無い列は全要素が有限の
    // 出力に依存しうるので、列の最大絶対値。
    let col_dep: Vec<(bool, f64)> = (0..ncols)
        .map(|col| {
            let xc = col_vals(col, &x_data);
            if col_nan[col] {
                lrn_col_scan(&xc, &col_vals(col, &up_data), win, beta)
            } else {
                (
                    true,
                    xc.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs())),
                )
            }
        })
        .collect();
    // 1 階・二階の全要素が入力の NaN で NaN になる列（有限の出力が無い）は外す。守る有限の要素が無いので、
    // スケールの段検査を含むすべての段検査と判定より前に外し、指数 0 としてチャンク数・係数・構造の定数へ
    // 影響させない。
    let col_skip: Vec<bool> = col_dep.iter().map(|&(fin, _)| !fin).collect();

    // 列 (n, inner) ごとのスケール指数。最大値は有限の出力が依存する入力 `x` だけから取る（NaN の要素と、
    // NaN の経路にしか入らない値は有限の出力に届かないので外す。反転窓・偶数 `size` の窓のずれ・冪 0 で
    // NaN が消える経路も [`lrn_col_scan`] の依存に入る）。
    let mut exps = vec![0i32; ncols];
    for (col, &(fin, mx)) in col_dep.iter().enumerate() {
        if fin {
            exps[col] = pow2_ceil_exponent(mx.max(floor));
        }
    }
    let scales: Vec<f32> = exps.iter().map(|&e| pow2(e)).collect();
    let kk: Vec<f32> = exps
        .iter()
        .map(|&e| (k / 2f64.powi(2 * e)) as f32)
        .collect();
    let l_exp: Vec<f64> = exps.iter().map(|&e| -2.0 * beta * f64::from(e)).collect();
    let chunks = chunk_factors(&l_exp);
    // 総括判定（第 2 軸）の係数（[`lrn_coeff`]）・走査量。走査量は、走査した分と判定する列（外す列を除く）の
    // 分の和。
    let coeff_judge = lrn_coeff(nt_f, nt_r, chunks.len(), beta);
    let ncols_judged = col_skip.iter().filter(|&&sk| !sk).count();
    judge_budget(
        scan_cost.saturating_add(lrn_judge_cost(ncols_judged, c, win, chunks.len())),
        "LocalResponseNorm",
    )?;

    // ホスト側の f64 参照（構造の各段を再現）と第 1 軸の判定、第 2 軸の m の計算。
    let xs: Vec<f32> = (0..x_data.len())
        .map(|i| {
            let (ni, rest) = (i / (c * inner), i % (c * inner));
            x_data[i] / scales[ni * inner + rest % inner]
        })
        .collect();
    let mut m_abs = vec![0.0f64; x_data.len()];
    for ni in 0..n {
        for s in 0..inner {
            let col = ni * inner + s;
            let col_idx: Vec<usize> = (0..c).map(|ci| at(ni, ci, s)).collect();
            // 有限の出力が無い列は、段検査（`s^{-2β}` を含む）より前に外す。`m` は NaN にして、ホストが有限
            // なら照合で拒否する側へ倒す。
            if col_skip[col] {
                for &i in &col_idx {
                    m_abs[i] = f64::NAN;
                }
                continue;
            }
            // NaN の入力は窓和 `d` を通って、窓が NaN を含むチャネルに届く。NaN の `d` は伝播として通し、
            // 有限の `d` だけを検査する。段の上界（`fold` の max／min は NaN を無視する。`NaN^0 = 1` は f64 の
            // `powf` でも 1）は有限の値の上界になる。
            let mut d = vec![0.0f64; c];
            for (ci, dc) in d.iter_mut().enumerate() {
                let lo = ci.saturating_sub(back_f);
                let hi = (ci + fwd_f).min(c - 1);
                let sum: f64 = (lo..=hi).map(|cj| f64::from(xs[col_idx[cj]]).powi(2)).sum();
                *dc = f64::from(kk[col]) + rho * sum;
            }
            // 有限の出力が残るのに有限の `d_s` が 1 つも無い列（β ∈ {0, −1, −2} で、冪の指数 0 により NaN が
            // 消える経路だけが残る列）は、段の上界（`d_s` の最小値）を定められないので拒否する（fail-closed）。
            if d.iter().all(|dc| dc.is_nan()) {
                return Err(invalid(
                    "LocalResponseNorm: NaN を含む列で、有限の d_s が無いのに有限の二階の経路が残る（冪の指数 0 で \
                     NaN が消える経路）ため拒否",
                ));
            }
            let big_f = l_exp[col].exp2();
            stage(big_f, "LRN の s^{-2β}")?;
            if d.iter()
                .any(|&dc| !dc.is_nan() && (dc <= 0.0 || (dc as f32) < f32::MIN_POSITIVE))
            {
                return Err(invalid(
                    "LocalResponseNorm のスケール後統計 d_s が 0 以下または f32 の非正規数",
                ));
            }
            let a: Vec<f64> = d.iter().map(|dc| dc.powf(-beta)).collect();
            let b: Vec<f64> = d.iter().map(|dc| dc.powf(-beta - 1.0)).collect();
            let (a_max, b_max) = (
                a.iter().cloned().fold(0.0, f64::max),
                b.iter().cloned().fold(0.0, f64::max),
            );
            let d_min = d.iter().cloned().fold(f64::INFINITY, f64::min);
            let col_u = col_idx
                .iter()
                .fold(0.0f64, |m, &i| m.max(f64::from(up_data[i]).abs()));
            let rb = 2.0 * (rho * beta).abs();
            let z = col_u * b_max;
            let sz = nt_r as f64 * z;
            let term2 = rb * sz;
            stage(col_u * a_max, "LRN の g·d^{-β}")?;
            stage(z, "LRN の g·x_s·d^{-β-1}")?;
            stage(sz, "LRN の窓和 Σ g·x_s·d^{-β-1}")?;
            stage(col_u * a_max + term2, "LRN の括弧内")?;
            stage(big_f * (col_u * a_max + term2), "LRN の dx")?;
            // 二階の逆伝播（seed ≤ 1）。
            let cot_inner = big_f;
            let cot_z = cot_inner * rb * nt_r as f64;
            stage(cot_inner * a_max + cot_z * b_max, "LRN の g への cotangent")?;
            let cot_a = cot_inner * col_u;
            let cot_b = cot_z * col_u;
            let cot_d = cot_a * beta.abs() * d_min.powf(-beta - 1.0)
                + cot_b * (beta + 1.0).abs() * d_min.powf(-beta - 2.0);
            stage(cot_d, "LRN の d_s への cotangent")?;
            let cot_sq = nt_f as f64 * rho.abs() * cot_d;
            stage(cot_sq, "LRN の x_s² への cotangent")?;
            let cot_xs = 2.0 * cot_sq + cot_inner * term2.max(rb * sz) + cot_z * col_u * b_max;
            stage(cot_xs, "LRN の x_s への cotangent")?;
            stage(cot_xs / f64::from(scales[col]), "LRN の x への cotangent")?;
            // 第 2 軸の本体: seed が 1 要素のときの二階の各要素を f32 で再現して参照と比べる。
            let col_x: Vec<f32> = col_idx.iter().map(|&i| x_data[i]).collect();
            let col_up: Vec<f32> = col_idx.iter().map(|&i| up_data[i]).collect();
            let col_chunks: Vec<f64> = chunks.iter().map(|ch| ch[col]).collect();
            lrn_col_judge(
                &col_x,
                &col_up,
                &LrnColConst {
                    s: f64::from(scales[col]),
                    kk: k / 2f64.powi(2 * exps[col]),
                    rho,
                    beta,
                    chunks: &col_chunks,
                },
                win,
                coeff_judge,
            )?;
            // 第 2 軸の m（各項の絶対値の和）。
            for (ci, &i) in col_idx.iter().enumerate() {
                let lo = ci.saturating_sub(back_r);
                let hi = (ci + fwd_r).min(c - 1);
                let acc: f64 = (lo..=hi)
                    .map(|cj| {
                        let ij = col_idx[cj];
                        f64::from(up_data[ij]).abs() * f64::from(xs[ij]).abs() * b[cj]
                    })
                    .sum();
                m_abs[i] = big_f
                    * (f64::from(up_data[i]).abs() * a[ci] + rb * f64::from(xs[i]).abs() * acc);
            }
        }
    }

    // 構造を積む。
    let xr = x_m.contiguous()?.reshape(&[n, c, inner])?;
    let upr = up.contiguous()?.reshape(&[n, c, inner])?;
    let col_shape = [n, 1, inner];
    let s_t = tensor_const(child, scales, &col_shape)?;
    let x_s = xr.div(&s_t)?;
    let sq = x_s.mul(&x_s)?;
    let sw = window_sum(&sq, c, back_f, fwd_f)?;
    let kk_t = tensor_const(child, kk, &col_shape)?;
    let d_s = sw.mul(&scalar_const(child, rho as f32)?)?.add(&kk_t)?;
    let a = d_s.pow_scalar((-beta) as f32)?;
    let b = d_s.pow_scalar((-beta - 1.0) as f32)?;
    let term1 = upr.mul(&a)?;
    let z = upr.mul(&x_s)?.mul(&b)?;
    let sz = window_sum(&z, c, back_r, fwd_r)?;
    let term2 = x_s
        .mul(&sz)?
        .mul(&scalar_const(child, (2.0 * rho * beta) as f32)?)?;
    let mut dx = term1.sub(&term2)?;
    for chunk in chunks {
        let chunk32: Vec<f32> = chunk.iter().map(|&v| v as f32).collect();
        dx = dx.mul(&tensor_const(child, chunk32, &col_shape)?)?;
    }
    let composed = dx.contiguous()?.reshape(&shape)?;
    shadow_check(
        child,
        &composed,
        &host_vec,
        &m_abs,
        rounding_coeff(nt_f + nt_r + 4),
        "LocalResponseNorm",
    )?;
    push_v3(child, &composed, host)
}

// ---------------------------------------------------------------------
// WeightNorm
// ---------------------------------------------------------------------

/// `dim` 軸（`None` は全体）をグループ軸にした正準化（`[groups, rest]`）の添字写像。
struct GroupMap {
    shape: Vec<usize>,
    dim: Option<usize>,
    groups: usize,
    rest: usize,
    /// グループ主順の位置 → 元の flat 添字。
    index: Vec<usize>,
    /// 軸を先頭へ寄せる置換（恒等のときは空）。
    perm: Vec<usize>,
}

impl GroupMap {
    fn new(shape: &[usize], dim: Option<usize>) -> Result<Self, AutodiffError> {
        let numel: usize = shape.iter().product();
        match dim {
            None => Ok(Self {
                shape: shape.to_vec(),
                dim,
                groups: 1,
                rest: numel,
                index: (0..numel).collect(),
                perm: Vec::new(),
            }),
            Some(d) => {
                if d >= shape.len() {
                    return Err(invalid("WeightNorm の dim が入力 rank の範囲外"));
                }
                let (outer, axis, inner) = (
                    shape[..d].iter().product::<usize>(),
                    shape[d],
                    shape[d + 1..].iter().product::<usize>(),
                );
                let mut index = Vec::with_capacity(numel);
                for g in 0..axis {
                    for o in 0..outer {
                        for i in 0..inner {
                            index.push((o * axis + g) * inner + i);
                        }
                    }
                }
                let perm = if d == 0 {
                    Vec::new()
                } else {
                    std::iter::once(d)
                        .chain((0..shape.len()).filter(|&x| x != d))
                        .collect()
                };
                Ok(Self {
                    shape: shape.to_vec(),
                    dim,
                    groups: axis,
                    rest: outer * inner,
                    index,
                    perm,
                })
            }
        }
    }

    fn to_group_major<T: Copy>(&self, data: &[T]) -> Vec<T> {
        self.index.iter().map(|&q| data[q]).collect()
    }

    fn to_form<'c>(&self, v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
        let src = if self.perm.is_empty() {
            *v
        } else {
            v.permute(&self.perm)?
        };
        src.contiguous()?.reshape(&[self.groups, self.rest])
    }

    fn restore<'c>(&self, v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
        let v = v.contiguous()?;
        if self.dim.is_none() || self.perm.is_empty() {
            return v.reshape(&self.shape);
        }
        let permuted: Vec<usize> = self.perm.iter().map(|&d| self.shape[d]).collect();
        let mut inv = vec![0usize; self.perm.len()];
        for (i, &p) in self.perm.iter().enumerate() {
            inv[p] = i;
        }
        v.reshape(&permuted)?.permute(&inv)?.contiguous()
    }
}

/// WeightNorm の 1 グループの構造の forward（二階の逆伝播が使う中間値と VJP 係数）。
struct WnFwd<T> {
    vs: Vec<T>,
    vh: Vec<T>,
    /// `v̂ = v_s/n_s` の分母側の Div 係数 `-(v_s/n_s)/n_s`。
    dvb: Vec<T>,
    up: Vec<T>,
    t: Vec<T>,
    inv_s: T,
    inv_n: T,
    /// Sqrt の係数 `0.5/n_s`。
    fsq: T,
    dg: T,
    /// `g/n_s` の分母側の Div 係数 `-(g/n_s)/n_s`。
    dgb: T,
    coef: T,
}

/// `weight_norm_cgrads` が積む構造（`v_s = v/s` 〜 `dv = (g/n_s/s)·(up − dg·v̂)`）の forward を再現する。
/// f32 の forward 値・VJP 係数のどれかが ±inf なら `None`（NaN は伝播として通す。[`no_inf`]）。
fn wn_fwd<T: Num>(v: &[f32], up: &[f32], g: f32, s: f64) -> Option<WnFwd<T>> {
    let s = T::k(s);
    let inv_s = T::k(1.0).over(s);
    let vs: Vec<T> = v.iter().map(|&x| T::k(f64::from(x)).over(s)).collect();
    let up: Vec<T> = up.iter().map(|&x| T::k(f64::from(x))).collect();
    let sq: Vec<T> = vs.iter().map(|&x| x.times(x)).collect();
    let ss = T::total(&sq);
    let n = ss.root();
    let fsq = n.root_grad();
    let inv_n = T::k(1.0).over(n);
    let vh: Vec<T> = vs.iter().map(|&x| x.over(n)).collect();
    let dvb: Vec<T> = vs.iter().map(|&x| x.over(n).over(n).negate()).collect();
    let dgt: Vec<T> = up.iter().zip(&vh).map(|(&u, &h)| u.times(h)).collect();
    let dg = T::total(&dgt);
    let t: Vec<T> = up
        .iter()
        .zip(&vh)
        .map(|(&u, &h)| u.minus(dg.times(h)))
        .collect();
    let gk = T::k(f64::from(g));
    let c1 = gk.over(n);
    let dgb = gk.over(n).over(n).negate();
    let coef = c1.over(s);
    let dv: Vec<T> = t.iter().map(|&x| coef.times(x)).collect();
    let ok = [ss, n, fsq, inv_n, inv_s, dg, c1, dgb, coef]
        .iter()
        .all(|v| v.no_inf())
        && [&vs, &vh, &dvb, &t, &dv].iter().all(|v| no_inf(v));
    ok.then_some(WnFwd {
        vs,
        vh,
        dvb,
        up,
        t,
        inv_s,
        inv_n,
        fsq,
        dg,
        dgb,
        coef,
    })
}

/// WeightNorm の seed（`dv_j` に 1、または `dg` に 1）。
#[derive(Clone, Copy)]
enum WnSeed {
    Dv(usize),
    Dg,
}

/// seed の二階の逆伝播を構造の演算順で再現する。`v`・`up` への要素を `hv`・`hu` へ書き、`g` への
/// 要素（`Dv` のときだけ）を返す。
fn wn_rev<T: Num>(f: &WnFwd<T>, seed: WnSeed, hv: &mut [T], hu: &mut [T]) -> Option<T> {
    let r = f.vs.len();
    let mut cot_vh = vec![T::k(0.0); r];
    for h in hu.iter_mut() {
        *h = T::k(0.0);
    }
    let (mut cot_n, cot_dg, h_g) = match seed {
        WnSeed::Dv(j) => {
            // dv = coef·t（coef は [groups, 1] のブロードキャスト）、coef = (g/n_s)/s。
            let cot_coef = f.t[j];
            let cot_t = f.coef;
            let cot_c1 = cot_coef.times(f.inv_s);
            let h_g = cot_c1.times(f.inv_n);
            let cot_n = cot_c1.times(f.dgb);
            // t = up − dg·v̂。
            hu[j] = cot_t;
            let cot_q = cot_t.negate();
            cot_vh[j] = cot_q.times(f.dg);
            (cot_n, cot_q.times(f.vh[j]), Some(h_g))
        }
        WnSeed::Dg => (T::k(0.0), T::k(1.0), None),
    };
    // dg = Σ up⊙v̂。
    for i in 0..r {
        hu[i] = hu[i].plus(cot_dg.times(f.vh[i]));
        cot_vh[i] = cot_vh[i].plus(cot_dg.times(f.up[i]));
    }
    // v̂ = v_s/n_s。
    let mut n_terms = vec![cot_n];
    for i in 0..r {
        hv[i] = cot_vh[i].times(f.inv_n);
        n_terms.push(cot_vh[i].times(f.dvb[i]));
    }
    cot_n = T::total(&n_terms);
    // n_s = √(Σ v_s²)、v_s = v/s。
    let cot_ss = cot_n.times(f.fsq);
    for (h, &x) in hv.iter_mut().zip(&f.vs) {
        let g = cot_ss.times(x);
        *h = T::total(&[*h, g, g]).times(f.inv_s);
    }
    h_g
}

/// `v` に NaN がある WeightNorm のグループの依存の走査（[`Dm`]。[`wn_fwd`]・[`wn_rev`] を全 seed で回す）。
/// 1 階 `dv`・`dg` と二階の全要素（`v`・`up`・`g` への要素）が入力の NaN で NaN になるとき `true`。
/// `true` のグループだけを段検査と判定から外してよい（守る有限の要素が無い）。NaN の伝わり方は値に
/// 依らないので、スケールは 1 で回す。
fn wn_group_all_nan(v: &[f32], up: &[f32], g: f32) -> bool {
    let Some(f) = wn_fwd::<Dm>(v, up, g, 1.0) else {
        // `Dm` は ±inf を持たないので到達しない。安全側として外さない側へ倒す。
        return false;
    };
    if !f.dg.nan || f.t.iter().any(|&t| !f.coef.times(t).nan) {
        return false;
    }
    let r = v.len();
    let (mut hv, mut hu) = (vec![Dm::CONST; r], vec![Dm::CONST; r]);
    for seed in (0..r).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
        if wn_rev(&f, seed, &mut hv, &mut hu).is_some_and(|h| !h.nan) {
            return false;
        }
        if hv.iter().chain(&hu).any(|h| !h.nan) {
            return false;
        }
    }
    true
}

/// WeightNorm の 1 グループの総括判定。seed が 1 要素（`dv` の各要素と `dg`）のときの二階
/// （`v`・`up`・`g` への要素）を判定する。`use_lin` なら丸めの一次伝播を事象ごとに追う [`Lin`]
/// （[`entry_check_lin`]。呼び出し側が幅 [`LIN_REST_MAX`] 以下かつ走査量 [`LIN_BUDGET`] 以内で選ぶ）、
/// そうでなければ従来の [`Rf`] と係数 `coeff`（[`entry_check`]）。
fn wn_group_judge(
    v: &[f32],
    up: &[f32],
    g: f32,
    s: f64,
    coeff: f64,
    use_lin: bool,
) -> Result<(), AutodiffError> {
    if use_lin {
        lin_reset();
        wn_group_judge_with::<Lin>(v, up, g, s, &|r, x, what| entry_check_lin(r, x, what))
    } else {
        wn_group_judge_with::<Rf>(v, up, g, s, &|r, x, what| entry_check(r, x, coeff, what))
    }
}

fn wn_group_judge_with<R: Num>(
    v: &[f32],
    up: &[f32],
    g: f32,
    s: f64,
    check: &dyn Fn(R, f32, &str) -> Result<(), AutodiffError>,
) -> Result<(), AutodiffError> {
    const WHAT: &str = "WeightNorm の二階";
    let (Some(fs), Some(fr)) = (wn_fwd::<f32>(v, up, g, s), wn_fwd::<R>(v, up, g, s)) else {
        return Err(nonfinite_struct(WHAT));
    };
    let nan_ok = [
        (&fs.vs, &fr.vs),
        (&fs.vh, &fr.vh),
        (&fs.dvb, &fr.dvb),
        (&fs.up, &fr.up),
        (&fs.t, &fr.t),
    ]
    .iter()
    .all(|(a, b)| same_nan(a, b))
        && same_nan(
            &[fs.inv_s, fs.inv_n, fs.fsq, fs.dg, fs.dgb, fs.coef],
            &[fr.inv_s, fr.inv_n, fr.fsq, fr.dg, fr.dgb, fr.coef],
        );
    if !nan_ok {
        return Err(nan_mismatch(WHAT));
    }
    let r = v.len();
    let (mut hv_s, mut hu_s) = (vec![0f32; r], vec![0f32; r]);
    // forward の事象は seed 間で共有し、逆伝播の事象だけ seed ごとに戻す（領域を持たない型では無害）。
    let mark = lin_mark();
    for seed in (0..r).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
        lin_truncate(mark);
        let (mut hv_r, mut hu_r) = (vec![R::k(0.0); r], vec![R::k(0.0); r]);
        let gs = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
        let gr = wn_rev(&fr, seed, &mut hv_r, &mut hu_r);
        if let (Some(gs), Some(gr)) = (gs, gr) {
            check(gr, gs, WHAT)?;
        }
        for i in 0..r {
            check(hv_r[i], hv_s[i], WHAT)?;
            check(hu_r[i], hu_s[i], WHAT)?;
        }
    }
    Ok(())
}

/// `WeightNorm` ノードの `(d_v, d_g)` を子テープへ記録する。`g_m` は子テープ上の `g` の写し。
#[allow(clippy::too_many_arguments)]
pub(super) fn weight_norm_cgrads<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    v_m: &Var<'c>,
    g_m: &Var<'c>,
    ids: (NodeId, NodeId),
    dim: Option<usize>,
    up: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let (v_id, g_id) = ids;
    let v_val = materialize_fallible(parent_nodes, parent.ops(), v_id)?;
    let g_val = materialize_fallible(parent_nodes, parent.ops(), g_id)?;
    let shape = v_val.shape().to_vec();
    let layout =
        wr::weight_norm_layout(&shape, g_val.shape(), dim).map_err(AutodiffError::Shape)?;
    if up.shape() != shape || v_m.shape() != shape || g_m.shape() != g_val.shape() {
        return Err(invalid("WeightNorm の shape 不整合（内部契約違反）"));
    }
    if shape.contains(&0) {
        return Ok((
            child.var_no_grad(&Tensor::zeros(&shape)?),
            child.var_no_grad(&Tensor::zeros(layout.g_shape())?),
        ));
    }
    let up_data = dense_vec(&child_value(child, up)?);
    check_finite(&up_data, "上流 g")?;
    let v_data = dense_vec(v_val);
    let g_data = dense_vec(g_val);
    let (dv_host, dg_host) = wr::weight_norm_vjp_host(&v_data, &g_data, &up_data, &layout)
        .map_err(AutodiffError::Shape)?;
    let gm = GroupMap::new(&shape, dim)?;
    let (groups, rest) = (gm.groups, gm.rest);

    let v_gm = gm.to_group_major(&v_data);
    let up_gm = gm.to_group_major(&up_data);
    let dv_gm = gm.to_group_major(&dv_host);
    // ホストの 1 階の非有限は、入力の NaN が届く位置の NaN だけを通す（[`host_nonfinite_check`]）。`dv` は
    // グループの `v` か `g` の NaN、`dg = Σup⊙v/n` はグループの `v` の NaN が届く（g に依存しない）。
    let v_nan: Vec<bool> = v_gm
        .chunks(rest)
        .map(|grp| grp.iter().any(|x| x.is_nan()))
        .collect();
    let reach_dv: Vec<bool> = (0..v_gm.len())
        .map(|i| v_nan[i / rest] || g_data[i / rest].is_nan())
        .collect();
    host_nonfinite_check(&dv_gm, &reach_dv, "WeightNorm の dv")?;
    host_nonfinite_check(&dg_host, &v_nan, "WeightNorm の dg")?;
    let exps: Vec<i32> = v_gm
        .chunks(rest)
        .map(|grp| {
            if grp.iter().any(|x| x.is_nan()) {
                0
            } else {
                pow2_ceil_exponent(grp.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())))
            }
        })
        .collect();
    let scales: Vec<f32> = exps.iter().map(|&e| pow2(e)).collect();

    // 第 1 軸（overflow）と第 2 軸の m・総括判定（係数は [`wn_coeff`]）。
    // 走査量は全グループ分を数える（`v` に NaN があるグループは判定の代わりに同じ規模の依存の走査
    // [`wn_group_all_nan`] を回す）。
    let coeff_judge = wn_coeff(rest);
    let use_lin = rest <= LIN_REST_MAX && groups.saturating_mul(lin_group_cost(rest)) <= LIN_BUDGET;
    judge_budget(wn_judge_cost(groups, rest), "WeightNorm")?;
    let nf = rest as f64;
    let mut m_dv = vec![0.0f64; v_gm.len()];
    let mut m_dg = vec![0.0f64; groups];
    for k in 0..groups {
        let lo = k * rest;
        let vs: Vec<f32> = v_gm[lo..lo + rest].iter().map(|&x| x / scales[k]).collect();
        // v に NaN があるグループは `n_s = √Σv_s²` が NaN になり、グループの 1 階・二階は全要素が入力の
        // NaN に依存する見込み。外す前に [`wn_group_all_nan`] で全要素の依存を確かめ、有限の出力が残る
        // 形（構造の演算が変わって NaN の消える経路ができた場合）は fail-closed で拒否する。グループは
        // 互いに独立（縮約は行ごと）。上流 `up` は有限（上の `check_finite`）。段検査より前に外し、`m` は
        // NaN にしてホストが有限なら照合で拒否する側へ倒す。
        if v_nan[k] {
            if !wn_group_all_nan(&v_gm[lo..lo + rest], &up_gm[lo..lo + rest], g_data[k]) {
                return Err(invalid(
                    "WeightNorm: v に NaN を含むグループに、入力の NaN に依存しない二階の要素が残るため拒否",
                ));
            }
            m_dv[lo..lo + rest].fill(f64::NAN);
            m_dg[k] = f64::NAN;
            continue;
        }
        let ss: f64 = vs.iter().map(|&x| f64::from(x).powi(2)).sum();
        let n_s = ss.sqrt();
        let s = f64::from(scales[k]);
        // g が NaN のグループは dv が NaN になる（1 階と同じ伝播）。g 依存の段だけを 0 として
        // 評価し、g に依存しない dg の段検査と m は通常どおり行う（NaN の g で段を拒否しない）。
        let gk = if g_data[k].is_nan() {
            0.0
        } else {
            f64::from(g_data[k]).abs()
        };
        let upk = &up_gm[lo..lo + rest];
        let u_max = upk.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let vhat: Vec<f64> = vs.iter().map(|&x| f64::from(x) / n_s).collect();
        let dgb: f64 = upk
            .iter()
            .zip(&vhat)
            .map(|(&x, &h)| f64::from(x).abs() * h.abs())
            .sum();
        let t_max = u_max + dgb;
        let coef = gk / n_s / s;
        stage(dgb, "WeightNorm の dg")?;
        stage(t_max, "WeightNorm の up − dg·v̂")?;
        stage(coef, "WeightNorm の g/(n_s·s)")?;
        stage(coef * t_max, "WeightNorm の dv")?;
        // 二階の逆伝播（seed ≤ 1）。
        let cot_coef = nf * t_max;
        stage(cot_coef / (n_s * s), "WeightNorm の g への cotangent")?;
        let cot_dg2 = nf * coef + 1.0;
        stage(coef + cot_dg2, "WeightNorm の g_up への cotangent")?;
        let b_vh = coef * dgb + cot_dg2 * u_max;
        stage(b_vh, "WeightNorm の v̂ への cotangent")?;
        let cot_n = nf * b_vh / n_s;
        let cot_ss = cot_n * 0.5 / n_s;
        let cot_vs = b_vh / n_s + 2.0 * cot_ss;
        stage(cot_vs, "WeightNorm の v_s への cotangent")?;
        stage(cot_vs / s, "WeightNorm の v への cotangent")?;
        let dg_val = upk
            .iter()
            .zip(&vhat)
            .map(|(&x, &h)| f64::from(x) * h)
            .sum::<f64>();
        for (j, (&x, &h)) in upk.iter().zip(&vhat).enumerate() {
            m_dv[lo + j] = coef * (f64::from(x).abs() + dg_val.abs() * h.abs());
        }
        m_dg[k] = dgb;
        // 第 2 軸の本体: seed が 1 要素のときの二階の各要素を f32 で再現して参照と比べる。
        // g が NaN のグループも判定する。g の NaN は `coef = (g/n_s)/s` を通って `dv` と、`dv` への seed
        // から `v`・`up` への二階へ届くが、`dg` への seed の二階（`v`・`up` への要素）と `dv` への seed の
        // `g` への要素は g に依存せず有限なので、通常どおり判定する（両側 NaN の要素だけ外す）。
        wn_group_judge(
            &v_gm[lo..lo + rest],
            upk,
            g_data[k],
            s,
            coeff_judge,
            use_lin,
        )?;
    }

    // 構造。
    let v2 = gm.to_form(v_m)?;
    let up2 = gm.to_form(up)?;
    let g2 = g_m.contiguous()?.reshape(&[groups, 1])?;
    let s_t = tensor_const(child, scales, &[groups, 1])?;
    let v_s = v2.div(&s_t)?;
    let n_s = v_s.mul(&v_s)?.sum(Some(1))?.unsqueeze(1)?.sqrt()?;
    let vhat = v_s.div(&n_s)?;
    let dg2 = up2.mul(&vhat)?.sum(Some(1))?.unsqueeze(1)?;
    let t = up2.sub(&dg2.mul(&vhat)?)?;
    let coef = g2.div(&n_s)?.div(&s_t)?;
    let dv2 = coef.mul(&t)?;

    let dv_composed = gm.restore(&dv2)?;
    let dg_composed = dg2.contiguous()?.reshape(layout.g_shape())?;
    let coeff = rounding_coeff(rest + 4);
    shadow_check(child, &dv2, &dv_gm, &m_dv, coeff, "WeightNorm の dv")?;
    shadow_check(
        child,
        &dg2.contiguous()?.reshape(&[groups])?,
        &dg_host,
        &m_dg,
        coeff,
        "WeightNorm の dg",
    )?;
    let dv = push_v3(child, &dv_composed, Tensor::new(dv_host, &shape)?)?;
    let dg = push_v3(child, &dg_composed, Tensor::new(dg_host, layout.g_shape())?)?;
    Ok((dv, dg))
}

// ---------------------------------------------------------------------
// SpectralNorm
// ---------------------------------------------------------------------

/// SpectralNorm の構造の forward（二階の逆伝播が使う中間値と VJP 係数）。
struct SnFwd<T> {
    ws: Vec<T>,
    p: Vec<T>,
    wh: Vec<T>,
    /// `Ŵ = W_s/σ_s` の分母側の Div 係数 `-(W_s/σ_s)/σ_s`。
    dwb: Vec<T>,
    up: Vec<T>,
    /// `t/σ_s` の分母側の Div 係数 `-(t/σ_s)/σ_s`。
    dob: Vec<T>,
    inv_s: T,
    inv_sig: T,
}

/// `spectral_norm_cgrad` が積む構造（`W_s = W/s` 〜 `dW = (up − Σ(up⊙Ŵ)·P)/σ_s/s`）の forward を再現する。
/// `p` は `P[i] = u_i·v_j` の f64 の厳密値（構造は f32 へ丸めた値を載せる）。
fn sn_fwd<T: Num>(w: &[f32], p: &[f64], up: &[f32], s: f64) -> Option<SnFwd<T>> {
    let s = T::k(s);
    let inv_s = T::k(1.0).over(s);
    let ws: Vec<T> = w.iter().map(|&x| T::k(f64::from(x)).over(s)).collect();
    let p: Vec<T> = p.iter().map(|&x| T::k(x)).collect();
    let up: Vec<T> = up.iter().map(|&x| T::k(f64::from(x))).collect();
    let mut sig = T::k(0.0);
    for (&a, &b) in ws.iter().zip(&p) {
        sig = sig.plus(a.times(b));
    }
    let inv_sig = T::k(1.0).over(sig);
    let wh: Vec<T> = ws.iter().map(|&x| x.over(sig)).collect();
    let dwb: Vec<T> = ws.iter().map(|&x| x.over(sig).over(sig).negate()).collect();
    let mut sp = T::k(0.0);
    for (&u, &h) in up.iter().zip(&wh) {
        sp = sp.plus(u.times(h));
    }
    let t: Vec<T> = up
        .iter()
        .zip(&p)
        .map(|(&u, &pi)| u.minus(sp.times(pi)))
        .collect();
    let dob: Vec<T> = t.iter().map(|&x| x.over(sig).over(sig).negate()).collect();
    let dw: Vec<T> = t.iter().map(|&x| x.over(sig).over(s)).collect();
    let ok = [sig, inv_sig, inv_s, sp].iter().all(|v| v.no_inf())
        && [&ws, &p, &wh, &dwb, &t, &dob, &dw]
            .iter()
            .all(|v| no_inf(v));
    ok.then_some(SnFwd {
        ws,
        p,
        wh,
        dwb,
        up,
        dob,
        inv_s,
        inv_sig,
    })
}

/// seed `e_j`（`dW_j` に 1）の二階の逆伝播を構造の演算順で再現し、`W`・`up` への要素を `hw`・`hu` へ書く。
fn sn_rev<T: Num>(f: &SnFwd<T>, j: usize, hw: &mut [T], hu: &mut [T]) {
    let n = f.ws.len();
    // dW = (t/σ_s)/s。
    let cot_o = f.inv_s;
    let cot_t = cot_o.times(f.inv_sig);
    let mut cot_sig = cot_o.times(f.dob[j]);
    // t = up − sp·P（sp は [1] のブロードキャスト）、sp = Σ up⊙Ŵ。
    let cot_sp = cot_t.negate().times(f.p[j]);
    for i in 0..n {
        hu[i] = cot_sp.times(f.wh[i]);
        let cot_wh = cot_sp.times(f.up[i]);
        hw[i] = cot_wh.times(f.inv_sig);
        cot_sig = cot_sig.plus(cot_wh.times(f.dwb[i]));
    }
    hu[j] = cot_t.plus(hu[j]);
    // σ_s = Σ W_s⊙P、W_s = W/s。
    for (h, &pi) in hw.iter_mut().zip(&f.p) {
        *h = h.plus(cot_sig.times(pi)).times(f.inv_s);
    }
}

/// `W` に NaN がある SpectralNorm の依存の走査（[`Dm`]。[`sn_fwd`]・[`sn_rev`] を全 seed で回す）。1 階
/// `dW` と二階の全要素（`W`・`up` への要素）が入力の NaN で NaN になるとき `true`。`true` のときだけ
/// テンソル単位で段検査と判定を外してよい。1 階 `dW = (t/σ_s)/s` の依存は `dob = −(t/σ_s)/σ_s` と同じ
/// （`s` は定数）なので `dob` で見る。スケールは 1 で回す。
fn sn_all_nan(w: &[f32], p: &[f64], up: &[f32]) -> bool {
    let Some(f) = sn_fwd::<Dm>(w, p, up, 1.0) else {
        // `Dm` は ±inf を持たないので到達しない。安全側として外さない側へ倒す。
        return false;
    };
    if f.dob.iter().any(|d| !d.nan) {
        return false;
    }
    let n = w.len();
    let (mut hw, mut hu) = (vec![Dm::CONST; n], vec![Dm::CONST; n]);
    for j in 0..n {
        sn_rev(&f, j, &mut hw, &mut hu);
        if hw.iter().chain(&hu).any(|h| !h.nan) {
            return false;
        }
    }
    true
}

/// SpectralNorm の総括判定。seed が 1 要素のときの二階（`W`・`up` への要素）を [`entry_check`] に掛ける。
fn sn_judge(w: &[f32], p: &[f64], up: &[f32], s: f64, coeff: f64) -> Result<(), AutodiffError> {
    const WHAT: &str = "SpectralNorm の二階";
    let (Some(fs), Some(fr)) = (sn_fwd::<f32>(w, p, up, s), sn_fwd::<Rf>(w, p, up, s)) else {
        return Err(nonfinite_struct(WHAT));
    };
    let nan_ok = [
        (&fs.ws, &fr.ws),
        (&fs.p, &fr.p),
        (&fs.wh, &fr.wh),
        (&fs.dwb, &fr.dwb),
        (&fs.up, &fr.up),
        (&fs.dob, &fr.dob),
    ]
    .iter()
    .all(|(a, b)| same_nan(a, b))
        && same_nan(&[fs.inv_s, fs.inv_sig], &[fr.inv_s, fr.inv_sig]);
    if !nan_ok {
        return Err(nan_mismatch(WHAT));
    }
    let n = w.len();
    let (mut hw_s, mut hu_s) = (vec![0f32; n], vec![0f32; n]);
    let (mut hw_r, mut hu_r) = (vec![Rf::k(0.0); n], vec![Rf::k(0.0); n]);
    for j in 0..n {
        sn_rev(&fs, j, &mut hw_s, &mut hu_s);
        sn_rev(&fr, j, &mut hw_r, &mut hu_r);
        for i in 0..n {
            entry_check(hw_r[i], hw_s[i], coeff, WHAT)?;
            entry_check(hu_r[i], hu_s[i], coeff, WHAT)?;
        }
    }
    Ok(())
}

/// `SpectralNorm` ノード（入力 `weight`）の d_weight を子テープへ記録する。`u`・`v` は forward 時点の
/// 非追跡スナップショット（定数扱い）。
#[allow(clippy::too_many_arguments)]
pub(super) fn spectral_norm_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    w_m: &Var<'c>,
    weight: NodeId,
    uv: (&Tensor<f32>, &Tensor<f32>),
    dim: usize,
    up: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (u, v) = uv;
    let w_val = materialize_fallible(parent_nodes, parent.ops(), weight)?;
    let shape = w_val.shape().to_vec();
    let layout = wr::spectral_norm_layout(&shape, dim).map_err(AutodiffError::Shape)?;
    if up.shape() != shape || w_m.shape() != shape {
        return Err(invalid("SpectralNorm の shape 不整合（内部契約違反）"));
    }
    let up_data = dense_vec(&child_value(child, up)?);
    check_finite(&up_data, "上流 g")?;
    let w_data = dense_vec(w_val);
    let (u_data, v_data) = (dense_vec(u), dense_vec(v));
    let host_vec = wr::spectral_norm_vjp_host(&w_data, &up_data, &layout, &u_data, &v_data)
        .map_err(AutodiffError::Shape)?;

    // P[i, j] = u_i·v_j を W と同じ並びに置く。
    let (h, cols) = (layout.rows(), layout.cols());
    let inner: usize = shape[dim + 1..].iter().product();
    let mut p = vec![0.0f32; w_data.len()];
    let mut p_exact = vec![0.0f64; w_data.len()];
    for (i, &ui) in u_data.iter().enumerate().take(h) {
        for (j, &vj) in v_data.iter().enumerate().take(cols) {
            let at = (j / inner * h + i) * inner + j % inner;
            p[at] = ui * vj;
            p_exact[at] = f64::from(ui) * f64::from(vj);
        }
    }
    let p_max = p.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    // W に NaN が 1 つでもあると `σ_s = ΣW_s⊙P` が NaN になり（`P` の成分が 0 でも `NaN·0 = NaN`）、`Ŵ`・
    // `Σup⊙Ŵ`・`dW` と二階は全要素 NaN になる見込みで、守る有限の要素が無い。外す前に [`sn_all_nan`] で
    // 全要素の依存を確かめる（下）。
    // 上流 `up` は有限（上の `check_finite`）。
    let has_nan = w_data.iter().any(|x| x.is_nan());
    // ホストの 1 階の非有限は、入力の NaN が届く位置の NaN だけを通す（[`host_nonfinite_check`]）。`W` か
    // スナップショット `u`・`v` の NaN は `σ` を通って全要素へ届く（`u`・`v` の NaN は下の `σ_s` の検査で
    // 拒否する）。それ以外の非有限（`Σup⊙W` の f32 積の overflow 等）は演算で生まれたものとして拒否する。
    let reach_all = has_nan || u_data.iter().chain(&v_data).any(|x| x.is_nan());
    host_nonfinite_check(&host_vec, &vec![reach_all; host_vec.len()], "SpectralNorm")?;

    let e = if has_nan {
        0
    } else {
        pow2_ceil_exponent(
            w_data
                .iter()
                .fold(0.0f64, |m, &x| m.max(f64::from(x).abs())),
        )
    };
    let s = pow2(e);
    let ws: Vec<f32> = w_data.iter().map(|&x| x / s).collect();
    // 走査量は判定（`W` に NaN が無いとき）と依存の走査（あるとき）で同じ規模なので、分岐の前に掛ける。
    judge_budget(sn_judge_cost(w_data.len()), "SpectralNorm")?;
    // `W` に NaN があるときは、外す前に [`sn_all_nan`] で全要素の依存を確かめ、有限の出力が残る形は
    // fail-closed で拒否する。
    if has_nan && !sn_all_nan(&w_data, &p_exact, &up_data) {
        return Err(invalid(
            "SpectralNorm: W に NaN を含むのに、入力の NaN に依存しない二階の要素が残るため拒否",
        ));
    }
    // `W` に NaN があると判定を外すので、`m` は NaN にしてホストが有限なら照合で拒否する側へ倒す。
    let mut m_abs = vec![if has_nan { f64::NAN } else { 0.0 }; w_data.len()];
    if !has_nan {
        let n = w_data.len() as f64;
        let sigma_s: f64 = ws
            .iter()
            .zip(&p)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        if sigma_s == 0.0 || !sigma_s.is_finite() {
            return Err(invalid("SpectralNorm のスケール後 σ_s が 0 または非有限"));
        }
        let sa = sigma_s.abs();
        let sf = f64::from(s);
        let u_max = up_data
            .iter()
            .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let w_hat_max = ws.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())) / sa;
        stage(w_hat_max, "SpectralNorm の Ŵ")?;
        let s_prime: f64 = up_data
            .iter()
            .zip(&ws)
            .map(|(&x, &w)| f64::from(x).abs() * f64::from(w).abs() / sa)
            .sum();
        stage(s_prime, "SpectralNorm の Σ g·Ŵ")?;
        let t_max = u_max + s_prime * p_max;
        stage(t_max / sa, "SpectralNorm の t/σ_s")?;
        stage(t_max / sa / sf, "SpectralNorm の dW")?;
        // 二階の逆伝播（seed ≤ 1）。
        let cot_t = 1.0 / (sa * sf);
        stage(cot_t, "SpectralNorm の t への cotangent")?;
        let cot_sigma_direct = n * t_max / (sa * sa * sf);
        stage(cot_sigma_direct, "SpectralNorm の σ_s への直接 cotangent")?;
        let cot_sp = n * p_max * cot_t;
        stage(cot_sp, "SpectralNorm の Σ g·Ŵ への cotangent")?;
        stage(
            cot_t + cot_sp * w_hat_max,
            "SpectralNorm の g への cotangent",
        )?;
        let cot_what = cot_sp * u_max;
        let cot_sigma = cot_sigma_direct + n * cot_what * w_hat_max / sa;
        stage(cot_sigma, "SpectralNorm の σ_s への cotangent")?;
        let cot_ws = cot_what / sa + cot_sigma * p_max;
        stage(cot_ws, "SpectralNorm の W_s への cotangent")?;
        stage(cot_ws / sf, "SpectralNorm の W への cotangent")?;
        let sp_val: f64 = up_data
            .iter()
            .zip(&ws)
            .map(|(&x, &w)| f64::from(x) * f64::from(w) / sigma_s)
            .sum();
        for (i, mi) in m_abs.iter_mut().enumerate() {
            *mi = (f64::from(up_data[i]).abs() + sp_val.abs() * f64::from(p[i]).abs()) / (sa * sf);
        }
        // 第 2 軸の本体: seed が 1 要素のときの二階の各要素を f32 で再現して参照と比べる。σ_s の相殺・
        // 係数行列 `I − ŴP` の相殺・Div 係数の overflow・中間値の underflow はここで見える。
        sn_judge(&w_data, &p_exact, &up_data, sf, sn_coeff(w_data.len()))?;
    }

    // 構造。
    let s_c = scalar_const(child, s)?;
    let p_t = tensor_const(child, p, &shape)?;
    let w_s = w_m.div(&s_c)?;
    let sigma_s = w_s.mul(&p_t)?.sum(None)?.reshape(&[1])?;
    let w_hat = w_s.div(&sigma_s)?;
    let sp = up.mul(&w_hat)?.sum(None)?.reshape(&[1])?;
    let t = up.sub(&sp.mul(&p_t)?)?;
    let composed = t.div(&sigma_s)?.div(&s_c)?;
    shadow_check(
        child,
        &composed,
        &host_vec,
        &m_abs,
        rounding_coeff(w_data.len() + 4),
        "SpectralNorm",
    )?;
    push_v3(child, &composed, Tensor::new(host_vec, &shape)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_graph_norm::STAGE_BOUND_GUARD;

    #[test]
    fn stage_rejects_large_or_nonfinite() {
        assert!(stage(1.0, "x").is_ok());
        assert!(stage(STAGE_BOUND_GUARD, "x").is_err());
        assert!(stage(f64::NAN, "x").is_err());
    }

    #[test]
    fn chunk_factors_split_large_exponents_and_skip_zero() {
        assert!(chunk_factors(&[0.0, 0.0]).is_empty());
        let one = chunk_factors(&[-60.0, 10.0]);
        assert_eq!(one.len(), 1);
        let many = chunk_factors(&[-250.0, 0.0]);
        assert_eq!(many.len(), 3);
        for ch in &many {
            assert!(ch.iter().all(|f| f.is_finite() && *f > 0.0));
        }
        let prod: f64 = many.iter().map(|c| c[0].log2()).sum();
        assert!((prod + 250.0).abs() < 1e-3, "{prod}");
    }

    #[test]
    fn struct_check_accepts_exact_and_rejects_cancellation() {
        assert!(struct_check(1.0, 1.0, 1.0, rounding_coeff(4), "t").is_ok());
        // 値 0 だが各項の大きさが 1e3: 桁落ちで二階が許容を外れうる。
        assert!(struct_check(0.0, 0.0, 1e3, rounding_coeff(4), "t").is_err());
        assert!(struct_check(f64::NAN, 1.0, 1.0, 0.0, "t").is_ok());
        assert!(struct_check(1.0, f64::NAN, 1.0, 0.0, "t").is_err());
        // 参照値が有限なら `m` だけの NaN でも拒否する（有限の要素を検査なしで通さない）。
        assert!(struct_check(1.0, 1.0, f64::NAN, 0.0, "t").is_err());
    }

    #[test]
    fn group_map_round_trips_positions() {
        let gm = GroupMap::new(&[2, 3, 2], Some(1)).unwrap();
        let data: Vec<u32> = (0..12).collect();
        let gmj = gm.to_group_major(&data);
        assert_eq!(gm.groups, 3);
        assert_eq!(gm.rest, 4);
        assert_eq!(gmj[..4], [0, 1, 6, 7]);
    }

    // --- 総括判定の f32 再現が子テープの実際の二階と一致すること（判定の前提の固定） ---------------

    use crate::test_support::test_ops;
    use crate::weight_reparam_ops::SpectralNormState;

    fn tt(v: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(v, shape).unwrap()
    }

    fn pattern(n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + phase) * 0.731).sin() * 0.9 + 0.2)
            .collect()
    }

    /// `cg.grad(x)` の要素 `j` を seed にした子テープの二階（`wrt` への行）。
    fn child_row(child: &Tape, gx: &Var<'_>, j: usize, wrt: &Var<'_>) -> Vec<f32> {
        let n = gx.shape().iter().product();
        let gj = gx
            .contiguous()
            .unwrap()
            .reshape(&[n])
            .unwrap()
            .narrow(0, j, 1)
            .unwrap()
            .sum(None)
            .unwrap();
        let g = child.backward(&gj).unwrap();
        dense_vec(g.get(wrt).unwrap().unwrap())
    }

    /// 再現値 `s` が子テープの実値 `a` から判定の見積り `coeff·M` を超えてずれないこと。
    fn assert_within(a: f32, s: f32, r: Rf, coeff: f64, ctx: &str) {
        let d = (f64::from(a) - f64::from(s)).abs();
        assert!(
            d <= coeff * r.m + f64::from(f32::MIN_POSITIVE),
            "{ctx}: 子テープ {a:e} と f32 再現 {s:e} の差 {d:e} が見積り {:e} を超える",
            coeff * r.m
        );
    }

    #[test]
    fn lrn_emulation_reproduces_child_tape_second_order() {
        // 奇数・偶数（非対称窓）・C を超える size と、β=−1（`-β-1 = 0` で PowScalar の係数が 0）。
        for beta in [0.75f32, 2.0, -1.0] {
            let compared = (1..=5usize)
                .filter(|&size| lrn_emulation_case(size, 0.5, beta, 1.0))
                .count();
            assert!(
                compared >= 4,
                "beta={beta}: 比べられた size が {compared} 件しかない"
            );
        }
    }

    fn lrn_emulation_case(size: usize, alpha: f32, beta: f32, k: f32) -> bool {
        let (n, c, inner) = (1usize, 4usize, 2usize);
        let shape = [n, c, inner];
        let x0 = pattern(8, 1.0);
        let w = pattern(8, 4.0);
        let tape = Tape::new_with_ops(test_ops());
        let child = Tape::new_with_ops(test_ops());
        let x = tape.var(&tt(x0.clone(), &shape));
        // 重み `wc` を追跡し、`up = wc`（Sum→Mul の VJP で bit 一致）への二階も子テープから読む。
        let wc = tape.var(&tt(w.clone(), &shape));
        let y = x.local_response_norm(size, alpha, beta, k).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        // 判定が保守的に拒否した入力は構造が積まれないので比べられない（呼び出し側で件数を確かめる）。
        let Ok(cg) = tape.backward_create_graph(&loss, &child) else {
            return false;
        };
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        let cwc = cg.child_var(&wc).unwrap().unwrap();
        let rho = f64::from(alpha) / size as f64;
        let floor = (f64::from(k).abs() * size as f64 / f64::from(alpha).abs()).sqrt();
        let win = LrnWin {
            bf: (size / 2).min(c - 1),
            ff: ((size - 1) / 2).min(c - 1),
            br: ((size - 1) / 2).min(c - 1),
            fr: (size / 2).min(c - 1),
        };
        let (nt_f, nt_r) = (
            window_terms(c, win.bf, win.ff),
            window_terms(c, win.br, win.fr),
        );
        for si in 0..inner {
            let idx: Vec<usize> = (0..c).map(|ci| ci * inner + si).collect();
            let col_x: Vec<f32> = idx.iter().map(|&i| x0[i]).collect();
            let col_up: Vec<f32> = idx.iter().map(|&i| w[i]).collect();
            let mx = col_x.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
            let e = pow2_ceil_exponent(mx.max(floor));
            let chunks: Vec<f64> = chunk_factors(&[-2.0 * f64::from(beta) * f64::from(e)])
                .iter()
                .map(|ch| ch[0])
                .collect();
            let kc = LrnColConst {
                s: f64::from(pow2(e)),
                kk: f64::from(k) / 2f64.powi(2 * e),
                rho,
                beta: f64::from(beta),
                chunks: &chunks,
            };
            let coeff = lrn_coeff(nt_f, nt_r, chunks.len(), f64::from(beta));
            let fs = lrn_col_fwd::<f32>(&col_x, &col_up, &kc, win).unwrap();
            let fr = lrn_col_fwd::<Rf>(&col_x, &col_up, &kc, win).unwrap();
            let (mut ss, mut sr) = (LrnScratch::<f32>::new(c), LrnScratch::<Rf>::new(c));
            for jc in 0..c {
                lrn_col_rev(&fs, win, jc, &mut ss);
                lrn_col_rev(&fr, win, jc, &mut sr);
                let row = child_row(&child, &gx, idx[jc], &cx);
                let row_up = child_row(&child, &gx, idx[jc], &cwc);
                let (ulo, uhi) = (jc.saturating_sub(win.br), (jc + win.fr).min(c - 1));
                let (xlo, xhi) = (ulo.saturating_sub(win.bf), (uhi + win.ff).min(c - 1));
                for (m, &i) in idx.iter().enumerate() {
                    let (s, r) = if (ulo..=uhi).contains(&m) {
                        (ss.up[m], sr.up[m])
                    } else {
                        (0.0, Rf::k(0.0))
                    };
                    assert_within(
                        row_up[i],
                        s,
                        r,
                        coeff,
                        &format!("lrn up size{size} beta{beta} col{si} j{jc} m{m}"),
                    );
                }
                for (l, &i) in idx.iter().enumerate() {
                    let (s, r) = if (xlo..=xhi).contains(&l) {
                        (ss.xs[l], sr.xs[l])
                    } else {
                        (0.0, Rf::k(0.0))
                    };
                    assert_within(
                        row[i],
                        s,
                        r,
                        coeff,
                        &format!("lrn size{size} beta{beta} col{si} j{jc} l{l}"),
                    );
                }
                // ほかの列への要素は 0。
                for (i, &a) in row.iter().enumerate() {
                    if i % inner != si {
                        assert_eq!(a, 0.0, "lrn: 列をまたぐ二階は 0");
                    }
                }
            }
        }
        true
    }

    #[test]
    fn weight_norm_emulation_reproduces_child_tape_second_order() {
        let (groups, rest) = (2usize, 3usize);
        let v0 = pattern(6, 2.0);
        let g0 = vec![0.7f32, -0.4];
        let w = pattern(6, 5.0);
        let tape = Tape::new_with_ops(test_ops());
        let child = Tape::new_with_ops(test_ops());
        let v = tape.var(&tt(v0.clone(), &[groups, rest]));
        let g = tape.var(&tt(g0.clone(), &[groups, 1]));
        let wc = tape.var(&tt(w.clone(), &[groups, rest]));
        let y = v.weight_norm(&g, Some(0)).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gv = cg.grad(&v).unwrap().unwrap();
        let gg = cg.grad(&g).unwrap().unwrap();
        let cv = cg.child_var(&v).unwrap().unwrap();
        let cgv = cg.child_var(&g).unwrap().unwrap();
        let cwc = cg.child_var(&wc).unwrap().unwrap();
        let coeff = wn_coeff(rest);
        for kg in 0..groups {
            let lo = kg * rest;
            let vk = &v0[lo..lo + rest];
            let mx = vk.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            let fs = wn_fwd::<f32>(vk, &w[lo..lo + rest], g0[kg], s).unwrap();
            let fr = wn_fwd::<Rf>(vk, &w[lo..lo + rest], g0[kg], s).unwrap();
            let (mut hv_s, mut hu_s) = (vec![0f32; rest], vec![0f32; rest]);
            let (mut hv_r, mut hu_r) = (vec![Rf::k(0.0); rest], vec![Rf::k(0.0); rest]);
            for seed in (0..rest).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
                let h_g_s = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
                let h_g_r = wn_rev(&fr, seed, &mut hv_r, &mut hu_r);
                let (row_v, row_u, row_g) = match seed {
                    WnSeed::Dv(j) => (
                        child_row(&child, &gv, lo + j, &cv),
                        child_row(&child, &gv, lo + j, &cwc),
                        Some(child_row(&child, &gv, lo + j, &cgv)),
                    ),
                    WnSeed::Dg => (
                        child_row(&child, &gg, kg, &cv),
                        child_row(&child, &gg, kg, &cwc),
                        None,
                    ),
                };
                for i in 0..rest {
                    assert_within(row_v[lo + i], hv_s[i], hv_r[i], coeff, "wn v");
                    assert_within(row_u[lo + i], hu_s[i], hu_r[i], coeff, "wn up");
                }
                // ほかのグループへの要素は 0。
                for (i, (&a, &b)) in row_v.iter().zip(&row_u).enumerate() {
                    if !(lo..lo + rest).contains(&i) {
                        assert_eq!((a, b), (0.0, 0.0), "wn: グループをまたぐ二階は 0");
                    }
                }
                if let (Some(row_g), Some(hs), Some(hr)) = (row_g, h_g_s, h_g_r) {
                    assert_within(row_g[kg], hs, hr, coeff, "wn g");
                }
            }
        }
    }

    #[test]
    fn spectral_norm_emulation_reproduces_child_tape_second_order() {
        let shape = [2usize, 3];
        let (u0, v0) = ([0.6f32, 0.8], [0.5f32, 0.3, 0.8]);
        let w0 = vec![0.9f32, 0.4, 0.5, 0.6, 0.7, 0.8];
        let up = pattern(6, 3.0);
        let tape = Tape::new_with_ops(test_ops());
        let child = Tape::new_with_ops(test_ops());
        let mut st = SpectralNormState::from_vectors(&shape, 0, &u0, &v0, 1, 1e-12).unwrap();
        let w = tape.var(&tt(w0.clone(), &shape));
        let wc = tape.var(&tt(up.clone(), &shape));
        let y = w.spectral_norm(&mut st, false).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gw = cg.grad(&w).unwrap().unwrap();
        let cw = cg.child_var(&w).unwrap().unwrap();
        let cwc = cg.child_var(&wc).unwrap().unwrap();
        // forward のスナップショット（`from_vectors` は u・v を正規化する。eval なので進まない）。
        let (us, vs) = (st.u().to_vec(), st.v().to_vec());
        let mut p = Vec::with_capacity(6);
        for &ui in &us {
            for &vj in &vs {
                p.push(f64::from(ui) * f64::from(vj));
            }
        }
        let mx = w0.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = f64::from(pow2(pow2_ceil_exponent(mx)));
        let coeff = sn_coeff(6);
        let fs = sn_fwd::<f32>(&w0, &p, &up, s).unwrap();
        let fr = sn_fwd::<Rf>(&w0, &p, &up, s).unwrap();
        let (mut hw_s, mut hu_s) = (vec![0f32; 6], vec![0f32; 6]);
        let (mut hw_r, mut hu_r) = (vec![Rf::k(0.0); 6], vec![Rf::k(0.0); 6]);
        for j in 0..6 {
            sn_rev(&fs, j, &mut hw_s, &mut hu_s);
            sn_rev(&fr, j, &mut hw_r, &mut hu_r);
            let row = child_row(&child, &gw, j, &cw);
            let row_up = child_row(&child, &gw, j, &cwc);
            for i in 0..6 {
                assert_within(row[i], hw_s[i], hw_r[i], coeff, "spectral W");
                assert_within(row_up[i], hu_s[i], hu_r[i], coeff, "spectral up");
            }
        }
    }

    /// NaN の要素の 1 対 1 の不変条件: 子テープの二階が有限な要素は f32 再現も有限で見積り内に一致し
    /// （再現の範囲外なら子テープは 0）、再現が NaN の要素は子テープも NaN。判定が両側 NaN の要素だけを
    /// 外すことの前提（NaN の到達範囲を構造の演算そのもので決める）。戻り値は（有限の要素数, NaN の要素数）。
    fn nan_pair(a: f32, s: Option<(f32, Rf)>, coeff: f64, ctx: &str) -> (usize, usize) {
        match s {
            Some((s, r)) if s.is_nan() => {
                assert!(r.v.is_nan(), "{ctx}: f32 再現だけが NaN");
                assert!(a.is_nan(), "{ctx}: 再現が NaN なのに子テープが有限 {a:e}");
                (0, 1)
            }
            Some((s, r)) => {
                if a.is_finite() {
                    assert_within(a, s, r, coeff, ctx);
                    (1, 0)
                } else {
                    // 再現の和に入らない `0·NaN` が子テープで届いた要素（厳しい側。判定は再現を検査する）。
                    assert!(a.is_nan(), "{ctx}: 子テープが ±inf");
                    (0, 1)
                }
            }
            None => {
                assert!(
                    a == 0.0 || a.is_nan(),
                    "{ctx}: 再現の範囲外は 0 か NaN: {a:e}"
                );
                (usize::from(a == 0.0), usize::from(a.is_nan()))
            }
        }
    }

    /// LRN の列に NaN が 1 つあるとき（窓は size 1・3）、上の不変条件が成り立ち、有限の要素も NaN の要素も
    /// 現れること（NaN の届かない窓の二階は判定され、子テープでも有限）。
    #[test]
    fn lrn_emulation_nan_reach_matches_child_tape() {
        let c = 5usize;
        let shape = [1usize, c, 1];
        let x0 = vec![f32::NAN, 0.6, -0.3, 0.8, 0.4];
        let w = vec![0.7f32, -0.5, 0.9, 0.3, -0.8];
        let (alpha, k) = (0.5f32, 1.0f32);
        // 偶数 size（窓のずれ）と、冪の指数 0 で NaN が消える β ∈ {0, −1, −2} を含む。
        let cases = [
            (1usize, 0.75f32),
            (3, 0.75),
            (2, 0.75),
            (2, -1.0),
            (3, -1.0),
            (3, 0.0),
            (2, 0.0),
            (3, -2.0),
            (2, -2.0),
        ];
        for (size, beta) in cases {
            let tape = Tape::new_with_ops(test_ops());
            let child = Tape::new_with_ops(test_ops());
            let x = tape.var(&tt(x0.clone(), &shape));
            let wc = tape.var(&tt(w.clone(), &shape));
            let y = x.local_response_norm(size, alpha, beta, k).unwrap();
            let loss = y.mul(&wc).unwrap().sum(None).unwrap();
            let cg = tape
                .backward_create_graph(&loss, &child)
                .unwrap_or_else(|e| {
                    panic!("size{size} beta{beta}: NaN の届かない窓は受理される: {e:?}")
                });
            let gx = cg.grad(&x).unwrap().unwrap();
            let cx = cg.child_var(&x).unwrap().unwrap();
            let cwc = cg.child_var(&wc).unwrap().unwrap();
            let rho = f64::from(alpha) / size as f64;
            let floor = (f64::from(k) * size as f64 / f64::from(alpha)).sqrt();
            let win = lrn_win(c, size);
            let mx = x0.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
            let e = pow2_ceil_exponent(mx.max(floor));
            let chunks: Vec<f64> = chunk_factors(&[-2.0 * f64::from(beta) * f64::from(e)])
                .iter()
                .map(|ch| ch[0])
                .collect();
            let kc = LrnColConst {
                s: f64::from(pow2(e)),
                kk: f64::from(k) / 2f64.powi(2 * e),
                rho,
                beta: f64::from(beta),
                chunks: &chunks,
            };
            let coeff = lrn_coeff(
                window_terms(c, win.bf, win.ff),
                window_terms(c, win.br, win.fr),
                chunks.len(),
                f64::from(beta),
            );
            let fs = lrn_col_fwd::<f32>(&x0, &w, &kc, win).unwrap();
            let fr = lrn_col_fwd::<Rf>(&x0, &w, &kc, win).unwrap();
            let fd = lrn_col_fwd::<Dm>(&x0, &w, &kc, win).unwrap();
            let (mut ss, mut sr) = (LrnScratch::<f32>::new(c), LrnScratch::<Rf>::new(c));
            let mut sd = LrnScratch::<Dm>::new(c);
            let (mut fin, mut nan) = (0usize, 0usize);
            for j in 0..c {
                lrn_col_rev(&fs, win, j, &mut ss);
                lrn_col_rev(&fr, win, j, &mut sr);
                lrn_col_rev(&fd, win, j, &mut sd);
                let row = child_row(&child, &gx, j, &cx);
                let row_up = child_row(&child, &gx, j, &cwc);
                let (ulo, uhi) = (j.saturating_sub(win.br), (j + win.fr).min(c - 1));
                let (xlo, xhi) = (ulo.saturating_sub(win.bf), (uhi + win.ff).min(c - 1));
                for i in 0..c {
                    let ctx = format!("lrn-nan size{size} beta{beta} j{j} i{i}");
                    // 依存の走査（`Dm`）の NaN は f32 の再現の NaN と一致する（冪の指数 0 の扱いを含む）。
                    if (xlo..=xhi).contains(&i) {
                        assert_eq!(sd.xs[i].nan, ss.xs[i].is_nan(), "{ctx}: Dm と f32（x）");
                    }
                    if (ulo..=uhi).contains(&i) {
                        assert_eq!(sd.up[i].nan, ss.up[i].is_nan(), "{ctx}: Dm と f32（up）");
                    }
                    let sx = (xlo..=xhi).contains(&i).then(|| (ss.xs[i], sr.xs[i]));
                    let su = (ulo..=uhi).contains(&i).then(|| (ss.up[i], sr.up[i]));
                    for (a, b) in [
                        nan_pair(row[i], sx, coeff, &ctx),
                        nan_pair(row_up[i], su, coeff, &format!("{ctx} up")),
                    ] {
                        fin += a;
                        nan += b;
                    }
                }
            }
            assert!(
                fin > 0 && nan > 0,
                "size{size} beta{beta}: 有限 {fin}・NaN {nan}"
            );
            // NaN から最も遠いチャネルの seed は、自分への二階が有限で判定に掛かる。
            assert!(ss.xs[c - 1].is_finite(), "size{size} beta{beta}");
        }
    }

    /// WeightNorm で g が NaN のグループ（と v が NaN のグループ）を含むとき、上の不変条件が成り立ち、
    /// g が NaN のグループでも `dg` への seed の `v`・`up` への二階と `dv` への seed の `g` への二階が有限で
    /// 再現と一致すること（判定から外さず検査する範囲）。
    #[test]
    fn weight_norm_emulation_nan_reach_matches_child_tape() {
        let (groups, rest) = (3usize, 3usize);
        let mut v0 = pattern(9, 2.0);
        v0[7] = f32::NAN;
        let g0 = vec![f32::NAN, -0.4, 0.5];
        let w = pattern(9, 5.0);
        let tape = Tape::new_with_ops(test_ops());
        let child = Tape::new_with_ops(test_ops());
        let v = tape.var(&tt(v0.clone(), &[groups, rest]));
        let g = tape.var(&tt(g0.clone(), &[groups, 1]));
        let wc = tape.var(&tt(w.clone(), &[groups, rest]));
        let y = v.weight_norm(&g, Some(0)).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gv = cg.grad(&v).unwrap().unwrap();
        let gg = cg.grad(&g).unwrap().unwrap();
        let cv = cg.child_var(&v).unwrap().unwrap();
        let cgv = cg.child_var(&g).unwrap().unwrap();
        let cwc = cg.child_var(&wc).unwrap().unwrap();
        let coeff = wn_coeff(rest);
        let (mut fin_g_nan_group, mut nan) = (0usize, 0usize);
        for kg in 0..groups {
            let lo = kg * rest;
            let vk = &v0[lo..lo + rest];
            let mx = vk.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            let fs = wn_fwd::<f32>(vk, &w[lo..lo + rest], g0[kg], s).unwrap();
            let fr = wn_fwd::<Rf>(vk, &w[lo..lo + rest], g0[kg], s).unwrap();
            let (mut hv_s, mut hu_s) = (vec![0f32; rest], vec![0f32; rest]);
            let (mut hv_r, mut hu_r) = (vec![Rf::k(0.0); rest], vec![Rf::k(0.0); rest]);
            for seed in (0..rest).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
                let h_g_s = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
                let h_g_r = wn_rev(&fr, seed, &mut hv_r, &mut hu_r);
                let (row_v, row_u, row_g) = match seed {
                    WnSeed::Dv(j) => (
                        child_row(&child, &gv, lo + j, &cv),
                        child_row(&child, &gv, lo + j, &cwc),
                        Some(child_row(&child, &gv, lo + j, &cgv)),
                    ),
                    WnSeed::Dg => (
                        child_row(&child, &gg, kg, &cv),
                        child_row(&child, &gg, kg, &cwc),
                        None,
                    ),
                };
                let mut pairs = Vec::new();
                for i in 0..rest {
                    pairs.push((row_v[lo + i], (hv_s[i], hv_r[i])));
                    pairs.push((row_u[lo + i], (hu_s[i], hu_r[i])));
                }
                if let (Some(row_g), Some(hs), Some(hr)) = (row_g, h_g_s, h_g_r) {
                    pairs.push((row_g[kg], (hs, hr)));
                }
                for (a, sr) in pairs {
                    let (f, n) = nan_pair(a, Some(sr), coeff, &format!("wn-nan group{kg}"));
                    if kg == 0 {
                        fin_g_nan_group += f;
                    }
                    nan += n;
                }
            }
        }
        // g が NaN のグループ: `dg` の seed の v・up への 2·rest と、`dv` の各 seed の g への rest。
        assert_eq!(fin_g_nan_group, 3 * rest, "g が NaN のグループの有限の要素");
        assert!(nan > 0);
    }

    /// Codex 指摘（PR #3950・未解決スレッド）の入力で、f32 再現がスケール復元の cotangent の
    /// underflow（2^-60·2^-100 = 2^-160 → 0）を再現し、判定が拒否することを直接確かめる。
    #[test]
    fn lrn_emulation_sees_scale_restoration_cotangent_underflow() {
        let x = [(-80f32).exp2()];
        let up = [1.0f32];
        let (alpha, beta) = (1e36f64, -1.0f64);
        let e = pow2_ceil_exponent(f64::from(x[0]));
        assert_eq!(e, -80);
        let chunks: Vec<f64> = chunk_factors(&[-2.0 * beta * f64::from(e)])
            .iter()
            .map(|ch| ch[0])
            .collect();
        assert_eq!(chunks.len(), 2, "2^-160 は 2 チャンクに分かれる");
        let kc = LrnColConst {
            s: f64::from(pow2(e)),
            kk: 0.0,
            rho: f64::from(alpha as f32),
            beta,
            chunks: &chunks,
        };
        let win = LrnWin {
            bf: 0,
            ff: 0,
            br: 0,
            fr: 0,
        };
        let fs = lrn_col_fwd::<f32>(&x, &up, &kc, win).unwrap();
        let fr = lrn_col_fwd::<Rf>(&x, &up, &kc, win).unwrap();
        assert_eq!(fs.ci, 0.0, "f32 の cotangent は 0 へ落ちる");
        assert!(fr.ci.v > 0.0);
        let (mut ss, mut sr) = (LrnScratch::<f32>::new(1), LrnScratch::<Rf>::new(1));
        lrn_col_rev(&fs, win, 0, &mut ss);
        lrn_col_rev(&fr, win, 0, &mut sr);
        assert_eq!(ss.xs[0], 0.0);
        let exact = 6.0 * f64::from(alpha as f32) * f64::from(x[0]);
        assert!(
            (sr.xs[0].v - exact).abs() <= 1e-6 * exact,
            "{:e}",
            sr.xs[0].v
        );
        assert!(entry_check(sr.xs[0], ss.xs[0], lrn_coeff(1, 1, 2, beta), "t").is_err());
    }

    #[test]
    fn entry_check_and_rf_rules() {
        let c = judge_coeff(4, 24, 0.0);
        assert!(entry_check(Rf::k(1.0), 1.0, c, "t").is_ok());
        // 値 0 だが大きさ 1e3（相殺）: 絶対許容を超える。
        assert!(entry_check(Rf::of(0.0, 1e3, false), 0.0, c, "t").is_err());
        // f32 だけが非有限・参照が非有限はどちらも拒否。
        assert!(entry_check(Rf::k(1.0), f32::NAN, c, "t").is_err());
        assert!(entry_check(Rf::k(f64::NAN), 1.0, c, "t").is_err());
        // 分母の相殺は κ として商の大きさへ入る（分母の m では割らない）。
        let den = Rf::k(1.0).minus(Rf::k(0.999));
        let q = Rf::k(1.0).over(den);
        assert!(q.m >= q.v.abs() * 1999.0 * 0.999, "{q:?}");
        assert!(judge_budget(JUDGE_BUDGET, "t").is_ok());
        assert!(judge_budget(JUDGE_BUDGET + 1, "t").is_err());
        // NaN の規則: 両側 NaN だけ外し、片側だけの NaN・±inf は拒否する。
        let nan = Rf::k(f64::NAN);
        assert!(nan.t, "入力の NaN の定数は taint 付き");
        // 演算で生まれた NaN（taint なし）は、両側 NaN でも外さず拒否する。
        let made = Rf::k(0.0).over(Rf::k(0.0));
        assert!(made.v.is_nan() && !made.t);
        assert!(entry_check(made, f32::NAN, c, "t").is_err());
        assert!(!same_nan(&[f32::NAN], &[made]));
        // taint は演算で伝わり、`pow_grad` の `e == 0`（定数 0）では伝わらない。
        assert!(nan.times(Rf::k(2.0)).t && !nan.pow_grad(0.0).t);
        assert!(entry_check(nan, f32::NAN, c, "t").is_ok());
        assert!(entry_check(Rf::k(1.0), f32::NAN, c, "t").is_err());
        assert!(entry_check(nan, 1.0, c, "t").is_err());
        assert!(entry_check(Rf::k(1.0), f32::INFINITY, c, "t").is_err());
        // forward の NaN の位置: 一致なら通し、f32 だけ・参照だけの NaN と長さ違いは拒否する。
        assert!(same_nan(&[1.0f32, f32::NAN], &[Rf::k(1.0), nan]));
        assert!(!same_nan(&[f32::NAN, 1.0f32], &[Rf::k(1.0), Rf::k(1.0)]));
        assert!(!same_nan(&[1.0f32, 1.0], &[Rf::k(1.0), nan]));
        assert!(!same_nan(&[1.0f32], &[Rf::k(1.0), Rf::k(1.0)]));
        assert!(no_inf(&[1.0f32, f32::NAN]) && !no_inf(&[f32::NEG_INFINITY]));
        assert!(!no_inf(&[Rf::of(1.0, f64::INFINITY, false)]));
    }

    /// ホストの 1 階の検査: 非有限は「NaN かつ到達範囲」だけ通し、到達範囲の inf・到達範囲外の NaN は拒否。
    #[test]
    fn host_nonfinite_check_rules() {
        assert!(host_nonfinite_check(&[1.0, f32::NAN], &[false, true], "t").is_ok());
        assert!(host_nonfinite_check(&[1.0, f32::NAN], &[true, false], "t").is_err());
        assert!(host_nonfinite_check(&[f32::INFINITY], &[true], "t").is_err());
        assert!(host_nonfinite_check(&[f32::NEG_INFINITY], &[false], "t").is_err());
        assert!(host_nonfinite_check(&[1.0], &[true, true], "t").is_err());
    }

    /// LRN の到達範囲は窓 2 段の和 `±(size−1)`、`col_nan` は NaN を含む列（[`lrn_col_scan`] で走査する列）。
    #[test]
    fn lrn_nan_masks_follow_windows() {
        let x = [f32::NAN, 1.0, 1.0, 1.0, 1.0];
        let (reach, col_nan) = lrn_nan_masks(&x, (1, 5, 1), 2);
        assert_eq!(reach, vec![true, true, false, false, false]);
        assert_eq!(col_nan, vec![true]);
        let (reach, _) = lrn_nan_masks(&x, (1, 5, 1), 1);
        assert_eq!(reach, vec![true, false, false, false, false]);
        // inner 軸の列は独立。
        let (reach, col_nan) = lrn_nan_masks(&[f32::NAN, 1.0, 1.0, 1.0], (1, 2, 2), 3);
        assert_eq!(
            (reach, col_nan),
            (vec![true, false, true, false], vec![true, false])
        );
    }

    /// NaN を含む LRN の列の分類。全窓 NaN でも偶数 size・`β = −1`（`b = d^0 = 1`）では二階に有限の経路が
    /// 残る（Codex P1・Bugbot の入力）ので外さず、奇数 size の同じ入力（P2）は全要素 NaN なので外す。
    #[test]
    fn lrn_col_scan_classifies_nan_columns() {
        let x = [f32::NAN, 1e30];
        let up = [1.0f32, 1.0];
        let (fin, mx) = lrn_col_scan(&x, &up, lrn_win(2, 2), -1.0);
        assert!(fin, "size 2・β = −1 は有限の二階の経路が残る");
        assert_eq!(mx, f64::from(1e30f32));
        let (fin, _) = lrn_col_scan(&x, &up, lrn_win(2, 3), -1.0);
        assert!(!fin, "size 3・β = −1 は全要素 NaN");
        let (fin, _) = lrn_col_scan(&x, &up, lrn_win(2, 2), 0.75);
        assert!(!fin, "size 2・β = 0.75 は全要素 NaN");
    }

    /// `v` に NaN がある WeightNorm のグループと `W` に NaN がある SpectralNorm は、1 階・二階の全要素が
    /// 入力の NaN に依存する（外してよい）。NaN の無い入力は外さない（述語が空に真にならない）。
    #[test]
    fn wn_sn_nan_scans_cover_all_entries() {
        let up = [1.0f32, -0.5, 0.25];
        assert!(wn_group_all_nan(&[f32::NAN, 1.0, 2.0], &up, 1.5));
        assert!(wn_group_all_nan(&[f32::NAN, 1.0, 2.0], &up, f32::NAN));
        assert!(!wn_group_all_nan(&[0.5, 1.0, 2.0], &up, 1.5));
        assert!(!wn_group_all_nan(&[0.5, 1.0, 2.0], &up, f32::NAN));
        let p = [0.25f64, 0.0, -0.5];
        assert!(sn_all_nan(&[f32::NAN, 1.0, 2.0], &p, &up));
        assert!(sn_all_nan(&[1.0, 2.0, f32::NAN], &p, &up));
        assert!(!sn_all_nan(&[0.5, 1.0, 2.0], &p, &up));
    }

    /// LRN の窓（`lrn_cgrad` と同じ組み立て）。
    fn lrn_win(c: usize, size: usize) -> LrnWin {
        LrnWin {
            bf: (size / 2).min(c - 1),
            ff: ((size - 1) / 2).min(c - 1),
            br: ((size - 1) / 2).min(c - 1),
            fr: (size / 2).min(c - 1),
        }
    }

    /// 走査量が forward の窓和（`win_sum`）と `lrn_col_rev` の seed ごとのループ（初期化・反転窓・二重走査・
    /// 戻し）・判定の反復回数を下回らないこと、Codex の入力（`C = 4096`・`size = 8191`）が上限を超えること（二重走査を数える）。
    #[test]
    fn judge_cost_counts_inner_loops() {
        for c in [1usize, 2, 3, 5, 8, 17] {
            for size in [1usize, 2, 3, 4, 5, 9, 2 * c + 3] {
                let w = lrn_win(c, size);
                let mut actual = c * (w.bf + w.ff + 1) + c * (w.br + w.fr + 1);
                for j in 0..c {
                    let (ulo, uhi) = (j.saturating_sub(w.br), (j + w.fr).min(c - 1));
                    let (xlo, xhi) = (ulo.saturating_sub(w.bf), (uhi + w.ff).min(c - 1));
                    let (sx, su) = (xhi - xlo + 1, uhi - ulo + 1);
                    actual += 3 * sx + 2 * su;
                    for m in ulo..=uhi {
                        actual += (m + w.ff).min(c - 1) - m.saturating_sub(w.bf) + 1;
                    }
                }
                let cost = lrn_judge_cost(1, c, w, 0);
                assert!(cost >= actual, "c={c} size={size}: {cost} < {actual}");
            }
        }
        let codex = lrn_judge_cost(1, 4096, lrn_win(4096, 8191), 1);
        assert!(codex > 4096 * 4096 * 4096, "{codex}");
        assert!(judge_budget(codex, "t").is_err());
        // 典型的な小さい窓は上限内。
        assert!(judge_budget(lrn_judge_cost(64, 16, lrn_win(16, 5), 1), "t").is_ok());
        // WeightNorm・SpectralNorm は seed 数 × 長さ（二乗）で数える。
        assert_eq!(wn_judge_cost(1, 2), 2 * 18 + 3 * 15);
        assert_eq!(sn_judge_cost(3), 3 * 23 + 3 * 12);
        assert!(judge_budget(wn_judge_cost(1, 4400), "t").is_err());
        assert!(judge_budget(sn_judge_cost(6000), "t").is_err());
    }
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

    /// 判定方式（[`Lin`]／従来の [`Rf`]。後者は幅の広いグループ・グループ数の多いテンソルの経路）のどちらでも、
    /// 相殺のない普通のグループは受理される。
    #[test]
    fn wn_group_judge_modes_agree_on_easy_group() {
        let (v, up) = ([0.9f32, 0.1, 0.6, 0.3], [0.7f32, -0.2, 0.4, 0.1]);
        for use_lin in [true, false] {
            assert!(wn_group_judge(&v, &up, 0.5, 1.0, wn_coeff(4), use_lin).is_ok());
        }
    }
}
