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
//!   SpectralNorm の照合の見積りは `Lin` の上界 + ホスト側の丸めの上界（[`sn_shadow_bound`]）で、出せない
//!   ときだけ従来の `rounding_coeff(N+4)·m_abs`（#4081。docs の決定記録 §81.9）。
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

use super::lin::{
    LIN_BUDGET, LIN_REST_MAX, LIN_U, Lin, Num, entry_check_lin, lin_events, lin_group_cost,
    lin_mark, lin_reset, lin_reset_sparse, lin_truncate,
};
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

pub(super) fn pow2(e: i32) -> f32 {
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
pub(super) fn wn_judge_cost(groups: usize, rest: usize) -> usize {
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

/// SpectralNorm の総括判定を [`Lin`] で行う走査量の見積り（[`Rf`] の 1 反復相当）。実測（`--release`・x86_64 Linux・
/// `spectral_norm_lin_judge_cost_measurement`。`σ_s`・`sp`・`σ_s` への cotangent の係数列を [`Num::settle`] で束ねた
/// 後）は N = 16 で約 6.7e4、64 で約 9.0e5、144 で約 4.9e6、256 で約 1.6e7、400 で約 3.5e7 で、N の 2 乗に比例する。
/// 全点を覆うよう 300·N² とする。
fn sn_lin_cost(n: usize) -> usize {
    300usize.saturating_mul(n).saturating_mul(n)
}

/// [`Lin`] 版の事象の総数の上界（forward の分と、seed 1 回分の逆伝播の分。実測は 20·N + 12）。事象の数は値に
/// ほとんど依らず N で決まる（定数の丸めの事象が無いときだけ減る）。テスト `sn_lin_events_stay_within_bound` が固定する。
fn sn_lin_events_max(n: usize) -> usize {
    20usize.saturating_mul(n).saturating_add(16)
}

/// WeightNorm の疎モードの [`Lin`] 版の事象数 `N`（forward の分と seed 1 回分の逆伝播の分）の上界。実測の最大は
/// 約 `34·rest + 12`（幅 2〜256・振幅 1e-3〜1e3。厳密モードの `18·rest + 17` の約 2 倍）に余裕を乗せて
/// `40·rest + 40` とする。テスト `wn_lin_events_stay_within_bound` が固定する（決定記録 §80.5）。
fn wn_lin_events_max(rest: usize) -> usize {
    40usize.saturating_mul(rest).saturating_add(40)
}

/// WeightNorm の総括判定で [`Lin`] を選ぶか（テンソル単位）。幅が [`LIN_REST_MAX`] 以下、走査量が
/// [`LIN_BUDGET`] 以内、かつ事象の総数が余裕係数の上限（`16·事象数·u < 0.01`）に収まるときだけ。収まらない
/// テンソルは従来の [`Rf`] のままにする（`sn_use_lin` と同じ方針。#4082・§80.5）。
fn wn_use_lin(groups: usize, rest: usize) -> bool {
    rest <= LIN_REST_MAX
        && groups.saturating_mul(lin_group_cost(rest)) <= LIN_BUDGET
        && 16.0 * wn_lin_events_max(rest) as f64 * LIN_U < 0.01
}

/// SpectralNorm の総括判定で [`Lin`] を選ぶか。走査量が [`LIN_BUDGET`] 以内で、かつ事象の総数が
/// [`entry_check_lin`] の余裕係数の上限（`16·事象数·u < 0.01`）に収まるときだけ。収まらないテンソルは従来の
/// [`Rf`] のままにする（`Lin` を選んで全要素が余裕係数の上限で拒否される、という拒否範囲の拡大を避ける）。
fn sn_use_lin(n: usize) -> bool {
    sn_lin_cost(n) <= LIN_BUDGET && 16.0 * sn_lin_events_max(n) as f64 * LIN_U < 0.01
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
pub(super) fn wn_coeff(rest: usize) -> f64 {
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
pub(super) struct WnFwd<T> {
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
pub(super) fn wn_fwd<T: Num>(v: &[f32], up: &[f32], g: f32, s: f64) -> Option<WnFwd<T>> {
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
pub(super) enum WnSeed {
    Dv(usize),
    Dg,
}

/// seed の二階の逆伝播を構造の演算順で再現する。`v`・`up` への要素を `hv`・`hu` へ書き、`g` への
/// 要素（`Dv` のときだけ）を返す。
pub(super) fn wn_rev<T: Num>(f: &WnFwd<T>, seed: WnSeed, hv: &mut [T], hu: &mut [T]) -> Option<T> {
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
/// （[`entry_check_lin`]。呼び出し側が [`wn_use_lin`] で選ぶ）、
/// そうでなければ従来の [`Rf`] と係数 `coeff`（[`entry_check`]）。
pub(super) fn wn_group_judge(
    v: &[f32],
    up: &[f32],
    g: f32,
    s: f64,
    coeff: f64,
    use_lin: bool,
) -> Result<(), AutodiffError> {
    if use_lin {
        // WeightNorm だけが疎モード（小さな誤差項を残余事象へまとめる。#4078・§80.4）を使う。
        lin_reset_sparse();
        wn_group_judge_with::<Lin>(v, up, g, s, &|r, x, what| {
            entry_check_lin(r, x, what, invalid)
        })
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
    let use_lin = wn_use_lin(groups, rest);
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
pub(super) struct SnFwd<T> {
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
    /// `σ_s = ΣW_s⊙P` と `sp = Σup⊙Ŵ`（[`Num::settle`] の anchors として逆伝播が使う）。
    sig: T,
    sp: T,
    /// 1 階 `dW = (t/σ_s)/s`（[`sn_shadow_bound`] が `Lin` の上界を読む）。
    dw: Vec<T>,
}

/// `spectral_norm_cgrad` が積む構造（`W_s = W/s` 〜 `dW = (up − Σ(up⊙Ŵ)·P)/σ_s/s`）の forward を再現する。
/// `p` は `P[i] = u_i·v_j` の f64 の厳密値（構造は f32 へ丸めた値を載せる）。
pub(super) fn sn_fwd<T: Num>(w: &[f32], p: &[f64], up: &[f32], s: f64) -> Option<SnFwd<T>> {
    let s = T::k(s);
    let inv_s = T::k(1.0).over(s);
    let ws: Vec<T> = w.iter().map(|&x| T::k(f64::from(x)).over(s)).collect();
    let p: Vec<T> = p.iter().map(|&x| T::k(x)).collect();
    let up: Vec<T> = up.iter().map(|&x| T::k(f64::from(x))).collect();
    // 縮約は [`Num::total`] で書く（f32・`Rf`・`Dm` は従来と同じ逐次和、`Lin` は和の順序に依らない上界）。
    let sig_terms: Vec<T> = ws.iter().zip(&p).map(|(&a, &b)| a.times(b)).collect();
    // 全要素に依存する `σ_s` の係数列は 1 つの事象へ束ねる（束ねた事象は共有の誤差として全経路で係数が残る）。
    let sig = T::total(&sig_terms).settle(&[]);
    let inv_sig = T::k(1.0).over(sig);
    let wh: Vec<T> = ws.iter().map(|&x| x.over(sig)).collect();
    let dwb: Vec<T> = ws.iter().map(|&x| x.over(sig).over(sig).negate()).collect();
    let sp_terms: Vec<T> = up.iter().zip(&wh).map(|(&u, &h)| u.times(h)).collect();
    let sp = T::total(&sp_terms).settle(&[sig, inv_s]);
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
        sig,
        sp,
        dw,
    })
}

/// 構造側の 1 階の丸めの上界 `B_str`（要素ごと）。[`sn_fwd`] を [`Lin`] で 1 回だけ回し（`O(N log N)`。
/// 二階の判定 [`sn_judge`] の `O(N²)` とは別）、`dW` の一次の上界に [`entry_check_lin`] と同じ二次項の
/// 余裕 `1 + 16·E·u` を掛ける。和の順序に依らない上界なので、他バックエンドの和の順序でも覆う。
/// `16·E·u ≥ 0.01`・非有限・上界を作れない場合は `None`（呼び出し側が従来の見積りへ戻す）。
fn sn_str_bound(w: &[f32], p: &[f64], up: &[f32], s: f64) -> Option<Vec<f64>> {
    // 事象数は N に比例する。上限を超える N では領域を作る前に諦める（[`sn_use_lin`] と同じ事象数の見積り）。
    if 16.0 * sn_lin_events_max(w.len()) as f64 * LIN_U >= 0.01 {
        return None;
    }
    lin_reset();
    let out = sn_fwd::<Lin>(w, p, up, s).and_then(|f| {
        let slack = 16.0 * lin_events() as f64 * LIN_U;
        let b: Vec<f64> = f.dw.iter().map(|d| d.bound() * (1.0 + slack)).collect();
        (slack < 0.01 && b.iter().all(|x| x.is_finite())).then_some(b)
    });
    lin_reset();
    out
}

/// ホスト側の丸めの上界 `B_host`（要素ごと）。`wr::spectral_norm_vjp_host` の演算列を数える（その実装を
/// 変えるときはここも更新する。ホスト側の健全性テストが食い違いを検出する）:
/// - `σ_h`: f64 の行和・外側の和。`|σ_h − σ| ≤ γ_{N+2}·A`（`A = Σ|W||P|`。`W·v`・`u·row` の積は f64 で
///   厳密または 1 丸め）。
/// - `S_h = Σ f64(f32(up·W))`: f32 の積の丸め `u/(1−u)·B + N·2^-150` と f64 の和 `γ_N·B`（`B = Σ|f32(up·W)|`）。
/// - `val = up/σ_h − (S_h/σ_h²)·u_i·v_j`: `σ_h` の相対誤差 `ε` を `1/σ` と `1/σ²` へ（`ε/(1−ε)`・
///   `ε(2+ε)/(1−ε)²`）、`S_h` の誤差、f64 の演算の丸め `γ_5·(|up/σ_h| + |S_h/σ_h²·P|)`。
/// - 最後の `as f32`: `u(1+2u)|host| + 2^-150`。
///
/// `ε ≥ 0.01`・非有限・`σ_h = 0` は `None`。一次近似の余裕 `1 + 8ε` を f64 評価の項に掛ける。
fn sn_host_bound(w: &[f32], p: &[f64], up: &[f32], sigma_h: f64, host: &[f32]) -> Option<Vec<f64>> {
    const D: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53
    const TINY: f64 = 7.006_492_321_624_085e-46; // 2^-150
    let n = w.len() as f64;
    let gamma = |k: f64| k * D / (1.0 - k * D);
    if !(sigma_h.is_finite() && sigma_h != 0.0 && (n + 2.0) * D < 0.01) || host.len() != w.len() {
        return None;
    }
    let (mut a, mut b, mut s_h) = (0.0f64, 0.0f64, 0.0f64);
    for ((&x, &pk), &g) in w.iter().zip(p).zip(up) {
        a += f64::from(x).abs() * pk.abs();
        let term = g * x;
        b += f64::from(term).abs();
        s_h += f64::from(term);
    }
    let inv = 1.0 / sigma_h.abs();
    let inv2 = inv * inv;
    let eps = gamma(n + 2.0) * a * inv;
    if !(eps.is_finite() && eps < 0.01) {
        return None;
    }
    let ds = LIN_U / (1.0 - LIN_U) * b + n * TINY + gamma(n) * b;
    let k1 = eps / (1.0 - eps);
    let k2 = eps * (2.0 + eps) / ((1.0 - eps) * (1.0 - eps));
    let out: Vec<f64> = (0..w.len())
        .map(|i| {
            let (g, pi) = (f64::from(up[i]).abs(), p[i].abs());
            let t1 = g * inv * k1;
            let t2 = (ds * inv2 + (s_h.abs() + ds) * inv2 * k2) * pi;
            let t3 = gamma(5.0) * (g * inv + s_h.abs() * inv2 * pi);
            let t4 = LIN_U * (1.0 + 2.0 * LIN_U) * f64::from(host[i]).abs() + TINY;
            (t1 + t2 + t3) * (1.0 + 8.0 * eps) + t4
        })
        .collect();
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// [`shadow_check`] の SpectralNorm 用の見積り `B_str·(1 + 16·E·u) + B_host`（要素ごと。係数 1 で渡す）。
/// `|composed − true| ≤ |composed − host| + |host − true|` の第 2 項を `B_host` が、他の和の順序の構造の
/// ずれを `B_str` が覆う。どちらかを出せない場合は `None`（従来の `rounding_coeff(N+4)·m_abs` のまま）。
/// `spectral_norm_cgrad` から呼ばれる。
pub(super) fn sn_shadow_bound(
    w: &[f32],
    p: &[f64],
    up: &[f32],
    s: f64,
    sigma_h: f64,
    host: &[f32],
) -> Option<Vec<f64>> {
    let st = sn_str_bound(w, p, up, s)?;
    let ho = sn_host_bound(w, p, up, sigma_h, host)?;
    let b: Vec<f64> = st.iter().zip(&ho).map(|(a, h)| a + h).collect();
    b.iter().all(|x| x.is_finite()).then_some(b)
}

/// seed `e_j`（`dW_j` に 1）の二階の逆伝播を構造の演算順で再現し、`W`・`up` への要素を `hw`・`hu` へ書く。
pub(super) fn sn_rev<T: Num>(f: &SnFwd<T>, j: usize, hw: &mut [T], hu: &mut [T]) {
    let n = f.ws.len();
    // dW = (t/σ_s)/s。
    let cot_o = f.inv_s;
    let cot_t = cot_o.times(f.inv_sig);
    // `σ_s` への cotangent は、分母側の Div 係数からの 1 項と `Ŵ` の n 項の縮約（[`Num::total`]）。
    let mut sig_terms = Vec::with_capacity(n + 1);
    sig_terms.push(cot_o.times(f.dob[j]));
    // t = up − sp·P（sp は [1] のブロードキャスト）、sp = Σ up⊙Ŵ。
    let cot_sp = cot_t.negate().times(f.p[j]);
    for i in 0..n {
        hu[i] = cot_sp.times(f.wh[i]);
        let cot_wh = cot_sp.times(f.up[i]);
        hw[i] = cot_wh.times(f.inv_sig);
        sig_terms.push(cot_wh.times(f.dwb[i]));
    }
    let cot_sig = T::total(&sig_terms).settle(&[f.sig, f.sp, f.inv_sig, f.inv_s, cot_t, cot_sp]);
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

/// SpectralNorm の総括判定。seed が 1 要素のときの二階（`W`・`up` への要素）を判定する。`use_lin` なら
/// 丸めの一次伝播を事象ごとに追う [`Lin`]（[`entry_check_lin`]。呼び出し側が [`sn_use_lin`] で選ぶ）、
/// そうでなければ従来の [`Rf`] と係数 `coeff`（[`entry_check`]）。`spectral_norm_cgrad` から呼ばれる。
pub(super) fn sn_judge(
    w: &[f32],
    p: &[f64],
    up: &[f32],
    s: f64,
    coeff: f64,
    use_lin: bool,
) -> Result<(), AutodiffError> {
    if use_lin {
        lin_reset();
        let r = sn_judge_with::<Lin>(w, p, up, s, &|r, x, what| {
            entry_check_lin(r, x, what, invalid)
        });
        // 係数列の領域は N に比例して伸びるので、判定後に解放する。
        lin_reset();
        r
    } else {
        sn_judge_with::<Rf>(w, p, up, s, &|r, x, what| entry_check(r, x, coeff, what))
    }
}

fn sn_judge_with<R: Num>(
    w: &[f32],
    p: &[f64],
    up: &[f32],
    s: f64,
    check: &dyn Fn(R, f32, &str) -> Result<(), AutodiffError>,
) -> Result<(), AutodiffError> {
    const WHAT: &str = "SpectralNorm の二階";
    let (Some(fs), Some(fr)) = (sn_fwd::<f32>(w, p, up, s), sn_fwd::<R>(w, p, up, s)) else {
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
    // forward の事象は seed 間で共有し、逆伝播の事象だけ seed ごとに戻す（領域を持たない型では無害）。
    let mark = lin_mark();
    for j in 0..n {
        lin_truncate(mark);
        let (mut hw_r, mut hu_r) = (vec![R::k(0.0); n], vec![R::k(0.0); n]);
        sn_rev(&fs, j, &mut hw_s, &mut hu_s);
        sn_rev(&fr, j, &mut hw_r, &mut hu_r);
        for i in 0..n {
            check(hw_r[i], hw_s[i], WHAT)?;
            check(hu_r[i], hu_s[i], WHAT)?;
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
    let mut sh_bound: Option<Vec<f64>> = None;
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
        sn_judge(
            &w_data,
            &p_exact,
            &up_data,
            sf,
            sn_coeff(w_data.len()),
            sn_use_lin(w_data.len()),
        )?;
        let sigma_h = wr::spectral_sigma_host(&w_data, &layout, &u_data, &v_data)
            .map_err(AutodiffError::Shape)?;
        sh_bound = sn_shadow_bound(&w_data, &p_exact, &up_data, sf, sigma_h, &host_vec);
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
    // 見積り: 出せるときは `Lin` の上界 + ホスト側の丸めの上界（係数 1）、出せないときは従来の
    // `rounding_coeff(N+4)·m_abs`。`W` に NaN があるときは従来どおり NaN の `m_abs` のまま。
    match &sh_bound {
        Some(b) => shadow_check(child, &composed, &host_vec, b, 1.0, "SpectralNorm")?,
        None => shadow_check(
            child,
            &composed,
            &host_vec,
            &m_abs,
            rounding_coeff(w_data.len() + 4),
            "SpectralNorm",
        )?,
    }
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

    // --- イシュー #4079: SpectralNorm の拒否の内訳の計測（docs/autodiff-higher-order-grad-decision.md §81） ---
    //
    // 計測の対象は本番の `spectral_norm_cgrad` の 2 段の保守的判定（構造を積む前の `sn_judge`・積んだ後の
    // `shadow_check`）と、その手前の走査量の上限・段検査。本番コードは一切変えず、(1) 公開経路
    // （`backward_create_graph`）の拒否を文面で分類し（計測用の分類であり、本番が文面に依存する意味ではない）、
    // (2) 各判定の「余裕」（`err / 閾値`。1 以上で拒否）を同じ式で再計算し、(3) 1 階・二階を f64 の閉じた式と
    // 突き合わせる。後続の #4080／#4081 が「受理に変える入力」として再利用する入力は
    // [`sn_followup_candidates`] に定義する。

    /// 上流の形。
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum SnUp {
        /// `pattern(N, 3.0)` に `scale` を掛けた一般形（`P` に平行でない）。
        General,
        /// `scale·P`（f32）。`sp = scale` で `t = up − sp·P` が相殺する対照。
        ParallelP,
    }

    /// 計測・後続用の 1 入力の定義。再構成の手順は §81.5。
    #[derive(Clone, Debug)]
    struct SnCase {
        shape: Vec<usize>,
        dim: usize,
        /// 重みの振幅。`w = amp·pattern(N, 1.0) + alpha·amp·P`。
        amp: f32,
        /// `P = u vᵀ` 成分の強さ（0 は σ が相殺しうる）。
        alpha: f32,
        up: SnUp,
        scale: f32,
    }

    struct SnInputs {
        w: Vec<f32>,
        up: Vec<f32>,
        p: Vec<f32>,
        p_exact: Vec<f64>,
        u: Vec<f32>,
        v: Vec<f32>,
    }

    /// `P[i, j] = u_i·v_j` を W と同じ並びに置く（`spectral_norm_cgrad` と同じ式）。
    fn sn_p(shape: &[usize], dim: usize, u: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f64>) {
        let layout = wr::spectral_norm_layout(shape, dim).unwrap();
        let (h, cols) = (layout.rows(), layout.cols());
        let inner: usize = shape[dim + 1..].iter().product();
        let n: usize = shape.iter().product();
        let (mut p, mut pe) = (vec![0.0f32; n], vec![0.0f64; n]);
        for (i, &ui) in u.iter().enumerate().take(h) {
            for (j, &vj) in v.iter().enumerate().take(cols) {
                let at = (j / inner * h + i) * inner + j % inner;
                p[at] = ui * vj;
                pe[at] = f64::from(ui) * f64::from(vj);
            }
        }
        (p, pe)
    }

    fn sn_state(c: &SnCase) -> SpectralNormState {
        let layout = wr::spectral_norm_layout(&c.shape, c.dim).unwrap();
        SpectralNormState::from_vectors(
            &c.shape,
            c.dim,
            &pattern(layout.rows(), 2.0),
            &pattern(layout.cols(), 6.0),
            1,
            1e-12,
        )
        .unwrap()
    }

    fn sn_inputs(c: &SnCase) -> SnInputs {
        let st = sn_state(c);
        let (u, v) = (st.u().to_vec(), st.v().to_vec());
        let (p, p_exact) = sn_p(&c.shape, c.dim, &u, &v);
        let n: usize = c.shape.iter().product();
        let base = pattern(n, 1.0);
        let w: Vec<f32> = (0..n)
            .map(|k| c.amp * base[k] + c.alpha * c.amp * p[k])
            .collect();
        let up: Vec<f32> = match c.up {
            SnUp::General => pattern(n, 3.0).iter().map(|x| c.scale * x).collect(),
            SnUp::ParallelP => p.iter().map(|x| c.scale * x).collect(),
        };
        SnInputs {
            w,
            up,
            p,
            p_exact,
            u,
            v,
        }
    }

    /// 1 階の f64 の閉じた式 `dW = (up − (Σ up·W/σ)·P)/σ`（`σ = Σ W·P`。`s` は約分される）。
    fn sn_dw64(w: &[f32], p: &[f64], up: &[f32]) -> Vec<f64> {
        let sigma: f64 = w.iter().zip(p).map(|(&a, &b)| f64::from(a) * b).sum();
        let a: f64 = up
            .iter()
            .zip(w)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        up.iter()
            .zip(p)
            .map(|(&x, &pi)| (f64::from(x) - a / sigma * pi) / sigma)
            .collect()
    }

    /// 二階の f64 の閉じた式。戻り値は（`∂dW_j/∂W_i` の行 j、`∂dW_j/∂up_i` の行 j）。
    fn sn_hess64(w: &[f32], p: &[f64], up: &[f32], j: usize) -> (Vec<f64>, Vec<f64>) {
        let sigma: f64 = w.iter().zip(p).map(|(&a, &b)| f64::from(a) * b).sum();
        let a: f64 = up
            .iter()
            .zip(w)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let n = w.len();
        let hw = (0..n)
            .map(|i| {
                -f64::from(up[j]) * p[i] / (sigma * sigma)
                    - (f64::from(up[i]) / (sigma * sigma) - 2.0 * a * p[i] / sigma.powi(3)) * p[j]
            })
            .collect();
        let hu = (0..n)
            .map(|i| {
                let d = if i == j { 1.0 / sigma } else { 0.0 };
                d - f64::from(w[i]) * p[j] / (sigma * sigma)
            })
            .collect();
        (hw, hu)
    }

    /// REQ-2 の統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）に対する `|差|` の比。1 未満なら合格。
    /// 分母は絶対 1e-5 と相対 1e-3·|真値| の大きい方。
    fn req2_ratio(actual: f64, truth: f64) -> f64 {
        let d = (actual - truth).abs();
        d / ABSOLUTE_RESCUE_THRESHOLD.max(RELATIVE_TOLERANCE * truth.abs())
    }

    /// 判定の最悪要素の余裕（`err / 閾値`。判定は [`entry_check`]・`struct_check_unscaled` と同じ式）。
    #[derive(Clone, Copy, Default, Debug)]
    struct Worst {
        ratio: f64,
        diff: f64,
        est: f64,
        v: f64,
        m: f64,
        set: bool,
    }

    impl Worst {
        fn take(&mut self, v: f64, v_str: f64, m: f64, coeff: f64) {
            self.take_with(v, v_str, m, coeff * m);
        }

        /// 誤差の見積り `est`（[`Lin`] の `bound·(1 + 16·N·u)`）を直接与える版。`m` は 0 で記録する。
        fn take_est(&mut self, v: f64, v_str: f64, est: f64) {
            self.take_with(v, v_str, 0.0, est);
        }

        fn take_with(&mut self, v: f64, v_str: f64, m: f64, est: f64) {
            let diff = (v_str - v).abs();
            let err = diff + est;
            let thr = ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * v.abs());
            let ratio = if err.is_finite() {
                err / thr
            } else {
                f64::INFINITY
            };
            if !self.set || ratio > self.ratio {
                *self = Worst {
                    ratio,
                    diff,
                    est,
                    v,
                    m,
                    set: true,
                };
            }
        }
    }

    /// `sn_judge` と同じループで最悪の余裕を求める。戻り値の `bool` は検査が全要素で通すか。`use_lin` なら
    /// [`Lin`] の判定（[`entry_check_lin`]）、そうでなければ [`Rf`] の判定（[`entry_check`]）。
    /// `sn_fwd` が非有限で `None` のときは `None`。
    fn sn_judge_margins(
        w: &[f32],
        p: &[f64],
        up: &[f32],
        s: f64,
        coeff: f64,
        use_lin: bool,
    ) -> Option<(Worst, bool)> {
        let n = w.len();
        let fs = sn_fwd::<f32>(w, p, up, s)?;
        let (mut hw_s, mut hu_s) = (vec![0f32; n], vec![0f32; n]);
        let (mut worst, mut all_ok) = (Worst::default(), true);
        if use_lin {
            lin_reset();
            let fl = sn_fwd::<Lin>(w, p, up, s)?;
            let mark = lin_mark();
            for j in 0..n {
                lin_truncate(mark);
                let (mut hw_l, mut hu_l) = (vec![Lin::k(0.0); n], vec![Lin::k(0.0); n]);
                sn_rev(&fs, j, &mut hw_s, &mut hu_s);
                sn_rev(&fl, j, &mut hw_l, &mut hu_l);
                let slack = 1.0 + 16.0 * lin_events() as f64 * LIN_U;
                for i in 0..n {
                    for (r, sv) in [(hw_l[i], hw_s[i]), (hu_l[i], hu_s[i])] {
                        all_ok &= entry_check_lin(r, sv, "m", invalid).is_ok();
                        if !(sv.is_nan() && r.input_nan()) {
                            worst.take_est(r.v, f64::from(sv), r.bound() * slack);
                        }
                    }
                }
            }
            lin_reset();
        } else {
            let fr = sn_fwd::<Rf>(w, p, up, s)?;
            let (mut hw_r, mut hu_r) = (vec![Rf::k(0.0); n], vec![Rf::k(0.0); n]);
            for j in 0..n {
                sn_rev(&fs, j, &mut hw_s, &mut hu_s);
                sn_rev(&fr, j, &mut hw_r, &mut hu_r);
                for i in 0..n {
                    for (r, sv) in [(hw_r[i], hw_s[i]), (hu_r[i], hu_s[i])] {
                        all_ok &= entry_check(r, sv, coeff, "m").is_ok();
                        if !(sv.is_nan() && r.input_nan()) {
                            worst.take(r.v, f64::from(sv), r.m, coeff);
                        }
                    }
                }
            }
        }
        Some((worst, all_ok))
    }

    /// 子テープに `spectral_norm_cgrad` の構造を同じ演算順で組み、`composed`（1 階）を返す。
    fn sn_build<'c>(
        child: &'c Tape,
        w_m: &Var<'c>,
        up: &Var<'c>,
        p: &[f32],
        s: f32,
        shape: &[usize],
    ) -> Var<'c> {
        let s_c = scalar_const(child, s).unwrap();
        let p_t = tensor_const(child, p.to_vec(), shape).unwrap();
        let w_s = w_m.div(&s_c).unwrap();
        let sigma_s = w_s
            .mul(&p_t)
            .unwrap()
            .sum(None)
            .unwrap()
            .reshape(&[1])
            .unwrap();
        let w_hat = w_s.div(&sigma_s).unwrap();
        let sp = up
            .mul(&w_hat)
            .unwrap()
            .sum(None)
            .unwrap()
            .reshape(&[1])
            .unwrap();
        let t = up.sub(&sp.mul(&p_t).unwrap()).unwrap();
        t.div(&sigma_s).unwrap().div(&s_c).unwrap()
    }

    /// 本番と同じ `m_abs`（1 階の項の絶対値和）。`σ_s` が 0／非有限なら `None`。
    fn sn_m_abs(inp: &SnInputs, s: f32) -> Option<Vec<f64>> {
        let ws: Vec<f32> = inp.w.iter().map(|&x| x / s).collect();
        let sigma_s: f64 = ws
            .iter()
            .zip(&inp.p)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        if sigma_s == 0.0 || !sigma_s.is_finite() {
            return None;
        }
        let sa = sigma_s.abs();
        let sp_val: f64 = inp
            .up
            .iter()
            .zip(&ws)
            .map(|(&x, &w)| f64::from(x) * f64::from(w) / sigma_s)
            .sum();
        Some(
            (0..inp.w.len())
                .map(|i| {
                    (f64::from(inp.up[i]).abs() + sp_val.abs() * f64::from(inp.p[i]).abs())
                        / (sa * f64::from(s))
                })
                .collect(),
        )
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum SnVerdict {
        Accepted,
        Budget,
        Stage,
        SnJudge,
        Shadow,
        Other,
    }

    fn sn_verdict(msg: &str) -> SnVerdict {
        if msg.contains("走査量") {
            SnVerdict::Budget
        } else if msg.contains("構造段の上界判定") {
            SnVerdict::Stage
        } else if msg.contains("SpectralNorm の二階: ") {
            SnVerdict::SnJudge
        } else if msg.contains("SpectralNorm: f32 構造の丸め・相殺・underflow") {
            SnVerdict::Shadow
        } else {
            SnVerdict::Other
        }
    }

    struct SnReport {
        verdict: SnVerdict,
        cost: usize,
        /// 本番が選ぶ方式（[`sn_use_lin`]）の `sn_judge` の余裕。
        sn: Option<(Worst, bool)>,
        /// 従来の `Rf` 版の `sn_judge` の余裕（`Lin` に置き換えた効果の比較用）。
        sn_rf: Option<(Worst, bool)>,
        sh: Option<(Worst, bool)>,
        host_err: f64,
        str_err: f64,
        str_req2: f64,
        flag: bool,
        /// 従来の見積り（`rounding_coeff(N+4)·m_abs`）が全要素を通すか（#4081 の前後比較用）。
        old_ok: Option<bool>,
        /// 本番が新しい見積りを使えたか（`None` なら従来式へ戻った）。
        used_new: bool,
    }

    /// 1 入力の計測。公開経路の判定・`sn_judge`／`shadow_check` の余裕・f64 参照との実誤差を求める。
    fn sn_measure(c: &SnCase) -> SnReport {
        let inp = sn_inputs(c);
        let n = inp.w.len();
        // 公開経路。
        let tape = Tape::new_with_ops(test_ops());
        let child = Tape::new_with_ops(test_ops());
        let mut st = sn_state(c);
        let w = tape.var(&tt(inp.w.clone(), &c.shape));
        let wc = tape.var(&tt(inp.up.clone(), &c.shape));
        let y = w.spectral_norm(&mut st, false).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let verdict = match tape.backward_create_graph(&loss, &child) {
            Ok(_) => SnVerdict::Accepted,
            Err(e) => {
                assert!(child.is_empty(), "拒否なのに子テープが空でない");
                sn_verdict(&e.to_string())
            }
        };
        let cost = sn_judge_cost(n);
        let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = pow2(pow2_ceil_exponent(mx));
        let m_abs = sn_m_abs(&inp, s);
        // sn_judge の余裕（走査量の上限内で σ_s が有限・非零のときだけ）。
        let judged = cost <= JUDGE_BUDGET && m_abs.is_some();
        let margins = |lin: bool| {
            if judged {
                sn_judge_margins(
                    &inp.w,
                    &inp.p_exact,
                    &inp.up,
                    f64::from(s),
                    sn_coeff(n),
                    lin,
                )
            } else {
                None
            }
        };
        let sn = margins(sn_use_lin(n));
        let sn_rf = margins(false);
        if let Some((_, ok)) = sn {
            match verdict {
                SnVerdict::SnJudge => assert!(!ok, "本番は sn_judge 拒否だが再計算は全要素通過"),
                SnVerdict::Accepted | SnVerdict::Shadow => {
                    assert!(ok, "本番は sn_judge を通過したが再計算は拒否")
                }
                _ => {}
            }
        }
        // shadow_check の余裕と f64 参照との実誤差。
        let layout = wr::spectral_norm_layout(&c.shape, c.dim).unwrap();
        let host = wr::spectral_norm_vjp_host(&inp.w, &inp.up, &layout, &inp.u, &inp.v).unwrap();
        let (mut sh, mut host_err, mut str_err, mut str_req2, mut flag) =
            (None, f64::NAN, f64::NAN, f64::NAN, false);
        let (mut old_ok, mut used_new) = (None, false);
        if let Some(m_abs) = m_abs {
            let ch = Tape::new_with_ops(test_ops());
            let w_m = ch.var(&tt(inp.w.clone(), &c.shape));
            let up_v = ch.var(&tt(inp.up.clone(), &c.shape));
            let composed = sn_build(&ch, &w_m, &up_v, &inp.p, s, &c.shape);
            let comp = dense_vec(&child_value(&ch, &composed).unwrap());
            // 本番と同じ見積り（出せれば `Lin` の上界 + ホスト側の丸めの上界、出せなければ従来式）。
            let sigma_h = wr::spectral_sigma_host(&inp.w, &layout, &inp.u, &inp.v).unwrap();
            let bound =
                sn_shadow_bound(&inp.w, &inp.p_exact, &inp.up, f64::from(s), sigma_h, &host);
            let (est_vec, coeff) = match &bound {
                Some(b) => (b.clone(), 1.0),
                None => (m_abs.clone(), rounding_coeff(n + 4)),
            };
            let res = shadow_check(&ch, &composed, &host, &est_vec, coeff, "SpectralNorm");
            let mut worst = Worst::default();
            let mut all_ok = true;
            for i in 0..n {
                let (v, vs) = (f64::from(host[i]), f64::from(comp[i]));
                if v.is_nan() {
                    continue;
                }
                let err = (vs - v).abs() + coeff * est_vec[i];
                all_ok &=
                    err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * v.abs();
                worst.take(v, vs, est_vec[i], coeff);
            }
            assert_eq!(res.is_ok(), all_ok, "shadow_check の再計算が本番と不一致");
            used_new = bound.is_some();
            let oc = rounding_coeff(n + 4);
            old_ok = Some((0..n).all(|i| {
                let v = f64::from(host[i]);
                let err = (f64::from(comp[i]) - v).abs() + oc * m_abs[i];
                v.is_nan()
                    || err < ABSOLUTE_RESCUE_THRESHOLD
                    || err <= (RELATIVE_TOLERANCE / 2.0) * v.abs()
            }));
            match verdict {
                SnVerdict::Accepted => assert!(res.is_ok()),
                SnVerdict::Shadow => assert!(res.is_err()),
                _ => {}
            }
            sh = Some((worst, all_ok));
            let t64 = sn_dw64(&inp.w, &inp.p_exact, &inp.up);
            host_err = (0..n)
                .map(|i| (f64::from(host[i]) - t64[i]).abs())
                .fold(0.0, f64::max);
            str_err = (0..n)
                .map(|i| (f64::from(comp[i]) - t64[i]).abs())
                .fold(0.0, f64::max);
            str_req2 = (0..n)
                .map(|i| req2_ratio(f64::from(comp[i]), t64[i]))
                .fold(0.0, f64::max);
            // 構造の 1 階が f64 参照から REQ-2 を外れるのに shadow_check が通す（要注意）。
            flag = all_ok && str_req2 >= 1.0;
        }
        SnReport {
            verdict,
            cost,
            sn,
            sn_rf,
            sh,
            host_err,
            str_err,
            str_req2,
            flag,
            old_ok,
            used_new,
        }
    }

    fn sn_case(shape: &[usize], amp: f32, alpha: f32, up: SnUp, scale: f32) -> SnCase {
        SnCase {
            shape: shape.to_vec(),
            dim: 0,
            amp,
            alpha,
            up,
            scale,
        }
    }

    fn sn_measure_cases() -> Vec<SnCase> {
        let mut out = Vec::new();
        for sh in [[2usize, 3], [4, 4], [8, 8], [16, 16]] {
            for amp in [1e-3f32, 1.0, 1e3] {
                for alpha in [0.0f32, 2.0] {
                    for up in [SnUp::General, SnUp::ParallelP] {
                        for scale in [1.0f32, 1e3] {
                            out.push(sn_case(&sh, amp, alpha, up, scale));
                        }
                    }
                }
            }
        }
        for sh in [[32usize, 32], [64, 64], [64, 96]] {
            for alpha in [0.0f32, 2.0] {
                for up in [SnUp::General, SnUp::ParallelP] {
                    out.push(sn_case(&sh, 1.0, alpha, up, 1.0));
                }
            }
        }
        for alpha in [0.0f32, 2.0] {
            for scale in [1.0f32, 1e3] {
                out.push(sn_case(&[4, 4, 4], 1.0, alpha, SnUp::General, scale));
            }
        }
        out
    }

    /// 計測（`#[ignore]`）: 新しい見積りの内訳（`|composed − host|`・`B_str`・`B_host`）を、余裕が最悪の要素で出す。
    /// `cargo test -p fandhe-ai-autodiff --release --lib spectral_norm_shadow_bound_breakdown_measurement -- --ignored --nocapture`
    #[test]
    #[ignore = "計測。--release で実行する"]
    fn spectral_norm_shadow_bound_breakdown_measurement() {
        for (c, _) in sn_followup_candidates() {
            let inp = sn_inputs(&c);
            let n = inp.w.len();
            let layout = wr::spectral_norm_layout(&c.shape, c.dim).unwrap();
            let host =
                wr::spectral_norm_vjp_host(&inp.w, &inp.up, &layout, &inp.u, &inp.v).unwrap();
            let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            let sigma_h = wr::spectral_sigma_host(&inp.w, &layout, &inp.u, &inp.v).unwrap();
            let st = sn_str_bound(&inp.w, &inp.p_exact, &inp.up, s);
            let ho = sn_host_bound(&inp.w, &inp.p_exact, &inp.up, sigma_h, &host);
            let ch = Tape::new_with_ops(test_ops());
            let w_m = ch.var(&tt(inp.w.clone(), &c.shape));
            let up_v = ch.var(&tt(inp.up.clone(), &c.shape));
            let composed = sn_build(
                &ch,
                &w_m,
                &up_v,
                &inp.p,
                pow2(pow2_ceil_exponent(mx)),
                &c.shape,
            );
            let comp = dense_vec(&child_value(&ch, &composed).unwrap());
            let (mut worst, mut at) = (0.0f64, 0usize);
            for i in 0..n {
                let v = f64::from(host[i]);
                let thr = ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * v.abs());
                let e = (f64::from(comp[i]) - v).abs()
                    + st.as_ref().map_or(f64::NAN, |b| b[i])
                    + ho.as_ref().map_or(f64::NAN, |b| b[i]);
                if e / thr > worst {
                    (worst, at) = (e / thr, i);
                }
            }
            eprintln!(
                "SNBRK shape={:?} amp={} alpha={} n={n} worst_ratio={worst:.3e} at={at} |c-h|={:.2e} B_str={:?} B_host={:?} |host|={:.2e}",
                c.shape,
                c.amp,
                c.alpha,
                (f64::from(comp[at]) - f64::from(host[at])).abs(),
                st.as_ref().map(|b| b[at]),
                ho.as_ref().map(|b| b[at]),
                f64::from(host[at]).abs()
            );
        }
    }

    /// 計測（`#[ignore]`）。出力は §81.2 の表の素データ。
    /// `cargo test -p fandhe-ai-autodiff --release --lib spectral_norm_rejection_breakdown_measurement -- --ignored --nocapture`
    #[test]
    #[ignore = "計測（O(N^2) の再計算を含む）。--release で実行する"]
    fn spectral_norm_rejection_breakdown_measurement() {
        eprintln!(
            "SNROW shape|amp|alpha|up|scale|verdict|cost/budget|sn_ratio(diff,est,|V|,kappa)|rf_ratio|sh_ratio(diff,est,|V|)|host_err|str_err|str_req2|flag"
        );
        for c in sn_measure_cases() {
            let r = sn_measure(&c);
            let sn = r.sn.map_or("-".to_string(), |(w, _)| {
                format!(
                    "{:.3e}({:.1e},{:.1e},{:.1e},{:.1e})",
                    w.ratio,
                    w.diff,
                    w.est,
                    w.v.abs(),
                    w.m / w.v.abs().max(f64::MIN_POSITIVE)
                )
            });
            let rf = r
                .sn_rf
                .map_or("-".to_string(), |(w, _)| format!("{:.3e}", w.ratio));
            let sh = r.sh.map_or("-".to_string(), |(w, _)| {
                format!(
                    "{:.3e}({:.1e},{:.1e},{:.1e})",
                    w.ratio,
                    w.diff,
                    w.est,
                    w.v.abs()
                )
            });
            eprintln!(
                "SNROW {:?}|{:e}|{}|{:?}|{:e}|{:?}|{:.3}|{}|{}|{}|{:.2e}|{:.2e}|{:.2e}|{}|old_ok={:?}|new_est={}",
                c.shape,
                c.amp,
                c.alpha,
                c.up,
                c.scale,
                r.verdict,
                r.cost as f64 / JUDGE_BUDGET as f64,
                sn,
                rf,
                sh,
                r.host_err,
                r.str_err,
                r.str_req2,
                if r.flag { "要注意" } else { "" },
                r.old_ok,
                r.used_new
            );
        }
    }

    /// `sn_judge` の直接テスト用に、`spectral_norm_cgrad` と同じ式で `(W, P の f64, s)` を作る。
    fn sn_probe(shape: &[usize], u: &[f32], v: &[f32], w: Vec<f32>) -> (Vec<f32>, Vec<f64>, f64) {
        let (_, p_exact) = sn_p(shape, 0, u, v);
        let mx = w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = f64::from(pow2(pow2_ceil_exponent(mx)));
        (w, p_exact, s)
    }

    /// `Lin` 版の `sn_judge` が、§81.5 の「`sn_judge` だけが拒否していた」2 入力を受理し、`Rf` 版は拒否のままである
    /// こと（#4080 の受入基準）。
    #[test]
    fn sn_judge_lin_accepts_what_rf_rejected() {
        for c in [
            sn_case(&[4, 4], 1.0, 2.0, SnUp::General, 1.0),
            sn_case(&[16, 16], 1e3, 0.0, SnUp::General, 1.0),
        ] {
            let inp = sn_inputs(&c);
            let n = inp.w.len();
            let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            assert!(sn_use_lin(n), "{c:?}: Lin を選ぶ条件内");
            assert!(
                sn_judge(&inp.w, &inp.p_exact, &inp.up, s, sn_coeff(n), true).is_ok(),
                "{c:?}: Lin 版は受理"
            );
            assert!(
                sn_judge(&inp.w, &inp.p_exact, &inp.up, s, sn_coeff(n), false).is_err(),
                "{c:?}: Rf 版は拒否のまま"
            );
        }
    }

    /// 既存の拒否プローブ 3 件（`tests/create_graph_norm_reparam.rs`）の入力を `sn_judge` 単体へ再構成し、`Lin` 版でも
    /// 拒否のままであること。`σ_s` の相殺（[2,2]）と係数行列の相殺（[1,2]）は `|V_str − V|` と一次の上界の項で、Div 係数の
    /// overflow（[1,2]）は `sn_fwd` の f32 の ±inf の検査で拒否される。
    #[test]
    fn sn_judge_lin_still_rejects_the_three_probes() {
        let w2 = (-33f32).exp2();
        let probes = [
            (
                vec![2usize, 2],
                vec![0.6f32, 0.8],
                vec![0.6f32, 0.8],
                vec![1.0f32, 1.0, 1.0, -2.06249],
            ),
            (
                vec![1, 2],
                vec![1.0],
                vec![4e-38, 1.0],
                vec![0.75 * 2f32.powi(127), 0.0],
            ),
            (vec![1, 2], vec![1.0], vec![1.0, 1e-8], vec![w2, w2]),
        ];
        for (shape, u, v, w) in probes {
            let n = w.len();
            let (w, p, s) = sn_probe(&shape, &u, &v, w);
            let up = vec![0.0f32; n];
            assert!(sn_use_lin(n));
            let r = sn_judge(&w, &p, &up, s, sn_coeff(n), true);
            assert!(r.is_err(), "{shape:?}: Lin 版でも拒否のまま");
        }
    }

    /// 事象の総数が [`sn_lin_events_max`] 以下であること（[`sn_use_lin`] の余裕係数の条件の根拠）。値の違う入力
    /// （定数の丸めの事象が出る・出ない）でも成り立つ。
    #[test]
    fn sn_lin_events_stay_within_bound() {
        for (side, amp) in [(2usize, 1.0f32), (4, 1.0), (8, 1e3), (16, 1.0), (16, 1e-3)] {
            let c = sn_case(&[side, side], amp, 2.0, SnUp::General, 1.0);
            let inp = sn_inputs(&c);
            let n = inp.w.len();
            let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx.max(1e-3))));
            lin_reset();
            let seen = std::cell::Cell::new(0usize);
            let _ = sn_judge_with::<Lin>(&inp.w, &inp.p_exact, &inp.up, s, &|_, _, _| {
                seen.set(seen.get().max(lin_events()));
                Ok(())
            });
            lin_reset();
            assert!(
                seen.get() <= sn_lin_events_max(n),
                "n={n}: 事象 {} が上界 {} を超えた",
                seen.get(),
                sn_lin_events_max(n)
            );
            assert!(seen.get() > 0);
        }
        // 走査量の上限と余裕係数の上限は、どちらも [16, 16] を含み、[32, 32] 以上を除く。
        assert!(sn_use_lin(256));
        assert!(!sn_use_lin(1024));
    }

    /// WeightNorm の疎モードの事象数（forward + seed 1 回分の逆伝播の最大）を数える。`Dg`・先頭と末尾の `Dv` の 3 seed。
    fn wn_events_of(v: &[f32], up: &[f32], g: f32, s: f64) -> usize {
        lin_reset_sparse();
        let Some(fl) = wn_fwd::<Lin>(v, up, g, s) else {
            return 0;
        };
        let r = v.len();
        let mark = lin_mark();
        let mut n = 0;
        for seed in [WnSeed::Dg, WnSeed::Dv(0), WnSeed::Dv(r - 1)] {
            lin_truncate(mark);
            let (mut hv, mut hu) = (vec![Lin::k(0.0); r], vec![Lin::k(0.0); r]);
            let _ = wn_rev(&fl, seed, &mut hv, &mut hu);
            n = n.max(lin_events());
        }
        lin_reset();
        n
    }

    /// 疎モードの事象数が [`wn_lin_events_max`] 以下であること（[`wn_use_lin`] の余裕係数の条件の根拠。#4082）。
    /// 幅 2〜[`LIN_REST_MAX`]・振幅 1e-3／1／1e3・`g` を振る。`wn_lin_events_max` を小さくすると失敗する。
    #[test]
    fn wn_lin_events_stay_within_bound() {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((st >> 33) as f64) / f64::from(1u32 << 31)
        };
        for rest in [2usize, 3, 8, 9, 16, 33, 64, LIN_REST_MAX] {
            for amp in [1e-3f64, 1.0, 1e3] {
                for _ in 0..2 {
                    let v: Vec<f32> = (0..rest)
                        .map(|_| ((next() * 2.0 - 1.0) * 2.0 * amp) as f32)
                        .collect();
                    let up: Vec<f32> = (0..rest)
                        .map(|_| ((next() * 2.0 - 1.0) * 3.0) as f32)
                        .collect();
                    let g = (0.05 + next() * 2.0) as f32;
                    let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
                    if mx == 0.0 {
                        continue;
                    }
                    let s = f64::from(pow2(pow2_ceil_exponent(mx)));
                    let n = wn_events_of(&v, &up, g, s);
                    assert!(n > 0);
                    assert!(
                        n <= wn_lin_events_max(rest),
                        "rest={rest} amp={amp}: 事象 {n} が上界 {} を超えた",
                        wn_lin_events_max(rest)
                    );
                }
            }
        }
    }

    /// 計測（`#[ignore]`・#4082）: 幅 9〜[`LIN_REST_MAX`] の決定的な入力で、グループ判定を従来の `Rf`（`use_lin = false`）と
    /// `Lin`（疎モード）の両方で行い、受理→拒否・拒否→受理の件数を幅ごとに `WIDEFLIP` 行で出す。判定は変えない。
    /// `cargo test -p fandhe-ai-autodiff --release --lib wn_wide_rejection_breakdown_measurement -- --ignored --nocapture`
    #[test]
    #[ignore = "計測。--release で実行する（決定記録 §80.5）"]
    fn wn_wide_rejection_breakdown_measurement() {
        let mut st = 0x1234_5678_9ABC_DEF1u64;
        let mut next = move || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((st >> 33) as f64) / f64::from(1u32 << 31)
        };
        for rest in [9usize, 12, 16, 24, 32, 48, 64, 96, 128] {
            let (mut both_ok, mut both_err, mut rf_only, mut lin_only) = (0, 0, 0, 0);
            for case in 0..60 {
                let amp = [1e-3f64, 1.0, 1e3][case % 3];
                // 0..3: 乱数、3..: `[0.2, 1.0)` の正の入力（相殺しやすい普通の入力）。
                let v: Vec<f32> = (0..rest)
                    .map(|_| {
                        let r = next();
                        (if case % 6 < 3 {
                            (r * 2.0 - 1.0) * 2.0
                        } else {
                            0.2 + 0.8 * r
                        } * amp) as f32
                    })
                    .collect();
                let up: Vec<f32> = (0..rest)
                    .map(|_| ((next() * 2.0 - 1.0) * 3.0) as f32)
                    .collect();
                let g = (0.3 + next() * 1.5) as f32;
                let mx = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
                if mx == 0.0 {
                    continue;
                }
                let s = f64::from(pow2(pow2_ceil_exponent(mx)));
                let rf = wn_group_judge(&v, &up, g, s, wn_coeff(rest), false).is_ok();
                let lin = wn_group_judge(&v, &up, g, s, wn_coeff(rest), true).is_ok();
                match (rf, lin) {
                    (true, true) => both_ok += 1,
                    (false, false) => both_err += 1,
                    (true, false) => rf_only += 1,
                    (false, true) => lin_only += 1,
                }
            }
            lin_reset();
            eprintln!(
                "WIDEFLIP rest={rest} both_accept={both_ok} both_reject={both_err} \
                 rf_accept_lin_reject={rf_only} rf_reject_lin_accept={lin_only}"
            );
        }
    }

    /// [`wn_use_lin`] の境界: 幅 [`LIN_REST_MAX`] とその +1、走査量 [`LIN_BUDGET`] の境界のグループ数、余裕係数の条件。
    #[test]
    fn wn_use_lin_boundaries() {
        assert!(wn_use_lin(1, 2));
        assert!(wn_use_lin(1, LIN_REST_MAX));
        assert!(!wn_use_lin(1, LIN_REST_MAX + 1));
        let per = lin_group_cost(16);
        assert!(wn_use_lin(LIN_BUDGET / per, 16));
        assert!(!wn_use_lin(LIN_BUDGET / per + 1, 16));
        // 余裕係数の条件は幅の上限の範囲内で満たされる（満たされなければ幅の上限が広すぎる）。
        assert!(16.0 * wn_lin_events_max(LIN_REST_MAX) as f64 * LIN_U < 0.01);
    }

    /// 計測（`#[ignore]`）: `sn_judge` の `Lin` 版と `Rf` 版の所要時間・事象数・§81.5 の入力の余裕。
    /// `cargo test -p fandhe-ai-autodiff --release --lib spectral_norm_lin_judge_cost_measurement -- --ignored --nocapture`
    #[test]
    #[ignore = "計測。--release で実行する"]
    fn spectral_norm_lin_judge_cost_measurement() {
        use std::time::Instant;
        for side in [2usize, 4, 8, 12, 16, 20, 22] {
            let c = sn_case(&[side, side], 1.0, 2.0, SnUp::General, 1.0);
            let inp = sn_inputs(&c);
            let n = inp.w.len();
            let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            let median = |f: &dyn Fn() -> bool| {
                let mut ts: Vec<f64> = (0..5)
                    .map(|_| {
                        let t = Instant::now();
                        let _ = f();
                        t.elapsed().as_secs_f64()
                    })
                    .collect();
                ts.sort_by(|a, b| a.total_cmp(b));
                ts[2]
            };
            // 全要素を走査させるため、検査は常に通す（早期拒否で時間が短くならないように）。
            let t_rf = median(&|| {
                sn_judge_with::<Rf>(&inp.w, &inp.p_exact, &inp.up, s, &|_, _, _| Ok(())).is_ok()
            });
            let ev = std::cell::Cell::new(0usize);
            let t_lin = median(&|| {
                lin_reset();
                let ok = sn_judge_with::<Lin>(&inp.w, &inp.p_exact, &inp.up, s, &|_, _, _| Ok(()))
                    .is_ok();
                ev.set(lin_events());
                lin_reset();
                ok
            });
            let cost = sn_judge_cost(n) as f64;
            eprintln!(
                "SNLIN events={} n={n} t_rf={t_rf:.4e}s t_lin={t_lin:.4e}s ns_per_rf_iter={:.2} lin_cost_equiv={:.3e} (cost={cost:.3e})",
                ev.get(),
                t_rf / cost * 1e9,
                t_lin / (t_rf / cost)
            );
        }
    }

    /// #4079（§81.5）が選んだ入力。各要素は（入力, 現在の判定の種別）。`sn_judge` を `Lin` に置き換えた
    /// #4080 で `sn_judge` だけが拒否していた 2 件が受理に変わり、`shadow_check` の見積りを `Lin` の上界 +
    /// ホスト側の丸めの上界に置き換えた #4081 で、`[8, 8]`（amp 1e3・α 0）が受理に変わった。`[16, 16]`
    /// （amp 1e3・α 2）は `B_str`（`Lin` の上界）が閾値を超えるため拒否のまま。最後の `[16, 16]`
    /// （amp 1e3・α 0・上流 1）は従来の見積りでは受理だったが、`B_str` が `σ_s` の相殺を含めて追うため
    /// #4081 で拒否に変わった（§81.9.3）。
    fn sn_followup_candidates() -> Vec<(SnCase, SnVerdict)> {
        vec![
            (
                sn_case(&[8, 8], 1e3, 0.0, SnUp::General, 1e3),
                SnVerdict::Accepted,
            ),
            (
                sn_case(&[16, 16], 1e3, 2.0, SnUp::General, 1e3),
                SnVerdict::Shadow,
            ),
            (
                sn_case(&[4, 4], 1.0, 2.0, SnUp::General, 1.0),
                SnVerdict::Accepted,
            ),
            (
                sn_case(&[16, 16], 1e3, 0.0, SnUp::General, 1.0),
                SnVerdict::Shadow,
            ),
        ]
    }

    /// 手組みの構造の 1 階・二階（行ごと）と、f64 の閉じた式との REQ-2 比の最大値。
    fn sn_structure_vs_f64(c: &SnCase) -> (f64, f64) {
        let inp = sn_inputs(c);
        let n = inp.w.len();
        let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = pow2(pow2_ceil_exponent(mx));
        let ch = Tape::new_with_ops(test_ops());
        let w_m = ch.var(&tt(inp.w.clone(), &c.shape));
        let up_v = ch.var(&tt(inp.up.clone(), &c.shape));
        let composed = sn_build(&ch, &w_m, &up_v, &inp.p, s, &c.shape);
        let comp = dense_vec(&child_value(&ch, &composed).unwrap());
        let t64 = sn_dw64(&inp.w, &inp.p_exact, &inp.up);
        let first = (0..n)
            .map(|i| req2_ratio(f64::from(comp[i]), t64[i]))
            .fold(0.0, f64::max);
        let mut second = 0.0f64;
        for j in 0..n {
            let (hw64, hu64) = sn_hess64(&inp.w, &inp.p_exact, &inp.up, j);
            let rw = child_row(&ch, &composed, j, &w_m);
            let ru = child_row(&ch, &composed, j, &up_v);
            for i in 0..n {
                second = second.max(req2_ratio(f64::from(rw[i]), hw64[i]));
                second = second.max(req2_ratio(f64::from(ru[i]), hu64[i]));
            }
        }
        (first, second)
    }

    /// 後続の入力が想定どおりの判定になり、実際の f32 の 1 階・二階は独立な f64 の閉じた式から REQ-2 の閾値の
    /// 1/4 以内に収まること（受理に変えても REQ-2 の範囲内）。`sn_judge` を `Lin` にして受理に変わった 2 件
    /// は、従来の `Rf` 版の余裕が 1 以上（拒否）だった入力でもある。tolerance は変えない。`--nocapture` で実測値を出す。
    #[test]
    fn spectral_norm_followup_candidates_are_req2_within_f64() {
        for (c, want) in sn_followup_candidates() {
            let r = sn_measure(&c);
            assert_eq!(r.verdict, want, "{c:?}: 判定の種別");
            let (sn, sh) = (r.sn.unwrap().0.ratio, r.sh.unwrap().0.ratio);
            let rf = r.sn_rf.unwrap().0.ratio;
            match want {
                SnVerdict::Shadow => assert!(sn < 1.0 && sh >= 1.0, "{c:?}: sn={sn} sh={sh}"),
                _ => assert!(sn < 1.0 && sh < 1.0, "{c:?}: sn={sn} sh={sh}"),
            }
            let (first, second) = sn_structure_vs_f64(&c);
            eprintln!(
                "SNCAND {c:?} lin={sn:.3e} rf={rf:.3e} sh={sh:.3e} req2_first={first:.3e} req2_second={second:.3e}"
            );
            assert!(first < 0.25, "{c:?}: 1 階の REQ-2 比 {first}");
            assert!(second < 0.25, "{c:?}: 二階の REQ-2 比 {second}");
        }
    }

    /// `Lin` で受理に変わった入力（`Rf` 版の `sn_judge` は拒否していたもの）の公開経路の二階が、f64 の閉じた式と
    /// REQ-2 の範囲で一致すること（拒否範囲を狭めても REQ-2 を外さない裏づけ）。
    #[test]
    fn spectral_norm_newly_accepted_by_lin_match_f64_second_order() {
        let cases = [
            sn_case(&[4, 4], 1e-3, 0.0, SnUp::General, 1.0),
            sn_case(&[4, 4], 1.0, 0.0, SnUp::General, 1e3),
            sn_case(&[8, 8], 1.0, 0.0, SnUp::ParallelP, 1.0),
            sn_case(&[8, 8], 1e3, 2.0, SnUp::General, 1.0),
            sn_case(&[16, 16], 1.0, 2.0, SnUp::ParallelP, 1.0),
            sn_case(&[16, 16], 1e3, 0.0, SnUp::ParallelP, 1.0),
        ];
        for c in cases {
            let r = sn_measure(&c);
            let rf = r.sn_rf.unwrap().0.ratio;
            assert_eq!(r.verdict, SnVerdict::Accepted, "{c:?}");
            assert!(rf >= 1.0, "{c:?}: Rf 版は拒否していたはず（{rf}）");
            let (first, second) = sn_structure_vs_f64(&c);
            assert!(first < 1.0, "{c:?}: 1 階 {first}");
            assert!(second < 1.0, "{c:?}: 二階 {second}");
        }
    }

    /// #4081 で `shadow_check` の見積りを置き換えて受理に変わった入力（従来の見積りは拒否していたもの）の
    /// 公開経路の 1 階・二階が、f64 の閉じた式と REQ-2 の範囲で一致すること。
    #[test]
    fn spectral_norm_newly_accepted_by_shadow_bound_match_f64() {
        let cases = [
            sn_case(&[8, 8], 1e3, 0.0, SnUp::General, 1e3),
            sn_case(&[8, 8], 1.0, 0.0, SnUp::General, 1.0),
        ];
        for c in cases {
            let r = sn_measure(&c);
            assert_eq!(r.verdict, SnVerdict::Accepted, "{c:?}");
            assert_eq!(
                r.old_ok,
                Some(false),
                "{c:?}: 従来の見積りは拒否していたはず"
            );
            assert!(r.used_new, "{c:?}: 新しい見積りが使われたはず");
            let (first, second) = sn_structure_vs_f64(&c);
            assert!(first < 1.0, "{c:?}: 1 階 {first}");
            assert!(second < 1.0, "{c:?}: 二階 {second}");
        }
    }

    /// 健全性（2 成分を別々に）。乱数掃引で (a) 構造側: f32 の再現と、和の順序を変えた再現（要素を逆順・
    /// 回転に並べ替えて回す）の 1 階が f64 の閉じた式から `B_str` 以内、(b) ホスト側: `spectral_norm_vjp_host`
    /// の値が f64 の閉じた式から `B_host` 以内。観測差（`|composed − host|`）を足した合成ではなく成分ごとに
    /// 見るので、どちらかの上界を 0 にすると落ちる。真値の f64 自体の誤差（約 `N·2^-53`）は小さい許容で足す。
    #[test]
    fn spectral_norm_shadow_bound_components_cover_deviations() {
        let (mut tot_bound, mut tot_err) = (0.0f64, 0.0f64);
        let (mut tot_hbound, mut tot_herr) = (0.0f64, 0.0f64);
        for shape in [[2usize, 3], [4, 4], [8, 8], [16, 16]] {
            for amp in [1e-3f32, 1.0, 1e3] {
                for alpha in [0.0f32, 2.0] {
                    for scale in [1.0f32, 1e3] {
                        let c = sn_case(&shape, amp, alpha, SnUp::General, scale);
                        let inp = sn_inputs(&c);
                        let n = inp.w.len();
                        let layout = wr::spectral_norm_layout(&c.shape, c.dim).unwrap();
                        let host =
                            wr::spectral_norm_vjp_host(&inp.w, &inp.up, &layout, &inp.u, &inp.v)
                                .unwrap();
                        let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
                        let s = f64::from(pow2(pow2_ceil_exponent(mx)));
                        let sigma_h =
                            wr::spectral_sigma_host(&inp.w, &layout, &inp.u, &inp.v).unwrap();
                        let t64 = sn_dw64(&inp.w, &inp.p_exact, &inp.up);
                        let st = sn_str_bound(&inp.w, &inp.p_exact, &inp.up, s);
                        let ho = sn_host_bound(&inp.w, &inp.p_exact, &inp.up, sigma_h, &host);
                        let (Some(st), Some(ho)) = (st, ho) else {
                            continue;
                        };
                        // 真値の f64 自体の誤差の許容（理由: 閉じた式の f64 評価は項の大きさに比例して `~N·2^-53`）。
                        let mag = |i: usize| {
                            (f64::from(inp.up[i]).abs() + f64::from(host[i]).abs())
                                * 8.0
                                * (n as f64 + 4.0)
                                * 2f64.powi(-53)
                        };
                        // 構造側: 元の順序・逆順・回転。
                        let orders: Vec<Vec<usize>> = vec![
                            (0..n).collect(),
                            (0..n).rev().collect(),
                            (0..n).map(|i| (i + n / 3) % n).collect(),
                        ];
                        for ord in &orders {
                            let w: Vec<f32> = ord.iter().map(|&i| inp.w[i]).collect();
                            let p: Vec<f64> = ord.iter().map(|&i| inp.p_exact[i]).collect();
                            let up: Vec<f32> = ord.iter().map(|&i| inp.up[i]).collect();
                            let f = sn_fwd::<f32>(&w, &p, &up, s).unwrap();
                            for (k, &i) in ord.iter().enumerate() {
                                let err = (f64::from(f.dw[k]) - t64[i]).abs();
                                assert!(
                                    err <= st[i] + mag(i),
                                    "{c:?} i={i}: 構造側の実誤差 {err:e} が B_str {:e} を超えた",
                                    st[i]
                                );
                                tot_err += err;
                                tot_bound += st[i];
                            }
                        }
                        for i in 0..n {
                            let err = (f64::from(host[i]) - t64[i]).abs();
                            assert!(
                                err <= ho[i] + mag(i),
                                "{c:?} i={i}: ホスト側の実誤差 {err:e} が B_host {:e} を超えた",
                                ho[i]
                            );
                            tot_herr += err;
                            tot_hbound += ho[i];
                        }
                    }
                }
            }
        }
        // 上界が桁違いに緩くないこと（0 に潰す変異や、無意味に大きい上界の両方を検出する）。
        assert!(
            tot_err > 0.0 && tot_bound / tot_err < 1e4,
            "{tot_bound:e} / {tot_err:e}"
        );
        assert!(
            tot_herr > 0.0 && tot_hbound / tot_herr < 1e4,
            "{tot_hbound:e} / {tot_herr:e}"
        );
    }

    /// 出せない条件では `None`（呼び出し側が従来の見積りへ戻す。`None` の入力では拒否範囲が広がらない）。
    /// 新見積り（B_str+B_host）を出せる入力では、従来式で受理だった入力が拒否に変わりうる（意図的。
    /// `min` を採らず置換にしたのは、従来式の見積りが Lin の上界に包含されない別系統の近似で、健全性を
    /// 新見積り 1 本の解析に揃えるため。`sn_followup_candidates` のコメントと docs §81.9.3 を参照）。
    #[test]
    fn spectral_norm_shadow_bound_falls_back_to_none() {
        // N が大きく `16·E·u ≥ 0.01` になる形は、領域を作る前に諦める。
        let n = 4096;
        let w = vec![1.0f32; n];
        let p = vec![1.0f64; n];
        let up = vec![1.0f32; n];
        assert!(sn_str_bound(&w, &p, &up, 1.0).is_none());
        assert!(sn_shadow_bound(&w, &p, &up, 1.0, f64::from(n as u32), &vec![0.0; n]).is_none());
        // `σ_h` が `A` に対して相殺しすぎ（ε ≥ 0.01）なら、ホスト側の上界を出せない。
        let (w, p, up) = ([1.0f32, -1.0], [1.0f64, 1.0], [1.0f32, 1.0]);
        assert!(sn_host_bound(&w, &p, &up, 1e-15, &[0.0, 0.0]).is_none());
        assert!(sn_host_bound(&w, &p, &up, 0.0, &[0.0, 0.0]).is_none());
        assert!(sn_host_bound(&w, &p, &up, 1.0, &[f32::NAN, 0.0]).is_none());
        assert!(sn_host_bound(&w, &p, &up, 1.0, &[0.0, 0.0]).is_some());
    }

    /// f64 の閉じた式（1 階・二階）が f64 の中央差分と一致すること（計測の真値の検算）。
    #[test]
    fn spectral_norm_closed_form_matches_f64_central_difference() {
        let c = sn_case(&[2, 3], 1.0, 1.0, SnUp::General, 1.0);
        let inp = sn_inputs(&c);
        let (w, up, p) = (&inp.w, &inp.up, &inp.p_exact);
        let n = w.len();
        let dw = |w: &[f64], up: &[f64], j: usize| -> f64 {
            let sigma: f64 = w.iter().zip(p).map(|(a, b)| a * b).sum();
            let a: f64 = up.iter().zip(w).map(|(x, y)| x * y).sum();
            (up[j] - a / sigma * p[j]) / sigma
        };
        let (w64, up64): (Vec<f64>, Vec<f64>) = (
            w.iter().map(|&x| f64::from(x)).collect(),
            up.iter().map(|&x| f64::from(x)).collect(),
        );
        let t64 = sn_dw64(w, p, up);
        let h = 1e-6;
        for (j, &tj) in t64.iter().enumerate() {
            assert!((dw(&w64, &up64, j) - tj).abs() < 1e-12);
            let (hw, hu) = sn_hess64(w, p, up, j);
            for i in 0..n {
                let (mut wp, mut wm) = (w64.clone(), w64.clone());
                wp[i] += h;
                wm[i] -= h;
                let fd_w = (dw(&wp, &up64, j) - dw(&wm, &up64, j)) / (2.0 * h);
                assert!(
                    (fd_w - hw[i]).abs() < 1e-6 * (1.0 + hw[i].abs()),
                    "W j={j} i={i}"
                );
                let (mut upp, mut upm) = (up64.clone(), up64.clone());
                upp[i] += h;
                upm[i] -= h;
                let fd_u = (dw(&w64, &upp, j) - dw(&w64, &upm, j)) / (2.0 * h);
                assert!(
                    (fd_u - hu[i]).abs() < 1e-6 * (1.0 + hu[i].abs()),
                    "up j={j} i={i}"
                );
            }
        }
    }

    /// 手組みの構造（計測の前提）が公開経路の二階と bit 一致すること（受理される入力で）。
    #[test]
    fn spectral_norm_hand_built_structure_matches_public_path() {
        let c = sn_case(&[2, 3], 1.0, 2.0, SnUp::General, 1.0);
        let inp = sn_inputs(&c);
        let n = inp.w.len();
        let tape = Tape::new_with_ops(test_ops());
        let child = Tape::new_with_ops(test_ops());
        let mut st = sn_state(&c);
        let w = tape.var(&tt(inp.w.clone(), &c.shape));
        let wc = tape.var(&tt(inp.up.clone(), &c.shape));
        let y = w.spectral_norm(&mut st, false).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gw = cg.grad(&w).unwrap().unwrap();
        let cw = cg.child_var(&w).unwrap().unwrap();
        let cwc = cg.child_var(&wc).unwrap().unwrap();
        let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = pow2(pow2_ceil_exponent(mx));
        let ch = Tape::new_with_ops(test_ops());
        let w_m = ch.var(&tt(inp.w.clone(), &c.shape));
        let up_v = ch.var(&tt(inp.up.clone(), &c.shape));
        let composed = sn_build(&ch, &w_m, &up_v, &inp.p, s, &c.shape);
        for j in 0..n {
            assert_eq!(
                child_row(&child, &gw, j, &cw),
                child_row(&ch, &composed, j, &w_m)
            );
            assert_eq!(
                child_row(&child, &gw, j, &cwc),
                child_row(&ch, &composed, j, &up_v)
            );
        }
    }
    // -----------------------------------------------------------------
    // 総括判定の走査時間の実測（イシュー #4085・決定記録 §83.1）。判定・定数は変えない
    // -----------------------------------------------------------------

    /// 決定的な LCG（[0, 1)）。計測の入力生成用（環境変数・ファイルは読まない）。
    struct TimingLcg(u64);
    impl TimingLcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f64) / f64::from(1u32 << 31)
        }
    }

    /// 条件 `ok(v)`（`v` が小さいほど真に単調）が成り立つ最大の `v >= lo`（倍々＋二分探索）。
    fn timing_max_by(lo: usize, ok: impl Fn(usize) -> bool) -> usize {
        let (mut lo, mut hi) = (lo, lo * 2);
        assert!(ok(lo));
        while ok(hi) {
            lo = hi;
            hi *= 2;
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// ウォームアップ 1 回の後の 5 回計測（1 回が 6 s 超なら 3 回）の中央値と各回の値（ns）。
    fn timing_median(mut f: impl FnMut()) -> (f64, Vec<f64>) {
        let mut once = || {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        };
        let first = once();
        let reps = if first > 6e9 { 3 } else { 5 };
        let mut xs: Vec<f64> = (0..reps).map(|_| once()).collect();
        let all = xs.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (xs[xs.len() / 2], all)
    }

    fn timing_runs_ms(all: &[f64]) -> Vec<f64> {
        all.iter().map(|v| (v / 1e5).round() / 10.0).collect()
    }

    /// 計測に使う異なる入力の最大数。グループ・列が多い形では、この数の入力を循環させて並べる
    /// （入力の生成と保持を抑えるため。判定は入力ごとに独立で、全件が走査される）。
    const TIMING_UNIQUE: usize = 256;

    /// WeightNorm の 1 グループの入力 `(v, up, g, s)`（`lin.rs::gen_input` と同じ分布。振幅 1・上流 3 程度）。
    fn timing_wn_input(rng: &mut TimingLcg, rest: usize) -> (Vec<f32>, Vec<f32>, f32, f64) {
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

    /// LRN の判定（`lrn_col_judge`）を、走査量 [`JUDGE_BUDGET`] ぎりぎりの入力で計測する。
    /// (a) 小さい窓（size 5・C 16）の列を上限まで並べる、(b) 1 列で C を増やし size = 2C + 1（二重走査
    /// `wr·wf` が効く形）。全列の判定を `lrn_col_judge` で回し `Ok`（全列を走査）を assert する。
    fn timing_lrn(label: &str, ncols_fixed: Option<usize>, c: usize, size: usize) {
        let win = lrn_win(c, size);
        let (alpha, beta, k) = (size as f64 * 1.0, 0.75f64, 1.0f64);
        let rho = alpha / size as f64;
        let floor = (k * size as f64 / alpha).sqrt();
        let mut rng = TimingLcg(0x4085 + (c * 7 + size) as u64);
        let n_unique = ncols_fixed.unwrap_or(TIMING_UNIQUE).min(TIMING_UNIQUE);
        let cols: Vec<(Vec<f32>, Vec<f32>)> = (0..n_unique)
            .map(|_| {
                let x: Vec<f32> = (0..c).map(|_| (0.3 + 0.7 * rng.next()) as f32).collect();
                let up: Vec<f32> = (0..c).map(|_| (rng.next() * 2.0 - 1.0) as f32).collect();
                (x, up)
            })
            .collect();
        let mx = cols
            .iter()
            .flat_map(|(x, _)| x.iter())
            .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
        let e = pow2_ceil_exponent(mx.max(floor));
        let chunks: Vec<f64> = chunk_factors(&[-2.0 * beta * f64::from(e)])
            .iter()
            .map(|ch| ch[0])
            .collect();
        let nchunks = chunks.len();
        let ncols = ncols_fixed.unwrap_or_else(|| {
            timing_max_by(1, |m| lrn_judge_cost(m, c, win, nchunks) <= JUDGE_BUDGET)
        });
        let cost = lrn_judge_cost(ncols, c, win, nchunks);
        assert!(cost <= JUDGE_BUDGET);
        if ncols_fixed.is_none() {
            assert!(lrn_judge_cost(ncols + 1, c, win, nchunks) > JUDGE_BUDGET);
        }
        let kc = LrnColConst {
            s: f64::from(pow2(e)),
            kk: k / 2f64.powi(2 * e),
            rho,
            beta,
            chunks: &chunks,
        };
        let coeff = lrn_coeff(
            window_terms(c, win.bf, win.ff),
            window_terms(c, win.br, win.fr),
            nchunks,
            beta,
        );
        let run = || {
            for col in 0..ncols {
                let (x, up) = &cols[col % n_unique];
                lrn_col_judge(x, up, &kc, win, coeff)?;
            }
            Ok::<(), AutodiffError>(())
        };
        // 全列を走査した（途中で拒否して短絡していない）ことの担保。
        let result = run();
        assert!(result.is_ok(), "{label}: {result:?}");
        let (med, all) = timing_median(|| {
            let _ = std::hint::black_box(run());
        });
        // 二階の要素数: seed（c 個）ごとの x・up への要素（帯の内側）。
        let entries = ncols
            * (0..c)
                .map(|j| {
                    let (ulo, uhi) = (j.saturating_sub(win.br), (j + win.fr).min(c - 1));
                    let (xlo, xhi) = (ulo.saturating_sub(win.bf), (uhi + win.ff).min(c - 1));
                    (xhi - xlo + 1) + (uhi - ulo + 1)
                })
                .sum::<usize>();
        eprintln!(
            "JUDGETIME kind=LRN {label} ncols={ncols} c={c} size={size} nchunks={nchunks} cost={cost} cost/2^27={:.4} entries={entries} result=accepted(full-scan) median_ms={:.1} runs_ms={:?} ns_per_iter={:.3} ns_per_entry={:.3}",
            cost as f64 / JUDGE_BUDGET as f64,
            med / 1e6,
            timing_runs_ms(&all),
            med / cost as f64,
            med / entries as f64,
        );
    }

    /// WeightNorm の `Rf` 経路（幅が [`LIN_REST_MAX`] 超）を、何もしない検査（全 seed を走査）で計測する。
    /// 併せて本番の検査（`wn_group_judge(use_lin = false)`）の受理件数を出す。
    fn timing_wn_rf(label: &str, groups: usize, rest: usize) {
        let cost = wn_judge_cost(groups, rest);
        assert!(cost <= JUDGE_BUDGET);
        let mut rng = TimingLcg(0x4085 + rest as u64);
        let n_unique = groups.min(TIMING_UNIQUE);
        let inputs: Vec<_> = (0..n_unique)
            .map(|_| timing_wn_input(&mut rng, rest))
            .collect();
        let run = || {
            for gi in 0..groups {
                let (v, up, g, s) = &inputs[gi % n_unique];
                wn_group_judge_with::<Rf>(v, up, *g, *s, &|_, _, _| Ok(()))?;
            }
            Ok::<(), AutodiffError>(())
        };
        assert!(run().is_ok());
        let (med, all) = timing_median(|| {
            let _ = std::hint::black_box(run());
        });
        let prod_ok = inputs
            .iter()
            .filter(|(v, up, g, s)| wn_group_judge(v, up, *g, *s, wn_coeff(rest), false).is_ok())
            .count();
        let entries = groups * (rest + 1) * 2 * rest;
        eprintln!(
            "JUDGETIME kind=WeightNorm(Rf) {label} groups={groups} rest={rest} cost={cost} cost/2^27={:.4} entries={entries} result=full-scan(no-op check) prod_accepts={prod_ok}/{n_unique}(unique inputs) median_ms={:.1} runs_ms={:?} ns_per_iter={:.3} ns_per_entry={:.3}",
            cost as f64 / JUDGE_BUDGET as f64,
            med / 1e6,
            timing_runs_ms(&all),
            med / cost as f64,
            med / entries as f64,
        );
    }

    /// LRN・WeightNorm（`Rf`）・SpectralNorm（`Rf`）の総括判定を、走査量 [`JUDGE_BUDGET`] ぎりぎりの入力で
    /// 計測する。`Lin` 経路は別の予算なので `judge_budget_timing_norm_lin` で測る。
    /// WeightNorm・SpectralNorm は検査を何もしない関数にして全走査させ、本番の検査の結果は別に出す。
    /// 実行: `cargo test -p fandhe-ai-autodiff --release --lib -- --ignored --nocapture
    /// --test-threads=1 judge_budget_timing`。出力は stderr の `JUDGETIME` 行のみ。
    #[test]
    #[ignore = "手動実行専用の計測（stderr 出力のみ。#4085・§83.1）"]
    fn judge_budget_timing_norm_rf() {
        // LRN: (a) 小さい窓で列数を上限まで、(b) 広い窓（size = 2C + 1）で C を上限まで。
        timing_lrn("shape=a(small-window)", None, 16, 5);
        let c_wide = timing_max_by(2, |c| {
            // 1 列。チャンク数は指数 0 の入力で 0 になる。
            let w = lrn_win(c, 2 * c + 1);
            lrn_judge_cost(1, c, w, 0) <= JUDGE_BUDGET
        });
        timing_lrn("shape=b(wide-window)", Some(1), c_wide, 2 * c_wide + 1);
        // WeightNorm（Rf）: (a) 1 グループの最大幅、(b) 幅 256 のグループを上限まで。
        let rest_max = timing_max_by(LIN_REST_MAX + 1, |r| wn_judge_cost(1, r) <= JUDGE_BUDGET);
        timing_wn_rf("shape=a(one-group)", 1, rest_max);
        let groups = timing_max_by(1, |g| wn_judge_cost(g, 256) <= JUDGE_BUDGET);
        assert!(wn_judge_cost(groups + 1, 256) > JUDGE_BUDGET);
        timing_wn_rf("shape=b(many-groups)", groups, 256);
        // SpectralNorm（Rf）: 上限内で最大の正方形（`sn_judge_cost` は N だけで決まる）。
        let side = timing_max_by(2, |m| sn_judge_cost(m * m) <= JUDGE_BUDGET);
        assert!(sn_judge_cost((side + 1) * (side + 1)) > JUDGE_BUDGET);
        let c = sn_case(&[side, side], 1.0, 2.0, SnUp::General, 1.0);
        let inp = sn_inputs(&c);
        let n = inp.w.len();
        let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        let s = f64::from(pow2(pow2_ceil_exponent(mx)));
        let cost = sn_judge_cost(n);
        let run = || sn_judge_with::<Rf>(&inp.w, &inp.p_exact, &inp.up, s, &|_, _, _| Ok(()));
        assert!(run().is_ok());
        let (med, all) = timing_median(|| {
            let _ = std::hint::black_box(run());
        });
        let prod = sn_judge(&inp.w, &inp.p_exact, &inp.up, s, sn_coeff(n), false);
        let entries = n * 2 * n;
        eprintln!(
            "JUDGETIME kind=SpectralNorm(Rf) shape={side}x{side} n={n} cost={cost} cost/2^27={:.4} entries={entries} result=full-scan(no-op check) prod={} median_ms={:.1} runs_ms={:?} ns_per_iter={:.3} ns_per_entry={:.3}",
            cost as f64 / JUDGE_BUDGET as f64,
            if prod.is_ok() { "accepted" } else { "rejected" },
            med / 1e6,
            timing_runs_ms(&all),
            med / cost as f64,
            med / entries as f64,
        );
    }

    /// `Lin` 経路（WeightNorm の疎モード・SpectralNorm）の総括判定を、`Lin` 側の予算 [`LIN_BUDGET`] ぎりぎり
    /// の入力で計測する。時間当たりのコストが `Rf` と違うので `Rf` とは別に測り、同じ入力の `Rf` の
    /// ns/反復との比も出す。検査は何もしない関数にして全走査させ、本番の判定の結果は別に出す。`Lin` の
    /// 領域は thread-local なので `--test-threads=1` で実行する。
    #[test]
    #[ignore = "手動実行専用の計測（stderr 出力のみ。#4085・§83.1）"]
    fn judge_budget_timing_norm_lin() {
        for rest in [8usize, 32, LIN_REST_MAX] {
            let unit = lin_group_cost(rest);
            let groups = LIN_BUDGET / unit;
            assert!(wn_use_lin(groups, rest) && !wn_use_lin(groups + 1, rest));
            let mut rng = TimingLcg(0x4085 + rest as u64);
            let n_unique = groups.min(TIMING_UNIQUE);
            let inputs: Vec<_> = (0..n_unique)
                .map(|_| timing_wn_input(&mut rng, rest))
                .collect();
            let run_lin = || {
                for gi in 0..groups {
                    let (v, up, g, s) = &inputs[gi % n_unique];
                    lin_reset_sparse();
                    wn_group_judge_with::<Lin>(v, up, *g, *s, &|_, _, _| Ok(()))?;
                }
                Ok::<(), AutodiffError>(())
            };
            assert!(run_lin().is_ok());
            let (med, all) = timing_median(|| {
                let _ = std::hint::black_box(run_lin());
            });
            // 同じ入力の Rf（何もしない検査）の ns/反復。グループ数は 64 件に絞る。
            let rf_groups = groups.min(64);
            let run_rf = || {
                for gi in 0..rf_groups {
                    let (v, up, g, s) = &inputs[gi % n_unique];
                    wn_group_judge_with::<Rf>(v, up, *g, *s, &|_, _, _| Ok(()))?;
                }
                Ok::<(), AutodiffError>(())
            };
            let (rf_med, _) = timing_median(|| {
                let _ = std::hint::black_box(run_rf());
            });
            let rf_ns_iter = rf_med / wn_judge_cost(rf_groups, rest) as f64;
            let prod_ok = inputs
                .iter()
                .filter(|(v, up, g, s)| wn_group_judge(v, up, *g, *s, wn_coeff(rest), true).is_ok())
                .count();
            lin_reset();
            let total_units = groups * unit;
            eprintln!(
                "JUDGETIME kind=WeightNorm(Lin,sparse) rest={rest} groups={groups} lin_group_cost={unit} units={total_units} units/LIN_BUDGET={:.4} rf_equiv_cost/2^27={:.4} result=full-scan(no-op check) prod_accepts={prod_ok}/{n_unique}(unique inputs) median_ms={:.1} runs_ms={:?} ns_per_unit={:.3} rf_ns_per_iter={:.3} ratio_unit_to_rf_iter={:.3}",
                total_units as f64 / LIN_BUDGET as f64,
                wn_judge_cost(groups, rest) as f64 / JUDGE_BUDGET as f64,
                med / 1e6,
                timing_runs_ms(&all),
                med / total_units as f64,
                rf_ns_iter,
                (med / total_units as f64) / rf_ns_iter,
            );
        }
        // SpectralNorm（Lin）: 走査量 `300·N²` が LIN_BUDGET 以内の最大の N（積で作れる形）。
        let n_max = timing_max_by(2, sn_use_lin);
        for shape in [[8usize, n_max / 8], [21, 21]] {
            let c = sn_case(&shape, 1.0, 2.0, SnUp::General, 1.0);
            let inp = sn_inputs(&c);
            let n = inp.w.len();
            assert!(sn_use_lin(n), "shape={shape:?} n={n}");
            let mx = inp.w.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let s = f64::from(pow2(pow2_ceil_exponent(mx)));
            let run_lin = || {
                lin_reset();
                let r = sn_judge_with::<Lin>(&inp.w, &inp.p_exact, &inp.up, s, &|_, _, _| Ok(()));
                lin_reset();
                r
            };
            assert!(run_lin().is_ok());
            let (med, all) = timing_median(|| {
                let _ = std::hint::black_box(run_lin());
            });
            let run_rf =
                || sn_judge_with::<Rf>(&inp.w, &inp.p_exact, &inp.up, s, &|_, _, _| Ok(()));
            let (rf_med, _) = timing_median(|| {
                let _ = std::hint::black_box(run_rf());
            });
            let rf_ns_iter = rf_med / sn_judge_cost(n) as f64;
            let prod = sn_judge(&inp.w, &inp.p_exact, &inp.up, s, sn_coeff(n), true);
            let units = sn_lin_cost(n);
            eprintln!(
                "JUDGETIME kind=SpectralNorm(Lin) shape={}x{} n={n} n_max_for_lin={n_max} sn_lin_cost={units} units/LIN_BUDGET={:.4} result=full-scan(no-op check) prod={} median_ms={:.1} runs_ms={:?} ns_per_unit={:.3} rf_ns_per_iter={:.3} ratio_unit_to_rf_iter={:.3}",
                shape[0],
                shape[1],
                units as f64 / LIN_BUDGET as f64,
                if prod.is_ok() { "accepted" } else { "rejected" },
                med / 1e6,
                timing_runs_ms(&all),
                med / units as f64,
                rf_ns_iter,
                (med / units as f64) / rf_ns_iter,
            );
        }
        lin_reset();
    }
}
