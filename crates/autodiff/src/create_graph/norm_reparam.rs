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
//!   VJP 係数が 1 つでも非有限なら（逆伝播は全要素に `cot·係数` を作るので `0·inf = NaN`）拒否する。
//!   走査量が `2^27` 要素を超える入力は計算量の fail-closed として拒否する。構造を積んだ後の 1 階の値の照合
//!   （[`shadow_check`]）も残す。
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

use super::{child_value, push_v3};
use crate::create_graph_norm::{STAGE_BOUND_GUARD, pow2_ceil_exponent};
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
    if v.is_finite() && v.abs() < STAGE_BOUND_GUARD {
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
    (2.0 * terms as f64 + 4.0) * (-23f64).exp2()
}

/// 1 要素の構造値 `v_str`（f32）が参照値 `v` に REQ-2 の統一複合判定で一致するかの保守的検査。
/// `m` は同じ式を各項の絶対値の和で評価した大きさ。NaN の参照は伝播の契約に従い素通しにする。
fn struct_check(v: f64, v_str: f64, m: f64, coeff: f64, what: &str) -> Result<(), AutodiffError> {
    if v.is_nan() || m.is_nan() {
        return Ok(());
    }
    let err = (v_str - v).abs() + coeff * m;
    if err.is_finite()
        && (err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * v.abs())
    {
        return Ok(());
    }
    Err(invalid(&format!(
        "{what}: f32 構造の丸め・相殺・underflow で二階が REQ-2 の判定を外れうるため拒否"
    )))
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
// 逆伝播は全要素に `cot·係数` を作るので、f32 の forward 値・VJP 係数が 1 つでも非有限なら `V_str` は
// NaN になる（`0·inf`）。これは forward の再現で非有限を見つけた時点で拒否する。

/// 総括判定で走査する二階の要素数の上限。超える入力は計算量の fail-closed として拒否する
/// （保守的。上限内の入力だけに保証を限る）。
const JUDGE_BUDGET: usize = 1 << 27;

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
    fn plus(self, o: Self) -> Self;
    fn minus(self, o: Self) -> Self;
    fn times(self, o: Self) -> Self;
    fn over(self, o: Self) -> Self;
    fn negate(self) -> Self;
    fn root(self) -> Self;
    /// `PowScalar` の forward `x^e`（構造は `e as f32` を使う）。
    fn powe(self, e: f64) -> Self;
    /// `PowScalar` の VJP 係数 `e·x^(e−1)`（`e == 0` は 0。`eval/scalar.rs::unary_grad_factor`）。
    fn pow_grad(self, e: f64) -> Self;
    /// `Sqrt` の VJP 係数 `0.5 / y`（`self` は forward の出力 `y`）。
    fn root_grad(self) -> Self;
    fn finite(self) -> bool;
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
    fn finite(self) -> bool {
        self.is_finite()
    }
}

/// f64 の参照値 `v` と、同じ式を各項の絶対値で評価した大きさ `m`（`m ≥ |v|`。相殺が無ければ `|v|`）。
/// 和・差は `m` を足し、積は `m` を掛ける。商・平方根・累乗は分母・底に**値** `|v|` を使い、分母・底の
/// 相殺は増幅率 `κ = m/|v|` として掛ける（`m` を分母に使うと相殺で判定が緩むため）。累乗 `x^e` が底の
/// 丸め誤差を `|e|` 倍する分は、`loss_class.rs` の Triplet と同じく係数側（[`judge_coeff`] の `amp`）に足す。
#[derive(Clone, Copy, Debug)]
struct Rf {
    v: f64,
    m: f64,
}

impl Rf {
    fn kappa(self) -> f64 {
        if self.v == 0.0 {
            1.0
        } else {
            (self.m / self.v.abs()).max(1.0)
        }
    }
}

impl Num for Rf {
    fn k(x: f64) -> Self {
        Rf { v: x, m: x.abs() }
    }
    fn plus(self, o: Self) -> Self {
        Rf {
            v: self.v + o.v,
            m: self.m + o.m,
        }
    }
    fn minus(self, o: Self) -> Self {
        Rf {
            v: self.v - o.v,
            m: self.m + o.m,
        }
    }
    fn times(self, o: Self) -> Self {
        Rf {
            v: self.v * o.v,
            m: self.m * o.m,
        }
    }
    fn over(self, o: Self) -> Self {
        Rf {
            v: self.v / o.v,
            m: self.m / o.v.abs() * o.kappa(),
        }
    }
    fn negate(self) -> Self {
        Rf {
            v: -self.v,
            m: self.m,
        }
    }
    fn root(self) -> Self {
        let v = self.v.sqrt();
        Rf {
            v,
            m: v.abs() * self.kappa(),
        }
    }
    fn powe(self, e: f64) -> Self {
        let v = self.v.powf(e);
        Rf {
            v,
            m: v.abs() * self.kappa(),
        }
    }
    fn pow_grad(self, e: f64) -> Self {
        if e == 0.0 {
            return Rf { v: 0.0, m: 0.0 };
        }
        let v = e * self.v.powf(e - 1.0);
        Rf {
            v,
            m: v.abs() * self.kappa(),
        }
    }
    fn root_grad(self) -> Self {
        let v = 0.5 / self.v;
        Rf {
            v,
            m: v.abs() * self.kappa(),
        }
    }
    fn finite(self) -> bool {
        self.v.is_finite() && self.m.is_finite()
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

fn all_finite<T: Num>(vals: &[T]) -> bool {
    vals.iter().all(|v| v.finite())
}

/// 二階の 1 要素の判定（`loss_class.rs::struct_check` と同じ形。scale は 1）。`r` は参照値 `V` と大きさ
/// `M`、`s` は構造の f32 の再現値 `V_str`。`(|V_str − V| + coeff·M)` が絶対 1e-5 未満か相対の半分以内で
/// なければ拒否する（tolerance 定数は変えない）。判定に掛けるのは NaN を含まないレーンだけなので、
/// どれかが非有限なら拒否する。
fn entry_check(r: Rf, s: f32, coeff: f64, what: &str) -> Result<(), AutodiffError> {
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
/// f32 の forward 値・VJP 係数のどれかが非有限なら `None`。
fn lrn_col_fwd<T: Num>(
    x: &[f32],
    up: &[f32],
    k: &LrnColConst<'_>,
    w: LrnWin,
) -> Option<LrnColFwd<T>> {
    let s = T::k(k.s);
    let inv_s = T::k(1.0).over(s);
    let xs: Vec<T> = x.iter().map(|&v| T::k(f64::from(v)).over(s)).collect();
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
    let mut fwd_ok = all_finite(&dx);
    for &ch in k.chunks {
        for v in &mut dx {
            *v = v.times(T::k(ch));
        }
        fwd_ok &= all_finite(&dx);
    }
    let mut ci = T::k(1.0);
    for &ch in k.chunks.iter().rev() {
        ci = ci.times(T::k(ch));
    }
    let ok = fwd_ok
        && inv_s.finite()
        && ci.finite()
        && [&xs, &sq, &d, &a, &b, &fa, &fb, &ux, &z, &sz]
            .iter()
            .all(|v| all_finite(v));
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

    // 列 (n, inner) ごとのスケール指数。
    let ncols = n * inner;
    let mut exps = vec![0i32; ncols];
    for ni in 0..n {
        for s in 0..inner {
            let mut mx = 0.0f64;
            let mut has_nan = false;
            for ci in 0..c {
                let v = x_data[at(ni, ci, s)];
                has_nan |= v.is_nan();
                mx = mx.max(f64::from(v).abs());
            }
            exps[ni * inner + s] = if has_nan {
                0
            } else {
                pow2_ceil_exponent(mx.max(floor))
            };
        }
    }
    let scales: Vec<f32> = exps.iter().map(|&e| pow2(e)).collect();
    let kk: Vec<f32> = exps
        .iter()
        .map(|&e| (k / 2f64.powi(2 * e)) as f32)
        .collect();
    let l_exp: Vec<f64> = exps.iter().map(|&e| -2.0 * beta * f64::from(e)).collect();
    let chunks = chunk_factors(&l_exp);
    // 総括判定（第 2 軸）の窓・係数（[`lrn_coeff`]）・走査量。
    let win = LrnWin {
        bf: back_f.min(c - 1),
        ff: fwd_f.min(c - 1),
        br: back_r.min(c - 1),
        fr: fwd_r.min(c - 1),
    };
    let coeff_judge = lrn_coeff(nt_f, nt_r, chunks.len(), beta);
    let per_elem = (win.br + win.fr + 1) + (win.br + win.fr + win.bf + win.ff + 1).min(c);
    judge_budget(x_data.len().saturating_mul(per_elem), "LocalResponseNorm")?;

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
            if col_idx
                .iter()
                .any(|&i| x_data[i].is_nan() || xs[i].is_nan())
            {
                continue;
            }
            let big_f = l_exp[col].exp2();
            stage(big_f, "LRN の s^{-2β}")?;
            let mut d = vec![0.0f64; c];
            for (ci, dc) in d.iter_mut().enumerate() {
                let lo = ci.saturating_sub(back_f);
                let hi = (ci + fwd_f).min(c - 1);
                let sum: f64 = (lo..=hi).map(|cj| f64::from(xs[col_idx[cj]]).powi(2)).sum();
                *dc = f64::from(kk[col]) + rho * sum;
            }
            if d.iter()
                .any(|&dc| dc.is_nan() || dc <= 0.0 || (dc as f32) < f32::MIN_POSITIVE)
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
/// f32 の forward 値・VJP 係数のどれかが非有限なら `None`。
fn wn_fwd<T: Num>(v: &[f32], up: &[f32], g: f32, s: f64) -> Option<WnFwd<T>> {
    let s = T::k(s);
    let inv_s = T::k(1.0).over(s);
    let vs: Vec<T> = v.iter().map(|&x| T::k(f64::from(x)).over(s)).collect();
    let up: Vec<T> = up.iter().map(|&x| T::k(f64::from(x))).collect();
    let mut ss = T::k(0.0);
    for &x in &vs {
        ss = ss.plus(x.times(x));
    }
    let n = ss.root();
    let fsq = n.root_grad();
    let inv_n = T::k(1.0).over(n);
    let vh: Vec<T> = vs.iter().map(|&x| x.over(n)).collect();
    let dvb: Vec<T> = vs.iter().map(|&x| x.over(n).over(n).negate()).collect();
    let mut dg = T::k(0.0);
    for (&u, &h) in up.iter().zip(&vh) {
        dg = dg.plus(u.times(h));
    }
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
        .all(|v| v.finite())
        && [&vs, &vh, &dvb, &t, &dv].iter().all(|v| all_finite(v));
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
    for i in 0..r {
        hv[i] = cot_vh[i].times(f.inv_n);
        cot_n = cot_n.plus(cot_vh[i].times(f.dvb[i]));
    }
    // n_s = √(Σ v_s²)、v_s = v/s。
    let cot_ss = cot_n.times(f.fsq);
    for (h, &x) in hv.iter_mut().zip(&f.vs) {
        let g = cot_ss.times(x);
        *h = h.plus(g).plus(g).times(f.inv_s);
    }
    h_g
}

/// WeightNorm の 1 グループの総括判定。seed が 1 要素（`dv` の各要素と `dg`）のときの二階
/// （`v`・`up`・`g` への要素）を [`entry_check`] に掛ける。
fn wn_group_judge(v: &[f32], up: &[f32], g: f32, s: f64, coeff: f64) -> Result<(), AutodiffError> {
    const WHAT: &str = "WeightNorm の二階";
    let (Some(fs), Some(fr)) = (wn_fwd::<f32>(v, up, g, s), wn_fwd::<Rf>(v, up, g, s)) else {
        return Err(nonfinite_struct(WHAT));
    };
    let r = v.len();
    let (mut hv_s, mut hu_s) = (vec![0f32; r], vec![0f32; r]);
    let (mut hv_r, mut hu_r) = (vec![Rf::k(0.0); r], vec![Rf::k(0.0); r]);
    for seed in (0..r).map(WnSeed::Dv).chain(std::iter::once(WnSeed::Dg)) {
        let gs = wn_rev(&fs, seed, &mut hv_s, &mut hu_s);
        let gr = wn_rev(&fr, seed, &mut hv_r, &mut hu_r);
        if let (Some(gs), Some(gr)) = (gs, gr) {
            entry_check(gr, gs, coeff, WHAT)?;
        }
        for i in 0..r {
            entry_check(hv_r[i], hv_s[i], coeff, WHAT)?;
            entry_check(hu_r[i], hu_s[i], coeff, WHAT)?;
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
    let coeff_judge = wn_coeff(rest);
    judge_budget(
        groups.saturating_mul(rest + 1).saturating_mul(2 * rest + 1),
        "WeightNorm",
    )?;
    let nf = rest as f64;
    let mut m_dv = vec![0.0f64; v_gm.len()];
    let mut m_dg = vec![0.0f64; groups];
    for k in 0..groups {
        let lo = k * rest;
        let vs: Vec<f32> = v_gm[lo..lo + rest].iter().map(|&x| x / scales[k]).collect();
        if vs.iter().any(|x| x.is_nan()) || up_gm[lo..lo + rest].iter().any(|x| x.is_nan()) {
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
        // g が NaN のグループは dv が NaN になる（1 階と同じ伝播）ので判定から外す。
        if !g_data[k].is_nan() {
            wn_group_judge(&v_gm[lo..lo + rest], upk, g_data[k], s, coeff_judge)?;
        }
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
    let ok = [sig, inv_sig, inv_s, sp].iter().all(|v| v.finite())
        && [&ws, &p, &wh, &dwb, &t, &dob, &dw]
            .iter()
            .all(|v| all_finite(v));
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

/// SpectralNorm の総括判定。seed が 1 要素のときの二階（`W`・`up` への要素）を [`entry_check`] に掛ける。
fn sn_judge(w: &[f32], p: &[f64], up: &[f32], s: f64, coeff: f64) -> Result<(), AutodiffError> {
    const WHAT: &str = "SpectralNorm の二階";
    let (Some(fs), Some(fr)) = (sn_fwd::<f32>(w, p, up, s), sn_fwd::<Rf>(w, p, up, s)) else {
        return Err(nonfinite_struct(WHAT));
    };
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
    let has_nan = w_data.iter().any(|x| x.is_nan()) || up_data.iter().any(|x| x.is_nan());

    let e = if w_data.iter().any(|x| x.is_nan()) {
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
    let mut m_abs = vec![0.0f64; w_data.len()];
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
        judge_budget(
            w_data.len().saturating_mul(2 * w_data.len()),
            "SpectralNorm",
        )?;
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
        let wc = tape.var_no_grad(&tt(w.clone(), &shape));
        let y = x.local_response_norm(size, alpha, beta, k).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        // 判定が保守的に拒否した入力は構造が積まれないので比べられない（呼び出し側で件数を確かめる）。
        let Ok(cg) = tape.backward_create_graph(&loss, &child) else {
            return false;
        };
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
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
                let (ulo, uhi) = (jc.saturating_sub(win.br), (jc + win.fr).min(c - 1));
                let (xlo, xhi) = (ulo.saturating_sub(win.bf), (uhi + win.ff).min(c - 1));
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
        let wc = tape.var_no_grad(&tt(w.clone(), &[groups, rest]));
        let y = v.weight_norm(&g, Some(0)).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gv = cg.grad(&v).unwrap().unwrap();
        let gg = cg.grad(&g).unwrap().unwrap();
        let cv = cg.child_var(&v).unwrap().unwrap();
        let cgv = cg.child_var(&g).unwrap().unwrap();
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
                let (row_v, row_g) = match seed {
                    WnSeed::Dv(j) => (
                        child_row(&child, &gv, lo + j, &cv),
                        Some(child_row(&child, &gv, lo + j, &cgv)),
                    ),
                    WnSeed::Dg => (child_row(&child, &gg, kg, &cv), None),
                };
                for i in 0..rest {
                    assert_within(row_v[lo + i], hv_s[i], hv_r[i], coeff, "wn v");
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
        let wc = tape.var_no_grad(&tt(up.clone(), &shape));
        let y = w.spectral_norm(&mut st, false).unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gw = cg.grad(&w).unwrap().unwrap();
        let cw = cg.child_var(&w).unwrap().unwrap();
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
            for i in 0..6 {
                assert_within(row[i], hw_s[i], hw_r[i], coeff, "spectral W");
            }
        }
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
        assert!(entry_check(Rf { v: 0.0, m: 1e3 }, 0.0, c, "t").is_err());
        // f32 だけが非有限・参照が非有限はどちらも拒否。
        assert!(entry_check(Rf::k(1.0), f32::NAN, c, "t").is_err());
        assert!(entry_check(Rf::k(f64::NAN), 1.0, c, "t").is_err());
        // 分母の相殺は κ として商の大きさへ入る（分母の m では割らない）。
        let den = Rf::k(1.0).minus(Rf::k(0.999));
        let q = Rf::k(1.0).over(den);
        assert!(q.m >= q.v.abs() * 1999.0 * 0.999, "{q:?}");
        assert!(judge_budget(JUDGE_BUDGET, "t").is_ok());
        assert!(judge_budget(JUDGE_BUDGET + 1, "t").is_err());
    }
}
