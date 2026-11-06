//! `backward_create_graph` の「総括判定」（二階の構造を子テープへ積む前に、f32 の overflow・
//! underflow・相殺で REQ-2 の統一複合判定を外れうる入力を型付き `Err(Backward)` で拒否する検査）の
//! 共通補助関数（イシュー #3955・設計記録 `docs/autodiff-higher-order-grad-decision.md` §77）。
//!
//! 呼び出し元: `loss_class.rs`（§40・§55）・`loss_regression.rs`（§39・§53）・
//! `loss_binary_margin.rs`（§39・§54）・`norm_reparam.rs`（§36・§57）。各モジュールは族名と節番号を
//! 含むエラー文面を持つため、ここの関数は `reject: fn(&str) -> AutodiffError`（各モジュールの `invalid`）
//! を受け取って文面を呼び出し元に委ねる。許容誤差（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）と
//! `STAGE_BOUND_GUARD` は変更しない。
//!
//! NaN の扱いが族ごとに異なる点は統合せず並置する（挙動不変の共通化）:
//! - loss 系 [`stage`]・[`struct_check`]: NaN レーンは伝播の契約に従い素通し。
//! - norm 系 [`stage_strict_ok`]・[`struct_check_unscaled`]: 段検査の NaN は拒否。参照値 `v` が NaN のときだけ
//!   素通しで、`m` が NaN の有限要素は拒否する。

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::Reduction;

/// 拒否時の文面を作る関数（各モジュールの `invalid`）。
pub(super) type Reject = fn(&str) -> AutodiffError;

/// loss 系の 1 段の上界 `v`（f64）を検査する。NaN は素通し、非有限または `STAGE_BOUND_GUARD` 以上は拒否。
pub(super) fn stage(v: f64, reject: Reject) -> Result<(), AutodiffError> {
    if v.is_nan() {
        return Ok(());
    }
    if !v.is_finite() || v.abs() >= STAGE_BOUND_GUARD {
        return Err(reject("構造の中間値が f32 範囲を超えうる入力のため拒否"));
    }
    Ok(())
}

/// norm 系の段検査: NaN を含め、有限かつ `STAGE_BOUND_GUARD` 未満のときだけ true。
pub(super) fn stage_strict_ok(v: f64) -> bool {
    v.is_finite() && v.abs() < STAGE_BOUND_GUARD
}

/// 丸め誤差係数 `(2·terms + 4)·2^-23`（`linalg_fft.rs::rounding_ok` と同じ形）。
/// `terms` は構造の最長の縮約長。
pub(super) fn rounding_coeff(terms: usize) -> f64 {
    (2.0 * terms as f64 + 4.0) * (-23f64).exp2()
}

/// 上流 `g`（スカラー）の絶対値を返す（有限・単一要素を要求し、段検査も通す）。
pub(super) fn upstream_abs(g_value: &Tensor<f32>, reject: Reject) -> Result<f64, AutodiffError> {
    let data = crate::eval::dense_vec(g_value);
    let Some(&gv) = data.first() else {
        return Err(reject("上流勾配が空"));
    };
    if data.len() != 1 || !gv.is_finite() {
        return Err(reject("上流勾配が非有限（または非スカラ）のため拒否"));
    }
    let gv = f64::from(gv).abs();
    stage(gv, reject)?;
    Ok(gv)
}

/// reduction 後の scale 上界 `|g|/n`（Mean）／`|g|`（Sum）。
pub(super) fn upstream_scale(
    g_value: &Tensor<f32>,
    reduction: Reduction,
    n: usize,
    reject: Reject,
) -> Result<f64, AutodiffError> {
    let gv = upstream_abs(g_value, reject)?;
    Ok(match reduction {
        Reduction::Mean => gv / n as f64,
        Reduction::Sum => gv,
    })
}

/// 子テープの `loss_scale_bcast` が f32 で作る `s_b`（Mean は `g / (n as f32)`、Sum は `g`）の再現。
pub(super) fn upstream_scale32(g_value: &Tensor<f32>, reduction: Reduction, n: usize) -> f32 {
    let g = crate::eval::dense_vec(g_value)
        .first()
        .copied()
        .unwrap_or(0.0);
    match reduction {
        Reduction::Mean => g / n as f32,
        Reduction::Sum => g,
    }
}

/// `g` への cotangent 上界（要素寄与の絶対値和 × 要素あたりの係数 `1/n`／`1`）を検査する。
pub(super) fn check_g_cotangent(
    sum_abs: f64,
    reduction: Reduction,
    n: usize,
    reject: Reject,
) -> Result<(), AutodiffError> {
    let v = match reduction {
        Reduction::Mean => sum_abs / n as f64,
        Reduction::Sum => sum_abs,
    };
    stage(v, reject)
}

/// 親テープのノード `id` の実値を連続ベクトルで取り出す。
pub(super) fn dense(
    parent: &Tape,
    parent_nodes: &[TapeNode],
    id: NodeId,
) -> Result<Vec<f32>, AutodiffError> {
    Ok(crate::eval::dense_vec(materialize_fallible(
        parent_nodes,
        parent.ops(),
        id,
    )?))
}

/// 入力に ±inf を含むなら拒否する（`0·inf = NaN` の混入を構造を積む前に止める）。
pub(super) fn reject_inf(values: &[&[f32]], reject: Reject) -> Result<(), AutodiffError> {
    if values.iter().any(|v| v.iter().any(|x| x.is_infinite())) {
        return Err(reject("入力に ±inf を含むため拒否"));
    }
    Ok(())
}

fn struct_reject(reject: Reject, what: &str) -> AutodiffError {
    reject(&format!(
        "{what}: f32 構造の丸め・相殺・underflow で二階が REQ-2 の判定を外れうるため拒否"
    ))
}

/// loss 系の要素検査（設計 §55.2）。誤差の見積り `(|v_str − v| + coeff·m)·scale` が絶対 1e-5 未満か、
/// 相対の半分以内でなければ拒否する。`v`・`m`・`scale` のいずれかが NaN なら素通し。
pub(super) fn struct_check(
    v: f64,
    v_str: f64,
    m: f64,
    scale: f64,
    coeff: f64,
    what: &str,
    reject: Reject,
) -> Result<(), AutodiffError> {
    if v.is_nan() || m.is_nan() || scale.is_nan() {
        return Ok(());
    }
    let err = (v_str - v).abs() + coeff * m;
    let bound = err * scale;
    if bound.is_finite()
        && (bound < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * v.abs())
    {
        return Ok(());
    }
    Err(struct_reject(reject, what))
}

/// norm 系の要素検査（scale なし、設計 §57）。参照値 `v` が NaN のときだけ素通しし、有限な `v` で
/// `m` が NaN なら拒否する。
pub(super) fn struct_check_unscaled(
    v: f64,
    v_str: f64,
    m: f64,
    coeff: f64,
    what: &str,
    reject: Reject,
) -> Result<(), AutodiffError> {
    if v.is_nan() {
        return Ok(());
    }
    let err = (v_str - v).abs() + coeff * m;
    if err.is_finite()
        && (err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * v.abs())
    {
        return Ok(());
    }
    Err(struct_reject(reject, what))
}
