//! 要素ごと損失 4 種（pos_weight 付き BCEWithLogits・HingeEmbedding・
//! SoftMargin・GaussianNLL）のホスト参照 forward・VJP カーネル
//! （イシュー #2652・親 #2651）。
//!
//! 呼び出し元は `crate::elementwise_loss_ops`（forward。値検査・shape 検査済みの
//! 入力のみを渡す）と `crate::grad::vjp`（`Op::BceWithLogitsPosWeightLoss` 等の
//! 分岐。`n == 0` の zeros 返却と `scale` 算出は呼び出し側で済ませる）。
//! `BackendOps` に対応メソッドを持たない融合対象外の演算のため、CPU／CUDA／Metal の
//! いずれの tape からも本モジュールのホスト計算で到達する
//! （`Op::PoissonNllLoss` と同型。`docs/autodiff-elementwise-loss-ops-decision.md` §2）。
//!
//! **数値契約**（`.claude/rules/coding-rust.md` の `f64` アキュムレータ契約）: 要素を
//! 先に `f64` へ昇格して要素式を評価し、index 順に `f64` で蓄積して最後に 1 回だけ
//! `f32` へ downcast する。`Mean` は `numel` で除算し、`numel == 0` は `0.0`
//! （既存 `mse_loss` 規約。呼び出し元が空テンソルの VJP を zeros で返す）。
//!
//! 公開入口の関数名（`bce_with_logits_loss_with` 等）とはワークスペース inventory
//! の完全一致検査を避けるため別名（`*_forward`／`*_vjp`）にしている。

use fandhe_ai_tensor_core::Tensor;

use super::{build_tensor, dense_vec, nan_propagating_max_f64};
use crate::var::Reduction;

/// `0.5 * ln(2π)`（`GaussianNLL` の `full = true` 定数項。PyTorch は
/// `0.5 * log(2π)` を `f32` 経由で足すが、本実装は `f64` で厳密に評価する）。
fn half_ln_2pi() -> f64 {
    0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// 縮約済みスカラー損失（shape `[]`）を作る。`Mean` は `numel == 0` で `0.0`。
fn finalize(total: f64, numel: usize, reduction: Reduction) -> Tensor<f32> {
    let out = match reduction {
        Reduction::Mean => {
            if numel == 0 {
                0.0
            } else {
                (total / numel as f64) as f32
            }
        }
        Reduction::Sum => total as f32,
    };
    build_tensor(vec![out], &[])
}

/// `softplus(−x) = ln(1 + exp(−x))` の安定形（`exp` の引数を常に非正にする）。
fn softplus_neg(x: f64) -> f64 {
    if x >= 0.0 {
        (-x).exp().ln_1p()
    } else {
        -x + x.exp().ln_1p()
    }
}

/// `σ(z) = 1 / (1 + exp(−z))` の安定形。
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------
// pos_weight 付き BCEWithLogits
// ---------------------------------------------------------------------

/// 要素損失 `l = (1 − y)·x + (1 + (p − 1)·y)·softplus(−x)`
/// （PyTorch `binary_cross_entropy_with_logits(pos_weight=)` と同じ式）。
fn bce_pw_elem(x: f64, y: f64, p: f64) -> f64 {
    let lw = 1.0 + (p - 1.0) * y;
    (1.0 - y) * x + lw * softplus_neg(x)
}

/// pos_weight 付き BCEWithLogits の forward。`pos_weight` は呼び出し元が
/// `input` と同 shape へ展開済み。
pub(crate) fn bce_with_logits_pos_weight_loss_forward(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    pos_weight: &Tensor<f32>,
    reduction: Reduction,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ys = dense_vec(target);
    let ps = dense_vec(pos_weight);
    let total: f64 = xs
        .iter()
        .zip(&ys)
        .zip(&ps)
        .map(|((&x, &y), &p)| bce_pw_elem(f64::from(x), f64::from(y), f64::from(p)))
        .sum();
    finalize(total, xs.len(), reduction)
}

/// pos_weight 付き BCEWithLogits の VJP。`dx = s·[(1 − y) − lw·σ(−x)]`・
/// `dy = s·[−x + (p − 1)·softplus(−x)]`（`pos_weight` は非追跡）。
pub(crate) fn bce_with_logits_pos_weight_loss_vjp(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    pos_weight: &Tensor<f32>,
    scale: f64,
) -> (Tensor<f32>, Tensor<f32>) {
    let xs = dense_vec(input);
    let ys = dense_vec(target);
    let ps = dense_vec(pos_weight);
    let mut dx = Vec::with_capacity(xs.len());
    let mut dy = Vec::with_capacity(xs.len());
    for ((&x, &y), &p) in xs.iter().zip(&ys).zip(&ps) {
        let (x, y, p) = (f64::from(x), f64::from(y), f64::from(p));
        let lw = 1.0 + (p - 1.0) * y;
        dx.push((scale * ((1.0 - y) - lw * sigmoid(-x))) as f32);
        dy.push((scale * (-x + (p - 1.0) * softplus_neg(x))) as f32);
    }
    (
        build_tensor(dx, input.shape()),
        build_tensor(dy, input.shape()),
    )
}

// ---------------------------------------------------------------------
// HingeEmbedding
// ---------------------------------------------------------------------

/// 要素損失（`y == 1` は `x`、それ以外〈`y == −1`〉は `max(0, margin − x)`。
/// `NaN` は伝播する）。
fn hinge_elem(x: f64, y: f64, margin: f64) -> f64 {
    if y == 1.0 {
        x
    } else {
        nan_propagating_max_f64(margin - x, 0.0)
    }
}

/// HingeEmbedding の forward（`y` は検証済みの ±1・`input` と同 shape）。
pub(crate) fn hinge_embedding_loss_forward(
    input: &Tensor<f32>,
    y: &Tensor<f32>,
    margin: f32,
    reduction: Reduction,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ys = dense_vec(y);
    let total: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| hinge_elem(f64::from(x), f64::from(y), f64::from(margin)))
        .sum();
    finalize(total, xs.len(), reduction)
}

/// HingeEmbedding の VJP。`y == 1` は `dx = s`、`y == −1` は
/// `margin − x > 0` のときのみ `dx = −s`。境界ちょうど（`x == margin`）の勾配は 0
/// （PyTorch 2.14.0 の実測。`margin_ranking_loss` の `clamp_min` 契約〈境界で勾配を通す〉
/// とは異なる。fixture `hinge_boundary` で確定）。
pub(crate) fn hinge_embedding_loss_vjp(
    input: &Tensor<f32>,
    y: &Tensor<f32>,
    margin: f32,
    scale: f64,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ys = dense_vec(y);
    let dx: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| {
            if y == 1.0 {
                scale as f32
            } else if f64::from(margin) - f64::from(x) > 0.0 {
                (-scale) as f32
            } else {
                0.0
            }
        })
        .collect();
    build_tensor(dx, input.shape())
}

// ---------------------------------------------------------------------
// SoftMargin
// ---------------------------------------------------------------------

/// 要素損失 `l = ln(1 + exp(z))`（`z = −y·x`）の安定形
/// `max(z, 0) + ln_1p(exp(−|z|))`。
fn soft_margin_elem(x: f64, y: f64) -> f64 {
    let z = -y * x;
    nan_propagating_max_f64(z, 0.0) + (-z.abs()).exp().ln_1p()
}

/// SoftMargin の forward（`y` は検証済みの ±1・`input` と同 shape）。
pub(crate) fn soft_margin_loss_forward(
    input: &Tensor<f32>,
    y: &Tensor<f32>,
    reduction: Reduction,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ys = dense_vec(y);
    let total: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| soft_margin_elem(f64::from(x), f64::from(y)))
        .sum();
    finalize(total, xs.len(), reduction)
}

/// SoftMargin の VJP。`dx = s·(−y)·σ(−y·x)`。
pub(crate) fn soft_margin_loss_vjp(
    input: &Tensor<f32>,
    y: &Tensor<f32>,
    scale: f64,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ys = dense_vec(y);
    let dx: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| {
            let (x, y) = (f64::from(x), f64::from(y));
            (scale * (-y) * sigmoid(-y * x)) as f32
        })
        .collect();
    build_tensor(dx, input.shape())
}

// ---------------------------------------------------------------------
// GaussianNLL
// ---------------------------------------------------------------------

/// 要素損失 `0.5·(ln c + (x − t)²/c)`（`c = max(var, eps)`。`full` は定数項
/// `0.5·ln(2π)` を加える）。
fn gaussian_nll_elem(x: f64, t: f64, v: f64, eps: f64, full: bool) -> f64 {
    let c = nan_propagating_max_f64(v, eps);
    let d = x - t;
    let base = 0.5 * (c.ln() + d * d / c);
    if full { base + half_ln_2pi() } else { base }
}

/// GaussianNLL の forward（3 入力は同 shape・`var >= 0` 検証済み）。
pub(crate) fn gaussian_nll_loss_forward(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    var: &Tensor<f32>,
    eps: f32,
    full: bool,
    reduction: Reduction,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ts = dense_vec(target);
    let vs = dense_vec(var);
    let total: f64 = xs
        .iter()
        .zip(&ts)
        .zip(&vs)
        .map(|((&x, &t), &v)| {
            gaussian_nll_elem(
                f64::from(x),
                f64::from(t),
                f64::from(v),
                f64::from(eps),
                full,
            )
        })
        .sum();
    finalize(total, xs.len(), reduction)
}

/// GaussianNLL の VJP。`dx = s·d/c`・`dt = −dx`・`dvar = s·0.5·(1/c − d²/c²)`。
/// `var < eps` の要素でも `dvar` を 0 にしない（PyTorch は `no_grad` 下の clamp で
/// 勾配を遮らない。fixture で確定済み）。
pub(crate) fn gaussian_nll_loss_vjp(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    var: &Tensor<f32>,
    eps: f32,
    scale: f64,
) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let xs = dense_vec(input);
    let ts = dense_vec(target);
    let vs = dense_vec(var);
    let mut dx = Vec::with_capacity(xs.len());
    let mut dt = Vec::with_capacity(xs.len());
    let mut dv = Vec::with_capacity(xs.len());
    for ((&x, &t), &v) in xs.iter().zip(&ts).zip(&vs) {
        let c = nan_propagating_max_f64(f64::from(v), f64::from(eps));
        let d = f64::from(x) - f64::from(t);
        let gx = scale * d / c;
        dx.push(gx as f32);
        dt.push((-gx) as f32);
        dv.push((scale * 0.5 * (1.0 / c - d * d / (c * c))) as f32);
    }
    let shape = input.shape();
    (
        build_tensor(dx, shape),
        build_tensor(dt, shape),
        build_tensor(dv, shape),
    )
}
