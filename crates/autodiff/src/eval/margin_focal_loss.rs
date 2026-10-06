//! MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss の
//! ホスト参照 forward・VJP カーネル（イシュー #2653・親 #2651）。
//!
//! 呼び出し元は `crate::margin_focal_loss_ops`（forward。値検査・shape 検査済みの
//! 入力のみを渡す）と `crate::grad::vjp`（`Op::MultiMarginLoss` 等の分岐。`n == 0` の
//! zeros 返却と `scale` 算出は呼び出し側で済ませる）。`BackendOps` に対応メソッドを
//! 持たない融合対象外の演算のため、CPU／CUDA／Metal のいずれの tape からも本モジュールの
//! ホスト計算で到達する（`crate::eval::elementwise_loss` と同型。
//! `docs/autodiff-margin-focal-loss-ops-decision.md` §2）。
//!
//! **数値契約**（`.claude/rules/coding-rust.md` の `f64` アキュムレータ契約）: 要素を
//! 先に `f64` へ昇格して要素式を評価し、行内・行間とも index 順に `f64` で蓄積して最後に
//! 1 回だけ `f32` へ downcast する。`Mean` の分母は multi 系 3 種が行数 `N`
//! （rank 1 入力は 1）、focal が `numel`（呼び出し側が `elementwise_loss_scale` へ渡す
//! `n` と本モジュールの `finalize` の分母は同じ値でなければならない）。
//!
//! マージン系の hinge 判定は `z > 0` の厳密不等号（PyTorch 2.14.0 実測: `z == 0`・`NaN` は
//! 損失にも勾配にも寄与しない。fixture `mm_boundary_*`・`mm_nan_inf` で確定）。
//!
//! 公開入口の関数名（`multi_margin_loss` 等）とはワークスペース inventory の完全一致検査を
//! 避けるため別名（`*_forward`／`*_vjp`）にしている。

use fandhe_ai_tensor_core::Tensor;

use super::elementwise_loss::{finalize, sigmoid, softplus_neg};
use super::{build_tensor, dense_vec};
use crate::var::Reduction;

/// 入力 shape から `(行数 N, クラス数 C)` を得る（rank 1 は `(1, C)`、rank 2 は
/// `(N, C)`。それ以外は呼び出し側が事前に拒否済みで、到達しない前提の防御として
/// `(1, numel)` を返す）。
pub(crate) fn rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape {
        [c] => (1, *c),
        [n, c] => (*n, *c),
        other => (1, other.iter().product()),
    }
}

// ---------------------------------------------------------------------
// MultiMargin
// ---------------------------------------------------------------------

/// 1 行の `Σ_{j≠y} h(z_j)`（`z_j = margin − x_y + x_j`。`z > 0` のときのみ寄与）。
fn multi_margin_row_sum(row: &[f32], y: usize, p: u8, margin: f64) -> f64 {
    let xy = f64::from(row[y]);
    let mut acc = 0.0f64;
    for (j, &xj) in row.iter().enumerate() {
        if j == y {
            continue;
        }
        let z = margin - xy + f64::from(xj);
        if z > 0.0 {
            acc += if p == 2 { z * z } else { z };
        }
    }
    acc
}

/// MultiMargin の forward。`targets` は検証済み（`0 <= t < C`・長さ `N`）、`weight` は
/// あれば長さ `C`。
pub(crate) fn multi_margin_loss_forward(
    input: &Tensor<f32>,
    targets: &[i32],
    weight: Option<&[f32]>,
    p: u8,
    margin: f32,
    reduction: Reduction,
) -> Tensor<f32> {
    let (n, c) = rows_cols(input.shape());
    let xs = dense_vec(input);
    let mut total = 0.0f64;
    for i in 0..n {
        let y = targets[i] as usize;
        let w = weight.map_or(1.0, |w| f64::from(w[y]));
        let row = &xs[i * c..(i + 1) * c];
        total += w * multi_margin_row_sum(row, y, p, f64::from(margin)) / c as f64;
    }
    finalize(total, n, reduction)
}

/// MultiMargin の VJP。`z_j > 0` の `j ≠ y` に `g_j = s·w[y]·(1 または 2·z_j)/C` を加え、
/// `x_y` から同量を引く。
pub(crate) fn multi_margin_loss_vjp(
    input: &Tensor<f32>,
    targets: &[i32],
    weight: Option<&[f32]>,
    p: u8,
    margin: f32,
    scale: f64,
) -> Tensor<f32> {
    let (n, c) = rows_cols(input.shape());
    let xs = dense_vec(input);
    let mut dx = vec![0f32; n * c];
    let mut row_grad = vec![0f64; c];
    for i in 0..n {
        let y = targets[i] as usize;
        let w = weight.map_or(1.0, |w| f64::from(w[y]));
        let row = &xs[i * c..(i + 1) * c];
        let xy = f64::from(row[y]);
        row_grad.iter_mut().for_each(|g| *g = 0.0);
        for (j, &xj) in row.iter().enumerate() {
            if j == y {
                continue;
            }
            let z = f64::from(margin) - xy + f64::from(xj);
            if z > 0.0 {
                let h = if p == 2 { 2.0 * z } else { 1.0 };
                let g = scale * w * h / c as f64;
                row_grad[j] += g;
                row_grad[y] -= g;
            }
        }
        for (j, g) in row_grad.iter().enumerate() {
            dx[i * c + j] = *g as f32;
        }
    }
    build_tensor(dx, input.shape())
}

// ---------------------------------------------------------------------
// MultiLabelMargin
// ---------------------------------------------------------------------

/// 1 行の target 列（先頭から最初の負値の手前まで。終端なしなら全 `C` 要素）と
/// target 集合マスクを返す。呼び出し側が行ごとに再利用するバッファ `mask` は長さ `C`。
fn multilabel_row_targets<'a>(row_targets: &'a [i32], mask: &mut [bool]) -> &'a [i32] {
    let len = row_targets
        .iter()
        .position(|&t| t < 0)
        .unwrap_or(row_targets.len());
    mask.iter_mut().for_each(|m| *m = false);
    for &t in &row_targets[..len] {
        mask[t as usize] = true;
    }
    &row_targets[..len]
}

/// MultiLabelMargin の forward。`targets` は検証済み（長さ `N·C`・全要素 `-1 <= t < C`）。
pub(crate) fn multilabel_margin_loss_forward(
    input: &Tensor<f32>,
    targets: &[i32],
    reduction: Reduction,
) -> Tensor<f32> {
    let (n, c) = rows_cols(input.shape());
    let xs = dense_vec(input);
    let mut mask = vec![false; c];
    let mut total = 0.0f64;
    for i in 0..n {
        let row = &xs[i * c..(i + 1) * c];
        let list = multilabel_row_targets(&targets[i * c..(i + 1) * c], &mut mask);
        let mut acc = 0.0f64;
        for &t in list {
            let xt = f64::from(row[t as usize]);
            for (d, &xd) in row.iter().enumerate() {
                if mask[d] {
                    continue;
                }
                let z = 1.0 - xt + f64::from(xd);
                if z > 0.0 {
                    acc += z;
                }
            }
        }
        total += acc / c as f64;
    }
    finalize(total, n, reduction)
}

/// MultiLabelMargin の VJP。`z > 0` の `(t, d)` 組ごとに `dx_t −= s/C`・`dx_d += s/C`
/// （重複 target 添字は重複分だけ加算される）。
pub(crate) fn multilabel_margin_loss_vjp(
    input: &Tensor<f32>,
    targets: &[i32],
    scale: f64,
) -> Tensor<f32> {
    let (n, c) = rows_cols(input.shape());
    let xs = dense_vec(input);
    let mut mask = vec![false; c];
    let mut dx = vec![0f32; n * c];
    let mut row_grad = vec![0f64; c];
    let g = scale / c as f64;
    for i in 0..n {
        let row = &xs[i * c..(i + 1) * c];
        let list = multilabel_row_targets(&targets[i * c..(i + 1) * c], &mut mask);
        row_grad.iter_mut().for_each(|v| *v = 0.0);
        for &t in list {
            let xt = f64::from(row[t as usize]);
            for (d, &xd) in row.iter().enumerate() {
                if mask[d] {
                    continue;
                }
                if 1.0 - xt + f64::from(xd) > 0.0 {
                    row_grad[t as usize] -= g;
                    row_grad[d] += g;
                }
            }
        }
        for (j, v) in row_grad.iter().enumerate() {
            dx[i * c + j] = *v as f32;
        }
    }
    build_tensor(dx, input.shape())
}

// ---------------------------------------------------------------------
// MultiLabelSoftMargin
// ---------------------------------------------------------------------

/// 要素損失 `l = t·softplus(−x) + (1−t)·softplus(x)`（`softplus(x) = softplus_neg(−x)`）。
/// `x = ±inf` と `t ∈ {0, 1}` の組で `0·inf = NaN` が出る挙動は PyTorch と同じ
/// （fixture `mlsm_inf*` で確定）。
fn soft_margin_ml_elem(x: f64, t: f64) -> f64 {
    t * softplus_neg(x) + (1.0 - t) * softplus_neg(-x)
}

/// MultiLabelSoftMargin の forward。`weight` はあれば長さ `C`。
pub(crate) fn multilabel_soft_margin_loss_forward(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    weight: Option<&[f32]>,
    reduction: Reduction,
) -> Tensor<f32> {
    let (n, c) = rows_cols(input.shape());
    let xs = dense_vec(input);
    let ts = dense_vec(target);
    let mut total = 0.0f64;
    for i in 0..n {
        let mut acc = 0.0f64;
        for j in 0..c {
            let w = weight.map_or(1.0, |w| f64::from(w[j]));
            acc += w * soft_margin_ml_elem(f64::from(xs[i * c + j]), f64::from(ts[i * c + j]));
        }
        total += acc / c as f64;
    }
    finalize(total, n, reduction)
}

/// MultiLabelSoftMargin の VJP。`dx = s·w_c·(σ(x) − t)/C`。
pub(crate) fn multilabel_soft_margin_loss_vjp(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    weight: Option<&[f32]>,
    scale: f64,
) -> Tensor<f32> {
    let (n, c) = rows_cols(input.shape());
    let xs = dense_vec(input);
    let ts = dense_vec(target);
    let mut dx = Vec::with_capacity(n * c);
    for i in 0..n {
        for j in 0..c {
            let w = weight.map_or(1.0, |w| f64::from(w[j]));
            let (x, t) = (f64::from(xs[i * c + j]), f64::from(ts[i * c + j]));
            dx.push((scale * w * (sigmoid(x) - t) / c as f64) as f32);
        }
    }
    build_tensor(dx, input.shape())
}

// ---------------------------------------------------------------------
// sigmoid focal loss
// ---------------------------------------------------------------------

/// `α_t = α·t + (1−α)(1−t)`（`alpha` 無効時は 1）。
fn focal_alpha_t(alpha: Option<f32>, t: f64) -> f64 {
    alpha.map_or(1.0, |a| {
        let a = f64::from(a);
        a * t + (1.0 - a) * (1.0 - t)
    })
}

/// `q = 1 − p_t = t·σ(−x) + (1−t)·σ(x)`（`1 − p_t` を引き算せず桁落ちを避ける形）。
fn focal_q(x: f64, t: f64) -> f64 {
    t * sigmoid(-x) + (1.0 - t) * sigmoid(x)
}

/// sigmoid focal loss の forward（`target` は `input` と同 shape）。
/// `l = α_t · ce · q^γ`（`ce = (1−t)·x + softplus(−x)`）。
pub(crate) fn sigmoid_focal_loss_forward(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    alpha: Option<f32>,
    gamma: f32,
    reduction: Reduction,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ts = dense_vec(target);
    let total: f64 = xs
        .iter()
        .zip(&ts)
        .map(|(&x, &t)| {
            let (x, t) = (f64::from(x), f64::from(t));
            let ce = (1.0 - t) * x + softplus_neg(x);
            focal_alpha_t(alpha, t) * ce * focal_q(x, t).powf(f64::from(gamma))
        })
        .sum();
    finalize(total, xs.len(), reduction)
}

/// sigmoid focal loss の VJP。
/// `dx = s·α_t·[(p − t)·q^γ + γ·ce·q^(γ−1)·(1 − 2t)·p·(1 − p)]`。
/// `γ == 0` のとき第 2 項は評価せず 0（`q^(γ−1)` が `q == 0` で発散するのを避ける）。
/// `0 < γ < 1` かつ `q == 0`（完全に飽和）の第 2 項は極限値 0 とする
/// （PyTorch は f32 評価で `∞·0 = NaN` になる差分。決定記録 §5）。
pub(crate) fn sigmoid_focal_loss_vjp(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    alpha: Option<f32>,
    gamma: f32,
    scale: f64,
) -> Tensor<f32> {
    let xs = dense_vec(input);
    let ts = dense_vec(target);
    let g = f64::from(gamma);
    let dx: Vec<f32> = xs
        .iter()
        .zip(&ts)
        .map(|(&x, &t)| {
            let (x, t) = (f64::from(x), f64::from(t));
            let p = sigmoid(x);
            let q = focal_q(x, t);
            let first = (p - t) * q.powf(g);
            let second = if g == 0.0 || (q == 0.0 && g < 1.0) {
                0.0
            } else {
                let ce = (1.0 - t) * x + softplus_neg(x);
                g * ce * q.powf(g - 1.0) * (1.0 - 2.0 * t) * p * (1.0 - p)
            };
            (scale * focal_alpha_t(alpha, t) * (first + second)) as f32
        })
        .collect();
    build_tensor(dx, input.shape())
}
