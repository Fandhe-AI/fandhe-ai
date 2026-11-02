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
//!   `s^{-2β}` は ±2^100 以下の 2 のべき乗チャンクへ分けて順に `Mul` する（f32 の非正規数を作らない）。
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
//! - **総括判定・第 2 軸（相殺・f32 underflow。[`shadow_check`]）**: 3 Op の構造はどれも引き算の形を
//!   している。構造を積んだ直後に、子テープ上の構造の実値（f32）と 1 階のホスト値（f64 起点）を要素ごとに
//!   比べ、`(|構造 − ホスト| + coeff·m)` が REQ-2 の許容の半分を超える入力を拒否する（`m` は各項の絶対値の和。
//!   `loss_class.rs::struct_check` と同じ形。tolerance 定数・閾値は変えない）。中間値の underflow・
//!   桁落ちはこの差として見える。
//! - 拒否は `AutodiffError::Backward`。`backward_create_graph` のロールバックで `child` は空へ戻り、
//!   安全な入力で再試行できる。
//!
//! 保証は判定を通った入力に限り、判定を通った入力では 1 階・`cg.grad`・二階（seed の各要素の絶対値 ≤ 1）・
//! 孫テープ再生が有限で、`cg.grad` は 1 階と bit 一致する（V3）。`g` を作る上流の Op の逆伝播、絶対値 1 を
//! 超える seed、三階以上の微分は対象外。
//!
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

/// 子テープ上の構造 `composed` の実値を 1 階のホスト値 `host` と要素ごとに突き合わせる（第 2 軸）。
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

/// 子テープ上の `Div(a, b)` の逆伝播が実際に使う偏微分係数（`eval/scalar.rs::binary_partials` の
/// `da = 1/b`・`db = -(a/b)/b` を **f32 の評価順のまま**）を、f64 の厳密値と突き合わせる検査。
/// `track_a`／`track_b` は当該オペランドが勾配追跡対象か（非追跡側の係数は使われないので検査しない）。
/// 係数単体の overflow（cotangent が 0 でも `0·inf = NaN`）と、中間商 `a/b` の underflow による
/// 係数の消失・精度劣化（二階項へ増幅される。PR #3950 codex 指摘 P1）を型付きエラーで拒否する。
/// 判定は相対のみ（絶対救済は使わない: 係数が小さくても後段で桁違いに増幅されうるため）。
fn div_partials_check(
    a: f64,
    b: f64,
    track_a: bool,
    track_b: bool,
    what: &str,
) -> Result<(), AutodiffError> {
    let (a32, b32) = (a as f32, b as f32);
    let da32 = 1.0f32 / b32;
    let db32 = -(a32 / b32) / b32;
    let da = 1.0 / b;
    let db = -(a / b) / b;
    let bad = |got: f32, exact: f64| -> bool {
        if !got.is_finite() || !exact.is_finite() {
            return true;
        }
        exact != 0.0 && (f64::from(got) - exact).abs() > (RELATIVE_TOLERANCE / 8.0) * exact.abs()
    };
    if (track_a && bad(da32, da)) || (track_b && bad(db32, db)) {
        return Err(invalid(&format!(
            "{what}: Div の偏微分係数が f32 で overflow／underflow し二階が REQ-2 の判定を外れうるため拒否"
        )));
    }
    Ok(())
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

/// `2^{L_lane}` を ±2^100 以下のチャンク列へ分ける（先頭から順に `Mul` する）。
fn chunk_factors(l: &[f64]) -> Vec<Vec<f32>> {
    let mut rem = l.to_vec();
    let mut out = Vec::new();
    for _ in 0..8 {
        if rem.iter().all(|r| r.abs() < 1e-12) {
            break;
        }
        let part: Vec<f64> = rem.iter().map(|r| r.clamp(-100.0, 100.0)).collect();
        out.push(part.iter().map(|p| 2f64.powf(*p) as f32).collect());
        for (r, p) in rem.iter_mut().zip(&part) {
            *r -= p;
        }
    }
    out
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
            // PowScalar の微分係数 `e·d^(e-1)` 単体は、実装（eval/scalar.rs）が f32 の
            // `powf` で先に評価する。cotangent との積（f64）を検査するだけでは、係数自身の
            // f32 overflow（例 d=1e-30 の e=-0.5 で -1.6e45）を取り逃がすため、同じ f32 評価順で拒否する。
            for &dc in &d {
                for e in [(-beta) as f32, (-beta - 1.0) as f32] {
                    let coef = if e == 0.0 {
                        0.0
                    } else {
                        e * (dc as f32).powf(e - 1.0)
                    };
                    stage(
                        f64::from(coef),
                        "LRN の PowScalar 微分係数 e·d^(e-1)（f32 評価）",
                    )?;
                }
            }
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
    for chunk in chunk_factors(&l_exp) {
        dx = dx.mul(&tensor_const(child, chunk, &col_shape)?)?;
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

    // 第 1 軸（overflow）と第 2 軸の m。
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
        // 構造中の Div（v_s/n_s・g/n_s・(·)/s）の偏微分係数を f32 評価順で検査する。
        for &x in &vs {
            div_partials_check(f64::from(x), n_s, true, true, "WeightNorm の v_s/n_s")?;
        }
        if !g_data[k].is_nan() {
            div_partials_check(f64::from(g_data[k]), n_s, true, true, "WeightNorm の g/n_s")?;
            div_partials_check(
                f64::from(g_data[k]) / n_s,
                s,
                true,
                false,
                "WeightNorm の (g/n_s)/s",
            )?;
        }
        div_partials_check(f64::from(vs[0]) * s, s, true, false, "WeightNorm の v/s")?;
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
    for (i, &ui) in u_data.iter().enumerate().take(h) {
        for (j, &vj) in v_data.iter().enumerate().take(cols) {
            p[(j / inner * h + i) * inner + j % inner] = ui * vj;
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
        // up への微分係数（W 側・up 側の両方）の独立検査。dW は up に線形で、係数行列は
        // σ_s = Σ W_s·P の相殺で決まる。up の値（全ゼロでも）に依らず、σ_s の丸め誤差
        // coeff·Σ|W_s·P| の相対値が dW の係数（1/σ_s と sp/σ_s の 2 重の 1/σ_s で最大 3 倍）を
        // 通して REQ-2 の許容の半分に収まらない入力は拒否する（tolerance 定数は不変）。
        let m_sigma: f64 = ws
            .iter()
            .zip(&p)
            .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
            .sum();
        let rel_sigma = rounding_coeff(w_data.len() + 4) * m_sigma / sa;
        if !rel_sigma.is_finite() || 3.0 * rel_sigma > RELATIVE_TOLERANCE / 2.0 {
            return Err(invalid(
                "SpectralNorm の σ_s の相殺で up への微分係数が REQ-2 の判定を外れうるため拒否",
            ));
        }
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
        // 構造中の Div（W/s・W_s/σ_s・t/σ_s・(·)/s）の偏微分係数を f32 評価順で検査する。
        let sp_chk: f64 = up_data
            .iter()
            .zip(&ws)
            .map(|(&x, &w)| f64::from(x) * f64::from(w) / sigma_s)
            .sum();
        div_partials_check(1.0, sf, true, false, "SpectralNorm の W/s")?;
        for (i, &w) in ws.iter().enumerate() {
            div_partials_check(f64::from(w), sigma_s, true, true, "SpectralNorm の W_s/σ_s")?;
            let t_i = f64::from(up_data[i]) - sp_chk * f64::from(p[i]);
            div_partials_check(t_i, sigma_s, true, true, "SpectralNorm の t/σ_s")?;
            div_partials_check(t_i / sigma_s, sf, true, false, "SpectralNorm の (t/σ_s)/s")?;
        }
        let sp_val: f64 = up_data
            .iter()
            .zip(&ws)
            .map(|(&x, &w)| f64::from(x) * f64::from(w) / sigma_s)
            .sum();
        for (i, mi) in m_abs.iter_mut().enumerate() {
            *mi = (f64::from(up_data[i]).abs() + sp_val.abs() * f64::from(p[i]).abs()) / (sa * sf);
        }
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
        let prod: f64 = many.iter().map(|c| f64::from(c[0]).log2()).sum();
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
}
