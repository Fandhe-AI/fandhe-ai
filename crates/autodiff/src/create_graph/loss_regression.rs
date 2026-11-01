//! 損失の回帰系 3 variant（`KlDivLoss`・`PoissonNllLoss`・`GaussianNllLoss`）の
//! `create_graph` 対応（イシュー #3798・親 #3719 系・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §39、実装記録は §53）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var::kl_div_loss`／
//! `kl_div_loss_with_log_target`／`poisson_nll_loss`／`gaussian_nll_loss` への委譲で足りるため
//! 親側に置く（forward の検証を迂回しない）。二値・マージン系の 6 variant は #3799 の担当で
//! ここには含めない。
//!
//! **方式**（`scale` は `loss_scale_bcast` と同じく子テープ上の `g` を通すので、損失が根でなくても
//! 二階が `g` へ流れる）:
//! - `KlDivLoss`（V1 + V2。値・構造とも f32。1 階は融合カーネル `kl_div_loss_backward` 経路）:
//!   `Probabilities` は `dinput = −(s⊙t)`・`dtarget = where(t≠0, s⊙((ln t_safe + 1) − x), 0)`、
//!   `LogProbabilities` は `dinput = −(s⊙e^t)`・`dtarget = s⊙(e^t⊙((t−x)+1))`。
//!   `t_safe` は `t == 0` のレーンだけ 1 に置き換えて `Log(0)` を作らない。
//!   融合カーネルの f32 scale と bit 一致は主張せず REQ-2 の統一複合判定で比べる。
//! - `PoissonNllLoss`（V3 + V2）: 値は 1 階のホスト関数 `poisson_nll_loss_vjp`（f64 内部）を
//!   `push_v3` で載せる。構造は `log_input`（payload の定数）で分岐し、`full` の Stirling 項は
//!   `t > 1` マスクの `where_cond`（V2）で選ぶ。
//! - `GaussianNllLoss`（V3）: 値は `gaussian_nll_loss_vjp`。構造は `c = max(var, eps)` を
//!   `(var − const(var)) + const(c)` の素通し形で表す（値はちょうど `c`・`var` についての微分は 1）。
//!   1 階が `var < eps` でも `dvar` を流すこと（PyTorch fixture で確定済み）と食い違わせないため
//!   `where_cond` の clamp は使わない。
//!
//! 子テープへ積む Op は `Add`・`Sub`・`Mul`・`Div`・`Neg`・`Log`・`Exp`・`Where`・`BroadcastTo`・
//! `Contiguous`・定数 `Leaf` で、すべて `supports_create_graph() == true`（閉包性。§39.11）。
//!
//! **総括判定（§39.12 の具体化）**: variant ごとに構造の各段と二階逆伝播の「cotangent × 微分係数」を
//! `f64` の三角不等式で上から押さえ、非有限または `STAGE_BOUND_GUARD`（2^125）以上の段があれば
//! 子テープへ最初に書く前に型付きエラーで拒否する（親の rollback で `child` は空に戻る）。
//! 入力の `±inf`・非有限の `g`・`x + eps <= 0`（Poisson 非 log）・`c == 0`（Gaussian）も拒否する。
//! NaN のレーンは段検査から外して伝播させる（§33.14 論点 3 と同じ規則）。追跡されていない入力も
//! 保守的に同じ検査へ含める。
//!
//! GPU: KlDiv は 1 階が融合カーネル経路（経路区分 K）、Poisson・Gaussian は 1 階がホスト関数
//! （経路区分 H）。実機 parity は未実測（Phase 10 の #3857・#3859）。

use fandhe_ai_tensor_core::{KlDivTarget, Tensor};

use super::{child_value, push_v3};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::{Reduction, Var};

fn invalid(msg: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {msg}（損失の回帰系。docs/autodiff-higher-order-grad-decision.md §39・§53 参照）"
    ))
}

/// 1 段の上界 `v`（f64）を検査する。NaN のレーンは伝播させるため素通し、非有限または
/// `STAGE_BOUND_GUARD` 以上は拒否する。
fn stage(v: f64) -> Result<(), AutodiffError> {
    if v.is_nan() {
        return Ok(());
    }
    if !v.is_finite() || v.abs() >= STAGE_BOUND_GUARD {
        return Err(invalid("構造の中間値が f32 範囲を超えうる入力のため拒否"));
    }
    Ok(())
}

/// 上流 `g`（スカラ）の値と、reduction 後の scale 上界 `|g|/n`（Mean）／`|g|`（Sum）を返す。
fn upstream_scale(
    g_value: &Tensor<f32>,
    reduction: Reduction,
    n: usize,
) -> Result<f64, AutodiffError> {
    let data = crate::eval::dense_vec(g_value);
    let Some(&gv) = data.first() else {
        return Err(invalid("上流勾配が空"));
    };
    if data.len() != 1 || !gv.is_finite() {
        return Err(invalid("上流勾配が非有限（または非スカラ）のため拒否"));
    }
    let gv = f64::from(gv).abs();
    stage(gv)?;
    Ok(match reduction {
        Reduction::Mean => gv / n as f64,
        Reduction::Sum => gv,
    })
}

/// `g` への cotangent 上界（要素寄与の絶対値和 × 要素あたりの係数 `1/n`／`1`）を検査する。
fn check_g_cotangent(sum_abs: f64, reduction: Reduction, n: usize) -> Result<(), AutodiffError> {
    let v = match reduction {
        Reduction::Mean => sum_abs / n as f64,
        Reduction::Sum => sum_abs,
    };
    stage(v)
}

fn dense(parent: &Tape, parent_nodes: &[TapeNode], id: NodeId) -> Result<Vec<f32>, AutodiffError> {
    Ok(crate::eval::dense_vec(materialize_fallible(
        parent_nodes,
        parent.ops(),
        id,
    )?))
}

fn reject_inf(values: &[&[f32]]) -> Result<(), AutodiffError> {
    if values.iter().any(|v| v.iter().any(|x| x.is_infinite())) {
        return Err(invalid("入力に ±inf を含むため拒否"));
    }
    Ok(())
}

fn zeros_like<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::zeros(shape)?))
}

fn ones_like<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    let n: usize = shape.iter().product();
    Ok(child.var_no_grad(&Tensor::new(vec![1f32; n], shape)?))
}

fn scalar(child: &Tape, v: f32) -> Var<'_> {
    child.var_no_grad(&Tensor::scalar(v))
}

/// 親ノードの shape が 2 入力で一致し、上流が単一要素であることを確かめて要素 shape を返す。
fn checked_shape(
    parent_nodes: &[TapeNode],
    ids: &[NodeId],
    g: &Var<'_>,
) -> Result<Vec<usize>, AutodiffError> {
    let shape = parent_nodes[ids[0].0].shape.clone();
    if ids.iter().any(|id| parent_nodes[id.0].shape != shape)
        || g.shape().iter().product::<usize>() != 1
    {
        return Err(invalid("shape 不整合（内部契約違反）"));
    }
    Ok(shape)
}

/// `KlDivLoss` の (d_input, d_target) を子テープへ記録する。
pub(super) fn kl_div_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Reduction),
    kind: KlDivTarget,
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let (input, target, reduction) = spec;
    let (x_m, t_m) = mirrors;
    let probabilities = match kind {
        KlDivTarget::Probabilities => true,
        KlDivTarget::LogProbabilities => false,
        _ => return Err(invalid("KlDivLoss: 未知の KlDivTarget は fail-closed")),
    };
    let shape = checked_shape(parent_nodes, &[input, target], g)?;
    if shape.contains(&0) {
        return Ok((zeros_like(child, &shape)?, zeros_like(child, &shape)?));
    }
    let n: usize = shape.iter().product();
    let xs = dense(parent, parent_nodes, input)?;
    let ts = dense(parent, parent_nodes, target)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs, &ts])?;
    let mut gsum = 0f64;
    for (&x, &t) in xs.iter().zip(&ts) {
        let (x, t) = (f64::from(x), f64::from(t));
        if x.is_nan() || t.is_nan() {
            continue;
        }
        if probabilities {
            let (a, inv) = if t == 0.0 {
                (1.0 + x.abs(), 1.0)
            } else {
                ((t.abs().ln()).abs() + 1.0 + x.abs(), 1.0 / t.abs())
            };
            stage(sb * t.abs())?;
            stage(sb * a)?;
            stage(sb * inv)?;
            gsum += t.abs() + if t == 0.0 { 0.0 } else { a };
        } else {
            let e = t.exp();
            let a = (t - x).abs() + 1.0;
            stage(e)?;
            stage(sb * e)?;
            stage(sb * e * a)?;
            gsum += e * (1.0 + a);
        }
    }
    check_g_cotangent(gsum, reduction, n)?;

    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let (dinput, dtarget) = if probabilities {
        let nz = super::mask_from_pred(&Tensor::new(ts.clone(), &shape)?, |t| t != 0.0)?;
        let zeros = zeros_like(child, &shape)?;
        let t_safe = Var::where_cond(&nz, t_m, &ones_like(child, &shape)?)?;
        let inner = t_safe.log()?.add(&scalar(child, 1.0))?.sub(x_m)?;
        let dt = Var::where_cond(&nz, &s_b.mul(&inner)?, &zeros)?;
        (s_b.mul(t_m)?.neg()?, dt)
    } else {
        let e = t_m.exp();
        let inner = t_m.sub(x_m)?.add(&scalar(child, 1.0))?;
        (s_b.mul(&e)?.neg()?, s_b.mul(&e.mul(&inner)?)?)
    };
    Ok((dinput, dtarget))
}

/// `PoissonNllLoss` の (d_input, d_target) を子テープへ記録する。
pub(super) fn poisson_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Reduction),
    options: &crate::loss_ops::PoissonNllOptions,
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let (input, target, reduction) = spec;
    let (x_m, t_m) = mirrors;
    let shape = checked_shape(parent_nodes, &[input, target], g)?;
    if shape.contains(&0) {
        return Ok((zeros_like(child, &shape)?, zeros_like(child, &shape)?));
    }
    let n: usize = shape.iter().product();
    let xs = dense(parent, parent_nodes, input)?;
    let ts = dense(parent, parent_nodes, target)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs, &ts])?;
    let (log_input, full) = (options.log_input_value(), options.full_value());
    let eps = f64::from(options.eps_value());
    let mut gsum = 0f64;
    for (&x, &t) in xs.iter().zip(&ts) {
        let (x, t) = (f64::from(x), f64::from(t));
        if x.is_nan() || t.is_nan() {
            continue;
        }
        let mut dt_extra = 0.0;
        if full && t > 1.0 {
            dt_extra = t.ln().abs() + 0.5 / t;
            stage(sb * dt_extra)?;
        }
        if log_input {
            let e = x.exp();
            stage(e)?;
            stage(sb * e)?;
            stage(sb * x.abs())?;
            stage(sb * t.abs())?;
            gsum += e + t.abs() + x.abs() + dt_extra;
        } else {
            let u = x + eps;
            if u <= 0.0 {
                return Err(invalid(
                    "PoissonNll(log_input=false): x + eps <= 0 のため拒否",
                ));
            }
            stage(1.0 / u)?;
            stage(sb * t.abs() / u)?;
            stage(sb * t.abs() / u / u)?;
            stage(sb * u.ln().abs())?;
            gsum += 1.0 + t.abs() / u + u.ln().abs() + dt_extra;
        }
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let (hx, ht) = crate::grad::poisson_nll_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        materialize_fallible(parent_nodes, parent.ops(), target)?,
        options,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let (dx, mut dt) = if log_input {
        (s_b.mul(&x_m.exp().sub(t_m)?)?, s_b.mul(x_m)?.neg()?)
    } else {
        let u = x_m.add(&scalar(child, options.eps_value()))?;
        (
            s_b.mul(&scalar(child, 1.0).sub(&t_m.div(&u)?)?)?,
            s_b.mul(&u.log()?)?.neg()?,
        )
    };
    if full {
        let big = super::mask_from_pred(&Tensor::new(ts.clone(), &shape)?, |t| t > 1.0)?;
        let t_safe = Var::where_cond(
            &big,
            t_m,
            &ones_like(child, &shape)?.mul(&scalar(child, 2.0))?,
        )?;
        let stirling = t_safe.log()?.add(&scalar(child, 0.5).div(&t_safe)?)?;
        let extra = Var::where_cond(&big, &s_b.mul(&stirling)?, &zeros_like(child, &shape)?)?;
        dt = dt.add(&extra)?;
    }
    let dx_v = push_v3(child, &dx, hx)?;
    let dt_v = push_v3(child, &dt, ht)?;
    Ok((dx_v, dt_v))
}

/// `GaussianNllLoss` の (d_input, d_target, d_var) を子テープへ記録する。
pub(super) fn gaussian_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, NodeId, Reduction),
    eps: f32,
    mirrors: (&Var<'c>, &Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>, Var<'c>), AutodiffError> {
    let (input, target, var, reduction) = spec;
    let (x_m, t_m, v_m) = mirrors;
    let shape = checked_shape(parent_nodes, &[input, target, var], g)?;
    if shape.contains(&0) {
        return Ok((
            zeros_like(child, &shape)?,
            zeros_like(child, &shape)?,
            zeros_like(child, &shape)?,
        ));
    }
    let n: usize = shape.iter().product();
    let xs = dense(parent, parent_nodes, input)?;
    let ts = dense(parent, parent_nodes, target)?;
    let vs = dense(parent, parent_nodes, var)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs, &ts, &vs])?;
    let mut c_data = Vec::with_capacity(n);
    let mut gsum = 0f64;
    for ((&x, &t), &v) in xs.iter().zip(&ts).zip(&vs) {
        let c32 = if v.is_nan() || eps.is_nan() {
            f32::NAN
        } else {
            v.max(eps)
        };
        c_data.push(c32);
        let (x, t, c) = (f64::from(x), f64::from(t), f64::from(c32));
        if x.is_nan() || t.is_nan() || c.is_nan() {
            continue;
        }
        if c == 0.0 {
            return Err(invalid("GaussianNll: max(var, eps) == 0 のため拒否"));
        }
        let d = x - t;
        let q = d / c;
        stage(d.abs())?;
        stage(q.abs())?;
        stage(q * q)?;
        stage(1.0 / c)?;
        stage(sb * q.abs())?;
        stage(sb / c)?;
        stage(sb * q * q)?;
        stage((sb + sb * q.abs()) * q.abs() / c)?;
        stage(0.5 * sb / c / c)?;
        stage(sb * q.abs() / c)?;
        gsum += q.abs() + 0.5 * (1.0 / c + q * q);
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let (hx, ht, hv) = crate::eval::elementwise_loss::gaussian_nll_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        materialize_fallible(parent_nodes, parent.ops(), target)?,
        materialize_fallible(parent_nodes, parent.ops(), var)?,
        eps,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let var_const = child.var_no_grad(&Tensor::new(vs, &shape)?);
    let c_const = child.var_no_grad(&Tensor::new(c_data, &shape)?);
    // 素通し形: 値はちょうど c・var についての微分は 1（1 階の clamp 区間の dvar と整合）。
    let c_m = v_m.sub(&var_const)?.add(&c_const)?;
    let q = x_m.sub(t_m)?.div(&c_m)?;
    let dx = s_b.mul(&q)?;
    let dt = dx.neg()?;
    let inv_c = ones_like(child, &shape)?.div(&c_m)?;
    let dv = s_b
        .mul(&scalar(child, 0.5))?
        .mul(&inv_c.sub(&q.mul(&q)?)?)?;
    Ok((
        push_v3(child, &dx, hx)?,
        push_v3(child, &dt, ht)?,
        push_v3(child, &dv, hv)?,
    ))
}
