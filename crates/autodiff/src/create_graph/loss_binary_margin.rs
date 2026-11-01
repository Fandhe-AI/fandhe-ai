//! 損失の二値・マージン系 6 variant（`BceLoss`・`BceWithLogitsPosWeightLoss`・`SoftMarginLoss`・
//! `HingeEmbeddingLoss`・`MarginRankingLoss`・`SigmoidFocalLoss`）の `create_graph` 対応
//! （イシュー #3799・親 #3719 系・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §39、実装記録は §54）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var` メソッド
//! （`bce_loss`／`bce_with_logits_loss`／`bce_with_logits_loss_with`／`soft_margin_loss`／
//! `hinge_embedding_loss`／`margin_ranking_loss`／`sigmoid_focal_loss`）への委譲で足りるため
//! 親側に置く（forward の検証を迂回しない）。回帰系 3 variant は #3798（`loss_regression`）の
//! 担当でここには含めない。
//!
//! **方式**（`s_b` は `loss_scale_bcast` と同じく子テープ上の `g` を通すので、損失が根でなくても
//! 二階が `g` へ流れる）:
//! - `BceLoss`（V1 + V2。値・構造とも f32。1 階は融合カーネル `bce_loss_backward` 経路）:
//!   `Logits` は `dinput = s_b⊙(σ(x) − t)`・`dtarget = −s_b⊙x`。`Probabilities` は
//!   `dinput = s_b⊙(x − t)/D`（`D` は `x(1−x) < 1e-12` のレーンで定数 `1e-12`、それ以外は
//!   `x(1−x)`）・`dtarget = −s_b⊙(L(x) − L(1−x))`（`L(u)` は `ln u < −100` のレーンで定数 `−100`、
//!   それ以外は `Log(u_safe)`）。融合カーネルの f32 scale と bit 一致は主張せず REQ-2 の統一複合
//!   判定で比べる。
//! - `BceWithLogitsPosWeightLoss`・`SoftMarginLoss`（V3）: 値は 1 階のホスト関数（f64 内部）を
//!   `push_v3` で載せ、構造は `Sigmoid`・`Softplus(β=1, threshold=20)` の合成。`pos_weight`・`y` は定数。
//! - `HingeEmbeddingLoss`・`MarginRankingLoss`（V3 + V2）: 入力について区分定数。区分は親の記録値から
//!   1 階と同じ f64 比較でホストが作り `where_cond` で選ぶ（境界は Hinge が 0・MarginRanking が通す）。
//! - `SigmoidFocalLoss`（V3 + V2）: `α_t`・`target` を定数とし、
//!   `dx = s_b⊙α_t⊙[(p−t)⊙q^γ + γ⊙ce⊙q^(γ−1)⊙(1−2t)⊙p⊙(1−p)]` を構造に積む。`γ == 0` は第 2 項を、
//!   `γ == 1` は `PowScalar` を積まない（payload の定数で分岐）。
//!
//! **飽和域の曲率保持**（PR #3944 レビュー対応）: `BceWithLogitsPosWeightLoss` の `dx` と focal の
//! `ce`・`σ(±x)` は `x` の符号別に桁落ちしない形へ分解する（Sigmoid の VJP は出力値から
//! `y(1−y)` を作るため、大きい側の `σ` を経由すると f32 で二階が 0 に丸まる）。focal は
//! `γ⊙ce` 自体も段検査に含める。
//!
//! 子テープへ積む Op は `Add`・`Sub`・`Mul`・`Div`・`Neg`・`Exp`・`Log`・`Sigmoid`・`Softplus`・
//! `PowScalar`・`Where`・`BroadcastTo`・`Contiguous`・定数 `Leaf` で、すべて
//! `supports_create_graph() == true`（閉包性。§39.11）。
//!
//! **総括判定（§39.12 の具体化）**: variant ごとに構造の各段と二階逆伝播の「cotangent × 微分係数」
//! （係数は f32 で先に作られるため単独でも）を `f64` の三角不等式で上から押さえ、非有限または
//! `STAGE_BOUND_GUARD`（2^125）以上の段があれば子テープへ最初に書く前に型付きエラーで拒否する
//! （親の rollback で `child` は空に戻る）。入力の `±inf`・非有限の `g` も拒否する。NaN のレーンは
//! 段検査から外して伝播させる（§33.14 論点 3 と同じ規則）。追跡されていない入力も保守的に同じ
//! 検査へ含める。focal の飽和域（`0<γ<1` または `1<γ<2` で `q` が f32 の正規最小値未満）は
//! `q^(γ−1)`・`q^(γ−2)` が発散するため拒否する（1 階が極限値 0 を返す差は §54.2 に記録）。
//!
//! GPU: Bce は 1 階が融合カーネル経路（経路区分 K）、残る 5 つは 1 階がホスト関数（経路区分 H）。
//! 実機 parity は未実測（Phase 10 の #3857・#3859）。

use fandhe_ai_tensor_core::{BceKind, Tensor};

use super::{child_value, push_v3};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::{Reduction, Var};

fn invalid(msg: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {msg}（損失の二値・マージン系。docs/autodiff-higher-order-grad-decision.md §39・§54 参照）"
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

/// 上流 `g`（スカラー）の値から、reduction 後の scale 上界 `|g|/n`（Mean）／`|g|`（Sum）を返す。
/// 非有限・非スカラーは拒否する。
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

fn full_like<'c>(child: &'c Tape, shape: &[usize], v: f32) -> Result<Var<'c>, AutodiffError> {
    let n: usize = shape.iter().product();
    Ok(child.var_no_grad(&Tensor::new(vec![v; n], shape)?))
}

fn scalar(child: &Tape, v: f32) -> Var<'_> {
    child.var_no_grad(&Tensor::scalar(v))
}

fn const_vec<'c>(
    child: &'c Tape,
    data: Vec<f32>,
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::new(data, shape)?))
}

/// 親ノードの shape が揃い、上流が単一要素であることを確かめて要素 shape を返す。
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

/// payload の定数テンソル（`y`・`target`・`pos_weight`）が入力 shape と一致することを確かめて
/// 要素列を返す。
fn payload_vec(t: &Tensor<f32>, shape: &[usize]) -> Result<Vec<f32>, AutodiffError> {
    if t.shape() != shape {
        return Err(invalid("payload の shape 不整合（内部契約違反）"));
    }
    Ok(crate::eval::dense_vec(t))
}

fn sigmoid64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `softplus(−x)`（安定形。上界の見積り用）。
fn softplus_neg64(x: f64) -> f64 {
    (-x).max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `BceLoss` の (d_input, d_target) を子テープへ記録する。
pub(super) fn bce_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Reduction),
    kind: BceKind,
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let (input, target, reduction) = spec;
    let (x_m, t_m) = mirrors;
    let probabilities = match kind {
        BceKind::Probabilities => true,
        BceKind::Logits => false,
        _ => return Err(invalid("BceLoss: 未知の BceKind は fail-closed")),
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
    // Probabilities 用のホスト側マスク（1 階 `bce_elem_grad_*` と同じ f32 式）。
    let mut prods = Vec::with_capacity(n);
    let mut one_minus = Vec::with_capacity(n);
    for (&x32, &t32) in xs.iter().zip(&ts) {
        let (x, t) = (f64::from(x32), f64::from(t32));
        if !probabilities {
            if x.is_nan() || t.is_nan() {
                continue;
            }
            let a = 1.0 + t.abs();
            stage(x.abs())?;
            stage(sb * a)?;
            stage(sb * x.abs())?;
            gsum += a + x.abs();
            continue;
        }
        let prod = x32 * (1.0 - x32);
        let u2 = 1.0 - x32;
        prods.push(prod);
        one_minus.push(u2);
        if x.is_nan() || t.is_nan() || prod.is_nan() {
            continue;
        }
        let d = f64::from(prod).max(1e-12);
        let num = (x - t).abs();
        stage(1.0 / d)?;
        stage(num / d)?;
        stage(num / d / d)?;
        stage(sb / d)?;
        stage(sb * num / d)?;
        stage(sb * num / d / d)?;
        let mut lsum = 0f64;
        for u in [x32, u2] {
            let ln32 = u.ln();
            if ln32.is_nan() || ln32 < crate::eval::BCE_LOG_CLAMP_MIN {
                lsum += 100.0;
                continue;
            }
            let u = f64::from(u);
            stage(1.0 / u)?;
            stage(sb / u)?;
            stage(u.ln().abs())?;
            stage(sb * u.ln().abs())?;
            lsum += u.ln().abs();
        }
        gsum += num / d + lsum;
    }
    check_g_cotangent(gsum, reduction, n)?;

    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    if !probabilities {
        let dinput = s_b.mul(&x_m.sigmoid().sub(t_m)?)?;
        let dtarget = s_b.mul(x_m)?.neg()?;
        return Ok((dinput, dtarget));
    }
    // dinput = s_b⊙(x − t)/D。`D` はクランプのレーンで定数 1e-12。
    let prod_t = Tensor::new(prods, &shape)?;
    let clamp_d = super::mask_from_pred(&prod_t, |v| v < 1e-12)?;
    let one = scalar(child, 1.0);
    let one_minus_x = one.sub(x_m)?;
    let d_var = Var::where_cond(
        &clamp_d,
        &full_like(child, &shape, 1e-12)?,
        &x_m.mul(&one_minus_x)?,
    )?;
    let dinput = s_b.mul(&x_m.sub(t_m)?.div(&d_var)?)?;
    // dtarget = −s_b⊙(L(x) − L(1−x))。`L(u)` は ln u < −100 のレーンで定数 −100。
    let ln_clamped = |values: Vec<f32>, u: &Var<'c>| -> Result<Var<'c>, AutodiffError> {
        let mask = super::mask_from_pred(&Tensor::new(values, &shape)?, |v| {
            let l = v.ln();
            l.is_nan() || l < crate::eval::BCE_LOG_CLAMP_MIN
        })?;
        let u_safe = Var::where_cond(&mask, &full_like(child, &shape, 1.0)?, u)?;
        Var::where_cond(&mask, &full_like(child, &shape, -100.0)?, &u_safe.log()?)
    };
    let l_x = ln_clamped(xs.clone(), x_m)?;
    let l_1mx = ln_clamped(one_minus, &one_minus_x)?;
    let dtarget = s_b.mul(&l_x.sub(&l_1mx)?)?.neg()?;
    Ok((dinput, dtarget))
}

/// `BceWithLogitsPosWeightLoss` の (d_input, d_target) を子テープへ記録する。
pub(super) fn bce_pos_weight_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Reduction),
    pos_weight: &Tensor<f32>,
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
    let ps = payload_vec(pos_weight, &shape)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs, &ts, &ps])?;
    let mut gsum = 0f64;
    for ((&x, &t), &p) in xs.iter().zip(&ts).zip(&ps) {
        let (x, t, p) = (f64::from(x), f64::from(t), f64::from(p));
        if x.is_nan() || t.is_nan() || p.is_nan() {
            continue;
        }
        let lw = (1.0 + (p - 1.0) * t).abs();
        let pm1 = (p - 1.0).abs();
        let sp = pm1 * softplus_neg64(x);
        stage(lw)?;
        stage(pm1)?;
        stage(x.abs())?;
        stage(sp)?;
        stage(sb * lw)?;
        stage(sb * (1.0 + t.abs() + lw))?;
        stage(sb * x.abs())?;
        stage(sb * (x.abs() + sp))?;
        stage(sb * pm1)?;
        gsum += 1.0 + t.abs() + lw + x.abs() + sp;
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let (hx, ht) = crate::eval::elementwise_loss::bce_with_logits_pos_weight_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        materialize_fallible(parent_nodes, parent.ops(), target)?,
        pos_weight,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let pm1_c = const_vec(child, ps.iter().map(|p| p - 1.0).collect(), &shape)?;
    let one = scalar(child, 1.0);
    let nx = x_m.neg()?;
    let lw = one.add(&pm1_c.mul(t_m)?)?;
    // dx = s_b⊙[(1−t) − lw⊙σ(−x)] を符号別の安定形へ分解する（二階の曲率を飽和域で保つ）。
    // x ≥ 0: そのまま（σ(−x) ≤ 0.5 で桁落ちなし）。
    // x < 0: (1−t) − lw = −pw⊙t なので lw⊙σ(x) − pw⊙t（σ(x) の VJP が y(1−y) で精度を保つ）。
    let pw_c = const_vec(child, ps.clone(), &shape)?;
    let nonneg = Tensor::new(xs.iter().map(|&x| x >= 0.0).collect::<Vec<bool>>(), &shape)?;
    let form_pos = one.sub(t_m)?.sub(&lw.mul(&nx.sigmoid())?)?;
    let form_neg = lw.mul(&x_m.sigmoid())?.sub(&pw_c.mul(t_m)?)?;
    let dx = s_b.mul(&Var::where_cond(&nonneg, &form_pos, &form_neg)?)?;
    let dt = s_b.mul(&nx.add(&pm1_c.mul(&nx.softplus(1.0, 20.0)?)?)?)?;
    Ok((push_v3(child, &dx, hx)?, push_v3(child, &dt, ht)?))
}

/// `SoftMarginLoss` の d_input を子テープへ記録する。
pub(super) fn soft_margin_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, Reduction),
    y: &Tensor<f32>,
    x_m: &Var<'c>,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (input, reduction) = spec;
    let shape = checked_shape(parent_nodes, &[input], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    let n: usize = shape.iter().product();
    let xs = dense(parent, parent_nodes, input)?;
    let ys = payload_vec(y, &shape)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs, &ys])?;
    let mut gsum = 0f64;
    for (&x, &yv) in xs.iter().zip(&ys) {
        let (x, yv) = (f64::from(x), f64::from(yv));
        if x.is_nan() || yv.is_nan() {
            continue;
        }
        stage((yv * x).abs())?;
        stage(sb * yv.abs())?;
        gsum += yv.abs() * 2.0;
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let hx = crate::eval::elementwise_loss::soft_margin_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        y,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let ny = const_vec(child, ys.iter().map(|v| -v).collect(), &shape)?;
    let dx = s_b.mul(&ny)?.mul(&ny.mul(x_m)?.sigmoid())?;
    push_v3(child, &dx, hx)
}

/// `HingeEmbeddingLoss` の d_input を子テープへ記録する。
pub(super) fn hinge_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, Reduction),
    hinge: (&Tensor<f32>, f32),
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (input, reduction) = spec;
    let (y, margin) = hinge;
    let shape = checked_shape(parent_nodes, &[input], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    let n: usize = shape.iter().product();
    let xs = dense(parent, parent_nodes, input)?;
    let ys = payload_vec(y, &shape)?;
    let g_value = child_value(child, g)?;
    upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs])?;
    check_g_cotangent(1.0, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let hx = crate::eval::elementwise_loss::hinge_embedding_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        y,
        margin,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    // 区分は 1 階と同じ f64 比較（`y == 1`、`margin − x > 0`。境界 `x == margin` は 0）。
    let pos = Tensor::new(ys.iter().map(|&v| v == 1.0).collect::<Vec<bool>>(), &shape)?;
    let active = Tensor::new(
        xs.iter()
            .map(|&x| f64::from(margin) - f64::from(x) > 0.0)
            .collect::<Vec<bool>>(),
        &shape,
    )?;
    let inner = Var::where_cond(&active, &s_b.neg()?, &zeros_like(child, &shape)?)?;
    let dx = Var::where_cond(&pos, &s_b, &inner)?;
    push_v3(child, &dx, hx)
}

/// `MarginRankingLoss` の (d_x1, d_x2) を子テープへ記録する。
pub(super) fn margin_ranking_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Reduction),
    ranking: (&Tensor<f32>, f32),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let (x1, x2, reduction) = spec;
    let (y, margin) = ranking;
    let shape = checked_shape(parent_nodes, &[x1, x2], g)?;
    if shape.contains(&0) {
        return Ok((zeros_like(child, &shape)?, zeros_like(child, &shape)?));
    }
    let n: usize = shape.iter().product();
    let x1s = dense(parent, parent_nodes, x1)?;
    let x2s = dense(parent, parent_nodes, x2)?;
    let ys = payload_vec(y, &shape)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&x1s, &x2s])?;
    let mut gsum = 0f64;
    for &yv in &ys {
        let yv = f64::from(yv);
        if yv.is_nan() {
            continue;
        }
        stage(sb * yv.abs())?;
        gsum += 2.0 * yv.abs();
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let (h1, h2) = crate::grad::margin_ranking_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), x1)?,
        materialize_fallible(parent_nodes, parent.ops(), x2)?,
        y,
        margin,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    // active = (−y·(x1 − x2) + margin ≥ 0)（f64。境界は通す＝`clamp_min` の契約）。
    let active = Tensor::new(
        x1s.iter()
            .zip(&x2s)
            .zip(&ys)
            .map(|((&a, &b), &yv)| {
                -f64::from(yv) * (f64::from(a) - f64::from(b)) + f64::from(margin) >= 0.0
            })
            .collect::<Vec<bool>>(),
        &shape,
    )?;
    let y_c = const_vec(child, ys, &shape)?;
    let sy = s_b.mul(&y_c)?;
    let zeros = zeros_like(child, &shape)?;
    let dx1 = Var::where_cond(&active, &sy.neg()?, &zeros)?;
    let dx2 = Var::where_cond(&active, &sy, &zeros)?;
    Ok((push_v3(child, &dx1, h1)?, push_v3(child, &dx2, h2)?))
}

/// `SigmoidFocalLoss` の d_input を子テープへ記録する。
pub(super) fn focal_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, Reduction),
    focal: (&Tensor<f32>, Option<f32>, f32),
    x_m: &Var<'c>,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (input, reduction) = spec;
    let (target, alpha, gamma) = focal;
    let shape = checked_shape(parent_nodes, &[input], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    let n: usize = shape.iter().product();
    let xs = dense(parent, parent_nodes, input)?;
    let ts = payload_vec(target, &shape)?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    reject_inf(&[&xs, &ts])?;
    let g64 = f64::from(gamma);
    // 飽和域（f32 の q が 0／非正規数）で q^(γ−1)・q^(γ−2) が発散する γ の範囲。
    let diverging_gamma = (g64 > 0.0 && g64 < 1.0) || (g64 > 1.0 && g64 < 2.0);
    let mut gsum = 0f64;
    let mut alpha_t = Vec::with_capacity(n);
    for (&x, &t) in xs.iter().zip(&ts) {
        let (x, t) = (f64::from(x), f64::from(t));
        let at = alpha.map_or(1.0, |a| {
            let a = f64::from(a);
            a * t + (1.0 - a) * (1.0 - t)
        });
        alpha_t.push(at as f32);
        if x.is_nan() || t.is_nan() {
            continue;
        }
        let p = sigmoid64(x);
        let q = t * sigmoid64(-x) + (1.0 - t) * p;
        if diverging_gamma && q < f64::from(f32::MIN_POSITIVE) {
            return Err(invalid(
                "SigmoidFocal: q が f32 の正規最小値未満の飽和域（q^(γ−1) が発散）のため拒否",
            ));
        }
        let ce = ((1.0 - t) * x).abs() + softplus_neg64(x);
        let qg = if g64 == 0.0 { 1.0 } else { q.powf(g64) };
        let (qg1, c1, c2) = if g64 == 0.0 {
            (0.0, 0.0, 0.0)
        } else if g64 == 1.0 {
            (1.0, 1.0, 0.0)
        } else {
            (
                q.powf(g64 - 1.0),
                g64 * q.powf(g64 - 1.0),
                (g64 - 1.0).abs() * q.powf(g64 - 2.0),
            )
        };
        let sba = sb * at.abs();
        let pt = (p - t).abs();
        let curv = (1.0 - 2.0 * t).abs() * p * (1.0 - p);
        stage(ce)?;
        stage(qg)?;
        stage(qg1)?;
        stage(c1)?;
        stage(c2)?;
        stage(sba)?;
        stage(sba * pt)?;
        stage(sba * pt * c1)?;
        stage(g64 * ce)?;
        stage(g64 * ce * qg1)?;
        stage(sba * g64 * ce * qg1)?;
        stage(sba * g64 * ce * curv)?;
        stage(sba * g64 * ce * curv * c2)?;
        stage(sba * g64 * qg1 * curv)?;
        gsum += at.abs() * (pt * qg + g64 * ce * qg1 * curv);
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let hx = crate::eval::margin_focal_loss::sigmoid_focal_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        target,
        alpha,
        gamma,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let t_c = const_vec(child, ts.clone(), &shape)?;
    let a_c = const_vec(child, alpha_t, &shape)?;
    let one = scalar(child, 1.0);
    let nx = x_m.neg()?;
    // σ(x)・σ(−x) は「小さい側」を直接 Sigmoid で作り、大きい側を `1 − 小` で作る。大きい側を
    // Sigmoid の出力から作ると VJP の y(1−y) が f32 の丸めで曲率を失う（x ≪ 0 の σ(−x) 等）。
    let nonneg = Tensor::new(xs.iter().map(|&x| x >= 0.0).collect::<Vec<bool>>(), &shape)?;
    let small = Var::where_cond(&nonneg, &nx.sigmoid(), &x_m.sigmoid())?;
    let big = one.sub(&small)?;
    let p = Var::where_cond(&nonneg, &big, &small)?;
    let sn = Var::where_cond(&nonneg, &small, &big)?;
    let q = t_c.mul(&sn)?.add(&one.sub(&t_c)?.mul(&p)?)?;
    // p − t = (1−t)⊙p − t⊙σ(−x)（t が 0／1 近傍の飽和域で桁落ちしない形）。
    let p_minus_t = one.sub(&t_c)?.mul(&p)?.sub(&t_c.mul(&sn)?)?;
    let first = if g64 == 0.0 {
        p_minus_t
    } else {
        p_minus_t.mul(&q.pow_scalar(gamma)?)?
    };
    let bracket = if g64 == 0.0 {
        first
    } else {
        // ce = (1−t)x + softplus(−x) を符号別に分解（x ≥ 0: そのまま、x < 0: softplus(x) − t⊙x）。
        // 単一式だと t = 0・x ≪ 0 で x + softplus(−x) が f32 で 0 に丸まり ce と微分を失う。
        let ce_pos = one.sub(&t_c)?.mul(x_m)?.add(&nx.softplus(1.0, 20.0)?)?;
        let ce_neg = x_m.softplus(1.0, 20.0)?.sub(&t_c.mul(x_m)?)?;
        let ce = Var::where_cond(&nonneg, &ce_pos, &ce_neg)?;
        let one_m_2t = one.sub(&t_c.mul(&scalar(child, 2.0))?)?;
        let mut second = scalar(child, gamma).mul(&ce)?;
        if g64 != 1.0 {
            second = second.mul(&q.pow_scalar(gamma - 1.0)?)?;
        }
        let second = second.mul(&one_m_2t)?.mul(&p)?.mul(&sn)?;
        first.add(&second)?
    };
    let dx = s_b.mul(&a_c)?.mul(&bracket)?;
    push_v3(child, &dx, hx)
}
