//! 統計縮約 3 Op（`LogSumExp`・`Nansum`・`Nanmean`）の `create_graph` 対応
//! （イシュー #3792・親 #3719 系・追跡表 #3717 項目 3。設計
//! `docs/autodiff-higher-order-grad-decision.md` §33、実装記録は §51）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var::logsumexp`／`nansum`／`nanmean` への
//! 1 行委譲で足りるため親側に置く（`Nansum` と `Nanmean` は Op タグを混同しない）。
//! `Var`／`Std`／`VectorNorm`／`PNorm` は別イシュー（#3791）の担当でここには含めない。
//!
//! **方式**（値は 1 階 `grad.rs` の VJP と同じホスト関数の結果を `push_v3` で載せる。
//! 契約は REQ-2 のまま）:
//! - `LogSumExp`（V3 + 非有限レーンは V2）: 構造は `g_b ⊙ softmax(x_safe)`。`x_safe` は
//!   `±inf` レーン（`grad::logsumexp_lane_masks`）だけを 0 に差し替える。`+inf` レーンの
//!   `+inf` 位置は `where_cond` で `g_b / inf_count`、それ以外の 0 位置は `where_cond` で 0
//!   （`inf * 0` を作らないため乗算マスクは使わない）。NaN レーンは守らず伝播させる。
//! - `Nansum`／`Nanmean`（V3 + 構造は V2）: 構造は `where_cond(isnan(x), 0, g_b [/ count])`。
//!   マスクと個数は親の記録値 `x` から作る定数で、x による二階は 0。`count == 0`（全 NaN）の
//!   レーンの個数は 1 に守る（構造は全 0 で有限・値はホスト関数の NaN で 1 階と同じ）。
//!
//! 子テープへ積む Op は `Reshape`・`BroadcastTo`・`Contiguous`・`Where`・`Mul`・
//! `Div`・`Softmax`・定数 `Leaf` で、すべて `supports_create_graph() == true`（閉包性。§33.11）。
//!
//! **総括判定（§33.12 の具体化）**: 有限の `g` なら各段は `|g|`（softmax は [0,1]・マスクは 0/1・
//! `1/count`・`1/inf_count` は 1 以下）で押さえられるため、`check_upstream_bound` の
//! 「`g` は有限かつ `max|g|·max(1, axis_len)` が `STAGE_BOUND_GUARD` 未満」だけを
//! 子テープへ書く前に課す。外れる入力は型付きエラーで拒否し、親の rollback で `child` は空に戻る。
//!
//! GPU: `logsumexp`／`stat_nansum`／`stat_nanmean` は CPU のみ実装で、CUDA／Metal では
//! ホストへフォールバックする（経路区分 `H`）。実機 parity は未実測（Phase 10 の #3856・#3858）。

use fandhe_ai_tensor_core::stat_reduce::{nan_reduce_vjp_host, stat_layout};
use fandhe_ai_tensor_core::{Tensor, reduce_out_shape};

use super::select_reduce::bcast_upstream;
use super::{child_value, push_v3};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::grad::{logsumexp_lane_masks, logsumexp_vjp};
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 縮約レーンの (outer, axis_len, inner)。`dim = None` は全要素を 1 レーンとする。
fn lane_dims(shape: &[usize], dim: Option<usize>, numel: usize) -> (usize, usize, usize) {
    match dim {
        None => (1, numel, 1),
        Some(axis) => (
            shape[..axis].iter().product(),
            shape[axis],
            shape[axis + 1..].iter().product(),
        ),
    }
}

/// 上流 `g` の総括判定。`g` に非有限要素がある、または `max|g|·max(1, axis_len)` が
/// `STAGE_BOUND_GUARD` 以上なら、子テープへ書く前に拒否する（§51.2）。
fn check_upstream_bound(g_value: &Tensor<f32>, axis_len: usize) -> Result<(), AutodiffError> {
    let data = crate::eval::dense_vec(g_value);
    let mut max_abs = 0f64;
    for &v in &data {
        if !v.is_finite() {
            return Err(invalid(
                "統計縮約（LogSumExp/Nansum/Nanmean）の上流勾配が非有限のため拒否（§33・§51 参照）"
                    .into(),
            ));
        }
        max_abs = max_abs.max(f64::from(v).abs());
    }
    if max_abs * (axis_len.max(1) as f64) >= STAGE_BOUND_GUARD {
        return Err(invalid(
            "統計縮約（LogSumExp/Nansum/Nanmean）の上流勾配が大きすぎ中間値が f32 範囲を超えうるため拒否（§33・§51 参照）"
                .into(),
        ));
    }
    Ok(())
}

/// 縮約ノード `id` の shape 整合を確かめ、入力 shape を返す。
fn checked_shapes(
    parent_nodes: &[TapeNode],
    id: NodeId,
    input: NodeId,
    dim: Option<usize>,
    g: &Var<'_>,
) -> Result<Vec<usize>, AutodiffError> {
    let in_shape = parent_nodes[input.0].shape.clone();
    let expect = reduce_out_shape(&in_shape, dim).map_err(AutodiffError::Shape)?;
    if parent_nodes[id.0].shape != expect || g.shape() != expect {
        return Err(invalid(format!(
            "統計縮約の shape 不整合（入力 {in_shape:?}・dim {dim:?}・期待 {expect:?}・記録値 {:?}・上流 {:?}。内部契約違反）",
            parent_nodes[id.0].shape,
            g.shape()
        )));
    }
    Ok(in_shape)
}

/// `LogSumExp` の d_input を子テープへ記録する（`x_m` は入力の子テープ上の写し）。
pub(super) fn logsumexp_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Option<usize>),
    x_m: &Var<'c>,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (id, input, dim) = spec;
    let in_shape = checked_shapes(parent_nodes, id, input, dim, g)?;
    let zeros = child.var_no_grad(&Tensor::zeros(&in_shape)?);
    if in_shape.contains(&0) {
        return Ok(zeros);
    }
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let numel: usize = in_shape.iter().product();
    let (_, axis_len, _) = lane_dims(&in_shape, dim, numel);
    let g_value = child_value(child, g)?;
    check_upstream_bound(&g_value, axis_len)?;

    let masks = logsumexp_lane_masks(input_val, dim);
    let guard = Tensor::new(masks.guard, &in_shape)?;
    let posinf = Tensor::new(masks.posinf_elem, &in_shape)?;
    let zero_elem = Tensor::new(masks.zero_elem, &in_shape)?;
    let cnt = child.var_no_grad(&Tensor::new(masks.inf_count, &in_shape)?);

    let g_b = bcast_upstream(g, &in_shape, dim)?;
    let x_safe = Var::where_cond(&guard, &zeros, x_m)?;
    let s = match dim {
        Some(d) => x_safe.softmax(d)?,
        None => x_safe.reshape(&[numel])?.softmax(0)?.reshape(&in_shape)?,
    };
    let finite = g_b.mul(&s)?;
    let zeroed = Var::where_cond(&zero_elem, &zeros, &finite)?;
    let composed = Var::where_cond(&posinf, &g_b.div(&cnt)?, &zeroed)?;
    let host = logsumexp_vjp(input_val, dim, &g_value);
    push_v3(child, &composed, host)
}

/// `Nansum`（`mean = false`）／`Nanmean`（`mean = true`）の d_input を子テープへ記録する。
pub(super) fn nan_reduce_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Option<usize>),
    mean: bool,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (id, input, dim) = spec;
    let in_shape = checked_shapes(parent_nodes, id, input, dim, g)?;
    let zeros = child.var_no_grad(&Tensor::zeros(&in_shape)?);
    if in_shape.contains(&0) {
        return Ok(zeros);
    }
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let numel: usize = in_shape.iter().product();
    let (outer, axis_len, inner) = lane_dims(&in_shape, dim, numel);
    let g_value = child_value(child, g)?;
    check_upstream_bound(&g_value, axis_len)?;

    let x = crate::eval::dense_vec(input_val);
    let nan_mask: Vec<bool> = x.iter().map(|v| v.is_nan()).collect();
    let g_b = bcast_upstream(g, &in_shape, dim)?;
    let scaled = if mean {
        let mut count = vec![1f32; x.len()];
        for o in 0..outer {
            for i in 0..inner {
                let c = (0..axis_len)
                    .filter(|&a| !nan_mask[(o * axis_len + a) * inner + i])
                    .count();
                // count == 0（全 NaN レーン）は 1 に守る（構造は全 0 で有限。値はホスト関数）。
                let c = c.max(1) as f32;
                for a in 0..axis_len {
                    count[(o * axis_len + a) * inner + i] = c;
                }
            }
        }
        g_b.div(&child.var_no_grad(&Tensor::new(count, &in_shape)?))?
    } else {
        g_b
    };
    let mask = Tensor::new(nan_mask, &in_shape)?;
    let composed = Var::where_cond(&mask, &zeros, &scaled)?;
    let layout = stat_layout(&in_shape, dim)?;
    let host = nan_reduce_vjp_host(&x, &crate::eval::dense_vec(&g_value), &layout, mean)?;
    push_v3(child, &composed, Tensor::new(host, &in_shape)?)
}
