//! 構造変換 3 Op（`Dropout`・`Interpolate`・`MaxUnpool`）の `create_graph` 対応
//! （イシュー #3793・親 #3726・決定記録 `docs/autodiff-higher-order-grad-decision.md`
//! §35・§56）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）と
//! `build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。3 Op はどれも入力 x について
//! 線形（`y = A·x`）なので cgrad は `Aᵀ g` で g だけに依存し、x についての二階の項は
//! 恒等的に 0 になる（二階は下流の非線形からだけ流れる）。payload の追加は要らない。
//!
//! 子テープへ積む Op は `Reshape`・`Contiguous`・`BroadcastTo`・`Mul`・`Where`・`Gather`・
//! `Scatter(Add)` と、写しの `Dropout`・`Interpolate`・`MaxUnpool` で、すべて
//! `supports_create_graph() == true`（閉包性。§21.1(4)）。
//!
//! **方式**:
//! - `Dropout`（V1+V2）: 写しは `dropout_with_mask(記録済みマスク)`。`Var::dropout` は
//!   マスクを引き直してグローバル RNG を消費するため呼ばない（`docs/rng-global-contract-design.md`
//!   §3.2）。cgrad は `where_cond(mask != 0, g·mask, 0)`。keep 位置は 1 階の
//!   `vjp_elementwise_mul` と同じ IEEE 乗算 1 回で bit 一致する。drop 位置は 1 階が `g·0`
//!   （g が非有限なら NaN、負の有限値なら `-0.0`）、cgrad は `+0.0` を返す（既知の差。§56.2）。
//! - `Interpolate` の `Nearest`・`NearestExact`（V1）: `zeros.scatter_add(index, g)`。
//!   固定 K タップの `Linear`・`Bilinear`・`Trilinear`・`Bicubic`（V1）:
//!   `g` を K 回複製して重みを掛け `scatter_add`（1 階 `interpolate_fixed_tap_scatter_vjp` と
//!   同じ並びと IEEE 演算）。`Area`（V3）: 値は 1 階の `interpolate_area_vjp`、構造は
//!   `pool` の窓構造を空間 N 次元へ一般化したもの。
//! - `MaxUnpool`（V1+V2）: 1 階と同じ `Scatter(Overwrite)` の d_src（`gather`＋last-writer
//!   マスク）。

use fandhe_ai_tensor_core::{InterpolateMode, ScatterReduce, ShapeError, Tensor};

use super::index_cum::{ScatterSpec, scatter_vjp};
use super::pool::{check_structure_contract, plan_adaptive_nd, window_scatter_structure};
use super::{f32_mask_to_bool, push_v3, value_of};
use crate::error::AutodiffError;
use crate::grad::{
    bicubic_src_index_and_weight_map, bilinear_src_index_and_weight_map, interpolate_area_vjp,
    linear_src_index_and_weight_map, nearest_exact_src_index_map, nearest_src_index_map,
    trilinear_src_index_and_weight_map,
};
use crate::tape::Tape;
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

fn overflow() -> AutodiffError {
    AutodiffError::Shape(ShapeError::ElementCountOverflow)
}

/// `shape` の要素数積（桁あふれは型付きエラー。素の `.iter().product()` は使わない）。
fn checked_numel(shape: &[usize]) -> Result<usize, AutodiffError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(overflow)
}

// ---------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------

/// `Dropout` の写し。記録済みマスクを固定して再生し、グローバル RNG は消費しない。
pub(super) fn replay_dropout<'c>(
    input: &Var<'c>,
    mask: &Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    input.dropout_with_mask(mask.clone())
}

/// `Dropout` の d_input。keep 位置は `g·mask`、drop 位置（`mask == 0`）は `0`。
pub(super) fn dropout_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    mask: &Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    let g_shape = g.shape();
    if mask.shape() != g_shape.as_slice() {
        return Err(invalid(format!(
            "Dropout の VJP: mask shape {:?} と上流 shape {g_shape:?} が不一致（内部契約違反）",
            mask.shape()
        )));
    }
    let keep = f32_mask_to_bool(mask)?;
    let mask_const = child.var_no_grad(&mask.contiguous());
    let zeros = child.var_no_grad(&Tensor::zeros(&g_shape)?);
    Var::where_cond(&keep, &g.mul(&mask_const)?, &zeros)
}

// ---------------------------------------------------------------------
// Interpolate
// ---------------------------------------------------------------------

/// `Interpolate` の d_input。入力検査の順序は 1 階 VJP（`grad.rs`）と同じ。
pub(super) fn interpolate_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    size: &[usize],
    mode: InterpolateMode,
) -> Result<Var<'c>, AutodiffError> {
    let rank = input_shape.len();
    let out_shape = g.shape();
    if size.is_empty() || size.len() > rank || out_shape.len() != rank {
        return Err(invalid(format!(
            "Interpolate の VJP: shape が整合しない（input={input_shape:?}, upstream={out_shape:?}, \
             size={size:?}。内部契約違反）"
        )));
    }
    let spatial_start = rank - size.len();
    if input_shape[..spatial_start] != out_shape[..spatial_start] {
        return Err(invalid(format!(
            "Interpolate の VJP: 空間軸以外が不一致（input={input_shape:?}, upstream={out_shape:?}）"
        )));
    }
    let outer = checked_numel(&input_shape[..spatial_start])?;
    // 空 batch は大きな確保をせずゼロ定数を返す（1 階の早期 return と同じ位置）。
    if outer == 0 {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    let sp_in_numel = checked_numel(&input_shape[spatial_start..])?;
    let sp_out_numel = checked_numel(&out_shape[spatial_start..])?;
    if sp_in_numel == 0 || sp_out_numel == 0 {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    // 固定タップ数 K（未知モードは fail-closed）。空間軸数がモードの要求と合わなければ拒否する
    // （`*_src_index_and_weight_map` は debug_assert で軸数を前提にするため）。
    let fixed = |axes: usize, k: usize| -> Result<usize, AutodiffError> {
        if size.len() == axes {
            Ok(k)
        } else {
            Err(invalid(format!(
                "Interpolate の VJP: 空間軸数 {} がモード {mode:?} の要求 {axes} と不一致",
                size.len()
            )))
        }
    };
    let g2 =
        |cols: usize| -> Result<Var<'c>, AutodiffError> { g.contiguous()?.reshape(&[outer, cols]) };

    match mode {
        InterpolateMode::Nearest | InterpolateMode::NearestExact => {
            check_sp_in(sp_in_numel)?;
            let index = if matches!(mode, InterpolateMode::Nearest) {
                nearest_src_index_map(input_shape, &out_shape, spatial_start, outer)
            } else {
                nearest_exact_src_index_map(input_shape, &out_shape, spatial_start, outer)
            };
            let zeros = child.var_no_grad(&Tensor::zeros(&[outer, sp_in_numel])?);
            zeros
                .scatter_add(1, &index, &g2(sp_out_numel)?)?
                .reshape(input_shape)
        }
        InterpolateMode::Linear { align_corners } => {
            let k = fixed(1, 2)?;
            check_sp_in(sp_in_numel)?;
            let (idx, w) = linear_src_index_and_weight_map(
                input_shape,
                &out_shape,
                spatial_start,
                align_corners,
            );
            fixed_tap_cgrad(
                child,
                g,
                input_shape,
                outer,
                sp_in_numel,
                sp_out_numel,
                k,
                &idx,
                &w,
            )
        }
        InterpolateMode::Bilinear { align_corners } => {
            let k = fixed(2, 4)?;
            check_sp_in(sp_in_numel)?;
            let (idx, w) = bilinear_src_index_and_weight_map(
                input_shape,
                &out_shape,
                spatial_start,
                align_corners,
            );
            fixed_tap_cgrad(
                child,
                g,
                input_shape,
                outer,
                sp_in_numel,
                sp_out_numel,
                k,
                &idx,
                &w,
            )
        }
        InterpolateMode::Trilinear { align_corners } => {
            let k = fixed(3, 8)?;
            check_sp_in(sp_in_numel)?;
            let (idx, w) = trilinear_src_index_and_weight_map(
                input_shape,
                &out_shape,
                spatial_start,
                align_corners,
            );
            fixed_tap_cgrad(
                child,
                g,
                input_shape,
                outer,
                sp_in_numel,
                sp_out_numel,
                k,
                &idx,
                &w,
            )
        }
        InterpolateMode::Bicubic { align_corners } => {
            let k = fixed(2, 16)?;
            check_sp_in(sp_in_numel)?;
            let (idx, w) = bicubic_src_index_and_weight_map(
                input_shape,
                &out_shape,
                spatial_start,
                align_corners,
            );
            fixed_tap_cgrad(
                child,
                g,
                input_shape,
                outer,
                sp_in_numel,
                sp_out_numel,
                k,
                &idx,
                &w,
            )
        }
        InterpolateMode::Area => {
            let sp_in = &input_shape[spatial_start..];
            let sp_out = &out_shape[spatial_start..];
            let plan = plan_adaptive_nd(sp_in, sp_out)?;
            // 疑似 shape `[outer, 1, 空間…]` で既存の窓構造（`nc = shape[0]·shape[1]`）を再利用する。
            let mut pseudo = vec![outer, 1];
            pseudo.extend_from_slice(sp_in);
            let composed = window_scatter_structure(child, g, &pseudo, &plan)?;
            let g_val = value_of(child, g)?;
            let host = interpolate_area_vjp(&g_val, input_shape, &out_shape, spatial_start, outer)?;
            check_structure_contract(&plan, &g_val, &host, &pseudo)?;
            let composed = composed.reshape(input_shape)?;
            push_v3(child, &composed, host)
        }
        _ => Err(invalid(format!(
            "Interpolate の VJP: 未知の InterpolateMode {mode:?} は fail-closed"
        ))),
    }
}

/// scatter の index dtype (`i32`) に収まらない入力空間は拒否する（1 階と同じ縦深防御）。
fn check_sp_in(sp_in_numel: usize) -> Result<(), AutodiffError> {
    if sp_in_numel > i32::MAX as usize {
        return Err(invalid(format!(
            "Interpolate の VJP: 空間軸要素数 {sp_in_numel} が i32::MAX を超え scatter_add の \
             index dtype (i32) に収まらない"
        )));
    }
    Ok(())
}

/// 固定 K タップ（`Linear`・`Bilinear`・`Trilinear`・`Bicubic`）の d_input。
/// 1 階の `interpolate_fixed_tap_scatter_vjp` と同じく、各出力位置の `g` に重みを f32 で掛け、
/// 出力位置 major・タップ minor の並びで `scatter_add`（f64 決定的集約）へ流す。
#[allow(clippy::too_many_arguments)]
fn fixed_tap_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    outer: usize,
    sp_in_numel: usize,
    sp_out_numel: usize,
    k: usize,
    index_row: &[i32],
    weight_row: &[f32],
) -> Result<Var<'c>, AutodiffError> {
    let sp_out_xk = sp_out_numel.checked_mul(k).ok_or_else(overflow)?;
    let total = outer.checked_mul(sp_out_xk).ok_or_else(overflow)?;
    if total > isize::MAX as usize / (2 * std::mem::size_of::<i32>()) {
        return Err(overflow());
    }
    if index_row.len() != sp_out_xk || weight_row.len() != sp_out_xk {
        return Err(invalid(
            "Interpolate の VJP: 索引・重み表の長さが不整合（内部契約違反）".into(),
        ));
    }
    let mut index_data = Vec::with_capacity(total);
    let mut weight_data = Vec::with_capacity(total);
    for _ in 0..outer {
        index_data.extend_from_slice(index_row);
        weight_data.extend_from_slice(weight_row);
    }
    let index = Tensor::<i32>::new(index_data, &[outer, sp_out_xk])?;
    let weight = child.var_no_grad(&Tensor::new(weight_data, &[outer, sp_out_xk])?);
    let src = g
        .contiguous()?
        .reshape(&[outer, sp_out_numel, 1])?
        .broadcast_to(&[outer, sp_out_numel, k])?
        .contiguous()?
        .reshape(&[outer, sp_out_xk])?
        .mul(&weight)?;
    let zeros = child.var_no_grad(&Tensor::zeros(&[outer, sp_in_numel])?);
    zeros.scatter_add(1, &index, &src)?.reshape(input_shape)
}

// ---------------------------------------------------------------------
// MaxUnpool
// ---------------------------------------------------------------------

/// `MaxUnpool` の d_input（入力 shape の `gather`＋last-writer マスク）。
pub(super) fn max_unpool_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    index: &Tensor<i32>,
) -> Result<Var<'c>, AutodiffError> {
    let out_shape = g.shape();
    if !(3..=5).contains(&input_shape.len())
        || out_shape.len() != input_shape.len()
        || input_shape[..2] != out_shape[..2]
        || index.shape() != input_shape
    {
        return Err(invalid(format!(
            "MaxUnpool の VJP: shape が整合しない（input={input_shape:?}, upstream={out_shape:?}, \
             index={:?}。内部契約違反）",
            index.shape()
        )));
    }
    let nc = input_shape[0]
        .checked_mul(input_shape[1])
        .ok_or_else(overflow)?;
    let in_plane = checked_numel(&input_shape[2..])?;
    let out_plane = checked_numel(&out_shape[2..])?;
    if nc == 0 || in_plane == 0 || out_plane == 0 {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    let index_flat = index.contiguous().reshape(&[nc, in_plane])?;
    for v in index_flat.host_slice().iter() {
        let vi = i64::from(*v);
        if vi < 0 || (vi as usize) >= out_plane {
            return Err(invalid(format!(
                "MaxUnpool の VJP: index の値 {vi} が [0, {out_plane}) の範囲外（契約違反）"
            )));
        }
    }
    let g_flat = g.contiguous()?.reshape(&[nc, out_plane])?;
    let grads = scatter_vjp(
        child,
        &g_flat,
        &ScatterSpec {
            input_shape: &[nc, out_plane],
            src_shape: &[nc, in_plane],
            dim: 1,
            index: &index_flat,
            reduce: ScatterReduce::Overwrite,
        },
        false,
        true,
    )?;
    grads
        .d_src
        .ok_or_else(|| {
            invalid("MaxUnpool の VJP: d_src が構築されなかった（内部契約違反）".into())
        })?
        .reshape(input_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    /// 空間軸数がモードの要求と合わない入力は型付きで拒否される（`child` へは何も書かない前提の
    /// 入口検査）。未知の `InterpolateMode` は外部から作れないため `_ =>` 腕はコード上で担保する。
    #[test]
    fn interpolate_rejects_axis_count_mismatch() {
        let child = Tape::new_with_ops(naive_ops());
        let g = child.var(&Tensor::new(vec![1.0f32; 8], &[1, 1, 2, 4]).unwrap());
        let r = interpolate_cgrad(
            &child,
            &g,
            &[1, 1, 1, 2],
            &[2, 4],
            InterpolateMode::Linear {
                align_corners: false,
            },
        );
        assert!(matches!(r, Err(AutodiffError::Backward(_))));
    }

    #[test]
    fn dropout_cgrad_rejects_shape_mismatch() {
        let child = Tape::new_with_ops(naive_ops());
        let g = child.var(&Tensor::new(vec![1.0f32; 4], &[4]).unwrap());
        let mask = Tensor::new(vec![1.0f32; 3], &[3]).unwrap();
        assert!(matches!(
            dropout_cgrad(&child, &g, &mask),
            Err(AutodiffError::Backward(_))
        ));
    }
}
