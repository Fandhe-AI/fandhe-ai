//! 選択縮約 3 Op（`OrderSelect`・`MedianAll`・`Quantile`）の `create_graph` 対応
//! （イシュー #3790・親 #3724・追跡表 #3717 項目 3。決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §34）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）・
//! `build_cgrads`（1 階 VJP の子テープ記録）・`validate_ancestors`（入口検査）から
//! 呼ばれる。`Max`／`Min`／`Amax`／`Amin` は別イシュー（#3789）の担当でここには含めない。
//!
//! **方式**（値は 1 階 `grad.rs` の VJP と bit 一致する。契約は REQ-2 のまま）:
//! - `OrderSelect` は **V1**: forward は `gather(dim, index_keep)` → `reshape`（値の選択のみ）、
//!   cgrad は 1 階と同じ `zeros.scatter_add`（`index_cum::gather_like_vjp`）。
//! - `MedianAll`・`Quantile` は **V3**: 値は 1 階と同じホスト関数
//!   （`median_all_vjp_host`・`quantile_vjp_host`）、構造は「x について区分定数のマスク／重み」
//!   による `where_cond`（`Single` lane は乗算を通さず `inf*0` を避ける）。マスクと重みは
//!   x に依存しない定数のため、二階は `g` を通る混合項だけが流れる。
//! - 子テープへ積む Op は `Gather`・`Reshape`・`BroadcastTo`・`Where`・`Mul`・`Add`・
//!   `Scatter`・`Contiguous`・定数 `Leaf` で、すべて `supports_create_graph() == true`
//!   （閉包性。孫テープでも閉じる）。
//! - 位置情報は tensor-core の `stat_reduce::quantile_select_host`（選択規則の単一情報源）。
//!
//! GPU: `BackendOps::stat_*` は CPU のみ実装で、CUDA／Metal ではホストへフォールバックする
//! （経路区分 `H`）。実機 parity は未実測（Phase 10 の #3856・#3858）。

use fandhe_ai_tensor_core::stat_reduce::{
    QuantileLaneSelect, median_all_host, median_all_vjp_host, quantile_select_host,
    quantile_vjp_host, stat_layout,
};
use fandhe_ai_tensor_core::{QuantileInterpolation, Tensor};

use super::{index_cum, push_v3, value_of};
use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 縮約軸（`None` は全軸）を長さ 1 にした keepdim 形。
fn keep_shape(input_shape: &[usize], dim: Option<usize>) -> Vec<usize> {
    match dim {
        Some(d) => {
            let mut s = input_shape.to_vec();
            if let Some(v) = s.get_mut(d) {
                *v = 1;
            }
            s
        }
        None => vec![1; input_shape.len()],
    }
}

fn keep_index(
    index: &Tensor<i32>,
    input_shape: &[usize],
    dim: usize,
) -> Result<Tensor<i32>, AutodiffError> {
    let keep = keep_shape(input_shape, Some(dim));
    Ok(Tensor::new(
        index.contiguous().host_slice().to_vec(),
        &keep,
    )?)
}

// ---------------------------------------------------------------------
// 入口検査（validate_ancestors から呼ぶ。子テープへ書く前に拒否する）
// ---------------------------------------------------------------------

/// `OrderSelect` の payload 検査: `dim` 範囲・`index` shape・各索引値の範囲・`i32` 収容。
pub(super) fn check_order_select(
    input_shape: &[usize],
    node_shape: &[usize],
    dim: usize,
    index: &Tensor<i32>,
) -> Result<(), AutodiffError> {
    let Some(&axis_len) = input_shape.get(dim) else {
        return Err(invalid(format!(
            "OrderSelect の dim={dim} が入力 rank {} の範囲外",
            input_shape.len()
        )));
    };
    if i32::try_from(axis_len).is_err() {
        return Err(invalid(format!(
            "OrderSelect の軸長 {axis_len} が i32 に収まらない"
        )));
    }
    if index.shape() != node_shape {
        return Err(invalid(format!(
            "OrderSelect の index shape {:?} がノード出力 shape {node_shape:?} と不一致",
            index.shape()
        )));
    }
    for &v in index.contiguous().host_slice().iter() {
        let ok = usize::try_from(v).map(|u| u < axis_len).unwrap_or(false);
        if !ok {
            return Err(invalid(format!(
                "OrderSelect の index 値 {v} が軸長 {axis_len} の範囲外"
            )));
        }
    }
    Ok(())
}

/// `Quantile` の payload 検査: `q` が有限で `[0, 1]`、補間方式が既知の 5 variant。
/// `QuantileInterpolation` は `#[non_exhaustive]` のため未知 variant は拒否する。
pub(super) fn check_quantile(
    q: f32,
    interpolation: QuantileInterpolation,
) -> Result<(), AutodiffError> {
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(invalid(format!(
            "Quantile の q={q} が有限かつ [0, 1] でない"
        )));
    }
    match interpolation {
        QuantileInterpolation::Linear
        | QuantileInterpolation::Lower
        | QuantileInterpolation::Higher
        | QuantileInterpolation::Midpoint
        | QuantileInterpolation::Nearest => Ok(()),
        _ => Err(invalid(format!(
            "Quantile の未知の補間方式 {interpolation:?}"
        ))),
    }
}

// ---------------------------------------------------------------------
// forward 再生
// ---------------------------------------------------------------------

/// `OrderSelect` の写し: `gather(dim, index_keep)` を出力 shape へ reshape する。
pub(super) fn replay_order_select<'c>(
    x: &Var<'c>,
    dim: usize,
    index: &Tensor<i32>,
    out_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let index_keep = keep_index(index, &x.shape(), dim)?;
    x.gather(dim, &index_keep)?.contiguous()?.reshape(out_shape)
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録
// ---------------------------------------------------------------------

/// `OrderSelect` の d_input（V1）: `zeros.scatter_add(dim, index_keep, g_keep)`。
pub(super) fn order_select_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    dim: usize,
    index: &Tensor<i32>,
) -> Result<Var<'c>, AutodiffError> {
    let index_keep = keep_index(index, input_shape, dim)?;
    let g_keep = g
        .contiguous()?
        .reshape(&keep_shape(input_shape, Some(dim)))?;
    index_cum::gather_like_vjp(child, &g_keep, input_shape, dim, &index_keep)
}

fn bool_mask(flags: Vec<bool>, shape: &[usize]) -> Result<Tensor<bool>, AutodiffError> {
    Ok(Tensor::new(flags, shape)?)
}

/// `MedianAll` の d_input（V3）。中央値と等しい要素（中央値が NaN なら NaN 要素）へ
/// `g / count` を均等分配する。`numel == 0` は 1 階（空ベクトル）に揃えゼロ定数。
pub(super) fn median_all_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x_val: &Tensor<f32>,
    input_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    if input_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    let layout = stat_layout(input_shape, None)?;
    let x = x_val.contiguous();
    let xs = x.host_slice();
    // 値: 1 階と同じホスト関数（bit 一致）。
    let g_val = value_of(child, g)?;
    let data = median_all_vjp_host(&xs, &g_val.contiguous().host_slice(), &layout)?;
    let host = Tensor::new(data, input_shape)?;
    // 構造: マスクと重み `1/count`（x について区分定数）。
    let m = median_all_host(&xs, &layout)?;
    let flags: Vec<bool> = xs
        .iter()
        .map(|&v| if m.is_nan() { v.is_nan() } else { v == m })
        .collect();
    let count = flags.iter().filter(|&&b| b).count();
    let share = (1.0_f64 / count.max(1) as f64) as f32;
    let weights: Vec<f32> = flags.iter().map(|&b| if b { share } else { 0.0 }).collect();
    let g_b = g
        .contiguous()?
        .reshape(&keep_shape(input_shape, None))?
        .broadcast_to(input_shape)?;
    let w = child.var_no_grad(&Tensor::new(weights, input_shape)?);
    let zeros = child.var_no_grad(&Tensor::zeros(input_shape)?);
    let composed = Var::where_cond(&bool_mask(flags, input_shape)?, &g_b.mul(&w)?, &zeros)?;
    push_v3(child, &composed, host)
}

/// `Quantile` の d_input（V3）。`Single` lane は `g` を直接（乗算なし）、`Pair` lane は
/// 下側へ `1-w`・上側へ `w` を乗じる。f32 で 0 に丸まる重みの位置はマスクから外す。
pub(super) fn quantile_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x_val: &Tensor<f32>,
    input_shape: &[usize],
    q: f32,
    dim: Option<usize>,
    interp: QuantileInterpolation,
) -> Result<Var<'c>, AutodiffError> {
    if input_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    let layout = stat_layout(input_shape, dim)?;
    let x = x_val.contiguous();
    let xs = x.host_slice();
    let g_val = value_of(child, g)?;
    let data = quantile_vjp_host(&xs, &g_val.contiguous().host_slice(), &layout, q, interp)?;
    let host = Tensor::new(data, input_shape)?;

    let numel = xs.len();
    let mut single = vec![false; numel];
    let mut pair = vec![false; numel];
    let mut weights = vec![0.0_f32; numel];
    for sel in quantile_select_host(&xs, &layout, q, interp)? {
        match sel {
            QuantileLaneSelect::Single { pos } => single[pos] = true,
            QuantileLaneSelect::Pair { lo, hi, w } => {
                let (wl, wh) = ((1.0 - w) as f32, w as f32);
                if wl != 0.0 {
                    pair[lo] = true;
                    weights[lo] = wl;
                }
                if wh != 0.0 {
                    pair[hi] = true;
                    weights[hi] = wh;
                }
            }
            _ => return Err(invalid("Quantile の未知の選択種別".into())),
        }
    }
    let g_b = g
        .contiguous()?
        .reshape(&keep_shape(input_shape, dim))?
        .broadcast_to(input_shape)?;
    let zeros = child.var_no_grad(&Tensor::zeros(input_shape)?);
    let mut composed = Var::where_cond(&bool_mask(single, input_shape)?, &g_b, &zeros)?;
    if pair.iter().any(|&b| b) {
        let w = child.var_no_grad(&Tensor::new(weights, input_shape)?);
        let term = Var::where_cond(&bool_mask(pair, input_shape)?, &g_b.mul(&w)?, &zeros)?;
        composed = composed.add(&term)?;
    }
    push_v3(child, &composed, host)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_order_select_rejects_bad_payloads() {
        let ok = Tensor::<i32>::new(vec![0, 2], &[2]).unwrap();
        assert!(check_order_select(&[2, 3], &[2], 1, &ok).is_ok());
        // dim 範囲外
        assert!(check_order_select(&[2, 3], &[2], 2, &ok).is_err());
        // 索引値が軸長以上
        let big = Tensor::<i32>::new(vec![0, 3], &[2]).unwrap();
        assert!(check_order_select(&[2, 3], &[2], 1, &big).is_err());
        // 負の索引
        let neg = Tensor::<i32>::new(vec![0, -1], &[2]).unwrap();
        assert!(check_order_select(&[2, 3], &[2], 1, &neg).is_err());
        // shape 不一致
        assert!(check_order_select(&[2, 3], &[3], 1, &ok).is_err());
    }

    #[test]
    fn check_quantile_rejects_bad_q() {
        assert!(check_quantile(0.5, QuantileInterpolation::Linear).is_ok());
        assert!(check_quantile(1.5, QuantileInterpolation::Linear).is_err());
        assert!(check_quantile(f32::NAN, QuantileInterpolation::Lower).is_err());
        assert!(check_quantile(-0.1, QuantileInterpolation::Nearest).is_err());
    }
}
