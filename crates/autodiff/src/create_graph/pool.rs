//! プーリング族（`MaxPool2d`・`MaxPool3d`・`AvgPool2d`・`AvgPool3d`・
//! `AdaptiveAvgPool2d`）の `create_graph` 対応（イシュー #3085・親 #3076・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §21・§29）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）と
//! `build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。子テープへ積む Op は
//! `Contiguous`・`Reshape`・`Mul`・`Gather`・`Scatter` だけで、すべて
//! `supports_create_graph() == true`（閉包性。§21.1(4)）。
//!
//! **方式**:
//! - Max 系は forward が確定した勝者 `index`（先勝ちタイ規則・NaN 規則込み）を
//!   ホスト定数として固定し、forward 再生は `gather`、VJP は `zeros.scatter_add`
//!   （1 階と同じ `scatter_with_fallback(Add)`。重複索引は `f64` の決定的集約で、
//!   値は 1 階と bit 一致する）。argmax を再計算しないためタイ規則は構造上 1 階と
//!   一致する。§21.2 の分類は V2 だが実態は **V1**（index 固定の bit 一致合成）。
//! - Avg 系（`AvgPool2d`・`AvgPool3d`・`AdaptiveAvgPool2d`）は **V3**: 値は 1 階と
//!   同じホスト関数（`f64` アキュムレータ・1 回 downcast）を `push_v3` で登録し、
//!   構造は [`window_scatter_structure`]（`g` に窓の逆 divisor を掛け、
//!   `gather` → `scatter_add` で入力位置へ戻す索引の合成）で積む。
//!   §21.3 は `AvgPool2d` の構造を `Fold` 経由としていたが、Adaptive は窓が可変・
//!   3D には fold が無いため 3 variant を同一の索引合成へ統一した（逸脱。§29 に理由）。

use fandhe_ai_tensor_core::pool3d::{Pool3dParams, pool3d_layout};
use fandhe_ai_tensor_core::{Pool2dParams, ShapeError, Tensor, adaptive_window};

use super::{push_v3, value_of};
use crate::error::AutodiffError;
use crate::grad::{adaptive_avg_pool2d_vjp, avg_pool2d_vjp, avg_pool3d_vjp, pool_window_input_pos};
use crate::tape::Tape;
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

fn overflow() -> AutodiffError {
    AutodiffError::Shape(ShapeError::ElementCountOverflow)
}

/// 空間軸（軸 2 以降）の要素数積。
fn spatial_numel(shape: &[usize]) -> Result<usize, AutodiffError> {
    shape
        .get(2..)
        .ok_or_else(overflow)?
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(overflow)
}

// ---------------------------------------------------------------------
// forward 再生（replay_op から呼ばれる）
// ---------------------------------------------------------------------

/// `MaxPool2d`／`MaxPool3d` の写し。記録済み `index` で勝者を `gather` するだけなので値は
/// 親と bit 一致する（子テープ上の Op タグだけが `Gather` になる。`Sort`／`Topk` と同型）。
pub(super) fn replay_max_pool<'c>(
    input: &Var<'c>,
    index: &Tensor<i32>,
    node_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    if node_shape.contains(&0) {
        return Ok(input.tape().var_no_grad(&Tensor::zeros(node_shape)?));
    }
    let in_shape = input.shape();
    let nc = in_shape[0].checked_mul(in_shape[1]).ok_or_else(overflow)?;
    let in_sp = spatial_numel(&in_shape)?;
    let out_sp = spatial_numel(node_shape)?;
    let index_flat = index.contiguous().reshape(&[nc, out_sp])?;
    input
        .contiguous()?
        .reshape(&[nc, in_sp])?
        .gather(1, &index_flat)?
        .reshape(node_shape)
}

/// `AvgPool2d` の写し（`ceil_mode` は forward が常に `false` に限る）。
pub(super) fn replay_avg_pool2d<'c>(
    input: &Var<'c>,
    params: &Pool2dParams,
    count_include_pad: bool,
) -> Result<Var<'c>, AutodiffError> {
    input.avg_pool2d(
        params.kernel_size(),
        Some(params.stride()),
        params.padding(),
        false,
        count_include_pad,
    )
}

/// `AvgPool3d` の写し。
pub(super) fn replay_avg_pool3d<'c>(
    input: &Var<'c>,
    params: &Pool3dParams,
    count_include_pad: bool,
) -> Result<Var<'c>, AutodiffError> {
    input.avg_pool3d(
        params.kernel_size(),
        Some(params.stride()),
        params.padding(),
        false,
        count_include_pad,
    )
}

/// `AdaptiveAvgPool2d` の写し。出力サイズはノード自身の shape から取る。
pub(super) fn replay_adaptive_avg_pool2d<'c>(
    input: &Var<'c>,
    node_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    if node_shape.len() != 4 {
        return Err(invalid(format!(
            "AdaptiveAvgPool2d のノード shape {node_shape:?} が rank 4 でない（内部契約違反）"
        )));
    }
    input.adaptive_avg_pool2d([node_shape[2], node_shape[3]])
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

/// `MaxPool2d`／`MaxPool3d` の d_input（1 階 `grad.rs` と同じ式）。
/// `g`・`index` を `[N·C, 出力空間]` へ平坦化し、`zeros([N·C, 入力空間]).scatter_add`。
/// index が `[0, 入力空間)` を外れる場合は子テープへ書く前に型付きエラーで拒否する
/// （security.md A08）。
pub(super) fn max_pool_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    index: &Tensor<i32>,
) -> Result<Var<'c>, AutodiffError> {
    if input_shape.len() < 4 {
        return Err(invalid(format!(
            "MaxPool の入力 shape {input_shape:?} が rank 4 以上でない（内部契約違反）"
        )));
    }
    let nc = input_shape[0]
        .checked_mul(input_shape[1])
        .ok_or_else(overflow)?;
    let in_sp = spatial_numel(input_shape)?;
    let g_shape = g.shape();
    let out_sp = spatial_numel(&g_shape)?;
    if nc == 0 || in_sp == 0 || out_sp == 0 {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    let flat_out = [nc, out_sp];
    let index_flat = index.contiguous().reshape(&flat_out)?;
    for v in index_flat.host_slice().iter() {
        let vi = i64::from(*v);
        if vi < 0 || (vi as usize) >= in_sp {
            return Err(invalid(format!(
                "MaxPool の VJP: index の値 {vi} が [0, {in_sp}) の範囲外（契約違反）"
            )));
        }
    }
    let g_flat = g.contiguous()?.reshape(&flat_out)?;
    super::index_cum::gather_like_vjp(child, &g_flat, &[nc, in_sp], 1, &index_flat)?
        .reshape(input_shape)
}

/// 窓構造の索引表（1 平面ぶん。全 `N·C` 平面で共通）。
struct WindowPlan {
    in_spatial: usize,
    out_spatial: usize,
    /// 窓に属する要素ごとの出力位置（`[0, out_spatial)`）。
    src: Vec<i32>,
    /// 同じく入力位置（`[0, in_spatial)`）。
    dst: Vec<i32>,
    /// 出力位置ごとの `1 / divisor`（長さ `out_spatial`）。
    inv: Vec<f32>,
}

impl WindowPlan {
    fn new(in_spatial: usize, out_spatial: usize) -> Result<Self, AutodiffError> {
        if in_spatial > i32::MAX as usize || out_spatial > i32::MAX as usize {
            return Err(AutodiffError::Shape(ShapeError::IndexRangeOverflow {
                index: in_spatial.max(out_spatial),
            }));
        }
        Ok(Self {
            in_spatial,
            out_spatial,
            src: Vec::new(),
            dst: Vec::new(),
            inv: Vec::with_capacity(out_spatial),
        })
    }

    /// 出力位置 `o`（昇順に呼ぶ）の窓（入力位置の列）と divisor を登録する。
    fn push_window(
        &mut self,
        o: usize,
        members: &[usize],
        divisor: usize,
    ) -> Result<(), AutodiffError> {
        if divisor == 0 || members.is_empty() {
            return Err(AutodiffError::Backward(
                "create_graph: プーリングの窓要素数または divisor が 0（契約違反）".into(),
            ));
        }
        for &m in members {
            self.src.push(o as i32);
            self.dst.push(m as i32);
        }
        self.inv.push((1.0 / divisor as f64) as f32);
        Ok(())
    }
}

/// `AvgPool2d` の窓表（`grad::avg_pool2d_vjp` と同じ窓・divisor 規約）。
fn plan_avg2d(
    input_shape: &[usize],
    out_shape: &[usize],
    params: &Pool2dParams,
    count_include_pad: bool,
) -> Result<WindowPlan, AutodiffError> {
    if input_shape.len() != 4 || out_shape.len() != 4 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 4,
            actual: input_shape.len().min(out_shape.len()),
        }));
    }
    let (h_in, w_in) = (input_shape[2], input_shape[3]);
    let (h_out, w_out) = (out_shape[2], out_shape[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let mut plan = WindowPlan::new(
        h_in.checked_mul(w_in).ok_or_else(overflow)?,
        h_out.checked_mul(w_out).ok_or_else(overflow)?,
    )?;
    let mut members = Vec::new();
    for oh in 0..h_out {
        for ow in 0..w_out {
            members.clear();
            for ki in 0..kh {
                let Some(h) = pool_window_input_pos(oh, sh, ki, dh, ph).filter(|&h| h < h_in)
                else {
                    continue;
                };
                for kj in 0..kw {
                    let Some(w) = pool_window_input_pos(ow, sw, kj, dw, pw).filter(|&w| w < w_in)
                    else {
                        continue;
                    };
                    members.push(h * w_in + w);
                }
            }
            let divisor = if count_include_pad {
                kh.checked_mul(kw).ok_or_else(overflow)?
            } else {
                members.len()
            };
            plan.push_window(oh * w_out + ow, &members, divisor)?;
        }
    }
    Ok(plan)
}

/// `AdaptiveAvgPool2d` の窓表（`adaptive_window` を単一情報源とする。divisor は窓要素数）。
fn plan_adaptive2d(
    input_shape: &[usize],
    out_shape: &[usize],
) -> Result<WindowPlan, AutodiffError> {
    if input_shape.len() != 4 || out_shape.len() != 4 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 4,
            actual: input_shape.len().min(out_shape.len()),
        }));
    }
    let (h_in, w_in) = (input_shape[2], input_shape[3]);
    let (h_out, w_out) = (out_shape[2], out_shape[3]);
    let mut plan = WindowPlan::new(
        h_in.checked_mul(w_in).ok_or_else(overflow)?,
        h_out.checked_mul(w_out).ok_or_else(overflow)?,
    )?;
    let mut members = Vec::new();
    for oh in 0..h_out {
        let (hs, he) = adaptive_window(oh, h_in, h_out).ok_or_else(overflow)?;
        for ow in 0..w_out {
            let (ws, we) = adaptive_window(ow, w_in, w_out).ok_or_else(overflow)?;
            members.clear();
            for h in hs..he {
                for w in ws..we {
                    members.push(h * w_in + w);
                }
            }
            let divisor = members.len();
            plan.push_window(oh * w_out + ow, &members, divisor)?;
        }
    }
    Ok(plan)
}

/// `AvgPool3d` の窓表（`tensor_core::pool3d` の窓走査・divisor 規約と同じ）。
fn plan_avg3d(
    input_shape: &[usize],
    params: &Pool3dParams,
    count_include_pad: bool,
) -> Result<WindowPlan, AutodiffError> {
    let layout = pool3d_layout(input_shape, params).map_err(AutodiffError::Shape)?;
    let [_, _, d_in, h_in, w_in] = layout.in_shape();
    let [_, _, d_out, h_out, w_out] = layout.out_shape();
    let [kd, kh, kw] = params.kernel_size();
    let [sd, sh, sw] = params.stride();
    let [pd, ph, pw] = params.padding();
    let [dd, dh, dw] = params.dilation();
    let mut plan = WindowPlan::new(
        d_in.checked_mul(h_in)
            .and_then(|v| v.checked_mul(w_in))
            .ok_or_else(overflow)?,
        d_out
            .checked_mul(h_out)
            .and_then(|v| v.checked_mul(w_out))
            .ok_or_else(overflow)?,
    )?;
    let mut members = Vec::new();
    for od in 0..d_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                members.clear();
                for ki in 0..kd {
                    let Some(d) = pool_window_input_pos(od, sd, ki, dd, pd).filter(|&d| d < d_in)
                    else {
                        continue;
                    };
                    for kj in 0..kh {
                        let Some(h) =
                            pool_window_input_pos(oh, sh, kj, dh, ph).filter(|&h| h < h_in)
                        else {
                            continue;
                        };
                        for kk in 0..kw {
                            let Some(w) =
                                pool_window_input_pos(ow, sw, kk, dw, pw).filter(|&w| w < w_in)
                            else {
                                continue;
                            };
                            members.push((d * h_in + h) * w_in + w);
                        }
                    }
                }
                let divisor = if count_include_pad {
                    kd.checked_mul(kh)
                        .and_then(|v| v.checked_mul(kw))
                        .ok_or_else(overflow)?
                } else {
                    members.len()
                };
                plan.push_window((od * h_out + oh) * w_out + ow, &members, divisor)?;
            }
        }
    }
    Ok(plan)
}

/// 平均系 VJP の構造: `g`（`[N, C, 出力空間…]`）を `[N·C, 出力空間]` へ平坦化し、窓の逆
/// divisor を掛けて `gather` → `zeros.scatter_add`（窓の重なりは `f64` 決定的集約）→
/// 入力 shape へ戻す。積む Op はすべて replayable。
fn window_scatter_structure<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    plan: &WindowPlan,
) -> Result<Var<'c>, AutodiffError> {
    let nc = input_shape[0]
        .checked_mul(input_shape[1])
        .ok_or_else(overflow)?;
    let e = plan.src.len();
    // N·C 平面ぶんへ複製する索引テンソルの要素数（確保前に桁あふれ・巨大確保を拒否する）。
    let total = nc.checked_mul(e).ok_or_else(overflow)?;
    if total > isize::MAX as usize / (2 * std::mem::size_of::<i32>()) {
        return Err(overflow());
    }
    let replicate = |one: &[i32]| -> Result<Tensor<i32>, AutodiffError> {
        let mut v = Vec::with_capacity(total);
        for _ in 0..nc {
            v.extend_from_slice(one);
        }
        Ok(Tensor::new(v, &[nc, e])?)
    };
    let src_index = replicate(&plan.src)?;
    let dst_index = replicate(&plan.dst)?;
    let inv = child.var_no_grad(&Tensor::new(plan.inv.clone(), &[1, plan.out_spatial])?);
    let g_flat = g.contiguous()?.reshape(&[nc, plan.out_spatial])?;
    let weighted = g_flat.mul(&inv)?;
    let picked = weighted.gather(1, &src_index)?;
    let zeros = child.var_no_grad(&Tensor::zeros(&[nc, plan.in_spatial])?);
    zeros
        .scatter_add(1, &dst_index, &picked)?
        .reshape(input_shape)
}

/// 空の上流・入力は 1 階の扱いに揃えゼロ定数を返す。
fn empty_case<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
) -> Result<Option<Var<'c>>, AutodiffError> {
    if g.shape().contains(&0) || input_shape.contains(&0) {
        return Ok(Some(child.var_no_grad(&Tensor::zeros(input_shape)?)));
    }
    Ok(None)
}

/// `AvgPool2d` の d_input（V3）。
pub(super) fn avg_pool2d_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    params: &Pool2dParams,
    count_include_pad: bool,
) -> Result<Var<'c>, AutodiffError> {
    if let Some(z) = empty_case(child, g, input_shape)? {
        return Ok(z);
    }
    let plan = plan_avg2d(input_shape, &g.shape(), params, count_include_pad)?;
    let composed = window_scatter_structure(child, g, input_shape, &plan)?;
    let host = avg_pool2d_vjp(&value_of(child, g)?, input_shape, params, count_include_pad)?;
    push_v3(child, &composed, host)
}

/// `AdaptiveAvgPool2d` の d_input（V3）。
pub(super) fn adaptive_avg_pool2d_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    if let Some(z) = empty_case(child, g, input_shape)? {
        return Ok(z);
    }
    let plan = plan_adaptive2d(input_shape, &g.shape())?;
    let composed = window_scatter_structure(child, g, input_shape, &plan)?;
    let host = adaptive_avg_pool2d_vjp(&value_of(child, g)?, input_shape)?;
    push_v3(child, &composed, host)
}

/// `AvgPool3d` の d_input（V3）。
pub(super) fn avg_pool3d_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    params: &Pool3dParams,
    count_include_pad: bool,
) -> Result<Var<'c>, AutodiffError> {
    if let Some(z) = empty_case(child, g, input_shape)? {
        return Ok(z);
    }
    let plan = plan_avg3d(input_shape, params, count_include_pad)?;
    let composed = window_scatter_structure(child, g, input_shape, &plan)?;
    let host = avg_pool3d_vjp(&value_of(child, g)?, input_shape, params, count_include_pad)?;
    push_v3(child, &composed, host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    fn data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.37 - 1.5)
            .collect()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            assert!(d < 1e-5 || d / y.abs().max(1e-30) < 1e-3, "{x} vs {y}");
        }
    }

    /// 構造側（索引合成）の値が 1 階のホスト関数と REQ-2 の範囲で一致すること。
    #[test]
    fn structure_matches_host_avg2d_adaptive_avg3d() {
        let tape = Tape::new_with_ops(naive_ops());
        for cip in [true, false] {
            let params = Pool2dParams::new([3, 3], Some([2, 2]), [1, 1], [1, 1]).unwrap();
            let ishape = [1usize, 2, 5, 6];
            let oshape = [1usize, 2, 3, 3];
            let g = tape.var(&Tensor::new(data(2 * 9), &oshape).unwrap());
            let plan = plan_avg2d(&ishape, &oshape, &params, cip).unwrap();
            let s = window_scatter_structure(&tape, &g, &ishape, &plan).unwrap();
            let h = avg_pool2d_vjp(&value_of(&tape, &g).unwrap(), &ishape, &params, cip).unwrap();
            close(&s.to_tensor().host_slice(), &h.host_slice());
        }
        let ishape = [1usize, 2, 5, 5];
        let oshape = [1usize, 2, 3, 3];
        let g = tape.var(&Tensor::new(data(2 * 9), &oshape).unwrap());
        let plan = plan_adaptive2d(&ishape, &oshape).unwrap();
        let s = window_scatter_structure(&tape, &g, &ishape, &plan).unwrap();
        let h = adaptive_avg_pool2d_vjp(&value_of(&tape, &g).unwrap(), &ishape).unwrap();
        close(&s.to_tensor().host_slice(), &h.host_slice());
        for cip in [true, false] {
            let params =
                Pool3dParams::new([2, 3, 2], Some([1, 2, 1]), [1, 1, 0], [1, 1, 1]).unwrap();
            let ishape = [1usize, 2, 3, 5, 4];
            let layout = pool3d_layout(&ishape, &params).unwrap();
            let oshape = layout.out_shape();
            let n: usize = oshape.iter().product();
            let g = tape.var(&Tensor::new(data(n), &oshape).unwrap());
            let plan = plan_avg3d(&ishape, &params, cip).unwrap();
            let s = window_scatter_structure(&tape, &g, &ishape, &plan).unwrap();
            let h = avg_pool3d_vjp(&value_of(&tape, &g).unwrap(), &ishape, &params, cip).unwrap();
            close(&s.to_tensor().host_slice(), &h.host_slice());
        }
    }

    /// Max の gather 再生が forward 出力と bit 一致する（NaN・タイ・負値を含む窓）。
    #[test]
    fn max_replay_is_bitwise_equal_to_forward() {
        let tape = Tape::new_with_ops(naive_ops());
        let mut x = data(2 * 6 * 6);
        x[3] = f32::NAN;
        x[4] = f32::NAN;
        x[10] = x[11];
        x[20] = -5.0;
        let xv = tape.var(&Tensor::new(x, &[1, 2, 6, 6]).unwrap());
        let (y, idx) = xv
            .max_pool2d([3, 3], Some([1, 1]), [1, 1], [1, 1], false)
            .unwrap();
        let r = replay_max_pool(&xv, &idx, &y.shape()).unwrap();
        let a: Vec<u32> = y
            .to_tensor()
            .host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b: Vec<u32> = r
            .to_tensor()
            .host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(a, b);
    }

    /// 範囲外 index は panic ではなく型付きエラー。
    #[test]
    fn max_vjp_rejects_out_of_range_index() {
        let tape = Tape::new_with_ops(naive_ops());
        let g = tape.var(&Tensor::new(vec![1.0f32; 4], &[1, 1, 2, 2]).unwrap());
        let idx = Tensor::<i32>::new(vec![0, 1, 2, 16], &[1, 1, 2, 2]).unwrap();
        assert!(max_pool_vjp(&tape, &g, &[1, 1, 4, 4], &idx).is_err());
        let idx = Tensor::<i32>::new(vec![0, 1, 2, -1], &[1, 1, 2, 2]).unwrap();
        assert!(max_pool_vjp(&tape, &g, &[1, 1, 4, 4], &idx).is_err());
    }
}
