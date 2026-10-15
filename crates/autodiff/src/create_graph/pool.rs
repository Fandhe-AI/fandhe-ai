//! プーリング族（`MaxPool2d`・`MaxPool3d`・`AvgPool2d`・`AvgPool3d`・
//! `AdaptiveAvgPool2d`）の `create_graph` 対応（イシュー #3085・親 #3076・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §21・§30）。
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
//!   構造は [`window_scatter_structure`]（窓を `(divisor, 窓要素数)` で分類し、
//!   `gather` → `1/s` → `scatter_add`〈`f64` 集約〉→ `s/d` で入力位置へ戻す索引の合成。
//!   除算前に集約して精度を保ち、`1/s` で中間和の overflow を避ける）で積む。
//!   §21.3 は `AvgPool2d` の構造を `Fold` 経由としていたが、Adaptive は窓が可変・
//!   3D には fold が無いため 3 variant を同一の索引合成へ統一した（逸脱。§30 に理由）。

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

/// 同一 divisor の窓が持つ（出力位置, 入力位置）の対応表。
#[derive(Default)]
struct ClassIndex {
    /// 窓に属する要素ごとの出力位置（`[0, out_spatial)`）。
    src: Vec<i32>,
    /// 同じく入力位置（`[0, in_spatial)`）。
    dst: Vec<i32>,
}

/// 窓構造の索引表（1 平面ぶん。全 `N·C` 平面で共通）。窓を divisor で
/// 分類して保持する（精度・overflow 方針は [`window_scatter_structure`] 参照）。
struct WindowPlan {
    in_spatial: usize,
    out_spatial: usize,
    classes: std::collections::BTreeMap<usize, ClassIndex>,
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
            classes: std::collections::BTreeMap::new(),
        })
    }

    /// 出力位置 `o` の窓（入力位置の列）と divisor を登録する。
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
        let class = self.classes.entry(divisor).or_default();
        for &m in members {
            class.src.push(o as i32);
            class.dst.push(m as i32);
        }
        Ok(())
    }

    /// 全分類の要素数合計。
    fn element_count(&self) -> usize {
        self.classes.values().map(|c| c.src.len()).sum()
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

/// 平均系 VJP の構造。`g`（`[N, C, 出力空間…]`）を `[N·C, 出力空間]` へ平坦化し、窓の
/// divisor `d` ごとに `zeros.scatter_add(dst, gather(g, src) · (1/s)) · (s/d)`
/// （`s = d` 以下の最大の 2 のべき乗。`s/d ∈ (0.5, 1]`）を作って足す。積む Op はすべて
/// replayable。窓要素数 `m ≤ d` の窓は同じ分類へ混在させる（`1/s` は窓要素数に依らない）。
///
/// 数値設計（codex-review 指摘・PR #3352。決定記録 §21.1(3)・(5)・§30.3）:
/// - **除算の前に窓内を集約する**: `g · (1/d)` を f32 で確定してから集約すると、相殺する上流
///   で丸め誤差が残り、孫テープ再生で 1 階のホスト値（`f64` 蓄積）と食い違う。`1/s` は 2 の
///   べき乗で厳密なので、`scatter_add` の `f64` 決定的集約で窓内の和を 1 回の丸めで作る。
/// - **集約前に `1/s` を掛ける**: 窓内の和が `m·|g|` まで膨らむのを `|g|` 程度に抑える。
/// - **重みは 1 以下**: 逆伝播で seed に掛かる `s/d` を増幅側（> 1）にしない。`s/d` が 1 超
///   だと有限な seed（例 `3e38`・`d = 3`）が最初の乗算で `inf` になる。逆向きの集約値は
///   `|seed|·m/d ≤ |seed|` に収まる。
/// - 残余: divisor が異なる分類の部分和は f32 で丸めてから足すため、桁違いの打ち消しが
///   起きる入力では 1 階値と食い違いうる。[`check_structure_contract`] がホストで構造を
///   模擬し、REQ-2 の範囲を外れる入力・f32 範囲を超える中間和は型付きエラーで拒否する。
fn window_scatter_structure<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    plan: &WindowPlan,
) -> Result<Var<'c>, AutodiffError> {
    let nc = input_shape[0]
        .checked_mul(input_shape[1])
        .ok_or_else(overflow)?;
    // N·C 平面ぶんへ複製する索引テンソルの総要素数（確保前に桁あふれ・巨大確保を拒否する）。
    let total_all = nc.checked_mul(plan.element_count()).ok_or_else(overflow)?;
    if total_all > isize::MAX as usize / (2 * std::mem::size_of::<i32>()) {
        return Err(overflow());
    }
    let g_flat = g.contiguous()?.reshape(&[nc, plan.out_spatial])?;
    let scalar = |v: f32| -> Result<Var<'c>, AutodiffError> {
        Ok(child.var_no_grad(&Tensor::new(vec![v], &[1, 1])?))
    };
    let mut acc: Option<Var<'c>> = None;
    for (&divisor, class) in &plan.classes {
        let e = class.src.len();
        let replicate = |one: &[i32]| -> Result<Tensor<i32>, AutodiffError> {
            let mut v = Vec::with_capacity(nc * e);
            for _ in 0..nc {
                v.extend_from_slice(one);
            }
            Ok(Tensor::new(v, &[nc, e])?)
        };
        let src_index = replicate(&class.src)?;
        let dst_index = replicate(&class.dst)?;
        let s = pow2_floor(divisor);
        let mut picked = g_flat.gather(1, &src_index)?;
        if s > 1 {
            picked = picked.mul(&scalar(1.0 / s as f32)?)?;
        }
        let zeros = child.var_no_grad(&Tensor::zeros(&[nc, plan.in_spatial])?);
        let mut part = zeros.scatter_add(1, &dst_index, &picked)?;
        let weight = (s as f64 / divisor as f64) as f32;
        if weight != 1.0 {
            part = part.mul(&scalar(weight)?)?;
        }
        acc = Some(match acc {
            Some(a) => a.add(&part)?,
            None => part,
        });
    }
    acc.ok_or_else(overflow)?.reshape(input_shape)
}

/// `d` 以下の最大の 2 のべき乗（`d ≥ 1`）。
fn pow2_floor(d: usize) -> usize {
    1usize << (usize::BITS - 1 - d.leading_zeros())
}

/// 構造（[`window_scatter_structure`]）をホストで模擬し、1 階のホスト値 `host` との REQ-2
/// 複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）と、集約前中間和の f32 範囲を検査する。
/// 範囲外は孫テープ再生で誤値が復活するため構築時に型付きエラーで拒否する（§27.3 と同型の
/// fail-closed。tolerance は変更しない）。
fn check_structure_contract(
    plan: &WindowPlan,
    g: &Tensor<f32>,
    host: &Tensor<f32>,
    input_shape: &[usize],
) -> Result<(), AutodiffError> {
    let nc = input_shape[0] * input_shape[1];
    let gv = g.host_slice();
    let hv = host.host_slice();
    let reject = |why: &str| invalid(format!("平均プーリングの構造合成が数値契約の範囲外: {why}"));
    for plane in 0..nc {
        let gp = &gv[plane * plan.out_spatial..(plane + 1) * plan.out_spatial];
        let mut total = vec![0.0f32; plan.in_spatial];
        for (&divisor, class) in &plan.classes {
            let s = pow2_floor(divisor);
            let mut sums = vec![0.0f64; plan.in_spatial];
            for (&o, &i) in class.src.iter().zip(&class.dst) {
                sums[i as usize] += f64::from(gp[o as usize]) / s as f64;
            }
            let weight = (s as f64 / divisor as f64) as f32;
            for (t, &sm) in total.iter_mut().zip(&sums) {
                if sm.abs() > f64::from(f32::MAX) {
                    return Err(reject("集約前の中間和が f32 の範囲を超える"));
                }
                *t += (sm as f32) * weight;
            }
        }
        let hp = &hv[plane * plan.in_spatial..(plane + 1) * plan.in_spatial];
        for (&sv, &h) in total.iter().zip(hp) {
            if h.is_nan() && sv.is_nan() {
                continue;
            }
            let d = (sv - h).abs();
            if h.is_finite() && !(d < 1e-5 || d / h.abs() < 1e-3) || h.is_finite() != sv.is_finite()
            {
                return Err(reject(
                    "分類間の f32 丸めが 1 階値と REQ-2 の範囲で一致しない",
                ));
            }
        }
    }
    Ok(())
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
    check_structure_contract(&plan, &value_of(child, g)?, &host, input_shape)?;
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
    check_structure_contract(&plan, &value_of(child, g)?, &host, input_shape)?;
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
    check_structure_contract(&plan, &value_of(child, g)?, &host, input_shape)?;
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

    /// 相殺する上流でも構造側（孫テープ再生で再計算される側）が 1 階のホスト値と一致する
    /// （codex-review 指摘・PR #3352。除算前に集約するため f32 積の丸めが残らない）。
    #[test]
    fn structure_keeps_cancellation_precision() {
        let tape = Tape::new_with_ops(naive_ops());
        let params = Pool2dParams::new([3, 3], Some([1, 1]), [0, 0], [1, 1]).unwrap();
        let ishape = [1usize, 1, 3, 5];
        let oshape = [1usize, 1, 1, 3];
        let g = tape.var(&Tensor::new(vec![1.0e8f32, 1.0e8, -199_999_984.0], &oshape).unwrap());
        let plan = plan_avg2d(&ishape, &oshape, &params, true).unwrap();
        let s = window_scatter_structure(&tape, &g, &ishape, &plan).unwrap();
        let h = avg_pool2d_vjp(&value_of(&tape, &g).unwrap(), &ishape, &params, true).unwrap();
        close(&s.to_tensor().host_slice(), &h.host_slice());
    }

    /// 窓内の和が f32 上限を超える入力でも構造側が有限で、二階の逆伝播（`gather` の VJP が
    /// 窓内の seed を集約する側）の中間和も overflow しない。
    #[test]
    fn structure_avoids_window_sum_overflow_forward_and_backward() {
        let tape = Tape::new_with_ops(naive_ops());
        let params = Pool2dParams::new([2, 2], Some([1, 1]), [0, 0], [1, 1]).unwrap();
        let ishape = [1usize, 1, 3, 3];
        let oshape = [1usize, 1, 2, 2];
        let g = tape.var(&Tensor::new(vec![1.0e38f32; 4], &oshape).unwrap());
        let plan = plan_avg2d(&ishape, &oshape, &params, true).unwrap();
        let s = window_scatter_structure(&tape, &g, &ishape, &plan).unwrap();
        // 中央は 4 窓が重なる: 真値は 4 · 1e38 / 4 = 1e38。
        assert!(s.to_tensor().host_slice().iter().all(|v| v.is_finite()));
        assert!((s.to_tensor().host_slice()[4] - 1.0e38).abs() < 1e31);
        // 逆向き: 各出力窓 4 要素へ seed 1e38 → d g[o] = Σ seed/4 = 1e38。
        let seed = tape.var_no_grad(&Tensor::new(vec![1.0e38f32; 9], &ishape).unwrap());
        let loss = s.mul(&seed).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dg = grads.get(&g).unwrap().expect("g");
        assert!(dg.host_slice().iter().all(|v| v.is_finite()));
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

    /// 逆伝播で seed を増幅しない（重み ≤ 1）: `d = 3` の平均で有限な seed `3e38` が有限のまま。
    #[test]
    fn backward_seed_is_not_amplified() {
        let tape = Tape::new_with_ops(naive_ops());
        let params = Pool2dParams::new([1, 3], Some([1, 1]), [0, 0], [1, 1]).unwrap();
        let ishape = [1usize, 1, 1, 4];
        let oshape = [1usize, 1, 1, 2];
        let g = tape.var(&Tensor::new(vec![1.0f32, 1.0], &oshape).unwrap());
        let plan = plan_avg2d(&ishape, &oshape, &params, true).unwrap();
        let s = window_scatter_structure(&tape, &g, &ishape, &plan).unwrap();
        let seed = tape.var_no_grad(&Tensor::new(vec![3.0e38f32; 4], &ishape).unwrap());
        let loss = s.mul(&seed).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dg = grads.get(&g).unwrap().expect("g");
        assert!(
            dg.host_slice().iter().all(|v| v.is_finite()),
            "{:?}",
            dg.host_slice()
        );
    }

    /// 分類間の f32 丸めで 1 階値から外れる入力は型付きエラーで拒否される。
    #[test]
    fn cross_class_rounding_is_rejected() {
        let tape = Tape::new_with_ops(naive_ops());
        let params = Pool2dParams::new([1, 3], Some([1, 1]), [0, 1], [1, 1]).unwrap();
        let ishape = [1usize, 1, 1, 3];
        let oshape = [1usize, 1, 1, 3];
        let g =
            tape.var(&Tensor::new(vec![100_000_008.0f32, -150_000_016.0, 0.0], &oshape).unwrap());
        let r = avg_pool2d_cgrad(&tape, &g, &ishape, &params, false);
        assert!(r.is_err());
    }
}
