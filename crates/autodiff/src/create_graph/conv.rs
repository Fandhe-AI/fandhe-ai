//! 2D 畳み込み族（`Conv2d`・`ConvTranspose2d`・`Unfold`・`Fold`）の
//! `create_graph` 対応（イシュー #3080・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §21・§22）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）と
//! `build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。本モジュールは
//! 「子テープへ積む Op を、すべて `supports_create_graph() == true` な
//! 構造 Op（`Narrow`・`Reshape`・`Permute`・`Contiguous`・`Concat`・rank 2
//! `MatMul`・`Unfold`・`Fold`・`Sum`）に限る」ことで閉包性（子テープ上の
//! 勾配をさらに `backward_create_graph` で微分できること。§21.1(4)）を保つ。
//!
//! **方式**（1 階 `grad.rs` の `Op::Conv2d`／`Op::ConvTranspose2d` VJP と同じ式）:
//! - d_input は **V1**: `unfold`／`fold`（groups = 1 の `Var` 演算。K 軸
//!   `(c, kh, kw)` の並びは groups 付き im2col の平坦化と一致する）と
//!   groups ごとの rank 2 `Var::matmul_fp32_strict` の合成。
//!   `Var::conv2d` を使わないのは forward が非 strict の `gemm_batched`
//!   を通り、backprop の FP32 厳密契約（CUDA TF32 opt-in 下でも不変）を
//!   破るため。
//! - d_weight／d_bias は **V3**: 構造（二階微分の経路）は上記合成で積み、
//!   ノードの**前方値**は 1 階 VJP と同じホスト関数
//!   （`grad::conv2d_dw_from_col`／`conv_transpose2d_dw_from_col`・
//!   `eval::reduce_bias_grad_rows`）が返す値を `Op::Contiguous`（VJP は
//!   恒等で前方値を読まない）のタグで `push_eager` する。これにより N·P 方向
//!   の長軸縮約が 1 階と同じ `f64` アキュムレータ契約（`.claude/rules/
//!   coding-rust.md`）のまま 1 階と `cg.grad` が bit 一致する。
//! - `output_padding`／`output_size` は `Op` に保持せず、再生対象ノード自身の
//!   shape から導出する（`Op::ConvTranspose2d` doc の方針を踏襲）。

use fandhe_ai_tensor_core::{Conv2dParams, ScalarDType, Tensor, im2col_out_shape};

use super::{reduce_bias_grad_var, value_of};
use crate::error::AutodiffError;
use crate::grad::{conv_transpose2d_dw_from_col, conv2d_dw_from_col, im2col_with_fallback};
use crate::tape::{Op, Tape};
use crate::var::Var;

/// 3 つの寄与のうち、親テープ側で `requires_grad` な（= 必要な）もの。
/// 不要な寄与は構造ノードもホスト計算も作らない。
#[derive(Clone, Copy)]
pub(super) struct ConvNeeds {
    pub(super) input: bool,
    pub(super) weight: bool,
    pub(super) bias: bool,
}

/// 子テープ上の VJP 結果（不要・未定義のものは `None`）。
pub(super) struct ConvGrads<'c> {
    pub(super) d_input: Option<Var<'c>>,
    pub(super) d_weight: Option<Var<'c>>,
    pub(super) d_bias: Option<Var<'c>>,
}

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

// ---------------------------------------------------------------------
// forward 再生（replay_op から呼ばれる。公開 `Var` メソッドへの薄い委譲）
// ---------------------------------------------------------------------

/// `Op::Conv2d` の再生。低精度は子テープへ精度情報を持たないため拒否する
/// （`validate_ancestors` と同条件の多層防御）。
pub(super) fn replay_conv2d<'c>(
    input: &Var<'c>,
    weight: &Var<'c>,
    bias: Option<&Var<'c>>,
    params: &Conv2dParams,
    compute_dtype: ScalarDType,
) -> Result<Var<'c>, AutodiffError> {
    if compute_dtype != ScalarDType::F32 {
        return Err(invalid(format!(
            "低精度 conv2d（{compute_dtype:?}）は子テープへ精度情報を持たず静かに f32 再生される\
             ため非対応（イシュー #2071・#3080）"
        )));
    }
    input.conv2d(
        weight,
        bias,
        params.stride(),
        params.padding(),
        params.dilation(),
        params.groups(),
    )
}

/// `Op::ConvTranspose2d` の再生。`output_padding` は出力 shape から導出する:
/// `op = out − ((in−1)·s − 2p + d·(k−1) + 1)`（各軸。`checked_*`）。
pub(super) fn replay_conv_transpose2d<'c>(
    out_shape: &[usize],
    input: &Var<'c>,
    weight: &Var<'c>,
    bias: Option<&Var<'c>>,
    params: &Conv2dParams,
) -> Result<Var<'c>, AutodiffError> {
    let in_shape = input.shape();
    if in_shape.len() != 4 || out_shape.len() != 4 {
        return Err(invalid(
            "conv_transpose2d の再生: rank 4 でない shape（内部契約違反）".into(),
        ));
    }
    let mut output_padding = [0usize; 2];
    for (i, slot) in output_padding.iter_mut().enumerate() {
        let (h_in, k, s, p, d) = (
            in_shape[2 + i],
            params.kernel_size()[i],
            params.stride()[i],
            params.padding()[i],
            params.dilation()[i],
        );
        let base = h_in
            .checked_sub(1)
            .and_then(|v| v.checked_mul(s))
            .and_then(|v| v.checked_add(d.checked_mul(k.checked_sub(1)?)?))
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_sub(p.checked_mul(2)?))
            .ok_or_else(|| invalid("conv_transpose2d の再生: 出力長の導出が桁あふれ／負".into()))?;
        *slot = out_shape[2 + i]
            .checked_sub(base)
            .ok_or_else(|| invalid("conv_transpose2d の再生: output_padding が負".into()))?;
    }
    input.conv_transpose2d(
        weight,
        bias,
        params.stride(),
        params.padding(),
        output_padding,
        params.dilation(),
        params.groups(),
    )
}

/// `Op::Unfold` の再生。
pub(super) fn replay_unfold<'c>(
    input: &Var<'c>,
    params: &Conv2dParams,
) -> Result<Var<'c>, AutodiffError> {
    input.unfold(
        params.kernel_size(),
        params.stride(),
        params.padding(),
        params.dilation(),
    )
}

/// `Op::Fold` の再生。`output_size` は再生対象ノード自身の shape から取る。
pub(super) fn replay_fold<'c>(
    out_shape: &[usize],
    input: &Var<'c>,
    params: &Conv2dParams,
) -> Result<Var<'c>, AutodiffError> {
    if out_shape.len() != 4 {
        return Err(invalid(
            "fold の再生: rank 4 でない出力 shape（内部契約違反）".into(),
        ));
    }
    input.fold(
        [out_shape[2], out_shape[3]],
        params.kernel_size(),
        params.stride(),
        params.padding(),
        params.dilation(),
    )
}

// ---------------------------------------------------------------------
// VJP（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

/// `Op::Unfold` の VJP（d_x = `fold`。1 階 VJP の `col2im` と同一関数）。
pub(super) fn unfold_vjp<'c>(
    g: &Var<'c>,
    input_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Var<'c>, AutodiffError> {
    if input_shape.len() != 4 {
        return Err(invalid("unfold の VJP: rank 4 でない入力 shape".into()));
    }
    g.fold(
        [input_shape[2], input_shape[3]],
        params.kernel_size(),
        params.stride(),
        params.padding(),
        params.dilation(),
    )
}

/// `Op::Fold` の VJP（d_x = `unfold`。純コピー）。
pub(super) fn fold_vjp<'c>(g: &Var<'c>, params: &Conv2dParams) -> Result<Var<'c>, AutodiffError> {
    g.unfold(
        params.kernel_size(),
        params.stride(),
        params.padding(),
        params.dilation(),
    )
}

fn cat_or_single<'c>(parts: Vec<Var<'c>>, dim: usize) -> Result<Var<'c>, AutodiffError> {
    if parts.len() == 1 {
        Ok(parts[0])
    } else {
        Var::cat(&parts, dim)
    }
}

/// `v: [N, C, ...]` の軸 1 を `[c0, c0+len)` で切り出し `[len, N·P]` へ並べ替える
/// （N 軸を列軸へ畳み込む。子テープの matmul は rank 2 限定のため）。
fn channel_rows<'c>(
    v: &Var<'c>,
    c0: usize,
    len: usize,
    n: usize,
    p: usize,
) -> Result<Var<'c>, AutodiffError> {
    v.narrow(1, c0, len)?
        .contiguous()?
        .reshape(&[n, len, p])?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[len, n * p])
}

/// `col: [N, G·K_g, P]` の group `gi` を `[N·P, K_g]` へ並べ替える。
fn col_np_by_k<'c>(
    col: &Var<'c>,
    gi: usize,
    k_g: usize,
    n: usize,
    p: usize,
) -> Result<Var<'c>, AutodiffError> {
    col.narrow(1, gi * k_g, k_g)?
        .permute(&[0, 2, 1])?
        .contiguous()?
        .reshape(&[n * p, k_g])
}

/// `[rows, N·P]` の matmul 結果を `[N, rows, P]` へ戻す。
fn np_cols_to_n_rows_p<'c>(
    m: &Var<'c>,
    rows: usize,
    n: usize,
    p: usize,
) -> Result<Var<'c>, AutodiffError> {
    m.contiguous()?
        .reshape(&[rows, n, p])?
        .permute(&[1, 0, 2])?
        .contiguous()
}

fn zeros_var<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::zeros(shape)?))
}

/// 要素数 0 の入力（`N = 0` 等）: 寄与は恒等的に 0（二階も 0 で整合）。
/// matmul を呼ばずゼロ定数で短絡する。
fn zero_grads<'c>(
    child: &'c Tape,
    x_shape: &[usize],
    w_shape: &[usize],
    bias_shape: Option<&[usize]>,
    need: ConvNeeds,
) -> Result<ConvGrads<'c>, AutodiffError> {
    Ok(ConvGrads {
        d_input: if need.input {
            Some(zeros_var(child, x_shape)?)
        } else {
            None
        },
        d_weight: if need.weight {
            Some(zeros_var(child, w_shape)?)
        } else {
            None
        },
        d_bias: match (need.bias, bias_shape) {
            (true, Some(b)) => Some(zeros_var(child, b)?),
            _ => None,
        },
    })
}

/// d_bias（`Conv2d`／`ConvTranspose2d` 共通）。値は 1 階と同じ
/// `eval::reduce_bias_grad_rows`（[`reduce_bias_grad_var`] が `Op::Sum` タグで登録）。
fn bias_grad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    bias_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let g_shape = g.shape();
    let p = g_shape[2]
        .checked_mul(g_shape[3])
        .ok_or_else(|| invalid("bias 勾配: 空間要素数の桁あふれ".into()))?;
    let rows = g
        .permute(&[0, 2, 3, 1])?
        .contiguous()?
        .reshape(&[g_shape[0] * p, g_shape[1]])?;
    reduce_bias_grad_var(child, &rows, bias_shape)
}

/// V3: 構造ノード `structure`（d_weight の二階微分経路）の前方値を、1 階と同じ
/// ホスト関数の値 `value` で置き換えたノードを返す。`Op::Contiguous` の VJP は
/// 恒等で前方値を読まないため微分値は構造側と一致する（`reduce_bias_grad_var` と同型）。
fn v3_node<'c>(
    child: &'c Tape,
    structure: &Var<'c>,
    value: Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    if value.shape() != structure.shape().as_slice() {
        return Err(invalid(
            "V3 ノードの値 shape が構造と不一致（内部契約違反）".into(),
        ));
    }
    let id = child.push_eager(
        Op::Contiguous {
            input: structure.node_id(),
        },
        value,
    );
    Ok(Var::from_raw(child, id))
}

/// `Op::Conv2d` の VJP を子テープへ記録する。`g`: `[N, Cout, Hout, Wout]`。
pub(super) fn conv2d_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x: &Var<'c>,
    w: &Var<'c>,
    bias_shape: Option<&[usize]>,
    params: &Conv2dParams,
    need: ConvNeeds,
) -> Result<ConvGrads<'c>, AutodiffError> {
    let (x_shape, w_shape, g_shape) = (x.shape(), w.shape(), g.shape());
    if x_shape.len() != 4 || w_shape.len() != 4 || g_shape.len() != 4 {
        return Err(invalid(
            "conv2d の VJP: rank 4 でない shape（内部契約違反）".into(),
        ));
    }
    if [&x_shape, &w_shape, &g_shape]
        .iter()
        .any(|s| s.contains(&0))
    {
        return zero_grads(child, &x_shape, &w_shape, bias_shape, need);
    }
    let groups = params.groups();
    let cout_g = w_shape[0] / groups.max(1);
    let k_g = w_shape[1] * w_shape[2] * w_shape[3];
    let n = g_shape[0];
    let p = g_shape[2]
        .checked_mul(g_shape[3])
        .ok_or_else(|| invalid("conv2d の VJP: 空間要素数の桁あふれ".into()))?;

    let mut out = ConvGrads {
        d_input: None,
        d_weight: None,
        d_bias: None,
    };

    if need.input {
        // d_col[g] = w_gᵀ · g_g → [N, G·K_g, P] → fold（col2im）。
        let mut parts = Vec::with_capacity(groups);
        for gi in 0..groups {
            let w_g = w
                .narrow(0, gi * cout_g, cout_g)?
                .contiguous()?
                .reshape(&[cout_g, k_g])?
                .transpose(0, 1)?;
            let g_g = channel_rows(g, gi * cout_g, cout_g, n, p)?;
            let cols = w_g.matmul_fp32_strict(&g_g)?;
            parts.push(np_cols_to_n_rows_p(&cols, k_g, n, p)?);
        }
        let d_col = cat_or_single(parts, 1)?;
        out.d_input = Some(d_col.fold(
            [x_shape[2], x_shape[3]],
            params.kernel_size(),
            params.stride(),
            params.padding(),
            params.dilation(),
        )?);
    }

    if need.weight {
        // 構造: dw[g] = g_g · col_g（[Cout_g, K_g]）。
        let col = x.unfold(
            params.kernel_size(),
            params.stride(),
            params.padding(),
            params.dilation(),
        )?;
        let mut parts = Vec::with_capacity(groups);
        for gi in 0..groups {
            let g_g = channel_rows(g, gi * cout_g, cout_g, n, p)?;
            let col_g = col_np_by_k(&col, gi, k_g, n, p)?;
            parts.push(g_g.matmul_fp32_strict(&col_g)?);
        }
        let structure = cat_or_single(parts, 0)?.contiguous()?.reshape(&w_shape)?;
        // 値: 1 階 VJP と同じホスト関数（f64 縮約）。
        let (g_val, x_val) = (value_of(child, g)?, value_of(child, x)?);
        let im2col_shape = im2col_out_shape(&x_shape, params)?;
        let col_val = im2col_with_fallback(child.ops(), &x_val, params, &im2col_shape)?;
        let g4 = g_val.contiguous().reshape(&[n, groups, cout_g, p])?;
        let value = conv2d_dw_from_col(child.ops(), &g4, &col_val, groups, cout_g, k_g, &w_shape)?;
        out.d_weight = Some(v3_node(child, &structure, value)?);
    }

    if need.bias
        && let Some(b) = bias_shape
    {
        out.d_bias = Some(bias_grad(child, g, b)?);
    }
    Ok(out)
}

/// `Op::ConvTranspose2d` の VJP を子テープへ記録する。`x`: `[N, Cin, H, W]`、
/// `w`: `[Cin, Cout_g, kH, kW]`、`g`: `[N, Cout, Hout, Wout]`。
pub(super) fn conv_transpose2d_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x: &Var<'c>,
    w: &Var<'c>,
    bias_shape: Option<&[usize]>,
    params: &Conv2dParams,
    need: ConvNeeds,
) -> Result<ConvGrads<'c>, AutodiffError> {
    let (x_shape, w_shape, g_shape) = (x.shape(), w.shape(), g.shape());
    if x_shape.len() != 4 || w_shape.len() != 4 || g_shape.len() != 4 {
        return Err(invalid(
            "conv_transpose2d の VJP: rank 4 でない shape（内部契約違反）".into(),
        ));
    }
    if [&x_shape, &w_shape, &g_shape]
        .iter()
        .any(|s| s.contains(&0))
    {
        return zero_grads(child, &x_shape, &w_shape, bias_shape, need);
    }
    let groups = params.groups();
    let cin_g = x_shape[1] / groups.max(1);
    let k_g = w_shape[1] * w_shape[2] * w_shape[3];
    let n = x_shape[0];
    let p_x = x_shape[2]
        .checked_mul(x_shape[3])
        .ok_or_else(|| invalid("conv_transpose2d の VJP: 空間要素数の桁あふれ".into()))?;

    let mut out = ConvGrads {
        d_input: None,
        d_weight: None,
        d_bias: None,
    };

    if need.input || need.weight {
        // col = im2col(g)（groups = 1 の unfold。K 軸は groups 付き im2col の平坦化と一致）。
        let col = g.unfold(
            params.kernel_size(),
            params.stride(),
            params.padding(),
            params.dilation(),
        )?;

        if need.input {
            // d_x[g] = w_g · col_g（[Cin_g, N·P_x]）。
            let mut parts = Vec::with_capacity(groups);
            for gi in 0..groups {
                let w_g = w
                    .narrow(0, gi * cin_g, cin_g)?
                    .contiguous()?
                    .reshape(&[cin_g, k_g])?;
                let c_g = channel_rows(&col, gi * k_g, k_g, n, p_x)?;
                let prod = w_g.matmul_fp32_strict(&c_g)?;
                parts.push(np_cols_to_n_rows_p(&prod, cin_g, n, p_x)?);
            }
            out.d_input = Some(cat_or_single(parts, 1)?.contiguous()?.reshape(&x_shape)?);
        }

        if need.weight {
            // 構造: dw[g] = x_g · col_g（[Cin_g, K_g]。col は [N·P_x, K_g] 並び）。
            let x3 = x.contiguous()?.reshape(&[n, x_shape[1], p_x])?;
            let mut parts = Vec::with_capacity(groups);
            for gi in 0..groups {
                let x_g = channel_rows(&x3, gi * cin_g, cin_g, n, p_x)?;
                let c_g = col_np_by_k(&col, gi, k_g, n, p_x)?;
                parts.push(x_g.matmul_fp32_strict(&c_g)?);
            }
            let structure = cat_or_single(parts, 0)?.contiguous()?.reshape(&w_shape)?;
            let (g_val, x_val) = (value_of(child, g)?, value_of(child, x)?);
            let im2col_shape = im2col_out_shape(&g_shape, params)?;
            let col_g_val =
                im2col_with_fallback(child.ops(), &g_val.contiguous(), params, &im2col_shape)?;
            let x4 = x_val.contiguous().reshape(&[n, groups, cin_g, p_x])?;
            let value = conv_transpose2d_dw_from_col(
                child.ops(),
                &x4,
                &col_g_val,
                groups,
                cin_g,
                k_g,
                &w_shape,
            )?;
            out.d_weight = Some(v3_node(child, &structure, value)?);
        }
    }

    if need.bias
        && let Some(b) = bias_shape
    {
        out.d_bias = Some(bias_grad(child, g, b)?);
    }
    Ok(out)
}
