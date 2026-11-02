//! 3D 畳み込み族（`Conv3d`・`ConvTranspose3d` と内部 Op `Unfold3d`・`Fold3d`・
//! `ConvWeightGradHost3d`）の `create_graph` 対応（イシュー #3797・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §38・§59）。[`super::conv`]（2D 版。§21・§23）の
//! 空間 3 軸一般化で、方式・式・fail-closed の方針は 2D 版と同一。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）と `build_cgrads`
//! （1 階 VJP の子テープ記録）から呼ばれる。子テープへ積む Op は、すべて
//! `supports_create_graph() == true` な構造 Op（`Narrow`・`Reshape`・`Permute`・`Contiguous`・
//! `Concat`・rank 2 `MatMul`・`Unfold3d`・`Fold3d`・`Sum`・`ConvWeightGradHost3d`）に限り、閉包性
//! （子テープ上の勾配をさらに `backward_create_graph` で微分できること。§21.1(4)）を保つ。
//!
//! **方式**（1 階 `grad.rs` の `Op::Conv3d`／`Op::ConvTranspose3d` VJP と同じ式）:
//! - d_input は **V1**: `unfold3d`／`fold3d`（groups = 1 の im2col3d／col2im3d。K 軸 `(c, kd, kh, kw)`
//!   の並びは groups 付き im2col3d の平坦化と一致する）と groups ごとの rank 2
//!   `Var::matmul_fp32_strict` の合成。`Var::conv3d` を合成に使わないのは、forward が非 strict の
//!   `gemm_batched` と `ops.conv3d` フックを通り、backprop の FP32 厳密契約（CUDA TF32 opt-in 下でも
//!   不変）を破るため（再生〔`replay_conv3d`〕だけは親の forward と同じ経路が必要なので使う）。
//! - d_weight は **V3**: 構造（二階微分経路）は上記合成で積み、ノードの前方値は 1 階 VJP と同じ
//!   ホスト関数（`grad::conv3d_dw_from_col`／`conv_transpose3d_dw_from_col`）が返す値を
//!   `Op::ConvWeightGradHost3d` として `push_eager` する。孫テープでの再生でも同じ関数で値を再計算する
//!   ため、N·P 方向の長軸縮約の `f64` アキュムレータ契約（`.claude/rules/coding-rust.md`）が何段目でも
//!   維持される。VJP は恒等で構造側のみへ渡す。
//! - d_bias は **V3**: `eval::reduce_bias_grad_rows`（[`reduce_bias_grad_var`] が `Op::Sum` タグで登録）。
//!   入力・重みが空でも上流 `g` が非空なら計算する。
//! - `output_padding`／`output_size` は `Op` に保持せず、再生対象ノード自身の shape から導出する。
//!
//! **GPU 経路**: `im2col3d`／`col2im3d` は CUDA・Metal・ROCm で override されておらずホストへ
//! フォールバックする（GEMM のみデバイス。§38.12）。GPU 実機 parity は Phase 10（#3857・#3859）で測る。
//!
//! **メモリ量の見積り**（上限は設けない。#3767 で承認。§38.12）: 入力 1 本の展開サイズを
//! `C_x = 4·N·Cin·kD·kH·kW·P` バイト（`P` は出力の空間要素数）とすると、子テープは約 4〜5·C_x、
//! d_weight の値の再計算で一時的に +1·C_x、孫テープではさらに加算される。巨大入力の確保前には
//! 要素数・確保長を `checked_mul` で検査し、桁あふれは型付きエラーで拒否する。

use fandhe_ai_tensor_core::{Conv3dParams, Tensor, im2col3d_out_shape};

use super::conv::{
    ConvGrads, ConvNeeds, cat_or_single, channel_rows, col_np_by_k, invalid, np_cols_to_n_rows_p,
    zeros_var,
};
use super::{reduce_bias_grad_var, value_of};
use crate::error::AutodiffError;
use crate::grad::{
    col2im3d_with_fallback, conv_transpose3d_dw_from_col, conv3d_dw_from_col,
    im2col3d_with_fallback, unfold3d_out_shape,
};
use crate::tape::{Op, Tape};
use crate::var::Var;

/// 各軸の積を `checked_mul` で求める（桁あふれは型付きで拒否）。
fn checked_dims_product(dims: &[usize], what: &str) -> Result<usize, AutodiffError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| invalid(format!("{what}: 要素数の桁あふれ")))
}

/// 同じ畳み込み設定の groups = 1 版（`unfold3d`／`fold3d` は常に groups = 1）。
fn params_g1(params: &Conv3dParams) -> Result<Conv3dParams, AutodiffError> {
    Conv3dParams::new(
        params.kernel_size(),
        params.stride(),
        params.padding(),
        params.dilation(),
        1,
    )
    .map_err(AutodiffError::Backend)
}

// ---------------------------------------------------------------------
// 内部 Op の入口（子テープ上の groups = 1 im2col3d／col2im3d）
// ---------------------------------------------------------------------

/// `[N, C, D, H, W]` → `[N, C·kvol, P]`（groups = 1 の im2col3d。純コピー）を子テープへ積む。
/// 検査（rank・窓数・確保サイズ）は実体化と push より前に行い、失敗時に孤児ノードを残さない。
pub(super) fn unfold3d_var<'c>(
    x: &Var<'c>,
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    let p1 = params_g1(params)?;
    let tape = x.tape();
    let x_shape = x.shape();
    let out_shape = unfold3d_out_shape(&x_shape, &p1)?;
    let value = if out_shape.contains(&0) || x_shape.contains(&0) {
        Tensor::zeros(&out_shape).map_err(AutodiffError::Shape)?
    } else {
        let x_val = value_of(tape, x)?;
        let col_shape = [out_shape[0], 1, out_shape[1], out_shape[2]];
        im2col3d_with_fallback(tape.ops(), &x_val.contiguous(), &p1, &col_shape)?
            .reshape(&out_shape)
            .map_err(AutodiffError::Shape)?
    };
    let id = tape.push_eager(
        Op::Unfold3d {
            input: x.node_id(),
            params: p1,
        },
        value,
    );
    Ok(Var::from_raw(tape, id))
}

/// `[N, C·kvol, P]` → `[N, C, D, H, W]`（groups = 1 の col2im3d。重なる窓は `f64` で加算）を
/// 子テープへ積む。`out_shape` は出力 shape（rank 5）。
pub(super) fn fold3d_var<'c>(
    col: &Var<'c>,
    out_shape: &[usize],
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    let p1 = params_g1(params)?;
    let tape = col.tape();
    let col_shape = col.shape();
    let expected = unfold3d_out_shape(out_shape, &p1)?;
    if col_shape != expected {
        return Err(invalid(format!(
            "fold3d: 入力 {col_shape:?} が出力 {out_shape:?} から導出した {expected:?} と一致しない"
        )));
    }
    let value = if expected.contains(&0) || out_shape.contains(&0) {
        Tensor::zeros(out_shape).map_err(AutodiffError::Shape)?
    } else {
        let d_col = value_of(tape, col)?
            .contiguous()
            .reshape(&[col_shape[0], 1, col_shape[1], col_shape[2]])
            .map_err(AutodiffError::Shape)?;
        col2im3d_with_fallback(tape.ops(), &d_col, out_shape, &p1)?
    };
    let id = tape.push_eager(
        Op::Fold3d {
            input: col.node_id(),
            params: p1,
        },
        value,
    );
    Ok(Var::from_raw(tape, id))
}

// ---------------------------------------------------------------------
// forward 再生（replay_op から呼ばれる）
// ---------------------------------------------------------------------

/// `Op::Unfold3d` の再生。
pub(super) fn replay_unfold3d<'c>(
    input: &Var<'c>,
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    unfold3d_var(input, params)
}

/// `Op::Fold3d` の再生。出力 shape は再生対象ノード自身の shape から取る。
pub(super) fn replay_fold3d<'c>(
    out_shape: &[usize],
    input: &Var<'c>,
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    if out_shape.len() != 5 {
        return Err(invalid(
            "fold3d の再生: rank 5 でない出力 shape（内部契約違反）".into(),
        ));
    }
    fold3d_var(input, out_shape, params)
}

/// `Op::Conv3d` の再生（公開 `Var::conv3d` への委譲。低精度 conv3d は存在しない）。
pub(super) fn replay_conv3d<'c>(
    input: &Var<'c>,
    weight: &Var<'c>,
    bias: Option<&Var<'c>>,
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    input.conv3d(
        weight,
        bias,
        params.stride(),
        params.padding(),
        params.dilation(),
        params.groups(),
    )
}

/// `Op::ConvTranspose3d` の再生。`output_padding` は出力 shape から導出する:
/// `op = out + 2p − ((in−1)·s + d·(k−1) + 1)`（各軸。`i128` の符号付き `checked_*`）。負・桁あふれ・
/// `>= stride` は拒否する。
pub(super) fn replay_conv_transpose3d<'c>(
    out_shape: &[usize],
    input: &Var<'c>,
    weight: &Var<'c>,
    bias: Option<&Var<'c>>,
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    let in_shape = input.shape();
    if in_shape.len() != 5 || out_shape.len() != 5 {
        return Err(invalid(
            "conv_transpose3d の再生: rank 5 でない shape（内部契約違反）".into(),
        ));
    }
    let mut output_padding = [0usize; 3];
    for (i, slot) in output_padding.iter_mut().enumerate() {
        let (h_in, k, s, p, d) = (
            in_shape[2 + i],
            params.kernel_size()[i],
            params.stride()[i],
            params.padding()[i],
            params.dilation()[i],
        );
        let to_i = |v: usize| i128::try_from(v).ok();
        let derived = (|| {
            let pos = to_i(h_in)?
                .checked_sub(1)?
                .checked_mul(to_i(s)?)?
                .checked_add(to_i(d)?.checked_mul(to_i(k)?.checked_sub(1)?)?)?
                .checked_add(1)?;
            let op = to_i(out_shape[2 + i])?
                .checked_add(to_i(p)?.checked_mul(2)?)?
                .checked_sub(pos)?;
            usize::try_from(op).ok()
        })();
        let op = derived.ok_or_else(|| {
            invalid("conv_transpose3d の再生: output_padding の導出が桁あふれ／負".into())
        })?;
        if op >= s {
            return Err(invalid(format!(
                "conv_transpose3d の再生: 導出した output_padding[{i}]={op} が stride={s} 以上"
            )));
        }
        *slot = op;
    }
    input.conv_transpose3d(
        weight,
        bias,
        params.stride(),
        params.padding(),
        output_padding,
        params.dilation(),
        params.groups(),
    )
}

// ---------------------------------------------------------------------
// VJP（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

/// `Op::Unfold3d` の VJP（d_x = `fold3d`）。
pub(super) fn unfold3d_vjp<'c>(
    g: &Var<'c>,
    input_shape: &[usize],
    params: &Conv3dParams,
) -> Result<Var<'c>, AutodiffError> {
    if input_shape.len() != 5 {
        return Err(invalid("unfold3d の VJP: rank 5 でない入力 shape".into()));
    }
    fold3d_var(g, input_shape, params)
}

/// `Op::Fold3d` の VJP（d_x = `unfold3d`。純コピー）。
pub(super) fn fold3d_vjp<'c>(g: &Var<'c>, params: &Conv3dParams) -> Result<Var<'c>, AutodiffError> {
    unfold3d_var(g, params)
}

/// d_bias（`Conv3d`／`ConvTranspose3d` 共通）。値は 1 階と同じ `eval::reduce_bias_grad_rows`。
fn bias_grad3d<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    bias_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let g_shape = g.shape();
    if g_shape.len() != 5 {
        return Err(invalid("bias 勾配: rank 5 でない上流 shape".into()));
    }
    let p = checked_dims_product(&g_shape[2..], "bias 勾配（空間）")?;
    let rows_n = g_shape[0]
        .checked_mul(p)
        .ok_or_else(|| invalid("bias 勾配: 行数の桁あふれ".into()))?;
    let rows = g
        .permute(&[0, 2, 3, 4, 1])?
        .contiguous()?
        .reshape(&[rows_n, g_shape[1]])?;
    reduce_bias_grad_var(child, &rows, bias_shape)
}

/// 要素数 0 の入力・重み・上流: d_input・d_weight は恒等的に 0。d_bias は上流 `g` が非空なら
/// 通常どおり縮約する（shape に 0 を含むときは `unfold3d_var` を呼ばずここで短絡する）。
fn zero_grads3d<'c>(
    child: &'c Tape,
    g: &Var<'c>,
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
            (true, Some(b)) if g.shape().contains(&0) => Some(zeros_var(child, b)?),
            (true, Some(b)) => Some(bias_grad3d(child, g, b)?),
            _ => None,
        },
    })
}

/// 1 階 VJP と同じホスト関数で d_weight の値を求める。`g`・`x` は子テープ上の写し
/// （`transposed == false` は `Conv3d`、`true` は `ConvTranspose3d`）。
fn weight_grad_value3d(
    child: &Tape,
    g: &Var<'_>,
    x: &Var<'_>,
    params: &Conv3dParams,
    transposed: bool,
    w_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    let (x_shape, g_shape) = (x.shape(), g.shape());
    if x_shape.len() != 5 || g_shape.len() != 5 || w_shape.len() != 5 {
        return Err(invalid("d_weight の値計算: rank 5 でない shape".into()));
    }
    let groups = params.groups().max(1);
    let k_g = checked_dims_product(&w_shape[1..], "d_weight（K_g）")?;
    let (g_val, x_val) = (value_of(child, g)?, value_of(child, x)?);
    if transposed {
        let n = x_shape[0];
        let p_x = checked_dims_product(&x_shape[2..], "conv_transpose3d の VJP（空間）")?;
        let cin_g = x_shape[1] / groups;
        let im2col_shape = im2col3d_out_shape(&g_shape, params)?;
        let col_g_val =
            im2col3d_with_fallback(child.ops(), &g_val.contiguous(), params, &im2col_shape)?;
        let x5 = x_val.contiguous().reshape(&[n, groups, cin_g, p_x])?;
        conv_transpose3d_dw_from_col(child.ops(), &x5, &col_g_val, groups, cin_g, k_g, w_shape)
    } else {
        let n = g_shape[0];
        let p = checked_dims_product(&g_shape[2..], "conv3d の VJP（空間）")?;
        let cout_g = w_shape[0] / groups;
        let im2col_shape = im2col3d_out_shape(&x_shape, params)?;
        let col_val = im2col3d_with_fallback(child.ops(), &x_val, params, &im2col_shape)?;
        let g5 = g_val.contiguous().reshape(&[n, groups, cout_g, p])?;
        conv3d_dw_from_col(child.ops(), &g5, &col_val, groups, cout_g, k_g, w_shape)
    }
}

/// d_weight ノード（`Op::ConvWeightGradHost3d`）を子テープへ登録する。`structure` は二階微分経路、
/// 前方値は [`weight_grad_value3d`]。孫テープでの再生（`replay_op`）からも同じ関数で呼ばれ、値は常に
/// ホスト関数で再計算される。
pub(super) fn host_weight_grad_node3d<'c>(
    child: &'c Tape,
    structure: &Var<'c>,
    g: &Var<'c>,
    x: &Var<'c>,
    params: &Conv3dParams,
    transposed: bool,
) -> Result<Var<'c>, AutodiffError> {
    let w_shape = structure.shape();
    let value = weight_grad_value3d(child, g, x, params, transposed, &w_shape)?;
    if value.shape() != w_shape.as_slice() {
        return Err(invalid(
            "d_weight ノードの値 shape が構造と不一致（内部契約違反）".into(),
        ));
    }
    let id = child.push_eager(
        Op::ConvWeightGradHost3d {
            structure: structure.node_id(),
            g: g.node_id(),
            x: x.node_id(),
            params: params.clone(),
            transposed,
        },
        value,
    );
    Ok(Var::from_raw(child, id))
}

/// `Op::Conv3d` の VJP を子テープへ記録する。`g`: `[N, Cout, Dout, Hout, Wout]`。
pub(super) fn conv3d_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x: &Var<'c>,
    w: &Var<'c>,
    bias_shape: Option<&[usize]>,
    params: &Conv3dParams,
    need: ConvNeeds,
) -> Result<ConvGrads<'c>, AutodiffError> {
    let (x_shape, w_shape, g_shape) = (x.shape(), w.shape(), g.shape());
    if x_shape.len() != 5 || w_shape.len() != 5 || g_shape.len() != 5 {
        return Err(invalid(
            "conv3d の VJP: rank 5 でない shape（内部契約違反）".into(),
        ));
    }
    if [&x_shape, &w_shape, &g_shape]
        .iter()
        .any(|s| s.contains(&0))
    {
        return zero_grads3d(child, g, &x_shape, &w_shape, bias_shape, need);
    }
    let groups = params.groups();
    let cout_g = w_shape[0] / groups.max(1);
    let k_g = checked_dims_product(&w_shape[1..], "conv3d の VJP（K_g）")?;
    let n = g_shape[0];
    let p = checked_dims_product(&g_shape[2..], "conv3d の VJP（空間）")?;

    let mut out = ConvGrads {
        d_input: None,
        d_weight: None,
        d_bias: None,
    };

    if need.input {
        // d_col[g] = w_gᵀ · g_g → [N, G·K_g, P] → fold3d（col2im3d）。
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
        out.d_input = Some(fold3d_var(&d_col, &x_shape, params)?);
    }

    if need.weight {
        // 構造: dw[g] = g_g · col_g（[Cout_g, K_g]）。
        let col = unfold3d_var(x, params)?;
        let mut parts = Vec::with_capacity(groups);
        for gi in 0..groups {
            let g_g = channel_rows(g, gi * cout_g, cout_g, n, p)?;
            let col_g = col_np_by_k(&col, gi, k_g, n, p)?;
            parts.push(g_g.matmul_fp32_strict(&col_g)?);
        }
        let structure = cat_or_single(parts, 0)?.contiguous()?.reshape(&w_shape)?;
        out.d_weight = Some(host_weight_grad_node3d(
            child, &structure, g, x, params, false,
        )?);
    }

    if need.bias
        && let Some(b) = bias_shape
    {
        out.d_bias = Some(bias_grad3d(child, g, b)?);
    }
    Ok(out)
}

/// `Op::ConvTranspose3d` の VJP を子テープへ記録する。`x`: `[N, Cin, D, H, W]`、
/// `w`: `[Cin, Cout_g, kD, kH, kW]`、`g`: `[N, Cout, Dout, Hout, Wout]`。
pub(super) fn conv_transpose3d_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    x: &Var<'c>,
    w: &Var<'c>,
    bias_shape: Option<&[usize]>,
    params: &Conv3dParams,
    need: ConvNeeds,
) -> Result<ConvGrads<'c>, AutodiffError> {
    let (x_shape, w_shape, g_shape) = (x.shape(), w.shape(), g.shape());
    if x_shape.len() != 5 || w_shape.len() != 5 || g_shape.len() != 5 {
        return Err(invalid(
            "conv_transpose3d の VJP: rank 5 でない shape（内部契約違反）".into(),
        ));
    }
    if [&x_shape, &w_shape, &g_shape]
        .iter()
        .any(|s| s.contains(&0))
    {
        return zero_grads3d(child, g, &x_shape, &w_shape, bias_shape, need);
    }
    let groups = params.groups();
    let cin_g = x_shape[1] / groups.max(1);
    let k_g = checked_dims_product(&w_shape[1..], "conv_transpose3d の VJP（K_g）")?;
    let n = x_shape[0];
    let p_x = checked_dims_product(&x_shape[2..], "conv_transpose3d の VJP（空間）")?;

    let mut out = ConvGrads {
        d_input: None,
        d_weight: None,
        d_bias: None,
    };

    if need.input || need.weight {
        // col = im2col3d(g)（groups = 1 の unfold3d。K 軸は groups 付き im2col3d の平坦化と一致）。
        let col = unfold3d_var(g, params)?;
        let col_shape = col.shape();
        let k_total = k_g
            .checked_mul(groups)
            .ok_or_else(|| invalid("conv_transpose3d の VJP: K 軸の桁あふれ".into()))?;
        if col_shape.len() != 3 || col_shape[1] != k_total || col_shape[2] != p_x {
            return Err(invalid(format!(
                "conv_transpose3d の VJP: upstream {g_shape:?} から導出した unfold3d shape \
                 {col_shape:?} が input {x_shape:?}／weight {w_shape:?} と整合しない"
            )));
        }

        if need.input {
            // d_x[g] = w_g · col_g（[Cin_g, N·P_x]）。
            let mut parts = Vec::with_capacity(groups);
            for gi in 0..groups {
                let w_g = w
                    .narrow(0, gi * cin_g, cin_g)?
                    .contiguous()?
                    .reshape(&[cin_g, k_g])?;
                let c_g = channel_rows(&col, gi * k_g, k_g, n, p_x)?;
                let prod_m = w_g.matmul_fp32_strict(&c_g)?;
                parts.push(np_cols_to_n_rows_p(&prod_m, cin_g, n, p_x)?);
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
            out.d_weight = Some(host_weight_grad_node3d(
                child, &structure, g, x, params, true,
            )?);
        }
    }

    if need.bias
        && let Some(b) = bias_shape
    {
        out.d_bias = Some(bias_grad3d(child, g, b)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7 + 3) % 11) as f32 - 5.0) * scale)
            .collect()
    }

    fn params(groups: usize) -> Conv3dParams {
        Conv3dParams::new([2, 3, 2], [1, 2, 1], [1, 1, 0], [1, 1, 2], groups).unwrap()
    }

    /// 前提: groups = 1 の unfold3d の K 軸 `(c, kd, kh, kw)` の並びは groups 付き im2col3d の
    /// `[N, G, K_g, P]` を `[N, G·K_g, P]` へ平坦化したものと一致する（d_input／d_weight の group 切り出しの前提）。
    #[test]
    fn unfold3d_groups1_matches_grouped_im2col3d_flatten() {
        let tape = Tape::new();
        let x_t = Tensor::new(seq(2 * 4 * 3 * 4 * 5, 0.25), &[2, 4, 3, 4, 5]).unwrap();
        let x = tape.var(&x_t);
        for groups in [1usize, 2] {
            let pg = params(groups);
            let u = unfold3d_var(&x, &pg).unwrap().to_tensor();
            let shape = im2col3d_out_shape(&[2, 4, 3, 4, 5], &pg).unwrap();
            let grouped = im2col3d_with_fallback(tape.ops(), &x_t, &pg, &shape).unwrap();
            let flat = grouped
                .contiguous()
                .reshape(&[shape[0], shape[1] * shape[2], shape[3]])
                .unwrap();
            assert_eq!(u.shape(), flat.shape());
            assert_eq!(u.host_slice().into_owned(), flat.host_slice().into_owned());
        }
    }

    /// `<unfold3d(x), y> = <x, fold3d(y)>`（随伴関係。f64 で内積を取る）。
    #[test]
    fn unfold3d_fold3d_are_adjoint() {
        let tape = Tape::new();
        let x_t = Tensor::new(seq(2 * 3 * 3 * 4 * 5, 0.5), &[2, 3, 3, 4, 5]).unwrap();
        let x = tape.var(&x_t);
        let pg = params(1);
        let ux = unfold3d_var(&x, &pg).unwrap().to_tensor();
        let y_t = Tensor::new(seq(ux.shape().iter().product(), 0.125), ux.shape()).unwrap();
        let y = tape.var(&y_t);
        let fy = fold3d_var(&y, &[2, 3, 3, 4, 5], &pg).unwrap().to_tensor();
        let dot = |a: &Tensor<f32>, b: &Tensor<f32>| -> f64 {
            a.host_slice()
                .iter()
                .zip(b.host_slice().iter())
                .map(|(&p, &q)| f64::from(p) * f64::from(q))
                .sum()
        };
        let (lhs, rhs) = (dot(&ux, &y_t), dot(&x_t, &fy));
        assert!(
            (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0),
            "{lhs} vs {rhs}"
        );
    }
}
