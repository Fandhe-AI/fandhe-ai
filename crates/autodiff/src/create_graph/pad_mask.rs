//! `Pad`・`PadMode`・`MaskedFill` の `create_graph` 対応（イシュー #3084・親 #3076・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §21・§29）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）
//! から呼ばれる。`replay_op`（forward 再生）は公開 `Var` メソッドへ薄く委譲するため
//! 本モジュールには置かない。子テープへ積む Op はすべて
//! `supports_create_graph() == true` な構造 Op（`Narrow`・`Where`・`Reshape`・
//! `Contiguous`・`Scatter`）に限り、閉包性（§21.1(4)）を保つ。
//!
//! **方式**（1 階 `grad.rs` の VJP と値が bit 一致する）:
//! - `Pad` は **V1**: 各軸に `narrow(axis, before, input_len)` を連鎖適用する
//!   （1 階と同じ順序。算術を含まない）。
//! - `MaskedFill` は **V1/V2**: `where_cond(mask, 0, g)`。`g * (mask == 0)` は上流が `inf`
//!   のとき `inf * 0 = NaN` を生むため使わない（§21.1(3)・§21.5）。
//! - `PadMode` は **V3**: 1 階は添字重複を `f64` アキュムレータで縮約し 1 回だけ downcast
//!   する契約（`docs/autodiff-pad-modes-decision.md`）を持つため、値は 1 階と同じ
//!   `pad_modes_vjp_host`、構造は `g` の平坦化 → `zeros.scatter_add(0, idx, ·)` →
//!   入力 shape へ reshape とし、`push_v3` で結合する。`idx` は tensor-core の
//!   `pad_modes_source_indices`（写像の情報源を 1 つに保つ）。

use fandhe_ai_tensor_core::pad_modes::{
    pad_modes_layout, pad_modes_source_indices, pad_modes_vjp_host,
};
use fandhe_ai_tensor_core::{PadMode, Tensor};

use super::{push_v3, value_of};
use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;

/// `Pad` の d_input: `g` を各軸 `narrow` で切り出す（1 階と同じ連鎖）。
pub(super) fn pad_cgrad<'c>(
    g: &Var<'c>,
    input_shape: &[usize],
    pads: &[(usize, usize)],
) -> Result<Var<'c>, AutodiffError> {
    let mut d = *g;
    for (axis, &(before, _after)) in pads.iter().enumerate() {
        d = d.narrow(axis, before, input_shape[axis])?;
    }
    Ok(d)
}

/// `MaskedFill` の d_input: fill 位置（`mask` が真）の勾配を 0 にする。
pub(super) fn masked_fill_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    mask: &Tensor<bool>,
) -> Result<Var<'c>, AutodiffError> {
    let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
    Var::where_cond(mask, &zeros, g)
}

/// `PadMode` の d_input（V3）。値は 1 階 `pad_modes_vjp_host`、構造は平坦 `scatter_add`。
/// 空テンソルは 1 階（出力が空なら勾配 0）に揃えゼロ定数を返す。
pub(super) fn pad_mode_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    pads: &[(usize, usize)],
    mode: PadMode,
) -> Result<Var<'c>, AutodiffError> {
    let layout = pad_modes_layout(input_shape, pads, mode)?;
    let (in_numel, out_numel) = (layout.in_numel(), layout.out_numel());
    if in_numel == 0 || out_numel == 0 {
        return Ok(child.var_no_grad(&Tensor::zeros(input_shape)?));
    }
    // 値: 1 階と同じ f64 縮約（1 回 downcast）。
    let g_val = value_of(child, g)?;
    let data = pad_modes_vjp_host(&g_val.contiguous().host_slice(), &layout)?;
    let host = Tensor::new(data, input_shape)?;
    // 構造: 添字は出力の行優先順の入力平坦添字。`i32` に収まらなければ型付きエラー。
    let idx: Vec<i32> = pad_modes_source_indices(&layout)?
        .into_iter()
        .map(|i| {
            i32::try_from(i).map_err(|_| {
                AutodiffError::Backward(format!(
                    "create_graph: PadMode の入力平坦添字 {i} が i32 に収まらない"
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    let idx = Tensor::<i32>::new(idx, &[out_numel])?;
    let zeros = child.var_no_grad(&Tensor::zeros(&[in_numel])?);
    let flat = g.contiguous()?.reshape(&[out_numel])?;
    let composed = zeros.scatter_add(0, &idx, &flat)?.reshape(input_shape)?;
    push_v3(child, &composed, host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    /// `(入力 shape, pads, mode)`。
    type Case<'a> = (&'a [usize], &'a [(usize, usize)], PadMode);

    /// V3 の構造側（平坦 `scatter_add`）の値が 1 階ホスト関数 `pad_modes_vjp_host` と
    /// REQ-2 統一複合判定で一致することを固定する。
    #[test]
    fn pad_mode_structure_value_matches_host() {
        let cases: [Case; 3] = [
            (&[5], &[(2, 2)], PadMode::Reflect),
            (&[3, 4], &[(4, 5), (1, 6)], PadMode::Replicate),
            (&[3, 4], &[(2, 3), (1, 4)], PadMode::Circular),
        ];
        for (shape, pads, mode) in cases {
            let tape = Tape::new_with_ops(naive_ops());
            let layout = pad_modes_layout(shape, pads, mode).unwrap();
            let data: Vec<f32> = (0..layout.out_numel())
                .map(|i| (i as f32) * 0.37 - 1.0)
                .collect();
            let g = tape.var(&Tensor::new(data, layout.out_shape()).unwrap());
            let g_val = value_of(&tape, &g).unwrap();
            let host = pad_modes_vjp_host(&g_val.contiguous().host_slice(), &layout).unwrap();
            let idx: Vec<i32> = pad_modes_source_indices(&layout)
                .unwrap()
                .into_iter()
                .map(|i| i as i32)
                .collect();
            let idx = Tensor::<i32>::new(idx, &[layout.out_numel()]).unwrap();
            let zeros = tape.var_no_grad(&Tensor::zeros(&[layout.in_numel()]).unwrap());
            let flat = g
                .contiguous()
                .unwrap()
                .reshape(&[layout.out_numel()])
                .unwrap();
            let got = zeros
                .scatter_add(0, &idx, &flat)
                .unwrap()
                .to_tensor()
                .host_slice()
                .into_owned();
            for (a, b) in got.iter().zip(host.iter()) {
                assert!(
                    fandhe_ai_tensor_core::tolerance::composite_close_f32(*a, *b),
                    "{shape:?} {mode:?}: {a} vs {b}"
                );
            }
        }
    }
}
