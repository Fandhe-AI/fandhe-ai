//! 選択縮約 4 Op（`Max`・`Min`・`Amax`・`Amin`）の `create_graph` 対応（イシュー #3789・
//! 親 #3076・設計 `docs/autodiff-higher-order-grad-decision.md` §34）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var::max`／`min`／`amax`／`amin` への
//! 1 行委譲で足りるため親側に置く。写しの Op タグを親と同じに保つことが重要で、
//! `Amax`／`Amin` を `Max` のタグで記録すると孫テープが先勝ち VJP を使ってしまう。
//!
//! **方式**（1 階の走査規則は `grad.rs` の `extremum_first_match_mask`／
//! `extremum_all_match_mask` に集約し、1 階 VJP 関数と同じ規則〈`==` 判定・NaN は全不一致・
//! outer×axis×inner 走査〉を共有する）:
//! - `Max`／`Min`（**V2**）: 親テープの記録値から「各 lane で `==` 一致する最初の 1 位置」の
//!   bool マスク `M` をホストで作り、`where_cond(M, g_b, 0)` を積む。`mul` を使わないのは
//!   `inf * 0` を作らないため。選ばれた位置は `g` の値そのまま、他は `+0.0` で 1 階と bit 一致する。
//! - `Amax`／`Amin`（**V3**）: 値は 1 階と同じホスト関数 `extremum_even_split_vjp` を
//!   `push_v3` で載せ、構造は `where_cond(M_all, g_b / k, 0)`（`M_all` は全一致マスク、`k` は
//!   lane の一致数。`k == 0` の lane はマスク全 false・`k = 1` で `0/1` となり Div の VJP が
//!   `NaN` を生まない）。
//!
//! 子テープへ積む Op は `Reshape`・`BroadcastTo`・`Contiguous`・`Where`・`ScalarBinary{Div}`・
//! 定数 `Leaf` のみで、すべて `supports_create_graph() == true`（閉包性。§34.11）。
//! 入力・出力・上流の shape 整合は子テープへ書く前に確かめ、崩れていれば型付きエラーで拒否する。

use fandhe_ai_tensor_core::{Tensor, reduce_out_shape};

use super::{child_value, push_v3};
use crate::error::AutodiffError;
use crate::grad::{extremum_all_match_mask, extremum_even_split_vjp, extremum_first_match_mask};
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 縮約 lane の分配規則。
#[derive(Clone, Copy)]
pub(super) enum Split {
    /// `Max`／`Min`: 先勝ちの 1 位置へ `g`（V2）。
    FirstMatch,
    /// `Amax`／`Amin`: 一致全位置へ `g / k`（V3）。
    EvenSplit,
}

/// 選択縮約ノードの記述（`build_cgrads` の腕が組み立てる）。
#[derive(Clone, Copy)]
pub(super) struct SelectReduceSpec {
    /// 縮約ノード自身（記録値 `out_value` の取得元）。
    pub(super) id: NodeId,
    /// 縮約の入力ノード。
    pub(super) input: NodeId,
    /// 縮約軸（`None` は全要素）。
    pub(super) dim: Option<usize>,
    pub(super) split: Split,
}

/// 上流 `g`（縮約後 shape）を入力 shape へ復元・broadcast する。
fn bcast_upstream<'c>(
    g: &Var<'c>,
    in_shape: &[usize],
    dim: Option<usize>,
) -> Result<Var<'c>, AutodiffError> {
    match dim {
        Some(axis) => g.contiguous()?.unsqueeze(axis)?.broadcast_to(in_shape),
        None => g
            .reshape(&vec![1usize; in_shape.len()])?
            .broadcast_to(in_shape),
    }
}

/// 選択縮約ノード `id`（入力 `input`・縮約軸 `dim`）の d_input を子テープへ記録する。
pub(super) fn select_reduce_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: SelectReduceSpec,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let SelectReduceSpec {
        id,
        input,
        dim,
        split,
    } = spec;
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let out_value = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let in_shape = input_val.shape().to_vec();
    let expect = reduce_out_shape(&in_shape, dim).map_err(AutodiffError::Shape)?;
    if out_value.shape() != expect.as_slice() || g.shape() != expect {
        return Err(invalid(format!(
            "選択縮約の shape 不整合（入力 {in_shape:?}・dim {dim:?}・期待 {expect:?}・記録値 {:?}・上流 {:?}。内部契約違反）",
            out_value.shape(),
            g.shape()
        )));
    }
    let zeros = child.var_no_grad(&Tensor::zeros(&in_shape)?);
    if in_shape.contains(&0) {
        return Ok(zeros);
    }
    let g_b = bcast_upstream(g, &in_shape, dim)?;
    match split {
        Split::FirstMatch => {
            let m = extremum_first_match_mask(input_val, dim, out_value);
            let mask = Tensor::new(m, &in_shape)?;
            Var::where_cond(&mask, &g_b, &zeros)
        }
        Split::EvenSplit => {
            let (m, k) = extremum_all_match_mask(input_val, dim, out_value);
            let mask = Tensor::new(m, &in_shape)?;
            let k_var = child.var_no_grad(&Tensor::new(k, &in_shape)?);
            let composed = Var::where_cond(&mask, &g_b.div(&k_var)?, &zeros)?;
            let g_value = child_value(child, g)?;
            let host = extremum_even_split_vjp(input_val, dim, out_value, &g_value);
            push_v3(child, &composed, host)
        }
    }
}
