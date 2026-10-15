//! 索引・累積族（`Gather`・`Scatter`・`Embedding`・`Sort`・`Topk`・`Cummax`・
//! `Cummin`・`Cumsum`）の `create_graph` 対応（イシュー #3083・親 #3076・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §21・§25）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）と
//! `build_cgrads`（1 階 VJP の子テープ記録）から呼ばれる。子テープへ積む Op は
//! すべて `supports_create_graph() == true` な構造 Op（`Gather`・`Scatter`・
//! `Where`・`Narrow`・`Contiguous`・`Cumsum`）に限り、閉包性（子テープ上の勾配を
//! さらに `backward_create_graph` で微分できること。§21.1(4)）を保つ。
//! `Gather` の cgrad は `Scatter{Add}`、`Scatter` の d_src は `Gather` を積むため、
//! この 2 variant は同時に対象にする必要がある。
//!
//! **方式**（1 階 `grad.rs` の同名 Op の VJP と同じ式。値は 1 階と bit 一致する）:
//! - 索引系（`Gather`・`Sort`・`Topk`・`Cummax`・`Cummin`）は **V1/V2**: `index` を
//!   ホストの定数として固定し、`zeros.scatter_add(dim, index, g)` を積む
//!   （1 階と同じ `scatter_with_fallback(Add)` を通る。重複索引は f64 決定的集約）。
//!   `Sort`／`Topk`／`Cummax`／`Cummin` の `Op` payload は `descending`／`largest` を
//!   持たないため、写しの forward は値の選択だけを行う `Var::gather` で再生する
//!   （値は親と bit 一致。子テープ上の Op タグだけが `Gather` になる）。
//! - `Scatter` は d_src が `gather`（`Overwrite` は last-writer マスクの
//!   `where_cond`。`mul` を使わず `inf*0` を避ける）、d_input は `Add` なら恒等、
//!   `Overwrite` なら `scatter(g, index, 0)`。
//! - `Embedding` は `Gather`（dim 0）として再生し、cgrad は `scatter_add` の後に
//!   `padding_idx` 行を `where_cond` で 0 にする。
//! - `Cumsum` は **V3**: 構造は `flip → cumsum → flip`、ノードの前方値は 1 階と
//!   同じホスト関数 `grad::cumsum_vjp_along`（lane ごとの f64 アキュムレータ）。

use fandhe_ai_tensor_core::{ScatterReduce, Tensor};

use super::{f32_mask_to_bool, push_v3, value_of};
use crate::error::AutodiffError;
use crate::grad::{cumsum_vjp_along, scatter_overwrite_last_writer_mask};
use crate::tape::Tape;
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

// ---------------------------------------------------------------------
// forward 再生（replay_op から呼ばれる。公開 `Var` メソッドへの薄い委譲）
// ---------------------------------------------------------------------

/// `Scatter` の写し。`reduce` に応じて `scatter_add`／`scatter` を呼ぶ。
/// `ScatterReduce` は `#[non_exhaustive]` のため未知 variant は `Err`
/// （`validate_ancestors` が入口で拒否済みの多層防御）。
pub(super) fn replay_scatter<'c>(
    input: &Var<'c>,
    src: &Var<'c>,
    dim: usize,
    index: &Tensor<i32>,
    reduce: ScatterReduce,
) -> Result<Var<'c>, AutodiffError> {
    match reduce {
        ScatterReduce::Add => input.scatter_add(dim, index, src),
        ScatterReduce::Overwrite => input.scatter(dim, index, src),
        _ => Err(invalid(format!("未知の ScatterReduce variant {reduce:?}"))),
    }
}

// ---------------------------------------------------------------------
// 1 階 VJP の子テープ記録（build_cgrads から呼ばれる）
// ---------------------------------------------------------------------

/// `Gather`／`Sort`／`Topk`／`Cummax`／`Cummin` の d_input:
/// `zeros(input_shape).scatter_add(dim, index, g)`。
pub(super) fn gather_like_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    input_shape: &[usize],
    dim: usize,
    index: &Tensor<i32>,
) -> Result<Var<'c>, AutodiffError> {
    let zeros = child.var_no_grad(&Tensor::zeros(input_shape)?);
    zeros.scatter_add(dim, index, g)
}

/// `Embedding` の d_weight: `scatter_add`（dim 0）の後、`padding_idx` 行を 0 にする
/// （1 階 `with_row_zeroed` と同じ選択。`where_cond` で選ぶため `inf*0` を生まない）。
pub(super) fn embedding_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    weight_shape: &[usize],
    index: &Tensor<i32>,
    padding_idx: Option<usize>,
) -> Result<Var<'c>, AutodiffError> {
    let scattered = gather_like_vjp(child, g, weight_shape, 0, index)?;
    let Some(p) = padding_idx else {
        return Ok(scattered);
    };
    if weight_shape.len() != 2 || p >= weight_shape[0] {
        return Err(invalid(format!(
            "Embedding の padding_idx={p} が weight shape {weight_shape:?} の範囲外（内部契約違反）"
        )));
    }
    let cols = weight_shape[1];
    let keep: Vec<bool> = (0..weight_shape[0] * cols).map(|i| i / cols != p).collect();
    let keep = Tensor::new(keep, weight_shape)?;
    let zeros = child.var_no_grad(&Tensor::zeros(weight_shape)?);
    Var::where_cond(&keep, &scattered, &zeros)
}

/// `Scatter` の `(d_input, d_src)`。必要な側だけ構築する。
pub(super) struct ScatterGrads<'c> {
    pub(super) d_input: Option<Var<'c>>,
    pub(super) d_src: Option<Var<'c>>,
}

/// `Scatter` ノードの静的な形（親テープの shape と `Op` payload）。
pub(super) struct ScatterSpec<'a> {
    pub(super) input_shape: &'a [usize],
    pub(super) src_shape: &'a [usize],
    pub(super) dim: usize,
    pub(super) index: &'a Tensor<i32>,
    pub(super) reduce: ScatterReduce,
}

/// `Scatter` の VJP（1 階 `grad.rs::Op::Scatter` と同じ式）。
pub(super) fn scatter_vjp<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    spec: &ScatterSpec<'_>,
    need_input: bool,
    need_src: bool,
) -> Result<ScatterGrads<'c>, AutodiffError> {
    let ScatterSpec {
        input_shape,
        src_shape,
        dim,
        index,
        reduce,
    } = *spec;
    // d_src: 非 dim 軸が input より小さい場合は g を先頭から narrow してから gather する。
    let raw_d_src = if need_src {
        let mut gs = *g;
        for (axis, &in_s) in input_shape.iter().enumerate() {
            if axis != dim && src_shape[axis] < in_s {
                gs = gs.narrow(axis, 0, src_shape[axis])?;
            }
        }
        Some(gs.gather(dim, index)?)
    } else {
        None
    };
    match reduce {
        ScatterReduce::Add => Ok(ScatterGrads {
            d_input: need_input.then_some(*g),
            d_src: raw_d_src,
        }),
        ScatterReduce::Overwrite => {
            let zeros_src = child.var_no_grad(&Tensor::zeros(src_shape)?);
            let d_input = if need_input {
                Some(g.scatter(dim, index, &zeros_src)?)
            } else {
                None
            };
            let d_src = match raw_d_src {
                Some(raw) => {
                    let mask = scatter_overwrite_last_writer_mask(index, dim, input_shape);
                    let keep = f32_mask_to_bool(&mask)?;
                    Some(Var::where_cond(&keep, &raw, &zeros_src)?)
                }
                None => None,
            };
            Ok(ScatterGrads { d_input, d_src })
        }
        _ => Err(invalid(format!(
            "Op::Scatter の VJP: 未知の ScatterReduce variant {reduce:?}"
        ))),
    }
}

/// `Cumsum` の d_input（V3）。値は 1 階 `cumsum_vjp_along`、構造は
/// `flip → cumsum → flip`。空テンソルは 1 階の早期 return に揃えゼロ定数を返す。
pub(super) fn cumsum_cgrad<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    dim: usize,
) -> Result<Var<'c>, AutodiffError> {
    let shape = g.shape();
    if shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&shape)?));
    }
    let host = cumsum_vjp_along(&value_of(child, g)?, dim);
    let composed = g.flip(&[dim])?.cumsum(dim)?.flip(&[dim])?;
    push_v3(child, &composed, host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    /// V3 の構造側（`flip → cumsum → flip`）の値が 1 階と同じホスト関数
    /// `cumsum_vjp_along` と bit 一致することを固定する。孫テープで
    /// `Op::Contiguous` が構造を再計算しても、f32 逐次和へ落ちて桁落ち・
    /// overflow が生じない前提（決定記録 §23.3 の型。イシュー #3083）。
    #[test]
    fn cumsum_structure_value_matches_host_bitwise() {
        let cases: [(&[f32], &[usize], usize); 4] = [
            (&[1.0, 1e8, -1e8, 1.0], &[4], 0),
            (&[3e38, -3e38, 3e38], &[3], 0),
            (&[1.0, 1e8, -1e8, 1.0, 2.0, -3.0], &[2, 3], 1),
            (&[0.5, 1e8, -1e8, 1.0, 2.0, -3.0, 1e7, -1e7], &[2, 2, 2], 1),
        ];
        for (data, shape, dim) in cases {
            let tape = Tape::new_with_ops(naive_ops());
            let g = tape.var(&Tensor::new(data.to_vec(), shape).unwrap());
            let host = cumsum_vjp_along(&value_of(&tape, &g).unwrap(), dim);
            let composed = g
                .flip(&[dim])
                .unwrap()
                .cumsum(dim)
                .unwrap()
                .flip(&[dim])
                .unwrap();
            let got = composed.to_tensor().host_slice().into_owned();
            let want = host.host_slice().into_owned();
            let a: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "shape={shape:?} dim={dim}");
        }
    }

    /// `ScatterReduce` は `#[non_exhaustive]` のため未知 variant は構築できない。
    /// 既知 variant が `replay_scatter` で正しく再生されることのみ確認する
    /// （未知 variant の拒否は `_ =>` の腕と `validate_ancestors` が担う）。
    #[test]
    fn replay_scatter_accepts_known_reduce_variants() {
        let tape = Tape::new_with_ops(naive_ops());
        let input = tape.var(&Tensor::new(vec![0.0f32; 3], &[3]).unwrap());
        let src = tape.var(&Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap());
        let idx = Tensor::<i32>::new(vec![0, 0], &[2]).unwrap();
        let add = replay_scatter(&input, &src, 0, &idx, ScatterReduce::Add).unwrap();
        let ow = replay_scatter(&input, &src, 0, &idx, ScatterReduce::Overwrite).unwrap();
        assert_eq!(
            add.to_tensor().host_slice().into_owned(),
            vec![3.0, 0.0, 0.0]
        );
        assert_eq!(
            ow.to_tensor().host_slice().into_owned(),
            vec![2.0, 0.0, 0.0]
        );
    }
}
