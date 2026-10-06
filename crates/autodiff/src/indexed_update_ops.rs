//! `scatter_reduce`・`index_add`・`index_copy`・`masked_scatter` の自由関数
//! （イシュー #2641・親 #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::scatter_reduce`／`index_add`／
//! `index_copy`／`masked_scatter` の委譲メソッドと `ScatterReduceMode` の再
//! エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
//! 本モジュールは内部クレート限定の入口で、`Var` に inherent メソッドを足さない。
//! 保留は `crates/facade/src/lib.rs` の `IndexedUpdateOpsHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定する
//! （`docs/autodiff-indexed-update-ops-decision.md`）。
//!
//! **PyTorch 相当と方式**:
//!
//! | 演算 | PyTorch 相当 | 方式 |
//! |---|---|---|
//! | [`scatter_reduce`] | `Tensor.scatter_reduce(dim, index, src, reduce, include_self)` | 専用 `Op::IndexedScatterReduce`＋共有ホストカーネル |
//! | [`index_add`] | `Tensor.index_add(dim, index, source)`（`alpha` なし） | 既存 `Var::scatter_add` の合成（新規 `Op` なし） |
//! | [`index_copy`] | `Tensor.index_copy(dim, index, source)` | 既存 `Var::scatter` の合成（新規 `Op` なし） |
//! | [`masked_scatter`] | `torch.masked_scatter(x, mask, source)` | view 系＋`narrow`＋`Var::scatter` の合成 |
//!
//! `dim` は `usize`、索引は既存慣例（`Var::scatter`／`gather`）に合わせ
//! `Tensor<i32>`（負の添字は拒否）。
//!
//! **経路**: [`scatter_reduce`] は ① `scatter_reduce_layout` による形状・確保サイズ
//! 検査 → ② `index` の値域検査 → ③ 入力・`src` の実体化 →
//! ④ `BackendOps::indexed_scatter_reduce`（`Unsupported` のときだけ共有ホスト
//! カーネル `indexed_update::scatter_reduce_host` へフォールバックし、他のエラーは
//! 伝播。戻り値 shape も検証）→ ⑤ `Op::IndexedScatterReduce` を積む（VJP は
//! `grad.rs`）。合成 3 演算は既存の `BackendOps::scatter`（CUDA／Metal は既存 GPU
//! カーネル、`Unsupported` のときは `eval::scatter`）を通り、新しい GPU カーネルは
//! 追加しない。**すべての検査は tape にノードを積む前に終える**（エラー時に孤児
//! ノードを残さない）。
//!
//! **数値契約・PyTorch 差分**: `scatter_reduce` の契約は
//! `fandhe_ai_tensor_core::indexed_update`、差分は決定記録 §5 を正とする。
//! `index_add` は `Add` の `f64` 決定的集約、`index_copy` は行優先で最後の書き手が
//! 勝つ（PyTorch は未定義。`index_put` と同じ「より強い契約」。勾配も最後の書き手
//! だけに流れる）。`index_add` の `alpha` は持たない（呼び出し側が `source` を事前に
//! スケールすれば等価。隠し定数ノードが葉プレフィックスに入る副作用を避ける）。
//! 高階微分（`create_graph`）・activation checkpoint・f64 自動微分経路は対象外。

use fandhe_ai_tensor_core::indexed_update::{self, ScatterReduceLayout};
use fandhe_ai_tensor_core::{BackendError, ScatterReduceMode, ShapeError, Tensor, broadcast_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::rearrange_ops::{checked_axis_len_as_i32, checked_index_alloc_len};
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// `index` の全値が `[0, dim_size)` に収まることを検査する（負値を含め範囲外は
/// `InvalidArgument`。`Var::scatter` の入口検査と同じ契約）。
fn check_index_range(op: &str, index: &Tensor<i32>, dim_size: usize) -> Result<(), AutodiffError> {
    for v in eval::dense_vec_i32(index) {
        if v < 0 || (v as usize) >= dim_size {
            return Err(AutodiffError::InvalidArgument(format!(
                "{op}: index 添字 {v} が範囲 [0, {dim_size}) を外れている"
            )));
        }
    }
    Ok(())
}

/// `scatter_reduce`（`Tensor.scatter_reduce` 相当）。`index`／`src` は同 shape、
/// `dim` 以外の軸で `index <= x`。`include_self == false` のとき、触れられた位置の
/// 入力値は縮約に含めない。
pub fn scatter_reduce<'t>(
    x: &Var<'t>,
    dim: usize,
    index: &Tensor<i32>,
    src: &Var<'t>,
    reduce: ScatterReduceMode,
    include_self: bool,
) -> Result<Var<'t>, AutodiffError> {
    x.check_same_tape(src)?;
    let in_shape = x.shape();
    let layout: ScatterReduceLayout =
        indexed_update::scatter_reduce_layout(&in_shape, index.shape(), &src.shape(), dim)
            .map_err(AutodiffError::Shape)?;
    // `scatter_reduce_layout` 成功時点で `dim < rank` が保証される。
    check_index_range("scatter_reduce", index, in_shape[dim])?;
    let index_c = index.contiguous();
    let (input_val, src_val) = {
        let nodes = x.tape().nodes.borrow();
        let ops = x.tape().ops();
        (
            materialize_fallible(&nodes, ops, x.node_id())?.clone(),
            materialize_fallible(&nodes, ops, src.node_id())?.clone(),
        )
    };
    let value = match x.tape().ops().indexed_scatter_reduce(
        &input_val,
        dim,
        &index_c,
        &src_val,
        reduce,
        include_self,
    ) {
        Ok(v) => {
            verify_shape(v.shape(), layout.shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data = indexed_update::scatter_reduce_host(
                &input_val.contiguous().host_slice(),
                &index_c.host_slice(),
                &src_val.contiguous().host_slice(),
                &layout,
                reduce,
                include_self,
            )
            .map_err(AutodiffError::Shape)?;
            Tensor::new(data, layout.shape()).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = x.tape().push_eager(
        Op::IndexedScatterReduce {
            input: x.node_id(),
            dim,
            index: index_c,
            src: src.node_id(),
            mode: reduce,
            include_self,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// `index_add` / `index_copy` の共通前処理。`index`（rank 1・長さ `source.shape[dim]`）
/// を `source` と同 shape へ拡張した `Tensor<i32>` を返す。検査はすべて tape に
/// ノードを積む前に終える。
fn plan_axis_index<'t>(
    op: &str,
    x: &Var<'t>,
    dim: usize,
    index: &Tensor<i32>,
    source: &Var<'t>,
) -> Result<Tensor<i32>, AutodiffError> {
    x.check_same_tape(source)?;
    let x_shape = x.shape();
    let s_shape = source.shape();
    let rank = x_shape.len();
    if dim >= rank {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: dim,
            rank,
        }));
    }
    if index.shape().len() != 1 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: index.shape().len(),
        }));
    }
    if s_shape.len() != rank {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: rank,
            actual: s_shape.len(),
        }));
    }
    let n = index.shape()[0];
    if n != s_shape[dim] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: index.shape().to_vec(),
            rhs: s_shape.clone(),
        }));
    }
    // `scatter_out_shape` は `<=` しか見ないため、ここで `dim` 以外の等号を要求する。
    for (axis, (&a, &b)) in x_shape.iter().zip(&s_shape).enumerate() {
        if axis != dim && a != b {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: x_shape.clone(),
                rhs: s_shape.clone(),
            }));
        }
    }
    check_index_range(op, index, x_shape[dim])?;
    checked_bytes_for::<f32>(&x_shape)?;
    checked_bytes_for::<f32>(&s_shape)?;
    checked_bytes_for::<i32>(&s_shape)?;
    let mut expand = vec![1usize; rank];
    expand[dim] = n;
    index
        .contiguous()
        .reshape(&expand)
        .and_then(|i| i.broadcast_to(&s_shape))
        .map_err(AutodiffError::Shape)
}

/// `index_add`（`Tensor.index_add` 相当。`alpha` なし）。`dim` 軸の `index[i]` 番目の
/// スライスへ `source` の `i` 番目のスライスを加算する。重複添字は `f64` の決定的
/// 集約（`ScatterReduce::Add`）。
pub fn index_add<'t>(
    x: &Var<'t>,
    dim: usize,
    index: &Tensor<i32>,
    source: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    let idx = plan_axis_index("index_add", x, dim, index, source)?;
    x.scatter_add(dim, &idx, source)
}

/// `index_copy`（`Tensor.index_copy` 相当）。`dim` 軸の `index[i]` 番目のスライスを
/// `source` の `i` 番目のスライスで置き換える。重複添字は行優先で最後の書き手が勝つ。
pub fn index_copy<'t>(
    x: &Var<'t>,
    dim: usize,
    index: &Tensor<i32>,
    source: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    let idx = plan_axis_index("index_copy", x, dim, index, source)?;
    x.scatter(dim, &idx, source)
}

/// `masked_scatter`（`torch.masked_scatter` 相当）。`x` と `mask` を共通 shape へ
/// ブロードキャストし、`mask` が真の位置へ `source` の要素を行優先で順に詰める。
/// `source` の要素数が真の個数より少なければ `InvalidArgument`、余った要素の勾配は 0。
pub fn masked_scatter<'t>(
    x: &Var<'t>,
    mask: &Tensor<bool>,
    source: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    x.check_same_tape(source)?;
    let out_shape = broadcast_shape(&x.shape(), mask.shape()).map_err(AutodiffError::Shape)?;
    checked_bytes_for::<f32>(&out_shape)?;
    checked_bytes_for::<bool>(&out_shape)?;
    let p: usize = out_shape.iter().product();
    checked_axis_len_as_i32(p)?;
    let mask_c = mask
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let mask_data = mask_c.host_slice();
    let m = mask_data.iter().filter(|&&b| b).count();
    let s_shape = source.shape();
    checked_bytes_for::<f32>(&s_shape)?;
    let s: usize = s_shape.iter().product();
    if m > s {
        return Err(AutodiffError::InvalidArgument(format!(
            "masked_scatter: source の要素数 {s} が mask の真の個数 {m} より少ない"
        )));
    }
    checked_index_alloc_len(m)?;
    let mut flat: Vec<i32> = Vec::with_capacity(m);
    for (i, &b) in mask_data.iter().enumerate() {
        if b {
            flat.push(
                i32::try_from(i)
                    .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))?,
            );
        }
    }
    let flat = Tensor::new(flat, &[m]).map_err(AutodiffError::Shape)?;
    let x_flat = x.broadcast_to(&out_shape)?.contiguous()?.reshape(&[p])?;
    let src_flat = source.contiguous()?.reshape(&[s])?.narrow(0, 0, m)?;
    x_flat.scatter(0, &flat, &src_flat)?.reshape(&out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    #[test]
    fn scatter_reduce_basic_values() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let s = tape.var(&t(vec![4.0, 5.0, 6.0], &[3]));
        let idx = ti(vec![0, 0, 2], &[3]);
        let out = scatter_reduce(&x, 0, &idx, &s, ScatterReduceMode::Amax, true).unwrap();
        assert_eq!(vals(&out), vec![5.0, 2.0, 6.0]);
    }

    #[test]
    fn index_add_and_copy_basic_values() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 6], &[3, 2]));
        let s = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let idx = ti(vec![2, 0], &[2]);
        let a = index_add(&x, 0, &idx, &s).unwrap();
        assert_eq!(vals(&a), vec![3.0, 4.0, 0.0, 0.0, 1.0, 2.0]);
        let dup = ti(vec![1, 1], &[2]);
        let c = index_copy(&x, 0, &dup, &s).unwrap();
        // 重複添字は最後の書き手が勝つ。
        assert_eq!(vals(&c), vec![0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_scatter_fills_in_row_major_order() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 6], &[2, 3]));
        let s = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
        let mask = Tensor::new(vec![true, false, true, false, true, false], &[2, 3]).unwrap();
        let out = masked_scatter(&x, &mask, &s).unwrap();
        assert_eq!(vals(&out), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        // 全偽・空 index でも合成経路が成立する。
        let none = Tensor::new(vec![false; 6], &[2, 3]).unwrap();
        let same = masked_scatter(&x, &none, &s).unwrap();
        assert_eq!(vals(&same), vec![0.0; 6]);
    }

    #[test]
    fn errors_leave_tape_untouched() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 4], &[4]));
        let s = tape.var(&t(vec![1.0, 2.0], &[2]));
        let before = tape.len();
        assert!(index_add(&x, 0, &ti(vec![0, 4], &[2]), &s).is_err());
        assert!(index_copy(&x, 0, &ti(vec![-1, 0], &[2]), &s).is_err());
        assert!(index_add(&x, 1, &ti(vec![0, 1], &[2]), &s).is_err());
        assert!(index_add(&x, 0, &ti(vec![0], &[1]), &s).is_err());
        assert!(
            scatter_reduce(
                &x,
                0,
                &ti(vec![0, 9], &[2]),
                &s,
                ScatterReduceMode::Sum,
                true
            )
            .is_err()
        );
        let mask = Tensor::new(vec![true; 4], &[4]).unwrap();
        assert!(masked_scatter(&x, &mask, &s).is_err());
        assert_eq!(tape.len(), before);
    }
}
