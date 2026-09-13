//! gather／scatter カーネル（`torch.gather`／`torch.scatter`／
//! `torch.scatter_add` 相当。イシュー #1776）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::gather`]／[`BackendOps::scatter`]
//! （`ops.rs`）の CPU 実装本体。呼び出し元（`ops.rs`）が `input`／
//! `index`／`src` の shape を [`fandhe_ai_tensor_core::gather_out_shape`]／
//! [`fandhe_ai_tensor_core::scatter_out_shape`] で再検査してから本モジュール
//! へ委譲する契約のため、本モジュール自身は呼び出し元が渡す `out_shape`
//! （検査・確定済み）をそのまま信頼し shape の再検査は行わない（`ops.rs`
//! 側の二重検査が fail-closed 境界。`.claude/rules/security.md` A08）。
//! `index` の値が `[0, input.shape()[dim])` 範囲内であることは呼び出し元
//! （`fandhe_ai_autodiff::var::Var::gather`／`scatter`／`scatter_add`）が
//! forward 時点で検査済みの前提（本モジュールは値検査を行わない）。
//!
//! **決定的集約順序・精度契約**（[`fandhe_ai_tensor_core::ScatterReduce`]
//! doc を正とする）: `scatter` は `index`（`Add` は `src` も同 shape）を
//! 行優先（row-major。`Tensor::contiguous()` と同じ順序）で走査し、
//! `Add` は出力位置ごとの `f64` アキュムレータへ逐次加算したうえで
//! **1 回だけ** `f32` へ downcast する。`Overwrite` は同じ走査順で
//! 「最後に処理された値」が残る単純代入。本実装は単一スレッド逐次
//! ループ（並列化は将来の性能最適化 issue のスコープ・`.claude/rules/
//! out-of-scope-tracking.md` 対象。並列化する場合は出力位置ごとの
//! 排他アキュムレータ・決定的な reduce 木を選定すること）。
//!
//! `Tensor::get`（境界チェック付き安全アクセス。REQ-8「境界検査を
//! 省略しない」）のみを用い、`unsafe`／`unwrap`／`expect` は使わない
//! （`.claude/rules/coding-rust.md`）。非 contiguous な `input`／
//! `index`／`src`（strided view）も `Tensor::get` で正しく読める
//! （`reduction.rs::gather_elements`／`unravel` と同じ「境界チェック
//! 付きアクセスで strided 入力を安全に読む」方針を踏襲する）。

use fandhe_ai_tensor_core::{ScatterReduce, ShapeError, Tensor};

/// 線形添字（行優先）を `shape` の多次元添字へ展開する
/// （`reduction.rs::unravel` と同じ方針の独立実装。gather／scatter
/// 専用にこのモジュール内で完結させる——`reduction.rs` 側の
/// `unravel` は private のためモジュール間で共有しない）。
fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for (axis, &d) in shape.iter().enumerate().rev() {
        if d == 0 {
            out[axis] = 0;
            continue;
        }
        out[axis] = idx % d;
        idx /= d;
    }
    out
}

/// 行優先（C-order）ストライドを計算する（`unravel` と対）。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// 多次元添字（行優先）を線形添字へ畳み込む（`unravel` の逆演算）。
fn ravel(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

/// [`fandhe_ai_tensor_core::BackendOps::gather`] の CPU 実装本体
/// （イシュー #1776）。`out_shape`（＝`index.shape()`）は呼び出し元
/// （`ops.rs`）が検査済みで渡す。各出力位置は独立読み出しのため
/// 決定的集約順序の契約は不要。
pub fn gather(
    input: &Tensor<f32>,
    dim: usize,
    index: &Tensor<i32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let numel: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    for flat in 0..numel {
        let coords = unravel(flat, out_shape);
        let dim_idx = index.get(&coords);
        debug_assert!(
            dim_idx.is_some(),
            "gather: index の走査ロジックにバグがあり範囲外になった（契約違反）"
        );
        let dim_idx = dim_idx.unwrap_or(0) as usize;
        let mut src_coords = coords;
        src_coords[dim] = dim_idx;
        let value = input.get(&src_coords);
        debug_assert!(
            value.is_some(),
            "gather: input の走査ロジックにバグがあり範囲外になった（契約違反。index 値検査は \
             呼び出し元 Var::gather が済ませている前提）"
        );
        out.push(value.unwrap_or(0.0));
    }
    Tensor::new(out, out_shape)
}

/// [`fandhe_ai_tensor_core::BackendOps::scatter`] の CPU 実装本体
/// （イシュー #1776）。出力 shape は `input.shape()` と恒等。`reduce`
/// で `Overwrite`／`Add` を分岐する（決定的集約順序・精度契約は
/// モジュール doc・[`fandhe_ai_tensor_core::ScatterReduce`] doc を
/// 正とする）。
pub fn scatter(
    input: &Tensor<f32>,
    dim: usize,
    index: &Tensor<i32>,
    src: &Tensor<f32>,
    reduce: ScatterReduce,
) -> Result<Tensor<f32>, ShapeError> {
    let out_shape = input.shape().to_vec();
    let numel: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);
    let index_shape = index.shape().to_vec();
    let index_numel: usize = index_shape.iter().product();

    // `dim` 軸を `dim_idx` へ差し替えた `out_shape` 上の多次元添字を
    // 導出し、`out_strides` で行優先の線形添字へ畳み込む（`Overwrite`・
    // `Add` 両分岐・未知 variant フォールバックで共有する）。
    let resolve_pos = |flat: usize| -> usize {
        let coords = unravel(flat, &index_shape);
        let dim_idx = index.get(&coords);
        debug_assert!(
            dim_idx.is_some(),
            "scatter: index の走査ロジックにバグがあり範囲外になった（契約違反）"
        );
        let dim_idx = dim_idx.unwrap_or(0) as usize;
        let mut dst_coords = coords;
        dst_coords[dim] = dim_idx;
        ravel(&dst_coords, &out_strides)
    };
    let read_src = |flat: usize| -> f32 {
        let coords = unravel(flat, &index_shape);
        let value = src.get(&coords);
        debug_assert!(
            value.is_some(),
            "scatter: src の走査ロジックにバグがあり範囲外になった（契約違反）"
        );
        value.unwrap_or(0.0)
    };

    match reduce {
        ScatterReduce::Add => {
            let mut acc = Vec::with_capacity(numel);
            for flat in 0..numel {
                let coords = unravel(flat, &out_shape);
                let value = input.get(&coords);
                debug_assert!(
                    value.is_some(),
                    "scatter: input の走査ロジックにバグがあり範囲外になった（契約違反）"
                );
                acc.push(value.unwrap_or(0.0) as f64);
            }
            for flat in 0..index_numel {
                let pos = resolve_pos(flat);
                acc[pos] += read_src(flat) as f64;
            }
            let out: Vec<f32> = acc.iter().map(|&v| v as f32).collect();
            Tensor::new(out, &out_shape)
        }
        // `Overwrite`、および `ScatterReduce`（`#[non_exhaustive]`）の
        // 未知 variant は同じ「上書き」意味論へフォールバックする
        // （`autodiff::eval::scatter` と同じ安全側の割り切り方針）。
        reduce => {
            debug_assert!(
                matches!(reduce, ScatterReduce::Overwrite),
                "scatter: 未知の ScatterReduce variant へフォールバックした（契約違反）"
            );
            let mut out = Vec::with_capacity(numel);
            for flat in 0..numel {
                let coords = unravel(flat, &out_shape);
                let value = input.get(&coords);
                debug_assert!(
                    value.is_some(),
                    "scatter: input の走査ロジックにバグがあり範囲外になった（契約違反）"
                );
                out.push(value.unwrap_or(0.0));
            }
            for flat in 0..index_numel {
                let pos = resolve_pos(flat);
                out[pos] = read_src(flat);
            }
            Tensor::new(out, &out_shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_basic_2d_dim1() {
        // input = [[1,2,3],[4,5,6]], index = [[0,2],[2,1]] along dim=1
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let index = Tensor::<i32>::new(vec![0, 2, 2, 1], &[2, 2]).unwrap();
        let out = gather(&input, 1, &index, &[2, 2]).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[1.0, 3.0, 6.0, 5.0]);
    }

    #[test]
    fn scatter_overwrite_basic() {
        let input = Tensor::new(vec![0.0; 6], &[2, 3]).unwrap();
        let index = Tensor::<i32>::new(vec![0, 2, 2, 1], &[2, 2]).unwrap();
        let src = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let out = scatter(&input, 1, &index, &src, ScatterReduce::Overwrite).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[1.0, 0.0, 2.0, 0.0, 4.0, 3.0]);
    }

    #[test]
    fn scatter_add_accumulates_duplicates() {
        // 全 index が同一位置 (row 0, col 0) を指す: 加算されるはず。
        let input = Tensor::new(vec![10.0, 0.0, 0.0], &[1, 3]).unwrap();
        let index = Tensor::<i32>::new(vec![0, 0, 0], &[1, 3]).unwrap();
        let src = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let out = scatter(&input, 1, &index, &src, ScatterReduce::Add).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[16.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_add_row_major_order_matches_manual_accumulation() {
        // dim=0 の 2 つの index がともに row=0 を指し、列 0 だけへ
        // 行優先順（`src[0]` → `src[1]`）で逐次加算されることを確認
        // する。
        let input = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
        let index = Tensor::<i32>::new(vec![0, 0], &[2, 1]).unwrap();
        let src = Tensor::new(vec![1.5, 2.5], &[2, 1]).unwrap();
        let out = scatter(&input, 0, &index, &src, ScatterReduce::Add).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[4.0, 0.0, 0.0, 0.0]);
    }
}
