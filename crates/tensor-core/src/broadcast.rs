//! NumPy 互換ブロードキャスト規則の純粋関数（TASK-1.4b・#12）。
//!
//! `Tensor::broadcast_to`／`Tensor::broadcast_with`（`tensor.rs`）から
//! 呼ばれ、shape 計算（本モジュール）と `storage`/`offset` を持つ
//! `Tensor` の view 構築（`tensor.rs` 側。private フィールドへの
//! アクセスが必要なため）とで責務を分離する。
//!
//! アルゴリズムは PoC-v2-1
//! （`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/rust/src/tensor.rs`）
//! の `broadcast_shape`／`broadcast_strides` を移植したもの。
//! `docs/public-api-design.md` §2.1 が定める「NumPy 互換ブロードキャスト
//! （stride 0 による同一要素の繰り返し読み）は PoC-v2-1 の確定事項を
//! 維持する」方針、および TASK-1.4a（#215）で導入済みの `strides: Vec<isize>`
//! （stride 0 ブロードキャストを見越した型）を土台とする。
//!
//! 消費側: `backend-cpu` の elementwise カーネル（#22・TASK-1.6b）・
//! `autodiff` の演算入口（#16〜#18）が、二項演算前に
//! `Tensor::broadcast_with` を経由して両オペランドの shape を揃える
//! 想定。

use crate::error::ShapeError;

/// NumPy 互換のブロードキャスト後 shape を計算する。
///
/// 末尾軸から比較し「両者同一」または「片方が 1」であれば大きい方を
/// 採用する。rank が異なる場合は短い方の先頭に暗黙の軸長 1 を補完する
/// （NumPy の broadcasting rule と同一）。いずれの条件も満たさない軸が
/// 1 つでもあれば `ShapeError::BroadcastIncompatible` を返す。
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for i in 0..rank {
        // 末尾から数えた位置。存在しない軸（rank 差分）は暗黙の 1 として扱う。
        let l = lhs
            .len()
            .checked_sub(rank - i)
            .and_then(|idx| lhs.get(idx))
            .copied()
            .unwrap_or(1);
        let r = rhs
            .len()
            .checked_sub(rank - i)
            .and_then(|idx| rhs.get(idx))
            .copied()
            .unwrap_or(1);
        let dim = match (l, r) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => {
                return Err(ShapeError::BroadcastIncompatible {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                });
            }
        };
        shape.push(dim);
    }
    Ok(shape)
}

/// 元 shape/strides を出力 rank へ揃え、ブロードキャストで拡張された
/// 軸（元の軸長が 1 で出力側が 1 より大きい軸）の stride を 0 にする。
///
/// stride 0 は「同一要素を繰り返し読む」ことを意味し、`Tensor::get` の
/// `offset + Σ idx[i] * strides[i]` 計算がそのまま成立する
/// （idx を掛けても 0 になるため常に同一要素を指す）。呼び出し元
/// （`Tensor::broadcast_to`）が事前に `broadcast_shape` 等で互換性を
/// 検査済みであることを前提とし、本関数自体は互換性検査を行わない
/// （非公開ヘルパーであり公開 API 契約はすべて `tensor.rs` 側が担う）。
pub(crate) fn broadcast_strides(shape: &[usize], strides: &[isize], out_rank: usize) -> Vec<isize> {
    let offset = out_rank - shape.len();
    (0..out_rank)
        .map(|i| {
            if i < offset {
                // rank 補完で追加された先頭軸は元データを持たないため stride 0。
                0
            } else {
                let src_axis = i - offset;
                if shape[src_axis] == 1 {
                    0
                } else {
                    strides[src_axis]
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_official_example() {
        // NumPy 公式ドキュメントの代表例。
        assert_eq!(
            broadcast_shape(&[8, 1, 6], &[1, 5, 1]).unwrap(),
            vec![8, 5, 6]
        );
    }

    #[test]
    fn rank_padding() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn trailing_mismatch_is_incompatible() {
        let err = broadcast_shape(&[3, 4], &[3]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::BroadcastIncompatible { lhs, rhs } if lhs == vec![3, 4] && rhs == vec![3]
        ));
    }

    #[test]
    fn scalar_broadcasts_to_any_shape() {
        assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn zero_size_axis_with_one_broadcasts_to_zero() {
        assert_eq!(broadcast_shape(&[0, 3], &[1, 3]).unwrap(), vec![0, 3]);
    }

    #[test]
    fn zero_size_axis_incompatible_with_nonone() {
        let err = broadcast_shape(&[0], &[2]).unwrap_err();
        assert!(matches!(err, ShapeError::BroadcastIncompatible { .. }));
    }

    #[test]
    fn broadcast_strides_marks_expanded_axes_zero() {
        // shape [1, 5, 1] / strides [5, 1, 1] を出力 rank 3（変化なし）へ。
        let strides = broadcast_strides(&[1, 5, 1], &[5, 1, 1], 3);
        assert_eq!(strides, vec![0, 1, 0]);
    }

    #[test]
    fn broadcast_strides_pads_leading_axes_with_zero() {
        // shape [3] / strides [1] を出力 rank 3 へ（先頭 2 軸は補完で stride 0）。
        let strides = broadcast_strides(&[3], &[1], 3);
        assert_eq!(strides, vec![0, 0, 1]);
    }
}
