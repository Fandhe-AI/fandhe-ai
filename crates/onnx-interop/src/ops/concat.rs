//! ONNX `Concat` オペ（TASK-7.2c）。
//!
//! `axis` 軸に沿って複数テンソルを連結する。`axis` 以外の軸は全入力で一致していなければ
//! ならない（ONNX Concat-13 仕様）。要素コピーのみで算術を伴わないため `T: Element` で
//! ジェネリック化する（イシュー #274。`interp::Value` の `I64`／`Bool`／`F16` にも適用）。

use fandhe_ai_tensor_core::{Element, Tensor};

use super::error::OpError;
use super::normalize_axis;

/// `Concat(inputs, axis)` を計算する。`inputs` が空の場合は [`OpError::EmptyInputs`]。
pub fn concat<T: Element>(inputs: &[&Tensor<T>], axis: i64) -> Result<Tensor<T>, OpError> {
    let first = *inputs
        .first()
        .ok_or(OpError::EmptyInputs { op: "Concat" })?;
    let rank = first.rank();
    let axis = normalize_axis(axis, rank).ok_or(OpError::AxisOutOfRange {
        op: "Concat",
        axis,
        rank,
    })?;

    for t in inputs.iter().skip(1) {
        if t.rank() != rank {
            return Err(OpError::RankMismatch {
                op: "Concat",
                expected: rank,
                actual: t.rank(),
            });
        }
        for (i, (&a, &b)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
            if i != axis && a != b {
                return Err(OpError::ConcatShapeMismatch {
                    axis,
                    lhs: first.shape().to_vec(),
                    rhs: t.shape().to_vec(),
                });
            }
        }
    }

    let outer: usize = first.shape()[..axis].iter().product();
    let inner: usize = first.shape()[axis + 1..].iter().product();

    let contiguous_inputs: Vec<Tensor<T>> = inputs.iter().map(|t| t.contiguous()).collect();
    let axis_sizes: Vec<usize> = inputs.iter().map(|t| t.shape()[axis]).collect();
    let total_axis: usize = axis_sizes.iter().sum();

    let mut out_shape = first.shape().to_vec();
    out_shape[axis] = total_axis;

    let mut out = Vec::with_capacity(outer * total_axis * inner);
    for o in 0..outer {
        for (t, &axis_size) in contiguous_inputs.iter().zip(axis_sizes.iter()) {
            let slice = t
                .as_slice()
                .ok_or(OpError::NonContiguousInternal("Concat"))?;
            let start = o * axis_size * inner;
            out.extend_from_slice(&slice[start..start + axis_size * inner]);
        }
    }

    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_axis0() {
        let a = Tensor::<f32>::new(vec![1.0, 2.0], &[1, 2]).unwrap();
        let b = Tensor::<f32>::new(vec![3.0, 4.0], &[1, 2]).unwrap();
        let y = concat(&[&a, &b], 0).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn concat_axis1_negative() {
        let a = Tensor::<f32>::new(vec![1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::<f32>::new(vec![3.0, 4.0], &[2, 1]).unwrap();
        let y = concat(&[&a, &b], -1).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 3.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn concat_three_inputs() {
        let a = Tensor::<f32>::new(vec![1.0], &[1]).unwrap();
        let b = Tensor::<f32>::new(vec![2.0, 3.0], &[2]).unwrap();
        let c = Tensor::<f32>::new(vec![4.0], &[1]).unwrap();
        let y = concat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(y.shape(), &[4]);
        for (i, e) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), *e);
        }
    }

    #[test]
    fn concat_single_input_is_identity() {
        // 実装計画 3.3 ギャップ観点「単一入力」。連結対象が 1 つのみでも
        // shape・値を変えずに（コピーとして）返す。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = concat(&[&a], 0).unwrap();
        assert_eq!(y.shape(), &[3]);
        for i in 0..3 {
            assert_eq!(y.get(&[i]).unwrap(), a.get(&[i]).unwrap());
        }
    }

    #[test]
    fn concat_rank_mismatch_rejected() {
        // 実装計画 3.3 ギャップ観点「非結合軸の形状不一致エラー」の rank 違い分
        // （`ConcatShapeMismatch` ではなく `RankMismatch` になることを固定化する）。
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[2, 3, 1]).unwrap();
        let err = concat(&[&a, &b], 0).unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn concat_empty_inputs_rejected() {
        let inputs: [&Tensor<f32>; 0] = [];
        let err = concat(&inputs, 0).unwrap_err();
        assert!(matches!(err, OpError::EmptyInputs { .. }));
    }

    #[test]
    fn concat_shape_mismatch_rejected() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[2, 4]).unwrap();
        let err = concat(&[&a, &b], 0).unwrap_err();
        assert!(matches!(err, OpError::ConcatShapeMismatch { .. }));
    }
}
