//! ONNX `Shape`／`Unsqueeze` オペ（TASK-7.2c）。

use tensor_core::Tensor;

use super::error::OpError;
use super::normalize_axis;

/// `Shape(x)`: `x` の各軸サイズを ONNX 仕様どおり `int64` 相当（`Vec<i64>`）の
/// 1 次元列で返す。`tensor-core::Element` は `i64` を実装しないため（TASK-1.4a 時点の
/// 対象型は `f32`/`f64`/`i32`/`half::f16` のみ）、本関数は `Tensor<i64>` ではなく生の
/// `Vec<i64>` を返す。デコード層が ONNX の `TensorProto`（int64 データ）へ変換する際は
/// この列をそのまま書き込める。
pub fn shape(x: &Tensor<f32>) -> Vec<i64> {
    x.shape().iter().map(|&d| d as i64).collect()
}

/// `Unsqueeze(x, axes)`: `axes`（ONNX 負軸表記対応）が指す位置にサイズ 1 の軸を挿入する。
/// `axes` は出力 rank（`x.rank() + axes.len()`）に対して正規化する（ONNX Unsqueeze-13
/// 仕様どおり）。重複軸は [`OpError::DuplicateAxis`] を返す。
pub fn unsqueeze(x: &Tensor<f32>, axes: &[i64]) -> Result<Tensor<f32>, OpError> {
    let out_rank = x.rank() + axes.len();
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let n = normalize_axis(axis, out_rank).ok_or(OpError::AxisOutOfRange {
            op: "Unsqueeze",
            axis,
            rank: out_rank,
        })?;
        normalized.push(n);
    }
    normalized.sort_unstable();
    for pair in normalized.windows(2) {
        if pair[0] == pair[1] {
            return Err(OpError::DuplicateAxis {
                op: "Unsqueeze",
                axis: pair[0],
            });
        }
    }

    let mut new_shape = Vec::with_capacity(out_rank);
    let mut src = x.shape().iter();
    for i in 0..out_rank {
        if normalized.binary_search(&i).is_ok() {
            new_shape.push(1);
        } else {
            // `normalized` の要素数は out_rank - x.rank() であり、挿入位置以外は
            // 必ず元の軸を 1 つずつ消費する（out_rank 回のループで src がちょうど
            // 尽きる不変条件は上記の重複検査・rank 計算により保証される）。
            match src.next() {
                Some(&d) => new_shape.push(d),
                None => {
                    return Err(OpError::AxisOutOfRange {
                        op: "Unsqueeze",
                        axis: i as i64,
                        rank: out_rank,
                    });
                }
            }
        }
    }

    x.contiguous().reshape(&new_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_returns_dims_as_i64() {
        let t = Tensor::<f32>::zeros(&[2, 3, 4]).unwrap();
        assert_eq!(shape(&t), vec![2, 3, 4]);
    }

    #[test]
    fn shape_of_rank_zero_is_empty() {
        let t = Tensor::<f32>::new(vec![1.0], &[]).unwrap();
        assert_eq!(shape(&t), Vec::<i64>::new());
    }

    #[test]
    fn unsqueeze_inserts_axis_at_position() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let y = unsqueeze(&t, &[0]).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3]);
        assert_eq!(y.get(&[0, 1, 2]).unwrap(), t.get(&[1, 2]).unwrap());
    }

    #[test]
    fn unsqueeze_negative_axis() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let y = unsqueeze(&t, &[-1]).unwrap();
        assert_eq!(y.shape(), &[2, 3, 1]);
    }

    #[test]
    fn unsqueeze_multiple_axes() {
        let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = unsqueeze(&t, &[0, 2]).unwrap();
        assert_eq!(y.shape(), &[1, 3, 1]);
    }

    #[test]
    fn unsqueeze_duplicate_axis_rejected() {
        let t = Tensor::<f32>::zeros(&[3]).unwrap();
        let err = unsqueeze(&t, &[0, 0]).unwrap_err();
        assert!(matches!(err, OpError::DuplicateAxis { axis: 0, .. }));
    }

    #[test]
    fn unsqueeze_axis_out_of_range_rejected() {
        let t = Tensor::<f32>::zeros(&[3]).unwrap();
        let err = unsqueeze(&t, &[5]).unwrap_err();
        assert!(matches!(err, OpError::AxisOutOfRange { axis: 5, .. }));
    }
}
