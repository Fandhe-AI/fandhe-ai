//! ONNX `Gather` オペ（TASK-7.2c）。
//!
//! `axis` 軸に沿って `indices` が指す要素を集める。出力 shape は
//! `data.shape[:axis] + indices_shape + data.shape[axis+1:]`（ONNX Gather-13 仕様）。
//! `indices` 自体は ONNX 仕様上つねに `int64` テンソルであり本関数の型パラメータとは
//! 別に生の `&[i64]` + 形状 `indices_shape` で受け取るが、`data`（集める対象）は
//! 要素コピーのみで算術を伴わないため `T: Element` でジェネリック化する
//! （イシュー #274。`interp::Value` の `F32`／`I64`／`Bool`／`F16` いずれの
//! `data` に対しても同じ実装で動作する）。

use tensor_core::{Element, Tensor};

use super::error::OpError;
use super::normalize_axis;

/// `Gather(data, indices, axis)` を計算する。
///
/// `indices` の各要素は負値許容（`index + data.shape()[axis]` で正規化。ONNX 仕様）。
/// 正規化後も範囲外の場合は [`OpError::IndexOutOfRange`]（OWASP A03: 外部入力の範囲検査を
/// 先に行い、`data_slice` への読み出し前に弾く。`.claude/rules/security.md`）。
pub fn gather<T: Element>(
    data: &Tensor<T>,
    indices: &[i64],
    indices_shape: &[usize],
    axis: i64,
) -> Result<Tensor<T>, OpError> {
    let rank = data.rank();
    let axis = normalize_axis(axis, rank).ok_or(OpError::AxisOutOfRange {
        op: "Gather",
        axis,
        rank,
    })?;
    let dim_size = data.shape()[axis];

    let expected_len: usize = indices_shape.iter().product();
    if indices.len() != expected_len {
        return Err(OpError::LengthMismatch {
            op: "Gather",
            name: "indices",
            expected: expected_len,
            actual: indices.len(),
        });
    }

    let mut normalized_indices = Vec::with_capacity(indices.len());
    for &idx in indices {
        let n = if idx < 0 { idx + dim_size as i64 } else { idx };
        if n < 0 || n as usize >= dim_size {
            return Err(OpError::IndexOutOfRange {
                op: "Gather",
                index: idx,
                dim_size,
            });
        }
        normalized_indices.push(n as usize);
    }

    let data_c = data.contiguous();
    let data_slice = data_c
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Gather"))?;

    let outer: usize = data.shape()[..axis].iter().product();
    let inner: usize = data.shape()[axis + 1..].iter().product();

    let mut out_shape = Vec::with_capacity(rank - 1 + indices_shape.len());
    out_shape.extend_from_slice(&data.shape()[..axis]);
    out_shape.extend_from_slice(indices_shape);
    out_shape.extend_from_slice(&data.shape()[axis + 1..]);

    let mut out = Vec::with_capacity(outer * normalized_indices.len() * inner);
    for o in 0..outer {
        for &gi in &normalized_indices {
            let base = (o * dim_size + gi) * inner;
            out.extend_from_slice(&data_slice[base..base + inner]);
        }
    }

    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_rows_axis0() {
        // data: [3,2], indices: [0,2] -> rows 0 and 2
        let data = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let y = gather(&data, &[0, 2], &[2], 0).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 5.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 6.0);
    }

    #[test]
    fn gather_axis1_with_negative_index() {
        let data = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        // axis 1, indices [-1, 0] -> column 2, column 0
        let y = gather(&data, &[-1, 0], &[2], 1).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 3.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 1.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 6.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn gather_negative_axis() {
        // 実装計画 3.3 ギャップ観点「axis 負値（`normalize_axis` 経由）」。
        // data: [2,3], axis=-1（axis 1 相当）, indices=[2,0] -> 列 2, 列 0。
        let data = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let y = gather(&data, &[2, 0], &[2], -1).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 3.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 1.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 6.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn gather_index_out_of_range_rejected() {
        let data = Tensor::<f32>::zeros(&[3, 2]).unwrap();
        let err = gather(&data, &[3], &[1], 0).unwrap_err();
        assert!(matches!(
            err,
            OpError::IndexOutOfRange {
                index: 3,
                dim_size: 3,
                ..
            }
        ));
    }

    #[test]
    fn gather_indices_length_mismatch_rejected() {
        let data = Tensor::<f32>::zeros(&[3, 2]).unwrap();
        let err = gather(&data, &[0, 1], &[3], 0).unwrap_err();
        assert!(matches!(
            err,
            OpError::LengthMismatch {
                expected: 3,
                actual: 2,
                ..
            }
        ));
    }
}
