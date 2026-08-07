//! ONNX `Slice` オペ（TASK-7.2c。動的境界対応）。
//!
//! ONNX Slice-10 以降は `starts`/`ends`/`axes`/`steps` を（proto 上は）入力テンソルとして
//! 受け取る（Slice-1 の属性方式とは異なる）。decode 層がこれらを実行時に確定した値として
//! 解決してから本関数を呼ぶ想定のため、本モジュールは decode 層に依存しないプレーンな
//! `&[i64]` スライスで受け取る（「動的境界」対応の要点は、この値が呼び出し時点まで
//! 定数化できない前提を型として強制しないことにある）。
//!
//! `axes`/`steps` は省略可（ONNX 仕様どおり `axes` 省略時は `0..starts.len()`、`steps`
//! 省略時は全軸 1）。境界のクランプは NumPy 拡張スライス互換の規則に従う:
//! `step > 0` は `[0, dim]`・`step < 0` は `[-1, dim-1]` へクランプする。

use tensor_core::Tensor;

use super::error::OpError;
use super::normalize_axis;

/// `Slice` の実行時パラメータ（デコード層が ONNX の `starts`/`ends`/`axes`/`steps`
/// 入力テンソルを解決した後の値。すべて要素数 `starts.len()` に揃っている必要がある
/// `axes`/`steps` を除く）。
pub struct SliceParams<'a> {
    pub starts: &'a [i64],
    pub ends: &'a [i64],
    pub axes: Option<&'a [i64]>,
    pub steps: Option<&'a [i64]>,
}

/// shape から行優先 strides（`usize`）を計算する内部ヘルパー。`tensor-core::Tensor` の
/// 走査には `as_slice()`（連続バッファ）を使うため、非公開の実装詳細としてここで独自に
/// 計算する（`tensor-core::tensor::row_major_strides` は非公開のため再利用不可）。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// `Slice(data, starts, ends, axes, steps)` を計算する。
///
/// `starts`/`ends` の各要素は範囲外・巨大値（ONNX の `INT64_MAX` センチネル等）でも
/// 常にクランプする（NumPy 拡張スライス互換）ため、決して panic しない。`steps` に 0 が
/// 含まれる場合のみ [`OpError::InvalidStep`] を返す（無限ループ回避）。
pub fn slice(data: &Tensor<f32>, params: &SliceParams<'_>) -> Result<Tensor<f32>, OpError> {
    let rank = data.rank();
    let n = params.starts.len();
    if params.ends.len() != n {
        return Err(OpError::LengthMismatch {
            op: "Slice",
            name: "ends",
            expected: n,
            actual: params.ends.len(),
        });
    }
    let axes_owned: Vec<i64>;
    let axes: &[i64] = match params.axes {
        Some(a) => {
            if a.len() != n {
                return Err(OpError::LengthMismatch {
                    op: "Slice",
                    name: "axes",
                    expected: n,
                    actual: a.len(),
                });
            }
            a
        }
        None => {
            axes_owned = (0..n as i64).collect();
            &axes_owned
        }
    };
    let steps_owned: Vec<i64>;
    let steps: &[i64] = match params.steps {
        Some(s) => {
            if s.len() != n {
                return Err(OpError::LengthMismatch {
                    op: "Slice",
                    name: "steps",
                    expected: n,
                    actual: s.len(),
                });
            }
            s
        }
        None => {
            steps_owned = vec![1; n];
            &steps_owned
        }
    };

    // per_axis[d] = (start, step, out_len)。指定されなかった軸は全範囲・step 1 のまま。
    let mut per_axis: Vec<(i64, i64, usize)> =
        data.shape().iter().map(|&d| (0i64, 1i64, d)).collect();
    let mut seen_axes: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        let axis = normalize_axis(axes[i], rank).ok_or(OpError::AxisOutOfRange {
            op: "Slice",
            axis: axes[i],
            rank,
        })?;
        if seen_axes.contains(&axis) {
            return Err(OpError::DuplicateAxis { op: "Slice", axis });
        }
        seen_axes.push(axis);

        let step = steps[i];
        if step == 0 {
            return Err(OpError::InvalidStep { op: "Slice", axis });
        }
        let dim = data.shape()[axis] as i64;
        // NumPy 拡張スライス互換のクランプ境界: 正 step は [0, dim]、負 step は [-1, dim-1]。
        let (lo, hi) = if step > 0 {
            (0i64, dim)
        } else {
            (-1i64, dim - 1)
        };
        let clamp = |raw: i64| -> i64 {
            let normalized = if raw < 0 {
                raw.saturating_add(dim)
            } else {
                raw
            };
            normalized.clamp(lo, hi)
        };
        let start = clamp(params.starts[i]);
        let end = clamp(params.ends[i]);

        let out_len = if step > 0 {
            if end > start {
                ((end - start) + step - 1) / step
            } else {
                0
            }
        } else if start > end {
            ((start - end) + (-step) - 1) / (-step)
        } else {
            0
        };

        per_axis[axis] = (start, step, out_len as usize);
    }

    let out_shape: Vec<usize> = per_axis.iter().map(|&(_, _, len)| len).collect();
    let total: usize = out_shape.iter().product();

    let data_c = data.contiguous();
    let data_slice = data_c
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Slice"))?;
    let in_strides = row_major_strides(data.shape());

    let mut out = Vec::with_capacity(total);
    let mut counters = vec![0usize; rank];
    for _ in 0..total {
        let mut flat: i64 = 0;
        for d in 0..rank {
            let (start, step, _) = per_axis[d];
            let coord = start + counters[d] as i64 * step;
            flat += coord * in_strides[d] as i64;
        }
        out.push(data_slice[flat as usize]);

        for d in (0..rank).rev() {
            counters[d] += 1;
            if counters[d] < out_shape[d] {
                break;
            }
            counters[d] = 0;
        }
    }

    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_positive_step() {
        // data: [0..10), slice [2,7) step 1 -> [2,3,4,5,6]
        let data = Tensor::<f32>::new((0..10).map(|v| v as f32).collect(), &[10]).unwrap();
        let y = slice(
            &data,
            &SliceParams {
                starts: &[2],
                ends: &[7],
                axes: None,
                steps: None,
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[5]);
        for (i, e) in [2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), *e);
        }
    }

    #[test]
    fn negative_indices_and_axes() {
        let data = Tensor::<f32>::new((0..10).map(|v| v as f32).collect(), &[10]).unwrap();
        // starts=-4, ends=-1 -> [6,7,8]
        let y = slice(
            &data,
            &SliceParams {
                starts: &[-4],
                ends: &[-1],
                axes: Some(&[-1]),
                steps: None,
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[3]);
        for (i, e) in [6.0, 7.0, 8.0].iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), *e);
        }
    }

    #[test]
    fn negative_step_reverses() {
        let data = Tensor::<f32>::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        // starts=4, ends=-6 (clamped -> -1), step=-1: reverses entire tensor
        let y = slice(
            &data,
            &SliceParams {
                starts: &[4],
                ends: &[-6],
                axes: None,
                steps: Some(&[-1]),
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[5]);
        for (i, e) in [4.0, 3.0, 2.0, 1.0, 0.0].iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), *e);
        }
    }

    #[test]
    fn int_max_sentinel_end_clamps() {
        let data = Tensor::<f32>::new((0..5).map(|v| v as f32).collect(), &[5]).unwrap();
        let y = slice(
            &data,
            &SliceParams {
                starts: &[1],
                ends: &[i64::MAX],
                axes: None,
                steps: None,
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[4]);
        for (i, e) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), *e);
        }
    }

    #[test]
    fn step_two_positive_dim() {
        let data = Tensor::<f32>::new((0..10).map(|v| v as f32).collect(), &[10]).unwrap();
        let y = slice(
            &data,
            &SliceParams {
                starts: &[0],
                ends: &[10],
                axes: None,
                steps: Some(&[2]),
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[5]);
        for (i, e) in [0.0, 2.0, 4.0, 6.0, 8.0].iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), *e);
        }
    }

    #[test]
    fn multi_axis_2d() {
        // data: [4,4] 0..16 row-major. slice rows [1,3) all step 1, cols [0,4) step 2.
        let data = Tensor::<f32>::new((0..16).map(|v| v as f32).collect(), &[4, 4]).unwrap();
        let y = slice(
            &data,
            &SliceParams {
                starts: &[1, 0],
                ends: &[3, 4],
                axes: Some(&[0, 1]),
                steps: Some(&[1, 2]),
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        // row1: [4,5,6,7] -> cols 0,2 = 4,6 ; row2: [8,9,10,11] -> 8,10
        assert_eq!(y.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 6.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 8.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 10.0);
    }

    #[test]
    fn step_zero_rejected() {
        let data = Tensor::<f32>::zeros(&[5]).unwrap();
        let err = slice(
            &data,
            &SliceParams {
                starts: &[0],
                ends: &[5],
                axes: None,
                steps: Some(&[0]),
            },
        )
        .unwrap_err();
        assert!(matches!(err, OpError::InvalidStep { .. }));
    }

    #[test]
    fn empty_range_yields_zero_length_axis() {
        let data = Tensor::<f32>::new((0..5).map(|v| v as f32).collect(), &[5]).unwrap();
        let y = slice(
            &data,
            &SliceParams {
                starts: &[3],
                ends: &[3],
                axes: None,
                steps: None,
            },
        )
        .unwrap();
        assert_eq!(y.shape(), &[0]);
    }

    #[test]
    fn duplicate_axis_rejected() {
        // 実装計画 3.3 ギャップ観点。同一軸を `axes` に重複指定した場合を固定化する。
        let data = Tensor::<f32>::zeros(&[5, 5]).unwrap();
        let err = slice(
            &data,
            &SliceParams {
                starts: &[0, 1],
                ends: &[2, 3],
                axes: Some(&[0, 0]),
                steps: None,
            },
        )
        .unwrap_err();
        assert!(matches!(err, OpError::DuplicateAxis { axis: 0, .. }));
    }

    #[test]
    fn lengths_mismatch_rejected() {
        let data = Tensor::<f32>::zeros(&[5, 5]).unwrap();
        let err = slice(
            &data,
            &SliceParams {
                starts: &[0, 0],
                ends: &[1],
                axes: None,
                steps: None,
            },
        )
        .unwrap_err();
        assert!(matches!(err, OpError::LengthMismatch { name: "ends", .. }));
    }
}
