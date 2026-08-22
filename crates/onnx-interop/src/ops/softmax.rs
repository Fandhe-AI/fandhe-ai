//! ONNX `Softmax` オペ（opset 13 семантик。TASK-7.3c・#84）。
//!
//! Attention 計算の `softmax(QK^T / sqrt(d))` に必須。opset 13 では `axis` 属性が指す
//! 単一軸方向にのみ正規化する（opset 13 未満の「入力全体を 2 次元へ強制変換
//! （coerced 2-D）してから正規化する」セマンティクスとは異なる）。本実装は
//! transformer.onnx（opset 13+ 前提。TASK-7.4）のみを対象とし、旧セマンティクスは
//! スコープ外とする（`out-of-scope-tracking.md`。必要が判明すれば Issue で追跡）。

use fandhe_ai_tensor_core::Tensor;

use super::error::OpError;
use super::normalize_axis;

/// `axis` 方向へ `Softmax` を計算する（opset 13 セマンティクス）。
///
/// `axis` の既定値（opset 13 は `-1`）の適用は decode 層結線タスクの責務とし、本関数は
/// 明示引数で受ける（`arith::modulo` の `fmod` 引数と同じ先例）。範囲外の軸指定は
/// [`OpError::AxisOutOfRange`]。
///
/// 数値安定化のため軸方向の最大値を減算してから `exp` を取る（オーバーフロー／
/// `NaN` 化を防ぐ標準的な実装。`security.md` の「静かな数値破損防止」に対応）。
pub fn softmax(x: &Tensor<f32>, axis: i64) -> Result<Tensor<f32>, OpError> {
    let rank = x.rank();
    let axis_norm = normalize_axis(axis, rank).ok_or(OpError::AxisOutOfRange {
        op: "Softmax",
        axis,
        rank,
    })?;

    let xc = x.contiguous();
    let shape = xc.shape();
    let slice = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Softmax"))?;

    // 行優先連続バッファを「軸より前（outer）× 軸自身（axis_len）× 軸より後（inner）」の
    // 3 重構造として扱う。ある outer/inner の組を固定したとき、軸方向の要素は
    // `inner` 要素おきに現れる（行優先ストライドの定義そのもの）。
    let outer: usize = shape[..axis_norm].iter().product();
    let axis_len = shape[axis_norm];
    let inner: usize = shape[axis_norm + 1..].iter().product();

    let mut out = vec![0f32; slice.len()];

    // `outer`・`axis_len`・`inner` のいずれかが 0（`slice` が空）なら結果も空であり、
    // ループ本体は何も書き込まない。しかし `checked_numel`（tensor-core）は
    // `[usize::MAX, 0]` のような「積が 0 になる巨大次元との組合せ」形状を正規に許容する
    // ため、素朴に `outer`/`inner` をループ境界に使うと空の `slice` に対して
    // `usize::MAX` 回反復するハングを引き起こしうる（PR #276 Bugbot 指摘。`MatMul` の
    // 形状検証方針と揃え、空データはループに入る前に早期リターンする。OWASP A03。
    // `.claude/rules/security.md`）。
    if slice.is_empty() {
        return Tensor::new(out, shape).map_err(OpError::from);
    }

    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            // 数値安定化: 軸方向の最大値を求めてから減算する（`exp` のオーバーフロー防止）。
            let mut max_v = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let v = slice[base + a * inner];
                if v > max_v {
                    max_v = v;
                }
            }
            let mut sum = 0f32;
            for a in 0..axis_len {
                let e = (slice[base + a * inner] - max_v).exp();
                out[base + a * inner] = e;
                sum += e;
            }
            for a in 0..axis_len {
                out[base + a * inner] /= sum;
            }
        }
    }

    Tensor::new(out, shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_1d_known_values() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = softmax(&x, 0).unwrap();
        let tol = 1e-6;
        assert!((y.get(&[0]).unwrap() - 0.090_030_57).abs() < tol);
        assert!((y.get(&[1]).unwrap() - 0.244_728_47).abs() < tol);
        assert!((y.get(&[2]).unwrap() - 0.665_240_96).abs() < tol);
    }

    #[test]
    fn softmax_2d_axis1_rows_sum_to_one() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0], &[2, 3]).unwrap();
        let y = softmax(&x, 1).unwrap();
        for row in 0..2 {
            let sum: f32 = (0..3).map(|c| y.get(&[row, c]).unwrap()).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        // 行間独立: row1 は一様分布に近い（[1,1] は全軸で同値ではないが対称性を確認）。
        assert!((y.get(&[1, 1]).unwrap() - y.get(&[1, 2]).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn softmax_negative_axis_matches_last_axis() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let y_neg = softmax(&x, -1).unwrap();
        let y_pos = softmax(&x, 1).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(y_neg.get(&[r, c]).unwrap(), y_pos.get(&[r, c]).unwrap());
            }
        }
    }

    #[test]
    fn softmax_3d_middle_axis() {
        // shape [2,3,2]・axis=1（中間軸）。inner=2・outer=2・axis_len=3 のループが
        // 正しく機能することを、各 (outer,inner) の組で総和 1.0 になることで確認する。
        let x = Tensor::<f32>::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 3, 2],
        )
        .unwrap();
        let y = softmax(&x, 1).unwrap();
        for o in 0..2 {
            for i in 0..2 {
                let sum: f32 = (0..3).map(|a| y.get(&[o, a, i]).unwrap()).sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn softmax_numerically_stable_for_large_values() {
        let x = Tensor::<f32>::new(vec![1000.0, 1001.0], &[2]).unwrap();
        let y = softmax(&x, 0).unwrap();
        let a = y.get(&[0]).unwrap();
        let b = y.get(&[1]).unwrap();
        assert!(a.is_finite() && b.is_finite());
        assert!((a - 0.268_94).abs() < 1e-4);
        assert!((b - 0.731_06).abs() < 1e-4);
    }

    #[test]
    fn axis_out_of_range_rejected() {
        let x = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = softmax(&x, 5).unwrap_err();
        assert!(matches!(err, OpError::AxisOutOfRange { .. }));
    }

    #[test]
    fn softmax_empty_axis_with_huge_sibling_dim_does_not_hang() {
        // `checked_numel`（tensor-core）は積が 0 になる形状（[usize::MAX, 0] 等）を
        // 正規に許容するため、`axis` 自体のサイズが 0 で兄弟次元（`outer`）が
        // `usize::MAX` のような巨大値の場合、素朴な実装だと空データにもかかわらず
        // `outer` を `usize::MAX` 回反復してハングしうる（PR #276 Bugbot 指摘）。
        // 空形状は早期リターンし、即座に空結果が返ることを確認する。
        let x = Tensor::<f32>::new(Vec::new(), &[usize::MAX, 0]).unwrap();
        let y = softmax(&x, 1).unwrap();
        assert_eq!(y.shape(), &[usize::MAX, 0]);
    }
}
