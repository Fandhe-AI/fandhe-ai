//! ONNX `Relu`／`Sigmoid` 要素ごと活性化オペ（TASK-7.2c）。
//!
//! いずれも属性を持たない純粋な要素ごと写像であり、入力の shape をそのまま維持する。

use tensor_core::Tensor;

use super::error::OpError;

/// 要素ごとに `f` を適用した新しいテンソルを返す共通実装。非 contiguous な入力
/// （transpose/narrow 後の view 等）は `contiguous()` で実体化してから走査する。
fn map_elementwise(
    op: &'static str,
    x: &Tensor<f32>,
    f: impl Fn(f32) -> f32,
) -> Result<Tensor<f32>, OpError> {
    let xc = x.contiguous();
    let slice = xc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let data: Vec<f32> = slice.iter().map(|&v| f(v)).collect();
    Tensor::new(data, xc.shape()).map_err(OpError::from)
}

/// NaN を伝播する `max`。`f32::max` は NaN 入力を暗黙に非 NaN 側へ潰す
/// （IEEE 754 の `maxNum` 系挙動）ため、`Relu` にそのまま使うと ONNX Runtime・
/// `autodiff::eval::relu`（`crates/autodiff/src/eval.rs`）の NaN 伝播動作と
/// 不整合になり、バックエンド数値一致検証で上流の数値破壊（NaN）が
/// 隠蔽されてしまう。両者の意味論を揃えるため同じ判定を用いる。
fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `Relu(x) = max(x, 0)`。NaN 入力は `nan_propagating_max` により NaN のまま返す
/// （`autodiff::eval::relu` と同じ意味論。ONNX Runtime とも整合する）。
pub fn relu(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_elementwise("Relu", x, |v| nan_propagating_max(v, 0.0))
}

/// `Sigmoid(x) = 1 / (1 + exp(-x))`。
pub fn sigmoid(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_elementwise("Sigmoid", x, |v| 1.0 / (1.0 + (-v).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clamps_negative_to_zero() {
        let x = Tensor::<f32>::new(vec![-2.0, -0.5, 0.0, 1.5, 3.0], &[5]).unwrap();
        let y = relu(&x).unwrap();
        assert_eq!(y.shape(), &[5]);
        let expected = [0.0, 0.0, 0.0, 1.5, 3.0];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), e);
        }
    }

    #[test]
    fn relu_propagates_nan() {
        // `f32::max` は NaN を暗黙に 0.0 へ潰すため NaN 非伝播バグの回帰確認
        // （autodiff::eval::relu・ONNX Runtime との整合。#272 レビュー指摘）。
        let x = Tensor::<f32>::new(vec![f32::NAN, -1.0, 2.0], &[3]).unwrap();
        let y = relu(&x).unwrap();
        assert!(y.get(&[0]).unwrap().is_nan());
        assert_eq!(y.get(&[1]).unwrap(), 0.0);
        assert_eq!(y.get(&[2]).unwrap(), 2.0);
    }

    #[test]
    fn relu_on_non_contiguous_view() {
        let t = Tensor::<f32>::new(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        let y = relu(&tt).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 0.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 2.0);
    }

    #[test]
    fn sigmoid_known_values() {
        let x = Tensor::<f32>::new(vec![0.0, 1.0, -1.0], &[3]).unwrap();
        let y = sigmoid(&x).unwrap();
        let tol = 1e-6;
        assert!((y.get(&[0]).unwrap() - 0.5).abs() < tol);
        assert!((y.get(&[1]).unwrap() - 0.731_058_6).abs() < 1e-5);
        assert!((y.get(&[2]).unwrap() - 0.268_941_4).abs() < 1e-5);
    }
}
