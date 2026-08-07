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

/// `Relu(x) = max(x, 0)`。
pub fn relu(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_elementwise("Relu", x, |v| v.max(0.0))
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
