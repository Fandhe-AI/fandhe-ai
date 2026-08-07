//! ONNX `Add`／`Mul`／`Div`／`Mod`／`Sqrt` MVP 算術オペ（TASK-7.3a・#82）。
//!
//! `Add`／`Mul`／`Div`／`Mod` は ONNX 仕様上 multidirectional broadcasting
//! （NumPy 互換）に対応する二項演算であり、`tensor_core::Tensor::broadcast_with`
//! （`broadcast_shape` 委譲。`crates/tensor-core/src/tensor.rs`）で出力 shape へ
//! 揃えた view を得てから要素ごとに計算する。`Sqrt` は `activation.rs` の
//! `Relu`/`Sigmoid` と同じ単項要素ごと写像パターンに従う。

use tensor_core::Tensor;

use super::error::OpError;

/// 二項要素ごと演算の共通実装。`lhs`／`rhs` を `broadcast_with` で共通 shape の
/// view へ揃え、`contiguous()` で実体化してから `f` を適用する（`activation.rs`
/// の `map_elementwise` と同様、非 contiguous view をそのまま走査しない）。
fn map_binary_elementwise(
    op: &'static str,
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor<f32>, OpError> {
    let (l, r) = lhs.broadcast_with(rhs)?;
    let lc = l.contiguous();
    let rc = r.contiguous();
    let l_slice = lc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let r_slice = rc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let data: Vec<f32> = l_slice
        .iter()
        .zip(r_slice.iter())
        .map(|(&a, &b)| f(a, b))
        .collect();
    Tensor::new(data, lc.shape()).map_err(OpError::from)
}

/// `Add(a, b) = a + b`（multidirectional broadcasting）。
pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_binary_elementwise("Add", a, b, |x, y| x + y)
}

/// `Mul(a, b) = a * b`（multidirectional broadcasting）。
pub fn mul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_binary_elementwise("Mul", a, b, |x, y| x * y)
}

/// `Div(a, b) = a / b`（multidirectional broadcasting）。IEEE 754 の除算規則を
/// そのまま透過し（0 除算は `inf`／`-inf`／`NaN`）、事前にゼロ検査でエラーには
/// しない。ONNX の `Div` 自体も floating point 入力に対する 0 除算の扱いを
/// IEEE 754 に委ねている。
pub fn div(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_binary_elementwise("Div", a, b, |x, y| x / y)
}

/// `Mod(a, b)`。ONNX `Mod-13` 仕様の `fmod` 属性を明示的に受け取る:
/// `fmod=1`（`fmod: true`）は C `fmod` 相当（結果の符号は被除数 `a` に従う。
/// Rust の `%` 演算子と同一）で、浮動小数点入力に対して仕様上有効な唯一の
/// モードである。`fmod=0`（`fmod: false`）は ONNX 仕様上「整数入力のみ」
/// 有効な Python 風モード（結果の符号は除数 `b` に従う）であり、`f32` 入力に
/// 対して要求された場合は黙って `rem_euclid` 等で代替せず
/// [`OpError::UnsupportedFmodMode`] を返す（誤った数値を静かに返さない。
/// `.claude/rules/security.md` A03 の「外部入力の検証」に準ずる）。
pub fn modulo(a: &Tensor<f32>, b: &Tensor<f32>, fmod: bool) -> Result<Tensor<f32>, OpError> {
    if !fmod {
        return Err(OpError::UnsupportedFmodMode { op: "Mod" });
    }
    map_binary_elementwise("Mod", a, b, |x, y| x % y)
}

/// `Sqrt(x) = sqrt(x)`。負値入力は ONNX 仕様上未定義域であり、`f32::sqrt` の
/// IEEE 754 準拠動作（`NaN`）をそのまま透過する（`Div` の 0 除算と同じ方針）。
pub fn sqrt(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    let xc = x.contiguous();
    let slice = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Sqrt"))?;
    let data: Vec<f32> = slice.iter().map(|&v| v.sqrt()).collect();
    Tensor::new(data, xc.shape()).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_same_shape() {
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let y = add(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 44.0);
    }

    #[test]
    fn add_broadcast_row_and_column() {
        // [3,1] + [1,4] -> [3,4]（NumPy 互換ブロードキャスト。ops_shape.rs の
        // elementwise_broadcast_row_and_column と同じ代表例）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
        let y = add(&a, &b).unwrap();
        assert_eq!(y.shape(), &[3, 4]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(y.get(&[2, 3]).unwrap(), 43.0);
    }

    #[test]
    fn add_broadcast_incompatible_rejected() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[4]).unwrap();
        let err = add(&a, &b).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }

    #[test]
    fn mul_broadcast_scalar() {
        // rank 差分（[2,3] と [3]）の暗黙先頭軸補完。
        let a = Tensor::<f32>::new((1..=6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let b = Tensor::<f32>::new(vec![2.0, 2.0, 2.0], &[3]).unwrap();
        let y = mul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(y.get(&[1, 2]).unwrap(), 12.0);
    }

    #[test]
    fn div_known_values() {
        let a = Tensor::<f32>::new(vec![10.0, 9.0, 8.0, 7.0], &[4]).unwrap();
        let b = Tensor::<f32>::new(vec![2.0, 3.0, 4.0, 7.0], &[4]).unwrap();
        let y = div(&a, &b).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 5.0);
        assert_eq!(y.get(&[1]).unwrap(), 3.0);
        assert_eq!(y.get(&[2]).unwrap(), 2.0);
        assert_eq!(y.get(&[3]).unwrap(), 1.0);
    }

    #[test]
    fn div_by_zero_follows_ieee754() {
        let a = Tensor::<f32>::new(vec![1.0, -1.0, 0.0], &[3]).unwrap();
        let b = Tensor::<f32>::new(vec![0.0, 0.0, 0.0], &[3]).unwrap();
        let y = div(&a, &b).unwrap();
        assert!(y.get(&[0]).unwrap().is_infinite() && y.get(&[0]).unwrap() > 0.0);
        assert!(y.get(&[1]).unwrap().is_infinite() && y.get(&[1]).unwrap() < 0.0);
        assert!(y.get(&[2]).unwrap().is_nan());
    }

    #[test]
    fn modulo_fmod_matches_c_fmod_semantics() {
        // fmod=1: 結果の符号は被除数（a）に従う（ONNX Mod-13 仕様）。
        let a = Tensor::<f32>::new(vec![7.0, -7.0, 7.0, -7.0], &[4]).unwrap();
        let b = Tensor::<f32>::new(vec![3.0, 3.0, -3.0, -3.0], &[4]).unwrap();
        let y = modulo(&a, &b, true).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 1.0);
        assert_eq!(y.get(&[1]).unwrap(), -1.0);
        assert_eq!(y.get(&[2]).unwrap(), 1.0);
        assert_eq!(y.get(&[3]).unwrap(), -1.0);
    }

    #[test]
    fn modulo_python_style_rejected_for_float_input() {
        let a = Tensor::<f32>::new(vec![7.0], &[1]).unwrap();
        let b = Tensor::<f32>::new(vec![3.0], &[1]).unwrap();
        let err = modulo(&a, &b, false).unwrap_err();
        assert!(matches!(err, OpError::UnsupportedFmodMode { op: "Mod" }));
    }

    #[test]
    fn sqrt_known_values() {
        let x = Tensor::<f32>::new(vec![0.0, 1.0, 4.0, 9.0], &[4]).unwrap();
        let y = sqrt(&x).unwrap();
        assert_eq!(y.get(&[0]).unwrap(), 0.0);
        assert_eq!(y.get(&[1]).unwrap(), 1.0);
        assert_eq!(y.get(&[2]).unwrap(), 2.0);
        assert_eq!(y.get(&[3]).unwrap(), 3.0);
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let x = Tensor::<f32>::new(vec![-1.0], &[1]).unwrap();
        let y = sqrt(&x).unwrap();
        assert!(y.get(&[0]).unwrap().is_nan());
    }
}
