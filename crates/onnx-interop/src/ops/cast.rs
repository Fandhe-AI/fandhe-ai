//! ONNX `Cast` オペ（TASK-7.3b・#83。イシュー #274 で型安全化・BOOL／FLOAT16 対応）。
//!
//! `tensor-core::Element` に `i64`／`bool` を追加した（イシュー #274・
//! `crates/tensor-core/src/element.rs`）ことに伴い、以前 `Vec<i64>` の素表現を
//! 経由していた `Cast` の i64 側も `Tensor<i64>` を直接扱う型安全な関数へ置き換える。
//! `to`（ONNX `TensorProto.DataType`）の解決・グラフ実行時の dtype 分岐は
//! インタープリタ結線（`onnx::interp`）の責務であり、本モジュールは decode 層・
//! 実行時型システムに関知しない「dtype 間の単体変換関数」のみを提供する
//! （`ops/mod.rs` 冒頭コメントの方針どおり）。対応範囲は `transformer.onnx`
//! （PyTorch エクスポート・opset 13 以降相当）が使用しうる `FLOAT(1)`／`INT64(7)`／
//! `BOOL(9)`／`FLOAT16(10)` の相互変換。

use half::f16;
use tensor_core::Tensor;

use super::error::OpError;

/// 非 contiguous な入力を実体化してスライスを取り出す共通ヘルパ。
fn contiguous_slice<T: tensor_core::Element>(
    op: &'static str,
    x: &Tensor<T>,
) -> Result<(Tensor<T>, Vec<T>), OpError> {
    let xc = x.contiguous();
    let data = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal(op))?
        .to_vec();
    Ok((xc, data))
}

/// `Cast(x, to=INT64)`: `f32` テンソルを `Tensor<i64>` へ変換する。
///
/// ONNX `Cast` 仕様は浮動小数点 → 整数をゼロ方向切り捨て（truncation）と定める。
/// Rust の `as` キャストは IEEE754 → 整数で同じくゼロ方向丸めを行い、かつ範囲外・NaN を
/// 未定義動作ではなく飽和（`i64::MIN`/`i64::MAX`）・0 に写像する（Rust Reference の
/// `numeric cast` 仕様）ため、追加の事前検査なしに安全に使える。
pub fn cast_to_int64(x: &Tensor<f32>) -> Result<Tensor<i64>, OpError> {
    let (xc, data) = contiguous_slice("Cast", x)?;
    let out: Vec<i64> = data.iter().map(|&v| v as i64).collect();
    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

/// `Cast(x, to=FLOAT)`: `Tensor<i64>` を `f32` テンソルへ変換する。
pub fn cast_to_float(x: &Tensor<i64>) -> Result<Tensor<f32>, OpError> {
    let (xc, data) = contiguous_slice("Cast", x)?;
    let out: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

/// `Cast(x, to=BOOL)`: `f32` テンソルを `Tensor<bool>` へ変換する。
/// ONNX/NumPy 慣習に従い非ゼロ（`NaN` を含む）を `true`、ゼロを `false` とする。
pub fn cast_to_bool(x: &Tensor<f32>) -> Result<Tensor<bool>, OpError> {
    let (xc, data) = contiguous_slice("Cast", x)?;
    let out: Vec<bool> = data.iter().map(|&v| v != 0.0).collect();
    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

/// `Cast(x, to=FLOAT)`: `Tensor<bool>` を `f32` テンソルへ変換する。
/// ONNX 仕様どおり `true -> 1.0`／`false -> 0.0`。
pub fn cast_bool_to_float(x: &Tensor<bool>) -> Result<Tensor<f32>, OpError> {
    let (xc, data) = contiguous_slice("Cast", x)?;
    let out: Vec<f32> = data.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

/// `Cast(x, to=FLOAT16)`: `f32` テンソルを `Tensor<half::f16>` へ変換する。
pub fn cast_to_f16(x: &Tensor<f32>) -> Result<Tensor<f16>, OpError> {
    let (xc, data) = contiguous_slice("Cast", x)?;
    let out: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

/// `Cast(x, to=FLOAT)`: `Tensor<half::f16>` を `f32` テンソルへ変換する。
pub fn cast_f16_to_float(x: &Tensor<f16>) -> Result<Tensor<f32>, OpError> {
    let (xc, data) = contiguous_slice("Cast", x)?;
    let out: Vec<f32> = data.iter().map(|&v| v.to_f32()).collect();
    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

/// `to`（ONNX `TensorProto.DataType` の整数値）が本モジュールの対応範囲
/// （`FLOAT(1)`／`INT64(7)`／`BOOL(9)`／`FLOAT16(10)`。イシュー #274 で BOOL／FLOAT16 を
/// 追加）内かを検査する。呼び出し元（`onnx::interp::compute_cast`）が
/// `cast_*` のどれへ分岐すべきかを判定する前段に使う。範囲外の値は
/// [`OpError::UnsupportedDataType`] を返す（未対応 dtype を握り潰さない）。
pub fn check_supported_cast_target(to: i64) -> Result<(), OpError> {
    const ONNX_DATA_TYPE_FLOAT: i64 = 1;
    const ONNX_DATA_TYPE_BOOL: i64 = 9;
    const ONNX_DATA_TYPE_INT64: i64 = 7;
    const ONNX_DATA_TYPE_FLOAT16: i64 = 10;
    match to {
        ONNX_DATA_TYPE_FLOAT
        | ONNX_DATA_TYPE_INT64
        | ONNX_DATA_TYPE_BOOL
        | ONNX_DATA_TYPE_FLOAT16 => Ok(()),
        _ => Err(OpError::UnsupportedDataType { op: "Cast", to }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_i64_truncates_toward_zero() {
        let x = Tensor::<f32>::new(vec![1.7, -1.7, 2.5, -2.5], &[4]).unwrap();
        let y = cast_to_int64(&x).unwrap();
        assert_eq!(y.as_slice().unwrap(), &[1, -1, 2, -2]);
    }

    #[test]
    fn i64_to_f32_converts() {
        let x = Tensor::<i64>::new(vec![1, -2, 3], &[3]).unwrap();
        let out = cast_to_float(&x).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[1.0, -2.0, 3.0]);
    }

    #[test]
    fn f32_to_bool_nonzero_is_true() {
        let x = Tensor::<f32>::new(vec![0.0, 1.0, -1.0, f32::NAN], &[4]).unwrap();
        let y = cast_to_bool(&x).unwrap();
        let s = y.as_slice().unwrap();
        assert_eq!(s, &[false, true, true, true]);
    }

    #[test]
    fn bool_to_f32_converts() {
        let x = Tensor::<bool>::new(vec![true, false], &[2]).unwrap();
        let y = cast_bool_to_float(&x).unwrap();
        assert_eq!(y.as_slice().unwrap(), &[1.0, 0.0]);
    }

    #[test]
    fn f32_to_f16_and_back_round_trips_representable_values() {
        let x = Tensor::<f32>::new(vec![1.0, -2.5, 0.0], &[3]).unwrap();
        let h = cast_to_f16(&x).unwrap();
        let back = cast_f16_to_float(&h).unwrap();
        assert_eq!(back.as_slice().unwrap(), &[1.0, -2.5, 0.0]);
    }

    #[test]
    fn supported_targets_accepted() {
        assert!(check_supported_cast_target(1).is_ok());
        assert!(check_supported_cast_target(7).is_ok());
        assert!(check_supported_cast_target(9).is_ok());
        assert!(check_supported_cast_target(10).is_ok());
    }

    #[test]
    fn unsupported_target_rejected() {
        let err = check_supported_cast_target(11 /* DOUBLE, 未対応 */).unwrap_err();
        assert!(matches!(
            err,
            OpError::UnsupportedDataType { op: "Cast", to: 11 }
        ));
    }
}
