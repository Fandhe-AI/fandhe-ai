//! ONNX `Cast` オペ（TASK-7.3b・#83）。
//!
//! `tensor-core::Element` は `i64` を実装しない（対象型は `f32`/`f64`/`i32`/
//! `half::f16` のみ。`crates/tensor-core/src/element.rs`）ため、本モジュールは
//! `Tensor<i64>` ではなく `shape_ops::shape` と同様に生の `Vec<i64>` で i64 側を
//! 表現する（`Element` への `i64` 追加は `tensor-core` 側の変更が必要であり、
//! `.claude/rules/delegation-impl.md` の担当分割上 core-builder の対象。本イシューの
//! スコープ外として記録する）。
//!
//! `to`（ONNX `TensorProto.DataType`）の解決・グラフ実行時の dtype 分岐は
//! インタープリタ結線（#78・未実装）の責務であり、本モジュールは decode 層・
//! 実行時型システムに関知しない「f32 テンソル ⇔ i64 列」の単体変換関数のみを提供する
//! （`ops/mod.rs` 冒頭コメントの方針どおり）。対応範囲は `transformer.onnx`
//! （PyTorch エクスポート・opset 13 以降相当）が使用する `FLOAT(1)` ⇔ `INT64(7)` のみ。

use tensor_core::Tensor;

use super::error::OpError;

/// `Cast(x, to=INT64)`: `f32` テンソルを `i64` 列へ変換する。
///
/// ONNX `Cast` 仕様は浮動小数点 → 整数をゼロ方向切り捨て（truncation）と定める。
/// Rust の `as` キャストは IEEE754 → 整数で同じくゼロ方向丸めを行い、かつ範囲外・NaN を
/// 未定義動作ではなく飽和（`i64::MIN`/`i64::MAX`）・0 に写像する（Rust Reference の
/// `numeric cast` 仕様）ため、追加の事前検査なしに安全に使える。
pub fn cast_to_int64(x: &Tensor<f32>) -> Result<Vec<i64>, OpError> {
    let contiguous = x.contiguous();
    let data = contiguous
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Cast"))?;
    Ok(data.iter().map(|&v| v as i64).collect())
}

/// `Cast(x, to=FLOAT)`: `i64` 列を `f32` テンソルへ変換する。
/// `shape` は `x` の要素数と一致する呼び出し元契約（ONNX `Cast` は形状を変えない。
/// `Tensor::new` が要素数不一致を `ShapeError` として検査する）。
pub fn cast_to_float(x: &[i64], shape: &[usize]) -> Result<Tensor<f32>, OpError> {
    let data: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    Tensor::new(data, shape).map_err(OpError::from)
}

/// `to`（ONNX `TensorProto.DataType` の整数値）が本モジュールの対応範囲
/// （`FLOAT(1)`／`INT64(7)`）内かを検査する。呼び出し元（#78 結線後）が
/// `cast_to_int64`/`cast_to_float` のどちらへ分岐すべきかを判定する前段に使う。
/// 範囲外の値は [`OpError::UnsupportedDataType`] を返す（未対応 dtype を握り潰さない）。
pub fn check_supported_cast_target(to: i64) -> Result<(), OpError> {
    const ONNX_DATA_TYPE_FLOAT: i64 = 1;
    const ONNX_DATA_TYPE_INT64: i64 = 7;
    match to {
        ONNX_DATA_TYPE_FLOAT | ONNX_DATA_TYPE_INT64 => Ok(()),
        _ => Err(OpError::UnsupportedDataType { op: "Cast", to }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_i64_truncates_toward_zero() {
        let x = Tensor::<f32>::new(vec![1.7, -1.7, 2.5, -2.5], &[4]).unwrap();
        assert_eq!(cast_to_int64(&x).unwrap(), vec![1, -1, 2, -2]);
    }

    #[test]
    fn i64_to_f32_converts() {
        let out = cast_to_float(&[1, -2, 3], &[3]).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[1.0, -2.0, 3.0]);
    }

    #[test]
    fn i64_to_f32_shape_mismatch_rejected() {
        let err = cast_to_float(&[1, 2, 3], &[2]).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }

    #[test]
    fn supported_targets_accepted() {
        assert!(check_supported_cast_target(1).is_ok());
        assert!(check_supported_cast_target(7).is_ok());
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
