//! ONNX `Constant` オペ（TASK-7.3a・#82）。
//!
//! `Constant` は入力テンソルを取らず、ノード属性からリテラルテンソルを生成する
//! （ONNX 仕様上は `value`／`value_float`／`value_floats`／`value_int` 等の
//! 排他的属性群を持つ）。他の 5 オペ（`add`/`mul`/`div`/`modulo`/`sqrt`。
//! `arith.rs`）とは異なり要素ごと写像ではないため独立モジュールとする。
//! 属性値は decode 層（ONNX proto の `AttributeProto`。TASK-7.2a）由来の型に
//! 依存しないプレーンな Rust enum として受け取り、decode 層の実装順序に
//! 依存しない（`ops/mod.rs` の設計方針を踏襲）。

use fandhe_ai_tensor_core::Tensor;

use super::error::OpError;

/// `Constant` が受け付ける属性値。本クレートが扱う対象型（`Tensor<f32>`）に
/// 対応する属性のみを対象とし、`value_int`/`value_string` 等の非 f32 系属性は
/// 対象外とする（decode 層で ONNX の他の属性型を扱う場合は別途変換する）。
#[derive(Debug, Clone)]
pub enum ConstantValue {
    /// `value` 属性: 任意 shape の `TensorProto` 相当（フラットデータ＋shape）。
    Tensor { data: Vec<f32>, shape: Vec<usize> },
    /// `value_float` 属性: rank 0 スカラー。
    Float(f32),
    /// `value_floats` 属性: 1 次元配列。
    Floats(Vec<f32>),
}

/// `Constant(value) -> y`。属性からテンソルを構築する。`Tensor` variant は
/// `data.len()` が `shape` の要素数積と一致しない場合 `tensor-core` 側の
/// `ShapeError::ElementCountMismatch` を透過する。
pub fn constant(value: &ConstantValue) -> Result<Tensor<f32>, OpError> {
    match value {
        ConstantValue::Tensor { data, shape } => {
            Tensor::new(data.clone(), shape).map_err(OpError::from)
        }
        ConstantValue::Float(v) => Tensor::new(vec![*v], &[]).map_err(OpError::from),
        ConstantValue::Floats(vs) => {
            let len = vs.len();
            Tensor::new(vs.clone(), &[len]).map_err(OpError::from)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_tensor_variant() {
        let v = ConstantValue::Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
        };
        let y = constant(&v).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn constant_float_variant_is_rank0() {
        let v = ConstantValue::Float(3.5);
        let y = constant(&v).unwrap();
        assert_eq!(y.shape(), &[] as &[usize]);
        assert_eq!(y.get(&[]).unwrap(), 3.5);
    }

    #[test]
    fn constant_floats_variant_is_rank1() {
        let v = ConstantValue::Floats(vec![1.0, 2.0, 3.0]);
        let y = constant(&v).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.get(&[2]).unwrap(), 3.0);
    }

    #[test]
    fn constant_tensor_shape_mismatch_rejected() {
        let v = ConstantValue::Tensor {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![2, 2],
        };
        let err = constant(&v).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }
}
