//! ONNX 8 オペ（`Gemm`／`Relu`／`Sigmoid`／`Shape`／`Gather`／`Unsqueeze`／`Concat`／`Slice`。
//! TASK-7.2c・#79）に加え、MVP 算術オペ（`Add`／`Mul`／`Div`／`Mod`／`Sqrt`／`Constant`。
//! TASK-7.3a・#82）、Attention 系オペ（`MatMul`／`Softmax`／`Erf`。TASK-7.3c・#84）を
//! `tensor-core::Tensor<f32>` 上の純粋関数として提供する。
//!
//! 各関数は「入力テンソル＋属性 → 出力テンソル」の単体演算に限定し、ONNX proto デコード
//! （TASK-7.2a）やグラフ実行順序の解決には関与しない。属性は proto 由来の型に依存しない
//! プレーンな Rust 構造体・スライスで受け取るため、decode 層（`AttributeProto` 等）の
//! 実装順序に依存せず本モジュール単体でテスト・使用できる。

mod activation;
mod arith;
mod concat;
mod constant;
mod error;
mod gather;
mod gemm;
mod matmul;
mod shape_ops;
mod slice;
mod softmax;

pub use activation::{erf, relu, sigmoid};
pub use arith::{add, div, modulo, mul, sqrt};
pub use concat::concat;
pub use constant::{ConstantValue, constant};
pub use error::OpError;
pub use gather::gather;
pub use gemm::{GemmAttrs, gemm};
pub use matmul::matmul;
pub use shape_ops::{shape, unsqueeze};
pub use slice::{SliceParams, slice};
pub use softmax::softmax;

/// ONNX の負軸表記（`axis < 0` の場合 `axis + rank`）を正規化し、`[0, rank)` の範囲を
/// 検査する。範囲外の場合は `None`（呼び出し元が `op` 名を添えて `OpError::AxisOutOfRange`
/// を構築する）。全オペ（`Gather`／`Unsqueeze`／`Concat`／`Slice`）が共有する規則
/// （ONNX Operators スキーマの axis 属性の共通仕様）。
pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        None
    } else {
        Some(normalized as usize)
    }
}

#[cfg(test)]
mod normalize_axis_tests {
    use super::normalize_axis;

    #[test]
    fn positive_within_range() {
        assert_eq!(normalize_axis(1, 3), Some(1));
    }

    #[test]
    fn negative_wraps_from_end() {
        assert_eq!(normalize_axis(-1, 3), Some(2));
        assert_eq!(normalize_axis(-3, 3), Some(0));
    }

    #[test]
    fn out_of_range_returns_none() {
        assert_eq!(normalize_axis(3, 3), None);
        assert_eq!(normalize_axis(-4, 3), None);
    }
}
