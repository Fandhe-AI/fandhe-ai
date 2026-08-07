//! ONNX 8 オペ（`Gemm`／`Relu`／`Sigmoid`／`Shape`／`Gather`／`Unsqueeze`／`Concat`／`Slice`。
//! TASK-7.2c・#79）に加え、MVP 算術オペ（`Add`／`Mul`／`Div`／`Mod`／`Sqrt`／`Constant`。
//! TASK-7.3a・#82）・MVP 形状操作オペ（`Cast`／`Reshape`／`Squeeze`／`Transpose`。
//! TASK-7.3b・#83）・Attention 系オペ（`MatMul`／`Softmax`／`Erf`。TASK-7.3c・#84）・
//! `LayerNormalization`（TASK-7.3d・#85）を提供する。算術・活性化・正規化系
//! （`arith`／`activation`／`gemm`／`matmul`／`softmax`／`layer_norm`）は
//! `tensor-core::Tensor<f32>` 専用の純粋関数のまま、形状系（`shape_ops`／
//! `shape_transform`／`gather`／`concat`／`slice`）は要素コピーのみで算術を伴わないため
//! `T: Element` でジェネリック化し、`Cast`（`cast`）は dtype ごとに型安全な変換関数を
//! 個別に提供する（イシュー #274）。
//!
//! 各関数は「入力テンソル＋属性 → 出力テンソル」の単体演算に限定し、ONNX proto デコード
//! （TASK-7.2a）やグラフ実行順序の解決には関与しない。属性は proto 由来の型に依存しない
//! プレーンな Rust 構造体・スライスで受け取るため、decode 層（`AttributeProto` 等）の
//! 実装順序に依存せず本モジュール単体でテスト・使用できる。インタープリタのディスパッチ
//! （op 名 → 本モジュール関数の解決）は [`crate::onnx::interp`]（TASK-7.2b・#78、
//! TASK-7.3 系 14 オペの結線は #274 で実装）が担い、全 22 オペがグラフ実行から
//! 到達可能である。

mod activation;
mod arith;
mod cast;
mod concat;
mod constant;
mod error;
mod gather;
mod gemm;
mod layer_norm;
mod matmul;
mod shape_ops;
mod shape_transform;
mod slice;
mod softmax;

pub use activation::{erf, relu, sigmoid};
pub use arith::{add, div, modulo, mul, sqrt};
pub use cast::{
    cast_bool_to_float, cast_f16_to_float, cast_to_bool, cast_to_f16, cast_to_float, cast_to_int64,
    check_supported_cast_target,
};
pub use concat::concat;
pub use constant::{ConstantValue, constant};
pub use error::OpError;
pub use gather::gather;
pub use gemm::{GemmAttrs, gemm};
pub use layer_norm::{LayerNormAttrs, layer_normalization};
pub use matmul::matmul;
pub use shape_ops::{shape, unsqueeze};
pub use shape_transform::{reshape, squeeze, transpose};
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
