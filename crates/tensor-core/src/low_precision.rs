//! 低精度 Linear forward（`half::f16`／`half::bf16`。イシュー #1960・親
//! #1626／#1648）。
//!
//! `crate::typed_ops::TypedOps<T>`（capability accessor
//! `BackendOps::typed_ops_f16`／`typed_ops_bf16`。イシュー #1687）は
//! `gemm`／`add`／`relu` 等の低レベル演算のみを提供し、`autodiff` から
//! 直接利用する箇所はこれまで存在しなかった（`docs/backend-dtype-
//! dispatch-design.md` §8「`Var`／`Tape` の dtype 一般化はスコープ外」）。
//! 本モジュールは `fandhe_ai_autodiff::nn::linear` から呼ばれる 1 本の
//! 関数（[`linear_forward_low_precision`]）として、Linear 層限定の
//! **opt-in** 低精度 forward（f32 master weight・f32 backward・forward
//! のみ低精度）を提供する。`Var`／`Tape` 自体の dtype は f32 のまま
//! 不変（`autodiff` 側は forward 値を f32 昇格済みで `Tensor<f32>`
//! として受け取り、通常の `Op::LinearAct`〈`compute_dtype` フィールド〉
//! ノードへそのまま記録する）。
//!
//! # 数値方式
//!
//! `input`／`weight`／`bias`（いずれも f32）を要素ごとに低精度へ降格
//! （`half::f16::from_f32`／`half::bf16::from_f32`。IEEE 754 最近接偶数
//! 丸め）し、`TypedOps<T>::gemm` → （`bias` があれば）`TypedOps<T>::add`
//! （NumPy 互換 broadcast。3 バックエンドとも f32 版と同一規則）→
//! （`Activation::Relu` なら）`TypedOps<T>::relu` の順に低精度カーネルへ
//! 委譲し、結果を 1 回だけ f32 へ昇格して返す。低精度の表現範囲を
//! 超える中間値は ±inf／非正規化数へ丸められる（IEEE 754 挙動をそのまま
//! 伝播させる。`crates/backend-cpu/src/typed_f16.rs` と同じ契約）。
//!
//! # fail-closed 方針
//!
//! - `BackendOps::typed_ops_f16`／`typed_ops_bf16` が `None`（低精度
//!   カーネル未実装のバックエンド）の場合は f32 へ静かにフォール
//!   バックせず [`BackendError::Unsupported`] を返す（opt-in が精度に
//!   ついて嘘をつかないため。`.claude/rules/security.md` A04）
//! - `dtype` が `ScalarDType::F16`／`Bf16` 以外、`act` が
//!   `Activation::None`／`Relu` 以外（[`crate::backend_ops::Activation`]
//!   は `#[non_exhaustive]`）は [`BackendError::InvalidArgument`] で
//!   拒否する
//!
//! # スコープ外
//!
//! `Var`／`Tape` の dtype 一般化・backward の低精度化・`DeviceParamStore`
//! 常駐経路・Linear 以外の層・facade 公開面の拡張は対象外（`docs/
//! autodiff-low-precision-linear-design.md` 参照）。

use half::{bf16, f16};

use crate::backend_ops::{Activation, BackendOps};
use crate::broadcast::broadcast_shape;
use crate::device::BackendError;
use crate::element::ScalarDType;
use crate::ops_shape::gemm_out_shape;
use crate::tensor::Tensor;
use crate::typed_ops::TypedOps;

/// `f16`／`bf16` を低精度 Linear forward の対象型として封印する
/// sealed trait。`f32`/`f64` はこの trait を実装しない（`Scalar` 自体は
/// 4 型あるが、本モジュールが扱うのは低精度 2 型のみ）。
mod private {
    pub trait Sealed {}
}

/// 低精度 Linear forward の対象型が持つべき f32 往復変換
/// （`half::f16`／`half::bf16` それぞれの `from_f32`／`to_f32` を
/// 共通シグネチャへ揃えるための内部 trait。`crate::element::Scalar`
/// の封印を再利用せず本モジュール専用に封印する）。
trait LowPrecisionScalar: crate::element::Scalar + private::Sealed {
    /// IEEE 754 最近接偶数丸めで f32 から降格する。
    fn from_f32_round(v: f32) -> Self;
    /// f32 へ昇格する（`half::{f16,bf16}::to_f32` そのもの）。
    fn to_f32_value(self) -> f32;
}

impl private::Sealed for f16 {}
impl LowPrecisionScalar for f16 {
    fn from_f32_round(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn to_f32_value(self) -> f32 {
        self.to_f32()
    }
}

impl private::Sealed for bf16 {}
impl LowPrecisionScalar for bf16 {
    fn from_f32_round(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn to_f32_value(self) -> f32 {
        self.to_f32()
    }
}

/// `Tensor<f32>` を低精度 `Tensor<T>` へ降格する（要素ごとの丸め。
/// `Tensor::host_slice` は contiguous なら借用・非 contiguous な view は
/// ここで 1 回だけ実体化する。shape 自体は不変のため `Tensor::new` の
/// 失敗は契約上到達しないはずだが、`Tensor` 実装の不変条件違反に対する
/// fail-safe として型付きエラーで受ける
/// `crates/backend-cpu/src/typed_f16.rs::upcast_f16` と同型の設計）。
fn downcast<T: LowPrecisionScalar>(t: &Tensor<f32>) -> Result<Tensor<T>, BackendError> {
    let rounded: Vec<T> = t
        .host_slice()
        .iter()
        .map(|&v| T::from_f32_round(v))
        .collect();
    Tensor::new(rounded, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "low_precision::downcast: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// 低精度 `Tensor<T>` を `Tensor<f32>` へ昇格する（[`downcast`] の対）。
fn upcast<T: LowPrecisionScalar>(t: &Tensor<T>) -> Result<Tensor<f32>, BackendError> {
    let promoted: Vec<f32> = t.host_slice().iter().map(|&v| v.to_f32_value()).collect();
    Tensor::new(promoted, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "low_precision::upcast: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// `TypedOps<T>` を型消去せずに直接呼ぶ本体（`T` が確定した後の
/// ジェネリック実装。呼び出し元の [`linear_forward_low_precision`] が
/// `ScalarDType` から `T` を選び本関数へディスパッチする）。
fn linear_forward_typed<T: LowPrecisionScalar>(
    ops: &dyn TypedOps<T>,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Result<Tensor<f32>, BackendError> {
    let x = downcast::<T>(input)?;
    let w = downcast::<T>(weight)?;
    let mut y = TypedOps::gemm(ops, &x, &w)?;
    if let Some(b) = bias {
        let b_t = downcast::<T>(b)?;
        y = TypedOps::add(ops, &y, &b_t)?;
    }
    // `Activation` は `#[non_exhaustive]`（外部クレート向けの非破壊拡張
    // マーカー）だが、本クレート（`tensor-core`）自身が定義元のため
    // ここでは網羅 match が成立する（`non_exhaustive` はクレート境界
    // 越しの match にのみ `_` 分岐を強制する）。`grad.rs::vjp` の
    // `Op::LinearAct` 分岐（`autodiff` クレート側・外部）はワイルド
    // カードで同じ contract を fail-closed に守る。
    let y = match act {
        Activation::None => y,
        Activation::Relu => TypedOps::relu(ops, &y)?,
    };
    upcast::<T>(&y)
}

/// Linear 層限定の opt-in 低精度 forward（イシュー #1960）。
///
/// `input`／`weight`／`bias`（すべて f32 のホスト常駐テンソル）を
/// `dtype`（[`ScalarDType::F16`]／[`ScalarDType::Bf16`] のみ）で
/// 指定した低精度へ丸めて `gemm`（＋`add`＋`act`）を計算し、結果を f32
/// へ昇格して返す。`fandhe_ai_autodiff::nn::linear::
/// linear_forward_low_precision`（自由関数。`LinearVars` へのフィールド
/// 追加を避けるための配置。設計 doc 参照）の唯一の呼び出し元。
///
/// shape 検査は `gemm_out_shape`／`broadcast_shape`（既存 f32 経路と
/// 同一の判定関数）で計算前に行う。`ops.typed_ops_f16()`／
/// `typed_ops_bf16()` が `None`（バックエンド未実装）の場合は
/// [`BackendError::Unsupported`] を返す（f32 へのフォールバックはしない。
/// モジュール doc「fail-closed 方針」参照）。
pub fn linear_forward_low_precision(
    ops: &dyn BackendOps,
    dtype: ScalarDType,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Result<Tensor<f32>, BackendError> {
    let out_shape =
        gemm_out_shape(input.shape(), weight.shape()).map_err(BackendError::ShapeMismatch)?;
    if let Some(b) = bias {
        broadcast_shape(&out_shape, b.shape()).map_err(BackendError::ShapeMismatch)?;
    }
    match dtype {
        ScalarDType::F16 => {
            let typed = ops.typed_ops_f16().ok_or_else(|| {
                BackendError::Unsupported(
                    "linear_forward_low_precision: typed_ops_f16 unavailable on this backend"
                        .into(),
                )
            })?;
            linear_forward_typed(typed, input, weight, bias, act)
        }
        ScalarDType::Bf16 => {
            let typed = ops.typed_ops_bf16().ok_or_else(|| {
                BackendError::Unsupported(
                    "linear_forward_low_precision: typed_ops_bf16 unavailable on this backend"
                        .into(),
                )
            })?;
            linear_forward_typed(typed, input, weight, bias, act)
        }
        other => Err(BackendError::InvalidArgument(format!(
            "linear_forward_low_precision: unsupported dtype ({other:?}); only F16/Bf16 are \
             accepted"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    /// `typed_ops_f16`／`typed_ops_bf16` とも `None` を返すだけの
    /// モック（accessor 不在の fail-closed 経路を検証するためのもの。
    /// `BackendOps` の他メソッドは到達しない想定のため `unimplemented!`
    /// で構わない）。
    struct NoTypedOpsBackend;

    impl BackendOps for NoTypedOpsBackend {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: gemm should not be reached")
        }
        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: add should not be reached")
        }
        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: mul should not be reached")
        }
        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: relu should not be reached")
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: exp should not be reached")
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: tanh should not be reached")
        }
        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: sum should not be reached")
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: max should not be reached")
        }
    }

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn f16_accessor_none_returns_unsupported() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F16, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn bf16_accessor_none_returns_unsupported() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::Bf16, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn f32_dtype_is_rejected_as_invalid_argument() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F32, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn f64_dtype_is_rejected_as_invalid_argument() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F64, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn shape_mismatch_is_rejected_before_accessor_lookup() {
        // 出力 shape がそもそも成立しない（k 不一致）ケースは
        // `typed_ops_f16()` を呼ぶ前に `ShapeMismatch` で拒否される
        // （accessor が `None` の `NoTypedOpsBackend` でも到達できる
        // ことで、shape 検査が accessor 取得より先に走る契約を確認）。
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0, 3.0], &[1, 3]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F16, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }
}
