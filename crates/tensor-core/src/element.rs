//! テンソルの要素型抽象化。
//!
//! `tensor.rs` の生成系 API（`zeros`/`ones`/`full`）がジェネリックな
//! `Tensor<T>` を返すには、`T` 自身が加法単位元・乗法単位元を生成できる
//! 必要がある。`Element` はこの capability を型境界として表現する
//! （spec 根拠: `docs/public-api-design.md` §2.3）。
//!
//! 実装対象は `f32`/`f64`/`i32`/`half::f16`（GPU バックエンド CUDA/Metal
//! が使用する半精度浮動小数点型）に加え、`i64`（ONNX `INT64` テンソル・
//! `onnx-interop` の shape／インデックス系値を型安全に表現するため。イシュー
//! #274）・`bool`（ONNX `BOOL` テンソルのデコード表現用。同じくイシュー #274）・
//! `half::bf16`（イシュー #1687。dtype 別演算 dispatch の対象型の 1 つ）。
//! `i64`/`bool` はいずれも `onnx-interop` の形状系オペ（コピーのみで算術を伴わない）
//! でのみ用いる想定であり、`backend_ops`／`dispatch` の算術カーネル dispatch 対象には
//! 含めない（算術・backend dispatch 対応は本イシューのスコープ外。#274 実装計画 §7）。
//!
//! [`Scalar`]（本ファイル下部）は `Element` の演算対象部分集合
//! （`f32`／`f64`／`half::f16`／`half::bf16`）に絞った sealed trait で、
//! `crate::typed_ops::TypedOps<T>`（イシュー #1687）の型境界として使う。

use half::{bf16, f16};

/// テンソルが扱える要素型の最小抽象化。
///
/// `Copy + Send + Sync + Debug + PartialEq + 'static` に加え、`zero()`/
/// `one()` を追加境界として要求する（`docs/public-api-design.md` §2.3
/// で「具体的なシグネチャは TASK-1.4 productize 時に確定する」とされた
/// 箇所を本イシュー（TASK-1.4a）で確定する）。
pub trait Element: Copy + Send + Sync + std::fmt::Debug + PartialEq + 'static {
    /// 加法単位元（`zeros` が使用する）。
    fn zero() -> Self;
    /// 乗法単位元（`ones` が使用する）。
    fn one() -> Self;
}

impl Element for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl Element for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl Element for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl Element for f16 {
    fn zero() -> Self {
        f16::ZERO
    }
    fn one() -> Self {
        f16::ONE
    }
}

impl Element for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl Element for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
}

impl Element for bf16 {
    fn zero() -> Self {
        bf16::ZERO
    }
    fn one() -> Self {
        bf16::ONE
    }
}

mod private {
    //! `Scalar` の実装先をこのクレート内の dtype に限定する封印
    //! （sealed trait パターン）。`TypedOps<T: Scalar>`（`crate::typed_ops`）の
    //! 型パラメータ `T` はこの `Scalar` に境界付けられるため、外部クレートは
    //! 独自の新規 dtype に対して `TypedOps<MyDtype>` を実装することはできない
    //! （`MyDtype` が封印済みの `Scalar` を実装できないため）。外部クレートに
    //! できるのは、既存 4 dtype（`f32`／`f64`／`half::f16`／`half::bf16`）の
    //! いずれかに対する `TypedOps<f64>` 等を、自前のバックエンド型
    //! （`Self` 側）に実装することのみである（封印されるのは dtype 側の
    //! `Scalar` であり、バックエンド型の `TypedOps` 実装先ではない）。

    /// 封印マーカー。このモジュール外からは実装できない。
    pub trait Sealed {}
}

/// 演算対象になりうる dtype の capability 境界（イシュー #1687）。
///
/// [`Element`] は unsealed であるため、後から演算境界（四則演算・比較・
/// 相互変換等）を非破壊追加できない（`docs/backend-dtype-dispatch-design.md`
/// §4.1 が指摘した制約）。`Scalar` はこの制約を再発させないよう
/// `private::Sealed` で封印し、実装対象を `f32`／`f64`／`half::f16`／
/// `half::bf16` の 4 型に限定する。`DTYPE` 以外の追加境界は、
/// `crate::typed_ops::TypedOps<T>` の複数実装で実際に共通化が必要に
/// なった時点で非破壊追加する（本イシューのスコープ外）。
pub trait Scalar: Element + private::Sealed {
    /// この型に対応する [`ScalarDType`] タグ。
    const DTYPE: ScalarDType;
}

/// 演算対象 dtype のタグ（イシュー #1687）。
///
/// [`crate::dispatch::DType`]（GEMM 経路選択規則専用・`#[non_exhaustive]`
/// なし）とは別の enum。こちらは `TypedOps<T>` の capability accessor
/// （[`crate::BackendOps::typed_ops_f64`] 等）が対応する dtype を実行時に
/// 識別するための汎用タグであり、GEMM 経路選択規則の対象外の dtype
/// （`F64`／`Bf16`）も表現する。
///
/// `#[non_exhaustive]`: 将来 dtype を追加する際に公開 API を破壊しない
/// ため（`crate::device::BackendError` 等、本クレートの他の公開 enum と
/// 同じ方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarDType {
    /// 32bit 浮動小数点。
    F32,
    /// 64bit 浮動小数点。
    F64,
    /// 16bit 浮動小数点（`half::f16`）。
    F16,
    /// bfloat16（`half::bf16`）。
    Bf16,
}

impl ScalarDType {
    /// `ScalarDType` → [`crate::dispatch::DType`] の明示マッピング
    /// （`docs/backend-dtype-dispatch-design.md` §4.4）。
    ///
    /// GEMM 経路選択規則（[`crate::dispatch::select_gemm_kernel`]）の
    /// 対象になりうるのは `F32`／`F16` のみ。`F64`／`Bf16` は規則
    /// エンジン非対象（呼び出し側が直接カーネル経路を選ぶ）ため `None`
    /// を返す。`crate::dispatch::DType` は `#[non_exhaustive]` を持たない
    /// ため（本クレート内では）網羅 match で足り、将来同 enum に
    /// variant が追加された場合はコンパイルエラーとして検知される。
    pub const fn dispatch_dtype(self) -> Option<crate::dispatch::DType> {
        match self {
            ScalarDType::F32 => Some(crate::dispatch::DType::F32),
            ScalarDType::F16 => Some(crate::dispatch::DType::F16),
            ScalarDType::F64 | ScalarDType::Bf16 => None,
        }
    }
}

impl private::Sealed for f32 {}
impl Scalar for f32 {
    const DTYPE: ScalarDType = ScalarDType::F32;
}

impl private::Sealed for f64 {}
impl Scalar for f64 {
    const DTYPE: ScalarDType = ScalarDType::F64;
}

impl private::Sealed for f16 {}
impl Scalar for f16 {
    const DTYPE: ScalarDType = ScalarDType::F16;
}

impl private::Sealed for bf16 {}
impl Scalar for bf16 {
    const DTYPE: ScalarDType = ScalarDType::Bf16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_zero_one_match_half_constants() {
        assert_eq!(<bf16 as Element>::zero(), bf16::ZERO);
        assert_eq!(<bf16 as Element>::one(), bf16::ONE);
    }

    #[test]
    fn scalar_dtype_maps_to_dispatch_dtype_correctly() {
        assert_eq!(
            ScalarDType::F32.dispatch_dtype(),
            Some(crate::dispatch::DType::F32)
        );
        assert_eq!(
            ScalarDType::F16.dispatch_dtype(),
            Some(crate::dispatch::DType::F16)
        );
        assert_eq!(ScalarDType::F64.dispatch_dtype(), None);
        assert_eq!(ScalarDType::Bf16.dispatch_dtype(), None);
    }

    #[test]
    fn scalar_dtype_const_matches_for_each_type() {
        assert_eq!(<f32 as Scalar>::DTYPE, ScalarDType::F32);
        assert_eq!(<f64 as Scalar>::DTYPE, ScalarDType::F64);
        assert_eq!(<f16 as Scalar>::DTYPE, ScalarDType::F16);
        assert_eq!(<bf16 as Scalar>::DTYPE, ScalarDType::Bf16);
    }
}
