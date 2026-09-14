//! dtype 変換基盤（f32 をハブとした相互変換。イシュー #1750・親 #1613）。
//!
//! `Var`（`autodiff`）は `Tensor<f32>` 専用であり、`Tensor<f64>`／`i32`／
//! `i64`／`bool` は [`crate::Element`] として生成できるものの相互変換
//! API が存在しなかった（`docs/compat-feature-gap.md` §2.12
//! 「`.to(dtype)`: なし」）。本モジュールはこの欠落を埋める型基盤で、
//! 8 方向（f32→{f64,i32,i64,bool}・{f64,i32,i64,bool}→f32）の cast を
//! (a) [`CastElement`]（sealed trait。要素単位の変換規則の単一情報源）、
//! (b) [`CastOps`]（`BackendOps` の capability accessor 経由でのみ到達
//! する dtype 別カーネル dispatch 面）、(c) ホスト参照実装（[`cast_from_f32`]／
//! [`cast_to_f32`]）として提供する。
//!
//! `backend-cpu` のカーネル実装は本モジュールのホスト参照実装へ委譲
//! する（`crate::interpolate` と同じ「乖離を構造的に排除する単一情報源」
//! の設計方針）。`autodiff::Var::cast`／`Tape::var_from`・facade 到達経路は
//! 本イシューで実装済み（`crates/autodiff/src/var.rs`・`tape.rs`・
//! `crates/facade/src/lib.rs`）。CUDA／Metal のカーネル実装は兄弟イシュー
//! #1751 が引き継ぐ。詳細な契約表・設計判断の比較は
//! `docs/tensor-core-cast-design.md` を参照。
//!
//! # 勾配契約（`autodiff` 側で強制。本クレートは非関知）
//!
//! 出力が f32 以外の cast は勾配を打ち切る（tape ノードを記録せず
//! detached な `Tensor<T>` を返す。`Var::argmax`／`Var::unique` と
//! 同型）。f32→f32（恒等）のみ勾配が伝播する。非 f32 → f32 は新しい
//! 葉として登録され、変換元テンソルへ勾配は流れない。
//!
//! # 数値契約（方向別。GPU 実装への注記は `docs/tensor-core-cast-design.md`）
//!
//! | 方向 | 規則（Rust `as` 意味論） |
//! |---|---|
//! | f32→f64 | 完全表現（exact） |
//! | f64→f32 | 最近接偶数丸め・範囲超過は ±inf |
//! | f32→i32／i64 | ゼロ方向切り捨て・範囲外は飽和・NaN→0 |
//! | i32／i64→f32 | 最近接偶数丸め（`\|v\| > 2^24` は非可逆） |
//! | f32→bool | `v != 0.0`（NaN→true・−0.0→false） |
//! | bool→f32 | `true→1.0`・`false→0.0` |
//!
//! バックエンド間 parity: 算術を含まない変換のため bit 完全一致契約
//! （NaN のみ payload がプラットフォーム依存のため「非 NaN は bit
//! 一致・NaN はクラス一致」）。REQ-2 複合判定・tolerance／baseline は
//! 不変。

use crate::Tensor;
use crate::device::BackendError;
use crate::element::Element;
use crate::error::ShapeError;
use crate::tensor::checked_numel_for;

mod private {
    //! [`super::CastElement`] の実装先をこのクレート内の dtype に
    //! 限定する封印（sealed trait パターン。`crate::element::private::
    //! Sealed` と同じ設計意図だが、対象 dtype 集合が異なる〈`Scalar`
    //! は演算対象 4 型・`CastElement` は cast 対象 5 型で `bool`／
    //! `i32`／`i64` を含む〉ため独立の封印を持つ）。

    /// 封印マーカー。このモジュール外からは実装できない。
    pub trait Sealed {}
}

/// cast 対象 dtype のタグ（イシュー #1750）。
///
/// [`crate::element::ScalarDType`]（演算対象 dtype 専用・`TypedOps<T>`／
/// GEMM 経路選択規則用）とは**別 enum**とする。`Scalar`（`ScalarDType`
/// の対象）は `f32`／`f64`／`half::f16`／`half::bf16` の 4 型に封印
/// されており、`CastDType` が持つ `I32`／`I64`／`Bool` を追加すると
/// `Scalar` の意味論（演算対象・GEMM 経路選択規則の対象）を汚染する
/// （`docs/backend-dtype-dispatch-design.md` §4.4 の整理を維持）。
///
/// `#[non_exhaustive]`: 将来 dtype（`half::f16`／`half::bf16` 等）を
/// 追加する際に公開 API を破壊しないため（`ScalarDType`・`BackendError`
/// 等、本クレートの他の公開 enum と同じ方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastDType {
    /// 32bit 浮動小数点（ハブ dtype）。
    F32,
    /// 64bit 浮動小数点。
    F64,
    /// 32bit 符号付き整数。
    I32,
    /// 64bit 符号付き整数。
    I64,
    /// 真偽値。
    Bool,
}

/// cast 対象になりうる dtype の capability 境界（イシュー #1750）。
///
/// [`Element`] は unsealed であるため、後から cast 境界を非破壊追加
/// できない（`Scalar` と同じ制約。`crate::element` モジュール doc
/// 参照）。`CastElement` はこの制約を再発させないよう
/// `private::Sealed` で封印し、実装対象を `f32`／`f64`／`i32`／`i64`／
/// `bool` の 5 型に限定する（`half::f16`／`half::bf16` は sealed かつ
/// `#[non_exhaustive]` の `CastDType` のため後続で非破壊追加可能。
/// 現時点はスコープ外）。
pub trait CastElement: Element + private::Sealed {
    /// この型に対応する [`CastDType`] タグ。
    const CAST_DTYPE: CastDType;

    /// f32 の 1 要素を `Self` へ変換する（要素単位の変換規則の単一
    /// 情報源。モジュール doc の数値契約表を参照）。
    fn from_f32(v: f32) -> Self;

    /// `Self` の 1 要素を f32 へ変換する（[`Self::from_f32`] の逆方向）。
    fn into_f32(self) -> f32;

    /// [`CastOps`]（動的ディスパッチ面）の f32→`Self` 方向の対応
    /// メソッドを呼ぶ橋渡し（型ごとに固定の 1 メソッドを呼ぶだけ。
    /// `f32` 自身の実装は恒等コピーで `ops` を使わない）。
    fn backend_cast_from_f32(
        ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<Self>, BackendError>;

    /// [`CastOps`] の `Self`→f32 方向の対応メソッドを呼ぶ橋渡し
    /// （[`Self::backend_cast_from_f32`] の逆方向）。
    fn backend_cast_to_f32(
        ops: &dyn CastOps,
        x: &Tensor<Self>,
    ) -> Result<Tensor<f32>, BackendError>;
}

/// dtype 別 cast カーネルの capability 面（イシュー #1750）。
///
/// [`crate::BackendOps::cast_ops`] 経由でのみ到達する（`TypedOps<T>`
/// と同じ 2 段構成: `BackendOps` 自体のメソッド数を 1 だけ増やし、
/// dtype ごとの分岐は本 trait 側に閉じ込める。
/// `docs/backend-dtype-dispatch-design.md` §4 案 D と同型）。
///
/// 8 メソッド全てに既定実装（`Unsupported` fail-safe）を持たせる
/// ため、バックエンドは対応する方向のみを部分的にオーバーライド
/// できる（例: Metal は `double` 非対応のため f64 の 2 方向のみ
/// 既定のまま残す。`docs/backend-dtype-dispatch-design.md` §14）。
/// フォールバック規則は [`cast_from_f32`]／[`cast_to_f32`]（ホスト
/// 参照実装）を参照する `autodiff` 側のヘルパーが担う（本クレートは
/// dispatch 面の定義のみ）。
pub trait CastOps {
    /// f32→f64（完全表現）。
    fn cast_f32_to_f64(&self, _x: &Tensor<f32>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_f32_to_f64: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// f32→i32（ゼロ方向切り捨て・範囲外は飽和・NaN→0）。
    fn cast_f32_to_i32(&self, _x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_f32_to_i32: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// f32→i64（同上）。
    fn cast_f32_to_i64(&self, _x: &Tensor<f32>) -> Result<Tensor<i64>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_f32_to_i64: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// f32→bool（`v != 0.0`。NaN→true・−0.0→false）。
    fn cast_f32_to_bool(&self, _x: &Tensor<f32>) -> Result<Tensor<bool>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_f32_to_bool: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// f64→f32（最近接偶数丸め・範囲超過は ±inf）。
    fn cast_f64_to_f32(&self, _x: &Tensor<f64>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_f64_to_f32: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// i32→f32（最近接偶数丸め。`|v| > 2^24` は非可逆）。
    fn cast_i32_to_f32(&self, _x: &Tensor<i32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_i32_to_f32: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// i64→f32（同上）。
    fn cast_i64_to_f32(&self, _x: &Tensor<i64>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_i64_to_f32: default fail-safe (no cast kernel available)".into(),
        ))
    }
    /// bool→f32（`true→1.0`・`false→0.0`）。
    fn cast_bool_to_f32(&self, _x: &Tensor<bool>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "cast_bool_to_f32: default fail-safe (no cast kernel available)".into(),
        ))
    }
}

impl private::Sealed for f32 {}
impl CastElement for f32 {
    const CAST_DTYPE: CastDType = CastDType::F32;

    fn from_f32(v: f32) -> Self {
        v
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn backend_cast_from_f32(
        _ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        // f32→f32 は恒等（`cast::<f32>()` の呼び出し規約）。カーネル
        // dispatch を経由せず単純にクローンする。
        Ok(x.clone())
    }

    fn backend_cast_to_f32(
        _ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Ok(x.clone())
    }
}

impl private::Sealed for f64 {}
impl CastElement for f64 {
    const CAST_DTYPE: CastDType = CastDType::F64;

    fn from_f32(v: f32) -> Self {
        v as f64
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn backend_cast_from_f32(
        ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<f64>, BackendError> {
        ops.cast_f32_to_f64(x)
    }

    fn backend_cast_to_f32(
        ops: &dyn CastOps,
        x: &Tensor<f64>,
    ) -> Result<Tensor<f32>, BackendError> {
        ops.cast_f64_to_f32(x)
    }
}

impl private::Sealed for i32 {}
impl CastElement for i32 {
    const CAST_DTYPE: CastDType = CastDType::I32;

    fn from_f32(v: f32) -> Self {
        // Rust 1.45+ の `as` は float→int を saturating（範囲外は
        // MIN/MAX 側へ飽和・NaN→0）で定義する。モジュール doc の
        // 数値契約表どおり。
        v as i32
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn backend_cast_from_f32(
        ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<i32>, BackendError> {
        ops.cast_f32_to_i32(x)
    }

    fn backend_cast_to_f32(
        ops: &dyn CastOps,
        x: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        ops.cast_i32_to_f32(x)
    }
}

impl private::Sealed for i64 {}
impl CastElement for i64 {
    const CAST_DTYPE: CastDType = CastDType::I64;

    fn from_f32(v: f32) -> Self {
        v as i64
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn backend_cast_from_f32(
        ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<i64>, BackendError> {
        ops.cast_f32_to_i64(x)
    }

    fn backend_cast_to_f32(
        ops: &dyn CastOps,
        x: &Tensor<i64>,
    ) -> Result<Tensor<f32>, BackendError> {
        ops.cast_i64_to_f32(x)
    }
}

impl private::Sealed for bool {}
impl CastElement for bool {
    const CAST_DTYPE: CastDType = CastDType::Bool;

    fn from_f32(v: f32) -> Self {
        // `f32 as bool` はコンパイル不能なため比較で明示する
        // （モジュール doc: NaN→true・−0.0→false）。
        v != 0.0
    }

    fn into_f32(self) -> f32 {
        // `bool as f32` もコンパイル不能なため明示分岐する。
        if self { 1.0 } else { 0.0 }
    }

    fn backend_cast_from_f32(
        ops: &dyn CastOps,
        x: &Tensor<f32>,
    ) -> Result<Tensor<bool>, BackendError> {
        ops.cast_f32_to_bool(x)
    }

    fn backend_cast_to_f32(
        ops: &dyn CastOps,
        x: &Tensor<bool>,
    ) -> Result<Tensor<f32>, BackendError> {
        ops.cast_bool_to_f32(x)
    }
}

/// [`Tensor<f32>`] → `Tensor<T>` のホスト参照実装（イシュー #1750）。
///
/// `backend-cpu::CastOps` 実装・`autodiff` のフォールバック経路が
/// 共有する単一情報源（`crate::interpolate` と同じ設計方針。本モジュール
/// doc 参照）。要素数積を `T` のバイトサイズで事前検査してから
/// [`Tensor::host_slice`]（非 contiguous view も稠密化）で読み出し、
/// 要素ごとに [`CastElement::from_f32`] を適用する。出力は常に
/// contiguous（strides は引き継がない）。
///
/// `T` が f32 より大きい型（`f64`／`i64`）の場合、入力 `x` 自体は
/// 妥当な shape（f32 として構築済み）でも `T` へキャストした際の
/// バイト量が `Vec` のアロケーション上限を超えうるため、`T` 基準で
/// 改めて要素数積を検査する（`unique.rs` の transpose 済み view 対策
/// と同種の防御。PR #1828 の教訓）。
pub fn cast_from_f32<T: CastElement>(x: &Tensor<f32>) -> Result<Tensor<T>, ShapeError> {
    checked_numel_for::<T>(x.shape())?;
    let slice = x.host_slice();
    let data: Vec<T> = slice.iter().map(|&v| T::from_f32(v)).collect();
    Tensor::new(data, x.shape())
}

/// `Tensor<T>` → [`Tensor<f32>`] のホスト参照実装（[`cast_from_f32`]
/// の逆方向。イシュー #1750）。
///
/// `T = bool` の場合も `Vec<bool>` → `Vec<f32>` へ通常の `Vec`
/// アロケーションで変換するため、生バイトの transmute／再解釈は
/// 行わない（Rust の `bool` 0／1 妥当性不変条件を破らない。
/// `docs/tensor-core-cast-design.md` の GPU 実装注記〈`u8` 経由
/// 実体化〉と対になる契約）。
pub fn cast_to_f32<T: CastElement>(x: &Tensor<T>) -> Result<Tensor<f32>, ShapeError> {
    checked_numel_for::<f32>(x.shape())?;
    let slice = x.host_slice();
    let data: Vec<f32> = slice.iter().map(|&v| v.into_f32()).collect();
    Tensor::new(data, x.shape())
}

impl Tensor<f32> {
    /// `self` を `Tensor<T>` へ変換する（[`cast_from_f32`] への薄い
    /// 委譲。イシュー #1750）。
    ///
    /// `T = f32` も許容するが、その場合は detached なコピーを返す
    /// だけであり（`autodiff::Var::cast::<f32>()` も同様に非微分・
    /// detached）、勾配を保ったまま f32 系の恒等射を得たい場合は
    /// `autodiff::Var::to_f32` を使うこと（本メソッドは値レベルの
    /// 変換のみを提供し、勾配伝播の意味論は `autodiff` 側の責務）。
    pub fn cast<T: CastElement>(&self) -> Result<Tensor<T>, ShapeError> {
        cast_from_f32(self)
    }
}

impl<T: CastElement> Tensor<T> {
    /// `self` を [`Tensor<f32>`] へ変換する（[`cast_to_f32`] への薄い
    /// 委譲。イシュー #1750）。`T = f32` の場合は恒等コピー。
    pub fn to_f32(&self) -> Result<Tensor<f32>, ShapeError> {
        cast_to_f32(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t<T: Element>(data: Vec<T>, shape: &[usize]) -> Tensor<T> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    // --- CastDType タグ ---

    #[test]
    fn cast_dtype_const_matches_for_each_type() {
        assert_eq!(<f32 as CastElement>::CAST_DTYPE, CastDType::F32);
        assert_eq!(<f64 as CastElement>::CAST_DTYPE, CastDType::F64);
        assert_eq!(<i32 as CastElement>::CAST_DTYPE, CastDType::I32);
        assert_eq!(<i64 as CastElement>::CAST_DTYPE, CastDType::I64);
        assert_eq!(<bool as CastElement>::CAST_DTYPE, CastDType::Bool);
    }

    // --- f32 <-> f64（完全表現・最近接偶数丸め） ---

    #[test]
    fn f32_to_f64_is_exact_round_trip() {
        let x = t(
            vec![1.5f32, -2.25, 0.0, -0.0, f32::MAX, f32::MIN_POSITIVE],
            &[6],
        );
        let y: Tensor<f64> = cast_from_f32(&x).unwrap();
        let back: Tensor<f32> = cast_to_f32(&y).unwrap();
        assert_eq!(back.host_slice().into_owned(), x.host_slice().into_owned());
        // f32→f64 は完全表現のため、個別要素も exact であるはず。
        assert_eq!(y.host_slice()[0], 1.5f64);
        assert_eq!(y.host_slice()[4], f32::MAX as f64);
    }

    #[test]
    fn f64_overflow_saturates_to_infinity_on_narrowing() {
        let x = t(vec![f64::MAX, f64::MIN], &[2]);
        let y: Tensor<f32> = cast_to_f32(&x).unwrap();
        let data = y.host_slice();
        assert_eq!(data[0], f32::INFINITY);
        assert_eq!(data[1], f32::NEG_INFINITY);
    }

    // --- f32 <-> i32／i64（飽和・ゼロ方向切り捨て・NaN→0） ---

    #[test]
    fn f32_to_i32_saturates_and_truncates_toward_zero() {
        let x = t(
            vec![
                1.9f32,
                -1.9,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1e20,
                -1e20,
            ],
            &[7],
        );
        let y: Tensor<i32> = cast_from_f32(&x).unwrap();
        let data = y.host_slice();
        assert_eq!(data[0], 1); // ゼロ方向切り捨て
        assert_eq!(data[1], -1);
        assert_eq!(data[2], 0); // NaN → 0
        assert_eq!(data[3], i32::MAX); // +inf は飽和
        assert_eq!(data[4], i32::MIN); // -inf は飽和
        assert_eq!(data[5], i32::MAX); // 範囲外は飽和
        assert_eq!(data[6], i32::MIN);
    }

    #[test]
    fn i32_min_max_round_trip_saturates_on_the_way_back() {
        let x = t(vec![i32::MIN, i32::MAX, 0i32], &[3]);
        let y: Tensor<f32> = cast_to_f32(&x).unwrap();
        // i32::MIN/MAX は f32 の仮数部（24bit）を超えるため丸められる
        // （`|v| > 2^24` は非可逆。モジュール doc 参照）。往復は
        // 「有限の近似値になる」ことのみ確認する。
        let data = y.host_slice();
        assert!(data[0].is_finite());
        assert!(data[1].is_finite());
        assert_eq!(data[2], 0.0);
    }

    #[test]
    fn f32_to_i64_saturates_and_truncates_toward_zero() {
        let x = t(vec![1.9f32, -1.9, f32::NAN, 1e30, -1e30], &[5]);
        let y: Tensor<i64> = cast_from_f32(&x).unwrap();
        let data = y.host_slice();
        assert_eq!(data[0], 1);
        assert_eq!(data[1], -1);
        assert_eq!(data[2], 0);
        assert_eq!(data[3], i64::MAX);
        assert_eq!(data[4], i64::MIN);
    }

    #[test]
    fn i64_extreme_values_round_trip_to_finite_f32() {
        let x = t(vec![i64::MIN, i64::MAX], &[2]);
        let y: Tensor<f32> = cast_to_f32(&x).unwrap();
        let data = y.host_slice();
        assert!(data[0].is_finite());
        assert!(data[1].is_finite());
    }

    #[test]
    fn i64_value_exceeding_2_pow_24_is_not_bit_exact_round_trip() {
        // `2^24 + 1` は f32 の仮数部で表現できない代表値
        // （モジュール doc「`|v| > 2^24` は非可逆」の直接確認）。
        let v: i64 = (1i64 << 24) + 1;
        let x = t(vec![v], &[1]);
        let y: Tensor<f32> = cast_to_f32(&x).unwrap();
        let back: Tensor<i64> = cast_from_f32(&y).unwrap();
        assert_ne!(back.host_slice()[0], v);
    }

    // --- f32 <-> bool ---

    #[test]
    fn f32_to_bool_nonzero_is_true_zero_is_false_nan_is_true() {
        let x = t(vec![0.0f32, -0.0, 1.0, -1.0, f32::NAN], &[5]);
        let y: Tensor<bool> = cast_from_f32(&x).unwrap();
        let data = y.host_slice();
        assert!(!data[0]); // 0.0 -> false
        assert!(!data[1]); // -0.0 -> false
        assert!(data[2]); // 1.0 -> true
        assert!(data[3]); // -1.0 -> true
        assert!(data[4]); // NaN -> true
    }

    #[test]
    fn bool_to_f32_true_is_one_false_is_zero() {
        let x = t(vec![true, false, true], &[3]);
        let y: Tensor<f32> = cast_to_f32(&x).unwrap();
        assert_eq!(y.host_slice().into_owned(), vec![1.0f32, 0.0, 1.0]);
    }

    // --- 恒等 f32→f32 ---

    #[test]
    fn f32_to_f32_cast_is_identity_value() {
        let x = t(vec![1.0f32, 2.5, -3.0], &[3]);
        let y: Tensor<f32> = cast_from_f32(&x).unwrap();
        assert_eq!(y.host_slice().into_owned(), x.host_slice().into_owned());
    }

    // --- 形状保存・非 contiguous・空テンソル ---

    #[test]
    fn cast_preserves_shape() {
        let x = t(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y: Tensor<i32> = cast_from_f32(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn cast_handles_non_contiguous_view() {
        let base = t(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let transposed = base
            .permute(&[1, 0])
            .expect("test fixture: rank 2 の permute は常に妥当");
        let y: Tensor<i32> = cast_from_f32(&transposed).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.host_slice().into_owned(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn cast_handles_empty_tensor() {
        let x = t(Vec::<f32>::new(), &[0]);
        let y: Tensor<i32> = cast_from_f32(&x).unwrap();
        assert_eq!(y.shape(), &[0]);
        assert!(y.host_slice().is_empty());
    }

    // --- Tensor 利便メソッド（`cast`／`to_f32`） ---

    #[test]
    fn tensor_cast_method_matches_free_function() {
        let x = t(vec![1.5f32, -2.5], &[2]);
        let via_method: Tensor<i32> = x.cast().unwrap();
        let via_fn: Tensor<i32> = cast_from_f32(&x).unwrap();
        assert_eq!(
            via_method.host_slice().into_owned(),
            via_fn.host_slice().into_owned()
        );
    }

    #[test]
    fn tensor_to_f32_method_matches_free_function() {
        let x = t(vec![i32::MIN, 0, i32::MAX], &[3]);
        let via_method = x.to_f32().unwrap();
        let via_fn = cast_to_f32(&x).unwrap();
        assert_eq!(
            via_method.host_slice().into_owned(),
            via_fn.host_slice().into_owned()
        );
    }

    // --- CastOps 既定実装（Unsupported fail-safe） ---

    struct NoCastOps;
    impl CastOps for NoCastOps {}

    #[test]
    fn cast_ops_default_methods_return_unsupported() {
        let ops = NoCastOps;
        let x = t(vec![1.0f32], &[1]);
        assert!(matches!(
            ops.cast_f32_to_f64(&x),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.cast_f32_to_i32(&x),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.cast_f32_to_i64(&x),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.cast_f32_to_bool(&x),
            Err(BackendError::Unsupported(_))
        ));
        let xf64 = t(vec![1.0f64], &[1]);
        assert!(matches!(
            ops.cast_f64_to_f32(&xf64),
            Err(BackendError::Unsupported(_))
        ));
        let xi32 = t(vec![1i32], &[1]);
        assert!(matches!(
            ops.cast_i32_to_f32(&xi32),
            Err(BackendError::Unsupported(_))
        ));
        let xi64 = t(vec![1i64], &[1]);
        assert!(matches!(
            ops.cast_i64_to_f32(&xi64),
            Err(BackendError::Unsupported(_))
        ));
        let xbool = t(vec![true], &[1]);
        assert!(matches!(
            ops.cast_bool_to_f32(&xbool),
            Err(BackendError::Unsupported(_))
        ));
    }

    /// `&dyn CastOps` として扱えること（object-safety のコンパイル時
    /// 検査。`typed_ops.rs::assert_dyn_compatible_f64` と同型）。
    fn assert_dyn_compatible(_ops: &dyn CastOps) {}

    #[test]
    fn cast_ops_is_object_safe() {
        let ops = NoCastOps;
        assert_dyn_compatible(&ops);
    }

    /// [`CastElement::backend_cast_from_f32`]／[`backend_cast_to_f32`]
    /// が既定実装（`Unsupported`）を正しく橋渡しすることを確認する
    /// （f32 自身は `ops` を使わない恒等経路のため対象外）。
    #[test]
    fn backend_cast_bridges_to_unsupported_default() {
        let ops = NoCastOps;
        let x = t(vec![1.0f32], &[1]);
        assert!(matches!(
            <f64 as CastElement>::backend_cast_from_f32(&ops, &x),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            <bool as CastElement>::backend_cast_from_f32(&ops, &x),
            Err(BackendError::Unsupported(_))
        ));
    }
}
