//! 決定的テンソル生成ユーティリティ（PyTorch `torch.arange`／
//! `torch.linspace`／`torch.eye`／`torch.zeros_like`／`torch.ones_like`
//! 相当。イシュー #1726。親 #1602）。
//!
//! # 位置づけ
//!
//! `Tensor::zeros`／`ones`／`full`（`tensor.rs`）と同じ「ホスト側だけで
//! 完結する生成系」レイヤーに属し、`rng.rs`（[`crate::rng`]）の
//! `randn`／`rand`／`randint` と並ぶ非乱数版のカウンターパートである。
//! `BackendOps` を経由せず、`Op`／VJP も追加しない（本モジュールが
//! 生成する値はいずれも微分不能な葉値であり、`torch.arange` 等に勾配が
//! 無いのと同じ理由。#1602 本文の設計方針）。生成した [`Tensor`] は
//! 既存の `Tape::var` アップロード経路でデバイスへ反映する。
//!
//! # 受入基準テンプレートの非適用（#1602 ツリー共通の注記）
//!
//! `#1602` 系列の他イシュー（例: 正規化・活性化関数の追加）が前提とする
//! 「`Op` 追加・`BackendOps` メソッド追加・`Var` メソッド追加・VJP 実装・
//! バックエンド間 parity テスト」という受入基準テンプレートは、本モジュール
//! （[`crate::rng`] の `randn`／`rand`／`randint` と同じ理由）には適用され
//! ない。「parity」は「ホスト生成 → CPU `Tape::var` アップロードの bit
//! 完全一致」＋ CUDA／Metal `#[ignore]` round-trip（`crates/facade/tests/
//! creation_tensor_generation.rs`）で満たす——バックエンド別カーネルが
//! 存在しないため REQ-2 複合判定の対象自体が無い（`docs/
//! rng-global-contract-design.md` §11 に集約して記録する）。
//!
//! # 数値契約
//!
//! - [`arange`]：長さを `f64` 中間計算（`ceil((end − start) / step)`）で
//!   求め、各要素は `(start as f64 + i as f64 * step as f64) as f32`
//!   （PyTorch CPU の `accscalar_t = double` と同じ方式）。IEEE 基本演算
//!   ＋ `ceil` のみで構成されるため、プラットフォーム横断で bit 同一
//!   （`rand` と同じくゴールデン値テストが可能）。
//! - [`linspace`]：PyTorch の 2 分割方式（前半は `start + i·step`、
//!   後半は `end − (steps − 1 − i)·step`。`step` は `f64`）を用い、
//!   先頭が `start`・末尾が `end` と bit 一致することを契約する。
//! - [`eye`]：`i / n == i % n` を満たす要素のみ `T::one()`、他は
//!   `T::zero()`。
//! - [`zeros_like`]／[`ones_like`]：`like` の shape のみを引き継ぎ、
//!   strides（転置・broadcast view 由来の stride 0 等）は一切保存しない
//!   新規 contiguous バッファを返す。

use crate::element::Element;
use crate::error::ShapeError;
use crate::tensor::{Tensor, checked_numel_for};

/// [`arange`]／[`linspace`] 専用のエラー型。`shape`（要素数）起因の
/// 不整合は `Tensor::zeros` 等と同じ [`ShapeError`] へ委譲し
/// （[`From<ShapeError>`] 実装）、引数自体の不正（`step == 0`・非有限値）
/// は `ShapeError` の対象外であるため本型で新設する
/// （`rng::RngError` と同型の設計。イシュー #1726）。
///
/// `#[non_exhaustive]`: `RngError`／`ShapeError` と同じ理由（公開 API
/// 非破壊。`.claude/rules/security.md`）で後続の検査項目追加に備える。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum CreationError {
    /// shape 起因の不整合（要素数積のオーバーフロー等）。
    Shape(ShapeError),
    /// `arange` の `step` がゼロまたは非有限（`NaN`／`±inf`）。
    InvalidStep { step: f32 },
    /// `arange`／`linspace` の `start`／`end` が非有限（`NaN`／`±inf`）。
    NonFiniteBound { start: f32, end: f32 },
}

impl From<ShapeError> for CreationError {
    fn from(err: ShapeError) -> Self {
        CreationError::Shape(err)
    }
}

impl std::fmt::Display for CreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreationError::Shape(err) => write!(f, "{err}"),
            CreationError::InvalidStep { step } => {
                write!(f, "arange の step が不正: step={step}")
            }
            CreationError::NonFiniteBound { start, end } => {
                write!(f, "start／end が非有限: start={start}, end={end}")
            }
        }
    }
}

impl std::error::Error for CreationError {}

/// `[start, end)` を `step` 刻みで並べたテンソルを生成する（PyTorch
/// `torch.arange` 相当。イシュー #1726）。
///
/// 長さは `ceil((end − start) / step)` を `f64` で計算する（負数への
/// 切り捨てを避けるため `0` 未満は空テンソルとして扱う——`start` と
/// `end` の大小関係が `step` の符号と矛盾する場合〈例:
/// `arange(0, 5, -1)`〉も同様に空テンソルを返す）。`step == 0` または
/// `start`／`end`／`step` のいずれかが非有限（`NaN`／`±inf`）の場合は
/// [`CreationError`] を返す。長さが `usize` の範囲・アロケーション可能
/// バイトサイズを超える場合はアロケーション前に
/// [`ShapeError::ElementCountOverflow`] を返す（[`CreationError::Shape`]
/// 経由）。
///
/// 数値契約はモジュール冒頭コメント参照（プラットフォーム横断で bit
/// 同一）。
pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor<f32>, CreationError> {
    if !start.is_finite() || !end.is_finite() {
        return Err(CreationError::NonFiniteBound { start, end });
    }
    if step == 0.0 || !step.is_finite() {
        return Err(CreationError::InvalidStep { step });
    }
    let start_f64 = start as f64;
    let end_f64 = end as f64;
    let step_f64 = step as f64;
    let raw_len = ((end_f64 - start_f64) / step_f64).ceil();
    let n: usize = if raw_len <= 0.0 {
        0
    } else if raw_len > usize::MAX as f64 {
        // アロケーション前検査は `checked_numel_for` に委ねるため、ここでは
        // `usize` へキャスト不能な巨大値を `usize::MAX` に丸めて渡す
        // （後続の `checked_numel_for::<f32>` がバイトサイズ超過として
        // 確実に `ElementCountOverflow` を返す）。
        usize::MAX
    } else {
        raw_len as usize
    };
    let numel = checked_numel_for::<f32>(&[n])?;
    let data: Vec<f32> = (0..numel)
        .map(|i| (start_f64 + i as f64 * step_f64) as f32)
        .collect();
    Ok(Tensor::new(data, &[numel])?)
}

/// `[start, end]`（両端を含む）を `steps` 個の等間隔値で埋めたテンソルを
/// 生成する（PyTorch `torch.linspace` 相当。イシュー #1726）。
///
/// `steps == 0` は空テンソル、`steps == 1` は `[start]` を返す。
/// `steps >= 2` は PyTorch の 2 分割方式（前半 `start + i·step`・後半
/// `end − (steps − 1 − i)·step`。`step = (end − start) / (steps − 1)`
/// を `f64` で計算）を用い、先頭が `start`・末尾が `end` と bit 一致
/// することを契約する。`start`／`end` が非有限の場合は
/// [`CreationError`] を返す。
pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Tensor<f32>, CreationError> {
    if !start.is_finite() || !end.is_finite() {
        return Err(CreationError::NonFiniteBound { start, end });
    }
    let numel = checked_numel_for::<f32>(&[steps])?;
    if numel == 0 {
        return Ok(Tensor::new(Vec::new(), &[0])?);
    }
    if numel == 1 {
        return Ok(Tensor::new(vec![start], &[1])?);
    }
    let start_f64 = start as f64;
    let end_f64 = end as f64;
    let step_f64 = (end_f64 - start_f64) / (numel - 1) as f64;
    let half = numel / 2;
    let data: Vec<f32> = (0..numel)
        .map(|i| {
            if i < half {
                (start_f64 + i as f64 * step_f64) as f32
            } else {
                (end_f64 - (numel - 1 - i) as f64 * step_f64) as f32
            }
        })
        .collect();
    Ok(Tensor::new(data, &[numel])?)
}

/// `n x n` の単位行列を生成する（PyTorch `torch.eye` 相当。長方形版
/// 〈`eye(rows, cols)`〉は対象外。イシュー #1726）。
///
/// `n == 0` は shape `[0, 0]` の空テンソルを返す。`n` が大きく
/// `n * n` がアロケーション不能な場合は [`ShapeError::
/// ElementCountOverflow`] を確保前に返す。
pub fn eye<T: Element>(n: usize) -> Result<Tensor<T>, ShapeError> {
    let numel = checked_numel_for::<T>(&[n, n])?;
    let mut data: Vec<T> = vec![T::zero(); numel];
    for i in 0..n {
        // 対角要素の平坦化添字は `i * n + i`。`checked_numel_for` が
        // `n * n` の計算可能性を既に検査済みのため、`n >= 1` のとき
        // `i < n` の範囲で `i * n + i < n * n` はオーバーフローしない。
        data[i * n + i] = T::one();
    }
    Tensor::new(data, &[n, n])
}

/// `like` と同じ shape・全要素 `T::zero()` の新規テンソルを生成する
/// （PyTorch `torch.zeros_like` 相当。イシュー #1726）。
///
/// `like` が転置・broadcast 由来の非 contiguous view であっても shape
/// のみを引き継ぎ、strides は保存しない新規 contiguous バッファを
/// 返す（モジュール冒頭コメント参照）。broadcast view は shape の
/// 要素数がストレージより大きくなりうるため、`like.shape()` の要素数が
/// アロケーション不能な場合は [`ShapeError::ElementCountOverflow`] を
/// 返す（infallible にはできない）。
pub fn zeros_like<T: Element>(like: &Tensor<T>) -> Result<Tensor<T>, ShapeError> {
    Tensor::zeros(like.shape())
}

/// `like` と同じ shape・全要素 `T::one()` の新規テンソルを生成する
/// （PyTorch `torch.ones_like` 相当。イシュー #1726）。契約は
/// [`zeros_like`] と同じ。
pub fn ones_like<T: Element>(like: &Tensor<T>) -> Result<Tensor<T>, ShapeError> {
    Tensor::ones(like.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn arange_basic_values() {
        let t = arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(t.shape(), &[5]);
        for i in 0..5 {
            assert_eq!(t.get(&[i]).unwrap(), i as f32);
        }

        let t = arange(0.0, 5.0, 2.0).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.get(&[0]).unwrap(), 0.0);
        assert_eq!(t.get(&[1]).unwrap(), 2.0);
        assert_eq!(t.get(&[2]).unwrap(), 4.0);
    }

    #[test]
    fn arange_fractional_step_uses_ceil_length_and_is_bit_identical_to_f64_formula() {
        let step: f32 = 0.3;
        let t = arange(0.0, 1.0, step).unwrap();
        assert_eq!(t.shape(), &[4]);
        for i in 0..4 {
            // 実装は引数として渡された f32（既に丸め済み）を f64 へ
            // 昇格して中間計算するため、期待値も同じ順序（f32 → f64）
            // で計算する（`0.3f64` という別途丸められた f64 リテラルとは
            // 異なる値になるため、それを直接使うと一致しない）。
            let expected = (0.0f64 + i as f64 * step as f64) as f32;
            assert_eq!(t.get(&[i]).unwrap(), expected);
        }
    }

    #[test]
    fn arange_negative_step() {
        let t = arange(5.0, 0.0, -1.0).unwrap();
        assert_eq!(t.shape(), &[5]);
        for i in 0..5 {
            assert_eq!(t.get(&[i]).unwrap(), 5.0 - i as f32);
        }
    }

    #[test]
    fn arange_direction_mismatch_is_empty() {
        assert_eq!(arange(0.0, 5.0, -1.0).unwrap().shape(), &[0]);
        assert_eq!(arange(5.0, 0.0, 1.0).unwrap().shape(), &[0]);
        assert_eq!(arange(3.0, 3.0, 1.0).unwrap().shape(), &[0]);
    }

    #[test]
    fn arange_rejects_zero_and_non_finite_step() {
        assert_eq!(
            arange(0.0, 5.0, 0.0).unwrap_err(),
            CreationError::InvalidStep { step: 0.0 }
        );
        // `f32::NAN` は `PartialEq` で自分自身とも一致しないため
        // （IEEE 754 の NaN 契約）、`assert_eq!` ではなく `matches!` で
        // variant 自体の判別のみを確認する。
        assert!(matches!(
            arange(0.0, 5.0, f32::NAN).unwrap_err(),
            CreationError::InvalidStep { step } if step.is_nan()
        ));
        assert_eq!(
            arange(0.0, 5.0, f32::INFINITY).unwrap_err(),
            CreationError::InvalidStep {
                step: f32::INFINITY
            }
        );
    }

    #[test]
    fn arange_rejects_non_finite_bounds() {
        assert!(matches!(
            arange(f32::NAN, 5.0, 1.0).unwrap_err(),
            CreationError::NonFiniteBound { .. }
        ));
        assert!(matches!(
            arange(0.0, f32::INFINITY, 1.0).unwrap_err(),
            CreationError::NonFiniteBound { .. }
        ));
    }

    #[test]
    fn arange_rejects_astronomical_length_without_allocating() {
        let err = arange(-3.0e38, 3.0e38, 1e-38).unwrap_err();
        assert!(matches!(
            err,
            CreationError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn linspace_endpoints_are_bit_exact() {
        let t = linspace(-1.5, 2.25, 7).unwrap();
        assert_eq!(t.shape(), &[7]);
        assert_eq!(t.get(&[0]).unwrap(), -1.5);
        assert_eq!(t.get(&[6]).unwrap(), 2.25);
        for i in 0..7 {
            let start_f64 = -1.5f64;
            let end_f64 = 2.25f64;
            let step_f64 = (end_f64 - start_f64) / 6.0;
            let expected = if i < 3 {
                (start_f64 + i as f64 * step_f64) as f32
            } else {
                (end_f64 - (6 - i) as f64 * step_f64) as f32
            };
            assert_eq!(t.get(&[i]).unwrap(), expected);
        }
    }

    #[test]
    fn linspace_steps_zero_and_one() {
        let t0 = linspace(1.0, 2.0, 0).unwrap();
        assert_eq!(t0.shape(), &[0]);

        let t1 = linspace(1.0, 2.0, 1).unwrap();
        assert_eq!(t1.shape(), &[1]);
        assert_eq!(t1.get(&[0]).unwrap(), 1.0);
    }

    #[test]
    fn linspace_rejects_non_finite_bounds() {
        assert!(matches!(
            linspace(f32::NAN, 1.0, 3).unwrap_err(),
            CreationError::NonFiniteBound { .. }
        ));
    }

    #[test]
    fn linspace_rejects_capacity_overflow() {
        let err = linspace(0.0, 1.0, usize::MAX).unwrap_err();
        assert!(matches!(
            err,
            CreationError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn eye_values_and_shape() {
        let t = eye::<f32>(3).unwrap();
        assert_eq!(t.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.get(&[i, j]).unwrap(), expected);
            }
        }

        let t_i32 = eye::<i32>(2).unwrap();
        assert_eq!(t_i32.get(&[0, 0]).unwrap(), 1);
        assert_eq!(t_i32.get(&[0, 1]).unwrap(), 0);

        let t_f16 = eye::<f16>(2).unwrap();
        assert_eq!(t_f16.get(&[1, 1]).unwrap(), f16::ONE);
        assert_eq!(t_f16.get(&[1, 0]).unwrap(), f16::ZERO);
    }

    #[test]
    fn eye_zero_is_empty() {
        let t = eye::<f32>(0).unwrap();
        assert_eq!(t.shape(), &[0, 0]);
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn eye_rejects_overflow() {
        let err = eye::<f32>(usize::MAX).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn zeros_like_ones_like_follow_shape_of_contiguous_and_transposed_views() {
        let base = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let z = zeros_like(&base).unwrap();
        assert_eq!(z.shape(), &[2, 3]);
        assert!(z.is_contiguous());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(z.get(&[i, j]).unwrap(), 0.0);
            }
        }

        let o = ones_like(&base).unwrap();
        assert_eq!(o.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(o.get(&[i, j]).unwrap(), 1.0);
            }
        }

        let transposed = base.transpose_2d().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        let z_t = zeros_like(&transposed).unwrap();
        assert_eq!(z_t.shape(), &[3, 2]);
        assert!(z_t.is_contiguous());
    }

    #[test]
    fn zeros_like_of_broadcast_view_allocates_full_shape_or_rejects() {
        let base = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let broadcasted = base.broadcast_to(&[4, 3]).unwrap();
        let z = zeros_like(&broadcasted).unwrap();
        assert_eq!(z.shape(), &[4, 3]);
        assert!(z.is_contiguous());
    }

    #[test]
    fn creation_error_from_shape_error_and_display() {
        let shape_err = ShapeError::ElementCountOverflow;
        let creation_err: CreationError = shape_err.clone().into();
        assert_eq!(creation_err, CreationError::Shape(shape_err));
        assert!(!format!("{creation_err}").is_empty());

        let step_err = CreationError::InvalidStep { step: 0.0 };
        assert!(format!("{step_err}").contains("step"));

        let bound_err = CreationError::NonFiniteBound {
            start: f32::NAN,
            end: 1.0,
        };
        assert!(!format!("{bound_err}").is_empty());
    }
}
