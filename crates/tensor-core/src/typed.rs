//! 固定次元 API（TASK-10.1b・#99）。
//!
//! `tensor-core` 基盤層の `Tensor<T>` は rank・shape を実行時値として
//! 保持し、コンパイル時には検査しない（PoC-v2-1 の確定事項・
//! `docs/public-api-design.md` §2.5「REQ-10 との関係」）。safetensors/ONNX
//! からロードする重みの shape は実行時にしか決まらないため、この設計は
//! 変更しない。本モジュールはその**上に積む後続レイヤー**として、
//! アーキテクチャ上コンパイル時に shape が固定される層（全結合層の
//! 重み・bias 等）に限定して const generics による型レベル shape 検証を
//! 提供する（spec 根拠: `docs/spec/04-requirements.md` REQ-10・
//! `docs/spec/05-tasks.md` TASK-10.1）。
//!
//! # 設計方針・適用境界
//!
//! v1 PoC-7（`docs/spec/03-poc/poc-7-type-safety/README.md`、基盤非依存の
//! 教訓として v2 に引き継ぐ）の実測に基づき、以下の境界を採用する:
//!
//! - **バッチ次元は型に載せない**: 可変バッチ推論と衝突するため、
//!   [`BatchedFeatures`] の第 1 軸（batch）は常に実行時次元のままとする
//!   （REQ-10 受け入れ基準・PoC-7 教訓）。
//! - **`generic_const_exprs` は使用しない**: nightly 限定機能のため
//!   stable Rust のみで構成する。`cat`（連結軸）等の算術 shape 検証は
//!   本モジュールの対象外。
//! - **デバイスタグの型パラメータ化は実装しない**: REQ-10 が「概念実証
//!   止まり」と明記しており、本イシュー（TASK-10.1b）では扱わない。
//!
//! # イシュー分割
//!
//! TASK-10.1 は 3 分割されている: #98（TASK-10.1a 型設計の文書化。着手時点
//! で未マージのため本モジュールは計画のフォールバック設計で実装した）→
//! 本モジュール（#99・TASK-10.1b 固定次元 API の実装）→ #100（TASK-10.1c
//! テスト整備。trybuild 等の手法検討を含む）。#98 が後日マージされた場合は
//! 本モジュールとの命名・型構成の整合を別途確認する必要がある
//! （out-of-scope。PR 本文に記録）。
//!
//! # fail-closed な境界コンストラクタ
//!
//! [`FixedVec::from_tensor`]・[`FixedMat::from_tensor`]・
//! [`BatchedFeatures::from_tensor`] はいずれも実行時 shape/rank を
//! const パラメータと突合してから受け入れ、不一致は既存の
//! [`ShapeError::RankMismatch`]/[`ShapeError::ShapeMismatch`] で拒否する
//! （`.claude/rules/security.md` A03: 外部入力検証を先に行う fail-closed
//! 方針。safetensors/ONNX ロード経路が固定次元層へ重みを渡す際の検証
//! 境界になる）。unchecked に構築する経路は設けない。

use crate::error::ShapeError;
use crate::tensor::Tensor;

/// 1 次元固定長テンソル（bias 等、アーキテクチャ上サイズが固定される
/// ベクトル）。内部 shape は常に `[N]`。
///
/// `N` はコンパイル時に確定するため、`N` が異なる `FixedVec` 同士を
/// 誤って渡す呼び出しはコンパイルエラーになる（後述の `compile_fail`
/// doctest 参照）。
#[derive(Clone, Debug)]
pub struct FixedVec<T: crate::element::Element, const N: usize> {
    inner: Tensor<T>,
}

impl<T: crate::element::Element, const N: usize> FixedVec<T, N> {
    /// 実行時 `Tensor<T>` の shape を `[N]` と突合して受け入れる
    /// 境界コンストラクタ。rank が 1 でなければ `RankMismatch`、
    /// 長さが `N` と異なれば `ShapeMismatch` を返す。
    pub fn from_tensor(t: Tensor<T>) -> Result<Self, ShapeError> {
        let rank = t.rank();
        if rank != 1 {
            return Err(ShapeError::RankMismatch {
                expected: 1,
                actual: rank,
            });
        }
        if t.shape()[0] != N {
            return Err(ShapeError::ShapeMismatch {
                lhs: vec![N],
                rhs: t.shape().to_vec(),
            });
        }
        Ok(FixedVec { inner: t })
    }

    /// 動的層（`Tensor<T>` を直接扱う既存 API）への脱出口。
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.inner
    }

    /// 所有権ごと動的層へ戻す。
    pub fn into_tensor(self) -> Tensor<T> {
        self.inner
    }
}

/// 2 次元固定次元テンソル（全結合層の重み等）。内部 shape は常に
/// `[IN, OUT]`。
///
/// `IN`/`OUT` の順序はそのまま型に載る。`FixedMat<T, IN, OUT>` を
/// 期待するシグネチャへ転置違い（`FixedMat<T, OUT, IN>`）を渡す呼び出しは
/// コンパイルエラーになる（後述の `compile_fail` doctest 参照）。
#[derive(Clone, Debug)]
pub struct FixedMat<T: crate::element::Element, const IN: usize, const OUT: usize> {
    inner: Tensor<T>,
}

impl<T: crate::element::Element, const IN: usize, const OUT: usize> FixedMat<T, IN, OUT> {
    /// 実行時 `Tensor<T>` の shape を `[IN, OUT]` と突合して受け入れる
    /// 境界コンストラクタ。
    pub fn from_tensor(t: Tensor<T>) -> Result<Self, ShapeError> {
        let rank = t.rank();
        if rank != 2 {
            return Err(ShapeError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        let shape = t.shape();
        if shape[0] != IN || shape[1] != OUT {
            return Err(ShapeError::ShapeMismatch {
                lhs: vec![IN, OUT],
                rhs: shape.to_vec(),
            });
        }
        Ok(FixedMat { inner: t })
    }

    /// 動的層への脱出口。
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.inner
    }

    /// 所有権ごと動的層へ戻す。
    pub fn into_tensor(self) -> Tensor<T> {
        self.inner
    }
}

/// バッチ入り特徴量テンソル（`[batch, F]`）。**batch は意図的に型パラ
/// メータへ含めない実行時次元**であり（モジュールドキュメント参照）、
/// `F`（特徴次元）のみを型で固定する。
#[derive(Clone, Debug)]
pub struct BatchedFeatures<T: crate::element::Element, const F: usize> {
    inner: Tensor<T>,
}

impl<T: crate::element::Element, const F: usize> BatchedFeatures<T, F> {
    /// 実行時 `Tensor<T>` の shape を `[batch, F]`（batch は任意）と
    /// 突合して受け入れる境界コンストラクタ。
    pub fn from_tensor(t: Tensor<T>) -> Result<Self, ShapeError> {
        let rank = t.rank();
        if rank != 2 {
            return Err(ShapeError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        if t.shape()[1] != F {
            return Err(ShapeError::ShapeMismatch {
                lhs: vec![t.shape()[0], F],
                rhs: t.shape().to_vec(),
            });
        }
        Ok(BatchedFeatures { inner: t })
    }

    /// `kernel` 実行後の出力を型付きに包む内部専用の境界コンストラクタ。
    /// `from_tensor` の rank/特徴次元検査に加えてバッチ次元
    /// （`expected_batch`）も突合する。`matmul_with`/`add_bias_with` が
    /// 「呼び出し元の入力バッチサイズと出力のバッチサイズが一致する」
    /// ことまで再検査するために使う（型と実体の乖離防止。REQ-8 の
    /// カーネル境界検査規約と同趣旨）。公開 `from_tensor` は入力ロード
    /// 境界用でバッチサイズを未知として許容するため、ここでは分離する。
    fn from_tensor_with_batch(t: Tensor<T>, expected_batch: usize) -> Result<Self, ShapeError> {
        let out = Self::from_tensor(t)?;
        if out.inner.shape()[0] != expected_batch {
            return Err(ShapeError::ShapeMismatch {
                lhs: vec![expected_batch, F],
                rhs: out.inner.shape().to_vec(),
            });
        }
        Ok(out)
    }

    /// 動的層への脱出口。
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.inner
    }

    /// 所有権ごと動的層へ戻す。
    pub fn into_tensor(self) -> Tensor<T> {
        self.inner
    }

    /// バッチサイズ（実行時次元）を返す。
    pub fn batch_size(&self) -> usize {
        self.inner.shape()[0]
    }

    /// 全結合層相当の型付き行列積: 内側次元 `F`（＝ `self` の特徴次元と
    /// `w` の入力次元）の一致を型で強制する（v1 PoC-7 の 6 誤りパターン
    /// 中 6 件検出方式を踏襲）。
    ///
    /// `tensor-core` は計算カーネルを持たない（GEMM は `backend-cpu` 等の
    /// 実装。`docs/public-api-design.md` §4）ため、実際の計算は
    /// 呼び出し元が渡す `kernel` クロージャに委譲する「基盤 `Tensor<T>`
    /// を計算する関数を型付きで包む」ジェネリック合成として提供する。
    /// `kernel` 実行後の出力 shape も `[batch, OUT]` であることを再検査
    /// してから型付きに包む（型と実体の乖離を防ぐ二重防御。境界検査を
    /// 省略しない方針は REQ-8 のカーネル境界検査規約と同趣旨）。
    pub fn matmul_with<const OUT: usize>(
        &self,
        w: &FixedMat<T, F, OUT>,
        kernel: impl FnOnce(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>, ShapeError>,
    ) -> Result<BatchedFeatures<T, OUT>, ShapeError> {
        let out = kernel(&self.inner, &w.inner)?;
        BatchedFeatures::from_tensor_with_batch(out, self.batch_size())
    }

    /// bias 加算の型付きシグネチャ: 特徴次元 `F` の一致を型で強制する。
    /// `matmul_with` と同様、実計算は `kernel` に委譲し出力 shape を
    /// 再検査してから型付きに包む。
    pub fn add_bias_with(
        &self,
        b: &FixedVec<T, F>,
        kernel: impl FnOnce(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>, ShapeError>,
    ) -> Result<BatchedFeatures<T, F>, ShapeError> {
        let out = kernel(&self.inner, &b.inner)?;
        BatchedFeatures::from_tensor_with_batch(out, self.batch_size())
    }
}

// --- コンパイルエラー実証（受け入れ条件: shape 不一致がコンパイル
// エラーになるケースの実証。`trybuild` 等の新規 dev-dependency は
// 許容依存 8 区分外でユーザー承認必須〈deps-policy.md〉のため、追加
// 依存ゼロで実証できる rustdoc `compile_fail` doctest を用いる） ---

/// 正常系: `matmul_with` は内側次元が一致していればコンパイル・実行とも
/// 成功する。
///
/// ```
/// use tensor_core::typed::{BatchedFeatures, FixedMat};
/// use tensor_core::Tensor;
///
/// let x = BatchedFeatures::<f32, 3>::from_tensor(
///     Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
/// )
/// .unwrap();
/// let w = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();
/// let y = x
///     .matmul_with(&w, |a, b| {
///         Ok(Tensor::zeros(&[a.shape()[0], b.shape()[1]]).unwrap())
///     })
///     .unwrap();
/// assert_eq!(y.batch_size(), 2);
/// ```
///
/// 誤りパターン 1: `matmul_with` の内側次元不一致
/// （`BatchedFeatures<f32, 3>` の特徴次元 3 に対し `FixedMat<f32, 4, 5>`
/// は入力次元 4 を要求するため型が合わず、コンパイルエラーになる）。
///
/// ```compile_fail
/// use tensor_core::typed::{BatchedFeatures, FixedMat};
/// use tensor_core::Tensor;
///
/// let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
/// let w = FixedMat::<f32, 4, 5>::from_tensor(Tensor::zeros(&[4, 5]).unwrap()).unwrap();
/// // 期待: FixedMat<f32, 3, _>。実際: FixedMat<f32, 4, 5> でコンパイルエラー。
/// let _y = x.matmul_with(&w, |a, b| {
///     Ok(Tensor::zeros(&[a.shape()[0], b.shape()[1]]).unwrap())
/// });
/// ```
///
/// 誤りパターン 2: bias 次元不一致（`BatchedFeatures<f32, 3>` に
/// `FixedVec<f32, 4>` を渡すとコンパイルエラーになる）。
///
/// ```compile_fail
/// use tensor_core::typed::{BatchedFeatures, FixedVec};
/// use tensor_core::Tensor;
///
/// let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
/// let b = FixedVec::<f32, 4>::from_tensor(Tensor::zeros(&[4]).unwrap()).unwrap();
/// // 期待: FixedVec<f32, 3>。実際: FixedVec<f32, 4> でコンパイルエラー。
/// let _y = x.add_bias_with(&b, |a, bias| {
///     Ok(Tensor::zeros(&[a.shape()[0], bias.shape()[0]]).unwrap())
/// });
/// ```
///
/// 誤りパターン 3: 型の取り違え（`FixedMat<f32, IN, OUT>` を期待する
/// 箇所へ転置形 `FixedMat<f32, OUT, IN>` を渡すとコンパイルエラーになる）。
///
/// ```compile_fail
/// use tensor_core::typed::{BatchedFeatures, FixedMat};
/// use tensor_core::Tensor;
///
/// fn apply(x: &BatchedFeatures<f32, 3>, w: &FixedMat<f32, 3, 4>) {
///     let _ = x.matmul_with(w, |a, b| {
///         Ok(Tensor::zeros(&[a.shape()[0], b.shape()[1]]).unwrap())
///     });
/// }
///
/// let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
/// // 転置形 FixedMat<f32, 4, 3> を渡す誤り。apply は FixedMat<f32, 3, 4> を要求する。
/// let w_transposed = FixedMat::<f32, 4, 3>::from_tensor(Tensor::zeros(&[4, 3]).unwrap()).unwrap();
/// apply(&x, &w_transposed);
/// ```
///
/// # なぜ `pub` かつ `#[doc(hidden)]` か
///
/// rustdoc は `--document-private-items` を渡さない限り非公開アイテムの
/// doctest をスキップし `cargo test --workspace` で実行されない
/// （実装レビュー #100 Bugbot 指摘）。本アイテムは上記 doctest を
/// `cargo test` で確実に実行させるためだけの錨（anchor）であり公開 API
/// ではないため、`pub` にしつつ `#[doc(hidden)]` で公開ドキュメントには
/// 出力しない。
#[doc(hidden)]
pub fn _doc_anchor() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_vec_from_tensor_ok_and_rank_mismatch() {
        let v = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap();
        assert_eq!(v.as_tensor().shape(), &[3]);

        let err = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn fixed_vec_from_tensor_shape_mismatch() {
        let err = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[4]).unwrap()).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { lhs, rhs }
            if lhs == vec![3] && rhs == vec![4]));
    }

    #[test]
    fn fixed_mat_from_tensor_ok_and_shape_mismatch() {
        let m = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();
        assert_eq!(m.as_tensor().shape(), &[3, 4]);

        let err = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 5]).unwrap()).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { lhs, rhs }
            if lhs == vec![3, 4] && rhs == vec![3, 5]));
    }

    #[test]
    fn fixed_mat_from_tensor_rank_mismatch() {
        let err = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn batched_features_from_tensor_ok_and_batch_size() {
        let bf = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[5, 3]).unwrap()).unwrap();
        assert_eq!(bf.batch_size(), 5);
        assert_eq!(bf.as_tensor().shape(), &[5, 3]);
    }

    #[test]
    fn batched_features_from_tensor_feature_mismatch() {
        let err =
            BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[5, 4]).unwrap()).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn batched_features_from_tensor_rank_mismatch() {
        let err = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn matmul_with_composes_types_and_checks_output_shape() {
        let x = BatchedFeatures::<f32, 3>::from_tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
        )
        .unwrap();
        let w = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();
        let y = x
            .matmul_with(&w, |a, b| {
                Ok(Tensor::zeros(&[a.shape()[0], b.shape()[1]]).unwrap())
            })
            .unwrap();
        assert_eq!(y.batch_size(), 2);
        assert_eq!(y.as_tensor().shape(), &[2, 4]);
    }

    #[test]
    fn matmul_with_rejects_kernel_output_shape_mismatch() {
        // kernel が誤った shape を返した場合、型と実体の乖離を二重防御で
        // 検出する（REQ-8 のカーネル境界検査規約と同趣旨: 出力 shape を
        // 再検査してから型付きに包む）。
        let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
        let w = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();
        let err = x
            .matmul_with(&w, |_a, _b| Ok(Tensor::zeros(&[2, 5]).unwrap()))
            .unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }

    #[test]
    fn add_bias_with_composes_types() {
        let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
        let b = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap();
        let y = x
            .add_bias_with(&b, |a, bias| {
                Ok(Tensor::zeros(&[a.shape()[0], bias.shape()[0]]).unwrap())
            })
            .unwrap();
        assert_eq!(y.as_tensor().shape(), &[2, 3]);
    }

    #[test]
    fn into_tensor_roundtrip() {
        let t = Tensor::<f32>::zeros(&[3]).unwrap();
        let v = FixedVec::<f32, 3>::from_tensor(t.clone()).unwrap();
        let back = v.into_tensor();
        assert_eq!(back.shape(), t.shape());
    }
}
