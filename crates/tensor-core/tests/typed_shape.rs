//! `fandhe_ai_tensor_core::typed`（TASK-10.1c・#100）の型レベル shape 検証テスト整備。
//!
//! `typed.rs` 内 `#[cfg(test)] mod tests`（#99・eca971b）は非公開フィールド
//! （`inner`）にアクセス可能なモジュール内白箱テストであり、`typed` を
//! 消費する外部クレートの視点（`pub` 経由でのみ到達可能な API）を検証して
//! いない。本ファイルはその境界を埋める統合テストとして、以下の観点を
//! 追加する（`tensor_views.rs` と同様、モジュール内テストと重複しない
//! 「複数 API の組合せ・外部消費者視点」観点に限定する）。
//!
//! - 公開 API のみで `FixedVec`/`FixedMat`/`BatchedFeatures` の
//!   構築・変換・型付き演算合成が完結すること（非公開フィールドに一切
//!   触れずに到達可能であることの実証）。
//! - `BatchedFeatures` のモジュールドキュメントが明記する設計判断
//!   （batch 次元は意図的に型パラメータへ含めない実行時次元。v1 PoC-7・
//!   REQ-10 受け入れ基準の教訓）を、異なる batch サイズが同一の型付き
//!   パイプラインを通ることで実証する。
//! - `add_bias_with` の出力 shape 再検査（型と実体の乖離を防ぐ二重防御。
//!   REQ-8 のカーネル境界検査規約と同趣旨）。`matmul_with` 側は #99 の
//!   `matmul_with_rejects_kernel_output_shape_mismatch` で検証済みだが、
//!   `add_bias_with` 側は未検証だったため対称に追加する。
//! - `kernel` クロージャが `Err` を返した場合の `?` 伝播経路
//!   （`matmul_with`/`add_bias_with` とも）。
//! - `BatchedFeatures::from_tensor` への rank-3 入力（既存テストは
//!   rank-1 のみを検証していたため、rank 不一致の別パターンを補う）。
//! - `matmul_with`/`add_bias_with` の出力バッチ次元再検査。`kernel` が
//!   特徴次元は正しいがバッチ次元だけ誤った出力を返す壊れ方は、特徴
//!   次元のみを見る `from_tensor` の再検査では検出できなかった
//!   （実装レビュー #100 指摘。バッチ不一致は `ShapeError::ShapeMismatch`
//!   で拒否する対称ケースを補う）。
//!
//! CI（self-hosted）は `docs/spec`（submodule）を checkout しないため、
//! 本ファイルは `docs/spec` 配下のいかなるファイルにも依存しない。
//!
//! # コンパイルエラー実証手法の検討（受け入れ条件対応）
//!
//! 受け入れ条件「コンパイル成功／失敗の両ケース（trybuild 等の手法検討
//! 含む）のテストを整備する」について、`trybuild` の採用を検討したが
//! 見送った。理由:
//!
//! - `trybuild` は許容依存 8 区分（`.claude/rules/deps-policy.md`）に
//!   含まれておらず、新規追加はユーザー承認が必須（同ファイル「バージョン
//!   固定」節）。本タスク単独でのユーザー承認取得は範囲外。
//! - `typed.rs` の `compile_fail` doctest 3 ケース（#99・eca971b で追加
//!   済み）で「内側次元不一致」「bias 次元不一致」「転置形の取り違え」の
//!   3 パターンを追加依存ゼロで実証できており、`trybuild` 固有の利点
//!   （stderr メッセージの厳密比較・複数ケースの一括実行）は本クレートの
//!   現行スコープでは必須ではない。
//!
//! 結論として `compile_fail` doctest を継続採用する。`trybuild` 導入が
//! 将来的に必要になった場合（例: エラーメッセージの回帰検知強化）は、
//! 依存追加としてユーザー承認を得たうえで別タスクで検討する
//! （`.claude/rules/out-of-scope-tracking.md` 準拠）。

use fandhe_ai_tensor_core::typed::{BatchedFeatures, FixedMat, FixedVec};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

// --- 公開 API のみでの構築・変換・型付き演算合成（外部消費者視点） ---

#[test]
fn public_api_only_builds_and_composes_typed_pipeline() {
    // 非公開フィールド（inner）に触れず、コンストラクタ・アクセサ・
    // 型付き演算のみで Linear 層相当（matmul + bias）を組み立てられる
    // ことを確認する。
    let x = BatchedFeatures::<f32, 3>::from_tensor(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
    )
    .unwrap();
    let w = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();
    let b = FixedVec::<f32, 4>::from_tensor(Tensor::zeros(&[4]).unwrap()).unwrap();

    let y = x
        .matmul_with(&w, |a, wt| {
            Ok(Tensor::zeros(&[a.shape()[0], wt.shape()[1]]).unwrap())
        })
        .unwrap();
    let z = y
        .add_bias_with(&b, |a, bias| {
            Ok(Tensor::zeros(&[a.shape()[0], bias.shape()[0]]).unwrap())
        })
        .unwrap();

    assert_eq!(z.batch_size(), 2);
    assert_eq!(z.as_tensor().shape(), &[2, 4]);
    let back = z.into_tensor();
    assert_eq!(back.shape(), &[2, 4]);
}

// --- batch 次元は実行時のまま（型に載せない設計）の実証 ---

#[test]
fn batch_dimension_varies_through_same_typed_pipeline() {
    // BatchedFeatures<T, F> は F のみ型で固定し、batch は実行時次元の
    // ままである（モジュールドキュメント・v1 PoC-7 の教訓）。異なる
    // batch サイズが同一の型（同一の FixedMat/FixedVec）を通じて処理
    // できることを確認する。
    let w = FixedMat::<f32, 3, 2>::from_tensor(Tensor::zeros(&[3, 2]).unwrap()).unwrap();

    for batch in [1usize, 2, 128] {
        let x =
            BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[batch, 3]).unwrap()).unwrap();
        assert_eq!(x.batch_size(), batch);

        let y = x
            .matmul_with(&w, |a, wt| {
                Ok(Tensor::zeros(&[a.shape()[0], wt.shape()[1]]).unwrap())
            })
            .unwrap();
        assert_eq!(y.batch_size(), batch);
        assert_eq!(y.as_tensor().shape(), &[batch, 2]);
    }
}

// --- add_bias_with の出力 shape 再検査（matmul_with と対称の二重防御） ---

#[test]
fn add_bias_with_rejects_kernel_output_shape_mismatch() {
    // kernel が誤った shape（特徴次元不一致）を返した場合、型と実体の
    // 乖離を出力 shape の再検査で検出する。matmul_with 側は #99 の
    // matmul_with_rejects_kernel_output_shape_mismatch で検証済みのため、
    // add_bias_with 側の対称ケースを補う。
    let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
    let b = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap();

    let err = x
        .add_bias_with(&b, |_a, _bias| Ok(Tensor::zeros(&[2, 5]).unwrap()))
        .unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

// --- matmul_with/add_bias_with の出力バッチ次元再検査 ---

#[test]
fn matmul_with_rejects_kernel_output_batch_mismatch() {
    // kernel が特徴次元は正しい（OUT=4）が batch を誤って書き換えた
    // 出力（期待 [2, 4] に対し [99, 4]）を返した場合、from_tensor の
    // 特徴次元検査だけでは検出できない。呼び出し元の batch_size() との
    // 突合で拒否されることを確認する。
    let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
    let w = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();

    let err = x
        .matmul_with(&w, |_a, _w| Ok(Tensor::zeros(&[99, 4]).unwrap()))
        .unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

#[test]
fn add_bias_with_rejects_kernel_output_batch_mismatch() {
    // matmul_with と対称のケース: bias 加算 kernel が特徴次元は正しい
    // が batch を誤って書き換えた出力を返した場合の拒否を確認する。
    let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
    let b = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap();

    let err = x
        .add_bias_with(&b, |_a, _bias| Ok(Tensor::zeros(&[99, 3]).unwrap()))
        .unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

// --- kernel クロージャの Err 伝播（? 演算子の経路） ---

#[test]
fn matmul_with_propagates_kernel_error() {
    let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
    let w = FixedMat::<f32, 3, 4>::from_tensor(Tensor::zeros(&[3, 4]).unwrap()).unwrap();

    let err = x
        .matmul_with(&w, |_a, _b| {
            Err(ShapeError::RankMismatch {
                expected: 2,
                actual: 3,
            })
        })
        .unwrap_err();
    assert!(matches!(
        err,
        ShapeError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn add_bias_with_propagates_kernel_error() {
    let x = BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3]).unwrap()).unwrap();
    let b = FixedVec::<f32, 3>::from_tensor(Tensor::zeros(&[3]).unwrap()).unwrap();

    let err = x
        .add_bias_with(&b, |_a, _bias| {
            Err(ShapeError::RankMismatch {
                expected: 2,
                actual: 1,
            })
        })
        .unwrap_err();
    assert!(matches!(
        err,
        ShapeError::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

// --- BatchedFeatures::from_tensor への rank 不一致の追加パターン ---

#[test]
fn batched_features_from_tensor_rejects_rank3_input() {
    // 既存の白箱テスト（#99）は rank-1 入力のみを RankMismatch で検証
    // していた。rank-3 入力（[2, 3, 4] 等）でも同様に拒否されることを
    // 補う。
    let err =
        BatchedFeatures::<f32, 3>::from_tensor(Tensor::zeros(&[2, 3, 4]).unwrap()).unwrap_err();
    assert!(matches!(
        err,
        ShapeError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}
