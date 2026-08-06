//! `backend-cpu::elementwise` の受け入れ条件対応テスト（#22・TASK-1.6b）。
//!
//! 親イシュー #20（TASK-1.6）の受け入れ条件「elementwise 演算の数値・
//! ブロードキャスト対応が期待値と一致する」を、公開 API（`add`/`mul`/
//! `relu`/`exp`/`tanh`）に対する統合テストとして検証する。単体テスト
//! （スライスカーネル・shape 走査ロジック単位の細かい検証）は
//! `src/elementwise.rs` 内の `#[cfg(test)]` に配置済みであり、本ファイルは
//! 公開 API を外部クレートと同じ経路で呼び出す統合テストに限定する。
//!
//! ## #25（TASK-1.6e）棚卸しメモ
//!
//! 空テンソル網羅（`mul`/`exp`/`tanh`・shape バリエーション）・
//! `PARALLEL_THRESHOLD` 境界・非正方 rank3 broadcast・長さ不一致
//! `should_panic` 契約は `PARALLEL_THRESHOLD` 等の非公開定数を参照する
//! ため `src/elementwise.rs` のインライン `#[cfg(test)]` へ追加した
//! （既存の配置規約どおり）。本ファイルには公開 API のみで組み立てられる
//! 統合テストとして `mul` の非 contiguous view 一致テストを追加する。

use backend_cpu::{add, exp, mul, relu, tanh};
use tensor_core::Tensor;

#[test]
fn add_matches_expected_values() {
    let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let out = add(&a, &b).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    let expected = [11.0, 22.0, 33.0, 44.0];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(out.get(&[i / 2, i % 2]).unwrap(), v);
    }
}

#[test]
fn mul_matches_expected_values() {
    let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = Tensor::<f32>::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
    let out = mul(&a, &b).unwrap();
    let expected = [2.0, 6.0, 12.0, 20.0];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(out.get(&[i]).unwrap(), v);
    }
}

#[test]
fn relu_matches_expected_values() {
    let a = Tensor::<f32>::new(vec![-2.0, -0.5, 0.0, 0.5, 2.0], &[5]).unwrap();
    let out = relu(&a).unwrap();
    let expected = [0.0, 0.0, 0.0, 0.5, 2.0];
    for (i, &v) in expected.iter().enumerate() {
        assert_eq!(out.get(&[i]).unwrap(), v);
    }
}

#[test]
fn exp_matches_std_f32_exp() {
    let a = Tensor::<f32>::new(vec![-1.0, 0.0, 0.5, 1.0, 3.0], &[5]).unwrap();
    let out = exp(&a).unwrap();
    for i in 0..5 {
        assert_eq!(out.get(&[i]).unwrap(), a.get(&[i]).unwrap().exp());
    }
}

#[test]
fn tanh_matches_std_f32_tanh() {
    let a = Tensor::<f32>::new(vec![-1.0, 0.0, 0.5, 1.0, 3.0], &[5]).unwrap();
    let out = tanh(&a).unwrap();
    for i in 0..5 {
        assert_eq!(out.get(&[i]).unwrap(), a.get(&[i]).unwrap().tanh());
    }
}

#[test]
fn add_broadcasts_row_and_column_vectors() {
    // 受け入れ条件対象の代表例: [3,1] + [1,4] -> [3,4]。
    let col = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let row = Tensor::<f32>::new(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
    let out = add(&col, &row).unwrap();
    assert_eq!(out.shape(), &[3, 4]);
    for i in 0..3 {
        for j in 0..4 {
            let expected = col.get(&[i, 0]).unwrap() + row.get(&[0, j]).unwrap();
            assert_eq!(out.get(&[i, j]).unwrap(), expected);
        }
    }
}

#[test]
fn add_broadcasts_bias_vector_over_matrix() {
    let matrix = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
    let bias = Tensor::<f32>::new(vec![100.0, 200.0, 300.0], &[3]).unwrap();
    let out = add(&matrix, &bias).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(
                out.get(&[i, j]).unwrap(),
                matrix.get(&[i, j]).unwrap() + bias.get(&[j]).unwrap()
            );
        }
    }
}

#[test]
fn mul_broadcasts_scalar_like_shape() {
    let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let scalar = Tensor::<f32>::new(vec![2.0], &[]).unwrap();
    let out = mul(&a, &scalar).unwrap();
    for i in 0..3 {
        assert_eq!(out.get(&[i]).unwrap(), a.get(&[i]).unwrap() * 2.0);
    }
}

#[test]
fn add_incompatible_shapes_returns_broadcast_incompatible() {
    let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
    let b = Tensor::<f32>::zeros(&[4]).unwrap();
    let err = add(&a, &b).unwrap_err();
    assert!(matches!(
        err,
        tensor_core::ShapeError::BroadcastIncompatible { .. }
    ));
}

#[test]
fn add_with_non_contiguous_view_matches_contiguous() {
    let a = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
    let a_t = a.transpose(0, 1).unwrap(); // shape [3, 2]、非 contiguous
    let b = Tensor::<f32>::new(vec![1.0; 6], &[3, 2]).unwrap();

    let out_view = add(&a_t, &b).unwrap();
    let out_contiguous = add(&a_t.contiguous(), &b).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(
                out_view.get(&[i, j]).unwrap(),
                out_contiguous.get(&[i, j]).unwrap()
            );
        }
    }
}

#[test]
fn relu_with_narrowed_view_matches_contiguous() {
    let a = Tensor::<f32>::new(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[8]).unwrap();
    let n = a.narrow(0, 2, 4).unwrap();
    let out_view = relu(&n).unwrap();
    let out_contiguous = relu(&n.contiguous()).unwrap();
    for i in 0..4 {
        assert_eq!(
            out_view.get(&[i]).unwrap(),
            out_contiguous.get(&[i]).unwrap()
        );
    }
}

#[test]
fn add_empty_tensor_is_empty() {
    let a = Tensor::<f32>::zeros(&[0, 3]).unwrap();
    let b = Tensor::<f32>::zeros(&[0, 3]).unwrap();
    let out = add(&a, &b).unwrap();
    assert_eq!(out.shape(), &[0, 3]);
    assert!(out.is_empty());
}

/// `mul` の非 contiguous view（transpose 後）が `contiguous()` 実体化後と
/// 一致することを確認する（`add`/`relu` の既存カバレッジを `mul` へ展開。
/// #25 棚卸しで特定したギャップ）。
#[test]
fn mul_with_non_contiguous_view_matches_contiguous() {
    let a = Tensor::<f32>::new((0..6).map(|v| v as f32 + 1.0).collect(), &[2, 3]).unwrap();
    let a_t = a.transpose(0, 1).unwrap(); // shape [3, 2]、非 contiguous
    let b = Tensor::<f32>::new(vec![2.0; 6], &[3, 2]).unwrap();

    let out_view = mul(&a_t, &b).unwrap();
    let out_contiguous = mul(&a_t.contiguous(), &b).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(
                out_view.get(&[i, j]).unwrap(),
                out_contiguous.get(&[i, j]).unwrap()
            );
        }
    }
}
