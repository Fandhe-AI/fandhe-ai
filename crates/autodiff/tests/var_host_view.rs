//! `Var::host_view`（イシュー #1335）の統合テスト。
//!
//! `to_tensor().contiguous().as_slice().to_vec()`（従来の値取り出し
//! 経路）と bit 同一であることを、contiguous な演算結果（借用分岐）と
//! 非 contiguous な `transpose` 後の値（所有分岐）の両方で確認する。
//! CPU（`Tape::new()` 既定バックエンド）で実行し、実機依存はない。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|v| v.to_bits()).collect()
}

/// contiguous な演算結果（`matmul`）では `host_view()` が
/// `to_tensor().contiguous().as_slice().to_vec()` と bit 同一。
#[test]
fn host_view_matches_to_tensor_for_contiguous_result() {
    let tape = Tape::new();
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
    let y = a.matmul(&b).unwrap();

    let expected = y.to_tensor().contiguous().as_slice().unwrap().to_vec();
    let view = y.host_view();

    assert_eq!(
        bits(&view),
        bits(&expected),
        "host_view は to_tensor().contiguous().as_slice() と bit 同一のはず"
    );
}

/// 非 contiguous な `transpose` 後の値（所有 `Cow::Owned` 相当分岐）でも
/// `host_view()` は同じ bit 同一契約を保つ。
#[test]
fn host_view_matches_to_tensor_for_transposed_result() {
    let tape = Tape::new();
    let x = tape.var(&t((0..6).map(|v| v as f32).collect(), &[2, 3]));
    let y = x.transpose(0, 1).unwrap();

    let expected = y.to_tensor().contiguous().as_slice().unwrap().to_vec();
    let view = y.host_view();

    assert_eq!(
        bits(&view),
        bits(&expected),
        "transpose 後（非 contiguous）でも host_view は bit 同一のはず"
    );
}

/// 空テンソルの `host_view()` は空スライスを返す。
#[test]
fn host_view_on_empty_tensor_is_empty() {
    let tape = Tape::new();
    let x = tape.var(&Tensor::<f32>::zeros(&[0, 3]).unwrap());
    let view = x.host_view();
    assert!(view.is_empty());
}
