//! 受け入れ条件「forward 実行時にテープへ演算が記録される」を直接検証
//! する統合テスト（TASK-1.5a・イシュー #16）。
//!
//! `Tape::len()` でノード数の増加・発生順記録を、`Var::to_tensor()` で
//! 各演算の forward 値が naive 計算の期待値と一致することを確認する。
//! shape 不整合・クロステープ検査の異常系、`RefCell` 借用モデルの
//! 回帰も併せて検証する（実機依存なし。CI 実行可能）。

mod common;

use autodiff::{AutodiffError, Tape};
use tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 1. forward 実行の連鎖（matmul → add → relu → mse_loss）で、テープ
///    ノード数が発生順に増加することを検証する（受け入れ条件の直接検証）。
#[test]
fn forward_execution_records_nodes_in_order() {
    let tape = Tape::new_with_ops(common::naive_ops());
    assert!(tape.is_empty());

    // x: [2,2], w: [2,2], b: [2] (bias broadcast), target: [2,2]
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let w = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let b = tape.var(&t(vec![10.0, 20.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));
    // 4 leaf ノードが登録された時点でノード数は 4。
    assert_eq!(tape.len(), 4);

    let y = x.matmul(&w).unwrap(); // node 5
    assert_eq!(tape.len(), 5);
    let y = y.add(&b).unwrap(); // node 6 (bias broadcast)
    assert_eq!(tape.len(), 6);
    let y = y.relu(); // node 7
    assert_eq!(tape.len(), 7);
    let loss = y.mse_loss(&target).unwrap(); // node 8
    assert_eq!(tape.len(), 8);

    // matmul(x, identity) == x, +bias, relu (全て正値のため不変)
    let expected_y = [11.0, 22.0, 13.0, 24.0];
    let sq_sum: f32 = expected_y.iter().map(|v| v * v).sum();
    let expected_loss = sq_sum / 4.0;
    assert_eq!(loss.to_tensor().get(&[]).unwrap(), expected_loss);
}

/// 2. 各 forward 演算（matmul/add/mul/relu/exp/tanh/sum/max）の出力値が
///    期待値と一致することを確認する。
#[test]
fn forward_values_match_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    let mm = a.matmul(&b).unwrap();
    // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(
        mm.to_tensor().get(&[0, 0]).unwrap(),
        19.0,
        "matmul(0,0) 不一致"
    );
    assert_eq!(mm.to_tensor().get(&[1, 1]).unwrap(), 50.0);

    let add = a.add(&b).unwrap();
    assert_eq!(add.to_tensor().get(&[0, 0]).unwrap(), 6.0);

    let mul = a.mul(&b).unwrap();
    assert_eq!(mul.to_tensor().get(&[0, 0]).unwrap(), 5.0);
    assert_eq!(mul.to_tensor().get(&[1, 1]).unwrap(), 32.0);

    let neg = tape.var(&t(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]));
    let relu = neg.relu();
    assert_eq!(relu.to_tensor().get(&[0, 0]).unwrap(), 0.0);
    assert_eq!(relu.to_tensor().get(&[0, 1]).unwrap(), 2.0);

    let zero = tape.var(&t(vec![0.0], &[1]));
    let exp = zero.exp();
    assert_eq!(exp.to_tensor().get(&[0]).unwrap(), 1.0);

    let tanh0 = zero.tanh();
    assert_eq!(tanh0.to_tensor().get(&[0]).unwrap(), 0.0);

    let sigmoid0 = zero.sigmoid();
    assert_eq!(sigmoid0.to_tensor().get(&[0]).unwrap(), 0.5);

    let s = a.sum(None).unwrap();
    assert_eq!(s.to_tensor().get(&[]).unwrap(), 10.0);
    let s_axis0 = a.sum(Some(0)).unwrap();
    assert_eq!(s_axis0.to_tensor().shape(), &[2]);
    assert_eq!(s_axis0.to_tensor().get(&[0]).unwrap(), 4.0); // 1+3
    assert_eq!(s_axis0.to_tensor().get(&[1]).unwrap(), 6.0); // 2+4

    let m = a.max(None).unwrap();
    assert_eq!(m.to_tensor().get(&[]).unwrap(), 4.0);
}

/// 3. shape 不整合（matmul の内側次元不一致・mse_loss の shape 不一致・
///    sum の dim 範囲外）が `AutodiffError::Shape(..)` を返すことを検証する。
#[test]
fn shape_mismatches_return_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());

    let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let bad_rhs = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
    let err = a.matmul(&bad_rhs).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));

    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = pred.mse_loss(&target).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));

    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let err = x.sum(Some(5)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

/// 4. 別 `Tape` 由来の `Var` を二項演算に渡すと `TapeMismatch` を返す
///    ことを検証する（クロステープ安全性。`docs/public-api-design.md` §3.1）。
#[test]
fn cross_tape_operations_return_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());

    let a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape_b.var(&t(vec![1.0, 2.0], &[2]));

    let err = a.add(&b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));

    let err = a.mul(&b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));

    let err = a.mse_loss(&b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));

    let ma = tape_a.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let mb = tape_b.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let err = ma.matmul(&mb).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

/// 5. `to_tensor()` 呼び出し直後にノード追加演算を呼んでも panic しない
///    ことを検証する（`RefCell` 借用モデルの回帰テスト）。
#[test]
fn to_tensor_does_not_hold_borrow_across_node_addition() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let owned = a.to_tensor();
    assert_eq!(owned.get(&[0]).unwrap(), 1.0);
    // to_tensor() は Ref を持ち越さないため、直後のノード追加演算
    // （relu が RefCell::borrow_mut を呼ぶ）が panic しないことを確認する。
    let r = a.relu();
    assert_eq!(r.to_tensor().get(&[0]).unwrap(), 1.0);

    // value() 経由でも、一時 Ref をその式文の終わりで解放していれば
    // 後続のノード追加演算が panic しないことを確認する。
    let val_sum: f32 = a.value().get(&[0]).unwrap() + a.value().get(&[1]).unwrap();
    assert_eq!(val_sum, 3.0);
    let r2 = a.exp();
    assert!(r2.to_tensor().get(&[0]).unwrap() > 0.0);
}

/// 6b. `Var::sigmoid`（TASK-9.1b・#92）がテープへノードを 1 個追記する
///     ことを検証する（受け入れ条件「forward 実行時にテープへ演算が
///     記録される」の Sigmoid 個別確認）。
#[test]
fn sigmoid_records_single_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -1.0], &[2]));
    let before = tape.len();
    let _ = x.sigmoid();
    assert_eq!(tape.len(), before + 1);
}

/// 6. ブロードキャスト付き `add`（bias 加算 `[N,M] + [M]`）の forward 値と
///    記録を検証する。
#[test]
fn broadcast_add_bias_over_matrix() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let bias = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let before = tape.len();
    let y = x.add(&bias).unwrap();
    assert_eq!(tape.len(), before + 1);
    let out = y.to_tensor();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.get(&[0, 0]).unwrap(), 11.0);
    assert_eq!(out.get(&[0, 1]).unwrap(), 22.0);
    assert_eq!(out.get(&[0, 2]).unwrap(), 33.0);
    assert_eq!(out.get(&[1, 0]).unwrap(), 14.0);
    assert_eq!(out.get(&[1, 1]).unwrap(), 25.0);
    assert_eq!(out.get(&[1, 2]).unwrap(), 36.0);
}

/// 7. `Tape::default()`（codex-review 第 19〜21 波・PR #403 の P1 是正で
///    追加した compat 経路）が `Tape::new_with_ops(default_ops::naive_ops())`
///    と同じ挙動（forward 演算が記録・実行できる）を持つこと。
///
///    **TASK-9.4（#411）での位置づけ**: このテストはもともと
///    `crates/autodiff/tests/compat_sequential.rs`（`autodiff::compat`
///    経由の統合テスト）にあったが、対象は `compat::Sequential`/
///    `compat::array` ではなく `Tape::default()` 単体の挙動であるため、
///    compat 層が `facade::compat` へ移設された後も本ファイル（`autodiff`
///    の公開 API のみを使う `Tape` 系テスト）に残置する。入力生成は
///    `compat::array`（移設先: `facade::compat::array`）ではなく
///    `tensor_core::Tensor::new`（`autodiff` の直接の依存先）を直接使う
///    よう置き換えた。
#[test]
fn tape_default_records_and_evaluates_ops() {
    let tape = Tape::default();
    let a = tape.var(&t(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]));
    let c = a.add(&b).unwrap();

    assert_eq!(dense_vec(&c.to_tensor()), vec![6.0, 8.0, 10.0, 12.0]);

    // `sum`（非 elementwise・`BackendOps::sum` 経由の即時実行経路。
    // `default_ops::NaiveOps::sum` の `#[cfg(test)]` 解除を検証する）。
    let total = c.sum(None).unwrap();
    assert_eq!(dense_vec(&total.to_tensor()), vec![36.0]);
}

/// テスト専用の連続化ヘルパー（`tensor_core::Tensor` の `pub` API のみを
/// 使用。`crates/facade/src/compat/array.rs` のテストヘルパーと同型）。
fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}
