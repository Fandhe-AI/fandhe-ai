//! `fandhe_ai_autodiff::bool_ops`（イシュー #2141）が PyTorch の
//! 文書化済み意味論（`torch.where`／`torch.logical_and`／
//! `torch.logical_not`／`torch.logical_or`）と一致することを、CI では
//! torch を実行できないため**手で導いた期待値**で固定する example
//! テスト（実装計画 §5.4）。
//!
//! 期待値の導出根拠は各テストのコメントに記す。参考として実行可能な
//! torch スニペットも併記するが、実際の torch 出力で検証済みとは
//! 主張しない（CI では未実行）。

use fandhe_ai::{Var, tape};
use fandhe_ai_autodiff::bool_ops::{gt_bool, lt_bool, masked_select};
use fandhe_ai_tensor_core::Tensor;

/// `torch.where(x > 0, x, torch.zeros_like(x))` 相当:
/// ```python
/// x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
/// y = torch.where(x > 0, x, torch.zeros_like(x))
/// y.sum().backward()
/// # y  == [0, 0, 0, 1, 2]
/// # x.grad == [0, 0, 0, 1, 1]（x > 0 の位置のみ 1。`torch.where` の
/// # 勾配は選択された分岐のみへ流れ、条件〈bool マスク〉自体は勾配を
/// # 持たない）
/// ```
///
/// 本実装（`gt_bool` の bool 出力を `Var::where_cond` の条件へ接続）が
/// 同じ forward・backward 挙動を持つことを確認する。
#[test]
fn where_cond_with_gt_bool_matches_torch_where_semantics() {
    let tape = tape();
    let x = tape.var(&Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap());
    let zero = tape.var(&Tensor::new(vec![0.0f32; 5], &[5]).unwrap());

    let mask = gt_bool(&x, &zero).expect("gt_bool は常に成功する");
    assert_eq!(
        mask.contiguous().host_slice().into_owned(),
        vec![false, false, false, true, true]
    );

    let y = Var::where_cond(&mask, &x, &zero).expect("where_cond は常に成功する");
    assert_eq!(
        y.to_tensor().contiguous().host_slice().into_owned(),
        vec![0.0, 0.0, 0.0, 1.0, 2.0]
    );

    let loss = y.sum(None).expect("sum は常に成功する");
    let grads = tape.backward(&loss).expect("backward は常に成功する");
    let dx = grads
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("loss から到達可能");
    assert_eq!(
        dx.contiguous().host_slice().into_owned(),
        vec![0.0, 0.0, 0.0, 1.0, 1.0]
    );
}

/// `torch.logical_and(x > a, x < b)` で範囲マスクを作り
/// `torch.masked_select` で抽出する相当:
/// ```python
/// x = torch.tensor([-1.0, 0.5, 1.5, 2.5, 3.5])
/// a, b = 0.0, 3.0
/// mask = torch.logical_and(x > a, x < b)
/// # mask == [False, True, True, True, False]
/// selected = torch.masked_select(x, mask)
/// # selected == [0.5, 1.5, 2.5]
/// ```
#[test]
fn logical_and_range_mask_with_masked_select_matches_torch_semantics() {
    let tape = tape();
    let x = tape.var(&Tensor::new(vec![-1.0, 0.5, 1.5, 2.5, 3.5], &[5]).unwrap());
    let a = tape.var(&Tensor::new(vec![0.0f32; 5], &[5]).unwrap());
    let b = tape.var(&Tensor::new(vec![3.0f32; 5], &[5]).unwrap());

    let gt_a = gt_bool(&x, &a).expect("gt_bool は常に成功する");
    let lt_b = lt_bool(&x, &b).expect("lt_bool は常に成功する");
    let mask = fandhe_ai_autodiff::bool_ops::logical_and(&gt_a, &lt_b)
        .expect("logical_and は常に成功する");
    assert_eq!(
        mask.contiguous().host_slice().into_owned(),
        vec![false, true, true, true, false]
    );

    let selected = masked_select(&x, &mask).expect("masked_select は常に成功する");
    assert_eq!(
        selected.contiguous().host_slice().into_owned(),
        vec![0.5, 1.5, 2.5]
    );
}

/// `torch.logical_not`／`torch.logical_or` の真理値表と `NaN` の扱い:
/// ```python
/// x = torch.tensor([float('nan'), 1.0])
/// gt0 = x > 0
/// # gt0 == [False, True]（NaN との比較は常に False。PyTorch も IEEE 754
/// # 準拠で NaN を含む比較は全て False）
/// not_gt0 = torch.logical_not(gt0)
/// # not_gt0 == [True, False]（NaN の位置は gt0 が False のため not で True）
/// either = torch.logical_or(gt0, not_gt0)
/// # either == [True, True]（排中律。gt0 と not_gt0 は互いに否定のため
/// # logical_or は常に True になる）
/// ```
#[test]
fn logical_not_or_truth_table_with_nan_matches_torch_semantics() {
    let tape = tape();
    let x = tape.var(&Tensor::new(vec![f32::NAN, 1.0], &[2]).unwrap());
    let zero = tape.var(&Tensor::new(vec![0.0f32; 2], &[2]).unwrap());

    let gt0 = gt_bool(&x, &zero).expect("gt_bool は常に成功する");
    assert_eq!(
        gt0.contiguous().host_slice().into_owned(),
        vec![false, true],
        "NaN を含む比較は IEEE 754 準拠で常に false"
    );

    let not_gt0 =
        fandhe_ai_autodiff::bool_ops::logical_not(&gt0).expect("logical_not は常に成功する");
    assert_eq!(
        not_gt0.contiguous().host_slice().into_owned(),
        vec![true, false]
    );

    let either = fandhe_ai_autodiff::bool_ops::logical_or(&gt0, &not_gt0)
        .expect("logical_or は常に成功する");
    assert_eq!(
        either.contiguous().host_slice().into_owned(),
        vec![true, true],
        "排中律: gt0 と logical_not(gt0) の logical_or は常に true"
    );
}
