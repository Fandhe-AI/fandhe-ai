//! `fandhe_ai::{logical_and, logical_or, logical_not}`（イシュー #2596・
//! 親 #2594・ルート #2499 コメント承認。`docs/autodiff-bool-ops-exposure-
//! decision.md` §6.2 案 B-1）が facade（`fandhe_ai`）だけで到達でき、内部
//! 実装（`fandhe_ai_autodiff::bool_ops`。#2141）と同じ意味論を持つことを
//! 固定する利用テスト。
//!
//! `fandhe_ai_autodiff` を import しない（facade 経由の到達性確認を兼ねる）。
//! 委譲のみでカーネルを新設せずホスト計算のため CPU 既定バックエンドで CI
//! 実行でき、`#[ignore]` の実機テストは不要（内部入口の GPU 側検証は
//! `bool_ops_backend_parity.rs` が引き続き扱う）。

use fandhe_ai::{AutodiffError, ShapeError, Tensor, logical_and, logical_not, logical_or, tape};

type Binary = fn(&Tensor<bool>, &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError>;
type Unary = fn(&Tensor<bool>) -> Result<Tensor<bool>, AutodiffError>;

fn host_bool(t: &Tensor<bool>) -> Vec<bool> {
    t.contiguous().host_slice().into_owned()
}

fn bools(v: Vec<bool>, shape: &[usize]) -> Tensor<bool> {
    Tensor::new(v, shape).expect("valid shape")
}

#[test]
fn signatures_are_pinned() {
    let _: Binary = logical_and;
    let _: Binary = logical_or;
    let _: Unary = logical_not;
}

#[test]
fn truth_tables() {
    let a = bools(vec![true, true, false, false], &[4]);
    let b = bools(vec![true, false, true, false], &[4]);
    assert_eq!(
        host_bool(&logical_and(&a, &b).unwrap()),
        vec![true, false, false, false]
    );
    assert_eq!(
        host_bool(&logical_or(&a, &b).unwrap()),
        vec![true, true, true, false]
    );
    let n = logical_not(&a).unwrap();
    assert_eq!(host_bool(&n), vec![false, false, true, true]);
    assert_eq!(host_bool(&logical_not(&n).unwrap()), host_bool(&a));
}

#[test]
fn broadcasts_and_accepts_non_contiguous_inputs() {
    let col = bools(vec![true, false], &[2, 1]);
    let row = bools(vec![true, true, false], &[1, 3]);
    let m = logical_and(&col, &row).unwrap();
    assert_eq!(m.shape(), &[2, 3]);
    assert_eq!(host_bool(&m), vec![true, true, false, false, false, false]);

    let base = bools(vec![true, false, false, true, true, false], &[2, 3]);
    let t = base.transpose(0, 1).expect("transpose");
    let n = logical_not(&t).unwrap();
    assert_eq!(n.shape(), &[3, 2]);
    assert_eq!(host_bool(&n), vec![false, false, true, false, true, true]);
}

#[test]
fn incompatible_shapes_return_shape_error() {
    let a = bools(vec![true, false], &[2]);
    let b = bools(vec![true, false, true], &[3]);
    assert!(matches!(logical_and(&a, &b), Err(AutodiffError::Shape(_))));
    assert!(matches!(logical_or(&a, &b), Err(AutodiffError::Shape(_))));
}

/// 巨大 broadcast view は確保前の要素数検査で `ElementCountOverflow` となり、
/// panic・巨大確保をしない（委譲が内部の事前検査を迂回していない証明）。
#[test]
fn huge_broadcast_view_is_rejected_before_allocation() {
    let one = bools(vec![true], &[1]);
    let huge_len = (isize::MAX as usize) / std::mem::size_of::<bool>() + 10;
    let huge = one
        .broadcast_to(&[huge_len])
        .expect("broadcast view は確保しない");
    assert!(matches!(
        logical_not(&huge),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
    assert!(matches!(
        logical_and(&huge, &one),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

/// facade だけで比較マスクの合成から `masked_select` まで通る利用例。
#[test]
fn composes_with_var_comparisons_and_masked_select() {
    let t = tape();
    let x = t.var(&Tensor::new(vec![1.0, 5.0, 9.0, 4.0], &[4]).unwrap());
    let lo = x
        .gt_bool(&t.var(&Tensor::new(vec![2.0; 4], &[4]).unwrap()))
        .unwrap();
    let hi = x
        .lt_bool(&t.var(&Tensor::new(vec![8.0; 4], &[4]).unwrap()))
        .unwrap();
    let mask = logical_and(&lo, &hi).unwrap();
    assert_eq!(host_bool(&mask), vec![false, true, false, true]);
    let picked = x.masked_select(&mask).unwrap();
    assert_eq!(
        picked.contiguous().host_slice().into_owned(),
        vec![5.0f32, 4.0]
    );
}
