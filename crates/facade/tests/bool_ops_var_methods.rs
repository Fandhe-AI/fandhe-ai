//! `Var::gt_bool`／`ge_bool`／`lt_bool`／`le_bool`／`eq_bool`／`ne_bool`／
//! `masked_select`（イシュー #2510。ルート #2499 一括承認）が facade
//! （`fandhe_ai`）だけで到達でき、既存の自由関数版（#2141）と同じ意味論を
//! 持つことを固定する利用テスト。
//!
//! `fandhe_ai_autodiff` を import しない（facade 経由の到達性確認を兼ねる）。
//! 委譲のみでカーネルを新設しないため CPU 既定バックエンドで CI 実行でき、
//! `#[ignore]` の実機テストは不要（GPU 側は `bool_ops_backend_parity.rs`
//! が引き続き扱う）。

use fandhe_ai::{AutodiffError, Tensor, Var, tape};

/// シグネチャ固定（コンパイル時）。`'t` は impl 側の early-bound lifetime
/// のため generic 補助関数内で関数ポインタへ強制変換する。
#[allow(dead_code)]
fn pin_signatures<'t>(_witness: &Var<'t>) {
    type Cmp<'t> = fn(&Var<'t>, &Var<'t>) -> Result<Tensor<bool>, AutodiffError>;
    let _: Cmp<'t> = Var::<'t>::gt_bool;
    let _: Cmp<'t> = Var::<'t>::ge_bool;
    let _: Cmp<'t> = Var::<'t>::lt_bool;
    let _: Cmp<'t> = Var::<'t>::le_bool;
    let _: Cmp<'t> = Var::<'t>::eq_bool;
    let _: Cmp<'t> = Var::<'t>::ne_bool;
    let _: fn(&Var<'t>, &Tensor<bool>) -> Result<Tensor<f32>, AutodiffError> =
        Var::<'t>::masked_select;
}

fn host_bool(t: &Tensor<bool>) -> Vec<bool> {
    t.contiguous().host_slice().into_owned()
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().host_slice().into_owned()
}

#[test]
fn comparisons_follow_ieee754_truth_table() {
    let tape = tape();
    let a = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, f32::NAN, -0.0, f32::NAN], &[6]).unwrap());
    let b = tape.var(&Tensor::new(vec![2.0, 2.0, 1.0, 1.0, 0.0, f32::NAN], &[6]).unwrap());

    assert_eq!(
        host_bool(&a.gt_bool(&b).unwrap()),
        vec![false, false, true, false, false, false]
    );
    assert_eq!(
        host_bool(&a.ge_bool(&b).unwrap()),
        vec![false, true, true, false, true, false]
    );
    assert_eq!(
        host_bool(&a.lt_bool(&b).unwrap()),
        vec![true, false, false, false, false, false]
    );
    assert_eq!(
        host_bool(&a.le_bool(&b).unwrap()),
        vec![true, true, false, false, true, false]
    );
    assert_eq!(
        host_bool(&a.eq_bool(&b).unwrap()),
        vec![false, true, false, false, true, false]
    );
    assert_eq!(
        host_bool(&a.ne_bool(&b).unwrap()),
        vec![true, false, true, true, false, true]
    );
}

#[test]
fn comparison_broadcasts_like_numpy() {
    let tape = tape();
    let a = tape.var(&Tensor::new(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]).unwrap());
    let b = tape.var(&Tensor::new(vec![2.0, 4.0, 3.0], &[3]).unwrap());
    let m = a.gt_bool(&b).unwrap();
    assert_eq!(m.shape(), &[2, 3]);
    assert_eq!(host_bool(&m), vec![false, true, false, true, false, true]);
}

#[test]
fn gt_bool_connects_to_where_cond_with_torch_semantics() {
    let tape = tape();
    let x = tape.var(&Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap());
    let zero = tape.var(&Tensor::new(vec![0.0f32; 5], &[5]).unwrap());
    let mask = x.gt_bool(&zero).unwrap();
    let y = Var::where_cond(&mask, &x, &zero).unwrap();
    assert_eq!(host_f32(&y.to_tensor()), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(host_f32(dx), vec![0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn masked_select_is_row_major_with_broadcast_mask_and_keeps_nan_bits() {
    let tape = tape();
    let x = tape.var(&Tensor::new(vec![1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap());
    let mask = Tensor::new(vec![true, true, false], &[3]).unwrap();
    let sel = x.masked_select(&mask).unwrap();
    assert_eq!(sel.shape(), &[4]);
    let got = host_f32(&sel);
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1].to_bits(), f32::NAN.to_bits());
    assert_eq!(&got[2..], &[4.0, 5.0]);
}

#[test]
fn masked_select_all_false_returns_empty_rank1() {
    let tape = tape();
    let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());
    let mask = Tensor::new(vec![false; 3], &[3]).unwrap();
    let sel = x.masked_select(&mask).unwrap();
    assert_eq!(sel.shape(), &[0]);
}

#[test]
fn error_paths_are_reported() {
    let t1 = tape();
    let t2 = tape();
    let a = t1.var(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
    let b = t2.var(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
    assert!(matches!(a.gt_bool(&b), Err(AutodiffError::TapeMismatch)));

    let c = t1.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());
    assert!(a.lt_bool(&c).is_err());
    let bad_mask = Tensor::new(vec![true; 3], &[3]).unwrap();
    assert!(a.masked_select(&bad_mask).is_err());
}
