//! `Var::cosine_similarity`／`Var::pairwise_distance`（イシュー #3821・親 #3813）の facade 到達性テスト。
//!
//! `fandhe_ai::{tape, Var, Tensor, AutodiffError}` だけで呼べること、手計算できる値（スタブでは通らない値）、
//! エラー系を確かめる。自由関数モジュール `similarity_ops` は facade へ再エクスポートされない
//! （`api_surface.rs` の `facade_does_not_reexport_or_declare_similarity_ops`）。

use fandhe_ai::{AutodiffError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致")
}

#[test]
fn cosine_similarity_hand_computed_via_facade() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![3.0, 4.0, 1.0, 0.0], &[2, 2]));
    let b = tape.var(&t(vec![4.0, 3.0, 0.0, 2.0], &[2, 2]));
    let s = a.cosine_similarity(&b, 1, 1e-8).unwrap();
    let got = s.value().host_slice().to_vec();
    assert!((got[0] - 0.96).abs() < 1e-6, "{got:?}");
    assert_eq!(got[1], 0.0);
    let loss = s.sum(None).unwrap();
    let g = tape.backward(&loss).unwrap();
    let da = g.get(&a).unwrap().unwrap().host_slice().to_vec();
    // 行 0: b/25 - 0.96 * a/25
    assert!(
        (da[0] - (4.0 / 25.0 - 0.96 * 3.0 / 25.0)).abs() < 1e-6,
        "{da:?}"
    );
}

#[test]
fn pairwise_distance_hand_computed_via_facade() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![3.0, 4.0, 1.0, 1.0], &[2, 2]));
    let b = tape.var(&t(vec![0.0, 0.0, 1.0, 1.0], &[2, 2]));
    let d = a.pairwise_distance(&b, 2.0, 0.0, false).unwrap();
    assert_eq!(d.value().host_slice().to_vec(), vec![5.0, 0.0]);
    let dk = a.pairwise_distance(&b, 2.0, 0.0, true).unwrap();
    assert_eq!(dk.value().shape(), &[2, 1]);
    let g = tape.backward(&d.sum(None).unwrap()).unwrap();
    let da = g.get(&a).unwrap().unwrap().host_slice().to_vec();
    assert!(
        (da[0] - 0.6).abs() < 1e-7 && (da[1] - 0.8).abs() < 1e-7,
        "{da:?}"
    );
}

#[test]
fn invalid_arguments_are_typed_errors_via_facade() {
    let tape = fandhe_ai::tape();
    let other = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let c = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let foreign = other.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert!(matches!(
        a.cosine_similarity(&foreign, 1, 1e-8),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(a.cosine_similarity(&c, 0, 1e-8).is_err());
    assert!(a.cosine_similarity(&a, 2, 1e-8).is_err());
    assert!(matches!(
        a.cosine_similarity(&a, 1, -1.0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        a.pairwise_distance(&a, 0.5, 0.0, false),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        a.pairwise_distance(&a, 2.0, f32::NAN, false),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
