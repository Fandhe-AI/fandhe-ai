//! `Var::batch_norm` 系の rank 5 受理（イシュー #3819・承認 #3768）を facade 到達経路
//! （`fandhe_ai::Var`・CPU 既定 tape）で固定する正のプローブ。
//!
//! 公開名を増やさない挙動拡張のため `api_surface.rs` のガードは無く、本ファイルが
//! 受理範囲のすり替え（rank 4 へ戻す・rank 6 以上へ広げる）を検出する。
//! `add_batch_norm3d` の正ガードは #3820。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn data() -> Vec<f32> {
    (0..48)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25)
        .collect()
}

fn forward(shape: &[usize]) -> (Tensor<f32>, Vec<u32>) {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(data(), shape).unwrap());
    let w = tape.var(&Tensor::new(vec![1.5f32, -0.8, 1.0], &[3]).unwrap());
    let b = tape.var(&Tensor::new(vec![0.1f32, -0.2, 0.3], &[3]).unwrap());
    let y = x.batch_norm(Some(&w), Some(&b), 1e-5).unwrap();
    let g = tape.backward(&y.sum(None).unwrap()).unwrap();
    let dw = g.get(&w).unwrap().unwrap().clone();
    (y.to_tensor(), bits(&dw))
}

#[test]
fn rank5_is_accepted_and_bit_identical_to_rank3_reshape() {
    let (y5, dw5) = forward(&[2, 3, 2, 2, 2]);
    let (y3, dw3) = forward(&[2, 3, 8]);
    assert_eq!(y5.shape(), &[2, 3, 2, 2, 2]);
    assert_eq!(bits(&y5), bits(&y3));
    assert_eq!(dw5, dw3);
}

#[test]
fn rank6_is_rejected_with_expected_5() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![0.5f32; 64], &[2; 6]).unwrap());
    let err = x.batch_norm(None, None, 1e-5).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 5,
            actual: 6
        })
    ));
}

#[test]
fn rank1_keeps_expected_2() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![0.5f32; 4], &[4]).unwrap());
    let err = x.batch_norm(None, None, 1e-5).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: 1
        })
    ));
}
