//! `Sequential::add_hardshrink`／`add_softshrink`（イシュー #3817・親 #3813）の層テスト。
//! forward（predict）の手計算値、`lambd` の拒否、学習経路（`fit`）が通ること、
//! 層経路が `Var` 演算と bit 一致することを確かめる。

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, Tensor};

fn x() -> Tensor<f32> {
    Tensor::new(vec![-1.0f32, -0.5, 0.25, 0.5, 0.75, 2.0], &[2, 3]).unwrap()
}

fn vals(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().host_slice().to_vec()
}

#[test]
fn hardshrink_layer_forward_matches_hand_computation() {
    let m = Sequential::new().add_hardshrink(0.5).unwrap();
    let y = m.predict(&x()).unwrap();
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(vals(&y), vec![-1.0, 0.0, 0.0, 0.0, 0.75, 2.0]);
}

#[test]
fn softshrink_layer_forward_matches_hand_computation() {
    let m = Sequential::new().add_softshrink(0.5).unwrap();
    let y = m.predict(&x()).unwrap();
    assert_eq!(vals(&y), vec![-0.5, 0.0, 0.0, 0.0, 0.25, 1.5]);
}

#[test]
fn invalid_lambd_is_rejected_at_construction() {
    for bad in [-0.1f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            Sequential::new().add_hardshrink(bad),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            Sequential::new().add_softshrink(bad),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    // -0.0 と 0 は受理する（恒等写像）。
    let m = Sequential::new().add_hardshrink(-0.0).unwrap();
    assert_eq!(vals(&m.predict(&x()).unwrap()), vals(&x()));
}

#[test]
fn layer_output_is_bit_identical_to_var_op() {
    let input = x();
    let tape = fandhe_ai::tape();
    let v = tape.var(&input);
    let hard = v.hardshrink(0.3).unwrap().to_tensor();
    let soft = v.softshrink(0.3).unwrap().to_tensor();
    let m_h = Sequential::new().add_hardshrink(0.3).unwrap();
    let m_s = Sequential::new().add_softshrink(0.3).unwrap();
    let bits = |t: &Tensor<f32>| vals(t).iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&m_h.predict(&input).unwrap()), bits(&hard));
    assert_eq!(bits(&m_s.predict(&input).unwrap()), bits(&soft));
}

#[test]
fn fit_runs_through_shrink_layers() {
    let xs: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.37).sin()).collect();
    let ys: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.11).cos()).collect();
    let x = Tensor::new(xs, &[8, 3]).unwrap();
    let y = Tensor::new(ys, &[8, 2]).unwrap();
    let mut model = Sequential::new()
        .add_linear(3, 4, 11)
        .unwrap()
        .add_hardshrink(0.1)
        .unwrap()
        .add_linear(4, 4, 12)
        .unwrap()
        .add_softshrink(0.05)
        .unwrap()
        .add_linear(4, 2, 13)
        .unwrap();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let h = model.fit(&x, &y, FitConfig::new(3, 8)).unwrap();
    assert_eq!(h.loss.len(), 3);
    assert!(h.loss.iter().all(|l| l.is_finite()));
}
