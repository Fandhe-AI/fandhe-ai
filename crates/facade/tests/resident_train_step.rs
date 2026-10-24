//! イシュー #3254: 常駐学習ステップの公開面（P1〜P7）をクレート外から公開 API だけで確かめる
//! CPU テスト（CI で実行）。
//!
//! # 役割
//! `facade` は唯一のサポートされる公開面のため、`Sequential::train_step_resident`・
//! `ResidentLoss::read`・`Tape::wait_device_param_store` を内部 API なしで使えることと、
//! 範囲外・古い世代・別 store の型付きエラー契約を固定する。数値の bit 一致（Tape 経路との比較）は
//! クレート内テスト（`compat/sequential/train_step_chain_tests_3251.rs`）が担い、実機の同期回数は
//! `#[ignore]` の実機テストが担う。決定記録は `docs/compat-resident-train-step-decision.md` §4・§14。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, BackendError, SgdConfig, Tensor};

fn model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, 11)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, 12)
        .unwrap()
}

fn data(len: usize, seed: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
}

#[test]
fn one_step_loss_is_finite_and_wait_is_ok() {
    let tape = fandhe_ai::tape();
    let model = model();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y = tape
        .upload_resident(&store, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let loss = model
        .train_step_resident(&mut store, &x, &y, &SgdConfig::new(0.05))
        .unwrap();
    tape.wait_device_param_store(&store).unwrap();
    assert!(loss.read(&mut store).unwrap().is_finite());
}

#[test]
fn stale_loss_handle_is_invalid_argument_and_new_handle_reads() {
    let tape = fandhe_ai::tape();
    let model = model();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y = tape
        .upload_resident(&store, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let config = SgdConfig::new(0.05);
    let first = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    // 2 回目の step の後、`train_step_resident` 内部の tape はすべて drop 済み。
    let second = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    assert!(matches!(
        first.read(&mut store),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    // 拒否の後も store は使い続けられる。
    assert!(second.read(&mut store).unwrap().is_finite());
    let third = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    assert!(third.read(&mut store).unwrap().is_finite());
}

#[test]
fn handles_of_another_store_are_invalid_argument() {
    let tape = fandhe_ai::tape();
    let model = model();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let mut other = model.init_device_param_store(&tape).unwrap();
    let x_other = tape
        .upload_resident(&other, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y_other = tape
        .upload_resident(&other, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let config = SgdConfig::new(0.05);
    // 別 store の入力。
    let r = model.train_step_resident(&mut store, &x_other, &y_other, &config);
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    // 別 store の loss。
    let loss_other = model
        .train_step_resident(&mut other, &x_other, &y_other, &config)
        .unwrap();
    assert!(matches!(
        loss_other.read(&mut store),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    assert!(loss_other.read(&mut other).unwrap().is_finite());
}

#[test]
fn out_of_scope_model_is_a_typed_error_and_store_is_reusable() {
    let tape = fandhe_ai::tape();
    let bad = Sequential::new()
        .add_linear(4, 8, 11)
        .unwrap()
        .add_sigmoid()
        .add_linear(8, 2, 12)
        .unwrap();
    let mut store = bad.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y = tape
        .upload_resident(&store, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let r = bad.train_step_resident(&mut store, &x, &y, &SgdConfig::new(0.05));
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::Unsupported(_)))
    ));
    // 拒否は store を変えず、同じ store で推論・次の入口を呼べる。
    assert!(tape.wait_device_param_store(&store).is_ok());
}
