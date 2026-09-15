//! `compat::Sequential` の train／eval モード（[`Sequential::set_training`]／
//! [`Sequential::train`]／[`Sequential::eval`]／[`Sequential::training`]）・
//! `named_parameters`（イシュー #1758）の公開 API 契約テストを、
//! `fandhe_ai`（facade）経由でのみ検証する。
//!
//! 本 issue は数値演算を追加しない（新規 `Op`／`BackendOps`／VJP／GPU
//! カーネルはいずれも該当なし）ため、ここでの「parity」に相当する
//! 検証は CPU 上の bit 完全一致（モード切替の前後で `predict`／
//! `forward`／`backward` の出力・勾配が変化しないこと）に限定する。
//! CUDA／Metal への申し送りは不要（数値経路に一切触れないため）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Tensor, tape};

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)
        .unwrap()
}

/// 初期値 `training()==true`・`eval()`→`false`・`train()`→`true`・
/// `set_training(b)` 往復。
#[test]
fn training_mode_toggle() {
    let mut model = build_model();
    assert!(model.training(), "既定は PyTorch の初期値と揃え true");

    model.eval();
    assert!(!model.training());

    model.train();
    assert!(model.training());

    model.set_training(false);
    assert!(!model.training());
    model.set_training(true);
    assert!(model.training());
}

/// index 接頭辞契約: `Linear→ReLU→Linear` で
/// `["0.weight","0.bias","2.weight","2.bias"]`（活性化を含む index。
/// PyTorch `nn.Sequential` と同じ規約）。
#[test]
fn named_parameters_index_prefix_contract() {
    let model = build_model();
    let params = model.named_parameters();
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["0.weight", "0.bias", "2.weight", "2.bias"]);
}

/// 順序契約: `named_parameters()` の tensor 列と
/// `trainable_parameters()` を要素ごとに `std::ptr::eq` で一致確認する
/// （`Sequential::named_parameters` doc「順序契約」参照）。
#[test]
fn named_parameters_order_matches_trainable_parameters() {
    let model = build_model();
    let named = model.named_parameters();
    let trainable = model.trainable_parameters();
    assert_eq!(named.len(), trainable.len());
    for ((_, named_tensor), trainable_tensor) in named.iter().zip(trainable.iter()) {
        assert!(std::ptr::eq(*named_tensor, *trainable_tensor));
    }
}

/// bit 同一（parity の代替）: 同一入力に対し `eval()` 前後・`train()`
/// 前後で `predict`（tape 不要経路）の出力が bit 完全一致する
/// （本 issue はモード依存層を一切導入しないため、`set_training` は
/// 数値経路に影響しない契約を固定する）。
#[test]
fn predict_output_unaffected_by_training_mode() {
    let mut model = build_model();
    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();

    let out_train = model.predict(&x).unwrap();

    model.eval();
    let out_eval = model.predict(&x).unwrap();

    model.train();
    let out_train_again = model.predict(&x).unwrap();

    assert_eq!(out_train.as_slice(), out_eval.as_slice());
    assert_eq!(out_train.as_slice(), out_train_again.as_slice());
}

/// bit 同一（parity の代替）: `bind → forward → mse_loss → backward`
/// の勾配が `eval()` 前後で bit 完全一致する。
#[test]
fn backward_gradients_unaffected_by_training_mode() {
    let mut model = build_model();
    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();
    let y = Tensor::new(vec![0.0_f32, 1.0, 1.0, 0.0], &[2, 2]).unwrap();

    let grads_train = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let yv = t.var(&y);
        let pred = bound.forward(&t, &xv).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let grads = t.backward(&loss).unwrap();
        let refs = bound.trainable_grads(&grads).unwrap();
        refs.into_iter().cloned().collect::<Vec<_>>()
    };

    model.eval();

    let grads_eval = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let yv = t.var(&y);
        let pred = bound.forward(&t, &xv).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let grads = t.backward(&loss).unwrap();
        let refs = bound.trainable_grads(&grads).unwrap();
        refs.into_iter().cloned().collect::<Vec<_>>()
    };

    assert_eq!(grads_train.len(), grads_eval.len());
    for (a, b) in grads_train.iter().zip(grads_eval.iter()) {
        assert_eq!(a.as_slice(), b.as_slice());
    }
}
