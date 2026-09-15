//! `nn::Module` の `set_training`／`training`／`named_parameters`
//! （イシュー #1758）の契約テスト。
//!
//! 本 issue は数値演算を追加しない（新規 `Op`／`BackendOps`／VJP／GPU
//! カーネルはいずれも該当なし）ため、ここでの「parity」に相当する
//! 検証は CPU 上の bit 完全一致（モード切替の前後で `forward` の出力・
//! 勾配が変化しないこと）に限定する。CUDA／Metal への申し送りは不要
//! （数値経路に一切触れないため）。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::activation::{Relu, Softmax};
use fandhe_ai_autodiff::nn::{Gru, Linear, Lstm, Module, MultiheadAttention, RmsNorm, Rnn};
use fandhe_ai_tensor_core::Tensor;

/// 既定契約（`Module::training`／`set_training` の trait doc）: 無状態
/// モジュール（`Relu`・`Softmax` 等）は `training()` が常に `true`
/// （`set_training` を呼んでも変化しない）・`named_parameters()` は空。
#[test]
fn stateless_modules_default_contract() {
    let mut relu = Relu;
    assert!(relu.training());
    relu.set_training(false);
    assert!(
        relu.training(),
        "無状態モジュールは set_training を呼んでも training() が変化しない契約"
    );
    assert!(relu.named_parameters().is_empty());

    let mut softmax = Softmax::new(0);
    assert!(softmax.training());
    softmax.set_training(false);
    assert!(softmax.training());
    assert!(softmax.named_parameters().is_empty());
}

/// object safety: `Vec<Box<dyn Module>>` に `Linear` と活性化関数を
/// 混在させ、`set_training`／`named_parameters` を dyn 経由で呼べる
/// ことを確認する（既存 `as_linear`／`as_relu` と同じ dyn 呼び出し
/// パターン）。
#[test]
fn dyn_module_set_training_and_named_parameters() {
    let linear = Linear::new(3, 2, true, 42).expect("valid ctor args");
    let mut layers: Vec<Box<dyn Module>> = vec![Box::new(linear), Box::new(Relu)];

    for layer in &mut layers {
        layer.set_training(false);
    }

    // `Linear::named_parameters()` は weight/bias を返し、`Relu` は空。
    assert_eq!(layers[0].named_parameters().len(), 2);
    assert!(layers[1].named_parameters().is_empty());
}

/// `Linear`（bias あり）の命名契約: `weight` → `bias` の順で、各参照が
/// accessor（`weight()`／`bias()`）と同一ポインタであること。
#[test]
fn linear_named_parameters_with_bias() {
    let linear = Linear::new(3, 2, true, 42).expect("valid ctor args");
    let params = linear.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "weight");
    assert!(std::ptr::eq(params[0].1, linear.weight()));
    assert_eq!(params[1].0, "bias");
    assert!(std::ptr::eq(
        params[1].1,
        linear.bias().expect("bias=true で構築した")
    ));
}

/// `Linear`（bias なし）の命名契約: `weight` のみ。
#[test]
fn linear_named_parameters_without_bias() {
    let linear = Linear::new(3, 2, false, 42).expect("valid ctor args");
    let params = linear.named_parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "weight");
    assert!(linear.bias().is_none());
}

/// `RmsNorm`（affine あり／なし）の命名契約: `weight`（`Some` の場合の
/// み）。
#[test]
fn rms_norm_named_parameters() {
    let with_affine = RmsNorm::new(4, 1e-5).expect("valid ctor args");
    let params = with_affine.named_parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "weight");
    assert!(std::ptr::eq(
        params[0].1,
        with_affine.weight().expect("affine=true で構築した")
    ));

    let without_affine = RmsNorm::without_affine(1e-5).expect("valid ctor args");
    assert!(without_affine.named_parameters().is_empty());
}

/// `LayerNorm`（affine あり／なし）の命名契約: `weight` → `bias`
/// の順（各 `Some` の場合のみ）。
#[test]
fn layer_norm_named_parameters() {
    use fandhe_ai_autodiff::nn::LayerNorm;

    let with_affine = LayerNorm::new(4, 1e-5).expect("valid ctor args");
    let params = with_affine.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "weight");
    assert_eq!(params[1].0, "bias");

    let without_affine = LayerNorm::without_affine(1e-5).expect("valid ctor args");
    assert!(without_affine.named_parameters().is_empty());
}

/// `MultiheadAttention` の命名契約: `q_proj.*` → `k_proj.*` →
/// `v_proj.*` → `out_proj.*`（各 `weight`→`bias`）の 4 層 × 2 =
/// 8 エントリ（bias あり構成）。各参照が対応する accessor
/// （`q_proj().weight()` 等）と同一ポインタであることも確認する。
#[test]
fn multihead_attention_named_parameters() {
    let mha = MultiheadAttention::new(4, 2, true, 7).expect("valid ctor args");
    let params = mha.named_parameters();
    assert_eq!(params.len(), 8);
    let expected_names = [
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
    ];
    for (i, expected) in expected_names.iter().enumerate() {
        assert_eq!(&params[i].0, expected);
    }
    assert!(std::ptr::eq(params[0].1, mha.q_proj().weight()));
    assert!(std::ptr::eq(
        params[1].1,
        mha.q_proj().bias().expect("bias=true")
    ));
    assert!(std::ptr::eq(params[6].1, mha.out_proj().weight()));
}

/// `Rnn`／`Lstm`／`Gru`（bias あり）の命名契約: `cell.weight_ih` →
/// `cell.weight_hh` → `cell.bias_ih` → `cell.bias_hh`。参照が
/// `cell().weight_ih()` 等と同一ポインタであることも確認する。
#[test]
fn rnn_family_named_parameters_with_bias() {
    let rnn = Rnn::new(3, 5, true, 11).expect("valid ctor args");
    let params = rnn.named_parameters();
    assert_eq!(params.len(), 4);
    assert_eq!(params[0].0, "cell.weight_ih");
    assert_eq!(params[1].0, "cell.weight_hh");
    assert_eq!(params[2].0, "cell.bias_ih");
    assert_eq!(params[3].0, "cell.bias_hh");
    assert!(std::ptr::eq(params[0].1, rnn.cell().weight_ih()));
    assert!(std::ptr::eq(
        params[2].1,
        rnn.cell().bias_ih().expect("bias=true")
    ));

    let lstm = Lstm::new(3, 5, true, 12).expect("valid ctor args");
    let lstm_params = lstm.named_parameters();
    assert_eq!(lstm_params.len(), 4);
    assert_eq!(lstm_params[0].0, "cell.weight_ih");
    assert_eq!(lstm_params[3].0, "cell.bias_hh");

    let gru = Gru::new(3, 5, true, 13).expect("valid ctor args");
    let gru_params = gru.named_parameters();
    assert_eq!(gru_params.len(), 4);
    assert_eq!(gru_params[0].0, "cell.weight_ih");
    assert_eq!(gru_params[3].0, "cell.bias_hh");
}

/// `Rnn`（bias なし）の命名契約: `cell.weight_ih` → `cell.weight_hh`
/// のみ（bias 系は列挙されない）。
#[test]
fn rnn_named_parameters_without_bias() {
    let rnn = Rnn::new(3, 5, false, 11).expect("valid ctor args");
    let params = rnn.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "cell.weight_ih");
    assert_eq!(params[1].0, "cell.weight_hh");
}

/// 不変性: `set_training` を切り替えても `Linear::bind(&tape).forward`
/// の出力が bit 完全一致すること（本 issue は数値経路に触れないため
/// の構造的な保証を固定する）。
#[test]
fn set_training_does_not_change_forward_output() {
    let mut linear = Linear::new(3, 2, true, 42).expect("valid ctor args");
    let x = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

    let tape_before = Tape::new_with_ops(common::naive_ops());
    let xv_before = tape_before.var(&x);
    let out_before = linear
        .bind(&tape_before)
        .forward(&xv_before)
        .unwrap()
        .to_tensor();

    linear.set_training(false);
    assert!(
        linear.training(),
        "Linear は無状態のため set_training(false) 後も training()==true"
    );

    let tape_after = Tape::new_with_ops(common::naive_ops());
    let xv_after = tape_after.var(&x);
    let out_after = linear
        .bind(&tape_after)
        .forward(&xv_after)
        .unwrap()
        .to_tensor();

    assert_eq!(out_before.as_slice(), out_after.as_slice());
}
