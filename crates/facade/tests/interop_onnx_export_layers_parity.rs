//! `fandhe_ai::interop::onnx::OnnxModel::from_sequential`（イシュー
//! #2037）の対応層拡大（イシュー #2076・親 #2034）の facade 経由 parity
//! テスト。
//!
//! **本ファイルは `fandhe_ai` と `fandhe_ai_backend_cpu::parity` のみを
//! import する**（`fandhe_ai_onnx_interop`・`fandhe_ai_autodiff` の内部
//! 型には依存しない。facade の公開 API のみを経由した検証という点で
//! `interop_onnx_internal_parity.rs`〈内部クレート直接呼び出しとの bit
//! 一致を目的とし意図的に内部クレートを import する〉とは目的が異なる）。
//!
//! `compat::Sequential::from_sequential(&m).run(..)`（export → interp
//! 経路）と `m.predict(&x)`（autodiff 直接 forward 経路）を REQ-2 統一
//! 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で突き合わせる
//! （`crates/facade/src/interop/onnx.rs` モジュール doc「`Sequential`
//! からの export」節の数値契約参照）。

use std::collections::HashMap;

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::onnx::{OnnxModel, OnnxValue};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn flat(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous な Tensor は as_slice できるはず")
        .to_vec()
}

/// `model.predict(&input)`（autodiff 直接 forward）と
/// `OnnxModel::from_sequential(model).run(..)`（export -> interp 経路）
/// の出力を REQ-2 統一複合判定で突き合わせる。
fn assert_export_parity(context: &str, model: &Sequential, input: &Tensor<f32>) {
    let expected = model.predict(input).expect("predict は成功するはず");

    let exported = OnnxModel::from_sequential(model).expect("from_sequential は成功するはず");
    let mut feeds = HashMap::new();
    feeds.insert("input".to_string(), OnnxValue::F32(input.clone()));
    let mut outputs = exported.run(feeds).expect("run は成功するはず");
    let actual = match outputs.remove("output").expect("output が実行結果に無い") {
        OnnxValue::F32(t) => t,
        other => panic!("output は F32 のはず（実際 {other:?}）"),
    };

    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: 出力 shape 不一致"
    );
    assert_parity(context, &flat(&actual), &flat(&expected));
}

#[test]
fn sigmoid_model_parity() {
    let model = Sequential::new()
        .add_linear(4, 6, 0x2076_f001)
        .expect("test fixture: add_linear に失敗")
        .add_sigmoid();
    let input =
        Tensor::<f32>::new((0..12).map(|i| (i as f32) * 0.1 - 0.6).collect(), &[3, 4]).unwrap();
    assert_export_parity("Linear -> Sigmoid", &model, &input);
}

#[test]
fn softmax_model_parity() {
    let model = Sequential::new()
        .add_linear(4, 6, 0x2076_f002)
        .expect("test fixture: add_linear に失敗")
        .add_softmax(1);
    let input =
        Tensor::<f32>::new((0..12).map(|i| (i as f32) * 0.1 - 0.6).collect(), &[3, 4]).unwrap();
    assert_export_parity("Linear -> Softmax(dim=1)", &model, &input);
}

#[test]
fn layer_norm_model_parity() {
    let model = Sequential::new()
        .add_linear(4, 6, 0x2076_f003)
        .expect("test fixture: add_linear に失敗")
        .add_layer_norm(6, 1e-5)
        .expect("test fixture: add_layer_norm に失敗");
    let input =
        Tensor::<f32>::new((0..12).map(|i| (i as f32) * 0.1 - 0.6).collect(), &[3, 4]).unwrap();
    assert_export_parity("Linear -> LayerNorm", &model, &input);
}

#[test]
fn gelu_model_parity() {
    let model = Sequential::new()
        .add_linear(4, 6, 0x2076_f004)
        .expect("test fixture: add_linear に失敗")
        .add_gelu()
        .add_linear(6, 3, 0x2076_f005)
        .expect("test fixture: add_linear に失敗");
    let input =
        Tensor::<f32>::new((0..12).map(|i| (i as f32) * 0.1 - 0.6).collect(), &[3, 4]).unwrap();
    assert_export_parity("Linear -> GELU -> Linear", &model, &input);
}

#[test]
fn conv2d_model_parity() {
    let model = Sequential::new()
        .add_conv2d(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 0x2076_f006)
        .expect("test fixture: add_conv2d に失敗")
        .add_relu();
    let input = Tensor::<f32>::new(
        (0..(2 * 5 * 5)).map(|i| (i as f32) * 0.05 - 1.0).collect(),
        &[1, 2, 5, 5],
    )
    .unwrap();
    assert_export_parity("Conv2d -> Relu", &model, &input);
}
