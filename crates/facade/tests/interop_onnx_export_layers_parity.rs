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

// `Sigmoid`（イシュー #3844・数値契約 (α)）は x<0 の分岐形の差があるため
// bit 一致ではなく複合判定で突き合わせる。

#[test]
fn sigmoid_model_parity() {
    let model = Sequential::new()
        .add_linear(4, 6, 0x3844_0101)
        .expect("test fixture: 層構築に失敗")
        .add_sigmoid();
    let input = Tensor::new(
        (0..12).map(|i| (i as f32 - 6.0) * 0.7).collect::<Vec<_>>(),
        &[3, 4],
    )
    .expect("test fixture");
    assert_export_parity("Linear -> Sigmoid", &model, &input);

    let model2 = Sequential::new()
        .add_linear(4, 6, 0x3844_0102)
        .expect("test fixture: 層構築に失敗")
        .add_sigmoid()
        .add_linear(6, 2, 0x3844_0103)
        .expect("test fixture: 層構築に失敗");
    assert_export_parity("Linear -> Sigmoid -> Linear", &model2, &input);
}

fn sigmoid_only_model() -> Sequential {
    Sequential::new().add_sigmoid()
}

fn column(values: Vec<f32>) -> Tensor<f32> {
    let n = values.len();
    Tensor::new(values, &[n, 1]).expect("test fixture")
}

#[test]
fn sigmoid_edge_sweep_parity() {
    let values = vec![
        -1000.0,
        -103.9,
        -88.7,
        -88.0,
        -87.3,
        -16.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        16.0,
        88.0,
        1000.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
    ];
    assert_export_parity("Sigmoid 境界値", &sigmoid_only_model(), &column(values));
}

#[test]
fn sigmoid_negative_strided_sweep_parity() {
    // 有限の負の f32（符号ビット 1・NaN/inf を除く）を bit パターン上で等間隔に走査する。
    let values: Vec<f32> = (0x8000_0000u32..=0xFF7F_FFFFu32)
        .step_by(4099)
        .map(f32::from_bits)
        .collect();
    assert!(values.len() > 100_000, "走査点数が不足: {}", values.len());
    assert_export_parity(
        "Sigmoid 負域 stride 走査",
        &sigmoid_only_model(),
        &column(values),
    );
}

#[test]
fn sigmoid_nan_class_matches() {
    // assert_parity は NaN 同士を fail と判定するため要素ごとの is_nan で見る。
    let model = sigmoid_only_model();
    let input = column(vec![f32::NAN, 0.5]);
    let expected = model.predict(&input).expect("predict 成功");
    let exported = OnnxModel::from_sequential(&model).expect("from_sequential 成功");
    let mut feeds = HashMap::new();
    feeds.insert("input".to_string(), OnnxValue::F32(input));
    let mut outputs = exported.run(feeds).expect("run 成功");
    let actual = match outputs.remove("output").expect("output 無し") {
        OnnxValue::F32(t) => t,
        other => panic!("output は F32 のはず（実際 {other:?}）"),
    };
    let (e, a) = (flat(&expected), flat(&actual));
    assert!(e[0].is_nan() && a[0].is_nan(), "NaN 入力は両経路で NaN");
    assert!(!e[1].is_nan() && !a[1].is_nan());
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
