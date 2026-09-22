//! `onnx::export_nn` の対応層拡大（イシュー #2076・親 #2034）の parity
//! テスト。`onnx_export_nn.rs`（Linear／ReLU の bit 完全一致契約）とは
//! 別ファイルに分離する（本ファイルは Sigmoid・Softmax・LayerNorm・
//! GELU・Conv2d を含む REQ-2 統一複合判定〈相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満〉のテストのみを扱う。`export_nn.rs` モジュール
//! 冒頭「bit 一致契約の前提」節参照）。
//!
//! 各テストは export → encode → decode → import → `interp::run` の
//! roundtrip 出力を、`nn::Module::forward_host`（tape 不要経路）による
//! 独立参照計算と `fandhe_ai_backend_cpu::parity::assert_parity` で
//! 突き合わせる。

use std::collections::HashMap;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::nn::activation::{Gelu, GeluTanh, LogSoftmax, Relu, Sigmoid, Softmax};
use fandhe_ai_autodiff::nn::{Conv2d, LayerNorm, Linear, Module};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_onnx_interop::onnx::export::{ExportError, ExportOptions, build_model_proto};
use fandhe_ai_onnx_interop::onnx::export_nn::{self, GRAPH_INPUT_NAME, GRAPH_OUTPUT_NAME};
use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, Value};
use fandhe_ai_onnx_interop::onnx::proto::ModelProto;
use fandhe_ai_tensor_core::Tensor;
use prost::Message;

// ---- テストユーティリティ（`onnx_export_nn.rs` と同型。別テスト
// バイナリのため独立に定義する） ----

fn linear_from_rng(
    rng: &mut Xorshift64Star,
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> Linear {
    let weight = Tensor::new(
        rng.fill_vec(in_features * out_features),
        &[in_features, out_features],
    )
    .expect("weight Tensor::new は成功するはず");
    let bias = if bias {
        Some(
            Tensor::new(rng.fill_vec(out_features), &[out_features])
                .expect("bias Tensor::new は成功するはず"),
        )
    } else {
        None
    };
    Linear::from_parameters(weight, bias).expect("Linear::from_parameters は成功するはず")
}

fn input_tensor(rng: &mut Xorshift64Star, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::new(rng.fill_vec(numel), shape).expect("input Tensor::new は成功するはず")
}

fn forward_host_chain(layers: &[Box<dyn Module>], input: &Tensor<f32>) -> Tensor<f32> {
    let ops = CpuBackendOps::new();
    let mut current = input.clone();
    for layer in layers {
        current = layer
            .forward_host(&ops, &current)
            .expect("forward_host は成功するはず");
    }
    current
}

/// export -> encode -> decode -> import -> `interp::run` の往復を行い、
/// `GRAPH_OUTPUT_NAME` の出力を返す。
fn export_roundtrip_output(layers: &[Box<dyn Module>], input: &Tensor<f32>) -> Tensor<f32> {
    let graph = export_nn::graph_from_layers(layers).expect("graph_from_layers は成功するはず");
    let model = build_model_proto(&graph, &ExportOptions::default())
        .expect("build_model_proto は成功するはず");
    let bytes = model.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("再 decode は成功するはず");
    let reimported = build_graph(&decoded).expect("build_graph は成功するはず");

    let mut feeds = HashMap::new();
    feeds.insert(GRAPH_INPUT_NAME.to_string(), Value::F32(input.clone()));
    let mut outputs = interp::run(&reimported, feeds).expect("interp::run は成功するはず");
    match outputs
        .remove(GRAPH_OUTPUT_NAME)
        .expect("GRAPH_OUTPUT_NAME が実行結果に無い")
    {
        Value::F32(t) => t,
        other => panic!("GRAPH_OUTPUT_NAME の出力は F32 のはず（実際 {other:?}）"),
    }
}

fn flat(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous な Tensor は as_slice できるはず")
        .to_vec()
}

/// `forward_host_chain`（独立参照経路）と `export_roundtrip_output`
/// （export → interp 経路）を REQ-2 統一複合判定で突き合わせる
/// （`export_nn.rs` モジュール冒頭「bit 一致契約の前提」節: Linear／
/// Relu 以外を含むモデルはこの複合判定で検証する）。
fn assert_layers_parity(context: &str, layers: &[Box<dyn Module>], input: &Tensor<f32>) {
    let expected = forward_host_chain(layers, input);
    let actual = export_roundtrip_output(layers, input);
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: 出力 shape 不一致"
    );
    assert_parity(context, &flat(&actual), &flat(&expected));
}

// ---- 単層 parity ----

#[test]
fn sigmoid_layer_parity() {
    let mut rng = Xorshift64Star::new(0x2076_0001);
    let l1 = linear_from_rng(&mut rng, 4, 6, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Sigmoid)];
    let input = input_tensor(&mut rng, &[3, 4]);
    assert_layers_parity("Linear -> Sigmoid", &layers, &input);
}

#[test]
fn softmax_layer_parity() {
    let mut rng = Xorshift64Star::new(0x2076_0002);
    let l1 = linear_from_rng(&mut rng, 4, 6, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Softmax::new(1))];
    let input = input_tensor(&mut rng, &[3, 4]);
    assert_layers_parity("Linear -> Softmax(dim=1)", &layers, &input);
}

#[test]
fn layer_norm_layer_parity() {
    let mut rng = Xorshift64Star::new(0x2076_0003);
    let l1 = linear_from_rng(&mut rng, 4, 6, true);
    let ln = LayerNorm::new(6, 1e-5).expect("LayerNorm::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(ln)];
    let input = input_tensor(&mut rng, &[3, 4]);
    assert_layers_parity("Linear -> LayerNorm", &layers, &input);
}

#[test]
fn gelu_layer_parity() {
    let mut rng = Xorshift64Star::new(0x2076_0004);
    let l1 = linear_from_rng(&mut rng, 4, 6, true);
    let l2 = linear_from_rng(&mut rng, 6, 3, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Gelu), Box::new(l2)];
    let input = input_tensor(&mut rng, &[3, 4]);
    assert_layers_parity("Linear -> GELU -> Linear", &layers, &input);
}

#[test]
fn conv2d_layer_parity_no_bias() {
    let mut rng = Xorshift64Star::new(0x2076_0005);
    let conv = Conv2d::new(2, 4, [3, 3], [1, 1], [0, 0], [1, 1], 1, false, 0x2076_0005)
        .expect("Conv2d::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(conv), Box::new(Relu)];
    let input = input_tensor(&mut rng, &[1, 2, 5, 5]);
    assert_layers_parity("Conv2d(no bias) -> Relu", &layers, &input);
}

#[test]
fn conv2d_layer_parity_with_bias_and_padding() {
    let mut rng = Xorshift64Star::new(0x2076_0006);
    let conv = Conv2d::new(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 0x2076_0006)
        .expect("Conv2d::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(conv)];
    let input = input_tensor(&mut rng, &[1, 2, 5, 5]);
    assert_layers_parity("Conv2d(bias, padding=1)", &layers, &input);
}

#[test]
fn conv2d_layer_parity_groups_and_dilation() {
    let mut rng = Xorshift64Star::new(0x2076_0007);
    // groups=2（Cin=4 を 2 群へ分割）・dilation=2。
    let conv = Conv2d::new(4, 4, [2, 2], [1, 1], [0, 0], [2, 2], 2, true, 0x2076_0007)
        .expect("Conv2d::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(conv)];
    let input = input_tensor(&mut rng, &[1, 4, 6, 6]);
    assert_layers_parity("Conv2d(groups=2, dilation=2)", &layers, &input);
}

#[test]
fn conv2d_layer_parity_stride2() {
    let mut rng = Xorshift64Star::new(0x2076_0008);
    let conv = Conv2d::new(2, 3, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, 0x2076_0008)
        .expect("Conv2d::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(conv)];
    let input = input_tensor(&mut rng, &[1, 2, 7, 7]);
    assert_layers_parity("Conv2d(stride=2, padding=1)", &layers, &input);
}

// ---- 合成 parity ----

#[test]
fn composite_conv_relu_then_flatten_free_linear_head() {
    // Flatten 非対応（`docs/onnx-export-op-mapping.md` §7「Conv2d 対応 ≠
    // CNN 対応」）のため、Conv2d の出力 shape をそのまま次層へ渡せる
    // 構成（Conv2d 単体または Conv2d -> 活性化）のみを検証する。
    let mut rng = Xorshift64Star::new(0x2076_0009);
    let conv = Conv2d::new(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 0x2076_0009)
        .expect("Conv2d::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(conv), Box::new(Sigmoid)];
    let input = input_tensor(&mut rng, &[1, 2, 5, 5]);
    assert_layers_parity("Conv2d -> Sigmoid", &layers, &input);
}

// ---- fail-closed テスト ----

#[test]
fn layer_norm_without_affine_is_rejected() {
    // ONNX `LayerNormalization` は `Scale` 必須のため、
    // `elementwise_affine=false`（`weight() == None`）の LayerNorm は
    // export できない（`export_nn.rs` モジュール冒頭「対応層」節）。
    let ln = LayerNorm::without_affine(1e-5).expect("LayerNorm::without_affine は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(ln)];
    let err = export_nn::export_parts_from_layers(&layers)
        .expect_err("without_affine の LayerNorm は拒否されるはず");
    assert!(
        matches!(err, ExportError::InvalidLayerParameter { index: 0, .. }),
        "予期しないエラー種別: {err:?}"
    );
}

#[test]
fn gelu_tanh_is_rejected_as_unsupported() {
    // `GeluTanh`（tanh 近似 GELU）は ONNX opset 17 に対応する演算が無い
    // ため `as_gelu` をオーバーライドせず、`Module::as_gelu()` は
    // `false` のまま fail-closed に拒否される。
    let layers: Vec<Box<dyn Module>> = vec![Box::new(GeluTanh)];
    let err = export_nn::export_parts_from_layers(&layers).expect_err("GeluTanh は未対応のはず");
    assert!(
        matches!(
            err,
            ExportError::UnsupportedLayer {
                index: 0,
                layer_kind: "unknown",
            }
        ),
        "予期しないエラー: {err:?}"
    );
}

#[test]
fn log_softmax_is_rejected_as_unsupported() {
    // `LogSoftmax` は対応する ONNX 演算が無いため `as_softmax` を
    // オーバーライドせず fail-closed に拒否される。
    let layers: Vec<Box<dyn Module>> = vec![Box::new(LogSoftmax::new(0))];
    let err = export_nn::export_parts_from_layers(&layers).expect_err("LogSoftmax は未対応のはず");
    assert!(
        matches!(
            err,
            ExportError::UnsupportedLayer {
                index: 0,
                layer_kind: "unknown",
            }
        ),
        "予期しないエラー: {err:?}"
    );
}

// ---- 決定性 ----

#[test]
fn gelu_composite_export_is_deterministic() {
    let mut rng = Xorshift64Star::new(0x2076_000a);
    let l1 = linear_from_rng(&mut rng, 4, 6, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Gelu)];

    let graph1 = export_nn::graph_from_layers(&layers).expect("1 回目の export は成功するはず");
    let graph2 = export_nn::graph_from_layers(&layers).expect("2 回目の export は成功するはず");
    let bytes1 = build_model_proto(&graph1, &ExportOptions::default())
        .expect("build_model_proto は成功するはず")
        .encode_to_vec();
    let bytes2 = build_model_proto(&graph2, &ExportOptions::default())
        .expect("build_model_proto は成功するはず")
        .encode_to_vec();
    assert_eq!(
        bytes1, bytes2,
        "GELU 合成ノードの export はバイト単位で決定的なはず"
    );
}
