//! `onnx::export_nn`（`nn::Module` 層列 -> `ExportNode` 列＋initializer＋
//! `Graph`。イシュー #2036）の統合テスト。公開 API のみを経由する:
//!
//! - roundtrip（層 A）: `export_nn::graph_from_layers -> build_model_proto ->
//!   encode -> decode -> build_graph -> interp::run` の出力を、
//!   `nn::Module::forward_host`（tape 不要経路）・`nn::Module::forward`
//!   （tape 経路）の 2 通りの独立な手動 forward と bit 完全一致で突き合わせる。
//! - 契約テスト: initializer 名の命名規約・属性の常時書き出し・決定性。
//! - fail-closed テスト: 空層列・未対応層。
//!
//! ## bit 一致契約の前提
//!
//! `export_nn.rs` モジュール冒頭ドキュメント「bit 一致契約の前提」節の
//! とおり、GEMM 出力に厳密な `±0.0` が現れない入力を用いる（`Relu` の
//! `max(0.0, x)` が `±0.0` 同士では実装依存になりうるため）。乱数生成器
//! （`bench_harness::rng::Xorshift64Star`）の出力は `[-1.0, 1.0)` の
//! 浮動小数点であり、有限次元の内積がちょうど `±0.0` になる確率は
//! 実用上ゼロ（本テストで使う全形状・全シードで実際に発生しないことを
//! 確認済み）。NaN／inf を生む値も使わない。

use std::collections::{HashMap, HashSet};

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::activation::{Relu, Tanh};
use fandhe_ai_autodiff::nn::{Linear, Module, RmsNorm, Sequential};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_onnx_interop::onnx::export::{ExportError, ExportOptions, build_model_proto};
use fandhe_ai_onnx_interop::onnx::export_nn::{self, GRAPH_INPUT_NAME, GRAPH_OUTPUT_NAME};
use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, Value};
use fandhe_ai_onnx_interop::onnx::proto::ModelProto;
use fandhe_ai_tensor_core::Tensor;
use prost::Message;

// ---- テストユーティリティ ----

/// 決定的シードから `Linear` を構築する（`bias: false` なら 2 入力 Gemm、
/// `true` なら 3 入力 Gemm になる）。
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

/// `Linear` を複製する（`Tensor<f32>` は `Clone`。`Linear` 自体は
/// `Clone` を実装しないため `weight()`/`bias()` の値を経由する）。
fn clone_linear(l: &Linear) -> Linear {
    Linear::from_parameters(l.weight().clone(), l.bias().cloned())
        .expect("Linear::from_parameters（複製）は成功するはず")
}

fn input_tensor(rng: &mut Xorshift64Star, batch: usize, features: usize) -> Tensor<f32> {
    Tensor::new(rng.fill_vec(batch * features), &[batch, features])
        .expect("input Tensor::new は成功するはず")
}

/// `nn::Module::forward_host`（tape 不要経路）を層順に適用する独立参照経路 1。
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

/// `nn::Module::forward`（tape 経路）を層順に適用する独立参照経路 2。
fn forward_tape_chain(layers: &[Box<dyn Module>], input: &Tensor<f32>) -> Tensor<f32> {
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let mut var = tape.var(input);
    for layer in layers {
        var = layer.forward(&tape, &var).expect("forward は成功するはず");
    }
    var.to_tensor()
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
        .expect("output feed が存在するはず")
    {
        Value::F32(t) => t,
        other => panic!("F32 出力を期待したが {other:?} だった"),
    }
}

fn assert_bit_exact(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape 不一致");
    let a_contig = a.contiguous();
    let b_contig = b.contiguous();
    let a_slice = a_contig.as_slice().expect("as_slice は成功するはず");
    let b_slice = b_contig.as_slice().expect("as_slice は成功するはず");
    assert_eq!(a_slice.len(), b_slice.len(), "{label}: 要素数不一致");
    for (i, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: 要素 {i} が bit 不一致（{x} vs {y}）"
        );
    }
}

// ---- 1. roundtrip（小形状。bias あり／なし混在） ----

#[test]
fn two_layer_mlp_roundtrip_matches_nn_bit_exact_small() {
    let mut weight_rng = Xorshift64Star::new(0x2036_0001);
    let mut data_rng = Xorshift64Star::new(0x2036_1001);

    let l1 = linear_from_rng(&mut weight_rng, 4, 5, true);
    let l2 = linear_from_rng(&mut weight_rng, 5, 2, false);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Relu), Box::new(l2)];

    let input = input_tensor(&mut data_rng, 3, 4);

    let via_host = forward_host_chain(&layers, &input);
    let via_tape = forward_tape_chain(&layers, &input);
    let via_onnx = export_roundtrip_output(&layers, &input);

    assert_bit_exact("host vs tape", &via_host, &via_tape);
    assert_bit_exact("host vs onnx roundtrip", &via_host, &via_onnx);
}

// ---- 2. roundtrip（大形状。BLIS ブロックタイル境界〈KC=256・MR/NR〉を跨ぐ） ----

#[test]
fn two_layer_mlp_roundtrip_matches_nn_bit_exact_large() {
    let mut weight_rng = Xorshift64Star::new(0x2036_0002);
    let mut data_rng = Xorshift64Star::new(0x2036_1002);

    // in=300 / hidden=257 / out=7 は CPU BLIS の KC=256（docs/cpu-gemm-*）を
    // 跨ぐ形状であり、interp（素朴ループ）と本番 GEMM 経路の縮約順序差が
    // 出やすい領域をカバーする。両層とも bias あり。
    let l1 = linear_from_rng(&mut weight_rng, 300, 257, true);
    let l2 = linear_from_rng(&mut weight_rng, 257, 7, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Relu), Box::new(l2)];

    let input = input_tensor(&mut data_rng, 5, 300);

    let via_host = forward_host_chain(&layers, &input);
    let via_tape = forward_tape_chain(&layers, &input);
    let via_onnx = export_roundtrip_output(&layers, &input);

    assert_bit_exact("host vs tape", &via_host, &via_tape);
    assert_bit_exact("host vs onnx roundtrip", &via_host, &via_onnx);
}

// ---- 3. Relu 単独（Linear なし）も構造上許容される ----

#[test]
fn relu_only_model_roundtrips() {
    let mut data_rng = Xorshift64Star::new(0x2036_2001);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(Relu)];
    let input = input_tensor(&mut data_rng, 2, 3);

    let via_host = forward_host_chain(&layers, &input);
    let via_onnx = export_roundtrip_output(&layers, &input);
    assert_bit_exact("relu only", &via_host, &via_onnx);

    let graph = export_nn::graph_from_layers(&layers).expect("graph_from_layers は成功するはず");
    assert_eq!(graph.inputs, vec![GRAPH_INPUT_NAME.to_string()]);
    assert_eq!(graph.outputs, vec![GRAPH_OUTPUT_NAME.to_string()]);
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.nodes[0].op_type, "Relu");
    assert_eq!(graph.nodes[0].input, vec![GRAPH_INPUT_NAME.to_string()]);
    assert_eq!(graph.nodes[0].output, vec![GRAPH_OUTPUT_NAME.to_string()]);
}

// ---- 契約テスト ----

#[test]
fn initializer_names_match_sequential_named_parameters() {
    let mut weight_rng = Xorshift64Star::new(0x2036_0003);

    let l1 = linear_from_rng(&mut weight_rng, 4, 5, true);
    let l2 = linear_from_rng(&mut weight_rng, 5, 2, false);

    let seq = Sequential::new()
        .add(clone_linear(&l1))
        .add(Relu)
        .add(clone_linear(&l2));
    let expected: HashSet<String> = seq
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Relu), Box::new(l2)];
    let parts = export_nn::export_parts_from_layers(&layers)
        .expect("export_parts_from_layers は成功するはず");
    let actual: HashSet<String> = parts
        .initializers
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    assert_eq!(actual, expected);
    // 命名契約（モジュール冒頭「名前規約」節）: layer0 は weight のみ
    // （bias なし）ではなく weight+bias、layer2 は weight のみ。
    assert!(actual.contains("0.weight"));
    assert!(actual.contains("0.bias"));
    assert!(actual.contains("2.weight"));
    assert!(!actual.contains("2.bias"));
}

#[test]
fn gemm_node_always_writes_default_attributes_and_bias_absent_has_two_inputs() {
    let mut weight_rng = Xorshift64Star::new(0x2036_0004);
    let l1 = linear_from_rng(&mut weight_rng, 3, 4, false);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1)];

    let graph = export_nn::graph_from_layers(&layers).expect("graph_from_layers は成功するはず");
    assert_eq!(graph.nodes.len(), 1);
    let node = &graph.nodes[0];
    assert_eq!(node.op_type, "Gemm");
    assert_eq!(node.domain, "");
    // bias なし Linear は 2 入力（input, weight）。
    assert_eq!(node.input.len(), 2);
    assert_eq!(node.input[0], GRAPH_INPUT_NAME);
    assert_eq!(node.input[1], "0.weight");
    assert_eq!(node.output, vec![GRAPH_OUTPUT_NAME.to_string()]);

    // `docs/onnx-export-op-mapping.md` §1「属性常時書き出し」契約:
    // alpha/beta/transA/transB の 4 属性が既定値でもすべて存在する。
    let attr_names: HashSet<&str> = node.attribute.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(
        attr_names,
        HashSet::from(["alpha", "beta", "transA", "transB"])
    );
    for attr in &node.attribute {
        match attr.name.as_str() {
            "alpha" | "beta" => assert_eq!(attr.f, 1.0),
            "transA" | "transB" => assert_eq!(attr.i, 0),
            other => panic!("想定外の属性: {other}"),
        }
    }
}

#[test]
fn export_is_deterministic_across_repeated_calls() {
    let mut weight_rng = Xorshift64Star::new(0x2036_0005);
    let l1 = linear_from_rng(&mut weight_rng, 4, 3, true);
    let l2 = linear_from_rng(&mut weight_rng, 3, 2, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Relu), Box::new(l2)];

    let graph1 = export_nn::graph_from_layers(&layers).expect("1 回目の export は成功するはず");
    let graph2 = export_nn::graph_from_layers(&layers).expect("2 回目の export は成功するはず");
    let bytes1 = build_model_proto(&graph1, &ExportOptions::default())
        .expect("build_model_proto は成功するはず")
        .encode_to_vec();
    let bytes2 = build_model_proto(&graph2, &ExportOptions::default())
        .expect("build_model_proto は成功するはず")
        .encode_to_vec();
    assert_eq!(bytes1, bytes2);

    // decode -> build_graph -> build_model_proto -> encode の不動点。
    let decoded = ModelProto::decode(bytes1.as_slice()).expect("decode は成功するはず");
    let reimported = build_graph(&decoded).expect("build_graph は成功するはず");
    let bytes3 = build_model_proto(&reimported, &ExportOptions::default())
        .expect("build_model_proto は成功するはず")
        .encode_to_vec();
    assert_eq!(bytes1, bytes3);
}

// ---- fail-closed テスト ----

#[test]
fn empty_layer_list_is_rejected() {
    let layers: Vec<Box<dyn Module>> = Vec::new();
    let err = export_nn::export_parts_from_layers(&layers).expect_err("空層列は拒否されるはず");
    assert_eq!(err, ExportError::EmptyModel);
}

#[test]
fn unsupported_layer_is_rejected_with_index_and_unknown_kind() {
    // `Tanh` は対応する ONNX 演算が無い層として未対応のまま（`Sigmoid` は
    // イシュー #2076 で対応層化したため負例に使えなくなった）。
    let layers: Vec<Box<dyn Module>> = vec![Box::new(Relu), Box::new(Tanh)];
    let err = export_nn::export_parts_from_layers(&layers).expect_err("Tanh は未対応のはず");
    assert_eq!(
        err,
        ExportError::UnsupportedLayer {
            index: 1,
            layer_kind: "unknown",
        }
    );
}

#[test]
fn unsupported_layer_reports_known_kind_when_downcast_hook_matches() {
    let norm = RmsNorm::new(4, 1e-5).expect("RmsNorm::new は成功するはず");
    let layers: Vec<Box<dyn Module>> = vec![Box::new(norm)];
    let err = export_nn::export_parts_from_layers(&layers).expect_err("RmsNorm は未対応のはず");
    assert_eq!(
        err,
        ExportError::UnsupportedLayer {
            index: 0,
            layer_kind: "RmsNorm",
        }
    );
}

#[test]
fn unsupported_layer_after_supported_prefix_does_not_return_partial_graph() {
    let mut weight_rng = Xorshift64Star::new(0x2036_0006);
    let l1 = linear_from_rng(&mut weight_rng, 4, 4, true);
    let layers: Vec<Box<dyn Module>> = vec![Box::new(l1), Box::new(Relu), Box::new(Tanh)];
    let err =
        export_nn::export_parts_from_layers(&layers).expect_err("末尾の Tanh で拒否されるはず");
    assert_eq!(
        err,
        ExportError::UnsupportedLayer {
            index: 2,
            layer_kind: "unknown",
        }
    );
}
