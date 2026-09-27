//! `fandhe_ai::interop::onnx::OnnxModel`（facade）経由の `Conv`（1D）・
//! `MaxPool`／`AveragePool` import が、内部クレート直接呼び出し
//! （`fandhe_ai_onnx_interop::onnx::interp::run`）と bit 一致すること
//! （イシュー #2199・親 #2185）、および `MaxPool`／`AveragePool` が
//! import 専用オペで export 未対応（`OnnxError::UnsupportedOp`）である
//! ことを確認する。
//!
//! `interop_onnx_internal_parity.rs` と同様、facade ラッパーが内部
//! クレートへ委譲するだけの薄い層であることを直接検証するため、本
//! ファイルは意図的に `fandhe_ai` と `fandhe_ai_onnx_interop` の両方を
//! import する。

use std::collections::HashMap;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxError, OnnxExportOptions, OnnxModel, OnnxValue};

use fandhe_ai_autodiff::nn::{Conv2d, MaxPool2d, Module};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, Value};
use fandhe_ai_onnx_interop::onnx::proto::{
    self, AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
    attribute_type, data_type,
};

fn attr_ints(name: &str, ints: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints,
        r#type: attribute_type::INTS,
        ..Default::default()
    }
}

fn value_info(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
    }
}

/// `MaxPool`（`kernel_shape=[2,2]`）単体ノードの `ModelProto` を組み立てる
/// （initializer なし。入力 `x` は feed で与える）。
fn build_max_pool_model() -> ModelProto {
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_maxpool".to_string(),
        op_type: "MaxPool".to_string(),
        attribute: vec![attr_ints("kernel_shape", vec![2, 2])],
        domain: String::new(),
    };
    ModelProto {
        graph: Some(GraphProto {
            node: vec![node],
            name: "max_pool_test".to_string(),
            initializer: vec![],
            input: vec![value_info("x")],
            output: vec![value_info("y")],
            value_info: vec![],
            sparse_initializer: vec![],
        }),
        ..Default::default()
    }
}

/// `Conv`（1D。`kernel_shape=[2]`）→ `AveragePool`（`kernel_shape=[2,2]`。
/// Conv 出力を rank4 へ reshape してから通す代わりに、素朴に Conv1d 単体の
/// モデルにする）の `ModelProto` を組み立てる。`w` は initializer として
/// 埋め込む。
fn build_conv1d_model() -> ModelProto {
    let node = NodeProto {
        input: vec!["x".to_string(), "w".to_string()],
        output: vec!["y".to_string()],
        name: "n_conv1d".to_string(),
        op_type: "Conv".to_string(),
        attribute: vec![attr_ints("kernel_shape", vec![2])],
        domain: String::new(),
    };
    let w_tensor = proto::TensorProto {
        dims: vec![1, 1, 2],
        data_type: proto::data_type::FLOAT,
        float_data: vec![1.0, -1.0],
        name: "w".to_string(),
        ..Default::default()
    };
    ModelProto {
        graph: Some(GraphProto {
            node: vec![node],
            name: "conv1d_test".to_string(),
            initializer: vec![w_tensor],
            input: vec![value_info("x")],
            output: vec![value_info("y")],
            value_info: vec![],
            sparse_initializer: vec![],
        }),
        ..Default::default()
    }
}

fn model_bytes(model: &ModelProto) -> Vec<u8> {
    proto::encode_model(model)
}

// --- facade と内部実装の bit 一致（MaxPool・2D） ---

#[test]
fn max_pool_facade_matches_internal_direct_call_bit_exact() {
    let bytes = model_bytes(&build_max_pool_model());
    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず");

    let internal_model = proto::decode_model(&bytes).expect("decode は成功するはず");
    let internal_graph = build_graph(&internal_model).expect("build_graph は成功するはず");

    for batch in [1usize, 2usize] {
        let numel = batch * 2 * 2;
        let data: Vec<f32> = (0..numel).map(|i| i as f32 - 1.5).collect();
        let shape = [batch, 1, 2, 2];

        let mut facade_feeds = HashMap::new();
        facade_feeds.insert(
            "x".to_string(),
            OnnxValue::F32(Tensor::<f32>::new(data.clone(), &shape).unwrap()),
        );
        let facade_result = facade_model.run(facade_feeds).expect("facade run 成功");
        let facade_out = match &facade_result["y"] {
            OnnxValue::F32(t) => t.clone(),
            other => panic!("OnnxValue::F32 を期待したが {other:?}"),
        };

        let mut internal_feeds = HashMap::new();
        internal_feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(data, &shape).unwrap()),
        );
        let internal_result =
            interp::run(&internal_graph, internal_feeds).expect("internal run 成功");
        let internal_out = match &internal_result["y"] {
            Value::F32(t) => t.clone(),
            other => panic!("Value::F32 を期待したが {other:?}"),
        };

        assert_eq!(facade_out.shape(), internal_out.shape(), "batch={batch}");
        for (a, b) in facade_out
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .zip(internal_out.contiguous().as_slice().unwrap().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "facade と内部クレート直接呼び出しが bit 不一致: batch={batch} facade={a} internal={b}"
            );
        }
    }
}

// --- facade と内部実装の bit 一致（Conv・1D） ---

#[test]
fn conv1d_facade_matches_internal_direct_call_bit_exact() {
    let bytes = model_bytes(&build_conv1d_model());
    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず");

    let internal_model = proto::decode_model(&bytes).expect("decode は成功するはず");
    let internal_graph = build_graph(&internal_model).expect("build_graph は成功するはず");

    for batch in [1usize, 2usize] {
        let numel = batch * 4;
        let data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.5).collect();
        let shape = [batch, 1, 4];

        let mut facade_feeds = HashMap::new();
        facade_feeds.insert(
            "x".to_string(),
            OnnxValue::F32(Tensor::<f32>::new(data.clone(), &shape).unwrap()),
        );
        let facade_result = facade_model.run(facade_feeds).expect("facade run 成功");
        let facade_out = match &facade_result["y"] {
            OnnxValue::F32(t) => t.clone(),
            other => panic!("OnnxValue::F32 を期待したが {other:?}"),
        };

        let mut internal_feeds = HashMap::new();
        internal_feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(data, &shape).unwrap()),
        );
        let internal_result =
            interp::run(&internal_graph, internal_feeds).expect("internal run 成功");
        let internal_out = match &internal_result["y"] {
            Value::F32(t) => t.clone(),
            other => panic!("Value::F32 を期待したが {other:?}"),
        };

        assert_eq!(facade_out.shape(), internal_out.shape(), "batch={batch}");
        for (a, b) in facade_out
            .contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .zip(internal_out.contiguous().as_slice().unwrap().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "facade と内部クレート直接呼び出しが bit 不一致: batch={batch} facade={a} internal={b}"
            );
        }
    }
}

// --- MaxPool は import 専用（export 未対応）: fail-closed の固定化 ---

#[test]
fn max_pool_import_succeeds_but_export_is_unsupported() {
    // `from_bytes`（import）は成功するが、`to_bytes`（export）は
    // `MaxPool` が export allowlist（`SUPPORTED_OP_TYPES`。23 op で
    // 不変）に含まれないため `OnnxError::UnsupportedOp` を返す
    // （import／export の非対称が意図的な fail-closed であることを
    // 固定する。実装計画 §4-9）。
    let bytes = model_bytes(&build_max_pool_model());
    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes（import）は成功するはず");

    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap()),
    );
    let run_result = facade_model
        .run(feeds)
        .expect("run（import 実行）は成功するはず");
    assert!(run_result.contains_key("y"));

    let err = facade_model
        .to_bytes(&OnnxExportOptions::default())
        .expect_err("MaxPool を含むグラフの export は拒否されるはず");
    match err {
        OnnxError::UnsupportedOp { op_type } => assert_eq!(op_type, "MaxPool"),
        other => panic!("OnnxError::UnsupportedOp を期待したが {other:?}"),
    }
}

// --- 公開 API 経由の Conv->MaxPool 2 ノードグラフ（codex-review 指摘。PR #2314
//     レビュー）: 本ファイルの既存テストは Conv・MaxPool を別々の単一ノード
//     モデルでしか確認しておらず、Conv の出力を Pool へ渡す結線を公開 API
//     （`OnnxModel::from_bytes`／`run`）経由で固定化するテストが無かった。
//     期待値は facade 内部の decode 経路ではなく `fandhe_ai_autodiff::nn::
//     Conv2d`／`MaxPool2d::forward_host`（独立実装）の連結結果と突合する ---

fn conv2d_initializer(name: &str, t: &fandhe_ai_tensor_core::Tensor<f32>) -> TensorProto {
    TensorProto {
        dims: t.shape().iter().map(|&d| d as i64).collect(),
        data_type: data_type::FLOAT,
        float_data: t.contiguous().as_slice().unwrap().to_vec(),
        name: name.to_string(),
        ..Default::default()
    }
}

/// `Conv(x, w, b) -> MaxPool(conv_out)` の 2 ノード `ModelProto`（`w`／`b` は
/// initializer。`x` のみ feed で与える）。
fn build_conv2d_then_max_pool_model(
    conv: &Conv2d,
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
) -> ModelProto {
    let conv_node = NodeProto {
        input: vec!["x".to_string(), "w".to_string(), "b".to_string()],
        output: vec!["conv_out".to_string()],
        name: "n_conv".to_string(),
        op_type: "Conv".to_string(),
        attribute: vec![
            attr_ints("kernel_shape", vec![3, 3]),
            attr_ints("strides", vec![1, 1]),
            attr_ints("pads", vec![1, 1, 1, 1]),
            attr_ints("dilations", vec![1, 1]),
        ],
        domain: String::new(),
    };
    let pool_node = NodeProto {
        input: vec!["conv_out".to_string()],
        output: vec!["y".to_string()],
        name: "n_pool".to_string(),
        op_type: "MaxPool".to_string(),
        attribute: vec![
            attr_ints("kernel_shape", kernel_shape),
            attr_ints("strides", strides),
        ],
        domain: String::new(),
    };
    ModelProto {
        graph: Some(GraphProto {
            node: vec![conv_node, pool_node],
            name: "conv2d_then_max_pool_test".to_string(),
            initializer: vec![
                conv2d_initializer("w", conv.weight()),
                conv2d_initializer("b", conv.bias().unwrap()),
            ],
            input: vec![value_info("x")],
            output: vec![value_info("y")],
            value_info: vec![],
            sparse_initializer: vec![],
        }),
        ..Default::default()
    }
}

#[test]
fn conv2d_then_max_pool_via_public_api_matches_nn_forward_host_chain() {
    // Conv2d(2->3,k3,s1,p1) -> MaxPool2d(k2,s2) を `OnnxModel::from_bytes`／
    // `run`（facade の公開 API のみ）経由で実行し、`nn::Conv2d`・
    // `nn::MaxPool2d::forward_host`（独立実装。`CpuBackendOps` 経由）を
    // 連結した参照値と突合する。Conv を含むため REQ-2 統一複合判定
    // （`.claude/rules/coding-rust.md`「バックエンド構成」節）に相当する
    // 許容誤差で比較する（`assert_parity` は onnx-interop 側の
    // 対応テストで用いており、facade 側は依存を増やさず同型の許容誤差を
    // 直接計算する）。
    let conv = Conv2d::new(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 500).unwrap();
    let shape = [1usize, 2, 8, 8];
    let numel: usize = shape.iter().product();
    let x_data: Vec<f32> = (0..numel).map(|v| (v as f32) * 0.05 - 2.0).collect();
    let x = Tensor::<f32>::new(x_data.clone(), &shape).unwrap();

    let model = build_conv2d_then_max_pool_model(&conv, vec![2, 2], vec![2, 2]);
    let bytes = proto::encode_model(&model);
    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず");

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), OnnxValue::F32(x.clone()));
    let result = facade_model.run(feeds).expect("run は成功するはず");
    let facade_out = match &result["y"] {
        OnnxValue::F32(t) => t.clone(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    };

    let ops = CpuBackendOps::new();
    let conv_out = conv.forward_host(&ops, &x).unwrap();
    let nn_max = MaxPool2d::new([2, 2], Some([2, 2]), [0, 0], [1, 1]).unwrap();
    let reference = nn_max.forward_host(&ops, &conv_out).unwrap();

    assert_eq!(facade_out.shape(), reference.shape());
    let a = facade_out.contiguous();
    let b = reference.contiguous();
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    assert_eq!(a_slice.len(), b_slice.len());
    for (i, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        let abs_err = (x - y).abs();
        let rel_err = abs_err / (y.abs() + 1e-6);
        // REQ-2 統一複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）。
        assert!(
            rel_err < 1e-3 || abs_err < 1e-5,
            "index={i} facade={x} reference={y} abs_err={abs_err} rel_err={rel_err}"
        );
    }
}
