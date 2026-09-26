//! `fandhe_ai::interop::onnx::OnnxModel` 経由の `GlobalAveragePool`／
//! `BatchNormalization`／`Flatten` import・実行を検証する（イシュー
//! #2200・親 #2185）。
//!
//! **本ファイルは `interop_onnx_internal_parity.rs` と同じく意図的に
//! `fandhe_ai` と `fandhe_ai_onnx_interop` の両方を import する**（合成
//! `ModelProto` を組み立てて `encode_model` でバイト列化するには
//! `onnx-interop` の proto 型が必要なため。`fandhe_ai`（facade）だけを
//! import するテストファイルではエンコードできない）。
//!
//! 検証範囲:
//! 1. PyTorch export 形の合成モデル（BN → Relu → GAP → Flatten）を
//!    `from_bytes` → `run` した結果が、内部クレート直接呼び出し
//!    （`fandhe_ai_onnx_interop::onnx::interp::run`）と bit 一致すること
//! 2. `compat::Sequential`（`add_adaptive_avg_pool2d([1,1])`．
//!    `add_flatten(1,3)`）の `predict` が、GAP → Flatten の ONNX モデルを
//!    `OnnxModel` 経由で実行した結果と bit 一致すること
//! 3. **非対称性**: 同じモデルで `from_bytes`／`run` は成功するが
//!    `to_bytes` は `OnnxError::UnsupportedOp` になること（export
//!    allowlist が 23 のままであることを固定する）

use std::collections::HashMap;

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::onnx::{OnnxError, OnnxExportOptions, OnnxModel, OnnxValue};

use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, Value};
use fandhe_ai_onnx_interop::onnx::proto::{
    self, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, data_type,
};

fn f32_tensor_proto(name: &str, data: Vec<f32>, dims: Vec<i64>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: data_type::FLOAT,
        dims,
        float_data: data,
        int64_data: vec![],
        raw_data: vec![],
    }
}

fn value_infos(names: &[&str]) -> Vec<ValueInfoProto> {
    names
        .iter()
        .map(|n| ValueInfoProto {
            name: n.to_string(),
        })
        .collect()
}

fn node(op_type: &str, name: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
    NodeProto {
        input: input.into_iter().map(String::from).collect(),
        output: output.into_iter().map(String::from).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        attribute: vec![],
        domain: String::new(),
    }
}

fn as_f32(v: &OnnxValue) -> Tensor<f32> {
    match v {
        OnnxValue::F32(t) => t.clone(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    }
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape());
    let ac = a.contiguous();
    let bc = b.contiguous();
    let a_slice = ac.as_slice().unwrap();
    let b_slice = bc.as_slice().unwrap();
    for (x, y) in a_slice.iter().zip(b_slice.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

/// PyTorch export 形の合成モデル（イシュー #2200 実装計画 §3.6）:
/// `X -> BatchNormalization -> Relu -> GlobalAveragePool -> Flatten -> y`。
/// `scale`／`bias`／`mean`／`var` は initializer として埋め込む。
fn build_bn_relu_gap_flatten_model(c: usize) -> ModelProto {
    let scale: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..c).map(|i| i as f32 * 0.05).collect();
    let mean: Vec<f32> = (0..c).map(|i| i as f32 * 0.2 - 1.0).collect();
    let var: Vec<f32> = vec![1.0; c];

    let nodes = vec![
        node(
            "BatchNormalization",
            "bn",
            vec!["x", "scale", "bias", "mean", "var"],
            vec!["bn_out"],
        ),
        node("Relu", "relu", vec!["bn_out"], vec!["relu_out"]),
        node(
            "GlobalAveragePool",
            "gap",
            vec!["relu_out"],
            vec!["gap_out"],
        ),
        node("Flatten", "flatten", vec!["gap_out"], vec!["y"]),
    ];

    ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: nodes,
            name: "g".to_string(),
            initializer: vec![
                f32_tensor_proto("scale", scale, vec![c as i64]),
                f32_tensor_proto("bias", bias, vec![c as i64]),
                f32_tensor_proto("mean", mean, vec![c as i64]),
                f32_tensor_proto("var", var, vec![c as i64]),
            ],
            input: value_infos(&["x"]),
            output: value_infos(&["y"]),
        }),
    }
}

#[test]
fn facade_bn_relu_gap_flatten_matches_internal_direct_call_bit_exact() {
    let (n, c, h, w) = (2usize, 3usize, 4usize, 4usize);
    let model = build_bn_relu_gap_flatten_model(c);
    let bytes = proto::encode_model(&model);

    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes 成功");
    let internal_model = proto::decode_model(&bytes).expect("decode_model 成功");
    let internal_graph = build_graph(&internal_model).expect("build_graph 成功");

    let x_data: Vec<f32> = (0..(n * c * h * w))
        .map(|v| (v as f32) * 0.1 - 2.0)
        .collect();

    let mut facade_feeds = HashMap::new();
    facade_feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(x_data.clone(), &[n, c, h, w]).unwrap()),
    );
    let facade_result = facade_model.run(facade_feeds).expect("facade run 成功");
    let facade_out = as_f32(&facade_result["y"]);
    assert_eq!(facade_out.shape(), &[n, c]);

    let mut internal_feeds = HashMap::new();
    internal_feeds.insert(
        "x".to_string(),
        Value::F32(Tensor::<f32>::new(x_data, &[n, c, h, w]).unwrap()),
    );
    let internal_result = interp::run(&internal_graph, internal_feeds).expect("internal run 成功");
    let internal_out = match &internal_result["y"] {
        Value::F32(t) => t.clone(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    };

    assert_bit_identical(&facade_out, &internal_out);
}

#[test]
fn facade_global_average_pool_flatten_matches_compat_sequential_predict() {
    let (n, c, h, w) = (2usize, 3usize, 5usize, 5usize);

    let nodes = vec![
        node("GlobalAveragePool", "gap", vec!["x"], vec!["gap_out"]),
        node("Flatten", "flatten", vec!["gap_out"], vec!["y"]),
    ];
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: nodes,
            name: "g".to_string(),
            initializer: vec![],
            input: value_infos(&["x"]),
            output: value_infos(&["y"]),
        }),
    };
    let bytes = proto::encode_model(&model);
    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes 成功");

    let x_data: Vec<f32> = (0..(n * c * h * w))
        .map(|v| (v as f32) * 0.37 - 3.0)
        .collect();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(x_data.clone(), &[n, c, h, w]).unwrap()),
    );
    let onnx_out = as_f32(&facade_model.run(feeds).expect("run 成功")["y"]);

    // 参照実装: `compat::Sequential`（`add_adaptive_avg_pool2d([1,1])` →
    // `add_flatten(1,3)`）の `predict`（tape 不要のホスト常駐 forward）。
    let seq = Sequential::new()
        .add_adaptive_avg_pool2d([1, 1])
        .expect("add_adaptive_avg_pool2d 成功")
        .add_flatten(1, 3);
    let x = Tensor::<f32>::new(x_data, &[n, c, h, w]).unwrap();
    let seq_out = seq.predict(&x).expect("predict 成功");

    assert_bit_identical(&onnx_out, &seq_out);
}

#[test]
fn facade_import_succeeds_but_export_rejects_global_average_pool_and_flatten() {
    // 非対称性テスト: import（`from_bytes`／`run`）は成功するが、export
    // （`to_bytes`）は export allowlist（23 op）外のため
    // `OnnxError::UnsupportedOp` で拒否される（イシュー #2200）。
    let nodes = vec![
        node("GlobalAveragePool", "gap", vec!["x"], vec!["gap_out"]),
        node("Flatten", "flatten", vec!["gap_out"], vec!["y"]),
    ];
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: nodes,
            name: "g".to_string(),
            initializer: vec![],
            input: value_infos(&["x"]),
            output: value_infos(&["y"]),
        }),
    };
    let bytes = proto::encode_model(&model);

    let facade_model =
        OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず（import 対応済み）");
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::zeros(&[1, 2, 3, 3]).unwrap()),
    );
    assert!(
        facade_model.run(feeds).is_ok(),
        "GlobalAveragePool -> Flatten の run は成功するはず"
    );

    let err = facade_model
        .to_bytes(&OnnxExportOptions::default())
        .unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnsupportedOp { op_type } if op_type == "GlobalAveragePool"),
        "export は最初の未対応 op（GlobalAveragePool）で拒否されるはずだが {err:?}"
    );
}
