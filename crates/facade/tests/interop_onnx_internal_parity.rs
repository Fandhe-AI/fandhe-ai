//! `fandhe_ai::interop::onnx::OnnxModel` と `fandhe_ai_onnx_interop` の
//! 直接呼び出し（`decode_model` → `build_graph` → `interp::run`。
//! `build_model_proto` → `encode_model`）の出力突合テスト（イシュー
//! #2017・AC1 / イシュー #2018・AC1）。
//!
//! **本ファイルは意図的に `fandhe_ai` と `fandhe_ai_onnx_interop` の両方を
//! import する**（facade ラッパーが内部クレートへ委譲するだけの薄い層で
//! あり、コピー・再計算を挟まないことを直接検証するため。
//! `tests/interop_onnx_import.rs`／`tests/interop_onnx_export.rs`
//! 〈facade のみ import〉とは目的が異なる）。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxError, OnnxExportOptions, OnnxModel, OnnxValue};

use fandhe_ai_onnx_interop::onnx::export::{ExportOptions, build_model_proto};
use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, Value};
use fandhe_ai_onnx_interop::onnx::proto::{
    self, AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
    attribute_type, data_type,
};

fn onnx_interop_fixture(rel: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures"
    ))
    .join(rel)
}

fn load_internal_model(rel: &str) -> ModelProto {
    let bytes = std::fs::read(onnx_interop_fixture(rel)).expect("fixture 読み込み失敗");
    proto::decode_model(&bytes).expect("decode は成功するはず")
}

// --- AC1: facade 経由の実行結果が内部クレート直接呼び出しと bit 一致する ---

#[test]
fn model_onnx_facade_matches_internal_direct_call_bit_exact() {
    // facade 経由。
    let facade_model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");

    // 内部クレート直接呼び出し。
    let internal_model = load_internal_model("model.onnx");
    let internal_graph = build_graph(&internal_model).expect("build_graph 成功");

    let samples: [([f32; 2], usize); 4] = [
        ([0.0, 0.0], 0),
        ([1.0, 1.0], 1),
        ([0.3, 0.7], 2),
        ([2.0, -1.0], 3),
    ];

    for (input, _) in samples {
        let mut facade_feeds = HashMap::new();
        facade_feeds.insert(
            "input".to_string(),
            OnnxValue::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
        );
        let facade_result = facade_model.run(facade_feeds).expect("facade run 成功");
        let facade_out = match &facade_result["output"] {
            OnnxValue::F32(t) => t.clone(),
            other => panic!("OnnxValue::F32 を期待したが {other:?}"),
        };

        let mut internal_feeds = HashMap::new();
        internal_feeds.insert(
            "input".to_string(),
            Value::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
        );
        let internal_result =
            interp::run(&internal_graph, internal_feeds).expect("internal run 成功");
        let internal_out = match &internal_result["output"] {
            Value::F32(t) => t.clone(),
            other => panic!("Value::F32 を期待したが {other:?}"),
        };

        assert_eq!(
            facade_out.shape(),
            internal_out.shape(),
            "shape が一致しない: input={input:?}"
        );
        assert!(
            !facade_out.as_slice().unwrap_or(&[]).is_empty(),
            "空虚 pass 防止: 出力要素数が 0"
        );
        for (a, b) in facade_out
            .as_slice()
            .unwrap()
            .iter()
            .zip(internal_out.as_slice().unwrap().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "facade 経由と内部クレート直接呼び出しの出力が bit 不一致: input={input:?} \
                 facade={a} internal={b}"
            );
        }
    }
}

#[test]
fn slice_repro_onnx_facade_matches_internal_direct_call_bit_exact() {
    let facade_model =
        OnnxModel::from_path(onnx_interop_fixture("slice_repro.onnx")).expect("from_path 成功");
    let internal_model = load_internal_model("slice_repro.onnx");
    let internal_graph = build_graph(&internal_model).expect("build_graph 成功");

    // 決定的な入力（乱数生成器を新規導入せず固定値パターンで多要素を
    // 網羅する。5x6 は slice_repro.onnx が要求する input_shape）。
    let flat: Vec<f32> = (0..30).map(|i| (i as f32) * 0.37 - 4.5).collect();

    let mut facade_feeds = HashMap::new();
    facade_feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(flat.clone(), &[5, 6]).unwrap()),
    );
    let facade_result = facade_model.run(facade_feeds).expect("facade run 成功");
    let facade_out = match &facade_result["output"] {
        OnnxValue::F32(t) => t.clone(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    };

    let mut internal_feeds = HashMap::new();
    internal_feeds.insert(
        "x".to_string(),
        Value::F32(Tensor::<f32>::new(flat, &[5, 6]).unwrap()),
    );
    let internal_result = interp::run(&internal_graph, internal_feeds).expect("internal run 成功");
    let internal_out = match &internal_result["output"] {
        Value::F32(t) => t.clone(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    };

    assert_eq!(facade_out.shape(), internal_out.shape());
    assert!(!facade_out.as_slice().unwrap().is_empty());
    for (a, b) in facade_out
        .as_slice()
        .unwrap()
        .iter()
        .zip(internal_out.as_slice().unwrap().iter())
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit 不一致: facade={a} internal={b}"
        );
    }
}

// --- AC2: 壊れた protobuf・未対応 op・未知 dtype の fail-closed 拒否（合成モデル） ---

fn minimal_model_with_node(node: NodeProto, inputs: Vec<&str>, outputs: Vec<&str>) -> ModelProto {
    ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            node: vec![node],
            name: "g".to_string(),
            initializer: vec![],
            input: inputs
                .into_iter()
                .map(|n| ValueInfoProto {
                    name: n.to_string(),
                })
                .collect(),
            output: outputs
                .into_iter()
                .map(|n| ValueInfoProto {
                    name: n.to_string(),
                })
                .collect(),
        }),
    }
}

#[test]
fn synthetic_model_with_unsupported_op_is_rejected_via_facade() {
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n1".to_string(),
        op_type: "LSTM".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let bytes = proto::encode_model(&model);

    let facade_model =
        OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず（構築自体は妥当）");
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    let err = facade_model.run(feeds).unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnsupportedOp { op_type } if op_type == "LSTM"),
        "OnnxError::UnsupportedOp を期待したが {err:?}"
    );
}

#[test]
fn synthetic_model_with_unknown_initializer_data_type_is_rejected_via_facade() {
    // `TensorProto.data_type` に本クレート未対応の値（999）を持つ
    // initializer を含む合成モデル（`GraphError::UnknownDataType` ->
    // `OnnxError::UnsupportedDataType` 写像の固定化）。
    let init = TensorProto {
        name: "w".to_string(),
        data_type: 999,
        dims: vec![1],
        float_data: vec![],
        int64_data: vec![],
        raw_data: vec![],
    };
    let node = NodeProto {
        input: vec!["w".to_string()],
        output: vec!["y".to_string()],
        name: "n_relu".to_string(),
        op_type: "Relu".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            node: vec![node],
            name: "g".to_string(),
            initializer: vec![init],
            input: vec![],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    let bytes = proto::encode_model(&model);

    let err = OnnxModel::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(
            &err,
            OnnxError::UnsupportedDataType { data_type, .. } if *data_type == 999
        ),
        "OnnxError::UnsupportedDataType を期待したが {err:?}"
    );
}

#[test]
fn synthetic_model_with_negative_dim_is_rejected_as_invalid_model_via_facade() {
    // `GraphError::NegativeDim` -> `OnnxError::InvalidModel` へ写像される
    // ことを固定する（UnsupportedDataType 以外の GraphError 全般の扱い）。
    let init = TensorProto {
        name: "w".to_string(),
        data_type: data_type::FLOAT,
        dims: vec![-1],
        float_data: vec![],
        int64_data: vec![],
        raw_data: vec![],
    };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "facade-test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            node: vec![],
            name: "g".to_string(),
            initializer: vec![init],
            input: vec![],
            output: vec![],
        }),
    };
    let bytes = proto::encode_model(&model);

    let err = OnnxModel::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(&err, OnnxError::InvalidModel { .. }),
        "OnnxError::InvalidModel を期待したが {err:?}"
    );
}

// --- F16: `OnnxValue::F16` が facade 経由で到達可能であることの固定化 ---

#[test]
fn synthetic_model_cast_to_float16_yields_onnx_value_f16_via_facade() {
    const ONNX_DATA_TYPE_FLOAT16: i64 = 10;

    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_cast".to_string(),
        op_type: "Cast".to_string(),
        attribute: vec![AttributeProto {
            name: "to".to_string(),
            f: 0.0,
            i: ONNX_DATA_TYPE_FLOAT16,
            s: vec![],
            t: None,
            floats: vec![],
            ints: vec![],
            r#type: attribute_type::INT,
        }],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let bytes = proto::encode_model(&model);

    let facade_model = OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず");
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(vec![1.5], &[1]).unwrap()),
    );
    let result = facade_model.run(feeds).expect("run は成功するはず");
    assert!(
        matches!(&result["y"], OnnxValue::F16(_)),
        "OnnxValue::F16 を期待したが {:?}",
        result["y"]
    );
}

// --- export（イシュー #2018）: facade ↔ 内部クレート直接呼び出し突合 ---

/// 1. facade `to_bytes(default)` == 内部クレート直接呼び出し
/// （`encode_model(&build_model_proto(&build_graph(&decode_model(bytes)),
/// &ExportOptions::default()))`）とバイト完全一致することを固定する
/// （`model.onnx`・`slice_repro.onnx`）。
#[test]
fn to_bytes_matches_internal_direct_call_byte_exact() {
    for fixture in ["model.onnx", "slice_repro.onnx"] {
        let facade_model = OnnxModel::from_path(onnx_interop_fixture(fixture))
            .unwrap_or_else(|e| panic!("{fixture} の from_path が失敗: {e}"));
        let facade_bytes = facade_model
            .to_bytes(&OnnxExportOptions::default())
            .unwrap_or_else(|e| panic!("{fixture} の to_bytes が失敗: {e}"));

        let internal_model = load_internal_model(fixture);
        let internal_graph = build_graph(&internal_model)
            .unwrap_or_else(|e| panic!("{fixture} の build_graph が失敗: {e}"));
        let internal_proto = build_model_proto(&internal_graph, &ExportOptions::default())
            .unwrap_or_else(|e| panic!("{fixture} の build_model_proto が失敗: {e}"));
        let internal_bytes = proto::encode_model(&internal_proto);

        assert_eq!(
            facade_bytes, internal_bytes,
            "{fixture}: facade to_bytes と内部クレート直接呼び出しがバイト不一致"
        );
    }
}

/// 2. `build_graph(decode_model(facade_bytes)) ==
/// build_graph(decode_model(元 bytes))`（`Graph: PartialEq`）に加え、
/// decode 結果の `value_info` が空・全 initializer の `float_data`／
/// `int64_data` が空であることを確認する（export の契約確認。
/// `docs/facade-onnx-export-exposure-decision.md` §4「value_info は常に
/// 空」・`export.rs` の「常に raw_data のみへ書き出す」契約）。
#[test]
fn to_bytes_output_graph_structurally_equal_and_satisfies_export_contract() {
    for fixture in ["model.onnx", "slice_repro.onnx"] {
        let facade_model = OnnxModel::from_path(onnx_interop_fixture(fixture))
            .unwrap_or_else(|e| panic!("{fixture} の from_path が失敗: {e}"));
        let facade_bytes = facade_model
            .to_bytes(&OnnxExportOptions::default())
            .unwrap_or_else(|e| panic!("{fixture} の to_bytes が失敗: {e}"));

        let reexported_model = proto::decode_model(&facade_bytes)
            .unwrap_or_else(|e| panic!("{fixture} の再 decode が失敗: {e}"));
        let reexported_graph = build_graph(&reexported_model)
            .unwrap_or_else(|e| panic!("{fixture} の再 build_graph が失敗: {e}"));

        let original_model = load_internal_model(fixture);
        let original_graph = build_graph(&original_model)
            .unwrap_or_else(|e| panic!("{fixture} の元 build_graph が失敗: {e}"));

        assert_eq!(
            reexported_graph, original_graph,
            "{fixture}: export→decode→build_graph が元 Graph と構造的に一致しない"
        );

        let graph_proto = reexported_model
            .graph
            .as_ref()
            .unwrap_or_else(|| panic!("{fixture}: 再 decode したモデルに graph が無い"));
        assert!(
            graph_proto.value_info.is_empty(),
            "{fixture}: value_info が空ではない（export 契約違反）"
        );
        assert!(
            !graph_proto.initializer.is_empty() || fixture == "slice_repro.onnx",
            "{fixture}: initializer が空虚 pass（少なくとも 1 件は想定）"
        );
        for tensor in &graph_proto.initializer {
            assert!(
                tensor.float_data.is_empty(),
                "{fixture}: initializer '{}' の float_data が空でない（raw_data 限定契約違反）",
                tensor.name
            );
            assert!(
                tensor.int64_data.is_empty(),
                "{fixture}: initializer '{}' の int64_data が空でない（raw_data 限定契約違反）",
                tensor.name
            );
        }
    }
}

/// 3. 既定値ドリフトガード: `OnnxExportOptions::default()` の 2 フィールド
/// == `ExportOptions::default()` の対応フィールド。
#[test]
fn onnx_export_options_default_matches_internal_export_options_default() {
    let facade_default = OnnxExportOptions::default();
    let internal_default = ExportOptions::default();
    assert_eq!(
        facade_default.ir_version, internal_default.ir_version,
        "OnnxExportOptions::default().ir_version が ExportOptions::default() とドリフトしている"
    );
    assert_eq!(
        facade_default.opset_version, internal_default.opset_version,
        "OnnxExportOptions::default().opset_version が ExportOptions::default() とドリフトしている"
    );
}

/// 4. 合成モデル: 未対応 op_type → `UnsupportedOp`、対応 op だが非既定
/// domain → `UnsupportedOp`（`op_type` が domain 修飾形）。いずれも拒否は
/// import 時ではなく export 時に起きることを assert する。
#[test]
fn synthetic_model_export_rejects_unsupported_op_and_non_default_domain() {
    // 4a. 未対応 op_type（allowlist 外）。
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_unsupported".to_string(),
        op_type: "LSTM".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let bytes = proto::encode_model(&model);
    let facade_model = OnnxModel::from_bytes(&bytes)
        .expect("import 時点では成功するはず（未対応判定は export 時）");
    let err = facade_model
        .to_bytes(&OnnxExportOptions::default())
        .unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnsupportedOp { op_type } if op_type == "LSTM"),
        "未対応 op_type で OnnxError::UnsupportedOp を期待したが {err:?}"
    );

    // 4b. 対応 op（Relu）だが非既定 domain。
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_custom_domain".to_string(),
        op_type: "Relu".to_string(),
        attribute: vec![],
        domain: "custom.domain".to_string(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let bytes = proto::encode_model(&model);
    let facade_model = OnnxModel::from_bytes(&bytes)
        .expect("import 時点では成功するはず（domain 検査も export 時のみ）");
    let err = facade_model
        .to_bytes(&OnnxExportOptions::default())
        .unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnsupportedOp { op_type } if op_type == "custom.domain::Relu"),
        "非既定 domain で domain 修飾形の OnnxError::UnsupportedOp を期待したが {err:?}"
    );
}

/// 5. 非既定 options が内部 `ExportOptions { ir_version, opset_version,
/// ..default }` の出力とバイト一致する。
#[test]
fn non_default_options_match_internal_export_options_byte_exact() {
    let facade_model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");
    let mut facade_options = OnnxExportOptions::default();
    facade_options.ir_version = 9;
    facade_options.opset_version = 18;
    let facade_bytes = facade_model
        .to_bytes(&facade_options)
        .expect("非既定 options での to_bytes 成功");

    let internal_model = load_internal_model("model.onnx");
    let internal_graph = build_graph(&internal_model).expect("build_graph 成功");
    let internal_options = ExportOptions {
        ir_version: 9,
        opset_version: 18,
        ..ExportOptions::default()
    };
    let internal_proto =
        build_model_proto(&internal_graph, &internal_options).expect("build_model_proto 成功");
    let internal_bytes = proto::encode_model(&internal_proto);

    assert_eq!(
        facade_bytes, internal_bytes,
        "非既定 options での facade to_bytes と内部クレート直接呼び出しがバイト不一致"
    );
}
