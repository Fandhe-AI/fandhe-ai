//! `onnx::export` の単体テスト（イシュー #1772）。
//!
//! 本モジュールが検証するのは「内部グラフ表現 `Graph` から `GraphProto`／
//! `ModelProto` への構造的な組み立てが正しいか」のみである。op_type ごとの
//! 意味論（内部 op -> `NodeProto` の属性マッピング）は #1773 のスコープであり、
//! ここでは対象にしない（`onnx::export` モジュール冒頭コメント参照）。
//! import -> export -> import の構造一致 roundtrip テスト・未対応 op の
//! fail-closed 確認という総合テストは `tests/onnx_export_roundtrip.rs`
//! （#1774）が担う。ここでは
//! (a) 既存 fixture（`model.onnx`）を decode -> build_graph -> export -> 再 decode
//! -> build_graph した結果が元の `Graph` と一致すること、(b) `encode_tensor`／
//! `decode_tensor` の dtype 網羅往復、(c) エラーパス、(d) 出力の決定性、を
//! 単体レベルで確認する。

use onnx_interop::onnx::export::{ExportError, ExportOptions, build_model_proto, encode_tensor};
use onnx_interop::onnx::graph::{RawTensor, build_graph};
use onnx_interop::onnx::proto::{GraphProto, ModelProto, OperatorSetIdProto};
use prost::Message;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_model(name: &str) -> ModelProto {
    let bytes = std::fs::read(fixture_path(name))
        .unwrap_or_else(|e| panic!("fixture 読み込み失敗 {name}: {e}"));
    ModelProto::decode(bytes.as_slice()).unwrap_or_else(|e| panic!("decode 失敗 {name}: {e}"))
}

/// `graph::decode_tensor` はクレート内部限定（`pub(crate)`）で本クレート外の
/// integration test からは直接呼べないため、公開 API である `build_graph` を
/// 経由してテンソルを decode する（`encode_tensor` -> 1 initializer だけの
/// 最小 `ModelProto` を組み立てて `build_graph` に通し、復号結果を取り出す）。
/// これにより往復（encode -> decode）を公開 API のみで検証できる。
fn roundtrip_via_build_graph(name: &str, tensor: &RawTensor) -> RawTensor {
    let proto = encode_tensor(name, tensor).expect("encode は成功するはず");
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![],
            name: "g".to_string(),
            initializer: vec![proto],
            input: vec![],
            output: vec![],
            value_info: vec![],
        }),
        opset_import: Vec::new(),
    };
    let mut graph = build_graph(&model).expect("build_graph は成功するはず");
    graph
        .initializers
        .remove(name)
        .expect("initializer が復号されているはず")
}

// --- (a) 既存 fixture のラウンドトリップ ---

#[test]
fn model_onnx_roundtrips_through_export_structurally() {
    let model = load_model("model.onnx");

    // decode 経路の実フィクスチャ自体（本モジュールの export を一切経由しない
    // `ModelProto`）に対して opset_import／value_info を直接検証する。以下の
    // encode -> decode 経路の検証は同一 prost 構造体を encode/decode 双方が
    // 共有するため、tag 番号（opset_import=8／value_info=13）を取り違えても
    // 自己整合的に pass してしまい得る（レビュー指摘）。ここで実ファイルの
    // decode 結果を直接 assert することで、tag 番号自体が正しく解釈されて
    // いることを本クレートの export に依存せず確認する。
    assert_eq!(
        model.opset_import,
        vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        "model.onnx（PyTorch 2.12.1 export）実測値と一致するはず（ExportOptions::default() のコメント参照）"
    );
    assert!(
        model
            .graph
            .as_ref()
            .expect("model.onnx に graph はあるはず")
            .value_info
            .is_empty(),
        "model.onnx（PyTorch 2.12.1 export）実測値は value_info を持たない"
    );

    let graph = build_graph(&model).expect("build_graph は model.onnx で成功するはず");

    let options = ExportOptions::default();
    let exported = build_model_proto(&graph, &options).expect("build_model_proto は成功するはず");

    // opset_import が既定オプション（フィクスチャ実測値と一致）で書き込まれて
    // いることを確認する。
    assert_eq!(exported.opset_import.len(), 1);
    assert_eq!(exported.opset_import[0].domain, "");
    assert_eq!(exported.opset_import[0].version, 17);
    assert_eq!(exported.ir_version, 8);

    // encode -> decode（実際の protobuf バイト列を経由）して再度 build_graph する。
    let bytes = exported.encode_to_vec();
    let redecoded = ModelProto::decode(bytes.as_slice()).expect("再 decode は成功するはず");
    let rebuilt_graph = build_graph(&redecoded).expect("再構築した Graph も妥当なはず");

    // 再デコードした opset_import も確認（decode 経路でも読めることの確認）。
    assert_eq!(redecoded.opset_import.len(), 1);
    assert_eq!(redecoded.opset_import[0].domain, "");
    assert_eq!(redecoded.opset_import[0].version, 17);

    // 元の Graph と構造一致（nodes・initializers・inputs・outputs）。
    assert_eq!(graph, rebuilt_graph);
}

// --- (b) encode_tensor／decode_tensor の往復（dtype 網羅） ---

#[test]
fn encode_tensor_f32_roundtrips_bit_exact_including_nan_and_denormals() {
    let data = vec![
        0.0_f32,
        -0.0_f32,
        1.5_f32,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE / 2.0, // 非正規化数
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let tensor = RawTensor::F32 {
        data: data.clone(),
        shape: vec![data.len() as i64],
    };
    let proto = encode_tensor("t", &tensor).expect("encode は成功するはず");
    assert!(proto.float_data.is_empty(), "float_data は常に空のはず");
    assert!(proto.int64_data.is_empty(), "int64_data は常に空のはず");

    let decoded = roundtrip_via_build_graph("t", &tensor);
    match decoded {
        RawTensor::F32 {
            data: decoded_data, ..
        } => {
            assert_eq!(decoded_data.len(), data.len());
            for (a, b) in data.iter().zip(decoded_data.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bit 完全一致のはず（NaN payload 込み）: {a} vs {b}"
                );
            }
        }
        other => panic!("F32 が期待されたが {other:?} だった"),
    }
}

#[test]
fn encode_tensor_i64_roundtrips_bit_exact() {
    let data = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let tensor = RawTensor::I64 {
        data: data.clone(),
        shape: vec![data.len() as i64],
    };
    let _proto = encode_tensor("t", &tensor).expect("encode は成功するはず");
    let decoded = roundtrip_via_build_graph("t", &tensor);
    match decoded {
        RawTensor::I64 {
            data: decoded_data, ..
        } => assert_eq!(decoded_data, data),
        other => panic!("I64 が期待されたが {other:?} だった"),
    }
}

#[test]
fn encode_tensor_bool_roundtrips() {
    let data = vec![true, false, true, true, false];
    let tensor = RawTensor::Bool {
        data: data.clone(),
        shape: vec![data.len() as i64],
    };
    let proto = encode_tensor("t", &tensor).expect("encode は成功するはず");
    assert_eq!(proto.raw_data, vec![1u8, 0, 1, 1, 0]);
    let decoded = roundtrip_via_build_graph("t", &tensor);
    match decoded {
        RawTensor::Bool {
            data: decoded_data, ..
        } => assert_eq!(decoded_data, data),
        other => panic!("Bool が期待されたが {other:?} だった"),
    }
}

#[test]
fn encode_tensor_f16_roundtrips_bit_exact_including_nan() {
    let data = vec![
        half::f16::from_f32(0.0),
        half::f16::from_f32(-0.0),
        half::f16::from_f32(1.5),
        half::f16::NAN,
        half::f16::INFINITY,
        half::f16::NEG_INFINITY,
    ];
    let tensor = RawTensor::F16 {
        data: data.clone(),
        shape: vec![data.len() as i64],
    };
    let _proto = encode_tensor("t", &tensor).expect("encode は成功するはず");
    let decoded = roundtrip_via_build_graph("t", &tensor);
    match decoded {
        RawTensor::F16 {
            data: decoded_data, ..
        } => {
            assert_eq!(decoded_data.len(), data.len());
            for (a, b) in data.iter().zip(decoded_data.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bit 完全一致のはず: {a:?} vs {b:?}"
                );
            }
        }
        other => panic!("F16 が期待されたが {other:?} だった"),
    }
}

#[test]
fn encode_tensor_empty_and_scalar_shapes_roundtrip() {
    // 空テンソル（shape = [0]）。
    let empty = RawTensor::F32 {
        data: vec![],
        shape: vec![0],
    };
    let proto = encode_tensor("empty", &empty).expect("空テンソルの encode は成功するはず");
    assert!(proto.raw_data.is_empty());
    let decoded = roundtrip_via_build_graph("empty", &empty);
    assert_eq!(
        decoded,
        RawTensor::F32 {
            data: vec![],
            shape: vec![0],
        }
    );

    // スカラー（shape = []、要素数 1）。
    let scalar = RawTensor::F32 {
        data: vec![3.25],
        shape: vec![],
    };
    let proto = encode_tensor("scalar", &scalar).expect("スカラーの encode は成功するはず");
    assert_eq!(proto.raw_data.len(), 4);
    let decoded = roundtrip_via_build_graph("scalar", &scalar);
    assert_eq!(
        decoded,
        RawTensor::F32 {
            data: vec![3.25],
            shape: vec![],
        }
    );
}

// --- (c) エラーパス ---

#[test]
fn encode_tensor_rejects_shape_data_mismatch() {
    // shape の期待要素数（2）と実データ長（3）が食い違う手組みデータ。
    let tensor = RawTensor::F32 {
        data: vec![1.0, 2.0, 3.0],
        shape: vec![2],
    };
    let err = encode_tensor("mismatched", &tensor).expect_err("食い違いは拒否されるはず");
    assert_eq!(
        err,
        ExportError::ShapeDataMismatch {
            tensor_name: "mismatched".to_string(),
            expected_elements: 2,
            actual_elements: 3,
        }
    );
}

#[test]
fn encode_tensor_rejects_negative_dim() {
    let tensor = RawTensor::F32 {
        data: vec![],
        shape: vec![-1],
    };
    let err = encode_tensor("neg", &tensor).expect_err("負の dim は拒否されるはず");
    assert_eq!(
        err,
        ExportError::NegativeDim {
            tensor_name: "neg".to_string(),
            dim: -1,
        }
    );
}

// --- (d) 決定性（HashMap 走査順に依存しないことの直接検証） ---

#[test]
fn build_model_proto_output_is_deterministic_across_multiple_initializers() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    // 複数 initializer を持つことを前提とするテスト（model.onnx は Gemm x2 の
    // 重み・バイアスを持つため initializer が複数存在する）。
    assert!(
        graph.initializers.len() > 1,
        "本テストは複数 initializer を前提とする"
    );

    let options = ExportOptions::default();
    let first = build_model_proto(&graph, &options).expect("1 回目の export は成功するはず");
    let second = build_model_proto(&graph, &options).expect("2 回目の export は成功するはず");

    let first_bytes = first.encode_to_vec();
    let second_bytes = second.encode_to_vec();
    assert_eq!(
        first_bytes, second_bytes,
        "同一 Graph からの export は毎回同じバイト列になるはず（initializer 名ソート）"
    );

    // initializer が名前順で並んでいることも直接確認する。
    let names: Vec<&str> = first
        .graph
        .as_ref()
        .expect("graph はあるはず")
        .initializer
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let mut sorted_names = names.clone();
    sorted_names.sort();
    assert_eq!(names, sorted_names, "initializer は名前順に並んでいるはず");
}

#[test]
fn build_model_proto_value_info_is_always_empty() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    let exported =
        build_model_proto(&graph, &ExportOptions::default()).expect("export は成功するはず");
    assert!(
        exported
            .graph
            .expect("graph はあるはず")
            .value_info
            .is_empty(),
        "value_info は常に空という契約（Graph が型／形状情報を保持しないため）"
    );
}
