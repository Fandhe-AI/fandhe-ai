//! `decode -> build_graph -> build_model_proto -> encode -> decode -> build_graph`
//! という総合 roundtrip の構造一致・数値一致テスト（イシュー #1774・親 #1653）。
//!
//! ## 既存テストとの切り分け
//!
//! - `tests/onnx_export.rs`（#1772）は `Graph -> GraphProto`／`ModelProto` への
//!   **組み立て自体**（`build_model_proto`・`encode_tensor`）の単体テストであり、
//!   fixture roundtrip（`model_onnx_roundtrips_through_export_structurally`）は
//!   既存の `Graph` の `PartialEq`（f32 の IEEE `==` に依存し NaN ≠ NaN・
//!   `-0.0 == 0.0`）に頼った 1 点のみ。
//! - `tests/onnx_export_ops.rs`（#1773）は内部 op（`ExportOp`）単位の意味論的
//!   対称性・`check_exportable` の allowlist／domain 検査を手組み `Graph` で
//!   単体検証する。
//! - 本ファイルは (a) 実 fixture（decode 由来の `NodeProto`・属性を含む）の
//!   import -> export -> import が **フィールド単位＋bit 同一**で構造一致する
//!   こと、(b) export したモデルの `interp::run` 結果が import 元と **bit 同一**
//!   であること、(c) 未対応 op・非既定 domain を含む**モデル**（decode 経路を
//!   通ったもの）の export が `ExportError::UnsupportedOp` で fail-closed に
//!   なること、の 3 点を総合的に固定する（`export.rs` モジュール冒頭コメントの
//!   「import -> export -> import の構造一致 roundtrip テスト・未対応 op の
//!   fail-closed 確認は #1774 のスコープ」を実装するファイル）。
//!
//! ## 判定式についての注記（REQ-2／REQ-7 との混同禁止）
//!
//! 本ファイルの比較はすべて **bit 同一**（`to_bits()` 一致、整数／真偽値は
//! `==`）である。`.claude/rules/coding-rust.md` の REQ-2 バックエンド間数値
//! 一致複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）とも、
//! `tests/onnx_interp.rs` 等が使う REQ-7 事前固定判定式
//! （`abs_err / (|ref| + 1e-6) <= 1e-3`）とも別軸の契約であり、tolerance は
//! 導入も変更もしない。ここで確認するのは「同一クレート内での import/export
//! の往復が値を一切変えない」という決定的な性質であって、外部参照実装との
//! 数値近似ではない。
//!
//! ## 契約根拠
//!
//! `export.rs` モジュール冒頭コメントの契約（`raw_data` のみへ書き出す・
//! initializer 名ソートで決定的・`value_info` 常に空）に依拠する。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use fandhe_ai_tensor_core::Tensor;
use onnx_interop::onnx::export::{
    ExportError, ExportOptions, SUPPORTED_OP_TYPES, build_model_proto, encode_tensor,
};
use onnx_interop::onnx::graph::{Graph, RawTensor, build_graph};
use onnx_interop::onnx::interp::{Value, run};
use onnx_interop::onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, ValueInfoProto,
};
use prost::Message;
use serde::Deserialize;

// ---- フィクスチャ読み込みヘルパ（既存テストと同形） ----

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn onnx_reference_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/onnx-reference")
        .join(name)
}

fn load_model(name: &str) -> ModelProto {
    let bytes = std::fs::read(fixture_path(name))
        .unwrap_or_else(|e| panic!("fixture 読み込み失敗 {name}: {e}"));
    ModelProto::decode(bytes.as_slice()).unwrap_or_else(|e| panic!("decode 失敗 {name}: {e}"))
}

// ---- roundtrip ヘルパ ----

/// `Graph` を export -> encode -> decode -> import した結果一式を返す。
/// `(exported ModelProto, そのバイト列, 再構築 Graph)`。
fn export_roundtrip(graph: &Graph) -> (ModelProto, Vec<u8>, Graph) {
    let exported = build_model_proto(graph, &ExportOptions::default())
        .expect("build_model_proto は成功するはず");
    let bytes = exported.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("再 decode は成功するはず");
    let rebuilt = build_graph(&decoded).expect("再構築した Graph も妥当なはず");
    (exported, bytes, rebuilt)
}

fn exported_bytes(graph: &Graph) -> Vec<u8> {
    build_model_proto(graph, &ExportOptions::default())
        .expect("build_model_proto は成功するはず")
        .encode_to_vec()
}

/// export の結果が不動点であることを確認する: `export(G)` のバイト列と
/// `export(import(export(G)))` のバイト列が完全一致するはず（`raw_data` のみ・
/// initializer 名ソートという決定的な組み立て契約から、2 回目以降の export は
/// 元の `G` のバイト列と一致するとは限らないが 1 回 export した結果に対しては
/// 不動点になる。`export.rs` モジュール冒頭コメント参照）。
/// 元 fixture ファイルのバイト列との一致は assert しない（typed data ->
/// `raw_data` 正規化で正当に変わりうるため。スコープ外整理は本ファイル末尾）。
fn assert_export_is_fixed_point(graph: &Graph) {
    let bytes1 = exported_bytes(graph);
    let decoded1 = ModelProto::decode(bytes1.as_slice()).expect("再 decode は成功するはず");

    let g1 = decoded1.graph.as_ref().expect("graph はあるはず");
    assert!(
        g1.value_info.is_empty(),
        "value_info は常に空のはず（export.rs 契約）"
    );
    for init in &g1.initializer {
        assert!(
            init.float_data.is_empty(),
            "float_data は常に空のはず（export.rs 契約, tensor={}）",
            init.name
        );
        assert!(
            init.int64_data.is_empty(),
            "int64_data は常に空のはず（export.rs 契約, tensor={}）",
            init.name
        );
    }

    let rebuilt1 = build_graph(&decoded1).expect("再構築した Graph も妥当なはず");
    let bytes2 = exported_bytes(&rebuilt1);
    assert_eq!(
        bytes1, bytes2,
        "export は不動点になるはず（initializer 名ソートで決定的な組み立て契約）"
    );
}

/// `check_exportable`（`export_ops`）を経由させずに `Graph` から `ModelProto`
/// を組み立てる（本番の `build_model_proto` 相当だが検査ステップを意図的に
/// スキップする）。テスト 8〜10 が「未対応 op・非既定 domain を含む**モデル**」
/// （decode 経路を通った `ModelProto`）を用意するための専用ヘルパであり、
/// 本番経路（`export.rs::build_model_proto`）は必ず検査を経由するため、この
/// バイパスは test-only。
fn to_model_proto_bypassing_check(graph: &Graph, options: &ExportOptions) -> ModelProto {
    let mut entries: Vec<(&String, &RawTensor)> = graph.initializers.iter().collect();
    entries.sort_by_key(|(name, _)| *name);
    let initializer = entries
        .into_iter()
        .map(|(name, tensor)| encode_tensor(name, tensor).expect("encode は成功するはず"))
        .collect();
    let input = graph
        .inputs
        .iter()
        .map(|name| ValueInfoProto { name: name.clone() })
        .collect();
    let output = graph
        .outputs
        .iter()
        .map(|name| ValueInfoProto { name: name.clone() })
        .collect();
    ModelProto {
        ir_version: options.ir_version,
        producer_name: options.producer_name.clone(),
        graph: Some(GraphProto {
            node: graph.nodes.clone(),
            name: options.graph_name.clone(),
            initializer,
            input,
            output,
            value_info: Vec::new(),
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: options.opset_domain.clone(),
            version: options.opset_version,
        }],
    }
}

// ---- bit 同一比較ヘルパ ----

fn assert_raw_tensor_bit_exact(name: &str, a: &RawTensor, b: &RawTensor) {
    match (a, b) {
        (
            RawTensor::F32 {
                data: ad,
                shape: r#as,
            },
            RawTensor::F32 {
                data: bd,
                shape: bs,
            },
        ) => {
            assert_eq!(r#as, bs, "shape 不一致 (initializer={name})");
            assert_eq!(ad.len(), bd.len(), "データ長不一致 (initializer={name})");
            for (i, (x, y)) in ad.iter().zip(bd.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "bit 不一致 (initializer={name}[{i}]): {x} vs {y}"
                );
            }
        }
        (
            RawTensor::I64 {
                data: ad,
                shape: r#as,
            },
            RawTensor::I64 {
                data: bd,
                shape: bs,
            },
        ) => {
            assert_eq!(r#as, bs, "shape 不一致 (initializer={name})");
            assert_eq!(ad, bd, "データ不一致 (initializer={name})");
        }
        (
            RawTensor::Bool {
                data: ad,
                shape: r#as,
            },
            RawTensor::Bool {
                data: bd,
                shape: bs,
            },
        ) => {
            assert_eq!(r#as, bs, "shape 不一致 (initializer={name})");
            assert_eq!(ad, bd, "データ不一致 (initializer={name})");
        }
        (
            RawTensor::F16 {
                data: ad,
                shape: r#as,
            },
            RawTensor::F16 {
                data: bd,
                shape: bs,
            },
        ) => {
            assert_eq!(r#as, bs, "shape 不一致 (initializer={name})");
            assert_eq!(ad.len(), bd.len(), "データ長不一致 (initializer={name})");
            for (i, (x, y)) in ad.iter().zip(bd.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "bit 不一致 (initializer={name}[{i}]): {x:?} vs {y:?}"
                );
            }
        }
        _ => panic!("variant 不一致 (initializer={name}): {a:?} vs {b:?}"),
    }
}

fn assert_attribute_bit_exact(node_name: &str, a: &AttributeProto, b: &AttributeProto) {
    assert_eq!(a.name, b.name, "属性名不一致 (node={node_name})");
    let attr_name = &a.name;
    assert_eq!(
        a.r#type, b.r#type,
        "属性型不一致 (node={node_name}, attr={attr_name})"
    );
    assert_eq!(
        a.i, b.i,
        "属性 i 不一致 (node={node_name}, attr={attr_name})"
    );
    assert_eq!(
        a.ints, b.ints,
        "属性 ints 不一致 (node={node_name}, attr={attr_name})"
    );
    assert_eq!(
        a.s, b.s,
        "属性 s 不一致 (node={node_name}, attr={attr_name})"
    );
    assert_eq!(
        a.f.to_bits(),
        b.f.to_bits(),
        "属性 f が bit 不一致 (node={node_name}, attr={attr_name}): {} vs {}",
        a.f,
        b.f
    );
    assert_eq!(
        a.floats.len(),
        b.floats.len(),
        "属性 floats 長さ不一致 (node={node_name}, attr={attr_name})"
    );
    for (i, (x, y)) in a.floats.iter().zip(b.floats.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "属性 floats[{i}] が bit 不一致 (node={node_name}, attr={attr_name}): {x} vs {y}"
        );
    }
    match (&a.t, &b.t) {
        (None, None) => {}
        (Some(at), Some(bt)) => {
            assert_eq!(
                at.dims, bt.dims,
                "属性 t.dims 不一致 (node={node_name}, attr={attr_name})"
            );
            assert_eq!(
                at.data_type, bt.data_type,
                "属性 t.data_type 不一致 (node={node_name}, attr={attr_name})"
            );
            assert_eq!(
                at.name, bt.name,
                "属性 t.name 不一致 (node={node_name}, attr={attr_name})"
            );
            assert_eq!(
                at.raw_data, bt.raw_data,
                "属性 t.raw_data 不一致 (node={node_name}, attr={attr_name})"
            );
            assert_eq!(
                at.int64_data, bt.int64_data,
                "属性 t.int64_data 不一致 (node={node_name}, attr={attr_name})"
            );
            assert_eq!(
                at.float_data.len(),
                bt.float_data.len(),
                "属性 t.float_data 長さ不一致 (node={node_name}, attr={attr_name})"
            );
            for (i, (x, y)) in at.float_data.iter().zip(bt.float_data.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "属性 t.float_data[{i}] が bit 不一致 (node={node_name}, attr={attr_name})"
                );
            }
        }
        _ => panic!("属性 t の Some/None が不一致 (node={node_name}, attr={attr_name})"),
    }
}

/// 2 つの `Graph` がフィールド単位・bit 同一で構造一致することを確認する。
///
/// NaN を含む initializer が存在する場合、末尾の `assert_eq!(a, b)`（既存
/// `Graph` の `PartialEq`。f32 の IEEE `==` に依存し NaN ≠ NaN）が false
/// negative になりうるため、本ヘルパは NaN を含まないフィクスチャ
/// （`model.onnx`／`slice_repro.onnx`）限定で使う。NaN を含む往復の bit 一致は
/// `typed_data_initializer_roundtrips_to_raw_data_bit_exact` が個別に
/// `to_bits()` で担保する。
fn assert_graph_structurally_identical(a: &Graph, b: &Graph) {
    assert_eq!(a.nodes.len(), b.nodes.len(), "node 数不一致");
    for (an, bn) in a.nodes.iter().zip(b.nodes.iter()) {
        assert_eq!(an.name, bn.name, "node.name 不一致");
        assert_eq!(
            an.op_type, bn.op_type,
            "node.op_type 不一致 (node={})",
            an.name
        );
        assert_eq!(
            an.domain, bn.domain,
            "node.domain 不一致 (node={})",
            an.name
        );
        assert_eq!(an.input, bn.input, "node.input 不一致 (node={})", an.name);
        assert_eq!(
            an.output, bn.output,
            "node.output 不一致 (node={})",
            an.name
        );
        assert_eq!(
            an.attribute.len(),
            bn.attribute.len(),
            "attribute 数不一致 (node={})",
            an.name
        );
        for (aa, ba) in an.attribute.iter().zip(bn.attribute.iter()) {
            assert_attribute_bit_exact(&an.name, aa, ba);
        }
    }

    let a_keys: HashSet<&String> = a.initializers.keys().collect();
    let b_keys: HashSet<&String> = b.initializers.keys().collect();
    assert_eq!(a_keys, b_keys, "initializer キー集合不一致");
    for name in a_keys {
        assert_raw_tensor_bit_exact(name, &a.initializers[name], &b.initializers[name]);
    }

    assert_eq!(a.inputs, b.inputs, "graph.inputs 不一致");
    assert_eq!(a.outputs, b.outputs, "graph.outputs 不一致");

    // 上記フィールド単位比較を経てなお、既存 `PartialEq` とも矛盾しないことの
    // 最終確認（NaN を含まない前提。本関数冒頭コメント参照）。
    assert_eq!(a, b, "Graph の PartialEq が不一致");
}

fn assert_value_bit_exact(output_name: &str, a: &Value, b: &Value) {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => {
            assert_eq!(x.shape(), y.shape(), "shape 不一致 (output={output_name})");
            let xc = x.contiguous();
            let yc = y.contiguous();
            let xs = xc.as_slice().expect("contiguous のはず");
            let ys = yc.as_slice().expect("contiguous のはず");
            assert_eq!(xs.len(), ys.len(), "データ長不一致 (output={output_name})");
            for (i, (p, q)) in xs.iter().zip(ys.iter()).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "bit 不一致 (output={output_name}[{i}]): {p} vs {q}"
                );
            }
        }
        (Value::I64(x), Value::I64(y)) => {
            assert_eq!(x.shape(), y.shape(), "shape 不一致 (output={output_name})");
            let xc = x.contiguous();
            let yc = y.contiguous();
            assert_eq!(
                xc.as_slice().expect("contiguous のはず"),
                yc.as_slice().expect("contiguous のはず"),
                "データ不一致 (output={output_name})"
            );
        }
        (Value::Bool(x), Value::Bool(y)) => {
            assert_eq!(x.shape(), y.shape(), "shape 不一致 (output={output_name})");
            let xc = x.contiguous();
            let yc = y.contiguous();
            assert_eq!(
                xc.as_slice().expect("contiguous のはず"),
                yc.as_slice().expect("contiguous のはず"),
                "データ不一致 (output={output_name})"
            );
        }
        (Value::F16(x), Value::F16(y)) => {
            assert_eq!(x.shape(), y.shape(), "shape 不一致 (output={output_name})");
            let xc = x.contiguous();
            let yc = y.contiguous();
            let xs = xc.as_slice().expect("contiguous のはず");
            let ys = yc.as_slice().expect("contiguous のはず");
            assert_eq!(xs.len(), ys.len(), "データ長不一致 (output={output_name})");
            for (i, (p, q)) in xs.iter().zip(ys.iter()).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "bit 不一致 (output={output_name}[{i}]): {p:?} vs {q:?}"
                );
            }
        }
        _ => panic!("variant 不一致 (output={output_name}): {a:?} vs {b:?}"),
    }
}

/// 元 `Graph` と rebuilt `Graph` の `run` 結果が bit 同一であることを確認する。
/// `run` は `feeds` を値で受け取る（消費する）ため `build_feeds` を都度呼ぶ。
fn assert_run_bit_identical(
    original: &Graph,
    rebuilt: &Graph,
    build_feeds: impl Fn() -> HashMap<String, Value>,
) {
    let result_a = run(original, build_feeds()).expect("original の run は成功するはず");
    let result_b = run(rebuilt, build_feeds()).expect("rebuilt の run は成功するはず");

    let mut keys_a: Vec<&String> = result_a.keys().collect();
    keys_a.sort();
    let mut expected_outputs: Vec<&String> = original.outputs.iter().collect();
    expected_outputs.sort();
    assert_eq!(
        keys_a, expected_outputs,
        "original の run 結果キーが graph.outputs と一致しない"
    );

    for name in &original.outputs {
        let a = result_a
            .get(name)
            .unwrap_or_else(|| panic!("original の run 結果に出力 {name} がない"));
        let b = result_b
            .get(name)
            .unwrap_or_else(|| panic!("rebuilt の run 結果に出力 {name} がない"));
        assert_value_bit_exact(name, a, b);
    }
}

// ==== (A) fixture roundtrip 構造一致（bit 同一） ====

#[test]
fn model_onnx_roundtrip_is_structurally_identical_bit_exact() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    let (_exported, _bytes, rebuilt) = export_roundtrip(&graph);
    assert_graph_structurally_identical(&graph, &rebuilt);

    // fixture README（tests/fixtures/README.md）の既知構造を rebuilt 側で直接
    // 固定する（比較が空虚でないことの担保）。
    let op_types: Vec<&str> = rebuilt.nodes.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(
        op_types,
        vec!["Gemm", "Relu", "Gemm", "Relu", "Gemm", "Sigmoid"]
    );
    assert_eq!(rebuilt.initializers.len(), 6);
    match &rebuilt.initializers["fc1.weight"] {
        RawTensor::F32 { shape, .. } => assert_eq!(shape, &vec![8, 2]),
        other => panic!("fc1.weight は F32 のはず: {other:?}"),
    }
    match &rebuilt.initializers["fc1.bias"] {
        RawTensor::F32 { shape, .. } => assert_eq!(shape, &vec![8]),
        other => panic!("fc1.bias は F32 のはず: {other:?}"),
    }
    match &rebuilt.initializers["fc3.weight"] {
        RawTensor::F32 { shape, .. } => assert_eq!(shape, &vec![1, 8]),
        other => panic!("fc3.weight は F32 のはず: {other:?}"),
    }
    match &rebuilt.initializers["fc3.bias"] {
        RawTensor::F32 { shape, .. } => assert_eq!(shape, &vec![1]),
        other => panic!("fc3.bias は F32 のはず: {other:?}"),
    }
}

#[test]
fn slice_repro_onnx_roundtrip_is_structurally_identical_bit_exact() {
    let model = load_model("slice_repro.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    let (_exported, _bytes, rebuilt) = export_roundtrip(&graph);
    assert_graph_structurally_identical(&graph, &rebuilt);

    // fixture README の既知構造を rebuilt 側で直接固定する。
    let op_types: Vec<&str> = rebuilt.nodes.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(
        op_types,
        vec!["Shape", "Gather", "Unsqueeze", "Concat", "Slice"]
    );
    assert_eq!(rebuilt.initializers.len(), 4);
    match &rebuilt.initializers["const_axes"] {
        RawTensor::I64 { data, shape } => {
            assert_eq!(shape, &vec![2]);
            assert_eq!(data, &vec![0, 1]);
        }
        other => panic!("const_axes は I64 のはず: {other:?}"),
    }
    match &rebuilt.initializers["const_4"] {
        RawTensor::I64 { data, shape } => {
            assert_eq!(shape, &vec![1]);
            assert_eq!(data, &vec![4]);
        }
        other => panic!("const_4 は I64 のはず: {other:?}"),
    }
    match &rebuilt.initializers["const_starts"] {
        RawTensor::I64 { data, shape } => {
            assert_eq!(shape, &vec![2]);
            assert_eq!(data, &vec![0, 0]);
        }
        other => panic!("const_starts は I64 のはず: {other:?}"),
    }
    match &rebuilt.initializers["const_gather_idx"] {
        RawTensor::I64 { data, shape } => {
            assert_eq!(shape, &vec![1]);
            assert_eq!(data, &vec![0]);
        }
        other => panic!("const_gather_idx は I64 のはず: {other:?}"),
    }
}

#[test]
fn model_onnx_exported_model_is_a_fixed_point_of_export() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    assert_export_is_fixed_point(&graph);
}

#[test]
fn slice_repro_onnx_exported_model_is_a_fixed_point_of_export() {
    let model = load_model("slice_repro.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    assert_export_is_fixed_point(&graph);
}

#[test]
fn typed_data_initializer_roundtrips_to_raw_data_bit_exact() {
    use onnx_interop::onnx::proto::TensorProto;
    use onnx_interop::onnx::proto::data_type;

    // 手組み `ModelProto`: `float_data`（NaN・-0.0・非正規化数込み）ベースの F32
    // initializer と `int64_data` ベースの I64 initializer（未使用でも
    // `build_graph` は initializer の消費を要求しない。`graph.rs::build_graph`
    // 参照）、それらを消費する Relu 1 node。`decode_tensor` の raw_data 優先・
    // typed data フォールバック解決順序（`graph.rs` モジュールコメント）を
    // typed data 側から検証する。
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![NodeProto {
                input: vec!["w".to_string()],
                output: vec!["y".to_string()],
                name: "relu1".to_string(),
                op_type: "Relu".to_string(),
                attribute: Vec::new(),
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![
                TensorProto {
                    dims: vec![6],
                    data_type: data_type::FLOAT,
                    float_data: vec![
                        0.0_f32,
                        -0.0_f32,
                        1.5_f32,
                        f32::NAN,
                        f32::MIN_POSITIVE / 2.0,
                        f32::INFINITY,
                    ],
                    int64_data: Vec::new(),
                    name: "w".to_string(),
                    raw_data: Vec::new(),
                },
                TensorProto {
                    dims: vec![3],
                    data_type: data_type::INT64,
                    float_data: Vec::new(),
                    int64_data: vec![i64::MIN, 0, i64::MAX],
                    name: "i".to_string(),
                    raw_data: Vec::new(),
                },
            ],
            input: Vec::new(),
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
            value_info: Vec::new(),
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
    };

    let bytes = model.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("decode は成功するはず");
    let graph1 = build_graph(&decoded).expect("build_graph は成功するはず（typed data 経路）");

    match &graph1.initializers["w"] {
        RawTensor::F32 { data, .. } => {
            let expected = [
                0.0_f32,
                -0.0_f32,
                1.5_f32,
                f32::NAN,
                f32::MIN_POSITIVE / 2.0,
                f32::INFINITY,
            ];
            assert_eq!(data.len(), expected.len());
            for (x, y) in data.iter().zip(expected.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "typed data 復号が bit 不一致");
            }
        }
        other => panic!("w は F32 のはず: {other:?}"),
    }
    match &graph1.initializers["i"] {
        RawTensor::I64 { data, .. } => assert_eq!(data, &vec![i64::MIN, 0, i64::MAX]),
        other => panic!("i は I64 のはず: {other:?}"),
    }

    // export -> 再 decode。export.rs の契約どおり raw_data のみへ正規化され、
    // それでも復号結果が typed data 経路の graph1 と bit 完全一致するはず。
    let (exported, _bytes2, graph2) = export_roundtrip(&graph1);
    let exported_graph = exported.graph.expect("graph はあるはず");
    for init in &exported_graph.initializer {
        assert!(
            init.float_data.is_empty(),
            "float_data は raw_data へ正規化されるはず ({})",
            init.name
        );
        assert!(
            init.int64_data.is_empty(),
            "int64_data は raw_data へ正規化されるはず ({})",
            init.name
        );
    }
    assert_raw_tensor_bit_exact("w", &graph1.initializers["w"], &graph2.initializers["w"]);
    assert_raw_tensor_bit_exact("i", &graph1.initializers["i"], &graph2.initializers["i"]);
}

// ==== (B) `interp::run` の bit 同一 ====

#[derive(Deserialize)]
struct OnnxReference {
    inputs: Vec<[f32; 2]>,
}

#[test]
fn model_onnx_exported_model_runs_bit_identical_to_import_source() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    let (_exported, _bytes, rebuilt) = export_roundtrip(&graph);

    let reference_json = std::fs::read_to_string(onnx_reference_path("onnx_reference.json"))
        .expect("onnx_reference.json 読み込み失敗");
    let reference: OnnxReference =
        serde_json::from_str(&reference_json).expect("onnx_reference.json パース失敗");
    assert!(
        !reference.inputs.is_empty(),
        "fixture が空では突合にならない"
    );

    for input in reference.inputs {
        assert_run_bit_identical(&graph, &rebuilt, move || {
            let mut feeds = HashMap::new();
            feeds.insert(
                "input".to_string(),
                Value::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
            );
            feeds
        });
    }
}

#[derive(Deserialize)]
struct SliceReproReference {
    inputs: Vec<Vec<f32>>,
    input_shape: Vec<usize>,
}

#[test]
fn slice_repro_onnx_exported_model_runs_bit_identical_to_import_source() {
    let model = load_model("slice_repro.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    let (_exported, _bytes, rebuilt) = export_roundtrip(&graph);

    let reference_json = std::fs::read_to_string(onnx_reference_path("slice_repro_reference.json"))
        .expect("slice_repro_reference.json 読み込み失敗");
    let reference: SliceReproReference =
        serde_json::from_str(&reference_json).expect("slice_repro_reference.json パース失敗");

    let flat: Vec<f32> = reference.inputs.iter().flatten().copied().collect();
    let shape = reference.input_shape.clone();

    assert_run_bit_identical(&graph, &rebuilt, move || {
        let mut feeds = HashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(flat.clone(), &shape).unwrap()),
        );
        feeds
    });
}

#[derive(Deserialize)]
struct TransformerReference {
    input_shape: Vec<usize>,
    input: Vec<Vec<Vec<f32>>>,
}

fn flatten3(nested: &[Vec<Vec<f32>>]) -> Vec<f32> {
    nested
        .iter()
        .flat_map(|batch| batch.iter().flat_map(|row| row.iter().copied()))
        .collect()
}

#[test]
#[ignore = "12MB の transformer.onnx を非コミット方針としているため。tests/fixtures/README.md の取得手順を参照"]
fn transformer_onnx_roundtrip_and_run_are_bit_identical() {
    // `cargo test -- --ignored`（`make test-ignored` 含む）でも非コミットの
    // transformer.onnx を取得していない環境では環境変数が未設定になるため、
    // fail ではなく早期 return でスキップする
    // （`tests/onnx_decode.rs`・`tests/onnx_transformer_e2e.rs` と同一運用）。
    let Ok(path) = std::env::var("ONNX_INTEROP_TRANSFORMER_ONNX") else {
        eprintln!(
            "skip: ONNX_INTEROP_TRANSFORMER_ONNX 未設定のため \
             transformer_onnx_roundtrip_and_run_are_bit_identical をスキップします \
             （tests/fixtures/README.md 参照）"
        );
        return;
    };

    let model_bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("読み込み失敗 {path}: {e}"));
    let model = ModelProto::decode(model_bytes.as_slice()).expect("decode は成功するはず");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    // 全 node の op_type が export allowlist に収まっていることの直接確認
    // （README 実測: 20 op 種別。allowlist ドリフト検出）。
    for node in &graph.nodes {
        assert!(
            node.domain.is_empty(),
            "domain は空のはず (node={})",
            node.name
        );
        assert!(
            SUPPORTED_OP_TYPES.contains(&node.op_type.as_str()),
            "{} が SUPPORTED_OP_TYPES に含まれない (node={})",
            node.op_type,
            node.name
        );
    }
    assert_eq!(graph.nodes.len(), 165, "README 実測値と一致するはず");
    assert_eq!(graph.initializers.len(), 12, "README 実測値と一致するはず");

    let (_exported, _bytes, rebuilt) = export_roundtrip(&graph);
    assert_graph_structurally_identical(&graph, &rebuilt);
    assert_export_is_fixed_point(&graph);

    let reference_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pytorch-transformer/reference.json");
    let reference_json = std::fs::read_to_string(&reference_path)
        .unwrap_or_else(|e| panic!("reference.json 読み込み失敗 {reference_path:?}: {e}"));
    let reference: TransformerReference =
        serde_json::from_str(&reference_json).expect("reference.json パース失敗");

    // `reference.json` の `output` との数値突合は行わない（既存 e2e の既知
    // REQ-7 超過は本テスト〈bit 同一の roundtrip 契約〉と無関係。本ファイル冒頭
    // `//!` 参照）。ここで固定するのは「同一 feed に対し import 元と export 後
    // のグラフが同じ結果を返す」ことのみ。
    let input_flat = flatten3(&reference.input);
    let shape = reference.input_shape.clone();

    assert_run_bit_identical(&graph, &rebuilt, move || {
        let mut feeds = HashMap::new();
        feeds.insert(
            "input".to_string(),
            Value::F32(Tensor::<f32>::new(input_flat.clone(), &shape).unwrap()),
        );
        feeds
    });
}

// ==== (C) 未対応 op を含むモデルの fail-closed ====

#[test]
fn model_with_unsupported_op_type_is_rejected_at_export() {
    let model = load_model("model.onnx");
    let mut graph = build_graph(&model).expect("build_graph は成功するはず");
    assert_eq!(
        graph.nodes[1].name, "/relu/Relu",
        "fixture README の既知順序が前提（tests/fixtures/README.md）"
    );
    graph.nodes[1].op_type = "Gelu".to_string();

    // 「モデル」としての fail-closed 検査: 手組み `Graph` へ直接
    // `build_model_proto` を呼ぶのではなく、check_exportable をバイパスした
    // `ModelProto` を encode -> decode -> build_graph してから export する
    // （decode 経路を通ったモデルであることの担保）。
    let tampered = to_model_proto_bypassing_check(&graph, &ExportOptions::default());
    let bytes = tampered.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("再 decode は成功するはず");
    let rebuilt = build_graph(&decoded)
        .expect("build_graph 自体は成功するはず（トポロジ検査は op_type を見ない）");

    let err = build_model_proto(&rebuilt, &ExportOptions::default())
        .expect_err("未対応 op_type を含むモデルは拒否されるはず");
    assert_eq!(
        err,
        ExportError::UnsupportedOp {
            node_name: "/relu/Relu".to_string(),
            op_type: "Gelu".to_string(),
            domain: String::new(),
        }
    );
}

#[test]
fn model_with_supported_op_in_custom_domain_is_rejected_at_export() {
    // 手組み `ModelProto`: op_type 自体は allowlist 内（`Gemm`）だが domain が
    // 既定 opset（空文字列）以外（`com.microsoft`）。decode 経路を通したモデル
    // として export を試みる。
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![NodeProto {
                input: vec!["a".to_string(), "b".to_string()],
                output: vec!["y".to_string()],
                name: "gemm_custom_domain".to_string(),
                op_type: "Gemm".to_string(),
                attribute: Vec::new(),
                domain: "com.microsoft".to_string(),
            }],
            name: "g".to_string(),
            initializer: Vec::new(),
            input: vec![
                ValueInfoProto {
                    name: "a".to_string(),
                },
                ValueInfoProto {
                    name: "b".to_string(),
                },
            ],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
            value_info: Vec::new(),
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
    };
    let bytes = model.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("decode は成功するはず");
    let graph = build_graph(&decoded).expect("build_graph は成功するはず");

    let err = build_model_proto(&graph, &ExportOptions::default())
        .expect_err("既定 opset 以外の domain を持つモデルは拒否されるはず");
    assert_eq!(
        err,
        ExportError::UnsupportedOp {
            node_name: "gemm_custom_domain".to_string(),
            op_type: "Gemm".to_string(),
            domain: "com.microsoft".to_string(),
        }
    );
}

#[test]
fn first_unsupported_node_in_topological_order_is_reported() {
    // 未対応 op を 2 箇所（順に Gelu・Conv）含むモデルで、`check_exportable`
    // （`export_ops.rs`）が nodes の並び順（トポロジカル順）で最初に見つかった
    // 違反ノードを報告することを固定する。
    let model = load_model("model.onnx");
    let mut graph = build_graph(&model).expect("build_graph は成功するはず");
    assert_eq!(
        graph.nodes[1].name, "/relu/Relu",
        "fixture README の既知順序が前提"
    );
    assert_eq!(
        graph.nodes[3].name, "/relu_1/Relu",
        "fixture README の既知順序が前提"
    );
    graph.nodes[1].op_type = "Gelu".to_string();
    graph.nodes[3].op_type = "Conv".to_string();

    let tampered = to_model_proto_bypassing_check(&graph, &ExportOptions::default());
    let bytes = tampered.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("再 decode は成功するはず");
    let rebuilt = build_graph(&decoded)
        .expect("build_graph 自体は成功するはず（トポロジ検査は op_type を見ない）");

    let err = build_model_proto(&rebuilt, &ExportOptions::default())
        .expect_err("複数の未対応 op を含むモデルは拒否されるはず");
    assert_eq!(
        err,
        ExportError::UnsupportedOp {
            node_name: "/relu/Relu".to_string(),
            op_type: "Gelu".to_string(),
            domain: String::new(),
        },
        "トポロジカル順で最初に現れる違反ノードが報告されるはず（2 番目の Conv ではない）"
    );
}

#[test]
fn exported_model_of_supported_ops_passes_check_exportable_for_all_fixtures() {
    // allowlist ドリフト検出: 両 fixture の全 node の op_type が
    // `SUPPORTED_OP_TYPES` に含まれ、domain が既定（空文字列）であることを
    // 直接 assert する（`transformer.onnx` は #[ignore] テスト側で同様に確認
    // 済み）。
    for fixture in ["model.onnx", "slice_repro.onnx"] {
        let model = load_model(fixture);
        let graph = build_graph(&model).expect("build_graph は成功するはず");
        for node in &graph.nodes {
            assert!(
                node.domain.is_empty(),
                "{fixture}: domain は空のはず (node={})",
                node.name
            );
            assert!(
                SUPPORTED_OP_TYPES.contains(&node.op_type.as_str()),
                "{fixture}: {} が SUPPORTED_OP_TYPES に含まれない (node={})",
                node.op_type,
                node.name
            );
        }
    }
}
