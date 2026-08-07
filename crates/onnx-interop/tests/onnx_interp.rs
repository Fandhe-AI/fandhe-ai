//! イシュー #78（TASK-7.2b・REQ-7）: `decode → build_graph → run`（ノード列を
//! `ops::*` へディスパッチして実行するグラフ実行インタープリタ）の全経路統合テスト。
//!
//! `tests/onnx_poc_v2_6_match.rs`（#80）は `ops::*` を直接呼び出す経路（decode 層を
//! 経由しない）の数値突合を、本ファイルは **ONNX proto デコード
//! （`onnx::proto::ModelProto::decode`）→ 内部グラフ構築（`onnx::graph::build_graph`）
//! → グラフ実行（`onnx::interp::run`）** の全経路突合を担う（受け入れ条件: 「ノード列が
//! オペ実装へディスパッチされ実行できる」の直接的な固定化）。
//!
//! ## 判定式についての注記（REQ-2 との混同禁止）
//!
//! `tests/onnx_poc_v2_6_match.rs` と同じ REQ-7 事前固定基準
//! `abs_err / (|ref| + 1e-6) <= 1e-3` を用いる。`.claude/rules/coding-rust.md` の
//! REQ-2 バックエンド間数値一致 OR 複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5
//! 未満）とは別指標であり、両者を混同してどちらかを緩和しない。

use std::collections::HashMap;
use std::path::PathBuf;

use onnx_interop::onnx::graph::{Graph, build_graph};
use onnx_interop::onnx::interp::{InterpError, Value, run};
use onnx_interop::onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
};
use prost::Message;
use serde::Deserialize;
use tensor_core::Tensor;

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

// --- model.onnx: decode -> build_graph -> run の全経路で PoC-v2-6 参照値と突合 ---

#[derive(Deserialize)]
struct OnnxReference {
    inputs: Vec<[f32; 2]>,
    outputs: Vec<f32>,
}

#[test]
fn model_onnx_end_to_end_matches_onnx_pytorch_reference_within_req7_tolerance() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    let reference_json = std::fs::read_to_string(onnx_reference_path("onnx_reference.json"))
        .expect("onnx_reference.json 読み込み失敗");
    let reference: OnnxReference =
        serde_json::from_str(&reference_json).expect("onnx_reference.json パース失敗");
    assert_eq!(reference.inputs.len(), reference.outputs.len());
    assert!(
        !reference.inputs.is_empty(),
        "fixture が空では突合にならない"
    );

    let mut max_rel_err = 0.0f32;
    for (input, &expected) in reference.inputs.iter().zip(reference.outputs.iter()) {
        let mut feeds = HashMap::new();
        feeds.insert(
            "input".to_string(),
            Value::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
        );
        let result = run(&graph, feeds).expect("run は成功するはず");
        let actual = match &result["output"] {
            Value::F32(t) => t.get(&[0, 0]).unwrap(),
            other => panic!("Value::F32 を期待したが {other:?}"),
        };
        let abs_err = (actual - expected).abs();
        // REQ-7 事前固定判定式（本ファイル冒頭 `//!` 参照）。
        let rel_err = abs_err / (expected.abs() + 1e-6);
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            rel_err <= 1e-3,
            "REQ-7 数値一致基準を超過: input={input:?} expected={expected} actual={actual} \
             abs_err={abs_err} rel_err={rel_err}"
        );
    }
    eprintln!("max_rel_err={max_rel_err} (threshold=1e-3)");
}

// --- slice_repro.onnx: 動的境界 Slice パターンの全経路突合 ---
// （Shape -> Gather -> Unsqueeze -> Concat -> Slice。v1 の burn-onnx 失敗パターンの
// グラフ実行レベル固定化。`tests/onnx_slice_dynamic_bounds.rs` は ops 手動連結のため
// decode 層を経由しない別経路であり、本テストと重複しない）。

#[derive(Deserialize)]
struct SliceReproReference {
    inputs: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

#[test]
fn slice_repro_onnx_end_to_end_matches_reference_within_req7_tolerance() {
    let model = load_model("slice_repro.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    let reference_json = std::fs::read_to_string(onnx_reference_path("slice_repro_reference.json"))
        .expect("slice_repro_reference.json 読み込み失敗");
    let reference: SliceReproReference =
        serde_json::from_str(&reference_json).expect("slice_repro_reference.json パース失敗");
    assert_eq!(reference.inputs.len(), reference.input_shape[0]);

    let flat: Vec<f32> = reference.inputs.iter().flatten().copied().collect();
    let x = Tensor::<f32>::new(flat, &reference.input_shape).unwrap();

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(x));
    let result = run(&graph, feeds).expect("run は成功するはず");
    let y = match &result["output"] {
        Value::F32(t) => t,
        other => panic!("Value::F32 を期待したが {other:?}"),
    };
    assert_eq!(y.shape(), reference.output_shape.as_slice());

    let mut max_rel_err = 0.0f32;
    for (i, row) in reference.outputs.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            let actual = y.get(&[i, j]).unwrap();
            let abs_err = (actual - expected).abs();
            let rel_err = abs_err / (expected.abs() + 1e-6);
            max_rel_err = max_rel_err.max(rel_err);
            assert!(
                rel_err <= 1e-3,
                "REQ-7 数値一致基準を超過: (i={i},j={j}) expected={expected} actual={actual} \
                 abs_err={abs_err} rel_err={rel_err}"
            );
        }
    }
    eprintln!("max_rel_err={max_rel_err} (threshold=1e-3)");
}

// --- エラー経路（no-silent-skip 契約・OWASP A03: 不正入力の fail-closed 拒否） ---

fn minimal_model_with_node(node: NodeProto, inputs: Vec<&str>, outputs: Vec<&str>) -> ModelProto {
    ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
fn run_rejects_unsupported_op_type() {
    // 未対応 op_type（TASK-7.3 系。#274 の担当）は UnsupportedOp で fail-closed に拒否する。
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n1".to_string(),
        op_type: "MatMul".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(matches!(err, InterpError::UnsupportedOp(op) if op == "MatMul"));
}

#[test]
fn run_rejects_missing_required_feed() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).unwrap();
    let err = run(&graph, HashMap::new()).unwrap_err();
    assert!(matches!(err, InterpError::MissingFeed { input } if input == "input"));
}

#[test]
fn run_rejects_unknown_feed_name() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        Value::F32(Tensor::<f32>::zeros(&[1, 2]).unwrap()),
    );
    feeds.insert(
        "not_a_real_input".to_string(),
        Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(matches!(err, InterpError::UnknownFeed { name } if name == "not_a_real_input"));
}

#[test]
fn run_rejects_type_mismatch_on_gemm_input() {
    // model.onnx の Gemm(A) は f32 テンソルを要求する。I64 を feed すると TypeMismatch。
    let model = load_model("model.onnx");
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        Value::I64 {
            data: vec![0, 0],
            shape: vec![1, 2],
        },
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(
        matches!(err, InterpError::TypeMismatch { node, expected } if node == "/fc1/Gemm" && expected == "f32")
    );
}

#[test]
fn run_rejects_concat_missing_axis_attribute() {
    // Concat の axis 属性は必須（ONNX 仕様。デフォルト値なし）。欠落は MissingAttribute。
    let node = NodeProto {
        input: vec!["a".to_string(), "b".to_string()],
        output: vec!["y".to_string()],
        name: "n_concat".to_string(),
        op_type: "Concat".to_string(),
        attribute: vec![], // axis 属性なし
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["a", "b"], vec!["y"]);
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "a".to_string(),
        Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    feeds.insert(
        "b".to_string(),
        Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(
        matches!(err, InterpError::MissingAttribute { node, attr } if node == "n_concat" && attr == "axis")
    );
}

#[test]
fn run_rejects_output_arity_mismatch() {
    // TASK-7.2 の 8 オペはすべて単一出力。node.output が 2 つ宣言されている場合
    // （不正な ONNX。実際の Relu は単一出力のみ）は OutputArityMismatch で拒否する。
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y1".to_string(), "y2".to_string()],
        name: "n_relu".to_string(),
        op_type: "Relu".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y1"]);
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(
        matches!(err, InterpError::OutputArityMismatch { node, expected: 1, actual: 2 } if node == "n_relu")
    );
}

#[test]
fn run_feed_overrides_initializer_pre_ir4_pattern() {
    // グラフ入力名が initializer 名と重複する pre-IR-4 パターン（`onnx/graph.rs` の
    // `graph_input_name_matching_initializer_name_is_accepted` と対称）。feed が
    // initializer を上書きすることを確認する（本モジュール `run` のドキュメント参照）。
    let init = TensorProto {
        name: "x".to_string(),
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
        dims: vec![1],
        float_data: vec![100.0],
        int64_data: vec![],
        raw_data: vec![],
    };
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_relu".to_string(),
        op_type: "Relu".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![node],
            name: "g".to_string(),
            initializer: vec![init],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        Value::F32(Tensor::<f32>::new(vec![7.0], &[1]).unwrap()),
    );
    let result = run(&graph, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => assert_eq!(
            t.get(&[0]).unwrap(),
            7.0,
            "feed が initializer(100.0) を上書きするはず"
        ),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

// `AttributeProto` を明示構築しても使われることを最低限確認する（型が実際に
// 到達可能であることの回帰確認。`onnx_decode.rs` と重複しない観点として、本ファイルは
// インタープリタが属性値を正しく読み取ることを確認する）。
#[test]
fn run_reads_gemm_attributes_correctly() {
    let node = NodeProto {
        input: vec!["a".to_string(), "b".to_string()],
        output: vec!["y".to_string()],
        name: "n_gemm".to_string(),
        op_type: "Gemm".to_string(),
        attribute: vec![
            AttributeProto {
                name: "alpha".to_string(),
                f: 2.0,
                i: 0,
                s: vec![],
                t: None,
                floats: vec![],
                ints: vec![],
                r#type: 1,
            },
            AttributeProto {
                name: "transB".to_string(),
                f: 0.0,
                i: 1,
                s: vec![],
                t: None,
                floats: vec![],
                ints: vec![],
                r#type: 2,
            },
        ],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["a", "b"], vec!["y"]);
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "a".to_string(),
        Value::F32(Tensor::<f32>::new(vec![1.0, 2.0], &[1, 2]).unwrap()),
    );
    feeds.insert(
        "b".to_string(),
        // transB=1 のため [1,2] を渡し B^T=[2,1] として使わせる。
        Value::F32(Tensor::<f32>::new(vec![3.0, 4.0], &[1, 2]).unwrap()),
    );
    let result = run(&graph, feeds).unwrap();
    match &result["y"] {
        // alpha=2.0 * (A @ B^T) = 2.0 * (1*3 + 2*4) = 2.0 * 11 = 22
        Value::F32(t) => assert_eq!(t.get(&[0, 0]).unwrap(), 22.0),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn run_rejects_graph_output_not_produced() {
    // `build_graph` は `GraphProto.output` が未生成の名前を参照していないことを
    // 静的検証する（`GraphError::UnknownGraphOutput`）ため、`run` 単体の
    // `InterpError::GraphOutputNotProduced` は通常経路では踏まない。
    // `Graph` の全フィールドが pub であることを利用し、その不変条件を
    // あえて破った `Graph` を直接構築して `run` の fail-closed 経路を演習する
    // （レビュー指摘: エラー variant のカバレッジ補強）。
    let graph = Graph {
        nodes: vec![],
        initializers: HashMap::new(),
        inputs: vec![],
        outputs: vec!["not_produced".to_string()],
    };
    let err = run(&graph, HashMap::new()).unwrap_err();
    assert!(matches!(err, InterpError::GraphOutputNotProduced { name } if name == "not_produced"));
}

#[test]
fn run_rejects_unsqueeze_i64_when_input_rank_is_not_one() {
    // i64 直接経路は入力 rank 1 のみサポートする（本ファイル冒頭 `//!`・`interp.rs`
    // モジュール冒頭コメント参照）。rank 2 の I64 値を feed すると
    // `I64ShapeUnsupported` で拒否される。
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_unsqueeze".to_string(),
        op_type: "Unsqueeze".to_string(),
        attribute: vec![AttributeProto {
            name: "axes".to_string(),
            f: 0.0,
            i: 0,
            s: vec![],
            t: None,
            floats: vec![],
            ints: vec![0],
            r#type: 7, // INTS
        }],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        Value::I64 {
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
        },
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(
        matches!(err, InterpError::I64ShapeUnsupported { node, op, shape } if node == "n_unsqueeze" && op == "Unsqueeze" && shape == vec![2, 2])
    );
}

#[test]
fn run_rejects_unsqueeze_i64_when_output_rank_exceeds_one() {
    // Medium レビュー指摘の回帰固定: 入力 rank 1 でも axes 挿入後の出力 rank が
    // 1 を超える場合（[3] + axes=[0] -> 本来 [1,3]）、i64 直接経路は多次元 shape を
    // 表現できないため `shape: vec![data.len()]` へ無言正規化せず fail-closed に
    // 拒否する（`I64ShapeUnsupported`）。
    let node = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "n_unsqueeze".to_string(),
        op_type: "Unsqueeze".to_string(),
        attribute: vec![AttributeProto {
            name: "axes".to_string(),
            f: 0.0,
            i: 0,
            s: vec![],
            t: None,
            floats: vec![],
            ints: vec![0],
            r#type: 7, // INTS
        }],
        domain: String::new(),
    };
    let model = minimal_model_with_node(node, vec!["x"], vec!["y"]);
    let graph = build_graph(&model).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        Value::I64 {
            data: vec![1, 2, 3],
            shape: vec![3],
        },
    );
    let err = run(&graph, feeds).unwrap_err();
    assert!(
        matches!(err, InterpError::I64ShapeUnsupported { node, op, shape } if node == "n_unsqueeze" && op == "Unsqueeze" && shape == vec![3])
    );
}
