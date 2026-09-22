//! `onnx::autograd`（イシュー #2078）の統合テスト。公開 API のみを経由する:
//!
//! - per-op forward bit 同一（`BoundGraph::run` vs `interp::run`）。
//! - `model.onnx`（`Gemm→Relu→Gemm→Relu→Gemm→Sigmoid`）end-to-end backward:
//!   `fc*.weight`／`fc*.bias` へ勾配が伝播し、有限差分（FD）近似と符号・オーダーが
//!   一致することを確認する（FD 自体の離散化誤差があるため REQ-2 の bit／複合判定
//!   ではなく、本テスト専用の緩い許容誤差を用いる）。
//! - `BindOptions::trainable` フィルタ・エラー経路（`MissingFeed`／`UnknownFeed`／
//!   `TapeMismatch`／`NotSingleInputOutput`／`UnsupportedInAutograd`）。

use std::collections::{HashMap, HashSet};

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_onnx_interop::onnx::autograd::{
    AutogradError, AutogradValue, BindOptions, BoundGraph,
};
use fandhe_ai_onnx_interop::onnx::graph::{Graph, build_graph};
use fandhe_ai_onnx_interop::onnx::interp::{self, Value};
use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, ModelProto, NodeProto};
use fandhe_ai_tensor_core::Tensor;
use prost::Message;

fn node(op_type: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
    NodeProto {
        input: input.into_iter().map(String::from).collect(),
        output: output.into_iter().map(String::from).collect(),
        name: format!("n_{op_type}"),
        op_type: op_type.to_string(),
        attribute: vec![],
        domain: String::new(),
    }
}

fn node_with_attrs(
    op_type: &str,
    input: Vec<&str>,
    output: Vec<&str>,
    attribute: Vec<AttributeProto>,
) -> NodeProto {
    let mut n = node(op_type, input, output);
    n.attribute = attribute;
    n
}

fn attr_i64(name: &str, i: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i,
        ..Default::default()
    }
}

fn single_node_graph(n: NodeProto, inputs: Vec<&str>, output: &str) -> Graph {
    Graph {
        nodes: vec![n],
        initializers: HashMap::new(),
        inputs: inputs.into_iter().map(String::from).collect(),
        outputs: vec![output.to_string()],
    }
}

fn f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// `interp::Value::F32` から要素 `Vec<f32>` を取り出す。
fn as_f32_vec(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(t) => t.contiguous().as_slice().unwrap().to_vec(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

// ================= per-op forward bit 同一 =================

/// 1 入力 op の forward bit 同一を `interp::run`（Const 経路）と
/// `BoundGraph::run`（Var 経路）で突き合わせる共通ヘルパー。
fn assert_unary_forward_bit_identical(n: NodeProto, x: Tensor<f32>) {
    let graph = single_node_graph(n, vec!["x"], "y");

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(x.clone()));
    let expected = interp::run(&graph, feeds).unwrap();
    let expected = as_f32_vec(&expected["y"]);

    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds2 = HashMap::new();
    feeds2.insert("x".to_string(), AutogradValue::Var(tape.var(&x)));
    let out = bound.run(feeds2).unwrap();
    let actual = match &out["y"] {
        AutogradValue::Var(v) => v.value().contiguous().as_slice().unwrap().to_vec(),
        AutogradValue::Const(_) => panic!("Var を期待した"),
    };

    assert_eq!(expected.len(), actual.len());
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert_eq!(
            e.to_bits(),
            a.to_bits(),
            "forward が bit 一致しない: expected={e} actual={a}"
        );
    }
}

fn assert_binary_forward_bit_identical(n: NodeProto, a: Tensor<f32>, b: Tensor<f32>) {
    let graph = single_node_graph(n, vec!["a", "b"], "y");

    let mut feeds = HashMap::new();
    feeds.insert("a".to_string(), Value::F32(a.clone()));
    feeds.insert("b".to_string(), Value::F32(b.clone()));
    let expected = interp::run(&graph, feeds).unwrap();
    let expected = as_f32_vec(&expected["y"]);

    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds2 = HashMap::new();
    feeds2.insert("a".to_string(), AutogradValue::Var(tape.var(&a)));
    feeds2.insert("b".to_string(), AutogradValue::Var(tape.var(&b)));
    let out = bound.run(feeds2).unwrap();
    let actual = match &out["y"] {
        AutogradValue::Var(v) => v.value().contiguous().as_slice().unwrap().to_vec(),
        AutogradValue::Const(_) => panic!("Var を期待した"),
    };

    assert_eq!(expected.len(), actual.len());
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert_eq!(
            e.to_bits(),
            a.to_bits(),
            "forward が bit 一致しない: expected={e} actual={a}"
        );
    }
}

#[test]
fn relu_forward_bit_identical() {
    assert_unary_forward_bit_identical(
        node("Relu", vec!["x"], vec!["y"]),
        f32(vec![-2.0, -0.25, 1.5, 3.0], &[4]),
    );
}

#[test]
fn sigmoid_forward_bit_identical() {
    assert_unary_forward_bit_identical(
        node("Sigmoid", vec!["x"], vec!["y"]),
        f32(vec![-1.0, 0.3, 2.0], &[3]),
    );
}

#[test]
fn sqrt_forward_bit_identical() {
    assert_unary_forward_bit_identical(
        node("Sqrt", vec!["x"], vec!["y"]),
        f32(vec![4.0, 9.0, 0.25], &[3]),
    );
}

#[test]
fn erf_forward_bit_identical() {
    assert_unary_forward_bit_identical(
        node("Erf", vec!["x"], vec!["y"]),
        f32(vec![-1.5, 0.0, 0.7, 2.2], &[4]),
    );
}

#[test]
fn add_forward_bit_identical_with_broadcast() {
    assert_binary_forward_bit_identical(
        node("Add", vec!["a", "b"], vec!["y"]),
        f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
        f32(vec![10.0, 20.0], &[2]),
    );
}

#[test]
fn mul_forward_bit_identical() {
    assert_binary_forward_bit_identical(
        node("Mul", vec!["a", "b"], vec!["y"]),
        f32(vec![1.0, 2.0, 3.0], &[3]),
        f32(vec![4.0, 5.0, 6.0], &[3]),
    );
}

#[test]
fn div_forward_bit_identical() {
    assert_binary_forward_bit_identical(
        node("Div", vec!["a", "b"], vec!["y"]),
        f32(vec![10.0, 21.0, 33.0], &[3]),
        f32(vec![2.0, 3.0, 4.0], &[3]),
    );
}

#[test]
fn matmul_forward_bit_identical() {
    assert_binary_forward_bit_identical(
        node("MatMul", vec!["a", "b"], vec!["y"]),
        f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
    );
}

#[test]
fn softmax_forward_bit_identical() {
    assert_unary_forward_bit_identical(
        node_with_attrs("Softmax", vec!["x"], vec!["y"], vec![attr_i64("axis", -1)]),
        f32(vec![1.0, 2.0, 3.0, 0.5, -1.0, 4.0], &[2, 3]),
    );
}

#[test]
fn layer_normalization_forward_bit_identical() {
    let graph = single_node_graph(
        node_with_attrs(
            "LayerNormalization",
            vec!["x", "scale", "bias"],
            vec!["y"],
            vec![attr_i64("axis", -1)],
        ),
        vec!["x", "scale", "bias"],
        "y",
    );
    let x = f32(vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.5], &[2, 4]);
    let scale = f32(vec![1.0, 1.1, 0.9, 1.2], &[4]);
    let bias = f32(vec![0.0, 0.1, -0.1, 0.05], &[4]);

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(x.clone()));
    feeds.insert("scale".to_string(), Value::F32(scale.clone()));
    feeds.insert("bias".to_string(), Value::F32(bias.clone()));
    let expected = as_f32_vec(&interp::run(&graph, feeds).unwrap()["y"]);

    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds2 = HashMap::new();
    feeds2.insert("x".to_string(), AutogradValue::Var(tape.var(&x)));
    feeds2.insert("scale".to_string(), AutogradValue::Var(tape.var(&scale)));
    feeds2.insert("bias".to_string(), AutogradValue::Var(tape.var(&bias)));
    let out = bound.run(feeds2).unwrap();
    let actual = match &out["y"] {
        AutogradValue::Var(v) => v.value().contiguous().as_slice().unwrap().to_vec(),
        AutogradValue::Const(_) => panic!("Var を期待した"),
    };
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert_eq!(e.to_bits(), a.to_bits());
    }
}

#[test]
fn gemm_forward_bit_identical_with_trans_b_and_bias() {
    let graph = single_node_graph(
        node_with_attrs(
            "Gemm",
            vec!["a", "b", "c"],
            vec!["y"],
            vec![attr_i64("transB", 1)],
        ),
        vec!["a", "b", "c"],
        "y",
    );
    let a = f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = f32(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = f32(vec![1.0, -1.0], &[2]);

    let mut feeds = HashMap::new();
    feeds.insert("a".to_string(), Value::F32(a.clone()));
    feeds.insert("b".to_string(), Value::F32(b.clone()));
    feeds.insert("c".to_string(), Value::F32(c.clone()));
    let expected = as_f32_vec(&interp::run(&graph, feeds).unwrap()["y"]);

    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds2 = HashMap::new();
    feeds2.insert("a".to_string(), AutogradValue::Var(tape.var(&a)));
    feeds2.insert("b".to_string(), AutogradValue::Var(tape.var(&b)));
    feeds2.insert("c".to_string(), AutogradValue::Var(tape.var(&c)));
    let out = bound.run(feeds2).unwrap();
    let actual = match &out["y"] {
        AutogradValue::Var(v) => v.value().contiguous().as_slice().unwrap().to_vec(),
        AutogradValue::Const(_) => panic!("Var を期待した"),
    };
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert_eq!(e.to_bits(), a.to_bits());
    }
}

// ================= model.onnx end-to-end =================

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_model_onnx_graph() -> Graph {
    let bytes = std::fs::read(fixture_path("model.onnx")).unwrap();
    let model = ModelProto::decode(bytes.as_slice()).unwrap();
    build_graph(&model).unwrap()
}

#[test]
fn model_onnx_forward_matches_interp_run_bit_identical() {
    let graph = load_model_onnx_graph();
    let inputs = [[0.3f32, -0.7], [1.2, 0.05], [-0.4, -0.9]];

    for input in inputs {
        let mut feeds = HashMap::new();
        feeds.insert(
            "input".to_string(),
            Value::F32(f32(input.to_vec(), &[1, 2])),
        );
        let expected = as_f32_vec(&interp::run(&graph, feeds).unwrap()["output"]);

        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
        let x = tape.var(&f32(input.to_vec(), &[1, 2]));
        let out = bound.forward(&x).unwrap();
        let actual = out.value().contiguous().as_slice().unwrap().to_vec();

        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert_eq!(
                e.to_bits(),
                a.to_bits(),
                "input={input:?} expected={e} actual={a}"
            );
        }
    }
}

/// `model.onnx` を `bind` し、`fc1.weight` の勾配を中心差分（FD, h=1e-2）と
/// 突き合わせる。FD 自体の離散化誤差があるため REQ-2 の bit／複合判定ではなく
/// 本テスト専用の緩い許容誤差（絶対誤差 5e-2 または相対誤差 5e-2）を用いる。
#[test]
fn model_onnx_backward_gradient_matches_finite_difference() {
    let graph = load_model_onnx_graph();
    let input = [0.3f32, -0.7];

    let loss_for_weight = |w_perturbed: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let mut trainable = HashSet::new();
        trainable.insert("fc1.weight".to_string());
        let bound =
            BoundGraph::bind(&graph, &tape, &BindOptions::with_trainable(trainable)).unwrap();
        // fc1.weight を手動で差し替えるため、feed として上書きする
        // （initializer と同名の feed は initializer を上書きする契約。
        // `BoundGraph::run` doc 参照）。
        let x = tape.var(&f32(input.to_vec(), &[1, 2]));
        let w = tape.var_no_grad(w_perturbed);
        let mut feeds = HashMap::new();
        feeds.insert("input".to_string(), AutogradValue::Var(x));
        feeds.insert("fc1.weight".to_string(), AutogradValue::Var(w));
        let out = bound.run(feeds).unwrap();
        match &out["output"] {
            AutogradValue::Var(v) => v.value().get(&[0, 0]).unwrap(),
            AutogradValue::Const(_) => panic!("Var を期待した"),
        }
    };

    // fc1.weight の初期値を取得する（bind() 経由ではなく graph.initializers から直接）。
    let fc1_weight = match graph.initializers.get("fc1.weight").unwrap() {
        fandhe_ai_onnx_interop::onnx::graph::RawTensor::F32 { data, shape } => f32(
            data.clone(),
            &shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        ),
        _ => panic!("fc1.weight は F32 のはず"),
    };

    // autograd 側の解析的勾配。
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let x = tape.var(&f32(input.to_vec(), &[1, 2]));
    let out = bound.forward(&x).unwrap();
    let grads = tape.backward(&out).unwrap();
    let w_var = bound
        .param("fc1.weight")
        .expect("fc1.weight は trainable のはず");
    let analytic = grads
        .get(&w_var)
        .unwrap()
        .expect("fc1.weight への勾配が存在するはず")
        .clone();
    let analytic = analytic.contiguous();
    let analytic_slice = analytic.as_slice().unwrap();

    let h = 1e-2f32;
    let mut checked = 0;
    let data = fc1_weight.contiguous().as_slice().unwrap().to_vec();
    let shape = fc1_weight.shape().to_vec();
    for i in 0..data.len().min(4) {
        let mut plus = data.clone();
        plus[i] += h;
        let mut minus = data.clone();
        minus[i] -= h;
        let loss_plus = loss_for_weight(&f32(plus, &shape));
        let loss_minus = loss_for_weight(&f32(minus, &shape));
        let fd = (loss_plus - loss_minus) / (2.0 * h);
        let a = analytic_slice[i];
        let abs_err = (fd - a).abs();
        let rel_err = abs_err / (a.abs() + 1e-6);
        assert!(
            abs_err < 5e-2 || rel_err < 5e-2,
            "fc1.weight[{i}] の勾配が FD と乖離: analytic={a} fd={fd} abs_err={abs_err} rel_err={rel_err}"
        );
        checked += 1;
    }
    assert!(checked > 0, "少なくとも 1 要素を検証したい");
}

#[test]
fn trainable_filter_excludes_non_listed_params_from_gradient_tracking() {
    let graph = load_model_onnx_graph();
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let mut trainable = HashSet::new();
    trainable.insert("fc1.weight".to_string());
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::with_trainable(trainable)).unwrap();

    assert!(bound.param("fc1.weight").is_some());
    assert!(bound.param("fc2.weight").is_none());

    let x = tape.var(&f32(vec![0.1, 0.2], &[1, 2]));
    let out = bound.forward(&x).unwrap();
    let grads = tape.backward(&out).unwrap();

    let w1 = bound.param("fc1.weight").unwrap();
    assert!(grads.get(&w1).unwrap().is_some());
}

// ================= エラー経路 =================

#[test]
fn missing_feed_is_rejected() {
    let graph = load_model_onnx_graph();
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let err = bound.run(HashMap::new()).unwrap_err();
    assert!(matches!(err, AutogradError::MissingFeed { .. }));
}

#[test]
fn unknown_feed_is_rejected() {
    let graph = load_model_onnx_graph();
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        AutogradValue::Var(tape.var(&f32(vec![0.1, 0.2], &[1, 2]))),
    );
    feeds.insert(
        "not_a_real_input".to_string(),
        AutogradValue::Var(tape.var(&f32(vec![1.0], &[1]))),
    );
    let err = bound.run(feeds).unwrap_err();
    assert!(matches!(err, AutogradError::UnknownFeed { .. }));
}

#[test]
fn cross_tape_var_is_rejected() {
    let graph = load_model_onnx_graph();
    let tape_a = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let tape_b = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape_a, &BindOptions::default()).unwrap();
    let x_wrong_tape = tape_b.var(&f32(vec![0.1, 0.2], &[1, 2]));
    let err = bound.forward(&x_wrong_tape).unwrap_err();
    assert!(matches!(err, AutogradError::Autodiff(_)));
}

#[test]
fn not_single_input_output_is_rejected_for_multi_output_graph() {
    // `graph.outputs` を 2 個にした Graph を手作りし、`forward` が拒否することを確認する。
    let n = node("Relu", vec!["x"], vec!["y"]);
    let graph = Graph {
        nodes: vec![n],
        initializers: HashMap::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string(), "y".to_string()],
    };
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let x = tape.var(&f32(vec![1.0, -1.0], &[2]));
    let err = bound.forward(&x).unwrap_err();
    assert!(matches!(err, AutogradError::NotSingleInputOutput { .. }));
}

#[test]
fn unsupported_op_is_rejected_fail_closed() {
    // `Reshape` は #2078 スコープでは未実装（モジュール冒頭コメント参照）。
    let graph = single_node_graph(
        node("Reshape", vec!["x", "shape"], vec!["y"]),
        vec!["x", "shape"],
        "y",
    );
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        AutogradValue::Var(tape.var(&f32(vec![1.0, 2.0], &[2]))),
    );
    feeds.insert(
        "shape".to_string(),
        AutogradValue::Const(Value::I64(Tensor::new(vec![2], &[1]).unwrap())),
    );
    let err = bound.run(feeds).unwrap_err();
    assert!(matches!(err, AutogradError::UnsupportedInAutograd { .. }));
}

#[test]
fn shape_and_cast_const_paths_work_without_severing_needed_gradients() {
    // `Shape`（Var 入力可・非勾配出力）と `Cast(F32->F32)`（恒等・Var のまま）の
    // const 経路 2 op を確認する。
    let graph = Graph {
        nodes: vec![
            node("Shape", vec!["x"], vec!["shp"]),
            node_with_attrs("Cast", vec!["x"], vec!["y"], vec![attr_i64("to", 1)]),
        ],
        initializers: HashMap::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["shp".to_string(), "y".to_string()],
    };
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let x = tape.var(&f32(vec![1.0, 2.0, 3.0], &[3]));
    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), AutogradValue::Var(x));
    let out = bound.run(feeds).unwrap();
    match &out["shp"] {
        AutogradValue::Const(Value::I64(t)) => {
            assert_eq!(t.contiguous().as_slice().unwrap(), &[3i64])
        }
        other => panic!("Value::I64(Const) を期待したが {other:?} のような値だった"),
    }
    match &out["y"] {
        AutogradValue::Var(_) => {}
        other => panic!("Cast(F32->F32) は Var のまま透過するはずが {other:?} のような値だった"),
    }
}

// ================= 有限差分 (FD) 勾配検証 =================
//
// `model_onnx_backward_gradient_matches_finite_difference` は Gemm/Relu/Sigmoid
// 経路のみを経由し、broadcast 縮約を伴う Add/Mul/Div・Sqrt/Erf・batched/broadcast
// MatMul・任意 axis Softmax・LayerNormalization の dx/dscale/dbias を検証しない
// （レビュー指摘: PR #2223 codex-review discussion_r4072652855。P2・ブロック対象外
// だが P1 修正〈数値契約不整合〉の再発防止のため追加する）。

/// 単一ノードグラフに対する汎用 FD 勾配検証ヘルパー。損失は `sum(output)`
/// （`Var::sum(None)`）とし、`inputs` の各要素を中心差分（`h`）で摂動して
/// analytic 勾配（`Tape::backward`）と数値勾配を突き合わせる。
/// `model_onnx_backward_gradient_matches_finite_difference` と同じく FD
/// 自体の離散化誤差があるため REQ-2 の bit／複合判定ではなく本ヘルパー専用の
/// 緩い許容誤差（`tol_abs` または `tol_rel`）を用いる。
fn assert_grad_matches_finite_difference(
    n: NodeProto,
    input_names: &[&str],
    inputs: &[Tensor<f32>],
    output_name: &str,
    h: f32,
    tol_abs: f32,
    tol_rel: f32,
) {
    let graph = single_node_graph(n, input_names.to_vec(), output_name);

    // 損失スカラー（sum(output)）のみを要する forward-only 評価。呼び出しごとに
    // 新規 Tape を張り直す（`model_onnx_backward_gradient_matches_finite_difference`
    // の `loss_for_weight` と同じ方式）。
    let forward_sum = |vals: &[Tensor<f32>]| -> f32 {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
        let mut feeds = HashMap::new();
        for (name, v) in input_names.iter().zip(vals.iter()) {
            feeds.insert((*name).to_string(), AutogradValue::Var(tape.var(v)));
        }
        let out = bound.run(feeds).unwrap();
        let out_var = match &out[output_name] {
            AutogradValue::Var(v) => *v,
            AutogradValue::Const(_) => panic!("Var を期待した"),
        };
        let loss = out_var.sum(None).unwrap();
        loss.value().contiguous().as_slice().unwrap()[0]
    };

    // analytic 勾配。
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let mut feeds = HashMap::new();
    let mut vars = Vec::new();
    for (name, v) in input_names.iter().zip(inputs.iter()) {
        let var = tape.var(v);
        // `Var` は `Copy`（clippy::clone_on_copy）。
        vars.push(var);
        feeds.insert((*name).to_string(), AutogradValue::Var(var));
    }
    let out = bound.run(feeds).unwrap();
    let out_var = match &out[output_name] {
        AutogradValue::Var(v) => *v,
        AutogradValue::Const(_) => panic!("Var を期待した"),
    };
    let loss = out_var.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    let mut checked = 0;
    for (idx, input) in inputs.iter().enumerate() {
        let analytic = grads
            .get(&vars[idx])
            .unwrap()
            .unwrap_or_else(|| panic!("{} への勾配が存在するはず", input_names[idx]))
            .clone();
        let analytic = analytic.contiguous();
        let analytic_slice = analytic.as_slice().unwrap();

        let data = input.contiguous().as_slice().unwrap().to_vec();
        let shape = input.shape().to_vec();
        for i in 0..data.len() {
            let mut plus_vals = inputs.to_vec();
            let mut plus_data = data.clone();
            plus_data[i] += h;
            plus_vals[idx] = f32(plus_data, &shape);
            let mut minus_vals = inputs.to_vec();
            let mut minus_data = data.clone();
            minus_data[i] -= h;
            minus_vals[idx] = f32(minus_data, &shape);

            let loss_plus = forward_sum(&plus_vals);
            let loss_minus = forward_sum(&minus_vals);
            let fd = (loss_plus - loss_minus) / (2.0 * h);
            let a = analytic_slice[i];
            let abs_err = (fd - a).abs();
            let rel_err = abs_err / (a.abs() + 1e-6);
            assert!(
                abs_err < tol_abs || rel_err < tol_rel,
                "{}[{i}] の勾配が FD と乖離: analytic={a} fd={fd} abs_err={abs_err} rel_err={rel_err}",
                input_names[idx]
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "少なくとも 1 要素を検証したい");
}

#[test]
fn softmax_backward_empty_axis_with_huge_sibling_dim_does_not_hang() {
    // forward 側の `softmax_empty_axis_with_huge_sibling_dim_does_not_hang`
    // （`ops/softmax.rs`）と対になる backward 側の回帰テスト。`axis` 自体の
    // サイズ（`inner`）が 0 で兄弟次元（`outer`）が `usize::MAX` の形状は
    // `checked_numel`（tensor-core）が正規に許容するため、backward にも forward
    // と同じ早期リターンが必要（レビュー指摘: PR #2223 Cursor Bugbot
    // discussion_r4072661705）。テスト自体が有限時間で完了すること
    // （反復ハングしないこと）が本テストの主張。
    let graph = single_node_graph(
        node_with_attrs("Softmax", vec!["x"], vec!["y"], vec![attr_i64("axis", 1)]),
        vec!["x"],
        "y",
    );
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let bound = BoundGraph::bind(&graph, &tape, &BindOptions::default()).unwrap();
    let x = tape.var(&f32(Vec::new(), &[usize::MAX, 0]));
    let out = bound.forward(&x).unwrap();
    assert_eq!(out.value().shape(), &[usize::MAX, 0]);
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap();
    assert_eq!(dx.shape(), &[usize::MAX, 0]);
}

#[test]
fn add_backward_gradient_matches_finite_difference_with_broadcast() {
    assert_grad_matches_finite_difference(
        node("Add", vec!["a", "b"], vec!["y"]),
        &["a", "b"],
        &[
            f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
            f32(vec![10.0, -5.0], &[2]),
        ],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn mul_backward_gradient_matches_finite_difference() {
    assert_grad_matches_finite_difference(
        node("Mul", vec!["a", "b"], vec!["y"]),
        &["a", "b"],
        &[
            f32(vec![1.0, 2.0, 3.0], &[3]),
            f32(vec![4.0, -1.0, 0.5], &[3]),
        ],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn div_backward_gradient_matches_finite_difference() {
    assert_grad_matches_finite_difference(
        node("Div", vec!["a", "b"], vec!["y"]),
        &["a", "b"],
        &[
            f32(vec![10.0, 21.0, -33.0], &[3]),
            f32(vec![2.0, 3.0, -4.0], &[3]),
        ],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn sqrt_backward_gradient_matches_finite_difference() {
    assert_grad_matches_finite_difference(
        node("Sqrt", vec!["x"], vec!["y"]),
        &["x"],
        &[f32(vec![4.0, 9.0, 2.25], &[3])],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn erf_backward_gradient_matches_finite_difference() {
    assert_grad_matches_finite_difference(
        node("Erf", vec!["x"], vec!["y"]),
        &["x"],
        &[f32(vec![-1.5, 0.3, 0.7, 2.0], &[4])],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn matmul_backward_gradient_matches_finite_difference_with_batch_broadcast() {
    // `a` shape=[2,2,3]（batch=2）・`b` shape=[3,2]（batch 次元なし。broadcast
    // 経路）で batched／broadcast MatMul backward を検証する。
    assert_grad_matches_finite_difference(
        node("MatMul", vec!["a", "b"], vec!["y"]),
        &["a", "b"],
        &[
            f32(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
                ],
                &[2, 2, 3],
            ),
            f32(vec![1.0, 0.5, -1.0, 2.0, 0.3, -0.7], &[3, 2]),
        ],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn softmax_backward_gradient_matches_finite_difference_on_middle_axis() {
    // axis=1（中間軸。`inner`>1 の縮約経路を検証する）。
    assert_grad_matches_finite_difference(
        node_with_attrs("Softmax", vec!["x"], vec!["y"], vec![attr_i64("axis", 1)]),
        &["x"],
        &[f32(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 3, 2],
        )],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}

#[test]
fn layer_normalization_backward_gradient_matches_finite_difference() {
    // dx／dscale／dbias の 3 入力すべてを同時に検証する。
    assert_grad_matches_finite_difference(
        node_with_attrs(
            "LayerNormalization",
            vec!["x", "scale", "bias"],
            vec!["y"],
            vec![attr_i64("axis", -1)],
        ),
        &["x", "scale", "bias"],
        &[
            f32(vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.5], &[2, 4]),
            f32(vec![1.0, 1.1, 0.9, 1.2], &[4]),
            f32(vec![0.0, 0.1, -0.1, 0.05], &[4]),
        ],
        "y",
        1e-2,
        5e-2,
        5e-2,
    );
}
