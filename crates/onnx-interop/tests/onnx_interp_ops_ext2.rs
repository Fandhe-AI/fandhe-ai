//! イシュー #2186（ONNX import op 拡大 2）の統合テスト: `Clip`／`Tanh`／
//! `Gelu`／`Where`／`Expand`／`ReduceMean`／`Pad`／`Resize` の op 単体
//! テスト（受理・拒否の両方）と `Gemm` の固め（属性型検証・入力数検査・
//! 旧 opset `broadcast` 属性）を対象にする。
//!
//! `onnx_interp_backend_dispatch.rs` と同型の直接 `Graph` 構築（decode
//! 層は経由しない。`interp::run` はグラフ実行の全経路であり decode 層の
//! 正しさは `onnx_decode.rs`／`onnx_interp.rs` が別途担保する）。
//!
//! 参照値の出自: ONNX 仕様の算術式（Clip／Gelu／ReduceMean／Pad は自明な
//! 手計算）または `crates/tensor-core::InterpolateMode` の doc が定める
//! 添字式（Nearest／NearestExact は整数演算で bit 完全一致、Bilinear は
//! REQ-2 統一複合判定）。いずれも `ops::*`／`Var::*` のコードを読まずに
//! 独立に導出した値であり、ORT／PyTorch 生成値ではない。

use std::collections::HashMap;

use fandhe_ai_onnx_interop::onnx::graph::Graph;
use fandhe_ai_onnx_interop::onnx::interp::{InterpError, Value, run};
use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, NodeProto, attribute_type};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

// ---- グラフ構築ヘルパ（`onnx_interp_backend_dispatch.rs` と同型） ----

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

fn node_with_attrs(
    op_type: &str,
    name: &str,
    input: Vec<&str>,
    output: Vec<&str>,
    attribute: Vec<AttributeProto>,
) -> NodeProto {
    let mut n = node(op_type, name, input, output);
    n.attribute = attribute;
    n
}

/// `r#type` を明示的に `FLOAT` へ設定した属性（`interp::attr_f32_typed`
/// の型検証を通す。`interp.rs` の `build_attr_f32` 相当だが `r#type` を
/// 設定する点が異なる）。
fn attr_f32_typed(name: &str, f: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f,
        r#type: attribute_type::FLOAT,
        ..Default::default()
    }
}

fn attr_i64_typed(name: &str, i: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i,
        r#type: attribute_type::INT,
        ..Default::default()
    }
}

fn attr_string_typed(name: &str, s: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        s: s.as_bytes().to_vec(),
        r#type: attribute_type::STRING,
        ..Default::default()
    }
}

fn attr_ints_typed(name: &str, ints: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints,
        r#type: attribute_type::INTS,
        ..Default::default()
    }
}

fn single_node_graph(n: NodeProto, inputs: &[&str]) -> Graph {
    let outputs = n.output.clone();
    Graph {
        nodes: vec![n],
        initializers: HashMap::new(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs,
    }
}

fn feed_f32(name: &str, data: Vec<f32>, shape: &[usize]) -> (String, Value) {
    (
        name.to_string(),
        Value::F32(Tensor::<f32>::new(data, shape).unwrap()),
    )
}

fn feed_i64(name: &str, data: Vec<i64>, shape: &[usize]) -> (String, Value) {
    (
        name.to_string(),
        Value::I64(Tensor::<i64>::new(data, shape).unwrap()),
    )
}

fn feed_bool(name: &str, data: Vec<bool>, shape: &[usize]) -> (String, Value) {
    (
        name.to_string(),
        Value::Bool(Tensor::<bool>::new(data, shape).unwrap()),
    )
}

fn as_f32_vec(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(t) => t.contiguous().as_slice().unwrap().to_vec(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

fn as_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::I64(t) => t.contiguous().as_slice().unwrap().to_vec(),
        other => panic!("Value::I64 を期待したが {other:?}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "要素数が一致しません");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: actual={a} expected={e} (tol={tol})"
        );
    }
}

// ================= Tanh =================

#[test]
fn tanh_matches_known_values() {
    let n = node("Tanh", "n", vec!["x"], vec!["y"]);
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![0.0, 1.0, -1.0], &[3])]);
    let result = run(&g, feeds).unwrap();
    // tanh(0)=0, tanh(1)≈0.7615941560, tanh(-1)≈-0.7615941560
    assert_close(
        &as_f32_vec(&result["y"]),
        &[0.0, 0.761_594_2, -0.761_594_2],
        1e-6,
    );
}

#[test]
fn tanh_rejects_wrong_arity() {
    let n = node("Tanh", "n", vec!["x", "extra"], vec!["y"]);
    let g = single_node_graph(n, &["x", "extra"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![0.0], &[1]),
        feed_f32("extra", vec![0.0], &[1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(
        err,
        InterpError::InputArityMismatch {
            min: 1,
            max: 1,
            actual: 2,
            ..
        }
    ));
}

// ================= Gelu =================

#[test]
fn gelu_default_approximate_none_matches_erf_formula() {
    let n = node("Gelu", "n", vec!["x"], vec!["y"]);
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![0.0, 1.0], &[2])]);
    let result = run(&g, feeds).unwrap();
    // Gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2))). Gelu(0)=0, Gelu(1)≈0.8413447461
    assert_close(&as_f32_vec(&result["y"]), &[0.0, 0.841_344_7], 1e-5);
}

#[test]
fn gelu_approximate_tanh_matches_tanh_formula() {
    let n = node_with_attrs(
        "Gelu",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_string_typed("approximate", "tanh")],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0], &[1])]);
    let result = run(&g, feeds).unwrap();
    // tanh 近似: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) ≈ 0.8411919906 for x=1
    assert_close(&as_f32_vec(&result["y"]), &[0.841_192], 1e-4);
}

#[test]
fn gelu_rejects_unsupported_approximate_value() {
    let n = node_with_attrs(
        "Gelu",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_string_typed("approximate", "sigmoid")],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0], &[1])]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(
        err,
        InterpError::InvalidAttribute { attr, .. } if attr == "approximate"
    ));
}

// ================= Clip =================

#[test]
fn clip_attr_form_clamps_to_min_max() {
    let n = node_with_attrs(
        "Clip",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_f32_typed("min", 0.0), attr_f32_typed("max", 6.0)],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![-2.0, 0.0, 3.0, 6.0, 10.0], &[5])]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["y"]), &[0.0, 0.0, 3.0, 6.0, 6.0], 1e-6);
}

#[test]
fn clip_input_form_min_only_uses_default_positive_infinity_for_max() {
    let n = node("Clip", "n", vec!["x", "min"], vec!["y"]);
    let g = single_node_graph(n, &["x", "min"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![-5.0, 5.0, 100.0], &[3]),
        feed_f32("min", vec![0.0], &[]),
    ]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["y"]), &[0.0, 5.0, 100.0], 1e-6);
}

#[test]
fn clip_min_greater_than_max_always_returns_max() {
    let n = node_with_attrs(
        "Clip",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_f32_typed("min", 5.0), attr_f32_typed("max", 1.0)],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![-10.0, 0.0, 10.0], &[3])]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["y"]), &[1.0, 1.0, 1.0], 1e-6);
}

#[test]
fn clip_rejects_mixed_attr_and_input_forms() {
    let n = node_with_attrs(
        "Clip",
        "n",
        vec!["x", "min"],
        vec!["y"],
        vec![attr_f32_typed("min", 0.0)],
    );
    let g = single_node_graph(n, &["x", "min"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0], &[1]),
        feed_f32("min", vec![0.0], &[]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { .. }));
}

#[test]
fn clip_rejects_non_scalar_min_input() {
    let n = node("Clip", "n", vec!["x", "min"], vec!["y"]);
    let g = single_node_graph(n, &["x", "min"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0], &[1]),
        feed_f32("min", vec![0.0, 0.0], &[2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "min"));
}

// ================= Where =================

#[test]
fn where_selects_elementwise_by_bool_cond() {
    let n = node("Where", "n", vec!["cond", "x", "y"], vec!["out"]);
    let g = single_node_graph(n, &["cond", "x", "y"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_bool("cond", vec![true, false, true], &[3]),
        feed_f32("x", vec![1.0, 2.0, 3.0], &[3]),
        feed_f32("y", vec![10.0, 20.0, 30.0], &[3]),
    ]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["out"]), &[1.0, 20.0, 3.0], 1e-6);
}

#[test]
fn where_broadcasts_cond_against_x_y() {
    // cond: [1] broadcast against x/y: [3]
    let n = node("Where", "n", vec!["cond", "x", "y"], vec!["out"]);
    let g = single_node_graph(n, &["cond", "x", "y"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_bool("cond", vec![true], &[1]),
        feed_f32("x", vec![1.0, 2.0, 3.0], &[3]),
        feed_f32("y", vec![10.0, 20.0, 30.0], &[3]),
    ]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["out"]), &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn where_rejects_non_bool_cond() {
    let n = node("Where", "n", vec!["cond", "x", "y"], vec!["out"]);
    let g = single_node_graph(n, &["cond", "x", "y"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("cond", vec![1.0], &[1]),
        feed_f32("x", vec![1.0], &[1]),
        feed_f32("y", vec![2.0], &[1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::TypeMismatch { expected, .. } if expected == "bool"));
}

// ================= Expand =================

#[test]
fn expand_bidirectional_broadcast_f32() {
    // data: [3,1], shape: [2,1,6] -> out: [2,3,6] (bidirectional)
    let n = node("Expand", "n", vec!["data", "shape"], vec!["y"]);
    let g = single_node_graph(n, &["data", "shape"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("data", vec![1.0, 2.0, 3.0], &[3, 1]),
        feed_i64("shape", vec![2, 1, 6], &[3]),
    ]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => assert_eq!(t.shape(), &[2, 3, 6]),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn expand_i64_uses_pure_copy_path() {
    let n = node("Expand", "n", vec!["data", "shape"], vec!["y"]);
    let g = single_node_graph(n, &["data", "shape"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_i64("data", vec![7], &[1]),
        feed_i64("shape", vec![3], &[1]),
    ]);
    let result = run(&g, feeds).unwrap();
    assert_eq!(as_i64_vec(&result["y"]), vec![7, 7, 7]);
}

#[test]
fn expand_rejects_negative_shape_element() {
    let n = node("Expand", "n", vec!["data", "shape"], vec!["y"]);
    let g = single_node_graph(n, &["data", "shape"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("data", vec![1.0], &[1]),
        feed_i64("shape", vec![-1], &[1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "shape"));
}

#[test]
fn expand_rejects_output_byte_size_overflow_instead_of_panicking() {
    // codex-review 指摘（PR #2313・イシュー #2186）: `broadcast_shape` は
    // 要素数積の `usize` オーバーフローのみを検査し、型ごとの確保バイト数
    // までは見ない。小さい入力（`[1]`）を極端に大きい shape へ拡張すると、
    // 要素数積自体は `usize` に収まっても `f32`（4 バイト）換算のバイト数が
    // `Vec` の allocation 上限（`isize::MAX` バイト）を超え、`contiguous()`
    // 内部の `Vec::with_capacity` が capacity overflow で panic し得た
    // （本番経路 panic 禁止方針 `.claude/rules/coding-rust.md` に反する）。
    // `check_expand_output_bytes` 導入後は型付きエラーを返すことを確認する。
    let huge = isize::MAX as usize / 4 + 1;
    let n = node("Expand", "n", vec!["data", "shape"], vec!["y"]);
    let g = single_node_graph(n, &["data", "shape"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("data", vec![1.0], &[1]),
        feed_i64("shape", vec![huge as i64], &[1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(
        err,
        InterpError::Shape(ShapeError::ElementCountOverflow)
    ));
}

#[test]
fn expand_rejects_output_within_isize_max_but_over_practical_byte_cap() {
    // レビュー指摘対応（codex P0・イシュー #2313）: `usize` オーバーフロー
    // も `isize::MAX` allocator 上限も超えない「数 GB 規模」の出力 shape
    // （小さい入力を極端に大きい shape へ拡張する攻撃的入力）は、従来の
    // オーバーフロー検査だけでは拒否できず `contiguous()` が実際に
    // 数 GB を確保してプロセスメモリを枯渇させ得た。実用的な上限
    // （`MAX_MATERIALIZE_BYTES` = 1 GiB）による拒否を確認する。
    let over_1gib_elements = (1usize << 30) / 4 + 1; // f32 換算で 1 GiB をわずかに超える要素数
    let n = node("Expand", "n", vec!["data", "shape"], vec!["y"]);
    let g = single_node_graph(n, &["data", "shape"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("data", vec![1.0], &[1]),
        feed_i64("shape", vec![over_1gib_elements as i64], &[1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "shape"));
}

// ================= ReduceMean =================

#[test]
fn reduce_mean_default_reduces_all_axes_with_keepdims() {
    let n = node("ReduceMean", "n", vec!["x"], vec!["y"]);
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => {
            assert_eq!(t.shape(), &[1, 1]);
            assert_close(t.contiguous().as_slice().unwrap(), &[2.5], 1e-6);
        }
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn reduce_mean_axes_attr_with_keepdims_false() {
    let n = node_with_attrs(
        "ReduceMean",
        "n",
        vec!["x"],
        vec!["y"],
        vec![
            attr_ints_typed("axes", vec![1]),
            attr_i64_typed("keepdims", 0),
        ],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => {
            assert_eq!(t.shape(), &[2]);
            assert_close(t.contiguous().as_slice().unwrap(), &[1.5, 3.5], 1e-6);
        }
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn reduce_mean_empty_axes_input_with_noop_is_identity() {
    let n = node_with_attrs(
        "ReduceMean",
        "n",
        vec!["x", "axes"],
        vec!["y"],
        vec![attr_i64_typed("noop_with_empty_axes", 1)],
    );
    let g = single_node_graph(n, &["x", "axes"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    feeds.insert(
        "axes".to_string(),
        Value::I64(Tensor::<i64>::new(vec![], &[0]).unwrap()),
    );
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["y"]), &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

#[test]
fn reduce_mean_omitted_axes_with_noop_is_identity() {
    // レビュー指摘対応（イシュー #2186 codex-review・PR #2313）: ONNX
    // ReduceMean-18 仕様上 `noop_with_empty_axes` は axes が「空」（省略・
    // 明示的な空リストのいずれも含む）の場合に適用される。axes を
    // 完全に省略（属性・第 2 入力ともになし）した場合も恒等演算になる
    // ことを確認する（`reduce_mean_empty_axes_input_with_noop_is_identity`
    // は空リスト明示のケースのみをカバーしていた）。
    let n = node_with_attrs(
        "ReduceMean",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_i64_typed("noop_with_empty_axes", 1)],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["y"]), &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

#[test]
fn reduce_mean_rejects_non_1d_axes_input() {
    // レビュー指摘対応（codex-review P0。security.md A03）: axes 入力は
    // ONNX 仕様上 1 次元テンソルでなければならない。shape [2,2]（要素数
    // 4）が長さ 4 の 1 次元 axes として誤って受理されないことを確認する。
    let n = node("ReduceMean", "n", vec!["x", "axes"], vec!["y"]);
    let g = single_node_graph(n, &["x", "axes"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0; 16], &[2, 2, 2, 2])]);
    feeds.insert(
        "axes".to_string(),
        Value::I64(Tensor::<i64>::new(vec![0, 1, 2, 3], &[2, 2]).unwrap()),
    );
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "axes"));
}

#[test]
fn reduce_mean_rejects_mixed_attr_and_input_axes() {
    let n = node_with_attrs(
        "ReduceMean",
        "n",
        vec!["x", "axes"],
        vec!["y"],
        vec![attr_ints_typed("axes", vec![0])],
    );
    let g = single_node_graph(n, &["x", "axes"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0], &[2])]);
    feeds.insert(
        "axes".to_string(),
        Value::I64(Tensor::<i64>::new(vec![0], &[1]).unwrap()),
    );
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "axes"));
}

// ================= Pad =================

#[test]
fn pad_attr_form_constant_default_value_zero() {
    let n = node_with_attrs(
        "Pad",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_ints_typed("pads", vec![1, 0, 0, 1])],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0], &[1, 2])]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_close(
                t.contiguous().as_slice().unwrap(),
                &[0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
                1e-6,
            );
        }
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn pad_input_form_with_constant_value() {
    let n = node("Pad", "n", vec!["x", "pads", "value"], vec!["y"]);
    let g = single_node_graph(n, &["x", "pads", "value"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![5.0], &[1]),
        feed_i64("pads", vec![1, 2], &[2]),
        feed_f32("value", vec![9.0], &[]),
    ]);
    let result = run(&g, feeds).unwrap();
    assert_close(&as_f32_vec(&result["y"]), &[9.0, 5.0, 9.0, 9.0], 1e-6);
}

#[test]
fn pad_rejects_mixed_value_attr_and_input_form() {
    // レビュー指摘対応（codex P0・イシュー #2313）: `pads` を第 2 入力
    // （入力形。Pad-11+）で渡しつつ、旧形式の `value` 属性（Pad-2）も
    // 同時に指定した混在ノードを拒否する。従来は `value` 属性を一切
    // 読まず埋め草値を暗黙的に 0.0 として計算しており、指定値（9.0）と
    // 異なる結果を無言で返していた（security.md A03）。
    let n = node_with_attrs(
        "Pad",
        "n",
        vec!["x", "pads"],
        vec!["y"],
        vec![attr_f32_typed("value", 9.0)],
    );
    let g = single_node_graph(n, &["x", "pads"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![5.0], &[1]),
        feed_i64("pads", vec![1, 1], &[2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "value"));
}

#[test]
fn pad_rejects_negative_pads() {
    let n = node_with_attrs(
        "Pad",
        "n",
        vec!["x"],
        vec!["y"],
        vec![attr_ints_typed("pads", vec![-1, 0])],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0], &[2])]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "pads"));
}

#[test]
fn pad_rejects_duplicate_axes() {
    // axes 入力形で同一軸を重複指定（[0, 0]）すると `result[n]` への
    // 代入が無言で後勝ち上書きされ、片方の pads 指定が消える。
    // レビュー指摘対応（イシュー #2186）: 重複軸は InvalidAttribute で
    // fail-closed に拒否する。
    let n = node("Pad", "n", vec!["x", "pads", "", "axes"], vec!["y"]);
    let g = single_node_graph(n, &["x", "pads", "axes"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0], &[1, 2]),
        feed_i64("pads", vec![1, 1, 0, 0], &[4]),
        feed_i64("axes", vec![0, 0], &[2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "axes"));
}

#[test]
fn pad_rejects_non_1d_pads_input() {
    // レビュー指摘対応（codex-review P0。security.md A03）: pads 入力は
    // ONNX 仕様上 1 次元テンソルでなければならない。shape [2,2]（要素数
    // 4）が長さ 4 の 1 次元 pads として誤って受理されないことを確認する。
    let n = node("Pad", "n", vec!["x", "pads"], vec!["y"]);
    let g = single_node_graph(n, &["x", "pads"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0], &[1, 2]),
        feed_i64("pads", vec![0, 0, 0, 0], &[2, 2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "pads"));
}

#[test]
fn pad_rejects_non_1d_axes_input() {
    // レビュー指摘対応（codex-review P0。security.md A03）: axes 入力も
    // 同様に 1 次元テンソルでなければならない。
    let n = node("Pad", "n", vec!["x", "pads", "", "axes"], vec!["y"]);
    let g = single_node_graph(n, &["x", "pads", "axes"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_i64("pads", vec![0, 0, 0, 0], &[4]),
        feed_i64("axes", vec![2, 3], &[1, 2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "axes"));
}

#[test]
fn pad_rejects_unsupported_mode() {
    let n = node_with_attrs(
        "Pad",
        "n",
        vec!["x"],
        vec!["y"],
        vec![
            attr_ints_typed("pads", vec![1, 1]),
            attr_string_typed("mode", "reflect"),
        ],
    );
    let g = single_node_graph(n, &["x"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0], &[2])]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "mode"));
}

// ================= Resize =================

#[test]
fn resize_nearest_asymmetric_floor_matches_integer_index_formula() {
    // ReLU6/mobilenet 相当の x2 nearest upsample。src = floor(dst*in/out)。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[4]),
    ]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => {
            assert_eq!(t.shape(), &[1, 1, 4, 4]);
            // 各入力画素が 2x2 ブロックへ複製される。
            assert_close(
                t.contiguous().as_slice().unwrap(),
                &[
                    1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
                ],
                1e-6,
            );
        }
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn resize_two_input_form_reads_second_input_as_scales() {
    // レビュー指摘対応（codex-review P1・Cursor Bugbot High。イシュー
    // #2186 PR #2313）: Resize-10 の 2 入力形式（`X`／`scales`）では
    // 第 2 入力を `roi` ではなく `scales` として読む必要がある。従来は
    // 常に第 2 入力を `roi` として検証していたため、正当な Resize-10
    // 2 入力モデルの `scales`（例: [2.0] のような非空値）が roi 検証
    // （空である必要がある）で誤って拒否されていた。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[4]),
    ]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => assert_eq!(t.shape(), &[1, 1, 4, 4]),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn resize_rejects_non_1d_scales_input() {
    // レビュー指摘対応（codex-review P0。security.md A03）: scales 入力
    // は ONNX 仕様上 1 次元テンソルでなければならない。shape [2,2]
    // （要素数 4）が長さ 4 の 1 次元 scales として誤って受理されない
    // ことを確認する。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![attr_string_typed("mode", "nearest")],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[2, 2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales"));
}

#[test]
fn resize_rejects_non_1d_sizes_input() {
    // レビュー指摘対応（codex-review P0。security.md A03）: sizes 入力
    // も同様に 1 次元テンソルでなければならない。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "", "sizes"],
        vec!["y"],
        vec![attr_string_typed("mode", "nearest")],
    );
    let g = single_node_graph(n, &["x", "sizes"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_i64("sizes", vec![1, 1, 4, 4], &[2, 2]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "sizes"));
}

#[test]
fn resize_rejects_nc_axis_scale_near_but_not_exactly_one() {
    // レビュー指摘対応（codex-review P1。イシュー #2186 PR #2313）:
    // N/C 軸は非対応（サイズ変更しない前提）のため倍率は厳密に 1.0 で
    // なければならない。旧実装は 1e-6 の許容差で判定していたため、
    // 1 以外の倍率（例: 0.9999995）を受理しつつ実際には N/C サイズを
    // 変更しない不整合が生じていた。厳密一致（`sv == 1.0`）で拒否
    // されることを確認する。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![0.999_999_5, 1.0, 2.0, 2.0], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales"));
}

#[test]
fn resize_sizes_form_accepted() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "", "sizes"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "sizes"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_i64("sizes", vec![1, 1, 4, 4], &[4]),
    ]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => assert_eq!(t.shape(), &[1, 1, 4, 4]),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn resize_rejects_output_over_practical_byte_cap() {
    // レビュー指摘対応（codex P0・イシュー #2313）: `Expand` と同じ
    // 実用的な上限（`MAX_MATERIALIZE_BYTES` = 1 GiB）を `interpolate`
    // 実体化前に検査する。`sizes` 入力（非信頼な外部 ONNX データ）に
    // よる H/W はオーバーフロー検査を通過しても数 GB 規模の確保を
    // 引き起こし得るため、`sizes` 経由でも拒否できることを確認する。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "", "sizes"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "sizes"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        // 1*1*40000*40000 要素 * 4 バイト（f32）≈ 6.4 GB > 1 GiB。
        feed_i64("sizes", vec![1, 1, 40_000, 40_000], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "sizes"));
}

#[test]
fn resize_linear_half_pixel_accepted() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![attr_string_typed("mode", "linear")],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[4]),
    ]);
    let result = run(&g, feeds).unwrap();
    match &result["y"] {
        Value::F32(t) => assert_eq!(t.shape(), &[1, 1, 4, 4]),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

#[test]
fn resize_rejects_default_onnx_nearest_combination() {
    // mode="nearest" の ONNX 既定（coordinate_transformation_mode=half_pixel・
    // nearest_mode=round_prefer_floor）は Nearest／NearestExact のどちらの
    // 添字式とも一致しないため拒否する（実装計画 §2.3 拒否表）。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![attr_string_typed("mode", "nearest")],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { .. }));
}

#[test]
fn resize_rejects_cubic_mode() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![attr_string_typed("mode", "cubic")],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "mode"));
}

#[test]
fn resize_rejects_non_integer_scale() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 1.5, 1.5], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales"));
}

#[test]
fn resize_rejects_scale_overflowing_usize_multiplication() {
    // レビュー指摘対応（イシュー #2186）: 非信頼な ONNX モデルの scales
    // 入力が usize::MAX を超える巨大な整数値浮動小数点数（例: 1e30）だと
    // `in_size * scale as usize` が usize 乗算オーバーフローを起こしうる
    // （debug ビルドでは panic、release ではラップして誤った shape）。
    // checked_mul による fail-closed 拒否を確認する。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 1e30, 1e30], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales"));
}

#[test]
fn resize_rejects_scale_overflowing_checked_mul_below_usize_max_guard() {
    // 上のテストは「usize::MAX を超える巨大な scale」の早期拒否ガードを
    // 通る。本テストは `2^63`（f32 で厳密に表現可能・`usize::MAX` 未満）
    // という早期ガードをすり抜ける値を使い、`in_size.checked_mul(scale)`
    // 自体が usize 乗算オーバーフローを検出して None を返す経路
    // （レビュー指摘の本丸）を直接確認する。
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        // in_size(H)=2 のときの H 方向 scale を 2^63 にすると
        // `2 * 2^63` が usize::MAX（2^64 - 1）を超えオーバーフローする。
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32(
            "scales",
            vec![1.0, 1.0, 9_223_372_036_854_775_808.0, 1.0],
            &[4],
        ),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales"));
}

#[test]
fn resize_rejects_nc_axis_scale_not_one() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![
            attr_string_typed("mode", "nearest"),
            attr_string_typed("coordinate_transformation_mode", "asymmetric"),
            attr_string_typed("nearest_mode", "floor"),
        ],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 2.0, 2.0, 2.0], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales"));
}

#[test]
fn resize_rejects_both_scales_and_sizes_specified() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales", "sizes"],
        vec!["y"],
        vec![attr_string_typed("mode", "nearest")],
    );
    let g = single_node_graph(n, &["x", "scales", "sizes"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        feed_f32("scales", vec![1.0, 1.0, 2.0, 2.0], &[4]),
        feed_i64("sizes", vec![1, 1, 4, 4], &[4]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "scales/sizes"));
}

#[test]
fn resize_rejects_rank_other_than_four() {
    let n = node_with_attrs(
        "Resize",
        "n",
        vec!["x", "", "scales"],
        vec!["y"],
        vec![attr_string_typed("mode", "nearest")],
    );
    let g = single_node_graph(n, &["x", "scales"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0], &[2]),
        feed_f32("scales", vec![2.0], &[1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "X"));
}

// ================= Gemm の固め（イシュー #2186） =================

#[test]
fn gemm_rejects_arity_outside_two_to_three() {
    let n = node("Gemm", "n", vec!["a"], vec!["y"]);
    let g = single_node_graph(n, &["a"]);
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("a", vec![1.0], &[1, 1])]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(
        err,
        InterpError::InputArityMismatch {
            min: 2,
            max: 3,
            actual: 1,
            ..
        }
    ));
}

#[test]
fn gemm_rejects_type_disguised_alpha_attribute() {
    // `alpha` を INT 型として送る（型偽装）。無検証読み取りなら黙って
    // ゼロ値の `f` を採用してしまう（イシュー #2076 の `Conv` P0 と同型の
    // 攻撃面。イシュー #2186 の Gemm 固め節）。
    let n = node_with_attrs(
        "Gemm",
        "n",
        vec!["a", "b"],
        vec!["y"],
        vec![attr_i64_typed("alpha", 5)],
    );
    let g = single_node_graph(n, &["a", "b"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 2.0], &[1, 2]),
        feed_f32("b", vec![1.0, 2.0], &[2, 1]),
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "alpha"));
}

#[test]
fn gemm_broadcast_zero_requires_c_shape_exactly_mn() {
    let n = node_with_attrs(
        "Gemm",
        "n",
        vec!["a", "b", "c"],
        vec!["y"],
        vec![attr_i64_typed("broadcast", 0)],
    );
    let g = single_node_graph(n, &["a", "b", "c"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 2.0], &[1, 2]),
        feed_f32("b", vec![1.0, 2.0], &[2, 1]),
        feed_f32("c", vec![1.0], &[1]), // [1] は [1,1] と異なる（broadcast=0 では不可）
    ]);
    let err = run(&g, feeds).unwrap_err();
    assert!(matches!(err, InterpError::InvalidAttribute { attr, .. } if attr == "broadcast"));
}

#[test]
fn gemm_broadcast_zero_accepts_c_shape_exactly_mn() {
    let n = node_with_attrs(
        "Gemm",
        "n",
        vec!["a", "b", "c"],
        vec!["y"],
        vec![attr_i64_typed("broadcast", 0)],
    );
    let g = single_node_graph(n, &["a", "b", "c"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 2.0], &[1, 2]),
        feed_f32("b", vec![1.0, 2.0], &[2, 1]),
        feed_f32("c", vec![10.0], &[1, 1]),
    ]);
    let result = run(&g, feeds).unwrap();
    // A@B = 1*1+2*2 = 5, +C(10) = 15
    assert_close(&as_f32_vec(&result["y"]), &[15.0], 1e-6);
}
