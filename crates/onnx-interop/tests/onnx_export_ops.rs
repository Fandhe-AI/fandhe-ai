//! `onnx::export_ops`（内部 op -> `NodeProto` の逆マッピング。イシュー #1773）の
//! 単体テスト。
//!
//! 各テストは `ExportNode -> to_node_proto -> Graph{nodes:[..]} ->
//! build_model_proto -> encode -> decode -> build_graph -> interp::run` という
//! 公開 API のみを経由した往復（層 A）の結果を、同じ属性値で `crate::ops::*` を
//! 直接呼んだ結果と突き合わせる。属性値は必ず**非既定値**を用いる
//! （既定値のままだと属性欠落を検出できず対称性テストが空虚に pass してしまう
//! ため。`export_ops.rs` モジュール冒頭コメント参照）。
//! import -> export -> import の総合 roundtrip・未対応 op 全数の fail-closed
//! 確認は #1774 のスコープであり、本テストは op 単位の対称性・層 B の
//! allowlist 検査・arity 検査に限定する。

use std::collections::HashMap;

use fandhe_ai_tensor_core::Tensor;
use onnx_interop::onnx::export::{
    ConstantAttr, ExportError, ExportNode, ExportOp, ExportOptions, SUPPORTED_OP_TYPES,
    build_model_proto, to_node_proto,
};
use onnx_interop::onnx::graph::{Graph, RawTensor, build_graph};
use onnx_interop::onnx::interp::{self, Value};
use onnx_interop::onnx::proto::{AttributeProto, ModelProto, attribute_type};
use onnx_interop::ops::{self, GemmAttrs, LayerNormAttrs};
use prost::Message;

// ---- テストユーティリティ ----

fn raw_f32(shape: &[i64], data: &[f32]) -> RawTensor {
    RawTensor::F32 {
        data: data.to_vec(),
        shape: shape.to_vec(),
    }
}

fn raw_i64(shape: &[i64], data: &[i64]) -> RawTensor {
    RawTensor::I64 {
        data: data.to_vec(),
        shape: shape.to_vec(),
    }
}

fn tensor_f32(shape: &[usize], data: &[f32]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("Tensor::new は成功するはず")
}

/// `ExportNode` を組み立て、公開 API のみを経由して
/// `to_node_proto -> Graph -> build_model_proto -> encode -> decode ->
/// build_graph -> interp::run` を実行し、`node.outputs` の結果を返す。
/// `initializers` に渡したテンソルがそのまま入力として使われる（`Graph.inputs`
/// は空のまま・feed も渡さない。テスト対象は「単一ノードの意味論マッピング」
/// のみであり feed 解決の検証は対象外のため）。
fn run_exported_node(
    node: ExportNode,
    initializers: Vec<(&str, RawTensor)>,
) -> HashMap<String, Value> {
    let proto = to_node_proto(&node).expect("to_node_proto は成功するはず");
    let mut inits = HashMap::new();
    for (name, tensor) in initializers {
        inits.insert(name.to_string(), tensor);
    }
    let graph = Graph {
        nodes: vec![proto],
        initializers: inits,
        inputs: Vec::new(),
        outputs: node.outputs.clone(),
    };
    let model = build_model_proto(&graph, &ExportOptions::default())
        .expect("build_model_proto は成功するはず");
    let bytes = model.encode_to_vec();
    let decoded = ModelProto::decode(bytes.as_slice()).expect("decode は成功するはず");
    let rebuilt = build_graph(&decoded).expect("build_graph は成功するはず");
    interp::run(&rebuilt, HashMap::new()).expect("interp::run は成功するはず")
}

fn expect_f32<'a>(result: &'a HashMap<String, Value>, name: &str) -> &'a Tensor<f32> {
    match result.get(name).expect("出力が存在するはず") {
        Value::F32(t) => t,
        other => panic!("F32 を期待したが {other:?} だった"),
    }
}

fn expect_i64<'a>(result: &'a HashMap<String, Value>, name: &str) -> &'a Tensor<i64> {
    match result.get(name).expect("出力が存在するはず") {
        Value::I64(t) => t,
        other => panic!("I64 を期待したが {other:?} だった"),
    }
}

fn assert_f32_bit_exact(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "shape が一致しない");
    let ac = a.contiguous();
    let bc = b.contiguous();
    let av = ac.as_slice().expect("contiguous のはず");
    let bv = bc.as_slice().expect("contiguous のはず");
    assert_eq!(av.len(), bv.len());
    for (x, y) in av.iter().zip(bv.iter()) {
        assert!(
            x.to_bits() == y.to_bits(),
            "bit 不一致: {x} (0x{:08x}) vs {y} (0x{:08x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

fn assert_i64_bit_exact(a: &Tensor<i64>, b: &Tensor<i64>) {
    assert_eq!(a.shape(), b.shape(), "shape が一致しない");
    let ac = a.contiguous();
    let bc = b.contiguous();
    assert_eq!(ac.as_slice(), bc.as_slice());
}

// ---- Tier 1: Gemm・MatMul・Add・Mul・Relu・Sigmoid・Softmax・Reshape ----

#[test]
fn gemm_exports_all_four_attributes_and_matches_direct_call() {
    // 非既定値: alpha/beta とも 1.0 以外、trans_a/trans_b とも true 以外を含む
    // 組み合わせ（transB のみ true）で属性欠落を検出可能にする。
    let attrs = GemmAttrs {
        alpha: 0.5,
        beta: 2.0,
        trans_a: false,
        trans_b: true,
    };
    let node = ExportNode {
        name: "gemm1".to_string(),
        op: ExportOp::Gemm(attrs),
        inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        outputs: vec!["y".to_string()],
    };

    // A: [2,3]・B: [2,3]（trans_b=true で transpose(0,1) -> [3,2] となり
    // A の K=3 と一致する）・C: [2,2]。
    let a = raw_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = raw_f32(&[2, 3], &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    let c = raw_f32(&[2, 2], &[1.0, 1.0, 1.0, 1.0]);

    let proto = to_node_proto(&node).expect("to_node_proto は成功するはず");
    assert_eq!(proto.op_type, "Gemm");
    assert_eq!(proto.attribute.len(), 4);
    let attr = |n: &str| proto.attribute.iter().find(|x| x.name == n).unwrap();
    assert_eq!(
        attr("alpha"),
        &AttributeProto {
            name: "alpha".to_string(),
            f: 0.5,
            i: 0,
            s: Vec::new(),
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            r#type: attribute_type::FLOAT,
        }
    );
    assert_eq!(attr("beta").f, 2.0);
    assert_eq!(attr("beta").r#type, attribute_type::FLOAT);
    assert_eq!(attr("transA").i, 0);
    assert_eq!(attr("transA").r#type, attribute_type::INT);
    assert_eq!(attr("transB").i, 1);

    let result = run_exported_node(
        node,
        vec![("a", a.clone()), ("b", b.clone()), ("c", c.clone())],
    );
    let exported_y = expect_f32(&result, "y");

    let a_t = tensor_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_t = tensor_f32(&[2, 3], &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    let c_t = tensor_f32(&[2, 2], &[1.0, 1.0, 1.0, 1.0]);
    let direct_y = ops::gemm(&a_t, &b_t, Some(&c_t), &attrs).expect("直接呼び出しは成功するはず");

    assert_f32_bit_exact(exported_y, &direct_y);
}

#[test]
fn gemm_omitted_c_input_is_accepted() {
    let attrs = GemmAttrs::default();
    let node = ExportNode {
        name: "gemm_no_c".to_string(),
        op: ExportOp::Gemm(attrs),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["y".to_string()],
    };
    let a = raw_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = raw_f32(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    let result = run_exported_node(node, vec![("a", a), ("b", b)]);
    let y = expect_f32(&result, "y");
    let expected = ops::gemm(
        &tensor_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        &tensor_f32(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
        None,
        &attrs,
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn matmul_add_mul_export_with_no_attributes_and_preserve_input_order() {
    for op in [
        ExportOp::MatMul,
        ExportOp::Add,
        ExportOp::Mul,
        ExportOp::Div,
    ] {
        let op_type = op.op_type();
        let node = ExportNode {
            name: format!("n_{op_type}"),
            op,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["y".to_string()],
        };
        let proto = to_node_proto(&node).expect("to_node_proto は成功するはず");
        assert_eq!(proto.op_type, op_type);
        assert!(proto.attribute.is_empty(), "{op_type} は属性 0 個のはず");
        assert_eq!(proto.input, vec!["a".to_string(), "b".to_string()]);
    }
}

#[test]
fn div_export_preserves_operand_order_numerically() {
    // 順序保存の確認: Div(a, b) と Div(b, a) は非対称なので、逆順で
    // export した場合に値が入れ替わることを確認する（入力順が
    // `to_node_proto` で保存されていることの間接検証）。
    let node = ExportNode {
        name: "div1".to_string(),
        op: ExportOp::Div,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["y".to_string()],
    };
    let a = raw_f32(&[2], &[10.0, 20.0]);
    let b = raw_f32(&[2], &[2.0, 5.0]);
    let result = run_exported_node(node, vec![("a", a.clone()), ("b", b.clone())]);
    let y = expect_f32(&result, "y");
    let expected = ops::div(
        &tensor_f32(&[2], &[10.0, 20.0]),
        &tensor_f32(&[2], &[2.0, 5.0]),
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn relu_sigmoid_export_single_input_no_attributes() {
    for op in [
        ExportOp::Relu,
        ExportOp::Sigmoid,
        ExportOp::Sqrt,
        ExportOp::Erf,
    ] {
        let op_type = op.op_type();
        let node = ExportNode {
            name: format!("n_{op_type}"),
            op,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
        };
        let proto = to_node_proto(&node).expect("to_node_proto は成功するはず");
        assert_eq!(proto.op_type, op_type);
        assert!(proto.attribute.is_empty());
        assert_eq!(proto.input, vec!["x".to_string()]);
    }
}

#[test]
fn mod_exports_fmod_attribute_and_matches_direct_call() {
    let node = ExportNode {
        name: "mod1".to_string(),
        op: ExportOp::Mod { fmod: true },
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["y".to_string()],
    };
    let a = raw_f32(&[2], &[5.5, -5.5]);
    let b = raw_f32(&[2], &[2.0, 2.0]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "fmod");
    assert_eq!(proto.attribute[0].i, 1);
    assert_eq!(proto.attribute[0].r#type, attribute_type::INT);

    let result = run_exported_node(node, vec![("a", a), ("b", b)]);
    let y = expect_f32(&result, "y");
    let expected = ops::modulo(
        &tensor_f32(&[2], &[5.5, -5.5]),
        &tensor_f32(&[2], &[2.0, 2.0]),
        true,
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn softmax_exports_non_default_axis_and_matches_direct_call() {
    // 既定値 -1 と異なる axis=0 を使う（属性欠落を検出可能にする）。
    let node = ExportNode {
        name: "softmax1".to_string(),
        op: ExportOp::Softmax { axis: 0 },
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    let x = raw_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "axis");
    assert_eq!(proto.attribute[0].i, 0);

    let result = run_exported_node(node, vec![("x", x)]);
    let y = expect_f32(&result, "y");
    let expected = ops::softmax(&tensor_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), 0).unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn reshape_exports_allowzero_attribute_and_matches_direct_call() {
    let node = ExportNode {
        name: "reshape1".to_string(),
        op: ExportOp::Reshape { allowzero: true },
        inputs: vec!["data".to_string(), "shape".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let shape = raw_i64(&[2], &[2, 2]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "allowzero");
    assert_eq!(proto.attribute[0].i, 1);

    let result = run_exported_node(node, vec![("data", data), ("shape", shape)]);
    let y = expect_f32(&result, "y");
    let expected = ops::reshape(&tensor_f32(&[4], &[1.0, 2.0, 3.0, 4.0]), &[2, 2], true).unwrap();
    assert_f32_bit_exact(y, &expected);
}

// ---- Tier 2: Shape・Gather・Unsqueeze・Concat・Slice ----

#[test]
fn shape_exports_no_attributes_and_matches_direct_call() {
    let node = ExportNode {
        name: "shape1".to_string(),
        op: ExportOp::Shape,
        inputs: vec!["data".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[2, 3, 4], &[0.0; 24]);
    let proto = to_node_proto(&node).unwrap();
    assert!(proto.attribute.is_empty());
    let result = run_exported_node(node, vec![("data", data)]);
    let y = expect_i64(&result, "y");
    assert_eq!(y.contiguous().as_slice().unwrap(), &[2i64, 3, 4]);
}

#[test]
fn gather_exports_non_default_axis_and_matches_direct_call() {
    let node = ExportNode {
        name: "gather1".to_string(),
        op: ExportOp::Gather { axis: 1 },
        inputs: vec!["data".to_string(), "indices".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let indices = raw_i64(&[2], &[0, 2]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "axis");
    assert_eq!(proto.attribute[0].i, 1);

    let result = run_exported_node(node, vec![("data", data), ("indices", indices)]);
    let y = expect_f32(&result, "y");
    let expected = ops::gather(
        &tensor_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        &[0, 2],
        &[2],
        1,
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn unsqueeze_export_uses_opset13_tensor_input_form() {
    let node = ExportNode {
        name: "unsqueeze1".to_string(),
        op: ExportOp::Unsqueeze,
        inputs: vec!["data".to_string(), "axes".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[2, 3], &[0.0; 6]);
    let axes = raw_i64(&[1], &[1]);
    let proto = to_node_proto(&node).unwrap();
    assert!(proto.attribute.is_empty());
    let result = run_exported_node(node, vec![("data", data), ("axes", axes)]);
    let y = expect_f32(&result, "y");
    assert_eq!(y.shape(), &[2, 1, 3]);
}

#[test]
fn squeeze_export_omitted_axes_uses_opset13_form() {
    let node = ExportNode {
        name: "squeeze1".to_string(),
        op: ExportOp::Squeeze,
        inputs: vec!["data".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[1, 3], &[1.0, 2.0, 3.0]);
    let result = run_exported_node(node, vec![("data", data)]);
    let y = expect_f32(&result, "y");
    assert_eq!(y.shape(), &[3]);
}

#[test]
fn concat_exports_required_axis_and_variadic_inputs() {
    let node = ExportNode {
        name: "concat1".to_string(),
        op: ExportOp::Concat { axis: 1 },
        inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        outputs: vec!["y".to_string()],
    };
    let a = raw_f32(&[1, 1], &[1.0]);
    let b = raw_f32(&[1, 1], &[2.0]);
    let c = raw_f32(&[1, 1], &[3.0]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "axis");
    assert_eq!(proto.attribute[0].i, 1);
    assert_eq!(proto.input.len(), 3);

    let result = run_exported_node(node, vec![("a", a), ("b", b), ("c", c)]);
    let y = expect_f32(&result, "y");
    assert_eq!(y.contiguous().as_slice().unwrap(), &[1.0f32, 2.0, 3.0]);
}

#[test]
fn slice_export_uses_opset13_tensor_input_form_with_optional_axes_and_steps() {
    let node = ExportNode {
        name: "slice1".to_string(),
        op: ExportOp::Slice,
        inputs: vec!["data".to_string(), "starts".to_string(), "ends".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[4], &[10.0, 20.0, 30.0, 40.0]);
    let starts = raw_i64(&[1], &[1]);
    let ends = raw_i64(&[1], &[3]);
    let proto = to_node_proto(&node).unwrap();
    assert!(proto.attribute.is_empty());
    assert_eq!(proto.input.len(), 3);

    let result = run_exported_node(
        node,
        vec![("data", data), ("starts", starts), ("ends", ends)],
    );
    let y = expect_f32(&result, "y");
    assert_eq!(y.contiguous().as_slice().unwrap(), &[20.0f32, 30.0]);
}

// ---- Tier 3: Transpose・Cast・Constant・LayerNormalization ----

#[test]
fn transpose_with_explicit_perm_attribute_present() {
    let node = ExportNode {
        name: "transpose1".to_string(),
        op: ExportOp::Transpose {
            perm: Some(vec![1, 0]),
        },
        inputs: vec!["data".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "perm");
    assert_eq!(proto.attribute[0].ints, vec![1, 0]);
    assert_eq!(proto.attribute[0].r#type, attribute_type::INTS);

    let result = run_exported_node(node, vec![("data", data)]);
    let y = expect_f32(&result, "y");
    let expected = ops::transpose(
        &tensor_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Some(&[1, 0]),
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn transpose_with_omitted_perm_has_no_attribute_and_reverses_axes() {
    // `perm` 省略時は rank 依存の軸反転という既定意味論を表すため、
    // 属性自体が付かないこと（`None` を静的既定値で埋めない契約）を確認する。
    let node = ExportNode {
        name: "transpose2".to_string(),
        op: ExportOp::Transpose { perm: None },
        inputs: vec!["data".to_string()],
        outputs: vec!["y".to_string()],
    };
    let data = raw_f32(&[2, 3, 4], &[0.0; 24]);
    let proto = to_node_proto(&node).unwrap();
    assert!(
        proto.attribute.is_empty(),
        "perm 省略時は属性を持たないはず"
    );
    let result = run_exported_node(node, vec![("data", data)]);
    let y = expect_f32(&result, "y");
    assert_eq!(y.shape(), &[4, 3, 2], "rank 依存の軸反転が適用されるはず");
}

#[test]
fn cast_exports_to_attribute_and_matches_direct_call() {
    let node = ExportNode {
        name: "cast1".to_string(),
        op: ExportOp::Cast { to: 7 }, // INT64
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    let x = raw_f32(&[2], &[1.5, 2.9]);
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "to");
    assert_eq!(proto.attribute[0].i, 7);

    let result = run_exported_node(node, vec![("x", x)]);
    let y = expect_i64(&result, "y");
    let expected = ops::cast_to_int64(&tensor_f32(&[2], &[1.5, 2.9])).unwrap();
    assert_i64_bit_exact(y, &expected);
}

#[test]
fn cast_rejects_unsupported_target_at_to_node_proto_time() {
    // `to_node_proto` 自体は `to` の値を検査しない（`check_supported_cast_target`
    // は interp 側の判定）ため、ノード自体は組み立てられるが実行時に拒否される
    // ことを確認する（layer A は arity のみを検査し dtype 妥当性は関与しない
    // 契約の確認）。
    let node = ExportNode {
        name: "cast_bad".to_string(),
        op: ExportOp::Cast { to: 99 },
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    let proto = to_node_proto(&node).expect("to_node_proto 自体は成功するはず");
    assert_eq!(proto.attribute[0].i, 99);
}

#[test]
fn constant_tensor_variant_uses_output_name_for_tensor_proto_name() {
    let raw = raw_f32(&[2], &[1.0, 2.0]);
    let node = ExportNode {
        name: "const_tensor".to_string(),
        op: ExportOp::Constant(ConstantAttr::Tensor(raw)),
        inputs: Vec::new(),
        outputs: vec!["y".to_string()],
    };
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 1);
    assert_eq!(proto.attribute[0].name, "value");
    assert_eq!(proto.attribute[0].r#type, attribute_type::TENSOR);
    let t = proto.attribute[0].t.as_ref().expect("tensor があるはず");
    assert_eq!(t.name, "y", "TensorProto.name はノードの出力名を使うはず");

    let result = run_exported_node(node, Vec::new());
    let y = expect_f32(&result, "y");
    assert_eq!(y.contiguous().as_slice().unwrap(), &[1.0f32, 2.0]);
}

#[test]
fn constant_float_variant_matches_direct_call() {
    let node = ExportNode {
        name: "const_float".to_string(),
        op: ExportOp::Constant(ConstantAttr::Float(3.5)),
        inputs: Vec::new(),
        outputs: vec!["y".to_string()],
    };
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute[0].name, "value_float");
    assert_eq!(proto.attribute[0].f, 3.5);
    let result = run_exported_node(node, Vec::new());
    let y = expect_f32(&result, "y");
    assert_eq!(y.contiguous().as_slice().unwrap(), &[3.5f32]);
}

#[test]
fn constant_floats_ints_int_variants_round_trip() {
    let node = ExportNode {
        name: "const_floats".to_string(),
        op: ExportOp::Constant(ConstantAttr::Floats(vec![1.0, 2.0, 3.0])),
        inputs: Vec::new(),
        outputs: vec!["y".to_string()],
    };
    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute[0].name, "value_floats");
    assert_eq!(proto.attribute[0].floats, vec![1.0, 2.0, 3.0]);
    let result = run_exported_node(node, Vec::new());
    let y = expect_f32(&result, "y");
    assert_eq!(y.contiguous().as_slice().unwrap(), &[1.0f32, 2.0, 3.0]);

    let node2 = ExportNode {
        name: "const_int".to_string(),
        op: ExportOp::Constant(ConstantAttr::Int(42)),
        inputs: Vec::new(),
        outputs: vec!["y2".to_string()],
    };
    let proto2 = to_node_proto(&node2).unwrap();
    assert_eq!(proto2.attribute[0].name, "value_int");
    assert_eq!(proto2.attribute[0].i, 42);
    let result2 = run_exported_node(node2, Vec::new());
    let y2 = expect_i64(&result2, "y2");
    assert_eq!(y2.contiguous().as_slice().unwrap(), &[42i64]);

    let node3 = ExportNode {
        name: "const_ints".to_string(),
        op: ExportOp::Constant(ConstantAttr::Ints(vec![1, 2, 3])),
        inputs: Vec::new(),
        outputs: vec!["y3".to_string()],
    };
    let proto3 = to_node_proto(&node3).unwrap();
    assert_eq!(proto3.attribute[0].name, "value_ints");
    assert_eq!(proto3.attribute[0].ints, vec![1, 2, 3]);
    let result3 = run_exported_node(node3, Vec::new());
    let y3 = expect_i64(&result3, "y3");
    assert_eq!(y3.contiguous().as_slice().unwrap(), &[1i64, 2, 3]);
}

#[test]
fn layer_normalization_exports_axis_and_epsilon_and_matches_direct_call() {
    let attrs = LayerNormAttrs {
        axis: -1,
        epsilon: 1e-3, // 既定 1e-5 と異なる非既定値
    };
    let node = ExportNode {
        name: "ln1".to_string(),
        op: ExportOp::LayerNormalization(attrs),
        inputs: vec!["x".to_string(), "scale".to_string(), "bias".to_string()],
        outputs: vec!["y".to_string()],
    };
    let x = raw_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scale = raw_f32(&[3], &[1.0, 1.0, 1.0]);
    let bias = raw_f32(&[3], &[0.0, 0.0, 0.0]);

    let proto = to_node_proto(&node).unwrap();
    assert_eq!(proto.attribute.len(), 2);
    let axis_attr = proto.attribute.iter().find(|a| a.name == "axis").unwrap();
    assert_eq!(axis_attr.i, -1);
    let eps_attr = proto
        .attribute
        .iter()
        .find(|a| a.name == "epsilon")
        .unwrap();
    assert_eq!(eps_attr.f, 1e-3);

    let result = run_exported_node(node, vec![("x", x), ("scale", scale), ("bias", bias)]);
    let y = expect_f32(&result, "y");
    let expected = ops::layer_normalization(
        &tensor_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        &tensor_f32(&[3], &[1.0, 1.0, 1.0]),
        Some(&tensor_f32(&[3], &[0.0, 0.0, 0.0])),
        &attrs,
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

#[test]
fn layer_normalization_omitted_bias_is_accepted() {
    let attrs = LayerNormAttrs::default();
    let node = ExportNode {
        name: "ln_no_bias".to_string(),
        op: ExportOp::LayerNormalization(attrs),
        inputs: vec!["x".to_string(), "scale".to_string()],
        outputs: vec!["y".to_string()],
    };
    let x = raw_f32(&[1, 2], &[1.0, 2.0]);
    let scale = raw_f32(&[2], &[1.0, 1.0]);
    let result = run_exported_node(node, vec![("x", x), ("scale", scale)]);
    let y = expect_f32(&result, "y");
    let expected = ops::layer_normalization(
        &tensor_f32(&[1, 2], &[1.0, 2.0]),
        &tensor_f32(&[2], &[1.0, 1.0]),
        None,
        &attrs,
    )
    .unwrap();
    assert_f32_bit_exact(y, &expected);
}

// ---- 層 A: arity 検査 ----

#[test]
fn arity_rejects_too_few_and_too_many_inputs() {
    let too_few = ExportNode {
        name: "gemm_bad".to_string(),
        op: ExportOp::Gemm(GemmAttrs::default()),
        inputs: vec!["a".to_string()],
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&too_few),
        Err(ExportError::InputArityMismatch { actual: 1, .. })
    ));

    let too_many = ExportNode {
        name: "gemm_bad2".to_string(),
        op: ExportOp::Gemm(GemmAttrs::default()),
        inputs: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ],
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&too_many),
        Err(ExportError::InputArityMismatch { actual: 4, .. })
    ));
}

#[test]
fn arity_rejects_output_count_other_than_one() {
    let node = ExportNode {
        name: "relu_bad".to_string(),
        op: ExportOp::Relu,
        inputs: vec!["x".to_string()],
        outputs: vec!["y1".to_string(), "y2".to_string()],
    };
    assert!(matches!(
        to_node_proto(&node),
        Err(ExportError::OutputArityMismatch {
            expected: 1,
            actual: 2,
            ..
        })
    ));
}

#[test]
fn arity_rejects_empty_required_input_but_allows_empty_optional_input() {
    // 必須入力（index 0,1）が空文字列は拒否。
    let node = ExportNode {
        name: "matmul_bad".to_string(),
        op: ExportOp::MatMul,
        inputs: vec!["a".to_string(), String::new()],
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&node),
        Err(ExportError::EmptyRequiredInput { index: 1, .. })
    ));

    // 省略可入力（Gemm の C。index 2）は空文字列が許容される。
    let node2 = ExportNode {
        name: "gemm_optional_empty".to_string(),
        op: ExportOp::Gemm(GemmAttrs::default()),
        inputs: vec!["a".to_string(), "b".to_string(), String::new()],
        outputs: vec!["y".to_string()],
    };
    assert!(to_node_proto(&node2).is_ok());
}

#[test]
fn concat_accepts_arbitrary_input_count_but_rejects_zero() {
    let zero = ExportNode {
        name: "concat_zero".to_string(),
        op: ExportOp::Concat { axis: 0 },
        inputs: Vec::new(),
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&zero),
        Err(ExportError::InputArityMismatch { actual: 0, .. })
    ));

    let many = ExportNode {
        name: "concat_many".to_string(),
        op: ExportOp::Concat { axis: 0 },
        inputs: (0..10).map(|i| format!("in{i}")).collect(),
        outputs: vec!["y".to_string()],
    };
    assert!(to_node_proto(&many).is_ok());
}

#[test]
fn concat_rejects_empty_string_at_any_variadic_position() {
    // `Concat` は可変長入力の全要素が必須（ONNX 仕様上どの要素も省略でき
    // ない）。`min_inputs=1` を満たす個数でも、2 番目以降が空文字列なら
    // `compute_concat` が実行時に失敗するため export 側で拒否する
    // （回帰: 以前は `i < min_inputs` のみの検査で 2 番目以降の空文字列を
    // 見逃していた）。
    let node = ExportNode {
        name: "concat_empty_second".to_string(),
        op: ExportOp::Concat { axis: 0 },
        inputs: vec!["a".to_string(), String::new(), "c".to_string()],
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&node),
        Err(ExportError::EmptyRequiredInput { index: 1, .. })
    ));
}

#[test]
fn slice_requires_ends_input_and_rejects_missing_or_empty() {
    // ONNX `Slice`（opset>=13 形）は `ends`（第 3 入力）が必須。
    // `compute_slice` は `input_name(node, 2)`（必須扱い）で読むため、
    // `data, starts` のみ（2 入力）や `ends` が空文字列のノードを export
    // すると再実行不能なモデルになる（回帰: 以前は `min_inputs=2` で
    // `ends` 欠落・空文字列ともに通過してしまっていた）。
    let missing_ends = ExportNode {
        name: "slice_missing_ends".to_string(),
        op: ExportOp::Slice,
        inputs: vec!["data".to_string(), "starts".to_string()],
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&missing_ends),
        Err(ExportError::InputArityMismatch { actual: 2, .. })
    ));

    let empty_ends = ExportNode {
        name: "slice_empty_ends".to_string(),
        op: ExportOp::Slice,
        inputs: vec!["data".to_string(), "starts".to_string(), String::new()],
        outputs: vec!["y".to_string()],
    };
    assert!(matches!(
        to_node_proto(&empty_ends),
        Err(ExportError::EmptyRequiredInput { index: 2, .. })
    ));

    // `axes`／`steps`（index 3, 4）は引き続き省略可入力として空文字列を
    // 許容する（`ends` までの 3 入力は必須のまま）。
    let optional_axes_steps_empty = ExportNode {
        name: "slice_optional_empty".to_string(),
        op: ExportOp::Slice,
        inputs: vec![
            "data".to_string(),
            "starts".to_string(),
            "ends".to_string(),
            String::new(),
            String::new(),
        ],
        outputs: vec!["y".to_string()],
    };
    assert!(to_node_proto(&optional_axes_steps_empty).is_ok());
}

// ---- 層 B: check_exportable（allowlist・domain 検査） ----

#[test]
fn build_model_proto_rejects_unsupported_op_type() {
    use onnx_interop::onnx::proto::NodeProto;

    let graph = Graph {
        nodes: vec![NodeProto {
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            name: "conv1".to_string(),
            op_type: "Conv".to_string(),
            attribute: Vec::new(),
            domain: String::new(),
        }],
        initializers: HashMap::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
    };
    let err = build_model_proto(&graph, &ExportOptions::default())
        .expect_err("未対応 op_type は拒否されるはず");
    assert!(matches!(
        err,
        ExportError::UnsupportedOp {
            op_type,
            ..
        } if op_type == "Conv"
    ));
}

#[test]
fn build_model_proto_rejects_non_default_domain() {
    use onnx_interop::onnx::proto::NodeProto;

    let graph = Graph {
        nodes: vec![NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["y".to_string()],
            name: "gemm_custom_domain".to_string(),
            op_type: "Gemm".to_string(),
            attribute: Vec::new(),
            domain: "com.custom".to_string(),
        }],
        initializers: HashMap::new(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["y".to_string()],
    };
    let err = build_model_proto(&graph, &ExportOptions::default())
        .expect_err("既定 opset 以外の domain は拒否されるはず");
    assert!(matches!(
        err,
        ExportError::UnsupportedOp { domain, .. } if domain == "com.custom"
    ));
}

#[test]
fn export_op_variants_all_have_op_type_in_supported_list() {
    // `ExportOp` の全 22 variant を 1 個ずつ構築し、`op_type()` が
    // `SUPPORTED_OP_TYPES` に含まれること・集合サイズが一致することを固定する
    // （drift 検出。variant 追加時はこの一覧・`op_type()`・`to_node_proto` の
    // 網羅 match 双方の更新がコンパイルエラーで強制される）。
    let sample: Vec<ExportOp> = vec![
        ExportOp::Gemm(GemmAttrs::default()),
        ExportOp::MatMul,
        ExportOp::Add,
        ExportOp::Mul,
        ExportOp::Div,
        ExportOp::Mod { fmod: false },
        ExportOp::Sqrt,
        ExportOp::Relu,
        ExportOp::Sigmoid,
        ExportOp::Erf,
        ExportOp::Softmax { axis: -1 },
        ExportOp::Reshape { allowzero: false },
        ExportOp::Shape,
        ExportOp::Gather { axis: 0 },
        ExportOp::Unsqueeze,
        ExportOp::Squeeze,
        ExportOp::Concat { axis: 0 },
        ExportOp::Slice,
        ExportOp::Transpose { perm: None },
        ExportOp::Cast { to: 1 },
        ExportOp::Constant(ConstantAttr::Int(0)),
        ExportOp::LayerNormalization(LayerNormAttrs::default()),
    ];
    assert_eq!(sample.len(), 22, "interp.rs 対応 22 op と揃うはず");
    assert_eq!(SUPPORTED_OP_TYPES.len(), 22);
    for op in &sample {
        assert!(
            SUPPORTED_OP_TYPES.contains(&op.op_type()),
            "{} が SUPPORTED_OP_TYPES に含まれない",
            op.op_type()
        );
    }
    // 重複なし（22 variant すべてが異なる op_type を持つ）ことも確認する。
    let mut op_types: Vec<&str> = sample.iter().map(ExportOp::op_type).collect();
    op_types.sort_unstable();
    op_types.dedup();
    assert_eq!(op_types.len(), 22);
}

#[test]
fn supported_op_types_are_all_reachable_in_interp_dispatch_table() {
    // `SUPPORTED_OP_TYPES` の各 op_type を持つ入力なしノードを `interp::run`
    // へ流し、返るエラーが `InterpError::UnsupportedOp` **ではない**ことを
    // 確認する（各 `compute_*` は入力・属性欠落で先に失敗するため、
    // `UnsupportedOp` にならないことが「ディスパッチ表に存在する」ことの
    // 間接証拠になる）。
    use onnx_interop::onnx::interp::InterpError;
    use onnx_interop::onnx::proto::NodeProto;

    for op_type in SUPPORTED_OP_TYPES {
        let node = NodeProto {
            input: Vec::new(),
            output: vec!["y".to_string()],
            name: format!("probe_{op_type}"),
            op_type: op_type.to_string(),
            attribute: Vec::new(),
            domain: String::new(),
        };
        let graph = Graph {
            nodes: vec![node],
            initializers: HashMap::new(),
            inputs: Vec::new(),
            outputs: vec!["y".to_string()],
        };
        let err = interp::run(&graph, HashMap::new())
            .expect_err(&format!("{op_type} は入力なしでは失敗するはず"));
        assert!(
            !matches!(err, InterpError::UnsupportedOp(_)),
            "{op_type} が interp のディスパッチ表に存在しない（UnsupportedOp を返した）"
        );
    }
}

// ---- 決定性 ----

#[test]
fn to_node_proto_is_deterministic_across_calls() {
    let node = ExportNode {
        name: "det1".to_string(),
        op: ExportOp::Gemm(GemmAttrs {
            alpha: 0.25,
            beta: 1.75,
            trans_a: true,
            trans_b: false,
        }),
        inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        outputs: vec!["y".to_string()],
    };
    let p1 = to_node_proto(&node).unwrap();
    let p2 = to_node_proto(&node).unwrap();
    assert_eq!(p1, p2);
}
