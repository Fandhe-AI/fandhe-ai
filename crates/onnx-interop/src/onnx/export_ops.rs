//! 内部 op（Rust ネイティブの属性表現）から `NodeProto`（op_type・属性）への
//! 逆マッピング（イシュー #1773。`onnx::export` の層 A）。
//!
//! `onnx::interp` が `NodeProto` から読む 22 op（`interp.rs` の `run` ディスパッチ表）と
//! 対称になるよう、[`ExportOp`] は同じ 22 op を Rust ネイティブの属性表現（`interp.rs`
//! の `attr_f32`／`attr_i64`／`attr_i64s`／`attr_i64_required` が読む値と同じ型）として
//! 保持する。属性は **常に全て書き出す**（既定値であっても省略しない。省略すると
//! 「属性欠落＝既定値」という対称性テストが空虚に pass してしまうため。唯一の例外は
//! [`ExportOp::Transpose`] の `perm`: 省略時の意味論が入力 rank に依存する
//! （`interp.rs::compute_transpose` -> `ops::transpose(t, None)` が rank 依存の軸反転を
//! 適用する）ため、`None` を静的な既定値で埋めずそのまま省略する）。
//!
//! autodiff の `Op`／`Tape` から本モジュールの `ExportOp` への橋渡しは親 #1653／#1775
//! のスコープであり、本モジュールはそれには関与しない（`onnx-interop` は crates.io
//! 非公開クレートであり facade へは一切公開しない。`export.rs` モジュール冒頭コメント
//! 参照）。

use super::export::{ExportError, encode_tensor};
use super::graph::{Graph, RawTensor};
use super::proto::{AttributeProto, NodeProto, attribute_type};
use crate::ops::{GemmAttrs, LayerNormAttrs};

/// `Constant` の排他属性群（ONNX Constant-13 仕様。`interp.rs::compute_constant` の
/// 逆方向）。1 属性のみを書き出す（複数指定は許容しない設計。呼び出し元が
/// この enum で択一を強制される）。
#[derive(Debug, Clone)]
pub enum ConstantAttr {
    /// `value`（TENSOR）。`t.name` は書き出し先ノードの出力名を決定的に用いる
    /// （`to_node_proto` 参照。`TensorProto.name` は import 側〈`decode_tensor`〉が
    /// エラーメッセージにしか使わないため衝突の実害はない）。
    Tensor(RawTensor),
    /// `value_float`（FLOAT）。
    Float(f32),
    /// `value_floats`（FLOATS）。
    Floats(Vec<f32>),
    /// `value_int`（INT）。
    Int(i64),
    /// `value_ints`（INTS）。
    Ints(Vec<i64>),
}

/// `interp.rs` が対応する 22 op を Rust ネイティブの属性表現として保持する。
/// 入力・出力の名前列は [`ExportNode`] 側が持つ（`ExportOp` 自体は op_type と
/// 属性のみの責務）。
#[derive(Debug, Clone)]
pub enum ExportOp {
    /// `Y = alpha * (A' @ B') + beta * C`。属性 4 つ（alpha／beta／transA／transB）。
    Gemm(GemmAttrs),
    /// `MatMul(A, B)`。属性なし。
    MatMul,
    /// `Add(A, B)`。属性なし。
    Add,
    /// `Mul(A, B)`。属性なし。
    Mul,
    /// `Div(A, B)`。属性なし。
    Div,
    /// `Mod(A, B)`。属性 `fmod`（INT。0/1）。
    Mod { fmod: bool },
    /// `Sqrt(X)`。属性なし。
    Sqrt,
    /// `Relu(X)`。属性なし。
    Relu,
    /// `Sigmoid(X)`。属性なし。
    Sigmoid,
    /// `Erf(X)`。属性なし。
    Erf,
    /// `Softmax(X)`。属性 `axis`（INT）。
    Softmax { axis: i64 },
    /// `Reshape(data, shape)`。属性 `allowzero`（INT。0/1）。`shape` は
    /// opset>=13 形（第 2 入力のテンソル名）で表す（attr 形〈opset<13〉は
    /// export しない。モジュール冒頭コメント参照）。
    Reshape { allowzero: bool },
    /// `Shape(data)`。属性なし。
    Shape,
    /// `Gather(data, indices)`。属性 `axis`（INT）。
    Gather { axis: i64 },
    /// `Unsqueeze(data, axes)`。opset>=13 形（`axes` は第 2 入力）。属性なし。
    Unsqueeze,
    /// `Squeeze(data, [axes])`。opset>=13 形（`axes` は省略可の第 2 入力）。
    /// 属性なし。
    Squeeze,
    /// `Concat(inputs...)`（1 個以上・可変長）。属性 `axis`（INT。必須）。
    Concat { axis: i64 },
    /// `Slice(data, starts, ends, [axes], [steps])`。opset>=13 形（すべて入力
    /// テンソル）。属性なし。
    Slice,
    /// `Transpose(data)`。属性 `perm`（INTS。省略可）。`None` は「rank 依存の
    /// 軸反転」という既定意味論を表し、属性自体を省略する（モジュール冒頭
    /// コメント参照）。
    Transpose { perm: Option<Vec<i64>> },
    /// `Cast(input)`。属性 `to`（INT。`ops::check_supported_cast_target` の
    /// 対応範囲 1／7／9／10 のみ）。
    Cast { to: i64 },
    /// `Constant()`。入力 0 個。排他属性群は [`ConstantAttr`] で表す。
    Constant(ConstantAttr),
    /// `LayerNormalization(X, Scale, [B])`。属性 `axis`（INT）／`epsilon`（FLOAT）。
    LayerNormalization(LayerNormAttrs),
}

impl ExportOp {
    /// この op が書き出す `NodeProto.op_type` 文字列。
    pub fn op_type(&self) -> &'static str {
        match self {
            ExportOp::Gemm(_) => "Gemm",
            ExportOp::MatMul => "MatMul",
            ExportOp::Add => "Add",
            ExportOp::Mul => "Mul",
            ExportOp::Div => "Div",
            ExportOp::Mod { .. } => "Mod",
            ExportOp::Sqrt => "Sqrt",
            ExportOp::Relu => "Relu",
            ExportOp::Sigmoid => "Sigmoid",
            ExportOp::Erf => "Erf",
            ExportOp::Softmax { .. } => "Softmax",
            ExportOp::Reshape { .. } => "Reshape",
            ExportOp::Shape => "Shape",
            ExportOp::Gather { .. } => "Gather",
            ExportOp::Unsqueeze => "Unsqueeze",
            ExportOp::Squeeze => "Squeeze",
            ExportOp::Concat { .. } => "Concat",
            ExportOp::Slice => "Slice",
            ExportOp::Transpose { .. } => "Transpose",
            ExportOp::Cast { .. } => "Cast",
            ExportOp::Constant(_) => "Constant",
            ExportOp::LayerNormalization(_) => "LayerNormalization",
        }
    }
}

/// 内部 op 1 個分の export 単位（`NodeProto` へ組み立てる前の中間表現）。
/// `name` はノード名（`NodeProto.name`）、`inputs`／`outputs` は入力・出力の
/// テンソル名列（`NodeProto.input`／`output`。省略可入力は空文字列で表す。
/// `interp.rs` の `node.input.get(N)` 由来の慣習と対称）。
#[derive(Debug, Clone)]
pub struct ExportNode {
    pub name: String,
    pub op: ExportOp,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// `interp.rs` が対応する 22 op の `op_type` 一覧（[`ExportOp::op_type`] が返す
/// 値の集合と同一）。`check_exportable` の allowlist として使う。両者のドリフトは
/// `#[cfg(test)]` のドリフト検出テストで固定する。
pub const SUPPORTED_OP_TYPES: &[&str] = &[
    "Gemm",
    "MatMul",
    "Add",
    "Mul",
    "Div",
    "Mod",
    "Sqrt",
    "Relu",
    "Sigmoid",
    "Erf",
    "Softmax",
    "Reshape",
    "Shape",
    "Gather",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Slice",
    "Transpose",
    "Cast",
    "Constant",
    "LayerNormalization",
];

fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: value,
        i: 0,
        s: Vec::new(),
        t: None,
        floats: Vec::new(),
        ints: Vec::new(),
        r#type: attribute_type::FLOAT,
    }
}

fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: value,
        s: Vec::new(),
        t: None,
        floats: Vec::new(),
        ints: Vec::new(),
        r#type: attribute_type::INT,
    }
}

fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        floats: Vec::new(),
        ints: values.to_vec(),
        r#type: attribute_type::INTS,
    }
}

fn attr_floats(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        floats: values.to_vec(),
        ints: Vec::new(),
        r#type: attribute_type::FLOATS,
    }
}

/// `value`（TENSOR）属性を組み立てる。`tensor_name` は書き出す `TensorProto.name`
/// （モジュール冒頭コメントのとおりノードの出力名を決定的に用いる。呼び出し元
/// `to_node_proto` 参照）。`encode_tensor` の検証（shape 非負性・要素数一致）を
/// そのまま継承する。
fn attr_tensor(
    name: &str,
    tensor_name: &str,
    raw: &RawTensor,
) -> Result<AttributeProto, ExportError> {
    let t = encode_tensor(tensor_name, raw)?;
    Ok(AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: Some(t),
        floats: Vec::new(),
        ints: Vec::new(),
        r#type: attribute_type::TENSOR,
    })
}

/// `inputs`／`outputs` の arity を検査する（`interp.rs` の各 `compute_*` が読む
/// 入力個数・`require_single_output` と対称）。`min_inputs..=max_inputs`
/// （`max_inputs = usize::MAX` は上限なし＝可変長。`Concat` 専用）の範囲外は
/// [`ExportError::InputArityMismatch`]、出力が 1 個以外は
/// [`ExportError::OutputArityMismatch`] を返す。`min_inputs` 未満の位置（必須
/// 入力）が空文字列の場合は [`ExportError::EmptyRequiredInput`]（省略可入力
/// 〈`min_inputs` 以降〉は空文字列を許容する。`interp.rs` の
/// `node.input.get(N)` が `Some(name) if !name.is_empty()` で判定する慣習と対称）。
fn check_arity(
    node_name: &str,
    op_type: &'static str,
    inputs: &[String],
    outputs: &[String],
    min_inputs: usize,
    max_inputs: usize,
) -> Result<(), ExportError> {
    if outputs.len() != 1 {
        return Err(ExportError::OutputArityMismatch {
            node_name: node_name.to_string(),
            op_type,
            expected: 1,
            actual: outputs.len(),
        });
    }
    if inputs.len() < min_inputs || inputs.len() > max_inputs {
        let expected = if max_inputs == usize::MAX {
            format!(">={min_inputs}")
        } else if min_inputs == max_inputs {
            format!("{min_inputs}")
        } else {
            format!("{min_inputs}..={max_inputs}")
        };
        return Err(ExportError::InputArityMismatch {
            node_name: node_name.to_string(),
            op_type,
            expected,
            actual: inputs.len(),
        });
    }
    for (i, name) in inputs.iter().enumerate() {
        if i < min_inputs && name.is_empty() {
            return Err(ExportError::EmptyRequiredInput {
                node_name: node_name.to_string(),
                op_type,
                index: i,
            });
        }
    }
    Ok(())
}

fn build_node(
    node: &ExportNode,
    op_type: &'static str,
    attribute: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        input: node.inputs.clone(),
        output: node.outputs.clone(),
        name: node.name.clone(),
        op_type: op_type.to_string(),
        attribute,
        // 既定 opset（"" = ai.onnx）限定。`check_exportable` が受理する domain と
        // 揃える（本モジュールは既定 opset の 22 op のみ書き出す）。
        domain: String::new(),
    }
}

/// [`ExportNode`] を `NodeProto` へ組み立てる（`interp.rs` の各 `compute_*` の
/// 逆方向）。属性は既定値であっても常に書き出す（`Transpose::perm` の `None`
/// のみ例外。モジュール冒頭コメント参照）。arity 検査で先に失敗させ
/// （`security.md` A03・不正入力の早期拒否）、その後 `NodeProto` を構築する。
pub fn to_node_proto(node: &ExportNode) -> Result<NodeProto, ExportError> {
    let op_type = node.op.op_type();
    match &node.op {
        ExportOp::Gemm(attrs) => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 3)?;
            let attribute = vec![
                attr_float("alpha", attrs.alpha),
                attr_float("beta", attrs.beta),
                attr_int("transA", i64::from(attrs.trans_a)),
                attr_int("transB", i64::from(attrs.trans_b)),
            ];
            Ok(build_node(node, op_type, attribute))
        }
        ExportOp::MatMul | ExportOp::Add | ExportOp::Mul | ExportOp::Div => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 2)?;
            Ok(build_node(node, op_type, Vec::new()))
        }
        ExportOp::Mod { fmod } => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 2)?;
            Ok(build_node(
                node,
                op_type,
                vec![attr_int("fmod", i64::from(*fmod))],
            ))
        }
        ExportOp::Sqrt | ExportOp::Relu | ExportOp::Sigmoid | ExportOp::Erf => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 1, 1)?;
            Ok(build_node(node, op_type, Vec::new()))
        }
        ExportOp::Softmax { axis } => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 1, 1)?;
            Ok(build_node(node, op_type, vec![attr_int("axis", *axis)]))
        }
        ExportOp::Reshape { allowzero } => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 2)?;
            Ok(build_node(
                node,
                op_type,
                vec![attr_int("allowzero", i64::from(*allowzero))],
            ))
        }
        ExportOp::Shape => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 1, 1)?;
            Ok(build_node(node, op_type, Vec::new()))
        }
        ExportOp::Gather { axis } => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 2)?;
            Ok(build_node(node, op_type, vec![attr_int("axis", *axis)]))
        }
        ExportOp::Unsqueeze => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 2)?;
            Ok(build_node(node, op_type, Vec::new()))
        }
        ExportOp::Squeeze => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 1, 2)?;
            Ok(build_node(node, op_type, Vec::new()))
        }
        ExportOp::Concat { axis } => {
            check_arity(
                &node.name,
                op_type,
                &node.inputs,
                &node.outputs,
                1,
                usize::MAX,
            )?;
            Ok(build_node(node, op_type, vec![attr_int("axis", *axis)]))
        }
        ExportOp::Slice => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 5)?;
            Ok(build_node(node, op_type, Vec::new()))
        }
        ExportOp::Transpose { perm } => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 1, 1)?;
            let attribute = match perm {
                Some(p) => vec![attr_ints("perm", p)],
                None => Vec::new(),
            };
            Ok(build_node(node, op_type, attribute))
        }
        ExportOp::Cast { to } => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 1, 1)?;
            Ok(build_node(node, op_type, vec![attr_int("to", *to)]))
        }
        ExportOp::Constant(attr) => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 0, 0)?;
            let attribute = match attr {
                ConstantAttr::Tensor(raw) => {
                    let out_name = node.outputs[0].clone();
                    vec![attr_tensor("value", &out_name, raw)?]
                }
                ConstantAttr::Float(v) => vec![attr_float("value_float", *v)],
                ConstantAttr::Floats(vs) => vec![attr_floats("value_floats", vs)],
                ConstantAttr::Int(v) => vec![attr_int("value_int", *v)],
                ConstantAttr::Ints(vs) => vec![attr_ints("value_ints", vs)],
            };
            Ok(build_node(node, op_type, attribute))
        }
        ExportOp::LayerNormalization(attrs) => {
            check_arity(&node.name, op_type, &node.inputs, &node.outputs, 2, 3)?;
            Ok(build_node(
                node,
                op_type,
                vec![
                    attr_int("axis", attrs.axis),
                    attr_float("epsilon", attrs.epsilon),
                ],
            ))
        }
    }
}

/// `Graph`（`graph.nodes: Vec<NodeProto>`）が本モジュールの export 対応範囲に
/// 収まっているかを fail-closed に検査する（層 B）。`build_model_proto`
/// （`export.rs`）が組み立て前に必ず呼ぶ。
///
/// 検査は 2 点: (1) `op_type` が [`SUPPORTED_OP_TYPES`] に含まれる、(2) `domain`
/// が既定 opset（空文字列。ONNX の `ai.onnx` ドメイン慣習）である。いずれかに
/// 違反するノードがあれば最初に見つかったものを [`ExportError::UnsupportedOp`]
/// で報告し、それ以降の処理を行わない（無言 skip はしない。`security.md` A03）。
pub fn check_exportable(graph: &Graph) -> Result<(), ExportError> {
    for node in &graph.nodes {
        let domain_ok = node.domain.is_empty();
        let op_ok = SUPPORTED_OP_TYPES.contains(&node.op_type.as_str());
        if !domain_ok || !op_ok {
            return Err(ExportError::UnsupportedOp {
                node_name: node.name.clone(),
                op_type: node.op_type.clone(),
                domain: node.domain.clone(),
            });
        }
    }
    Ok(())
}
