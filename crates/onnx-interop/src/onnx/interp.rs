//! ONNX グラフ実行インタープリタ（TASK-7.2b・REQ-7・イシュー #78）。
//!
//! `onnx::graph::build_graph` が構築した [`Graph`]（トポロジカル順検証済み・SSA
//! 検証済み）を受け取り、各ノードを `op_type` 名で `ops::*`（TASK-7.2c・#79）へ
//! ディスパッチして実行する。`graph` モジュールが「グラフは既に妥当である」前提を
//! 保証しているため、本モジュールは値の解決（feed／initializer／先行ノード出力）と
//! 演算ディスパッチのみに専念できる（`onnx/graph.rs` 冒頭コメント参照）。
//!
//! PoC-v2-6 方式B（`docs/spec/03-poc/poc-v2-6-interop/code/rust/src/onnx/interp.rs`）の
//! productize。実装対象は TASK-7.2 の 8 オペ（`Gemm`／`Relu`／`Sigmoid`／`Shape`／
//! `Gather`／`Unsqueeze`／`Concat`／`Slice`）の結線のみであり、TASK-7.3 系 14 オペの
//! ディスパッチ結線はイシュー #274 で追跡する（未対応 `op_type` は
//! [`InterpError::UnsupportedOp`] で fail-closed に拒否し、無言 skip はしない）。
//!
//! ## 実行時値モデルと i64 の扱いについて
//!
//! `tensor_core::Element` は `i64` を実装しない（#274）ため、`Shape` の出力や
//! `Gather`／`Slice` のインデックス系入力は [`Value::I64`]（生の `Vec<i64>` +
//! `Vec<usize>` shape）として保持する。`Gather`／`Unsqueeze`／`Concat` は i64
//! データに対しても直接（f32 へブリッジせず）動作する専用実装を持つ。これは
//! PoC-v2-6 参照実装（`interp.rs`）が i64 経路を専用ロジックで扱っていたのと
//! 同じ方針であり、f32 ブリッジによる精度損失（大きな shape/index 値が f32 の
//! 24bit 仮数部を超えて丸まる問題）を避けるための意図的な設計判断である。
//!
//! ただし i64 経路は `tests/fixtures/slice_repro.onnx`（動的境界 Slice パターン。
//! `Shape -> Gather -> Unsqueeze -> Concat -> Slice`）が要求する 1 次元ベクトル
//! （形状問い合わせ結果・インデックス列）の範囲に限定する。`Unsqueeze`/`Concat` の
//! i64 経路は「常に 1 次元」という不変条件を前提に、`Unsqueeze` はデータを保持した
//! まま shape を要素数へ正規化し（`axes` の妥当性は `ops::normalize_axis` 経由で
//! 検証するが、多次元 rank の追跡はしない）、`Concat` は 1 次元入力のみを連結する。
//! 一般的な多次元 `Tensor<i64>` サポートは #274 のスコープ。
//!
//! ## エラー方針
//!
//! 本モジュールは ONNX という信頼できない外部フォーマット（OWASP A03）を実行する
//! ため、`unwrap()`/`expect()` を本番経路で使わない（`.claude/rules/coding-rust.md`）。
//! 値の欠落・型不一致・属性欠落・feed の過不足はすべて型付きエラー
//! （[`InterpError`]）で拒否し、無言 skip はしない（no-silent-skip 契約。
//! `onnx/graph.rs` と同じ方針）。

use std::collections::{HashMap, HashSet};
use std::fmt;

use tensor_core::{ShapeError, Tensor};

use super::graph::{Graph, RawTensor};
use super::proto::NodeProto;
use crate::ops::{self, GemmAttrs, OpError, SliceParams};

/// 実行時に env（変数束縛）へ格納される値。ONNX の `TensorProto.data_type` の
/// うち本クレートが対応する 2 種類（`FLOAT`／`INT64`）に対応する
/// （`onnx::proto::data_type`）。`I64` は `Vec<i64>` + shape の素表現（本モジュール
/// 冒頭コメント参照。`tensor_core::Element` が `i64` 非対応のため）。
#[derive(Clone, Debug)]
pub enum Value {
    F32(Tensor<f32>),
    I64 { data: Vec<i64>, shape: Vec<usize> },
}

/// インタープリタの実行時エラー。`#[non_exhaustive]`: `OpError`／`GraphError` と
/// 同じ理由（公開 API 非破壊。TASK-7.3 系オペ追加時の variant 追加に備える）。
#[non_exhaustive]
#[derive(Debug)]
pub enum InterpError {
    /// ディスパッチ表（本モジュールが実装する 8 オペ）に無い `op_type`。
    /// TASK-7.3 系 14 オペの結線は #274（未実装のまま到達すると本 variant で拒否）。
    UnsupportedOp(String),
    /// ノードの入力名が env（feed／initializer／先行ノード出力の集合）に存在しない。
    /// `build_graph` はトポロジカル順を検証済みのため通常は到達しないが、
    /// 位置指定の入力が省略（空文字列）されている場合等に発生しうる。
    MissingInput { node: String, input: String },
    /// env に値は存在するが期待する [`Value`] variant と異なる
    /// （例: `Gemm` の入力に `Value::I64` が渡された）。
    TypeMismatch {
        node: String,
        expected: &'static str,
    },
    /// 必須属性（例: `Concat` の `axis`）が `NodeProto.attribute` に存在しない。
    MissingAttribute { node: String, attr: String },
    /// `graph.inputs` のうち initializer を持たない入力に対応する feed が
    /// `run` の呼び出し元から渡されなかった（no-silent-skip 契約）。
    MissingFeed { input: String },
    /// `run` に渡された feed 名が `graph.inputs`／initializer 名のいずれにも
    /// 属さない（呼び出し元の取り違えを無言で吸収しない）。
    UnknownFeed { name: String },
    /// ノードの宣言出力数が 1 以外（TASK-7.2 の 8 オペはすべて単一出力。
    /// 実装計画 5.3 節参照）。
    OutputArityMismatch {
        node: String,
        expected: usize,
        actual: usize,
    },
    /// i64 直接実装（`Gather`／`Unsqueeze`／`Concat`）が前提とする「1 次元」
    /// 不変条件が崩れている（本モジュール冒頭コメント参照）。一般的な多次元
    /// i64 サポートは #274 のスコープ。
    I64ShapeUnsupported {
        node: String,
        op: &'static str,
        shape: Vec<usize>,
    },
    /// `graph.outputs` に列挙された名前が実行後の env に存在しない。
    /// `build_graph` が生成可能性を検証済みのため到達しないはずだが、
    /// 防御的に型付きエラーで報告する（`unwrap()` を避けるため。coding-rust.md）。
    GraphOutputNotProduced { name: String },
    /// `ops::*`（TASK-7.2c）が返したオペ固有エラーをそのまま透過する。
    Op(OpError),
    /// initializer の `RawTensor` を [`Value`] へ変換する際の shape 不整合
    /// （`tensor_core::Tensor::new` が返す）。`build_graph` が既に検証済みのため
    /// 通常到達しない防御的経路。
    Shape(ShapeError),
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::UnsupportedOp(op) => write!(f, "未対応の op_type: {op}"),
            InterpError::MissingInput { node, input } => {
                write!(f, "ノード '{node}' の入力 '{input}' が見つかりません")
            }
            InterpError::TypeMismatch { node, expected } => {
                write!(f, "ノード '{node}': 型不一致（期待: {expected}）")
            }
            InterpError::MissingAttribute { node, attr } => {
                write!(f, "ノード '{node}': 必須属性 '{attr}' がありません")
            }
            InterpError::MissingFeed { input } => {
                write!(f, "グラフ入力 '{input}' に対応する feed がありません")
            }
            InterpError::UnknownFeed { name } => {
                write!(
                    f,
                    "feed '{name}' はグラフ入力にも initializer にも属しません"
                )
            }
            InterpError::OutputArityMismatch {
                node,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "ノード '{node}': 出力数不一致（期待 {expected}、実際 {actual}）"
                )
            }
            InterpError::I64ShapeUnsupported { node, op, shape } => {
                write!(
                    f,
                    "ノード '{node}' ({op}): i64 直接実装は 1 次元のみ対応（実際の shape={shape:?}）"
                )
            }
            InterpError::GraphOutputNotProduced { name } => {
                write!(f, "グラフ出力 '{name}' が実行結果に存在しません")
            }
            InterpError::Op(e) => write!(f, "{e}"),
            InterpError::Shape(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for InterpError {}

impl From<OpError> for InterpError {
    fn from(e: OpError) -> Self {
        InterpError::Op(e)
    }
}

impl From<ShapeError> for InterpError {
    fn from(e: ShapeError) -> Self {
        InterpError::Shape(e)
    }
}

/// `RawTensor`（`graph::build_graph` が復号した initializer）を実行時値へ変換する。
/// `build_graph` が dims の非負性・要素数整合を検証済みのため通常は失敗しないが、
/// `Tensor::new` の結果を `unwrap()` せず型付きエラーで伝播する（coding-rust.md）。
fn raw_to_value(raw: &RawTensor) -> Result<Value, InterpError> {
    match raw {
        RawTensor::F32 { data, shape } => {
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let t = Tensor::new(data.clone(), &shape_usize)?;
            Ok(Value::F32(t))
        }
        RawTensor::I64 { data, shape } => {
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            Ok(Value::I64 {
                data: data.clone(),
                shape: shape_usize,
            })
        }
    }
}

/// `node.input[idx]` を取得する。ONNX は省略可能入力を空文字列で表す規約
/// （`onnx/graph.rs` と同じ規約）のため、範囲外・空文字列のいずれも
/// [`InterpError::MissingInput`] として扱う。
fn input_name(node: &NodeProto, idx: usize) -> Result<&str, InterpError> {
    node.input
        .get(idx)
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| InterpError::MissingInput {
            node: node.name.clone(),
            input: format!("<input[{idx}]>"),
        })
}

fn get_value<'a>(
    env: &'a HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a Value, InterpError> {
    env.get(name).ok_or_else(|| InterpError::MissingInput {
        node: node.name.clone(),
        input: name.to_string(),
    })
}

fn get_f32<'a>(
    env: &'a HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a Tensor<f32>, InterpError> {
    match get_value(env, node, name)? {
        Value::F32(t) => Ok(t),
        Value::I64 { .. } => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f32",
        }),
    }
}

fn get_i64<'a>(
    env: &'a HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<(&'a [i64], &'a [usize]), InterpError> {
    match get_value(env, node, name)? {
        Value::I64 { data, shape } => Ok((data.as_slice(), shape.as_slice())),
        Value::F32(_) => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "i64",
        }),
    }
}

fn attr_f32(node: &NodeProto, name: &str, default: f32) -> f32 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.f)
        .unwrap_or(default)
}

fn attr_i64(node: &NodeProto, name: &str, default: i64) -> i64 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

fn attr_i64_required(node: &NodeProto, name: &str) -> Result<i64, InterpError> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .ok_or_else(|| InterpError::MissingAttribute {
            node: node.name.clone(),
            attr: name.to_string(),
        })
}

/// `Unsqueeze` opset<13 の `axes` 属性（`AttributeProto.ints`）を読む。
fn attr_i64s<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

/// `axes`（`Unsqueeze` 等）の妥当性を `ops::normalize_axis` 経由で検証する
/// （範囲外・重複軸の拒否）。`ops::unsqueeze` 内部の検証ロジックと同じ規則を
/// i64 直接経路（多次元 shape を追跡しない）にも適用し、両経路で同じ入力に
/// 対して同じ合否判定になるようにする。
fn validate_axes_for_rank(
    op: &'static str,
    in_rank: usize,
    axes: &[i64],
) -> Result<(), InterpError> {
    let out_rank = in_rank + axes.len();
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let n = ops::normalize_axis(axis, out_rank).ok_or(OpError::AxisOutOfRange {
            op,
            axis,
            rank: out_rank,
        })?;
        normalized.push(n);
    }
    normalized.sort_unstable();
    for pair in normalized.windows(2) {
        if pair[0] == pair[1] {
            return Err(OpError::DuplicateAxis { op, axis: pair[0] }.into());
        }
    }
    Ok(())
}

fn compute_gemm(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let a = get_f32(env, node, input_name(node, 0)?)?;
    let b = get_f32(env, node, input_name(node, 1)?)?;
    let c = match node.input.get(2) {
        Some(name) if !name.is_empty() => Some(get_f32(env, node, name)?),
        _ => None,
    };
    let attrs = GemmAttrs {
        alpha: attr_f32(node, "alpha", 1.0),
        beta: attr_f32(node, "beta", 1.0),
        trans_a: attr_i64(node, "transA", 0) != 0,
        trans_b: attr_i64(node, "transB", 0) != 0,
    };
    Ok(Value::F32(ops::gemm(a, b, c, &attrs)?))
}

fn compute_relu(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    Ok(Value::F32(ops::relu(x)?))
}

fn compute_sigmoid(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    Ok(Value::F32(ops::sigmoid(x)?))
}

fn compute_shape(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let name = input_name(node, 0)?;
    let dims: Vec<i64> = match get_value(env, node, name)? {
        Value::F32(t) => ops::shape(t),
        Value::I64 { shape, .. } => shape.iter().map(|&d| d as i64).collect(),
    };
    let len = dims.len();
    Ok(Value::I64 {
        data: dims,
        shape: vec![len],
    })
}

fn compute_gather(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    let idx_name = input_name(node, 1)?;
    let axis = attr_i64(node, "axis", 0);
    match get_value(env, node, data_name)? {
        Value::F32(data_t) => {
            let (idx, idx_shape) = get_i64(env, node, idx_name)?;
            Ok(Value::F32(ops::gather(data_t, idx, idx_shape, axis)?))
        }
        Value::I64 { data, shape } => {
            if shape.len() != 1 {
                return Err(InterpError::I64ShapeUnsupported {
                    node: node.name.clone(),
                    op: "Gather",
                    shape: shape.clone(),
                });
            }
            // rank 1 前提での axis 妥当性検査（0／-1 のみ許容）。
            ops::normalize_axis(axis, 1).ok_or(OpError::AxisOutOfRange {
                op: "Gather",
                axis,
                rank: 1,
            })?;
            let (idx, idx_shape) = get_i64(env, node, idx_name)?;
            // f32 パス（`ops::gather`）と同様、`idx_shape` の積が実データ長と
            // 一致するかを検査してから使う。ここを省略すると `idx.len()` から
            // 構築した `data` と、素通しした `idx_shape` が矛盾した `Value::I64`
            // （不変条件違反）を生成しうる（レビュー指摘: Cursor Bugbot、PR #298）。
            let expected_len: usize = idx_shape.iter().product();
            if idx.len() != expected_len {
                return Err(OpError::LengthMismatch {
                    op: "Gather",
                    name: "indices",
                    expected: expected_len,
                    actual: idx.len(),
                }
                .into());
            }
            let dim = data.len() as i64;
            let mut out = Vec::with_capacity(idx.len());
            for &raw in idx {
                let n = if raw < 0 { raw + dim } else { raw };
                if n < 0 || n as usize >= data.len() {
                    return Err(OpError::IndexOutOfRange {
                        op: "Gather",
                        index: raw,
                        dim_size: data.len(),
                    }
                    .into());
                }
                out.push(data[n as usize]);
            }
            Ok(Value::I64 {
                data: out,
                shape: idx_shape.to_vec(),
            })
        }
    }
}

fn compute_unsqueeze(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    // opset>=13: axes は第 2 入力（テンソル）。opset<13: axes は属性（ints）。
    let axes: Vec<i64> = match node.input.get(1) {
        Some(name) if !name.is_empty() => get_i64(env, node, name)?.0.to_vec(),
        _ => attr_i64s(node, "axes")
            .map(<[i64]>::to_vec)
            .ok_or_else(|| InterpError::MissingAttribute {
                node: node.name.clone(),
                attr: "axes".to_string(),
            })?,
    };

    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::unsqueeze(t, &axes)?)),
        Value::I64 { data, shape } => {
            if shape.len() != 1 {
                return Err(InterpError::I64ShapeUnsupported {
                    node: node.name.clone(),
                    op: "Unsqueeze",
                    shape: shape.clone(),
                });
            }
            validate_axes_for_rank("Unsqueeze", shape.len(), &axes)?;
            // i64 直接経路は `Value::I64.shape` を「真の ONNX 論理 shape」ではなく
            // 要素数のみを保持する平坦なブックキーピングとして扱う（本モジュール
            // 冒頭コメント参照）。axes 挿入後の出力 rank が 1 を超える場合、その
            // まま `shape: vec![data.len()]` へ丸めると本来の多次元 shape（例:
            // [3] + axes=[0] -> 本来 [1,3]）を無言で破棄してしまうため、
            // no-silent-skip 契約に従い fail-closed で拒否する（レビュー指摘）。
            //
            // ただし `data.len() == 1` の場合のみ例外的に許容する: `tests/fixtures/
            // slice_repro.onnx` の実 Gather ノード（`onnx_decode.rs` の
            // `slice_repro_onnx_decodes_expected_graph_structure` が固定化する
            // `const_gather_idx` は shape=[1]・data=[0]。真のスカラー index
            // （dims=[]）ではなく 1 要素配列 index のため、Gather 後も rank 1 の
            // まま Unsqueeze へ渡る）でも、この後段 `Concat` の i64 分岐は
            // shape.len()==1 の入力のみ受理する（`compute_concat` 参照）。1 要素
            // データは rank の解釈が結果に影響しない（並び替えの余地がない）ため、
            // ここで丸めても no-silent-skip 契約の実害（多次元情報の消失による
            // 誤ったデータ順序・要素数の混同）は生じない。#274 で真の多次元 i64
            // shape 追跡を導入する際に本例外の要否を再検討する。
            let out_rank = shape.len() + axes.len();
            if out_rank != 1 && data.len() != 1 {
                return Err(InterpError::I64ShapeUnsupported {
                    node: node.name.clone(),
                    op: "Unsqueeze",
                    shape: shape.clone(),
                });
            }
            Ok(Value::I64 {
                data: data.clone(),
                shape: vec![data.len()],
            })
        }
    }
}

fn compute_concat(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    if node.input.is_empty() {
        return Err(OpError::EmptyInputs { op: "Concat" }.into());
    }
    let axis = attr_i64_required(node, "axis")?;
    match get_value(env, node, &node.input[0])? {
        Value::F32(_) => {
            let mut tensors = Vec::with_capacity(node.input.len());
            for name in &node.input {
                tensors.push(get_f32(env, node, name)?);
            }
            Ok(Value::F32(ops::concat(&tensors, axis)?))
        }
        Value::I64 { .. } => {
            let mut combined = Vec::new();
            for name in &node.input {
                let (data, shape) = get_i64(env, node, name)?;
                if shape.len() != 1 {
                    return Err(InterpError::I64ShapeUnsupported {
                        node: node.name.clone(),
                        op: "Concat",
                        shape: shape.to_vec(),
                    });
                }
                combined.extend_from_slice(data);
            }
            ops::normalize_axis(axis, 1).ok_or(OpError::AxisOutOfRange {
                op: "Concat",
                axis,
                rank: 1,
            })?;
            let len = combined.len();
            Ok(Value::I64 {
                data: combined,
                shape: vec![len],
            })
        }
    }
}

fn compute_slice(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data = get_f32(env, node, input_name(node, 0)?)?;
    let (starts, _) = get_i64(env, node, input_name(node, 1)?)?;
    let (ends, _) = get_i64(env, node, input_name(node, 2)?)?;
    let axes = match node.input.get(3) {
        Some(name) if !name.is_empty() => Some(get_i64(env, node, name)?.0),
        _ => None,
    };
    let steps = match node.input.get(4) {
        Some(name) if !name.is_empty() => Some(get_i64(env, node, name)?.0),
        _ => None,
    };
    let params = SliceParams {
        starts,
        ends,
        axes,
        steps,
    };
    Ok(Value::F32(ops::slice(data, &params)?))
}

/// `node.output` が単一出力であることを検査し、その名前を返す（TASK-7.2 の
/// 8 オペはすべて単一出力。実装計画 5.3 節）。
fn require_single_output(node: &NodeProto) -> Result<&str, InterpError> {
    if node.output.len() != 1 {
        return Err(InterpError::OutputArityMismatch {
            node: node.name.clone(),
            expected: 1,
            actual: node.output.len(),
        });
    }
    Ok(node.output[0].as_str())
}

/// `Graph`（`build_graph` が構築したトポロジカル順検証済みグラフ）を実行する。
///
/// 呼び出し元は `onnx-interop` 利用者（将来の `onnx_interop::run_model` 等の
/// 上位 API・TASK-7.4）。`feeds` は `graph.inputs` のうち initializer を持たない
/// 入力に対応する実行時の値（feeds 検証は以下の順序で行う。no-silent-skip 契約）:
///
/// 1. `graph.inputs` のうち initializer を持たない入力に feed が無ければ
///    [`InterpError::MissingFeed`]
/// 2. `graph.inputs`（および initializer 名）に属さない feed 名は
///    [`InterpError::UnknownFeed`] で拒否
///
/// initializer と同名の feed が渡された場合（pre-IR-4 パターン）は feed が
/// initializer を上書きする（ONNX 仕様のデフォルト値セマンティクス）。
pub fn run(
    graph: &Graph,
    feeds: HashMap<String, Value>,
) -> Result<HashMap<String, Value>, InterpError> {
    for input in &graph.inputs {
        if !graph.initializers.contains_key(input) && !feeds.contains_key(input) {
            return Err(InterpError::MissingFeed {
                input: input.clone(),
            });
        }
    }
    let input_set: HashSet<&str> = graph.inputs.iter().map(String::as_str).collect();
    for name in feeds.keys() {
        if !input_set.contains(name.as_str()) && !graph.initializers.contains_key(name) {
            return Err(InterpError::UnknownFeed { name: name.clone() });
        }
    }

    let mut env: HashMap<String, Value> =
        HashMap::with_capacity(graph.initializers.len() + feeds.len());
    for (k, v) in &graph.initializers {
        env.insert(k.clone(), raw_to_value(v)?);
    }
    for (k, v) in feeds {
        // feed が initializer を上書きする（ONNX のデフォルト値セマンティクス。
        // 本関数冒頭ドキュメント参照）。
        env.insert(k, v);
    }

    for node in &graph.nodes {
        let out_value = match node.op_type.as_str() {
            "Gemm" => compute_gemm(&env, node)?,
            "Relu" => compute_relu(&env, node)?,
            "Sigmoid" => compute_sigmoid(&env, node)?,
            "Shape" => compute_shape(&env, node)?,
            "Gather" => compute_gather(&env, node)?,
            "Unsqueeze" => compute_unsqueeze(&env, node)?,
            "Concat" => compute_concat(&env, node)?,
            "Slice" => compute_slice(&env, node)?,
            other => return Err(InterpError::UnsupportedOp(other.to_string())),
        };
        let output_name = require_single_output(node)?.to_string();
        env.insert(output_name, out_value);
    }

    let mut result = HashMap::with_capacity(graph.outputs.len());
    for name in &graph.outputs {
        let v = env
            .get(name)
            .cloned()
            .ok_or_else(|| InterpError::GraphOutputNotProduced { name: name.clone() })?;
        result.insert(name.clone(), v);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

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

    fn empty_graph(nodes: Vec<NodeProto>, inputs: Vec<&str>, outputs: Vec<&str>) -> Graph {
        Graph {
            nodes,
            initializers: StdHashMap::new(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
        }
    }

    #[test]
    fn raw_to_value_converts_f32_and_i64() {
        let f = raw_to_value(&RawTensor::F32 {
            data: vec![1.0, 2.0],
            shape: vec![2],
        })
        .unwrap();
        assert!(matches!(f, Value::F32(_)));

        let i = raw_to_value(&RawTensor::I64 {
            data: vec![1, 2, 3],
            shape: vec![3],
        })
        .unwrap();
        match i {
            Value::I64 { data, shape } => {
                assert_eq!(data, vec![1, 2, 3]);
                assert_eq!(shape, vec![3]);
            }
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn attr_helpers_fall_back_to_default_when_absent() {
        let n = node("Gemm", vec!["a", "b"], vec!["y"]);
        assert_eq!(attr_f32(&n, "alpha", 1.0), 1.0);
        assert_eq!(attr_i64(&n, "transB", 0), 0);
        assert!(attr_i64_required(&n, "axis").is_err());
        assert_eq!(attr_i64s(&n, "axes"), None);
    }

    #[test]
    fn run_rejects_missing_feed() {
        let n = node("Relu", vec!["x"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let err = run(&g, StdHashMap::new()).unwrap_err();
        assert!(matches!(err, InterpError::MissingFeed { input } if input == "x"));
    }

    #[test]
    fn run_rejects_unknown_feed() {
        let n = node("Relu", vec!["x"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
        );
        feeds.insert(
            "bogus".to_string(),
            Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(err, InterpError::UnknownFeed { name } if name == "bogus"));
    }

    #[test]
    fn run_rejects_unsupported_op() {
        let n = node("Dropout", vec!["x"], vec!["y", "mask"]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(err, InterpError::UnsupportedOp(op) if op == "Dropout"));
    }

    #[test]
    fn compute_gather_i64_rejects_indices_shape_length_mismatch() {
        // レビュー指摘（Cursor Bugbot、PR #298）: i64 Gather 分岐が `idx.len()`
        // から出力を構築する一方 `idx_shape` を素通ししていたため、両者が矛盾する
        // `Value::I64` を無検査で生成できてしまっていた。f32 パス（`ops::gather`）
        // 同様に `OpError::LengthMismatch` で拒否されることを固定化する。
        let n = node("Gather", vec!["data", "idx"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["data", "idx"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "data".to_string(),
            Value::I64 {
                data: vec![10, 20, 30],
                shape: vec![3],
            },
        );
        // idx_shape の積は 3 だが、実データ長は 2（矛盾）。
        feeds.insert(
            "idx".to_string(),
            Value::I64 {
                data: vec![0, 1],
                shape: vec![3],
            },
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(
            err,
            InterpError::Op(OpError::LengthMismatch {
                op: "Gather",
                name: "indices",
                expected: 3,
                actual: 2,
            })
        ));
    }

    #[test]
    fn run_end_to_end_relu() {
        let n = node("Relu", vec!["x"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(vec![-1.0, 2.0], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::F32(t) => {
                assert_eq!(t.get(&[0]).unwrap(), 0.0);
                assert_eq!(t.get(&[1]).unwrap(), 2.0);
            }
            _ => panic!("Value::F32 を期待"),
        }
    }
}
