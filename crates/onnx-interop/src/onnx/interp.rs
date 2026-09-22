//! ONNX グラフ実行インタープリタ（TASK-7.2b・REQ-7・イシュー #78。
//! TASK-7.3 系 14 オペの結線・`Value::I64` の型安全化はイシュー #274）。
//!
//! `onnx::graph::build_graph` が構築した [`Graph`]（トポロジカル順検証済み・SSA
//! 検証済み）を受け取り、各ノードを `op_type` 名で `ops::*`（TASK-7.2c・#79）へ
//! ディスパッチして実行する。`graph` モジュールが「グラフは既に妥当である」前提を
//! 保証しているため、本モジュールは値の解決（feed／initializer／先行ノード出力）と
//! 演算ディスパッチのみに専念できる（`onnx/graph.rs` 冒頭コメント参照）。
//!
//! PoC-v2-6 方式B（`docs/spec/03-poc/poc-v2-6-interop/code/rust/src/onnx/interp.rs`）の
//! productize。TASK-7.2 の 8 オペ（`Gemm`／`Relu`／`Sigmoid`／`Shape`／`Gather`／
//! `Unsqueeze`／`Concat`／`Slice`）に加え、TASK-7.3 系 14 オペ（`Add`／`Mul`／`Div`／
//! `Mod`／`Sqrt`／`Constant`／`Cast`／`Reshape`／`Squeeze`／`Transpose`／`MatMul`／
//! `Softmax`／`Erf`／`LayerNormalization`）をイシュー #274 で結線した。イシュー
//! #2076（親 #2034）で `Conv`（2 次元畳み込み）を追加し、全 23 オペがグラフ実行
//! から到達可能（未対応 `op_type` は引き続き [`InterpError::UnsupportedOp`]
//! で fail-closed に拒否し、無言 skip はしない）。
//!
//! ## 実行時値モデルと dtype の扱いについて
//!
//! `fandhe_ai_tensor_core::Element` にイシュー #274 で `i64`／`bool` を追加した（`element.rs`）
//! ことに伴い、[`Value::I64`]／[`Value::Bool`] は生の `Vec` + shape 表現ではなく
//! `Tensor<i64>`／`Tensor<bool>` を直接保持する型安全な表現へ置き換えた。
//! `Gather`／`Unsqueeze`／`Concat`／`Slice`（`ops::*` が `T: Element` でジェネリック化
//! 済み。`ops/mod.rs`）はいずれの dtype に対しても同じ実装で動作するため、以前の
//! 「i64 直接経路は 1 次元のみ」という不変条件（`I64ShapeUnsupported`）は撤廃した。
//! `Value::F16` は `half::f16`（`Cast(to=FLOAT16)`・BOOL/FLOAT16 initializer decode。
//! `onnx/graph.rs::decode_tensor`）向けに追加した。
//!
//! 算術・活性化・正規化系オペのうち `Add`／`Mul`／`Div`／`Mod` は `f32`／`i64` の
//! 両経路に対応する（`ops::*_i64`。イシュー #87 残作業〈TASK-7.4a〉。
//! `transformer.onnx` の `Shape -> Gather -> Div` 等 head_dim 算出が `i64` 同士の
//! 算術を要求するため）。`MatMul`／`Softmax`／`Erf`／`LayerNormalization`・
//! `Sqrt`（ONNX 仕様上 float 型のみ対応）は `ops::*` 側が `Tensor<f32>` 専用のまま
//! のため、非 F32 入力は [`InterpError::TypeMismatch`] で拒否する。いずれも
//! `f32`/`i64` 間の暗黙変換はしない（`Cast` を明示的に経由させる。混在ペアも
//! 同様に拒否する。`compute_add`／`compute_mul`／`compute_div`／`compute_mod`
//! 参照）。
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

use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};
use half::f16;

use super::graph::{Graph, GraphError, RawTensor};
use super::interp_device;
use super::proto::{AttributeProto, NodeProto, attribute_type};
use crate::ops::{self, ConstantValue, ConvAttrs, GemmAttrs, LayerNormAttrs, OpError, SliceParams};

/// 実行時に env（変数束縛）へ格納される値。ONNX の `TensorProto.data_type` の
/// うち本クレートが対応する 4 種類（`FLOAT`／`INT64`／`BOOL`／`FLOAT16`）に対応する
/// （`onnx::proto::data_type`）。いずれも `fandhe_ai_tensor_core::Element` を実装する型を
/// 直接保持する型安全な表現（本モジュール冒頭コメント参照。イシュー #274）。
#[derive(Clone, Debug)]
pub enum Value {
    F32(Tensor<f32>),
    I64(Tensor<i64>),
    Bool(Tensor<bool>),
    F16(Tensor<f16>),
}

/// インタープリタの実行時エラー。`#[non_exhaustive]`: `OpError`／`GraphError` と
/// 同じ理由（公開 API 非破壊。後続オペ追加時の variant 追加に備える）。
#[non_exhaustive]
#[derive(Debug)]
pub enum InterpError {
    /// ディスパッチ表（本モジュールが実装する全 22 オペ）に無い `op_type`。
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
    /// ノードの宣言出力数が 1 以外（本モジュールが実装する全オペは単一出力。
    /// `LayerNormalization` の任意出力 `Mean`／`InvStdDev` 宣言もここで拒否する。
    /// 実装計画 5.3 節・#274 実装計画スコープ外節参照）。
    OutputArityMismatch {
        node: String,
        expected: usize,
        actual: usize,
    },
    /// `graph.outputs` に列挙された名前が実行後の env に存在しない。
    /// `build_graph` が生成可能性を検証済みのため到達しないはずだが、
    /// 防御的に型付きエラーで報告する（`unwrap()` を避けるため。coding-rust.md）。
    GraphOutputNotProduced { name: String },
    /// `ops::*`（TASK-7.2c）が返したオペ固有エラーをそのまま透過する。
    Op(OpError),
    /// initializer の `RawTensor` を [`Value`] へ変換する際の shape 不整合
    /// （`fandhe_ai_tensor_core::Tensor::new` が返す）。`build_graph` が既に検証済みのため
    /// 通常到達しない防御的経路。
    Shape(ShapeError),
    /// `Constant` の `value`（TENSOR 型）属性が保持する `TensorProto` の復号エラー。
    /// `onnx::graph::decode_tensor` をそのまま透過する。
    Graph(GraphError),
    /// STRING 属性（`Conv` の `auto_pad`。イシュー #2076）が UTF-8 として不正
    /// だった。`AttributeProto.s: Vec<u8>` は任意バイト列を許容するため、
    /// 文字列属性を読む側（`attr_string`。本モジュール内 private 関数のため
    /// intra-doc link ではなくコードスパンで参照する）で明示的に検証する
    /// （OWASP A03。`.claude/rules/security.md`）。
    InvalidAttribute {
        node: String,
        attr: String,
        reason: String,
    },
    /// `run_with_ops`（イシュー #2077）の device 実行経路で発生した、
    /// `BackendError::Unsupported`／`BackendError::ShapeMismatch` 以外の
    /// 実行時エラー（driver 不在・カーネル起動失敗・デバイスメモリ確保
    /// 失敗等）。`interp_device::try_device` が発生源。`Unsupported`／
    /// `ShapeMismatch` はホスト実装へフォールバックするため本 variant に
    /// はならない（GPU 故障を隠蔽しない fail-closed 方針。OWASP A08）。
    Backend { node: String, message: String },
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
            InterpError::GraphOutputNotProduced { name } => {
                write!(f, "グラフ出力 '{name}' が実行結果に存在しません")
            }
            InterpError::Op(e) => write!(f, "{e}"),
            InterpError::Shape(e) => write!(f, "{e}"),
            InterpError::Graph(e) => write!(f, "{e}"),
            InterpError::InvalidAttribute { node, attr, reason } => {
                write!(f, "ノード '{node}': 属性 '{attr}' が不正です（{reason}）")
            }
            InterpError::Backend { node, message } => {
                write!(f, "ノード '{node}': device 実行エラー: {message}")
            }
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

impl From<GraphError> for InterpError {
    fn from(e: GraphError) -> Self {
        InterpError::Graph(e)
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
            let t = Tensor::new(data.clone(), &shape_usize)?;
            Ok(Value::I64(t))
        }
        RawTensor::Bool { data, shape } => {
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let t = Tensor::new(data.clone(), &shape_usize)?;
            Ok(Value::Bool(t))
        }
        RawTensor::F16 { data, shape } => {
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let t = Tensor::new(data.clone(), &shape_usize)?;
            Ok(Value::F16(t))
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
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f32",
        }),
    }
}

fn get_i64<'a>(
    env: &'a HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a Tensor<i64>, InterpError> {
    match get_value(env, node, name)? {
        Value::I64(t) => Ok(t),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "i64",
        }),
    }
}

/// `name` が指す i64 テンソルを、非 contiguous な view でも安全に読めるよう
/// 実体化して所有権付きの `(データ, shape)` として返す（`Gather` のインデックス・
/// `Slice` の `starts`/`ends`/`axes`/`steps`・`Reshape` の `shape` 等、`ops::*` の
/// スライス引数へそのまま渡せる形。値は shape 問い合わせ結果・インデックス列等の
/// 小規模データのため、参照を返す代わりに複製するコストは無視できる）。
fn i64_vec_and_shape(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<(Vec<i64>, Vec<usize>), InterpError> {
    let t = get_i64(env, node, name)?;
    let tc = t.contiguous();
    let data = tc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("(i64 input)"))?;
    Ok((data.to_vec(), tc.shape().to_vec()))
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

/// `Unsqueeze`／`Squeeze`（opset<13）・`Transpose` の `axes`/`perm` 属性
/// （`AttributeProto.ints`）を読む。
fn attr_i64s<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

/// `name` に一致する属性をちょうど 1 件だけ取得する（`Conv` 属性の型検証系
/// 関数が共用する。イシュー #2076 codex-review 指摘）。同名属性が複数宣言され
/// ている場合、どちらを採用すべきかは ONNX 仕様上一意に決まらないため、無言で
/// 先頭／末尾いずれかを採用せず [`InterpError::InvalidAttribute`] で fail-closed
/// に拒否する（OWASP A03。外部フォーマット由来の曖昧な入力を検証せずに通さない）。
fn find_attr_unique<'a>(
    node: &'a NodeProto,
    name: &str,
) -> Result<Option<&'a AttributeProto>, InterpError> {
    let mut matches = node.attribute.iter().filter(|a| a.name == name);
    let first = matches.next();
    if matches.next().is_some() {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: name.to_string(),
            reason: "同名属性が複数宣言されています".to_string(),
        });
    }
    Ok(first)
}

/// `attr.r#type`（`AttributeProto.AttributeType`。`onnx/proto.rs::attribute_type`）
/// が `expected` と一致することを検証する（`Conv` 属性の型検証系関数が共用。
/// イシュー #2076 codex-review 指摘）。
///
/// `AttributeProto` は `f`／`i`／`s`／`ints` 等のフィールドを protobuf の
/// oneof 相当として持つが、prost 生成コードでは実体は常に全フィールドを
/// 持つ構造体であり、未設定フィールドは型のゼロ値（`0`／`""`／`Vec::new()`）
/// になる。このため `r#type` を確認せず特定フィールドだけを読むと、本来
/// 別の型であるべき属性（例: 本来 INTS であるべき `kernel_shape` を STRING
/// 型として送り `ints` を空のまま残す）が「省略された」場合と区別できず、
/// 無言で ONNX 既定値へ fallback してしまう（OWASP A03。外部 ONNX モデルは
/// 非信頼な入力）。
fn check_attr_type(
    node: &NodeProto,
    attr: &AttributeProto,
    expected: i32,
    expected_name: &str,
) -> Result<(), InterpError> {
    if attr.r#type != expected {
        return Err(InterpError::InvalidAttribute {
            node: node.name.clone(),
            attr: attr.name.clone(),
            reason: format!(
                "AttributeType が {expected_name} ({expected}) ではありません（実際の値: {actual}）",
                actual = attr.r#type
            ),
        });
    }
    Ok(())
}

/// `Conv` の `kernel_shape`／`strides`／`pads`／`dilations`（INTS 型属性）を読む。
/// 属性が省略されていれば `None` を返し（呼び出し側 `compute_conv` が ONNX
/// 既定値へ fallback する）、存在する場合は `r#type == INTS` であることを
/// [`check_attr_type`] で検証してから `ints` を返す。`attr_i64s`（他オペ用の
/// 無検証版）との違いは、型不一致を「省略」と同一視せず fail-closed に拒否する
/// 点（イシュー #2076 codex-review 指摘。P0: 型偽装した不正 ONNX モデルの
/// 属性が無言で別の畳み込み条件〈既定値〉へ変更される問題への対処）。
///
/// `r#type == INTS` が確認できても `ints` が空の場合は拒否する（本来の
/// INTS 属性が空リストで存在する ONNX モデルは想定されず、`export_ops.rs`
/// も非空の値のみ書き出す。空 `ints` を許すと `.unwrap_or(&[])` 経由で
/// 「省略」時と同じ既定値 fallback に合流してしまい、`r#type` 検証だけでは
/// 塞げない同型の無言 fallback 抜け道になる）。
fn attr_ints_typed<'a>(node: &'a NodeProto, name: &str) -> Result<Option<&'a [i64]>, InterpError> {
    match find_attr_unique(node, name)? {
        None => Ok(None),
        Some(a) => {
            check_attr_type(node, a, attribute_type::INTS, "INTS")?;
            if a.ints.is_empty() {
                return Err(InterpError::InvalidAttribute {
                    node: node.name.clone(),
                    attr: name.to_string(),
                    reason: "INTS 属性が存在しますが要素が空です".to_string(),
                });
            }
            Ok(Some(a.ints.as_slice()))
        }
    }
}

/// `Conv` の `group`（INT 型属性）を読む。属性が省略されていれば `default`
/// を返し、存在する場合は `r#type == INT` であることを検証してから `i` を
/// 返す（`attr_ints_typed` と同型の型検証。イシュー #2076 codex-review 指摘）。
fn attr_i64_typed(node: &NodeProto, name: &str, default: i64) -> Result<i64, InterpError> {
    match find_attr_unique(node, name)? {
        None => Ok(default),
        Some(a) => {
            check_attr_type(node, a, attribute_type::INT, "INT")?;
            Ok(a.i)
        }
    }
}

/// STRING 属性（`AttributeProto.s: Vec<u8>`。`Conv` の `auto_pad`。
/// イシュー #2076）を読む。属性が無ければ `default` をそのまま返す
/// （ONNX 仕様の既定値セマンティクス。`attr_f32`／`attr_i64` と同型）。
/// 存在する場合は `r#type == STRING` であることを検証してから UTF-8 として
/// 復号する（`check_attr_type` 参照。型が異なる場合に空の `s` を「省略」と
/// 同一視して既定値へ fallback しない）。バイト列が UTF-8 として不正な場合も
/// 無言でエラー化せず [`InterpError::InvalidAttribute`] を返す（OWASP A03。
/// 外部フォーマット由来のバイト列を検証せずに文字列化しない）。同名属性が
/// 複数宣言されている場合も [`find_attr_unique`] が同様に拒否する。
fn attr_string(node: &NodeProto, name: &str, default: &str) -> Result<String, InterpError> {
    match find_attr_unique(node, name)? {
        None => Ok(default.to_string()),
        Some(a) => {
            check_attr_type(node, a, attribute_type::STRING, "STRING")?;
            String::from_utf8(a.s.clone()).map_err(|e| InterpError::InvalidAttribute {
                node: node.name.clone(),
                attr: name.to_string(),
                reason: format!("UTF-8 として不正なバイト列 ({e})"),
            })
        }
    }
}

fn compute_gemm(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
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
    // opt-in ON（`dev_ops` が `Some`）のときのみ device 経路を試みる。
    // `Ok(None)`（未対応 shape・`Unsupported`）はホスト実装へそのまま
    // フォールバックし、opt-in OFF（`dev_ops` が `None`）時の経路・出力は
    // 一切変更しない（イシュー #2077 実装計画 §3.2）。
    if let Some(ops) = dev_ops
        && let Some(out) = interp_device::device_gemm(&node.name, ops, a, b, c, &attrs)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((Value::F32(ops::gemm(a, b, c, &attrs)?), false))
}

fn compute_relu(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    if let Some(ops) = dev_ops
        && let Some(out) = interp_device::device_relu(&node.name, ops, x)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((Value::F32(ops::relu(x)?), false))
}

fn compute_sigmoid(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    if let Some(ops) = dev_ops
        && let Some(out) = interp_device::device_sigmoid(&node.name, ops, x)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((Value::F32(ops::sigmoid(x)?), false))
}

fn compute_shape(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let name = input_name(node, 0)?;
    let dims: Vec<i64> = match get_value(env, node, name)? {
        Value::F32(t) => ops::shape(t),
        Value::I64(t) => ops::shape(t),
        Value::Bool(t) => ops::shape(t),
        Value::F16(t) => ops::shape(t),
    };
    let len = dims.len();
    Ok(Value::I64(Tensor::new(dims, &[len])?))
}

fn compute_gather(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    let idx_name = input_name(node, 1)?;
    let axis = attr_i64(node, "axis", 0);
    let (idx, idx_shape) = i64_vec_and_shape(env, node, idx_name)?;
    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::gather(t, &idx, &idx_shape, axis)?)),
        Value::I64(t) => Ok(Value::I64(ops::gather(t, &idx, &idx_shape, axis)?)),
        Value::Bool(t) => Ok(Value::Bool(ops::gather(t, &idx, &idx_shape, axis)?)),
        Value::F16(t) => Ok(Value::F16(ops::gather(t, &idx, &idx_shape, axis)?)),
    }
}

fn compute_unsqueeze(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    // opset>=13: axes は第 2 入力（テンソル）。opset<13: axes は属性（ints）。
    let axes: Vec<i64> = match node.input.get(1) {
        Some(name) if !name.is_empty() => i64_vec_and_shape(env, node, name)?.0,
        _ => attr_i64s(node, "axes")
            .map(<[i64]>::to_vec)
            .ok_or_else(|| InterpError::MissingAttribute {
                node: node.name.clone(),
                attr: "axes".to_string(),
            })?,
    };

    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::unsqueeze(t, &axes)?)),
        Value::I64(t) => unsqueeze_i64_with_scalar_gather_shim(t, &axes),
        Value::Bool(t) => Ok(Value::Bool(ops::unsqueeze(t, &axes)?)),
        Value::F16(t) => Ok(Value::F16(ops::unsqueeze(t, &axes)?)),
    }
}

/// `Value::I64` の `Unsqueeze` 専用の互換シム。
///
/// 真の ONNX `Unsqueeze` 意味論（`ops::unsqueeze`。イシュー #274 で一般の
/// `Tensor<i64>` に対しても適用可能にした）は、入力 rank 1・`axes=[0]` に対し
/// 出力 rank 2（`[1] -> [1,1]`）を返す。しかし `tests/fixtures/slice_repro.onnx`
/// （`Shape -> Gather -> Unsqueeze -> Concat -> Slice` パターン）は `Gather` の
/// インデックスが真のスカラー（`dims=[]`）ではなく 1 要素配列（`shape=[1]`）と
/// いう非正規な形でエクスポートされているため、`Gather` 出力も `shape=[1]`
/// のまま `Unsqueeze` に渡る。この入力に真の `Unsqueeze` 意味論を適用すると
/// 出力が `[1,1]`（rank 2）になり、後続 `Concat` が同じ経路で作る `shape=[1]`
/// の定数（`const_4`）と rank が食い違い `RankMismatch` で失敗する
/// （このモデルが「shape ベクトルを組み立てる」という設計意図を実現できなくなる）。
///
/// 旧実装（PR #298）はこの非正規パターンを吸収するため、入力データが 1 要素
/// （`numel() == 1`）の場合のみ rank を追跡せず `shape: [1]` のフラット表現を
/// 維持する互換シムを備えていた（`interp.rs` 冒頭コメントが「#274 で本例外の
/// 要否を再検討する」と明記していた箇所）。#274 で `ops::unsqueeze` を真の
/// 多次元対応にした後もこの特定パターンとの互換性は必要なため、シム自体は
/// 維持しつつ「`slice_repro.onnx` が実際に生成する形（入力 rank 1・
/// axes 1 個・1 要素）」のみに適用範囲を限定する（`numel() == 1` だけを条件に
/// すると rank ≥ 2 の shape=[1,1] 入力や `axes.len() >= 2` の呼び出しにも誤って
/// 発火し、真の意味論〈`out_rank = t.rank() + axes.len()`〉より小さい rank を
/// 無言で返してしまう。no-silent-skip 契約〈`graph.rs` 冒頭コメント〉に反するため
/// `t.rank() == 1 && axes.len() == 1` も併せて要求する）。この条件を外れる
/// 入力は `ops::unsqueeze` の真の意味論へフォールスルーする。
/// `run_unsqueeze_i64_supports_multi_dim_input_rank`／
/// `run_unsqueeze_i64_supports_output_rank_exceeding_one`〈`tests/onnx_interp.rs`〉
/// が対象パターン外の入力で真の意味論が使われることを固定化する。
fn unsqueeze_i64_with_scalar_gather_shim(
    t: &Tensor<i64>,
    axes: &[i64],
) -> Result<Value, InterpError> {
    if !(t.rank() == 1 && axes.len() == 1 && t.numel() == 1) {
        return Ok(Value::I64(ops::unsqueeze(t, axes)?));
    }
    // 出力 rank（`t.rank() + axes.len()`）に対する axes 妥当性のみ検査し
    // （範囲外・重複軸を拒否）、shape 自体は `[1]` のまま維持する。
    let out_rank = t.rank() + axes.len();
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let n = ops::normalize_axis(axis, out_rank).ok_or(OpError::AxisOutOfRange {
            op: "Unsqueeze",
            axis,
            rank: out_rank,
        })?;
        normalized.push(n);
    }
    normalized.sort_unstable();
    for pair in normalized.windows(2) {
        if pair[0] == pair[1] {
            return Err(OpError::DuplicateAxis {
                op: "Unsqueeze",
                axis: pair[0],
            }
            .into());
        }
    }
    let tc = t.contiguous();
    let data = tc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Unsqueeze"))?;
    Ok(Value::I64(Tensor::new(data.to_vec(), &[data.len()])?))
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
        Value::I64(_) => {
            let mut tensors = Vec::with_capacity(node.input.len());
            for name in &node.input {
                tensors.push(get_i64(env, node, name)?);
            }
            Ok(Value::I64(ops::concat(&tensors, axis)?))
        }
        Value::Bool(_) => {
            let mut tensors = Vec::with_capacity(node.input.len());
            for name in &node.input {
                tensors.push(get_bool(env, node, name)?);
            }
            Ok(Value::Bool(ops::concat(&tensors, axis)?))
        }
        Value::F16(_) => {
            let mut tensors = Vec::with_capacity(node.input.len());
            for name in &node.input {
                tensors.push(get_f16(env, node, name)?);
            }
            Ok(Value::F16(ops::concat(&tensors, axis)?))
        }
    }
}

fn compute_slice(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    let (starts, _) = i64_vec_and_shape(env, node, input_name(node, 1)?)?;
    let (ends, _) = i64_vec_and_shape(env, node, input_name(node, 2)?)?;
    let axes = match node.input.get(3) {
        Some(name) if !name.is_empty() => Some(i64_vec_and_shape(env, node, name)?.0),
        _ => None,
    };
    let steps = match node.input.get(4) {
        Some(name) if !name.is_empty() => Some(i64_vec_and_shape(env, node, name)?.0),
        _ => None,
    };
    let params = SliceParams {
        starts: &starts,
        ends: &ends,
        axes: axes.as_deref(),
        steps: steps.as_deref(),
    };
    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::slice(t, &params)?)),
        Value::I64(t) => Ok(Value::I64(ops::slice(t, &params)?)),
        Value::Bool(t) => Ok(Value::Bool(ops::slice(t, &params)?)),
        Value::F16(t) => Ok(Value::F16(ops::slice(t, &params)?)),
    }
}

fn get_bool<'a>(
    env: &'a HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a Tensor<bool>, InterpError> {
    match get_value(env, node, name)? {
        Value::Bool(t) => Ok(t),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "bool",
        }),
    }
}

fn get_f16<'a>(
    env: &'a HashMap<String, Value>,
    node: &NodeProto,
    name: &str,
) -> Result<&'a Tensor<f16>, InterpError> {
    match get_value(env, node, name)? {
        Value::F16(t) => Ok(t),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f16",
        }),
    }
}

// ---- TASK-7.3a: MVP 算術オペ（イシュー #274 結線。`i64` 対応はイシュー #87
// 残作業〈TASK-7.4a〉。`transformer.onnx` の `Shape -> Gather -> Div` 等
// head_dim 算出経路が `i64` 同士の算術を要求する。実測確認済み〈イシュー #87
// reopen コメント〉）----

/// `(F32, F32)` は既存の `f32` 経路、`(I64, I64)` は `i64` 経路（`ops::*_i64`）
/// へ分岐する。混在（`F32`/`I64` 相互）は ONNX 仕様上そもそも許容されない
/// 組み合わせであり、`Cast` を明示的に経由させる既存方針（`compute_cast` 冒頭
/// コメント）に合わせて暗黙変換せず [`InterpError::TypeMismatch`] で拒否する
/// （no-silent-skip 契約）。
fn compute_add(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let a = get_value(env, node, input_name(node, 0)?)?;
    let b = get_value(env, node, input_name(node, 1)?)?;
    match (a, b) {
        (Value::F32(a), Value::F32(b)) => {
            if let Some(ops) = dev_ops
                && let Some(out) = interp_device::device_add(&node.name, ops, a, b)?
            {
                return Ok((Value::F32(out), true));
            }
            Ok((Value::F32(ops::add(a, b)?), false))
        }
        (Value::I64(a), Value::I64(b)) => Ok((Value::I64(ops::add_i64(a, b)?), false)),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f32 or i64 (matching pair)",
        }),
    }
}

fn compute_mul(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let a = get_value(env, node, input_name(node, 0)?)?;
    let b = get_value(env, node, input_name(node, 1)?)?;
    match (a, b) {
        (Value::F32(a), Value::F32(b)) => {
            if let Some(ops) = dev_ops
                && let Some(out) = interp_device::device_mul(&node.name, ops, a, b)?
            {
                return Ok((Value::F32(out), true));
            }
            Ok((Value::F32(ops::mul(a, b)?), false))
        }
        (Value::I64(a), Value::I64(b)) => Ok((Value::I64(ops::mul_i64(a, b)?), false)),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f32 or i64 (matching pair)",
        }),
    }
}

fn compute_div(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let a = get_value(env, node, input_name(node, 0)?)?;
    let b = get_value(env, node, input_name(node, 1)?)?;
    match (a, b) {
        (Value::F32(a), Value::F32(b)) => {
            if let Some(ops) = dev_ops
                && let Some(out) = interp_device::device_div(&node.name, ops, a, b)?
            {
                return Ok((Value::F32(out), true));
            }
            Ok((Value::F32(ops::div(a, b)?), false))
        }
        (Value::I64(a), Value::I64(b)) => Ok((Value::I64(ops::div_i64(a, b)?), false)),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f32 or i64 (matching pair)",
        }),
    }
}

fn compute_mod(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let a = get_value(env, node, input_name(node, 0)?)?;
    let b = get_value(env, node, input_name(node, 1)?)?;
    let fmod = attr_i64(node, "fmod", 0) != 0;
    match (a, b) {
        (Value::F32(a), Value::F32(b)) => Ok(Value::F32(ops::modulo(a, b, fmod)?)),
        (Value::I64(a), Value::I64(b)) => Ok(Value::I64(ops::mod_i64(a, b, fmod)?)),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "f32 or i64 (matching pair)",
        }),
    }
}

fn compute_sqrt(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    if let Some(ops) = dev_ops
        && let Some(out) = interp_device::device_sqrt(&node.name, ops, x)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((Value::F32(ops::sqrt(x)?), false))
}

/// `Constant(value|value_float|value_floats|value_int|value_ints) -> y`。
/// ONNX Constant-13 仕様は排他的な属性群を定義する。`value`（`TENSOR`）は
/// `onnx::graph::decode_tensor` を再利用し `raw_to_value` で `Value` 化する
/// （dtype に応じ `F32`／`I64`／`Bool`／`F16` のいずれにもなりうる）。`value_int`／
/// `value_ints` は `Value::I64` を構築する（`ops::constant`〈f32 専用〉では扱えない
/// ため、この 2 属性のみ本関数内で直接処理する）。属性が一つも見つからない場合は
/// [`InterpError::MissingAttribute`]（`attr` は代表として `"value"` を報告）で
/// fail-closed に拒否する。
fn compute_constant(node: &NodeProto) -> Result<Value, InterpError> {
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
        let t = attr
            .t
            .as_ref()
            .ok_or_else(|| InterpError::MissingAttribute {
                node: node.name.clone(),
                attr: "value".to_string(),
            })?;
        let raw = super::graph::decode_tensor(t)?;
        return raw_to_value(&raw);
    }
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_float") {
        return Ok(Value::F32(ops::constant(&ConstantValue::Float(attr.f))?));
    }
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_floats") {
        return Ok(Value::F32(ops::constant(&ConstantValue::Floats(
            attr.floats.clone(),
        ))?));
    }
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_int") {
        return Ok(Value::I64(Tensor::new(vec![attr.i], &[])?));
    }
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "value_ints") {
        let len = attr.ints.len();
        return Ok(Value::I64(Tensor::new(attr.ints.clone(), &[len])?));
    }
    Err(InterpError::MissingAttribute {
        node: node.name.clone(),
        attr: "value".to_string(),
    })
}

// ---- TASK-7.3b: MVP 形状操作オペ（イシュー #274 結線）----

/// `Cast(x, to)`。`to`（ONNX `TensorProto.DataType`）を
/// [`ops::check_supported_cast_target`] で範囲検査してから、入力 [`Value`] variant と
/// `to` の組で `ops::cast_*` へ分岐する（型安全化。イシュー #274）。恒等 Cast
/// （同一 dtype 間）は変換なしでそのまま透過する。未対応の組（例: `Bool -> INT64`）は
/// [`InterpError::TypeMismatch`] で fail-closed に拒否する（f32 ブリッジ等の暗黙変換は
/// 行わない）。
fn compute_cast(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    const ONNX_DATA_TYPE_FLOAT: i64 = 1;
    const ONNX_DATA_TYPE_INT64: i64 = 7;
    const ONNX_DATA_TYPE_BOOL: i64 = 9;
    const ONNX_DATA_TYPE_FLOAT16: i64 = 10;

    let name = input_name(node, 0)?;
    let to = attr_i64_required(node, "to")?;
    ops::check_supported_cast_target(to)?;
    let value = get_value(env, node, name)?;
    match (value, to) {
        (Value::F32(t), ONNX_DATA_TYPE_FLOAT) => Ok(Value::F32(t.clone())),
        (Value::F32(t), ONNX_DATA_TYPE_INT64) => Ok(Value::I64(ops::cast_to_int64(t)?)),
        (Value::F32(t), ONNX_DATA_TYPE_BOOL) => Ok(Value::Bool(ops::cast_to_bool(t)?)),
        (Value::F32(t), ONNX_DATA_TYPE_FLOAT16) => Ok(Value::F16(ops::cast_to_f16(t)?)),
        (Value::I64(t), ONNX_DATA_TYPE_INT64) => Ok(Value::I64(t.clone())),
        (Value::I64(t), ONNX_DATA_TYPE_FLOAT) => Ok(Value::F32(ops::cast_to_float(t)?)),
        (Value::Bool(t), ONNX_DATA_TYPE_BOOL) => Ok(Value::Bool(t.clone())),
        (Value::Bool(t), ONNX_DATA_TYPE_FLOAT) => Ok(Value::F32(ops::cast_bool_to_float(t)?)),
        (Value::F16(t), ONNX_DATA_TYPE_FLOAT16) => Ok(Value::F16(t.clone())),
        (Value::F16(t), ONNX_DATA_TYPE_FLOAT) => Ok(Value::F32(ops::cast_f16_to_float(t)?)),
        _ => Err(InterpError::TypeMismatch {
            node: node.name.clone(),
            expected: "supported Cast source/target dtype combination",
        }),
    }
}

fn compute_reshape(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    let (shape, _) = i64_vec_and_shape(env, node, input_name(node, 1)?)?;
    let allowzero = attr_i64(node, "allowzero", 0) != 0;
    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::reshape(t, &shape, allowzero)?)),
        Value::I64(t) => Ok(Value::I64(ops::reshape(t, &shape, allowzero)?)),
        Value::Bool(t) => Ok(Value::Bool(ops::reshape(t, &shape, allowzero)?)),
        Value::F16(t) => Ok(Value::F16(ops::reshape(t, &shape, allowzero)?)),
    }
}

fn compute_squeeze(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    // opset>=13: axes は第 2 入力（テンソル、省略可）。opset<13: axes は属性（ints）。
    let axes: Option<Vec<i64>> = match node.input.get(1) {
        Some(name) if !name.is_empty() => Some(i64_vec_and_shape(env, node, name)?.0),
        _ => attr_i64s(node, "axes").map(<[i64]>::to_vec),
    };
    let axes_ref = axes.as_deref();
    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::squeeze(t, axes_ref)?)),
        Value::I64(t) => Ok(Value::I64(ops::squeeze(t, axes_ref)?)),
        Value::Bool(t) => Ok(Value::Bool(ops::squeeze(t, axes_ref)?)),
        Value::F16(t) => Ok(Value::F16(ops::squeeze(t, axes_ref)?)),
    }
}

fn compute_transpose(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let data_name = input_name(node, 0)?;
    let perm = attr_i64s(node, "perm").map(<[i64]>::to_vec);
    let perm_ref = perm.as_deref();
    match get_value(env, node, data_name)? {
        Value::F32(t) => Ok(Value::F32(ops::transpose(t, perm_ref)?)),
        Value::I64(t) => Ok(Value::I64(ops::transpose(t, perm_ref)?)),
        Value::Bool(t) => Ok(Value::Bool(ops::transpose(t, perm_ref)?)),
        Value::F16(t) => Ok(Value::F16(ops::transpose(t, perm_ref)?)),
    }
}

// ---- TASK-7.3c: Attention 系オペ（イシュー #274 結線）----

fn compute_matmul(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let a = get_f32(env, node, input_name(node, 0)?)?;
    let b = get_f32(env, node, input_name(node, 1)?)?;
    if let Some(ops) = dev_ops
        && let Some(out) = interp_device::device_matmul(&node.name, ops, a, b)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((Value::F32(ops::matmul(a, b)?), false))
}

fn compute_softmax(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    let axis = attr_i64(node, "axis", -1);
    if let Some(ops) = dev_ops
        && let Some(out) = interp_device::device_softmax(&node.name, ops, x, axis)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((Value::F32(ops::softmax(x, axis)?), false))
}

fn compute_erf(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    Ok(Value::F32(ops::erf(x)?))
}

// ---- TASK-7.3d: LayerNormalization（イシュー #274 結線）----

fn compute_layer_normalization(
    env: &HashMap<String, Value>,
    node: &NodeProto,
    dev_ops: Option<&dyn BackendOps>,
) -> Result<(Value, bool), InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    let scale = get_f32(env, node, input_name(node, 1)?)?;
    let bias = match node.input.get(2) {
        Some(name) if !name.is_empty() => Some(get_f32(env, node, name)?),
        _ => None,
    };
    let attrs = LayerNormAttrs {
        axis: attr_i64(node, "axis", -1),
        epsilon: attr_f32(node, "epsilon", 1e-5),
    };
    if let Some(ops) = dev_ops
        && let Some(out) =
            interp_device::device_layer_norm(&node.name, ops, x, scale, bias, &attrs)?
    {
        return Ok((Value::F32(out), true));
    }
    Ok((
        Value::F32(ops::layer_normalization(x, scale, bias, &attrs)?),
        false,
    ))
}

/// `Conv(X, W, [B])`（イシュー #2076・親 #2034。`ops::conv` の結線）。
/// 属性は「常に全て書き出す」export 側規約（`export_ops.rs` モジュール
/// 冒頭コメント）に従い export したグラフでは常に存在するが、本関数は
/// interp 単体でも呼べるよう ONNX 仕様の既定値（`strides`／`dilations`
/// 省略時 `[1,1]`・`pads` 省略時 `[0,0,0,0]`・`group` 省略時 `1`・
/// `auto_pad` 省略時 `"NOTSET"`）へ fallback する（`attr_ints_typed`／
/// `attr_i64_typed`／`attr_string` が `None`／属性欠落を透過的に扱う）。
///
/// `kernel_shape`／`strides`／`pads`／`dilations`（INTS）・`group`（INT）・
/// `auto_pad`（STRING）はいずれも `AttributeProto.r#type` を期待型と照合する
/// 型検証版のヘルパーで読む（`attr_i64s`／`attr_i64` の無検証版は使わない）。
/// 型を偽装した不正な ONNX モデル（例: `kernel_shape` を STRING 型として
/// 送り `ints` を空のまま残す）が、型不一致を「省略」と同一視されて無言で
/// 既定値へ変更され、別の畳み込み条件で実行されてしまうのを防ぐ
/// （イシュー #2076 codex-review 指摘。OWASP A03。`.claude/rules/security.md`）。
fn compute_conv(env: &HashMap<String, Value>, node: &NodeProto) -> Result<Value, InterpError> {
    let x = get_f32(env, node, input_name(node, 0)?)?;
    let w = get_f32(env, node, input_name(node, 1)?)?;
    let b = match node.input.get(2) {
        Some(name) if !name.is_empty() => Some(get_f32(env, node, name)?),
        _ => None,
    };
    let attrs = ConvAttrs {
        kernel_shape: attr_ints_typed(node, "kernel_shape")?
            .unwrap_or(&[])
            .to_vec(),
        strides: attr_ints_typed(node, "strides")?.unwrap_or(&[]).to_vec(),
        pads: attr_ints_typed(node, "pads")?.unwrap_or(&[]).to_vec(),
        dilations: attr_ints_typed(node, "dilations")?.unwrap_or(&[]).to_vec(),
        group: attr_i64_typed(node, "group", 1)?,
        auto_pad: attr_string(node, "auto_pad", "NOTSET")?,
    };
    Ok(Value::F32(ops::conv(x, w, b, &attrs)?))
}

/// `node.output` が単一出力であることを検査し、その名前を返す（本モジュールが実装する
/// 全オペは単一出力。`LayerNormalization` の任意出力 `Mean`／`InvStdDev` 宣言もここで
/// 一律拒否する。実装計画 5.3 節・#274 実装計画スコープ外節）。
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

/// `run_with_ops_report` が返す、ノードごとの実行経路の可観測点
/// （イシュー #2077）。テスト・診断専用の内部クレート限定型であり facade
/// へは公開しない（`docs/onnx-gpu-execution-decision.md` §3.1 「dispatcher
/// の配置」承認事項 2 参照）。
#[derive(Debug, Clone, Default)]
pub struct DispatchReport {
    /// device（`BackendOps`）実行に到達したノード名。
    pub device_nodes: Vec<String>,
    /// ホスト実装（`ops::*`）で実行されたノード名（`dev_ops` が `None`
    /// の場合は全ノードがここに入る）。
    pub host_nodes: Vec<String>,
}

/// `run`／`run_with_ops`／`run_with_ops_report` の共通実装。`dev_ops` が
/// `Some` の場合のみ f32 系ノード（`MatMul`／`Gemm`／`Add`／`Mul`／`Div`／
/// `Sqrt`／`Relu`／`Sigmoid`／`Softmax`／`LayerNormalization`）で device
/// 経路を試みる（`interp_device` モジュール冒頭コメント参照）。`report` が
/// `Some` の場合のみノードごとの実行経路を記録する（`run`／`run_with_ops`
/// は記録コストを払わない）。
///
/// `dev_ops` が `None` の場合、各 `compute_*` は device 分岐へ一切入らず
/// 既存コードをそのまま実行するため、本関数の出力は `dev_ops` 追加前の
/// `run` と bit 完全に不変である（イシュー #2077 実装計画 §3.6 契約 (a)）。
fn run_impl(
    graph: &Graph,
    feeds: HashMap<String, Value>,
    dev_ops: Option<&dyn BackendOps>,
    mut report: Option<&mut DispatchReport>,
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
        // `used_device` は各 `compute_*` が実際に device 経路（`interp_
        // device::device_*` が `Ok(Some(_))` を返した場合）を採用したかを
        // 正確に表す（device 結線対象外の op は常に `false` で揃える。
        // 推測・近似ではなく `compute_*` の戻り値そのものから得る）。
        // `Conv`（イシュー #2076）は device 実行の対象外（#2077／#2222 の
        // 結線範囲は `interp_device` モジュール冒頭コメント参照）のため
        // 常にホスト実装（`ops::conv`）で実行し `false` を報告する。
        let (out_value, used_device) = match node.op_type.as_str() {
            "Gemm" => compute_gemm(&env, node, dev_ops)?,
            "Relu" => compute_relu(&env, node, dev_ops)?,
            "Sigmoid" => compute_sigmoid(&env, node, dev_ops)?,
            "Shape" => (compute_shape(&env, node)?, false),
            "Gather" => (compute_gather(&env, node)?, false),
            "Unsqueeze" => (compute_unsqueeze(&env, node)?, false),
            "Concat" => (compute_concat(&env, node)?, false),
            "Slice" => (compute_slice(&env, node)?, false),
            "Add" => compute_add(&env, node, dev_ops)?,
            "Mul" => compute_mul(&env, node, dev_ops)?,
            "Div" => compute_div(&env, node, dev_ops)?,
            "Mod" => (compute_mod(&env, node)?, false),
            "Sqrt" => compute_sqrt(&env, node, dev_ops)?,
            "Constant" => (compute_constant(node)?, false),
            "Cast" => (compute_cast(&env, node)?, false),
            "Reshape" => (compute_reshape(&env, node)?, false),
            "Squeeze" => (compute_squeeze(&env, node)?, false),
            "Transpose" => (compute_transpose(&env, node)?, false),
            "MatMul" => compute_matmul(&env, node, dev_ops)?,
            "Softmax" => compute_softmax(&env, node, dev_ops)?,
            "Erf" => (compute_erf(&env, node)?, false),
            "LayerNormalization" => compute_layer_normalization(&env, node, dev_ops)?,
            "Conv" => (compute_conv(&env, node)?, false),
            other => return Err(InterpError::UnsupportedOp(other.to_string())),
        };
        if let Some(r) = report.as_deref_mut() {
            if used_device {
                r.device_nodes.push(node.name.clone());
            } else {
                r.host_nodes.push(node.name.clone());
            }
        }
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

/// `Graph`（`build_graph` が構築したトポロジカル順検証済みグラフ）をホスト
/// CPU 実装（`ops::*`）のみで実行する。
///
/// 呼び出し元は `onnx-interop` 利用者（`fandhe_ai::interop::onnx::OnnxModel::
/// run` の opt-in OFF 経路〈既定〉・facade 経由の GPU 実行 opt-in が無効な
/// 場合。TASK-7.4）。`feeds` は `graph.inputs` のうち initializer を持たない
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
    run_impl(graph, feeds, None, None)
}

/// [`run`] と同じグラフ実行だが、`ops`（CPU／CUDA／Metal いずれの
/// `BackendOps` 実装でもよい）経由の device 実行を試みる（イシュー
/// #2077。`fandhe_ai::set_cuda_onnx_gpu_execution_enabled` 等の facade
/// opt-in から到達する）。
///
/// 対象 op（`interp_device` モジュール冒頭コメント参照）の f32 経路のみ
/// device を試み、`BackendError::Unsupported`／`ShapeMismatch` はホスト
/// 実装（`ops::*`）へフォールバックする。それ以外のバックエンドエラーは
/// [`InterpError::Backend`] として伝播する（黙示フォールバックしない。
/// OWASP A08）。非 f32 経路・形状操作系（`Shape`／`Gather`／`Unsqueeze`／
/// `Concat`／`Slice`／`Cast`／`Reshape`／`Squeeze`／`Transpose`／
/// `Constant`）・`Mod`／`Erf` は常にホスト実行のまま（`run` と同一実装）。
pub fn run_with_ops(
    graph: &Graph,
    feeds: HashMap<String, Value>,
    ops: &dyn BackendOps,
) -> Result<HashMap<String, Value>, InterpError> {
    run_impl(graph, feeds, Some(ops), None)
}

/// [`run_with_ops`] と同じだが、ノードごとの実行経路（[`DispatchReport`]）
/// も返す（テスト・診断専用。`ops-interop` の内部クレート限定 API であり
/// facade へは公開しない）。
pub fn run_with_ops_report(
    graph: &Graph,
    feeds: HashMap<String, Value>,
    ops: &dyn BackendOps,
) -> Result<(HashMap<String, Value>, DispatchReport), InterpError> {
    let mut report = DispatchReport::default();
    let result = run_impl(graph, feeds, Some(ops), Some(&mut report))?;
    Ok((result, report))
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

    fn node_with_attrs(
        op_type: &str,
        input: Vec<&str>,
        output: Vec<&str>,
        attribute: Vec<super::super::proto::AttributeProto>,
    ) -> NodeProto {
        let mut n = node(op_type, input, output);
        n.attribute = attribute;
        n
    }

    fn build_attr_i64(name: &str, i: i64) -> super::super::proto::AttributeProto {
        super::super::proto::AttributeProto {
            name: name.to_string(),
            i,
            ..Default::default()
        }
    }

    fn build_attr_i64s(name: &str, ints: Vec<i64>) -> super::super::proto::AttributeProto {
        super::super::proto::AttributeProto {
            name: name.to_string(),
            ints,
            ..Default::default()
        }
    }

    fn build_attr_f32(name: &str, f: f32) -> super::super::proto::AttributeProto {
        super::super::proto::AttributeProto {
            name: name.to_string(),
            f,
            ..Default::default()
        }
    }

    /// `r#type` を明示的に `INTS` へ設定した属性を組み立てる（`Conv` の
    /// `kernel_shape`／`strides`／`pads`／`dilations` 用。`build_attr_i64s`
    /// は意図的に `r#type` を設定しない他オペ用ビルダーのため、型検証を
    /// 経由する `attr_ints_typed` のテストには使えない）。
    fn build_attr_ints_typed(name: &str, ints: Vec<i64>) -> super::super::proto::AttributeProto {
        super::super::proto::AttributeProto {
            name: name.to_string(),
            ints,
            r#type: super::super::proto::attribute_type::INTS,
            ..Default::default()
        }
    }

    /// `r#type` を明示的に `INT` へ設定した属性を組み立てる（`Conv` の
    /// `group` 用）。
    fn build_attr_i64_typed(name: &str, i: i64) -> super::super::proto::AttributeProto {
        super::super::proto::AttributeProto {
            name: name.to_string(),
            i,
            r#type: super::super::proto::attribute_type::INT,
            ..Default::default()
        }
    }

    /// `r#type` を明示的に `STRING` へ設定した属性を組み立てる（`Conv` の
    /// `auto_pad` 用）。
    fn build_attr_string_typed(name: &str, s: &str) -> super::super::proto::AttributeProto {
        super::super::proto::AttributeProto {
            name: name.to_string(),
            s: s.as_bytes().to_vec(),
            r#type: super::super::proto::attribute_type::STRING,
            ..Default::default()
        }
    }

    /// `Conv` の全属性を（型検証込みで）正しく設定したノードを組み立てる。
    fn conv_node_with_attrs(extra: Vec<super::super::proto::AttributeProto>) -> NodeProto {
        let mut attrs = vec![
            build_attr_ints_typed("kernel_shape", vec![1, 1]),
            build_attr_ints_typed("strides", vec![1, 1]),
            build_attr_ints_typed("pads", vec![0, 0, 0, 0]),
            build_attr_ints_typed("dilations", vec![1, 1]),
            build_attr_i64_typed("group", 1),
            build_attr_string_typed("auto_pad", "NOTSET"),
        ];
        attrs.retain(|a| !extra.iter().any(|e| e.name == a.name));
        attrs.extend(extra);
        node_with_attrs("Conv", vec!["x", "w"], vec!["y"], attrs)
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
    fn raw_to_value_converts_all_dtypes() {
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
            Value::I64(t) => {
                assert_eq!(t.as_slice().unwrap(), &[1, 2, 3]);
                assert_eq!(t.shape(), &[3]);
            }
            _ => panic!("Value::I64 を期待"),
        }

        let b = raw_to_value(&RawTensor::Bool {
            data: vec![true, false],
            shape: vec![2],
        })
        .unwrap();
        assert!(matches!(b, Value::Bool(_)));

        let h = raw_to_value(&RawTensor::F16 {
            data: vec![f16::from_f32(1.5)],
            shape: vec![1],
        })
        .unwrap();
        assert!(matches!(h, Value::F16(_)));
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
    fn compute_gather_i64_accepts_consistent_indices_shape() {
        // `Value::I64` 経路（`compute_gather` → `ops::gather`）が idx_shape と
        // 実データ長が一致する妥当な入力で正常に動作することを固定化する
        // （イシュー #274 で `Value::I64` が `Tensor<i64>` 化された後の回帰）。
        // 長さ不一致（`OpError::LengthMismatch`）を実際に拒否する検査自体は
        // `crates/onnx-interop/src/ops/gather.rs` 側のテストが担う（PR #298・
        // Cursor Bugbot 指摘の回帰ガードはそちらに現存する）。
        let n = node("Gather", vec!["data", "idx"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["data", "idx"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "data".to_string(),
            Value::I64(Tensor::<i64>::new(vec![10, 20, 30], &[3]).unwrap()),
        );
        feeds.insert(
            "idx".to_string(),
            Value::I64(Tensor::<i64>::new(vec![0, 1], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[10, 20]),
            _ => panic!("Value::I64 を期待"),
        }
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

    #[test]
    fn unsqueeze_i64_with_scalar_gather_shim_keeps_flat_shape_for_single_element() {
        // `unsqueeze_i64_with_scalar_gather_shim` 単体の直接固定化。入力
        // shape=[1]（`numel()==1`）・axes=[0] は、真の ONNX Unsqueeze 意味論
        // （出力 rank 2・shape=[1,1]）ではなく `slice_repro.onnx` 互換のため
        // shape=[1] のフラット表現を維持する（このシムがないと
        // `slice_repro_onnx_end_to_end_matches_reference_within_req7_tolerance`
        // 〈`tests/onnx_interp.rs`〉が `Concat` 内で `RankMismatch` により失敗する。
        // シムを削除・変更する場合は本テストの意図〈関数ドキュメンテーション
        // コメント参照〉を踏まえること）。
        let t = Tensor::<i64>::new(vec![7], &[1]).unwrap();
        let v = unsqueeze_i64_with_scalar_gather_shim(&t, &[0]).unwrap();
        match v {
            Value::I64(out) => {
                assert_eq!(out.shape(), &[1]);
                assert_eq!(out.as_slice().unwrap(), &[7]);
            }
            other => panic!("Value::I64 を期待したが {other:?}"),
        }
    }

    #[test]
    fn unsqueeze_i64_with_scalar_gather_shim_delegates_multi_element_to_generic_unsqueeze() {
        // `numel() != 1` はシムを経由せず `ops::unsqueeze`（真の意味論）へ委譲する。
        let t = Tensor::<i64>::new(vec![1, 2, 3], &[3]).unwrap();
        let v = unsqueeze_i64_with_scalar_gather_shim(&t, &[0]).unwrap();
        match v {
            Value::I64(out) => assert_eq!(out.shape(), &[1, 3]),
            other => panic!("Value::I64 を期待したが {other:?}"),
        }
    }

    #[test]
    fn run_gather_multidim_i64_shape_no_longer_restricted() {
        // 旧実装は i64 直接経路を 1 次元のみに制限していた（`I64ShapeUnsupported`）。
        // イシュー #274 で `Tensor<i64>` 化した `ops::gather` ジェネリック実装に
        // 委譲するようになったため、多次元 i64 data でも成功することを固定化する。
        let n = node("Gather", vec!["data", "idx"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["data", "idx"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        // data: shape [3,2] の i64 (行ごとに集める)
        feeds.insert(
            "data".to_string(),
            Value::I64(Tensor::<i64>::new(vec![1, 2, 3, 4, 5, 6], &[3, 2]).unwrap()),
        );
        feeds.insert(
            "idx".to_string(),
            Value::I64(Tensor::<i64>::new(vec![0, 2], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert_eq!(t.as_slice().unwrap(), &[1, 2, 5, 6]);
            }
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_add() {
        let n = node("Add", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 2.0], &[2]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::F32(Tensor::<f32>::new(vec![10.0, 20.0], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::F32(t) => {
                assert_eq!(t.get(&[0]).unwrap(), 11.0);
                assert_eq!(t.get(&[1]).unwrap(), 22.0);
            }
            _ => panic!("Value::F32 を期待"),
        }
    }

    // ---- i64 版算術オペの結線（イシュー #87 残作業。transformer.onnx の
    // head_dim 算出〈Shape -> Gather -> Div〉が要求する `i64` 同士の Add/Mul/Div/Mod。
    // `ops::arith` 側の数値仕様の単体テストは `ops::arith::tests` を参照し、ここでは
    // `run` 経由のディスパッチ（`compute_add`/`compute_mul`/`compute_div`/
    // `compute_mod` の (Value, Value) 分岐）のみを確認する）----

    #[test]
    fn run_end_to_end_add_i64() {
        let n = node("Add", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::I64(Tensor::<i64>::new(vec![1, 2], &[2]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::I64(Tensor::<i64>::new(vec![10, 20], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[11, 22]),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_mul_i64() {
        let n = node("Mul", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::I64(Tensor::<i64>::new(vec![3, 4], &[2]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::I64(Tensor::<i64>::new(vec![5, 6], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[15, 24]),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_div_i64() {
        // transformer.onnx の `/layers.0/self_attn/Div`（Gather 由来の head_dim を
        // 定数で割る整数除算経路）と同型の再現ケース（イシュー #87 実測確認済み）。
        let n = node("Div", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::I64(Tensor::<i64>::new(vec![8], &[]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::I64(Tensor::<i64>::new(vec![2], &[]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[4]),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_mod_i64() {
        let n = node_with_attrs(
            "Mod",
            vec!["a", "b"],
            vec!["y"],
            vec![build_attr_i64("fmod", 1)],
        );
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::I64(Tensor::<i64>::new(vec![7], &[]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::I64(Tensor::<i64>::new(vec![3], &[]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[1]),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_rejects_mixed_f32_i64_div() {
        // f32/i64 混在は暗黙変換せず拒否する（`Cast` を明示的に経由させる既存方針。
        // compute_div の (Value, Value) 分岐コメント参照）。
        let n = node("Div", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::F32(Tensor::<f32>::new(vec![8.0], &[]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::I64(Tensor::<i64>::new(vec![2], &[]).unwrap()),
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(err, InterpError::TypeMismatch { .. }));
    }

    #[test]
    fn run_rejects_i64_div_by_zero() {
        let n = node("Div", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::I64(Tensor::<i64>::new(vec![1], &[]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::I64(Tensor::<i64>::new(vec![0], &[]).unwrap()),
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(
            err,
            InterpError::Op(OpError::IntegerDivisionFailed { op: "Div" })
        ));
    }

    #[test]
    fn run_end_to_end_cast_f32_to_i64() {
        let n = node_with_attrs("Cast", vec!["x"], vec!["y"], vec![build_attr_i64("to", 7)]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.7, -2.3], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[1, -2]),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_rejects_unsupported_cast_combination() {
        // Bool -> INT64 は未対応の組（`ops::cast` は F32<->I64/Bool/F16 のみ対応）。
        let n = node_with_attrs("Cast", vec!["x"], vec!["y"], vec![build_attr_i64("to", 7)]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::Bool(Tensor::<bool>::new(vec![true, false], &[2]).unwrap()),
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(err, InterpError::TypeMismatch { .. }));
    }

    #[test]
    fn run_end_to_end_reshape() {
        let n = node("Reshape", vec!["x", "shape"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["x", "shape"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap()),
        );
        feeds.insert(
            "shape".to_string(),
            Value::I64(Tensor::<i64>::new(vec![3, 2], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::F32(t) => assert_eq!(t.shape(), &[3, 2]),
            _ => panic!("Value::F32 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_transpose_default_reverses_axes() {
        let n = node("Transpose", vec!["x"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::F32(t) => assert_eq!(t.shape(), &[3, 2]),
            _ => panic!("Value::F32 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_constant_tensor_attribute() {
        let t = super::super::proto::TensorProto {
            dims: vec![2],
            data_type: super::super::proto::data_type::FLOAT,
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };
        let attr = super::super::proto::AttributeProto {
            name: "value".to_string(),
            t: Some(t),
            ..Default::default()
        };
        let n = node_with_attrs("Constant", vec![], vec!["y"], vec![attr]);
        let g = empty_graph(vec![n], vec![], vec!["y"]);
        let result = run(&g, StdHashMap::new()).unwrap();
        match &result["y"] {
            Value::F32(t) => assert_eq!(t.as_slice().unwrap(), &[1.0, 2.0]),
            _ => panic!("Value::F32 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_constant_value_int() {
        let attr = build_attr_i64("value_int", 42);
        let n = node_with_attrs("Constant", vec![], vec!["y"], vec![attr]);
        let g = empty_graph(vec![n], vec![], vec!["y"]);
        let result = run(&g, StdHashMap::new()).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.get(&[]).unwrap(), 42),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_constant_value_ints() {
        let attr = build_attr_i64s("value_ints", vec![1, 2, 3]);
        let n = node_with_attrs("Constant", vec![], vec!["y"], vec![attr]);
        let g = empty_graph(vec![n], vec![], vec!["y"]);
        let result = run(&g, StdHashMap::new()).unwrap();
        match &result["y"] {
            Value::I64(t) => assert_eq!(t.as_slice().unwrap(), &[1, 2, 3]),
            _ => panic!("Value::I64 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_matmul() {
        let n = node("MatMul", vec!["a", "b"], vec!["y"]);
        let g = empty_graph(vec![n], vec!["a", "b"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "a".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap()),
        );
        feeds.insert(
            "b".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::F32(t) => assert_eq!(t.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]),
            _ => panic!("Value::F32 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_softmax() {
        let n = node_with_attrs(
            "Softmax",
            vec!["x"],
            vec!["y"],
            vec![build_attr_i64("axis", -1)],
        );
        let g = empty_graph(vec![n], vec!["x"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        match &result["y"] {
            Value::F32(t) => {
                assert!((t.get(&[0]).unwrap() - 0.5).abs() < 1e-6);
                assert!((t.get(&[1]).unwrap() - 0.5).abs() < 1e-6);
            }
            _ => panic!("Value::F32 を期待"),
        }
    }

    #[test]
    fn run_end_to_end_layer_normalization() {
        let attrs = vec![build_attr_i64("axis", -1), build_attr_f32("epsilon", 1e-5)];
        let n = node_with_attrs("LayerNormalization", vec!["x", "scale"], vec!["y"], attrs);
        let g = empty_graph(vec![n], vec!["x", "scale"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap()),
        );
        feeds.insert(
            "scale".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap()),
        );
        let result = run(&g, feeds).unwrap();
        assert!(matches!(&result["y"], Value::F32(_)));
    }

    #[test]
    fn run_layer_normalization_multi_output_rejected() {
        // `LayerNormalization` の任意出力 `Mean`／`InvStdDev` 宣言は
        // `require_single_output` により一律 fail-closed で拒否する（#274 実装計画
        // スコープ外節）。
        let attrs = vec![build_attr_i64("axis", -1)];
        let n = node_with_attrs(
            "LayerNormalization",
            vec!["x", "scale"],
            vec!["y", "mean", "invstddev"],
            attrs,
        );
        let g = empty_graph(vec![n], vec!["x", "scale"], vec!["y"]);
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 2.0], &[1, 2]).unwrap()),
        );
        feeds.insert(
            "scale".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap()),
        );
        let err = run(&g, feeds).unwrap_err();
        assert!(matches!(
            err,
            InterpError::OutputArityMismatch {
                expected: 1,
                actual: 3,
                ..
            }
        ));
    }

    /// `run_end_to_end_conv` の feed（`X: [1,1,2,2]`・`W: [1,1,1,1]`。
    /// `kernel_shape=[1,1]`・`stride=1`・`pad=0` の恒等写像に近い最小構成）
    /// を組み立てる（型検証を通過する正常系の固定に使う）。
    fn conv_feeds() -> StdHashMap<String, Value> {
        let mut feeds = StdHashMap::new();
        feeds.insert(
            "x".to_string(),
            Value::F32(Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap()),
        );
        feeds.insert(
            "w".to_string(),
            Value::F32(Tensor::<f32>::new(vec![2.0], &[1, 1, 1, 1]).unwrap()),
        );
        feeds
    }

    #[test]
    fn run_end_to_end_conv_accepts_correctly_typed_attrs() {
        // すべての Conv 属性が期待型（INTS／INT／STRING）で宣言されている
        // 正常系（`export_ops.rs` が実際に export する形と同じ）。
        let n = conv_node_with_attrs(vec![]);
        let g = empty_graph(vec![n], vec!["x", "w"], vec!["y"]);
        let result = run(&g, conv_feeds()).unwrap();
        match &result["y"] {
            Value::F32(t) => assert_eq!(t.as_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]),
            _ => panic!("Value::F32 を期待"),
        }
    }

    #[test]
    fn compute_conv_rejects_kernel_shape_with_mismatched_attribute_type() {
        // P0（イシュー #2076 codex-review 指摘）: `kernel_shape` を INTS では
        // なく STRING 型として送る型偽装。`ints` は空のままだが、これを
        // 「省略された」場合と同一視して既定値へ fallback してはならない。
        let n = conv_node_with_attrs(vec![build_attr_string_typed("kernel_shape", "")]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "kernel_shape"
        ));
    }

    #[test]
    fn compute_conv_rejects_strides_with_mismatched_attribute_type() {
        let n = conv_node_with_attrs(vec![build_attr_string_typed("strides", "")]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "strides"
        ));
    }

    #[test]
    fn compute_conv_rejects_pads_with_mismatched_attribute_type() {
        let n = conv_node_with_attrs(vec![build_attr_string_typed("pads", "")]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "pads"
        ));
    }

    #[test]
    fn compute_conv_rejects_dilations_with_mismatched_attribute_type() {
        let n = conv_node_with_attrs(vec![build_attr_string_typed("dilations", "")]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "dilations"
        ));
    }

    #[test]
    fn compute_conv_rejects_group_with_mismatched_attribute_type() {
        // `group`（本来 INT）を INTS 型として送る型偽装。
        let n = conv_node_with_attrs(vec![build_attr_ints_typed("group", vec![1])]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "group"
        ));
    }

    #[test]
    fn compute_conv_rejects_auto_pad_with_mismatched_attribute_type() {
        // `auto_pad`（本来 STRING）を INT 型として送る型偽装。`s` は空の
        // ままだが、空文字列 `""` を UTF-8 として妥当に復号できてしまうため
        // 型検証がなければ無言で通過してしまう（codex-review 指摘の core）。
        let n = conv_node_with_attrs(vec![build_attr_i64_typed("auto_pad", 0)]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "auto_pad"
        ));
    }

    #[test]
    fn compute_conv_rejects_duplicate_attribute() {
        // 同名属性の複数宣言はどちらを採用すべきか ONNX 仕様上一意に決まら
        // ないため、無言で先頭／末尾を採用せず fail-closed に拒否する。
        let mut n = conv_node_with_attrs(vec![]);
        n.attribute
            .push(build_attr_ints_typed("kernel_shape", vec![3, 3]));
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "kernel_shape"
        ));
    }

    #[test]
    fn compute_conv_rejects_present_but_empty_ints_attribute() {
        // `r#type == INTS` を正しく宣言しつつ `ints` を空のまま残す型偽装。
        // `check_attr_type` だけでは通過してしまい `.unwrap_or(&[])` 経由で
        // 「省略された」場合と同じ既定値 fallback に合流するため、空リストは
        // 型検証と別に明示的に拒否する（advisor 指摘。P0 の「期待フィールド
        // が空」という不正表現に対応）。
        let n = conv_node_with_attrs(vec![build_attr_ints_typed("strides", vec![])]);
        let err = compute_conv(&conv_feeds(), &n).unwrap_err();
        assert!(matches!(
            err,
            InterpError::InvalidAttribute { ref attr, .. } if attr == "strides"
        ));
    }

    #[test]
    fn compute_conv_falls_back_to_defaults_when_attrs_omitted() {
        // 属性が丸ごと省略されている場合（型検証の対象外）は従来どおり
        // ONNX 仕様の既定値へ fallback する。
        let n = node("Conv", vec!["x", "w"], vec!["y"]);
        let result = compute_conv(&conv_feeds(), &n).unwrap();
        match result {
            Value::F32(t) => assert_eq!(t.as_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]),
            _ => panic!("Value::F32 を期待"),
        }
    }
}
