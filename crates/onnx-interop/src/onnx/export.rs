//! 内部グラフ表現 `Graph` -> `GraphProto`／`ModelProto` への降下（REQ-7・イシュー #1772）。
//!
//! `graph::build_graph`（`ModelProto` -> `Graph`）の逆方向を担う。親イシュー #1653
//! （ONNX export）の最初の分解であり、本モジュールが担うのは
//! **「既に確定した `NodeProto` のリストを `GraphProto.node` へ詰める」という
//! 構造的・機械的な組み立てのみ**である。`Graph.nodes: Vec<NodeProto>` は decode
//! 由来の `NodeProto` をそのまま保持しており、op ごとの属性（`AttributeProto`）は
//! 既にプロトコル型として確定済みのため、本モジュールは op_type の意味論
//! （属性の作り方）には一切関与しない。
//!
//! ## スコープ境界（兄弟イシューとの切り分け）
//!
//! - 内部 op（Rust ネイティブの属性表現）-> `NodeProto`（op_type・属性）という
//!   意味論的な逆マッピングは #1773 のスコープ。
//! - import -> export -> import の構造一致 roundtrip テスト・未対応 op の
//!   fail-closed 確認は #1774 のスコープ。
//! - facade 公開は #1775（#1652 の判断待ち）。本モジュールは `onnx-interop`
//!   内部 API のみを提供し facade へは一切公開しない
//!   （`onnx-interop` は crates.io 非公開クレート・`docs/compat-api-scope.md` の
//!   対象外）。
//!
//! ## `value_info` を常に空にする契約
//!
//! `Graph` は中間テンソルの型／形状情報を保持していない（`graph::build_graph` は
//! `GraphProto.value_info` を読まずに `Graph` を構築する）。`ValueInfoProto` 自体も
//! 本クレートでは `name` のみの部分実装であり `TypeProto` を意図的に持たない
//! （`proto.rs` 冒頭コメント）。したがって本モジュールが export したモデルは、
//! 本クレート自身の decode（`graph::build_graph`）に対しては構造的にラウンド
//! トリップ可能だが、`onnx.checker`（型必須）等の外部 ONNX ツールでの厳密な
//! 妥当性検証は保証しない。この制約はスコープ外事項としてイシュー側で追跡する
//! （`.claude/rules/out-of-scope-tracking.md`）。
//!
//! ## tensor データの書き出し形式（契約）
//!
//! `encode_tensor` は常に `raw_data`（リトルエンディアンのバイト列）のみへ
//! 書き出し、`float_data`／`int64_data` は常に空のままにする。理由:
//! (a) `decode_tensor` は既に `raw_data` を typed data より優先する解決順序を
//! 採用しており、単一の書き出し経路にすることで decode 側の分岐を経ずに常に
//! 同じパスで bit-exact に再現できる。
//! (b) 新規の手書き protobuf encode を増やさず（`derive(Message)` の `encode`
//! のみを使う方針を維持）、`Vec<u8>` へのバイト直列化という通常の Rust コードに
//! 留められる。

use super::graph::{self, Graph, RawTensor};
use super::proto::{GraphProto, ModelProto, OperatorSetIdProto, TensorProto, ValueInfoProto};
use std::fmt;

/// export 処理で発生しうるエラー。本番経路で `unwrap()` / `expect()` を使わない
/// 方針（`coding-rust.md`）に従い、不正な `Graph`（#1773 以降で手組みされる
/// ケースを含む）は必ずこの型で呼び出し元へ伝播する。
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportError {
    /// `RawTensor` の `shape` に負の値が含まれる（不正な形状）。
    NegativeDim { tensor_name: String, dim: i64 },
    /// `shape` の積（要素数）が `usize` の範囲を超える。
    ElementCountOverflow { tensor_name: String },
    /// `RawTensor` の実データ長（要素数）が `shape` から導出した期待要素数と
    /// 一致しない（`decode_tensor` の逆方向。fail-closed に拒否する）。
    ShapeDataMismatch {
        tensor_name: String,
        expected_elements: usize,
        actual_elements: usize,
    },
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::NegativeDim { tensor_name, dim } => {
                write!(f, "dims に負の値（tensor={tensor_name}）: {dim}")
            }
            ExportError::ElementCountOverflow { tensor_name } => {
                write!(
                    f,
                    "dims の積（要素数）がオーバーフロー（tensor={tensor_name}）"
                )
            }
            ExportError::ShapeDataMismatch {
                tensor_name,
                expected_elements,
                actual_elements,
            } => write!(
                f,
                "データ長不整合（tensor={tensor_name}）: shape から期待される要素数={expected_elements} 実データ要素数={actual_elements}"
            ),
        }
    }
}
impl std::error::Error for ExportError {}

/// `element_count` の失敗を `ExportError` へマップする（`graph::element_count` は
/// `GraphError` を返すため、export 側の型へ変換する薄いアダプタ）。
fn element_count(tensor_name: &str, shape: &[i64]) -> Result<usize, ExportError> {
    graph::element_count(tensor_name, shape).map_err(|e| match e {
        graph::GraphError::NegativeDim { tensor_name, dim } => {
            ExportError::NegativeDim { tensor_name, dim }
        }
        graph::GraphError::ElementCountOverflow { tensor_name } => {
            ExportError::ElementCountOverflow { tensor_name }
        }
        // `graph::element_count` はこの 2 variant 以外を返さない（`graph.rs`
        // 実装参照）。到達不能だが `GraphError` は `#[non_exhaustive]` のため
        // 網羅的な match として fallback を用意する（将来 variant が増えても
        // ここでコンパイルが壊れず、実行時にも安全側〈オーバーフロー扱い〉へ倒す）。
        _ => ExportError::ElementCountOverflow {
            tensor_name: tensor_name.to_string(),
        },
    })
}

/// export 時の付随情報（`Graph` 自体が持たないメタデータ）。
///
/// `Graph` はグラフ名を持たない（`build_graph` が `GraphProto.name` を破棄して
/// いる）ため、グラフ名はここで指定する。既定値はイシュー #1772 の事前調査で
/// 実測したフィクスチャ `tests/fixtures/model.onnx`（PyTorch 2.12.1 export）の
/// 値（`ir_version=8`・`opset_import=[{domain: "", version: 17}]`）と一致させる。
#[derive(Clone, Debug, PartialEq)]
pub struct ExportOptions {
    pub ir_version: i64,
    pub producer_name: String,
    pub graph_name: String,
    pub opset_domain: String,
    pub opset_version: i64,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            ir_version: 8,
            producer_name: "fandhe-ai-onnx-interop".to_string(),
            graph_name: "fandhe_ai_export".to_string(),
            opset_domain: String::new(),
            opset_version: 17,
        }
    }
}

/// `RawTensor` を `TensorProto` へ復号する（`graph::decode_tensor` の逆方向）。
///
/// 検証順序は `decode_tensor` と対称: `shape` の非負性・要素数の `checked_mul`
/// （`element_count`）を先に行い、実データ長との一致を確認してから初めて
/// バイト列へ変換する（長さ・形状検証の先行。`security.md` A03）。
///
/// 契約: 常に `raw_data` のみへ書き出す（モジュール冒頭コメント参照）。
pub fn encode_tensor(name: &str, tensor: &RawTensor) -> Result<TensorProto, ExportError> {
    match tensor {
        RawTensor::F32 { data, shape } => {
            let expected = element_count(name, shape)?;
            if data.len() != expected {
                return Err(ExportError::ShapeDataMismatch {
                    tensor_name: name.to_string(),
                    expected_elements: expected,
                    actual_elements: data.len(),
                });
            }
            let mut raw_data = Vec::with_capacity(data.len() * 4);
            for v in data {
                raw_data.extend_from_slice(&v.to_le_bytes());
            }
            Ok(TensorProto {
                dims: shape.clone(),
                data_type: super::proto::data_type::FLOAT,
                float_data: Vec::new(),
                int64_data: Vec::new(),
                name: name.to_string(),
                raw_data,
            })
        }
        RawTensor::I64 { data, shape } => {
            let expected = element_count(name, shape)?;
            if data.len() != expected {
                return Err(ExportError::ShapeDataMismatch {
                    tensor_name: name.to_string(),
                    expected_elements: expected,
                    actual_elements: data.len(),
                });
            }
            let mut raw_data = Vec::with_capacity(data.len() * 8);
            for v in data {
                raw_data.extend_from_slice(&v.to_le_bytes());
            }
            Ok(TensorProto {
                dims: shape.clone(),
                data_type: super::proto::data_type::INT64,
                float_data: Vec::new(),
                int64_data: Vec::new(),
                name: name.to_string(),
                raw_data,
            })
        }
        RawTensor::Bool { data, shape } => {
            let expected = element_count(name, shape)?;
            if data.len() != expected {
                return Err(ExportError::ShapeDataMismatch {
                    tensor_name: name.to_string(),
                    expected_elements: expected,
                    actual_elements: data.len(),
                });
            }
            // decode 側（`graph::decode_tensor` BOOL 分岐）の `b != 0` と対称に
            // 1 バイト/要素・0/1 で書き出す（ONNX/NumPy の bool テンソル慣習）。
            let raw_data: Vec<u8> = data.iter().map(|&b| u8::from(b)).collect();
            Ok(TensorProto {
                dims: shape.clone(),
                data_type: super::proto::data_type::BOOL,
                float_data: Vec::new(),
                int64_data: Vec::new(),
                name: name.to_string(),
                raw_data,
            })
        }
        RawTensor::F16 { data, shape } => {
            let expected = element_count(name, shape)?;
            if data.len() != expected {
                return Err(ExportError::ShapeDataMismatch {
                    tensor_name: name.to_string(),
                    expected_elements: expected,
                    actual_elements: data.len(),
                });
            }
            let mut raw_data = Vec::with_capacity(data.len() * 2);
            for v in data {
                raw_data.extend_from_slice(&v.to_le_bytes());
            }
            Ok(TensorProto {
                dims: shape.clone(),
                data_type: super::proto::data_type::FLOAT16,
                float_data: Vec::new(),
                int64_data: Vec::new(),
                name: name.to_string(),
                raw_data,
            })
        }
    }
}

/// `Graph` から `ModelProto` を構築する（`build_graph` の逆方向）。
///
/// 呼び出し元は #1773 以降（内部 op -> `NodeProto` マッピング後の export
/// パイプライン全体）・将来の codegen。本関数自体は `graph.nodes`（既に
/// `NodeProto` として確定済み）を素通しするのみで、op_type の意味論には
/// 一切関与しない（モジュール冒頭コメント参照。未対応 op の検査は #1773／#1774
/// のスコープ）。
pub fn build_model_proto(
    graph: &Graph,
    options: &ExportOptions,
) -> Result<ModelProto, ExportError> {
    // `graph.initializers` は `HashMap` のため走査順が保証されない。同じ
    // `Graph` から呼び出すたびに異なるバイト列が生成されるのを避けるため、
    // テンソル名でソートしてから `encode_tensor` を適用する（決定的な出力。
    // reproducibility・テスト容易性のための明示的な設計判断）。
    // `(name, tensor)` を直接 `Vec` へ集めてソートすることで、名前だけを
    // 先にソートしてから同じ `HashMap` を再度 `get` で引き直す（`Some` が
    // 保証されるにもかかわらずエラー分岐を用意する必要が生じる）間接参照を
    // 避ける（レビュー指摘）。
    let mut entries: Vec<(&String, &RawTensor)> = graph.initializers.iter().collect();
    entries.sort_by_key(|(name, _)| *name);
    let mut initializer = Vec::with_capacity(entries.len());
    for (name, tensor) in entries {
        initializer.push(encode_tensor(name, tensor)?);
    }

    let input: Vec<ValueInfoProto> = graph
        .inputs
        .iter()
        .map(|name| ValueInfoProto { name: name.clone() })
        .collect();
    let output: Vec<ValueInfoProto> = graph
        .outputs
        .iter()
        .map(|name| ValueInfoProto { name: name.clone() })
        .collect();

    let graph_proto = GraphProto {
        node: graph.nodes.clone(),
        name: options.graph_name.clone(),
        initializer,
        input,
        output,
        // 契約: 常に空（モジュール冒頭コメント参照）。
        value_info: Vec::new(),
    };

    Ok(ModelProto {
        ir_version: options.ir_version,
        producer_name: options.producer_name.clone(),
        graph: Some(graph_proto),
        opset_import: vec![OperatorSetIdProto {
            domain: options.opset_domain.clone(),
            version: options.opset_version,
        }],
    })
}
