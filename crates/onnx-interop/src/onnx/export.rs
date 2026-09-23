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
//!   意味論的な逆マッピングは #1773 のスコープ（`export_ops` モジュール）。
//!   `build_model_proto` は `export_ops::check_exportable` で
//!   `graph.nodes`（decode 由来・または #1773 以降に手組みされた `NodeProto`）が
//!   `interp.rs` 対応 23 op の allowlist（`export_ops::SUPPORTED_OP_TYPES`）・
//!   既定 opset（`domain` が空文字列）に収まっているかを fail-closed に検査して
//!   から組み立てる（詳細対応表は `docs/onnx-export-op-mapping.md`）。
//! - import -> export -> import の構造一致 roundtrip テスト・未対応 op の
//!   fail-closed 確認は `tests/onnx_export_roundtrip.rs` で固定済み（#1774）。
//! - facade 公開は #1775 で publish 承認取得後の段階 0 として整理し、
//!   #2018 で `fandhe_ai::interop::onnx::{OnnxModel::to_bytes,
//!   OnnxModel::to_path, OnnxExportOptions}` として実装済み（`docs/facade-
//!   onnx-export-exposure-decision.md`。案 B・薄いラッパー型）。facade は
//!   `build_model_proto` → `proto::encode_model` への委譲のみで、本
//!   モジュール自体は非公開クレート `onnx-interop` 内に留まる（facade は
//!   `crates/facade/src/interop/onnx.rs` の `pub use` で本モジュールの型
//!   を再エクスポートしない）。
//! - `fandhe_ai_autodiff::nn::Module` の層列から `Graph`（本モジュールの
//!   `build_model_proto` が受け取れる形）を組み立てる橋渡しは、当初
//!   本モジュールのスコープ外としていたが、#2036 で `super::export_nn`
//!   （本クレート内部限定。facade 未接続）として実装済み（`docs/facade-
//!   onnx-export-exposure-decision.md` §15・§16）。
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

use super::export_ops;
use super::graph::{self, Graph, RawTensor};
use super::proto::{GraphProto, ModelProto, OperatorSetIdProto, TensorProto, ValueInfoProto};
use std::fmt;

// `export_ops`（イシュー #1773。内部 op -> `NodeProto` の意味論的マッピング）の
// 公開面を本モジュールから再エクスポートする（`onnx::export_ops::X` ではなく
// `onnx::export::X` として利用可能にする。呼び出し元は #1773 以降のテスト・
// 将来の codegen・facade 結線〈#1775 が判断済み・publish 承認待ちの段階 0。
// `docs/facade-onnx-export-exposure-decision.md`〉）。
pub use export_ops::{ConstantAttr, ExportNode, ExportOp, SUPPORTED_OP_TYPES, to_node_proto};

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
    /// `op_type` が `export_ops::SUPPORTED_OP_TYPES`（`interp.rs` 対応 23 op）に
    /// 含まれない、または `domain` が既定 opset（空文字列）以外
    /// （`export_ops::check_exportable`。イシュー #1773 の層 B）。
    UnsupportedOp {
        node_name: String,
        op_type: String,
        domain: String,
    },
    /// 必須入力の個数が op ごとの arity 範囲外
    /// （`export_ops::to_node_proto`。イシュー #1773）。
    InputArityMismatch {
        node_name: String,
        op_type: &'static str,
        expected: String,
        actual: usize,
    },
    /// 出力の個数が 1 以外（本モジュールが対応する全 op は単一出力。
    /// `interp.rs::require_single_output` と対称）。
    OutputArityMismatch {
        node_name: String,
        op_type: &'static str,
        expected: usize,
        actual: usize,
    },
    /// 必須入力（省略不可の位置）が空文字列
    /// （`interp.rs` の `node.input.get(N)` が `Some(name) if !name.is_empty()`
    /// で判定する慣習と対称。省略可入力〈末尾のみ〉は空文字列を許容する）。
    EmptyRequiredInput {
        node_name: String,
        op_type: &'static str,
        index: usize,
    },
    /// `onnx::export_nn::export_parts_from_layers`（イシュー #2036）が
    /// 空の層列（`layers.is_empty()`）を渡された。
    EmptyModel,
    /// `onnx::export_nn`（#2036）が `fandhe_ai_autodiff::nn::Module` の
    /// 層列を走査した際、`as_linear`／`as_relu` のいずれにも該当しない
    /// 層に遭遇した（fail-closed 拒否。§3.1「未対応層は型付き `Err`」）。
    /// `layer_kind` は判別可能な範囲（`as_conv2d` 等の既存ダウンキャスト
    /// フック 8 種）でのみ具体名を報告し、それ以外は `"unknown"`。
    UnsupportedLayer {
        index: usize,
        layer_kind: &'static str,
    },
    /// `onnx::export_nn`（#2036）が `Linear::weight`／`bias` の shape を
    /// 検証した際の不整合（weight が rank 2 でない・bias が rank 1 でない・
    /// `bias.len() != weight.shape()[1]`・shape 次元が `i64` へ収まらない等）。
    InvalidLayerParameter { index: usize, reason: String },
    /// `onnx::export_nn`（#2036）が構築するテンソル名（graph input／output・
    /// 中間テンソル名・initializer 名）が重複した。現行の命名規約
    /// （`layer{i}`／`layer{i}_out`／`{i}.weight`／`{i}.bias`）では構造上
    /// 発生し得ないが、設計要件として fail-closed に検査する（§3.1）。
    DuplicateTensorName { name: String },
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
            ExportError::UnsupportedOp {
                node_name,
                op_type,
                domain,
            } => write!(
                f,
                "export 未対応の op（node={node_name}）: op_type={op_type} domain={domain:?}"
            ),
            ExportError::InputArityMismatch {
                node_name,
                op_type,
                expected,
                actual,
            } => write!(
                f,
                "入力個数不整合（node={node_name}・op_type={op_type}）: 期待={expected} 実際={actual}"
            ),
            ExportError::OutputArityMismatch {
                node_name,
                op_type,
                expected,
                actual,
            } => write!(
                f,
                "出力個数不整合（node={node_name}・op_type={op_type}）: 期待={expected} 実際={actual}"
            ),
            ExportError::EmptyRequiredInput {
                node_name,
                op_type,
                index,
            } => write!(
                f,
                "必須入力が空（node={node_name}・op_type={op_type}）: index={index}"
            ),
            ExportError::EmptyModel => {
                write!(f, "export 対象の層列が空（`nn::Module` が 0 件）")
            }
            ExportError::UnsupportedLayer { index, layer_kind } => write!(
                f,
                "export 未対応の層（index={index}）: layer_kind={layer_kind}"
            ),
            ExportError::InvalidLayerParameter { index, reason } => {
                write!(f, "層パラメータが不正（index={index}）: {reason}")
            }
            ExportError::DuplicateTensorName { name } => {
                write!(f, "テンソル名の重複: {name}")
            }
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
/// パイプライン全体）・将来の codegen。組み立て自体（`graph.nodes` を
/// `GraphProto.node` へ詰める処理）は機械的な素通しのみで、op_type の意味論
/// には関与しないが、組み立てに先立ち `export_ops::check_exportable`
/// （イシュー #1773 の層 B）で `graph.nodes` が `interp.rs` 対応 23 op の
/// allowlist・既定 opset（`domain` が空文字列）に収まっているかを fail-closed
/// に検査する（無言 skip はしない。`security.md` A03）。
pub fn build_model_proto(
    graph: &Graph,
    options: &ExportOptions,
) -> Result<ModelProto, ExportError> {
    export_ops::check_exportable(graph)?;

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
        // 契約: 常に空（内部 `Graph` は sparse テンソルを保持しない設計のため。
        // イシュー #2079）。
        sparse_initializer: Vec::new(),
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
