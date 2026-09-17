//! ONNX import ラッパー（イシュー #2017・`docs/facade-onnx-import-
//! exposure-decision.md` §4 案 B の実装）。
//!
//! `fandhe_ai_onnx_interop`（内部クレート。crates.io 公開名
//! `fandhe-ai-onnx-interop`）の `onnx::proto::decode_model` →
//! `onnx::graph::build_graph` → `onnx::interp::run` を 1 つの薄い型
//! [`OnnxModel`] に束ねる。**推論専用・ホスト CPU 実行のみ**であり
//! `BackendOps`／`Device` を経由しない（GPU 実行にはならない）。
//! **autograd 未接続**（入出力は [`crate::Tensor`] であり `Var` ではない。
//! 勾配は取れない）。
//!
//! 数値契約は REQ-7 判定式（`abs_err/(|ref|+1e-6) <= 1e-3`。
//! `crates/onnx-interop/tests/onnx_poc_v2_6_match.rs` 系と同一）であり、
//! `.claude/rules/coding-rust.md` の REQ-2 統一複合判定（バックエンド間
//! 数値一致）とは別指標である（両者を混同しない）。
//!
//! 対応 op は 22 種（`fandhe_ai_onnx_interop::onnx::interp` 冒頭コメント
//! 参照）。未対応 `op_type` は無言 skip せず [`OnnxError::UnsupportedOp`]
//! で fail-closed に拒否する（no-silent-skip 契約。`.claude/rules/
//! security.md` A03）。`run` の `feeds` は ONNX の pre-IR-4 セマンティクス
//! どおり同名 initializer を上書きする。
//!
//! [`OnnxValue::F16`] は `half::f16` を素通しする。facade は `half` を
//! 再エクスポートしないため、`half::f16` を名指しして扱うには利用者側が
//! `half` クレートへ直接依存する必要がある（`TypedOps<half::f16>` と同じ
//! 扱い。`docs/backend-dtype-dispatch-design.md` §16 の同種注記参照）。
//!
//! ## 非信頼入力の扱い
//!
//! ONNX は非信頼な外部フォーマットである。本ラッパーは検証を迂回・複製
//! せず、`fandhe_ai_onnx_interop::onnx::graph::build_graph` の既存検査
//! （dims 非負・要素数 overflow 拒否・データ長／バイト長一致・名前
//! 重複／SSA／トポロジカル順検証）と `onnx::interp::run` の
//! no-silent-skip 契約をそのまま通す（迂回・複製しない）。`from_path` は
//! パスを `std::fs::read` へそのまま渡すのみでシェル展開・パス連結は
//! 行わない。入力総バイト数・要素数の明示上限は導入していない
//! （`build_graph` の長さ整合検査がバイト長を初期入力長で抑える。値の
//! 決定にユーザー承認が要るため本 issue のスコープ外。
//! `docs/facade-onnx-import-exposure-decision.md` §6.3 参照）。

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use fandhe_ai_onnx_interop::onnx::graph::{Graph, GraphError, build_graph};
use fandhe_ai_onnx_interop::onnx::interp::{InterpError, Value as InterpValue, run as interp_run};
use fandhe_ai_onnx_interop::onnx::proto::decode_model;
use fandhe_ai_tensor_core::f16;

use crate::Tensor;

/// 読み込み済み ONNX モデル（内部的にはトポロジカル順検証済みの
/// `Graph` を保持する。フィールドは private——`fandhe_ai_onnx_interop`
/// の内部型〈`NodeProto` 等〉を facade の公開シグネチャへ出さないため）。
#[derive(Debug)]
pub struct OnnxModel {
    graph: Graph,
}

impl OnnxModel {
    /// `.onnx` ファイルのバイト列からモデルを構築する。
    ///
    /// protobuf デコード（[`OnnxError::Decode`]）→ 内部グラフ構築
    /// （形状・トポロジ検証。該当する `OnnxError` variant）の順で検証する。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        let model = decode_model(bytes).map_err(|e| OnnxError::Decode {
            message: e.to_string(),
        })?;
        let graph = build_graph(&model).map_err(map_graph_error)?;
        Ok(Self { graph })
    }

    /// ファイルパスから `.onnx` モデルを読み込む（`std::fs::read` →
    /// [`OnnxModel::from_bytes`]。パスをシェル展開・連結せずそのまま
    /// 渡す）。
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path.as_ref()).map_err(OnnxError::Io)?;
        Self::from_bytes(&bytes)
    }

    /// グラフを実行する。`feeds` はグラフ入力名 → 値の対応（未対応の
    /// initializer 上書きを含む pre-IR-4 セマンティクス）。
    ///
    /// 推論専用・ホスト CPU 実行のみ（`BackendOps`／`Device` 非経由）・
    /// autograd 未接続（戻り値は [`crate::Tensor`] であり `Var` ではない）。
    pub fn run(
        &self,
        feeds: HashMap<String, OnnxValue>,
    ) -> Result<HashMap<String, OnnxValue>, OnnxError> {
        let interp_feeds: HashMap<String, InterpValue> = feeds
            .into_iter()
            .map(|(k, v)| (k, onnx_value_to_interp(v)))
            .collect();
        let outputs = interp_run(&self.graph, interp_feeds).map_err(map_interp_error)?;
        Ok(outputs
            .into_iter()
            .map(|(k, v)| (k, interp_value_to_onnx(v)))
            .collect())
    }
}

/// [`OnnxModel::run`] が受け付ける／返す実行時値。ONNX の
/// `TensorProto.data_type` のうち本クレートが対応する 4 種類に対応する
/// （`fandhe_ai_onnx_interop::onnx::interp::Value` と同じ集合。
/// 内部クレートの `Value` 自体は facade の公開面に出さない）。
#[derive(Clone, Debug)]
pub enum OnnxValue {
    F32(Tensor<f32>),
    I64(Tensor<i64>),
    Bool(Tensor<bool>),
    F16(Tensor<f16>),
}

/// `OnnxValue` → 内部 `interp::Value` への変換（move。コピー・再計算を
/// 挟まないため出力は内部クレート直接呼び出しと bit 一致する）。`pub`
/// な `From` impl にはしない——内部クレートの `interp::Value` が facade の
/// 公開 trait impl として露出するのを避けるため（private 関数に留める）。
fn onnx_value_to_interp(v: OnnxValue) -> InterpValue {
    match v {
        OnnxValue::F32(t) => InterpValue::F32(t),
        OnnxValue::I64(t) => InterpValue::I64(t),
        OnnxValue::Bool(t) => InterpValue::Bool(t),
        OnnxValue::F16(t) => InterpValue::F16(t),
    }
}

/// `interp::Value` → `OnnxValue` への逆変換（同じく move）。
fn interp_value_to_onnx(v: InterpValue) -> OnnxValue {
    match v {
        InterpValue::F32(t) => OnnxValue::F32(t),
        InterpValue::I64(t) => OnnxValue::I64(t),
        InterpValue::Bool(t) => OnnxValue::Bool(t),
        InterpValue::F16(t) => OnnxValue::F16(t),
    }
}

/// [`OnnxModel`] の失敗を表す型付きエラー。`#[non_exhaustive]`:
/// 内部クレート（`GraphError`／`InterpError`）も `#[non_exhaustive]` で
/// あり、対応 op・対応 dtype の拡張に伴う variant 追加に備える。
///
/// 設計上の判断（`docs/facade-onnx-import-exposure-decision.md` §4.2）:
/// 内部クレートのエラー enum（`GraphError`／`InterpError`）をそのまま
/// ペイロードに持たせず、facade だけで名指し・`match` できる自己完結
/// 型として定義する。承認済み公開面は `OnnxModel`／`OnnxValue`／
/// `OnnxError` の 3 型のみであり、内部エラー型をペイロードに含めると
/// 利用者がそれらを名指しできず「型付き `Err` を fail-closed に拒否
/// する」という受け入れ条件を facade 単独では満たせないため。
#[non_exhaustive]
#[derive(Debug)]
pub enum OnnxError {
    /// `from_path` でのファイル読み込み失敗。
    Io(std::io::Error),
    /// protobuf デコード失敗（壊れたバイト列等）。`prost::DecodeError` は
    /// `Display` 文字列のみを保持する（`prost` 型を公開面に出さない）。
    Decode { message: String },
    /// テンソルの `data_type` が本クレートの対応範囲外（`GraphError::
    /// UnknownDataType`。initializer decode 経由・`Constant` 属性テンソル
    /// decode 経由〈`InterpError::Graph(GraphError::UnknownDataType)`〉の
    /// 両方をこの variant へ写像する）。
    UnsupportedDataType { tensor_name: String, data_type: i32 },
    /// 未対応の `op_type`（`InterpError::UnsupportedOp`）。
    UnsupportedOp { op_type: String },
    /// `run` の呼び出し元がグラフ入力に対応する feed を渡さなかった
    /// （`InterpError::MissingFeed`）。
    MissingFeed { input: String },
    /// `run` に渡された feed 名がグラフ入力にも initializer にも属さない
    /// （`InterpError::UnknownFeed`）。
    UnknownFeed { name: String },
    /// 上記以外のモデル構築時エラー（トポロジ矛盾・形状不正・名前重複
    /// 等。`GraphError` の `Display` 文字列を保持する）。
    InvalidModel { message: String },
    /// 上記以外の実行時エラー（型不一致・属性欠落・未対応 dtype の演算
    /// 等。`InterpError` の `Display` 文字列を保持する）。
    Execution { message: String },
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "ONNX ファイル読み込み失敗: {e}"),
            OnnxError::Decode { message } => write!(f, "ONNX protobuf デコード失敗: {message}"),
            OnnxError::UnsupportedDataType {
                tensor_name,
                data_type,
            } => write!(
                f,
                "未対応の ONNX data_type（tensor={tensor_name}）: {data_type}"
            ),
            OnnxError::UnsupportedOp { op_type } => write!(f, "未対応の ONNX op_type: {op_type}"),
            OnnxError::MissingFeed { input } => {
                write!(f, "グラフ入力 '{input}' に対応する feed がありません")
            }
            OnnxError::UnknownFeed { name } => {
                write!(
                    f,
                    "feed '{name}' はグラフ入力にも initializer にも属しません"
                )
            }
            OnnxError::InvalidModel { message } => write!(f, "不正な ONNX モデル: {message}"),
            OnnxError::Execution { message } => write!(f, "ONNX 実行時エラー: {message}"),
        }
    }
}

impl std::error::Error for OnnxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// `GraphError` → `OnnxError` 写像（モデル構築時のエラー経路）。
fn map_graph_error(e: GraphError) -> OnnxError {
    match e {
        GraphError::UnknownDataType {
            tensor_name,
            data_type,
        } => OnnxError::UnsupportedDataType {
            tensor_name,
            data_type,
        },
        other => OnnxError::InvalidModel {
            message: other.to_string(),
        },
    }
}

/// `InterpError` → `OnnxError` 写像（`run` 実行時のエラー経路）。
fn map_interp_error(e: InterpError) -> OnnxError {
    match e {
        InterpError::UnsupportedOp(op_type) => OnnxError::UnsupportedOp { op_type },
        InterpError::MissingFeed { input } => OnnxError::MissingFeed { input },
        InterpError::UnknownFeed { name } => OnnxError::UnknownFeed { name },
        // `Constant` 属性テンソルの decode 失敗（`GraphError` 由来）も
        // モデル構築時と同じ写像規則（未対応 dtype は名指し・それ以外は
        // InvalidModel）を適用する。
        InterpError::Graph(graph_err) => map_graph_error(graph_err),
        other => OnnxError::Execution {
            message: other.to_string(),
        },
    }
}
