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
//! ## ONNX export（イシュー #2018）
//!
//! [`OnnxModel::to_bytes`]／[`OnnxModel::to_path`] は import 済みモデルの
//! roundtrip export ラッパーである（`docs/facade-onnx-export-exposure-
//! decision.md` §4 案 B。承認事項は同 doc §10・#2018 承認コメント）。
//! `fandhe_ai_onnx_interop::onnx::export::build_model_proto`（allowlist
//! による fail-closed 検査込み）→ `onnx::proto::encode_model` への薄い
//! 委譲のみで、以下を doc として明記する:
//!
//! - `value_info` は常に空。本クレート内 roundtrip は bit 同一で保証する
//!   が、`onnx.checker` 等の外部ツールでの厳密な妥当性検証は保証しない
//!   （`export.rs` モジュール冒頭コメント）
//! - [`OnnxExportOptions`] の既定値（`ir_version=8`／`opset_version=17`）
//!   が書き出される。**import 時に元モデルの `opset_import`／
//!   `ir_version`／`producer_name`／グラフ名は [`fandhe_ai_onnx_interop::
//!   onnx::graph::Graph`] に保持されない**ため、export 結果は options の
//!   値（と内部固定の producer／graph 名）になる。元モデルの opset と
//!   合わせる責任は利用者側にある
//! - tensor は常に `raw_data`（リトルエンディアン）のみで書き出す。
//!   initializer は名前順で決定的に出力する（同一モデルの `to_bytes` は
//!   何度呼んでも同一バイト列）
//! - ホスト CPU 実行のみ・`BackendOps`／`Device` 非経由。`to_bytes`／
//!   `to_path` で export できるのは import 済みモデル（`OnnxModel`）の
//!   みで、学習済み `Sequential`／`nn` から直接 `OnnxModel` を構築する
//!   経路は [`OnnxModel::from_sequential`]（次節・#2037）を使う
//! - allowlist（`interp` 対応 22 op・既定 domain）外のノードを含む
//!   モデルは `from_bytes` では構築できても **export 時に**
//!   [`OnnxError::UnsupportedOp`] により fail-closed に拒否する（無言
//!   skip しない）
//!
//! ## `Sequential` からの export（イシュー #2037・親 #2034）
//!
//! [`OnnxModel::from_sequential`] は学習済み [`crate::compat::Sequential`]
//! から直接 [`OnnxModel`] を構築する（`torch.onnx.export` 相当の
//! ユーザーストーリー。`docs/facade-onnx-export-exposure-decision.md`
//! §17）。`fandhe_ai_onnx_interop::onnx::export_nn::graph_from_layers`
//! （#2036）への 1 段委譲のみで、構築した [`Graph`] を encode／decode
//! による正規化を挟まず直接 [`OnnxModel`] が保持する。以下を doc として
//! 明記する:
//!
//! - **対応層は `Linear`／`ReLU` の 2 種のみ**（Sigmoid・Tanh・Conv2d・
//!   LayerNorm 等は非対応）。1 つでも非対応層を含む場合は `Graph` を
//!   一切構築せず [`OnnxError::UnsupportedLayer`] で fail-closed に
//!   拒否する（部分的に構築されたモデルを返さない）
//! - graph input 名は常に `"input"`・output 名は常に `"output"`
//!   （最終層の出力）。initializer 名は `{i}.weight`／`{i}.bias`
//!   （`i` は層の位置。`Sequential::state_dict` と同じ命名規約）
//! - 空の `Sequential`（層 0 個）は [`OnnxError::InvalidModel`] で拒否
//!   される（`ExportError::EmptyModel` の写像）
//! - `value_info` は常に空（前節と同じ制約）
//! - ホスト CPU 実行のみ・`BackendOps`／`Device` 非経由（学習済み
//!   パラメータの値をそのままコピーするのみで算術を行わない）
//! - bit 完全一致契約: `from_sequential(&m).to_bytes(opts)` →
//!   `from_bytes` → `run` の出力は、`Linear→ReLU` 入力に NaN が現れず
//!   GEMM 出力に厳密な `±0.0` が現れない限り `m.predict(&x)` と bit
//!   完全一致する（`export_nn` モジュール doc「bit 一致契約の前提」
//!   参照）
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

use fandhe_ai_onnx_interop::onnx::export::{ExportError, ExportOptions, build_model_proto};
use fandhe_ai_onnx_interop::onnx::export_nn::graph_from_layers;
use fandhe_ai_onnx_interop::onnx::graph::{Graph, GraphError, build_graph};
use fandhe_ai_onnx_interop::onnx::interp::{InterpError, Value as InterpValue, run as interp_run};
use fandhe_ai_onnx_interop::onnx::proto::{decode_model, encode_model};
use fandhe_ai_tensor_core::f16;

use crate::Tensor;
use crate::compat::Sequential;

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

    /// 学習済み [`crate::compat::Sequential`] から [`OnnxModel`] を構築
    /// する（イシュー #2037。モジュール doc「`Sequential` からの
    /// export」節参照）。
    ///
    /// `fandhe_ai_onnx_interop::onnx::export_nn::graph_from_layers` への
    /// 1 段委譲のみ（薄いラッパー原則）。対応層は `Linear`／`ReLU` の
    /// 2 種のみで、それ以外（Sigmoid・Tanh・Conv2d 等）を 1 つでも
    /// 含む場合は [`Graph`] を一切構築せず
    /// [`OnnxError::UnsupportedLayer`] を返す（全層事前検証・
    /// fail-closed。`security.md` A08）。空の `Sequential` は
    /// [`OnnxError::InvalidModel`] で拒否される。
    pub fn from_sequential(model: &Sequential) -> Result<Self, OnnxError> {
        let graph = graph_from_layers(model.layers()).map_err(map_export_error)?;
        Ok(Self { graph })
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

    /// 保持しているグラフを `.onnx`（protobuf）バイト列へ書き出す
    /// （イシュー #2018。roundtrip export 限定。モジュール doc「ONNX
    /// export」節参照）。
    ///
    /// `fandhe_ai_onnx_interop::onnx::export::build_model_proto`（allowlist
    /// による fail-closed 検査を含む）→ `proto::encode_model` への委譲の
    /// み。allowlist 外 op を含む場合は [`OnnxError::UnsupportedOp`] を
    /// 返す（無言 skip しない）。
    pub fn to_bytes(&self, options: &OnnxExportOptions) -> Result<Vec<u8>, OnnxError> {
        let model =
            build_model_proto(&self.graph, &options.to_internal()).map_err(map_export_error)?;
        Ok(encode_model(&model))
    }

    /// [`OnnxModel::to_bytes`] の結果をファイルパスへ書き出す（`to_bytes`
    /// が成功してから `std::fs::write` する順序。export 失敗時にファイル
    /// を作成・切り詰めない）。パスをシェル展開・連結せずそのまま渡す
    /// （[`OnnxModel::from_path`] と対称）。既存ファイルは上書きされる。
    pub fn to_path(
        &self,
        path: impl AsRef<Path>,
        options: &OnnxExportOptions,
    ) -> Result<(), OnnxError> {
        let bytes = self.to_bytes(options)?;
        std::fs::write(path.as_ref(), bytes).map_err(OnnxError::Io)
    }
}

/// [`OnnxModel::to_bytes`]／[`OnnxModel::to_path`] の export オプション
/// （イシュー #2018・承認事項 1）。内部クレートの `ExportOptions`
/// （`producer_name`／`graph_name`／`opset_domain` を持つ）を承認済み 2
/// フィールドのみへ縮小した自己完結型（薄いラッパー原則）。
///
/// `#[non_exhaustive]`: 将来のフィールド追加を非破壊にするため
/// （`OnnxError` と同じ方針）。値の検証は行わない（薄いラッパー原則。
/// `ir_version`／`opset_version` はそのまま書き出される）。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxExportOptions {
    pub ir_version: i64,
    pub opset_version: i64,
}

impl Default for OnnxExportOptions {
    /// `fandhe_ai_onnx_interop::onnx::export::ExportOptions::default()` と
    /// 同じ既定値（`ir_version=8`・`opset_version=17`）。
    fn default() -> Self {
        OnnxExportOptions {
            ir_version: 8,
            opset_version: 17,
        }
    }
}

impl OnnxExportOptions {
    /// 内部クレートの `ExportOptions` へ変換する（`producer_name`／
    /// `graph_name`／`opset_domain` は内部既定値のまま。`ExportOptions`
    /// を facade 公開面へ出さないための private ヘルパ）。
    fn to_internal(&self) -> ExportOptions {
        ExportOptions {
            ir_version: self.ir_version,
            opset_version: self.opset_version,
            ..ExportOptions::default()
        }
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
    /// ファイル I/O 失敗（[`OnnxModel::from_path`] での読み込み失敗、または
    /// [`OnnxModel::to_path`] での書き込み失敗）。読み込み・書き込みを
    /// 区別する専用 variant は設けず（薄いラッパー原則。`std::io::Error`
    /// 自体は操作の別を保持しない）、[`fmt::Display`] 側で「I/O 失敗」と
    /// 中立に表現する。
    Io(std::io::Error),
    /// protobuf デコード失敗（壊れたバイト列等）。`prost::DecodeError` は
    /// `Display` 文字列のみを保持する（`prost` 型を公開面に出さない）。
    Decode { message: String },
    /// テンソルの `data_type` が本クレートの対応範囲外（`GraphError::
    /// UnknownDataType`。initializer decode 経由・`Constant` 属性テンソル
    /// decode 経由〈`InterpError::Graph(GraphError::UnknownDataType)`〉の
    /// 両方をこの variant へ写像する）。
    UnsupportedDataType { tensor_name: String, data_type: i32 },
    /// 未対応の `op_type`（`InterpError::UnsupportedOp`。import 実行時）、
    /// または export 時の allowlist 外 op（`ExportError::UnsupportedOp`。
    /// `op_type` が既定 domain 以外の場合は `"{domain}::{op_type}"`
    /// 形式で domain を含める。情報を落とさないための写像判断）。
    UnsupportedOp { op_type: String },
    /// `run` の呼び出し元がグラフ入力に対応する feed を渡さなかった
    /// （`InterpError::MissingFeed`）。
    MissingFeed { input: String },
    /// `run` に渡された feed 名がグラフ入力にも initializer にも属さない
    /// （`InterpError::UnknownFeed`）。
    UnknownFeed { name: String },
    /// [`OnnxModel::from_sequential`] が対応しない層（`Linear`／`ReLU`
    /// 以外）を含む場合の拒否（`ExportError::UnsupportedLayer`。イシュー
    /// #2037・承認事項 4）。`layer_kind` は `Module` trait の既存
    /// ダウンキャストフックで判別できる範囲のみ具体名を報告し、それ
    /// 以外は `"unknown"`（`export_nn` モジュール doc 参照）。
    UnsupportedLayer { index: usize, layer_kind: String },
    /// 上記以外のモデル構築時エラー（トポロジ矛盾・形状不正・名前重複
    /// 等。`GraphError` の `Display` 文字列を保持する）、または上記以外
    /// の export 時エラー（`ExportError` の `Display` 文字列を保持する。
    /// `check_exportable` 済みの `Graph` からは通常到達しないが
    /// `ExportError` は `#[non_exhaustive]` のため fallback として写像
    /// する）。
    InvalidModel { message: String },
    /// 上記以外の実行時エラー（型不一致・属性欠落・未対応 dtype の演算
    /// 等。`InterpError` の `Display` 文字列を保持する）。
    Execution { message: String },
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "ONNX ファイル I/O 失敗（読み込みまたは書き込み）: {e}"),
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
            OnnxError::UnsupportedLayer { index, layer_kind } => {
                write!(f, "未対応の層（index={index}）: {layer_kind}")
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

/// `ExportError` → `OnnxError` 写像（`OnnxModel::to_bytes`／`to_path`・
/// `OnnxModel::from_sequential` の export 時エラー経路）。承認済みの
/// 新規 variant は `UnsupportedLayer`（イシュー #2037・承認事項 4）の
/// 1 件のみ——`EmptyModel`／`InvalidLayerParameter`／
/// `DuplicateTensorName`（`from_sequential` 経由でのみ到達しうる）は
/// 既存 fallback（`InvalidModel { message }`）へ写像し、承認済み公開面
/// 〈#2018／#2037 承認事項〉の範囲に留める。
fn map_export_error(e: ExportError) -> OnnxError {
    match e {
        ExportError::UnsupportedOp {
            op_type, domain, ..
        } => {
            // 既定 domain（空文字列）以外は `"{domain}::{op_type}"` 形式で
            // domain 情報を落とさずに `op_type` へ畳み込む（`OnnxError`
            // へ新規 variant を追加しない設計上の制約下での情報保持）。
            let op_type = if domain.is_empty() {
                op_type
            } else {
                format!("{domain}::{op_type}")
            };
            OnnxError::UnsupportedOp { op_type }
        }
        ExportError::UnsupportedLayer { index, layer_kind } => OnnxError::UnsupportedLayer {
            index,
            layer_kind: layer_kind.to_string(),
        },
        other => OnnxError::InvalidModel {
            message: other.to_string(),
        },
    }
}
