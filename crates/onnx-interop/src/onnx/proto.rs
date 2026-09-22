//! ONNX protobuf メッセージの手書き `prost::Message` 実装（部分実装）。
//!
//! ## 生成方法についての設計判断（REQ-7・deps-policy.md・PoC-v2-6）
//!
//! `prost-build`（`protoc` へのビルド時依存）は使わず、TASK-7.2a（本モジュール）が
//! 必要とする 7 メッセージ（`ModelProto` / `GraphProto` / `NodeProto` /
//! `AttributeProto` / `TensorProto` / `ValueInfoProto` / `OperatorSetIdProto`）のみを
//! フィールド番号を一致させた `#[derive(prost::Message)]` 構造体として手書きする。
//! `protoc` 非依存の手書き derive は OWASP A06（サプライチェーン）の観点でもむしろ
//! 縮小になる（PoC-v2-6 advisor レビュー由来の判断）。
//!
//! `ModelProto.opset_import`（tag=8）・`GraphProto.value_info`（tag=13）はイシュー
//! #1772（`onnx::export`。内部グラフ表現 -> `GraphProto`／`ModelProto` への降下）で
//! 追加した。フィールド番号の出典は本モジュール既存フィールドと同じ `onnx==1.22.0`
//! 同梱の `onnx/onnx.proto`（`OperatorSetIdProto{ domain: string tag=1, version:
//! int64 tag=2 }`）。`value_info` は decode 方向では常に空 `Vec` のまま扱われ
//! （`graph::build_graph` は `value_info` を読まない）、export 方向（`onnx::export`）が
//! 書き込む契約は同モジュールのドキュメンテーションコメントを参照。
//!
//! `prost::Message::decode` は構造体に未宣言のフィールド番号を protobuf のワイヤ
//! フォーマット仕様どおり自動的にスキップするため、`TypeProto`（`ValueInfoProto.type`）
//! 等、本クレートが現時点で使わない再帰的メッセージは意図的に定義しない。`AttributeProto`
//! の `g` / `graphs`（If/Loop/Scan のサブグラフ属性）も同様に #78 のスコープ外として
//! 意図的に未定義とする。#78（インタープリタ基盤）・#79（8 オペ実装）で必要になった
//! 時点で拡張する。**例外**: `GraphProto.sparse_initializer`（tag=15。
//! `SparseTensorProto`）は中身を一切使わないが、無言スキップに委ねると
//! sparse initializer だけが「最初から存在しない」ものとして扱われ
//! no-silent-skip 契約に反するため、**存在検出のためだけに意図的に宣言する**
//! （`graph::build_graph` が非空を fail-closed 拒否する。イシュー #2079）。
//! この例外により本モジュールが宣言するメッセージ数は 7 から 9 になった
//! （`SparseTensorProto` に加え、下記の `SparseTensorValueName` probe 型）。
//!
//! **メモリ増幅対策（イシュー #2079 codex-review 指摘。2026-09-22 是正）**:
//! `sparse_initializer` を無条件に `Vec<SparseTensorProto>` として decode 対象に
//! 含めると、非信頼入力の `values`/`indices`（`TensorProto`）の `raw_data`・
//! packed `float_data`/`int64_data` が `build_graph` の非空検査に到達する前に
//! 構造体へ完全展開され、巨大な sparse payload によるメモリ枯渇（DoS。
//! security.md A03・AGENTS.md「外部フォーマットのパース検証（P0）」）を招く。
//! この対策は 2 層で構成する:
//! 1. **主対策（`decode_model` 経由の非信頼入力）**: [`decode_model`] は
//!    `ModelProto::decode` を呼ぶ**前**に、`prost::encoding` の公開
//!    プリミティブ（`decode_key`/`decode_varint`/`skip_field`）だけを使う
//!    bounded なワイヤスキャン（`prescan_sparse_initializer`（非公開関数））で
//!    `sparse_initializer`（tag=15）の存在有無だけを検出し、検出時は
//!    `ModelProto` を一切構築せず [`DecodeModelError::
//!    SparseInitializerNotSupported`] で fail-closed に拒否する。**この
//!    「検出済み」判定は、同じ走査中に検出**後**（同一 graph 内・別 graph
//!    出現・`ModelProto` 直下のいずれでも）不正な形式に遭遇した場合でも
//!    確定して優先される**（2026-09-22 codex-review P0 是正）: 一度でも
//!    `sparse_initializer` を検出したら `ModelProto::decode` へは一切
//!    フォールバックしない。フォールバックすると、`ModelProto::decode` が
//!    同じ不正入力を独立に検証して最終的に拒否するとしても、拒否が確定
//!    する**前**に構造体側（下記層 2）の `SparseTensorValueName.name` が
//!    上限なく構造体へ展開されてしまい、巨大な `values.name` を持つ
//!    sparse_initializer の直後に不正フィールドを置くだけで層 2 の
//!    縮小策を迂回した巨大確保（DoS）を誘発できたため（詳細は
//!    `prescan_sparse_initializer` のドキュメンテーションコメント）。
//!    走査未検出のまま不正な形式に遭遇した場合（「判定不能」）は従来どおり
//!    `ModelProto::decode` へフォールバックする（同じ地点で同じ不正入力を
//!    独立に検証し拒否するため安全）。
//! 2. **構造体側の縮小（`ModelProto::decode` を直接呼ぶ経路への多層防御）**:
//!    `SparseTensorProto.values` は `TensorProto` 全体ではなく `name`
//!    （tag=8）のみを宣言した [`SparseTensorValueName`] として decode する。
//!    `prost::Message::decode` は構造体に未宣言のフィールド番号を自動的に
//!    スキップする（本コメント冒頭段落）ため、`raw_data`/`float_data`/
//!    `int64_data`/`dims`/`data_type` は Vec へ展開されずバイト列として
//!    読み飛ばされるのみとなる。`indices`（COO インデックス）は
//!    `build_graph` がそもそも参照しないためフィールド自体を宣言せず、
//!    tag=2 を丸ごと未宣言フィールドのスキップに委ねる。
//!
//! フィールド番号の出典: `onnx==1.22.0` 同梱の `onnx/onnx.proto` を実際に読み、
//! 該当 6 メッセージのフィールド番号を転記した（PoC-v2-6 で実ファイル
//! `model.onnx` / `slice_repro.onnx` / `transformer.onnx` を parse してノード数・
//! op_type 列が期待どおりか検証済み。`docs/spec/03-poc/poc-v2-6-interop/evidence/`
//! 配下の各ログ参照）。

use std::fmt;

use prost::Message;
use prost::bytes::Buf;
use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};

/// ONNX モデル全体（`.onnx` ファイルのトップレベルメッセージ）。
///
/// `graph.rs::build_graph` の入力。`decode` の呼び出し元は `onnx-interop` 利用者
/// （#78 のインタープリタ・将来の codegen）。
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    /// このモデルが要求する opset（ドメイン・バージョンの組。複数ドメインを持つ
    /// モデルもありうるため `repeated`）。decode 方向では現状どの呼び出し元も
    /// 参照していない（#78 のインタープリタは opset を見ずに `op_type` 名で直接
    /// ディスパッチする）が、export 方向（`onnx::export::build_model_proto`。
    /// イシュー #1772）が書き込む。
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

/// モデルが要求する opset の 1 エントリ（ドメイン・バージョンの組）。
/// `ModelProto.opset_import`（tag=8）の要素型。
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    /// opset のドメイン。既定 opset は空文字列（`onnx==1.22.0` の慣習。
    /// `ExportOptions::default()` も同じ既定値を使う）。
    #[prost(string, tag = "1")]
    pub domain: String,
    /// opset バージョン番号。
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// 計算グラフ本体。ONNX 仕様は `node` がトポロジカル順であることを要求するが、
/// 本クレートはこれを信頼せず `graph::build_graph` で自前検証する。
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    /// 中間テンソルの型・形状ヒント（ONNX 仕様上は任意）。`graph::build_graph`
    /// は読まない（`Graph` は中間テンソルの型／形状情報を保持しない設計）。
    /// export 方向（`onnx::export`。イシュー #1772）は常に空のまま書き出す
    /// （理由は `onnx::export` モジュールのドキュメンテーションコメント参照）。
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
    /// sparse 形式の initializer。本クレートは sparse テンソルを非対応
    /// （`docs/tensor-core-sparse-complex-decision.md`）とし、dense
    /// initializer と異なり中身は一切解釈しない。decode 方向: 非空であれば
    /// `graph::build_graph` が `GraphError::SparseInitializerNotSupported` で
    /// fail-closed に拒否する（存在検出のみの意図的な宣言。他の未使用
    /// フィールドのように無言スキップに委ねると sparse initializer だけが
    /// 「最初から存在しない」ものとして扱われ no-silent-skip 契約〈A03／A08〉
    /// に反するため。イシュー #2079）。export 方向（`onnx::export`）は常に
    /// 空のまま書き出す（内部 `Graph` は sparse を保持しない設計のため）。
    #[prost(message, repeated, tag = "15")]
    pub sparse_initializer: Vec<SparseTensorProto>,
}

/// sparse 形式テンソル（COO 形式: `values`・`indices`・`dims`）。
///
/// 本クレートでは **検出専用**（`GraphProto.sparse_initializer` の非空検査の
/// ためだけに宣言する）。`values` の中身（`name` 以外）・`indices`・`dims` は
/// 一切 decode・解釈しない（sparse テンソルの実装自体はスコープ外。
/// `docs/tensor-core-sparse-complex-decision.md` REQ-9）。フィールド番号の
/// 出典は `onnx==1.22.0` 同梱の `onnx/onnx.proto`（`SparseTensorProto{
/// values: TensorProto tag=1, indices: TensorProto tag=2, dims: repeated
/// int64 tag=3 }`）。イシュー #2079。
///
/// **`values` の型が `TensorProto` ではなく [`SparseTensorValueName`] な
/// 理由（メモリ増幅対策。2026-09-22 codex-review 是正）**: `TensorProto`
/// 全体を decode すると非信頼入力の `raw_data`/`float_data`/`int64_data`
/// まで構造体へ展開されてしまう（本モジュール冒頭コメント参照）ため、
/// エラー診断に使う `name`（tag=8）のみを宣言した軽量型で decode する。
/// `indices`（tag=2）・`dims`（tag=3）は `build_graph` が読まないため、
/// フィールド自体を宣言せず prost の自動スキップに委ねる。
#[derive(Clone, PartialEq, Message)]
pub struct SparseTensorProto {
    /// 非ゼロ要素の値。`graph::build_graph` は `values.name` をエラー診断用
    /// のテンソル名として使うのみ（[`SparseTensorValueName`] のコメント
    /// 参照）。
    #[prost(message, optional, tag = "1")]
    pub values: Option<SparseTensorValueName>,
}

/// `SparseTensorProto.values`（本来は `TensorProto`）から `name`（tag=8）
/// のみを取り出す軽量 probe 型。`raw_data`/`float_data`/`int64_data`/
/// `dims`/`data_type` を意図的に未宣言のままにし、`prost::Message::decode`
/// の自動フィールドスキップ（本モジュール冒頭コメント）に委ねることで、
/// 非信頼入力の巨大な sparse payload を構造体へ展開せずに済ませる
/// （イシュー #2079 codex-review 指摘。メモリ枯渇 DoS 対策の詳細は
/// `SparseTensorProto` のドキュメンテーションコメント参照）。
#[derive(Clone, PartialEq, Message)]
pub struct SparseTensorValueName {
    #[prost(string, tag = "8")]
    pub name: String,
}

/// 演算グラフの 1 ノード（1 オペレータ呼び出し）。
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// ノード属性（オペレータのパラメータ）。本クレートが使うフィールドのみ部分定義。
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes, tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

/// テンソル（initializer／定数）の protobuf 表現。`graph::decode_tensor` が
/// `RawTensor` へ復号する前段。
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes, tag = "9")]
    pub raw_data: Vec<u8>,
}

/// グラフ入出力の名前（型情報 `TypeProto` は未使用のため意図的に定義しない）。
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
}

/// onnx.proto3 `TensorProto.DataType`（本クレートが扱う値のみ抜粋）。
/// 未対応の値は `graph::decode_tensor` が `GraphError::UnknownDataType` で拒否する
/// （無言 skip は A03〈インジェクション／不正入力〉の観点で禁止。security.md）。
/// `BOOL`／`FLOAT16` はイシュー #274 で追加（`Cast` の対応範囲拡張に伴う initializer
/// decode 側の対応）。
pub mod data_type {
    pub const FLOAT: i32 = 1;
    pub const INT64: i32 = 7;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
}

/// onnx.proto3 `AttributeProto.AttributeType`（本クレートが書き出す値のみ抜粋）。
/// 出典は本モジュール冒頭コメントと同じ `onnx==1.22.0` 同梱の `onnx/onnx.proto`。
/// `onnx::export`（イシュー #1773）が `AttributeProto.r#type` を設定する際に使う。
/// 既存テスト（`tests/onnx_decode.rs`・`tests/onnx_interp.rs`）はこの追加以前から
/// 同値のローカル定数を使っており、ここでの追加はそれらの重複定義を置き換える
/// ものではない（`r#type` を export が正しく設定することを新規テストで固定する）。
pub mod attribute_type {
    pub const FLOAT: i32 = 1;
    pub const INT: i32 = 2;
    pub const STRING: i32 = 3;
    pub const TENSOR: i32 = 4;
    pub const FLOATS: i32 = 6;
    pub const INTS: i32 = 7;
}

/// `decode_model` のエラー型（イシュー #2079 codex-review 是正）。
///
/// `prost::DecodeError`（壊れたバイト列）に加え、`ModelProto::decode` を
/// 呼ぶ**前**の bounded 事前走査（`prescan_sparse_initializer`（非公開関数））が
/// `GraphProto.sparse_initializer` の存在を検出した専用分岐を持つ。
/// この分岐は `graph::build_graph` の `GraphError::
/// SparseInitializerNotSupported` と同一の診断情報（`tensor_name`・
/// `count`）を運び、facade（`crates/facade/src/interop/onnx.rs`）は
/// これを `OnnxError::SparseInitializerNotSupported` へそのまま写像する。
#[derive(Debug)]
pub enum DecodeModelError {
    /// `ModelProto::decode` 自体の失敗（壊れたバイト列・不正なワイヤ
    /// フォーマット等）。
    Wire(prost::DecodeError),
    /// bounded 事前走査で `sparse_initializer`（tag=15）の非空を検出し、
    /// `ModelProto::decode` を呼ぶ前に fail-closed で拒否した。
    SparseInitializerNotSupported { tensor_name: String, count: usize },
}

impl fmt::Display for DecodeModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeModelError::Wire(e) => write!(f, "{e}"),
            DecodeModelError::SparseInitializerNotSupported { tensor_name, count } => write!(
                f,
                "sparse_initializer は非対応（tensor={tensor_name}・count={count}）: sparse テンソルは対象外のため fail-closed に拒否"
            ),
        }
    }
}

impl std::error::Error for DecodeModelError {}

/// `ModelProto::decode` への薄い委譲入口（イシュー #2017）。
///
/// facade（`crates/facade`）は crates.io 公開クレートの依存面を絞るため
/// `prost` へ直接依存せず、呼び出し側が `prost::Message` を `use` しなくても
/// `.onnx` バイト列を復号できるこの関数だけを経由する。検証（dims 非負・
/// 要素数整合・トポロジカル順等）は本関数の責務外で、後続の
/// `graph::build_graph` が担う（本モジュール冒頭コメント・`graph.rs` 冒頭
/// コメント参照）。
///
/// **`sparse_initializer` の早期 fail-closed 拒否（イシュー #2079
/// codex-review 是正）**: `ModelProto::decode` を呼ぶ前に
/// `prescan_sparse_initializer`（非公開関数） で `GraphProto.sparse_initializer`
/// （tag=15）の存在だけを bounded に検出する。検出時は `ModelProto` を
/// 一切構築せず [`DecodeModelError::SparseInitializerNotSupported`] を
/// 返す（本モジュール冒頭コメント「メモリ増幅対策」節参照）。**検出は
/// 走査中の後続の不正な形式より優先される**（`PrescanOutcome::Found` は
/// 同じ走査が後で `PrescanOutcome::Indeterminate` になっても上書きされない。
/// `prescan_sparse_initializer` のドキュメンテーションコメント参照。
/// 2026-09-22 codex-review P0 是正）。
pub fn decode_model(bytes: &[u8]) -> Result<ModelProto, DecodeModelError> {
    if let PrescanOutcome::Found { tensor_name, count } = prescan_sparse_initializer(bytes) {
        return Err(DecodeModelError::SparseInitializerNotSupported { tensor_name, count });
    }
    ModelProto::decode(bytes).map_err(DecodeModelError::Wire)
}

/// [`prescan_sparse_initializer`] の三値判定結果（2026-09-22 codex-review
/// P0 是正。旧実装は `Option<(String, usize)>` で「検出」と「未検出／
/// 判定不能」の 2 値しか表現できず、「検出**後**に走査が不正な形式へ
/// 遭遇した」場合に検出結果ごと `None`（判定不能）へ握りつぶされていた）。
enum PrescanOutcome {
    /// 走査全体が正常に完了し、`sparse_initializer` は 1 件も出現しなかった。
    NotFound,
    /// `sparse_initializer`（tag=15）を 1 件以上検出した。走査がこの後
    /// 不正な形式に遭遇したかどうかに関わらず、検出という事実は確定する
    /// （`count` は少なくともこの件数が存在したことの下界。走査が
    /// 途中で打ち切られた場合、実際の総数はこれ以上かもしれない）。
    Found { tensor_name: String, count: usize },
    /// `sparse_initializer` を 1 件も検出しないまま、走査自体が不正な形式
    /// （バッファ終端超過・未対応 wire type 等）で継続不能になった。
    /// 「なし」と断定できないため `decode_model` は `ModelProto::decode`
    /// へフォールバックする（本モジュール冒頭コメント「メモリ増幅対策」
    /// 節参照）。
    Indeterminate,
}

/// `ModelProto::decode` より前に `GraphProto.sparse_initializer`（tag=15）
/// の存在だけを bounded に検出する事前走査（イシュー #2079 codex-review
/// 是正。主対策。本モジュール冒頭コメント「メモリ増幅対策」節参照）。
///
/// `prost::encoding` の公開プリミティブ（`decode_key`/`decode_varint`/
/// `skip_field`）だけを使い、`ModelProto` トップレベル -> 各 `graph`
/// （tag=7）出現 -> その直下の `sparse_initializer`（tag=15）出現、という
/// 固定 3 階層だけを歩く（ネストしたメッセージへは再帰しない。深さが
/// 固定のためスタック消費も定数）。protobuf は同一の非 repeated
/// メッセージフィールドが複数回出現した場合にマージする仕様のため、
/// `graph`（tag=7）の全出現を走査して `sparse_initializer` の出現数を
/// 合算する（1 回の出現だけを見ると分割された悪意ある入力を見逃す）。
///
/// 各 length-delimited フィールドの長さは残りバッファ長と必ず照合する
/// （超過は即座に不正入力と判定）。
///
/// **検出後に走査自体が不正な形式へ遭遇した場合の扱い（2026-09-22
/// codex-review P0 是正）**: `count`（この時点までに検出した
/// `sparse_initializer` の件数）・`first_name`（最初の要素の診断名）は
/// 呼び出し元の `&mut` 変数へ都度書き込まれるため、走査ループが
/// 不正な形式（バッファ終端超過・未対応 wire type 等）に遭遇して早期
/// 終了した後も値は保持されている。この関数は走査終了後に必ず
/// `count > 0` を最優先でチェックし、`count > 0` であれば走査の成否に
/// 関わらず [`PrescanOutcome::Found`] を返す（走査が正常終了したか
/// 不正な形式で打ち切られたかは区別しない。**一度でも tag=15 を検出した
/// 走査は `ModelProto::decode` へ絶対にフォールバックしない**という
/// 不変条件を守るため）。旧実装は走査失敗を `?` でそのまま関数全体の
/// 早期リターンへ伝播させており、検出済みの `count`／`first_name` を
/// 呼び出し元へ返さずに握りつぶしていた。これにより「sparse_initializer
/// を検出したが直後の走査で不正な形式に遭遇した」入力が「sparse
/// なし」と同一視されて `decode_model` が `ModelProto::decode` へ
/// フォールバックし、巨大な `values.name`（`SparseTensorValueName.name`
/// は構造体側では長さ上限なしで decode される。本モジュール冒頭コメント
/// 「メモリ増幅対策」節の層 2 参照）を持つ sparse_initializer の直後に
/// 不正フィールドを配置するだけで層 2 の縮小策を迂回でき、拒否が確定する
/// **前**に巨大確保が発生していた（codex-review 指摘）。
/// `count == 0` のまま走査が不正な形式に遭遇した場合のみ
/// [`PrescanOutcome::Indeterminate`] を返し、`decode_model` は従来どおり
/// `ModelProto::decode` が同じ不正入力を独立に検証し拒否することに委ねる
/// （`decode_model` 全体としては安全側に倒れる）。
///
/// 検出時（`count > 0`）の `tensor_name` は、最初の要素の `values.name`
/// （tag=1 の中の tag=8）だけを同じく bounded に読み取ったもの（`TensorProto`
/// の他フィールド（`raw_data` 等）へは一切踏み込まない。`values` 自体が
/// 存在しない・`name` が無い場合は空文字列。`graph::build_graph` の既存
/// 挙動と同じ fallback）。名前は [`SPARSE_TENSOR_NAME_DIAG_CAP`]
/// （`graph::build_graph` と共有する上限）バイトで切り詰める
/// （[`cap_sparse_tensor_diag_name`] 経由。診断契約の統一。codex-review
/// P2 是正・2026-09-22）。値が同一メッセージ内に複数回出現した場合は
/// protobuf の後勝ちマージ規則に従い最後の出現を採用する
/// （`scan_sparse_tensor_bytes_for_values_name`／
/// `scan_tensor_bytes_for_name` のドキュメンテーションコメント参照）。
/// **「最初の要素」判定は「名前を読み取れた
/// 最初の要素」ではなく「出現順で最初の要素」**（`graph::build_graph` の
/// `g.sparse_initializer[0]` と同じ意味）であるため、最初の要素に
/// `values.name` が無い場合は空文字列のまま確定し、2 番目以降の要素へは
/// 名前を探しに行かない（探索を続けると `build_graph` 側の
/// `tensor_name` と食い違いうる。Cursor Bugbot 指摘・#2079 是正）。
fn prescan_sparse_initializer(bytes: &[u8]) -> PrescanOutcome {
    let mut buf: &[u8] = bytes;
    let mut count = 0usize;
    let mut first_name: Option<String> = None;
    let mut first_recorded = false;
    // 走査ループ全体を `Option<()>` のクロージャへ閉じ込め、`?` による
    // 早期離脱を「関数全体の早期リターン」ではなく「このクロージャの
    // 早期リターン」に限定する。こうすることで、離脱後も `count`・
    // `first_name`（いずれも `&mut` 経由でクロージャ外の変数を直接
    // 更新している）の値がそのまま呼び出し元スコープに残る。
    let scan_completed = (|| -> Option<()> {
        while buf.has_remaining() {
            let (tag, wire_type) = decode_key(&mut buf).ok()?;
            match wire_type {
                WireType::LengthDelimited => {
                    let field_bytes = take_length_delimited(&mut buf)?;
                    if tag == 7 {
                        scan_graph_bytes_for_sparse_initializer(
                            field_bytes,
                            &mut count,
                            &mut first_name,
                            &mut first_recorded,
                        )?;
                    }
                }
                _ => skip_field(wire_type, tag, &mut buf, DecodeContext::default()).ok()?,
            }
        }
        Some(())
    })()
    .is_some();

    if count > 0 {
        // 検出は走査の成否より優先する（上記ドキュメンテーションコメント
        // 「検出後に走査自体が不正な形式へ遭遇した場合の扱い」参照）。
        PrescanOutcome::Found {
            tensor_name: first_name.unwrap_or_default(),
            count,
        }
    } else if scan_completed {
        PrescanOutcome::NotFound
    } else {
        PrescanOutcome::Indeterminate
    }
}

/// `prescan_sparse_initializer` から呼ばれる。`GraphProto` 直下（ネストした
/// メッセージへは再帰しない）を走査し、`sparse_initializer`（tag=15）の
/// 出現ごとに `count` を加算し、出現順で最初の 1 件からのみ `values.name`
/// を `first_name` へ格納する。`first_recorded`（`first_name` とは別の
/// bool）で「最初の要素を既に処理したか」を追跡する
/// （`first_name.is_none()` だけで判定すると、最初の要素に名前が
/// 無かった場合に「未処理」と誤認して 2 番目以降の要素まで探しに行って
/// しまい、`graph::build_graph` の `g.sparse_initializer[0]` 基準の
/// `tensor_name`〈最初の要素が無名なら空文字列〉と食い違う。Cursor
/// Bugbot 指摘・#2079 是正）。
fn scan_graph_bytes_for_sparse_initializer(
    bytes: &[u8],
    count: &mut usize,
    first_name: &mut Option<String>,
    first_recorded: &mut bool,
) -> Option<()> {
    let mut buf: &[u8] = bytes;
    while buf.has_remaining() {
        let (tag, wire_type) = decode_key(&mut buf).ok()?;
        match wire_type {
            WireType::LengthDelimited => {
                let field_bytes = take_length_delimited(&mut buf)?;
                if tag == 15 {
                    *count += 1;
                    if !*first_recorded {
                        *first_recorded = true;
                        *first_name = scan_sparse_tensor_bytes_for_values_name(field_bytes);
                    }
                }
            }
            _ => skip_field(wire_type, tag, &mut buf, DecodeContext::default()).ok()?,
        }
    }
    Some(())
}

/// `SparseTensorProto` 直下を走査し `values`（tag=1）の中身を取り出して
/// `scan_tensor_bytes_for_name` へ渡す。`indices`（tag=2）・`dims`（tag=3）
/// には踏み込まない（`SparseTensorValueName` のコメント参照）。
///
/// **最後の出現を採用する理由（codex-review P2 是正。2026-09-22）**:
/// protobuf のワイヤフォーマット仕様上、`values`（singular message field。
/// tag=1）が同一 `SparseTensorProto` 内に複数回出現した場合は「後勝ち」
/// でマージされる（`prost::Message::merge` の singular message field
/// 実装は出現ごとに既存値へ再帰マージし、内部の singular scalar field
/// は最後に decode した値で上書きされる）。最初の出現だけを見て即座に
/// `return` すると、`ModelProto::decode` を直接呼ぶ経路（`build_graph`
/// が `SparseTensorProto::decode` 由来の値を読む経路。本モジュール冒頭
/// コメント「メモリ増幅対策」節の層 2）が報告する `tensor_name` と、
/// この事前走査（層 1・`decode_model` 経由）が報告する `tensor_name` が
/// 複数回出現の悪意ある／不正な入力で食い違いうる（codex-review 指摘）。
/// 全出現を走査し最後の出現の結果を採用することで、両経路の診断
/// payload を一致させる。
fn scan_sparse_tensor_bytes_for_values_name(bytes: &[u8]) -> Option<String> {
    let mut buf: &[u8] = bytes;
    let mut last: Option<String> = None;
    while buf.has_remaining() {
        let (tag, wire_type) = decode_key(&mut buf).ok()?;
        match wire_type {
            WireType::LengthDelimited => {
                let field_bytes = take_length_delimited(&mut buf)?;
                if tag == 1 {
                    // 後勝ちマージ: この出現に `name` が無ければ（内側
                    // `Option` が `None`）以前の出現で得た `last` を保持
                    // する（protobuf は出現ごとにメッセージをマージする
                    // だけで、フィールド不在の出現が既存値を消すことは
                    // ない）。
                    if let Some(name) = scan_tensor_bytes_for_name(field_bytes)? {
                        last = Some(name);
                    }
                }
            }
            _ => skip_field(wire_type, tag, &mut buf, DecodeContext::default()).ok()?,
        }
    }
    last
}

/// `sparse_initializer` の診断用テンソル名に適用する共通の長さ上限
/// （バイト単位。UTF-8 文字境界は考慮しない）。事前走査（本モジュール
/// `scan_tensor_bytes_for_name`。層 1・`decode_model` 経由）と
/// `graph::build_graph`（層 2・`ModelProto::decode` を直接呼ぶ経路が
/// 読む `values.name`）の両方がこの定数・[`cap_sparse_tensor_diag_name`]
/// を経由することで、両経路が報告する `tensor_name` payload を一致させる
/// （codex-review P2 是正。2026-09-22。「診断契約の統一」）。
pub(crate) const SPARSE_TENSOR_NAME_DIAG_CAP: usize = 256;

/// `sparse_initializer` の診断名（`values.name`）を
/// [`SPARSE_TENSOR_NAME_DIAG_CAP`] バイトへ切り詰める。`String::
/// from_utf8_lossy` を経由するため、切り詰め位置が UTF-8 文字境界を跨いだ
/// 場合は置換文字（U+FFFD）になる（事前走査側の元実装と同じ挙動）。
/// `graph::build_graph` は既に UTF-8 検証済みの `String`（`values.name.
/// as_bytes()`）を渡し、事前走査側は非信頼入力の生バイト列をそのまま
/// 渡す。いずれも入力が同一バイト列であれば出力が一致する（上記
/// ドキュメンテーションコメント参照）。
pub(crate) fn cap_sparse_tensor_diag_name(bytes: &[u8]) -> String {
    let cap = bytes.len().min(SPARSE_TENSOR_NAME_DIAG_CAP);
    String::from_utf8_lossy(&bytes[..cap]).into_owned()
}

/// `TensorProto` 直下を走査し `name`（tag=8）だけを取り出す。`raw_data`
/// （tag=9）・packed `float_data`/`int64_data`（tag=4/7）・`dims`（tag=1）・
/// `data_type`（tag=2）には一切踏み込まない（該当タグは `skip_field` で
/// 読み飛ばすのみ。メモリ増幅対策の核心部分）。
///
/// `name`（singular string field）が同一メッセージ内に複数回出現した
/// 場合、protobuf は後勝ちでマージする（`scan_sparse_tensor_bytes_for_
/// values_name` のドキュメンテーションコメント参照）。本関数は
/// **`Option<Option<String>>` を返さず全出現を走査して最後の出現を
/// 採用する**ことで同じ後勝ち規則に従う（最初の出現で即 `return` しない）。
/// 走査自体が不正入力（バッファ終端超過等）で失敗した場合のみ `None` を
/// 早期に返す（`?` 演算子）。この失敗は呼び出し元
/// `scan_sparse_tensor_bytes_for_values_name` の `?` でさらに伝播するが、
/// その時点で `scan_graph_bytes_for_sparse_initializer` は既に `count` を
/// 加算済み（`*count += 1` の**後**にこの呼び出し系列へ入る）であり、
/// `*first_name` へは単純代入（`?` 非使用）で結果が書き込まれるため、
/// この失敗は診断名が空文字列になるだけで `prescan_sparse_initializer` の
/// `PrescanOutcome::Found` 判定（`count > 0`）には影響しない（2026-09-22
/// codex-review P0 是正の対象範囲外。`PrescanOutcome` のドキュメンテーション
/// コメント参照）。
fn scan_tensor_bytes_for_name(bytes: &[u8]) -> Option<Option<String>> {
    let mut buf: &[u8] = bytes;
    let mut last: Option<String> = None;
    while buf.has_remaining() {
        let (tag, wire_type) = decode_key(&mut buf).ok()?;
        match wire_type {
            WireType::LengthDelimited => {
                let field_bytes = take_length_delimited(&mut buf)?;
                if tag == 8 {
                    last = Some(cap_sparse_tensor_diag_name(field_bytes));
                }
            }
            _ => skip_field(wire_type, tag, &mut buf, DecodeContext::default()).ok()?,
        }
    }
    Some(last)
}

/// length-delimited フィールドの中身を、残りバッファ長と照合したうえで
/// 取り出す（`buf` はその分だけ前進する）。長さがバッファ終端を超える
/// 場合は不正入力として `None`（呼び出し元の走査は打ち切られる。
/// `prescan_sparse_initializer` は打ち切り時点までに検出済みの `count`
/// を優先し、`count == 0` のときだけ `ModelProto::decode` 本体へ判定を
/// 委ねる。`PrescanOutcome` のドキュメンテーションコメント参照）。
fn take_length_delimited<'b>(buf: &mut &'b [u8]) -> Option<&'b [u8]> {
    let len = decode_varint(buf).ok()?;
    let len = usize::try_from(len).ok()?;
    if len > buf.remaining() {
        return None;
    }
    let field_bytes = &buf[..len];
    buf.advance(len);
    Some(field_bytes)
}

/// `ModelProto::encode_to_vec` への薄い委譲入口（イシュー #2017）。
///
/// `decode_model` と対称の書き出し方向。facade からの再利用に加え、本クレート
/// 内テスト（合成モデルのバイト列化）でも `prost::Message` を都度 `use` せず
/// 済む便宜のために公開する（`onnx::export::build_model_proto` が組み立てた
/// `ModelProto` をワイヤフォーマットへ落とす最終段。ONNX export 自体の
/// facade 公開可否は別 issue のスコープ。`docs/facade-onnx-export-exposure-
/// decision.md`）。
pub fn encode_model(model: &ModelProto) -> Vec<u8> {
    model.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_model` -> `decode_model` の往復が構造的に等価（`PartialEq`
    /// derive）であることを最小構成の `ModelProto` で確認する（イシュー
    /// #2017 §5 ステップ 1）。
    #[test]
    fn encode_then_decode_roundtrips() {
        let model = ModelProto {
            ir_version: 7,
            producer_name: "fandhe-ai-test".to_string(),
            graph: Some(GraphProto {
                node: Vec::new(),
                name: "g".to_string(),
                initializer: Vec::new(),
                input: Vec::new(),
                output: Vec::new(),
                value_info: Vec::new(),
                sparse_initializer: Vec::new(),
            }),
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            }],
        };

        let bytes = encode_model(&model);
        let decoded = decode_model(&bytes).expect("decode は成功するはず");

        assert_eq!(decoded, model);
    }

    /// 壊れたバイト列（有効な protobuf ワイヤフォーマットではない）を渡すと
    /// `Err` で fail-closed に拒否されることを確認する（no-silent-skip
    /// 契約。`onnx::graph`／`onnx::interp` と同じ方針）。
    #[test]
    fn decode_rejects_garbage_bytes() {
        // タグ 1（varint）を宣言しつつ後続 varint を打ち切る不完全なバイト列。
        let garbage = [0x08u8, 0xffu8];
        let result = decode_model(&garbage);
        assert!(
            result.is_err(),
            "壊れたバイト列を decode_model が受理してしまった"
        );
    }
}
