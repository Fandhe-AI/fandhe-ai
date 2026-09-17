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
//! 時点で拡張する。
//!
//! フィールド番号の出典: `onnx==1.22.0` 同梱の `onnx/onnx.proto` を実際に読み、
//! 該当 6 メッセージのフィールド番号を転記した（PoC-v2-6 で実ファイル
//! `model.onnx` / `slice_repro.onnx` / `transformer.onnx` を parse してノード数・
//! op_type 列が期待どおりか検証済み。`docs/spec/03-poc/poc-v2-6-interop/evidence/`
//! 配下の各ログ参照）。

use prost::Message;

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

/// `ModelProto::decode` への薄い委譲入口（イシュー #2017）。
///
/// facade（`crates/facade`）は crates.io 公開クレートの依存面を絞るため
/// `prost` へ直接依存せず、呼び出し側が `prost::Message` を `use` しなくても
/// `.onnx` バイト列を復号できるこの関数だけを経由する。検証（dims 非負・
/// 要素数整合・トポロジカル順等）は本関数の責務外で、後続の
/// `graph::build_graph` が担う（本モジュール冒頭コメント・`graph.rs` 冒頭
/// コメント参照）。
pub fn decode_model(bytes: &[u8]) -> Result<ModelProto, prost::DecodeError> {
    ModelProto::decode(bytes)
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
