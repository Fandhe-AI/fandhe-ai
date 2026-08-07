//! ONNX protobuf メッセージの手書き `prost::Message` 実装（部分実装）。
//!
//! ## 生成方法についての設計判断（REQ-7・deps-policy.md・PoC-v2-6）
//!
//! `prost-build`（`protoc` へのビルド時依存）は使わず、TASK-7.2a（本モジュール）が
//! 必要とする 6 メッセージ（`ModelProto` / `GraphProto` / `NodeProto` /
//! `AttributeProto` / `TensorProto` / `ValueInfoProto`）のみをフィールド番号を
//! 一致させた `#[derive(prost::Message)]` 構造体として手書きする。`protoc` 非依存の
//! 手書き derive は OWASP A06（サプライチェーン）の観点でもむしろ縮小になる（PoC-v2-6
//! advisor レビュー由来の判断）。
//!
//! `prost::Message::decode` は構造体に未宣言のフィールド番号を protobuf のワイヤ
//! フォーマット仕様どおり自動的にスキップするため、`TypeProto`（`ValueInfoProto.type`）
//! 等、本クレートが現時点で使わない再帰的メッセージは意図的に定義しない。#78（インタープリタ
//! 基盤）・#79（8 オペ実装）で必要になった時点で拡張する。
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
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
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
pub mod data_type {
    pub const FLOAT: i32 = 1;
    pub const INT64: i32 = 7;
}
