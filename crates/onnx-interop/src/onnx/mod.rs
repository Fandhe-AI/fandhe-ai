//! ONNX 自前取り込みの構成要素（REQ-7）。
//!
//! - `proto`: protobuf デコード（`prost` 手書き derive、`protoc` 非依存。TASK-7.2a）
//! - `graph`: `ModelProto` -> 内部グラフ表現（トポロジカル順検証・initializer 復号。TASK-7.2a）
//!
//! グラフ実行（インタープリタ基盤）は #78、8 オペ実装は #79、PoC 数値突合テストは
//! #80 で追加予定（本モジュールはグラフ構造のデコードまでを担う。イシュー #77）。

pub mod graph;
pub mod proto;
