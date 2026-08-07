//! ONNX 自前取り込みの構成要素（REQ-7）。
//!
//! - `proto`: protobuf デコード（`prost` 手書き derive、`protoc` 非依存。TASK-7.2a）
//! - `graph`: `ModelProto` -> 内部グラフ表現（トポロジカル順検証・initializer 復号。TASK-7.2a）
//! - `interp`: グラフ実行インタープリタ（`Graph` のノード列を `ops::*` へディスパッチ。
//!   TASK-7.2b・イシュー #78）
//!
//! 8 オペ実装は #79（`crate::ops`）、PoC 数値突合テストは #80
//! （`tests/onnx_poc_v2_6_match.rs`・`tests/onnx_slice_dynamic_bounds.rs`）で追加済み。
//! TASK-7.3 系 14 オペのディスパッチ結線は #274 で追跡する（`interp` モジュール
//! 冒頭コメント参照）。

pub mod graph;
pub mod interp;
pub mod proto;
