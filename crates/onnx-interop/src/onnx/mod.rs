//! ONNX 自前取り込みの構成要素（REQ-7）。
//!
//! - `proto`: protobuf デコード（`prost` 手書き derive、`protoc` 非依存。TASK-7.2a）
//! - `graph`: `ModelProto` -> 内部グラフ表現（トポロジカル順検証・initializer 復号。TASK-7.2a）
//! - `interp`: グラフ実行インタープリタ（`Graph` のノード列を `ops::*` へディスパッチ。
//!   TASK-7.2b・イシュー #78。import 対応は 26 op〈イシュー #2200 で `GlobalAveragePool`／
//!   `BatchNormalization`／`Flatten` を追加〉）
//! - `export`: 内部グラフ表現 `Graph` -> `GraphProto`／`ModelProto` への降下
//!   （`graph` の逆方向。イシュー #1772）。`build_model_proto` は `export_ops`
//!   （内部 op -> `NodeProto` の意味論的マッピング。#1773）の
//!   `check_exportable` を経由してから組み立てる。
//! - `export_ops`: `interp` が対応する 26 op のうち export allowlist に含まれる
//!   ものの逆マッピング（`ExportOp` -> `NodeProto`。イシュー #1773）。`export` から
//!   `pub use` で再エクスポートする。**import（`interp`）と export
//!   （`export_ops`）は 26 op で対称**（イシュー #2200 で import が 26 op へ
//!   拡大した際に生じた非対称〈`GlobalAveragePool`／`BatchNormalization`／
//!   `Flatten` が import のみ対応〉を、イシュー #2187 で export 側へも
//!   追加して解消済み。`tests/onnx_export_ops.rs` の
//!   `supported_op_types_are_all_reachable_in_interp_dispatch_table` が
//!   対称性を fail-closed に固定する。E2 の 8 op〈#2313〉・MaxPool／
//!   AveragePool〈#2314〉は import 側 PR が未マージのため、export 側も
//!   未対応のまま残る〈#2187 実装計画 §0 参照〉）。
//! - `export_nn`: `fandhe_ai_autodiff::nn::Module` の層列（`Linear`／`ReLU`
//!   限定）から `export` が受け取れる `Graph` を組み立てる橋渡し
//!   （イシュー #2036。本モジュール自体は本クレート内部限定のまま
//!   facade へ再エクスポートしないが、facade
//!   `OnnxModel::from_sequential`〈イシュー #2037〉が薄く委譲して呼ぶ）。
//!
//! 8 オペ実装は #79（`crate::ops`）、PoC 数値突合テストは #80
//! （`tests/onnx_poc_v2_6_match.rs`・`tests/onnx_slice_dynamic_bounds.rs`）で追加済み。
//! TASK-7.3 系 14 オペのディスパッチ結線は #274 で追跡する（`interp` モジュール
//! 冒頭コメント参照）。

pub mod autograd;
pub mod export;
pub mod export_nn;
pub mod export_ops;
pub mod graph;
pub mod interp;
// `interp` から呼ばれる `BackendOps` 経由の device 実行ヘルパ（非公開。
// イシュー #2077）。`interp::run_with_ops` の内部実装詳細であり facade
// 公開面には出さない。
mod interp_device;
pub mod proto;
