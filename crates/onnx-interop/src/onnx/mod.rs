//! ONNX 自前取り込みの構成要素（REQ-7）。
//!
//! - `proto`: protobuf デコード（`prost` 手書き derive、`protoc` 非依存。TASK-7.2a）
//! - `graph`: `ModelProto` -> 内部グラフ表現（トポロジカル順検証・initializer 復号。TASK-7.2a）
//! - `interp`: グラフ実行インタープリタ（`Graph` のノード列を `ops::*` へディスパッチ。
//!   TASK-7.2b・イシュー #78。import 対応は 36 op〈イシュー #2200 で `GlobalAveragePool`／
//!   `BatchNormalization`／`Flatten`、イシュー #2199 で `MaxPool`／`AveragePool`
//!   （`Conv` の 1D 対応追加を含む）、イシュー #2186 で `Clip`／`Tanh`／`Gelu`／
//!   `Where`／`Expand`／`ReduceMean`／`Pad`／`Resize` を追加〉）
//! - `export`: 内部グラフ表現 `Graph` -> `GraphProto`／`ModelProto` への降下
//!   （`graph` の逆方向。イシュー #1772）。`build_model_proto` は `export_ops`
//!   （内部 op -> `NodeProto` の意味論的マッピング。#1773）の
//!   `check_exportable` を経由してから組み立てる。
//! - `export_ops`: `interp` が対応する 36 op のうち export allowlist（`SUPPORTED_OP_TYPES`。
//!   イシュー #1773）に含まれる 26 op の逆マッピング（`ExportOp` -> `NodeProto`）。`export`
//!   から `pub use` で再エクスポートする。**import（`interp`）と export（`export_ops`）
//!   の対応 op 数は非対称**（イシュー #2200 で `GlobalAveragePool`／`BatchNormalization`／
//!   `Flatten` を import・export 双方へ追加し対称化した〈#2187〉一方、イシュー #2186
//!   （E2・PR #2313）で追加した `Clip`／`Tanh`／`Gelu`／`Where`／`Expand`／`ReduceMean`／
//!   `Pad`／`Resize`、およびイシュー #2199（PR #2314）で追加した `MaxPool`／
//!   `AveragePool` はいずれも import 側のみで、export allowlist は 26 op のまま
//!   拡張していない。これらを含むグラフの export は `OnnxError::UnsupportedOp` で
//!   拒否される〈無言 skip しない〉。この非対称は計画時点で `docs/
//!   onnx-export-op-mapping.md` §8 が想定済みの残作業であり〈export 側 PR
//!   マージ後に別イシューで追う〉。`tests/onnx_export_ops.rs` 参照）。
//! - `export_nn`: `fandhe_ai_autodiff::nn::Module` の層列（`Linear`／`ReLU`
//!   限定）から `export` が受け取れる `Graph` を組み立てる橋渡し
//!   （イシュー #2036。本モジュール自体は本クレート内部限定のまま
//!   facade へ再エクスポートしないが、facade
//!   `OnnxModel::from_sequential`〈イシュー #2037〉が薄く委譲して呼ぶ）。
//! - `external_data`: ONNX external data（外部 `.onnx.data` ファイル）の
//!   initializer／Constant 属性テンソル読み込みに fail-closed で対応する
//!   新しい import 入口（`build_graph_with_external_data`。イシュー
//!   #2347）。`onnx-interop` 内部限定（facade へは非公開。承認待ち事項は
//!   `docs/compat-api-scope.md` §5 参照）。
//!
//! 8 オペ実装は #79（`crate::ops`）、PoC 数値突合テストは #80
//! （`tests/onnx_poc_v2_6_match.rs`・`tests/onnx_slice_dynamic_bounds.rs`）で追加済み。
//! TASK-7.3 系 14 オペのディスパッチ結線は #274 で追跡する（`interp` モジュール
//! 冒頭コメント参照）。

pub mod autograd;
pub mod export;
pub mod export_nn;
pub mod export_ops;
// external data（外部 `.data` ファイル）の initializer／Constant 属性テンソル
// 読み込みに fail-closed で対応する新しい import 入口（イシュー #2347）。
// `onnx-interop` 内部限定（facade へは非公開。承認待ち事項は
// `docs/compat-api-scope.md` §5 参照）。
pub mod external_data;
pub mod graph;
pub mod interp;
// `interp` から呼ばれる `BackendOps` 経由の device 実行ヘルパ（非公開。
// イシュー #2077）。`interp::run_with_ops` の内部実装詳細であり facade
// 公開面には出さない。
mod interp_device;
// `interp` のディスパッチ表が委譲する追加 8 op（`Clip`／`Tanh`／`Gelu`／
// `Where`／`Expand`／`ReduceMean`／`Pad`／`Resize`。イシュー #2186）。
// `interp_device` と同じ位置づけの非公開実装詳細（`interp::run_impl`
// のディスパッチ先であり facade 公開面には出さない）。
mod interp_ext;
pub mod proto;
