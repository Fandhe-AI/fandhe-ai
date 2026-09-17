//! 相互運用（interop）公開面（イシュー #2017）。
//!
//! 現時点は `onnx`（ONNX import／export。import は #2017・export は
//! #2018）のみを提供する。safetensors save／load を追加する場合はここへ
//! 兄弟モジュールとして差し込む設計とし（`docs/facade-safetensors-
//! exposure-decision.md`。ユーザー承認待ちの段階 0）、本ファイル自体は
//! 各サブモジュールの `pub mod` 宣言のみに留める。

pub mod onnx;
