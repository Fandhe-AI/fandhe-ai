//! 相互運用（interop）公開面（イシュー #2017）。
//!
//! 現時点は [`onnx`]（ONNX import）のみを提供する。ONNX export・
//! safetensors save／load を追加する場合はここへ兄弟モジュールとして
//! 差し込む設計とし（`docs/facade-onnx-export-exposure-decision.md`・
//! `docs/facade-safetensors-exposure-decision.md`。いずれもユーザー
//! 承認待ちの段階 0）、本ファイル自体は各サブモジュールの `pub mod`
//! 宣言のみに留める。

pub mod onnx;
