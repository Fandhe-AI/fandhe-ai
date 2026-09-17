//! 相互運用（interop）公開面（イシュー #2017・#2019）。
//!
//! `onnx`（ONNX import。#2017）に加え、`safetensors`（safetensors
//! save／load 純再エクスポート。#2019・`docs/facade-safetensors-
//! exposure-decision.md` §11 案 A）を提供する。ONNX export を追加する
//! 場合はここへ兄弟モジュールとして差し込む設計とし
//! （`docs/facade-onnx-export-exposure-decision.md`。ユーザー承認待ちの
//! 段階 0）、本ファイル自体は各サブモジュールの `pub mod` 宣言のみに
//! 留める（`pub use` は置かない。`tests/api_surface.rs::
//! interop_module_exposes_only_approved_onnx_surface` が機械的に固定）。

pub mod onnx;
pub mod safetensors;
