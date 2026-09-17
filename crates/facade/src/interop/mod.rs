//! 相互運用（interop）公開面（イシュー #2017・#2018・#2019）。
//!
//! `onnx`（ONNX import／export。import は #2017・export は #2018）に加え、
//! `safetensors`（safetensors save／load 純再エクスポート。#2019・
//! `docs/facade-safetensors-exposure-decision.md` §11 案 A）を提供する。
//! 本ファイル自体は各サブモジュールの `pub mod` 宣言のみに留める
//! （`pub use` は置かない。`tests/api_surface.rs::
//! interop_module_exposes_only_approved_onnx_surface` が機械的に固定）。

pub mod onnx;
pub mod safetensors;
