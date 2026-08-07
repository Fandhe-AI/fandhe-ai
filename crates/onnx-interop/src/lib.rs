//! ONNX / safetensors 相互運用層。
//!
//! `safetensors` はワイヤフォーマットの読み書きのみに用い、`tensor-core` のテンソル型への
//! マッピングは自作する。ONNX の protobuf デコードは `prost` を用いるが、`prost-build`
//! （`protoc` へのビルド時依存）は使わず手書き derive で取り込む（PoC-v2-6。
//! `.claude/rules/deps-policy.md`）。外部フォーマットのパースは長さ・形状の検証を先に行い
//! 不正入力を弾く（OWASP A03。`.claude/rules/security.md`）。
//!
//! ## モジュール構成
//!
//! - [`st_load`]: safetensors パース → `tensor-core::Tensor<f32>` へのマッピング
//!   （TASK-7.1a・#73・REQ-7）。PyTorch で保存した重みファイルのロード経路を提供する。
//! - ONNX 経路（TASK-7.2 以降）は本イシューのスコープ外で未実装。

pub mod st_load;
