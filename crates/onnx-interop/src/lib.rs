//! ONNX / safetensors 相互運用層。
//!
//! `safetensors` はワイヤフォーマットの読み書きのみに用い、`tensor-core` のテンソル型への
//! マッピングは自作する。ONNX の protobuf デコードは `prost` を用いるが、`prost-build`
//! （`protoc` へのビルド時依存）は使わず手書き derive で取り込む（PoC-v2-6。
//! `.claude/rules/deps-policy.md`）。外部フォーマットのパースは長さ・形状の検証を先に行い
//! 不正入力を弾く（OWASP A03。`.claude/rules/security.md`）。
//!
//! 雛形段階（TASK-1.1 部分実装。許容依存の `Cargo.toml` 反映を除く。反映はユーザー承認を
//! 要するため別イシューで対応する）では型・実装を持たない（spec 根拠: `docs/spec/05-tasks.md`
//! TASK-1.1、REQ-7）。
//!
//! `require_keys`（TASK-7.1b・イシュー #74）は safetensors ロード結果のキーマップに対する
//! 期待キー集合の充足検査を提供する。キー不足を無言 skip せず型付きエラー
//! （[`LoadError`]）で報告する（v1 PoC-6 詰まりポイント #2 対策。詳細は
//! `require_keys` モジュールのドキュメンテーションコメント参照）。
//!
//! `require_keys` モジュールは非公開（`mod`）とし、型・関数のみ `pub use` で
//! クレートルートへ再エクスポートする。モジュールを `pub mod` にすると
//! `onnx_interop::require_keys` がモジュール（型名前空間）と関数（値名前空間）の
//! 同名で並存し、`use onnx_interop::require_keys;` が両方を意図せずインポートして
//! 読み手（Claude を含む）を混乱させるため（イシュー #74 レビュー指摘）。

mod require_keys;
pub use require_keys::{LoadError, require_keys};
