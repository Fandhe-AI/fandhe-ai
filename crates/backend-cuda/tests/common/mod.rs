//! `crates/backend-cuda/tests/*.rs` は独立クレート扱いのため、各テスト
//! ファイルはここを `mod common;` で読み込んで共有フィクスチャへアクセス
//! する（`crates/autodiff/tests/common/mod.rs` と同型のパターン）。
//!
//! サブモジュール構成の理由: 本ディレクトリは現状 parity 非後退契約
//! （イシュー #491）のフィクスチャのみを持つが、将来他の共通フィクスチャが
//! 増えた場合に単一ファイルへ詰め込まず追加できるよう、役割単位で
//! サブモジュール分割する（`parity_baseline` は fixture・検査ユーティリティ
//! の役割に閉じる）。

pub mod parity_baseline;
