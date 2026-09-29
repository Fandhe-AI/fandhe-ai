//! facade 統合テスト共有ヘルパー（`tests/common/mod.rs` 形式。
//! `crates/backend-cuda/tests/common/mod.rs` と同型のサブディレクトリ構成のため、
//! 独立したテストターゲットにはならない）。
//!
//! 各 `tests/*.rs` は個別の bin crate になり、`mod common;` で取り込んだ `pub` 項目の
//! うち当該 binary で未使用のものは `dead_code` 警告（clippy `-D warnings` で失敗）に
//! なる。公開 API を増やす場合は、取り込む全テストファイルが使うことを確認する。

pub mod temp_dir;
