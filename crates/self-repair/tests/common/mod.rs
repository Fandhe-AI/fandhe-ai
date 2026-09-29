//! 結合テスト共通ヘルパ（イシュー #2382、親 #2363）。各 `tests/*.rs` は別バイナリのため、
//! 未使用の pub 項目は dead_code になる。公開 API は実際に使う `new`・`path` のみに絞っている。
//!
//! - `temp_dir`: 全バイナリで使えるので `pub mod` で宣言する
//! - `temp_file.rs`: ファイル用ガードは一部バイナリだけが使うため、ここでは宣言しない。使う側が
//!   `mod common;` に加えて `#[path = "common/temp_file.rs"] mod temp_file;` で読み込む
//!   （dead_code を `#[allow]` で黙らせないための構成。coding-rust.md）

pub mod temp_dir;
