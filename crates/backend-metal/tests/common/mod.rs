//! `crates/backend-metal/tests/*.rs` は独立クレート扱いのため、各テスト
//! ファイルから `mod common;` で本モジュール群を共有する
//! （`crates/backend-cuda/tests/common/mod.rs` と同型のパターン）。

pub mod splitk_parity_baseline;
