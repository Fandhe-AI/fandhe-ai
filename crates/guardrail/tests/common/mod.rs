//! 結合テスト共通ヘルパ（イシュー #2380）。各 `tests/*.rs` は別バイナリのため、
//! 未使用の pub 項目は dead_code になる。公開 API は実際に使う `new`・`path` のみに絞っている。

pub mod temp_dir;
