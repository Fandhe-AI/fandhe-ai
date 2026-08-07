//! measured 経路（実シグナル計測）専用のチェック群（TASK-4.1c・イシュー #106）。
//!
//! [`crate::check`] の measured オーケストレーションから呼ばれる。
//! - [`diff_lines`][]: 変更行数（`git diff --numstat` 合算）。
//! - [`api_stability`][]: 公開 API 破壊検出（baseline ツリー全 `.rs` の
//!   `pub fn`/`pub struct`/`pub enum` 行シグネチャ比較。PoC-3 パリティ）。
//!
//! いずれも `git` 呼び出しは [`crate::exclusion_match::run_git`] に一本化し、
//! diff 出力汚染対策（`-c core.quotePath=false` 等）を独自実装しない。

pub(crate) mod api_stability;
pub(crate) mod diff_lines;
