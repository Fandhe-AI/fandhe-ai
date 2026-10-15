//! REQ-2 統一複合判定の許容誤差定数の正本（単一定義）。
//!
//! `backend-cpu::parity`（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` として再エクスポート）と
//! `autodiff`（具体バックエンドクレートへ依存できないため、両クレートが参照できる
//! `tensor-core` へ正本を置く）が共有する。
//!
//! **閾値の変更はユーザー承認必須**（`.claude/rules/coding-rust.md`
//! 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」・
//! `.claude/rules/security.md` A08）。

/// REQ-2 統一複合判定の相対誤差閾値（全ペア共通）。変更はユーザー承認必須。
pub const RELATIVE_TOLERANCE: f64 = 1e-3;

/// REQ-2 統一複合判定の絶対誤差救済閾値（0 近傍の相対誤差跳ね上がり対策。全ペア共通）。
/// 変更はユーザー承認必須。
pub const ABSOLUTE_RESCUE_THRESHOLD: f64 = 1e-5;
