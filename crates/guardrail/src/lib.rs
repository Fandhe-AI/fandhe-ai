//! 自己修復ループのガードレール。
//!
//! `self-repair` クレートが取り込む AI 生成変更を 3 分岐（自動適用・エスカレーション・
//! 却下）で判定し、判定の迂回経路を作らない（REQ-4。OWASP A08。`.claude/rules/security.md`）。
//! 判定閾値・ポリシー除外リストの変更は必ず人間（ユーザー）承認を経る運用とし、
//! 本クレート自体はその契約を強制する側であって、閾値を自己判断で緩和しない
//! （`.claude/rules/delegation-impl.md`）。v1（`Fandhe-AI/rust-ai-library-v1`）資産の
//! 移植先でもある。
//!
//! # モジュール構成（TASK-4.1c・イシュー #106 時点）
//! - [`decision`][]: 3 分岐判定ロジック本体（[`decision::decide`]）。評価済み 5 条件
//!   シグナルから [`decision::Verdict`] を導出する純粋関数。
//! - [`exit_code`][]: `guardrail check` の終了コード契約（[`exit_code::GuardrailExitCode`]）。
//!   `Verdict` → 終了コードの変換をここ 1 箇所に閉じ込める。
//! - [`report`][]: 判定レポート JSON の「判定結果」セクション（[`report::VerdictSection`]）。
//! - [`error`][]: クレート共通の型付きエラー（[`error::GuardrailError`]）。
//!
//! CLI 引数・設定パース・シグナル実測（TASK-4.1a／TASK-4.1b、イシュー #104／#105）、
//! ポリシー除外リスト評価（TASK-5.2 系）、`guardrail eval`・`self-repair` 連携
//! （TASK-4.2 以降）は本クレートの他モジュールが順次追加する想定であり、本イシューの
//! スコープ外（`.claude/rules/out-of-scope-tracking.md`）。

pub mod decision;
pub mod error;
pub mod exit_code;
pub mod report;
