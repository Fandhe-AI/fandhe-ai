//! 自己修復ループのガードレール。
//!
//! `self-repair` クレートが取り込む AI 生成変更を 3 分岐（採用・保留・却下等）で判定し、
//! 判定の迂回経路を作らない（REQ-4。OWASP A08。`.claude/rules/security.md`）。
//! 判定閾値・ポリシー除外リストの変更は必ず人間（ユーザー）承認を経る運用とし、
//! 本クレート自体はその契約を強制する側であって、閾値を自己判断で緩和しない
//! （`.claude/rules/delegation-impl.md`）。v1 資産の移植先でもある。
//!
//! # TASK-4.1a（本イシュー）のスコープ
//!
//! CLI 骨格・引数解析・設定（TOML）パースと検証・シグナル JSON 入力型・
//! 判定レポート JSON 出力・終了コード契約の型定義のみを実装する
//! （`docs/guardrail-self-repair-cli.md` が正本の設計文書）。5 条件の判定
//! ロジック本体は TASK-4.1b（#105）、3 分岐出力・判定根拠の作り込みは
//! TASK-4.1c（#106）、ベンチ計測の bench-harness 付け替えは TASK-4.1d（#107）
//! で実装する。`self-repair` は本クレートを **lib として直接呼び出す**
//! （3.4 節。サブプロセス起動は行わない）ため、`main.rs`（バイナリ）とは
//! 独立して公開 API（`decide` 相当）を提供する設計を維持する。
//!
//! `bin/guardrail`（`main.rs`）はここで公開するモジュールを組み合わせて
//! CLI フローを構成するのみで、判定ロジック自体はライブラリ側に置く
//! （`self-repair` からの lib 呼び出しと CLI 実行が同じロジックを共有するため）。

pub mod cli;
pub mod config;
pub mod error;
pub mod exit_code;
pub mod report;
pub mod signals;
pub mod toml_lite;
