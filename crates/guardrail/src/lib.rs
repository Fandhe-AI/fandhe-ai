//! 自己修復ループのガードレール。
//!
//! `self-repair` クレートが取り込む AI 生成変更を 3 分岐（自動適用・
//! エスカレーション・却下）で判定し、判定の迂回経路を作らない（REQ-4。
//! OWASP A08。`.claude/rules/security.md`）。判定閾値・ポリシー除外リストの
//! 変更は必ず人間（ユーザー）承認を経る運用とし、本クレート自体はその契約を
//! 強制する側であって、閾値を自己判断で緩和しない
//! （`.claude/rules/delegation-impl.md`）。v1（`Fandhe-AI/rust-ai-library-v1`）
//! 資産の移植先でもある。
//!
//! TASK-4.1b（イシュー #105）で閾値体系（[`config`]）・判定ロジック本体
//! （[`decision`]）・シグナル入力型（[`signals`]）・共有エラー型（[`error`]）を
//! 移植し、TASK-4.1c（イシュー #106）で終了コード契約（[`exit_code`]）・
//! 判定レポート JSON の「判定結果」セクション（[`report`]）を追加した。CLI 配線・
//! 設定ファイル（`guardrail.toml`）パース・実シグナル計測（git 差分・ゲート実行等）
//! は別イシュー（#104・#107 等）の管轄であり、本クレートは lib モジュールのみで
//! 完結する（spec 根拠: `docs/spec/05-tasks.md` TASK-4.1、REQ-4）。
//!
//! # モジュール構成
//! - [`error`][]: クレート全体で共有する型付きエラー（`GuardrailError`）
//! - [`config`][]: 5 条件のうち行数・ベンチの 2 条件が持つ閾値体系
//!   （`Thresholds`・プリセット・値域検証）
//! - [`decision`][]: 3 分岐判定ロジック本体（`decide`・`Verdict`・`Reason` 等）
//! - [`signals`][]: シグナル入力型（`Signals`）から `decision::DecisionInput`
//!   への変換
//! - [`exit_code`][]: `guardrail check` の終了コード契約
//!   （[`exit_code::GuardrailExitCode`]）。`Verdict` → 終了コードの変換を
//!   ここ 1 箇所に閉じ込める。
//! - [`report`][]: 判定レポート JSON の「判定結果」セクション
//!   （[`report::VerdictSection`]）。
//!
//! ポリシー除外リスト評価（TASK-5.2 系）・`guardrail eval`・`self-repair` 連携
//! （TASK-4.2 以降）は本クレートの他モジュールが順次追加する想定であり、
//! 本イシューのスコープ外（`.claude/rules/out-of-scope-tracking.md`）。

pub mod config;
pub mod decision;
pub mod error;
pub mod exit_code;
pub mod report;
pub mod signals;

pub use config::{MIN_BENCH_RUNS, PresetName, Thresholds, ThresholdsRaw};
pub use decision::{
    AUTO_APPLY_FALLBACK_REASON, BenchSignal, Decision, DecisionInput, GateSignal, GateSignals,
    Reason, Verdict, decide,
};
pub use error::GuardrailError;
pub use exit_code::GuardrailExitCode;
pub use report::VerdictSection;
pub use signals::Signals;
