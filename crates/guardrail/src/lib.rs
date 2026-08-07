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
//! # モジュール構成（TASK-4.1a／TASK-4.1b／TASK-4.1c／TASK-4.1d／TASK-4.3a／TASK-4.4a／
//! TASK-4.4b・イシュー #104／#105／#106／#107／#112／#113／#115 合流時点）
//! - [`cli`][]: CLI 引数解析（`check`／`eval` サブコマンド。#104 管轄）。
//! - [`config`][]: `--config` の TOML 設定パース・検証、および判定閾値
//!   （[`config::Thresholds`]・プリセット・値域検証。#104 管轄。#105 の
//!   [`decision`] はこの型を契約 API としてそのまま受け取る）。
//! - [`signals`][]: `--signals` の JSON シグナル入力型（#104 管轄）。
//! - [`toml_lite`][]: 依存追加なしの最小 TOML パーサ（`config`／`eval::dataset`
//!   から利用。#104 管轄。文字列配列対応は #115 が追加）。
//! - [`decision`][]: 3 分岐判定ロジック本体（[`decision::decide`]）。評価済み 5 条件
//!   シグナルから [`decision::Verdict`] を導出する純粋関数（#105 管轄）。
//! - [`exit_code`][]: `guardrail check`／`guardrail eval` の終了コード契約
//!   （[`exit_code::GuardrailExitCode`]／[`exit_code::EvalExitCode`]）。
//!   `Verdict` → 終了コードの変換をここ 1 箇所に閉じ込める。
//! - [`report`][]: 判定レポート JSON のフルスキーマ（[`report::Report`]。#104 管轄）
//!   および 3 分岐判定の出力セクション（[`report::VerdictSection`]。#106 管轄）。
//! - [`eval`][]: `guardrail eval`（ラベル付きデータセット一括評価）の評価
//!   オーケストレーション（[`eval::run`]）・データセット読み込み
//!   （[`eval::dataset`]）・出力レポート型（[`eval::report`]。TASK-4.3a・#115 管轄）。
//! - [`error`][]: クレート共通の型付きエラー（[`error::GuardrailError`]）。
//! - [`median_gate`][]: 5 回以上計測の劣化率系列を検証し、`decision::BenchSignal::Measured`
//!   （中央値のみ受け取る受け口）を構築する唯一の公開経路（REQ-4「単発計測での閾値判定は
//!   行わないこと」。TASK-4.4a・イシュー #112）。
//! - [`determinism`][]: 学習系回帰テスト向け決定的シード設定ユーティリティ
//!   （TASK-4.4b・イシュー #113 管轄）。`self-repair` は
//!   `pub use guardrail::determinism;` で再輸出し双方から利用する。
//! - [`bench_gate`][]: ベンチゲート計測系（TASK-4.1d・#107 管轄）。`bench-harness` の
//!   計測 API（`run`・`MeasurementConfig`・`median_q1_q3`）を呼び出す実行系。判定ロジック
//!   本体（[`decision`]）・3 分岐出力（[`report::VerdictSection`]）とは並行実装のため
//!   未結線（同モジュールのドキュメント「スコープ境界」参照）。
//! - [`exclusion_match`][]: ポリシー除外リスト（REQ-5・`policy-exclusion.toml`。
//!   TASK-5.1a・#119 が定義）のうち `test-tolerance-loosening` ルールの match
//!   述語（[`exclusion_match::test_assertion_relaxation_without_prod_change`]）。
//!   テスト許容誤差の**単独**緩和（本番コード変更を伴わない）を検知し、
//!   REQ-4 ゲーミング検知（本番・テスト**同時**変更が対象）がすり抜ける
//!   PoC-3 既知ブラインドスポット G5 を補う（TASK-5.2b・イシュー #123）。
//!   `MatchRule` 列挙・`policy-exclusion.toml` ロード・`decide()` への配線は
//!   `policy_exclusion` モジュール（TASK-5.2a／c・#122／#124）のスコープ。
//!
//! `self-repair` は本クレートを **lib として直接呼び出す**（3.4 節。サブプロセス
//! 起動は行わない）ため、`main.rs`（バイナリ）とは独立して公開 API（`decide` 相当）
//! を提供する設計を維持する。`bin/guardrail`（`main.rs`）はここで公開するモジュール
//! を組み合わせて CLI フローを構成するのみで、判定ロジック自体はライブラリ側に置く
//! （`self-repair` からの lib 呼び出しと CLI 実行が同じロジックを共有するため）。
//!
//! ポリシー除外リスト評価（TASK-5.2 系）・`self-repair` 連携（TASK-4.2 以降）は
//! 本クレートの他モジュールが順次追加する想定であり、本 PR 群のスコープ外
//! （`.claude/rules/out-of-scope-tracking.md`）。

pub mod bench_gate;
pub mod cli;
pub mod config;
pub mod decision;
pub mod determinism;
pub mod error;
pub mod eval;
pub mod exclusion_match;
pub mod exit_code;
pub mod median_gate;
pub mod report;
pub mod signals;
pub mod toml_lite;

pub use config::{PresetName, Thresholds};
pub use decision::{
    AUTO_APPLY_FALLBACK_REASON, BenchSignal, Decision, DecisionInput, GateSignal, GateSignals,
    Reason, Verdict, decide,
};
pub use error::GuardrailError;
pub use exit_code::{EvalExitCode, GuardrailExitCode};
pub use report::VerdictSection;
pub use signals::Signals;
