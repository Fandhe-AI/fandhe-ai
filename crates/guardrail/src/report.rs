//! `guardrail check --output` が書き出す判定レポート JSON（v2 版）。
//!
//! `docs/guardrail-self-repair-cli.md` 2.1 節のスキーマをそのまま型として
//! 定義する。REQ-3 データ要件・REQ-6 の回帰テストセット根拠データとして
//! 再利用可能な形式を維持するため、フィールド名・型は文書と 1 対 1 対応させる。
//! シリアライズは `serde_json` に一任し、文字列連結で JSON を組み立てない
//! （2.5 節 A03 対策）。
//!
//! [`VerdictSection`] は 3 分岐判定の出力（`verdict`・`reason`・
//! `reason_conditions`・`applied_exclusion_rule_ids`）専用の派生型
//! （TASK-4.1c・イシュー #106）。[`Report`] は現段階では `verdict`/`reason`
//! を直接フィールドとして持つ（TASK-4.1a・イシュー #104 の骨格段階では
//! 判定ロジック〈#105/#106〉を未結線のため、`main.rs` が `Verdict::Escalate`
//! を暫定固定で埋める）。`VerdictSection::from_decision` は
//! [`crate::decision::decide`] の結果から一意にこれらの値を導出する経路を
//! 提供し、CLI 層が判定ロジックを結線する際（#105/#106 以降）に
//! `Report` へ合流させる想定（`.claude/rules/security.md` A08「判定の
//! 迂回経路を作らない」）。

use serde::{Deserialize, Serialize};

use crate::decision::{AUTO_APPLY_FALLBACK_REASON, Decision, Verdict};

/// シグナルの出所。`--signals`（1.2 節 CI 契約検証パス）経由なら `Injected`、
/// 実シグナル計測（本番相当経路。TASK-4.1a では未実装）なら `Measured`。
/// REQ-6 の回帰テストセット根拠データは `Measured` のみを採用する（2.1 節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSource {
    Measured,
    Injected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    Pass,
    Fail,
}

/// 判定レポート JSON のスキーマバージョン。2.1 節の記載どおり `"1"`（初版）で
/// 据え置く（新規設計文書のため既存消費者との互換負担がない）。
pub const SCHEMA_VERSION: &str = "1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: String,
    pub signal_source: SignalSource,
    pub change_id: Option<String>,
    pub lines_changed: u64,
    pub public_api_broken: bool,
    pub gaming_suspected: bool,
    pub build_result: GateOutcome,
    pub test_result: GateOutcome,
    pub clippy_result: GateOutcome,
    pub bench_measurements_pct: Vec<f64>,
    pub bench_median_pct: f64,
    pub applied_exclusion_rule_ids: Vec<String>,
    pub verdict: Verdict,
    pub reason: String,
    /// 機械可読の理由 ID 一覧（[`Decision::reason_conditions`] を転記。§2.1
    /// 追記フィールド。前方互換の追加のため `schema_version` は `"1"` 据え置き）。
    /// `Report` は `Deserialize` を要する（`report_round_trips_through_json`
    /// 等）ため `String` 所有型で持つ（`VerdictSection::reason_conditions`
    /// は `&'static str` のまま。TASK-4.1c・イシュー #106）。
    pub reason_conditions: Vec<String>,
}

/// [`Report::from_decision`] が受け取る、判定に依存しない実測値一式。
///
/// `verdict`/`reason`/`reason_conditions`/`applied_exclusion_rule_ids` は
/// 判定結果（[`Decision`]）から一意に導出するため本構造体には含めない
/// （引数の数が増えても呼び出し側が判定結果と実測値を取り違えないように、
/// 責務ごとに引数を分離する）。
#[derive(Debug, Clone)]
pub struct ReportInputs {
    pub signal_source: SignalSource,
    pub change_id: Option<String>,
    pub lines_changed: u64,
    pub public_api_broken: bool,
    pub gaming_suspected: bool,
    pub build_result: GateOutcome,
    pub test_result: GateOutcome,
    pub clippy_result: GateOutcome,
    pub bench_measurements_pct: Vec<f64>,
    pub bench_median_pct: f64,
}

impl Report {
    /// [`ReportInputs`]（実測値一式）と [`Decision`]（判定結果）から
    /// `Report` を構築する唯一の合流口（TASK-4.1c・イシュー #106）。
    ///
    /// `verdict`/`reason`/`reason_conditions`/`applied_exclusion_rule_ids` は
    /// 必ず [`VerdictSection::from_decision`] を経由して埋める
    /// （`.claude/rules/security.md` A08「判定の迂回経路を作らない」。
    /// `crate::check` の injected／measured 両経路がこの関数のみを通して
    /// `Report` を構築する）。
    pub fn from_decision(inputs: ReportInputs, decision: &Decision) -> Self {
        let section = VerdictSection::from_decision(decision);
        Report {
            schema_version: SCHEMA_VERSION.to_string(),
            signal_source: inputs.signal_source,
            change_id: inputs.change_id,
            lines_changed: inputs.lines_changed,
            public_api_broken: inputs.public_api_broken,
            gaming_suspected: inputs.gaming_suspected,
            build_result: inputs.build_result,
            test_result: inputs.test_result,
            clippy_result: inputs.clippy_result,
            bench_measurements_pct: inputs.bench_measurements_pct,
            bench_median_pct: inputs.bench_median_pct,
            applied_exclusion_rule_ids: section.applied_exclusion_rule_ids,
            verdict: decision.verdict(),
            reason: section.reason,
            reason_conditions: section
                .reason_conditions
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }
}

/// 計測値列から中央値を求める。TASK-4.1a では `report.rs` がレポートの整形にのみ
/// 用い、判定ロジック（#105/#106）には使わない。
///
/// 中央値の定義は `bench_harness::median_q1_q3`（ソート後 `idx = round(p * (n-1))`
/// 番目の要素を採用する median-of-halves 方式。PoC-v2-1 実測踏襲）にそのまま委譲する。
/// 本関数がここで独自に「中央 2 値の平均」方式を再実装すると、`bench_gate.rs` の
/// `BenchSignal::validate`（同じく `median_q1_q3` を用いて `bench_median_pct` の
/// 再計算・一致検証を行う）と定義が食い違い、偶数件の計測で正当な中央値を
/// 誤って reject しうる（Bugbot 指摘・#107 PR #305 レビュー）。計測系を
/// `bench-harness` へ付け替える本イシューの趣旨（`docs/guardrail-self-repair-cli.md`
/// 1.4 節）にも沿い、中央値の定義は `bench-harness` 側に一本化する。
///
/// 事前条件: `values` は空でないこと（呼び出し元の `signals.rs` が REQ-4
/// 「5 回以上」の受け入れ基準として非空・5 件以上を検証済み）。
///
/// # Errors
///
/// `values` が空、または NaN が混入している場合 `bench_harness::BenchError`
/// をそのまま返す（本番経路で `unwrap()`/`expect()` を使わない方針。
/// `.claude/rules/coding-rust.md`）。
pub fn median(values: &[f64]) -> Result<f64, bench_harness::BenchError> {
    bench_harness::median_q1_q3(values).map(|q| q.median)
}

/// 判定レポート JSON の「判定結果」セクション（§2.1 の `verdict`・`reason`
/// 相当フィールド群。TASK-4.1c・イシュー #106）。
///
/// `serde_json` のエスケープに出力を一任し、`reason` はテキスト連結時も
/// [`crate::decision::Reason`] の `Display` 実装（固定フォーマット文字列 +
/// 実測値の埋め込み）のみを経由する（外部由来の任意文字列をそのまま
/// 混入させない。ただし `ExclusionMatch.rule_id` は設定ファイル
/// `policy-exclusion.toml` 由来であり任意の外部入力ではない。
/// `.claude/rules/security.md` A03）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct VerdictSection {
    /// 機械可読の分岐識別子（`"auto_apply"` | `"escalate"` | `"reject"`。
    /// §2.1 `verdict` フィールド）。
    pub verdict: &'static str,
    /// 人間可読の判定理由（§2.1 `reason` フィールド）。逸脱条件が複数ある
    /// 場合は `"; "` で連結する。自動適用時（逸脱なし）は
    /// [`AUTO_APPLY_FALLBACK_REASON`] を用いる。
    pub reason: String,
    /// 機械可読の理由 ID 一覧（CI・自己修復ループの照合用）。
    pub reason_conditions: Vec<&'static str>,
    /// マッチしたポリシー除外リストのルール ID 一覧（空 = マッチなし。
    /// §2.1 `applied_exclusion_rule_ids` フィールド）。
    pub applied_exclusion_rule_ids: Vec<String>,
}

impl VerdictSection {
    /// [`Decision`] から判定結果セクションを導出する。
    ///
    /// `decide()` の戻り値をそのまま変換するのみで判定ロジックを持たない
    /// （表示・シリアライズ専任。`.claude/rules/security.md` A08）。
    pub fn from_decision(decision: &Decision) -> Self {
        let reason_conditions = decision.reason_conditions();
        let reason = if decision.reasons().is_empty() {
            AUTO_APPLY_FALLBACK_REASON.to_string()
        } else {
            decision
                .reasons()
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        };

        VerdictSection {
            verdict: decision.verdict().as_machine_id(),
            reason,
            reason_conditions,
            applied_exclusion_rule_ids: decision.exclusion_rule_ids().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PresetName, Thresholds};
    use crate::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals, decide};

    #[test]
    fn median_of_odd_length() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
    }

    #[test]
    fn median_of_even_length() {
        // `bench_harness::median_q1_q3` の median-of-halves 方式（idx = round(p*(n-1))）に
        // 委譲するため、n=4 では idx=round(1.5)=2 -> sorted[2]=3.0（「中央 2 値の平均」方式の
        // 2.5 とは異なる。bench_gate.rs の `BenchSignal::validate` と定義を統一した結果）。
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 3.0);
    }

    #[test]
    fn median_rejects_empty_values() {
        assert!(median(&[]).is_err());
    }

    #[test]
    fn median_rejects_nan_values() {
        assert!(median(&[1.0, f64::NAN, 2.0]).is_err());
    }

    #[test]
    fn report_round_trips_through_json() {
        let report = Report {
            schema_version: SCHEMA_VERSION.to_string(),
            signal_source: SignalSource::Injected,
            change_id: Some("abc123".to_string()),
            lines_changed: 10,
            public_api_broken: false,
            gaming_suspected: false,
            build_result: GateOutcome::Pass,
            test_result: GateOutcome::Pass,
            clippy_result: GateOutcome::Pass,
            bench_measurements_pct: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            bench_median_pct: 1.0,
            applied_exclusion_rule_ids: vec![],
            verdict: Verdict::Escalate,
            reason: "判定ロジック未実装（TASK-4.1b/#105 で移植）".to_string(),
            reason_conditions: vec!["gate_skipped".to_string()],
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.verdict, Verdict::Escalate);
        assert_eq!(parsed.signal_source, SignalSource::Injected);
    }

    fn thresholds() -> Thresholds {
        Thresholds::builtin(PresetName::Default)
    }

    fn all_passed_gates() -> GateSignals {
        GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Passed,
            clippy: GateSignal::Passed,
        }
    }

    #[test]
    fn auto_apply_uses_fallback_reason_and_serializes_machine_id() {
        let t = thresholds();
        let input = DecisionInput::new(
            &t,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");
        let decision = decide(&input).expect("判定に失敗");

        let section = VerdictSection::from_decision(&decision);
        assert_eq!(section.verdict, "auto_apply");
        assert_eq!(section.reason, AUTO_APPLY_FALLBACK_REASON);
        assert!(section.reason_conditions.is_empty());
        assert!(section.applied_exclusion_rule_ids.is_empty());

        let json = serde_json::to_value(&section).expect("シリアライズに失敗");
        assert_eq!(json["verdict"], "auto_apply");
    }

    #[test]
    fn reject_reason_joins_multiple_gate_failures() {
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Failed,
            clippy: GateSignal::Skipped,
        };
        let t = thresholds();
        let input =
            DecisionInput::new(&t, 10, gates, false, false, BenchSignal::NotRun, Vec::new())
                .expect("矛盾なし入力の構築に失敗");
        let decision = decide(&input).expect("判定に失敗");

        let section = VerdictSection::from_decision(&decision);
        assert_eq!(section.verdict, "reject");
        assert!(section.reason.contains("build"));
        assert!(section.reason.contains("; "));
        assert_eq!(
            section.reason_conditions,
            vec!["gate_build_failed", "gate_test_failed"]
        );

        let json = serde_json::to_value(&section).expect("シリアライズに失敗");
        assert_eq!(json["verdict"], "reject");
    }

    #[test]
    fn escalate_records_applied_exclusion_rule_ids() {
        let t = thresholds();
        let input = DecisionInput::new(
            &t,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            vec!["arch-hyperparameter-change".to_string()],
        )
        .expect("矛盾なし入力の構築に失敗");
        let decision = decide(&input).expect("判定に失敗");

        let section = VerdictSection::from_decision(&decision);
        assert_eq!(section.verdict, "escalate");
        assert_eq!(
            section.applied_exclusion_rule_ids,
            vec!["arch-hyperparameter-change".to_string()]
        );
    }
}
