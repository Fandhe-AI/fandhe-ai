//! 判定レポート JSON（`docs/guardrail-self-repair-cli.md` §2.1）のうち、
//! 3 分岐判定の出力に対応する部分（TASK-4.1c・イシュー #106）。
//!
//! §2.1 のフルスキーマ（`schema_version`・`signal_source`・`change_id`・
//! `lines_changed`・`build_result`・`bench_measurements_pct` 等）は
//! シグナル収集・CLI 骨格を担う #104（TASK-4.1a）・#105（TASK-4.1b）が
//! 定義する `Report` 構造体の管轄であり、本モジュールはそこに
//! `#[serde(flatten)]` で合流させる想定の「判定結果セクション」のみを
//! 提供する（`.claude/rules/delegation-impl.md`: 同一ファイルの並行編集を
//! 避けるための分割）。
//!
//! `verdict`（機械可読 ID）・`reason`（人間可読理由。複数逸脱時は `; ` 区切り
//! で連結）・`reason_conditions`（機械可読理由 ID の配列）・
//! `applied_exclusion_rule_ids`（マッチしたポリシー除外ルール ID）の 4
//! フィールドは [`crate::decision::Decision`] から一意に導出され、
//! [`crate::decision::decide`] を経由しない別経路からこれらの値を組み立てる
//! ことはない（`.claude/rules/security.md` A08「判定の迂回経路を作らない」）。

use serde::Serialize;

use crate::decision::{AUTO_APPLY_FALLBACK_REASON, Decision};

/// 判定レポート JSON の「判定結果」セクション（§2.1 の `verdict`・`reason`
/// 相当フィールド群）。
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
    use crate::config::{PresetName, Thresholds, ThresholdsRaw};
    use crate::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals, decide};

    fn thresholds() -> Thresholds {
        Thresholds::from_raw(
            PresetName::Default,
            ThresholdsRaw {
                lines_max: 200,
                bench_max_pct: 5.0,
                bench_runs: 5,
            },
        )
        .expect("固定値の検証に失敗")
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
