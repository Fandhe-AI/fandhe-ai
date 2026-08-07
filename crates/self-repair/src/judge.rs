//! guardrail 3 分岐判定を取り込み判断へ接続するアダプタ（TASK-3.1d・イシュー #135・REQ-3）。
//!
//! v1（`Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/judge.rs`）の
//! `GuardrailAdoptionJudge` を v2 API へ差し替えて移植する。`guardrail` は
//! サブプロセスでなく lib として直接呼び出す（`docs/guardrail-self-repair-cli.md`
//! §3.4）。[`GuardrailAdoptionJudge::judge`] は [`crate::outcome::VerifiedEvidence`]
//! の 6 シグナルをそのまま `guardrail::DecisionInput::new` へ渡し
//! `guardrail::decide` を呼ぶだけであり、`decide` の結果を経由しない
//! [`crate::outcome::AdoptionVerdict`] 生成経路をモジュール内に存在させない
//! （A08: 自己修復ループが取り込む変更はガードレール判定を必ず経由し、判定の
//! 迂回経路を作らない）。

use crate::error::SelfRepairError;
use crate::outcome::{AdoptionVerdict, VerifiedEvidence};
use crate::stages::AdoptionJudge;

/// `guardrail::decision::decide` を取り込み判断として使うアダプタ。
///
/// 閾値（[`guardrail::Thresholds`]）は保持するのみで一切緩和・再定義しない
/// （`.claude/rules/security.md`「ガードレール閾値の変更はユーザー承認必須」）。
/// `Copy` を実装する `Thresholds` をそのまま値で保持し、`judge` の都度
/// `&self.thresholds` として `guardrail::DecisionInput::new` へ渡す。
#[derive(Debug, Clone, Copy)]
pub struct GuardrailAdoptionJudge {
    thresholds: guardrail::Thresholds,
}

impl GuardrailAdoptionJudge {
    /// 検証済みの `guardrail::Thresholds` を受け取って構築する。
    ///
    /// 閾値の妥当性検証（値域チェック）は呼び出し元（`guardrail::config`
    /// 側。`Thresholds::builtin`／TOML 読み込み経路）の責務であり、本型では
    /// 再検証しない（`crates/guardrail/src/config.rs` の `validate` を正とする）。
    pub fn new(thresholds: guardrail::Thresholds) -> Self {
        GuardrailAdoptionJudge { thresholds }
    }
}

impl AdoptionJudge for GuardrailAdoptionJudge {
    /// `evidence` の 6 シグナルを `guardrail::DecisionInput::new` へそのまま渡し、
    /// `guardrail::decide` の結果を [`AdoptionVerdict`] へ変換する。
    ///
    /// `guardrail::Verdict` の 3 variant を網羅 match で変換する（`_ =>` は
    /// 使わない。variant 追加時にコンパイルエラーで検出する fail-closed 設計。
    /// `.claude/rules/security.md` A05）。
    fn judge(&self, evidence: &VerifiedEvidence) -> Result<AdoptionVerdict, SelfRepairError> {
        let input = guardrail::DecisionInput::new(
            &self.thresholds,
            evidence.lines_changed(),
            evidence.gates(),
            evidence.api_broken(),
            evidence.gaming_suspect(),
            *evidence.bench(),
            evidence.exclusion_rule_ids().to_vec(),
        )
        .map_err(|error| SelfRepairError::Judgement {
            attempt: evidence.attempt(),
            reason: format!(
                "guardrail::DecisionInput の構築に失敗しました（矛盾入力。fail-closed）: {error}"
            ),
        })?;

        let decision = guardrail::decide(&input).map_err(|error| SelfRepairError::Judgement {
            attempt: evidence.attempt(),
            reason: format!("guardrail::decide の実行に失敗しました: {error}"),
        })?;

        // 理由破棄禁止（v1 PR #171 レビュー指摘の踏襲）: `Escalate`/`Reject`
        // では `decision.reasons()` を必ず `LoopReport` へ伝わる形へ変換する。
        let reason_text = || {
            decision
                .reasons()
                .iter()
                .map(|reason| reason.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        };

        match decision.verdict() {
            guardrail::Verdict::AutoApply => Ok(AdoptionVerdict::Adopt),
            guardrail::Verdict::Escalate => Ok(AdoptionVerdict::Escalate {
                reason: reason_text(),
            }),
            guardrail::Verdict::Reject => Ok(AdoptionVerdict::Reject {
                // guardrail の却下（build/test/clippy いずれかの失敗）はゲート
                // 失敗のみが理由であり、別の修正案であれば次の試行で通過しうる
                // ため常に再試行可能とする（`retryable: false` は本アダプタでは
                // 生成しない。「対応不能と判明した」等の確定却下は本イシューの
                // スコープ外）。
                retryable: true,
                reason: reason_text(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use guardrail::{BenchSignal, GateSignal, GateSignals, PresetName, Thresholds};

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

    #[allow(clippy::too_many_arguments)]
    fn evidence(
        attempt: u32,
        gates: GateSignals,
        bench: BenchSignal,
        lines_changed: u64,
        api_broken: bool,
        gaming_suspect: bool,
        exclusion_rule_ids: Vec<String>,
    ) -> VerifiedEvidence {
        VerifiedEvidence::new(
            attempt,
            "test proposal",
            "gates: build=pass test=pass clippy=pass",
            gates,
            bench,
            lines_changed,
            api_broken,
            gaming_suspect,
            exclusion_rule_ids,
        )
    }

    #[test]
    fn all_green_yields_adopt() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::NotRun,
            10,
            false,
            false,
            Vec::new(),
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        assert_eq!(verdict, AdoptionVerdict::Adopt);
    }

    #[test]
    fn gate_failure_yields_retryable_reject_with_reason() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let ev = evidence(1, gates, BenchSignal::NotRun, 10, false, false, Vec::new());

        let verdict = judge.judge(&ev).expect("judge should not error");
        match verdict {
            AdoptionVerdict::Reject { retryable, reason } => {
                assert!(retryable);
                assert!(reason.contains("build"));
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn lines_max_exceeded_yields_escalate_with_reason() {
        let thresholds = thresholds();
        let judge = GuardrailAdoptionJudge::new(thresholds);
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::NotRun,
            thresholds.lines_max + 1,
            false,
            false,
            Vec::new(),
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        match verdict {
            AdoptionVerdict::Escalate { reason } => assert!(!reason.is_empty()),
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    #[test]
    fn api_broken_yields_escalate() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::NotRun,
            10,
            true,
            false,
            Vec::new(),
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        assert!(matches!(verdict, AdoptionVerdict::Escalate { .. }));
    }

    #[test]
    fn gaming_suspect_yields_escalate() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::NotRun,
            10,
            false,
            true,
            Vec::new(),
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        assert!(matches!(verdict, AdoptionVerdict::Escalate { .. }));
    }

    #[test]
    fn bench_median_exceeded_yields_escalate() {
        let thresholds = thresholds();
        let judge = GuardrailAdoptionJudge::new(thresholds);
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::Measured {
                median_pct: thresholds.bench_median_max_pct + 1.0,
            },
            10,
            false,
            false,
            Vec::new(),
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        assert!(matches!(verdict, AdoptionVerdict::Escalate { .. }));
    }

    #[test]
    fn bench_non_finite_yields_escalate() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::Measured {
                median_pct: f64::NAN,
            },
            10,
            false,
            false,
            Vec::new(),
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        assert!(matches!(verdict, AdoptionVerdict::Escalate { .. }));
    }

    /// REQ-5: ポリシー除外リスト match は機械判定条件によらず無条件で
    /// エスカレーションへ回る（全ゲート green・全指標が閾値内でも）。
    #[test]
    fn exclusion_match_yields_escalate_even_when_all_signals_clean() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let ev = evidence(
            1,
            all_passed_gates(),
            BenchSignal::NotRun,
            10,
            false,
            false,
            vec!["arch-hyperparameter-change".to_string()],
        );

        let verdict = judge.judge(&ev).expect("judge should not error");
        match verdict {
            AdoptionVerdict::Escalate { reason } => {
                assert!(reason.contains("arch-hyperparameter-change"));
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    /// 矛盾入力（ゲート未全通過 + `BenchSignal::Measured`）は
    /// `guardrail::DecisionInput::new` が fail-closed に拒否し、本アダプタは
    /// これを `SelfRepairError::Judgement` として伝播する（attempt は
    /// `evidence.attempt()` と一致する）。
    #[test]
    fn inconsistent_input_is_rejected_fail_closed() {
        let judge = GuardrailAdoptionJudge::new(thresholds());
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let ev = evidence(
            7,
            gates,
            BenchSignal::Measured { median_pct: 1.0 },
            10,
            false,
            false,
            Vec::new(),
        );

        let error = judge
            .judge(&ev)
            .expect_err("inconsistent input should error");
        match error {
            SelfRepairError::Judgement { attempt, .. } => assert_eq!(attempt, 7),
            other => panic!("expected Judgement error, got {other:?}"),
        }
    }
}
