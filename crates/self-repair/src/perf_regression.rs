//! 性能回帰種別（[`RepairKind::PerfRegression`]）の検出・修正生成
//! （TASK-3.1b・イシュー #133・REQ-3/REQ-4。移植元は v1
//! `Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/perf.rs`。
//! `docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//!
//! [`crate::stages`] が定義した種別非依存 trait のうち、`Detector` と
//! `FixGenerator` を性能回帰専用に実装する。`VerificationGate`/`AdoptionJudge`
//! の実ゲート統合は #134（検証 3 ゲート実実行）・#135（guardrail 3 分岐判定
//! 統合）のスコープ。
//!
//! # PoC-2 題材 (b) との対応
//! `docs/spec/03-poc/poc-2-ai-self-maintenance/README.md:44-54` の
//! 「不要な再計算の挿入」は build/test/clippy を素通りし bench のみが検出する
//! （+63〜69% 劣化）。修正試行 1 は部分修正で不合格（+24〜30%）→ 却下・再試行
//! → 試行 2 で完全修正 → 取り込み、という 2 試行構造を持つ。
//! [`PerfRegressionFixGenerator::poc2_default`] はこの 2 段階の修正戦略を
//! そのまま写像した決定的な `Proposal` 列を返す（実 AI による修正生成の接続は
//! 本イシューのスコープ外。out-of-scope-tracking.md 準拠）。
//!
//! # 新 API 差し替え（TASK-3.1b の実体）
//! v1 はローカル定数（`DEFAULT_BENCH_MAX_PCT`/`DEFAULT_MIN_BENCH_RUNS`）と
//! 自前の中央値算出（`median_change_pct`）を持っていたが、v2 では
//! `guardrail::Thresholds::builtin(PresetName::Default)`
//! （`bench_median_max_pct`/`bench_runs_min`）・
//! `guardrail::median_gate::bench_signal_from_measurements`
//! （REQ-4「5 回以上」の下限検証・非有限値拒否・中央値算出を一元化した唯一の
//! 定義）へ置き換える。閾値・中央値算出ロジックの複製を持たないことで、
//! `.claude/rules/coding-rust.md`「バックエンド間許容誤差の定義を単独変更
//! しない」と同種の「定義の二重管理防止」を満たす。
//!
//! [`BenchMeasurer`] の契約も `Vec<BenchSample>`（v1）から `Vec<f64>`（%。
//! `guardrail::median_gate` の `measurements_pct` 語彙）へ揃える。
//!
//! # 本モジュールが担わない責務（out-of-scope-tracking.md 準拠）
//! - `cargo bench` 実測・5 回計測の実行そのもの → 実ベンチ計測系の接続
//!   （TASK-3.3・#136 系）。本モジュールは計測結果を受け取る [`BenchMeasurer`]
//!   trait（seam）のみ定義し、実装はテストのみで注入する。
//! - build/test/clippy/bench 実ゲート・`guardrail` 3 分岐判定との配線
//!   → #134/#135

use crate::error::SelfRepairError;
use crate::kind::RepairKind;
use crate::stages::{DetectionOutcome, Detector, Finding, FixGenerator, Proposal};

use guardrail::{BenchSignal, PresetName, Thresholds};

/// ベンチ計測への呼び出し口（seam）。
///
/// 「ベースライン比の変化率サンプル列（％）を返す」ことだけを契約とし、実際の
/// `cargo bench` 実行・Criterion 連携は持たない。実装は [`PerfRegressionDetector`]
/// へ注入し、本モジュールのテストでは決定的なサンプル列を返すテストダブルを
/// 使う（Criterion 実行を CI に持ち込まない）。`guardrail::median_gate` の
/// `measurements_pct: &[f64]` 語彙に揃えるため、戻り値は `Vec<f64>`（％。
/// 正 = 劣化、負 = 改善）とする。
pub trait BenchMeasurer {
    /// ベースラインとの比較計測を実行し、変化率サンプル列（％）を返す。
    ///
    /// 計測そのものの失敗（プロセス起動失敗・パース不能等）は
    /// [`SelfRepairError::Detection`] で返す（fail-closed。判定不能を
    /// 「劣化なし」に丸めない）。
    fn measure(&self) -> Result<Vec<f64>, SelfRepairError>;
}

/// 性能回帰種別の検出段階（[`Detector`] 実装）。
///
/// [`BenchMeasurer`] から得たサンプル列を
/// `guardrail::median_gate::bench_signal_from_measurements` で検証・中央値化し、
/// `thresholds.bench_median_max_pct` と比較する。中央値が閾値を超えていれば
/// [`DetectionOutcome::Finding`]、以内なら [`DetectionOutcome::NoActionNeeded`]
/// を返す。
///
/// # fail-closed の契約
/// - 計測自体の失敗（`measurer.measure()` の `Err`）はそのまま伝播する。
/// - サンプル数が `thresholds.bench_runs_min` 未満・非有限値混入は
///   `bench_signal_from_measurements` が `MedianGateError` として拒否し、
///   `NoActionNeeded` に丸めず `Err(SelfRepairError::Detection)` へ変換する
///   （REQ-4「5 回以上計測」違反を検出不能として握りつぶさない）。
/// - `kind` が `PerfRegression` 以外で呼ばれた場合も、対応外の種別に対して
///   誤って `NoActionNeeded` を返さないよう `Err` とする。
pub struct PerfRegressionDetector<M: BenchMeasurer> {
    measurer: M,
    thresholds: Thresholds,
}

impl<M: BenchMeasurer> PerfRegressionDetector<M> {
    /// `thresholds` はガードレール閾値そのもの（`guardrail::Thresholds`）。
    /// 単独緩和・新規定義はしない（`.claude/rules/security.md`）。
    pub fn new(measurer: M, thresholds: Thresholds) -> Self {
        PerfRegressionDetector {
            measurer,
            thresholds,
        }
    }

    /// ガードレール既定プリセット（`PresetName::Default`。5.0%・5 回）で構築する。
    pub fn with_default_thresholds(measurer: M) -> Self {
        PerfRegressionDetector::new(measurer, Thresholds::builtin(PresetName::Default))
    }
}

impl<M: BenchMeasurer> Detector for PerfRegressionDetector<M> {
    fn detect(&self, kind: RepairKind) -> Result<DetectionOutcome, SelfRepairError> {
        if kind != RepairKind::PerfRegression {
            return Err(SelfRepairError::Detection {
                kind: kind.as_machine_id(),
                reason: "PerfRegressionDetector は PerfRegression 種別のみを扱います".to_string(),
            });
        }

        let measurements = self.measurer.measure()?;

        let signal = guardrail::median_gate::bench_signal_from_measurements(&measurements)
            .map_err(|source| SelfRepairError::Detection {
                kind: RepairKind::PerfRegression.as_machine_id(),
                reason: source.to_string(),
            })?;

        // `bench_signal_from_measurements` は検証成功時に必ず `Measured` を
        // 返す契約だが（`guardrail::median_gate` の doc 参照）、`BenchSignal`
        // 自体は `NotRun` を持つ 2 variant の enum であるため、契約が破られた
        // 場合も NoActionNeeded に丸めず fail-closed で拒否する（`_ =>` を
        // 使わない網羅列挙。`.claude/rules/security.md` A05 と同じ考え方）。
        let median_pct = match signal {
            BenchSignal::Measured { median_pct } => median_pct,
            BenchSignal::NotRun => {
                return Err(SelfRepairError::Detection {
                    kind: RepairKind::PerfRegression.as_machine_id(),
                    reason:
                        "bench_signal_from_measurements が Measured を返しませんでした（契約違反）"
                            .to_string(),
                });
            }
        };

        if median_pct > self.thresholds.bench_median_max_pct {
            Ok(DetectionOutcome::Finding(Finding::new(
                RepairKind::PerfRegression,
                format!(
                    "ベンチ劣化中央値 {median_pct:.2}% が閾値 {:.2}% を超過しました（{} 回計測: {measurements:?}）",
                    self.thresholds.bench_median_max_pct,
                    measurements.len()
                ),
            )))
        } else {
            Ok(DetectionOutcome::NoActionNeeded)
        }
    }
}

/// 性能回帰種別の修正生成段階（[`FixGenerator`] 実装）。
///
/// PoC-2 題材 (b) の「部分修正 → 完全修正」という段階的試行を、attempt 番号
/// （1 始まり）で選択する決定的な戦略列として実装する。実 AI（LLM）による
/// 修正生成の接続は本イシューのスコープ外（out-of-scope-tracking.md 準拠。
/// 本クレートはハーネスとしての完走検証のみを担う）。
pub struct PerfRegressionFixGenerator {
    /// index 0 が attempt=1 の修正内容。戦略が尽きた attempt では
    /// `SelfRepairError::FixGeneration` を返す。
    strategies: Vec<String>,
}

impl PerfRegressionFixGenerator {
    pub fn new(strategies: Vec<String>) -> Self {
        PerfRegressionFixGenerator { strategies }
    }

    /// PoC-2 題材 (b) の実測（部分修正 +24〜30% → 完全修正で劣化解消）を
    /// そのまま写像した既定の 2 段階戦略。
    pub fn poc2_default() -> Self {
        PerfRegressionFixGenerator::new(vec![
            "部分修正: ホットパスの一部にのみキャッシュを追加（不要な再計算が一部残存）"
                .to_string(),
            "完全修正: 不要な再計算箇所を全て除去しベースライン相当の呼び出し回数に戻す"
                .to_string(),
        ])
    }
}

impl FixGenerator for PerfRegressionFixGenerator {
    fn generate(&self, finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
        if finding.kind() != RepairKind::PerfRegression {
            return Err(SelfRepairError::FixGeneration {
                attempt,
                reason:
                    "PerfRegressionFixGenerator は PerfRegression 種別の Finding のみを扱います"
                        .to_string(),
            });
        }

        // `attempt` は 1 始まり契約（stages.rs の `FixGenerator` doc・
        // `runner.rs` の `1..=max_attempts` ループ）だが、本番経路で
        // `attempt - 1` を無条件の usize 減算にすると、契約が破られて 0 が
        // 渡された場合に本番経路でパニック・アンダーフローを招く
        // （`.claude/rules/coding-rust.md`: 本番経路で unwrap/expect/パニックを
        // 起こさない）。`checked_sub` で型付きエラーに変換し、契約違反も
        // fail-closed で拒否する。
        let idx = match attempt.checked_sub(1) {
            Some(idx) => idx as usize,
            None => {
                return Err(SelfRepairError::FixGeneration {
                    attempt,
                    reason: "attempt は 1 始まりである必要があります（0 は不正な入力）".to_string(),
                });
            }
        };
        match self.strategies.get(idx) {
            Some(description) => Ok(Proposal {
                attempt,
                description: description.clone(),
            }),
            None => Err(SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "性能回帰種別の修正戦略が尽きました（試行 {attempt}・登録戦略数 {}）",
                    self.strategies.len()
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::outcome::{AdoptionVerdict, LoopOutcome, VerifiedEvidence};
    use crate::runner::SelfRepairLoop;
    use crate::stages::{AdoptionJudge, VerificationGate, VerificationOutcome};

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("テスト用の試行上限は正の値を渡すこと")
    }

    struct FixedMeasurer(Vec<f64>);
    impl BenchMeasurer for FixedMeasurer {
        fn measure(&self) -> Result<Vec<f64>, SelfRepairError> {
            Ok(self.0.clone())
        }
    }

    struct FailingMeasurer;
    impl BenchMeasurer for FailingMeasurer {
        fn measure(&self) -> Result<Vec<f64>, SelfRepairError> {
            Err(SelfRepairError::Detection {
                kind: RepairKind::PerfRegression.as_machine_id(),
                reason: "bench backend unavailable (scripted failure)".to_string(),
            })
        }
    }

    // --- 中央値ベース検出テスト ---

    #[test]
    fn median_above_threshold_is_detected_even_with_low_outliers() {
        // 中央値 +60% > 5%。平均・最大値ではなく中央値であることの証明対。
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            1.0, 2.0, 60.0, 61.0, 62.0,
        ]));
        let outcome = detector
            .detect(RepairKind::PerfRegression)
            .expect("detect should not error");
        assert!(matches!(outcome, DetectionOutcome::Finding(_)));
    }

    #[test]
    fn median_within_threshold_is_no_action_needed_despite_high_outliers() {
        // 中央値 +2% <= 5%。外れ値（+60・+61%）に引きずられず中央値で判定する。
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            60.0, 61.0, 2.0, 1.0, 1.0,
        ]));
        let outcome = detector
            .detect(RepairKind::PerfRegression)
            .expect("detect should not error");
        assert_eq!(outcome, DetectionOutcome::NoActionNeeded);
    }

    #[test]
    fn threshold_exactly_at_limit_is_not_a_regression() {
        // 中央値がちょうど 5.0%（閾値と同値）は超過ではないため回帰としない
        // （guardrail の「超過」判定と同じ境界の扱い）。
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            5.0, 5.0, 5.0, 5.0, 5.0,
        ]));
        let outcome = detector
            .detect(RepairKind::PerfRegression)
            .expect("detect should not error");
        assert_eq!(outcome, DetectionOutcome::NoActionNeeded);
    }

    // --- fail-closed テスト（下限・非有限値の検証は guardrail::median_gate 側） ---

    #[test]
    fn fewer_than_min_bench_runs_is_fail_closed_error() {
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            60.0, 61.0, 62.0, 63.0,
        ]));
        let error = detector
            .detect(RepairKind::PerfRegression)
            .expect_err("fewer than 5 samples must be rejected fail-closed");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn measurement_failure_propagates_as_error() {
        let detector = PerfRegressionDetector::with_default_thresholds(FailingMeasurer);
        let error = detector
            .detect(RepairKind::PerfRegression)
            .expect_err("measurer failure must propagate");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn nan_sample_is_rejected_fail_closed() {
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            1.0,
            2.0,
            f64::NAN,
            4.0,
            5.0,
        ]));
        let error = detector
            .detect(RepairKind::PerfRegression)
            .expect_err("NaN sample must be rejected fail-closed");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn wrong_kind_is_rejected_fail_closed() {
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            60.0, 61.0, 62.0, 63.0, 64.0,
        ]));
        let error = detector
            .detect(RepairKind::BugFix)
            .expect_err("mismatched kind must be rejected fail-closed");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    // --- FixGenerator の戦略列テスト ---

    #[test]
    fn fix_generator_exhausts_after_registered_strategies() {
        let generator = PerfRegressionFixGenerator::poc2_default();
        let finding = Finding::new(RepairKind::PerfRegression, "dummy");

        assert!(generator.generate(&finding, 1).is_ok());
        assert!(generator.generate(&finding, 2).is_ok());
        let error = generator
            .generate(&finding, 3)
            .expect_err("3rd attempt must exhaust the strategy list");
        assert!(matches!(
            error,
            SelfRepairError::FixGeneration { attempt: 3, .. }
        ));
    }

    #[test]
    fn fix_generator_rejects_mismatched_kind() {
        let generator = PerfRegressionFixGenerator::poc2_default();
        let finding = Finding::new(RepairKind::BugFix, "dummy");
        let error = generator
            .generate(&finding, 1)
            .expect_err("mismatched kind must be rejected fail-closed");
        assert!(matches!(error, SelfRepairError::FixGeneration { .. }));
    }

    #[test]
    fn fix_generator_rejects_zero_attempt_fail_closed() {
        // attempt=0 は契約違反（1 始まり）。`checked_sub` により本番経路の
        // 減算アンダーフロー・パニックを経ずに型付きエラーへ変換されることを
        // 確認する。
        let generator = PerfRegressionFixGenerator::poc2_default();
        let finding = Finding::new(RepairKind::PerfRegression, "dummy");
        let error = generator
            .generate(&finding, 0)
            .expect_err("attempt=0 must be rejected fail-closed");
        assert!(matches!(
            error,
            SelfRepairError::FixGeneration { attempt: 0, .. }
        ));
    }

    // --- 完走テスト ---

    /// PoC-2 題材 (b) の bench ゲート相当。**`proposal.description` の内容**
    /// で合否を決める（attempt 番号では決めない）。「完全修正」戦略の提案
    /// のみ合格とすることで、このテストが `PerfRegressionFixGenerator` の
    /// 戦略順序・内容を実際に検証する。
    struct DescriptionBasedBenchGate;
    impl VerificationGate for DescriptionBasedBenchGate {
        fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
            if proposal.description.contains("完全修正") {
                Ok(VerificationOutcome::Passed(VerifiedEvidence::new(
                    proposal.attempt,
                    proposal.description.clone(),
                    "gates: build=pass test=pass clippy=pass bench=pass(median<=5%)",
                    guardrail::GateSignals {
                        build: guardrail::GateSignal::Passed,
                        test: guardrail::GateSignal::Passed,
                        clippy: guardrail::GateSignal::Passed,
                    },
                    guardrail::BenchSignal::Measured { median_pct: 0.0 },
                    0,
                    false,
                    false,
                    Vec::new(),
                )))
            } else {
                Ok(VerificationOutcome::Failed {
                    reason: format!(
                        "attempt {} は部分修正のため bench 劣化が残存（scripted）",
                        proposal.attempt
                    ),
                })
            }
        }
    }

    struct AlwaysAdopt;
    impl AdoptionJudge for AlwaysAdopt {
        fn judge(&self, _evidence: &VerifiedEvidence) -> Result<AdoptionVerdict, SelfRepairError> {
            Ok(AdoptionVerdict::Adopt)
        }
    }

    #[test]
    fn perf_regression_loop_completes_via_partial_then_full_fix_poc2_material_b() {
        // PoC-2 題材 (b) 実測相当: 劣化 +63〜69%（5 回計測）を検出 →
        // 試行 1（部分修正）は bench ゲート不合格 → 却下・再試行 →
        // 試行 2（完全修正）で検証通過 → 取り込み。
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            63.0, 65.0, 67.0, 68.0, 69.0,
        ]));

        // Finding の summary に中央値が埋め込まれていることを確認する
        // （検出器が計測結果を握りつぶさず引き渡していることの証跡）。
        let finding = match detector
            .detect(RepairKind::PerfRegression)
            .expect("detect should not error")
        {
            DetectionOutcome::Finding(finding) => finding,
            DetectionOutcome::NoActionNeeded => panic!("regression must be detected"),
        };
        assert!(
            finding.summary.contains("67.00"),
            "summary must carry the median (67.0), got: {}",
            finding.summary
        );

        let fix_generator = PerfRegressionFixGenerator::poc2_default();
        let verification_gate = DescriptionBasedBenchGate;
        let adoption_judge = AlwaysAdopt;

        let loop_ = SelfRepairLoop::new(
            detector,
            fix_generator,
            verification_gate,
            adoption_judge,
            nz(3),
        );

        let report = loop_
            .run(RepairKind::PerfRegression)
            .expect("perf regression loop must complete without interruption");

        assert_eq!(report.outcome, LoopOutcome::Adopted);
        assert_eq!(report.attempt_count(), 2);
        assert!(matches!(
            report.attempts[0].outcome,
            crate::report::AttemptOutcome::VerificationFailed { .. }
        ));
        assert!(matches!(
            report.attempts[1].outcome,
            crate::report::AttemptOutcome::Adopted
        ));
    }

    #[test]
    fn perf_regression_below_threshold_short_circuits_without_any_attempt() {
        // 中央値が閾値以内なら検出段階で NoActionNeeded となり、修正生成・
        // 検証・取り込み判断のいずれも呼ばれない。
        let detector = PerfRegressionDetector::with_default_thresholds(FixedMeasurer(vec![
            1.0, 2.0, 1.0, 2.0, 1.0,
        ]));
        let fix_generator = PerfRegressionFixGenerator::poc2_default();
        let verification_gate = DescriptionBasedBenchGate;
        let adoption_judge = AlwaysAdopt;

        let loop_ = SelfRepairLoop::new(
            detector,
            fix_generator,
            verification_gate,
            adoption_judge,
            nz(3),
        );

        let report = loop_
            .run(RepairKind::PerfRegression)
            .expect("run should not error");
        assert_eq!(report.outcome, LoopOutcome::NoActionNeeded);
        assert_eq!(report.attempt_count(), 0);
    }
}
