//! 自己修復ループのオーケストレータ（TASK-3.1a・イシュー #132・REQ-3）。
//!
//! [`SelfRepairLoop`] は [`crate::stages`] の 4 trait を組み合わせ、PoC-2
//! （`docs/spec/03-poc/poc-2-ai-self-maintenance/README.md`）の「検出 → 修正
//! 試行（複数回。失敗時は却下して再試行）→ 検証 → 取り込み/却下」を種別非依存
//! に実行する。呼び出し元は guardrail 統合後の CLI（後続タスク）・検出/生成
//! 実装を担う #133・検証ゲート実実行を担う #134 の統合テストを想定する。

use std::num::NonZeroU32;
use std::time::Instant;

use crate::error::SelfRepairError;
use crate::kind::RepairKind;
use crate::outcome::{AdoptionVerdict, LoopOutcome};
use crate::report::{AttemptOutcome, AttemptRecord, LoopFailure, LoopReport};
use crate::stages::{
    AdoptionJudge, DetectionOutcome, Detector, FixGenerator, VerificationGate, VerificationOutcome,
};

/// 自己修復ループ 1 回分を実行するオーケストレータ。
///
/// 4 段階の実装をジェネリクスで受け取る（種別ごとの実装差し替えは呼び出し元が
/// 型パラメータを変えることで行う。種別別の `Detector`/`FixGenerator`
/// （#133 のスコープ）を実装しても本体のオーケストレーションは変更不要という
/// 設計上の意図）。
pub struct SelfRepairLoop<D, F, V, J>
where
    D: Detector,
    F: FixGenerator,
    V: VerificationGate,
    J: AdoptionJudge,
{
    detector: D,
    fix_generator: F,
    verification_gate: V,
    adoption_judge: J,
    /// 修正試行の上限回数（PoC-2: 際限のない再試行を許さない。閾値そのものは
    /// ガードレール設定の一部であり `.claude/rules/security.md` の「ガードレール
    /// 閾値の変更はユーザー承認必須」の対象。本イシューでは呼び出し元が
    /// コンストラクタで指定する単純な値として扱い、設定ファイル化はスコープ外
    /// とする）。`NonZeroU32` により「0 回では検出後に一切試行できず
    /// Exhausted の契約と矛盾する」という制約を型で保証し、本番経路での
    /// 実行時パニック（`assert!`/`unwrap`/`expect`）を使わずに済ませる
    /// （`.claude/rules/coding-rust.md`: 本番経路で unwrap/expect を使わない方針）。
    max_attempts: NonZeroU32,
}

impl<D, F, V, J> SelfRepairLoop<D, F, V, J>
where
    D: Detector,
    F: FixGenerator,
    V: VerificationGate,
    J: AdoptionJudge,
{
    pub fn new(
        detector: D,
        fix_generator: F,
        verification_gate: V,
        adoption_judge: J,
        max_attempts: NonZeroU32,
    ) -> Self {
        SelfRepairLoop {
            detector,
            fix_generator,
            verification_gate,
            adoption_judge,
            max_attempts,
        }
    }

    /// `kind` に対する自己修復ループを 1 回実行する。
    ///
    /// 段階の呼び出し順序は `stages.rs` モジュールコメントの契約に従う。
    /// [`crate::error::SelfRepairError`] は「段階の実行自体が失敗した」場合のみ返し、
    /// 「検証に落ちた」「却下された」という想定内の否定的結果は
    /// `Ok(LoopReport { outcome, .. })` として返す（呼び出し元が
    /// エラーハンドリングと業務判断を混同しないようにする）。
    ///
    /// エラー終了時（[`LoopFailure`]）も、それまでに蓄積した試行記録
    /// （`attempts`）を保持したまま返す（`Result<LoopReport, SelfRepairError>`
    /// のみでは early return とともに過去の試行履歴が失われ、
    /// `.claude/rules/security.md` の「取り込み判断の根拠を追跡可能にする」
    /// という要求を満たせないため。v1 イシュー #40 レビュー指摘を踏襲）。
    pub fn run(&self, kind: RepairKind) -> Result<LoopReport, LoopFailure> {
        let loop_start = Instant::now();

        let finding = match self.detector.detect(kind).map_err(|error| LoopFailure {
            error,
            attempts: Vec::new(),
        })? {
            DetectionOutcome::NoActionNeeded => {
                return Ok(LoopReport {
                    kind,
                    outcome: LoopOutcome::NoActionNeeded,
                    attempts: Vec::new(),
                    total_duration: loop_start.elapsed(),
                });
            }
            DetectionOutcome::Finding(finding) => finding,
        };

        // `LoopReport.kind` は引数 `kind`（呼び出し元が指定した種別）をそのまま
        // 記録する一方、後続の `fix_generator`/`verification_gate`/`adoption_judge`
        // は `finding`（`detector.detect` の戻り値）を消費する。`Detector` の
        // 実装（#133 のスコープ）が引数と異なる種別の `Finding` を誤って返すと、
        // 報告される repair kind（`LoopReport.kind`）と実際に行われた作業
        // （`finding.kind()` が示す種別）が静かに乖離しうる。`proposal.attempt`・
        // `evidence.attempt()` に対して既に設けている「単一の真実源からの逸脱を
        // fail-closed で検出する」監査証跡保護（上記コメント参照）と同種の
        // 問題であるため、ここでも段階実行自体の失敗として扱う（レビュー指摘）。
        if finding.kind() != kind {
            return Err(LoopFailure {
                error: SelfRepairError::Detection {
                    kind: kind.as_machine_id(),
                    reason: format!(
                        "detector が kind={} の呼び出しに対し finding.kind()={} を \
                         返しました（repair kind の単一の真実源が破られています）",
                        kind.as_machine_id(),
                        finding.kind().as_machine_id()
                    ),
                },
                attempts: Vec::new(),
            });
        }

        let mut attempts = Vec::new();

        for attempt in 1..=self.max_attempts.get() {
            let attempt_start = Instant::now();

            let proposal = self
                .fix_generator
                .generate(&finding, attempt)
                .map_err(|error| LoopFailure {
                    error,
                    attempts: attempts.clone(),
                })?;

            // attempt 番号の単一の真実源はループ側のカウンタ（`attempt`）である。
            // `Proposal.attempt`（`FixGenerator::generate` が返す pub フィールド）
            // は本来この値と一致するはずだが、型で強制されておらず、将来の
            // `FixGenerator` 実装（#133）が誤った/古い値を設定すると
            // `LoopReport.attempts`（ループ側の attempt を記録）と
            // `SelfRepairError::Judgement { attempt }`（`VerifiedEvidence::attempt()`
            // 経由で Proposal 由来の値を記録）の attempt 番号が食い違い、
            // `.claude/rules/security.md` の「取り込み判断の根拠を追跡可能に
            // する」を静かに損ないうる（レビュー指摘）。ここで不一致を
            // fail-closed で検出し、段階実行自体の失敗として扱う。
            if proposal.attempt != attempt {
                return Err(LoopFailure {
                    error: SelfRepairError::FixGeneration {
                        attempt,
                        reason: format!(
                            "fix generator が attempt={attempt} の呼び出しに対し \
                             proposal.attempt={} を返しました（attempt 番号の単一の \
                             真実源が破られています）",
                            proposal.attempt
                        ),
                    },
                    attempts: attempts.clone(),
                });
            }

            match self
                .verification_gate
                .verify(&proposal)
                .map_err(|error| LoopFailure {
                    error,
                    attempts: attempts.clone(),
                })? {
                VerificationOutcome::Failed { reason } => {
                    attempts.push(AttemptRecord {
                        attempt,
                        duration: attempt_start.elapsed(),
                        outcome: AttemptOutcome::VerificationFailed { reason },
                    });
                    // PoC-2: 検証落ちは取り込み判断へ進めず、再試行する
                    // （AdoptionJudge を呼ばないことが「検証を経ない取り込み
                    // 経路を作らない」契約の実行時側の担保でもある）。
                    continue;
                }
                VerificationOutcome::Passed(evidence) => {
                    // `VerifiedEvidence::new` の `attempt` はコンストラクタの
                    // 自由な `u32` 引数であり、`Proposal.attempt` から自動導出
                    // されるわけではない（#134 の実実装が誤った値を渡しうる）。
                    // `SelfRepairError::Judgement { attempt: evidence.attempt() }`
                    // はこの値をそのまま使うため、`Proposal.attempt` の検査
                    // （上記）だけでは監査ログの食い違いを防ぎきれない。ループ側
                    // の attempt カウンタを単一の真実源として、ここでも
                    // 不一致を fail-closed で検出する。
                    if evidence.attempt() != attempt {
                        return Err(LoopFailure {
                            error: SelfRepairError::Verification {
                                attempt,
                                reason: format!(
                                    "verification gate が attempt={attempt} の呼び出しに対し \
                                     evidence.attempt()={} を返しました（attempt 番号の単一の \
                                     真実源が破られています）",
                                    evidence.attempt()
                                ),
                            },
                            attempts: attempts.clone(),
                        });
                    }

                    match self
                        .adoption_judge
                        .judge(&evidence)
                        .map_err(|error| LoopFailure {
                            error,
                            attempts: attempts.clone(),
                        })? {
                        AdoptionVerdict::Adopt => {
                            attempts.push(AttemptRecord {
                                attempt,
                                duration: attempt_start.elapsed(),
                                outcome: AttemptOutcome::Adopted,
                            });
                            return Ok(LoopReport {
                                kind,
                                outcome: LoopOutcome::Adopted,
                                attempts,
                                total_duration: loop_start.elapsed(),
                            });
                        }
                        AdoptionVerdict::Escalate { reason } => {
                            attempts.push(AttemptRecord {
                                attempt,
                                duration: attempt_start.elapsed(),
                                outcome: AttemptOutcome::Escalated {
                                    reason: reason.clone(),
                                },
                            });
                            return Ok(LoopReport {
                                kind,
                                outcome: LoopOutcome::Escalated { reason },
                                attempts,
                                total_duration: loop_start.elapsed(),
                            });
                        }
                        AdoptionVerdict::Reject {
                            retryable: true,
                            reason,
                        } => {
                            attempts.push(AttemptRecord {
                                attempt,
                                duration: attempt_start.elapsed(),
                                outcome: AttemptOutcome::AdoptionRejectedRetryable { reason },
                            });
                            continue;
                        }
                        AdoptionVerdict::Reject {
                            retryable: false,
                            reason,
                        } => {
                            attempts.push(AttemptRecord {
                                attempt,
                                duration: attempt_start.elapsed(),
                                outcome: AttemptOutcome::RejectedFinal {
                                    reason: reason.clone(),
                                },
                            });
                            return Ok(LoopReport {
                                kind,
                                outcome: LoopOutcome::Rejected {
                                    stage: "adoption_judge",
                                    reason,
                                },
                                attempts,
                                total_duration: loop_start.elapsed(),
                            });
                        }
                    }
                }
            }
        }

        Ok(LoopReport {
            kind,
            outcome: LoopOutcome::Exhausted,
            attempts,
            total_duration: loop_start.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;
    use crate::error::SelfRepairError;
    use crate::outcome::VerifiedEvidence;
    use crate::stages::{Finding, Proposal};

    /// テスト用の `NonZeroU32` 構築ヘルパー。`unwrap` はテストコード限定
    /// （`.claude/rules/coding-rust.md` の unwrap/expect 禁止は本番経路が対象）。
    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("テスト用の試行上限は正の値を渡すこと")
    }

    /// 検出段階のテストダブル。固定の [`DetectionOutcome`] を返す。
    struct FixedDetector(DetectionOutcome);
    impl Detector for FixedDetector {
        fn detect(&self, _kind: RepairKind) -> Result<DetectionOutcome, SelfRepairError> {
            Ok(self.0.clone())
        }
    }

    /// 修正生成段階のテストダブル。試行回数を記録するだけで内容は固定。
    struct RecordingFixGenerator {
        calls: RefCell<Vec<u32>>,
    }
    impl RecordingFixGenerator {
        fn new() -> Self {
            RecordingFixGenerator {
                calls: RefCell::new(Vec::new()),
            }
        }
    }
    impl FixGenerator for RecordingFixGenerator {
        fn generate(&self, finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
            self.calls.borrow_mut().push(attempt);
            Ok(Proposal {
                attempt,
                description: format!("fix for {}", finding.summary),
            })
        }
    }

    /// 検証段階のテストダブル。`attempt` ごとに事前設定した結果を返す
    /// （1 始まりの試行番号でスクリプト化する。範囲外は毎回不合格扱い）。
    struct ScriptedVerificationGate {
        /// index 0 が attempt=1 の結果。
        script: Vec<bool>,
        judge_input_calls: RefCell<Vec<u32>>,
    }
    impl ScriptedVerificationGate {
        fn new(script: Vec<bool>) -> Self {
            ScriptedVerificationGate {
                script,
                judge_input_calls: RefCell::new(Vec::new()),
            }
        }

        /// 検証を通過し [`VerifiedEvidence`] を発行した試行番号の記録
        /// （迂回不能性テストが「検証不合格の試行では証跡が一切発行されない
        /// = AdoptionJudge へ到達しない」ことを確認するために使う）。
        fn passed_attempts(&self) -> Vec<u32> {
            self.judge_input_calls.borrow().clone()
        }
    }
    impl VerificationGate for ScriptedVerificationGate {
        fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
            let idx = (proposal.attempt - 1) as usize;
            let passed = self.script.get(idx).copied().unwrap_or(false);
            if passed {
                self.judge_input_calls.borrow_mut().push(proposal.attempt);
                // runner.rs のオーケストレーション（呼び出し順序・迂回不能性）
                // のみを検証するテストダブルであり、guardrail 統合後の 3 分岐
                // 判定の中身は #135 が別途検証するため、ここでは
                // guardrail 非依存の S1 形（本イシューのスコープ）の証跡を積む。
                Ok(VerificationOutcome::Passed(VerifiedEvidence::new(
                    proposal.attempt,
                    proposal.description.clone(),
                    "gates: build=pass test=pass clippy=pass",
                )))
            } else {
                Ok(VerificationOutcome::Failed {
                    reason: format!(
                        "attempt {} verification failed (scripted)",
                        proposal.attempt
                    ),
                })
            }
        }
    }

    /// 取り込み判断段階のテストダブル。呼び出し回数を数え、固定の
    /// [`AdoptionVerdict`] を返す（迂回不能性の検証に呼び出し回数を使う）。
    struct FixedAdoptionJudge {
        verdict_for_first_call: AdoptionVerdict,
        call_count: Cell<u32>,
    }
    impl FixedAdoptionJudge {
        fn always(verdict: AdoptionVerdict) -> Self {
            FixedAdoptionJudge {
                verdict_for_first_call: verdict,
                call_count: Cell::new(0),
            }
        }
    }
    impl AdoptionJudge for FixedAdoptionJudge {
        fn judge(&self, _evidence: &VerifiedEvidence) -> Result<AdoptionVerdict, SelfRepairError> {
            self.call_count.set(self.call_count.get() + 1);
            Ok(self.verdict_for_first_call.clone())
        }
    }

    /// 取り込み判断段階のテストダブル。呼び出し順に固定の
    /// [`AdoptionVerdict`] 列を返す（`FixedAdoptionJudge` は毎回同じ verdict
    /// しか返せず、`Reject { retryable: true }` → 再試行 → 別 verdict という
    /// runner.rs の状態機械の分岐を検証できないため、これを補うテストダブル。
    /// レビュー指摘: `AdoptionVerdict::Reject { retryable: true }` の再試行
    /// 分岐が未テストだった）。
    struct ScriptedAdoptionJudge {
        verdicts: RefCell<std::collections::VecDeque<AdoptionVerdict>>,
    }
    impl ScriptedAdoptionJudge {
        fn new(verdicts: Vec<AdoptionVerdict>) -> Self {
            ScriptedAdoptionJudge {
                verdicts: RefCell::new(verdicts.into()),
            }
        }
    }
    impl AdoptionJudge for ScriptedAdoptionJudge {
        fn judge(&self, _evidence: &VerifiedEvidence) -> Result<AdoptionVerdict, SelfRepairError> {
            Ok(self
                .verdicts
                .borrow_mut()
                .pop_front()
                .unwrap_or(AdoptionVerdict::Escalate {
                    reason: "スクリプトの verdict が尽きました（テスト設定不備）".to_string(),
                }))
        }
    }

    fn finding(kind: RepairKind) -> Finding {
        Finding::new(kind, "dummy finding for test")
    }

    /// 検出段階のテストダブル。常に [`SelfRepairError::Detection`] を返す
    /// （4 trait のうち「段階の実行自体が失敗する」経路の検証用）。
    struct FailingDetector;
    impl Detector for FailingDetector {
        fn detect(&self, _kind: RepairKind) -> Result<DetectionOutcome, SelfRepairError> {
            Err(SelfRepairError::Detection {
                kind: "test",
                reason: "detection backend unavailable (scripted failure)".to_string(),
            })
        }
    }

    /// 修正生成段階のテストダブル。`fail_attempt` に一致する試行番号でのみ
    /// [`SelfRepairError::FixGeneration`] を返し、それ以外は成功する
    /// （エラー発生前に蓄積された attempts の検証に使う）。
    struct FixGeneratorFailsOnAttempt {
        fail_attempt: u32,
    }
    impl FixGenerator for FixGeneratorFailsOnAttempt {
        fn generate(&self, finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
            if attempt == self.fail_attempt {
                Err(SelfRepairError::FixGeneration {
                    attempt,
                    reason: "fix generation backend unavailable (scripted failure)".to_string(),
                })
            } else {
                Ok(Proposal {
                    attempt,
                    description: format!("fix for {}", finding.summary),
                })
            }
        }
    }

    /// 検証段階のテストダブル。`fail_attempt` に一致する試行番号でのみ
    /// [`SelfRepairError::Verification`] を返し、それ未満は検証不合格
    /// （`Failed`）を返す（`fix_generation_error_preserves_prior_attempt_history`
    /// と同様、エラー発生前に蓄積された attempts の検証に使う）。
    struct VerificationGateFailsOnAttempt {
        fail_attempt: u32,
    }
    impl VerificationGate for VerificationGateFailsOnAttempt {
        fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
            if proposal.attempt == self.fail_attempt {
                Err(SelfRepairError::Verification {
                    attempt: proposal.attempt,
                    reason: "verification gate spawn failed (scripted failure)".to_string(),
                })
            } else {
                Ok(VerificationOutcome::Failed {
                    reason: format!(
                        "attempt {} verification failed (scripted)",
                        proposal.attempt
                    ),
                })
            }
        }
    }

    /// 修正生成段階のテストダブル。ループが渡した `attempt` とは無関係に
    /// `wrong_attempt` を `Proposal.attempt` に設定する（attempt 番号の単一の
    /// 真実源の不一致検出テスト用。レビュー指摘 Medium #1 の
    /// `Proposal.attempt` 側の回帰防止）。
    struct FixGeneratorReturnsWrongAttempt {
        wrong_attempt: u32,
    }
    impl FixGenerator for FixGeneratorReturnsWrongAttempt {
        fn generate(&self, finding: &Finding, _attempt: u32) -> Result<Proposal, SelfRepairError> {
            Ok(Proposal {
                attempt: self.wrong_attempt,
                description: format!("fix for {}", finding.summary),
            })
        }
    }

    /// 検証段階のテストダブル。検証は常に合格させるが、
    /// [`VerifiedEvidence::new`] に渡す `attempt` をループが渡した
    /// `proposal.attempt` とは無関係な `wrong_attempt` にすり替える
    /// （attempt 番号の単一の真実源の不一致検出テスト用。レビュー指摘
    /// Medium #1 の `evidence.attempt()` 側の回帰防止。`VerifiedEvidence::new`
    /// が `pub(crate)` のため本クレート内のテストダブルから直接呼べる
    /// ことが、outcome.rs の doc が説明する「クレート境界までしか型で
    /// 強制されない」ことの実地証明でもある）。
    struct VerificationGateReturnsWrongAttemptEvidence {
        wrong_attempt: u32,
    }
    impl VerificationGate for VerificationGateReturnsWrongAttemptEvidence {
        fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
            Ok(VerificationOutcome::Passed(VerifiedEvidence::new(
                self.wrong_attempt,
                proposal.description.clone(),
                "gates: build=pass test=pass clippy=pass",
            )))
        }
    }

    /// 取り込み判断段階のテストダブル。常に [`SelfRepairError::Judgement`] を返す。
    ///
    /// `attempt` を固定値ではなく `evidence.attempt()` から取る（v1 PR #170 での
    /// Bugbot 指摘対応を踏襲）。`judge` は `VerifiedEvidence` 以外から試行番号を
    /// 知る手段を持たないという契約（stages.rs の `AdoptionJudge` doc 参照）を
    /// テストダブル側でも検証できるよう、実装として正しい参照方法を示す。
    struct FailingAdoptionJudge;
    impl AdoptionJudge for FailingAdoptionJudge {
        fn judge(&self, evidence: &VerifiedEvidence) -> Result<AdoptionVerdict, SelfRepairError> {
            Err(SelfRepairError::Judgement {
                attempt: evidence.attempt(),
                reason: "adoption judge backend unavailable (scripted failure)".to_string(),
            })
        }
    }

    #[test]
    fn happy_path_reaches_adopted_on_first_attempt() {
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let report = loop_.run(RepairKind::BugFix).expect("run should not error");
        assert_eq!(report.outcome, LoopOutcome::Adopted);
        assert_eq!(report.attempt_count(), 1);
        assert!(matches!(
            report.attempts[0].outcome,
            AttemptOutcome::Adopted
        ));
    }

    #[test]
    fn verification_failure_is_rejected_and_retried_then_adopted() {
        // attempt 1: 検証不合格 → 却下して再試行、attempt 2: 検証合格 → 採用
        // （PoC-2 題材 (a) の「試行 1 失敗 → 却下 → 試行 2」構造）。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![false, true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let report = loop_.run(RepairKind::BugFix).expect("run should not error");
        assert_eq!(report.outcome, LoopOutcome::Adopted);
        assert_eq!(report.attempt_count(), 2);
        assert!(matches!(
            report.attempts[0].outcome,
            AttemptOutcome::VerificationFailed { .. }
        ));
        assert!(matches!(
            report.attempts[1].outcome,
            AttemptOutcome::Adopted
        ));
    }

    #[test]
    fn exhausts_after_max_attempts_when_verification_always_fails() {
        let judge = FixedAdoptionJudge::always(AdoptionVerdict::Adopt);
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(
                RepairKind::PerfRegression,
            ))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![false, false, false]),
            judge,
            nz(3),
        );

        let report = loop_
            .run(RepairKind::PerfRegression)
            .expect("run should not error");
        assert_eq!(report.outcome, LoopOutcome::Exhausted);
        assert_eq!(report.attempt_count(), 3);
        // 迂回不能性: 検証が一度も合格しないので VerifiedEvidence は一度も
        // 発行されず（passed_attempts が空）、取り込み判断も一度も呼ばれない
        // （AdoptionJudge::judge へ到達する経路が verify の Passed 分岐にしか
        // 存在しないことの実行時側の担保）。
        let SelfRepairLoop {
            verification_gate,
            adoption_judge,
            ..
        } = &loop_;
        assert!(verification_gate.passed_attempts().is_empty());
        assert_eq!(adoption_judge.call_count.get(), 0);
    }

    #[test]
    fn no_action_needed_short_circuits_before_any_attempt() {
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::NoActionNeeded),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let report = loop_
            .run(RepairKind::FeatureAddition)
            .expect("run should not error");
        assert_eq!(report.outcome, LoopOutcome::NoActionNeeded);
        assert_eq!(report.attempt_count(), 0);
        assert_eq!(loop_.fix_generator.calls.borrow().len(), 0);
    }

    #[test]
    fn non_retryable_rejection_terminates_immediately() {
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true, true, true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Reject {
                retryable: false,
                reason: "対応不能と判明".to_string(),
            }),
            nz(3),
        );

        let report = loop_.run(RepairKind::BugFix).expect("run should not error");
        assert_eq!(report.attempt_count(), 1);
        assert!(matches!(
            report.outcome,
            LoopOutcome::Rejected {
                stage: "adoption_judge",
                ..
            }
        ));
    }

    #[test]
    fn retryable_rejection_retries_and_then_adopts() {
        // attempt 1: 検証合格するが取り込み判断が再試行可能な却下を返す
        // → AdoptionRejectedRetryable を記録して次の試行へ、attempt 2:
        // 検証合格・取り込み判断が承認 → Adopted。
        // レビュー指摘: `Reject { retryable: true }` → continue 分岐が
        // 12 件のテストのいずれからも経由されていなかった。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true, true]),
            ScriptedAdoptionJudge::new(vec![
                AdoptionVerdict::Reject {
                    retryable: true,
                    reason: "一時的な要因のため再試行可能".to_string(),
                },
                AdoptionVerdict::Adopt,
            ]),
            nz(3),
        );

        let report = loop_.run(RepairKind::BugFix).expect("run should not error");
        assert_eq!(report.outcome, LoopOutcome::Adopted);
        assert_eq!(report.attempt_count(), 2);
        assert!(matches!(
            report.attempts[0].outcome,
            AttemptOutcome::AdoptionRejectedRetryable { .. }
        ));
        assert!(matches!(
            report.attempts[1].outcome,
            AttemptOutcome::Adopted
        ));
    }

    #[test]
    fn escalation_terminates_immediately_without_retry() {
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(
                RepairKind::FeatureAddition,
            ))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true, true, true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Escalate {
                reason: "人間レビューが必要".to_string(),
            }),
            nz(3),
        );

        let report = loop_
            .run(RepairKind::FeatureAddition)
            .expect("run should not error");
        assert_eq!(report.attempt_count(), 1);
        assert_eq!(
            report.outcome,
            LoopOutcome::Escalated {
                reason: "人間レビューが必要".to_string()
            }
        );
    }

    #[test]
    fn runner_is_kind_agnostic_across_all_repair_kinds() {
        for kind in [
            RepairKind::BugFix,
            RepairKind::PerfRegression,
            RepairKind::FeatureAddition,
        ] {
            let loop_ = SelfRepairLoop::new(
                FixedDetector(DetectionOutcome::Finding(finding(kind))),
                RecordingFixGenerator::new(),
                ScriptedVerificationGate::new(vec![true]),
                FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
                nz(3),
            );

            let report = loop_.run(kind).expect("run should not error");
            assert_eq!(report.outcome, LoopOutcome::Adopted);
            assert_eq!(report.kind, kind);
        }
    }

    #[test]
    fn detection_error_propagates_with_no_prior_attempts() {
        // 検出段階の失敗はループ開始前なので attempts は必ず空
        // （v1 レビュー指摘: 段階の Err 経路が未検証だった領域）。
        let loop_ = SelfRepairLoop::new(
            FailingDetector,
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("detection failure should propagate as Err");
        assert!(matches!(failure.error, SelfRepairError::Detection { .. }));
        assert!(failure.attempts.is_empty());
    }

    #[test]
    fn fix_generation_error_preserves_prior_attempt_history() {
        // attempt 1: 検証不合格で却下・再試行、attempt 2: 修正生成自体が失敗。
        // レビュー指摘の core: attempt 1 の VerificationFailed 記録が
        // LoopFailure::attempts に残ることを確認する。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            FixGeneratorFailsOnAttempt { fail_attempt: 2 },
            ScriptedVerificationGate::new(vec![false]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("fix generation failure should propagate as Err");
        assert!(matches!(
            failure.error,
            SelfRepairError::FixGeneration { attempt: 2, .. }
        ));
        assert_eq!(failure.attempts.len(), 1);
        assert!(matches!(
            failure.attempts[0].outcome,
            AttemptOutcome::VerificationFailed { .. }
        ));
    }

    #[test]
    fn fix_generator_attempt_mismatch_is_rejected_fail_closed() {
        // attempt 1 から FixGenerator がループの attempt（1）とは異なる
        // proposal.attempt（99）を返す。レビュー指摘 Medium #1: attempt 番号の
        // 単一の真実源（ループ側のカウンタ）と Proposal.attempt の不一致を
        // fail-closed で検出することを確認する回帰テスト。verification_gate
        // へは到達しないため attempts は空のまま。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            FixGeneratorReturnsWrongAttempt { wrong_attempt: 99 },
            ScriptedVerificationGate::new(vec![true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("attempt mismatch should propagate as Err");
        assert!(matches!(
            failure.error,
            SelfRepairError::FixGeneration { attempt: 1, .. }
        ));
        assert!(failure.attempts.is_empty());
    }

    #[test]
    fn detector_kind_mismatch_is_rejected_fail_closed() {
        // 呼び出し引数は BugFix だが、detector が誤って PerfRegression の
        // Finding を返す。Bugbot 指摘: `run` が `finding.kind()` と引数
        // `kind` の一致を検査していなかったため、`LoopReport.kind`（引数を
        // そのまま記録）と後続処理が実際に消費する `finding` の種別が静かに
        // 乖離しうる回帰。`proposal.attempt`/`evidence.attempt()` と同種の
        // fail-closed 検出を確認する。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(
                RepairKind::PerfRegression,
            ))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![true]),
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("kind mismatch should propagate as Err");
        assert!(matches!(failure.error, SelfRepairError::Detection { .. }));
        assert!(failure.attempts.is_empty());
        assert_eq!(loop_.fix_generator.calls.borrow().len(), 0);
    }

    #[test]
    fn verification_evidence_attempt_mismatch_is_rejected_fail_closed() {
        // attempt 1 で検証が合格するが、VerifiedEvidence が保持する
        // attempt（99）がループの attempt（1）と食い違う。レビュー指摘
        // Medium #1 の `evidence.attempt()` 側（advisor 追加指摘）: この
        // 不一致も fail-closed で検出し、AdoptionJudge へは到達しないことを
        // 確認する回帰テスト。
        let judge = FixedAdoptionJudge::always(AdoptionVerdict::Adopt);
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            VerificationGateReturnsWrongAttemptEvidence { wrong_attempt: 99 },
            judge,
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("evidence attempt mismatch should propagate as Err");
        assert!(matches!(
            failure.error,
            SelfRepairError::Verification { attempt: 1, .. }
        ));
        assert!(failure.attempts.is_empty());
        let SelfRepairLoop { adoption_judge, .. } = &loop_;
        assert_eq!(adoption_judge.call_count.get(), 0);
    }

    #[test]
    fn verification_error_propagates_and_preserves_prior_attempt_history() {
        // attempt 1: 検証不合格で却下・再試行、attempt 2: 検証段階自体が失敗。
        // fix_generation 系のテストと同じ観点で、verify 呼び出しでも過去の
        // attempts が LoopFailure に残ることを確認する。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            VerificationGateFailsOnAttempt { fail_attempt: 2 },
            FixedAdoptionJudge::always(AdoptionVerdict::Adopt),
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("verification failure should propagate as Err");
        assert!(matches!(
            failure.error,
            SelfRepairError::Verification { attempt: 2, .. }
        ));
        assert_eq!(failure.attempts.len(), 1);
        assert!(matches!(
            failure.attempts[0].outcome,
            AttemptOutcome::VerificationFailed { .. }
        ));
    }

    #[test]
    fn judgement_error_propagates_and_preserves_prior_attempt_history() {
        // attempt 1: 検証不合格で却下・再試行、attempt 2: 検証合格して
        // 取り込み判断へ進むが judge 自体が失敗する。judge 呼び出しでも過去の
        // attempts が LoopFailure に残ることを確認する（未検証だった 3
        // 呼び出し site の最後の 1 つ）。
        let loop_ = SelfRepairLoop::new(
            FixedDetector(DetectionOutcome::Finding(finding(RepairKind::BugFix))),
            RecordingFixGenerator::new(),
            ScriptedVerificationGate::new(vec![false, true]),
            FailingAdoptionJudge,
            nz(3),
        );

        let failure = loop_
            .run(RepairKind::BugFix)
            .expect_err("judgement failure should propagate as Err");
        // v1 PR #170 での Bugbot 指摘の回帰防止: judge が失敗するのは
        // attempt=2（1 は検証不合格で AdoptionJudge に到達すらしない）。
        // `VerifiedEvidence::attempt()` 経由でなければ 2 を報告できないため、
        // 固定値へ戻す変更をここで検出する。
        assert!(matches!(
            failure.error,
            SelfRepairError::Judgement { attempt: 2, .. }
        ));
        assert_eq!(failure.attempts.len(), 1);
        assert!(matches!(
            failure.attempts[0].outcome,
            AttemptOutcome::VerificationFailed { .. }
        ));
    }
}
