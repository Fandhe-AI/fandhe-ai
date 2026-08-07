//! 検証フェーズ 3 ゲート（build/test/clippy）の実実行（TASK-3.1c・
//! イシュー #134・REQ-3）。
//!
//! [`crate::stages::VerificationGate`] の実装 [`CargoVerificationGate`] を
//! 提供する。移植元は v1 `Fandhe-AI/rust-ai-library-v1`
//! `tools/self-repair/src/cargo_gate.rs`（`docs/spec/v1-assets-inventory.md`
//! L17「改修して再利用」判定）。
//!
//! # スコープ境界
//! - 検証フェーズ 4 ゲートのうちベンチゲート（[`crate::verify_bench::SelfRepairBenchGate`]）
//!   は本ゲートに含まない。ベンチゲートの `VerificationGate` への結線
//!   （4 ゲート合成）は #136 系（TASK-3.2）のスコープ。
//! - guardrail 3 分岐判定との統合・[`crate::outcome::VerifiedEvidence`] への
//!   構造化シグナル（`gates`/`bench`/`lines_changed` 等）追加は #135
//!   （TASK-3.1d）のスコープ。本ゲートは guardrail 非依存の v1 S1 形
//!   （`attempt`・`proposal_summary`・`gate_report` の 3 フィールド）のまま
//!   [`VerifiedEvidence`] を発行する。
//!
//! # fail-closed 契約
//! 3 ゲートは build → test --release → clippy の順で逐次実行し、いずれかが
//! 失敗した時点で後続ゲートを実行せず [`crate::stages::VerificationOutcome::Failed`]
//! を返す。spawn 自体の失敗（[`crate::exec::CommandRunner::run`] が `Err`
//! を返す場合）は `Failed`/`Passed` のどちらにも丸めず
//! [`crate::error::SelfRepairError::Verification`] として伝播する
//! （`.claude/rules/security.md` A08: 判定不能と否定的結果を混同しない）。

use std::path::PathBuf;

use crate::error::SelfRepairError;
use crate::exec::CommandRunner;
use crate::outcome::VerifiedEvidence;
use crate::stages::{Proposal, VerificationGate, VerificationOutcome};

/// 1 ゲートの実行結果（内部処理用。`?` による早期リターンに使う）。
enum GateStep {
    Passed,
    Failed {
        gate_name: &'static str,
        log_tail: String,
    },
}

/// `cargo build`/`cargo test --release`/`cargo clippy` の 3 ゲートを逐次
/// 実行する [`VerificationGate`] 実装。
///
/// `R: CommandRunner` はテスト時にスクリプト化したテストダブルを注入できる
/// よう総称化している（本番経路は [`crate::exec::SystemCommandRunner`]）。
pub struct CargoVerificationGate<R: CommandRunner> {
    workspace: PathBuf,
    runner: R,
}

impl<R: CommandRunner> CargoVerificationGate<R> {
    pub fn new(workspace: PathBuf, runner: R) -> Self {
        CargoVerificationGate { workspace, runner }
    }

    /// 1 ゲートを実行し、spawn 失敗は `Err`、非 0 終了は `Ok(GateStep::Failed)`
    /// として返す（`verify` 側で `?` を使い spawn 失敗を早期リターンできる
    /// ようにするための内部ヘルパー）。
    fn run_gate(
        &self,
        gate_name: &'static str,
        args: &[&str],
        proposal: &Proposal,
    ) -> Result<GateStep, SelfRepairError> {
        let output = self
            .runner
            .run("cargo", args, &self.workspace)
            .map_err(|error| SelfRepairError::Verification {
                attempt: proposal.attempt,
                reason: format!("{gate_name} ゲートの起動に失敗しました: {error}"),
            })?;

        if output.success() {
            Ok(GateStep::Passed)
        } else {
            Ok(GateStep::Failed {
                gate_name,
                log_tail: output.log_tail().to_string(),
            })
        }
    }
}

impl<R: CommandRunner> VerificationGate for CargoVerificationGate<R> {
    fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
        // 1. build
        match self.run_gate("build", &["build"], proposal)? {
            GateStep::Passed => {}
            GateStep::Failed {
                gate_name,
                log_tail,
            } => {
                return Ok(VerificationOutcome::Failed {
                    reason: format!("{gate_name} が失敗しました: {log_tail}"),
                });
            }
        }

        // 2. test --release
        match self.run_gate("test", &["test", "--release"], proposal)? {
            GateStep::Passed => {}
            GateStep::Failed {
                gate_name,
                log_tail,
            } => {
                return Ok(VerificationOutcome::Failed {
                    reason: format!("{gate_name} が失敗しました: {log_tail}"),
                });
            }
        }

        // 3. clippy
        match self.run_gate(
            "clippy",
            &["clippy", "--all-targets", "--", "-D", "warnings"],
            proposal,
        )? {
            GateStep::Passed => {}
            GateStep::Failed {
                gate_name,
                log_tail,
            } => {
                return Ok(VerificationOutcome::Failed {
                    reason: format!("{gate_name} が失敗しました: {log_tail}"),
                });
            }
        }

        // 全ゲート通過。既存 runner テスト（`runner.rs` の
        // `ScriptedVerificationGate`）の gate_report 文字列形式と揃え、
        // #135 側の差し替えコストを局所化する（実装計画 4.3 節）。
        Ok(VerificationOutcome::Passed(VerifiedEvidence::new(
            proposal.attempt,
            proposal.description.clone(),
            "build=pass test=pass clippy=pass",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{CommandOutput, ExecError};
    use std::cell::RefCell;
    use std::path::Path;

    /// スクリプト化した `CommandRunner` テストダブル。
    ///
    /// `program`/`args` の呼び出し履歴を記録し、`args[0]`（cargo サブ
    /// コマンド）に応じて事前設定した結果を返す。未設定のサブコマンドは
    /// テスト設定不備として扱い、常に成功を返す。
    struct ScriptedCommandRunner {
        /// (サブコマンド, 成功可否) の対応表。
        results: Vec<(&'static str, bool)>,
        calls: RefCell<Vec<String>>,
    }

    impl ScriptedCommandRunner {
        fn new(results: Vec<(&'static str, bool)>) -> Self {
            ScriptedCommandRunner {
                results,
                calls: RefCell::new(Vec::new()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.borrow().len()
        }
    }

    impl CommandRunner for ScriptedCommandRunner {
        fn run(
            &self,
            program: &str,
            args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            let subcommand = args.first().copied().unwrap_or("");
            self.calls
                .borrow_mut()
                .push(format!("{program} {}", args.join(" ")));

            let success = self
                .results
                .iter()
                .find(|(name, _)| *name == subcommand)
                .map(|(_, success)| *success)
                .unwrap_or(true);

            let log = if success {
                format!("{subcommand}: ok")
            } else {
                format!("{subcommand}: FAILED (scripted)")
            };
            Ok(CommandOutput::from_captured(success, log.into_bytes()))
        }
    }

    /// spawn 自体が失敗するテストダブル（`build` ゲートの起動から常に
    /// `Err` を返す）。
    struct FailingSpawnRunner;
    impl CommandRunner for FailingSpawnRunner {
        fn run(
            &self,
            _program: &str,
            _args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            Err(ExecError::new("spawn failed (scripted)"))
        }
    }

    fn proposal(attempt: u32) -> Proposal {
        Proposal {
            attempt,
            description: "test proposal".to_string(),
        }
    }

    #[test]
    fn all_gates_pass_yields_verified_evidence() {
        let gate = CargoVerificationGate::new(
            PathBuf::from("."),
            ScriptedCommandRunner::new(vec![("build", true), ("test", true), ("clippy", true)]),
        );
        let outcome = gate.verify(&proposal(1)).expect("verify should not error");
        match outcome {
            VerificationOutcome::Passed(evidence) => {
                assert_eq!(evidence.attempt(), 1);
                assert_eq!(evidence.gate_report(), "build=pass test=pass clippy=pass");
            }
            VerificationOutcome::Failed { reason } => {
                panic!("expected Passed, got Failed: {reason}")
            }
        }
        assert_eq!(gate.runner.call_count(), 3);
    }

    #[test]
    fn build_failure_short_circuits_before_test_and_clippy() {
        let gate = CargoVerificationGate::new(
            PathBuf::from("."),
            ScriptedCommandRunner::new(vec![("build", false), ("test", true), ("clippy", true)]),
        );
        let outcome = gate.verify(&proposal(1)).expect("verify should not error");
        match outcome {
            VerificationOutcome::Failed { reason } => {
                assert!(reason.contains("build"));
            }
            VerificationOutcome::Passed(_) => panic!("expected Failed"),
        }
        // build のみ呼ばれ、test/clippy には到達しない（fail-fast）。
        assert_eq!(gate.runner.call_count(), 1);
    }

    #[test]
    fn test_failure_short_circuits_before_clippy() {
        let gate = CargoVerificationGate::new(
            PathBuf::from("."),
            ScriptedCommandRunner::new(vec![("build", true), ("test", false), ("clippy", true)]),
        );
        let outcome = gate.verify(&proposal(1)).expect("verify should not error");
        match outcome {
            VerificationOutcome::Failed { reason } => {
                assert!(reason.contains("test"));
            }
            VerificationOutcome::Passed(_) => panic!("expected Failed"),
        }
        assert_eq!(gate.runner.call_count(), 2);
    }

    #[test]
    fn clippy_failure_reports_log_tail_in_reason() {
        let gate = CargoVerificationGate::new(
            PathBuf::from("."),
            ScriptedCommandRunner::new(vec![("build", true), ("test", true), ("clippy", false)]),
        );
        let outcome = gate.verify(&proposal(1)).expect("verify should not error");
        match outcome {
            VerificationOutcome::Failed { reason } => {
                assert!(reason.contains("clippy"));
                assert!(reason.contains("FAILED"));
            }
            VerificationOutcome::Passed(_) => panic!("expected Failed"),
        }
        assert_eq!(gate.runner.call_count(), 3);
    }

    #[test]
    fn spawn_failure_propagates_as_verification_error_not_failed_outcome() {
        let gate = CargoVerificationGate::new(PathBuf::from("."), FailingSpawnRunner);
        let result = gate.verify(&proposal(3));
        match result {
            Err(SelfRepairError::Verification { attempt, .. }) => {
                assert_eq!(attempt, 3);
            }
            other => panic!("expected SelfRepairError::Verification, got {other:?}"),
        }
    }

    #[test]
    fn verified_evidence_attempt_matches_proposal_attempt() {
        let gate = CargoVerificationGate::new(
            PathBuf::from("."),
            ScriptedCommandRunner::new(vec![("build", true), ("test", true), ("clippy", true)]),
        );
        let outcome = gate.verify(&proposal(7)).expect("verify should not error");
        match outcome {
            VerificationOutcome::Passed(evidence) => assert_eq!(evidence.attempt(), 7),
            VerificationOutcome::Failed { .. } => panic!("expected Passed"),
        }
    }
}
