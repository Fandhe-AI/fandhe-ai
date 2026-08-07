//! build/test/clippy ゲートの順次実行（TASK-4.1c・イシュー #106）。
//!
//! [`crate::check`] の measured 経路から呼ばれ、`docs/guardrail-self-repair-cli.md`
//! §2.1 の `build_result`/`test_result`/`clippy_result` を実測する。
//! 実行コマンドは §2.1 の記載（`cargo build`／`cargo test --release`／
//! `cargo clippy --all-targets -- -D warnings`）と一字一句一致させる
//! （運用者が期待するコマンドと実装が乖離しないよう、コマンド文字列自体を
//! 契約の正本とする。`.claude/rules/coding-rust.md` の
//! `cargo clippy --workspace --all-targets --all-features -- -D warnings`
//! は本リポ自身の CI ポリシーであり、`guardrail check` が任意の対象
//! リポジトリに対して実行するゲートとは別契約）。
//! `cargo build` が失敗した場合、`cargo test`/`cargo clippy` は起動しない
//! （PoC-3 の実行順序契約: build 失敗時 test/clippy はスキップ。
//! `decision::GateSignal::Skipped` は「実行されなかった」ことを表し、
//! 自動適用の根拠には使われない。`decision.rs` モジュールコメント参照）。
//! `cargo test` が失敗した場合も同様に `clippy` を起動しない。
//!
//! v1（`rust-ai-library-v1/crates/guardrail/src/gates.rs`）の実行順序を
//! そのまま踏襲する。

use std::path::Path;

use crate::decision::{GateSignal, GateSignals};
use crate::error::GuardrailError;
use crate::exec::CommandRunner;

/// `runner` を使って `repo_root` 上で build → test → clippy を順に実行する。
///
/// 子プロセス起動自体の失敗（[`GuardrailError::GateSpawn`]）はそのまま
/// 呼び出し元へ伝播する（fail-closed。「ゲートを実行できなかった」ことを
/// `GateSignal::Failed` へ丸めない）。
pub(crate) fn run_gates(
    runner: &dyn CommandRunner,
    repo_root: &Path,
) -> Result<GateSignals, GuardrailError> {
    let build = runner.run(repo_root, "cargo", &["build"])?;
    if !build.success {
        log_gate_failure("build", &build);
        return Ok(GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        });
    }

    let test = runner.run(repo_root, "cargo", &["test", "--release"])?;
    if !test.success {
        log_gate_failure("test", &test);
        return Ok(GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Failed,
            clippy: GateSignal::Skipped,
        });
    }

    let clippy = runner.run(
        repo_root,
        "cargo",
        &["clippy", "--all-targets", "--", "-D", "warnings"],
    )?;
    if !clippy.success {
        log_gate_failure("clippy", &clippy);
    }

    Ok(GateSignals {
        build: GateSignal::Passed,
        test: GateSignal::Passed,
        clippy: if clippy.success {
            GateSignal::Passed
        } else {
            GateSignal::Failed
        },
    })
}

/// ゲート失敗時の診断出力（stderr。判定レポート JSON には含めない —
/// §2.1 スキーマに `gate_stderr` 相当のフィールドはなく、レポートは
/// あくまで pass/fail の 2 値のみを持つ契約のため）。運用者が CI ログから
/// 原因を追えるようにする目的のみ。
fn log_gate_failure(gate: &str, output: &crate::exec::CommandOutput) {
    eprintln!("guardrail: gate '{gate}' failed");
    if !output.stderr.is_empty() {
        eprintln!("{}", output.stderr);
    }
    if !output.stdout.is_empty() {
        eprintln!("{}", output.stdout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::tests_support::ScriptedRunner;
    use std::path::PathBuf;

    fn dummy_repo() -> PathBuf {
        PathBuf::from(".")
    }

    #[test]
    fn all_pass_yields_all_passed() {
        let runner = ScriptedRunner::new(vec![true, true, true]);
        let gates = run_gates(&runner, &dummy_repo()).expect("ゲート実行に失敗");
        assert_eq!(gates.build, GateSignal::Passed);
        assert_eq!(gates.test, GateSignal::Passed);
        assert_eq!(gates.clippy, GateSignal::Passed);
        assert_eq!(runner.calls.borrow().len(), 3);
    }

    #[test]
    fn build_failure_skips_test_and_clippy() {
        let runner = ScriptedRunner::new(vec![false]);
        let gates = run_gates(&runner, &dummy_repo()).expect("ゲート実行に失敗");
        assert_eq!(gates.build, GateSignal::Failed);
        assert_eq!(gates.test, GateSignal::Skipped);
        assert_eq!(gates.clippy, GateSignal::Skipped);
        // build のみ 1 回呼ばれ、test/clippy は起動されない。
        assert_eq!(runner.calls.borrow().len(), 1);
    }

    #[test]
    fn test_failure_skips_clippy() {
        let runner = ScriptedRunner::new(vec![true, false]);
        let gates = run_gates(&runner, &dummy_repo()).expect("ゲート実行に失敗");
        assert_eq!(gates.build, GateSignal::Passed);
        assert_eq!(gates.test, GateSignal::Failed);
        assert_eq!(gates.clippy, GateSignal::Skipped);
        assert_eq!(runner.calls.borrow().len(), 2);
    }

    #[test]
    fn clippy_failure_is_reported_after_build_and_test_pass() {
        let runner = ScriptedRunner::new(vec![true, true, false]);
        let gates = run_gates(&runner, &dummy_repo()).expect("ゲート実行に失敗");
        assert_eq!(gates.build, GateSignal::Passed);
        assert_eq!(gates.test, GateSignal::Passed);
        assert_eq!(gates.clippy, GateSignal::Failed);
        assert_eq!(runner.calls.borrow().len(), 3);
    }

    #[test]
    fn command_spawn_failure_propagates_as_error() {
        struct FailingRunner;
        impl CommandRunner for FailingRunner {
            fn run(
                &self,
                _cwd: &Path,
                program: &str,
                args: &[&str],
            ) -> Result<crate::exec::CommandOutput, GuardrailError> {
                Err(GuardrailError::GateSpawn {
                    command: format!("{program} {}", args.join(" ")),
                    source: std::io::Error::other("not found"),
                })
            }
        }
        let err = run_gates(&FailingRunner, &dummy_repo()).unwrap_err();
        assert!(matches!(err, GuardrailError::GateSpawn { .. }));
    }
}
