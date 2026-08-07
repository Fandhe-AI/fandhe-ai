//! サブプロセス実行の抽象化（`CommandRunner`）。
//!
//! [`crate::gates`] が `cargo build`/`test`/`clippy` を起動する際の実行口を
//! trait で抽象化し、単体テストではモック実装（成功・失敗・スキップ順序の
//! 固定検証）を注入できるようにする（TASK-4.1c・イシュー #106。v1
//! `rust-ai-library-v1/crates/guardrail/src/exec.rs` からの移植）。
//!
//! 本モジュール自体は `cargo` 専用ではなく任意コマンドの起動口だが、
//! `git` 呼び出しは [`crate::exclusion_match::run_git`]（`-c core.quotePath=false`
//! 等の diff 出力汚染対策済み）に一本化しているため、本モジュール経由では
//! 呼ばない（[`crate::gates`] 以外からは使わない契約）。

use std::path::Path;
use std::process::Command;

use crate::error::GuardrailError;

/// コマンド実行結果。終了コードは `success` の bool のみに単純化する
/// （[`crate::decision::GateSignal`] は pass/fail の 2 値のみを要求するため）。
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// コマンド実行の抽象境界。[`SystemCommandRunner`] が本番経路、テストでは
/// モック実装を注入して `cargo` を実際に起動せずゲート順序・スキップ挙動を
/// 検証する。
pub trait CommandRunner {
    fn run(
        &self,
        cwd: &Path,
        program: &str,
        args: &[&str],
    ) -> Result<CommandOutput, GuardrailError>;
}

/// 実際に子プロセスを起動する本番実装。
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    /// `program`（`cargo` 等）を `args` で `cwd` 上で起動する。子プロセス
    /// 起動自体の失敗（実行ファイル不在等）は
    /// [`GuardrailError::GateSpawn`] として fail-closed で伝播する
    /// （`success = false` へ丸めない。`.claude/rules/security.md` A08:
    /// 「ゲートを実行できなかった」ことと「ゲートが失敗した」ことを混同しない）。
    fn run(
        &self,
        cwd: &Path,
        program: &str,
        args: &[&str],
    ) -> Result<CommandOutput, GuardrailError> {
        let command_display = format!("{program} {}", args.join(" "));
        let output = Command::new(program)
            .args(args)
            .current_dir(cwd)
            .output()
            .map_err(|source| GuardrailError::GateSpawn {
                command: command_display,
                source,
            })?;
        Ok(CommandOutput {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    //! `gates.rs` の単体テストから使うモック `CommandRunner`（順序・成否を
    //! 固定検証するため、実プロセスは起動しない）。
    use super::*;
    use std::cell::RefCell;

    pub struct ScriptedRunner {
        /// 呼び出し順に返す `(success, stdout, stderr)`。呼び出し回数が
        /// 用意した件数を超えた場合はパニックする（テストの想定漏れを
        /// 早期検出するため）。
        scripts: RefCell<std::collections::VecDeque<bool>>,
        pub calls: RefCell<Vec<String>>,
    }

    impl ScriptedRunner {
        pub fn new(results: Vec<bool>) -> Self {
            ScriptedRunner {
                scripts: RefCell::new(results.into()),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl CommandRunner for ScriptedRunner {
        fn run(
            &self,
            _cwd: &Path,
            program: &str,
            args: &[&str],
        ) -> Result<CommandOutput, GuardrailError> {
            self.calls
                .borrow_mut()
                .push(format!("{program} {}", args.join(" ")));
            let success = self
                .scripts
                .borrow_mut()
                .pop_front()
                .expect("ScriptedRunner: 用意したスクリプトを超えて呼び出された");
            Ok(CommandOutput {
                success,
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }
}
