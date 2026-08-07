//! 種別ごとの検出段階（[`crate::bug_fix`]・[`crate::feature_addition`]）が使う
//! コマンド実行の抽象化（TASK-3.1b・イシュー #133・REQ-3）。
//!
//! v1（`Fandhe-AI/rust-ai-library-v1` `crates/guardrail/src/exec.rs`）は本 seam を
//! `guardrail::exec::CommandRunner` として `guardrail` 側に置いていたが、v2
//! `guardrail` クレートには本イシュー時点で `exec` モジュールが未移植であり
//! （guardrail 自体の CLI 実行系〈TASK-4.1〉はイシュー #103 が別途追跡）、
//! guardrail 側への新規追加は並行 guardrail トラック（#134〜#136・#145 等）と
//! 競合しうる（`.claude/rules/delegation-impl.md`「複数 Agent に同一ファイルを
//! 並行編集させない」）。そのため本クレート内に v2 新設として置く。検証ゲート
//! 実実行（#134）が build/test/clippy を起動する際も同一 seam の再利用を想定
//! する（呼び出し元候補: `crate::stages::VerificationGate` の #134 実装）。
//! `guardrail` への移設が必要になった場合は `out-of-scope-tracking.md` の
//! 規約に従い Issue へ記録する（実装計画セクション 7 参照）。
//!
//! # 呼び出し文脈
//! - 呼び出し元: [`crate::bug_fix::BugFixDetector`]・
//!   [`crate::feature_addition::FeatureAdditionDetector`]（いずれも
//!   `cargo test --release` の起動に使う）
//! - 呼び出し先: [`SystemCommandRunner`] が `std::process::Command` を実行する

use std::path::Path;
use std::process::Command;

/// キャプチャするコマンド出力（stdout/stderr 結合）の上限バイト数。
///
/// 巨大な `cargo test` ログを無制限に保持するとメモリを圧迫する
/// （`.claude/rules/security.md` A03 の外部入力検証と同種の DoS 耐性の思想。
/// 末尾を優先して保持するのは、失敗原因は末尾に現れることが多いため。
/// v1 `guardrail::exec::MAX_CAPTURED_LOG_BYTES` と同値）。
pub const MAX_CAPTURED_LOG_BYTES: usize = 256 * 1024;

/// [`truncate_to_tail`] が切り詰め発生時に先頭へ付与するマーカー文字列。
const TRUNCATED_LOG_PREFIX: &str = "...(先頭を切り詰め)...";

/// コマンド実行結果。exit 成否と結合ログ（上限切り詰め済み）を保持する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    success: bool,
    /// stdout/stderr を結合し [`MAX_CAPTURED_LOG_BYTES`] で末尾切り詰めしたログ。
    log_tail: String,
}

impl CommandOutput {
    /// テスト・呼び出し元から任意の成否とログで組み立てるための構築子。
    /// 実プロセス実行結果は [`SystemCommandRunner::run`] が組み立てる。
    pub fn new(success: bool, log: impl Into<String>) -> Self {
        let log = log.into();
        let log_tail = truncate_to_tail(&log, MAX_CAPTURED_LOG_BYTES);
        CommandOutput { success, log_tail }
    }

    /// コマンドが 0 終了したかどうか。
    pub fn success(&self) -> bool {
        self.success
    }

    /// 末尾切り詰め済みの結合ログ（[`crate::stages::Finding::summary`] 等、
    /// 失敗理由の要約表示に使う）。
    pub fn log_tail(&self) -> &str {
        &self.log_tail
    }
}

/// バイト列を UTF-8 境界を壊さない範囲で末尾 `limit` バイトに切り詰める。
fn truncate_to_tail(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let start = s.len() - limit;
    let mut start = start;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("{TRUNCATED_LOG_PREFIX}{}", &s[start..])
}

/// 各 `Detector` 実装から見た「コマンドを 1 つ実行する」ことの抽象。
///
/// 実装は [`SystemCommandRunner`]（実プロセス起動）と、テスト用の
/// [`crate::test_support::ScriptedCommand`] の 2 種類を想定する。
pub trait CommandRunner {
    /// `program` を `args` 付きで `cwd` を作業ディレクトリとして実行する。
    ///
    /// spawn 自体に失敗した場合（コマンド未インストール等）は `Err` を返す。
    /// 呼び出し元（[`crate::stages::Detector`] 実装）はこれを
    /// [`crate::error::SelfRepairError::Detection`] へ変換する（fail-closed。
    /// `.claude/rules/security.md` A08: 判定不能を通過に倒す経路を作らない）。
    fn run(&self, program: &str, args: &[&str], cwd: &Path) -> Result<CommandOutput, String>;
}

/// `std::process::Command` を用いた実プロセス実行の [`CommandRunner`] 実装。
///
/// 引数配列で起動しシェル経由の文字列展開をしない
/// （`.claude/rules/security.md` A03。v1 `guardrail::exec::SystemCommandRunner`
/// と同一方針）。`self-repair` 自身が lefthook の pre-push フック等、既存の
/// git/cargo フック処理系の子プロセスとして呼ばれる可能性があるため、cargo
/// 起動に影響する環境変数を明示的に除去してから起動する（祖先プロセスの
/// `CARGO_TARGET_DIR` 等が検出対象ワークスペースの target ディレクトリ・
/// ロックを意図せず差し替えるのを防ぐ）。
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run(&self, program: &str, args: &[&str], cwd: &Path) -> Result<CommandOutput, String> {
        let output = Command::new(program)
            .env_remove("CARGO_TARGET_DIR")
            .env_remove("CARGO_BUILD_TARGET_DIR")
            .env_remove("CARGO_MAKEFLAGS")
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTFLAGS")
            .current_dir(cwd)
            .args(args)
            .output()
            .map_err(|source| {
                format!("コマンド起動に失敗しました（program={program}）: {source}")
            })?;

        let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&output.stderr));

        Ok(CommandOutput::new(output.status.success(), log))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_long_log_to_tail_and_keeps_utf8_boundary() {
        let long = "あ".repeat(200_000);
        let output = CommandOutput::new(true, long.clone());
        assert!(output.log_tail().len() <= MAX_CAPTURED_LOG_BYTES + TRUNCATED_LOG_PREFIX.len());
        assert!(output.log_tail().starts_with(TRUNCATED_LOG_PREFIX));
        assert!(long.len() > MAX_CAPTURED_LOG_BYTES);
    }

    #[test]
    fn keeps_short_log_unchanged() {
        let output = CommandOutput::new(false, "short log");
        assert_eq!(output.log_tail(), "short log");
        assert!(!output.success());
    }

    #[test]
    fn system_runner_reports_spawn_failure_as_err() {
        let runner = SystemCommandRunner;
        let err = runner
            .run(
                "this-binary-should-not-exist-self-repair-test",
                &[],
                Path::new("."),
            )
            .expect_err("存在しないコマンドは Err を返すこと");
        assert!(err.contains("コマンド起動に失敗しました"));
    }

    #[test]
    fn system_runner_runs_real_command() {
        // 実機非依存の軽量コマンド（`cargo --version`）でスモークする
        // （実装計画セクション 4 ステップ 9）。
        let runner = SystemCommandRunner;
        let output = runner
            .run("cargo", &["--version"], Path::new("."))
            .expect("`cargo --version` の起動に失敗");
        assert!(output.success());
    }
}
