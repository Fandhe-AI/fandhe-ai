//! コマンド実行抽象（TASK-3.1b・イシュー #133、TASK-3.1c・イシュー #134・REQ-3）。
//!
//! [`crate::verify_gates::CargoVerificationGate`] が `cargo build`/`cargo
//! test`/`cargo clippy` を起動する際の実行系であり、種別ごとの検出段階
//! （[`crate::bug_fix::BugFixDetector`]・[`crate::feature_addition::FeatureAdditionDetector`]）
//! が `cargo test --release` を起動する際にも同一 seam を再利用する。
//! 移植元は v1 `Fandhe-AI/rust-ai-library-v1` `crates/guardrail/src/exec.rs`
//! （`docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//!
//! v1 では `guardrail::exec` だったが、本クレート（`self-repair`）内に
//! 実装を置く。理由: `guardrail` クレート側は本イシューと並行する他イシュー
//! （`guardrail check` 実シグナル計測経路・TASK-6.1c・#199 等）の編集対象と
//! なりうるため、`.claude/rules/delegation-impl.md`「複数 Agent に同一
//! ファイルを並行編集させない」に従い編集対象を本クレートに閉じる。
//! guardrail 側との実装共通化は out-of-scope（イシュー #134 の PR 参照）。
//!
//! # A03（インジェクション）対応
//! [`CommandRunner::run`] はプログラム名と引数を配列で受け取り
//! `std::process::Command` へそのまま渡す。シェル（`sh -c` 等）を経由しない
//! ため、引数中の `;`・`|`・`$()` 等がシェルメタ文字として再解釈されることは
//! ない（`.claude/rules/security.md` A03・`stages.rs` の #134 向け契約）。

use std::path::Path;
use std::process::Command;

/// 取り込むログの上限（256 KiB）。
///
/// `cargo test`・`cargo clippy` の出力は数百 KB に達しうるため、上限なしで
/// 保持すると監査ログ（`LoopReport`）やエラーメッセージの肥大化・DoS 耐性の
/// 低下を招く（`.claude/rules/security.md` A03「巨大出力による DoS」）。
/// v1 `guardrail/src/exec.rs` と同一の上限値を踏襲する。
const MAX_CAPTURED_LOG_BYTES: usize = 256 * 1024;

/// ログを上限超過で切り詰めた際に先頭へ付与するマーカー。
const TRUNCATED_LOG_PREFIX: &str = "...(truncated)...\n";

/// コマンド実行結果（stdout/stderr 結合ログ＋成功可否）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    success: bool,
    log_tail: String,
}

impl CommandOutput {
    /// プロセスが 0 終了コードで終了したか（非 0 終了・シグナル終了は
    /// `false`）。
    pub fn success(&self) -> bool {
        self.success
    }

    /// stdout/stderr を結合し末尾 256 KiB に切り詰めたログ（UTF-8 文字境界
    /// を尊重して切り詰める。マルチバイト文字の途中で分断しない）。
    pub fn log_tail(&self) -> &str {
        &self.log_tail
    }

    /// stdout/stderr の生バイト列から [`CommandOutput`] を構築する。
    ///
    /// `SystemCommandRunner::run`（本番経路）と `verify_gates.rs` の
    /// `CommandRunner` テストダブル（スクリプト化したログを構築する
    /// ユニットテスト）の双方から呼べるよう `pub(crate)` とする。
    pub(crate) fn from_captured(success: bool, mut combined: Vec<u8>) -> Self {
        let log_tail = if combined.len() <= MAX_CAPTURED_LOG_BYTES {
            String::from_utf8_lossy(&combined).into_owned()
        } else {
            // 末尾 MAX_CAPTURED_LOG_BYTES 相当を残す。UTF-8 文字境界を跨いで
            // 分断すると `String::from_utf8_lossy` が置換文字を挿入し
            // ログが読みにくくなるため、境界に達するまで先頭を切り詰める
            // （v1 `exec.rs` と同一方針）。
            let cut = combined.len() - MAX_CAPTURED_LOG_BYTES;
            let mut boundary = cut;
            // UTF-8 継続バイト（先頭 2 ビットが `10`）の間は境界とみなさず
            // 前進する。`[u8]` は `str::is_char_boundary` を持たないため
            // バイトパターンを直接判定する（`combined` はプロセス出力の生
            // バイト列であり、必ずしも UTF-8 として妥当とは限らないが、
            // 継続バイト判定自体は妥当性を前提としない）。
            while boundary < combined.len() && combined[boundary] & 0b1100_0000 == 0b1000_0000 {
                boundary += 1;
            }
            combined.drain(0..boundary);
            format!(
                "{TRUNCATED_LOG_PREFIX}{}",
                String::from_utf8_lossy(&combined)
            )
        };
        CommandOutput { success, log_tail }
    }
}

/// コマンド実行の抽象。[`crate::verify_gates::CargoVerificationGate`] は
/// この trait 経由でのみ子プロセスを起動し、本番経路は
/// [`SystemCommandRunner`]、ユニットテストはスクリプト化したテストダブル
/// （`verify_gates.rs`・`candidate.rs` のテストモジュール参照）を注入する。
pub trait CommandRunner {
    /// `program` を `args`（配列）とともに `cwd` で起動し、結果を返す。
    ///
    /// spawn 自体に失敗した場合（実行ファイルが存在しない等）は `Err` を
    /// 返す。プロセスが起動できたが非 0 終了した場合は `Ok` の
    /// `CommandOutput::success() == false` で表現し、両者を区別する
    /// （呼び出し元の [`crate::verify_gates::CargoVerificationGate`] が
    /// spawn 失敗を `SelfRepairError::Verification` として伝播し、
    /// ゲート不合格〈`Ok`〉と区別する契約と対応する）。
    fn run(&self, program: &str, args: &[&str], cwd: &Path) -> Result<CommandOutput, ExecError>;
}

/// [`CommandRunner::run`] の spawn 失敗を表す内部エラー型。
///
/// `SelfRepairError` へ変換する責務は呼び出し元（`verify_gates.rs`）が持つ
/// （`SelfRepairError::Verification { attempt, reason }` は `attempt` を
/// 知らないと構築できないため、`exec.rs` 単体では `SelfRepairError` を
/// 直接返さない）。
#[derive(Debug, Clone)]
pub struct ExecError {
    message: String,
}

impl ExecError {
    /// `verify_gates.rs` の spawn 失敗テストダブルからも構築できるよう
    /// `pub(crate)` とする（クレート外からは構築不能。呼び出し元は
    /// `SystemCommandRunner::run` とテストダブルのみに限られる）。
    pub(crate) fn new(message: impl Into<String>) -> Self {
        ExecError {
            message: message.into(),
        }
    }

    /// 人間可読なエラー内容（`SelfRepairError::Verification.reason` へ
    /// 埋め込む用途）。
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "コマンド実行に失敗しました: {}", self.message)
    }
}

impl std::error::Error for ExecError {}

/// `std::process::Command` による実プロセス実行（[`CommandRunner`] 本番実装）。
pub struct SystemCommandRunner;

impl SystemCommandRunner {
    pub fn new() -> Self {
        SystemCommandRunner
    }
}

impl Default for SystemCommandRunner {
    fn default() -> Self {
        SystemCommandRunner::new()
    }
}

impl CommandRunner for SystemCommandRunner {
    fn run(&self, program: &str, args: &[&str], cwd: &Path) -> Result<CommandOutput, ExecError> {
        let mut command = Command::new(program);
        command.args(args).current_dir(cwd);

        // 祖先プロセス（lefthook フック・CI・テストハーネス）が
        // `CARGO_TARGET_DIR` 等のビルド出力先を変える環境変数を設定して
        // いる場合、子プロセスの `cargo build`/`test`/`clippy` の実行先が
        // 呼び出し元の意図（`cwd` 配下）とすり替わりうる。検証ゲートが
        // 検証対象と異なる workspace を build/test する事態を防ぐため、
        // 明示的に除去してから起動する（v1 `exec.rs` と同一の理由）。
        // `CARGO_ENCODED_RUSTFLAGS` は `RUSTFLAGS` と同じ rustc フラグを
        // 伝える等価チャンネルであり、Cargo は両者を等価に扱う
        // （どちらか一方が設定されていればそちらが優先される）。
        // `RUSTFLAGS` のみ除去して `CARGO_ENCODED_RUSTFLAGS` を継承したまま
        // にすると、祖先プロセスがこちらを設定していた場合に検証ビルドの
        // 挙動が変わりうるため、同様に除去する。
        // `CARGO_MAKEFLAGS` は cargo が子プロセスへ jobserver（GNU make
        // 互換のトークンパイプ）を伝搬させる環境変数であり、継承すると
        // 検証用ビルドが祖先の jobserver からトークンを取得しに行き、
        // 祖先側の並列度制約や、祖先プロセス終了後のパイプ切断で
        // ハング／異常終了しうる。`RUSTC_WRAPPER` は sccache 等の
        // コンパイララッパーを指定する環境変数であり、継承すると
        // 検証ビルドが祖先と同じラッパー・キャッシュを経由してしまい、
        // 検証対象と異なるキャッシュ状態の影響を受けうる。いずれも
        // 検証ゲートの再現性を損なうため、他の変数と同様に除去する。
        command.env_remove("CARGO_TARGET_DIR");
        command.env_remove("CARGO_BUILD_TARGET_DIR");
        command.env_remove("RUSTFLAGS");
        command.env_remove("CARGO_ENCODED_RUSTFLAGS");
        command.env_remove("CARGO_MAKEFLAGS");
        command.env_remove("RUSTC_WRAPPER");

        let output = command
            .output()
            .map_err(|error| ExecError::new(format!("{program} の起動に失敗しました: {error}")))?;

        let mut combined = output.stdout;
        combined.extend_from_slice(&output.stderr);

        Ok(CommandOutput::from_captured(
            output.status.success(),
            combined,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_tail_keeps_short_output_unmodified() {
        let output = CommandOutput::from_captured(true, b"hello world".to_vec());
        assert_eq!(output.log_tail(), "hello world");
        assert!(output.success());
    }

    #[test]
    fn log_tail_truncates_and_prefixes_marker_when_over_limit() {
        // 上限を大幅に超える ASCII 文字列（文字境界問題を起こさない前提）で
        // 切り詰め・マーカー付与を確認する。
        let big = vec![b'a'; MAX_CAPTURED_LOG_BYTES + 100];
        let output = CommandOutput::from_captured(false, big);
        assert!(!output.success());
        assert!(output.log_tail().starts_with(TRUNCATED_LOG_PREFIX));
        // マーカー分を除いた残りは MAX_CAPTURED_LOG_BYTES 以下。
        assert!(output.log_tail().len() <= MAX_CAPTURED_LOG_BYTES + TRUNCATED_LOG_PREFIX.len());
    }

    #[test]
    fn log_tail_truncation_respects_utf8_char_boundary() {
        // マルチバイト文字（3 バイトの日本語文字）が上限境界をまたぐ配置で
        // 構築し、置換文字（U+FFFD）が挿入されないことを確認する。
        let filler = vec![b'x'; MAX_CAPTURED_LOG_BYTES - 1];
        let mut bytes = filler;
        // 境界をまたぐようにマルチバイト文字を追加する。
        bytes.extend_from_slice("あ".as_bytes());
        let output = CommandOutput::from_captured(true, bytes);
        assert!(!output.log_tail().contains('\u{FFFD}'));
    }

    #[test]
    fn system_command_runner_reports_success_for_zero_exit() {
        let runner = SystemCommandRunner::new();
        let cwd = std::env::current_dir().expect("current_dir should be available in tests");
        let output = runner
            .run("cargo", &["--version"], &cwd)
            .expect("cargo --version should spawn successfully");
        assert!(output.success());
        assert!(output.log_tail().to_lowercase().contains("cargo"));
    }

    #[test]
    fn system_command_runner_reports_failure_for_nonzero_exit() {
        let runner = SystemCommandRunner::new();
        let cwd = std::env::current_dir().expect("current_dir should be available in tests");
        // 存在しないサブコマンドを渡し、非 0 終了を確実に発生させる。
        let output = runner
            .run("cargo", &["__self_repair_nonexistent_subcommand__"], &cwd)
            .expect("cargo should still spawn even if the subcommand is invalid");
        assert!(!output.success());
    }

    #[test]
    fn system_command_runner_errors_on_missing_program() {
        let runner = SystemCommandRunner::new();
        let cwd = std::env::current_dir().expect("current_dir should be available in tests");
        let result = runner.run("self_repair_definitely_not_a_real_binary_xyz", &[], &cwd);
        assert!(result.is_err());
    }
}
