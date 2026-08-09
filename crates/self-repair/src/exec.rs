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

use crate::isolation::ExecIsolation;

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
    truncated: bool,
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

    /// `log_tail()` が [`MAX_CAPTURED_LOG_BYTES`] 超過により先頭側を切り詰め
    /// 済みか（イシュー #137 Codex レビュー指摘）。
    ///
    /// `crate::diff_signals::run_git` はこの値を見て、`git diff --numstat`／
    /// `git diff -U0` 等の構造化出力を切り詰め済みログとして解析しないよう
    /// fail-closed で拒否する（先頭側のファイル・変更行が黙って欠落し
    /// `lines_changed` 過少計上・`api_broken` 見逃しに繋がるため。
    /// `.claude/rules/security.md` A08「判定の迂回経路を作らない」）。
    /// `CargoVerificationGate`（build/test/clippy ログ）はログを人間可読な
    /// 診断メッセージとしてのみ扱うため、この値を無視して従来どおり
    /// 切り詰め済みログをそのまま使ってよい。
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// stdout/stderr の生バイト列から [`CommandOutput`] を構築する。
    ///
    /// `SystemCommandRunner::run`（本番経路）と `verify_gates.rs` の
    /// `CommandRunner` テストダブル（スクリプト化したログを構築する
    /// ユニットテスト）の双方から呼べるよう `pub(crate)` とする。
    pub(crate) fn from_captured(success: bool, combined: Vec<u8>) -> Self {
        // `combined` はプロセス出力の生バイト列であり、必ずしも UTF-8 として
        // 妥当とは限らない。不正なバイト列を含む場合、上限適用前（バイト列の
        // 段階）で切り詰めてから `String::from_utf8_lossy` を呼ぶと、不正な
        // 各バイトが置換文字 U+FFFD（3 バイト）へ個別に展開されうるため、
        // 変換後の文字列長が `MAX_CAPTURED_LOG_BYTES` を大きく超過しうる
        // （最悪 1 バイトあたり 3 バイトへ膨張＝ 3 倍）。これを避けるため、
        // 先に lossy 変換して妥当な UTF-8 文字列に正規化してから、文字列側
        // （`char_indices` による文字境界）で上限を適用する。
        let decoded = String::from_utf8_lossy(&combined).into_owned();
        let (log_tail, truncated) = if decoded.len() <= MAX_CAPTURED_LOG_BYTES {
            (decoded, false)
        } else {
            // 末尾 MAX_CAPTURED_LOG_BYTES 相当を残す。`decoded` は既に妥当な
            // UTF-8 文字列なので `char_indices` で文字境界を安全に走査できる
            // （バイト列時点でのバイトパターン判定は不要）。
            let cut = decoded.len() - MAX_CAPTURED_LOG_BYTES;
            let boundary = decoded
                .char_indices()
                .map(|(index, _)| index)
                .find(|&index| index >= cut)
                .unwrap_or(decoded.len());
            (
                format!("{TRUNCATED_LOG_PREFIX}{}", &decoded[boundary..]),
                true,
            )
        };
        CommandOutput {
            success,
            log_tail,
            truncated,
        }
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
///
/// `Clone` を実装する（イシュー #137）。`crate::verify_direct_composite::
/// RepairCompositeGate<R>` は試行ごとに `R: CommandRunner + Clone` を要求する
/// `crate::verify_gates::CargoVerificationGate<R>` を**新規構築**する（diff 由来
/// シグナルを試行ごとに実測し直すため。`verify_direct_composite` モジュール冒頭
/// ドキュメント参照）。`isolation` フィールドを持つが複製に副作用はない
/// （`ExecIsolation` は設定値のみを保持する）。
///
/// # `new()` と `isolated()` の使い分け（イシュー #414）
/// [`SystemCommandRunner::new`] は従来どおりの挙動（`run` 内の個別 deny-list・
/// 網羅遮断なし）を維持し、既存の呼び出し元・テストとの互換を保つ。
/// [`SystemCommandRunner::isolated`] は環境変数 allowlist・`HOME`/`TMPDIR`
/// 付け替え・（設定されていれば）`unshare` によるネットワーク隔離を適用する
/// 候補実行専用の構成であり、`main.rs::run_run` が `--candidates` を実行する
/// 3 経路（`gate_spec.runner`・`BugFixDetector`・`FeatureAdditionDetector`）に
/// のみ使う（`isolation` モジュール冒頭ドキュメント参照）。
#[derive(Debug, Clone)]
pub struct SystemCommandRunner {
    isolation: Option<ExecIsolation>,
}

impl SystemCommandRunner {
    pub fn new() -> Self {
        SystemCommandRunner { isolation: None }
    }

    /// 候補実行の縦深防御（`isolation` モジュール参照）を適用する
    /// `SystemCommandRunner` を構築する。
    pub fn isolated(isolation: ExecIsolation) -> Self {
        SystemCommandRunner {
            isolation: Some(isolation),
        }
    }
}

impl Default for SystemCommandRunner {
    fn default() -> Self {
        SystemCommandRunner::new()
    }
}

impl CommandRunner for SystemCommandRunner {
    fn run(&self, program: &str, args: &[&str], cwd: &Path) -> Result<CommandOutput, ExecError> {
        // `isolation` が設定されている場合のみ、ネットワーク隔離が有効なら
        // argv を `unshare` ラップへ差し替える（`Inherit` の場合は no-op で
        // 元の `program`/`args` のまま。`isolation.rs::ExecIsolation::
        // wrap_argv_for_network_isolation` doc 参照）。
        let (effective_program, owned_args);
        let (program, args): (&str, &[&str]) = match &self.isolation {
            Some(isolation) => {
                let (wrapped_program, wrapped_args) =
                    isolation.wrap_argv_for_network_isolation(program, args);
                effective_program = wrapped_program;
                owned_args = wrapped_args;
                (effective_program, owned_args.as_slice())
            }
            None => (program, args),
        };

        let mut command = Command::new(program);
        command.args(args).current_dir(cwd);

        if let Some(isolation) = &self.isolation {
            // 候補実行経路: 環境変数を allowlist 方式で遮断し、`HOME`/`TMPDIR`
            // を sandbox 配下へ付け替える（`isolation.rs::ExecIsolation::apply`
            // doc 参照）。以降の `env_remove` 個別指定は `env_clear` 後は
            // 不要だが、`new()` 経路との差分を最小化するため分岐で早期
            // return はせず、`apply` 適用後にそのまま従来の spawn ロジックへ
            // 合流させる。
            isolation.apply(&mut command);
            let output = command.output().map_err(|error| {
                ExecError::new(format!("{program} の起動に失敗しました: {error}"))
            })?;
            let mut combined = output.stdout;
            combined.extend_from_slice(&output.stderr);
            return Ok(CommandOutput::from_captured(
                output.status.success(),
                combined,
            ));
        }

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

        // 祖先プロセス（lefthook の pre-push フック等）から `GIT_DIR`／
        // `GIT_WORK_TREE`／`GIT_INDEX_FILE` 等の `GIT_*` 環境変数を継承した
        // 状態で子プロセスを起動すると、本 runner 経由で呼ばれる git 操作
        // （`crate::diff_signals`・`crate::verify_bench_direct` が `git diff`／
        // `git worktree add` 等を `SystemCommandRunner::run("git", ..)` 経由で
        // 起動する。イシュー #137）が `cwd` 指定を無視して呼び出し元プロセスの
        // リポジトリ（本 worktree の `.git`）を対象にしてしまう
        // （`tests/revalidation_bug_fix.rs`・
        // `tests/feature_addition_loop_completion_task_3_3c.rs` の
        // `sandboxed_git_command` 系ヘルパーが既に踏んだ事故パターンと同一。
        // 2026-08-07 実測: 対応前に sandbox の `git commit` が実リポジトリの
        // 現在ブランチ HEAD へ実際にコミットしてしまう事故が発生している）。
        // `CARGO_TARGET_DIR` 等と同じ理由で本番経路（`SystemCommandRunner`）
        // 自体に除去を持たせ、呼び出し元ごとに個別対処させない。
        for (key, _) in std::env::vars_os() {
            if let Some(key_str) = key.to_str()
                && key_str.starts_with("GIT_")
            {
                command.env_remove(key_str);
            }
        }

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
        assert!(!output.truncated(), "上限未満は切り詰めなし扱いのはず");
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
        assert!(
            output.truncated(),
            "上限超過は truncated() で検出できるはず（イシュー #137 Codex レビュー指摘）"
        );
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
    fn log_tail_stays_within_bound_when_invalid_utf8_expands_via_replacement_chars() {
        // 不正な UTF-8 バイト列（`0xFF` は単独では継続バイトと誤認されない
        // ため、旧実装のバイト境界判定だけでは切り詰め対象に残ってしまう）
        // を上限超過分含める。`String::from_utf8_lossy` は不正バイト 1 個を
        // U+FFFD（3 バイト）へ展開するため、バイト列側で先に切り詰めてから
        // 変換すると変換後の文字列が上限の最大約 3 倍まで膨張しうる
        // （Cursor Bugbot review #4885590407 が指摘した回帰）。文字列化後に
        // 切り詰める本実装ではこの膨張が起きないことを確認する。
        let mut bytes = vec![0xFFu8; MAX_CAPTURED_LOG_BYTES + 1024];
        bytes.extend_from_slice(b"tail-marker");
        let output = CommandOutput::from_captured(true, bytes);
        assert!(
            output.log_tail().len() <= MAX_CAPTURED_LOG_BYTES + TRUNCATED_LOG_PREFIX.len(),
            "log_tail should stay within the documented bound even for invalid UTF-8 input, got {} bytes",
            output.log_tail().len()
        );
        assert!(output.log_tail().ends_with("tail-marker"));
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
