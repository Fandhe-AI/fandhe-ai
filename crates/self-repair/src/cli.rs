//! `self-repair` バイナリの自作コマンドライン引数パーサ（TASK-3.4 残作業・
//! イシュー #145）。
//!
//! `clap` は `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれず、
//! 依存追加はユーザー承認事項のため、`crates/guardrail/src/cli.rs` と同じ
//! 方針で `std::env::args` ベースの自作パースを行う。
//!
//! 現時点で結線済みのサブコマンドは `verify-log`
//! （`docs/guardrail-self-repair-cli.md` 3.2 節）のみである。`run`
//! （同 3.1 節）は別イシューのスコープ（`crate::lib` モジュールコメント
//! 「CLI バイナリ（self-repair run/verify-log）→ 後続タスク」参照）のため
//! 未実装であり、[`Command`] へ variant を追加する形で拡張する想定
//! （guardrail の `check`/`eval` 二本立てと同じ拡張パターン）。
//!
//! # 引数の受け渡しは `OsString`（PR #356 codex-review P1 指摘対応）
//!
//! `std::env::args()` は非 UTF-8 引数が渡ると panic する（exit 101。fail-closed
//! 契約の usage エラー exit 2・内部エラー exit 1 のいずれとも異なる非文書化の
//! 終了コードになってしまう）。ファイルシステム上は有効な非 UTF-8 パスの
//! ログファイルを `--log` に渡すケースを panic させないため、本モジュールは
//! `OsString` を受け取り、サブコマンド名・フラグ名のみを `to_str()` で UTF-8
//! 検証する（不正なら usage エラーへ変換。exit コードへの写像は `main.rs` に
//! 一本化）。`--log` の値自体は UTF-8 検証せず `PathBuf` へそのまま渡す。

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

/// CLI 引数の解析・検証に失敗した（未知引数・値欠落・不正なサブコマンド等）。
/// `docs/guardrail-self-repair-cli.md` 3.2 節には verify-log 固有の終了コード
/// 定義がないため、guardrail の usage エラー区分（終了コード `2`）に整合する
/// 契約を実装計画（イシュー #145 差し戻し分・完走判定基準 4）側で新たに定め、
/// `main.rs::report_error_and_exit` がこの型を終了コードへ写像する
/// （写像は 1 箇所に閉じ込め、fail-closed 契約を保つ。`.claude/rules/security.md` A08）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UsageError {}

/// `self-repair verify-log` の引数（3.2 節）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyLogArgs {
    pub log: PathBuf,
    /// `--allow-empty-log`（任意フラグ・値なし）。レコード 0 件のログを
    /// 明示的に許容する場合のみ指定する。指定なしでは空ログを exit 1
    /// （検証不合格扱い）にする（PR #356 codex-review P1 指摘対応:
    /// 空ログを無条件 exit 0 で通すと、ログ全削除による改竄を終了コードのみ
    /// 見る監査スクリプトが「検証成功」として見逃す経路になっていたため。
    /// `main.rs::run_verify_log` 参照）。
    pub allow_empty_log: bool,
}

/// 本バイナリが受理するサブコマンド。`run`（3.1 節）は未実装
/// （モジュールコメント参照）のため variant を持たない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    VerifyLog(VerifyLogArgs),
}

/// `std::env::args_os()` の実引数（プログラム名を除く）を受け取ってパースする。
/// `main.rs` から呼ばれ、返った [`Command`] に応じて実行フローを分岐する。
/// `OsString` を受け取る理由はモジュール冒頭ドキュメント参照（非 UTF-8 引数で
/// panic させないため）。
pub fn parse<I, S>(args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut it = args.into_iter().map(|s| s.as_ref().to_os_string());
    let subcommand = it
        .next()
        .ok_or_else(|| UsageError("missing subcommand (verify-log)".to_string()))?;
    let subcommand = subcommand.to_str().ok_or_else(|| {
        UsageError("subcommand must be valid UTF-8 (expected verify-log)".to_string())
    })?;

    match subcommand {
        "verify-log" => parse_verify_log(it).map(Command::VerifyLog),
        other => Err(UsageError(format!(
            "unknown subcommand '{other}' (expected verify-log)"
        ))),
    }
}

fn parse_verify_log<I>(args: I) -> Result<VerifyLogArgs, UsageError>
where
    I: Iterator<Item = OsString>,
{
    let mut log: Option<PathBuf> = None;
    let mut allow_empty_log = false;

    let mut it = args.peekable();
    while let Some(flag) = it.next() {
        let flag_str = flag.to_str().ok_or_else(|| {
            UsageError(format!(
                "argument '{}' must be valid UTF-8",
                flag.to_string_lossy()
            ))
        })?;
        match flag_str {
            "--log" => log = Some(PathBuf::from(take_value(&mut it, flag_str)?)),
            "--allow-empty-log" => allow_empty_log = true,
            unknown => {
                return Err(UsageError(format!(
                    "unknown argument '{unknown}' for 'self-repair verify-log'"
                )));
            }
        }
    }

    let log = log.ok_or_else(|| UsageError("missing required argument '--log'".to_string()))?;
    Ok(VerifyLogArgs {
        log,
        allow_empty_log,
    })
}

/// `--flag value` 形式で次のトークンを値として取り出す。値欠落は usage エラー。
/// 値は UTF-8 検証しない（`--log` のファイルパスが非 UTF-8 でもそのまま
/// `OsString` として受理する。モジュール冒頭ドキュメント参照）。
fn take_value<I>(it: &mut std::iter::Peekable<I>, flag: &str) -> Result<OsString, UsageError>
where
    I: Iterator<Item = OsString>,
{
    it.next()
        .ok_or_else(|| UsageError(format!("missing value for '{flag}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_verify_log_with_log_arg() {
        let cmd = parse(["verify-log", "--log", "trial.jsonl"]).unwrap();
        let Command::VerifyLog(args) = cmd;
        assert_eq!(args.log, PathBuf::from("trial.jsonl"));
        assert!(!args.allow_empty_log);
    }

    #[test]
    fn parses_verify_log_with_allow_empty_log_flag() {
        let cmd = parse(["verify-log", "--log", "trial.jsonl", "--allow-empty-log"]).unwrap();
        let Command::VerifyLog(args) = cmd;
        assert!(args.allow_empty_log);
    }

    #[test]
    fn rejects_missing_log_arg() {
        let err = parse(["verify-log"]).unwrap_err();
        assert!(err.0.contains("--log"));
    }

    #[test]
    fn rejects_missing_value_for_log() {
        let err = parse(["verify-log", "--log"]).unwrap_err();
        assert!(err.0.contains("--log"));
    }

    #[test]
    fn rejects_unknown_argument() {
        let err = parse(["verify-log", "--bogus", "x"]).unwrap_err();
        assert!(err.0.contains("--bogus"));
    }

    #[test]
    fn rejects_missing_subcommand() {
        let err = parse(Vec::<&str>::new()).unwrap_err();
        assert!(err.0.contains("subcommand"));
    }

    #[test]
    fn rejects_unknown_subcommand() {
        let err = parse(["bogus"]).unwrap_err();
        assert!(err.0.contains("bogus"));
    }

    /// PR #356 codex-review P1 指摘対応: 非 UTF-8 の `--log` 値は panic せず
    /// `PathBuf` としてそのまま受理される（ファイルシステム上は有効な非 UTF-8
    /// パスの拒否は `verify_chain` の I/O エラー経路〈exit 1〉に委ねる）。
    #[cfg(unix)]
    #[test]
    fn accepts_non_utf8_log_value_without_panicking() {
        use std::os::unix::ffi::OsStrExt;

        let non_utf8_log = OsStr::from_bytes(b"log-\xff\xfe.jsonl").to_os_string();
        let cmd = parse([
            OsString::from("verify-log"),
            OsString::from("--log"),
            non_utf8_log.clone(),
        ])
        .unwrap();
        let Command::VerifyLog(args) = cmd;
        assert_eq!(args.log, PathBuf::from(non_utf8_log));
    }

    /// PR #356 codex-review P1 指摘対応: サブコマンド名自体が非 UTF-8 の場合は
    /// panic ではなく usage エラー（exit 2）に写像される。
    #[cfg(unix)]
    #[test]
    fn rejects_non_utf8_subcommand_without_panicking() {
        use std::os::unix::ffi::OsStrExt;

        let non_utf8_subcommand = OsStr::from_bytes(b"\xff\xfe").to_os_string();
        let err = parse([non_utf8_subcommand]).unwrap_err();
        assert!(err.0.contains("UTF-8"));
    }
}
