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
}

/// 本バイナリが受理するサブコマンド。`run`（3.1 節）は未実装
/// （モジュールコメント参照）のため variant を持たない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    VerifyLog(VerifyLogArgs),
}

/// `std::env::args()` の実引数（プログラム名を除く）を受け取ってパースする。
/// `main.rs` から呼ばれ、返った [`Command`] に応じて実行フローを分岐する。
pub fn parse<I, S>(args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut it = args.into_iter().map(|s| s.as_ref().to_string());
    let subcommand = it
        .next()
        .ok_or_else(|| UsageError("missing subcommand (verify-log)".to_string()))?;

    match subcommand.as_str() {
        "verify-log" => parse_verify_log(it).map(Command::VerifyLog),
        other => Err(UsageError(format!(
            "unknown subcommand '{other}' (expected verify-log)"
        ))),
    }
}

fn parse_verify_log<I>(args: I) -> Result<VerifyLogArgs, UsageError>
where
    I: Iterator<Item = String>,
{
    let mut log: Option<PathBuf> = None;

    let mut it = args.peekable();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--log" => log = Some(PathBuf::from(take_value(&mut it, &flag)?)),
            unknown => {
                return Err(UsageError(format!(
                    "unknown argument '{unknown}' for 'self-repair verify-log'"
                )));
            }
        }
    }

    let log = log.ok_or_else(|| UsageError("missing required argument '--log'".to_string()))?;
    Ok(VerifyLogArgs { log })
}

/// `--flag value` 形式で次のトークンを値として取り出す。値欠落は usage エラー。
fn take_value<I>(it: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, UsageError>
where
    I: Iterator<Item = String>,
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
}
