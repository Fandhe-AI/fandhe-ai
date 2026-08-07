//! 自作コマンドライン引数パーサ。
//!
//! `clap` は `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれないため
//! （TASK-4.1a 計画 2.1 節）、`std::env::args` ベースで
//! `docs/guardrail-self-repair-cli.md` 1.2〜1.3 節の `check`／`eval` 引数一覧を
//! 自作パースする。usage エラー（未知引数・値欠落・不正なサブコマンド）は
//! 2.3 節の終了コード契約における `2` に対応する
//! `GuardrailError::UsageError`／`GuardrailError::InjectedSignalsNotAllowed`
//! を返す。`main.rs` から呼ばれ、返った `Command` に応じて実行フローを分岐する。

use std::path::PathBuf;

use crate::error::GuardrailError;

/// `--format` の出力形式（1.2〜1.3 節で `check`／`eval` 共通）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    fn parse(s: &str) -> Result<Self, GuardrailError> {
        match s {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            other => Err(GuardrailError::UsageError(format!(
                "unknown --format value '{other}' (expected text|json)"
            ))),
        }
    }
}

/// `guardrail check` の引数（1.2 節）。
#[derive(Debug, Clone, PartialEq)]
pub struct CheckArgs {
    pub baseline: String,
    pub change_id: Option<String>,
    pub config: Option<PathBuf>,
    pub preset: String,
    pub repo: PathBuf,
    /// `--signals` の値。入口ガード（環境変数 `GUARDRAIL_ALLOW_INJECTED_SIGNALS=1`）を
    /// 通過した場合のみ `Some` になる。ガードを通過しない `--signals` 指定は
    /// `parse` の時点で `GuardrailError::InjectedSignalsNotAllowed` を返す。
    pub signals: Option<PathBuf>,
    pub format: OutputFormat,
    pub output: Option<PathBuf>,
}

/// `guardrail eval` の引数（1.3 節）。
#[derive(Debug, Clone, PartialEq)]
pub struct EvalArgs {
    pub dataset: PathBuf,
    pub config: Option<PathBuf>,
    pub preset: String,
    pub repo: PathBuf,
    pub format: OutputFormat,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Check(CheckArgs),
    Eval(EvalArgs),
}

const DEFAULT_DATASET_DIR: &str = "crates/guardrail/tests/fixtures/labeled-changes";

/// `std::env::args()` の実引数（プログラム名を除く）と、`--signals` 入口ガード用の
/// 環境変数参照を受け取ってパースする。環境変数を引数化しているのはテスト容易性のため
/// （`main.rs` からは `std::env::var` をそのまま渡す）。
pub fn parse<I, S, E>(args: I, env_lookup: E) -> Result<Command, GuardrailError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
    E: Fn(&str) -> Option<String>,
{
    let mut it = args.into_iter().map(|s| s.as_ref().to_string());
    let subcommand = it
        .next()
        .ok_or_else(|| GuardrailError::UsageError("missing subcommand (check|eval)".to_string()))?;

    match subcommand.as_str() {
        "check" => parse_check(it, env_lookup).map(Command::Check),
        "eval" => parse_eval(it).map(Command::Eval),
        other => Err(GuardrailError::UsageError(format!(
            "unknown subcommand '{other}' (expected check|eval)"
        ))),
    }
}

fn parse_check<I, E>(args: I, env_lookup: E) -> Result<CheckArgs, GuardrailError>
where
    I: Iterator<Item = String>,
    E: Fn(&str) -> Option<String>,
{
    let mut baseline = "baseline".to_string();
    let mut change_id = None;
    let mut config = None;
    let mut preset = "default".to_string();
    let mut repo = PathBuf::from(".");
    let mut signals_raw: Option<String> = None;
    let mut format = OutputFormat::Text;
    let mut output = None;

    let mut it = args.peekable();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--baseline" => baseline = take_value(&mut it, &flag)?,
            "--change-id" => change_id = Some(take_value(&mut it, &flag)?),
            "--config" => config = Some(PathBuf::from(take_value(&mut it, &flag)?)),
            "--preset" => preset = take_value(&mut it, &flag)?,
            "--repo" => repo = PathBuf::from(take_value(&mut it, &flag)?),
            "--signals" => signals_raw = Some(take_value(&mut it, &flag)?),
            "--format" => format = OutputFormat::parse(&take_value(&mut it, &flag)?)?,
            "--output" => output = Some(PathBuf::from(take_value(&mut it, &flag)?)),
            unknown => {
                return Err(GuardrailError::UsageError(format!(
                    "unknown argument '{unknown}' for 'guardrail check'"
                )));
            }
        }
    }

    // `--signals` の迂回防止入口ガード（1.2 節）。環境変数未設定時は usage エラー
    // （終了コード 2）で拒否し、CI 契約検証ジョブ以外からの注入を防ぐ（A08）。
    let signals = match signals_raw {
        Some(raw) => {
            let allowed = env_lookup("GUARDRAIL_ALLOW_INJECTED_SIGNALS").as_deref() == Some("1");
            if !allowed {
                return Err(GuardrailError::InjectedSignalsNotAllowed);
            }
            Some(PathBuf::from(raw))
        }
        None => None,
    };

    Ok(CheckArgs {
        baseline,
        change_id,
        config,
        preset,
        repo,
        signals,
        format,
        output,
    })
}

fn parse_eval<I>(args: I) -> Result<EvalArgs, GuardrailError>
where
    I: Iterator<Item = String>,
{
    let mut dataset = PathBuf::from(DEFAULT_DATASET_DIR);
    let mut config = None;
    let mut preset = "default".to_string();
    let mut repo = PathBuf::from(".");
    let mut format = OutputFormat::Text;
    let mut output = None;

    let mut it = args.peekable();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--dataset" => dataset = PathBuf::from(take_value(&mut it, &flag)?),
            "--config" => config = Some(PathBuf::from(take_value(&mut it, &flag)?)),
            "--preset" => preset = take_value(&mut it, &flag)?,
            "--repo" => repo = PathBuf::from(take_value(&mut it, &flag)?),
            "--format" => format = OutputFormat::parse(&take_value(&mut it, &flag)?)?,
            "--output" => output = Some(PathBuf::from(take_value(&mut it, &flag)?)),
            unknown => {
                return Err(GuardrailError::UsageError(format!(
                    "unknown argument '{unknown}' for 'guardrail eval'"
                )));
            }
        }
    }

    Ok(EvalArgs {
        dataset,
        config,
        preset,
        repo,
        format,
        output,
    })
}

/// `--flag value` 形式で次のトークンを値として取り出す。値欠落は usage エラー
/// （終了コード 2）とする。
fn take_value<I>(it: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, GuardrailError>
where
    I: Iterator<Item = String>,
{
    it.next()
        .ok_or_else(|| GuardrailError::UsageError(format!("missing value for '{flag}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env_allow(key: &str) -> Option<String> {
        if key == "GUARDRAIL_ALLOW_INJECTED_SIGNALS" {
            Some("1".to_string())
        } else {
            None
        }
    }

    #[test]
    fn parses_check_with_defaults() {
        let cmd = parse(["check"], no_env).unwrap();
        match cmd {
            Command::Check(args) => {
                assert_eq!(args.baseline, "baseline");
                assert_eq!(args.preset, "default");
                assert_eq!(args.repo, PathBuf::from("."));
                assert_eq!(args.format, OutputFormat::Text);
                assert!(args.signals.is_none());
            }
            _ => panic!("expected Check"),
        }
    }

    #[test]
    fn parses_check_with_all_arguments() {
        let cmd = parse(
            [
                "check",
                "--baseline",
                "main",
                "--change-id",
                "abc",
                "--config",
                "g.toml",
                "--preset",
                "strict",
                "--repo",
                "/repo",
                "--format",
                "json",
                "--output",
                "out.json",
            ],
            no_env,
        )
        .unwrap();
        let Command::Check(args) = cmd else {
            panic!("expected Check")
        };
        assert_eq!(args.baseline, "main");
        assert_eq!(args.change_id.as_deref(), Some("abc"));
        assert_eq!(args.config, Some(PathBuf::from("g.toml")));
        assert_eq!(args.preset, "strict");
        assert_eq!(args.repo, PathBuf::from("/repo"));
        assert_eq!(args.format, OutputFormat::Json);
        assert_eq!(args.output, Some(PathBuf::from("out.json")));
    }

    #[test]
    fn rejects_signals_without_env_guard() {
        let err = parse(["check", "--signals", "s.json"], no_env).unwrap_err();
        assert!(matches!(err, GuardrailError::InjectedSignalsNotAllowed));
    }

    #[test]
    fn accepts_signals_with_env_guard() {
        let cmd = parse(["check", "--signals", "s.json"], env_allow).unwrap();
        let Command::Check(args) = cmd else {
            panic!("expected Check")
        };
        assert_eq!(args.signals, Some(PathBuf::from("s.json")));
    }

    #[test]
    fn rejects_unknown_argument() {
        let err = parse(["check", "--bogus", "x"], no_env).unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn rejects_missing_value() {
        let err = parse(["check", "--baseline"], no_env).unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn rejects_missing_subcommand() {
        let err = parse(Vec::<&str>::new(), no_env).unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn rejects_unknown_subcommand() {
        let err = parse(["bogus"], no_env).unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn parses_eval_with_defaults() {
        let cmd = parse(["eval"], no_env).unwrap();
        let Command::Eval(args) = cmd else {
            panic!("expected Eval")
        };
        assert_eq!(args.dataset, PathBuf::from(DEFAULT_DATASET_DIR));
        assert_eq!(args.preset, "default");
    }

    #[test]
    fn rejects_invalid_format_value() {
        let err = parse(["check", "--format", "xml"], no_env).unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }
}
