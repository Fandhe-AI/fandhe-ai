//! `guardrail` バイナリのエントリポイント。
//!
//! CI・`self-repair` から呼び出される本番相当経路（`docs/guardrail-self-repair-cli.md`
//! 1.2 節）の CLI 実行フローを構成する。判定ロジック自体は `crates/guardrail`
//! の lib 側（`self-repair` が直接呼び出す経路。3.4 節）に置く方針のため、
//! ここでは `cli::parse` → 設定解決 → `guardrail::check::run_injected`／
//! `run_measured` 呼び出し → レポート生成・出力 → 終了コード変換、という
//! 薄い統合のみを担う。
//!
//! TASK-4.1c（イシュー #106）で `guardrail::check`（injected／measured 両経路）
//! が [`guardrail::decision::decide`] へ結線された。3 分岐（auto_apply/0・
//! escalate/10・reject/20）は代表ケースで CLI 統合テスト
//! （`tests/cli_three_branch.rs`）が固定する。判定不能（`--baseline` 実在
//! 確認失敗・シグナル入力の検証失敗等）は `GuardrailError` として伝播し、
//! `0`/`10`/`20` のいずれへも丸めず内部エラー（終了コード `1`）または usage
//! エラー（終了コード `2`）となる（fail-closed 契約。`.claude/rules/security.md`
//! A08。`report_error_and_exit` 参照）。
//!
//! `eval` の評価ロジック本体（全 fixture 一括評価・率集計）は
//! `guardrail::eval::run`（TASK-4.3a・イシュー #115）へ委譲する。ここでは
//! 設定解決（`--config`/`--preset`）→ `eval::run` 呼び出し → 出力
//! （`--format`/`--output`）→ 終了コード変換（`EvalExitCode::from_pass`）
//! という薄い統合のみを担う（`run_check` と同じ「lib 側にロジックを置き
//! bin は統合層に留める」方針を踏襲）。

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use guardrail::check;
use guardrail::cli::{self, CheckArgs, Command, EvalArgs, OutputFormat};
use guardrail::config::{self, PresetName};
use guardrail::error::GuardrailError;
use guardrail::eval;
use guardrail::eval::report::EvalReport;
use guardrail::exit_code::{EvalExitCode, GuardrailExitCode};
use guardrail::report::Report;
use guardrail::signals::Signals;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env_lookup = |key: &str| std::env::var(key).ok();

    match cli::parse(args, env_lookup) {
        Ok(Command::Check(check_args)) => run_check(check_args),
        Ok(Command::Eval(eval_args)) => run_eval(eval_args),
        Err(err) => report_error_and_exit(&err),
    }
}

/// `guardrail check` の実行フロー。
///
/// 入力解決（引数 → 設定 → シグナル）と検証・レポート出力までを実装し、
/// verdict は常に `escalate` の暫定固定とする（実装計画 2.2 節）。
fn run_check(args: CheckArgs) -> ExitCode {
    match run_check_inner(&args) {
        Ok(report) => {
            emit_report(&report, args.format, args.output.as_deref());
            GuardrailExitCode::from_verdict(report.verdict).into_process_exit_code()
        }
        Err(err) => report_error_and_exit(&err),
    }
}

fn run_check_inner(args: &CheckArgs) -> Result<Report, GuardrailError> {
    let preset = PresetName::parse(&args.preset)?;
    let config = config::resolve(args.config.as_deref(), &args.repo, preset)?;

    match &args.signals {
        Some(path) => {
            let raw = fs::read_to_string(path).map_err(|source| GuardrailError::Io {
                path: path.clone(),
                source,
            })?;
            let signals = Signals::from_json_str(&raw)?;
            check::run_injected(&signals, &config, args.change_id.clone())
        }
        None => check::run_measured(&args.repo, &args.baseline, &config, args.change_id.clone()),
    }
}

/// `guardrail eval` の実行フロー。評価ロジック本体は `guardrail::eval::run`
/// （TASK-4.3a・イシュー #115）に委譲し、ここでは設定解決・出力・終了コード
/// 変換のみを行う（モジュールコメント参照）。
fn run_eval(args: EvalArgs) -> ExitCode {
    match run_eval_inner(&args) {
        Ok(report) => {
            emit_eval_report(&report, args.format, args.output.as_deref());
            EvalExitCode::from_pass(report.pass()).into_process_exit_code()
        }
        Err(err) => report_error_and_exit(&err),
    }
}

fn run_eval_inner(args: &EvalArgs) -> Result<EvalReport, GuardrailError> {
    let preset = PresetName::parse(&args.preset)?;
    let config = config::resolve(args.config.as_deref(), &args.repo, preset)?;
    eval::run(&args.dataset, &config.thresholds)
}

/// `eval::EvalReport` を `--format`/`--output` に応じて出力する。`emit_report`
/// （`check` 用）と同じく serde_json のエスケープに一任し文字列連結で JSON を
/// 組み立てない（2.5 節 A03 対策）。
fn emit_eval_report(report: &EvalReport, format: OutputFormat, output: Option<&Path>) {
    let json = match serde_json::to_string_pretty(report) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("internal error: failed to serialize eval report: {e}");
            return;
        }
    };

    match format {
        OutputFormat::Json => println!("{json}"),
        OutputFormat::Text => {
            println!(
                "total={} miss_rate={:.2}%(ok={}) false_positive_rate={:.2}%(ok={}) pass={}",
                report.total_count,
                report.miss_rate_pct,
                report.miss_rate_ok,
                report.false_positive_rate_pct,
                report.false_positive_rate_ok,
                report.pass(),
            );
            for item in &report.items {
                println!(
                    "  {} expected={} actual={} correct={} known_blind_spot={}",
                    item.change_id,
                    item.expected_verdict,
                    item.actual_verdict,
                    item.correct,
                    item.known_blind_spot
                );
            }
        }
    }

    if let Some(path) = output
        && let Err(e) = write_file(path, &json)
    {
        eprintln!(
            "warning: failed to write --output '{}': {e}",
            path.display()
        );
    }
}

fn emit_report(report: &Report, format: OutputFormat, output: Option<&Path>) {
    // レポートは常に serde_json 経由でシリアライズする（文字列連結で JSON を
    // 組み立てない。`docs/guardrail-self-repair-cli.md` 2.5 節 A03 対策）。
    let json = match serde_json::to_string_pretty(report) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("internal error: failed to serialize report: {e}");
            return;
        }
    };

    match format {
        OutputFormat::Json => println!("{json}"),
        OutputFormat::Text => println!(
            "verdict={:?} reason=\"{}\" signal_source={:?}",
            report.verdict, report.reason, report.signal_source
        ),
    }

    if let Some(path) = output
        && let Err(e) = write_file(path, &json)
    {
        eprintln!(
            "warning: failed to write --output '{}': {e}",
            path.display()
        );
    }
}

fn write_file(path: &Path, content: &str) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(content.as_bytes())
}

/// `GuardrailError` を人間可読なメッセージとして stderr へ出力し、
/// `docs/guardrail-self-repair-cli.md` 2.3 節の終了コード契約へ変換する。
/// 変換はこの関数のみが行い、他の経路から `0` を返さない
/// （fail-closed。`.claude/rules/security.md` A08）。
fn report_error_and_exit(err: &GuardrailError) -> ExitCode {
    eprintln!("guardrail: {err}");
    match err {
        GuardrailError::UsageError(_) | GuardrailError::InjectedSignalsNotAllowed => {
            GuardrailExitCode::UsageError.into_process_exit_code()
        }
        GuardrailError::InvalidInput(_)
        | GuardrailError::Io { .. }
        | GuardrailError::InconsistentDecisionInput { .. }
        | GuardrailError::DiffSpawn { .. }
        | GuardrailError::DiffFailed { .. }
        | GuardrailError::DiffUnexpectedFormat { .. }
        | GuardrailError::GateSpawn { .. } => {
            GuardrailExitCode::InternalError.into_process_exit_code()
        }
    }
}
