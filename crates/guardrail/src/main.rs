//! `guardrail` バイナリのエントリポイント。
//!
//! CI・`self-repair` から呼び出される本番相当経路（`docs/guardrail-self-repair-cli.md`
//! 1.2 節）の CLI 実行フローを構成する。判定ロジック自体は `crates/guardrail`
//! の lib 側（`self-repair` が直接呼び出す経路。3.4 節）に置く方針のため、
//! ここでは `cli::parse` → 設定解決 → 入力解決 → レポート生成・出力 →
//! 終了コード変換、という薄い統合のみを担う。
//!
//! TASK-4.1a（本イシュー）のスコープでは 5 条件判定ロジック（#105）が
//! 未実装のため、`check` は常に `Verdict::Escalate`（終了コード `10`）を
//! 返す暫定固定とする。「判定不能時に自動適用へ倒れない」fail-closed 契約
//! （`.claude/rules/security.md` A08）を骨格段階から満たすための設計判断
//! （実装計画 2.2 節）。
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

use guardrail::cli::{self, CheckArgs, Command, EvalArgs, OutputFormat};
use guardrail::config::{self, PresetName};
use guardrail::decision::Verdict;
use guardrail::error::GuardrailError;
use guardrail::eval;
use guardrail::eval::report::EvalReport;
use guardrail::exit_code::{EvalExitCode, GuardrailExitCode};
use guardrail::report::{GateOutcome, Report, SCHEMA_VERSION, SignalSource};
use guardrail::signals::{GateResult, Signals};

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
    // 設定は現段階では判定ロジック（#105）に渡していないが、値域検証
    // （config.rs）自体は骨格段階でも通す（不正な guardrail.toml を早期検出する）。
    let _config = config::resolve(args.config.as_deref(), &args.repo, preset)?;

    let (signal_source, signals) = match &args.signals {
        Some(path) => {
            let raw = fs::read_to_string(path).map_err(|source| GuardrailError::Io {
                path: path.clone(),
                source,
            })?;
            (SignalSource::Injected, Some(Signals::from_json_str(&raw)?))
        }
        None => (SignalSource::Measured, None),
    };

    let reason = match &signals {
        // #105/#106 で判定ロジックが移植されるまでは、シグナル入力があっても
        // 判定は行わず常に escalate とする（実装計画 2.2 節）。
        Some(_) => "判定ロジック未実装（5 条件判定は TASK-4.1b/#105 で移植予定）。\
             シグナル入力は受理済みだが判定には未使用"
            .to_string(),
        None => "シグナル未取得: 実シグナル計測（本番相当経路）は TASK-4.1b/#105 以降で実装する。\
             骨格段階では escalate に固定する"
            .to_string(),
    };

    let report = match &signals {
        Some(s) => Report {
            schema_version: SCHEMA_VERSION.to_string(),
            signal_source,
            change_id: args.change_id.clone(),
            lines_changed: s.lines_changed,
            public_api_broken: s.public_api_broken,
            gaming_suspected: s.gaming_suspected,
            build_result: gate_outcome(s.build_result),
            test_result: gate_outcome(s.test_result),
            clippy_result: gate_outcome(s.clippy_result),
            bench_measurements_pct: s.bench_measurements_pct.clone(),
            bench_median_pct: guardrail::report::median(&s.bench_measurements_pct).map_err(
                |e| {
                    GuardrailError::InvalidInput(format!(
                        "bench_measurements_pct の中央値算出に失敗: {e}"
                    ))
                },
            )?,
            applied_exclusion_rule_ids: Vec::new(),
            verdict: Verdict::Escalate,
            reason,
        },
        None => Report {
            schema_version: SCHEMA_VERSION.to_string(),
            signal_source,
            change_id: args.change_id.clone(),
            lines_changed: 0,
            public_api_broken: false,
            gaming_suspected: false,
            build_result: GateOutcome::Fail,
            test_result: GateOutcome::Fail,
            clippy_result: GateOutcome::Fail,
            bench_measurements_pct: Vec::new(),
            bench_median_pct: 0.0,
            applied_exclusion_rule_ids: Vec::new(),
            verdict: Verdict::Escalate,
            reason,
        },
    };

    Ok(report)
}

fn gate_outcome(result: GateResult) -> GateOutcome {
    match result {
        GateResult::Pass => GateOutcome::Pass,
        GateResult::Fail => GateOutcome::Fail,
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
        | GuardrailError::DiffUnexpectedFormat { .. } => {
            GuardrailExitCode::InternalError.into_process_exit_code()
        }
    }
}
