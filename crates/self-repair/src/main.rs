//! `self-repair` バイナリのエントリポイント（TASK-3.4 残作業・イシュー #145）。
//!
//! `docs/guardrail-self-repair-cli.md` 3.2 節「self-repair verify-log」の
//! CLI エントリポイント。試行ログ（JSON Lines・SHA-256 ハッシュチェーン。
//! 3.3 節）の整合性検証（改竄検知）を、`cargo test` を経由せず監査担当者が
//! 直接実行できる手段として提供する（`.claude/rules/security.md`
//! 「ループ試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を
//! 追跡可能にする」への対応）。
//!
//! 検証ロジック本体は lib 側の [`self_repair::verify_chain`]（`logging.rs`。
//! TASK-3.4・#145 本体・PR #340）の単一実装のみを呼び出し、CLI 側で検証を
//! 二重実装・迂回する経路は持たない（判定の迂回経路を作らない。
//! `.claude/rules/security.md` A08）。`run`（3.1 節）は別イシューのスコープの
//! ため未実装（`cli.rs` モジュールコメント参照）。
//!
//! # 終了コード契約（`verify-log`）
//! `docs/guardrail-self-repair-cli.md` 3.5 節は `run` の 3 分岐契約
//! （0/10/20/1）であり `verify-log` には verdict がないため、guardrail の
//! usage エラー区分（2.3 節）と整合する以下の契約を本イシュー差し戻し分で
//! 新たに定める（3.2 節へ追記済み）:
//!
//! | 値 | 意味 |
//! |---|---|
//! | `0` | チェーン整合（改竄なし） |
//! | `1` | 検証不合格（`LogError::ChainViolation` = 改竄・欠落検知）および
//!         内部エラー（I/O・パース失敗）。fail-closed: 読めない・壊れた
//!         ログも一律に非 0 とする |
//! | `2` | usage エラー（`--log` 欠落・未知引数） |
//!
//! この変換は本関数（[`report_verify_log_error`]）のみが行い、他の経路から
//! `0` を返さない（fail-closed。`.claude/rules/security.md` A08）。

use std::process::ExitCode;

use self_repair::cli::{self, Command, UsageError, VerifyLogArgs};
use self_repair::{LogError, verify_chain};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match cli::parse(args) {
        Ok(Command::VerifyLog(verify_log_args)) => run_verify_log(verify_log_args),
        Err(err) => report_usage_error_and_exit(&err),
    }
}

/// `self-repair verify-log` の実行フロー。`verify_chain` の結果を stdout/stderr
/// への報告と終了コードへ変換するのみの薄い統合とする（`guardrail::main::run_check`
/// と同一方針。ロジックは lib 側 [`self_repair::verify_chain`] に一本化）。
fn run_verify_log(args: VerifyLogArgs) -> ExitCode {
    match verify_chain(&args.log) {
        Ok(()) => {
            println!(
                "OK: ログチェーンの整合性を確認しました（log={}）",
                args.log.display()
            );
            ExitCode::from(0)
        }
        Err(err) => report_verify_log_error(&err),
    }
}

/// [`LogError`] を人間可読なメッセージとして stderr へ出力し、終了コード `1`
/// （検証不合格・内部エラーの両方を含む fail-closed 契約。モジュール冒頭
/// ドキュメント参照）へ変換する。`ChainViolation`（改竄・欠落検知）の内容は
/// `seq`・理由のみを含み、`LogError::Display`（`logging.rs`）実装同様
/// payload 本文は出力しない。
fn report_verify_log_error(err: &LogError) -> ExitCode {
    eprintln!("self-repair verify-log: {err}");
    ExitCode::from(1)
}

/// CLI 引数の usage エラーを stderr へ出力し、終了コード `2` へ変換する
/// （guardrail の usage エラー区分と整合。モジュール冒頭ドキュメント参照）。
fn report_usage_error_and_exit(err: &UsageError) -> ExitCode {
    eprintln!("self-repair: {err}");
    ExitCode::from(2)
}
