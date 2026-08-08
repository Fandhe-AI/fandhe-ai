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
//! | `0` | チェーン整合（改竄なし）。レコード 0 件（空ログ）は `--allow-empty-log` を明示指定した場合のみ `0` とする |
//! | `1` | 検証不合格（`LogError::ChainViolation` = 改竄・欠落検知）・内部エラー（I/O・パース失敗）・`--allow-empty-log` 未指定での空ログ検知。fail-closed: 読めない・壊れたログも一律に非 0 とする |
//! | `2` | usage エラー（`--log` 欠落・未知引数） |
//!
//! この変換は本関数（[`report_verify_log_error`]）のみが行い、他の経路から
//! `0` を返さない（fail-closed。`.claude/rules/security.md` A08）。
//!
//! 空ログの扱いは PR #356 codex-review P1 指摘対応で変更した:
//! 当初は「空ログか末尾切り詰めかを本コマンド単体では区別できない」ことを
//! 理由に `WARN:` メッセージ付きで exit 0 としていたが、終了コードのみを
//! 見る監査自動化（`.claude/rules/security.md` A08）にとってはログ全削除の
//! 改竄を「検証成功」として素通しする経路になっていた。既定を fail-closed
//! な exit 1 へ変え、空ログを正当な運用として扱いたい呼び出し元は
//! `--allow-empty-log` を明示指定する（`docs/guardrail-self-repair-cli.md`
//! 3.2 節も同時更新）。

use std::process::ExitCode;

use self_repair::cli::{self, Command, UsageError, VerifyLogArgs};
use self_repair::{LogError, VerifyChainSummary, verify_chain};

fn main() -> ExitCode {
    // `std::env::args()` ではなく `args_os()` を使う理由は `cli` モジュール
    // 冒頭ドキュメント参照（非 UTF-8 引数での panic を避けるため。PR #356
    // codex-review P1 指摘対応）。
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

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
        // レコード 0 件は「チェーン検証エラーなし」であって「改竄されていない
        // ことの証明」ではない（末尾切り詰めで空になったログも同じ結果になる。
        // `verify_chain` モジュール冒頭ドキュメント参照）。ログ全削除による
        // 改竄と正当な空ログを区別できないため、既定では fail-closed に
        // exit 1 とする（PR #356 codex-review P1 指摘対応: 終了コードのみを
        // 見る監査自動化がこれまでの無条件 exit 0 を「検証成功」として
        // 見逃していた）。空ログを正当な運用として扱いたい場合のみ
        // `--allow-empty-log` を明示指定させ、その場合は `WARN:` 付きで
        // exit 0 のまま外部アンカー突合（docs/self-repair-log-format.md §7）を
        // 促す。
        Ok(summary) if summary.record_count == 0 && !args.allow_empty_log => {
            eprintln!(
                "self-repair verify-log: ログチェーンにレコードがありません（log={}, records=0）。空ログか末尾切り詰め（ログ全削除を含む改竄）かは本コマンド単体では区別できないため fail-closed で不合格とします。空ログを許容する場合は --allow-empty-log を指定してください",
                args.log.display()
            );
            ExitCode::from(1)
        }
        Ok(summary) if summary.record_count == 0 => {
            println!(
                "WARN: ログチェーンにレコードがありません（log={}, records=0）。--allow-empty-log が指定されているため exit 0 とします。空ログか末尾切り詰めかは本コマンド単体では区別できません。外部アンカー運用（docs/self-repair-log-format.md 7 節）との突合を確認してください",
                args.log.display()
            );
            ExitCode::from(0)
        }
        // `record_count > 0` の分岐でのみ到達する。`verify_chain` の実装上
        // `record_count > 0` は `last_seq.is_some()` と等価であり、この分岐に
        // 限れば `None` は生じない（`unwrap_or_default()` で握り潰すと万一の
        // 不整合時に `last_seq=0` を誤表示しうるため、`Some` を明示的に照合する）。
        Ok(VerifyChainSummary {
            record_count,
            last_seq: Some(last_seq),
            last_hash,
        }) => {
            // 監査担当者が外部アンカー（書き込み直後に別経路へ記録した最終
            // hash・seq）と突合できるよう、成功メッセージにレコード件数・
            // 最終 seq・最終 hash を含める（security.md A08 の意図。
            // `verify_chain` モジュール冒頭ドキュメント参照）。
            println!(
                "OK: ログチェーンの整合性を確認しました（log={}, records={}, last_seq={}, last_hash={}）",
                args.log.display(),
                record_count,
                last_seq,
                last_hash
            );
            ExitCode::from(0)
        }
        // record_count > 0 かつ last_seq == None は verify_chain の不変条件上
        // 到達しないはずだが、将来の実装変更でこの不変条件が崩れた場合に
        // 静かに誤った「OK」を出さないよう、fail-closed でエラー扱いにする
        // （security.md A08「判定の迂回経路を作らない」と同じ思想）。
        Ok(summary) => {
            eprintln!(
                "self-repair verify-log: 内部不整合を検知しました（log={}, records={}, last_seq=None）。verify_chain の実装を確認してください",
                args.log.display(),
                summary.record_count
            );
            ExitCode::from(1)
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
