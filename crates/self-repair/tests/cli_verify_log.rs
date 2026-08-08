//! `self-repair verify-log` CLI 実行経路の統合テスト（TASK-3.4 残作業・
//! イシュー #145。完走判定基準 4 対応）。
//!
//! `crates/self-repair/tests/logging_chain.rs` が `self_repair::verify_chain`
//! を lib として直接呼ぶのに対し、本ファイルは「監査担当者が `cargo test`
//! 経由でなく CLI から直接検証できる」という受け入れ条件
//! （`docs/guardrail-self-repair-cli.md` 3.2 節）を、実バイナリの起動
//! （`env!("CARGO_BIN_EXE_self-repair")`）を通して固定する
//! （`crates/guardrail/tests/cli_three_branch.rs` と同一パターン）。
//! 実機（CUDA・Metal）依存はないため `#[ignore]` 分離は不要
//! （`.claude/rules/coding-rust.md`）。

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use self_repair::outcome::LoopOutcome;
use self_repair::report::{AttemptOutcome, AttemptRecord, LoopReport};
use self_repair::{LogWriter, RepairKind};

fn self_repair_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_self-repair"))
}

/// テストごとに衝突しない一時ファイルパスを作る
/// （`tests/logging_chain.rs::unique_log_path` と同一方式）。
fn unique_log_path(test_name: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "self-repair-cli-verify-log-it-{}-{test_name}-{seq}.jsonl",
        std::process::id()
    ))
}

/// 正当な 1 レコードのログを新規作成する。
fn write_valid_log(path: &std::path::Path) {
    let report = LoopReport {
        kind: RepairKind::FeatureAddition,
        outcome: LoopOutcome::Adopted,
        attempts: vec![AttemptRecord {
            attempt: 1,
            duration: Duration::from_millis(10),
            outcome: AttemptOutcome::Adopted,
        }],
        total_duration: Duration::from_millis(123),
    };
    LogWriter::open(path)
        .expect("新規ログファイルを開けること")
        .append_report(&report)
        .expect("LoopReport の追記に失敗しないこと");
}

/// 受け入れ条件: 正常チェーン → exit 0。
#[test]
fn verify_log_on_valid_chain_exits_zero() {
    let path = unique_log_path("valid_chain");
    write_valid_log(&path);

    let output = self_repair_bin()
        .args(["verify-log", "--log", path.to_str().unwrap()])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(0));
    let _ = std::fs::remove_file(&path);
}

/// 受け入れ条件: 改竄チェーン（`hash` フィールドの書き換え）→ exit 1・
/// stderr に改竄検知メッセージ。
#[test]
fn verify_log_on_tampered_hash_field_exits_one_with_message() {
    let path = unique_log_path("tampered_hash");
    write_valid_log(&path);

    let content = std::fs::read_to_string(&path).expect("読めること");
    let bogus_hash = "0".repeat(64);
    let hash_key = "\"hash\":\"";
    let start = content.find(hash_key).expect("hash フィールドがあること") + hash_key.len();
    let end = content[start..].find('"').expect("終端があること") + start;
    let tampered = format!("{}{}{}", &content[..start], bogus_hash, &content[end..]);
    assert_ne!(content, tampered, "実際に書き換わっていること");
    std::fs::write(&path, tampered).expect("書き戻せること");

    let output = self_repair_bin()
        .args(["verify-log", "--log", path.to_str().unwrap()])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("改竄") || stderr.contains("欠落"),
        "stderr に改竄検知メッセージが含まれること: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// 受け入れ条件: レコード削除（中間行の削除。末尾からの切り詰めではない）
/// による改竄が検知される。`append_report` は 1 回の呼び出しで
/// `loop_start → detection → attempt → loop_outcome` の複数レコードを
/// 書き出す（`logging.rs::append_stages`）ため、中間の 1 行を取り除くと
/// `seq` の連続性が崩れ `ChainViolation` になる（`docs/self-repair-log-format.md`
/// 6 節「レコード削除・順序入れ替え（いずれも `verify_chain` が検知）」）。
#[test]
fn verify_log_on_deleted_middle_record_exits_one() {
    let path = unique_log_path("deleted_middle_record");
    write_valid_log(&path);

    let content = std::fs::read_to_string(&path).expect("読めること");
    let lines: Vec<&str> = content.lines().collect();
    assert!(
        lines.len() >= 3,
        "1 回の append_report で複数レコードが書かれること: {} 行",
        lines.len()
    );
    // 中間の 1 レコード（index 1）を取り除く（末尾切り詰めではなく削除）。
    let mut remaining: Vec<&str> = lines;
    remaining.remove(1);
    let tampered = format!("{}\n", remaining.join("\n"));
    std::fs::write(&path, tampered).expect("書き戻せること");

    let output = self_repair_bin()
        .args(["verify-log", "--log", path.to_str().unwrap()])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("改竄") || stderr.contains("欠落"),
        "stderr に改竄検知メッセージが含まれること: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
}

/// 既知の限界: 末尾切り詰め（末尾レコードの削除）は `seq` 連続性・
/// `prev_hash` チェーンのいずれも破らないため `verify_chain` 単体では
/// 検知できない（`docs/self-repair-log-format.md` 6〜7 節。外部アンカー
/// 運用でカバーする対象であり、実装計画 §8 のスコープ外）。回帰検出のため
/// 「削除後も exit 0 のままである」という現状の挙動を固定する（将来
/// 誤って挙動が変わった場合にこのテストが失敗して気付けるようにする）。
#[test]
fn verify_log_on_tail_truncation_exits_zero_known_limitation() {
    let path = unique_log_path("tail_truncation");
    let report_a = LoopReport {
        kind: RepairKind::FeatureAddition,
        outcome: LoopOutcome::Adopted,
        attempts: vec![AttemptRecord {
            attempt: 1,
            duration: Duration::from_millis(1),
            outcome: AttemptOutcome::Adopted,
        }],
        total_duration: Duration::from_millis(1),
    };
    let report_b = report_a.clone();
    let mut writer = LogWriter::open(&path).expect("開けること");
    writer.append_report(&report_a).expect("1 件目の追記");
    writer.append_report(&report_b).expect("2 件目の追記");

    // 1 件目のレコード群のみを残す形で末尾を切り詰める（2 件目の削除を模擬）。
    let content = std::fs::read_to_string(&path).expect("読めること");
    let first_report_line_count = content.lines().count() / 2;
    let split_at = content
        .lines()
        .take(first_report_line_count)
        .map(|l| l.len() + 1)
        .sum::<usize>();
    std::fs::write(&path, &content[..split_at]).expect("書き戻せること");

    let output = self_repair_bin()
        .args(["verify-log", "--log", path.to_str().unwrap()])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(0));
    let _ = std::fs::remove_file(&path);
}

/// 受け入れ条件: `--log` 未指定 → exit 2（usage エラー）。
#[test]
fn verify_log_without_log_arg_exits_two() {
    let output = self_repair_bin()
        .args(["verify-log"])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--log"),
        "stderr に --log 欠落の説明が含まれること: {stderr}"
    );
}

/// 受け入れ条件: 存在しないファイルパス → exit 1（I/O エラーも fail-closed
/// で非 0）。
#[test]
fn verify_log_on_missing_file_exits_one() {
    let path = unique_log_path("missing_file");
    // ファイルを作らずそのままパスを渡す。

    let output = self_repair_bin()
        .args(["verify-log", "--log", path.to_str().unwrap()])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(1));
}

/// 受け入れ条件: 未知のサブコマンド → exit 2（usage エラー）。
#[test]
fn unknown_subcommand_exits_two() {
    let output = self_repair_bin()
        .args(["bogus"])
        .output()
        .expect("failed to run self-repair binary");

    assert_eq!(output.status.code(), Some(2));
}
