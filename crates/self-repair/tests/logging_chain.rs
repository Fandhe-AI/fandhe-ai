//! ログ形式（TASK-3.4・イシュー #145）の受け入れ条件を、クレート公開 API
//! （`self_repair::{LogWriter, LogError, verify_chain}`）のみを使って検証する
//! 統合テスト。
//!
//! `crates/self-repair/src/logging.rs` 内の `#[cfg(test)]` ユニットテストは
//! クレート内部の `LogRecord`（非公開）を直接パースして段階列・ハッシュ
//! フィールドまで検証するのに対し、本テストは「外部の呼び出し元（将来の
//! CLI バイナリ想定。`docs/guardrail-self-repair-cli.md` 3 節）が公開 API
//! だけで意図どおり使えるか」を確認する（実装計画 6 章「検証方法」3 の
//! 統合テスト観点）。実機（CUDA・Metal）依存はないため `#[ignore]` 分離は
//! 不要（`.claude/rules/coding-rust.md`）。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use self_repair::outcome::LoopOutcome;
use self_repair::report::{AttemptOutcome, AttemptRecord, LoopFailure, LoopReport};
use self_repair::{LogError, LogWriter, RepairKind, SelfRepairError};

/// テストごとに衝突しない一時ファイルパスを作る。`tempfile` クレート
/// （`.claude/rules/deps-policy.md` の許容依存 8 区分外）を使わず、
/// `crates/self-repair/tests/verify_gates_integration.rs` 等の既存統合
/// テストと同じ `std::env::temp_dir()` + プロセス ID + 単調増加カウンタ
/// 方式で一意性を確保する。
fn unique_log_path(test_name: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "self-repair-logging-chain-it-{}-{test_name}-{seq}.jsonl",
        std::process::id()
    ))
}

fn sample_report(outcome: LoopOutcome, attempts: Vec<AttemptRecord>) -> LoopReport {
    LoopReport {
        kind: RepairKind::FeatureAddition,
        outcome,
        attempts,
        total_duration: Duration::from_millis(456),
    }
}

/// 受け入れ条件: 実ファイルへの追記 → `verify_chain` 通過。
#[test]
fn append_report_then_verify_chain_succeeds() {
    let path = unique_log_path("append_then_verify");
    let report = sample_report(
        LoopOutcome::Adopted,
        vec![AttemptRecord {
            attempt: 1,
            duration: Duration::from_millis(10),
            outcome: AttemptOutcome::Adopted,
        }],
    );

    LogWriter::open(&path)
        .expect("新規ログファイルを開けること")
        .append_report(&report)
        .expect("LoopReport の追記に失敗しないこと");

    self_repair::verify_chain(&path).expect("追記直後の verify_chain が成功すること");
    let _ = std::fs::remove_file(&path);
}

/// 受け入れ条件: `LoopFailure` 経路（段階の実行自体が失敗したケース）でも
/// 追記・検証が成立する。
#[test]
fn append_failure_then_verify_chain_succeeds() {
    let path = unique_log_path("append_failure_then_verify");
    let failure = LoopFailure {
        error: SelfRepairError::FixGeneration {
            attempt: 1,
            reason: "修正生成に失敗".to_string(),
        },
        attempts: Vec::new(),
    };

    LogWriter::open(&path)
        .expect("新規ログファイルを開けること")
        .append_failure(&failure)
        .expect("LoopFailure の追記に失敗しないこと");

    self_repair::verify_chain(&path).expect("LoopFailure 経路でも verify_chain が成功すること");
    let _ = std::fs::remove_file(&path);
}

/// 受け入れ条件: 改竄（生のバイト列レベルでの書き換え）を `verify_chain`
/// が `LogError::ChainViolation` として検知する。ログの中身は公開 API から
/// 見えないため、`hash` フィールドの値そのものを別の妥当なハッシュ値
/// （genesis 相当の 64 桁 16 進文字列）へ書き換える、フォーマットを壊さない
/// 改竄を模擬する。
#[test]
fn tampering_the_hash_field_is_detected_via_public_api() {
    let path = unique_log_path("tamper_hash_field");
    let report = sample_report(
        LoopOutcome::Adopted,
        vec![AttemptRecord {
            attempt: 1,
            duration: Duration::from_millis(5),
            outcome: AttemptOutcome::Adopted,
        }],
    );
    LogWriter::open(&path)
        .expect("開けること")
        .append_report(&report)
        .expect("追記に失敗しないこと");

    let content = std::fs::read_to_string(&path).expect("読めること");
    let last_line_start = content.trim_end().rfind('\n').map(|i| i + 1).unwrap_or(0);
    // 最終行の `"hash":"..."` を全く別の妥当な形の 64 桁 16 進文字列へ
    // 書き換える（フィールド自体は残すため JSON の整形式は壊れない。
    // ハッシュ再計算との不一致のみで検知されることを確認する）。
    let bogus_hash = "0".repeat(64);
    let (head, last_line) = content.split_at(last_line_start);
    let tampered_last_line = {
        let hash_key = "\"hash\":\"";
        let start = last_line
            .find(hash_key)
            .expect("hash フィールドが存在すること")
            + hash_key.len();
        let end = last_line[start..]
            .find('"')
            .expect("hash 値の終端があること")
            + start;
        format!("{}{}{}", &last_line[..start], bogus_hash, &last_line[end..])
    };
    assert_ne!(
        last_line, tampered_last_line,
        "hash フィールドが実際に書き換わっていること"
    );
    std::fs::write(&path, format!("{head}{tampered_last_line}")).expect("書き戻せること");

    let result = self_repair::verify_chain(&path);
    assert!(
        matches!(result, Err(LogError::ChainViolation { .. })),
        "hash フィールドの改竄が ChainViolation として検知されること（fail-closed）"
    );
    let _ = std::fs::remove_file(&path);
}

/// 受け入れ条件: `LogWriter::open` を 2 回に分けて呼んでも（追記継続）
/// チェーンが繋がったままであること。
#[test]
fn reopening_across_process_boundary_style_calls_keeps_chain_connected() {
    let path = unique_log_path("reopen_keeps_chain");

    LogWriter::open(&path)
        .expect("1 回目のオープン")
        .append_report(&sample_report(
            LoopOutcome::Adopted,
            vec![AttemptRecord {
                attempt: 1,
                duration: Duration::from_millis(1),
                outcome: AttemptOutcome::Adopted,
            }],
        ))
        .expect("1 回目の追記");

    LogWriter::open(&path)
        .expect("2 回目のオープン（既存ファイルへの追記継続）")
        .append_report(&sample_report(LoopOutcome::NoActionNeeded, Vec::new()))
        .expect("2 回目の追記");

    self_repair::verify_chain(&path).expect("複数回に分けた追記全体で verify_chain が成功すること");
    let _ = std::fs::remove_file(&path);
}

/// 存在しないログファイルに対する `verify_chain` は `LogError::Io` を返す
/// （ファイル未作成を「空ログ = 検証成功」と誤認しない。fail-closed）。
#[test]
fn verify_chain_on_missing_file_returns_io_error() {
    let path = unique_log_path("missing_file");
    let result = self_repair::verify_chain(&path);
    assert!(
        matches!(result, Err(LogError::Io { .. })),
        "存在しないファイルは LogError::Io として報告されること"
    );
}
