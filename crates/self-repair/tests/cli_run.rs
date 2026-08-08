//! `self-repair run` CLI 引数の usage 契約テスト（イシュー #142 差し戻し分・
//! 実装計画 §3.3）。
//!
//! `crates/self-repair/tests/cli_verify_log.rs` と同じパターンで実バイナリを
//! 起動する（`env!("CARGO_BIN_EXE_self-repair")`）。本ファイルは usage エラー
//! （`--kind` 欠落・不正値・`--log` 欠落・`--max-attempts 0`・未知引数）の
//! 終了コード契約（`docs/guardrail-self-repair-cli.md` 3.5 節: usage エラー=2）
//! と、非 UTF-8 引数で panic（exit 101）しないことを固定する軽量テストであり、
//! 実際にループを 1 回完走させる重い実証（sandbox 構築・release ビルド・
//! ベンチ実測）は行わない（それは
//! `tests/feature_addition_loop_completion_task_3_3c.rs` の責務）。実機依存は
//! ないため通常 CI で実行する（`#[ignore]` 分離は不要）。

use std::process::Command;

fn self_repair_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_self-repair"))
}

/// `--kind` 欠落は usage エラー（exit 2）になる（`--log`／`--candidates`／
/// `--bench-bin`／`--workload-source` も併せて欠落させ、最初に検出される
/// `--kind` 欠落を固定する）。
#[test]
fn run_without_kind_exits_with_usage_error() {
    let output = self_repair_bin()
        .args(["run"])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--kind"), "stderr: {stderr}");
}

/// `--kind` の不正値は usage エラー（exit 2）になる。
#[test]
fn run_with_invalid_kind_value_exits_with_usage_error() {
    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "not-a-real-kind",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
        ])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--kind"), "stderr: {stderr}");
}

/// `--log` 欠落は usage エラー（exit 2）になる。
#[test]
fn run_without_log_exits_with_usage_error() {
    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "feature-addition",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
        ])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--log"), "stderr: {stderr}");
}

/// `--max-attempts 0`（`NonZeroU32` 違反）は usage エラー（exit 2）になる。
#[test]
fn run_with_zero_max_attempts_exits_with_usage_error() {
    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
            "--max-attempts",
            "0",
        ])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--max-attempts"), "stderr: {stderr}");
}

/// 未知引数は usage エラー（exit 2）になる。
#[test]
fn run_with_unknown_argument_exits_with_usage_error() {
    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
            "--bogus",
            "x",
        ])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--bogus"), "stderr: {stderr}");
}

/// PR #356 codex-review P1 指摘対応と同じ観点: 非 UTF-8 引数（`--kind` の値）
/// が渡っても panic（exit 101）せず、usage エラー（exit 2）へ変換される
/// （`cli.rs` モジュール冒頭ドキュメント参照）。
#[cfg(unix)]
#[test]
fn run_with_non_utf8_kind_value_does_not_panic() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let non_utf8_kind = OsStr::from_bytes(b"kind-\xff\xfe");
    let output = self_repair_bin()
        .args([
            OsStr::new("run"),
            OsStr::new("--kind"),
            non_utf8_kind,
            OsStr::new("--log"),
            OsStr::new("trial.jsonl"),
            OsStr::new("--candidates"),
            OsStr::new("candidates.json"),
            OsStr::new("--bench-bin"),
            OsStr::new("bench_workload"),
            OsStr::new("--workload-source"),
            OsStr::new("src/bin/bench_workload.rs"),
        ])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
}

/// PR #361 codex-review P0 指摘対応の回帰防止（main.rs:272 相当。
/// `docs/guardrail-self-repair-cli.md` 3.7 節「候補実行の信頼境界」）:
/// `--allow-candidate-exec` を指定しない場合、他の必須引数（`--repo` を
/// 含む）を満たしていても usage エラー（exit 2）で拒否されることを実
/// バイナリ起動で確認する。`--repo` に実在しないパスを渡しているため、
/// `main.rs::run_run`（`resolve_baseline_commit`・sandbox 構築）まで
/// 到達していれば `git rev-parse` の失敗で exit 1 になるはずであり、
/// exit 2 が返ることは `cli::parse_run` の段階（sandbox 構築・候補コード
/// 実行のいずれにも到達する前）で拒否されていることの証拠になる。
#[test]
fn run_without_allow_candidate_exec_is_rejected_before_touching_repo() {
    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "feature-addition",
            "--repo",
            "/nonexistent/self-repair-cli-run-test-repo",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
        ])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--allow-candidate-exec"),
        "stderr: {stderr}"
    );
}

/// `run`／`verify-log` のいずれでもないサブコマンドは usage エラー（exit 2）
/// になる（`cli.rs::parse` のエラーメッセージが `run`／`verify-log` 両方を
/// 案内することの回帰確認）。
#[test]
fn unknown_subcommand_mentions_both_run_and_verify_log() {
    let output = self_repair_bin()
        .args(["bogus-subcommand"])
        .output()
        .expect("failed to run self-repair binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("run") && stderr.contains("verify-log"),
        "stderr: {stderr}"
    );
}
