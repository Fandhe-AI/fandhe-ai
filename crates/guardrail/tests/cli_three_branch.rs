//! `guardrail check` CLI 実行経路からの 3 分岐（auto_apply/0・escalate/10・
//! reject/20）到達を固定する統合テスト（TASK-4.1c・イシュー #106）。
//!
//! #106 は「3 分岐出力が CLI 実行経路から到達不能」（`run_check_inner` が
//! 常に `Verdict::Escalate` を返す固定実装のまま）という理由で reopen
//! された（#103 コメント参照）。本ファイルはその受け入れ条件
//! 「3 分岐が代表ケースで正しく出力される」を固定する。
//!
//! # injected 経路（`--signals`）: 常時実行
//! 決定的・高速（`decide()` を直接経由する経路の到達確認が主目的であり、
//! `git`/`cargo` の実プロセス起動を要さない）。
//!
//! # measured 経路: `#[ignore]`
//! 実際に一時 git リポジトリ上で `cargo build`/`test --release`/`clippy` を
//! 起動するため、1 ケースで数十秒〜要する（self-hosted runner・並列 worktree
//! 実行下でのコスト。`.claude/rules/coding-rust.md` の `#[ignore]` は本来
//! 実機依存テスト向けだが、本テストは実機非依存でもコストが高いため同じ
//! 分離機構を転用する。CI での有効化は `cargo test -- --ignored` 等の
//! 明示実行を想定し、通常の `cargo test` では実行しない）。auto_apply の
//! 疎通（measured 経路が `decide()` へ到達すること）のみを 1 ケースで
//! 確認し、escalate/reject の measured 分岐は `check.rs`／`gates.rs`／
//! `checks/*.rs` の単体テストが個別に固定する。

use std::process::Command;

fn guardrail_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_guardrail"))
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run_check_with_signals(fixture: &str) -> (Option<i32>, serde_json::Value) {
    let signals_path = fixtures_dir().join(fixture);
    let output = guardrail_bin()
        .args([
            "check",
            "--signals",
            signals_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .env("GUARDRAIL_ALLOW_INJECTED_SIGNALS", "1")
        .output()
        .expect("failed to run guardrail binary");
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout should be valid JSON");
    (output.status.code(), value)
}

#[test]
fn injected_all_clean_signals_yield_auto_apply_exit_0() {
    let (code, value) = run_check_with_signals("signals-ok.json");
    assert_eq!(code, Some(0));
    assert_eq!(value["verdict"], "auto_apply");
    assert_eq!(value["signal_source"], "injected");
    assert!(
        value["reason_conditions"]
            .as_array()
            .expect("reason_conditions should be an array")
            .is_empty(),
        "auto_apply の reason_conditions は空のはず"
    );
}

#[test]
fn injected_lines_exceeded_signals_yield_escalate_exit_10() {
    let (code, value) = run_check_with_signals("signals-escalate.json");
    assert_eq!(code, Some(10));
    assert_eq!(value["verdict"], "escalate");
    assert_eq!(value["signal_source"], "injected");
    let reason_conditions = value["reason_conditions"]
        .as_array()
        .expect("reason_conditions should be an array");
    assert!(
        reason_conditions.iter().any(|c| c == "lines_max_exceeded"),
        "reason_conditions に lines_max_exceeded が含まれるはず: {reason_conditions:?}"
    );
}

#[test]
fn injected_build_failed_signals_yield_reject_exit_20() {
    let (code, value) = run_check_with_signals("signals-reject.json");
    assert_eq!(code, Some(20));
    assert_eq!(value["verdict"], "reject");
    assert_eq!(value["signal_source"], "injected");
    let reason_conditions = value["reason_conditions"]
        .as_array()
        .expect("reason_conditions should be an array");
    assert!(
        reason_conditions.iter().any(|c| c == "gate_build_failed"),
        "reason_conditions に gate_build_failed が含まれるはず: {reason_conditions:?}"
    );
}

/// 迂回防止の回帰: `--signals` は環境変数ガードなしでは拒否される
/// （終了コード 2）。3 分岐（0/10/20）のいずれへも丸めない。
#[test]
fn injected_signals_without_env_guard_is_rejected_not_a_verdict_code() {
    let signals_path = fixtures_dir().join("signals-ok.json");
    let output = guardrail_bin()
        .args(["check", "--signals", signals_path.to_str().unwrap()])
        .env_remove("GUARDRAIL_ALLOW_INJECTED_SIGNALS")
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(2));
}

/// measured 経路: 存在しない baseline ref は判定不能として内部エラー
/// （終了コード 1）になる。3 分岐（0/10/20）のいずれへも丸めない
/// （fail-closed。`.claude/rules/security.md` A08）。
#[test]
fn measured_nonexistent_baseline_is_internal_error_not_a_verdict_code() {
    // カレントの git リポジトリ（本 worktree）を対象に、実在しない baseline
    // ref を指定する。measured 経路は `changeset::resolve_baseline_commit`
    // で早期に fail-closed 拒否するため、`cargo build` 等は起動されない
    // （高コストな実プロセス起動を避けつつ fail-closed 経路を固定できる）。
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."); // repository root
    let output = guardrail_bin()
        .args([
            "check",
            "--repo",
            repo_root.to_str().unwrap(),
            "--baseline",
            "this-ref-does-not-exist-106",
        ])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

/// measured 経路の疎通確認（auto_apply 到達の唯一の測定ケース）。
/// 一時 git リポジトリ上で実際に `cargo build`/`test --release`/`clippy` を
/// 起動するため高コスト（モジュール冒頭コメント参照）。通常 CI では
/// スキップし、`cargo test -- --ignored` 等の明示実行でのみ検証する。
#[test]
#[ignore = "cargo build/test --release/clippy を実起動するため高コスト（実機依存ではないが CI 既定では skip）"]
fn measured_clean_tiny_crate_yields_auto_apply_exit_0() {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-cli-three-branch-measured-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();

    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"tiny\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();

    let git = |args: &[&str]| {
        let status = Command::new("git")
            .args(args)
            .current_dir(&dir)
            .status()
            .expect("git 起動に失敗");
        assert!(status.success(), "git {args:?} が失敗");
    };
    git(&["init", "-q"]);
    git(&["add", "-A"]);
    git(&[
        "-c",
        "user.email=guardrail-test@example.invalid",
        "-c",
        "user.name=guardrail-test",
        "commit",
        "-q",
        "-m",
        "baseline",
    ]);

    // コメントのみの小変更（3 分岐の判定条件をいずれも逸脱しない）。
    std::fs::write(
        dir.join("src/lib.rs"),
        "// コメント追加\npub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();

    let target_dir = dir.join("target-isolated");
    let output = guardrail_bin()
        .args([
            "check",
            "--repo",
            dir.to_str().unwrap(),
            "--baseline",
            "HEAD",
            "--format",
            "json",
        ])
        .env("CARGO_TARGET_DIR", &target_dir)
        .output()
        .expect("failed to run guardrail binary");

    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout should be valid JSON");
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(value["verdict"], "auto_apply");
    assert_eq!(value["signal_source"], "measured");

    std::fs::remove_dir_all(&dir).ok();
}
