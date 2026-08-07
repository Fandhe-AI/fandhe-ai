//! `guardrail` バイナリの起動・終了コード統合テスト（TASK-4.1a・TASK-4.1c）。
//!
//! `env!("CARGO_BIN_EXE_guardrail")` でビルド済みバイナリを直接起動し、
//! `docs/guardrail-self-repair-cli.md` 2.3 節の終了コード契約・1.2 節の
//! `--signals` 入口ガードが CLI プロセス境界で機能することを検証する。
//! 単体テスト（`src/*.rs` 内 `#[cfg(test)]`）はロジック単位の検証を担い、
//! 本ファイルはプロセス起動・環境変数・ファイル I/O を含む結合経路を担う。
//!
//! TASK-4.1c（イシュー #106）で `run_check_inner` が `decide()` へ結線された
//! ことに伴い、`check_without_signals_escalates_with_exit_code_10`・
//! `check_with_signals_and_env_guard_escalates_and_writes_report`・
//! `check_with_valid_config_override_still_escalates` の 3 テストは実際の
//! 判定結果（fail-closed の内部エラー／auto_apply）へ更新した
//! （代表的な 3 分岐到達の固定は `tests/cli_three_branch.rs` が担う）。

use std::process::Command;

fn guardrail_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_guardrail"))
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn check_without_signals_and_missing_baseline_is_internal_error_exit_code_1() {
    // 引数なし実行は measured 経路（既定 `--baseline "baseline"`）に入るが、
    // このテストプロセスの cwd（`crates/guardrail`）には `baseline` という
    // ref は存在しないため、`changeset::resolve_baseline_commit` が
    // fail-closed で拒否する（0/10/20 のいずれへも丸めない。
    // `.claude/rules/security.md` A08）。
    //
    // 「`baseline` という ref が存在しない」という前提はリポジトリの状態に
    // 依存し将来変わりうるため（誰かが `baseline` ブランチ／タグを作った
    // 場合、本テストは意味が変わり `cargo build` 等を実起動してしまう）、
    // `check_with_explicit_nonexistent_baseline_is_internal_error_exit_code_1`
    // で明示的に存在しない ref を指定する頑健な版も別途固定する。
    let output = guardrail_bin()
        .args(["check"])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn check_with_explicit_nonexistent_baseline_is_internal_error_exit_code_1() {
    let output = guardrail_bin()
        .args([
            "check",
            "--baseline",
            "this-ref-definitely-does-not-exist-106",
        ])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn check_with_signals_without_env_guard_is_rejected_with_exit_code_2() {
    let signals_path = fixtures_dir().join("signals-ok.json");
    let output = guardrail_bin()
        .args(["check", "--signals", signals_path.to_str().unwrap()])
        .env_remove("GUARDRAIL_ALLOW_INJECTED_SIGNALS")
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn check_with_signals_and_env_guard_yields_auto_apply_and_writes_report() {
    // `signals-ok.json`（全ゲート pass・42 行・bench 中央値 1.05%）は既定
    // プリセットの閾値内であり除外リスト評価も行わない（injected 経路）ため
    // `decide()` は `auto_apply` を返す（TASK-4.1c・イシュー #106 で結線）。
    let signals_path = fixtures_dir().join("signals-ok.json");
    let out_dir = std::env::temp_dir().join(format!(
        "guardrail-cli-skeleton-test-{}-report-ok",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).unwrap();
    let out_path = out_dir.join("report.json");

    let output = guardrail_bin()
        .args([
            "check",
            "--signals",
            signals_path.to_str().unwrap(),
            "--format",
            "json",
            "--output",
            out_path.to_str().unwrap(),
        ])
        .env("GUARDRAIL_ALLOW_INJECTED_SIGNALS", "1")
        .output()
        .expect("failed to run guardrail binary");

    assert_eq!(output.status.code(), Some(0));
    let report_json = std::fs::read_to_string(&out_path).expect("report file should be written");
    let value: serde_json::Value = serde_json::from_str(&report_json).unwrap();
    assert_eq!(value["signal_source"], "injected");
    assert_eq!(value["verdict"], "auto_apply");
    assert_eq!(value["schema_version"], "1");

    std::fs::remove_dir_all(&out_dir).ok();
}

#[test]
fn check_with_missing_required_signal_field_is_internal_error_exit_code_1() {
    let signals_path = fixtures_dir().join("signals-missing-field.json");
    let output = guardrail_bin()
        .args(["check", "--signals", signals_path.to_str().unwrap()])
        .env("GUARDRAIL_ALLOW_INJECTED_SIGNALS", "1")
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn check_with_unknown_argument_is_usage_error_exit_code_2() {
    let output = guardrail_bin()
        .args(["check", "--not-a-real-flag", "x"])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn check_with_invalid_preset_is_usage_error_exit_code_2() {
    let output = guardrail_bin()
        .args(["check", "--preset", "bogus"])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn check_with_config_unknown_field_is_internal_error_exit_code_1() {
    let config_path = fixtures_dir().join("guardrail-unknown-field.toml");
    let output = guardrail_bin()
        .args(["check", "--config", config_path.to_str().unwrap()])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn check_with_valid_config_override_and_missing_baseline_is_internal_error() {
    // `--config` override は `guardrail.toml` の値域検証を通すのみで、measured
    // 経路の baseline 解決には影響しない。`check_without_signals_and_missing_baseline_is_internal_error_exit_code_1`
    // と同じ理由で fail-closed（終了コード 1）になる。
    let config_path = fixtures_dir().join("guardrail-ok.toml");
    let output = guardrail_bin()
        .args(["check", "--config", config_path.to_str().unwrap()])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn eval_with_unresolvable_default_dataset_path_is_internal_error_exit_code_1() {
    // `--dataset` 既定値（`cli::DEFAULT_DATASET_DIR`）はリポジトリルートからの
    // 相対パスであり、本テストバイナリの cwd（cargo test 実行時は
    // `crates/guardrail` パッケージルート）から解決すると存在しない。
    // `guardrail eval` の評価ロジック本体（`guardrail::eval::run`。
    // TASK-4.3a・イシュー #115）が実 dataset を一括評価すること自体は
    // `tests/eval_harness.rs`（`--dataset` を明示指定）で検証する。
    let output = guardrail_bin()
        .args(["eval"])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn missing_subcommand_is_usage_error_exit_code_2() {
    let output = guardrail_bin()
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(2));
}
