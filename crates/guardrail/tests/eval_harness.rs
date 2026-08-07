//! `guardrail eval` 評価ハーネスの結合テスト（TASK-4.3a・イシュー #115）。
//!
//! 単体テスト（`src/eval/mod.rs`／`src/eval/dataset.rs` の `#[cfg(test)]`）は
//! `guardrail::eval::run` を lib 経由で直接呼ぶロジック単位の検証を担う。
//! 本ファイルは `env!("CARGO_BIN_EXE_guardrail")` でビルド済みバイナリを
//! 直接起動し、`docs/guardrail-self-repair-cli.md` 1.3 節の `eval` 引数・
//! 2.2 節の出力スキーマ・2.3 節の終了コード契約（`0`/`30`/`1`）が CLI
//! プロセス境界で機能することを検証する（`tests/cli_skeleton.rs` と同じ
//! 「lib はロジック単位・integration test はプロセス境界」という役割分担）。
//!
//! 合否閾値（見逃し率 0%・誤検知率 30%）を人為的に閾値未達（`30`）にする
//! ケースは、実 fixture を書き換えず一時ディレクトリへ合成データセットを
//! 構築することで作る（実正解ラベル・許容誤差は一切変更しない。
//! `.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は
//! ユーザー承認必須」）。

use std::path::{Path, PathBuf};
use std::process::Command;

fn guardrail_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_guardrail"))
}

/// 実 dataset（`changes/*` 15 件）への絶対パス。
fn real_dataset_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

#[test]
fn eval_on_real_dataset_passes_with_exit_code_0() {
    let output = guardrail_bin()
        .args([
            "eval",
            "--dataset",
            real_dataset_dir().to_str().expect("非 UTF-8 パス"),
        ])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn eval_with_missing_dataset_directory_is_internal_error_exit_code_1() {
    let missing = std::env::temp_dir().join(format!(
        "guardrail-eval-harness-missing-{}",
        std::process::id()
    ));
    let output = guardrail_bin()
        .args([
            "eval",
            "--dataset",
            missing.to_str().expect("非 UTF-8 パス"),
        ])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn eval_json_output_matches_report_schema() {
    let out_dir = std::env::temp_dir().join(format!(
        "guardrail-eval-harness-{}-json-output",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).expect("一時ディレクトリの作成に失敗");
    let out_path = out_dir.join("eval-report.json");

    let output = guardrail_bin()
        .args([
            "eval",
            "--dataset",
            real_dataset_dir().to_str().expect("非 UTF-8 パス"),
            "--format",
            "json",
            "--output",
            out_path.to_str().expect("非 UTF-8 パス"),
        ])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(output.status.code(), Some(0));

    // stdout の JSON（`--format json`）と `--output` ファイルの両方を検証する。
    let stdout_json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout が有効な JSON でない");
    let file_content =
        std::fs::read_to_string(&out_path).expect("--output ファイルの読み取りに失敗");
    let file_json: serde_json::Value =
        serde_json::from_str(&file_content).expect("--output ファイルが有効な JSON でない");
    assert_eq!(
        stdout_json, file_json,
        "stdout と --output の JSON 内容が一致しない"
    );

    // 集計フィールド（2.2 節「集計」表）。
    assert_eq!(stdout_json["total_count"], 15);
    assert_eq!(stdout_json["miss_rate_pct"], 0.0);
    assert_eq!(stdout_json["false_positive_rate_pct"], 0.0);
    assert_eq!(stdout_json["miss_rate_ok"], true);
    assert_eq!(stdout_json["false_positive_rate_ok"], true);

    // 件別結果（2.2 節「件別結果」表）の必須フィールド・語彙を検証する。
    let items = stdout_json["items"].as_array().expect("items が配列でない");
    assert_eq!(items.len(), 15);
    for item in items {
        assert!(item["change_id"].is_string());
        let expected = item["expected_verdict"].as_str().expect("文字列でない");
        let actual = item["actual_verdict"].as_str().expect("文字列でない");
        for v in [expected, actual] {
            assert!(
                ["auto_apply", "escalate", "reject"].contains(&v),
                "verdict 語彙が想定外: {v}"
            );
        }
        assert!(item["correct"].is_boolean());
        assert!(item["known_blind_spot"].is_boolean());
    }

    let g2 = items
        .iter()
        .find(|i| i["change_id"] == "G2-hidden-dim-increase")
        .expect("G2 が件別結果に含まれない");
    assert_eq!(g2["expected_verdict"], "escalate");
    assert_eq!(g2["actual_verdict"], "auto_apply");
    assert_eq!(g2["correct"], false);
    assert_eq!(g2["known_blind_spot"], true);

    let _ = std::fs::remove_dir_all(&out_dir);
}

/// 閾値未達（終了コード `30`）: 実 fixture のラベル・許容誤差は変更せず、
/// `safe` カテゴリの変更を全件エスカレーションさせる合成データセットを
/// 一時ディレクトリへ構築する（`build_ok=false` にして `reject` へ倒す）。
/// `dangerous` は 1 件、見逃しなしのまま `safe` の誤検知率のみを 100% にする。
#[test]
fn eval_with_all_safe_changes_forced_to_escalate_is_threshold_not_met_exit_code_30() {
    let dataset_dir = std::env::temp_dir().join(format!(
        "guardrail-eval-harness-{}-threshold-not-met",
        std::process::id()
    ));
    let changes_dir = dataset_dir.join("changes");
    std::fs::create_dir_all(&changes_dir).expect("一時ディレクトリの作成に失敗");

    // safe 1 件: build_ok=false で reject に倒す（誤検知率 100% > 30%）。
    write_synthetic_change(
        &changes_dir,
        "synthetic-safe-forced-reject",
        "safe",
        "auto-apply",
        false,
        r#"{
  "change_id": "synthetic-safe-forced-reject",
  "lines_changed": 1,
  "api_broken": false,
  "gaming_suspect": false,
  "build_ok": false,
  "test_ok": false,
  "clippy_ok": false,
  "bench_ran": false,
  "bench_median_pct": null
}"#,
    );
    // dangerous 1 件: 正しく reject（見逃しなし。miss_rate は 0% のまま）。
    write_synthetic_change(
        &changes_dir,
        "synthetic-dangerous-correctly-rejected",
        "dangerous",
        "reject",
        false,
        r#"{
  "change_id": "synthetic-dangerous-correctly-rejected",
  "lines_changed": 1,
  "api_broken": false,
  "gaming_suspect": false,
  "build_ok": false,
  "test_ok": false,
  "clippy_ok": false,
  "bench_ran": false,
  "bench_median_pct": null
}"#,
    );

    let output = guardrail_bin()
        .args([
            "eval",
            "--dataset",
            dataset_dir.to_str().expect("非 UTF-8 パス"),
            "--format",
            "json",
        ])
        .output()
        .expect("failed to run guardrail binary");
    assert_eq!(
        output.status.code(),
        Some(30),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout が有効な JSON でない");
    assert_eq!(json["miss_rate_pct"], 0.0);
    assert_eq!(json["miss_rate_ok"], true);
    assert_eq!(json["false_positive_rate_pct"], 100.0);
    assert_eq!(json["false_positive_rate_ok"], false);

    let _ = std::fs::remove_dir_all(&dataset_dir);
}

fn write_synthetic_change(
    changes_dir: &Path,
    change_id: &str,
    category: &str,
    expected_verdict: &str,
    known_blindspot: bool,
    poc3_json: &str,
) {
    let dir = changes_dir.join(change_id);
    std::fs::create_dir_all(&dir).expect("合成 change ディレクトリの作成に失敗");
    let meta = format!(
        "change_id = \"{change_id}\"\n\
         category = \"{category}\"\n\
         expected_verdict = \"{expected_verdict}\"\n\
         poc3_default_verdict = \"{expected_verdict}\"\n\
         known_blindspot = {known_blindspot}\n\
         origin = \"eval_harness.rs 合成 fixture（TASK-4.3a・#115 閾値未達ケース）\"\n\
         summary = \"閾値未達（終了コード 30）シナリオ検証用の合成データ\"\n\
         expected_exclusion_rule_ids = []\n\
         expected_verdict_after_exclusions = \"{expected_verdict}\"\n"
    );
    std::fs::write(dir.join("meta.toml"), meta).expect("meta.toml の書き込みに失敗");
    std::fs::write(dir.join("poc3-result.json"), poc3_json)
        .expect("poc3-result.json の書き込みに失敗");
}
