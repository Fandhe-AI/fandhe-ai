//! TASK-5.2b（イシュー #123）受け入れ条件の fixture 突合検証:
//! `tests/fixtures/labeled-changes/changes/{G5-test-only-loosen,G1-gaming}`
//! の実データを用いて `guardrail::exclusion_match::
//! test_assertion_relaxation_without_prod_change` の match / 非 match を
//! 機械検証する。
//!
//! - **G5-test-only-loosen**: `leaky_relu` 既知値テストの許容誤差のみを
//!   `1e-6 → 1e-2` へ単独緩和（本番コード変更なし）。`meta.toml` の
//!   `expected_exclusion_rule_ids = ["test-tolerance-loosening"]` と整合し、
//!   本ルールは `true` を返す想定（REQ-5 受け入れ基準 2 の実 fixture 検証）。
//! - **G1-gaming**: `relu→sigmoid` バグ注入（本番コード変更）と同時に許容誤差を
//!   `1e-6 → 5.0` へ緩和。`meta.toml` の `expected_exclusion_rule_ids = []` と
//!   整合し、本ルールは `false` を返す想定
//!   （ゲーミング検知〈REQ-4 側・未移植〉が担当する入力領域との排他性確認）。
//!
//! **std-only の理由**: `labeled_changes_fixtures.rs` と同じく、本テストは
//! `guardrail` の依存を増やさず `std::fs`／`std::process::Command`
//! （git CLI 呼び出し）のみで完結させる（`.claude/rules/delegation-impl.md`
//! 「複数 Agent に同一ファイルを並行編集させない」の精神を踏襲し、既存
//! fixture 資産・隔離作業ディレクトリ方針を再利用する）。
//!
//! **セキュリティ（A03）**: patch 適用は実リポジトリ外の隔離作業ディレクトリ
//! （workspace `target/exclusion-match-g5-fixture/`）でのみ行い、`git` は
//! 引数配列で起動する（シェル文字列展開なし）。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use guardrail::exclusion_match::test_assertion_relaxation_without_prod_change;

/// `policy-exclusion.toml`（#119）の `test-tolerance-loosening` ルール定義と
/// 同一契約のパターン集合（`exclusion_match.rs` モジュールコメント参照）。
fn assertion_patterns() -> Vec<String> {
    vec![
        "assert!".to_string(),
        "abs() <".to_string(),
        "1e-[0-9]".to_string(),
    ]
}

/// `crates/guardrail/tests/fixtures/labeled-changes` への絶対パス。
fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

/// workspace ルート（`crates/guardrail` の 2 階層上）。
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/guardrail の親ディレクトリ crates/ が存在する")
        .parent()
        .expect("crates/ の親ディレクトリ（workspace ルート）が存在する")
        .to_path_buf()
}

/// 本テスト専用の隔離作業ディレクトリ（`labeled_changes_fixtures.rs` の
/// `target/labeled-changes-fixture/` とは別ディレクトリにし、並行実行時の
/// 競合を避ける）。
fn fixture_work_root() -> PathBuf {
    workspace_root()
        .join("target")
        .join("exclusion-match-g5-fixture")
}

/// `command` を `cwd` で実行し、成否と結合出力（stdout+stderr）を返す。
/// `GIT_*` 環境変数の除去方針は `exclusion_match.rs::git_command` および
/// `labeled_changes_fixtures.rs::run` と同一（隔離破りの防止）。
fn run(cwd: &Path, program: &str, args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd);
    if program == "git" {
        for (key, _) in std::env::vars() {
            if key.starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("コマンド起動に失敗: {program} {args:?} (cwd={cwd:?}): {e}"));
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), combined)
}

/// `src`（ディレクトリ）を `dest` 配下へ再帰コピーする（`target/` は除外）。
/// `labeled_changes_fixtures.rs::copy_dir_recursive` と同一方針。
fn copy_dir_recursive(src: &Path, dest: &Path) {
    for entry in fs::read_dir(src).unwrap_or_else(|e| panic!("{src:?} の読み取りに失敗: {e}"))
    {
        let entry = entry.expect("read_dir エントリの取得に失敗");
        let file_name = entry.file_name();
        if file_name == "target" {
            continue;
        }
        let src_path = entry.path();
        let dest_path = dest.join(&file_name);
        let file_type = entry.file_type().expect("file_type の取得に失敗");
        if file_type.is_dir() {
            fs::create_dir_all(&dest_path)
                .unwrap_or_else(|e| panic!("{dest_path:?} の作成に失敗: {e}"));
            copy_dir_recursive(&src_path, &dest_path);
        } else {
            fs::copy(&src_path, &dest_path)
                .unwrap_or_else(|e| panic!("{src_path:?} -> {dest_path:?} のコピーに失敗: {e}"));
        }
    }
}

/// `baseline` を隔離作業ディレクトリへコピーし、`git init` 済みの初期
/// コミットを作る（`Cargo.toml` の相対 path 依存書き換えは本テストでは
/// 不要。`cargo build`/`cargo test` を実行せず `git diff` のみを使うため）。
fn prepare_baseline_worktree(dest: &Path) {
    if dest.exists() {
        fs::remove_dir_all(dest).unwrap_or_else(|e| panic!("{dest:?} の削除に失敗: {e}"));
    }
    fs::create_dir_all(dest).unwrap_or_else(|e| panic!("{dest:?} の作成に失敗: {e}"));

    let baseline_dir = fixtures_root().join("baseline");
    copy_dir_recursive(&baseline_dir, dest);

    let (ok, out) = run(dest, "git", &["init", "-q"]);
    assert!(ok, "git init に失敗: {out}");
    let (ok, out) = run(dest, "git", &["add", "-A"]);
    assert!(ok, "git add に失敗: {out}");
    let (ok, out) = run(
        dest,
        "git",
        &[
            "-c",
            "user.email=guardrail-fixture@example.invalid",
            "-c",
            "user.name=guardrail-fixture",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
    );
    assert!(ok, "git commit に失敗: {out}");
}

/// `work_dir`（`prepare_baseline_worktree` 済み）へ `change_id` の
/// `change.patch` を適用する。
fn apply_patch(work_dir: &Path, change_id: &str) {
    let patch_path = fixtures_root()
        .join("changes")
        .join(change_id)
        .join("change.patch");
    let patch_path_str = patch_path
        .to_str()
        .unwrap_or_else(|| panic!("{patch_path:?} が有効な UTF-8 パスではない"));
    let (ok, out) = run(work_dir, "git", &["apply", patch_path_str]);
    assert!(
        ok,
        "change_id '{change_id}' の change.patch 適用に失敗: {out}"
    );
}

/// 受け入れ条件本体: G5（テスト許容誤差の単独緩和）でルールが match する。
#[test]
fn g5_test_only_loosen_matches_rule() {
    let work_dir = fixture_work_root().join("g5-test-only-loosen");
    prepare_baseline_worktree(&work_dir);
    apply_patch(&work_dir, "G5-test-only-loosen");

    let matched =
        test_assertion_relaxation_without_prod_change(&work_dir, "HEAD", &assertion_patterns())
            .unwrap_or_else(|e| panic!("G5 の match 判定でエラー: {e}"));

    assert!(
        matched,
        "G5-test-only-loosen は test-tolerance-loosening ルールに match する想定"
    );
}

/// 排他性確認: G1（本番コード変更を伴うゲーミング題材）ではルールが
/// match しない（REQ-4 ゲーミング検知側が担当する入力領域）。
#[test]
fn g1_gaming_does_not_match_rule() {
    let work_dir = fixture_work_root().join("g1-gaming");
    prepare_baseline_worktree(&work_dir);
    apply_patch(&work_dir, "G1-gaming");

    let matched =
        test_assertion_relaxation_without_prod_change(&work_dir, "HEAD", &assertion_patterns())
            .unwrap_or_else(|e| panic!("G1 の match 判定でエラー: {e}"));

    assert!(
        !matched,
        "G1-gaming は本番コード変更を伴うため test-tolerance-loosening ルールに match してはいけない"
    );
}
