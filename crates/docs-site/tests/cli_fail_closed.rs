//! `docs-site` バイナリの E2E テスト（実装計画 §4 手順 4）。
//!
//! `env!("CARGO_BIN_EXE_docs-site")` でビルド済みバイナリを起動し、
//! 「valid フィクスチャ → exit 0・出力ディレクトリ生成」「不正フィクスチャ 3 種・
//! `--out` 欠落 → 非 0 終了 + stderr に理由」を検証する。出力先は各テストが
//! 一意な一時ディレクトリを使うため、libtest の並列実行と衝突しない。

mod common;

use common::TempOutDir;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_docs-site"))
}

#[test]
fn valid_fixture_exits_zero_and_creates_output_dir() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid");

    let output = bin()
        .arg("--root")
        .arg(&root)
        .arg("--out")
        .arg(out.out())
        .output()
        .expect("docs-site binary should launch");

    assert!(
        output.status.success(),
        "expected exit 0, got {:?}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.out().is_dir());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("validated 3 page"));
}

#[test]
fn missing_out_flag_exits_nonzero_with_usage_on_stderr() {
    let root = fixture_root("valid");
    let output = bin()
        .arg("--root")
        .arg(&root)
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--out"));
}

#[test]
fn unknown_key_fixture_exits_nonzero() {
    let root = fixture_root("unknown-key");
    let out = TempOutDir::new("unknown-key");

    let output = bin()
        .arg("--root")
        .arg(&root)
        .arg("--out")
        .arg(out.out())
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("build failed"));
}

#[test]
fn missing_key_fixture_exits_nonzero() {
    let root = fixture_root("missing-key");
    let out = TempOutDir::new("missing-key");

    let output = bin()
        .arg("--root")
        .arg(&root)
        .arg("--out")
        .arg(out.out())
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("build failed"));
}

/// イシュー #872 受け入れ基準: リンク切れ fixture（実在しないページへの
/// ルート相対リンクを含む）はビルド内蔵 linkcheck により非 0 終了し、
/// `out` ディレクトリが一切作成されない（fail-closed。`--out` を渡していても
/// `open_out_root_dir` の呼び出し自体に到達しないため。`build.rs`・
/// `linkcheck.rs` のモジュールコメント参照）。
#[test]
fn broken_link_fixture_exits_nonzero_and_creates_no_output_directory() {
    let root = fixture_root("broken-link");
    let out = TempOutDir::new("broken-link");

    let output = bin()
        .arg("--root")
        .arg(&root)
        .arg("--out")
        .arg(out.out())
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    assert!(
        !out.out().exists(),
        "out directory must not be created when linkcheck fails"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("build failed"));
    assert!(stderr.contains("does-not-exist"));
}

#[test]
fn missing_source_fixture_exits_nonzero() {
    let root = fixture_root("missing-source");
    let out = TempOutDir::new("missing-source");

    let output = bin()
        .arg("--root")
        .arg(&root)
        .arg("--out")
        .arg(out.out())
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does-not-exist.md"));
}
