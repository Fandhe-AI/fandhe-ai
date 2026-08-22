//! `docs-site` バイナリの E2E テスト（実装計画 §4 手順 4）。
//!
//! `env!("CARGO_BIN_EXE_docs-site")` でビルド済みバイナリを起動し、
//! 「valid フィクスチャ → exit 0・出力ディレクトリ生成」「不正フィクスチャ 3 種・
//! `--out` 欠落 → 非 0 終了 + stderr に理由」を検証する。出力先は各テストが
//! 一意な一時ディレクトリを使うため、libtest の並列実行と衝突しない。

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// テスト専用の一時出力ディレクトリパスを払い出す（プロセス起動先として渡すのみ・
/// 事前作成は `docs-site` バイナリ側の責務）。プロセス固有サフィックスで
/// 並列テスト間の衝突を避ける（`tempfile` 等の外部クレートは追加しない）。
struct TempOutDir(PathBuf);

impl TempOutDir {
    fn new(tag: &str) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self(std::env::temp_dir().join(format!(
            "rust-ai-library-docs-site-cli-test-{tag}-{}-{unique}",
            std::process::id()
        )))
    }
}

impl Drop for TempOutDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
        .arg(&out.0)
        .output()
        .expect("docs-site binary should launch");

    assert!(
        output.status.success(),
        "expected exit 0, got {:?}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.0.is_dir());
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
        .arg(&out.0)
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
        .arg(&out.0)
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("build failed"));
}

#[test]
fn missing_source_fixture_exits_nonzero() {
    let root = fixture_root("missing-source");
    let out = TempOutDir::new("missing-source");

    let output = bin()
        .arg("--root")
        .arg(&root)
        .arg("--out")
        .arg(&out.0)
        .output()
        .expect("docs-site binary should launch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does-not-exist.md"));
}
