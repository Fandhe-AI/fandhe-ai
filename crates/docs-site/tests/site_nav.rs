//! `docs_site::nav` / `docs_site::build` のライブラリ API を `tests/fixtures/` の
//! 固定フィクスチャ経由で検証する統合テスト（実装計画 §4 手順 4）。
//!
//! バイナリ起動を伴う CLI 全体の E2E は `tests/cli_fail_closed.rs` が担う。
//! 本ファイルはライブラリ関数（[`docs_site::nav::parse_nav`] /
//! [`docs_site::build::build_site`]）を直接呼び、フィクスチャルートは
//! `CARGO_MANIFEST_DIR` からの相対パスで解決する（本イシューではリポジトリルートに
//! `site/nav.toml` を作らない方針のため、実体は本クレート配下のフィクスチャのみ）。

use std::path::{Path, PathBuf};

use docs_site::build::{BuildError, build_site};
use docs_site::nav::NavError;

fn fixture_root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// テスト専用の一時出力ディレクトリ。`Drop` でベストエフォート削除する。
/// 外部クレート（`tempfile` 等）を追加せず `std::env::temp_dir()` +
/// プロセス固有サフィックスで代用する（REQ-1 v2: 外部依存ゼロを維持する）。
struct TempOutDir(PathBuf);

impl TempOutDir {
    fn new(tag: &str) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self(std::env::temp_dir().join(format!(
            "rust-ai-library-docs-site-site-nav-test-{tag}-{}-{unique}",
            std::process::id()
        )))
    }
}

impl Drop for TempOutDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn valid_fixture_builds_and_reports_all_pages() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid");
    let report = build_site(&root, &out.0).expect("valid fixture should build");
    assert_eq!(report.pages, 3);
    assert!(out.0.is_dir());
}

#[test]
fn unknown_key_fixture_fails_closed_with_parse_error() {
    let root = fixture_root("unknown-key");
    let out = TempOutDir::new("unknown-key");
    match build_site(&root, &out.0) {
        Err(BuildError::Nav(NavError::Parse { .. })) => {}
        other => panic!("expected Nav(Parse), got {other:?}"),
    }
}

#[test]
fn missing_key_fixture_fails_closed_with_missing_key_error() {
    let root = fixture_root("missing-key");
    let out = TempOutDir::new("missing-key");
    match build_site(&root, &out.0) {
        Err(BuildError::Nav(NavError::MissingKey { context, key })) => {
            assert_eq!(context, "section.page");
            assert_eq!(key, "source");
        }
        other => panic!("expected Nav(MissingKey), got {other:?}"),
    }
}

#[test]
fn missing_source_fixture_fails_closed_with_missing_source_error() {
    let root = fixture_root("missing-source");
    let out = TempOutDir::new("missing-source");
    match build_site(&root, &out.0) {
        Err(BuildError::Nav(NavError::MissingSource(source))) => {
            assert_eq!(source, "site/does-not-exist.md");
        }
        other => panic!("expected Nav(MissingSource), got {other:?}"),
    }
}
