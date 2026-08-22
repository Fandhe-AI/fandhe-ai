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
        // `std::env::temp_dir()` は macOS では `/var/folders/...`
        // （`/var` 自体が `/private/var` への symlink）を返す。`build_site`
        // の `open_out_root_dir` は `out` 配下の全コンポーネントを fd 相対
        // `O_NOFOLLOW` で辿り経路上のどの symlink も拒否する（P0 修正・
        // PR #899。`crates/docs-site/src/build.rs` の同関数ドキュメント
        // コメント参照）ため、symlink を含む一時ディレクトリを `out` へ
        // そのまま渡すとテストが偽陽性で失敗する。まだ存在しない出力先
        // 自体は `canonicalize` できないため、既に実在する
        // `std::env::temp_dir()` の方を先に symlink 無しの実パスへ解決し、
        // その上へ一意なサフィックス付きコンポーネントを追加する
        // （テストフィクスチャ自身の正規化であり、本番コードの symlink
        // 拒否ロジックを弱めるものではない）。
        let base = std::fs::canonicalize(std::env::temp_dir())
            .expect("canonicalize std::env::temp_dir() for site_nav test");
        Self(base.join(format!(
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

/// 受け入れ基準 2: 生成 HTML が nav.toml のセクション構造どおりのサイドバーを
/// 持つこと（イシュー #870）。
#[test]
fn valid_fixture_generates_sidebar_matching_nav_structure() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid-sidebar");
    build_site(&root, &out.0).expect("valid fixture should build");

    let html = std::fs::read_to_string(out.0.join("guide/intro/index.html"))
        .expect("intro page should be written");

    // セクション見出し・ページリンクが nav.toml の宣言順どおり現れる。
    let guide_pos = html.find("Guide").expect("Guide section heading present");
    let reference_pos = html
        .find("Reference")
        .expect("Reference section heading present");
    assert!(
        guide_pos < reference_pos,
        "sections must keep declaration order"
    );

    assert!(html.contains("href=\"/guide/intro/\""));
    assert!(html.contains("href=\"/guide/getting-started/\""));
    assert!(html.contains("href=\"/reference/api/\""));
    // 現在ページ（intro）に aria-current="page" が付与されている。
    assert!(html.contains("aria-current=\"page\">Introduction</a>"));
}

/// Markdown 由来の各種タグが本文（`<article>`）に反映されていることの確認
/// （受け入れ基準 1 の統合テスト側カバレッジ）。
#[test]
fn valid_fixture_renders_markdown_syntax_into_html() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid-markdown");
    build_site(&root, &out.0).expect("valid fixture should build");

    let intro_html = std::fs::read_to_string(out.0.join("guide/intro/index.html")).unwrap();
    assert!(intro_html.contains("<h1>Introduction</h1>"));
    assert!(intro_html.contains("<ul><li>first item</li>"));
    assert!(intro_html.contains("<blockquote>"));
    assert!(intro_html.contains("<table>"));

    let getting_started_html =
        std::fs::read_to_string(out.0.join("guide/getting-started/index.html")).unwrap();
    assert!(getting_started_html.contains("<pre><code class=\"language-rust\">"));
    assert!(getting_started_html.contains("<strong>bold</strong>"));
    assert!(getting_started_html.contains("<em>em</em>"));
    assert!(getting_started_html.contains("<strong><em>bold+em</em></strong>"));
    assert!(getting_started_html.contains("<a href=\"/guide/intro/\">link</a>"));
}

/// `assets/site.css` が生成されることの確認（テーマ CSS 書き出し。実装計画 §2.5）。
#[test]
fn valid_fixture_writes_theme_css_asset() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid-css");
    build_site(&root, &out.0).expect("valid fixture should build");

    let css = std::fs::read_to_string(out.0.join("assets/site.css"))
        .expect("assets/site.css should be written");
    assert!(css.contains(":root"));
    assert!(css.contains("prefers-color-scheme: dark"));
}

/// イシュー #871: `assets/site.js`（テーマ切替・全文検索 JS）が生成される
/// ことの確認。
#[test]
fn valid_fixture_writes_site_js_asset() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid-site-js");
    build_site(&root, &out.0).expect("valid fixture should build");

    let js = std::fs::read_to_string(out.0.join("assets/site.js"))
        .expect("assets/site.js should be written");
    assert!(js.contains("docs-theme-toggle"));
    assert!(js.contains("docs-search-input"));
}

/// イシュー #871 受け入れ基準 1: `search-index.json` が nav.toml 宣言の全ページ
/// （fixture 3 ページ）を含むこと。`serialize_index` はキー順固定の決定的
/// 直列化のため、文字列 contains での検証が可能（`search.rs` モジュール doc
/// 参照）。
#[test]
fn valid_fixture_search_index_contains_all_declared_pages() {
    let root = fixture_root("valid");
    let out = TempOutDir::new("valid-search-index");
    build_site(&root, &out.0).expect("valid fixture should build");

    let index_json = std::fs::read_to_string(out.0.join("assets/search-index.json"))
        .expect("assets/search-index.json should be written");
    assert!(index_json.contains(r#""href":"/guide/intro/""#));
    assert!(index_json.contains(r#""title":"Introduction""#));
    assert!(index_json.contains(r#""href":"/guide/getting-started/""#));
    assert!(index_json.contains(r#""title":"Getting Started""#));
    assert!(index_json.contains(r#""href":"/reference/api/""#));
    assert!(index_json.contains(r#""title":"API""#));
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
