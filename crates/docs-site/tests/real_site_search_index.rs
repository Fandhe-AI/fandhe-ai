//! イシュー #871 受け入れ基準 1 の本命テスト:
//! リポジトリルート実体 `site/nav.toml`（#873〜#875 で作成済みの本番原稿ツリー）
//! を実ビルドし、`search-index.json` が nav.toml 宣言の**全ページ**を含むことを
//! 検証する。`tests/site_nav.rs` は `tests/fixtures/valid`（3 ページの固定
//! フィクスチャ）を対象にする一方、本ファイルは本番原稿ツリー自体の全ページ数
//! （`nav::parse_nav` が返す実際のページ数）と索引内の `"href":` 出現数が一致する
//! ことまで確認する（フィクスチャ側のカバレッジと相補的）。
//!
//! 実機（CUDA/Metal）非依存・外部ネットワーク非依存のため `#[ignore]` は不要。

use std::path::{Path, PathBuf};

use docs_site::build::build_site;
use docs_site::layout::asset_href;
use docs_site::nav::parse_nav;

/// `crates/docs-site` から見たリポジトリルート（`CARGO_MANIFEST_DIR` の 2 つ
/// 上。`crates/docs-site` -> `crates` -> リポジトリルート）。
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root should be resolvable from CARGO_MANIFEST_DIR")
}

/// テスト専用の一時出力ディレクトリ。`Drop` でベストエフォート削除する。
/// `tests/site_nav.rs`・`tests/cli_fail_closed.rs` と同じパターン
/// （`std::env::temp_dir()` を先に `canonicalize` してから一意サフィックスを
/// 付ける。macOS の `/var` symlink による `open_out_root_dir` の偽陽性拒否を
/// 回避する。同コメント参照）。外部クレート（`tempfile` 等）は追加しない。
struct TempOutDir(PathBuf);

impl TempOutDir {
    fn new(tag: &str) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let base = std::fs::canonicalize(std::env::temp_dir())
            .expect("canonicalize std::env::temp_dir() for real_site_search_index test");
        Self(base.join(format!(
            "rust-ai-library-docs-site-real-site-search-index-test-{tag}-{}-{unique}",
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
fn real_site_nav_toml_builds_and_search_index_covers_every_declared_page() {
    let root = repo_root();
    let nav_toml_path = root.join("site/nav.toml");
    assert!(
        nav_toml_path.is_file(),
        "expected repository root site/nav.toml at {}",
        nav_toml_path.display()
    );

    // `build_site` 自体は fail-closed で `nav.toml` を再パースするが、期待値
    // （全ページの href・title 一覧）を独立に導出するため、ここでも
    // `parse_nav` を直接呼ぶ（`build_site` の内部実装詳細に依存しない
    // ブラックボックス検証）。
    let nav_toml_input =
        std::fs::read_to_string(&nav_toml_path).expect("read repository root site/nav.toml");
    let nav = parse_nav(&nav_toml_input).expect("repository root site/nav.toml should parse");

    let expected_pages: Vec<(String, String)> = nav
        .sections
        .iter()
        .flat_map(|section| section.pages.iter())
        .map(|page| {
            (
                asset_href(&nav.site.base_path, &page.path),
                page.title.clone(),
            )
        })
        .collect();
    assert!(
        !expected_pages.is_empty(),
        "repository root site/nav.toml should declare at least one page"
    );

    let out = TempOutDir::new("real-site");
    let report = build_site(&root, &out.0).expect("repository root site/ should build");
    assert_eq!(
        report.pages,
        expected_pages.len(),
        "build_site page count should match nav.toml declared page count"
    );

    let index_json = std::fs::read_to_string(out.0.join("assets/search-index.json"))
        .expect("assets/search-index.json should be written");

    // 受け入れ基準 1 の核心: `nav::parse_nav` が返す全ページ数と索引内の
    // `"href":` 出現数が一致すること（1 ページ 1 エントリが漏れなく収載されて
    // いることの数的な保証。個々のページの href・title が実際に含まれること
    // も併せて確認する）。
    let href_occurrences = index_json.matches("\"href\":").count();
    assert_eq!(
        href_occurrences,
        expected_pages.len(),
        "search-index.json must contain exactly one \"href\" entry per declared page"
    );

    for (href, title) in &expected_pages {
        let href_needle = format!("\"href\":\"{href}\"");
        assert!(
            index_json.contains(&href_needle),
            "search-index.json should contain page href {href}"
        );
        let title_needle = format!("\"title\":\"{}\"", escape_for_contains_check(title));
        assert!(
            index_json.contains(&title_needle),
            "search-index.json should contain page title {title}"
        );
    }
}

/// 索引 JSON 内の `title` 値は `search::escape_json_string` でエスケープ
/// 済み（`<` `>` `&` 等）だが、本テストが対象にする `site/nav.toml` の実際の
/// ページタイトルはいずれもこれらの文字を含まない日本語・英数字の見出し文言
/// のため、素の文字列一致で十分照合できる。将来タイトルにエスケープ対象文字が
/// 混入した場合にこの前提が崩れたことを検知できるよう、ここで明示的に契約化
/// する（本関数は現状恒等関数だが、意図を残すため独立させる）。
fn escape_for_contains_check(title: &str) -> String {
    assert!(
        !title
            .chars()
            .any(|c| matches!(c, '"' | '\\' | '<' | '>' | '&')),
        "this test's plain string matching assumes page titles contain no JSON/HTML escape target characters; got {title:?}"
    );
    title.to_string()
}
