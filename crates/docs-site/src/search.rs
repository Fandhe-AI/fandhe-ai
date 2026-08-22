//! 依存ゼロの全文検索インデックス生成（イシュー #871）。
//!
//! # 役割・呼び出し文脈
//!
//! `crate::build::build_site` がページループの中で各ページの本文
//! （`markdown::markdown_to_nodes` の戻り値。`layout::docs_page` へ渡す前の
//! `Vec<Node>`）から [`extract_plain_text`] でプレーンテキストを抽出し、
//! [`SearchPage`] として蓄積する。全ページ収集後に [`serialize_index`] で
//! 決定的な JSON へ直列化し、[`validate_index_size`] でサイズ上限を検証して
//! から `<out>/`[`INDEX_REL_PATH`] へ書き出す（書き出し自体は `build.rs` が
//! 担う。本モジュールは sans-I/O な純関数のみを提供する）。
//!
//! `crate::layout::docs_page` は [`INDEX_REL_PATH`] を
//! [`crate::layout::asset_href`] 経由で検索入力欄の `data-search-index`
//! 属性へ埋め込み、`crate::script::SITE_JS` が実行時に本インデックスを
//! `fetch` して部分一致検索する（ビルド時生成 + 実行時 fetch という構成を
//! 取ることで、外部 JS ライブラリ・追加クレート依存を一切増やさない。
//! deps-policy.md の「外部依存ゼロ」方針を docs ビルドへ準用する）。
//!
//! 参照実装 `fandhe-backend` `crates/docs-site/src/search.rs`（イシュー #396
//! 相当）からの移植だが、本リポジトリの `markdown.rs` は見出しに `id` を
//! 生成せず TOC も存在しないため、参照実装が持つ `SearchSection`（見出し
//! アンカー索引）は含めない（実装計画 §4.1）。見出しテキスト自体は本文
//! プレーンテキスト抽出に含まれるため検索到達性は損なわれない。将来
//! セクション単位の索引が必要になった場合は [`INDEX_VERSION`] を上げて
//! スキーマを拡張する。
//!
//! # セキュリティ不変条件（`.claude/rules/security.md`）
//!
//! 索引 JSON は HTML へインライン埋め込みせず独立ファイルとして配信するため
//! `<script>` コンテキストへの混入経路は無いが、多層防御として
//! [`escape_json_string`] は JSON 必須エスケープ（`"` `\` 制御文字）に加えて
//! `<` `>` `&`（HTML 混入時の実害を無くす）と `U+2028`/`U+2029`
//! （JS 内テンプレートリテラル・行分割制御文字として悪用され得る）も
//! エスケープする。
//!
//! 二段のサイズ上限（[`MAX_PAGE_TEXT_BYTES`]・[`MAX_INDEX_BYTES`]）は
//! 無自覚な索引肥大化・DoS 化を防ぐ fail-closed 設計（`build.rs` が
//! [`validate_index_size`] の `Err` をビルド失敗として扱う）。

use crate::html::Node;

/// 索引スキーマのバージョン。将来スキーマを変更する際、JS 側の互換判定に使う。
pub(crate) const INDEX_VERSION: u32 = 1;

/// 1 ページあたりの本文プレーンテキスト上限（バイト）。超過分は
/// [`truncate_at_char_boundary`] で決定的に切り詰める（エラーにはしない。
/// ページ全文を索引に含めることは目的ではなく、冒頭からの部分一致・到達性を
/// 確保することが目的のため）。
pub(crate) const MAX_PAGE_TEXT_BYTES: usize = 4096;

/// 索引全体（直列化済み JSON）の上限（バイト）。超過はビルド失敗
/// （[`validate_index_size`] が `Err` を返し、`build.rs` が fail-closed で
/// `out` に一切書き出さない）。
pub(crate) const MAX_INDEX_BYTES: usize = 1024 * 1024;

/// 索引の出力先（`out` 起点の相対パス）。`crate::script::SCRIPT_REL_PATH` と
/// 同様に `build.rs` が書き出す固定の相対パス。
pub(crate) const INDEX_REL_PATH: &str = "assets/search-index.json";

/// 索引 1 ページ分のエントリ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SearchPage {
    /// ページの遷移先 href（`crate::layout::asset_href` 経由で構築済み。
    /// `base_path` を反映済みのため JS 側はそのまま `<a href>` へ使える）。
    pub(crate) href: String,
    /// ページタイトル。
    pub(crate) title: String,
    /// 本文プレーンテキスト（[`MAX_PAGE_TEXT_BYTES`] 以下に切り詰め済み）。
    pub(crate) text: String,
}

/// サイト全体の検索インデックス。[`serialize_index`] の入力。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SearchIndex {
    /// `site/nav.toml` の `[site] base_path`（JS 側が遷移先 URL の妥当性判定に
    /// 使う基準情報。実際の href は各ページで既に `base_path` を反映済み）。
    pub(crate) base_path: String,
    /// 宣言順を保持したページ列。
    pub(crate) pages: Vec<SearchPage>,
}

/// `Node` 木から本文プレーンテキストを抽出する。
///
/// 要素の子を処理した後にブロック境界の区切り（半角スペース 1 個）を挿入し、
/// 最後に連続空白を 1 個へ正規化する。区切りを入れないと隣接ブロックの
/// テキストが「導入本文です」のように癒着し、部分一致・可読性が劣化する。
pub(crate) fn extract_plain_text(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        extract_plain_text_into(node, &mut out);
    }
    normalize_whitespace(&out)
}

/// [`extract_plain_text`] の内部再帰実装。
fn extract_plain_text_into(node: &Node, out: &mut String) {
    match node {
        Node::Text(text) => out.push_str(text),
        Node::Element { children, .. } => {
            for child in children {
                extract_plain_text_into(child, out);
            }
            // ブロック境界の区切り。`normalize_whitespace` が連続空白を
            // 1 個へ畳み込むため、テキストを持たない要素の後でも安全に
            // 挿入できる。
            out.push(' ');
        }
    }
}

/// 連続する空白（改行・タブを含む Unicode 空白）を半角スペース 1 個へ畳み込み、
/// 先頭・末尾の空白を除去する。
fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = true; // 先頭の空白を捨てるため true から開始
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(c);
            last_was_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// UTF-8 文字境界を跨がずに `s` を `max_bytes` 以下へ決定的に切り詰める。
///
/// `String::truncate`・スライスへのバイト添字直指定は文字境界を跨ぐと panic
/// するため（日本語等のマルチバイト文字を含む本文で実際に起こり得る）、
/// `char_indices()` を走査して `max_bytes` 以下の最大境界を探す。
pub(crate) fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = 0;
    for (idx, ch) in s.char_indices() {
        let next = idx + ch.len_utf8();
        if next > max_bytes {
            break;
        }
        end = next;
    }
    &s[..end]
}

/// JSON 文字列リテラルへエスケープして `out` へ push する（呼び出し側が `"`
/// で囲む契約。本関数自体は囲み `"` を出力しない）。
///
/// JSON 必須（`"` `\` および `U+0000`〜`U+001F` 制御文字）に加え、多層防御
/// として `<` `>` `&` `U+2028`（LINE SEPARATOR）`U+2029`（PARAGRAPH
/// SEPARATOR）も `\uXXXX` 形式でエスケープする（モジュール doc 参照）。
pub(crate) fn escape_json_string(value: &str, out: &mut String) {
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' | '>' | '&' | '\u{2028}' | '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

/// [`SearchIndex`] をキー順固定の決定的 JSON へ直列化する。
///
/// 外部依存（`serde_json` 等）は追加しない（`Cargo.toml` の「外部依存ゼロ」
/// 方針コメント参照）。キー順は `version` → `base_path` → `pages`、ページは
/// `href` → `title` → `text` に固定する（同一入力に対して常に同一バイト列を
/// 返す決定性。2 回ビルドしてのバイト同一比較で検証可能）。
pub(crate) fn serialize_index(index: &SearchIndex) -> String {
    let mut out = String::new();
    out.push('{');
    out.push_str("\"version\":");
    out.push_str(&INDEX_VERSION.to_string());
    out.push_str(",\"base_path\":\"");
    escape_json_string(&index.base_path, &mut out);
    out.push_str("\",\"pages\":[");
    for (i, page) in index.pages.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str("\"href\":\"");
        escape_json_string(&page.href, &mut out);
        out.push_str("\",\"title\":\"");
        escape_json_string(&page.title, &mut out);
        out.push_str("\",\"text\":\"");
        escape_json_string(&page.text, &mut out);
        out.push_str("\"}");
    }
    out.push_str("]}");
    out
}

/// 直列化済み索引 JSON の失敗理由（上限超過）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexTooLarge {
    /// 直列化済み JSON の実バイト数。
    pub(crate) bytes: usize,
    /// 許容上限（バイト）。
    pub(crate) max: usize,
}

/// 直列化済み JSON `json` のバイト長が `max_bytes` 以下かを検証する
/// （fail-closed。上限を引数で受け取る純関数にすることで、実サイトでは
/// 到達しない上限超過経路を小さい上限を注入したテストで直接検証できる）。
pub(crate) fn validate_index_size(json: &str, max_bytes: usize) -> Result<(), IndexTooLarge> {
    let bytes = json.len();
    if bytes > max_bytes {
        Err(IndexTooLarge {
            bytes,
            max: max_bytes,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_at_char_boundary_cuts_japanese_text_at_a_char_boundary() {
        let s = "こんにちは"; // 3 バイト/文字 × 5 文字 = 15 バイト
        let truncated = truncate_at_char_boundary(s, 7);
        assert!(s.is_char_boundary(truncated.len()));
        assert_eq!(truncated, "こん");
    }

    #[test]
    fn truncate_at_char_boundary_keeps_string_unchanged_when_within_limit() {
        assert_eq!(truncate_at_char_boundary("hello", 100), "hello");
        assert_eq!(truncate_at_char_boundary("hello", 5), "hello");
    }

    #[test]
    fn escape_json_string_escapes_required_and_defense_in_depth_characters() {
        let mut out = String::new();
        escape_json_string("a\"b\\c<d>e&f\u{2028}g\u{2029}h", &mut out);
        assert_eq!(out, "a\\\"b\\\\c\\u003cd\\u003ee\\u0026f\\u2028g\\u2029h");
    }

    #[test]
    fn escape_json_string_escapes_control_characters() {
        let mut out = String::new();
        escape_json_string("a\nb\tc\rd\u{0}e", &mut out);
        assert_eq!(out, "a\\nb\\tc\\rd\\u0000e");
    }

    #[test]
    fn serialize_index_is_deterministic_with_fixed_key_order() {
        let index = SearchIndex {
            base_path: "/rust-ai-library".to_string(),
            pages: vec![SearchPage {
                href: "/rust-ai-library/guides/".to_string(),
                title: "Guides".to_string(),
                text: "hello world".to_string(),
            }],
        };
        let first = serialize_index(&index);
        let second = serialize_index(&index);
        assert_eq!(first, second);
        assert!(
            first.starts_with(r#"{"version":1,"base_path":"/rust-ai-library","pages":[{"href":"#)
        );
        assert!(first.contains(r#""text":"hello world"}"#));
    }

    #[test]
    fn validate_index_size_rejects_json_exceeding_a_small_injected_limit() {
        let json = "0123456789";
        assert!(validate_index_size(json, 5).is_err());
        let err = validate_index_size(json, 5).unwrap_err();
        assert_eq!(err.bytes, 10);
        assert_eq!(err.max, 5);
    }

    #[test]
    fn validate_index_size_accepts_json_within_a_sufficient_limit() {
        assert!(validate_index_size("0123456789", 1024).is_ok());
    }

    #[test]
    fn extract_plain_text_normalizes_whitespace_across_blocks() {
        let nodes = vec![
            Node::element("p", vec![], vec![Node::text("  Hello   world  ")]),
            Node::element("p", vec![], vec![Node::text("Second\nparagraph")]),
        ];
        assert_eq!(extract_plain_text(&nodes), "Hello world Second paragraph");
    }

    #[test]
    fn extract_plain_text_truncates_long_page_body_to_the_configured_limit() {
        let long = "あ".repeat(2000); // 3 バイト × 2000 文字 = 6000 バイト（> 4096）
        let nodes = vec![Node::element("p", vec![], vec![Node::text(long)])];
        let extracted = extract_plain_text(&nodes);
        let truncated = truncate_at_char_boundary(&extracted, MAX_PAGE_TEXT_BYTES);
        assert!(truncated.len() <= MAX_PAGE_TEXT_BYTES);
        assert!(extracted.len() > MAX_PAGE_TEXT_BYTES);
    }
}
