//! ビルド内蔵 linkcheck（fail-closed。イシュー #872）。
//!
//! # 役割・呼び出し文脈
//!
//! `crate::build::build_site` が全ページを `markdown::markdown_to_nodes` →
//! `layout::docs_page` でレンダリングし終えた直後（`out` への書き出し・
//! `open_out_root_dir` の呼び出しより前）に `check_links`（`pub(crate)`。
//! 本モジュールは `pub` だが本関数は非公開のため、private-item への
//! intra-doc link を避けコードスパンで参照する）を呼ぶ。戻り値が
//! 非空であれば `build_site` は `BuildError::BrokenLinks` で早期 return し、
//! `out` へ 1 バイトも書き出さない（ディレクトリの作成すら行わない）。これが
//! 本イシューの fail-closed 契約の核: リンク切れのある生成物が GitHub Pages
//! へ公開されるパイプラインへ流れる経路を構造的に塞ぐ。
//!
//! # 検証対象・分類
//!
//! 各ページの最終形 `html::Node`（`layout::docs_page` の戻り値。ヘッダの
//! セクションメニュー・サイドバー・本文・`<link>`/`<script>` を含む）から
//! `href`・`src` 属性値を `crate::html::find_attr_values`（`pub(crate)`。
//! private-item への intra-doc link を避けコードスパンで参照する）で再帰収集し、
//! 以下のとおり分類する。
//!
//! - スキームあり（`http:`/`https:`/`mailto:` 等）・protocol-relative
//!   （`//...`）→ 対象外（外部リンクのネットワーク到達性検証はスコープ外。
//!   参照実装 fandhe-backend `crates/docs-site/src/linkcheck.rs` と同判断）
//! - `href` はまず `split_fragment` で `path?query` 部分と `#fragment` 部分へ
//!   分け、`path?query` 側はさらに `strip_query` で `?` 以降を切り落として
//!   から解決する（レビュー指摘・PR #901・github-actions(bot) P2 /
//!   cursor(bot) Low: query を残したまま解決すると `/guide/?tab=cpu` のような
//!   UI 状態を表す query 付き既存ページへのリンクが既知ターゲット表〈query を
//!   含まない `page.path` ベース〉と一致せず誤って壊れたリンク判定される。
//!   query はサイト内リンク解決に無関係な値として無視する）
//! - `#fragment` のみ → 解決先ページ（`#fragment` のみなら現在ページ自身。
//!   `resolve_target` が空パスを現在ページへ解決するため自然に扱える）の
//!   `id` 属性値集合と突合する。本リポの `markdown.rs` は見出し `id` を
//!   生成しないため、fragment リンクは実在 `id` が無い限り fail-closed に
//!   落ちる（これは正しい挙動。実装計画 §2.2）。ただし **空 fragment**
//!   （`href` が `...#` で終わる形。同レビュー指摘）は HTML 仕様上「文書
//!   先頭」を意味し対応する `id` の実在を要求しないため、検証対象から除く
//! - ルート相対（`/...`）・相対パス → `resolve_segments`（非公開関数。
//!   private-item への intra-doc link を避けコードスパンで参照する。
//!   `.`/`..` 解決。
//!   ルートより上への `..` エスケープは即 broken）で解決し、既知ターゲット表
//!   と突合。fragment 付き（空 fragment を除く）は解決先ページの `id` 集合
//!   とも突合する
//! - 既知ターゲット表 = 全ページの `layout::asset_href(base_path, &page.path)`
//!   とアセット 3 種（呼び出し元 `build.rs` が `asset_hrefs` として渡す
//!   `assets/site.css`・`script::SCRIPT_REL_PATH`・`search::INDEX_REL_PATH`
//!   の `asset_href` 適用済み値）の和集合。末尾 `/` の有無は
//!   `normalize_target_key`（非公開関数。private-item への intra-doc link を
//!   避けコードスパンで参照する）で正規化して同一視する（ルート `/` は除く）
//!
//! ヘッダの `index_path` リンクは全ページに埋め込まれるため、`index_path` の
//! ページ実在突合（`nav::Section::index_path` ドキュメンテーションコメントが
//! 本イシューへ委ねた責務）はこの走査で自然に担保される。
//!
//! 壊れたリンクは 1 件目で打ち切らず全件収集して返す（是正効率のため。
//! 実装計画 §2.2）。
//!
//! # セキュリティ考慮（`.claude/rules/security.md` A03・多層防御）
//!
//! 本モジュールはリンク文字列の読み取り・比較のみを行い、ファイルシステム
//! アクセス・シェル呼び出しは一切行わない。`resolve_segments`（非公開関数。
//! private-item への intra-doc link を避けコードスパンで参照する）が `..` に
//! よるルートエスケープを拒否するのは、`nav::validate_sources` のパス
//! トラバーサル対策（`page.source` 側）を代替・迂回するものではなく、
//! 生成済み HTML 側のリンク検証という別レイヤーでの多層防御として位置づける。
//! `href` の生成・エスケープ自体（`html::render` の一元エスケープ・`Node` の
//! `pub(crate)` 契約）には一切変更を加えない。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::html::{self, Node};
use crate::layout;

/// 検出した壊れたリンク 1 件分。`Display` はページパス・href・短い理由のみを
/// 含み、絶対パス・環境変数・入力全文は含めない（`NavError`・`BuildError` と
/// 同じ機微情報露出防止契約。`.claude/rules/security.md`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokenLink {
    /// リンクが埋め込まれていたページの `page.path`（`base_path` を含まない
    /// 生の値。診断上どのページを直すべきかが分かるようにするため）。
    pub page_path: String,
    /// 壊れていた `href`／`src` の生の値（レンダリング済み HTML から採取した
    /// ままの文字列。`base_path` 反映済みの場合はそのまま）。
    pub href: String,
    /// 壊れている理由（人間可読な短い説明）。
    pub reason: String,
}

impl fmt::Display for BrokenLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: link `{}`: {}",
            self.page_path, self.href, self.reason
        )
    }
}

/// `href` がスキーム付き絶対 URL（`http:`／`https:`／`mailto:` 等）・
/// protocol-relative（`//...`）であるかを判定する。真ならリンク検証の対象外
/// （ネットワーク到達性検証はスコープ外。モジュール doc 参照）。
fn is_external_url(href: &str) -> bool {
    if href.starts_with("//") {
        return true;
    }
    if let Some(colon_idx) = href.find(':') {
        let scheme_candidate = &href[..colon_idx];
        // スキームはコロンより前に `/` を含まない（`RFC 3986` §3.1 の
        // `scheme` 生成規則: 先頭は英字必須、2 文字目以降は英数字・`+`・
        // `-`・`.` を許容）。相対パス中にコロンが現れる非現実的なケース
        // （`/a:b` 等）を誤って外部リンク扱いしないための条件。
        //
        // レビュー指摘（PR #901・github-actions(bot) P2）: 先頭文字にも
        // 英数字を許すと `123:missing` のような数字始まりの相対パスを
        // scheme 付き外部 URL と誤認し、検証対象から除外してしまう
        // （壊れたリンクが linkcheck をすり抜ける）。先頭文字は
        // `is_ascii_alphabetic()` で検証し、2 文字目以降のみ現行の文字
        // 集合（英数字・`+`・`-`・`.`）を適用する。
        let mut chars = scheme_candidate.chars();
        let starts_with_alpha = chars.next().is_some_and(|c| c.is_ascii_alphabetic());
        if starts_with_alpha
            && !scheme_candidate.contains('/')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        {
            return true;
        }
    }
    false
}

/// `s`（percent-encoding を含みうる文字列）を percent-decode する。
///
/// レビュー指摘（PR #901・github-actions(bot) P2）: fragment を
/// percent-decode せず `id` 属性値と直接比較すると、`#foo%20bar`
/// のようなブラウザ上は `foo bar` として解決可能なリンクを誤って
/// 壊れたリンクと判定してしまう（fail-closed でビルド全体が失敗しうる）。
/// `href`／`id` 双方とも percent-decode 後の値で突合するのが正しい。
///
/// 不正な percent-encoding（`%` の後ろに 2 桁の 16 進数が続かない、または
/// デコード結果が不正な UTF-8 バイト列になる）は `None` を返し、
/// 呼び出し元がその fragment を明示的に broken と判定する（あいまいな
/// 入力を許容側へ倒さない。`.claude/rules/security.md` A03 準拠）。
fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let hex_str = std::str::from_utf8(hex).ok()?;
            let value = u8::from_str_radix(hex_str, 16).ok()?;
            out.push(value);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `href` を `(path 部分, fragment 部分)` へ分割する（`#` の有無で判定）。
/// `path` 部分には query 文字列（`?...`）が含まれうる（`strip_query` で
/// 別途除去する。URL は `path?query#fragment` の順であるため、まず `#` で
/// fragment を切り離してから `?` を扱う）。
fn split_fragment(href: &str) -> (&str, Option<&str>) {
    match href.find('#') {
        Some(idx) => (&href[..idx], Some(&href[idx + 1..])),
        None => (href, None),
    }
}

/// `path_part`（`split_fragment` 適用済み。fragment を含まない）から query
/// 文字列（`?` 以降）を取り除く。
///
/// レビュー指摘（PR #901・github-actions(bot) P2 / cursor(bot) Low）:
/// query を除去せず解決すると `/guide/?tab=cpu` のような UI 状態を表す
/// query 付きリンクが既知ターゲット表（query を含まない `page.path` ベース）
/// と一致せず、実在するページへのリンクが誤って壊れたリンクと判定され
/// fail-closed でビルド全体が失敗しうる。query はサイト内リンク解決に
/// 無関係な値のため、path のみを既知ターゲット表と突合する。
fn strip_query(path_part: &str) -> &str {
    match path_part.find('?') {
        Some(idx) => &path_part[..idx],
        None => path_part,
    }
}

/// `path` の末尾 `/` の有無を正規化する（ルート `/` 単体は除く）。既知
/// ターゲット表への登録・解決済みリンクとの突合の両方で使う単一の正規化点
/// （実装計画 §2.2「末尾 `/` の有無は正規化して同一視」）。
fn normalize_target_key(path: &str) -> String {
    if path == "/" {
        path.to_string()
    } else {
        path.strip_suffix('/').unwrap_or(path).to_string()
    }
}

/// `base`（ディレクトリ扱いのセグメント列）を起点に `relative`（`/` 区切り。
/// `.`/`..` を含みうる）を解決する。`..` が `base` より上へエスケープする
/// 場合は `None`（呼び出し元が「ルートより上への `..` エスケープ」として
/// broken 判定する）。
fn resolve_segments(base: &[&str], relative: &str) -> Option<Vec<String>> {
    let mut segments: Vec<String> = base.iter().map(|segment| segment.to_string()).collect();
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            other => segments.push(other.to_string()),
        }
    }
    Some(segments)
}

/// `current_href`（現在ページの `asset_href` 適用済み href。常に `/` 始まり）
/// を起点に `path_part`（`href` の fragment を除いた部分）を解決し、`/` 始まりの
/// サイト絶対パスへ正規化する。
///
/// - `path_part` が `/` 始まりならルート相対として絶対解決する（`layout.rs`
///   の `rewrite_root_relative_hrefs` が本文中の `/` 始まりリンクへ既に
///   `base_path` を反映済みのため、ここで得られる絶対パスは常に `base_path`
///   を含む形になる。ヘッダ・サイドバーの nav・asset リンクも同様）
/// - それ以外（相対パス。実サイト原稿には存在しないが多層防御として対応する）
///   は `current_href` 自身のディレクトリ（page.path は常に `/` 終わりの
///   ため、ページ自身をディレクトリとみなす）を起点に解決する
/// - `path_part` が空（`href` が `#fragment` のみ、または空文字）の場合は
///   `current_href` 自身（同一ページ）へ解決する
fn resolve_target(current_href: &str, path_part: &str) -> Option<String> {
    let (base, relative): (Vec<&str>, &str) = if let Some(stripped) = path_part.strip_prefix('/') {
        (Vec::new(), stripped)
    } else {
        let current_dir: Vec<&str> = current_href
            .trim_matches('/')
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        (current_dir, path_part)
    };
    let resolved = resolve_segments(&base, relative)?;
    Some(format!("/{}", resolved.join("/")))
}

/// 全ページのリンク（`href`／`src`）を既知ターゲット表・同一/他ページの `id`
/// 集合と突合し、壊れたリンクを全件収集して返す。空なら「壊れたリンクなし」。
///
/// - `pages`: `(page.path, layout::docs_page` の戻り値）` の列（宣言順で
///   構わない。突合はページパス単位の集合演算のため順序に依存しない）
/// - `base_path`: `nav.toml` の `[site] base_path`（`layout::asset_href` と
///   同じ形式規則）
/// - `asset_hrefs`: 呼び出し元（`build.rs`）が `layout::asset_href` 経由で
///   構築済みの静的アセット href（`assets/site.css`・`assets/site.js`・
///   `assets/search-index.json`）
pub(crate) fn check_links(
    pages: &[(String, Node)],
    base_path: &str,
    asset_hrefs: &[String],
) -> Vec<BrokenLink> {
    // ページ href（既知ターゲット表の一部）→ そのページの `id` 属性値集合。
    // fragment 付きリンクの解決先突合（他ページの見出し等への `#id` リンク）
    // にも使う。
    let mut page_ids_by_key: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (page_path, node) in pages {
        let key = normalize_target_key(&layout::asset_href(base_path, page_path));
        let ids: BTreeSet<String> = html::find_attr_values(node, "id").into_iter().collect();
        page_ids_by_key.insert(key, ids);
    }

    let mut known_targets: BTreeSet<String> = page_ids_by_key.keys().cloned().collect();
    for asset_href in asset_hrefs {
        known_targets.insert(normalize_target_key(asset_href));
    }

    let mut broken = Vec::new();

    for (page_path, node) in pages {
        let current_href = layout::asset_href(base_path, page_path);

        let mut link_values = html::find_attr_values(node, "href");
        link_values.extend(html::find_attr_values(node, "src"));

        for raw in link_values {
            if raw.is_empty() {
                broken.push(BrokenLink {
                    page_path: page_path.clone(),
                    href: raw,
                    reason: "href/src attribute is empty".to_string(),
                });
                continue;
            }

            if is_external_url(&raw) {
                continue;
            }

            let (path_part, fragment_part) = split_fragment(&raw);
            let path_part = strip_query(path_part);

            let Some(resolved) = resolve_target(&current_href, path_part) else {
                broken.push(BrokenLink {
                    page_path: page_path.clone(),
                    href: raw.clone(),
                    reason: "path escapes above the site root via `..`".to_string(),
                });
                continue;
            };
            let key = normalize_target_key(&resolved);

            if !known_targets.contains(&key) {
                broken.push(BrokenLink {
                    page_path: page_path.clone(),
                    href: raw.clone(),
                    reason: format!("target `{key}` does not exist in the built site"),
                });
                continue;
            }

            // レビュー指摘（PR #901・cursor(bot) Low）: 空 fragment（`href` が
            // `...#` で終わる形。HTML 仕様上「文書先頭」を意味し、ページ側に
            // 対応する `id` 実在を要求すべきではない）は検証対象から除く。
            if let Some(fragment) = fragment_part.filter(|fragment| !fragment.is_empty()) {
                // レビュー指摘（PR #901・github-actions(bot) P2）: fragment は
                // percent-encoding されうる（`#foo%20bar` 等）ため、`id`
                // 属性値（percent-decode 前提の生の値）と比較する前に
                // percent-decode する。不正な percent-encoding は明示的に
                // broken と判定する（`percent_decode` doc 参照）。
                let Some(decoded_fragment) = percent_decode(fragment) else {
                    broken.push(BrokenLink {
                        page_path: page_path.clone(),
                        href: raw.clone(),
                        reason: format!("fragment `#{fragment}` is not validly percent-encoded"),
                    });
                    continue;
                };
                match page_ids_by_key.get(&key) {
                    Some(target_ids) if target_ids.contains(&decoded_fragment) => {}
                    Some(_) => broken.push(BrokenLink {
                        page_path: page_path.clone(),
                        href: raw.clone(),
                        reason: format!("fragment `#{fragment}` not found on target page"),
                    }),
                    None => broken.push(BrokenLink {
                        page_path: page_path.clone(),
                        href: raw.clone(),
                        reason: format!(
                            "target `{key}` is not a page; fragment `#{fragment}` cannot resolve"
                        ),
                    }),
                }
            }
        }
    }

    broken
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(path: &str, node: Node) -> (String, Node) {
        (path.to_string(), node)
    }

    fn link(href: &str) -> Node {
        Node::element(
            "a",
            vec![("href".to_string(), href.to_string())],
            vec![Node::text("link")],
        )
    }

    fn heading_with_id(id: &str) -> Node {
        Node::element(
            "h2",
            vec![("id".to_string(), id.to_string())],
            vec![Node::text("Heading")],
        )
    }

    #[test]
    fn valid_root_relative_and_asset_links_report_no_broken_links() {
        let pages = vec![
            page(
                "/guide/",
                Node::element(
                    "html",
                    vec![],
                    vec![
                        link("/guide/backends/"),
                        Node::element(
                            "link",
                            vec![("href".to_string(), "/assets/site.css".to_string())],
                            vec![],
                        ),
                    ],
                ),
            ),
            page("/guide/backends/", Node::element("html", vec![], vec![])),
        ];
        let asset_hrefs = vec!["/assets/site.css".to_string()];
        assert!(check_links(&pages, "", &asset_hrefs).is_empty());
    }

    #[test]
    fn valid_relative_link_within_same_directory_reports_no_broken_links() {
        let pages = vec![
            page(
                "/guide/",
                Node::element("html", vec![], vec![link("backends/")]),
            ),
            page("/guide/backends/", Node::element("html", vec![], vec![])),
        ];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn missing_page_target_is_reported_broken() {
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("/does-not-exist/")]),
        )];
        let broken = check_links(&pages, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("does not exist"));
        assert_eq!(broken[0].href, "/does-not-exist/");
    }

    #[test]
    fn parent_escape_above_site_root_is_reported_broken() {
        let pages = vec![page(
            "/",
            Node::element("html", vec![], vec![link("../escape/")]),
        )];
        let broken = check_links(&pages, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains(".."));
    }

    #[test]
    fn missing_fragment_on_same_page_is_reported_broken() {
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("#missing")]),
        )];
        let broken = check_links(&pages, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("fragment"));
    }

    #[test]
    fn existing_fragment_on_same_page_reports_no_broken_links() {
        let pages = vec![page(
            "/guide/",
            Node::element(
                "html",
                vec![],
                vec![link("#section"), heading_with_id("section")],
            ),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn fragment_on_another_existing_page_is_validated_against_that_pages_ids() {
        let pages = vec![
            page(
                "/guide/",
                Node::element("html", vec![], vec![link("/reference/#anchor")]),
            ),
            page(
                "/reference/",
                Node::element("html", vec![], vec![heading_with_id("anchor")]),
            ),
        ];
        assert!(check_links(&pages, "", &[]).is_empty());

        let pages_missing = vec![
            page(
                "/guide/",
                Node::element("html", vec![], vec![link("/reference/#missing")]),
            ),
            page(
                "/reference/",
                Node::element("html", vec![], vec![heading_with_id("anchor")]),
            ),
        ];
        let broken = check_links(&pages_missing, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("fragment"));
    }

    #[test]
    fn fragment_pointing_at_an_asset_target_is_reported_broken() {
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("/assets/site.css#missing")]),
        )];
        let broken = check_links(&pages, "", &["/assets/site.css".to_string()]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("is not a page"));
    }

    #[test]
    fn external_links_are_ignored() {
        let pages = vec![page(
            "/guide/",
            Node::element(
                "html",
                vec![],
                vec![
                    link("https://github.com/Fandhe-AI/rust-ai-library"),
                    link("mailto:someone@example.com"),
                    link("//cdn.example.com/asset.js"),
                ],
            ),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn empty_href_is_reported_broken() {
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("")]),
        )];
        let broken = check_links(&pages, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("empty"));
    }

    #[test]
    fn trailing_slash_presence_is_normalized_as_equivalent() {
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("/assets/site.css")]),
        )];
        // 既知ターゲット表側に末尾 `/` が付いていても同一視する。
        let asset_hrefs = vec!["/assets/site.css/".to_string()];
        assert!(check_links(&pages, "", &asset_hrefs).is_empty());
    }

    #[test]
    fn query_string_on_an_existing_page_link_is_ignored() {
        // レビュー指摘（PR #901・github-actions(bot) P2 / cursor(bot) Low）:
        // `/guide/?tab=cpu` のような query 付きリンクは query を無視して
        // `/guide/` として解決すべきで、query の有無で誤って壊れたリンク
        // 判定してはならない。
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("/guide/?tab=cpu")]),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn query_string_before_fragment_is_ignored_and_fragment_still_validated() {
        let pages = vec![page(
            "/guide/",
            Node::element(
                "html",
                vec![],
                vec![link("/guide/?tab=cpu#section"), heading_with_id("section")],
            ),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());

        let pages_missing = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("/guide/?tab=cpu#missing")]),
        )];
        let broken = check_links(&pages_missing, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("fragment"));
    }

    #[test]
    fn empty_fragment_on_an_existing_page_reports_no_broken_links() {
        // レビュー指摘（PR #901・cursor(bot) Low）: `href="/guide/#"` の
        // ような空 fragment は HTML 仕様上「文書先頭」を意味し、対応する
        // `id` の実在を要求すべきではない。
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("/guide/#")]),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn empty_fragment_on_the_current_page_reports_no_broken_links() {
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("#")]),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn percent_encoded_fragment_matching_an_id_reports_no_broken_links() {
        // レビュー指摘（PR #901・github-actions(bot) P2）: `#foo%20bar` は
        // ブラウザ上 `id="foo bar"` へ解決可能なため、percent-decode 後に
        // 突合すれば壊れたリンクと誤判定されない。
        let pages = vec![page(
            "/guide/",
            Node::element(
                "html",
                vec![],
                vec![link("#foo%20bar"), heading_with_id("foo bar")],
            ),
        )];
        assert!(check_links(&pages, "", &[]).is_empty());
    }

    #[test]
    fn invalid_percent_encoded_fragment_is_reported_broken() {
        // 不正な percent-encoding（`%` の後ろに 2 桁の 16 進数が続かない）は
        // 曖昧な入力を許容側へ倒さず明示的に broken とする。
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("#foo%2")]),
        )];
        let broken = check_links(&pages, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("percent-encoded"));
    }

    #[test]
    fn scheme_starting_with_a_digit_is_not_treated_as_external_and_is_validated() {
        // レビュー指摘（PR #901・github-actions(bot) P2）: RFC 3986 は
        // scheme の先頭を英字必須とするため、`123:missing` のような
        // 数字始まりは外部 URL 扱いせずサイト内リンクとして検証すべき
        // （既知ターゲット表に存在しないため壊れたリンクとして検出される）。
        let pages = vec![page(
            "/guide/",
            Node::element("html", vec![], vec![link("123:missing")]),
        )];
        let broken = check_links(&pages, "", &[]);
        assert_eq!(broken.len(), 1);
        assert!(broken[0].reason.contains("does not exist"));
    }

    #[test]
    fn base_path_is_applied_when_resolving_root_relative_links() {
        let pages = vec![
            page(
                "/guide/",
                Node::element(
                    "html",
                    vec![],
                    // `layout::rewrite_root_relative_hrefs` は本文中のルート
                    // 相対リンクへ `base_path` を反映済みにする。ここではその
                    // 反映済みの形をそのまま模する。
                    vec![link("/rust-ai-library/reference/")],
                ),
            ),
            page("/reference/", Node::element("html", vec![], vec![])),
        ];
        assert!(check_links(&pages, "/rust-ai-library", &[]).is_empty());
    }
}
