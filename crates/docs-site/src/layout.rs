//! ページ骨格（`<html>` 全体）の組み立て: ヘッダ・サイドバー・本文。
//!
//! # 呼び出し文脈
//!
//! `build.rs` が [`nav::parse_nav`](crate::nav::parse_nav) 済みの [`Nav`] と
//! `markdown::markdown_to_nodes` の変換結果（本文ノード列）を `docs_page` へ渡し、
//! 得られた `crate::html::Node` を `html::render` で文字列化してから
//! `<!DOCTYPE html>` を前置してファイルへ書き出す（実装計画 §2.4・§2.6）。
//!
//! # スコープ境界（イシュー #870 → #871 で解消）
//!
//! イシュー #870 時点ではヘッダのアクション領域・CSS 側の `data-theme` フック
//! （`theme.rs`/`assets/site.css`）のみを用意し、3 カラム TOC・全文検索 UI・
//! テーマトグルボタン・FOUC 抑止スクリプトは実装しなかった（兄弟イシュー
//! #871 のスコープとして先送りしていた）。本イシュー（#871）でテーマ
//! トグルボタン・検索 UI・`<head>` への FOUC 抑止スクリプト埋め込みを実装
//! した（3 カラム TOC は依然スコープ外。ページ内目次は `markdown.rs` が
//! 見出し `id` を生成しないため対象外。実装計画 §4.3・§7）。
//!
//! # ヘッダーメニュー・サイドバー設計（Fandhe-AI/fandhe-backend・fandhe-frontend
//! の docs-site と同一方針）
//!
//! - ヘッダーは各セクションをトリガー（`a.docs-header-trigger`）とし、
//!   `:hover`/`:focus-within` でセクション配下ページの一覧
//!   （`ul.docs-header-dropdown`）を表示するホバーメニュー方式にする
//!   （[`header`]）。開閉状態を JS で管理せず CSS のみで完結させるのは、
//!   参照実装と同じく `role`/`aria-expanded`/`aria-haspopup` 等の
//!   「JS が更新する動的状態」を偽装しないため（静的マークアップに
//!   支援技術向けの虚偽の動的状態を持たせない。`assets/site.css` の
//!   `.docs-header-dropdown` セレクタ群コメント参照）
//! - サイドバー（[`sidebar`]）はヘッダーとは対照的に「現在ページが属する
//!   セクションのページのみ」を描画する。ヘッダーのドロップダウンで別
//!   セクションへ遷移すると、遷移後のページ描画時にサイドバーの内容も
//!   その新しいセクションへ切り替わる（参照実装と同じ「ヘッダー＝全体
//!   俯瞰・サイドバー＝現在地の詳細」という役割分担）
//! - モバイル（`assets/site.css` の `@media (max-width: 48rem)`。既存の
//!   イシュー #900 対応ブレークポイントと統一し、参照実装の `767px` とは
//!   別ブレークポイントを増設しない）ではドロップダウンを無効化し、
//!   トリガー列の横スクロールのみ残す。`:hover` はタッチ環境で信頼できず、
//!   トリガー自体がセクション索引ページへのリンクとして導線を担うため
//!   （トリガー到達後はサイドバーが当該セクションの全ページを表示する。
//!   導線をゼロにしない fail-closed 方針）

use crate::html::Node;
use crate::nav::{Nav, Section};
use crate::script;
use crate::search;

/// rust-ai-library の GitHub リポジトリ URL。ヘッダの外部リンク先として使う。
const GITHUB_REPO_URL: &str = "https://github.com/Fandhe-AI/rust-ai-library";

/// `base_path` と `relative`（`/` 始まりでも始まりでなくてもよい）を結合し、
/// GitHub Pages プロジェクトサイト等の `base_path` プレフィックスを考慮した
/// href を組み立てる単一実装点（実装計画 §2.4「asset_href」）。
///
/// - `base_path` は `""` または `/` 始まり・`/` 終わりでない文字列
///   （[`crate::nav::Nav`] の `site.base_path` はパース時点でこの形式を保証済み）
/// - `relative` の先頭 `/` は正規化のため一旦除去してから結合する（二重スラッシュ
///   防止）。`relative` が `/` 単体（サイトトップ）の場合は末尾スラッシュを保つ
pub fn asset_href(base_path: &str, relative: &str) -> String {
    let trimmed = relative.trim_start_matches('/');
    format!("{base_path}/{trimmed}")
}

/// ヘッダのアクション領域に置くテーマトグルボタン。
///
/// `hidden` を既定属性として持つ: `crate::script::SITE_JS` がイベント配線
/// 完了後にのみ `hidden` を解除する fail-closed 設計（`script.rs` モジュール
/// doc の「責務（テーマトグル）」手順 6 参照）。JS 不達・実行前は非表示のまま
/// とし、「押しても何も起きないボタン」を利用者に見せない。
fn theme_toggle_button() -> Node {
    Node::element(
        "button",
        vec![
            ("type".to_string(), "button".to_string()),
            ("class".to_string(), "docs-theme-toggle".to_string()),
            ("hidden".to_string(), "".to_string()),
            ("aria-pressed".to_string(), "false".to_string()),
        ],
        vec![],
    )
}

/// ヘッダのアクション領域に置く全文検索 UI（入力欄 + 結果表示領域）。
///
/// `index_href`（`crate::search::INDEX_REL_PATH` を [`asset_href`] 経由で
/// `base_path` 反映済みにした URL）を `data-search-index` 属性へ埋め込み、
/// `crate::script::SITE_JS` の `initSearch` がここから索引 URL を読み取る
/// （`script.rs` モジュール doc「責務（全文検索）」手順 2）。コンテナ自体も
/// テーマトグルと同じ理由で既定 `hidden`（JS 配線完了後に解除）。
fn search_ui(index_href: &str) -> Node {
    Node::element(
        "div",
        vec![
            ("class".to_string(), "docs-search".to_string()),
            ("hidden".to_string(), "".to_string()),
        ],
        vec![
            Node::element(
                "label",
                vec![("for".to_string(), "docs-search-input".to_string())],
                vec![Node::text("Search")],
            ),
            Node::element(
                "input",
                vec![
                    ("id".to_string(), "docs-search-input".to_string()),
                    ("class".to_string(), "docs-search-input".to_string()),
                    ("type".to_string(), "search".to_string()),
                    ("data-search-index".to_string(), index_href.to_string()),
                    ("autocomplete".to_string(), "off".to_string()),
                    (
                        "aria-controls".to_string(),
                        "docs-search-results".to_string(),
                    ),
                ],
                vec![],
            ),
            Node::element(
                "div",
                vec![
                    ("id".to_string(), "docs-search-results".to_string()),
                    ("class".to_string(), "docs-search-results".to_string()),
                    ("hidden".to_string(), "".to_string()),
                ],
                vec![],
            ),
        ],
    )
}

/// サイトタイトルのリンク先（「ホーム」への href）。`nav.toml` は `page.path
/// = "/"` の宣言を必須としない（`nav::Nav` の型が保証する唯一の不変条件は
/// 「1 件以上のセクション」「各セクション 1 件以上のページ」）ため、無条件に
/// `asset_href(base_path, "/")` を使うと、ルートページを宣言していない
/// nav.toml では実在しないページへのリンクになってしまう
/// （イシュー #872 の linkcheck 実装時に発見: `docs-site` 自身の単体・統合
/// テスト用 fixture の大半は `page.path = "/"` を宣言していないため、
/// linkcheck 導入前は誰にも検出されない潜在的なリンク切れだった）。
///
/// 実サイト（`site/nav.toml`）は先頭セクションの先頭ページが `path = "/"`
/// （Home）であるため実質的な変更はないが、`path = "/"` を宣言しない
/// nav.toml（テスト fixture 含む）でも常に実在するリンクにするため、
/// 「先頭セクションの先頭ページ」（`nav::parse_nav` が保証する不変条件により
/// 必ず存在する）を安全なホームリンク先として採用する。
fn home_href(nav: &Nav) -> String {
    let base_path = nav.site.base_path.as_str();
    nav.sections
        .first()
        .and_then(|section| section.pages.first())
        .map(|page| asset_href(base_path, &page.path))
        .unwrap_or_else(|| asset_href(base_path, "/"))
}

/// セクションの「着地点」href。`section.index_path` があればそれを、
/// なければ先頭ページ（[`crate::nav::parse_nav`] が保証する「1 セクション
/// 1 ページ以上」の不変条件により必ず存在する）を使う。
///
/// [`header`] のトリガーリンクと [`sidebar`] の見出しリンクの双方から呼ぶ
/// 単一実装点にすることで、同じセクションを指す 2 箇所のリンク先が実装の
/// 分岐によって食い違う事故を防ぐ（advisor 指摘）。
fn section_landing_href(base_path: &str, section: &Section) -> String {
    match section.index_path.as_deref() {
        Some(index_path) => asset_href(base_path, index_path),
        None => section
            .pages
            .first()
            .map(|page| asset_href(base_path, &page.path))
            .unwrap_or_else(|| asset_href(base_path, "/")),
    }
}

/// `current_path` が属するセクションを返す（[`sidebar`] の絞り込み表示に使う）。
/// `nav.toml` は同一 `page.path` の重複を禁止する（`nav::parse_nav` の
/// `DuplicatePath` 検証）ため、一致するセクションは高々 1 件。
fn section_for_path<'a>(nav: &'a Nav, current_path: &str) -> Option<&'a Section> {
    nav.sections
        .iter()
        .find(|section| section.pages.iter().any(|page| page.path == current_path))
}

/// ヘッダーのアクション領域に置く GitHub リポジトリリンク。
fn github_link() -> Node {
    Node::element(
        "a",
        vec![
            ("class".to_string(), "docs-github-link".to_string()),
            ("href".to_string(), GITHUB_REPO_URL.to_string()),
            ("target".to_string(), "_blank".to_string()),
            // tabnabbing 対策（.claude/rules/security.md A05）。
            ("rel".to_string(), "noopener noreferrer".to_string()),
        ],
        vec![Node::text("GitHub")],
    )
}

/// ヘッダーのセクションメニュー（構造:
/// `nav.docs-header-nav` の内側に `ul.docs-header-menu`、その内側に
/// `li.docs-header-group` が並ぶ）。各セクションのトリガー
/// （`a.docs-header-trigger`。href は [`section_landing_href`]）+
/// セクション配下ページのドロップダウン（`ul.docs-header-dropdown`）からなる。
///
/// - トリガーには現在ページがそのセクションに属するとき `aria-current="true"`
///   を付与する（ページ完全一致を表す `"page"` とは意味軸を分離し、
///   「現在のセクション」というより粗い粒度を表す。参照実装
///   `fandhe-backend::docs_site::nav::header_nav` と同じ契約）
/// - ドロップダウン内リンクは `page.path == current_path` のとき
///   `aria-current="page"` を付与する（[`sidebar`] と同一契約）
/// - 開閉は JS を使わず CSS の `:hover`/`:focus-within` のみで行うため、
///   `role`/`aria-expanded`/`aria-haspopup` は付与しない（モジュール冒頭
///   「ヘッダーメニュー・サイドバー設計」節参照）
fn header_nav(nav: &Nav, current_path: &str) -> Node {
    let base_path = nav.site.base_path.as_str();

    let groups: Vec<Node> = nav
        .sections
        .iter()
        .map(|section| {
            let in_section = section.pages.iter().any(|page| page.path == current_path);
            let trigger_href = section_landing_href(base_path, section);
            let mut trigger_attrs = vec![
                ("class".to_string(), "docs-header-trigger".to_string()),
                ("href".to_string(), trigger_href),
            ];
            if in_section {
                trigger_attrs.push(("aria-current".to_string(), "true".to_string()));
            }
            let trigger =
                Node::element("a", trigger_attrs, vec![Node::text(section.title.clone())]);

            let items: Vec<Node> = section
                .pages
                .iter()
                .map(|page| {
                    let mut attrs = vec![("href".to_string(), asset_href(base_path, &page.path))];
                    if page.path == current_path {
                        attrs.push(("aria-current".to_string(), "page".to_string()));
                    }
                    Node::element(
                        "li",
                        vec![],
                        vec![Node::element(
                            "a",
                            attrs,
                            vec![Node::text(page.title.clone())],
                        )],
                    )
                })
                .collect();
            let dropdown = Node::element(
                "ul",
                vec![("class".to_string(), "docs-header-dropdown".to_string())],
                items,
            );

            Node::element(
                "li",
                vec![("class".to_string(), "docs-header-group".to_string())],
                vec![trigger, dropdown],
            )
        })
        .collect();

    Node::element(
        "nav",
        vec![
            ("class".to_string(), "docs-header-nav".to_string()),
            ("aria-label".to_string(), "Site sections".to_string()),
        ],
        vec![Node::element(
            "ul",
            vec![("class".to_string(), "docs-header-menu".to_string())],
            groups,
        )],
    )
}

/// ヘッダ: サイトタイトル（[`home_href`] が指す「ホーム」へのリンク）・
/// セクションメニュー（[`header_nav`]）・検索 UI・GitHub リポジトリリンク・
/// テーマトグルの順で構成する（アクション領域内の順序は参照実装
/// `fandhe-backend` の `div.docs-header-actions` と同じ「検索 → GitHub →
/// テーマ」順）。
fn header(nav: &Nav, current_path: &str) -> Node {
    let base_path = nav.site.base_path.as_str();
    let index_href = asset_href(base_path, search::INDEX_REL_PATH);

    Node::element(
        "header",
        vec![("class".to_string(), "site-header".to_string())],
        vec![
            Node::element(
                "div",
                vec![("class".to_string(), "site-title".to_string())],
                vec![Node::element(
                    "a",
                    vec![("href".to_string(), home_href(nav))],
                    vec![Node::text(nav.site.title.clone())],
                )],
            ),
            header_nav(nav, current_path),
            Node::element(
                "div",
                vec![("class".to_string(), "site-header-actions".to_string())],
                vec![search_ui(&index_href), github_link(), theme_toggle_button()],
            ),
        ],
    )
}

/// サイドバー: 現在ページ（`current_path`。生の `page.path` と比較する。
/// `base_path` を含まない）が属するセクションのみを描画する
/// （[`section_for_path`]）。セクション見出し（`h2`）は
/// [`section_landing_href`] へのリンクにする（要件 2: ヘッダーのトリガーと
/// 同じ着地点を指す）。ページの `<a>` には `page.path == current_path` の
/// ときのみ `aria-current="page"` を付与する。
///
/// `current_path` が `nav` 中のどの `page.path` にも一致しない場合は
/// 全セクション・全ページの列挙へフォールバックする（fail-open。参照実装
/// `fandhe-backend::docs_site::nav::sidebar` と同方針: サイトナビゲーション
/// 表示はアクセス境界ではなく、nav 未登録ページで導線を全損させるより
/// 全件表示の方が安全側。実運用の `site/nav.toml` は全ページを列挙するため
/// この分岐は本番経路では通常発生しない）。
fn sidebar(nav: &Nav, current_path: &str) -> Node {
    let base_path = nav.site.base_path.as_str();
    let sections: Vec<&Section> = match section_for_path(nav, current_path) {
        Some(section) => vec![section],
        None => nav.sections.iter().collect(),
    };

    let mut children = Vec::new();
    for section in sections {
        children.push(Node::element(
            "h2",
            vec![],
            vec![Node::element(
                "a",
                vec![("href".to_string(), section_landing_href(base_path, section))],
                vec![Node::text(section.title.clone())],
            )],
        ));

        let page_items: Vec<Node> = section
            .pages
            .iter()
            .map(|page| {
                let mut attrs = vec![("href".to_string(), asset_href(base_path, &page.path))];
                if page.path == current_path {
                    attrs.push(("aria-current".to_string(), "page".to_string()));
                }
                Node::element(
                    "li",
                    vec![],
                    vec![Node::element(
                        "a",
                        attrs,
                        vec![Node::text(page.title.clone())],
                    )],
                )
            })
            .collect();
        children.push(Node::element("ul", vec![], page_items));
    }

    Node::element(
        "aside",
        vec![("class".to_string(), "site-sidebar".to_string())],
        vec![Node::element("nav", vec![], children)],
    )
}

/// `nodes` 内の `<a href="/...">`（ルート相対・`//` プロトコル相対を除く）を
/// 再帰的に探し、`asset_href` と同じ規則で `base_path` を反映する。
///
/// `markdown::markdown_to_nodes` は `site.base_path` を知らない（Markdown 変換を
/// サイト設定から独立させる設計。`markdown.rs` モジュールコメント参照）ため、
/// ヘッダ・サイドバーの nav リンク／CSS の asset リンクと同じ `asset_href` 経由の
/// プレフィックス付与を、本文（`article` 直下）だけここで別途適用する。適用しない
/// と GitHub Pages のプロジェクトサイト配下（本番 `site/nav.toml` は
/// `base_path = "/rust-ai-library"`）で本文内のルート相対リンクだけ `base_path` を
/// 反映せず素通しになり、nav・asset リンクとの扱いが不整合になる（Cursor Bugbot
/// 指摘・イシュー #870）。`href` 以外の属性・`a` 以外のタグは対象外。
fn rewrite_root_relative_hrefs(base_path: &str, nodes: Vec<Node>) -> Vec<Node> {
    nodes
        .into_iter()
        .map(|node| match node {
            Node::Element {
                tag,
                attrs,
                children,
            } => {
                let children = rewrite_root_relative_hrefs(base_path, children);
                let attrs = if tag == "a" {
                    attrs
                        .into_iter()
                        .map(|(name, value)| {
                            if name == "href" && value.starts_with('/') && !value.starts_with("//")
                            {
                                (name, asset_href(base_path, &value))
                            } else {
                                (name, value)
                            }
                        })
                        .collect()
                } else {
                    attrs
                };
                Node::Element {
                    tag,
                    attrs,
                    children,
                }
            }
            text @ Node::Text(_) => text,
        })
        .collect()
}

/// ページ全体（`<html>`）を組み立てる。`body` は本文（`markdown::markdown_to_nodes`
/// の戻り値）。`current_path` はサイドバーの `aria-current` 判定に使う生の
/// `page.path`（`base_path` を含まない）。
///
/// `<!DOCTYPE html>` はここでは前置しない（`build.rs` が書き出し時に前置する。
/// モジュール冒頭コメント参照）。`pub(crate)` 限定（`html::Node` が
/// `pub(crate)` のため。`html.rs` の安全性契約コメント参照）: 呼び出し元は
/// 同一クレート内の `build.rs` のみ。
pub(crate) fn docs_page(nav: &Nav, page_title: &str, current_path: &str, body: Vec<Node>) -> Node {
    let base_path = nav.site.base_path.as_str();
    let css_href = asset_href(base_path, "assets/site.css");
    let script_href = asset_href(base_path, script::SCRIPT_REL_PATH);

    let mut head_children = vec![
        Node::element(
            "meta",
            vec![("charset".to_string(), "utf-8".to_string())],
            vec![],
        ),
        Node::element(
            "meta",
            vec![
                ("name".to_string(), "viewport".to_string()),
                (
                    "content".to_string(),
                    "width=device-width, initial-scale=1".to_string(),
                ),
            ],
            vec![],
        ),
        Node::element(
            "title",
            vec![],
            vec![Node::text(format!("{page_title} | {}", nav.site.title))],
        ),
    ];

    // FOUC 抑止スクリプトは stylesheet `<link>` より前に配置する（同期実行の
    // インライン `<script>` は後続のパース・レンダリングをブロックするため、
    // ここで `<html data-theme>` を確定させてからスタイルシートを適用させる。
    // `script.rs` の `inline_theme_bootstrap` は escape-safe 検証に落ちた場合
    // `None` を返し、その場合は `<script>` 自体を出力せず CSS 側の
    // `prefers-color-scheme` 追従へ fail-closed に退避する（壊れた JS を
    // 配信しない。`script.rs` モジュール doc 参照）。
    if let Some(bootstrap) = script::inline_theme_bootstrap() {
        head_children.push(Node::element("script", vec![], vec![Node::text(bootstrap)]));
    }

    head_children.push(Node::element(
        "link",
        vec![
            ("rel".to_string(), "stylesheet".to_string()),
            ("href".to_string(), css_href),
        ],
        vec![],
    ));

    // `crate::script::SITE_JS` 本体は `defer` で読み込む（DOM 構築をブロック
    // せず、`DOMContentLoaded` 相当のタイミングで実行される。`script.rs` の
    // `ready()` 自体も `document.readyState` を見て二重に安全側へ倒す）。
    head_children.push(Node::element(
        "script",
        vec![
            ("defer".to_string(), "".to_string()),
            ("src".to_string(), script_href),
        ],
        vec![],
    ));

    let head = Node::element("head", vec![], head_children);

    let body_node = Node::element(
        "body",
        vec![],
        vec![
            header(nav, current_path),
            Node::element(
                "div",
                vec![("class".to_string(), "site-body".to_string())],
                vec![
                    sidebar(nav, current_path),
                    Node::element(
                        "main",
                        vec![("class".to_string(), "site-main".to_string())],
                        vec![Node::element(
                            "article",
                            vec![],
                            rewrite_root_relative_hrefs(base_path, body),
                        )],
                    ),
                ],
            ),
        ],
    );

    Node::element(
        "html",
        vec![("lang".to_string(), "ja".to_string())],
        vec![head, body_node],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::html::render;
    use crate::nav::parse_nav;

    const SAMPLE_NAV: &str = r#"
[site]
title = "rust-ai-library"
base_path = "/rust-ai-library"

[[section]]
title = "Guides"
index_path = "/guides/"

[[section.page]]
title = "Guides"
source = "guides.md"
path = "/guides/"

[[section.page]]
title = "Backends"
source = "backends.md"
path = "/guides/backends/"

[[section]]
title = "API"

[[section.page]]
title = "API"
source = "api.md"
path = "/api/"
"#;

    #[test]
    fn asset_href_joins_base_path_and_relative() {
        assert_eq!(
            asset_href("/rust-ai-library", "assets/site.css"),
            "/rust-ai-library/assets/site.css"
        );
        assert_eq!(asset_href("", "assets/site.css"), "/assets/site.css");
        assert_eq!(
            asset_href("/rust-ai-library", "/guides/"),
            "/rust-ai-library/guides/"
        );
        assert_eq!(asset_href("", "/"), "/");
        assert_eq!(asset_href("/rust-ai-library", "/"), "/rust-ai-library/");
    }

    /// 要件 2: サイドバーは現在ページが属するセクションのページのみを描画する
    /// （ヘッダーのドロップダウンで別セクションへ遷移すると切り替わる設計。
    /// モジュール冒頭「ヘッダーメニュー・サイドバー設計」節参照）。
    #[test]
    fn sidebar_scopes_to_current_section_only() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&sidebar(&nav, "/guides/backends/"));
        assert!(html.contains("Guides"));
        assert!(html.contains("href=\"/rust-ai-library/guides/\""));
        assert!(html.contains("href=\"/rust-ai-library/guides/backends/\""));
        // 現在ページが属さない API セクションは含まれない。
        assert!(!html.contains(">API<"));
        assert!(!html.contains("href=\"/rust-ai-library/api/\""));
    }

    /// 要件 2: セクション見出し（`h2`）はそのセクションの着地点（`index_path`。
    /// 無ければ先頭ページ）へのリンクにする（[`section_landing_href`]）。
    #[test]
    fn sidebar_section_heading_links_to_section_landing_href() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&sidebar(&nav, "/guides/backends/"));
        assert!(html.contains("<h2><a href=\"/rust-ai-library/guides/\">Guides</a></h2>"));
    }

    /// 要件 2 のフォールバック（fail-open）: `current_path` がどのページにも
    /// 一致しない場合は全セクション・全ページを描画する（参照実装
    /// `fandhe-backend::docs_site::nav::sidebar` と同方針。`sidebar` doc 参照）。
    #[test]
    fn sidebar_falls_back_to_all_sections_when_current_path_is_unknown() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&sidebar(&nav, "/unknown/"));
        let guides_pos = html.find("Guides").expect("Guides section present");
        let api_pos = html.find(">API<").expect("API section present");
        assert!(
            guides_pos < api_pos,
            "sections must appear in declaration order"
        );
        assert!(html.contains("href=\"/rust-ai-library/guides/\""));
        assert!(html.contains("href=\"/rust-ai-library/guides/backends/\""));
        assert!(html.contains("href=\"/rust-ai-library/api/\""));
    }

    #[test]
    fn sidebar_marks_current_page_with_aria_current() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&sidebar(&nav, "/guides/backends/"));
        assert!(html.contains(
            "<a href=\"/rust-ai-library/guides/backends/\" aria-current=\"page\">Backends</a>"
        ));
        // 現在ページでないリンクには aria-current を付与しない。
        assert!(!html.contains("<a href=\"/rust-ai-library/guides/\" aria-current=\"page\">"));
    }

    /// 要件 1: ヘッダーは各セクションをトリガーとするドロップダウンメニュー
    /// になる。`index_path` を持たないセクション（API）は先頭ページへ
    /// フォールバックする（[`section_landing_href`]）。
    #[test]
    fn header_includes_dropdown_for_each_section_and_github_link_with_safe_rel() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&header(&nav, "/guides/backends/"));
        assert!(html.contains("class=\"docs-header-nav\""));
        assert!(html.contains("class=\"docs-header-menu\""));
        assert!(
            html.contains(r#"class="docs-header-trigger" href="/rust-ai-library/guides/" aria-current="true">Guides</a>"#)
        );
        // ドロップダウンにセクション配下ページが含まれる。
        assert!(html.contains("class=\"docs-header-dropdown\""));
        assert!(html.contains(
            "<a href=\"/rust-ai-library/guides/backends/\" aria-current=\"page\">Backends</a>"
        ));
        // index_path を持たないセクション（API）は先頭ページへフォールバックし、
        // 現在ページはこのセクションに属さないため aria-current は付かない。
        assert!(
            html.contains(r#"class="docs-header-trigger" href="/rust-ai-library/api/">API</a>"#)
        );
        assert!(html.contains(&format!("href=\"{GITHUB_REPO_URL}\"")));
        assert!(html.contains("class=\"docs-github-link\""));
        assert!(html.contains("target=\"_blank\""));
        assert!(html.contains("rel=\"noopener noreferrer\""));
    }

    /// イシュー #872 回帰テスト: `SAMPLE_NAV` は `page.path = "/"` を宣言
    /// していない（`nav.rs`・`nav::Nav` の不変条件はそれを必須にしない）。
    /// このような nav.toml でもサイトタイトルのリンク先（`home_href`）が
    /// 実在するページ（先頭セクションの先頭ページ）を指すこと（無条件
    /// `asset_href(base_path, "/")` を使うと linkcheck が壊れたリンクとして
    /// 検出する。`home_href` ドキュメンテーションコメント参照）。
    #[test]
    fn header_site_title_links_to_first_page_when_root_path_is_not_declared() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&header(&nav, "/guides/"));
        assert!(html.contains("class=\"site-title\""));
        assert!(html.contains("<a href=\"/rust-ai-library/guides/\">rust-ai-library</a>"));
    }

    /// `home_href` 単体テスト: 先頭セクションの先頭ページの `asset_href`
    /// 適用済み href を返す。
    #[test]
    fn home_href_returns_first_section_first_page_href() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        assert_eq!(home_href(&nav), "/rust-ai-library/guides/");
    }

    #[test]
    fn docs_page_wraps_body_and_links_stylesheet() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&docs_page(
            &nav,
            "Guides",
            "/guides/",
            vec![Node::element("h1", vec![], vec![Node::text("Guides")])],
        ));
        assert!(html.starts_with("<html lang=\"ja\">"));
        assert!(html.contains("<title>Guides | rust-ai-library</title>"));
        assert!(html.contains("href=\"/rust-ai-library/assets/site.css\""));
        assert!(html.contains("<article><h1>Guides</h1></article>"));
    }

    /// Cursor Bugbot 指摘（PR #899）の回帰テスト: 本文（`markdown::markdown_to_nodes`
    /// 由来）中のルート相対リンク（`/` 始まり）が `site.base_path` を反映せず
    /// 素通しで出力されると、nav・asset リンクとの扱いが不整合になり GitHub
    /// Pages のプロジェクトサイト配下（`base_path = "/rust-ai-library"`）で本文内
    /// リンクが壊れる。`docs_page` が `article` へ埋め込む前に本文中の `<a>` の
    /// ルート相対 `href` へも `base_path` を反映することを確認する。
    #[test]
    fn docs_page_rewrites_root_relative_links_in_body_with_base_path() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let body = vec![Node::element(
            "p",
            vec![],
            vec![Node::element(
                "a",
                vec![("href".to_string(), "/guides/backends/".to_string())],
                vec![Node::text("Backends")],
            )],
        )];
        let html = render(&docs_page(&nav, "Guides", "/guides/", body));
        assert!(html.contains("<a href=\"/rust-ai-library/guides/backends/\">Backends</a>"));
        // base_path を反映しない生の `/guides/backends/` はもう出現しない
        // （本文中の href としては。sidebar 側の同一パスは既に base_path 反映
        // 済みのため、上の assert と併せて素通し混入がないことを確認する）。
        assert!(!html.contains("href=\"/guides/backends/\""));
    }

    /// `base_path = ""`（GitHub Pages のユーザー/組織サイト等、サブパスなし配置）
    /// では本文中のルート相対リンクを変更しない（`asset_href("", href)` は
    /// 元の値へ戻る）ことを確認する回帰テスト。
    #[test]
    fn docs_page_leaves_root_relative_links_unchanged_when_base_path_is_empty() {
        let nav = parse_nav(
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "intro.md"
path = "/intro/"
"#,
        )
        .expect("valid nav.toml");
        let body = vec![Node::element(
            "a",
            vec![("href".to_string(), "/intro/".to_string())],
            vec![Node::text("Intro")],
        )];
        let html = render(&docs_page(&nav, "Top", "/", body));
        assert!(html.contains("<a href=\"/intro/\">Intro</a>"));
    }

    /// イシュー #871 回帰テスト: FOUC 抑止インラインスクリプトが `<head>` 内で
    /// stylesheet `<link>` より前に位置すること（`docs_page` モジュール doc
    /// 「FOUC 抑止スクリプトは stylesheet `<link>` より前に配置する」の固定）。
    #[test]
    fn docs_page_places_inline_theme_bootstrap_before_stylesheet_link() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&docs_page(&nav, "Guides", "/guides/", vec![]));
        let script_pos = html
            .find("<script>")
            .expect("inline theme bootstrap script should be present");
        let link_pos = html
            .find("<link rel=\"stylesheet\"")
            .expect("stylesheet link should be present");
        assert!(
            script_pos < link_pos,
            "FOUC 抑止スクリプトは stylesheet より前に位置する必要がある"
        );
        assert!(html.contains(crate::script::THEME_STORAGE_KEY));
    }

    /// イシュー #871 回帰テスト: `<script defer src>` で `site.js` を
    /// `base_path` 反映済み URL で読み込むこと。
    #[test]
    fn docs_page_loads_site_js_with_defer_and_base_path() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&docs_page(&nav, "Guides", "/guides/", vec![]));
        assert!(html.contains("<script defer=\"\" src=\"/rust-ai-library/assets/site.js\""));
    }

    /// イシュー #871 回帰テスト: 検索入力欄の `data-search-index` が
    /// `base_path` を反映した索引 URL であること。
    #[test]
    fn header_search_input_reflects_base_path_in_index_url() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&header(&nav, "/guides/"));
        assert!(html.contains("data-search-index=\"/rust-ai-library/assets/search-index.json\""));
    }

    /// イシュー #871 回帰テスト: テーマトグルボタン・検索 UI コンテナは
    /// 既定で `hidden`（JS 配線完了後にのみ解除する fail-closed 設計。
    /// `layout.rs` の `theme_toggle_button`/`search_ui` doc 参照）。
    #[test]
    fn header_theme_toggle_and_search_ui_default_to_hidden() {
        let nav = parse_nav(SAMPLE_NAV).expect("valid nav.toml");
        let html = render(&header(&nav, "/guides/"));
        assert!(html.contains("class=\"docs-theme-toggle\" hidden=\"\""));
        assert!(html.contains("class=\"docs-search\" hidden=\"\""));
        assert!(
            html.contains("id=\"docs-search-results\" class=\"docs-search-results\" hidden=\"\"")
        );
    }
}
