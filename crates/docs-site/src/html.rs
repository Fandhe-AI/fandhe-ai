//! 最小 HTML ノード層。`markdown`・`layout` の両モジュールが生成物を組み立てる際の
//! 唯一の HTML 文字列化経路を担う。
//!
//! # 呼び出し文脈
//!
//! `markdown::markdown_to_nodes` が Markdown 原稿を `Node` 木へ変換し、
//! `layout::docs_page` がそれをページ骨格（ヘッダ・サイドバー・本文）に組み込む。
//! 最終的な HTML 文字列化は必ず `render` を経由し、`build.rs` はその戻り値を
//! そのままファイルへ書き出す（`format!` によるタグ組み立ての迂回経路を
//! このモジュール外に作らないことで XSS 対策を一元化する。実装計画 §2.1・
//! イシュー #870）。`render_all`（兄弟ノード列の一括連結）は本番経路
//! （`build.rs`）では使わず、`markdown` モジュールの単体テストが変換結果を
//! まとめて文字列比較する用途の `#[cfg(test)]` 限定ヘルパーとする。
//!
//! # 安全性契約（イシュー #870 の設計判断）
//!
//! - `Node` は `Element` と `Text` の 2 バリアントのみで構成し、**生 HTML を
//!   注入できるバリアント（`RawHtml` 相当）を持たない**。これにより
//!   「エスケープをスキップする経路」自体が型として存在しない
//!   （参照実装 fandhe-backend の `fandhe_frontend_core::Node` より安全側に単純化。
//!   実装計画 §2.1）
//! - テキスト・属性値は `render` 系関数の内部でのみ HTML エスケープする。
//!   呼び出し側（`markdown`・`layout`）はエスケープ済みでない生文字列を
//!   `Node::Text` / 属性値としてそのまま渡してよい
//! - `tag` 名・属性キーはエスケープ対象外（`render_into` はそのまま出力する）
//!   ため、**任意の外部入力から `tag`／属性キーを構築してはならない**契約に
//!   なっている。この契約を型で強制するため、`Node::element`・
//!   `Node::text` は `pub(crate)` に限定し、`docs-site` クレート外から
//!   構築できないようにする（呼び出し元は `markdown`・`layout`・`build` の
//!   同一クレート内コードのみで、いずれもタグ名・属性キーは固定文字列
//!   リテラルしか渡さない。codex-review 指摘・PR #899・P0）

/// HTML ノード木。`Element` は開始・終了タグと属性・子ノードを持ち、`Text` は
/// レンダリング時に必ずエスケープされる素のテキストを保持する。
///
/// `pub(crate)` 限定（型自体・バリアントのフィールドとも）: Rust の enum は
/// バリアントのフィールドだけを個別に非公開化できず、型が `pub` だと
/// `Node::Element { tag, .. }` の直接構築・パターン分解のどちらも外部から
/// 素通りしてしまう。`Node::element`／`Node::text` を `pub(crate)` に
/// 限定しても型自体が `pub` のままではこのフィールド直接構築で迂回できて
/// しまうため、型そのものをクレート内限定にして迂回経路を閉じる
/// （`docs-site` を利用する他クレートは存在せず本クレート内でのみ完結する
/// SSG 実装のため外部公開は不要。codex-review 指摘・PR #899・P0）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Node {
    /// 要素ノード（`tag` は小文字の HTML タグ名を想定）。
    Element {
        tag: String,
        attrs: Vec<(String, String)>,
        children: Vec<Node>,
    },
    /// テキストノード。`render` 時に `&` `<` `>` をエスケープする。
    Text(String),
}

impl Node {
    /// 要素ノードを組み立てる便利コンストラクタ。
    ///
    /// `pub(crate)` 限定（`docs-site` クレート外から呼べない）: `tag`・
    /// 属性キーは `render_into` でエスケープされずそのまま出力されるため、
    /// 外部からの任意文字列を受け付けると HTML 注入（XSS）の経路になる
    /// （モジュール冒頭コメント「安全性契約」参照。codex-review 指摘・
    /// PR #899・P0）。クレート内の呼び出し元（`markdown`・`layout`）は
    /// タグ名・属性キーに固定文字列リテラルしか渡さない。
    pub(crate) fn element(
        tag: impl Into<String>,
        attrs: Vec<(String, String)>,
        children: Vec<Node>,
    ) -> Node {
        Node::Element {
            tag: tag.into(),
            attrs,
            children,
        }
    }

    /// テキストノードを組み立てる便利コンストラクタ。`pub(crate)` 限定の理由は
    /// `Node::element` のドキュメンテーションコメント参照（テキスト自体は
    /// `render` 時にエスケープされるため注入経路にはならないが、コンストラクタ
    /// を型として揃えるため同様に限定する）。
    pub(crate) fn text(value: impl Into<String>) -> Node {
        Node::Text(value.into())
    }
}

/// void 要素（終了タグを持たない要素）のホワイトリスト。この一覧に一致するタグは
/// 常に自己終端（`<tag ... />`）でレンダリングし、子ノードがあっても出力しない
/// （呼び出し側が void 要素へ子ノードを渡すのは呼び出し側のバグであり、ここで
/// 黙って握りつぶす。子を渡さない契約は呼び出し側〈layout.rs〉が守る）。
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// テキストコンテンツの HTML エスケープ（`&` `<` `>` の 3 種類）。
fn escape_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
    out
}

/// 属性値の HTML エスケープ（`&` `<` `>` `"` `'` の 5 種類。属性値はダブルクォートで
/// 囲むため `"` を、`'` も念のため無害化する）。
fn escape_attr(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// 単一ノードを HTML 文字列へレンダリングする。`pub(crate)` 限定の理由は
/// `Node` のドキュメンテーションコメント参照（`Node` 自体が `pub(crate)`
/// のため、シグネチャの一貫性としても外部公開しない）。
pub(crate) fn render(node: &Node) -> String {
    let mut out = String::new();
    render_into(node, &mut out);
    out
}

/// ノード列を連結して HTML 文字列へレンダリングする（`<article>` 本文等、
/// 兄弟ノード列をまとめて扱う呼び出し元向け）。`#[cfg(test)]` 限定:
/// 本番経路（`build.rs`）は個々のページ全体を 1 つの `Node` へ組み立てて
/// から `render` を呼ぶため使わない。`markdown` モジュールの単体テストが
/// `markdown_to_nodes` の戻り値（`Vec<Node>`）をまとめて文字列比較する際の
/// 専用ヘルパー。
#[cfg(test)]
pub(crate) fn render_all(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        render_into(node, &mut out);
    }
    out
}

fn render_into(node: &Node, out: &mut String) {
    match node {
        Node::Text(text) => out.push_str(&escape_text(text)),
        Node::Element {
            tag,
            attrs,
            children,
        } => {
            out.push('<');
            out.push_str(tag);
            for (key, value) in attrs {
                out.push(' ');
                out.push_str(key);
                out.push_str("=\"");
                out.push_str(&escape_attr(value));
                out.push('"');
            }
            if VOID_ELEMENTS.contains(&tag.as_str()) {
                out.push_str(" />");
                return;
            }
            out.push('>');
            for child in children {
                render_into(child, out);
            }
            out.push_str("</");
            out.push_str(tag);
            out.push('>');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_text_content() {
        let node = Node::text("<script>alert('x')&\"y\"</script>");
        assert_eq!(
            render(&node),
            "&lt;script&gt;alert('x')&amp;\"y\"&lt;/script&gt;"
        );
    }

    #[test]
    fn escapes_attribute_values() {
        let node = Node::element(
            "a",
            vec![("href".to_string(), "\"><script>x</script>".to_string())],
            vec![Node::text("link")],
        );
        assert_eq!(
            render(&node),
            "<a href=\"&quot;&gt;&lt;script&gt;x&lt;/script&gt;\">link</a>"
        );
    }

    #[test]
    fn renders_nested_elements() {
        let node = Node::element(
            "ul",
            vec![],
            vec![
                Node::element("li", vec![], vec![Node::text("a")]),
                Node::element("li", vec![], vec![Node::text("b")]),
            ],
        );
        assert_eq!(render(&node), "<ul><li>a</li><li>b</li></ul>");
    }

    #[test]
    fn renders_void_elements_self_closing_without_children() {
        let node = Node::element(
            "link",
            vec![
                ("rel".to_string(), "stylesheet".to_string()),
                ("href".to_string(), "/assets/site.css".to_string()),
            ],
            vec![],
        );
        assert_eq!(
            render(&node),
            "<link rel=\"stylesheet\" href=\"/assets/site.css\" />"
        );
    }

    #[test]
    fn render_all_concatenates_sibling_nodes() {
        let nodes = vec![
            Node::element("h1", vec![], vec![Node::text("Title")]),
            Node::element("p", vec![], vec![Node::text("Body")]),
        ];
        assert_eq!(render_all(&nodes), "<h1>Title</h1><p>Body</p>");
    }
}
