//! `api_surface.rs` 専用の「正ガード」共通部品（イシュー #3782）。
//!
//! 否定ガード・保留 doctest ガード（`*HoldDoctestGuard`）を、承認形だけを許す正ガードへ
//! 反転するときに毎回ほぼ同形で書いていた走査（宣言の数え上げ・`pub use` の固定・
//! 「各類型を検出する」自己テスト・doctest の実在検査）をまとめる。反転の手順と先例は
//! 同ディレクトリの `README.md` を正とし、ここには書き写さない。
//!
//! - 呼び出し元: `api_surface.rs` の `mod surface_guard;` だけ。`api_surface` の子モジュールなので、
//!   既存の private な走査部品（`strip_comments_and_literals` 等）を `super::` で再利用し、
//!   判定ロジックは複製しない（既存ガードの判定は不変）。
//! - 契約: すべて `facade_src_files()` と同形の `(src からの相対パス, 内容)` の組を入力にし、
//!   違反を `Vec<String>` で返す（空なら合格）。panic するのは自己テストの試験台だけ。
//! - `tests/common/` に置かない理由: `common` は多数のテストバイナリが取り込み、使わない
//!   `pub` 項目が `dead_code` で clippy `-D warnings` を落とすため。

// 本体は後続の反転イシューが使う部品。現時点では同ファイルの等価性テストだけが呼ぶため、
// 非 test ビルドでは未使用になる。
#![allow(dead_code)]

use super::{
    approved_reexport_surface_violations, count_fn_declarations_by_name, doc_lines_above_fn,
    doctest_probe_violations, facade_crate_root, inner_doc_lines, read_to_string_or_panic,
    strip_comments_and_literals, tokenize_including_punctuation,
};

/// コメント・リテラルを除いたトークン列にする（宣言走査の共通前処理）。
fn tokens_of(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    tokenize_including_punctuation(&cleaned)
}

/// 手順 1: `fn <name>` の宣言を含むファイルを、宣言の数だけ返す。
pub fn fn_declaration_sites(files: &[(String, String)], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (rel, content) in files {
        let n = count_fn_declarations_by_name(&tokens_of(content), name);
        out.extend(std::iter::repeat_n(rel.clone(), n));
    }
    out
}

/// 手順 1: `trait|struct|enum|type|union <name>` の宣言を含むファイルを、宣言の数だけ返す。
pub fn type_declaration_sites(files: &[(String, String)], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (rel, content) in files {
        let tokens = tokens_of(content);
        let n = tokens
            .windows(2)
            .filter(|w| {
                matches!(
                    w[0].as_str(),
                    "trait" | "struct" | "enum" | "type" | "union"
                ) && w[1] == name
            })
            .count();
        out.extend(std::iter::repeat_n(rel.clone(), n));
    }
    out
}

/// 手順 1: 宣言場所の多重集合が `expected` と完全一致しなければ違反（fail-closed）。
/// 0 件（取り下げ・改名）・承認外の場所・重複はいずれも違反。`expected = &[]` は
/// 「宣言が残っていない」の確認に使う。
pub fn declaration_count_violations(
    label: &str,
    sites: &[String],
    expected: &[&str],
) -> Vec<String> {
    let mut got: Vec<&str> = sites.iter().map(String::as_str).collect();
    let mut want: Vec<&str> = expected.to_vec();
    got.sort_unstable();
    want.sort_unstable();
    if got == want {
        Vec::new()
    } else {
        vec![format!(
            "{label}: 宣言場所 {got:?} が承認形 {want:?} と一致しない"
        )]
    }
}

/// 手順 2: 承認済み `pub use` 文だけを許す検査の入口（`approved_reexport_surface_violations`
/// をそのまま呼ぶ。判定は不変）。独自ラッパー型を許す場合は `type_declaration_sites` と
/// 組み合わせ、`names` には内部型の名前だけを渡す。
pub fn reexport_violations(
    files: &[(String, String)],
    names: &[&str],
    approved: &[(&str, &str)],
) -> Vec<String> {
    approved_reexport_surface_violations(files, names, approved)
}

/// 手順 3: 検出器が「常に通る」状態にならないことを固定する自己テストの試験台。
/// `base` は違反 0 件、`missing` なら 1 ファイル欠落ごとに違反、`injected` は各組を足すと違反、
/// `benign` は足しても違反 0 件でなければならない。満たさなければ panic する。
pub fn assert_detects_each_category(
    label: &str,
    checker: impl Fn(&[(String, String)]) -> Vec<String>,
    base: &[(String, String)],
    missing: bool,
    injected: &[(&str, &str)],
    benign: &[(&str, &str)],
) {
    let v = checker(base);
    assert!(v.is_empty(), "{label}: 承認形 base が違反になる: {v:?}");
    if missing {
        for i in 0..base.len() {
            let mut files = base.to_vec();
            let removed = files.remove(i).0;
            assert!(
                !checker(&files).is_empty(),
                "{label}: {removed} の欠落を検出できない"
            );
        }
    }
    let with = |extra: &(&str, &str)| {
        let mut files = base.to_vec();
        files.push((extra.0.to_string(), extra.1.to_string()));
        files
    };
    for extra in injected {
        assert!(
            !checker(&with(extra)).is_empty(),
            "{label}: 混入 {extra:?} を検出できない"
        );
    }
    for extra in benign {
        let v = checker(&with(extra));
        assert!(
            v.is_empty(),
            "{label}: 無害な {extra:?} を誤検出した: {v:?}"
        );
    }
}

/// 手順 5: doctest を探す doc の位置。
pub enum DocAnchor<'a> {
    /// ファイル先頭の `//!` モジュール doc。
    Module,
    /// `pub struct AveragedModel` のように、この文字列で始まる行の直前の `///` doc。
    Item(&'a str),
}

/// 手順 5: 実在を固定する利用例 doctest 1 件分。
pub struct UsageDoctest<'a> {
    pub label: &'a str,
    /// `crates/facade/src` からの相対パス。
    pub file: &'a str,
    pub anchor: DocAnchor<'a>,
    pub needles: &'a [&'a str],
}

/// 内容と anchor から doc 行を取り出して `doctest_probe_violations` に掛ける（合成入力の自己テスト用に分離）。
fn doc_probe_violations_for(content: &str, entry: &UsageDoctest<'_>) -> Vec<String> {
    let lines = match entry.anchor {
        DocAnchor::Module => inner_doc_lines(content),
        DocAnchor::Item(decl) => doc_lines_above_fn(content, decl),
    };
    doctest_probe_violations(entry.label, &lines, entry.needles)
}

/// 手順 5: 各エントリの利用例 doctest が実在し、実際にコンパイルされる形であることを検査する。
/// 空入力が違反になること（検出器の空振り防止）の自己確認も先に行う。
pub fn usage_doctest_violations(entries: &[UsageDoctest<'_>]) -> Vec<String> {
    assert!(
        !doctest_probe_violations("t", &[], &["x"]).is_empty(),
        "doctest_probe_violations が空の doc を違反にしない"
    );
    let root = facade_crate_root();
    let mut v = Vec::new();
    for e in entries {
        let content = read_to_string_or_panic(&root.join("src").join(e.file));
        v.extend(doc_probe_violations_for(&content, e));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::super::{
        PHASE4_TRAINING_DATA_APPROVED_REEXPORTS, PHASE4_TRAINING_DATA_NAMES, facade_src_files,
    };
    use super::*;

    fn f(rel: &str, body: &str) -> (String, String) {
        (rel.to_string(), body.to_string())
    }

    // ---- 等価性: 実 facade src で既存ガードと同じ判定になる ----

    #[test]
    fn equivalent_to_accumulate_steps_guards() {
        let files = facade_src_files();
        let sites = fn_declaration_sites(&files, "accumulate_steps");
        assert_eq!(sites, vec!["compat/training.rs".to_string()]);
        assert!(declaration_count_violations("a", &sites, &["compat/training.rs"]).is_empty());
        let none = fn_declaration_sites(&files, "with_accumulate_steps_for_test");
        assert!(declaration_count_violations("b", &none, &[]).is_empty());
    }

    #[test]
    fn equivalent_to_swa_declaration_guards() {
        let files = facade_src_files();
        assert_eq!(
            type_declaration_sites(&files, "AveragedModel"),
            vec!["optim_swa.rs".to_string()]
        );
        assert_eq!(
            type_declaration_sites(&files, "SwaCallback"),
            vec!["compat/callbacks.rs".to_string()]
        );
        assert!(fn_declaration_sites(&files, "use_swa").is_empty());
    }

    #[test]
    fn equivalent_to_phase4_reexport_guard() {
        let v = reexport_violations(
            &facade_src_files(),
            &PHASE4_TRAINING_DATA_NAMES,
            &PHASE4_TRAINING_DATA_APPROVED_REEXPORTS,
        );
        assert!(v.is_empty(), "{v:?}");
    }

    #[test]
    fn equivalent_to_swa_usage_doctest_guard() {
        let v = usage_doctest_violations(&[
            UsageDoctest {
                label: "optim.rs モジュール doc（SWA 節）",
                file: "optim.rs",
                anchor: DocAnchor::Module,
                needles: &["AveragedModel", "SwaLr", "SwaAnneal"],
            },
            UsageDoctest {
                label: "AveragedModel の doc",
                file: "optim_swa.rs",
                anchor: DocAnchor::Item("pub struct AveragedModel"),
                needles: &[
                    "use fandhe_ai::optim::AveragedModel",
                    "AveragedModel::new",
                    "averaged_parameters",
                ],
            },
            UsageDoctest {
                label: "SwaCallback の doc",
                file: "compat/callbacks.rs",
                anchor: DocAnchor::Item("pub struct SwaCallback"),
                needles: &["Callback::Swa", "SwaCallback::new", "raw_state_dict"],
            },
        ]);
        assert!(v.is_empty(), "{v:?}");
    }

    // ---- 自己テスト: 空振り防止 ----

    #[test]
    fn declaration_helpers_detect_zero_duplicate_and_foreign_sites() {
        let files = vec![
            f(
                "a.rs",
                "pub fn foo() {}\n// fn foo() {}\nconst S: &str = \"fn foo()\";",
            ),
            f("b.rs", "struct Bar;\nenum Baz {}\n"),
        ];
        let sites = fn_declaration_sites(&files, "foo");
        assert_eq!(sites, vec!["a.rs".to_string()]);
        assert!(declaration_count_violations("x", &sites, &["a.rs"]).is_empty());
        assert!(!declaration_count_violations("x", &sites, &["b.rs"]).is_empty());
        assert!(!declaration_count_violations("x", &[], &["a.rs"]).is_empty());
        let dup = vec!["a.rs".to_string(), "a.rs".to_string()];
        assert!(!declaration_count_violations("x", &dup, &["a.rs"]).is_empty());
        assert!(!declaration_count_violations("x", &sites, &[]).is_empty());
        assert_eq!(
            type_declaration_sites(&files, "Bar"),
            vec!["b.rs".to_string()]
        );
        assert_eq!(
            type_declaration_sites(&files, "Baz"),
            vec!["b.rs".to_string()]
        );
        assert!(type_declaration_sites(&files, "Qux").is_empty());
    }

    fn reexport_checker(files: &[(String, String)]) -> Vec<String> {
        reexport_violations(files, &["Foo"], &[("m.rs", "pub use inner::Foo;")])
    }

    fn reexport_base() -> Vec<(String, String)> {
        vec![f("m.rs", "pub use inner::Foo;\n")]
    }

    #[test]
    fn assert_detects_each_category_accepts_working_checker() {
        assert_detects_each_category(
            "synthetic",
            reexport_checker,
            &reexport_base(),
            true,
            &[
                ("x.rs", "pub use inner::Foo;"),
                ("x.rs", "pub use inner::Foo as Bar;"),
                ("x.rs", "pub use inner::{Foo};"),
                ("x.rs", "pub struct Foo;"),
            ],
            &[
                ("x.rs", "// pub use inner::Foo;"),
                ("x.rs", "const S: &str = \"pub use inner::Foo;\";"),
                ("x.rs", "use inner::Foo;"),
                ("x.rs", "pub use inner::Other;"),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "混入")]
    fn assert_detects_each_category_panics_when_checker_always_passes() {
        assert_detects_each_category(
            "always-pass",
            |_| Vec::new(),
            &reexport_base(),
            false,
            &[("x.rs", "pub use inner::Foo;")],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "承認形 base")]
    fn assert_detects_each_category_panics_when_base_is_violating() {
        assert_detects_each_category(
            "bad-base",
            |_| vec!["v".to_string()],
            &reexport_base(),
            false,
            &[],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "誤検出")]
    fn assert_detects_each_category_panics_on_false_positive() {
        assert_detects_each_category(
            "false-positive",
            |files| {
                if files.len() > 1 {
                    vec!["v".to_string()]
                } else {
                    Vec::new()
                }
            },
            &reexport_base(),
            false,
            &[],
            &[("x.rs", "// comment")],
        );
    }

    #[test]
    fn doc_probe_violations_for_detects_missing_needle_and_non_compiled_blocks() {
        let entry = UsageDoctest {
            label: "t",
            file: "unused.rs",
            anchor: DocAnchor::Item("pub struct S"),
            needles: &["S::new"],
        };
        let ok = "/// ```\n/// let s = S::new();\n/// ```\npub struct S;\n";
        assert!(doc_probe_violations_for(ok, &entry).is_empty());
        let ignored = "/// ```ignore\n/// let s = S::new();\n/// ```\npub struct S;\n";
        assert!(!doc_probe_violations_for(ignored, &entry).is_empty());
        let missing = "/// ```\n/// let s = T::new();\n/// ```\npub struct S;\n";
        assert!(!doc_probe_violations_for(missing, &entry).is_empty());
        let module = UsageDoctest {
            anchor: DocAnchor::Module,
            ..entry
        };
        let inner = "//! ```\n//! let s = S::new();\n//! ```\n";
        assert!(doc_probe_violations_for(inner, &module).is_empty());
        assert!(!doc_probe_violations_for("", &module).is_empty());
    }
}
