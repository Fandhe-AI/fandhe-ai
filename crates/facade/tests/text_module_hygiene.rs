//! `text` モジュール（語彙 lookup 型のテキスト変換。イシュー #2897）の衛生契約を
//! CI で常時検査する。
//!
//! `docs/facade-text-vectorization-design.md` §8 の機械検査契約
//! （`crates/facade/src/text/` にファイル・ネットワーク・スレッド・並列・
//! 危険なブロックが現れない）と、`text` の公開が承認形（`lib.rs` の `pub mod text;`
//! 1 件と `text/mod.rs` の 6 名の `pub use` のみ。イシュー #2937・設計記録 §16.2）に
//! 限られることを、コメントを含む生テキストの照合で固定する。走査対象は固定パスのみで外部入力を取らない。

use std::path::{Path, PathBuf};

const FORBIDDEN: [&str; 6] = [
    "std::fs",
    "std::net",
    "std::thread",
    "std::sync::mpsc",
    "rayon",
    "unsafe",
];

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir);
    assert!(entries.is_ok(), "read_dir 失敗: {}", dir.display());
    let Ok(entries) = entries else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn text_module_has_no_forbidden_constructs() {
    let mut files = Vec::new();
    collect_rs(&src_dir().join("text"), &mut files);
    assert!(!files.is_empty(), "src/text/ に .rs が 1 件もない");
    for f in files {
        let body = std::fs::read_to_string(&f);
        assert!(body.is_ok(), "読み取り失敗: {}", f.display());
        let body = body.unwrap_or_default();
        for word in FORBIDDEN {
            assert!(
                !body.contains(word),
                "{} に禁止語 {word:?} が含まれる",
                f.display()
            );
        }
    }
}

#[test]
fn text_module_is_exposed_only_in_approved_shape() {
    let lib = std::fs::read_to_string(src_dir().join("lib.rs")).unwrap_or_default();
    assert_eq!(
        lib.matches("\npub mod text;").count(),
        1,
        "lib.rs に `pub mod text;` がちょうど 1 件ない"
    );
    assert_eq!(lib.matches("\nmod text;").count(), 0);
    assert!(!lib.contains("pub(crate) mod text"));

    // text/mod.rs の行頭 `pub use` は承認済みの 6 名だけ（別名・glob・pub mod を拒否）。
    let text_mod =
        std::fs::read_to_string(src_dir().join("text").join("mod.rs")).unwrap_or_default();
    let mut names: Vec<String> = Vec::new();
    for line in text_mod.lines() {
        assert!(
            !line.starts_with("pub mod"),
            "text/mod.rs にサブモジュールの公開がある: {line}"
        );
        if let Some(rest) = line.strip_prefix("pub use ") {
            assert!(
                !rest.contains(" as ") && !rest.contains('*') && !rest.contains("crate::text"),
                "承認外の形の pub use: {line}"
            );
            let (_, tail) = rest.split_once("::").unwrap_or(("", rest));
            let tail = tail
                .trim_end_matches(';')
                .trim_matches(|c| c == '{' || c == '}');
            names.extend(
                tail.split(',')
                    .map(|n| n.trim().to_string())
                    .filter(|n| !n.is_empty()),
            );
        }
    }
    names.sort();
    assert_eq!(
        names,
        [
            "Split",
            "Standardize",
            "TextError",
            "TextLimits",
            "TextVectorization",
            "TextVectorizationConfig"
        ]
    );

    // サブモジュールは公開せず、クレートルートへの再エクスポートもしない。
    let mut files = Vec::new();
    collect_rs(&src_dir(), &mut files);
    for f in files {
        let body = std::fs::read_to_string(&f).unwrap_or_default();
        assert!(
            !body.contains("pub use crate::text"),
            "{} が text を再エクスポートしている",
            f.display()
        );
        if f.starts_with(src_dir().join("text")) {
            assert!(
                !body.contains("\npub mod "),
                "{} がサブモジュールを公開している",
                f.display()
            );
        }
    }
}
