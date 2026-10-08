//! `text` モジュール（語彙 lookup 型のテキスト変換。イシュー #2897）の衛生契約を
//! CI で常時検査する。
//!
//! `docs/facade-text-vectorization-design.md` §8 の機械検査契約
//! （`crates/facade/src/text/` にファイル・ネットワーク・スレッド・並列・
//! 危険なブロックが現れない）と、公開形が未承認の間は `text` を facade へ
//! 公開しないこと（`lib.rs` が非公開の `mod text;` のみ）を、コメントを含む
//! 生テキストの照合で固定する。走査対象は固定パスのみで外部入力を取らない。

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
fn text_module_is_not_exposed_from_facade() {
    let lib = std::fs::read_to_string(src_dir().join("lib.rs")).unwrap_or_default();
    assert!(
        lib.contains("\nmod text;"),
        "lib.rs に非公開 `mod text;` がない"
    );
    assert!(!lib.contains("pub mod text"));
    assert!(!lib.contains("pub(crate) mod text"));
    let mut files = Vec::new();
    collect_rs(&src_dir(), &mut files);
    for f in files {
        let body = std::fs::read_to_string(&f).unwrap_or_default();
        assert!(
            !body.contains("pub use crate::text"),
            "{} が text を再エクスポートしている",
            f.display()
        );
    }
}
