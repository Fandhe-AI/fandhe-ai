//! hf-hub の公開面・依存形状・TLS 構成の機械検査（イシュー #3393）。
//!
//! 役割: 第 11 区分 `ureq` を hf-hub で使う条件（deps-policy.md）が、将来の変更で黙って崩れない
//! ことをソース走査で固定する。facade の `tests/api_surface.rs` と同型だが、test helper を共有
//! できないため自己完結で持つ。否定ガード（「現れない」検査）は、合成入力で違反を検出できる
//! 正のプローブと、走査件数の下限（inventory）を併設する。
//!
//! 検査: (1) `ureq` の語は `src/transport/ureq_json.rs` だけに現れる (2) `lib.rs` の公開項目に
//! HTTP 系の型が出ない (3) 転送層モジュールに制限なしの `pub` がない (4) TLS 構成の固定設定が
//! 本番部分に揃い、無効化手段がない (5) `Cargo.toml` の `ureq` 宣言が `cfg(unix)` 限定・workspace
//! 継承のみ。

use std::fs;
use std::path::{Path, PathBuf};

const UREQ_FILE: &str = "src/transport/ureq_json.rs";

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    fs::read_to_string(crate_dir().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// コメント・文字列・文字リテラルを空白へ置換する（行構造は保つ）。
fn strip(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut depth = 0;
            while i < b.len() {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    if b[i] == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            }
        } else if c == 'r' && matches!(next, Some('"') | Some('#')) && is_raw_start(&b, i) {
            let mut j = i + 1;
            let mut hashes = 0;
            while b.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            j += 1; // opening quote
            loop {
                if j >= b.len() {
                    break;
                }
                if b[j] == '"' && (0..hashes).all(|k| b.get(j + 1 + k) == Some(&'#')) {
                    j += 1 + hashes;
                    break;
                }
                if b[j] == '\n' {
                    out.push('\n');
                }
                j += 1;
            }
            out.push(' ');
            i = j;
        } else if c == '"' {
            i += 1;
            while i < b.len() && b[i] != '"' {
                if b[i] == '\\' {
                    i += 1;
                }
                if i < b.len() && b[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 1;
            out.push(' ');
        } else if c == '\'' && is_char_literal(&b, i) {
            i += 1;
            while i < b.len() && b[i] != '\'' {
                if b[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            out.push(' ');
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

fn is_raw_start(b: &[char], i: usize) -> bool {
    let prev_ident = i > 0 && (b[i - 1].is_alphanumeric() || b[i - 1] == '_');
    if prev_ident {
        return false;
    }
    let mut j = i + 1;
    while b.get(j) == Some(&'#') {
        j += 1;
    }
    b.get(j) == Some(&'"')
}

fn is_char_literal(b: &[char], i: usize) -> bool {
    // `'a'`・`'\n'` は文字リテラル、`'a`（ライフタイム）は違う。
    matches!(
        (b.get(i + 1), b.get(i + 2)),
        (Some('\\'), _) | (Some(_), Some('\''))
    )
}

/// 識別子境界つきで `word` が現れるか。
fn has_word(code: &str, word: &str) -> bool {
    let bytes = code.as_bytes();
    let mut start = 0;
    while let Some(p) = code[start..].find(word) {
        let s = start + p;
        let e = s + word.len();
        let is_id = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let before = s == 0 || !is_id(bytes[s - 1]);
        let after = e >= bytes.len() || !is_id(bytes[e]);
        if before && after {
            return true;
        }
        start = e;
    }
    false
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let p = e.expect("entry").path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// 制限なしの `pub` 項目のシグネチャ（`pub` から最初の `{` か `;` まで）を集める。
fn unrestricted_pub_signatures(code: &str) -> Vec<String> {
    let mut sigs = Vec::new();
    let mut rest = code;
    while let Some(p) = rest.find("pub ") {
        let before_ok = p == 0 || !rest.as_bytes()[p - 1].is_ascii_alphanumeric();
        let tail = &rest[p..];
        if before_ok {
            let end = tail.find(['{', ';']).unwrap_or(tail.len());
            sigs.push(tail[..end].split_whitespace().collect::<Vec<_>>().join(" "));
        }
        rest = &tail[4..];
    }
    sigs
}

const FORBIDDEN_IN_PUBLIC: &[&str] = &[
    "ureq",
    "http",
    "TlsConfig",
    "disable_verification",
    "serde_json",
];

fn leaking_public_signatures(code: &str) -> Vec<String> {
    unrestricted_pub_signatures(code)
        .into_iter()
        .filter(|s| FORBIDDEN_IN_PUBLIC.iter().any(|w| has_word(s, w)))
        .collect()
}

/// コメント除去後のコードから、最初の `#[cfg(test)]` より前（本番部分）を返す。
fn production_part(code: &str) -> &str {
    code.find("#[cfg(test)]").map_or(code, |p| &code[..p])
}

/// 本番部分に足りない必須トークン、または禁止トークンを返す。
fn tls_config_violations(raw_src: &str) -> Vec<String> {
    let mut v = Vec::new();
    let stripped = strip(raw_src);
    let prod_code = production_part(&stripped);
    for need in [
        "https_only(true)",
        "max_redirects(0)",
        "proxy(None)",
        "RootCerts::PlatformVerifier",
        "TlsProvider::NativeTls",
        "timeout_global(Some(",
        "timeout_recv_body(Some(",
        "timeout_connect(Some(",
    ] {
        if !prod_code.contains(need) {
            v.push(format!("missing: {need}"));
        }
    }
    if prod_code.contains("https_only(false)") {
        v.push("https_only(false) in production part".into());
    }
    let all = strip(raw_src).to_ascii_lowercase();
    for bad in ["danger", "insecure", "accept_invalid"] {
        if all.contains(bad) {
            v.push(format!("forbidden identifier: {bad}"));
        }
    }
    let mut rest = all.as_str();
    while let Some(p) = rest.find("disable_verification") {
        let after = &rest[p + "disable_verification".len()..];
        if !after.starts_with("()") {
            v.push("disable_verification is used as non-getter".into());
        }
        rest = after;
    }
    v
}

/// `ureq` 宣言の形状違反を返す（`Cargo.toml` 本文を入力）。
fn cargo_ureq_violations(toml: &str) -> Vec<String> {
    let mut v = Vec::new();
    let mut section = String::new();
    let mut found = 0;
    for raw in toml.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(h) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = h.trim().to_owned();
            if section.ends_with(".ureq") {
                found += 1;
                v.push(format!("[{section}] table form is not allowed"));
            }
            continue;
        }
        let key = line.split(['=', '.']).next().unwrap_or("").trim();
        if key == "ureq" {
            found += 1;
            if section != "target.'cfg(unix)'.dependencies" {
                v.push(format!("ureq declared under [{section}]"));
            }
            let norm: String = line.chars().filter(|c| !c.is_whitespace()).collect();
            if norm != "ureq.workspace=true" && norm != "ureq={workspace=true}" {
                v.push("ureq must be `ureq.workspace = true` only".into());
            }
        }
    }
    if found != 1 {
        v.push(format!("ureq must appear exactly once (found {found})"));
    }
    v
}

#[test]
fn ureq_token_is_confined_to_one_file() {
    let mut files = Vec::new();
    rs_files(&crate_dir().join("src"), &mut files);
    rs_files(&crate_dir().join("tests"), &mut files);
    // inventory: 走査対象が空振りしていないこと（src の主要ファイル + 本ファイル）。
    assert!(files.len() >= 9, "scanned only {} files", files.len());
    let mut hit_allowed = false;
    for f in &files {
        let rel = f
            .strip_prefix(crate_dir())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let code = strip(&fs::read_to_string(f).expect("read"));
        let has = has_word(&code, "ureq");
        if rel == UREQ_FILE {
            hit_allowed = has;
        } else {
            assert!(!has, "ureq token leaked into {rel}");
        }
    }
    assert!(hit_allowed, "probe: {UREQ_FILE} should contain the token");
}

#[test]
fn lib_public_surface_has_no_http_types() {
    let code = strip(&read("src/lib.rs"));
    assert!(!unrestricted_pub_signatures(&code).is_empty(), "inventory");
    assert!(
        leaking_public_signatures(&code).is_empty(),
        "{:?}",
        leaking_public_signatures(&code)
    );
    assert!(!code.contains("pub mod "), "pub mod is not allowed");
    assert!(!code.contains("pub use transport"), "transport is internal");
}

#[test]
fn transport_modules_have_no_unrestricted_pub() {
    for f in ["src/transport.rs", UREQ_FILE] {
        let code = strip(&read(f));
        assert!(
            unrestricted_pub_signatures(&code).is_empty(),
            "{f}: {:?}",
            unrestricted_pub_signatures(&code)
        );
        assert!(code.contains("pub(crate)") || code.contains("pub(super)"));
    }
}

#[test]
fn tls_configuration_is_pinned() {
    let v = tls_config_violations(&read(UREQ_FILE));
    assert!(v.is_empty(), "{v:?}");
}

#[test]
fn cargo_toml_declares_ureq_only_via_unix_workspace_inheritance() {
    let v = cargo_ureq_violations(&read("Cargo.toml"));
    assert!(v.is_empty(), "{v:?}");
}

// ---- 否定ガードの自己テスト（合成入力。違反を検出できること・正例で 0 件であること）----

#[test]
fn strip_removes_comments_and_literals() {
    let s =
        strip("let a = \"ureq\"; // ureq\n/* ureq */ let b = 'x'; let c: &'a str; r#\"ureq\"#;");
    assert!(!has_word(&s, "ureq"));
    assert!(s.contains("let c"));
    assert!(has_word(&strip("use ureq::Agent;"), "ureq"));
    assert!(!has_word(&strip("use ureq_json::X;"), "ureq"));
}

#[test]
fn probe_public_signature_leak_is_detected() {
    let bad = strip("pub fn f(a: ureq::Agent) -> u8 { 0 }\npub fn g() -> http::Uri { todo!() }");
    assert_eq!(leaking_public_signatures(&bad).len(), 2);
    let good = strip("pub fn f(a: u8) -> u8 { a }\npub(crate) fn g(a: ureq::Agent) {}");
    assert!(leaking_public_signatures(&good).is_empty());
    let multi = strip("pub fn f(\n a: serde_json::Value,\n) {}");
    assert_eq!(leaking_public_signatures(&multi).len(), 1);
}

#[test]
fn probe_unrestricted_pub_is_detected() {
    assert_eq!(unrestricted_pub_signatures("pub fn a() {}").len(), 1);
    assert!(unrestricted_pub_signatures("pub(crate) fn a() {}").is_empty());
}

#[test]
fn probe_tls_violations_are_detected() {
    let good = "fn b() { c.https_only(true).max_redirects(0).proxy(None)\
        .root_certs(RootCerts::PlatformVerifier).provider(TlsProvider::NativeTls)\
        .timeout_global(Some(a)).timeout_recv_body(Some(a)).timeout_connect(Some(a)); \
        let _ = x.disable_verification(); }\n#[cfg(test)]\nmod t { fn p() { https_only(false) } }";
    assert!(
        tls_config_violations(good).is_empty(),
        "{:?}",
        tls_config_violations(good)
    );
    let bad_http = good.replace("https_only(true)", "https_only(false)");
    assert!(!tls_config_violations(&bad_http).is_empty());
    let bad_missing = good.replace(".proxy(None)", "");
    assert!(!tls_config_violations(&bad_missing).is_empty());
    let bad_setter = good.replace("x.disable_verification()", "x.disable_verification(true)");
    assert!(!tls_config_violations(&bad_setter).is_empty());
    let bad_name = good.replace("fn b()", "fn danger_accept()");
    assert!(!tls_config_violations(&bad_name).is_empty());
}

#[test]
fn probe_cargo_toml_shapes() {
    let good = "[dependencies]\nserde.workspace = true\n\n[target.'cfg(unix)'.dependencies]\nureq.workspace = true\n";
    assert!(cargo_ureq_violations(good).is_empty());
    let inline = "[target.'cfg(unix)'.dependencies]\nureq = { workspace = true }\n";
    assert!(cargo_ureq_violations(inline).is_empty());
    for bad in [
        "[dependencies]\nureq.workspace = true\n",
        "[target.'cfg(unix)'.dependencies]\nureq = \"=3.4.2\"\n",
        "[target.'cfg(unix)'.dependencies]\nureq = { workspace = true, features = [\"x\"] }\n",
        "[target.'cfg(unix)'.dependencies.ureq]\nworkspace = true\n",
        "[dependencies]\nserde.workspace = true\n",
        "[target.'cfg(unix)'.dependencies]\nureq.workspace = true\n[dependencies]\nureq.workspace = true\n",
    ] {
        assert!(!cargo_ureq_violations(bad).is_empty(), "{bad}");
    }
}
