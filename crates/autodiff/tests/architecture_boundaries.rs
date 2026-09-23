//! アーキテクチャ境界の固定テスト（TASK-12.1d・#164）。
//!
//! `docs/fusion-graph-design.md` §3.4「`autodiff` は具体クレートへの依存
//! を一切持たない」・`.claude/rules/coding-rust.md`「本番経路で
//! `unwrap()`/`expect()` を使わない」を、grep ベースで機械的に固定する
//! 回帰ガード。
//!
//! **A03 インジェクション対策の一環**でもある: `crates/autodiff/` 配下
//! （`Cargo.toml`・`src/`）以外は走査しない固定パスのみを対象とし、
//! 外部入力を受け取らない（`.claude/rules/security.md`）。

use std::path::Path;

fn autodiff_crate_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_to_string_or_panic(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("test fixture: {} が読めない: {e}", path.display()))
}

/// `crates/autodiff/Cargo.toml` が具体バックエンドクレート
/// （`backend-cpu`／`backend-cuda`／`backend-metal`）へ依存していないこと
/// を固定する（`docs/fusion-graph-design.md` §3.4「`autodiff` は
/// `backend-cpu`／`backend-cuda`／`backend-metal` のいずれにも依存しない
/// （`tensor-core` への既存依存のみを保つ）」）。
///
/// `[dev-dependencies]` も対象に含める: 境界検査テスト（本ファイル）自体
/// と矛盾しないよう、`autodiff` の src 内 `#[cfg(test)]` フィクスチャ
/// （`test_support.rs`）・統合テストのフィクスチャ（`tests/common/`）は
/// いずれも `eval.rs`／`tensor-core` の `pub` API のみで naive 実装を
/// 独立に持つ設計とし、`backend-cpu` を dev-dependency に追加していない
/// （実装計画の設計判断。`crates/onnx-interop`・`crates/guardrail`・
/// `crates/self-repair` の fixture とは異なる）。
#[test]
fn autodiff_cargo_toml_does_not_depend_on_concrete_backends() {
    let cargo_toml = autodiff_crate_root().join("Cargo.toml");
    let content = read_to_string_or_panic(&cargo_toml);
    for forbidden in ["backend-cpu", "backend-cuda", "backend-metal"] {
        assert!(
            !content.contains(forbidden),
            "crates/autodiff/Cargo.toml が具体バックエンドクレート {forbidden} に依存している\
             （docs/fusion-graph-design.md §3.4 の不変条件違反）"
        );
    }
}

/// `crates/autodiff/src/` 配下の `.rs` ファイルが `backend_cpu`／
/// `backend_cuda`／`backend_metal`（クレート名。Rust 識別子は `-` を `_`
/// に正規化する）を参照していないことを固定する。
#[test]
fn autodiff_src_does_not_reference_concrete_backend_crates() {
    let src_dir = autodiff_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for forbidden in ["backend_cpu", "backend_cuda", "backend_metal"] {
            if content.contains(forbidden) {
                offending.push(format!("{}: {forbidden}", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "crates/autodiff/src/ が具体バックエンドクレートを直接参照している: {offending:?}"
    );
}

/// `crates/autodiff/src/eval.rs` に `panic!`／`unwrap()`／`expect()`／
/// `unreachable!()` が存在しないことを固定する（TASK-12.1d・#164。
/// `docs/fusion-graph-design.md` §2.5「eval.rs 非 panic 化の設計方針」・
/// §3.5.3 (iii)「`materialize_non_fallible` が `eval.rs` を最終手段として
/// 使う経路が構造的に失敗しないための前提」）。
///
/// `#[cfg(test)]` が付与された項目（関数単位の `#[cfg(test)] fn ...`・
/// 末尾 `mod tests { ... }` のいずれも）はテストコード自身（`.unwrap()`
/// を使うテストアサーション）とみなし対象外とする——本テストの対象は
/// 「本番経路（`#[cfg(test)]` が付与されていない項目）」のみである。
///
/// ファイル冒頭からの単純な prefix 切り出し（最初の `#[cfg(test)]` より
/// 前のみを走査）は、`reduce_axis`／`sum`／`max`（TASK-12.1d・#164）の
/// ように個別関数へ `#[cfg(test)]` を付与しつつその後ろに本番コード
/// （`mse_loss`／`softmax_along`／`cross_entropy_loss` 等）が続く構成では
/// 後続の本番コードを丸ごとガード対象から取りこぼす（実装計画時の
/// 見落とし）。本実装は `#[cfg(test)]` 属性行を検出するたびに、その
/// 直後に続く項目（次に現れる `{` から対応する `}` までの波括弧ブロック）
/// だけを中括弧の深さで追跡してスキャン対象から除外し、それ以外の行は
/// すべて「本番経路」として扱う（`mod tests { ... }` の網羅除外・個別
/// `#[cfg(test)] fn` の網羅除外の両方をこの単一ロジックで扱える）。
#[test]
fn eval_rs_has_no_panic_macros_outside_test_module() {
    let eval_rs = autodiff_crate_root().join("src/eval.rs");
    let content = read_to_string_or_panic(&eval_rs);
    let production_code = strip_cfg_test_items(&content);

    for forbidden in ["panic!(", "unwrap()", "expect(", "unreachable!("] {
        assert!(
            !production_code.contains(forbidden),
            "eval.rs の本番経路（#[cfg(test)] が付与されていない項目）に {forbidden} が含まれている\
             （eval.rs 非 panic 化の契約違反。docs/fusion-graph-design.md §2.5）"
        );
    }
}

/// `content` から `#[cfg(test)]` が付与された項目（関数・`mod` 等の
/// 波括弧ブロック）を丸ごと除去し、残りを「本番経路」の行のみを結合した
/// 文字列として返す（コメント行 `//`／`///`／`//!` も除外する。規約を
/// 説明する散文が `unwrap()` 等の語を含みうるため誤検知防止）。
///
/// `eval_rs_has_no_panic_macros_outside_test_module`（上記）専用の
/// テストユーティリティ。中括弧の深さのみで項目境界を追跡する単純な
/// 実装のため、文字列リテラル中の `{`/`}` は非対応（`eval.rs` に該当
/// パターンが無いことを前提とする）。
fn strip_cfg_test_items(content: &str) -> String {
    let mut out = String::new();
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("#[cfg(test)]") {
            // この属性が付与された項目（次の `{` から対応する `}` まで）
            // を丸ごと読み飛ばす。属性直後に別の属性（`#[derive(...)]`
            // 等）が続く場合もあるため、`{` が現れるまで行を読み進める。
            let mut depth = 0usize;
            let mut entered = false;
            for skip_line in lines.by_ref() {
                for ch in skip_line.chars() {
                    match ch {
                        '{' => {
                            depth += 1;
                            entered = true;
                        }
                        '}' => depth = depth.saturating_sub(1),
                        _ => {}
                    }
                }
                if entered && depth == 0 {
                    break;
                }
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `content` から行コメント（`//`）とブロックコメント（`/* ... */`。
/// ネスト対応）を空白へ置換して取り除いた文字列を返す（codex-review
/// 指摘・PR #2212 その 6）: `extract_trait_body`／
/// `extract_supertrait_bound_tokens` が `content.find("pub trait
/// CustomFunction")` で最初の一致を採用するため、実定義より前にブロック
/// コメントで偽の `pub trait CustomFunction { ... }` を置かれると、
/// コメント除去前の走査ではその偽定義を境界検査の対象として抜き出して
/// しまう（実定義の検査を素通りできる）。行コメント（`//` 始まりの行）
/// のみを除去する旧来の `no_comments` 相当の処理ではブロックコメントに
/// 対応できないため、本関数は両方を対象にする。文字列リテラル中の
/// `//`／`/*` は非対応（`strip_top_level_statements` と同じ簡易実装
/// 方針。`custom.rs` に該当パターンが無いことを前提とする）。
///
/// **ブロックコメントは前後に半角スペース 1 個を挿入してから除去する**
/// （codex-review 指摘・PR #2212 その 10）: 旧実装はコメント本体を単に
/// 読み飛ばすだけだったため `use crate::BackendOps/**/as Send;` のような
/// 隣接トークンが `BackendOpsas` へ連結し、後段のトークナイザ
/// （`tokenize_including_punctuation`／`extract_identifier_tokens`）が
/// 1 個の識別子として誤認識して `as` エイリアス検出・`use`／`type`
/// キーワード判定（`strip_keyword_prefix`）をすり抜けさせてしまう。
/// スペース挿入によりコメント除去後もトークン境界を保つ。行コメントは
/// 常に行末（`\n` の直前）で終わるため、既存の「`\n` を保持したまま
/// スキップする」実装のままでもトークン連結は起きない。
fn strip_comments(content: &str) -> String {
    let chars: Vec<char> = content.chars().collect();
    let len = chars.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;
    while i < len {
        let c = chars[i];
        if c == '/' && i + 1 < len && chars[i + 1] == '/' {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && i + 1 < len && chars[i + 1] == '*' {
            out.push(' ');
            let mut depth = 1i32;
            i += 2;
            while i < len && depth > 0 {
                if i + 1 < len && chars[i] == '/' && chars[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if i + 1 < len && chars[i] == '*' && chars[i + 1] == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    if chars[i] == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            }
            out.push(' ');
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// `content`（`strip_comments` 済み想定）中の文字列リテラル（`"..."`）
/// および char リテラル（`'x'`／`'"'`／`'\\''`／`'\u{XXXX}'` 等）の
/// **内容のみ**を空白へ置換し、開始・終了のクォート文字自体は保持
/// する（ライフタイム `'a` 等の閉じクォートを伴わないトークンは対象外
/// でそのまま残す）。文字列側のエスケープシーケンス（`\"`／`\\` 等）は
/// バックスラッシュの直後の 1 文字をまとめて消費し文字列終端と誤認
/// しない（raw 文字列リテラル `r"..."`／`r#"..."#` は非対応。
/// `custom.rs` および `crates/autodiff/src` 配下に該当パターンが無い
/// ことを前提とする `strip_comments` と同じ簡易実装方針）。
///
/// [`var_impl_block_bodies`] の中括弧深さ追跡はトークン列上の `{`／`}`
/// のみを見て文字列・char リテラルの中身を特別扱いしないため、次の
/// 2 通りのバイパス（いずれも PR #2212 追加 P1 の是正時に自己レビューで
/// 判明）が成立しうる:
///
/// 1. `impl Var { fn helper() { let _ = "}"; } pub fn custom(&self) {}
///    }` のように**文字列**リテラル内に `}` を含むと、リテラル内の
///    `}` を実際の中括弧として誤カウントして impl 本体の終端を実際
///    より手前で確定してしまい、本体末尾の `pub fn custom` を本体の
///    外へ追い出して検出をすり抜けさせる。
/// 2. `impl Var { fn h() -> char { '}' } pub fn custom(&self) {} }`
///    のように**char** リテラル `'}'` がトークン化後に単独の `}`
///    トークンとして現れる場合も同様に中括弧カウントを狂わせる。
///    `'"'` の場合は下の `"` 検査に「文字列の開始」と誤認識され、
///    離れた場所の次の `"` までを丸ごと空白化してしまう。
///
/// 本関数は両リテラルの中身を空白化することでこれらを塞ぐ。開始・
/// 終了のクォート文字自体は残すため、`skip_fn_declaration_qualifiers`
/// の ABI 文字列リテラル判定（`extern "C"` のクォートをトークンとして
/// 検出する処理）とは整合したまま保たれる。
///
/// **対象外（本関数では扱わない・報告のみ）**: `strip_comments` は
/// `content` に対して本関数より前段で適用される前提のため、文字列
/// リテラル内の `//`／`/*` を実コメントの開始と誤認して行末までを
/// 読み飛ばしてしまう既存の限界（`impl Var { fn s() -> &str { "//" }
/// pub fn custom(&self) {} }` 等）はそのまま残る。これは
/// `strip_comments` を利用する本ファイルの全テストに共通する既知の
/// 簡易実装方針であり、単一パスの本格的な字句解析器を要する別軸の
/// 改善であるため本 P1 修正の対象外とする。
fn strip_string_literal_contents(content: &str) -> String {
    let chars: Vec<char> = content.chars().collect();
    let len = chars.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;
    while i < len {
        let c = chars[i];
        if c == '\'' {
            // 文字リテラル（`'"'`／`'\\''`／`'\u{XXXX}'` 等）とライフタイム
            // （`'a` 等・閉じクォートを伴わない）を判別する（PR #2212
            // 追加 P1 の是正時に自己レビューで判明）: 判別せず `'` を
            // 無視すると、
            // `'"'` のような「文字列引用符 1 文字を表す char リテラル」
            // が下の `"` 検査に「文字列の開始」と誤認識され、以降で偶然
            // 現れる次の `"` までを丸ごと文字列扱いして空白化してしまう
            // （本来の文字列リテラルではないコードが消えてしまう）。
            if chars.get(i + 1) == Some(&'\\') {
                // エスケープシーケンス。`\u{XXXX}` は可変長のため `}` まで
                // 読み進めてから閉じクォートを消費する。中身（バック
                // スラッシュ以降・閉じクォート未満のすべての文字）は
                // 空白へ置換し開始・終了クォートのみ残す——`'\u{7b}'`
                // のような unicode escape はソーステキスト上に `{`／`}`
                // という文字がそのまま現れるため、中身を素通しすると
                // [`var_impl_block_bodies`] の中括弧深さ追跡を実際の
                // 中括弧と誤って狂わせてしまう（PR #2212 追加 P1 の是正時に自己レビューで判明・PR #2212
                // 追加 P1）。
                out.push('\'');
                let mut j = i + 1;
                while j < len && chars[j] != '\'' {
                    out.push(' ');
                    j += 1;
                }
                if j < len {
                    out.push('\'');
                    j += 1;
                }
                i = j;
                continue;
            }
            if chars.get(i + 2) == Some(&'\'') {
                // 単純な 1 文字の char リテラル（`'"'`／`'{'`／`'}'` 等）。
                // 中身の 1 文字を空白へ置換する（`'{'`／`'}'` を素通しする
                // と、トークン化後に単独の `{`／`}` トークンとして現れ
                // [`var_impl_block_bodies`] の中括弧深さ追跡を誤らせて
                // しまう。`'"'` を素通しした場合の文字列開始誤認識と
                // 同種のバイパスであり、いずれも中身を残す理由がない）。
                out.push('\'');
                out.push(' ');
                out.push('\'');
                i += 3;
                continue;
            }
            // 閉じクォートが直後に無い ⇒ ライフタイム（`'a` 等）。その
            // まま素通しし、後続の識別子文字は通常のトークンとして
            // `tokenize_including_punctuation` に処理させる。
            out.push(c);
            i += 1;
            continue;
        }
        if c != '"' {
            out.push(c);
            i += 1;
            continue;
        }
        out.push('"');
        i += 1;
        while i < len {
            let cur = chars[i];
            if cur == '\\' && i + 1 < len {
                out.push(' ');
                out.push(' ');
                i += 2;
                continue;
            }
            if cur == '"' {
                out.push('"');
                i += 1;
                break;
            }
            out.push(if cur == '\n' { '\n' } else { ' ' });
            i += 1;
        }
    }
    out
}

/// `statement` 先頭の属性列（`strip_leading_attributes` と同じ `[`／`]`
/// の深さ追跡）を走査し、いずれかの属性の**先頭識別子（属性パス名）**が
/// `cfg`／`cfg_attr` と一致するかを判定する（codex-review 指摘・
/// PR #2212 その 15）。
///
/// 旧実装 `statement_has_cfg_test_attribute` はステートメント全体を
/// トークン化し `cfg`・`(`・`test`・`)` という並びが**任意の位置**に
/// 現れるかで判定していたため、次の 2 通りの迂回を許していた:
///
/// 1. `#[cfg(any())]`（常に無効・実質デッドコードの条件）が付いた
///    canonical import は `cfg(test)` という並びを含まないため
///    「条件付き属性なし」と判定され、canonical import として素通り
///    してしまう。
/// 2. `#[cfg_attr(any(), cfg(test))]` のように、属性の**引数内**に
///    `cfg(test)` というトークン列が現れるだけの `cfg_attr` 属性まで
///    「`#[cfg(test)]` 属性が付いている」と誤判定し、当該 import 文を
///    まるごと検証対象から外してしまう（本来の目的は「テスト専用
///    import を canonical 判定から除外する」ことだが、これにより
///    パス不一致の検証自体を迂回できていた）。
///
/// 本関数は属性ごとに `[` 直後の**先頭識別子のみ**（`cfg(...)` の
/// `cfg`、`cfg_attr(...)` の `cfg_attr`）を見るため、上記いずれの
/// 迂回も塞ぐ。判定は「`cfg`／`cfg_attr` が付いているか否か」のみで
/// あり、`cfg(test)` のような特定条件への一致は問わない（条件付き
/// Tensor import はテスト専用スコープに限らずすべて拒否する設計。
/// `find_canonical_tensor_import` のドキュメントコメント参照）。
fn statement_has_conditional_attribute(statement: &str) -> bool {
    let mut rest = statement.trim_start();
    while let Some(after_hash) = rest.strip_prefix('#') {
        let after_hash = after_hash.trim_start();
        let Some(after_bracket) = after_hash.strip_prefix('[') else {
            break;
        };
        let mut depth = 1i32;
        let mut end = None;
        for (idx, ch) in after_bracket.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(idx);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            // `strip_leading_attributes` と同じ fail-closed 方針:
            // 対応する `]` が無い不正な属性は読み飛ばさず走査を打ち切る。
            break;
        };
        let attr_body = &after_bracket[..end];
        let attr_name = extract_identifier_tokens(attr_body).into_iter().next();
        if matches!(attr_name.as_deref(), Some("cfg") | Some("cfg_attr")) {
            return true;
        }
        rest = after_bracket[end + 1..].trim_start();
    }
    false
}

/// ステートメント（またはその可視性修飾除去後の残り）の先頭が識別子
/// トークンとして `keyword`（`"use"`／`"type"`）と一致するかを判定し、
/// 一致すればキーワード以降の残り文字列（前後の空白は trim 済み）を
/// 返す（codex-review 指摘・PR #2212 その 7）: `starts_with("use ")`／
/// `starts_with("type ")` は半角スペース 1 個固定の文字列一致のため、
/// `use\ncrate::...`／`type\nTensor = ...` のように改行を挟んだ有効な
/// Rust 記法を見逃す。本関数は `trim_start()` で任意の空白（改行・タブ
/// 含む）を読み飛ばしたうえで `keyword` の文字列一致を取り、直後が
/// 識別子構成文字（英数字／`_`）でないこと（`used`／`typeof` 等の無関係
/// な識別子ではないこと）を確認してから残りを返す。
fn strip_keyword_prefix<'a>(statement: &'a str, keyword: &str) -> Option<&'a str> {
    let trimmed = statement.trim_start();
    let rest = trimmed.strip_prefix(keyword)?;
    if rest
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }
    Some(rest.trim_start())
}

fn visit_rs_files(dir: &Path, f: &mut impl FnMut(&Path, &str)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files(&path, f);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let content = read_to_string_or_panic(&path);
            f(&path, &content);
        }
    }
}

/// `crates/autodiff/src/custom.rs` の
/// `pub trait CustomFunction: <supertrait 境界> { ... }` を、トレイト
/// 宣言（`pub trait` から続く supertrait 境界のヘッダー部分を含む）から
/// 閉じ括弧までまるごと部分文字列として抜き出す（イシュー #2064 §12.5
/// (b) 第 3 項「新規 trait が `BackendOps` 等を引数に取らないことの機械
/// 検査」の前段）。**ヘッダー（`pub trait Name: Send + Sync + 'static`
/// の supertrait 境界部分）を検査対象から取りこぼさない**（codex-review
/// 指摘・PR #2212: 開始 `{` 以降のみを返す旧実装では `Send + Sync +
/// 'static` 境界の削除も `BackendOps` 等の混入もこのテストで検出できな
/// かった）。トレイト本体の開始 `{` から対応する `}` までは中括弧の深さ
/// で追跡する（`strip_cfg_test_items` と同じ単純な深さ追跡方式。トレイ
/// ト本体は文字列リテラル中に `{`/`}` を含まないため対応不要）。トレイ
/// ト定義が見つからない場合は空文字列を返す（呼び出し側のアサーションで
/// 検出不能を明示的に fail させるため、黙って全文を返さない）。
///
/// **`content` は呼び出し側で `strip_comments` 済みであることを前提と
/// する**（codex-review 指摘・PR #2212 その 6）: コメント除去前の生
/// テキストに対して `content.find(&needle)` で最初の一致を採用すると、
/// 実定義より前にブロックコメントで偽の `pub trait CustomFunction { ...
/// }` を置かれた場合にその偽定義を抜き出してしまい、境界検査（denylist・
/// allowlist・supertrait 境界）を素通りできてしまう。
fn extract_trait_body(content: &str, trait_name: &str) -> String {
    let needle = format!("pub trait {trait_name}");
    let Some(start) = content.find(&needle) else {
        return String::new();
    };
    let after_needle = &content[start..];
    let Some(brace_offset) = after_needle.find('{') else {
        return String::new();
    };
    let body_start = start + brace_offset;
    let mut depth = 0i32;
    let mut end = None;
    for (i, ch) in content[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(body_start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    match end {
        // `start`（`pub trait Name` の先頭）から返すことで、開始 `{` の
        // 手前にある supertrait 境界（`: Send + Sync + 'static` 等の
        // ヘッダー）を検査対象に含める。
        Some(end) => content[start..end].to_string(),
        None => String::new(),
    }
}

/// `crates/autodiff/src/custom.rs` の `pub trait CustomFunction: <supertrait
/// 境界> { ... }` の **supertrait 境界リストのみ**（トレイト名〈および
/// ジェネリクス `<...>` があればそれも読み飛ばした後〉直後の `:` から、
/// `where` 節／本体開始 `{` の手前までの区間を `+` 区切りで分割したトー
/// クン列）を抜き出す（イシュー #2064 §12.5 (b)。codex-review 指摘・
/// PR #2212 その 3: ヘッダー全体〈`pub trait Name` から開始 `{` 手前まで〉
/// に対する単純なトークン存在判定では、`pub trait CustomFunction<Send,
/// Sync, T> where T: 'static` のように `Send`／`Sync` がジェネリクス
/// 仮引数名・`'static` が where 節側の境界として現れる「偽装」ケースで
/// も supertrait 自体を削除した検査を素通りできてしまう。本実装は
/// ジェネリクス区間・where 節を構造的に除外し、実際の supertrait 境界
/// リストの区間だけを供出源とすることで、この偽装を防ぐ）。
///
/// - トレイト名の直後に `<...>`（ジェネリクス仮引数リスト）が続く場合は
///   `<`/`>` の深さ追跡でバランスよく読み飛ばす
/// - 続く最初の非空白文字が `:` でなければ supertrait 境界が存在しない
///   （空の `Vec` を返す。呼び出し側で「必須境界が見つからない」として
///   fail させるため、黙って全文を対象にしない）
/// - `:` の後は `where`（キーワード）・開始 `{` のうち先に現れる方の
///   手前までを境界リスト区間とし、`+` で分割してトリムしたトークン列
///   を返す
///
/// **`content` は呼び出し側で `strip_comments` 済みであることを前提と
/// する**（`extract_trait_body` と同じ理由。codex-review 指摘・PR #2212
/// その 6: コメント除去前の生テキストではブロックコメント中の偽ヘッダー
/// を最初の一致として抜き出してしまう）。
fn extract_supertrait_bound_tokens(content: &str, trait_name: &str) -> Vec<String> {
    let needle = format!("pub trait {trait_name}");
    let Some(start) = content.find(&needle) else {
        return Vec::new();
    };
    let mut cursor = start + needle.len();
    let after_needle = &content[cursor..];
    let leading_ws = after_needle.len() - after_needle.trim_start().len();
    if after_needle.trim_start().starts_with('<') {
        let generics_start = cursor + leading_ws;
        let mut depth = 0i32;
        for (i, ch) in content[generics_start..].char_indices() {
            match ch {
                '<' => depth += 1,
                '>' => {
                    depth -= 1;
                    if depth == 0 {
                        cursor = generics_start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    let rest_trimmed = content[cursor..].trim_start();
    let Some(after_colon) = rest_trimmed.strip_prefix(':') else {
        return Vec::new();
    };
    let where_pos = after_colon.find("where");
    let brace_pos = after_colon.find('{');
    let end = match (where_pos, brace_pos) {
        (Some(w), Some(b)) => w.min(b),
        (Some(w), None) => w,
        (None, Some(b)) => b,
        (None, None) => after_colon.len(),
    };
    after_colon[..end]
        .split('+')
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .collect()
}

/// `text` 中の識別子トークン（`[A-Za-z_][A-Za-z0-9_]*`。ライフタイムは
/// 先頭の `'` を含めて 1 トークンとして扱う。例: `'static`）を出現順に
/// 列挙する（`custom_function_trait_signatures_are_host_tensor_only`
/// 専用の allowlist 判定ユーティリティ。codex-review 指摘・PR #2212
/// その 2「型エイリアス経由の禁止型混入」対策。denylist〈固定識別子の
/// 文字列検索〉だけでは `use ... as Ops` のような別名を素通りするため、
/// シグネチャに現れる識別子を allowlist と突き合わせる fail-closed 方式
/// を追加する）。
fn extract_identifier_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' || c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            if c == '\'' {
                i += 1;
            }
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();
            if token != "'" {
                tokens.push(token);
            }
        } else {
            i += 1;
        }
    }
    tokens
}

/// `content` 中の `needle` が識別子境界で一致しているか判定する
/// （`needle` の前後が英数字／`_` でなければ独立した識別子とみなす）。
/// 部分一致（例: `Device` に対する `DeviceAllocator`）を誤検出しない
/// ための最小限のトークン境界チェック。
fn contains_identifier(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut start = 0usize;
    while let Some(rel) = haystack[start..].find(needle) {
        let idx = start + rel;
        let before_ok =
            idx == 0 || !bytes[idx - 1].is_ascii_alphanumeric() && bytes[idx - 1] != b'_';
        let after_idx = idx + needle_bytes.len();
        let after_ok = after_idx >= bytes.len()
            || !bytes[after_idx].is_ascii_alphanumeric() && bytes[after_idx] != b'_';
        if before_ok && after_ok {
            return true;
        }
        start = idx + 1;
    }
    false
}

/// `content` を波括弧の深さ 0 の `;` 区切りでステートメント単位に分割し、
/// 各断片（前後の空白を除去）を返す（`custom_function_trait_signatures_
/// are_host_tensor_only` 専用。codex-review 指摘・PR #2212 その 3「複数行
/// または可視性付き alias で禁止型検査を迂回できる」対策）。
///
/// `use crate::{A, B as C};` のように波括弧を含む文は、内側の `{`/`}`
/// で深さが上下する間は `;` があっても分割せず、深さが 0 に戻った後の
/// `;`（または波括弧そのものの閉じ）で初めて 1 断片として確定する。
/// これにより複数行にまたがる `use` 文もひと続きの文字列として保持され、
/// 行単位の `starts_with` 判定では検出できなかった複数行エイリアス
/// import を後続の `.contains(" as ")` 判定で捕捉できる。
///
/// 単純な深さ追跡のみのため文字列リテラル中の `{`/`}`/`;` には非対応
/// （`custom.rs` に該当パターンが無いことを前提とする。`strip_cfg_test_
/// items` と同じ簡易実装方針）。
fn split_top_level_statements(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in content.chars() {
        match ch {
            '{' => {
                depth += 1;
                current.push(ch);
            }
            '}' => {
                depth -= 1;
                current.push(ch);
                if depth <= 0 {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        out.push(trimmed.to_string());
                    }
                    current.clear();
                    depth = 0;
                }
            }
            ';' if depth == 0 => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

/// `text` を「識別子トークン」と「区切り文字（1 文字）」に分解した
/// トークン列へ変換する（空白は読み飛ばす）。`declares_pub_fn` 専用の
/// ユーティリティ。`extract_identifier_tokens`（識別子のみを抽出する
/// 既存関数）と異なり `(`／`<` などの区切り文字もトークンとして残す
/// ことで、`pub`・`fn`・関数名の間に改行やコメント除去後の空白を挟んだ
/// 有効な Rust 記法を、固定文字列一致ではなくトークン列の連続一致で
/// 検出できるようにする（codex-review 指摘・PR #2212 その 5:
/// `contains("pub fn custom(")`／`contains("pub fn custom<")` の固定
/// 文字列一致は `pub\nfn custom(` のような改行を挟んだ宣言を見逃す）。
fn tokenize_including_punctuation(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '\'' || c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            if c == '\'' {
                i += 1;
            }
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        } else {
            tokens.push(c.to_string());
            i += 1;
        }
    }
    tokens
}

/// `tokens[j..]` の先頭から連続する `fn` 宣言の修飾子トークン
/// （`unsafe`／`const`／`async`／`extern` およびその ABI 文字列リテラル。
/// 任意順・0 個以上の繰り返し）を読み飛ばし、修飾子列の直後の index を
/// 返す（codex-review 指摘・PR #2212 その 11）: 旧実装は `unsafe`／
/// `const`／`async` の 3 種のみを読み飛ばしており `pub extern "C" fn
/// custom` のような ABI 指定付き宣言を見逃していた（`extern` トークンが
/// `fn` 直前の位置に来るため `tokens.get(j) == Some("fn")` の一致に
/// 失敗し `declares_pub_fn` が false を返す）。`extern` は ABI 文字列
/// リテラル（`"C"` 等）を伴う場合と伴わない場合の両方が有効な Rust
/// 記法であるため、`extern` の直後にトークン化された文字列リテラル
/// （`"` 開始トークンから対応する `"` 終了トークンまで）が続けばそれも
/// まとめて読み飛ばす。
fn skip_fn_declaration_qualifiers(tokens: &[String], mut j: usize) -> usize {
    loop {
        match tokens.get(j).map(String::as_str) {
            Some("unsafe" | "const" | "async") => {
                j += 1;
            }
            Some("extern") => {
                j += 1;
                if tokens.get(j).map(String::as_str) == Some("\"") {
                    j += 1;
                    while let Some(tok) = tokens.get(j) {
                        j += 1;
                        if tok == "\"" {
                            break;
                        }
                    }
                }
            }
            _ => break,
        }
    }
    j
}

/// `content`（コメント除去済み想定）に `pub fn <fn_name>(` または
/// `pub fn <fn_name><`（ジェネリクス付き）宣言が存在するかをトークン列
/// の連続一致で判定する。`pub`・`fn`・`fn_name` の間の空白量（改行を
/// 含む）に影響されない。`pub`・`fn` の間に `unsafe`／`const`／`async`／
/// `extern`（ABI 文字列リテラル付き含む）修飾子（0 個以上・任意順の
/// 繰り返し）が挟まる宣言も検出する（`skip_fn_declaration_qualifiers`。
/// `crates/facade/tests/api_surface.rs::
/// unapproved_onnx_pub_fn_with_qualifiers_is_flagged` が既に固定して
/// いる `pub async fn`／`pub unsafe fn` の扱いに合わせる）。
/// `pub(crate) fn ...` のようなスコープ付き可視性は「独立した `pub`
/// トークンの直後に修飾子または `fn` トークンが続かない」ため一致しない
/// （`pub` の直後に `(` が来る）——本関数の検査対象はあくまで無条件
/// `pub fn` 宣言のみで、旧実装（固定文字列一致）と同じ可視性スコープの
/// 扱いを保つ。
fn declares_pub_fn(content: &str, fn_name: &str) -> bool {
    tokens_declare_pub_fn(&tokenize_including_punctuation(content), fn_name)
}

/// [`declares_pub_fn`] の本体（トークン列を直接受け取る版）。
/// `content` 全体を対象にする [`declares_pub_fn`] とは別に、
/// [`var_impl_block_bodies`] が抜き出した impl ブロック本体トークン列
/// （部分スライス）に対しても同じ判定ロジックを適用できるようにする
/// ため分離した（codex-review 追加 P1 指摘・PR #2212: `Var` への
/// inherent／trait impl はクレート内の任意ファイルに書けるため、単一
/// ファイル全体ではなく impl ブロック単位で走査する必要がある）。
fn tokens_declare_pub_fn(tokens: &[String], fn_name: &str) -> bool {
    for (i, token) in tokens.iter().enumerate() {
        if token != "pub" {
            continue;
        }
        let j = skip_fn_declaration_qualifiers(tokens, i + 1);
        if tokens.get(j).map(String::as_str) == Some("fn")
            && tokens.get(j + 1).map(String::as_str) == Some(fn_name)
            && matches!(tokens.get(j + 2).map(String::as_str), Some("(") | Some("<"))
        {
            return true;
        }
    }
    false
}

/// `tokens`（コメント除去済み想定のファイル全体トークン列）から、
/// `Var` 型に対する impl ブロック（`impl<...> Var { ... }` の inherent
/// impl・`impl<...> Trait for Var { ... }` の trait impl の両方を含む）
/// の本体トークン列をすべて抜き出す（codex-review 追加 P1 指摘・PR
/// #2212: `var_rs_does_not_declare_pub_fn_custom` は `src/var.rs` のみを
/// 走査していたため、`Var` への inherent impl を同クレート内の別ファイル
/// （新規 module 等）に追加すると facade が再エクスポートする `Var` から
/// 未承認の `Var::custom`／`Var::add_custom` へ到達できる一方、この否定
/// ガードは通過してしまっていた）。
///
/// 判定手順:
/// 1. `impl` トークンを見つけたら、直後の generic parameter リスト
///    （`impl<T: Trait<U>> ...` 等）を山括弧の深さ追跡で読み飛ばす。
///    これにより `impl<T: Trait<Var>> Foo` のような無関係な型パラメータ
///    境界に現れる `Var` を、impl 対象そのものと誤認しない。
/// 2. 続くヘッダー（型パス・`for`・`where` 節等）を開始 `{` まで走査し、
///    独立したトークンとして `Var` が現れるかを記録する（`where` 節中の
///    `Var` 出現は許容——過剰検出側に倒す fail-closed 方針。本関数は
///    「否定ガードの検出漏れを防ぐ」ことが目的であり、対象を広げすぎて
///    body が空振りするだけなら実害はない）。
/// 3. 開始 `{` から対応する `}` までを中括弧の深さ追跡で本体として
///    抜き出す（`extract_trait_body` と同じ方式）。**`tokens` は呼び
///    出し側で `strip_comments` → `strip_string_literal_contents` 済み
///    の content をトークン化したものであることを前提とする**——
///    文字列・char リテラル中の `{`/`}` を素通しした生トークン列を
///    渡すと、リテラル内の中括弧を実際の中括弧として誤カウントして
///    しまう（理由は [`strip_string_literal_contents`] のドキュメン
///    テーションコメント参照）。
/// 4. ヘッダーに `Var` が含まれていた場合のみ本体トークン列を返す。
fn var_impl_block_bodies(tokens: &[String]) -> Vec<Vec<String>> {
    let mut bodies = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != "impl" {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        if tokens.get(j).map(String::as_str) == Some("<") {
            let mut depth = 0i32;
            while j < tokens.len() {
                match tokens[j].as_str() {
                    "<" => depth += 1,
                    ">" => depth -= 1,
                    _ => {}
                }
                j += 1;
                if depth == 0 {
                    break;
                }
            }
        }
        let mut header_has_var = false;
        while j < tokens.len() && tokens[j] != "{" {
            if tokens[j] == "Var" {
                header_has_var = true;
            }
            j += 1;
        }
        let Some("{") = tokens.get(j).map(String::as_str) else {
            // 対応する impl 本体（`{`）が見つからない不正な入力は
            // これ以上走査を続けられないため打ち切る（fail-closed:
            // クラッシュや無限ループより安全側の「検査終了」を選ぶ）。
            break;
        };
        let body_start = j + 1;
        let mut depth = 1i32;
        let mut k = body_start;
        while k < tokens.len() && depth > 0 {
            match tokens[k].as_str() {
                "{" => depth += 1,
                "}" => depth -= 1,
                _ => {}
            }
            k += 1;
        }
        let body_end = if depth == 0 { k - 1 } else { tokens.len() };
        if header_has_var {
            bodies.push(tokens[body_start..body_end].to_vec());
        }
        // `body_end + 1`（本体を丸ごと読み飛ばす）にはしない: Rust では
        // 関数本体の中に別の impl ブロックをネストできる（`impl Other {
        // fn g() { impl Var { pub fn custom(&self) {} } } }`）ため、
        // 本体を丸ごとスキップすると内側の `impl Var` を見逃してしまう
        // （codex-review 追加 P1 指摘・PR #2212）。`body_start` から
        // 再走査することで、この外側ループ自身がネストした `impl` トー
        // クンを別途検出する。
        i = body_start;
    }
    bodies
}

/// `line` 先頭の可視性修飾（`pub`／`pub(crate)`／`pub(super)`／
/// `pub(in ...)` 等）を読み飛ばした残りを返す（可視性修飾が無ければ
/// そのまま `trim_start()` した文字列を返す）。`pub type Tensor = ...;`
/// のような可視性付き宣言を `type ` 始まりの行と同一視して判定するために
/// 使う（codex-review 指摘・PR #2212 その 3。旧実装は `type ` 始まりの
/// 行のみを対象としており `pub type` を取りこぼしていた）。
fn strip_visibility_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();
    let Some(after_pub) = trimmed.strip_prefix("pub") else {
        return trimmed;
    };
    let after_pub = after_pub.trim_start();
    if let Some(after_paren_open) = after_pub.strip_prefix('(')
        && let Some(close_rel) = after_paren_open.find(')')
    {
        return after_paren_open[close_rel + 1..].trim_start();
    }
    after_pub
}

/// `statement` 先頭に連続する外部属性（`#[...]`。`#[derive(Debug)]`・
/// `#[allow(unused_imports)]` 等）を、`[`／`]` の深さ追跡（ネスト対応。
/// `#[cfg(feature = "x")]` のような属性内の文字列リテラル中の `[`／`]`
/// は非対応——`custom.rs` に該当パターンが無いことを前提とする単純な
/// 実装方針は `split_top_level_statements` と同じ）で読み飛ばし、
/// 属性の後に残る item 本体（可視性修飾・`use`／`type` キーワード等）
/// の先頭を返す（codex-review 指摘・PR #2212 その 13）:
/// `split_top_level_statements` は属性とその直後の item を 1 ステート
/// メントとして結合するため、`#[allow(unused_imports)] use crate::
/// BackendOps as Ops;` のように外部属性が先頭に付くと、後続の
/// `strip_visibility_prefix`／`strip_keyword_prefix` が期待する
/// 「可視性修飾または `use`／`type` キーワードで始まる」という前提が
/// 崩れ、alias 検出・type エイリアス検査の両ループを素通りしてしまう。
fn strip_leading_attributes(statement: &str) -> &str {
    let mut rest = statement.trim_start();
    while let Some(after_hash) = rest.strip_prefix('#') {
        let after_hash = after_hash.trim_start();
        let Some(after_bracket) = after_hash.strip_prefix('[') else {
            break;
        };
        let mut depth = 1i32;
        let mut end = None;
        for (idx, ch) in after_bracket.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(idx);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            // 対応する `]` が見つからない不正な属性は読み飛ばさず、
            // そのまま残りをキーワード判定へ渡す（fail-closed: 判定に
            // 失敗させて検出漏れを起こすより、後続処理へ委ねる）。
            break;
        };
        rest = after_bracket[end + 1..].trim_start();
    }
    rest
}

/// `crates/autodiff/src/custom.rs` の `CustomFunction` trait が
/// `BackendOps`／`Tape`／`Var`／`Device`／`NodeId`（グラフ構築・デバイス
/// 結線に関わる型）を一切シグネチャへ含まないことを固定する（イシュー
/// #2064 §12.5 (b) 第 3 項）。`docs/autodiff-custom-function-decision.md`
/// §12.4「契約」が定める「`forward`／`backward` は host `Tensor<f32>`
/// のみを受け渡す（`BackendOps` 非露出）」「`'static` 境界により `&Tape`・
/// `Var<'t>` を捕捉できない」という設計の構造的裏付けを、trait 定義
/// そのものの grep で確認する（実装 `CustomFn`〈`pub(crate)` newtype。
/// `Arc<dyn CustomFunction>` を保持〉やモジュール doc コメントは走査
/// 対象に含めない——trait 定義の外側に `Tape` 等の語が現れても本テスト
/// の関心事ではないため、`extract_trait_body` で trait 本体のみへ絞る）。
/// trait 本体自身のドキュメンテーションコメント（`///`。§12.4「契約」の
/// 説明文が `Tape`・`Var` 等の語を含む）はシグネチャではないため、
/// `//` 行を除去してからシグネチャのみを検査する。
///
/// **ヘッダー（supertrait 境界）は専用の `extract_trait_header` で
/// 独立判定する**（codex-review 指摘・PR #2212 その 1）: ヘッダー＋
/// メソッド本体全体（`extract_trait_body`）に対する `contains` 判定では、
/// supertrait から `Send`／`Sync`／`'static` を落としても同じ文字列が
/// メソッドシグネチャ側に残っていれば検査を素通りしてしまうため、
/// 必須境界の有無判定はヘッダー行のみに限定する。
///
/// **禁止識別子の denylist に加えて allowlist 判定も行う**（codex-review
/// 指摘・PR #2212 その 2）: 固定識別子の文字列検索（denylist）のみでは
/// `use ... as Ops` のような型エイリアス経由で禁止型を別名で混入させて
/// もこのテストで検出できない。import／alias 解決は行わず、シグネチャに
/// 現れる識別子トークンを現行シグネチャの実際のトークン集合そのもの
/// （`ALLOWED_SIGNATURE_TOKENS`）と突き合わせる fail-closed 方式で
/// 補完し、未知の識別子（エイリアス経由の混入を含む）を検出する。
/// さらに、`use ... as ...`（エイリアス import）・`type ... = <禁止型>;`
/// （型エイリアスによる禁止型の再エクスポート）が `custom.rs` に
/// 存在しないことも構造的に固定し、別名混入の経路自体を塞ぐ。
#[test]
fn custom_function_trait_signatures_are_host_tensor_only() {
    let custom_rs = autodiff_crate_root().join("src/custom.rs");
    let content = read_to_string_or_panic(&custom_rs);
    // 行コメント・ブロックコメント（ネスト対応）を除去した全文（`content`
    // 中の位置に依存する走査は以降すべてこの `no_comments` を対象にする。
    // codex-review 指摘・PR #2212 その 6: ブロックコメント中の偽トレイト
    // 定義を `extract_trait_body`／`extract_supertrait_bound_tokens` が
    // 誤って最初の一致として抜き出す迂回を塞ぐ）。
    let no_comments = strip_comments(&content);

    let supertrait_bounds = extract_supertrait_bound_tokens(&no_comments, "CustomFunction");
    assert!(
        !supertrait_bounds.is_empty(),
        "src/custom.rs から `pub trait CustomFunction` の supertrait 境界（`:` から\
         `where`／`{{` 手前まで）を抽出できなかった（テスト自体が検査対象を見失っている、\
         または supertrait 境界自体が存在しない。ファイル構成が変わっていないか確認）"
    );
    for required_bound in ["Send", "Sync", "'static"] {
        assert!(
            supertrait_bounds
                .iter()
                .any(|bound| bound == required_bound),
            "CustomFunction trait の supertrait 境界に必須の {required_bound} が\
             見つからない（ジェネリクス仮引数名・where 節側の同名トークンでは代替できない\
             構造的判定。§12.4「'static 境界により &Tape・Var<'t> を捕捉できない」契約違反）"
        );
    }

    let trait_body = extract_trait_body(&no_comments, "CustomFunction");
    assert!(
        !trait_body.is_empty(),
        "src/custom.rs から `pub trait CustomFunction` 本体を抽出できなかった\
         （テスト自体が検査対象を見失っている。ファイル構成が変わっていないか確認）"
    );
    // `trait_body` は `no_comments`（コメント除去済み）から抽出済みのため
    // 行単位の `//` 再フィルタは不要。
    let signatures_only: String = trait_body;

    // denylist 判定（既存。defense in depth として allowlist 判定と併用）。
    const FORBIDDEN_IDENTIFIERS: &[&str] = &["BackendOps", "Tape", "Var", "Device", "NodeId"];
    for forbidden in FORBIDDEN_IDENTIFIERS {
        assert!(
            !contains_identifier(&signatures_only, forbidden),
            "CustomFunction trait のシグネチャに {forbidden} が含まれている\
             （§12.4 の「host Tensor<f32> のみを受け渡す」契約違反の疑い）"
        );
    }

    // allowlist 判定: 現行シグネチャの識別子トークン集合そのもの。
    // シグネチャ変更時は本配列の更新も 1 行差分としてレビュー対象になる。
    const ALLOWED_SIGNATURE_TOKENS: &[&str] = &[
        "'static",
        "AutodiffError",
        "CustomFunction",
        "Option",
        "Result",
        "Send",
        "Sync",
        "Tensor",
        "Vec",
        "backward",
        "bool",
        "f32",
        "fn",
        "forward",
        "input_shapes",
        "inputs",
        "name",
        "out_value",
        "output_shape",
        "pub",
        "requires_grad",
        "self",
        "str",
        "trait",
        "upstream",
        "usize",
    ];
    for token in extract_identifier_tokens(&signatures_only) {
        assert!(
            ALLOWED_SIGNATURE_TOKENS.contains(&token.as_str()),
            "CustomFunction trait のシグネチャに allowlist 外の識別子 `{token}` が含まれている\
             （型エイリアス経由の禁止型混入の疑い。意図的なシグネチャ変更であれば\
             ALLOWED_SIGNATURE_TOKENS の更新漏れなのでレビューのうえ追加すること）"
        );
    }

    // 別名混入の経路自体を構造的に塞ぐ（allowlist 判定は現行シグネチャに
    // 対する静的な検査であり、将来 `use ... as` で禁止型を別名 import
    // されると allowlist 側の更新と一緒に通ってしまう。よってファイル
    // 全体を対象に、エイリアス import 自体を禁止する）。
    //
    // 文単位（波括弧の深さ 0 の `;` まで）でステートメントを再構成して
    // から判定する（codex-review 指摘・PR #2212 その 3: 行単位
    // `trimmed.starts_with("use ")`／`starts_with("type ")` は
    // `use crate::{\n    BackendOps as Tensor,\n};` のような複数行
    // 波括弧 import や `pub type Tensor = BackendOps;` のような可視性
    // 修飾付き宣言を検出できなかった。`split_top_level_statements` は
    // 波括弧のペアをまたいで `;` まで 1 文として保持するため、複数行
    // `use` もひと続きの文字列として判定できる）。
    //
    // `use` 文自体の判定は `strip_visibility_prefix` を先に適用する
    // （codex-review 指摘・PR #2212 その 4／Bugbot 指摘: 旧実装は
    // `statement.starts_with("use ")` のみを見ており、`pub use`・
    // `pub(crate) use` のような可視性修飾付き use 文を素通りしていた。
    // `type ` 判定と同じ「可視性修飾を読み飛ばしてから判定する」方針に
    // 揃える）。
    //
    // alias 検出は `.contains(" as ")` の固定文字列一致ではなく、
    // `extract_identifier_tokens` によるトークン化を用いる
    // （codex-review 指摘・PR #2212 その 4: タブ区切り
    // `BackendOps\tas\tTensor` や `BackendOps as/* alias */Tensor`
    // のようにコメントが `as` 直後へ隙間なく挟まるケースは、前後に
    // 半角スペースを要求する `" as "` 一致では見逃す。`as` は Rust の
    // 予約語で識別子として現れないため、トークン列に独立した `as`
    // トークンが 1 つでも含まれていれば alias import とみなせる。
    // トークナイザは英数字／`_`／先頭の `'` のみを識別子境界とするため、
    // 空白種別（スペース・タブ・改行）やコメント区切り文字（`/*`・`*/`）
    // の違いに影響されない）。
    //
    // `use ` 判定は固定スペース 1 個の `starts_with` ではなく
    // `strip_keyword_prefix`（トークン境界での識別子一致）を使う
    // （codex-review 指摘・PR #2212 その 7: `use\ncrate::{...}` のように
    // `use` の直後に改行が来る有効な Rust 記法を `starts_with("use ")`
    // は見逃す）。
    for statement in split_top_level_statements(&no_comments) {
        let after_attributes = strip_leading_attributes(&statement);
        let after_visibility = strip_visibility_prefix(after_attributes);
        if strip_keyword_prefix(after_visibility, "use").is_some() {
            let has_as_token = extract_identifier_tokens(after_visibility)
                .iter()
                .any(|token| token == "as");
            assert!(
                !has_as_token,
                "src/custom.rs の use 文にエイリアス（`use ... as ...`）が含まれている: \
                 {statement}（複数行の波括弧 import・可視性修飾・タブ区切り・コメント挟み込み\
                 を含む。禁止型を別名で混入させる経路になりうるため、本ファイルではエイリアス\
                 import を使わない設計とする）"
            );
        }
        // `pub`／`pub(crate)` 等の可視性修飾を読み飛ばしてから `type` を
        // 判定する（旧実装は `type ` 始まりの行しか見ておらず `pub type`
        // を取りこぼしていた。さらに `strip_keyword_prefix` により
        // `type\nTensor = ...` のような改行を挟んだ宣言も見逃さない）。
        if strip_keyword_prefix(after_visibility, "type").is_some() {
            for forbidden in FORBIDDEN_IDENTIFIERS {
                assert!(
                    !contains_identifier(after_visibility, forbidden),
                    "src/custom.rs の type エイリアス宣言が禁止型 {forbidden} を参照している: \
                     {statement}"
                );
            }
        }
    }

    // ローカル型定義による同名別型の混入を遮断する（codex-review 指摘・
    // PR #2212 その 5）: `ALLOWED_SIGNATURE_TOKENS` は名前ベースの許可
    // のため、`use ... as ...` によるエイリアスを禁止しても
    // `pub struct Tensor(BackendOps);` のように custom.rs 内で `Tensor`
    // という名前の別型を直接定義されると、シグネチャ上は allowlist を
    // 素通りしたまま実質的に禁止型（`BackendOps`）を混入できてしまう。
    // `strip_prefix` によるステートメント先頭一致（旧実装）は
    // `#[derive(Debug)]` 等の属性行や `struct`／`Tensor` 間の改行がある
    // と検出漏れになるため、コメント除去済みの全文をトークン化し
    // `struct`／`enum`／`type` トークンの直後に `Tensor` トークンが続く
    // 箇所を走査する（空白・改行・属性行の位置に依存しない）。
    //
    // `type` も対象に含める（codex-review 指摘・PR #2212 その 8）:
    // 直前の type エイリアス検査（上記ループ内）は右辺が
    // `FORBIDDEN_IDENTIFIERS` の固定 denylist に一致する場合のみ拒否
    // するため、`type Tensor = SomeOtherType;` のように denylist に
    // 載っていない任意の型へすり替える宣言を見逃す。`Tensor` という
    // 名前のローカル型エイリアス自体を構造的に禁止すれば、右辺の型が
    // 何であっても（denylist の有無に関係なく）「シグネチャの `Tensor`
    // は必ず canonical import 由来」という不変条件を保てる。
    let all_tokens = tokenize_including_punctuation(&no_comments);
    let local_tensor_type_declared = all_tokens
        .windows(2)
        .any(|w| (w[0] == "struct" || w[0] == "enum" || w[0] == "type") && w[1] == "Tensor");
    assert!(
        !local_tensor_type_declared,
        "src/custom.rs に Tensor という名前のローカル型定義（struct／enum／type エイリアス）\
         が見つかった（tensor_core::Tensor と同名の別型でラップして禁止型を混入させる経路に\
         なりうるため、本ファイルでは定義しない設計とする）"
    );

    // シグネチャの `Tensor` トークンが指す型を一意に固定する（import 元
    // の完全パス検査。codex-review 指摘・PR #2212 その 5）: `Tensor` を
    // import する `use` 文が 1 つ以上あり、そのすべてが
    // `fandhe_ai_tensor_core::Tensor` という完全パスと一致することを
    // 検査する（`crate::Tensor` や別クレートの同名型を経由した再
    // エクスポートでは一致しない）。判定ロジックは
    // `find_canonical_tensor_import` に切り出し、回帰テスト
    // （`architecture_boundary_bypass_scenarios_are_detected`）から
    // 任意の合成入力に対しても同じ経路で検証できるようにする。
    match find_canonical_tensor_import(&no_comments) {
        Ok(true) => {}
        Ok(false) => panic!(
            "src/custom.rs に fandhe_ai_tensor_core::Tensor の import が見つからない\
             （シグネチャの Tensor トークンが指す型を一意に固定できない）"
        ),
        Err(message) => panic!("src/custom.rs: {message}"),
    }
}

/// `content` のトップレベル `use` 文を走査し、`Tensor` を import する
/// 文がすべて無条件（`cfg`／`cfg_attr` 属性が付いていない）かつ
/// `fandhe_ai_tensor_core::Tensor` という完全パスと一致することを検査
/// する（`custom_function_trait_signatures_are_host_tensor_only` と、
/// その迂回シナリオ回帰テスト `architecture_boundary_bypass_scenarios_
/// are_detected` の双方から呼ぶ共通ロジック）。
///
/// - 条件付き Tensor import（`cfg`／`cfg_attr` のいずれかが付いた
///   `use ... Tensor;`）が 1 つでも見つかった場合は `Err` を返す
///   （codex-review 指摘・PR #2212 その 15）: パスが
///   `fandhe_ai_tensor_core::Tensor` と一致していても拒否する。旧実装は
///   「`#[cfg(test)]` 配下の import だけを canonical 判定から除外する」
///   という設計だったが、`statement_has_cfg_test_attribute` の判定単位
///   （ステートメント全体のトークン列に `cfg(test)` という並びが**任意の
///   位置**に現れるか）では (1) 常に無効な `#[cfg(any())]` が付いた
///   canonical import を素通しし、(2) `#[cfg_attr(any(), cfg(test))]`
///   のように属性の**引数内**に `cfg(test)` を含むだけの別属性まで
///   「cfg(test) 属性付き」と誤判定してパス不一致の検証ごと迂回できて
///   いた。本関数は「トップレベルの Tensor import には `cfg`／
///   `cfg_attr` のいずれの属性も一切許さない」という、より単純で
///   構文的に検証可能な不変条件へ置き換える（テスト専用 import が
///   必要な場合は `#[cfg(test)] mod` 内に置く前提。`mod` ブロックは
///   `split_top_level_statements` が 1 ステートメントとして丸ごと
///   グループ化し、`strip_keyword_prefix(_, "use")` が `mod` 始まりの
///   文には一致しないため、この検査の走査対象に元から入らない）。
/// - 無条件かつ完全パス不一致の import が見つかった場合も `Err` を返す。
/// - 無条件かつ完全パス一致の import が 1 つ以上見つかれば `Ok(true)`、
///   1 つも見つからなければ `Ok(false)` を返す。
fn find_canonical_tensor_import(content: &str) -> Result<bool, String> {
    let mut canonical_found = false;
    for statement in split_top_level_statements(content) {
        let after_attributes = strip_leading_attributes(&statement);
        let after_visibility = strip_visibility_prefix(after_attributes);
        let Some(use_body) = strip_keyword_prefix(after_visibility, "use") else {
            continue;
        };
        let use_tokens = extract_identifier_tokens(use_body);
        if use_tokens.last().map(String::as_str) != Some("Tensor") {
            continue;
        }
        if statement_has_conditional_attribute(&statement) {
            return Err(format!(
                "Tensor を import する use 文に cfg／cfg_attr 等の条件付き属性が付いている\
                 （トップレベルの Tensor import は無条件でなければならない。テスト専用\
                 import が必要な場合は #[cfg(test)] mod 内に置くこと）: {statement}"
            ));
        }
        if use_tokens != vec!["fandhe_ai_tensor_core".to_string(), "Tensor".to_string()] {
            return Err(format!(
                "use 文が Tensor を import しているが完全パスが fandhe_ai_tensor_core::Tensor\
                 と一致しない: {statement}"
            ));
        }
        canonical_found = true;
    }
    Ok(canonical_found)
}

/// `crates/autodiff/src` 配下のいずれのファイルにも、`Var` への impl
/// ブロック（inherent／trait 双方）が `pub fn custom(`／`pub fn custom<`・
/// `pub fn add_custom(`／`pub fn add_custom<` 宣言を持たないことを固定
/// する（イシュー #2064 §12.5 (b) 第 3 項）。facade は `Var` を型ごと
/// 再エクスポートしているため、`Var::custom`（または `add_custom`）が
/// 生えると `Tape::custom` の facade 転送メソッド不在ガード
/// （`crates/facade/tests/api_surface.rs::
/// facade_tape_does_not_expose_custom_forwarding_method`）を経由せずに
/// 承認 (b) 前の到達経路が生まれてしまう（`docs/autodiff-custom-
/// function-decision.md` §12.4「入口」の設計根拠）。
///
/// **`src/var.rs` のみを走査する旧実装の死角**（codex-review 追加 P1
/// 指摘・PR #2212）: inherent `impl Var { ... }` は Rust の言語仕様上
/// 同一クレート内のどのファイルにも書ける（`var.rs` に定義する必要が
/// ない）。旧実装は `src/var.rs` の内容にしか `declares_pub_fn` を
/// 適用していなかったため、たとえば新規モジュール（`src/foo.rs`）に
/// `impl Var { pub fn custom(...) { ... } }` を追加しても本ガードは
/// 素通りしてしまい、facade 再エクスポート経由で未承認の入口へ到達
/// 可能になっていた。本実装は `visit_rs_files` で `src/` 全体を再帰
/// 走査し、各ファイルを `strip_comments` → `strip_string_literal_
/// contents`（文字列リテラル内の `{`／`}` による中括弧深さ追跡の誤り
/// を防ぐ）した上で [`var_impl_block_bodies`] により「ヘッダーに
/// `Var` を含む impl ブロック」の本体のみを抜き出し、その本体トークン
/// 列に対して [`tokens_declare_pub_fn`] を適用する（`impl Other { pub
/// fn custom() {} }` のような無関係な型への同名メソッドは検出対象外の
/// まま。ネストした impl ブロック——関数本体の中に書かれた `impl Var`
/// 等——も [`var_impl_block_bodies`] が取りこぼさない）。
#[test]
fn autodiff_src_does_not_declare_pub_fn_custom_on_var() {
    let src_dir = autodiff_crate_root().join("src");
    let mut violations: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        // 行コメント（`//`）のみを除去する旧実装は `pub/* separator */ fn
        // custom` のようなブロックコメントを挟んだ宣言を素通りさせていた
        // （codex-review 指摘・PR #2212 その 12）。`strip_comments`
        // （行・ブロック両対応、トークン境界を保つ）を使う。
        let no_comments = strip_comments(content);
        // 文字列リテラル内の `{`／`}` が中括弧深さ追跡を誤らせるバイパス
        // （codex-review 追加 P1 指摘・PR #2212。`strip_string_literal_
        // contents` のドキュメンテーションコメント参照）を防ぐため、
        // `var_impl_block_bodies` へ渡す前に文字列リテラルの中身を
        // 空白化する。
        let no_strings = strip_string_literal_contents(&no_comments);
        let tokens = tokenize_including_punctuation(&no_strings);
        for body in var_impl_block_bodies(&tokens) {
            for fn_name in ["custom", "add_custom"] {
                if tokens_declare_pub_fn(&body, fn_name) {
                    violations.push(format!(
                        "{}: impl Var に pub fn {fn_name} 宣言",
                        path.display()
                    ));
                }
            }
        }
    });
    assert!(
        violations.is_empty(),
        "crates/autodiff/src 配下の impl Var ブロックに未承認の pub fn custom／\
         add_custom 宣言が見つかった（§12.5 (b) 未承認のまま Var 経由の到達口を\
         設けてしまっている。`pub`・`fn`・関数名の間に改行・ブロックコメントを\
         挟んだ宣言も、`src/var.rs` 以外のファイルに書かれた impl ブロックも\
         検出する）: {violations:?}"
    );
}

/// [`autodiff_src_does_not_declare_pub_fn_custom_on_var`] のブロックコメント経由の
/// バイパス（codex-review 指摘・PR #2212 その 12）を、実際に
/// `strip_comments` + `declares_pub_fn` の組み合わせで検出できることを
/// 固定する回帰テスト。`src/var.rs` 本体を変更せずに検証するため、
/// 同じパターンの合成入力に対して直接アサートする。
#[test]
fn declares_pub_fn_detects_declaration_split_by_block_comment() {
    let forged = "pub/* separator */fn custom(&self) {}";
    let no_comments = strip_comments(forged);
    assert!(
        declares_pub_fn(&no_comments, "custom"),
        "ブロックコメントで pub と fn を分断した宣言を検出できていない: {no_comments}"
    );
}

/// [`autodiff_src_does_not_declare_pub_fn_custom_on_var`] が使う
/// [`var_impl_block_bodies`] が、`src/var.rs` 以外のファイル相当の合成
/// 入力（同一クレート内の別モジュール・別 impl ブロック）に対しても
/// 正しく検出・除外できることを固定する回帰テスト（codex-review 追加
/// P1 指摘・PR #2212: 単一ファイル走査の死角を塞いだことの検証）。
#[test]
fn var_impl_block_bodies_detects_declaration_in_other_module_file() {
    // 1) `src/var.rs` とは別ファイル相当の `mod extra { impl Var { ... } }`
    //    に書かれた `pub fn custom` を検出できる（ファイル単位ではなく
    //    impl ブロック単位で走査しているため、module 宣言の有無に依らず
    //    トークン列上は同じ検出結果になる）。
    let forged_in_extra_module = "mod extra { impl Var { pub fn custom() {} } }";
    let tokens = tokenize_including_punctuation(&strip_comments(forged_in_extra_module));
    let bodies = var_impl_block_bodies(&tokens);
    assert_eq!(
        bodies.len(),
        1,
        "mod extra 内の impl Var ブロックを 1 件検出できていない: {bodies:?}"
    );
    assert!(
        tokens_declare_pub_fn(&bodies[0], "custom"),
        "mod extra 内の impl Var {{ pub fn custom() {{}} }} を検出できていない"
    );

    // 2) generic parameter・ライフタイム付き impl ヘッダー
    //    （`impl<'a> Var { ... }`）と、修飾子付き宣言（`pub extern "C"
    //    fn add_custom`）の組み合わせも検出できる。実運用の
    //    `autodiff_src_does_not_declare_pub_fn_custom_on_var` と同じ
    //    `strip_comments → strip_string_literal_contents → tokenize`
    //    パイプラインを通す（PR #2212 追加 P1 の是正時に自己レビューで
    //    判明: ABI 文字列リテラル `"C"` が `strip_string_literal_
    //    contents` を経由しても `skip_fn_declaration_qualifiers` の
    //    判定と整合することを固定する）。
    let forged_with_lifetime_and_abi = "impl<'a> Var { pub extern \"C\" fn add_custom() {} }";
    let no_comments = strip_comments(forged_with_lifetime_and_abi);
    let no_strings = strip_string_literal_contents(&no_comments);
    let tokens = tokenize_including_punctuation(&no_strings);
    let bodies = var_impl_block_bodies(&tokens);
    assert_eq!(
        bodies.len(),
        1,
        "impl<'a> Var ブロックを 1 件検出できていない: {bodies:?}"
    );
    assert!(
        tokens_declare_pub_fn(&bodies[0], "add_custom"),
        "impl<'a> Var {{ pub extern \"C\" fn add_custom() {{}} }} を検出できていない"
    );

    // 3) 無関係な型への同名メソッドは誤検出しない（`impl Other { ... }`
    //    はヘッダーに `Var` トークンを含まないため対象外のまま）。
    let unrelated_impl = "impl Other { pub fn custom() {} }";
    let tokens = tokenize_including_punctuation(&strip_comments(unrelated_impl));
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies.is_empty(),
        "impl Other への pub fn custom を Var 向けと誤検出した: {bodies:?}"
    );

    // 4) 関数本体の中にネストされた `impl Var` ブロック（PR #2212 追加
    //    P1 の是正時に自己レビューで判明。`impl Other { fn g() { impl
    //    Var { pub fn custom(&self) {} } } }`）
    //    も見逃さない（本体を丸ごと読み飛ばすと外側 `impl Other` の
    //    走査終了と同時に内側 `impl Var` も飛ばしてしまうため、
    //    `var_impl_block_bodies` は本体開始位置から再走査する）。
    let nested_impl = "impl Other { fn g() { impl Var { pub fn custom(&self) {} } } }";
    let tokens = tokenize_including_punctuation(&strip_comments(nested_impl));
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "関数本体にネストされた impl Var 内の pub fn custom を検出できていない: {bodies:?}"
    );

    // 5) 文字列リテラル内の `}` による中括弧深さ追跡の誤り（PR #2212
    //    追加 P1 の是正時に自己レビューで判明）: `strip_string_literal_
    //    contents` を適用せずに
    //    `var_impl_block_bodies` へ渡すと、リテラル内の `}` を実際の
    //    中括弧として誤カウントし、本体末尾の `pub fn custom` を本体の
    //    外へ追い出してしまう。`strip_string_literal_contents` を通す
    //    ことで正しく検出できることを固定する。
    let forged_with_brace_in_string =
        "impl Var { fn helper() { let _ = \"}\"; } pub fn custom(&self) {} }";
    let no_comments = strip_comments(forged_with_brace_in_string);
    let no_strings = strip_string_literal_contents(&no_comments);
    let tokens = tokenize_including_punctuation(&no_strings);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "文字列リテラル内の }} を実際の中括弧と誤カウントし、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );
    // 対照実験: `strip_string_literal_contents` を適用しない場合は
    // このバイパスが実際に成立する（誤って検出漏れになる）ことも
    // 併せて固定し、上記の修正が実際に効いていることを裏付ける。
    let tokens_without_fix = tokenize_including_punctuation(&no_comments);
    let bodies_without_fix = var_impl_block_bodies(&tokens_without_fix);
    assert!(
        !bodies_without_fix
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "strip_string_literal_contents 抜きでも検出できてしまっている\
         （回帰テストの前提が崩れている）: {bodies_without_fix:?}"
    );

    // 6) char リテラル中の `"`（`'"'`）による文字列開始の誤検出
    //    （PR #2212 追加 P1 の是正時に自己レビューで判明）: char
    //    リテラルを判別せず `'` を無視すると、1 つめの `'"'` を
    //    「文字列の開始」と誤認し、離れた場所にある
    //    2 つめの `'"'` までを丸ごと文字列として空白化してしまい、
    //    その間に挟まれた `pub fn custom` が消えてしまう。
    let forged_with_char_literal_quote =
        "impl Var { fn q() -> char { '\"' } pub fn custom(&self) {} fn r() -> char { '\"' } }";
    let no_comments = strip_comments(forged_with_char_literal_quote);
    let no_strings = strip_string_literal_contents(&no_comments);
    let tokens = tokenize_including_punctuation(&no_strings);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "char リテラル '\"' を文字列の開始と誤認識し、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );

    // 7) char リテラル `'}'`（`'{'`）による中括弧深さ追跡の誤り
    //    （PR #2212 追加 P1 の是正時に自己レビューで判明）: char
    //    リテラルの中身をそのまま素通しするとトークン化後に単独の
    //    `}` トークンとして現れ、impl 本体の
    //    終端を実際より手前で確定してしまい、本体末尾の `pub fn
    //    custom` を本体の外へ追い出してしまう。
    let forged_with_char_literal_brace =
        "impl Var { fn h() -> char { '}' } pub fn custom(&self) {} }";
    let no_comments = strip_comments(forged_with_char_literal_brace);
    let no_strings = strip_string_literal_contents(&no_comments);
    let tokens = tokenize_including_punctuation(&no_strings);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "char リテラル '}}' を実際の中括弧と誤カウントし、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );
}

/// codex-review 指摘（PR #2212 その 3・その 4）・Bugbot 指摘（PR #2212）
/// が挙げたバイパスシナリオ（ジェネリクス／where 節側の偽装トークン・
/// 複数行または可視性付き alias・タブ区切りやコメント挟み込みの alias・
/// `pub use`／`pub(crate) use` 等の可視性修飾付き alias import）を、
/// 実際に新実装が検出できることを固定する回帰テスト。
#[test]
fn architecture_boundary_bypass_scenarios_are_detected() {
    // 1) supertrait 自体を削除し、ジェネリクス仮引数名・where 節側に
    //    Send／Sync／'static を偽装したヘッダーからは境界が抽出されない
    //    （＝呼び出し側で「必須境界が見つからない」として fail する）。
    let forged_header =
        "pub trait CustomFunction<Send, Sync, T> where T: 'static {\n    fn forward(&self);\n}\n";
    let bounds = extract_supertrait_bound_tokens(forged_header, "CustomFunction");
    assert!(
        bounds.is_empty(),
        "偽装ヘッダーから supertrait 境界が抽出されてしまった: {bounds:?}"
    );

    // 2) 複数行の波括弧 use エイリアスが 1 ステートメントとして結合され、
    //    トークン化した識別子列に `as` トークンが含まれる。
    let forged_use = "use crate::{\n    BackendOps as Tensor,\n};\n";
    let statements = split_top_level_statements(forged_use);
    assert_eq!(
        statements.len(),
        1,
        "複数行 use 文が 1 文として結合されていない: {statements:?}"
    );
    assert!(statements[0].starts_with("use "));
    assert!(
        extract_identifier_tokens(&statements[0])
            .iter()
            .any(|token| token == "as")
    );

    // 3) 可視性修飾付き type alias（`pub type ...`）も `type ` 始まりと
    //    同一視して検出される。
    let forged_type = "pub type Tensor = BackendOps;\n";
    for statement in split_top_level_statements(forged_type) {
        let after_vis = strip_visibility_prefix(&statement);
        assert!(
            after_vis.starts_with("type "),
            "pub type を検出できていない: {after_vis}"
        );
        assert!(contains_identifier(after_vis, "BackendOps"));
    }

    // 4) タブ区切り alias（codex-review 指摘・PR #2212 その 4）:
    //    `" as "`（前後半角スペース固定）の文字列一致では見逃すが、
    //    トークン化すれば `as` が独立した識別子として抽出される。
    let forged_use_tab = "use crate::BackendOps\tas\tTensor;\n";
    for statement in split_top_level_statements(forged_use_tab) {
        let after_vis = strip_visibility_prefix(&statement);
        assert!(after_vis.starts_with("use "));
        assert!(
            !after_vis.contains(" as "),
            "このシナリオは前後スペース固定の文字列一致では検出できないことの前提確認"
        );
        assert!(
            extract_identifier_tokens(after_vis)
                .iter()
                .any(|token| token == "as"),
            "タブ区切り alias の as トークンを検出できていない: {after_vis}"
        );
    }

    // 5) コメント挟み込み alias（codex-review 指摘・PR #2212 その 4）:
    //    `as` の直後に空白なしでブロックコメントが続くと `" as "` 一致
    //    では見逃すが、トークン化すれば影響されない。
    let forged_use_comment = "use crate::BackendOps as/* alias */Tensor;\n";
    for statement in split_top_level_statements(forged_use_comment) {
        let after_vis = strip_visibility_prefix(&statement);
        assert!(after_vis.starts_with("use "));
        assert!(
            !after_vis.contains(" as "),
            "このシナリオは前後スペース固定の文字列一致では検出できないことの前提確認"
        );
        assert!(
            extract_identifier_tokens(after_vis)
                .iter()
                .any(|token| token == "as"),
            "コメント挟み込み alias の as トークンを検出できていない: {after_vis}"
        );
    }

    // 6) 可視性修飾付き alias import（Bugbot 指摘。`pub use`・
    //    `pub(crate) use` は旧実装の `statement.starts_with("use ")`
    //    判定を素通りしていた）。`strip_visibility_prefix` を先に適用
    //    すればどちらも `use ` 始まりとして検出対象になる。
    for forged_pub_use in [
        "pub use crate::BackendOps as Tensor;\n",
        "pub(crate) use crate::BackendOps as Tensor;\n",
    ] {
        for statement in split_top_level_statements(forged_pub_use) {
            let after_vis = strip_visibility_prefix(&statement);
            assert!(
                after_vis.starts_with("use "),
                "可視性修飾付き use 文を検出できていない: {statement} -> {after_vis}"
            );
            assert!(
                extract_identifier_tokens(after_vis)
                    .iter()
                    .any(|token| token == "as"),
                "可視性修飾付き alias import の as トークンを検出できていない: {after_vis}"
            );
        }
    }

    // 7) 改行を挟んだ `pub fn custom(` 宣言（codex-review 指摘・
    //    PR #2212 その 5）: 固定文字列一致 `contains("pub fn custom(")`
    //    は見逃すが、`declares_pub_fn` はトークン列の連続一致で検出する。
    assert!(
        declares_pub_fn("pub\nfn custom(&self) {}", "custom"),
        "改行を挟んだ pub fn custom( 宣言を検出できていない"
    );
    assert!(
        declares_pub_fn("pub\n    fn\ncustom<'t>(&self) {}", "custom"),
        "改行を挟んだジェネリクス付き pub fn custom< 宣言を検出できていない"
    );
    assert!(
        !declares_pub_fn("pub fn custom_foo(&self) {}", "custom"),
        "custom_foo のような無関係な関数名を誤検出してはならない"
    );
    assert!(
        !declares_pub_fn("pub(crate) fn custom(&self) {}", "custom"),
        "pub(crate) fn custom はスコープ付き可視性のため対象外（旧実装と同じ扱い）"
    );
    assert!(
        declares_pub_fn("pub unsafe fn custom(&self) {}", "custom"),
        "pub unsafe fn custom 宣言を検出できていない"
    );
    assert!(
        declares_pub_fn("pub const fn custom() {}", "custom"),
        "pub const fn custom 宣言を検出できていない"
    );
    // 7b) `extern`（ABI 文字列リテラル付き・なし双方）を挟んだ宣言
    //     （codex-review 指摘・PR #2212 その 11）: 旧実装は `unsafe`／
    //     `const`／`async` の 3 種のみ読み飛ばすため、`extern` トークン
    //     が `fn` 直前に残り一致に失敗していた。
    assert!(
        declares_pub_fn("pub extern \"C\" fn custom() {}", "custom"),
        "ABI 文字列リテラル付き pub extern \"C\" fn custom 宣言を検出できていない"
    );
    assert!(
        declares_pub_fn("pub unsafe extern \"C\" fn custom() {}", "custom"),
        "pub unsafe extern \"C\" fn custom（修飾子併記）宣言を検出できていない"
    );
    assert!(
        declares_pub_fn("pub extern fn custom() {}", "custom"),
        "ABI 文字列リテラルなし pub extern fn custom 宣言を検出できていない"
    );

    // 8) 同名別型によるシグネチャ混入（codex-review 指摘・PR #2212
    //    その 5）: `struct Tensor(BackendOps);` のようなローカル定義を
    //    トークン列の連続一致で検出できることの確認。ステートメント
    //    先頭一致（`strip_prefix`）では見逃す属性行（`#[derive(...)]`）
    //    付き・`struct`/`Tensor` 間に改行を挟んだケースも対象にする。
    for forged_wrapper in [
        "pub struct Tensor(BackendOps);\n",
        "#[derive(Debug)]\npub struct Tensor(BackendOps);\n",
        "pub struct\nTensor(BackendOps);\n",
        "enum Tensor { Wrapped(BackendOps) }\n",
    ] {
        let tokens = tokenize_including_punctuation(forged_wrapper);
        assert!(
            tokens
                .windows(2)
                .any(|w| (w[0] == "struct" || w[0] == "enum") && w[1] == "Tensor"),
            "ローカル Tensor 型定義をトークン列で検出できていない: {forged_wrapper}"
        );
    }

    // 9) import 元の完全パス検査（codex-review 指摘・PR #2212 その 5）:
    //    `Tensor` を import する use 文の完全パスが
    //    `fandhe_ai_tensor_core::Tensor` と異なる場合（別クレート・
    //    再エクスポート経由）はトークン列が一致しないことの確認。
    let forged_other_crate_import = "use some_other_crate::Tensor;\n";
    for statement in split_top_level_statements(forged_other_crate_import) {
        let after_vis = strip_visibility_prefix(&statement);
        let use_body = after_vis
            .strip_prefix("use ")
            .expect("use 文の前提が崩れている");
        let use_tokens = extract_identifier_tokens(use_body);
        assert_ne!(
            use_tokens,
            vec!["fandhe_ai_tensor_core".to_string(), "Tensor".to_string()],
            "別クレート由来の Tensor import が誤って正規 import と一致してしまっている: \
             {use_body}"
        );
    }

    // 10) `strip_keyword_prefix` は `use`／`type` の直後に改行が来る
    //     有効な Rust 記法も検出する（codex-review 指摘・PR #2212
    //     その 7: `starts_with("use ")`／`starts_with("type ")` は
    //     半角スペース 1 個固定のため `use\ncrate::...`・
    //     `type\nTensor = ...` を見逃していた）。
    assert_eq!(
        strip_keyword_prefix("use\ncrate::BackendOps as Tensor;", "use"),
        Some("crate::BackendOps as Tensor;")
    );
    assert_eq!(
        strip_keyword_prefix("type\nTensor = BackendOps;", "type"),
        Some("Tensor = BackendOps;")
    );
    // `used`／`typeof` のような無関係な識別子には一致しない。
    assert_eq!(strip_keyword_prefix("used_value = 1;", "use"), None);
    assert_eq!(strip_keyword_prefix("typeof_value = 1;", "type"), None);

    // 11) `strip_comments` はブロックコメント（ネスト対応）を除去する
    //     ため、実定義より前に置かれた偽トレイト定義（ブロックコメント
    //     内）は `extract_trait_body`／`extract_supertrait_bound_tokens`
    //     の走査対象から外れる（codex-review 指摘・PR #2212 その 6）。
    let forged_comment_then_real = "/* pub trait CustomFunction { fn evil(&self); } */\n\
         pub trait CustomFunction: Send + Sync + 'static {\n    fn forward(&self);\n}\n";
    let cleaned = strip_comments(forged_comment_then_real);
    assert!(
        !cleaned.contains("evil"),
        "ブロックコメント内の偽トレイト定義が除去されていない: {cleaned}"
    );
    let bounds = extract_supertrait_bound_tokens(&cleaned, "CustomFunction");
    assert!(
        bounds.iter().any(|b| b == "Send")
            && bounds.iter().any(|b| b == "Sync")
            && bounds.iter().any(|b| b == "'static"),
        "ブロックコメント除去後は実定義の supertrait 境界を正しく抽出できるはず: {bounds:?}"
    );
    let body = extract_trait_body(&cleaned, "CustomFunction");
    assert!(
        !body.contains("evil") && body.contains("forward"),
        "抽出したトレイト本体が実定義（forward のみ）ではなく偽定義（evil）を含んでいる: {body}"
    );

    // 12) `type Tensor = <denylist 外の型>;` のようなローカル型エイリアス
    //     も、右辺が denylist に一致しなくても構造的に検出される
    //     （codex-review 指摘・PR #2212 その 8: 旧実装は右辺の
    //     `FORBIDDEN_IDENTIFIERS` 一致のみを拒否しており、denylist に
    //     載っていない任意の型へのすり替えを見逃していた）。
    for forged_type_alias in [
        "type Tensor = SomeUnlistedType;\n",
        "pub type Tensor = SomeUnlistedType;\n",
        "type\nTensor = SomeUnlistedType;\n",
    ] {
        let tokens = tokenize_including_punctuation(forged_type_alias);
        assert!(
            tokens
                .windows(2)
                .any(|w| (w[0] == "struct" || w[0] == "enum" || w[0] == "type") && w[1] == "Tensor"),
            "denylist 外の型へすり替える type Tensor エイリアスを検出できていない: \
             {forged_type_alias}"
        );
    }

    // 13) `#[cfg(test)]` 配下の `use fandhe_ai_tensor_core::Tensor;` は
    //     canonical import 判定から除外される（codex-review 指摘・
    //     PR #2212 その 8: トップレベル import を非 canonical な別名へ
    //     差し替えつつ、テスト専用スコープの正規 import だけで判定を
    //     通過させる迂回を塞ぐ）。パスは canonical だが `#[cfg(test)]`
    //     という条件付き属性が付いているため `find_canonical_tensor_
    //     import` は `Err` を返す（新設計: 条件付き Tensor import は
    //     パスの正誤に関わらずすべて拒否する。旧実装は「無条件の別名
    //     import が無ければ canonical 扱いしない」という消極的な保護
    //     だったが、新設計では条件付き import の存在自体を積極的に
    //     エラーとして報告する）。
    let cfg_test_only_import =
        "#[cfg(test)]\nuse fandhe_ai_tensor_core::Tensor;\n\npub type Tensor = BackendOps;\n";
    assert!(
        find_canonical_tensor_import(cfg_test_only_import).is_err(),
        "#[cfg(test)] 配下の canonical import が誤って許容されている"
    );

    // 14) codex-review 追加指摘（PR #2212 その 15）が挙げたバイパス
    //     シナリオを検出できることを固定する: 旧実装
    //     `statement_has_cfg_test_attribute` はステートメント全体の
    //     トークン列に `cfg`・`(`・`test`・`)` の並びが**任意の位置**に
    //     現れるかで判定していたため、(a) 常に無効な `#[cfg(any())]`
    //     が付いた canonical import を誤って通過させ、(b)
    //     `#[cfg_attr(any(), cfg(test))]` のように属性の**引数内**に
    //     `cfg(test)` を含むだけの別属性まで「cfg(test) 属性付き」と
    //     誤判定してパス不一致の検証ごと迂回できていた。
    //     `statement_has_conditional_attribute` は属性の先頭識別子
    //     （属性パス名）のみで `cfg`／`cfg_attr` 判定するため、
    //     いずれも検出できる。
    assert!(
        statement_has_conditional_attribute("#[cfg(any())]\nuse fandhe_ai_tensor_core::Tensor;"),
        "常に無効な #[cfg(any())] 属性を検出できていない（(a) の迂回シナリオ）"
    );
    assert!(
        statement_has_conditional_attribute(
            "#[cfg_attr(any(), cfg(test))]\nuse other_crate::Tensor;"
        ),
        "cfg_attr 属性の引数内に cfg(test) を含むだけの別属性を誤って見逃している\
         （(b) の迂回シナリオ）"
    );
    assert!(
        statement_has_conditional_attribute("#[cfg( test )]\nuse fandhe_ai_tensor_core::Tensor;"),
        "空白入り #[cfg( test )] 属性を検出できていない"
    );
    assert!(
        !statement_has_conditional_attribute("use fandhe_ai_tensor_core::Tensor;"),
        "条件付き属性が無い文を誤って検出している"
    );
    assert!(
        !statement_has_conditional_attribute(
            "#[allow(unused_imports)]\nuse fandhe_ai_tensor_core::Tensor;"
        ),
        "cfg／cfg_attr 以外の属性（allow）を誤って条件付きと判定している"
    );

    // 上記 (a)・(b) を実際に `find_canonical_tensor_import` へ通した
    // 統合シナリオ: 常に無効な `#[cfg(any())]` が付いた「パスは正しい」
    // canonical import と、`#[cfg_attr(any(), cfg(test))]` が付いた
    // 「パスが異なる」import を両方トップレベルに置いても、どちらも
    // 条件付き属性の時点で拒否され、非正規型がシグネチャへ入るのを
    // 検出できずに素通りすることはない。
    let forged_conditional_scenario = "#[cfg(any())]\n\
         use fandhe_ai_tensor_core::Tensor;\n\n\
         #[cfg_attr(any(), cfg(test))]\n\
         use other_crate::Tensor;\n";
    assert!(
        find_canonical_tensor_import(forged_conditional_scenario).is_err(),
        "条件付き属性付きの Tensor import（canonical パスであっても）が誤って\
         許容されている: {forged_conditional_scenario}"
    );

    // 無条件の canonical import のみが存在する通常ケースは引き続き
    // `Ok(true)` を返す（今回の変更による既存の正常経路への回帰が
    // ないことの確認）。
    assert_eq!(
        find_canonical_tensor_import("use fandhe_ai_tensor_core::Tensor;\n"),
        Ok(true),
        "無条件の canonical import が誤って拒否されている"
    );
    assert_eq!(
        find_canonical_tensor_import("use crate::BackendOps;\n"),
        Ok(false),
        "Tensor を import しない場合は canonical import 不在（Ok(false)）を返すはず"
    );

    // 15) 外部属性（`#[allow(unused_imports)]` 等）が先頭に付いた
    //     `use`／`type` 宣言も alias 検出・type エイリアス検査の対象に
    //     なる（codex-review 指摘・PR #2212 その 13）: `split_top_level_
    //     statements` は属性と後続 item を 1 ステートメントに結合する
    //     ため、`strip_leading_attributes` で属性を読み飛ばしてから
    //     `strip_visibility_prefix`／`strip_keyword_prefix` を適用しない
    //     と、属性文字列が先頭に残ったままキーワード一致に失敗し検査を
    //     素通りしてしまう。
    let forged_attr_use = "#[allow(unused_imports)]\nuse crate::BackendOps as Ops;\n";
    let mut alias_detected_via_attr = false;
    for statement in split_top_level_statements(forged_attr_use) {
        let after_attributes = strip_leading_attributes(&statement);
        let after_visibility = strip_visibility_prefix(after_attributes);
        if strip_keyword_prefix(after_visibility, "use").is_some()
            && extract_identifier_tokens(after_visibility)
                .iter()
                .any(|token| token == "as")
        {
            alias_detected_via_attr = true;
        }
    }
    assert!(
        alias_detected_via_attr,
        "外部属性付き use 文の alias（as トークン）を検出できていない: {forged_attr_use}"
    );

    let forged_attr_type = "#[allow(dead_code)]\npub type Tensor = BackendOps;\n";
    let mut type_detected_via_attr = false;
    for statement in split_top_level_statements(forged_attr_type) {
        let after_attributes = strip_leading_attributes(&statement);
        let after_visibility = strip_visibility_prefix(after_attributes);
        if strip_keyword_prefix(after_visibility, "type").is_some()
            && contains_identifier(after_visibility, "BackendOps")
        {
            type_detected_via_attr = true;
        }
    }
    assert!(
        type_detected_via_attr,
        "外部属性付き pub type 宣言の禁止型参照を検出できていない: {forged_attr_type}"
    );

    // 複数属性・複数行属性も読み飛ばせる（ネスト括弧・改行を含む）。
    let forged_multi_attr = "#[allow(unused_imports)]\n#[cfg_attr(test, allow(dead_code))]\nuse crate::BackendOps as Ops;\n";
    let mut alias_detected_via_multi_attr = false;
    for statement in split_top_level_statements(forged_multi_attr) {
        let after_attributes = strip_leading_attributes(&statement);
        let after_visibility = strip_visibility_prefix(after_attributes);
        if strip_keyword_prefix(after_visibility, "use").is_some()
            && extract_identifier_tokens(after_visibility)
                .iter()
                .any(|token| token == "as")
        {
            alias_detected_via_multi_attr = true;
        }
    }
    assert!(
        alias_detected_via_multi_attr,
        "複数行・複数個の外部属性を読み飛ばせていない: {forged_multi_attr}"
    );
}
