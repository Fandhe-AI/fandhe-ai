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
/// **本関数を単独で境界検査の前処理として使わない**（codex-review 追加
/// P1 指摘・PR #2212 是正の続き）: 文字列リテラルの中身を関知しない
/// 単純な文字走査のため、`impl Var { fn s() -> &'static str { "//" } pub
/// fn custom(&self) {} }` のように文字列リテラルの中身に `//` が含まれ
/// ると、本関数はそれを実コメントの開始と誤認して**行末までを丸ごと**
/// 読み飛ばしてしまう（本関数の時点で該当行の残り——`} pub fn
/// custom(&self) {}` を含む——が既に失われるため、文字列内容を別途
/// 空白化する後処理を挟んでも手遅れである）。この構造的欠陥（コメント
/// 検出とリテラル境界検出を別パスに分ける設計そのものの限界）を塞いだ
/// 単一パスの [`normalize_source`] が現行の唯一の本番経路であり、本関数
/// （`strip_comments`）はこのバイパスを実演する回帰テスト
/// （`var_impl_block_bodies_detects_declaration_in_other_module_file` の
/// 「対照実験」・`architecture_boundary_bypass_scenarios_are_detected` の
/// 文字列内 `//`／文字列内偽トレイト定義シナリオ）専用の歴史的
/// ユーティリティとして残す。
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

/// 単一パスの字句正規化: コメント（行・ブロック。ネスト対応）と、
/// 文字列・raw 文字列・バイト文字列・char・ライフタイムを判別した
/// うえでリテラルの中身のみを空白（改行は `\n` のまま保持）へ置換した
/// 文字列を返す（codex-review 指摘・PR #2212 是正の続き。第 2 ラウンド
/// レビュー指摘 1・2）。
///
/// **本ファイルの全ての境界検査（否定ガード・`extract_trait_body`／
/// `extract_supertrait_bound_tokens`・canonical import 検査・alias
/// 検査等）は本関数の出力のみを走査対象とする**——`strip_comments` →
/// `strip_string_literal_contents` の 2 パス方式は、1 パス目
/// （`strip_comments`）の時点で文字列リテラル中の `//` を実コメントの
/// 開始と誤認して行末までを削ってしまうため、2 パス目を後段に挟んでも
/// 手遅れという構造的欠陥を持つ（`strip_comments` のドキュメンテーション
/// コメント参照）。さらに、コメント除去より前に `content.find("pub
/// trait CustomFunction")` 等の部分文字列一致を行う設計では、文字列
/// リテラルの中身に偽の定義テキストを埋め込んだ場合にそれを実定義より
/// 先に抜き出してしまう（本関数は文字列内容を空白化するため、この
/// 偽装テキスト自体が走査対象から消える）。本関数はコメント検出と
/// リテラル境界検出を単一の走査で行うことで、リテラルの中身が
/// コメント検出ロジックへ一切渡らないようにし、両方のバイパスを構造的に
/// 塞ぐ。
///
/// 対応するリテラル形式:
/// - 行コメント（`//` から行末まで）・ブロックコメント（`/* ... */`。
///   ネスト対応）
/// - 通常文字列（`"..."`。エスケープシーケンス対応。開始・終了の
///   クォート文字自体は保持する——`skip_fn_declaration_qualifiers` の
///   ABI 文字列リテラル判定〈`extern "C"` のクォートをトークンとして
///   検出する処理〉と整合させるため）
/// - raw 文字列（`r"..."`／`r#"..."#`／`r##"..."##` 等。`#` の個数に
///   応じて開始・終了区切りをバランスさせる。raw 文字列は仕様上
///   エスケープ処理を行わないため、中身は単純に空白化する）
/// - バイト文字列（`b"..."`）・raw バイト文字列（`br"..."`／
///   `br#"..."#` 等）——`b`／`r` が識別子の途中（例: `for` の `r`）で
///   はなく独立したリテラル接頭辞として現れる場合のみ対象とする
///   （直前の文字が識別子構成文字でないことを条件にする）
/// - C 文字列（`c"..."`）・raw C 文字列（`cr"..."`／`cr#"..."#` 等。
///   Bugbot 指摘・PR #2212: 旧実装は `c`／`cr` 接頭辞を認識せず、
///   `c` を素通しした直後の `r` が識別子継続文字に見えるため
///   `cr#"..."#` が単一の raw リテラルとして扱われず、内部の `"` が
///   偽の通常文字列を開いて後続ソースを丸ごと飲み込んでいた。`b`／
///   `br` と同じ「直前が識別子構成文字でない」境界条件のもとで判定する
///   （`abc"..."` の `c` のように識別子途中の `c` は対象外）
/// - char（`'a'`／`'\n'`／`'\''`／`'\u{7b}'` 等）・バイト char（`b'a'`
///   等）
/// - ライフタイム（`'a`・`'static` 等。閉じクォートを伴わないため char
///   と区別してそのまま素通しする）
///
/// トークン境界と行構造（改行位置）を保つため、コメント除去は前後に
/// 半角スペース 1 個を挿入する（`strip_comments` の既存方針を踏襲。
/// 隣接トークンの意図しない連結を防ぐ）。
fn normalize_source(content: &str) -> String {
    fn is_ident_continue(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    // `chars[start..]` から連続する `#` の個数を数え、その直後が `"`
    // であれば `Some(個数)` を返す（raw 文字列の開始判定用）。
    fn raw_hash_run(chars: &[char], start: usize) -> Option<usize> {
        let mut j = start;
        while chars.get(j) == Some(&'#') {
            j += 1;
        }
        if chars.get(j) == Some(&'"') {
            Some(j - start)
        } else {
            None
        }
    }

    let chars: Vec<char> = content.chars().collect();
    let len = chars.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;
    while i < len {
        let c = chars[i];
        let prev_is_ident = i > 0 && is_ident_continue(chars[i - 1]);

        // 行コメント。
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // ブロックコメント（ネスト対応）。
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            out.push(' ');
            i += 2;
            let mut depth = 1i32;
            while i < len && depth > 0 {
                if chars.get(i) == Some(&'/') && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars.get(i) == Some(&'*') && chars.get(i + 1) == Some(&'/') {
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

        // リテラル接頭辞（`b`／`r`／`br`／`c`／`cr`）は、直前が識別子
        // 構成文字でない（＝独立した新しいトークンの先頭である）場合
        // のみ判定する。`for`・`return` の `r` のように識別子の一部で
        // ある場合は対象外（`prev_is_ident` 判定に加え、`raw_hash_run`
        // が直後に `"` を伴わない通常の識別子継続文字を見た時点で
        // `None` を返すため、いずれの条件でも誤爆しない）。
        if !prev_is_ident {
            // raw バイト文字列（`br"..."`／`br#"..."#` 等）。
            if c == 'b'
                && chars.get(i + 1) == Some(&'r')
                && let Some(hashes) = raw_hash_run(&chars, i + 2)
            {
                i = emit_raw_literal(&chars, &mut out, i, 2 + hashes, hashes);
                continue;
            }
            // raw C 文字列（`cr"..."`／`cr#"..."#` 等。Bugbot 指摘・PR
            // #2212: `br` と同様の 2 文字接頭辞として、単純な `r` 判定
            // より先に判定する必要がある——先に `r` 単独判定を行うと
            // `c` の直後にある `r` を見落として通常の `c"..."` 判定へ
            // フォールスルーしてしまい、`cr#"..."#` の `#` 以降が
            // ソースへそのまま漏れる）。
            if c == 'c'
                && chars.get(i + 1) == Some(&'r')
                && let Some(hashes) = raw_hash_run(&chars, i + 2)
            {
                i = emit_raw_literal(&chars, &mut out, i, 2 + hashes, hashes);
                continue;
            }
            // raw 文字列（`r"..."`／`r#"..."#` 等）。
            if c == 'r'
                && let Some(hashes) = raw_hash_run(&chars, i + 1)
            {
                i = emit_raw_literal(&chars, &mut out, i, 1 + hashes, hashes);
                continue;
            }
            // バイト文字列（`b"..."`）・バイト char（`b'...'`）: `b` は
            // そのまま出力し、直後の通常文字列／char 処理へフォール
            // スルーする（次のループ反復で下の `"`／`'` 分岐に入る）。
            if c == 'b' && matches!(chars.get(i + 1), Some('"') | Some('\'')) {
                out.push('b');
                i += 1;
                continue;
            }
            // C 文字列（`c"..."`）: `c` はそのまま出力し、直後の通常
            // 文字列処理へフォールスルーする（Rust に `c'...'` という
            // 単一文字 C 文字リテラル形式は存在しないため char 分岐は
            // 不要）。
            if c == 'c' && chars.get(i + 1) == Some(&'"') {
                out.push('c');
                i += 1;
                continue;
            }
        }

        // char リテラル（`'"'`／`'\\''`／`'\u{XXXX}'` 等）とライフタイム
        // （`'a` 等・閉じクォートを伴わない）の判別
        // （`strip_string_literal_contents` から継承した判定ロジック）。
        if c == '\'' {
            if chars.get(i + 1) == Some(&'\\') {
                // エスケープシーケンス。`\u{XXXX}` は可変長のため `}` を
                // 含め閉じクォートまで読み進める。中身はソーステキスト
                // 上に `{`／`}` という文字がそのまま現れうる
                // （`'\u{7b}'` 等）ため空白化し、[`var_impl_block_bodies`]
                // の中括弧深さ追跡を誤らせない。バックスラッシュ直後の
                // 1 文字（エスケープ本体の先頭）は、それが `'\''` の
                // `\'` のように閉じクォートと同じ文字であっても判定せず
                // 無条件に消費する。これをしないと `'\''` の 2 文字目の
                // `'` を閉じクォートと誤認識し、直後の実際の閉じ
                // クォートから再同期してしまう（`('\'','"')` の `"` を
                // 文字列リテラル開始と誤認識する不具合）。
                out.push('\'');
                let mut j = i + 1; // バックスラッシュの位置
                out.push(' ');
                j += 1;
                if j < len {
                    out.push(' '); // エスケープ本体の先頭 1 文字（無条件消費）
                    j += 1;
                }
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
                out.push('\'');
                out.push(' ');
                out.push('\'');
                i += 3;
                continue;
            }
            // 閉じクォートが直後に無い ⇒ ライフタイム（`'a`／`'static`
            // 等）。そのまま素通しする。
            out.push(c);
            i += 1;
            continue;
        }

        // 通常の文字列リテラル。`//`／`/*` を含む中身も、この分岐に
        // 入った時点で閉じクォートまで一気に消費するため、上のコメント
        // 判定へ渡ることはない（2 パス方式の構造的欠陥を単一パス化で
        // 塞ぐ本関数の核心）。
        if c == '"' {
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
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

/// [`normalize_source`] の raw 文字列（`r"..."`／`r#"..."#`／
/// `br"..."`／`br#"..."#` 等）処理を切り出したヘルパー。`start` は
/// 接頭辞の先頭（`b`／`r` のいずれか）の位置、`prefix_len` は `b`／
/// `r`／`#`（`hashes` 個）を含む開始区切り全体の長さ（開始 `"` 手前
/// まで）、`hashes` は開始区切りに含まれる `#` の個数（終端 `"` の
/// 直後に同数の `#` が続く箇所を終端とみなす。仕様どおり raw 文字列は
/// エスケープ処理を行わない）。開始・終了の区切り文字列（`b`／`r`／
/// `#`／`"`）はそのまま出力へコピーし、内容のみを空白（改行は `\n` の
/// まま保持）へ置換する。戻り値は処理後の走査位置（呼び出し元の `i`
/// に代入する）。
fn emit_raw_literal(
    chars: &[char],
    out: &mut String,
    start: usize,
    prefix_len: usize,
    hashes: usize,
) -> usize {
    let len = chars.len();
    let quote_pos = start + prefix_len;
    for &ch in &chars[start..=quote_pos] {
        out.push(ch);
    }
    let mut i = quote_pos + 1;
    while i < len {
        if chars[i] == '"' && chars[i + 1..].iter().take(hashes).all(|&c| c == '#') {
            out.push('"');
            for _ in 0..hashes {
                out.push('#');
            }
            return i + 1 + hashes;
        }
        out.push(if chars[i] == '\n' { '\n' } else { ' ' });
        i += 1;
    }
    i
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
///    出し側で [`normalize_source`] 済みの content をトークン化した
///    ものであることを前提とする**——文字列・char リテラル中の
///    `{`/`}` を素通しした生トークン列を渡すと、リテラル内の中括弧を
///    実際の中括弧として誤カウントしてしまう（理由は
///    [`normalize_source`] のドキュメンテーションコメント参照）。
/// 4. ヘッダーに `Var` が含まれていた場合のみ本体トークン列を返す。
///
/// **ヘッダー走査中の `<...>`／`[...]`／`(...)` の深さ追跡**（Bugbot
/// 指摘・PR #2212 第 2 ラウンド）: 手順 2 のヘッダー走査は「最初に現れ
/// た `{` を本体開始」とみなす単純な実装だったため、`impl<const N:
/// usize> Var where [(); { N }]: Sized { pub fn custom() {} }` のように
/// where 節内の const ジェネリクス式（`[(); { N }]`）が `{`／`}` を
/// 含むと、その `{` を本体開始と誤認してヘッダー走査を早期終了して
/// しまい、実際の本体（`pub fn custom` を含む）を取りこぼす。本実装は
/// ヘッダー走査中も `<`／`[`／`(` の深さを個別に追跡し、いずれも深さ 0
/// の位置で現れた最初の `{` のみを本体開始として扱う。深さが 0 でない
/// 間に現れる `{`／`}`（`[(); { N }]` 内の const ブロック等）は単に
/// 読み飛ばす（対応する `[`／`]` の深さで囲まれている限り、内部の
/// `{`／`}` 自体を追跡する必要はない）。
///
/// **ヘッダー走査中の `{...}`（const ジェネリクス既定値式）の深さ追跡**
/// （Bugbot 指摘・PR #2212 第 3 ラウンド）: 上記修正後も、ジェネリクス
/// リストを事前に読み飛ばす別ループ（旧実装）が `<`／`>` トークンのみで
/// 深さを数えていたため、`impl<const B: bool = { 3 > 2 }, const N:
/// usize = { 1 }> Var { pub fn custom() {} }` のように const ジェネリ
/// クス既定値式の中に比較演算子由来の `>`（`3 > 2`）が現れると、その
/// `>` を誤って山括弧の閉じと数えてジェネリクスリストの途中で走査を
/// 打ち切ってしまい、残りの `, const N: usize = { 1 }` 部分に含まれる
/// `{`（第 2 引数の既定値式の開始）を本体開始と誤認してしまう
/// （実際の本体である `pub fn custom` を取りこぼす）。本実装は事前
/// スキップ用の別ループを廃止し、単一のヘッダー走査ループへ統合した
/// うえで `{`／`}` の深さ（`brace_depth`）も追跡する: `brace_depth > 0`
/// の間は `<`／`>`／`[`／`]`／`(`／`)` を深さ追跡の対象外とし
/// （ネストした const 式の内部にある構文要素として無視する）、`{` は
/// 「他のすべての深さが 0」の場合のみ本体開始とみなし、それ以外の
/// `{` は単に `brace_depth` を 1 増やして読み飛ばす（対応する `}` で
/// 減らす）。ジェネリクスの有無に依らずこの単一ループがヘッダー全体
/// （`<...>`・型パス・`for`・`where` 節）を走査するため、事前スキップ
/// ループは不要になった。
///
/// [`var_impl_block_bodies_with_aliases`] の `extra_var_aliases` を空
/// スライスで呼ぶ薄いラッパー（既存の呼び出し元・回帰テストの互換性を
/// 保つため。alias 判定が不要な合成入力のテストはこちらを使い続ける）。
fn var_impl_block_bodies(tokens: &[String]) -> Vec<Vec<String>> {
    var_impl_block_bodies_with_aliases(tokens, &[])
}

/// `tokens`（`fn` 宣言を含む任意の本体トークン列）に `fn <fn_name>(`
/// または `fn <fn_name><`（ジェネリクス付き）宣言が存在するかを、
/// 可視性修飾（`pub` の有無）を一切問わずトークン列の連続一致で判定
/// する（codex-review P1 指摘・PR #2212: trait impl のメソッドは
/// `impl Trait for Var { fn custom(&self) {} }` のように可視性修飾子を
/// 一切書かずに宣言でき、それでもトレイトの可視性がそのまま公開 API と
/// して機能する——`Var: Trait` かつ `Trait` が到達可能なら `Var::
/// custom` は外部から呼べる。[`tokens_declare_pub_fn`] は `pub` 必須で
/// 判定するため trait impl のメソッドを検出できず、[`var_impl_block_
/// bodies_with_aliases_and_kind`] が trait impl と判定した本体にはこちら
/// を使う）。`unsafe`／`const`／`async`／`extern` 等の修飾子は `fn`
/// トークンの直前に来るため、修飾子の有無に関わらず `fn` トークン自体を
/// 起点に走査すれば足りる（[`skip_fn_declaration_qualifiers`] を経由
/// する必要がない）。
fn tokens_declare_fn(tokens: &[String], fn_name: &str) -> bool {
    for (i, token) in tokens.iter().enumerate() {
        if token == "fn"
            && tokens.get(i + 1).map(String::as_str) == Some(fn_name)
            && matches!(tokens.get(i + 2).map(String::as_str), Some("(") | Some("<"))
        {
            return true;
        }
    }
    false
}

/// [`var_impl_block_bodies`] の本体（codex-review P1 指摘・PR #2212 その
/// 続き。イシュー #2064）: ヘッダー判定は文字どおりの `Var` トークンの
/// 有無のみを見ていたため、同一クレート内の別ファイルで `use crate::Var
/// as V; impl V<'_> { pub fn custom(...) {} }` のように import alias を
/// 経由すると、`header_has_var` が false のまま否定ガード
/// （`autodiff_src_does_not_declare_pub_fn_custom_on_var`）を素通りできて
/// しまっていた。本関数は呼び出し元が同一ファイルから事前に収集した
/// `Var` の alias 名集合（[`find_var_alias_declarations`]。`use ... Var as
/// X`／`type X = Var...;` を検出する）を `extra_var_aliases` として受け取り、
/// ヘッダー走査中に `Var` そのものだけでなくこれらの alias 名のいずれかが
/// 独立したトークンとして現れた場合も impl 対象を `Var` とみなす。
/// alias 解決自体は行わず、同一ファイル内で検出済みの alias 名との文字列
/// 一致のみで判定する（`autodiff_src_does_not_alias_var` が alias 宣言
/// 自体を fail-closed に拒否するため、alias 経由の impl も両テストの
/// 組み合わせで検出できる）。
fn var_impl_block_bodies_with_aliases(
    tokens: &[String],
    extra_var_aliases: &[String],
) -> Vec<Vec<String>> {
    var_impl_block_bodies_with_aliases_and_kind(tokens, extra_var_aliases)
        .into_iter()
        .map(|(body, _is_trait_impl)| body)
        .collect()
}

/// [`var_impl_block_bodies_with_aliases`] の本体（codex-review P1
/// 指摘・PR #2212 第 4 ラウンド）。trait impl（`impl<...> Trait for Var
/// { ... }`）と inherent impl（`impl<...> Var { ... }`）を区別せず本体
/// トークン列のみを返す旧実装では、`autodiff_src_does_not_declare_pub_
/// fn_custom_on_var` が [`tokens_declare_pub_fn`]（`pub` 必須）のみを
/// 適用していたため、trait impl 経由で `pub` を書かずに宣言された
/// `fn custom`（`impl SomeTrait for Var { fn custom(&self) {} }`。
/// トレイト実装のメソッドは可視性修飾子を持たず、トレイト自体の可視性が
/// そのまま公開 API として機能する）を見逃していた。本関数はヘッダー
/// 走査中に独立した `for` トークン（深さ 0。`impl<...> Trait for Var`
/// の区切り）の出現を記録し、本体トークン列と合わせて `is_trait_impl`
/// フラグをタプルで返す。呼び出し側（否定ガード本体）はこのフラグに
/// 応じて trait impl には [`tokens_declare_fn`]（可視性不問）、
/// inherent impl には従来どおり [`tokens_declare_pub_fn`]（`pub` 必須）
/// を使い分ける。
///
/// **過剰検出側に倒す既知の限界**: where 節中の高階トレイト境界
/// （HRTB。例: `impl<F> Var where F: for<'a> Fn(&'a i32) { fn custom(&self)
/// {} }`）に現れる `for` も深さ 0 として拾ってしまい、実際には inherent
/// impl であっても trait impl と誤分類しうる。この場合 `pub` の有無を
/// 問わず `fn custom` を検出するため、本来は非公開で無害な同名メソッド
/// を誤って違反と報告する可能性があるが、逆方向（trait impl 経由の
/// 本物の公開漏れを見逃す）よりは安全側であり、本ファイル全体が採用する
/// fail-closed 方針（過剰検出を許容し検出漏れを避ける）と整合する。
fn var_impl_block_bodies_with_aliases_and_kind(
    tokens: &[String],
    extra_var_aliases: &[String],
) -> Vec<(Vec<String>, bool)> {
    let mut bodies = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != "impl" {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut header_has_var = false;
        let mut is_trait_impl = false;
        let mut angle_depth = 0i32;
        let mut bracket_depth = 0i32;
        let mut paren_depth = 0i32;
        let mut brace_depth = 0i32;
        while j < tokens.len() {
            match tokens[j].as_str() {
                // `>` は `saturating_sub` では 0 未満に飽和しない
                // （`i32::saturating_sub` は型の最小値方向にのみ飽和する
                // ため、`0i32.saturating_sub(1)` は `-1` になる）。where 節
                // に現れる `->`（戻り値型矢印。トークナイザは `-`／`>` の
                // 2 トークンに分解する）や比較演算子由来の対応しない `>`
                // で `angle_depth` が負に振れると、以降ずっと `angle_
                // depth == 0` の条件を満たせなくなり、実際の本体開始
                // `{` を永久に見失ってしまう（自己レビューで判明）。
                // `(depth - 1).max(0)` で 0 未満に飽和させることで、
                // 対応しない `>` を無害化しつつ、正しく対応する `<...>`
                // の深さ追跡は従来どおり機能する。
                //
                // `brace_depth > 0`（const ジェネリクス既定値式
                // `{ 3 > 2 }` 等のネストした中括弧の内側）の間は
                // `<`／`>`／`[`／`]`／`(`／`)` を一切カウントしない
                // （ドキュメンテーションコメント「ヘッダー走査中の
                // `{...}`」節参照）。式の中身に現れる比較演算子由来の
                // `>` 等を山括弧の閉じと誤認しないようにするための
                // ガードで、対応する `[`／`]`・`(`／`)` の深さ追跡自体
                // には影響しない。
                "<" if brace_depth == 0 => angle_depth += 1,
                ">" if brace_depth == 0 => angle_depth = (angle_depth - 1).max(0),
                "[" if brace_depth == 0 => bracket_depth += 1,
                "]" if brace_depth == 0 => bracket_depth = (bracket_depth - 1).max(0),
                "(" if brace_depth == 0 => paren_depth += 1,
                ")" if brace_depth == 0 => paren_depth = (paren_depth - 1).max(0),
                "{" if angle_depth == 0
                    && bracket_depth == 0
                    && paren_depth == 0
                    && brace_depth == 0 =>
                {
                    break;
                }
                "{" => brace_depth += 1,
                "}" => brace_depth = (brace_depth - 1).max(0),
                // `for`（深さ 0）は `impl<...> Trait for Var` の trait
                // impl 区切りキーワード。where 節中の HRTB（`for<'a> ...`）
                // も同じ深さ 0 で出現しうるため誤って trait impl と判定
                // する場合があるが、本関数の doc コメント「過剰検出側に
                // 倒す既知の限界」節が示すとおり fail-closed 方針上
                // 許容する。
                "for"
                    if angle_depth == 0
                        && bracket_depth == 0
                        && paren_depth == 0
                        && brace_depth == 0 =>
                {
                    is_trait_impl = true;
                }
                other if other == "Var" || extra_var_aliases.iter().any(|alias| alias == other) => {
                    header_has_var = true;
                }
                _ => {}
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
            bodies.push((tokens[body_start..body_end].to_vec(), is_trait_impl));
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
    // コメント（行・ブロック。ネスト対応）と文字列・char リテラルの
    // 中身を単一パスで正規化した全文（`content` 中の位置に依存する
    // 走査は以降すべてこの `normalized` を対象にする。codex-review
    // 指摘・PR #2212 その 6・第 2 ラウンド指摘 1・2: ブロックコメント
    // 中や文字列リテラル中の偽トレイト定義を `extract_trait_body`／
    // `extract_supertrait_bound_tokens` が誤って最初の一致として抜き
    // 出す迂回、および文字列リテラル中の `//` が実コメントの開始と
    // 誤認され後続コードが失われる迂回を、単一パスの [`normalize_
    // source`] で構造的に塞ぐ）。
    let normalized = normalize_source(&content);

    let supertrait_bounds = extract_supertrait_bound_tokens(&normalized, "CustomFunction");
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

    let trait_body = extract_trait_body(&normalized, "CustomFunction");
    assert!(
        !trait_body.is_empty(),
        "src/custom.rs から `pub trait CustomFunction` 本体を抽出できなかった\
         （テスト自体が検査対象を見失っている。ファイル構成が変わっていないか確認）"
    );
    // `trait_body` は `normalized`（正規化済み）から抽出済みのため
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
    for statement in split_top_level_statements(&normalized) {
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
    let all_tokens = tokenize_including_punctuation(&normalized);
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
    match find_canonical_tensor_import(&normalized) {
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
/// 走査し、各ファイルを [`normalize_source`]（コメント除去・文字列／
/// char リテラル内容の空白化を単一パスで行う。旧 `strip_comments` →
/// `strip_string_literal_contents` の 2 パス方式は、1 パス目の時点で
/// 文字列リテラル中の `//` を実コメントと誤認して後続コードごと削って
/// しまう構造的欠陥を持つため使わない）した上で [`var_impl_block_
/// bodies`] により「ヘッダーに `Var` を含む impl ブロック」の本体のみを
/// 抜き出し、その本体トークン列に対して [`tokens_declare_pub_fn`] を
/// 適用する（`impl Other { pub fn custom() {} }` のような無関係な型への
/// 同名メソッドは検出対象外のまま。ネストした impl ブロック——関数本体
/// の中に書かれた `impl Var` 等——も [`var_impl_block_bodies`] が
/// 取りこぼさない）。
#[test]
fn autodiff_src_does_not_declare_pub_fn_custom_on_var() {
    let src_dir = autodiff_crate_root().join("src");
    let mut violations: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        // 単一パスの正規化（コメント除去・文字列／char リテラル中身の
        // 空白化。`pub/* separator */ fn custom` のようなブロックコメント
        // を挟んだ宣言、`"//"` を含む文字列リテラル、文字列・char リテ
        // ラル内の `{`／`}` のいずれも構造的に扱える。codex-review 指摘・
        // PR #2212 その 12・第 2 ラウンド指摘 1）。
        let normalized = normalize_source(content);
        let tokens = tokenize_including_punctuation(&normalized);
        // 同一ファイル内で `use ... Var as X;`／`type X = Var...;` により
        // 宣言された alias 名を先に収集し、ヘッダー判定（`header_has_var`）
        // に加える（codex-review P1 指摘・PR #2212 その続き。イシュー
        // #2064: 文字どおりの `Var` トークンのみを見るヘッダー判定では
        // `use crate::Var as V; impl V { pub fn custom() {} }` のような
        // alias 経由の宣言を見逃す。`autodiff_src_does_not_alias_var` が
        // alias 宣言自体を fail-closed に拒否するのと合わせた二重の対策）。
        let var_aliases = find_var_alias_declarations(&normalized);
        // `is_trait_impl` により判定関数を使い分ける（codex-review P1
        // 指摘・PR #2212 第 4 ラウンド）: inherent impl（`impl Var { ... }`）
        // は従来どおり [`tokens_declare_pub_fn`]（`pub` 必須）で判定する
        // 一方、trait impl（`impl Trait for Var { ... }`）はメソッドに
        // 可視性修飾子を書かないのが通常の Rust 記法であり、トレイト自体
        // の可視性がそのまま公開 API として機能するため、`pub` の有無を
        // 問わない [`tokens_declare_fn`] で判定する
        // （[`var_impl_block_bodies_with_aliases_and_kind`] のドキュメン
        // テーションコメント参照）。
        for (body, is_trait_impl) in
            var_impl_block_bodies_with_aliases_and_kind(&tokens, &var_aliases)
        {
            for fn_name in ["custom", "add_custom"] {
                let declared = if is_trait_impl {
                    tokens_declare_fn(&body, fn_name)
                } else {
                    tokens_declare_pub_fn(&body, fn_name)
                };
                if declared {
                    violations.push(format!(
                        "{}: impl Var（または alias {var_aliases:?}。trait impl={is_trait_impl}）に \
                         {fn_name} 宣言",
                        path.display()
                    ));
                }
            }
        }
    });
    assert!(
        violations.is_empty(),
        "crates/autodiff/src 配下の impl Var ブロックに未承認の custom／\
         add_custom 宣言が見つかった（§12.5 (b) 未承認のまま Var 経由の到達口を\
         設けてしまっている。`pub`・`fn`・関数名の間に改行・ブロックコメントを\
         挟んだ宣言も、`src/var.rs` 以外のファイルに書かれた impl ブロックも、\
         同一ファイル内の import alias 経由の宣言も、trait impl 経由の\
         可視性修飾子なし宣言も検出する）: {violations:?}"
    );
}

/// [`autodiff_src_does_not_declare_pub_fn_custom_on_var`] のブロックコメント経由の
/// バイパス（codex-review 指摘・PR #2212 その 12）を、実際に
/// [`normalize_source`] + `declares_pub_fn` の組み合わせ（本番経路と
/// 同一のパイプライン）で検出できることを固定する回帰テスト。
/// `src/var.rs` 本体を変更せずに検証するため、同じパターンの合成入力に
/// 対して直接アサートする。
#[test]
fn declares_pub_fn_detects_declaration_split_by_block_comment() {
    let forged = "pub/* separator */fn custom(&self) {}";
    let normalized = normalize_source(forged);
    assert!(
        declares_pub_fn(&normalized, "custom"),
        "ブロックコメントで pub と fn を分断した宣言を検出できていない: {normalized}"
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
    let tokens = tokenize_including_punctuation(&normalize_source(forged_in_extra_module));
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
    //    `normalize_source → tokenize` パイプラインを通す（PR #2212
    //    追加 P1 の是正時に自己レビューで判明: ABI 文字列リテラル `"C"`
    //    が正規化を経由しても `skip_fn_declaration_qualifiers` の判定と
    //    整合することを固定する）。
    let forged_with_lifetime_and_abi = "impl<'a> Var { pub extern \"C\" fn add_custom() {} }";
    let normalized = normalize_source(forged_with_lifetime_and_abi);
    let tokens = tokenize_including_punctuation(&normalized);
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
    let tokens = tokenize_including_punctuation(&normalize_source(unrelated_impl));
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
    let tokens = tokenize_including_punctuation(&normalize_source(nested_impl));
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "関数本体にネストされた impl Var 内の pub fn custom を検出できていない: {bodies:?}"
    );

    // 5) 文字列リテラル内の `}` による中括弧深さ追跡の誤り（PR #2212
    //    追加 P1 の是正時に自己レビューで判明）: 文字列リテラルの中身を
    //    空白化せずに `var_impl_block_bodies` へ渡すと、リテラル内の
    //    `}` を実際の中括弧として誤カウントし、本体末尾の `pub fn
    //    custom` を本体の外へ追い出してしまう。[`normalize_source`]
    //    （文字列リテラルの中身を空白化する）を通すことで正しく検出
    //    できることを固定する。
    let forged_with_brace_in_string =
        "impl Var { fn helper() { let _ = \"}\"; } pub fn custom(&self) {} }";
    let normalized = normalize_source(forged_with_brace_in_string);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "文字列リテラル内の }} を実際の中括弧と誤カウントし、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );
    // 対照実験（歴史的バイパスの実演）: 文字列リテラルの中身を関知しない
    // `strip_comments` 単独（コメント除去のみ）では、このバイパスが実際に
    // 成立する（誤って検出漏れになる）ことも併せて固定し、上記の
    // `normalize_source` による修正が実際に効いていることを裏付ける。
    let comments_only = strip_comments(forged_with_brace_in_string);
    let tokens_without_fix = tokenize_including_punctuation(&comments_only);
    let bodies_without_fix = var_impl_block_bodies(&tokens_without_fix);
    assert!(
        !bodies_without_fix
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "コメント除去のみ（文字列リテラルの中身を空白化しない）でも検出できてしまっている\
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
    let normalized = normalize_source(forged_with_char_literal_quote);
    let tokens = tokenize_including_punctuation(&normalized);
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
    let normalized = normalize_source(forged_with_char_literal_brace);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "char リテラル '}}' を実際の中括弧と誤カウントし、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );

    // 8) 文字列リテラルの中身に `//` を含む場合（第 2 ラウンド codex-review
    //    指摘 1）: `strip_comments` 単独では `"//"` を実コメントの開始と
    //    誤認し、行末（`} }`。この合成入力は 1 行）までを丸ごと削って
    //    しまうため、`pub fn custom` の宣言自体が消えて検出漏れになる。
    //    `strip_string_literal_contents` を後段に挟んでも、`strip_comments`
    //    の時点で既に失われた `pub fn custom(&self) {} }` は復元できない
    //    （2 パス方式の構造的欠陥）。[`normalize_source`] は文字列リテラル
    //    に入った時点で閉じクォートまで一気に消費するため、内部の `//`
    //    が行コメント判定へ渡ることがなく、後続の `pub fn custom` を
    //    正しく検出できる。
    let forged_with_line_comment_marker_in_string =
        "impl Var { fn s() -> &'static str { \"//\" } pub fn custom(&self) {} }";
    let normalized = normalize_source(forged_with_line_comment_marker_in_string);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "文字列リテラル中の \"//\" を実コメントと誤認し、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );
    // 対照実験（歴史的バイパスの実演）: `strip_comments` 単独では
    // `"//"` 以降（`} pub fn custom(&self) {} }` を含む）が行末まで
    // 丸ごと削られ、impl 本体の閉じ括弧が失われた不完全な入力になる
    // ため、`pub fn custom` は検出できない（誤って検出漏れになる）。
    let comments_only = strip_comments(forged_with_line_comment_marker_in_string);
    assert!(
        !comments_only.contains("pub fn custom"),
        "strip_comments 単独では文字列内の \"//\" 以降が削られないはずがない\
         （回帰テストの前提が崩れている）: {comments_only:?}"
    );
    let tokens_without_fix = tokenize_including_punctuation(&comments_only);
    let bodies_without_fix = var_impl_block_bodies(&tokens_without_fix);
    assert!(
        !bodies_without_fix
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "strip_comments 単独でも検出できてしまっている（回帰テストの前提が崩れている）: \
         {bodies_without_fix:?}"
    );

    // 9) raw 文字列（`"`／`//` を含む）・char `'"'`・ライフタイム
    //    `'static` が混在するケース（第 2 ラウンド codex-review 指摘 1
    //    が要求する追加ケース）。外側の raw 文字列は `#` 2 個
    //    （`r##"..."##`）でこのテストソース自身を記述し、内側（解析対象
    //    の合成入力）に `#` 1 個の raw 文字列（`r#"has "quotes" and //
    //    not a comment"#`）・char リテラル `'"'`・ライフタイム `'static`
    //    を埋め込む。いずれも `pub fn custom` の検出を妨げないことを
    //    固定する。
    let forged_raw_string_mix = r##"impl Var { fn s() -> &'static str { r#"has "quotes" and // not a comment"# } fn q() -> char { '"' } pub fn custom(&self) {} }"##;
    let normalized = normalize_source(forged_raw_string_mix);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "raw 文字列・char・ライフタイム混在ケースで\
         pub fn custom の検出を見逃した: {bodies:?}"
    );
}

/// [`var_impl_block_bodies_with_aliases_and_kind`] が trait impl
/// （`impl Trait for Var { ... }`）と inherent impl（`impl Var { ... }`）
/// を区別し、trait impl のメソッドは可視性修飾子（`pub`）を伴わなくても
/// 検出することを固定する回帰テスト（codex-review P1 指摘・PR #2212
/// 第 4 ラウンド）。
#[test]
fn var_impl_block_bodies_with_aliases_and_kind_detects_trait_impl_without_pub() {
    // 1) trait impl 経由で `pub` を書かずに宣言された `fn custom` は
    //    `is_trait_impl == true` として検出できる。
    let trait_impl = "impl SomeTrait for Var { fn custom(&self) {} }";
    let tokens = tokenize_including_punctuation(&normalize_source(trait_impl));
    let bodies = var_impl_block_bodies_with_aliases_and_kind(&tokens, &[]);
    assert_eq!(
        bodies.len(),
        1,
        "impl SomeTrait for Var ブロックを 1 件検出できていない: {bodies:?}"
    );
    let (body, is_trait_impl) = &bodies[0];
    assert!(
        *is_trait_impl,
        "impl SomeTrait for Var を trait impl として判定できていない"
    );
    assert!(
        tokens_declare_fn(body, "custom"),
        "trait impl 内の可視性修飾子なし fn custom を検出できていない: {body:?}"
    );
    // `tokens_declare_pub_fn`（`pub` 必須）ではこの trait impl のメソッド
    // を検出できないことも併せて固定する（否定ガード本体が
    // `is_trait_impl` で判定関数を使い分ける必要性の裏付け）。
    assert!(
        !tokens_declare_pub_fn(body, "custom"),
        "pub を伴わない trait impl のメソッドが tokens_declare_pub_fn で\
         誤って検出されている（本テストの前提が崩れている）: {body:?}"
    );

    // 2) inherent impl（`for` を含まない）で `pub` を伴わない `fn custom`
    //    は `is_trait_impl == false` のままであり、否定ガード本体は
    //    `tokens_declare_pub_fn` を適用するため違反として扱わない
    //    （非公開のヘルパーメソッドを誤検出しないことの固定）。
    let inherent_impl_without_pub = "impl Var { fn custom() {} }";
    let tokens = tokenize_including_punctuation(&normalize_source(inherent_impl_without_pub));
    let bodies = var_impl_block_bodies_with_aliases_and_kind(&tokens, &[]);
    assert_eq!(
        bodies.len(),
        1,
        "impl Var ブロックを 1 件検出できていない: {bodies:?}"
    );
    let (body, is_trait_impl) = &bodies[0];
    assert!(
        !*is_trait_impl,
        "for を含まない impl Var を trait impl と誤判定した"
    );
    assert!(
        !tokens_declare_pub_fn(body, "custom"),
        "pub を伴わない inherent impl のメソッドが tokens_declare_pub_fn で\
         誤って検出されている: {body:?}"
    );
}

/// `use_body`（`use` キーワード除去後・`;` 除去済みの残り。例:
/// `crate::Var as V`・`crate::{Tape, Var as V}`・`path::Var::{self as V}`）
/// を走査し、`Var` に対する alias をすべて収集して返す（[`find_var_
/// alias_declarations`] 専用。codex-review P1 指摘・PR #2212 その続き・
/// Bugbot 指摘 PR #2212 第 4 ラウンド。イシュー #2064）。
///
/// 構造保持トークン列（[`tokenize_including_punctuation`]。`{`／`}`／
/// `:` を個別トークンとして残す）を使う（後述の `Var::{self as V}` 判定
/// に構造情報が必要なため、識別子のみを残す [`extract_identifier_
/// tokens`] からは切り替えた）。検出パターンは 2 通り:
///
/// 1. **単純形**（`Var as V`）: `Var` の直後（トークン列上で隣接）が
///    `as` であれば、その次の識別子を alias とする。`crate::{Var as V,
///    Tape}` のような波括弧グループ内の `Var as V` も、`{`／`,` が
///    `Var` と `as` の間に挟まらない限り同じ隣接パターンで一致する。
/// 2. **`self` 再エクスポート形**（`Var::{self as V}`。Bugbot 指摘）:
///    `Var` の直後に `:`／`:`（`::`）・`{` のみが連続して現れ、その後に
///    `self` が続き、さらにその直後が `as` であれば、`self` は `Var`
///    自身を指すため `as` の次の識別子を `Var` の alias とする。`Var`
///    の直後に上記以外のトークン（`,` 等）が挟まる場合——例えば
///    `use crate::{Var, self as X}` の `self as X` は `crate` モジュール
///    自身の再エクスポートであり `Var` の alias ではない——は対象外の
///    ままとする（`Var` からの隣接判定が `,` で途切れるため）。
///
/// `use crate::Var;`（alias なし）・`use crate::Variance as V;`（`Var` とは
/// 異なる識別子）はいずれも "Var" の直後に上記いずれのパターンも続かない
/// ため誤検出しない。
fn find_var_aliases_in_use_body(use_body: &str) -> Vec<String> {
    let tokens = tokenize_including_punctuation(use_body);
    let mut aliases = Vec::new();
    for i in 0..tokens.len() {
        if tokens[i] != "Var" {
            continue;
        }
        if tokens.get(i + 1).map(String::as_str) == Some("as")
            && let Some(alias) = tokens.get(i + 2)
        {
            aliases.push(alias.clone());
            continue;
        }
        // `Var::{self as V}` 形: "Var" の直後に ":"（"::" は 1 文字ずつ
        // 2 トークンに分解される）・"{" のみが連続して現れ、その先に
        // "self" が続く場合のみ「self が Var 自身を指す」とみなす。
        let mut j = i + 1;
        let mut reached_self = false;
        loop {
            match tokens.get(j).map(String::as_str) {
                Some(":") | Some("{") => j += 1,
                Some("self") => {
                    j += 1;
                    reached_self = true;
                    break;
                }
                _ => break,
            }
        }
        if reached_self
            && tokens.get(j).map(String::as_str) == Some("as")
            && let Some(alias) = tokens.get(j + 1)
        {
            aliases.push(alias.clone());
        }
    }
    aliases
}

/// `rest`（`type` キーワード除去後の残り。例: `V = Var;`・
/// `V<'a> = Var<'a>;`・`V = crate::Var;`・`V = crate::var::Var<'a>;`・
/// `V = self::Var;`）が `Var`（パス修飾・ジェネリクス適用形を含む）を
/// 指す type alias 宣言であれば、alias 名（`type` の直後の識別子）を
/// 返す（[`find_var_alias_declarations`] 専用）。
///
/// alias 名の直後に続きうる generic parameter リスト（`<...>`）は
/// [`var_impl_block_bodies_with_aliases`] と同じ山括弧の深さ追跡で読み
/// 飛ばす——`type V<T = Foo> = Var<T>;` のようにデフォルト型引数の中に
/// `=` が現れても、山括弧の深さが 0 に戻るまではその `=` を type alias
/// 本体の代入演算子とみなさない。
///
/// **`=` 直後がパス修飾された `Var`（Bugbot 指摘・PR #2212）**: 旧実装は
/// `=` の直後のトークンが文字どおり `"Var"` かどうかしか見ておらず、
/// `type V = crate::Var;`・`type V = crate::var::Var<'a>;`・
/// `type V = self::Var;` のようにパスを経由すると `=` 直後が `crate`
/// 等になり検出漏れになっていた。本実装は `=` の後ろを `::` 区切りの
/// パスセグメント列として読み進め（`ident` の次が `::` ならさらに
/// セグメントを読み、`::` 以外〈`<` によるジェネリクス開始・`;`・入力
/// 終端等〉に達したら打ち切る）、最終セグメントが `Var` であれば alias
/// とみなす。パスの先頭が `crate`／`self`／`super` のいずれであっても
/// 先頭 `::` の絶対パス（`::fandhe_ai_autodiff::Var`）も同様に扱う。
/// パス自体の妥当性検証は行わず、最終セグメント名のみで判定する
/// （import alias 解決を行わない他関数群と同じ「名前一致のみ」の方針）。
fn type_alias_target_is_var(rest: &str) -> Option<String> {
    let tokens = tokenize_including_punctuation(rest);
    let alias_name = tokens.first()?.clone();
    let mut i = 1usize;
    let mut angle_depth = 0i32;
    while i < tokens.len() {
        match tokens[i].as_str() {
            "<" => angle_depth += 1,
            ">" => angle_depth = (angle_depth - 1).max(0),
            "=" if angle_depth == 0 => break,
            _ => {}
        }
        i += 1;
    }
    if tokens.get(i).map(String::as_str) != Some("=") {
        return None;
    }
    // `=` の直後から `::` 区切りのパスセグメント列を読み進め、最終
    // セグメント（`::` の後続が続かない直前のセグメント）を得る。
    let mut j = i + 1;
    // 絶対パス（`type V = ::fandhe_ai_autodiff::Var;`）の先頭 `::`
    // （`:` `:` の 2 トークン）を読み飛ばす。読み飛ばさないと先頭の `:` を
    // セグメントとみなして直後に打ち切り、alias を見逃す（Bugbot 指摘・
    // PR #2212）。
    if tokens.get(j).map(String::as_str) == Some(":")
        && tokens.get(j + 1).map(String::as_str) == Some(":")
    {
        j += 2;
    }
    let mut last_segment: Option<&str> = None;
    while let Some(segment) = tokens.get(j).map(String::as_str) {
        if segment == "::" {
            // 単独の `::` トークンは現れない（tokenize_including_
            // punctuation は `:` を 1 文字ずつ 2 トークンに分解する）
            // ため到達しないが、将来のトークナイザ変更に対する保険と
            // して残す。
            j += 1;
            continue;
        }
        last_segment = Some(segment);
        j += 1;
        // `:` `:`（`::` 区切り）が続く場合のみ次のセグメントへ進む。
        if tokens.get(j).map(String::as_str) == Some(":")
            && tokens.get(j + 1).map(String::as_str) == Some(":")
        {
            j += 2;
            continue;
        }
        break;
    }
    if last_segment == Some("Var") {
        Some(alias_name)
    } else {
        None
    }
}

/// `content`（[`normalize_source`] 済みの入力を想定）中に宣言された
/// `Var` 型への alias 名をすべて収集して返す（`autodiff_src_does_not_
/// alias_var`・[`var_impl_block_bodies_with_aliases`] の呼び出し元専用。
/// codex-review P1 指摘・PR #2212 その続き。イシュー #2064 §「`var_impl_
/// block_bodies` は impl ヘッダに文字どおり `Var` トークンがある場合だけ
/// 本体を検査する。同一クレート内の別ファイルで `use crate::Var as V;`
/// のように import alias を使うと否定ガードをすり抜ける」）。
///
/// 検出対象:
/// - `use ... Var as X;`（`crate::`／`super::`／`self::`／`crate::var::Var`
///   等パス不問。`use crate::{Var as X, …}` のような波括弧グループ内の
///   `Var as X` も対象——[`find_var_aliases_in_use_body`] 参照）
/// - `type X = Var...;`／`type X<...> = Var...;`
/// - `pub use ... Var as X;`（`strip_visibility_prefix` で可視性修飾を
///   読み飛ばしてから判定するため、`pub`／`pub(crate)` 等いずれも対象）
///
/// ネストした `mod x { ... }` の内部も再帰的に走査する
/// （[`split_top_level_statements`] は `mod` ブロックを波括弧ごと 1
/// ステートメントとして返すため、`mod` キーワード除去後の残りから本体
/// `{...}` を切り出して再帰する）。
fn find_var_alias_declarations(content: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    for statement in split_top_level_statements(content) {
        let after_attributes = strip_leading_attributes(&statement);
        let after_visibility = strip_visibility_prefix(after_attributes);
        if let Some(use_body) = strip_keyword_prefix(after_visibility, "use") {
            aliases.extend(find_var_aliases_in_use_body(use_body));
        } else if let Some(type_rest) = strip_keyword_prefix(after_visibility, "type")
            && let Some(alias) = type_alias_target_is_var(type_rest)
        {
            aliases.push(alias);
        }
        // ブロックスコープ（`fn`／`const`／`static`／`impl`／`trait`／
        // `mod` 等、本体に `{...}` を持つ任意のステートメント）は種類を
        // 限定せず再帰走査する（Bugbot 指摘・PR #2212 第 4 ラウンド:
        // 旧実装は `mod { ... }` のみを再帰対象としており、`fn f() {
        // use crate::Var as V; impl V { pub fn custom() {} } }` のように
        // 関数本体・impl／trait 本体の中に隠された alias 宣言を見逃して
        // いた）。ステートメント中で最初に現れる top-level `{` から最後
        // の `}` までを本体として切り出し、無条件に再帰する。`use`／
        // `type` 分岐と独立した処理であるため、`use crate::{...}` の
        // ような use 文自体が波括弧グループを持つ場合は二重に走査され
        // うるが、alias 収集は同じ alias 名を複数回 push するだけで
        // 冪等であり実害はない（本ファイル全体が採用する fail-closed
        // 方針を優先し、ステートメント種別ごとの個別実装を増やさない）。
        if let (Some(brace_start), Some(brace_end)) =
            (after_visibility.find('{'), after_visibility.rfind('}'))
            && brace_end > brace_start
        {
            aliases.extend(find_var_alias_declarations(
                &after_visibility[brace_start + 1..brace_end],
            ));
        }
    }
    aliases
}

/// `crates/autodiff/src/` 配下のいずれの `.rs` ファイルにも `Var` 型への
/// alias（`use ... Var as X`／`type X = Var...;`。可視性修飾・`mod` 内
/// ネスト・波括弧グループ import を含む）が存在しないことを固定する
/// （codex-review P1 指摘・PR #2212 その続き。イシュー #2064）。
///
/// [`autodiff_src_does_not_declare_pub_fn_custom_on_var`] の否定ガード
/// （文字どおりの `Var` トークンのみを見る）は、alias 経由で `impl V { pub
/// fn custom() {} }` のように書かれた宣言を取りこぼす死角を持つ。alias
/// import 自体を src 全体で fail-closed に禁止することで、この死角を
/// 構造的に塞ぐ（[`var_impl_block_bodies_with_aliases`] による同一
/// ファイル内 alias 名の考慮と合わせた二重の対策）。
#[test]
fn autodiff_src_does_not_alias_var() {
    let src_dir = autodiff_crate_root().join("src");
    let mut violations: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let normalized = normalize_source(content);
        for alias in find_var_alias_declarations(&normalized) {
            violations.push(format!("{}: Var の alias `{alias}`", path.display()));
        }
    });
    assert!(
        violations.is_empty(),
        "crates/autodiff/src 配下に Var 型への import alias／type alias が見つかった\
         （否定ガード `autodiff_src_does_not_declare_pub_fn_custom_on_var` の alias 経由\
         迂回を防ぐため、本ファイルでは Var の alias を作らない設計とする）: {violations:?}"
    );
}

/// [`find_var_alias_declarations`] の単体テスト（合成入力）。
/// `use crate::Var as V;`・`use crate::{Tape, Var as V};`・
/// `type V = Var;`・`mod inner { use super::Var as W; }` を alias として
/// 検出し、`use crate::Var;`（alias なし）・`use crate::Variance as V;`
/// （`Var` とは異なる識別子）はいずれも検出しないことを固定する。
#[test]
fn find_var_alias_declarations_detects_use_and_type_aliases() {
    let simple_use_alias = normalize_source("use crate::Var as V;");
    assert_eq!(
        find_var_alias_declarations(&simple_use_alias),
        vec!["V".to_string()],
        "use crate::Var as V; の alias V を検出できていない"
    );

    let grouped_use_alias = normalize_source("use crate::{Tape, Var as V};");
    assert_eq!(
        find_var_alias_declarations(&grouped_use_alias),
        vec!["V".to_string()],
        "use crate::{{Tape, Var as V}}; の alias V を検出できていない"
    );

    let type_alias = normalize_source("type V = Var;");
    assert_eq!(
        find_var_alias_declarations(&type_alias),
        vec!["V".to_string()],
        "type V = Var; の alias V を検出できていない"
    );

    let nested_mod_alias = normalize_source("mod inner { use super::Var as W; }");
    assert_eq!(
        find_var_alias_declarations(&nested_mod_alias),
        vec!["W".to_string()],
        "mod inner {{ use super::Var as W; }} の alias W を検出できていない"
    );

    let no_alias = normalize_source("use crate::Var;");
    assert!(
        find_var_alias_declarations(&no_alias).is_empty(),
        "alias を伴わない use crate::Var; を誤って alias 宣言として検出した"
    );

    let different_type = normalize_source("use crate::Variance as V;");
    assert!(
        find_var_alias_declarations(&different_type).is_empty(),
        "Var とは異なる識別子 Variance の alias 宣言を誤って Var の alias として検出した"
    );
}

/// [`type_alias_target_is_var`] がパス修飾された `Var`（`crate::Var`・
/// `crate::var::Var<'a>`・`self::Var`）を検出することを固定する回帰
/// テスト（Bugbot 指摘・PR #2212）。パス修飾なしの `type X = Var;` は
/// 既存の [`find_var_alias_declarations_detects_use_and_type_aliases`]
/// が固定済みのため、ここではパス修飾形のみを扱う。
#[test]
fn find_var_alias_declarations_detects_path_qualified_type_aliases() {
    let crate_qualified = normalize_source("type X = crate::Var;");
    assert_eq!(
        find_var_alias_declarations(&crate_qualified),
        vec!["X".to_string()],
        "type X = crate::Var; の alias X を検出できていない"
    );

    let module_and_generic_qualified = normalize_source("type X = crate::var::Var<'a>;");
    assert_eq!(
        find_var_alias_declarations(&module_and_generic_qualified),
        vec!["X".to_string()],
        "type X = crate::var::Var<'a>; の alias X を検出できていない"
    );

    let self_qualified = normalize_source("type X = self::Var;");
    assert_eq!(
        find_var_alias_declarations(&self_qualified),
        vec!["X".to_string()],
        "type X = self::Var; の alias X を検出できていない"
    );

    let absolute_path = normalize_source("type X = ::fandhe_ai_autodiff::Var;");
    assert_eq!(
        find_var_alias_declarations(&absolute_path),
        vec!["X".to_string()],
        "type X = ::fandhe_ai_autodiff::Var; の alias X を検出できていない"
    );

    let absolute_bare = normalize_source("pub(crate) type X<'a> = ::Var<'a>;");
    assert_eq!(
        find_var_alias_declarations(&absolute_bare),
        vec!["X".to_string()],
        "type X<'a> = ::Var<'a>; の alias X を検出できていない"
    );

    // 絶対パス alias 経由の `impl X { pub fn custom }` も alias 対応の
    // impl 走査で検出されること（alias 禁止ガードと impl 走査の両層）。
    let absolute_impl =
        normalize_source("type X = ::fandhe_ai_autodiff::Var;\nimpl X { pub fn custom(&self) {} }");
    let aliases = find_var_alias_declarations(&absolute_impl);
    let tokens = tokenize_including_punctuation(&absolute_impl);
    assert!(
        var_impl_block_bodies_with_aliases(&tokens, &aliases)
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "絶対パス alias 経由の impl X {{ pub fn custom }} を検出できていない"
    );

    // 対照実験: パスの最終セグメントが `Var` 以外なら検出しない
    // （`crate::Variance` は `Var` とは異なる識別子）。
    let path_qualified_different_type = normalize_source("type X = crate::Variance;");
    assert!(
        find_var_alias_declarations(&path_qualified_different_type).is_empty(),
        "crate::Variance を誤って Var の alias として検出した"
    );
}

/// [`find_var_alias_declarations`] が `mod` 以外のブロックスコープ
/// （`fn` 本体・`impl` 本体等）の中に隠された alias 宣言も検出することを
/// 固定する回帰テスト（Bugbot 指摘・PR #2212 第 4 ラウンド）。
#[test]
fn find_var_alias_declarations_detects_alias_inside_fn_body() {
    let alias_inside_fn =
        normalize_source("fn f() { use crate::Var as V; impl V { pub fn custom() {} } }");
    assert_eq!(
        find_var_alias_declarations(&alias_inside_fn),
        vec!["V".to_string()],
        "fn f() {{ use crate::Var as V; impl V {{ pub fn custom() {{}} }} }} の\
         alias V を検出できていない"
    );

    // 上記に加えて、alias 経由の impl（`impl V { pub fn custom() {} }`）
    // 自体も var_impl_block_bodies_with_aliases_and_kind との組み合わせ
    // で検出できることを確認する（本テストは alias 収集単体の固定だが、
    // 実運用パイプラインとの整合を裏付けるために合わせて検証する）。
    let tokens = tokenize_including_punctuation(&alias_inside_fn);
    let aliases = find_var_alias_declarations(&alias_inside_fn);
    let bodies = var_impl_block_bodies_with_aliases_and_kind(&tokens, &aliases);
    assert!(
        bodies
            .iter()
            .any(|(body, _is_trait_impl)| tokens_declare_pub_fn(body, "custom")),
        "fn 本体内の alias 経由 impl V {{ pub fn custom() {{}} }} を検出できていない: \
         {bodies:?}"
    );
}

/// [`find_var_aliases_in_use_body`] が `Var::{self as V}`（Bugbot 指摘・
/// PR #2212 第 4 ラウンド）を alias として検出し、`Var` と無関係な兄弟
/// 項目の `self as X`（`use crate::{Var, self as X}` の `self` は `crate`
/// モジュール自身を指し `Var` の alias ではない）を誤検出しないことを
/// 固定する回帰テスト。
#[test]
fn find_var_aliases_in_use_body_detects_self_as_form() {
    let self_as_form = "path::Var::{self as V}";
    assert_eq!(
        find_var_aliases_in_use_body(self_as_form),
        vec!["V".to_string()],
        "path::Var::{{self as V}} の alias V を検出できていない"
    );

    // 対照実験: `Var` の直後が `,` で途切れる場合、後続の `self as X` は
    // `Var` とは無関係な兄弟項目（`crate` モジュール自身の再エクスポート）
    // であり alias として検出してはならない。
    let unrelated_self_as = "crate::{Var, self as X}";
    assert!(
        find_var_aliases_in_use_body(unrelated_self_as).is_empty(),
        "crate::{{Var, self as X}} の self as X を誤って Var の alias として検出した: \
         {:?}",
        find_var_aliases_in_use_body(unrelated_self_as)
    );

    // 単純形（`Var as V`）は引き続き検出できる（回帰確認）。
    let simple_form = "crate::Var as V";
    assert_eq!(
        find_var_aliases_in_use_body(simple_form),
        vec!["V".to_string()],
        "crate::Var as V の alias V を検出できていない"
    );
}

/// [`var_impl_block_bodies_with_aliases`] が、同一ファイル内で検出済みの
/// `Var` alias 名（[`find_var_alias_declarations`] の出力）をヘッダー
/// 判定に使うことで、`use crate::Var as V; impl V<'_> { pub fn custom()
/// {} }` のように alias 経由で書かれた impl ブロックも検出できることを
/// 固定する（codex-review P1 指摘・PR #2212 その続き。イシュー #2064）。
#[test]
fn var_impl_block_bodies_with_aliases_detects_impl_via_use_alias() {
    let forged_alias_impl = "use crate::Var as V; impl V<'_> { pub fn custom() {} }";
    let normalized = normalize_source(forged_alias_impl);
    let aliases = find_var_alias_declarations(&normalized);
    assert_eq!(
        aliases,
        vec!["V".to_string()],
        "use crate::Var as V; から alias V を検出できていない"
    );

    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies_with_aliases(&tokens, &aliases);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "alias V 経由の impl V {{ pub fn custom() {{}} }} を検出できていない: {bodies:?}"
    );

    // 対照実験: alias 名を渡さない従来版（`var_impl_block_bodies`）では
    // ヘッダーに文字どおりの `Var` トークンが無いため検出できない
    // （alias 未考慮の死角が実在することの裏付け）。
    let bodies_without_aliases = var_impl_block_bodies(&tokens);
    assert!(
        !bodies_without_aliases
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "alias 名を考慮しない従来版が検出できてしまっている（回帰テストの前提が崩れている）: \
         {bodies_without_aliases:?}"
    );
}

/// [`var_impl_block_bodies`] のヘッダー走査が、where 節内の const
/// ジェネリクス式（`[(); { N }]` のような、`[`／`]` の中に `{`／`}` を
/// 含む構造）を正しく読み飛ばし、実際の impl 本体を取りこぼさないことを
/// 固定する回帰テスト（Bugbot 指摘・PR #2212 第 2 ラウンド指摘 3）:
/// 旧実装は「最初に現れた `{` を本体開始」とみなす単純な走査だったため、
/// where 節内の const ブロックが持つ `{` を本体開始と誤認してヘッダー
/// 走査を早期終了し、本来の本体末尾にある `pub fn custom` を見逃して
/// いた。
#[test]
fn var_impl_block_bodies_handles_where_clause_const_generic_block() {
    let forged_where_const_block =
        "impl<const N: usize> Var where [(); { N }]: Sized { pub fn custom() {} }";
    let normalized = normalize_source(forged_where_const_block);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert_eq!(
        bodies.len(),
        1,
        "where 節内に const ジェネリクス式（[(); {{ N }}]）を持つ impl Var ブロックを\
         1 件検出できていない: {bodies:?}"
    );
    assert!(
        tokens_declare_pub_fn(&bodies[0], "custom"),
        "where 節内の const ブロックの {{ を本体開始と誤認し、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );

    // 同様に `(...)`（括弧）の深さもヘッダー走査中に追跡する必要がある
    // ことの確認: where 節に関数ポインタ型の引数リスト（`(...)`）を含む
    // ケースでも本体開始 `{` を正しく特定できる。
    let forged_where_fn_pointer_bound =
        "impl Var where fn(usize) -> usize: Sized { pub fn custom() {} }";
    let normalized = normalize_source(forged_where_fn_pointer_bound);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert_eq!(
        bodies.len(),
        1,
        "where 節内に関数ポインタ型境界を持つ impl Var ブロックを 1 件検出できていない: \
         {bodies:?}"
    );
    assert!(
        tokens_declare_pub_fn(&bodies[0], "custom"),
        "where 節内の関数ポインタ型境界の括弧を誤って扱い、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );
}

/// [`var_impl_block_bodies`] のヘッダー走査が、const ジェネリクス
/// パラメータの既定値式に含まれる比較演算子由来の `>`（`{ 3 > 2 }`
/// 等）を山括弧の閉じと誤認しないことを固定する回帰テスト（Bugbot
/// 指摘・PR #2212 第 4 ラウンド）: 旧実装はジェネリクスリストを事前に
/// 読み飛ばす別ループが `<`／`>` トークンのみで深さを数えていたため、
/// 1 個目の const パラメータの既定値式内の `>` で早期に山括弧の閉じと
/// 誤認し、走査位置が本来のジェネリクスリスト終端より手前に残ってしまう。
/// その結果、2 個目の const パラメータの既定値式が持つ `{` を本体開始
/// と誤認してヘッダー走査を打ち切り、実際の本体（`pub fn custom` を
/// 含む）を取りこぼす。
#[test]
fn var_impl_block_bodies_handles_const_generic_default_value_comparison() {
    let forged_const_generic_comparison =
        "impl<const B: bool = { 3 > 2 }, const N: usize = { 1 }> Var { pub fn custom() {} }";
    let normalized = normalize_source(forged_const_generic_comparison);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert_eq!(
        bodies.len(),
        1,
        "const ジェネリクス既定値式の比較演算子由来の `>` を含む impl Var ブロックを\
         1 件検出できていない: {bodies:?}"
    );
    assert!(
        tokens_declare_pub_fn(&bodies[0], "custom"),
        "const ジェネリクス既定値式内の `>` を山括弧の閉じと誤認し、\
         2 個目の既定値式の {{ を本体開始と誤認して pub fn custom の検出を\
         見逃した: {bodies:?}"
    );
}

/// [`normalize_source`] 自体の単体テスト（`var_impl_block_bodies` 等の
/// 本番パイプライン経由の固定に加え、正規化そのものの性質を直接固定
/// する。第 2 ラウンド codex-review 指摘 1）。
#[test]
fn normalize_source_handles_all_literal_forms() {
    // 行コメント・ブロックコメント（ネスト対応）は空白へ置換される。
    let with_comments = "let a = 1; // trailing\n/* outer /* inner */ still comment */let b = 2;";
    let normalized = normalize_source(with_comments);
    assert!(!normalized.contains("trailing"));
    assert!(!normalized.contains("still comment"));
    assert!(normalized.contains("let a = 1;"));
    assert!(normalized.contains("let b = 2;"));

    // 通常文字列: エスケープシーケンスを含んでいても閉じクォートを
    // 正しく検出し、中身は空白化されるが開始・終了のクォートは残る。
    let with_string = r#"let s = "a\"b // not a comment"; let t = 1;"#;
    let normalized = normalize_source(with_string);
    assert!(normalized.contains("let t = 1;"));
    assert!(!normalized.contains("not a comment"));
    // クォート自体は保持される（`skip_fn_declaration_qualifiers` の
    // ABI 文字列リテラル判定との整合のため）。
    let quote_count = normalized.chars().filter(|&c| c == '"').count();
    assert_eq!(
        quote_count, 2,
        "開始・終了のクォートが保持されていない: {normalized:?}"
    );

    // raw 文字列（`#` の個数が異なる場合も終端を正しく判定する）。
    let with_raw_string =
        r###"let s = r##"contains "one hash" -> "# still inside"##; let t = 2;"###;
    let normalized = normalize_source(with_raw_string);
    assert!(normalized.contains("let t = 2;"));
    assert!(!normalized.contains("still inside"));

    // バイト文字列・バイト char。
    let with_byte_literals = r#"let b = b"raw // bytes"; let c = b'\n'; let d = 1;"#;
    let normalized = normalize_source(with_byte_literals);
    assert!(normalized.contains("let d = 1;"));
    assert!(!normalized.contains("raw // bytes"));

    // raw バイト文字列。
    let with_raw_byte_string = r##"let b = br#"raw "// bytes"#; let d = 1;"##;
    let normalized = normalize_source(with_raw_byte_string);
    assert!(normalized.contains("let d = 1;"));
    assert!(!normalized.contains("bytes"));

    // char リテラルと識別子途中の `r`／`b`（`for`／`bar` 等）を混同
    // しない: raw 文字列・バイト文字列の接頭辞判定は「直前が識別子構成
    // 文字でない」場合に限る。
    let with_ident_r_and_b = "for bar in 0..1 { let r = 1; let b = 2; }";
    let normalized = normalize_source(with_ident_r_and_b);
    assert_eq!(normalized, with_ident_r_and_b);

    // ライフタイムと char リテラルの判別。
    let with_lifetime_and_char =
        "fn f<'a>(c: char) -> &'a str { if c == '\\'' { \"q\" } else { \"n\" } }";
    let normalized = normalize_source(with_lifetime_and_char);
    assert!(
        normalized.contains("'a"),
        "ライフタイムが保持されていない: {normalized:?}"
    );

    // unicode escape char リテラル（`'\u{7b}'` は `{` を表す）の中身が
    // 空白化され、実際の中括弧深さ追跡を誤らせないことを確認する。
    let with_unicode_escape_char =
        "impl Var { fn h() -> char { '\\u{7b}' } pub fn custom(&self) {} }";
    let normalized = normalize_source(with_unicode_escape_char);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "unicode escape char リテラル中の {{ を実際の中括弧と誤カウントし、\
         pub fn custom の検出を見逃した: {bodies:?}"
    );

    // C 文字列（`c"..."`）: 中身は通常文字列と同様に空白化されるが
    // 開始・終了のクォートと `c` 接頭辞は保持される（Bugbot 指摘・
    // PR #2212 第 4 ラウンド）。
    let with_c_string = r#"let s = c"raw // not a comment"; let t = 3;"#;
    let normalized = normalize_source(with_c_string);
    assert!(normalized.contains("let t = 3;"));
    assert!(!normalized.contains("not a comment"));
    assert!(
        normalized.contains("c\""),
        "C 文字列の `c` 接頭辞が保持されていない: {normalized:?}"
    );

    // raw C 文字列（`cr#"..."#`）: 内部に `"`／`//` を含んでいても、
    // 単一の raw リテラルとして消費され後続のソースを飲み込まない
    // ことを固定する。旧実装は `c` の直後の `r` を識別子継続とみなし
    // `cr#"..."#` を認識できず、内部の `"` が偽の通常文字列を開いて
    // `pub fn custom` を隠していた（Bugbot 指摘・PR #2212 第 4
    // ラウンド）。
    let forged_raw_c_string = r####"impl Var { fn s() -> &'static [u8] { cr#"has "quotes" and // not a comment"# } pub fn custom(&self) {} }"####;
    let normalized = normalize_source(forged_raw_c_string);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies(&tokens);
    assert!(
        bodies
            .iter()
            .any(|body| tokens_declare_pub_fn(body, "custom")),
        "raw C 文字列（cr#\"...\"#）中の \"／// を実際の文字列・コメント境界と\
         誤認し、pub fn custom の検出を見逃した: {bodies:?}"
    );

    // 識別子途中の `c`（`abc"..."` の `c`）は C 文字列の接頭辞と
    // 誤認しない（`prev_is_ident` 判定の回帰確認）。
    let with_ident_c = "let abc = 1; let d = 2;";
    let normalized = normalize_source(with_ident_c);
    assert_eq!(normalized, with_ident_c);
}

/// [`normalize_source`] のエスケープされた閉じクォート
/// （`'\''`。バックスラッシュ＋シングルクォートで単一引用符を表す char
/// リテラル）の後続トークン復元を、本番経路（`normalize_source` →
/// `tokenize_including_punctuation` → `var_impl_block_bodies_with_aliases_
/// and_kind`）で固定する回帰テスト。旧実装はエスケープ本体の先頭 1 文字
/// （`\'` の `'`）を閉じクォートと誤認識し、後続の実際の閉じクォートから
/// 再同期していたため、`('\'','"')` のようなタプルリテラル直後の `"` を
/// 文字列リテラルの開始と誤認識して後続コード（`impl CustomExt for Var
/// { fn custom(&self) {} }` 等）を丸ごと呑み込み、否定ガードの検出漏れを
/// 招いていた。`b'\''`（バイト char 版）も同型の迂回経路として併せて
/// 固定する。
#[test]
fn normalize_source_recovers_tokens_after_escaped_quote_char_literal() {
    let forged_via_char_literal = "const Q: (char, char) = ('\\'','\"'); \
         impl CustomExt for Var { fn custom(&self) {} } const S: &str = \"\";";
    let normalized = normalize_source(forged_via_char_literal);
    let tokens = tokenize_including_punctuation(&normalized);
    let bodies = var_impl_block_bodies_with_aliases_and_kind(&tokens, &[]);
    assert!(
        bodies
            .iter()
            .any(|(body, _is_trait_impl)| tokens_declare_fn(body, "custom")),
        "`'\\''`（エスケープされた閉じクォート）直後の `('\\'','\"')` の `\"` を\
         文字列リテラル開始と誤認識し、後続の impl CustomExt ブロック内の\
         fn custom を検出できなかった: normalized={normalized:?}, bodies={bodies:?}"
    );

    // バイト char 版（`b'\''`）も同型の迂回経路として固定する。
    let forged_via_byte_char_literal = "const Q: (u8, char) = (b'\\'','\"'); \
         impl CustomExt for Var { fn custom(&self) {} } const S: &str = \"\";";
    let normalized_byte = normalize_source(forged_via_byte_char_literal);
    let tokens_byte = tokenize_including_punctuation(&normalized_byte);
    let bodies_byte = var_impl_block_bodies_with_aliases_and_kind(&tokens_byte, &[]);
    assert!(
        bodies_byte
            .iter()
            .any(|(body, _is_trait_impl)| tokens_declare_fn(body, "custom")),
        "`b'\\''`（エスケープされた閉じクォートを含むバイト char リテラル）\
         直後の後続コードを誤って呑み込み、fn custom の検出を見逃した: \
         normalized={normalized_byte:?}, bodies={bodies_byte:?}"
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

    // 11) [`normalize_source`] はブロックコメント（ネスト対応）を除去する
    //     ため、実定義より前に置かれた偽トレイト定義（ブロックコメント
    //     内）は `extract_trait_body`／`extract_supertrait_bound_tokens`
    //     の走査対象から外れる（codex-review 指摘・PR #2212 その 6）。
    let forged_comment_then_real = "/* pub trait CustomFunction { fn evil(&self); } */\n\
         pub trait CustomFunction: Send + Sync + 'static {\n    fn forward(&self);\n}\n";
    let normalized = normalize_source(forged_comment_then_real);
    assert!(
        !normalized.contains("evil"),
        "ブロックコメント内の偽トレイト定義が除去されていない: {normalized}"
    );
    let bounds = extract_supertrait_bound_tokens(&normalized, "CustomFunction");
    assert!(
        bounds.iter().any(|b| b == "Send")
            && bounds.iter().any(|b| b == "Sync")
            && bounds.iter().any(|b| b == "'static"),
        "ブロックコメント除去後は実定義の supertrait 境界を正しく抽出できるはず: {bounds:?}"
    );
    let body = extract_trait_body(&normalized, "CustomFunction");
    assert!(
        !body.contains("evil") && body.contains("forward"),
        "抽出したトレイト本体が実定義（forward のみ）ではなく偽定義（evil）を含んでいる: {body}"
    );

    // 11b) 第 2 ラウンド codex-review 指摘 2: 文字列リテラルの中身に
    //      偽トレイト定義（`pub trait CustomFunction { ... }` というテキ
    //      スト）を埋め込み、実定義より前に置いた場合。`strip_comments`
    //      はコメントしか除去しないため文字列の中身はそのまま残り、
    //      `extract_trait_body`／`extract_supertrait_bound_tokens` の
    //      `content.find("pub trait CustomFunction")` が文字列内の偽定義
    //      を最初の一致として抜き出してしまう（denylist・allowlist・
    //      supertrait 境界の検査を偽定義に対して行い、実定義の検査を
    //      素通りできてしまう）。[`normalize_source`] は文字列リテラル
    //      の中身を空白化するため、この偽装テキスト自体が走査対象から
    //      消え、実定義のみが抽出される。
    let forged_fake_trait_in_string = "const S: &str = \"pub trait CustomFunction { fn evil(&self); }\";\n\
         pub trait CustomFunction: Send + Sync + 'static {\n    fn forward(&self);\n}\n";
    let normalized = normalize_source(forged_fake_trait_in_string);
    let bounds = extract_supertrait_bound_tokens(&normalized, "CustomFunction");
    assert!(
        bounds.iter().any(|b| b == "Send")
            && bounds.iter().any(|b| b == "Sync")
            && bounds.iter().any(|b| b == "'static"),
        "文字列リテラル内の偽トレイト定義を実定義と誤って抽出している（境界抽出が空か\
         偽定義由来のはず）: {bounds:?}"
    );
    let body = extract_trait_body(&normalized, "CustomFunction");
    assert!(
        !body.contains("evil") && body.contains("forward"),
        "抽出したトレイト本体が実定義（forward のみ）ではなく文字列内の偽定義（evil）を\
         含んでいる: {body}"
    );
    // 対照実験（歴史的バイパスの実演）: コメント除去のみでは文字列の
    // 中身が保持されたままのため、`content.find` は文字列内の偽定義を
    // 最初の一致として拾ってしまう。
    let comments_only = strip_comments(forged_fake_trait_in_string);
    let bounds_without_fix = extract_supertrait_bound_tokens(&comments_only, "CustomFunction");
    assert!(
        bounds_without_fix.is_empty(),
        "strip_comments 単独では文字列内の偽定義（supertrait 境界を持たない）が\
         最初の一致として抽出され、境界抽出は空になるはず（回帰テストの前提が崩れている）: \
         {bounds_without_fix:?}"
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
