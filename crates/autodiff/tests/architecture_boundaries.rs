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

    let supertrait_bounds = extract_supertrait_bound_tokens(&content, "CustomFunction");
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

    let trait_body = extract_trait_body(&content, "CustomFunction");
    assert!(
        !trait_body.is_empty(),
        "src/custom.rs から `pub trait CustomFunction` 本体を抽出できなかった\
         （テスト自体が検査対象を見失っている。ファイル構成が変わっていないか確認）"
    );
    let signatures_only: String = trait_body
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

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
    // `use` もひと続きの文字列として `.contains(" as ")` 判定できる）。
    let no_comments: String = content
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for statement in split_top_level_statements(&no_comments) {
        if statement.starts_with("use ") {
            assert!(
                !statement.contains(" as "),
                "src/custom.rs の use 文にエイリアス（`use ... as ...`）が含まれている: \
                 {statement}（複数行の波括弧 import を含む。禁止型を別名で混入させる経路に\
                 なりうるため、本ファイルではエイリアス import を使わない設計とする）"
            );
        }
        // `pub`／`pub(crate)` 等の可視性修飾を読み飛ばしてから `type ` を
        // 判定する（旧実装は `type ` 始まりの行しか見ておらず `pub type`
        // を取りこぼしていた）。
        let after_visibility = strip_visibility_prefix(&statement);
        if after_visibility.starts_with("type ") {
            for forbidden in FORBIDDEN_IDENTIFIERS {
                assert!(
                    !contains_identifier(after_visibility, forbidden),
                    "src/custom.rs の type エイリアス宣言が禁止型 {forbidden} を参照している: \
                     {statement}"
                );
            }
        }
    }
}

/// `crates/autodiff/src/var.rs` に `pub fn custom(`／`pub fn custom<`
/// 宣言が存在しないことを固定する（イシュー #2064 §12.5 (b) 第 3 項）。
/// facade は `Var` を型ごと再エクスポートしているため、`Var::custom` が
/// 生えると `Tape::custom` の facade 転送メソッド不在ガード
/// （`crates/facade/tests/api_surface.rs::
/// facade_tape_does_not_expose_custom_forwarding_method`）を経由せずに
/// 承認 (b) 前の到達経路が生まれてしまう（`docs/autodiff-custom-
/// function-decision.md` §12.4「入口」の設計根拠）。
#[test]
fn var_rs_does_not_declare_pub_fn_custom() {
    let var_rs = autodiff_crate_root().join("src/var.rs");
    let content = read_to_string_or_panic(&var_rs);
    let no_comments: String = content
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !no_comments.contains("pub fn custom(") && !no_comments.contains("pub fn custom<"),
        "src/var.rs に pub fn custom 宣言が見つかった\
         （§12.5 (b) 未承認のまま Var 経由の到達口を設けてしまっている）"
    );
}

/// codex-review 指摘（PR #2212 その 3）が挙げた 2 つのバイパスシナリオ
/// （ジェネリクス／where 節側の偽装トークン・複数行または可視性付き
/// alias）を、実際に新実装が検出できることを固定する回帰テスト。
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
    //    `.contains(" as ")` 判定の対象になる。
    let forged_use = "use crate::{\n    BackendOps as Tensor,\n};\n";
    let statements = split_top_level_statements(forged_use);
    assert_eq!(
        statements.len(),
        1,
        "複数行 use 文が 1 文として結合されていない: {statements:?}"
    );
    assert!(statements[0].starts_with("use "));
    assert!(statements[0].contains(" as "));

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
}
