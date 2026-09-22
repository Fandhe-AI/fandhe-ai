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
/// 境界> { ... }` の**ヘッダー行のみ**（`pub trait CustomFunction` から
/// 開始 `{` の手前まで、メソッド本体を含まない）を抜き出す（イシュー
/// #2064 §12.5 (b)。codex-review 指摘・PR #2212 その 2: `extract_trait_body`
/// が返すヘッダー＋メソッド本体全体に対して `contains` 判定すると、
/// supertrait 境界から `Send`／`Sync`／`'static` を削除しても、同じ文字列
/// がメソッドシグネチャ側に偶然残っていれば検査を素通りしてしまう。
/// 必須境界の有無判定はヘッダーのみに限定して行う）。トレイト定義が
/// 見つからない場合は空文字列を返す（呼び出し側で検出不能を明示的に
/// fail させるため）。
fn extract_trait_header(content: &str, trait_name: &str) -> String {
    let needle = format!("pub trait {trait_name}");
    let Some(start) = content.find(&needle) else {
        return String::new();
    };
    let after_needle = &content[start..];
    let Some(brace_offset) = after_needle.find('{') else {
        return String::new();
    };
    content[start..start + brace_offset].to_string()
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

    let header = extract_trait_header(&content, "CustomFunction");
    assert!(
        !header.is_empty(),
        "src/custom.rs から `pub trait CustomFunction` のヘッダーを抽出できなかった\
         （テスト自体が検査対象を見失っている。ファイル構成が変わっていないか確認）"
    );
    for required_bound in ["Send", "Sync"] {
        assert!(
            contains_identifier(&header, required_bound),
            "CustomFunction trait のヘッダーに必須の supertrait 境界 {required_bound} が\
             見つからない（§12.4「'static 境界により &Tape・Var<'t> を捕捉できない」契約違反）"
        );
    }
    // `'static` はアポストロフィを含むライフタイムトークンのため
    // `contains_identifier`（英数字／`_` の識別子境界判定）はそのまま
    // 適用できない。ヘッダーのみへ限定済みのため単純な部分文字列一致で
    // 十分（メソッド本体を含まないので誤検出の余地がない）。
    assert!(
        header.contains("'static"),
        "CustomFunction trait のヘッダーに必須の supertrait 境界 'static が\
         見つからない（§12.4「'static 境界により &Tape・Var<'t> を捕捉できない」契約違反）"
    );

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
    let no_comments: String = content
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for line in no_comments.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("use ") {
            assert!(
                !trimmed.contains(" as "),
                "src/custom.rs の use 文にエイリアス（`use ... as ...`）が含まれている: \
                 {trimmed}（禁止型を別名で混入させる経路になりうるため、本ファイルでは\
                 エイリアス import を使わない設計とする）"
            );
        }
        if trimmed.starts_with("type ") {
            for forbidden in FORBIDDEN_IDENTIFIERS {
                assert!(
                    !contains_identifier(trimmed, forbidden),
                    "src/custom.rs の type エイリアス宣言が禁止型 {forbidden} を参照している: \
                     {trimmed}"
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
