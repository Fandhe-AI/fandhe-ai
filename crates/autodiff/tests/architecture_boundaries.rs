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
/// `#[cfg(test)]` ブロック内のテストコード自身（`.unwrap()` を使う
/// テストアサーション）は対象外とする——本テストの対象は「本番経路
/// （`#[cfg(test)]` の外側）」のみであり、`mod tests { ... }` 以降は走査
/// しない（同ファイル内でテストモジュールが末尾にまとまっている構成を
/// 前提とする単純な行ベース切り出し）。
#[test]
fn eval_rs_has_no_panic_macros_outside_test_module() {
    let eval_rs = autodiff_crate_root().join("src/eval.rs");
    let content = read_to_string_or_panic(&eval_rs);
    let before_test_module = content
        .split("#[cfg(test)]")
        .next()
        .expect("split は必ず 1 要素以上を返す");
    // コメント行（`//`／`///`／`//!`）は規約を説明する散文として
    // "unwrap()" 等の語を含みうるため、実コードのみを対象とするよう
    // 行頭が `//` の行を除外する（誤検知防止）。
    let production_code: String = before_test_module
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    for forbidden in ["panic!(", "unwrap()", "expect(", "unreachable!("] {
        assert!(
            !production_code.contains(forbidden),
            "eval.rs の本番経路（#[cfg(test)] より前）に {forbidden} が含まれている\
             （eval.rs 非 panic 化の契約違反。docs/fusion-graph-design.md §2.5）"
        );
    }
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
