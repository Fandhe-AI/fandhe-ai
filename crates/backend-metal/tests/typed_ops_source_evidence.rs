//! `backend-metal` の `TypedOps` dtype 結線に関する文字列証跡検査
//! （イシュー #1705）。
//!
//! `crates/backend-metal/src/{ops.rs,typed_f16.rs,lib.rs}` はいずれも
//! `#[cfg(target_os = "macos")]` 限定のため、Linux CI では型検査（`cargo
//! check --target aarch64-apple-darwin`）はできても `cargo test` 実行は
//! できない。本ファイルはソースを文字列として読み込むだけで cfg に触れず、
//! Linux（GitHub ホステッド CI）でも常時実行できる（`tests/
//! elementwise_gemm_bias_act_source_evidence.rs`・`tests/
//! shader_source_evidence.rs` と同型の設計判断）。
//!
//! 検査するのは次の 2 点の恒久事実（設計 `docs/backend-dtype-dispatch-
//! design.md` §5・§14）:
//!
//! 1. **`typed_ops_f64` は恒久 `Unsupported`（accessor 既定 `None`）**:
//!    `ops.rs` に `fn typed_ops_f64(` のオーバーライドが存在せず、
//!    クレート内に `impl TypedOps<f64>` が存在しない
//! 2. **`typed_ops_f16` は `dispatch_f16_auto_unverified` へ内部結線
//!    済み**: `ops.rs` に `fn typed_ops_f16(` が存在し、`typed_f16.rs`
//!    が `dispatch_f16_auto_unverified` を呼び出し、暗黙の f32
//!    フォールバック（`dispatch_auto`／`dispatch_backend_auto` の
//!    メソッド呼び出し）を経由しない
//!
//! doc comment 中の言及（本ファイル自身の説明文や `typed_f16.rs` の
//! モジュール doc が `dispatch_auto` 等の語を含みうる）で誤検出しない
//! よう、`//` で始まる行を除去してから検査する。

use std::fs;
use std::path::Path;

/// 対象ソースを読み込み、`//` コメント行（先頭の空白を除去した後に `//`
/// で始まる行）を取り除いた本文を返す。doc comment（`///`／`//!`）内の
/// 語句が検査対象へ混入するのを防ぐ（`#[doc(hidden)]` 属性行自体は
/// `//` で始まらないため除去されない）。
fn read_source_without_comment_lines(relative_path: &str) -> String {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join(relative_path);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    content
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn typed_ops_f64_accessor_is_not_overridden_in_ops_rs() {
    let ops_src = read_source_without_comment_lines("src/ops.rs");
    assert!(
        !ops_src.contains("fn typed_ops_f64("),
        "ops.rs は typed_ops_f64 をオーバーライドしてはならない（恒久 Unsupported。既定 None のまま）"
    );
}

#[test]
fn no_typed_ops_f64_impl_exists_in_crate() {
    for path in ["src/ops.rs", "src/typed_f16.rs", "src/lib.rs"] {
        let src = read_source_without_comment_lines(path);
        assert!(
            !src.contains("TypedOps<f64>"),
            "{path} は TypedOps<f64> を実装してはならない（設計 §5・§14: Metal f64 は恒久 Unsupported）"
        );
    }
}

#[test]
fn typed_ops_f16_accessor_is_wired_in_ops_rs() {
    let ops_src = read_source_without_comment_lines("src/ops.rs");
    assert!(
        ops_src.contains("fn typed_ops_f16("),
        "ops.rs は typed_ops_f16 accessor をオーバーライドしているはず"
    );
}

#[test]
fn typed_f16_module_is_declared_in_lib_rs() {
    let lib_src = read_source_without_comment_lines("src/lib.rs");
    assert!(
        lib_src.contains("mod typed_f16;"),
        "lib.rs は typed_f16 モジュールを宣言しているはず"
    );
}

#[test]
fn typed_f16_gemm_calls_dispatch_f16_auto_unverified_without_implicit_f32_fallback() {
    let src = read_source_without_comment_lines("src/typed_f16.rs");
    assert!(
        src.contains(".dispatch_f16_auto_unverified("),
        "typed_f16.rs::gemm は dispatch_f16_auto_unverified を呼ぶはず（結線証跡）"
    );
    assert!(
        !src.contains(".dispatch_auto("),
        "typed_f16.rs は f32 経路 dispatch_auto を呼んではならない（暗黙フォールバック禁止）"
    );
    assert!(
        !src.contains(".dispatch_backend_auto("),
        "typed_f16.rs は f32 経路 dispatch_backend_auto を呼んではならない（暗黙フォールバック禁止）"
    );
}
