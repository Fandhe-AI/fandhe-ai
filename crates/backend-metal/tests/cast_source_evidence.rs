//! `backend-metal` の cast カーネル（イシュー #1751・親 #1613）に
//! 関する文字列証跡検査。
//!
//! `crates/backend-metal/src/{ops.rs,cast.rs}` はいずれも
//! `#[cfg(target_os = "macos")]` 限定のため、Linux CI では型検査
//! （`cargo check --target aarch64-apple-darwin`）はできても `cargo
//! test` 実行はできない。本ファイルはソースを文字列として読み込む
//! だけで cfg に触れず、Linux（GitHub ホステッド CI）でも常時実行
//! できる（`tests/typed_ops_source_evidence.rs` と同型の設計判断）。
//!
//! 検査する恒久事実（`docs/tensor-core-cast-design.md` §3.3・
//! `shaders/cast.metal` 冒頭コメント参照）:
//!
//! 1. `shaders/cast.metal` が 6 カーネル・境界検査・NaN 判定の bit
//!    パターン方式・`double` 非含有を満たす
//! 2. `ops.rs`／`cast.rs` に `fn cast_f32_to_f64(`／`fn cast_f64_to_f32(`
//!    が現れない（f64 2 方向の恒久 `Unsupported` フォールバックの
//!    固定。CPU の 8 方向実装〈`backend-cpu::cast`〉との非対称性を
//!    機械的に検証する）

use std::fs;
use std::path::Path;

/// `typed_ops_source_evidence.rs::read_source_without_comment_lines`
/// と同型（`//` コメント行除去。doc comment 中の言及による誤検出を
/// 防ぐ）。
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

fn cast_shader_source() -> String {
    read_source_without_comment_lines("src/shaders/cast.metal")
}

#[test]
fn shader_has_metal_stdlib_header() {
    let src = cast_shader_source();
    assert!(src.contains("#include <metal_stdlib>"));
    assert!(src.contains("using namespace metal;"));
}

#[test]
fn shader_declares_all_six_kernels_with_expected_buffer_indices() {
    let src = cast_shader_source();
    for (name, in_ty, out_ty) in [
        ("cast_f32_to_i32", "const float", "int"),
        ("cast_f32_to_i64", "const float", "long"),
        ("cast_f32_to_bool", "const float", "uchar"),
        ("cast_i32_to_f32", "const int", "float"),
        ("cast_i64_to_f32", "const long", "float"),
        ("cast_bool_to_f32", "const uchar", "float"),
    ] {
        assert!(
            src.contains(&format!("kernel void {name}(")),
            "missing kernel: {name}"
        );
        assert!(
            src.contains(&format!("device {in_ty}* in [[buffer(0)]]")),
            "{name}: input buffer index/type mismatch"
        );
        assert!(
            src.contains(&format!("device {out_ty}* out [[buffer(1)]]")),
            "{name}: output buffer index/type mismatch"
        );
    }
    // numel は全カーネル共通で buffer index 2 の constant uint&。
    assert_eq!(
        src.matches("constant uint& numel [[buffer(2)]]").count(),
        6,
        "all 6 kernels must declare numel at buffer index 2"
    );
}

#[test]
fn shader_retains_bounds_check_in_all_kernels() {
    let src = cast_shader_source();
    assert_eq!(
        src.matches("if (gid >= numel) {").count(),
        6,
        "all 6 kernels must retain the REQ-8 bounds check"
    );
}

#[test]
fn nan_sensitive_kernels_use_bit_pattern_checks_not_isnan() {
    let src = cast_shader_source();
    // NaN 判定を要する 3 カーネル（f32 が入力側）のみ `as_type<uint>`
    // を使う。`isnan()` は一切使わない（モジュール doc 規則 1）。
    assert!(src.contains("as_type<uint>(v)"));
    assert!(src.contains("as_type<uint>(in[gid])"));
    assert!(!src.contains("isnan"));
}

#[test]
fn saturating_int_casts_use_literal_bounds() {
    let src = cast_shader_source();
    assert!(src.contains("2147483647"));
    assert!(src.contains("(-2147483647 - 1)"));
    assert!(src.contains("9223372036854775807L"));
    assert!(src.contains("(-9223372036854775807L - 1)"));
}

#[test]
fn shader_does_not_use_double_type() {
    // MSL は `double` 非対応。コメント除去後の本文に `double` という
    // 字面が一切現れないことを機械的に固定する（f64 方向は本ファイル
    // に実装しないため、そもそも登場しないはずの不変条件）。
    let src = cast_shader_source();
    assert!(
        !src.contains("double"),
        "cast.metal must not use the double type (MSL has no double support)"
    );
}

#[test]
fn shader_bool_directions_use_uchar() {
    let src = cast_shader_source();
    assert!(src.contains("device uchar* out [[buffer(1)]]"));
    assert!(src.contains("device const uchar* in [[buffer(0)]]"));
}

/// `ops.rs`／`cast.rs` のいずれにも f64 2 方向のオーバーライドが
/// 存在しない（恒久 `Unsupported` フォールバックの固定。ファイル自体が
/// 存在しない場合は cast.rs 未実装の signal になるため、存在確認込みで
/// 検査する）。
#[test]
fn f64_directions_are_not_overridden_anywhere_in_crate() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in ["src/ops.rs", "src/cast.rs"] {
        let path = manifest_dir.join(relative);
        assert!(path.exists(), "expected file to exist: {relative}");
        let src = read_source_without_comment_lines(relative);
        assert!(
            !src.contains("fn cast_f32_to_f64("),
            "{relative} must not override cast_f32_to_f64 (double not supported by MSL)"
        );
        assert!(
            !src.contains("fn cast_f64_to_f32("),
            "{relative} must not override cast_f64_to_f32 (double not supported by MSL)"
        );
    }
}

/// `ops.rs` が `cast_ops` accessor を `Some(self)` へオーバーライド
/// していることの文字列証跡（6 方向のみ実装済みでも accessor 自体は
/// `Some` を返す契約。設計 §2.3「意図した挙動変更」と対になる）。
#[test]
fn ops_rs_overrides_cast_ops_accessor() {
    let src = read_source_without_comment_lines("src/ops.rs");
    assert!(src.contains("fn cast_ops(&self)"));
}
