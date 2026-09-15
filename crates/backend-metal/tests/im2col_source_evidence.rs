//! イシュー #1768: `crates/backend-metal/src/shaders/im2col.metal` が
//! 想定どおりのカーネル・数値契約を実装していることを文字列パターンで
//! 機械検証する（`scan_source_evidence.rs`・`constant_pad_source_
//! evidence.rs` と同型。`objc2` 系 FFI に触れないため Linux（本実装
//! 環境・CI）でも実行できる）。
//!
//! カーネル実行結果そのものの正しさ（bit 一致）は macOS 実機限定の
//! `im2col_col2im_parity.rs`（`#[ignore]`）が担う。本テストは
//! ソーステキストの構造的な契約（`#include` の順序・`Im2colDims` の
//! フィールド数・REQ-8 境界検査の存在・`im2col_f32` が算術を含まない
//! こと・`col2im_f32` が binary64 ソフトウェアエミュレーションを使う
//! こと・座標計算の符号判定順序）を固定する。

const SRC: &str = include_str!("../src/shaders/im2col.metal");

/// `#include <metal_stdlib>` → `using namespace metal;` の順（他の
/// shader ファイルと同じ規約）。
#[test]
fn include_precedes_using_namespace() {
    let include_pos = SRC.find("#include <metal_stdlib>").expect("include");
    let using_pos = SRC.find("using namespace metal;").expect("using");
    assert!(include_pos < using_pos);
}

/// カーネル 2 つとも宣言されている。
#[test]
fn declares_both_kernels() {
    assert!(SRC.contains("kernel void im2col_f32("));
    assert!(SRC.contains("kernel void col2im_f32("));
}

/// バッファ index 0〜2（`in`／`d_col`・`out`・`dims`）の宣言。
#[test]
fn declares_expected_buffer_indices() {
    assert!(SRC.contains("device const float* in [[buffer(0)]]"));
    assert!(SRC.contains("device const float* d_col [[buffer(0)]]"));
    assert_eq!(SRC.matches("device float* out [[buffer(1)]]").count(), 2);
    assert_eq!(
        SRC.matches("constant Im2colDims& dims [[buffer(2)]]")
            .count(),
        2
    );
}

/// `struct Im2colDims` は 19 フィールド（`crate::im2col_model::
/// Im2colDims` と同数）。
#[test]
fn im2col_dims_struct_has_19_fields() {
    let start = SRC.find("struct Im2colDims {").expect("struct Im2colDims");
    let end = SRC[start..].find("};").expect("struct end") + start;
    let body = &SRC[start..end];
    let field_count = body.matches("uint ").count();
    assert_eq!(field_count, 19, "Im2colDims は 19 フィールドのはず");
}

/// REQ-8 境界検査（`gid >= dims.numel`）が両カーネルに存在する。
#[test]
fn both_kernels_have_bounds_check() {
    assert_eq!(SRC.matches("if (gid >= dims.numel) {").count(), 2);
}

/// `im2col_f32` 本体は binary64 ソフトウェアエミュレーション
/// （`im2col_f64_` 系関数）を一切呼ばない（算術を含まない純粋コピー。
/// モジュール冒頭コメントの数値契約）。
#[test]
fn im2col_kernel_body_contains_no_f64_emulation_calls() {
    let start = SRC.find("kernel void im2col_f32(").expect("im2col_f32");
    let end = SRC[start..]
        .find("kernel void col2im_f32(")
        .expect("col2im_f32 follows")
        + start;
    let body = &SRC[start..end];
    assert!(!body.contains("im2col_f64_widen("));
    assert!(!body.contains("im2col_f64_add("));
    assert!(!body.contains("im2col_f64_narrow("));
}

/// `col2im_f32` 本体は binary64 ソフトウェアエミュレーション
/// （`widen`／`add`／`narrow`）を使う（`mul` は使わない。scan.metal
/// と異なり col2im は乗算を伴わないため）。
#[test]
fn col2im_kernel_body_uses_f64_emulation() {
    let start = SRC.find("kernel void col2im_f32(").expect("col2im_f32");
    let body = &SRC[start..];
    assert!(body.contains("im2col_f64_widen("));
    assert!(body.contains("im2col_f64_add("));
    assert!(body.contains("im2col_f64_narrow("));
    assert!(!body.contains("im2col_f64_mul("));
}

/// 座標計算は剰余を取る前に符号判定する（C の負数 `%` の符号曖昧性を
/// 避けるため。`kernels_im2col.rs::col2im_checks_sign_before_modulo`
/// と同じ意図の Metal 版検証）。
#[test]
fn col2im_checks_sign_before_modulo() {
    let idx_h = SRC
        .find("if (num_h < 0) continue;")
        .expect("num_h sign check");
    let mod_h = SRC
        .find("if (num_h % (long)dims.sh != 0) continue;")
        .expect("num_h modulo");
    assert!(idx_h < mod_h);
    let idx_w = SRC
        .find("if (num_w < 0) continue;")
        .expect("num_w sign check");
    let mod_w = SRC
        .find("if (num_w % (long)dims.sw != 0) continue;")
        .expect("num_w modulo");
    assert!(idx_w < mod_w);
}

/// 座標計算・添字演算は `long`（符号付き 64bit）で行う（`ulong` の
/// 即座のラップアラウンドを避けるため。`kernels_im2col.rs` の
/// `long long` と同じ理由）。
#[test]
fn coordinate_arithmetic_uses_signed_long() {
    assert!(SRC.contains("long h = oh * (long)dims.sh"));
    assert!(SRC.contains("long w = ow * (long)dims.sw"));
}
