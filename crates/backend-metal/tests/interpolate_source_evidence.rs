//! イシュー #1757: interpolate カーネル（MSL）の文字列証跡テスト。
//! `gather_scatter_source_evidence.rs`（#1778）と同方針: `include_str!`
//! によるビルド時文字列埋め込みへの contains 検査のみで完結するため、
//! Metal 実機・`cfg(target_os = "macos")` を必要とせず Linux CI
//! （GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、`crates/backend-metal/
//! src/shaders/interpolate.metal` 冒頭コメントが明記するアルゴリズム
//! 契約（座標配列を保持しない末尾軸剥がし方式・添字計算は `ulong`〈イシュー
//! #1834 codex-review P0 是正で `long` から変更。u32 収容の 2 軸積が
//! `i64::MAX` を超えうるため〉）のロックを兼ねる。

/// `crates/backend-metal/src/shaders/interpolate.metal` のソース全文。
const INTERPOLATE_METAL_SOURCE: &str = include_str!("../src/shaders/interpolate.metal");

/// `MetalInterpolate::new`（`crate::interpolate`）はソース全文をそのまま
/// `newLibraryWithSource_options_error` へ渡して実行時コンパイルする
/// ため、`#include <metal_stdlib>`／`using namespace metal;` の宣言が
/// 欠けていると全経路が `LibraryCompilation` エラーになる
/// （`gather_scatter.metal` と同じ構成が必須）。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        INTERPOLATE_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "interpolate.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        INTERPOLATE_METAL_SOURCE.contains("using namespace metal;"),
        "interpolate.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = INTERPOLATE_METAL_SOURCE
        .find("#include <metal_stdlib>")
        .unwrap();
    let using_pos = INTERPOLATE_METAL_SOURCE
        .find("using namespace metal;")
        .unwrap();
    assert!(
        include_pos < using_pos,
        "`#include <metal_stdlib>` は `using namespace metal;` より前に置く"
    );
}

#[test]
fn kernel_name_and_buffer_order_are_declared() {
    assert!(
        INTERPOLATE_METAL_SOURCE.contains("kernel void interpolate_nearest_f32("),
        "interpolate_nearest_f32 カーネルの宣言が見つかりません"
    );
    // バッファ index の宣言順（`interpolate.rs::encode_interpolate_dispatch`
    // と一致させる契約）。
    assert!(INTERPOLATE_METAL_SOURCE.contains("device const float* input [[buffer(0)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("device float* out [[buffer(1)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant uint* shapes [[buffer(2)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant uint& rank [[buffer(3)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant uint& spatial_start [[buffer(4)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant uint& numel [[buffer(5)]]"));
}

/// REQ-8 境界検査: `gid >= numel` の早期 return（末尾ブロックの余剰
/// スレッド対策。手動境界チェックを省略しない）。
#[test]
fn kernel_has_grid_boundary_guard() {
    assert!(
        INTERPOLATE_METAL_SOURCE.contains("if (gid >= numel) {"),
        "interpolate_nearest_f32 の `gid >= numel` 境界検査が見つかりません"
    );
}

/// 添字計算に `ulong`（64bit 符号なし）を使うことをロックする
/// （REQ-8・イシュー #1834 codex-review P0 是正: u32 収容の 2 軸積
/// `c * in_size` が `i64::MAX` を超えうるため `long` から変更した）。
#[test]
fn stride_arithmetic_uses_unsigned_64bit() {
    assert!(INTERPOLATE_METAL_SOURCE.contains("ulong rem = (ulong)gid;"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("ulong src_c;"));
}

/// 空間軸の src 添字は縦深防御クランプ（`min(src_c, in_shape[a]-1)`
/// 相当）を持つ（REQ-8）。
#[test]
fn kernel_clamps_spatial_src_coord() {
    assert!(INTERPOLATE_METAL_SOURCE.contains("if (src_c > max_c) {"));
}

/// 空間軸以外（`a < spatial_start`）は `src_c = c`（素通し）。
#[test]
fn kernel_passes_through_non_spatial_axes() {
    assert!(INTERPOLATE_METAL_SOURCE.contains("src_c = c;"));
}

// --- bilinear（イシュー #1762） ---

#[test]
fn bilinear_kernel_name_and_buffer_order_are_declared() {
    assert!(
        INTERPOLATE_METAL_SOURCE.contains("kernel void interpolate_bilinear_f32("),
        "interpolate_bilinear_f32 カーネルの宣言が見つかりません"
    );
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant uint& align_corners [[buffer(4)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant float& scale_h [[buffer(6)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant float& scale_w [[buffer(7)]]"));
}

#[test]
fn bilinear_kernel_has_grid_boundary_guard() {
    let bilinear_start = INTERPOLATE_METAL_SOURCE
        .find("kernel void interpolate_bilinear_f32(")
        .expect("bilinear kernel must exist");
    assert!(
        INTERPOLATE_METAL_SOURCE[bilinear_start..].contains("if (gid >= numel) {"),
        "interpolate_bilinear_f32 の `gid >= numel` 境界検査が見つかりません"
    );
}

#[test]
fn bilinear_kernel_clamps_spatial_src_coord() {
    assert!(INTERPOLATE_METAL_SOURCE.contains("if (i0y > (long)in_h - 1) {"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("if (i1y > (long)in_h - 1) {"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("if (i0x > (long)in_w - 1) {"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("if (i1x > (long)in_w - 1) {"));
}

#[test]
fn bilinear_kernel_uses_fma_for_blend() {
    assert!(INTERPOLATE_METAL_SOURCE.contains("fma(l1x, v01, l0x * v00)"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("fma(l1x, v11, l0x * v10)"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("fma(l1y, row1, l0y * row0)"));
}

#[test]
fn bilinear_kernel_takes_precomputed_scale_arguments() {
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant float& scale_h [[buffer(6)]]"));
    assert!(INTERPOLATE_METAL_SOURCE.contains("constant float& scale_w [[buffer(7)]]"));
}
