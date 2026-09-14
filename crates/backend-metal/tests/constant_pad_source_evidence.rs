//! イシュー #1756: pad カーネル（MSL）の文字列証跡テスト。
//! `gather_scatter_source_evidence.rs`（#1778）と同方針: `include_str!`
//! によるビルド時文字列埋め込みへの contains 検査のみで完結するため、
//! Metal 実機・`cfg(target_os = "macos")` を必要とせず Linux CI
//! （GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、`crates/backend-metal/
//! src/shaders/constant_pad.metal` 冒頭コメントが明記するアルゴリズム
//! 契約（座標配列を保持しない末尾軸剥がし方式・添字計算は `long`）の
//! ロックを兼ねる。

/// `crates/backend-metal/src/shaders/constant_pad.metal` のソース全文。
const CONSTANT_PAD_METAL_SOURCE: &str = include_str!("../src/shaders/constant_pad.metal");

/// `MetalConstantPad::new`（`crate::constant_pad`）はソース全文を
/// そのまま `newLibraryWithSource_options_error` へ渡して実行時
/// コンパイルするため、`#include <metal_stdlib>`／`using namespace
/// metal;` の宣言が欠けていると全経路が `LibraryCompilation` エラーに
/// なる（`gather_scatter.metal` と同じ構成が必須）。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        CONSTANT_PAD_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "constant_pad.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        CONSTANT_PAD_METAL_SOURCE.contains("using namespace metal;"),
        "constant_pad.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = CONSTANT_PAD_METAL_SOURCE
        .find("#include <metal_stdlib>")
        .unwrap();
    let using_pos = CONSTANT_PAD_METAL_SOURCE
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
        CONSTANT_PAD_METAL_SOURCE.contains("kernel void constant_pad_f32("),
        "constant_pad_f32 カーネルの宣言が見つかりません"
    );
    // バッファ index の宣言順（`constant_pad.rs::encode_pad_dispatch` と
    // 一致させる契約）。
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("device const float* input [[buffer(0)]]"));
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("device float* out [[buffer(1)]]"));
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("constant uint* shapes [[buffer(2)]]"));
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("constant uint& rank [[buffer(3)]]"));
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("constant uint& numel [[buffer(4)]]"));
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("constant float& value [[buffer(5)]]"));
}

/// REQ-8 境界検査: `gid >= numel` の早期 return（末尾ブロックの余剰
/// スレッド対策。手動境界チェックを省略しない）。
#[test]
fn kernel_has_grid_boundary_guard() {
    assert!(
        CONSTANT_PAD_METAL_SOURCE.contains("if (gid >= numel) {"),
        "constant_pad_f32 の `gid >= numel` 境界検査が見つかりません"
    );
}

/// 添字計算に `long`（64bit 符号付き）を使うことをロックする（`before`
/// を引いた際のアンダーフロー検出・`u32` の中間ストライド積
/// オーバーフロー対策。REQ-8）。
#[test]
fn stride_arithmetic_uses_signed_64bit() {
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("long rem = (long)gid;"));
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("long src_c = c - (long)before[a];"));
}

/// パディング領域は `value` を書く分岐を持つ。
#[test]
fn kernel_has_padding_fallback_to_value() {
    assert!(CONSTANT_PAD_METAL_SOURCE.contains("out[gid] = inside ? input[in_flat] : value;"));
}
