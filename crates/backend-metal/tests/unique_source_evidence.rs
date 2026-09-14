//! イシュー #1734: `bitonic_step_u32` カーネル（MSL）の文字列証跡
//! テスト。`gather_scatter_source_evidence.rs` と同方針: `include_str!`
//! によるビルド時文字列埋め込みへの contains 検査のみで完結するため、
//! Metal 実機・`cfg(target_os = "macos")` を必要とせず Linux CI
//! （GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証（`i >= n`／`ixj <= i ||
//! ixj >= n` の両境界検査）と、整数 compare/swap のみで浮動小数点演算を
//! 一切含まないこと（決定性の根拠）のロックを兼ねる。

/// `crates/backend-metal/src/shaders/unique.metal` のソース全文。
const UNIQUE_METAL_SOURCE: &str = include_str!("../src/shaders/unique.metal");

/// `MetalUnique::new`（`crate::unique`）はソース全文をそのまま
/// `newLibraryWithSource_options_error` へ渡して実行時コンパイルする
/// ため、`#include <metal_stdlib>`／`using namespace metal;` の宣言が
/// 欠けていると `LibraryCompilation` エラーになる（`gather_scatter.metal`
/// と同じ構成が必須）。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        UNIQUE_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "unique.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        UNIQUE_METAL_SOURCE.contains("using namespace metal;"),
        "unique.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = UNIQUE_METAL_SOURCE.find("#include <metal_stdlib>").unwrap();
    let using_pos = UNIQUE_METAL_SOURCE.find("using namespace metal;").unwrap();
    assert!(
        include_pos < using_pos,
        "`#include <metal_stdlib>` は `using namespace metal;` より前に置く"
    );
}

#[test]
fn kernel_name_and_buffer_order_are_declared() {
    assert!(
        UNIQUE_METAL_SOURCE.contains("kernel void bitonic_step_u32("),
        "bitonic_step_u32 カーネルの宣言が見つかりません"
    );
    // buffer index 0（keys）・constant index 1〜3（j／k／n）。
    assert!(UNIQUE_METAL_SOURCE.contains("[[buffer(0)]]"));
    assert!(UNIQUE_METAL_SOURCE.contains("[[buffer(1)]]"));
    assert!(UNIQUE_METAL_SOURCE.contains("[[buffer(2)]]"));
    assert!(UNIQUE_METAL_SOURCE.contains("[[buffer(3)]]"));
}

/// REQ-8（境界検査規約）: `i >= n`（自スレッド添字の範囲検査）と
/// `ixj <= i || ixj >= n`（比較相手添字の範囲検査・同一ペア二重処理
/// 防止）の両方がソース中に存在することを機械検証する（最適化を理由に
/// 省略しない契約のロック）。
#[test]
fn boundary_checks_are_present() {
    assert!(
        UNIQUE_METAL_SOURCE.contains("if (i >= n)"),
        "自スレッド添字 i の境界検査（i >= n）が見つかりません"
    );
    assert!(
        UNIQUE_METAL_SOURCE.contains("if (ixj <= i || ixj >= n)"),
        "比較相手添字 ixj の境界検査（ixj <= i || ixj >= n）が見つかりません"
    );
}

/// カーネル本体（`kernel void bitonic_step_u32` の定義区間）が浮動小数点
/// 演算を一切含まないことを確認する（決定性の根拠。整数 compare/swap
/// のみで NVRTC の math モードに非依存な CUDA 側 `kernels_unique.rs`
/// と同じ設計方針）。
#[test]
fn kernel_body_contains_no_floating_point_types() {
    let start = UNIQUE_METAL_SOURCE
        .find("kernel void bitonic_step_u32(")
        .expect("kernel declaration must exist");
    let body = &UNIQUE_METAL_SOURCE[start..];
    assert!(
        !body.contains("float"),
        "bitonic_step_u32 は整数演算のみを行うはずだが `float` を含む"
    );
    assert!(
        !body.contains("half"),
        "bitonic_step_u32 は整数演算のみを行うはずだが `half` を含む"
    );
}
