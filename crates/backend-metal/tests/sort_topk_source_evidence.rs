//! イシュー #1741: `sort_build_keys_u64`／`bitonic_step_u64`／
//! `sort_finalize_f32` カーネル（MSL）の文字列証跡テスト。
//! `unique_source_evidence.rs` と同方針: `include_str!` によるビルド
//! 時文字列埋め込みへの contains 検査のみで完結するため、Metal 実機・
//! `cfg(target_os = "macos")` を必要とせず Linux CI（GitHub ホステッド）
//! 上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、`bitonic_step_u64`
//! （比較・swap のみを行う中核カーネル）が浮動小数点演算を一切含まない
//! こと（決定性の根拠。合成キーの算術は整数演算のみ）のロックを兼ねる。

/// `crates/backend-metal/src/shaders/sort.metal` のソース全文。
const SORT_METAL_SOURCE: &str = include_str!("../src/shaders/sort.metal");

/// `MetalSort::new`（`crate::sort`）はソース全文をそのまま
/// `newLibraryWithSource_options_error` へ渡して実行時コンパイルする
/// ため、`#include <metal_stdlib>`／`using namespace metal;` の宣言が
/// 欠けていると `LibraryCompilation` エラーになる。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        SORT_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "sort.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        SORT_METAL_SOURCE.contains("using namespace metal;"),
        "sort.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = SORT_METAL_SOURCE.find("#include <metal_stdlib>").unwrap();
    let using_pos = SORT_METAL_SOURCE.find("using namespace metal;").unwrap();
    assert!(
        include_pos < using_pos,
        "`#include <metal_stdlib>` は `using namespace metal;` より前に置く"
    );
}

#[test]
fn all_three_kernels_are_declared() {
    assert!(
        SORT_METAL_SOURCE.contains("kernel void sort_build_keys_u64("),
        "sort_build_keys_u64 カーネルの宣言が見つかりません"
    );
    assert!(
        SORT_METAL_SOURCE.contains("kernel void bitonic_step_u64("),
        "bitonic_step_u64 カーネルの宣言が見つかりません"
    );
    assert!(
        SORT_METAL_SOURCE.contains("kernel void sort_finalize_f32("),
        "sort_finalize_f32 カーネルの宣言が見つかりません"
    );
}

/// `bitonic_step_u64`（`sort.rs::MetalSort::run_sort_f32` がホスト側
/// ループで繰り返しエンコードする中核カーネル）の境界検査
/// （`(ulong)gid >= total`・`ixj <= i || ixj >= padded`）が存在する
/// ことを機械検証する（最適化を理由に省略しない契約のロック）。
#[test]
fn bitonic_step_u64_boundary_checks_are_present() {
    let start = SORT_METAL_SOURCE
        .find("kernel void bitonic_step_u64(")
        .expect("kernel declaration must exist");
    let end = SORT_METAL_SOURCE[start..]
        .find("kernel void sort_finalize_f32(")
        .map(|off| start + off)
        .unwrap_or(SORT_METAL_SOURCE.len());
    let body = &SORT_METAL_SOURCE[start..end];
    assert!(
        body.contains("if ((ulong)gid >= total)"),
        "bitonic_step_u64 のグローバル添字境界検査が見つかりません"
    );
    assert!(
        body.contains("if (ixj <= i || ixj >= padded)"),
        "bitonic_step_u64 の比較相手添字境界検査が見つかりません"
    );
}

/// `bitonic_step_u64` 本体（整数 compare/swap のみを行う中核カーネル）
/// が浮動小数点演算を一切含まないことを確認する（決定性の根拠。
/// `unique_source_evidence.rs::kernel_body_contains_no_floating_point_types`
/// と同型）。`sort_build_keys_u64`／`sort_finalize_f32` は `input`
/// （`float*`）を扱うため対象外——合成キーの比較・swap のみが
/// 決定性の要となる区間。
#[test]
fn bitonic_step_u64_body_contains_no_floating_point_types() {
    let start = SORT_METAL_SOURCE
        .find("kernel void bitonic_step_u64(")
        .expect("kernel declaration must exist");
    let end = SORT_METAL_SOURCE[start..]
        .find("kernel void sort_finalize_f32(")
        .map(|off| start + off)
        .unwrap_or(SORT_METAL_SOURCE.len());
    let body = &SORT_METAL_SOURCE[start..end];
    assert!(
        !body.contains("float"),
        "bitonic_step_u64 は整数演算のみを行うはずだが `float` を含む"
    );
    assert!(
        !body.contains("half"),
        "bitonic_step_u64 は整数演算のみを行うはずだが `half` を含む"
    );
}

/// `sort_build_keys_u64`／`sort_finalize_f32` にも境界検査（`numel_in`／
/// `numel_out` 検査）が存在することを機械検証する。
#[test]
fn build_and_finalize_have_numel_boundary_checks() {
    assert!(
        SORT_METAL_SOURCE.contains("if (in_pos >= numel_in) {"),
        "sort_build_keys_u64 の numel_in 境界検査が見つかりません"
    );
    assert!(
        SORT_METAL_SOURCE.contains("if (in_pos >= numel_in || out_pos >= numel_out) {"),
        "sort_finalize_f32 の numel_in／numel_out 境界検査が見つかりません"
    );
}
