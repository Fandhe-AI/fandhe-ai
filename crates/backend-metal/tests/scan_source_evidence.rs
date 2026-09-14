//! イシュー #1740: `cumsum_f32`／`cumprod_f32` カーネル（MSL）の文字列
//! 証跡テスト。`unique_source_evidence.rs`／`gather_scatter_source_
//! evidence.rs` と同方針: `include_str!` によるビルド時文字列埋め込み
//! への contains 検査のみで完結するため、Metal 実機・
//! `cfg(target_os = "macos")` を必要とせず Linux CI（GitHub
//! ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証（`gid >= lanes` 境界検査）
//! と、`crate::soft_f64`／`crate::scan_model` と bit 完全一致する契約の
//! 根拠となる binary64 ソフトウェアエミュレーションアキュムレータ
//! （`scan_f64_widen`／`scan_f64_add`／`scan_f64_mul`／
//! `scan_f64_narrow`）の使用のロックを兼ねる。

/// `crates/backend-metal/src/shaders/scan.metal` のソース全文。
const SCAN_METAL_SOURCE: &str = include_str!("../src/shaders/scan.metal");

/// `MetalScan::new`（`crate::scan`）はソース全文をそのまま
/// `newLibraryWithSource_options_error` へ渡して実行時コンパイルする
/// ため、`#include <metal_stdlib>`／`using namespace metal;` の宣言が
/// 欠けていると `LibraryCompilation` エラーになる（`unique.metal`・
/// `gather_scatter.metal` と同じ構成が必須）。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        SCAN_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "scan.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        SCAN_METAL_SOURCE.contains("using namespace metal;"),
        "scan.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = SCAN_METAL_SOURCE.find("#include <metal_stdlib>").unwrap();
    let using_pos = SCAN_METAL_SOURCE.find("using namespace metal;").unwrap();
    assert!(
        include_pos < using_pos,
        "`#include <metal_stdlib>` は `using namespace metal;` より前に置く"
    );
}

/// `crate::scan::MetalScan::new` が `pipeline::make_pipeline` へ渡す
/// カーネル名（`"cumsum_f32"`／`"cumprod_f32"`）とバッファ index
/// （`x`＝0・`out`＝1・`lanes`＝2・`axis_len`＝3・`inner`＝4）の宣言を
/// 機械検証する。
#[test]
fn kernel_names_and_buffer_order_are_declared() {
    for kernel in ["cumsum_f32", "cumprod_f32"] {
        assert!(
            SCAN_METAL_SOURCE.contains(&format!("kernel void {kernel}(")),
            "{kernel} カーネルの宣言が見つかりません"
        );
    }
    // buffer index 0（x）〜4（inner）。
    for idx in 0..=4 {
        assert!(
            SCAN_METAL_SOURCE.contains(&format!("[[buffer({idx})]]")),
            "buffer({idx}) の宣言が見つかりません"
        );
    }
}

/// REQ-8（境界検査規約）: 両カーネルとも `gid >= lanes`（グローバル
/// スレッド id の境界検査）を持つことを機械検証する（最適化を理由に
/// 省略しない契約のロック）。
#[test]
fn boundary_checks_are_present() {
    let occurrences = SCAN_METAL_SOURCE.matches("if (gid >= lanes)").count();
    assert_eq!(
        occurrences, 2,
        "cumsum_f32／cumprod_f32 の両方に `if (gid >= lanes)` 境界検査が必要（{occurrences} 箇所検出）"
    );
}

/// binary64 ソフトウェアエミュレーションアキュムレータ
/// （`scan_f64_widen`／`scan_f64_add`／`scan_f64_mul`／
/// `scan_f64_narrow`）がカーネル本体で使われていることを機械検証する
/// （MSL は `double` 非対応のため、ホスト `f64` 参照実装との bit 完全
/// 一致契約はこのソフトウェアエミュレーション経路に依存する。
/// `.claude/rules/coding-rust.md`「勾配の長軸縮約」節と同じ設計）。
#[test]
fn kernels_use_soft_f64_accumulator() {
    for f in [
        "scan_f64_widen",
        "scan_f64_add",
        "scan_f64_mul",
        "scan_f64_narrow",
    ] {
        assert!(
            SCAN_METAL_SOURCE.contains(f),
            "scan.metal に `{f}`（binary64 ソフトウェアエミュレーション）が見つかりません"
        );
    }
    // cumsum は加算のみ（乗算なし）・cumprod は乗算を使う、という
    // カーネルごとの使い分けを本体区間で確認する。
    let cumsum_start = SCAN_METAL_SOURCE
        .find("kernel void cumsum_f32(")
        .expect("cumsum_f32 declaration must exist");
    let cumprod_start = SCAN_METAL_SOURCE
        .find("kernel void cumprod_f32(")
        .expect("cumprod_f32 declaration must exist");
    let cumsum_body = &SCAN_METAL_SOURCE[cumsum_start..cumprod_start];
    let cumprod_body = &SCAN_METAL_SOURCE[cumprod_start..];
    assert!(
        cumsum_body.contains("scan_f64_add"),
        "cumsum_f32 は scan_f64_add を使うはず"
    );
    assert!(
        cumprod_body.contains("scan_f64_mul"),
        "cumprod_f32 は scan_f64_mul を使うはず"
    );
}
