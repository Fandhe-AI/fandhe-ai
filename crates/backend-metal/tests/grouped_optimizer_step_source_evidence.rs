//! イシュー #3837: Metal の param groups 向け区間テーブル更新
//! （`sgd_step_device_grouped`／`adam_step_device_grouped` とその `_tracked`）が、
//! 既存 MSL カーネルをバイトオフセット付きで区間ごとに起動する構造であること、および
//! `ops.rs` の override 結線が実在することを機械検査する証跡テスト
//! （`adam_source_evidence.rs` と同型）。
//!
//! `include_str!` による contains 検査のみで完結するため、Metal 実機・
//! `cfg(target_os = "macos")` を必要とせず Linux CI 上でも green になる。`ops.rs` 自体が
//! macOS 限定で Linux では型検査すら走らないため、override 結線の漏れ（既定
//! `Unsupported` に落ちる回帰）を Linux で検知する意図的な文字列検査である。
//! 実機 parity は `grouped_optimizer_step_parity.rs`（`#[ignore]`・#3863 で実測）。

const SGD_METAL: &str = include_str!("../src/shaders/sgd.metal");
const ADAM_METAL: &str = include_str!("../src/shaders/adam.metal");
const SGD_RS: &str = include_str!("../src/sgd.rs");
const ADAM_RS: &str = include_str!("../src/adam.rs");
const OPS_SOURCE: &str = include_str!("../src/ops.rs");

fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| l.trim_start())
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// grouped 経路は新規 MSL を足さず既存カーネルを使う。REQ-8 の手動境界チェックと意図した
/// `fma(` が既存カーネルに残っていること（grouped もこれに依存する）。Adam はさらに FP 縮約禁止
/// （`#pragma clang fp contract(off)`）と `precise::sqrt` を持つ（SGD シェーダにはもともと無い）。
#[test]
fn existing_kernels_keep_bound_check_and_fma() {
    for (src, kernel) in [(SGD_METAL, "sgd_step_f32"), (ADAM_METAL, "adam_step_f32")] {
        let s = strip_line_comments(src);
        let pos = s
            .find(&format!("kernel void {kernel}"))
            .expect("kernel not found");
        let body = &s[pos..];
        assert!(body.contains("if (idx < numel)"), "{kernel}: bound check");
        assert!(body.contains("fma("), "{kernel}: intended fma");
    }
    let adam = strip_line_comments(ADAM_METAL);
    let pos = adam.find("kernel void adam_step_f32").unwrap();
    assert!(adam[..pos].contains("#pragma clang fp contract(off)"));
    assert!(adam.contains("precise::sqrt"));
}

/// 区間ごとの起動がバッファオフセット（`区間先頭 * 4` バイト）と区間 `numel` を渡すこと、
/// 区間の合計長を検証すること（REQ-8）をロックする。
#[test]
fn grouped_dispatch_uses_segment_offsets_and_numel() {
    for (name, src) in [("sgd.rs", SGD_RS), ("adam.rs", ADAM_RS)] {
        let s = strip_line_comments(src);
        assert!(s.contains("pub fn run_grouped("), "{name}: run_grouped");
        assert!(
            s.contains("off * std::mem::size_of::<f32>()"),
            "{name}: byte offset"
        );
        assert!(s.contains("seg.numel as u32"), "{name}: per-segment numel");
        assert!(s.contains("offset_bytes"), "{name}: offset plumbing");
    }
    assert!(OPS_SOURCE.contains("fn check_grouped_segment_total("));
}

/// `ops.rs` の override 結線（capability と 4 メソッド）が実在すること。
#[test]
fn ops_wires_grouped_overrides_and_capability() {
    let s = strip_line_comments(OPS_SOURCE);
    for f in [
        "fn sgd_step_device_grouped(",
        "fn sgd_step_device_grouped_tracked(",
        "fn adam_step_device_grouped(",
        "fn adam_step_device_grouped_tracked(",
        "fn supports_grouped_optimizer_step(",
    ] {
        assert!(s.contains(f), "{f} の override が ops.rs にありません");
    }
    assert!(s.contains("sgd.run_grouped("));
    assert!(s.contains("adam.run_grouped("));
}
