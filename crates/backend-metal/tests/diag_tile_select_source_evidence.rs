//! イシュー #4219: 診断テスト（`src/*diag_tests*.rs`）の tile 選択が、本番入口
//! （`verified_tile_table_device` + `select_for_verified_device`／
//! `select_route_for_verified_device`）に揃っていることを固定する棚卸しガード。
//!
//! 旧入口（`verified_m4_max_gpu_core_count`・`select_for_device`・
//! `select_route_for_device`）は M4 Max の表しか通らず、M4 無印（Mac mini。#3206）では
//! 診断が汎用表・本番が M4Base 表になる。`DIAG_BITEQ` は構成差を検出できないため
//! （#4153 で発覚）、診断側の旧入口の再混入をソース検査で防ぐ。
//!
//! 許可リストへの追加は「意図的に旧入口を使う理由」をファイル冒頭コメントに書き、
//! レビューを経ること。ソース走査のみで Metal 実機・macOS を要さず Linux CI で green になる。

use std::fs;
use std::path::Path;

/// 呼び出し形（`(` 付き）で数える旧入口。文字列リテラル中の `select_for_device=` 等は除く。
const LEGACY_CALLS: [&str; 3] = [
    "verified_m4_max_gpu_core_count(",
    "select_for_device(",
    "select_route_for_device(",
];

/// 意図的に旧入口を使うファイルと、その非コメント行での呼び出し総数（棚卸し #4219）。
/// `gemm_tile_select_macmini_diag_tests.rs`: #3181 の入力側実測で RULE 固定済みのため変えない。
const ALLOWED: [(&str, usize); 1] = [("gemm_tile_select_macmini_diag_tests.rs", 5)];

fn legacy_call_count(src: &str) -> usize {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .map(|l| {
            LEGACY_CALLS
                .iter()
                .map(|p| l.matches(p).count())
                .sum::<usize>()
        })
        .sum()
}

fn diag_files() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for e in fs::read_dir(&dir).expect("src を読めること") {
        let p = e.expect("dir entry").path();
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if name.ends_with(".rs") && name.contains("diag_tests") {
            out.push((
                name,
                fs::read_to_string(&p).expect("診断ファイルを読めること"),
            ));
        }
    }
    out
}

#[test]
fn diag_tests_do_not_use_legacy_tile_entry_points() {
    let files = diag_files();
    assert!(
        files.len() >= 20,
        "走査対象が少なすぎる（空振り防止）: {}",
        files.len()
    );
    for (name, src) in &files {
        let n = legacy_call_count(src);
        let allowed = ALLOWED
            .iter()
            .find(|(f, _)| f == name)
            .map_or(0, |(_, c)| *c);
        assert_eq!(
            n, allowed,
            "{name}: 旧入口の呼び出し数 {n}（許可 {allowed}）。verified_tile_table_device / \
             select_for_verified_device へ揃えるか、許可リストへ理由付きで追加する"
        );
    }
}

#[test]
fn allowlist_and_probe_are_live() {
    let files = diag_files();
    for (allowed, _) in ALLOWED {
        assert!(
            files.iter().any(|(n, _)| n == allowed),
            "許可リストのファイルが無い: {allowed}"
        );
    }
    let (_, reuse) = files
        .iter()
        .find(|(n, _)| n == "gemm_reuse_phase_diag_tests.rs")
        .expect("代表ファイルが存在する");
    assert!(reuse.contains("verified_tile_table_device"));
    assert!(reuse.contains("ROUTE_CFG_CHECK"));
}
