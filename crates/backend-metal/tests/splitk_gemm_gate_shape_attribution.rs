//! イシュー #1521: Metal GEMM candle 比ゲート（旧 #1037・`docs/perf/
//! metal-gemm-candle-gate-remeasurement.md`）が対象とする NN 正方形状
//! （N=1024/2048/4096・`(m,n,k)=(N,N,N)`）が、split-K 本番結線（イシュー
//! #1516・#1530）後の `crate::tile::select_route_for_device` の観点で
//! なお classic 経路（[`tile::GemmRoute::Classic`]）へ到達することを
//! 機械的に固定する「帰属表」テスト。
//!
//! # 位置づけ
//!
//! `docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/attribution.md` は
//! `fandhe-ai =0.8.0` タグ時点（`SPLIT_K_NUMERIC_CONTRACT_APPROVED=
//! false`）での非到達根拠を記録した。その後 #1527
//! （`SPLIT_K_NUMERIC_CONTRACT_APPROVED=true`）・#1530（split-K 本番
//! 結線）を経た現在の HEAD では、数値契約ゲート自体は真になったが、
//! (a) `MetalGemm::dispatch_auto` の既定構成（`split_k_auto_enabled`＝
//! `tile::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED`＝`false`）では
//! split-K 分岐そのものへ入らない、(b) 仮に入ったとしても本テストの
//! 対象形状は `should_split_k` の並列度条件（`tile.rs`
//! `should_split_k_rejects_large_square_and_wide_shapes` が正方
//! 512〜4096 を対象に回帰確認済み）で `None` になる、という**独立した
//! 2 つの理由**により、v0.7.0/v0.8.0 時点と同じ classic 経路を通ることを
//! 本テストで確定する（`docs/perf/logs/
//! metal-gemm-candle-gate-head-1521/attribution.md` §「split-K が
//! 非到達である根拠」の一次証跡）。
//!
//! `crate::tile::{should_split_k, select_route_for_device, GemmRoute,
//! select_for_device}` はいずれも `cfg(target_os = "macos")` を持たない
//! ため（`tile.rs` モジュール冒頭 doc 参照）、本統合テストは Linux
//! （CI・本実装環境）でもコンパイル・実行できる。
//! `crates/backend-metal/tests/splitk_train_shape_attribution.rs`
//! （イシュー #1517）と同型の構成。
//!
//! 実行例（帰属表を標準出力へ表示する）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --test splitk_gemm_gate_shape_attribution -- --nocapture
//! ```

use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{GemmRoute, TileConfig};

/// GEMM ゲート（旧 #1037）の対象形状。`scripts/bench/framework-compare/
/// run_gemm_gate.sh` の `_SIZES_BY_DEVICE["metal"]`（N=1024/2048/4096）
/// に対応し、`(m,n,k)=(N,N,N)` の NN 正方として `bench-fandhe gemm metal
/// <N> reuse` から呼ばれる。
const GATE_SIZES: &[usize] = &[1024, 2048, 4096];

/// 1 形状分の帰属表 1 行。`should_split_k` の Some/None・
/// `select_route_for_device` が返す [`GemmRoute`] の判別（`Classic`／
/// `SplitK`）を機械出力する。
fn print_row(n: usize) {
    let plan = tile::should_split_k(n, n, n);
    let route = tile::select_route_for_device(n, n, n, None);
    let route_kind = match &route {
        GemmRoute::Classic(_) => "Classic",
        GemmRoute::SplitK(_) => "SplitK",
    };
    println!(
        "| N={n} | ({n},{n},{n}) | should_split_k={} | select_route_for_device={route_kind} |",
        if plan.is_some() { "Some" } else { "None" }
    );
}

/// 帰属表の機械生成（`--nocapture` で標準出力へ Markdown 表を出す）。
/// アサーション自体は行わない（`assert_*` は個別テストで行う。本テストは
/// 表の可視化専用。`splitk_train_shape_attribution.rs::
/// print_attribution_table` と同じ役割分担）。
#[test]
fn print_attribution_table() {
    println!("| shape | (m,n,k) | should_split_k | select_route_for_device |");
    println!("|---|---|---|---|");
    for &n in GATE_SIZES {
        print_row(n);
    }
}

/// GEMM ゲート対象の NN 正方 3 形状すべてで `tile::should_split_k` が
/// `None`（split-K 対象外）であることを固定する（帰属表の理由 (b)）。
#[test]
fn gate_shapes_are_not_split_k_candidates() {
    for &n in GATE_SIZES {
        let plan = tile::should_split_k(n, n, n);
        assert!(
            plan.is_none(),
            "N={n} 正方の should_split_k が Some だった（帰属表の前提\
             『GEMM ゲート対象形状は split-K 対象外』が崩れている）"
        );
    }
}

/// GEMM ゲート対象の NN 正方 3 形状すべてで `select_route_for_device` が
/// `GemmRoute::Classic(select_for_device(..))` と一致することを固定する
/// （`should_split_k` が `None` の場合の `select_route_for_device` の
/// 契約どおりの委譲を直接検証する。帰属表の理由 (b) の裏取り）。
#[test]
fn gate_shapes_route_to_classic_matching_select_for_device() {
    for &n in GATE_SIZES {
        let route = tile::select_route_for_device(n, n, n, None);
        let expected: TileConfig = tile::select_for_device(n, n, n, None);
        match route {
            GemmRoute::Classic(cfg) => assert_eq!(
                cfg, expected,
                "N={n} の select_route_for_device が返す TileConfig が \
                 select_for_device の直接呼び出しと一致しない"
            ),
            GemmRoute::SplitK(_) => panic!(
                "N={n} 正方の select_route_for_device が GemmRoute::SplitK を\
                 返した（帰属表の前提が崩れている）"
            ),
        }
    }
}
