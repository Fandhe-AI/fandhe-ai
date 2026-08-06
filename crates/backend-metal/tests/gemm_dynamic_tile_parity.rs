//! `backend-metal` 動的タイル選択 GEMM（TASK-1.8f・#188）の受け入れ条件検証:
//! 「`gemm_simdgroup_tiled` の全候補構成（`crate::tile` の候補セット・
//! `dispatch_auto` の自動選択）の数値が CPU 参照実装と複合判定（相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満。REQ-2）で一致する」。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・
//! Linux）では `#![cfg(target_os = "macos")]` によりコンパイル対象外に
//! なり、`#[ignore]` により通常の `cargo test` からも除外される
//! （実機依存テストの分離。`.claude/rules/coding-rust.md`。
//! `tests/gemm_simdgroup_parity.rs`〈#40〉と同じ方針）。実行するには
//! macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p backend-metal -- --ignored --nocapture
//! ```
//!
//! CPU 参照は `backend_cpu::parity::matmul_reference_fma`（FMA 契約の
//! 唯一の参照点）、判定は `backend_cpu::parity::assert_parity`（REQ-2
//! 統一複合判定の唯一の実体。閾値の独自定義・緩和は禁止。
//! `.claude/rules/security.md`）を使う。入力生成は
//! `bench_harness::rng::Xorshift64Star`（決定的シード）。

#![cfg(target_os = "macos")]

use backend_cpu::parity::{assert_parity, matmul_reference_fma};
use backend_metal::{GemmVariant, MetalContext, MetalGemm, TileConfig};
use bench_harness::rng::Xorshift64Star;

/// `variant`（[`GemmVariant::SimdgroupTiled`]）・`(seed_a, seed_b, m, n, k)`
/// の 1 ケースを実行し、CPU 参照実装との複合判定 PASS を確認する。
fn run_case(cfg: TileConfig, seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b = Xorshift64Star::new(seed_b).fill_vec(k * n);

    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    let actual = gemm
        .dispatch_variant(&ctx, GemmVariant::SimdgroupTiled(cfg), &a, &b, m, n, k)
        .unwrap_or_else(|err| {
            panic!("Metal SimdgroupTiled({cfg:?}) GEMM のディスパッチに失敗した: {err}")
        });

    assert_parity(
        &format!("metal SimdgroupTiled({cfg:?}) gemm m={m} n={n} k={k}"),
        &actual,
        &expected,
    );
}

/// `crate::tile` の候補セット（大形状・縦長・横長・中形状・単一
/// simdgroup）を全て、8 の倍数の中規模形状で検証する（構成別の一致確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn all_tile_candidates_match_cpu_reference_medium_shape() {
    let candidates = [
        TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        },
        TileConfig {
            bm: 64,
            bn: 32,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        },
        TileConfig {
            bm: 32,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        },
        TileConfig {
            bm: 32,
            bn: 32,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        },
        TileConfig::SINGLE_SIMDGROUP_8X8,
    ];
    for (i, cfg) in candidates.into_iter().enumerate() {
        run_case(cfg, 10 + i as u64, 20 + i as u64, 256, 256, 256);
    }
}

/// 直接ロード経路（`staged=false`）を明示指定し、協調ロード経路と別に
/// 検証する（計画「設計方針」節: 両経路を実装し実測で選択するため、
/// 少なくとも数値正しさは両方で担保する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn direct_load_path_matches_cpu_reference() {
    let cfg = TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: false,
    };
    run_case(cfg, 30, 31, 256, 256, 256);
}

/// threadgroup サイズ（BM/BN/BK）いずれの倍数でもない境界形状。
/// `shaders/gemm.metal` の `gemm_simdgroup_tiled` 手動境界チェック
/// （ブロック端の早期 return・K タイル端の 0 埋め）が実際に効くケース
/// （REQ-8）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tiled_matches_cpu_reference_non_multiple_of_tile() {
    let cfg = TileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };
    run_case(cfg, 1, 2, 100, 130, 70);
    run_case(cfg, 3, 4, 65, 129, 33);
}

/// `dispatch_auto`（`crate::tile::select` による自動選択入口）が
/// 複数の形状帯（微小・中形状・縦長・横長・大形状）で CPU 参照実装と
/// 一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_auto_matches_cpu_reference_across_shape_bands() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for (i, &(m, n, k)) in [
        (7usize, 13usize, 5usize), // 微小形状
        (128, 128, 128),           // 中形状（正方）
        (1024, 128, 256),          // 縦長
        (128, 1024, 256),          // 横長
        (1024, 1024, 1024),        // 大形状（正方）
    ]
    .iter()
    .enumerate()
    {
        let a = Xorshift64Star::new(40 + i as u64).fill_vec(m * k);
        let b = Xorshift64Star::new(50 + i as u64).fill_vec(k * n);

        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a, &b, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        let actual = gemm
            .dispatch_auto(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|err| panic!("dispatch_auto(m={m}, n={n}, k={k}) に失敗した: {err}"));

        assert_parity(
            &format!("metal dispatch_auto gemm m={m} n={n} k={k}"),
            &actual,
            &expected,
        );
    }
}

/// K ストレスケース（PoC-v2-5 の FMA 契約実測ケースに対応。長い内積での
/// 丸め誤差蓄積が CPU 参照実装〈`f32::mul_add`〉と一致することを確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tiled_matches_cpu_reference_k_stress() {
    let cfg = TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };
    run_case(cfg, 7, 8, 64, 64, 4096);
}
