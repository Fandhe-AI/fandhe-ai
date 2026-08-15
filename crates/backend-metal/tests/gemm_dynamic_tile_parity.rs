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
///
/// `dispatch_variant` だけを呼ぶと `MetalGemm::pipeline_for_tile` が構成の
/// 検証・コンパイル・パイプライン上限確認のいずれかに失敗した場合
/// `crate::tile::fallback_chain` で `TileConfig::SINGLE_SIMDGROUP_8X8` へ
/// サイレントにフォールバックしても数値一致自体は通ってしまい、`cfg` が
/// 実際に採用されたことを保証しない（イシュー #532・PR #651 codex-review
/// 指摘 P2）。この「実際に採用された構成の検証」（`MetalGemm::
/// resolve_tile_config` を用いる）は、統合テスト（本ファイル。クレート境界
/// の外）から内部実装（パイプライン構築・フォールバック）が公開 API として
/// 露出してしまう問題（PR #651 codex-review 再指摘 P1）を避けるため、
/// `backend_metal` クレート内部の `crate::tile` モジュール
/// `#[cfg(test)] mod tests`（クレート内テスト。`resolve_tile_config` は
/// `pub(crate)`）へ集約済み
/// （`all_tile_candidates_match_cpu_reference_medium_shape` 等）。本関数は
/// 数値一致（CPU 参照実装との複合判定）の確認に限定する。
///
/// `crate::tile::CANDIDATES` を巡回する上記クレート内テストは全構成が
/// `staged=true` のため、本ファイルが使う `staged=false`（直接ロード経路）
/// の構成はその巡回対象に含まれない。この構成のフォールバック検知は
/// `crate::tile` の `direct_load_path_config_resolves_without_fallback`
/// （クレート内テスト）が別途担う（codex-review 再指摘対応。イシュー
/// #532・PR #651。`BUGBOT_BUG_ID: c65127ea-56c2-4c52-95c2-604b5739cf40`）。
/// 下記 `direct_load_path_*` 系のテストが使う `TileConfig` を変更する場合は
/// 同テストの `cfg` も同期させること。
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

// `crate::tile::CANDIDATES` を全て、8 の倍数の中規模形状で検証するテスト
// （構成別の一致確認）は、`CANDIDATES` がクレート内部表現のため `pub(crate)`
// であり（codex-review 指摘・PR #651）、本統合テスト（クレート外）からは
// 参照できなくなったため、`crates/backend-metal/src/tile.rs` の
// `#[cfg(test)] mod tests`（`all_tile_candidates_match_cpu_reference_medium_shape`）
// へ移設済み。

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
        pad: 0,
    };
    run_case(cfg, 30, 31, 256, 256, 256);
}

/// 直接ロード経路（`staged=false`）かつ `dims.m`/`dims.n`（8 の倍数へ
/// pad8 済みの実効次元）が `BM`/`BN` を割り切らない境界形状。
/// `shaders/gemm.metal` の `gemm_simdgroup_tiled` 直接ロード経路が
/// `a_row < dims.m`・`b_col < dims.n` の境界チェックなしに
/// `simdgroup_load` すると範囲外読み出しになるケース（レビュー指摘。
/// #188 PR review。`m=n=k=256`・`bm=bn=32` の組で境界に到達しない
/// `direct_load_path_matches_cpu_reference` の見落としを補う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn direct_load_path_matches_cpu_reference_non_multiple_of_tile() {
    let cfg = TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: false,
        pad: 0,
    };
    // pad8(100)=104（32 の倍数でない）・pad8(70)=72 で
    // BM/BN=32 を割り切らない実効次元を作る。
    run_case(cfg, 32, 33, 100, 70, 70);
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
        pad: 4,
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
        pad: 4,
    };
    run_case(cfg, 7, 8, 64, 64, 4096);
}

// --- イシュー #532: MLX classic 経路の未収録 3 構成 ---

/// `bk=32`（本実装初採用。イシュー #532）の境界形状ケース: `k` が 32 の
/// 倍数にならない実効次元（pad8(70)=72）で `bk_eff` 末尾 0 埋め経路
/// （`shaders/gemm.metal` の境界チェック。REQ-8）に到達することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bk32_candidate_matches_cpu_reference_non_multiple_of_tile() {
    let cfg = TileConfig {
        bm: 64,
        bn: 32,
        bk: 32,
        wm: 2,
        wn: 2,
        staged: true,
        pad: 4,
    };
    run_case(cfg, 60, 61, 100, 70, 70);
}

/// `bk=32`（イシュー #532 の性能動機）の K ストレスケース: K=4096 の長い
/// 内積で `threadgroup_barrier` 往復半減の狙いが数値正しさを損なわないこと
/// を確認する（`tiled_matches_cpu_reference_k_stress` の bk=16 版に対応）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bk32_candidate_matches_cpu_reference_k_stress() {
    let cfg = TileConfig {
        bm: 64,
        bn: 32,
        bk: 32,
        wm: 2,
        wn: 2,
        staged: true,
        pad: 4,
    };
    run_case(cfg, 62, 63, 64, 64, 4096);
}

/// `(64,32,8,4,1)`（wm=4 縦分担・bk=8 小刻み。イシュー #532）の境界形状:
/// m/n/k いずれも bm/bn/bk の倍数でない形状で REQ-8 手動境界チェックの
/// 実効を確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn wm4_bk8_candidate_matches_cpu_reference_non_multiple_of_tile() {
    let cfg = TileConfig {
        bm: 64,
        bn: 32,
        bk: 8,
        wm: 4,
        wn: 1,
        staged: true,
        pad: 4,
    };
    run_case(cfg, 64, 65, 100, 70, 70);
}

/// `(64,64,16,1,2)`（少 simdgroup・acc_rows が `MAX_ACC` ちょうどの境界。
/// イシュー #532）の境界形状: m/n/k いずれも bm/bn/bk の倍数でない形状で
/// REQ-8 手動境界チェックの実効を確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn wm1_wn2_candidate_matches_cpu_reference_non_multiple_of_tile() {
    let cfg = TileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 1,
        wn: 2,
        staged: true,
        pad: 4,
    };
    run_case(cfg, 66, 67, 100, 130, 70);
}

// デバイス上限直接検証（イシュー #532 受け入れ基準「SMEM 上限内の実機
// 確認」）は `crate::tile::CANDIDATES` を直接参照する必要があるが、
// `CANDIDATES` はクレート内部表現のため `pub(crate)` であり
// （codex-review 指摘・PR #651）、本統合テスト（クレート外）からは
// 参照できなくなったため、`crates/backend-metal/src/tile.rs` の
// `#[cfg(test)] mod tests`
// （`all_tile_candidates_validate_under_actual_device_shared_memory_limit`）
// へ移設済み。
