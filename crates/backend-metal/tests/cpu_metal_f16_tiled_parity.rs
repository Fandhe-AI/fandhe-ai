//! CPU-Metal ペアの数値一致回帰テスト: f16 タイル化 GEMM
//! `gemm_simdgroup_tiled_f16`（イシュー #796）。
//!
//! `tests/cpu_metal_f16_parity.rs`（非タイル `gemm_simdgroup_f16`）と同じ
//! 判定基盤（`backend_cpu::assert_parity`。REQ-2 統一複合判定「相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満」の唯一の実体）・同じ 3 段階の参照値
//! 構築方法（f16 入力を f32 化 → `matmul_reference_fma` → f16 丸め →
//! f32 化して比較。同ファイル冒頭コメント参照）を使う。本ファイルが追加で
//! 検証するのは 2 点:
//!
//! 1. `gemm_simdgroup_tiled_f16` の出力が CPU 参照実装と REQ-2 統一複合
//!    判定で一致すること（direct-load・staged 双方の代表構成 × 複数形状）。
//!    `crate::tile::CANDIDATES` は `pub(crate)`（クレート外非公開）のため
//!    全構成の横断巡回は `crates/backend-metal/src/tile.rs` の
//!    `#[cfg(test)] mod tests`
//!    （`all_tile_candidates_match_cpu_reference_f16_tiled_medium_shape`）
//!    へ集約する（f32 版 `gemm_dynamic_tile_parity.rs` 冒頭コメントと同じ
//!    判断根拠。本ファイルでは個別に選んだ代表構成のみを検証する）。
//! 2. イシュー #796 受け入れ条件 1「既存 8x8 カーネル
//!    〈`gemm_simdgroup_f16`〉と統一複合判定で一致する」: 同一入力に対して
//!    `dispatch_f16_unverified`（既存）と `dispatch_f16_tiled_unverified`
//!    （本イシュー新設）の出力を直接照合する。
//!
//! # 累算精度契約
//!
//! `gemm_simdgroup_tiled_f16` は `gemm_simdgroup_f16` と同じ精度契約
//! （A/B は `MM_T`＝`simdgroup_half8x8`、アキュムレータは
//! `ACC_T`＝`simdgroup_float8x8`。f32 累算）を共有する（`shaders/gemm.metal`
//! 該当カーネル冒頭コメント参照）ため、参照値構築の丸めタイミングも
//! `cpu_metal_f16_parity.rs` と同一にする。
//!
//! # 実行環境
//!
//! `cpu_metal_f16_parity.rs` と同じく `#![cfg(target_os = "macos")]` で
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。全ケース
//! `#[ignore]` とし、`cargo test -p backend-metal --release -- --ignored
//! --nocapture` で実行する（実装計画 §6.2）。実機アクセス不可時の扱いは
//! 実装計画 §6.3（テスト実装・クロスビルド確認済みの状態で実機実測を
//! 後続へ引き継ぐ）を参照。

#![cfg(target_os = "macos")]

use backend_cpu::parity::assert_parity;
use backend_metal::tile::TileConfig;
use backend_metal::{MetalContext, MetalGemm};
use bench_harness::rng::Xorshift64Star;
use half::f16;

/// 決定的シードで A・B（f16）を生成し、f16→f32→参照 matmul→f16 丸め→f32 の
/// 経路で得た参照値と `gemm_simdgroup_tiled_f16`（`cfg` で指定した
/// `TileConfig`）の出力（f16→f32）を `assert_parity` で照合する
/// （`cpu_metal_f16_parity.rs::assert_metal_f16_parity` と同型。`cfg` 引数が
/// 追加された差分のみ）。
#[allow(clippy::too_many_arguments)]
fn assert_metal_f16_tiled_parity(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    context: &str,
    seed: u64,
    m: usize,
    n: usize,
    k: usize,
    cfg: TileConfig,
) {
    let mut rng = Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16(m * k);
    let b_f16: Vec<f16> = rng.fill_vec_f16(k * n);

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; m * n];
    backend_cpu::parity::matmul_reference_fma(&a_f32, &b_f32, &mut c_ref_f32, m, n, k)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .dispatch_f16_tiled_unverified(ctx, &a_f16, &b_f16, m, n, k, cfg)
        .expect(
            "MetalGemm::dispatch_f16_tiled_unverified must succeed on Metal-equipped test runner",
        );
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    assert_parity(context, &c_gpu_f32, &c_ref_rounded);
}

/// staged 経路の代表構成（32x32/bk16/wm2/wn2。`crate::tile::CANDIDATES[3]`
/// と同値）の PoC-v2-5 基準規模（512x512x512）での数値一致。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_tiled_parity_staged_baseline_shape_512() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    let cfg = TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };
    assert_metal_f16_tiled_parity(
        &ctx,
        &gemm,
        "f16 tiled staged baseline 512x512x512",
        200,
        512,
        512,
        512,
        cfg,
    );
}

/// 単一 simdgroup 8x8（direct-load 分岐。`USE_TGP_STAGING=false`）の数値
/// 一致。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_tiled_parity_direct_load_single_simdgroup_8x8() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_metal_f16_tiled_parity(
        &ctx,
        &gemm,
        "f16 tiled direct-load 64x64x128",
        211,
        64,
        64,
        128,
        TileConfig::SINGLE_SIMDGROUP_8X8,
    );
}

/// BM/BN/BK の倍数でない非正方形状（REQ-8 手動境界検査・`pad8` パディング
/// 経路の回帰）。staged 構成（32x32/bk16/wm2/wn2）を明示指定する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_tiled_parity_boundary_shape_non_multiple_of_tile() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    let cfg = TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };
    assert_metal_f16_tiled_parity(
        &ctx,
        &gemm,
        "f16 tiled boundary 100x50x72",
        221,
        100,
        50,
        72,
        cfg,
    );
}

/// K=4096 ストレスケース（PoC-v2-5 準拠の積和蓄積検証。f32 累算の桁落ち
/// 耐性を確認する中核ケース。`cpu_metal_f16_parity.rs::f16_k4096_stress`
/// と同じ規模で staged 構成〈64x64/bk16/wm2/wn2〉を使う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_tiled_k4096_stress() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    let cfg = TileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };
    assert_metal_f16_tiled_parity(
        &ctx,
        &gemm,
        "f16 tiled K4096 stress 256x256x4096",
        231,
        256,
        256,
        4096,
        cfg,
    );
}

/// イシュー #796 受け入れ条件 1: 既存 `gemm_simdgroup_f16`
/// （`dispatch_f16_unverified`）と `gemm_simdgroup_tiled_f16`
/// （`dispatch_f16_tiled_unverified`）が同一入力に対して統一複合判定
/// （REQ-2）で一致する結果を返すことを直接照合する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_tiled_parity_matches_existing_8x8_kernel() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    let (m, n, k) = (256, 192, 320);
    let mut rng = Xorshift64Star::new(241);
    let a: Vec<f16> = rng.fill_vec_f16(m * k);
    let b: Vec<f16> = rng.fill_vec_f16(k * n);

    let cfg = TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    };

    let baseline = gemm
        .dispatch_f16_unverified(&ctx, &a, &b, m, n, k)
        .expect("既存 gemm_simdgroup_f16（dispatch_f16_unverified）の dispatch に失敗した");
    let tiled = gemm
        .dispatch_f16_tiled_unverified(&ctx, &a, &b, m, n, k, cfg)
        .expect("gemm_simdgroup_tiled_f16（dispatch_f16_tiled_unverified）の dispatch に失敗した");

    let baseline_f32: Vec<f32> = baseline.iter().map(|x| x.to_f32()).collect();
    let tiled_f32: Vec<f32> = tiled.iter().map(|x| x.to_f32()).collect();
    assert_parity(
        "f16 tiled vs 既存 8x8 カーネル 256x192x320",
        &tiled_f32,
        &baseline_f32,
    );
}
