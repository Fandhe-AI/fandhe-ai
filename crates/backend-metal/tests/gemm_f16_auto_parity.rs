//! f16 動的タイル選択の自動入口 `MetalGemm::dispatch_f16_auto`（イシュー
//! #798）の数値一致回帰テスト。
//!
//! `tests/cpu_metal_f16_tiled_parity.rs`（明示 `TileConfig` 指定の
//! `dispatch_f16_tiled_unverified`）と同じ判定基盤
//! （`backend_cpu::parity::assert_parity`。REQ-2 統一複合判定「相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満」の唯一の実体。許容誤差は変更しない
//! `.claude/rules/coding-rust.md`）・同じ 3 段階の参照値構築方法（f16 入力を
//! f32 化 → `matmul_reference_fma` → f16 丸め → f32 化して比較）を使う。
//!
//! 本ファイルが追加で検証するのは「`dispatch_f16_auto` が `tile::select`
//! による動的タイル選択で dispatch した結果が CPU 参照実装と一致すること」
//! （イシュー #798 受け入れ条件 1・2）であり、`tile::select` の分岐
//! （縦長・横長・正方立方・準正方大形状長方形・微小形状）を代表する形状を
//! 選定してケース化する（`crates/backend-metal/src/tile.rs::
//! select_with_occupancy` 本体コメントの分岐条件と同一）。8 非整列の端数
//! 形状（`pad8` パディング・アンパディング経路の回帰）と、既存
//! `dispatch_f16_unverified`（非タイル 8x8。後方互換方針で存置）との直接
//! 照合ケースも合わせて含む。
//!
//! `gemm_auto_parity.rs`（f32 `dispatch_backend_auto` の対応版）と同じく
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`#![cfg(target_os = "macos")]` ＋ 全ケース `#[ignore]`）。実機アクセス
//! 不可時は `cargo test -p backend-metal --release -- --ignored --nocapture`
//! の実行を #799 実機セッションへ引き継ぐ（#796 の
//! `tests/cpu_metal_f16_tiled_parity.rs` 引き継ぎと同一様式）。
//!
//! ```sh
//! cargo test -p backend-metal --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use backend_cpu::parity::{assert_parity, matmul_reference_fma};
use backend_metal::tile::TileConfig;
use backend_metal::{MetalContext, MetalGemm};
use bench_harness::rng::Xorshift64Star;
use half::f16;

/// 決定的シードで A・B（f16）を生成し、f16→f32→参照 matmul→f16 丸め→f32 の
/// 経路で得た参照値と `dispatch_f16_auto` の出力（f16→f32）を
/// `assert_parity` で照合する（`cpu_metal_f16_tiled_parity.rs::
/// assert_metal_f16_tiled_parity` と同型。`cfg` を明示せず `dispatch_f16_auto`
/// 内部の `tile::select` に委ねる点のみ異なる）。
fn assert_dispatch_f16_auto_parity(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    context: &str,
    seed: u64,
    m: usize,
    n: usize,
    k: usize,
) {
    let mut rng = Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16(m * k);
    let b_f16: Vec<f16> = rng.fill_vec_f16(k * n);

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; m * n];
    matmul_reference_fma(&a_f32, &b_f32, &mut c_ref_f32, m, n, k)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .dispatch_f16_auto(ctx, &a_f16, &b_f16, m, n, k)
        .expect("MetalGemm::dispatch_f16_auto must succeed on Metal-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    assert_parity(context, &c_gpu_f32, &c_ref_rounded);
}

/// 正方立方 512³（`tile::select` は `CANDIDATES[3]` を選択。PoC-v2-5 基準
/// 規模）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_cpu_reference_square_cube_512() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_dispatch_f16_auto_parity(
        &ctx,
        &gemm,
        "f16 dispatch_f16_auto 正方立方 512x512x512",
        500,
        512,
        512,
        512,
    );
}

/// 縦長形状（`tile::select` は `CANDIDATES[1]` を選択）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_cpu_reference_tall_shape() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_dispatch_f16_auto_parity(
        &ctx,
        &gemm,
        "f16 dispatch_f16_auto 縦長 2048x256x512",
        501,
        2048,
        256,
        512,
    );
}

/// 横長形状（`tile::select` は `CANDIDATES[2]` を選択）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_cpu_reference_wide_shape() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_dispatch_f16_auto_parity(
        &ctx,
        &gemm,
        "f16 dispatch_f16_auto 横長 256x2048x512",
        502,
        256,
        2048,
        512,
    );
}

/// 準正方大形状長方形（`tile::select` は `CANDIDATES[0]` を選択）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_cpu_reference_large_near_square_rectangle() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_dispatch_f16_auto_parity(
        &ctx,
        &gemm,
        "f16 dispatch_f16_auto 準正方大形状長方形 1536x1024x512",
        503,
        1536,
        1024,
        512,
    );
}

/// 微小形状（`tile::select` は `TileConfig::SINGLE_SIMDGROUP_8X8` へ縮退）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_cpu_reference_tiny_shape() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_dispatch_f16_auto_parity(
        &ctx,
        &gemm,
        "f16 dispatch_f16_auto 微小形状 32x32x32",
        504,
        32,
        32,
        32,
    );
}

/// 8 非整列の端数形状（`pad8` パディング・アンパディングと手動境界検査の
/// 回帰。REQ-8）。`m`/`n`/`k` すべて `SMALL`（64）以上にして
/// `SINGLE_SIMDGROUP_8X8`（微小形状フォールバック）を経由させず、
/// staged 経路（`CANDIDATES[3]`。`tile.rs::tests::
/// dispatch_f16_auto_shapes_resolve_to_expected_candidate` で
/// `select(521, 265, 131) == CANDIDATES[3]` を Linux 上でも固定している）で
/// `pad8` パディング・アンパディングを踏む（旧 130x70x54 は k=54 < 64 で
/// `SINGLE_SIMDGROUP_8X8` へ縮退し微小形状ケースと重複していたため是正。
/// advisor 指摘）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_cpu_reference_non_multiple_of_8_boundary_shape() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    assert_dispatch_f16_auto_parity(
        &ctx,
        &gemm,
        "f16 dispatch_f16_auto 端数形状 521x265x131",
        505,
        521,
        265,
        131,
    );
}

/// イシュー #798 後方互換方針: 既存 `gemm_simdgroup_f16`
/// （`dispatch_f16_unverified`。非タイル 8x8。自動経路の縮退先ではなく
/// 明示入口専用の計測・回帰基線として存置する）と `dispatch_f16_auto`
/// が同一入力に対して統一複合判定（REQ-2）で一致する結果を返すことを
/// 直接照合する（`cpu_metal_f16_tiled_parity.rs::
/// f16_tiled_parity_matches_existing_8x8_kernel` の自動経路版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_existing_non_tiled_8x8_kernel() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    let (m, n, k) = (256, 192, 320);
    let mut rng = Xorshift64Star::new(506);
    let a: Vec<f16> = rng.fill_vec_f16(m * k);
    let b: Vec<f16> = rng.fill_vec_f16(k * n);

    let baseline = gemm
        .dispatch_f16_unverified(&ctx, &a, &b, m, n, k)
        .expect("既存 gemm_simdgroup_f16（dispatch_f16_unverified）の dispatch に失敗した");
    let auto = gemm
        .dispatch_f16_auto(&ctx, &a, &b, m, n, k)
        .expect("dispatch_f16_auto の dispatch に失敗した");

    let baseline_f32: Vec<f32> = baseline.iter().map(|x| x.to_f32()).collect();
    let auto_f32: Vec<f32> = auto.iter().map(|x| x.to_f32()).collect();
    assert_parity(
        "f16 dispatch_f16_auto vs 既存 8x8 カーネル 256x192x320",
        &auto_f32,
        &baseline_f32,
    );
}

/// `MetalGemm::dispatch_f16_tiled_unverified`（明示 `TileConfig` 指定入口）
/// が `tile::select` の出力をそのまま受け取っても `dispatch_f16_auto` と
/// 同一の結果を返すことを確認する（`dispatch_f16_auto` 自体が内部で
/// `tile::select(m, n, k)` → `dispatch_f16_tiled_unverified` へ委譲する薄い
/// ラッパーであることの直接照合。`gemm.rs::MetalGemm::dispatch_f16_auto`
/// ドキュメンテーションコメント参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_f16_auto_matches_explicit_tile_select_dispatch() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

    let (m, n, k) = (512, 512, 512);
    let mut rng = Xorshift64Star::new(507);
    let a: Vec<f16> = rng.fill_vec_f16(m * k);
    let b: Vec<f16> = rng.fill_vec_f16(k * n);

    let cfg: TileConfig = backend_metal::tile::select(m, n, k);
    let explicit = gemm
        .dispatch_f16_tiled_unverified(&ctx, &a, &b, m, n, k, cfg)
        .expect("dispatch_f16_tiled_unverified の dispatch に失敗した");
    let auto = gemm
        .dispatch_f16_auto(&ctx, &a, &b, m, n, k)
        .expect("dispatch_f16_auto の dispatch に失敗した");

    let explicit_f32: Vec<f32> = explicit.iter().map(|x| x.to_f32()).collect();
    let auto_f32: Vec<f32> = auto.iter().map(|x| x.to_f32()).collect();
    assert_parity(
        "f16 dispatch_f16_auto vs 明示 tile::select 出力の dispatch 512x512x512",
        &auto_f32,
        &explicit_f32,
    );
}
