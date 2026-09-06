//! `FINE_BARRIER_ENABLED`（simdgroup 細粒度同期。イシュー #809・#1278）の
//! 数値契約: base（`fine_barrier_enabled=false`）/head（`true`）の出力が
//! ビット単位で一致することを実機で確認する受け入れテスト（AC-1）。
//!
//! `examples/gemm_fine_barrier_ab_bench.rs` のフェーズ 0 は `dispatch_auto`
//! 経路のみを毎回の A/B 計測前セルフチェックとして自己検証するが、本ファイル
//! は独立した `#[ignore]` テストとして `dispatch_auto`・`dispatch_tiled_prepared`
//! の両経路を size 512/1024/2048/4096 で固定する（イシュー #1278 実装計画
//! §3.1 (2)）。`simdgroup_barrier(mem_flags::mem_none)` の挿入は
//! `tile_a`/`tile_b`（threadgroup メモリ）確定後の演算オペランド列を一切
//! 変えないため（`docs/perf/metal-gemm-fine-barrier-ab.md` 参照）、両経路
//! とも `assert_eq!` の厳密一致（tolerance 不使用）で検証する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（GitHub
//! ホステッド・ubuntu-latest）では `#![cfg(target_os = "macos")]` により
//! コンパイル対象外になり、`#[ignore]` により通常の `cargo test` からも
//! 除外される（`tests/gemm_dynamic_tile_parity.rs` と同じ方針）。実行するには
//! macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_fine_barrier_bit_match -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};

/// `gemm_fine_barrier_ab_bench.rs::SEED` と同一値（決定的シード。同一
/// プロトコルであることを明示するため揃える）。
const SEED: u64 = 0xC0FFEE;

/// `dispatch_auto`（本番既定の自動タイル選択経路）で base/head の出力が
/// ビット単位で一致することを size 512/1024/2048/4096 で確認する。
/// `examples/gemm_fine_barrier_ab_bench.rs::phase0_bit_match_selfcheck`
/// （256/512/1024/2048/4096 を A/B 計測直前に自己検証）と重複するが、
/// 本テストは実機実行のたびに独立して実行可能な受け入れテストとして
/// `#[ignore]` で分離しておく（イシュー #1278 AC-1）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn fine_barrier_on_off_bit_match_dispatch_auto() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base_gemm = MetalGemm::new_with_fine_barrier(&ctx, false)
        .expect("base GEMM パイプラインの構築に失敗した");
    let head_gemm = MetalGemm::new_with_fine_barrier(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    for size in [512usize, 1024, 2048, 4096] {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        let base_out = base_gemm
            .dispatch_auto(&ctx, &a, &b, size, size, size)
            .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
        let head_out = head_gemm
            .dispatch_auto(&ctx, &a, &b, size, size, size)
            .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

        assert_eq!(
            base_out, head_out,
            "size={size}: dispatch_auto で FINE_BARRIER_ENABLED の有無により出力がビット単位で\
             一致しなかった。演算オペランド列が変わっている疑いがあるため、shaders/gemm.metal の\
             FINE_BARRIER_ENABLED 挿入箇所を確認すること。"
        );
    }
}

/// `dispatch_tiled_prepared`（`examples/gemm_fine_barrier_ab_bench.rs` の
/// フェーズ 2 が実際に A/B 計測する入口。アップロード済みバッファ・確定
/// `TileConfig` 直接指定）で base/head の出力がビット単位で一致することを
/// size 512/1024/2048/4096 で確認する。`dispatch_auto` とは異なるコード
/// パス（バッファ確保・エンコードの分離）を通るため独立に検証する必要が
/// ある（イシュー #1278 AC-1）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn fine_barrier_on_off_bit_match_dispatch_tiled_prepared() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base_gemm = MetalGemm::new_with_fine_barrier(&ctx, false)
        .expect("base GEMM パイプラインの構築に失敗した");
    let head_gemm = MetalGemm::new_with_fine_barrier(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    for size in [512usize, 1024, 2048, 4096] {
        let cfg = tile::select_for_device(size, size, size, ctx.verified_m4_max_gpu_core_count());

        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        let a_buf = MetalBuffer::new_with_data(&ctx, &a)
            .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b)
            .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let base_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
            .expect("base C バッファの確保に失敗した（実機でのみ実行する前提）");
        let head_c_buf = MetalBuffer::new_zeroed(&ctx, size * size)
            .expect("head C バッファの確保に失敗した（実機でのみ実行する前提）");

        base_gemm
            .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &base_c_buf, size, size, size, cfg)
            .expect("base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
        head_gemm
            .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &head_c_buf, size, size, size, cfg)
            .expect("head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");

        let base_out = base_c_buf.read_to_vec();
        let head_out = head_c_buf.read_to_vec();

        assert_eq!(
            base_out, head_out,
            "size={size}: dispatch_tiled_prepared で FINE_BARRIER_ENABLED の有無により出力が\
             ビット単位で一致しなかった。演算オペランド列が変わっている疑いがあるため、\
             shaders/gemm.metal の FINE_BARRIER_ENABLED 挿入箇所を確認すること。"
        );
    }
}
