//! `SWIZZLE_ENABLED`（threadgroup ID スウィズル。イシュー #540・#795・
//! #1279）の数値契約: base（`swizzle_enabled=false`）/head（`true`）の
//! 出力がビット単位で一致することを実機で確認する受け入れテスト（AC-1）。
//!
//! `examples/gemm_swizzle_ab_bench.rs` のフェーズ 0 は `dispatch_auto`
//! 経路のみを毎回の A/B 計測前セルフチェックとして自己検証するが、本ファイル
//! は独立した `#[ignore]` テストとして `dispatch_auto`・`dispatch_tiled_prepared`
//! の両経路を size 512/1024/2048/4096 で固定する
//! （`tests/gemm_fine_barrier_bit_match.rs`〈#809・#1278〉と同型構成）。
//!
//! threadgroup ID スウィズルは各 C タイルを担当する threadgroup の
//! tgid→タイル座標写像のみを変え、その threadgroup が実行する演算
//! （K ループ順・FMA 契約・simdgroup 分担）自体は変わらない。また各出力
//! 要素は正確に 1 threadgroup が 1 回書くため（`shaders/gemm.metal` の
//! `tiled_block_out_of_range` 早期 return により余剰 threadgroup は無害化
//! される。`docs/perf/metal-gemm-tgid-swizzle-ab.md` 参照）、base/head の
//! 出力は演算オペランド列レベルで不変であり、両経路とも `assert_eq!` の
//! 厳密一致（tolerance 不使用）で検証する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（GitHub
//! ホステッド・ubuntu-latest）では `#![cfg(target_os = "macos")]` により
//! コンパイル対象外になり、`#[ignore]` により通常の `cargo test` からも
//! 除外される（`tests/gemm_fine_barrier_bit_match.rs` と同じ方針）。実行するには
//! macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_swizzle_bit_match -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};

/// `gemm_swizzle_ab_bench.rs::SEED` と同一値（決定的シード。同一
/// プロトコルであることを明示するため揃える）。
const SEED: u64 = 0xC0FFEE;

/// AC-1 のビット単位一致検証を `to_bits()` 経由で行う（`assert_eq!` の
/// `f32` 数値比較は IEEE 754 の `+0.0 == -0.0` を区別できず、符号ビットの
/// 差異を見逃しうるため。`gemm_fine_barrier_bit_match.rs::assert_bit_exact`
/// と同じ理由。PR #1372 codex-review P2 指摘 discussion r3943195893 の
/// 教訓を踏襲）。`NaN` は `to_bits()` がビットパターンをそのまま返すため
/// `NaN != NaN` の数値比較特性に引きずられず、ビットパターンが完全一致
/// する場合のみ一致と判定する。
fn assert_bit_exact(base: &[f32], head: &[f32], size: usize, dispatch_name: &str) {
    let base_bits: Vec<u32> = base.iter().map(|v| v.to_bits()).collect();
    let head_bits: Vec<u32> = head.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        base_bits, head_bits,
        "size={size}: {dispatch_name} で SWIZZLE_ENABLED の有無により出力が\
         ビット単位で一致しなかった。tgid→タイル座標写像以外の演算オペランド列が\
         変わっている疑いがあるため、shaders/gemm.metal の SWIZZLE_ENABLED \
         挿入箇所を確認すること。"
    );
}

/// `dispatch_auto`（本番既定の自動タイル選択経路）で base/head の出力が
/// ビット単位で一致することを size 512/1024/2048/4096 で確認する。
/// `examples/gemm_swizzle_ab_bench.rs::phase0_bit_match_selfcheck`
/// （256/512/1024/2048/4096 を A/B 計測直前に自己検証）と重複するが、
/// 本テストは実機実行のたびに独立して実行可能な受け入れテストとして
/// `#[ignore]` で分離しておく（イシュー #1279 AC-1）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn swizzle_on_off_bit_match_dispatch_auto() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base_gemm =
        MetalGemm::new_with_swizzle(&ctx, false).expect("base GEMM パイプラインの構築に失敗した");
    let head_gemm =
        MetalGemm::new_with_swizzle(&ctx, true).expect("head GEMM パイプラインの構築に失敗した");

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

        assert_bit_exact(&base_out, &head_out, size, "dispatch_auto");
    }
}

/// `dispatch_tiled_prepared`（`examples/gemm_swizzle_ab_bench.rs` のフェーズ
/// 2 が実際に A/B 計測する入口。アップロード済みバッファ・確定
/// `TileConfig` 直接指定。余剰 threadgroup を生む `tiles_m` 非倍数の形状も
/// swizzle 時の grid 拡張〈`tiles_n << SWIZZLE_LOG`〉を通じて発生しうるため、
/// `tiled_block_out_of_range` ガードの正当性もここで実機検証する）で
/// base/head の出力がビット単位で一致することを size 512/1024/2048/4096
/// で確認する。`dispatch_auto` とは異なるコードパス（バッファ確保・
/// エンコードの分離）を通るため独立に検証する必要がある（イシュー #1279
/// AC-1）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn swizzle_on_off_bit_match_dispatch_tiled_prepared() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base_gemm =
        MetalGemm::new_with_swizzle(&ctx, false).expect("base GEMM パイプラインの構築に失敗した");
    let head_gemm =
        MetalGemm::new_with_swizzle(&ctx, true).expect("head GEMM パイプラインの構築に失敗した");

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

        assert_bit_exact(&base_out, &head_out, size, "dispatch_tiled_prepared");
    }
}
