//! `crate::tile::CANDIDATES` の追加タイル候補（`[0]`/`[3]`/`[4]`/`[5]`/`[6]`）
//! を `MetalGemm::dispatch_tiled_prepared`（転送非計測）で比較する再実行用
//! example（イシュー #1036）。
//!
//! `docs/perf/metal-gemm-bottleneck-rediagnosis.md` §4.4 の追加タイル候補
//! スイープは、当初「一時計測バイナリ（`dispatch_tiled_prepared` を直接
//! 呼ぶ最小 example・本 PR には含めない）」で計測しソースを削除していた
//! ため、第三者が再現できないという codex-review 指摘（PR #1096・P2）を
//! 受けて本 example として収録する。計測対象・計測プロトコルは同 doc §4.4
//! の生ログ（`docs/perf/logs/metal-gemm-rediagnosis-1036/step4b_extra_candidates.log`）
//! と同一にする:
//!
//! - `TileConfig` 5 種は `crate::tile::CANDIDATES`（`pub(crate)` のため本
//!   example では値を直接複製する。`gemm_bench.rs` の候補比較セクションと
//!   同じ手法）の index 0・3・4・5・6 と同一の値
//! - size ∈ {1024, 2048, 4096}
//! - `bench-harness::protocol::run`・`MeasurementConfig::default()`
//!   （warmup 20 回・計測 20 回・中央値。TASK-8.1）
//! - 決定的シード `SEED = 0xC0FFEE`（`gemm_bench.rs`・
//!   `gemm_f32_prepared_bench.rs` と同一値）
//!
//! 出力は `size=<N> candidate=<label> tflops=<中央値> resolved_matches_requested=<bool>`
//! （生ログと同一形式）。`resolved_matches_requested` は要求した
//! `TileConfig` が `pipeline_for_tile` のフォールバック chain を経ても
//! そのまま採用されたか（`resolved_cfg == cfg`）を示す（`gemm_bench.rs`
//! の `measure_tiled_prepared` と同じ検証観点）。
//!
//! 実行はプロセス間で GPU クロック状態が変動しうる（同 doc §3.4 の
//! 「原因未確定のプロセス間変動」）ため、本 example の再実行値は絶対値では
//! なく候補間の相対順位の確認用として扱う（同 doc §4.4 参照）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs`・
//! `gemm_f32_prepared_bench.rs` と同一（self-hosted runner をベンチ実行で
//! 占有しない・Linux CI でもビルド検証のみ通す）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_tile_sweep --release
//! ```

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, TileConfig};

    /// 決定的シード（`gemm_bench.rs`・`gemm_f32_prepared_bench.rs` と同一値）。
    const SEED: u64 = 0xC0FFEE;

    fn tflops(size: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (size as f64).powi(3);
        flops / median_secs / 1e12
    }

    /// `crate::tile::CANDIDATES`（`pub(crate)`）の index 0・3・4・5・6 と
    /// 同一の値を複製したラベル付き候補列。`select` の添字依存（`tile.rs`
    /// 冒頭コメント参照）とは無関係な計測専用の複製であり、`tile.rs` 側の
    /// 配列が変わった場合は本 example 側も追従する必要がある（値は
    /// #1036 doc §4.4 の生ログ・`select` 未経由の明示指定という同 doc の
    /// 計測方針に合わせて固定）。
    fn candidates() -> [(&'static str, TileConfig); 5] {
        [
            (
                "cand0_64x64x16_wm2wn2_staged",
                TileConfig {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand3_32x32x16_wm2wn2_staged",
                TileConfig {
                    bm: 32,
                    bn: 32,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand4_64x64x16_wm1wn2_staged",
                TileConfig {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                    wm: 1,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand5_64x32x32_wm2wn2_staged",
                TileConfig {
                    bm: 64,
                    bn: 32,
                    bk: 32,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand6_64x32x8_wm4wn1_staged",
                TileConfig {
                    bm: 64,
                    bn: 32,
                    bk: 8,
                    wm: 4,
                    wn: 1,
                    staged: true,
                },
            ),
        ]
    }

    /// `gemm_bench.rs::measure_tiled_prepared` と同一の計測境界（バッファは
    /// ループ外で 1 回だけ確保・アップロードし、計測対象はディスパッチの
    /// みに絞る）。候補比較の目的上、転送コストを含めない。
    fn measure_tiled_prepared(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        cfg: TileConfig,
        size: usize,
        config: &MeasurementConfig,
    ) -> (f64, TileConfig) {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        let a_buf = MetalBuffer::new_with_data(ctx, &a)
            .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(ctx, &b)
            .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(ctx, size * size)
            .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");

        // resolved_cfg は cfg・デバイス限界のみに依存し計測ループ中は不変
        // なため、warmup/計測ループへ入る前に 1 回だけ確定させる
        // （`gemm_bench.rs::measure_tiled_prepared` と同じ扱い）。
        let resolved_cfg = gemm
            .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
            .expect("Metal GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");

        let measurement = bench_run(config, || {
            gemm.dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
                .expect("Metal GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        (tflops(size, measurement.median_secs), resolved_cfg)
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for size in [1024usize, 2048, 4096] {
            let config = MeasurementConfig::default();
            for (label, cfg) in candidates() {
                let (tflops, resolved_cfg) =
                    measure_tiled_prepared(&gemm, &ctx, cfg, size, &config);
                println!(
                    "size={size} candidate={label} tflops={tflops:.4} resolved_matches_requested={}",
                    resolved_cfg == cfg,
                );
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`objc2` 系は `cfg(target_os = "macos")` 限定の
/// ため本クレートの GEMM 実装自体がコンパイル対象外になる。Linux CI の
/// `cargo build --workspace --all-targets`／`cargo clippy --all-targets`
/// をこの example も含めて通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_tile_sweep example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p fandhe-ai-backend-metal --example gemm_tile_sweep --release"
    );
}
