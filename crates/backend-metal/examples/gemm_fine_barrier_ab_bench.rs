//! simdgroup 細粒度同期（イシュー #809）の A/B 再計測バイナリ。
//!
//! 親ツリー #735（GEMM 第 2 次最適化）→ #790（phase-5 補完・低優先改修）
//! 配下。MLX steel `mma.h`（`docs/perf/metal-gemm-bottleneck-diagnosis.md`
//! 〈#487〉・`docs/backend-metal-mlx-classic-nax-decision.md`〈#549〉で構成
//! 対比済み）が simdgroup フラグメントロード間に用いる
//! `simdgroup_barrier(mem_flags::mem_none)`（`threadgroup_barrier` より
//! 軽量な simdgroup スコープのフェンス）を、`gemm_simdgroup_tiled` の
//! staged 経路 kk ループへ適用する構成（`FINE_BARRIER_ENABLED` function
//! constant。`shaders/gemm.metal`）の性能効果を実測する。
//!
//! `gemm_swizzle_ab_bench.rs`（イシュー #540・#746）と同一プロトコル
//! （`bench_harness::ab`・`docs/perf/metal-bench-noise-protocol.md`）を
//! 踏襲する: フェーズ 0（bit 一致の自己検証）→ フェーズ 1（安定性
//! セルフチェック）→ フェーズ 2（prepared 境界 A/B）。
//!
//! フェーズ 0 を追加している理由: barrier 挿入は演算オペランド列を変えない
//! ため理論上 base/head の出力はビット単位で一致するはずである
//! （`shaders/gemm.metal` の `FINE_BARRIER_ENABLED` 宣言コメント参照）。
//! この数値契約を計測前に自己検証し、崩れている場合は A/B 計測へ進まない
//! （安全側判断: 数値が一致しない構成の性能を比較しても意味がないため）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs`
//! ドキュメンテーションコメント（同ディレクトリ）と同一。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_fine_barrier_ab_bench --release
//! ```
//!
//! 実行前後に `pmset -g therm` でサーマル状態を記録すること
//! （`docs/perf/metal-gemm-fine-barrier-ab.md` 実行手順参照）。フェーズ 1 で
//! いずれかのサイズが安定性ゲート（`bench_harness::ab::STABILITY_SPREAD_GATE`）
//! を超過した場合、フェーズ 2 の A/B 判定には進まない（「判定不可」を出力
//! して終了する。安全側判断: 判定を無効化して中断する方向のみ許す）。

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{AbConfig, run_ab, run_stability};
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};
    use std::time::Duration;

    /// `gemm_swizzle_ab_bench.rs` と同一値（決定的シード。過去 PoC・CPU 実装
    /// ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// フェーズ 1・フェーズ 2 共通のラウンド数・cooldown・時間ベース
    /// ウォームアップ下限（`gemm_swizzle_ab_bench.rs` と同一初期値。実機で
    /// 安定性ゲートを満たせない場合は手順書（`docs/perf/
    /// metal-bench-noise-protocol.md`）の調整手順に従い増やす方向のみ許す）。
    const ROUNDS: usize = 6;
    const COOLDOWN: Duration = Duration::from_secs(2);
    const MIN_WARMUP: Duration = Duration::from_secs(1);

    fn tflops(size: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (size as f64).powi(3);
        flops / median_secs / 1e12
    }

    /// `gemm_swizzle_ab_bench.rs::head_over_base_tflops_ratio` と同一ロジック
    /// （TFLOPS 比は実行時間比の逆数。取り違え防止のため 1 箇所に集約する）。
    fn head_over_base_tflops_ratio(size: usize, median_a_secs: f64, median_b_secs: f64) -> f64 {
        tflops(size, median_b_secs) / tflops(size, median_a_secs)
    }

    /// フェーズ 0: base（`fine_barrier_enabled=false`）/head（`true`）の出力が
    /// ビット単位で一致することを計測前に自己検証する（本ファイル冒頭
    /// ドキュメンテーションコメント参照）。不一致の場合は `panic` し、
    /// フェーズ 1・2 へは進まない（安全側判断）。
    fn phase0_bit_match_selfcheck(ctx: &MetalContext) {
        println!("--- フェーズ 0: base/head 出力 bit 一致の自己検証 ---");

        let base_gemm = MetalGemm::new_with_fine_barrier(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_fine_barrier(ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        for size in [256usize, 512, 1024] {
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let base_out = base_gemm
                .dispatch_auto(ctx, &a, &b, size, size, size)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(ctx, &a, &b, size, size, size)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            assert_eq!(
                base_out, head_out,
                "size={size}: FINE_BARRIER_ENABLED の有無で出力がビット単位で一致しなかった。\
                 演算オペランド列が変わっている疑いがあるため、A/B 計測（性能比較）は無意味になる。\
                 shaders/gemm.metal の FINE_BARRIER_ENABLED 挿入箇所を確認すること。"
            );
            println!("size={size} bit-exact match: OK");
        }
    }

    /// フェーズ 1: `gemm.dispatch_variant(GemmVariant::SimdgroupTiled(tile::select(..)))`
    /// 相当の `dispatch_auto`（本番既定経路と同一構成選択）を `size` ごとに
    /// [`run_stability`] で計測し、spread を出力する（`gemm_swizzle_ab_bench.rs`
    /// と同一構造）。
    fn phase1_stability_selfcheck(ctx: &MetalContext, gemm: &MetalGemm) -> bool {
        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        println!(
            "--- フェーズ 1: 安定性セルフチェック（対照カーネル: SimdgroupTiled auto 選択）---"
        );

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        let mut all_within_gate = true;
        for size in [256usize, 512, 1024, 2048, 4096] {
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let result = run_stability(&ab_config, &measurement_config, || {
                gemm.dispatch_auto(ctx, &a, &b, size, size, size)
                    .expect("Metal GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            })
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            let within_gate = result.spread <= SPREAD_GATE;
            all_within_gate &= within_gate;

            let round_tflops: Vec<f64> = result
                .round_medians_secs
                .iter()
                .map(|&secs| tflops(size, secs))
                .collect();
            println!(
                "size={size} spread={:.4} ({}) round_tflops={round_tflops:.4?}",
                result.spread,
                if within_gate { "OK" } else { "NG: gate 超過" }
            );
        }

        if !all_within_gate {
            println!(
                "--- フェーズ 1 判定: 一部サイズが spread ≤{SPREAD_GATE:.2} 相当を満たさなかった。\
                 フェーズ 2（A/B 判定）はスキップする（安全側判断: 判定不可のまま採否を確定しない）。"
            );
        }
        all_within_gate
    }

    /// フェーズ 2: `MetalGemm::new_with_fine_barrier` で base
    /// （`fine_barrier_enabled=false`）/head（`true`）の 2 インスタンスを
    /// 同一プロセス内に構築し、[`run_ab`] で interleaved 計測する
    /// （`gemm_swizzle_ab_bench.rs::phase2_swizzle_ab` と同一構造・同一計測
    /// 境界〈prepared。アップロード済みバッファを base/head で共有〉）。
    fn phase2_fine_barrier_ab(ctx: &MetalContext) {
        println!("--- フェーズ 2: simdgroup 細粒度同期 A/B（base=off / head=on）---");

        let base_gemm = MetalGemm::new_with_fine_barrier(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_fine_barrier(ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        for size in [256usize, 512, 1024, 2048, 4096] {
            let cfg =
                tile::select_for_device(size, size, size, ctx.verified_m4_max_gpu_core_count());

            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let a_buf = MetalBuffer::new_with_data(ctx, &a)
                .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let b_buf = MetalBuffer::new_with_data(ctx, &b)
                .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let c_buf = MetalBuffer::new_zeroed(ctx, size * size)
                .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");

            let base_resolved = base_gemm
                .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
                .expect("base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
            let head_resolved = head_gemm
                .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
                .expect("head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");

            let result = run_ab(
                &ab_config,
                &measurement_config,
                || {
                    base_gemm
                        .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
                        .expect("base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
                },
                || {
                    head_gemm
                        .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
                        .expect("head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
                },
            )
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            let base_tflops: Vec<f64> = result
                .a_round_medians_secs
                .iter()
                .map(|&secs| tflops(size, secs))
                .collect();
            let head_tflops: Vec<f64> = result
                .b_round_medians_secs
                .iter()
                .map(|&secs| tflops(size, secs))
                .collect();

            let base_median_tflops = tflops(size, result.median_a_secs);
            let head_median_tflops = tflops(size, result.median_b_secs);
            let head_over_base =
                head_over_base_tflops_ratio(size, result.median_a_secs, result.median_b_secs);

            println!(
                "size={size} base_resolved=({}x{}, {}) head_resolved=({}x{}, {}) \
                 base_median_tflops={:.4} head_median_tflops={:.4} head_over_base={:.4} \
                 spread_base={:.4} spread_head={:.4} base_round_tflops={base_tflops:.4?} \
                 head_round_tflops={head_tflops:.4?}",
                base_resolved.bm,
                base_resolved.bn,
                if base_resolved == cfg {
                    "resolved=requested"
                } else {
                    "resolved!=requested(fallback)"
                },
                head_resolved.bm,
                head_resolved.bn,
                if head_resolved == cfg {
                    "resolved=requested"
                } else {
                    "resolved!=requested(fallback)"
                },
                base_median_tflops,
                head_median_tflops,
                head_over_base,
                result.spread_a,
                result.spread_b,
            );
        }

        println!(
            "--- フェーズ 2 完了。採否判定基準（size 2048/4096 の中央値改善があり、かつ他\
             サイズで劣化中央値 5% 超がないこと。イシュー #809 計画 §3.1）に従い、\
             `docs/perf/metal-gemm-fine-barrier-ab.md` へ記録すること。"
        );
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

        phase0_bit_match_selfcheck(&ctx);

        // フェーズ 1 は本番既定（fine barrier off）の `MetalGemm::new` で行う
        // （対照カーネルの計測プロトコル自体を検証する目的のため、fine
        // barrier の有無に依存しない構成を使う。`gemm_swizzle_ab_bench.rs`
        // と同じ判断）。
        let default_gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let phase1_ok = phase1_stability_selfcheck(&ctx, &default_gemm);
        if !phase1_ok {
            return;
        }

        phase2_fine_barrier_ab(&ctx);
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`gemm_bench.rs` と同じ理由。`objc2` 系は
/// `cfg(target_os = "macos")` 限定のため本クレートの GEMM 実装自体が
/// コンパイル対象外になる。Linux CI の `cargo build --workspace --all-targets`
/// をこの example も含めて通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_fine_barrier_ab_bench example requires macOS (Apple Silicon). \
         See docs/perf/metal-bench-noise-protocol.md and \
         docs/perf/metal-gemm-fine-barrier-ab.md for the real-hardware execution procedure."
    );
}
