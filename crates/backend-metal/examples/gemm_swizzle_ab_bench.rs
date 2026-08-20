//! Metal ベンチ計測プロトコルのノイズ対策セルフチェック + tgid swizzle
//! （イシュー #540）A/B 再計測バイナリ（イシュー #746）。
//!
//! 親イシュー #737 配下: 2026-08-19 の M4 Max 実機計測で、tgid swizzle の
//! A/B と無関係な対照カーネル（naive/tiled/simdgroup）が計測実行間で
//! 最大 70% 超変動し（256/512 で顕著）、「劣化中央値 5% 以内」の
//! （#540 が定める）判定が成立しなかった。本バイナリはサーマル・GPU クロック
//! （DVFS）挙動の系統誤差を抑える計測プロトコル
//! （`bench_harness::ab`・`docs/perf/metal-bench-noise-protocol.md`）で
//! フェーズ 1（安定性セルフチェック）・フェーズ 2（swizzle A/B）を行う。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs`
//! ドキュメンテーションコメント（同ディレクトリ）と同一。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_swizzle_ab_bench --release
//! ```
//!
//! 実行前後に `pmset -g therm` でサーマル状態を記録すること
//! （`docs/perf/metal-gemm-tgid-swizzle-ab.md` 実行手順参照。`sudo` 不要の
//! 非特権コマンドのみを使う設計。`powermetrics` は `sudo` 必須のため不採用）。
//! フェーズ 1 でいずれかのサイズが安定性ゲート（spread ≤5% 程度）を超過した
//! 場合、フェーズ 2 の A/B 判定には進まない（「判定不可」を出力して終了する。
//! 安全側判断: 判定を無効化して中断する方向のみ許す）。

#[cfg(target_os = "macos")]
mod macos_impl {
    use backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{AbConfig, run_ab, run_stability};
    use bench_harness::rng::Xorshift64Star;
    use std::time::Duration;

    /// `crates/backend-metal/examples/gemm_bench.rs` と同一値
    /// （決定的シード。過去 PoC・CPU 実装ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// フェーズ 1・フェーズ 2 共通のラウンド数・cooldown・時間ベース
    /// ウォームアップ下限（イシュー #746 実装計画 §4.2 の初期値）。実機で
    /// 安定性ゲートを満たせない場合は手順書（`docs/perf/
    /// metal-bench-noise-protocol.md`）の調整手順に従い増やす方向のみ許す
    /// （減らす調整は spread 実測 green が条件。実装計画 §4.2）。
    const ROUNDS: usize = 6;
    const COOLDOWN: Duration = Duration::from_secs(2);
    const MIN_WARMUP: Duration = Duration::from_secs(1);

    /// フェーズ 1（安定性セルフチェック）対象の対照カーネル
    /// （tgid swizzle と無関係。#746 イシュー本文がばらつきを観測した対象）。
    fn tflops(size: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (size as f64).powi(3);
        flops / median_secs / 1e12
    }

    /// フェーズ 1: `gemm.dispatch_variant(GemmVariant::SimdgroupTiled(tile::select(..)))`
    /// （`dispatch_auto` の本番既定経路と同一構成選択）を `size` ごとに
    /// [`run_stability`] で計測し、spread を出力する。
    ///
    /// 戻り値は「全サイズで spread ≤5% 相当を満たしたか」（フェーズ 2 へ
    /// 進めるかどうかの判定材料。閾値そのものは呼び出し元 `main` が
    /// 出力メッセージとして明示するのみで、本関数はブール判定のみ返す）。
    fn phase1_stability_selfcheck(ctx: &MetalContext, gemm: &MetalGemm) -> bool {
        // 安定性ゲートの値自体は `bench_harness::ab::STABILITY_SPREAD_GATE` を
        // 単一真実源とする（`docs/perf/metal-bench-noise-protocol.md` と同じ値を
        // example 内に直接複製すると、閾値変更時にコードと文書が独立に乖離しうる
        // ため。codex-review 指摘対応。イシュー #746 PR #763）。
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

    /// フェーズ 2: `MetalGemm::new_with_swizzle` で base（`swizzle_enabled=false`）/
    /// head（`swizzle_enabled=true`）の 2 インスタンスを同一プロセス内に
    /// 構築し、[`run_ab`] で interleaved 計測する。`tile::select` が選ぶ構成を
    /// `dispatch_tiled_prepared` へ明示指定して使う（`#744` 是正後の
    /// `tile::select` 結果。バッファは base/head で共有し、swizzle の影響が
    /// パイプライン・grid 計算のみに閉じることを利用してアップロード
    /// コストを計測対象から除く。`gemm_bench.rs::measure_tiled_prepared`
    /// と同じ計測範囲の判断）。
    fn phase2_swizzle_ab(ctx: &MetalContext) {
        println!("--- フェーズ 2: tgid swizzle A/B（base=off / head=on）---");

        let base_gemm = MetalGemm::new_with_swizzle(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm =
            MetalGemm::new_with_swizzle(ctx, true).expect("head GEMM パイプラインの構築に失敗した");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        for size in [256usize, 512, 1024, 2048, 4096] {
            let cfg = tile::select(size, size, size);

            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let a_buf = MetalBuffer::new_with_data(ctx, &a)
                .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let b_buf = MetalBuffer::new_with_data(ctx, &b)
                .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let c_buf = MetalBuffer::new_zeroed(ctx, size * size)
                .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");

            // resolved_cfg は base/head 双方とも同じ `cfg`・デバイス限界に
            // 依存するため計測前に 1 回ずつ確定させ、フォールバックの
            // 有無を確認する（`gemm_bench.rs::measure_tiled_prepared` と
            // 同じ狙い）。
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
            // `result.b_over_a_ratio`（= median_b_secs / median_a_secs）は実行時間の比
            // （レイテンシ比）であり、TFLOPS は時間の逆数のため TFLOPS 比の
            // head/base はその逆数（median_a_secs / median_b_secs）になる。
            // ここを取り違えて `b_over_a_ratio` をそのまま `head_over_base` として
            // 出力すると、#540 の採否判定基準（`head_over_base` > 1.0 で採用）が
            // 逆転する（head が実際に高速でも 1.0 未満になる）ため、
            // TFLOPS 値同士の比として明示的に計算する（codex-review・Cursor Bugbot
            // 指摘対応。イシュー #746 PR #763）。
            let head_over_base = head_median_tflops / base_median_tflops;

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
            "--- フェーズ 2 完了。採否判定は #540 既存基準（2048/4096 の中央値改善で採用、\
             なければ revert）に従い、`docs/perf/metal-gemm-tgid-swizzle-ab.md` へ記録すること。"
        );
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        // フェーズ 1 は本番既定（swizzle off）の `MetalGemm::new` で行う
        // （対照カーネルの計測プロトコル自体を検証する目的のため、
        // swizzle の有無に依存しない構成を使う）。
        let default_gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let phase1_ok = phase1_stability_selfcheck(&ctx, &default_gemm);
        if !phase1_ok {
            return;
        }

        phase2_swizzle_ab(&ctx);
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
        "backend-metal gemm_swizzle_ab_bench example requires macOS (Apple Silicon). \
         See docs/perf/metal-bench-noise-protocol.md and \
         docs/perf/metal-gemm-tgid-swizzle-ab.md for the real-hardware execution procedure."
    );
}
