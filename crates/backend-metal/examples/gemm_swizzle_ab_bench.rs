//! Metal ベンチ計測プロトコルのノイズ対策セルフチェック + tgid swizzle
//! （イシュー #540）A/B 再計測バイナリ（イシュー #746）。
//!
//! 親イシュー #737 配下: 2026-08-19 の M4 Max 実機計測で、tgid swizzle の
//! A/B と無関係な対照カーネル（naive/tiled/simdgroup）が計測実行間で
//! 最大 70% 超変動し（256/512 で顕著）、「劣化中央値 5% 以内」の
//! （#540 が定める）判定が成立しなかった。本バイナリはサーマル・GPU クロック
//! （DVFS）挙動の系統誤差を抑える計測プロトコル
//! （`bench_harness::ab`・`docs/perf/metal-bench-noise-protocol.md`）で
//! フェーズ 0（bit 一致の自己検証。イシュー #1279）・フェーズ 1（安定性
//! セルフチェック）・フェーズ 2（prepared 境界 swizzle A/B）・フェーズ 3
//! （転送込み境界 swizzle A/B。イシュー #795）を行う。
//!
//! フェーズ 0 を追加している理由: threadgroup ID スウィズルは各 C タイルを
//! 担当する threadgroup の tgid→タイル座標写像のみを変え、その threadgroup
//! が実行する演算（K ループ順・FMA 契約・simdgroup 分担）自体は変わらない。
//! また各出力要素は正確に 1 threadgroup が 1 回書くため、base（swizzle
//! off）/head（on）の出力は理論上ビット単位で一致するはずである
//! （`gemm_fine_barrier_ab_bench.rs` と同じ論法。#809・#1278 の先例に倣い、
//! この数値契約を計測前に自己検証する。崩れている場合は A/B 計測（性能
//! 比較）へ進まない安全側判断）。
//!
//! フェーズ 3 はフェーズ 2 の `dispatch_tiled_prepared`（アップロード済み
//! バッファ・確定 `TileConfig` を直接渡す prepared 境界）と異なり、
//! `dispatch_auto`（ホストスライス入力・アップロード + GEMM + 読み戻しを
//! 1 計測区間に含む本番相当の呼び出し経路）を使う。計測境界の定義は
//! `docs/perf/oss-gemm-comparison-baseline.md` §計測境界と同じ「転送込み」
//! 区分に合わせている（本番 `MetalGemm::dispatch_auto` 呼び出しがまさに
//! この境界で計測されるため、prepared 境界だけでは見えない転送コストの
//! 影響を含めた採否判断材料として使う。イシュー #795 計画 Step 1）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs`
//! ドキュメンテーションコメント（同ディレクトリ）と同一。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo build --release -p fandhe-ai-backend-metal --example gemm_swizzle_ab_bench
//! ./target/release/examples/gemm_swizzle_ab_bench
//! ```
//!
//! 実行前後に `pmset -g therm` でサーマル状態を記録すること
//! （`docs/perf/metal-gemm-tgid-swizzle-ab.md` 実行手順参照。`sudo` 不要の
//! 非特権コマンドのみを使う設計。`powermetrics` は `sudo` 必須のため不採用）。
//! フェーズ 1 でいずれかのサイズが安定性ゲート（spread ≤5% 程度）を超過した
//! 場合、フェーズ 2 の A/B 判定には進まない（「判定不可」を出力して終了する。
//! 安全側判断: 判定を無効化して中断する方向のみ許す）。

/// フェーズ 2（prepared 境界）・フェーズ 3（転送込み境界）双方の 1 run 分
/// 集計から単一 run の参考 verdict を計算する純関数（`docs/perf/
/// metal-gemm-tgid-swizzle-ab.md` §判断基準〈#795 改定版〉をそのまま
/// 定数化したもので、新規閾値の導入ではない）。`gemm_fine_barrier_ab_bench.
/// rs::single_run_verdict`（フェーズ 2 単独入力）と異なり、本関数は
/// フェーズ 2・フェーズ 3 **双方**の比列を受け取る（#795 判断基準が
/// 「prepared 境界・転送込み境界の両方で size 2048/4096 の改善」を採用の
/// 必要条件とするため）。macOS 依存部分（Metal コンテキスト・計測）を
/// 一切持たないため `#[cfg(target_os = "macos")]` の外に置き、Linux CI
/// （`cargo test --workspace`）でもユニットテストとして検査対象にする。
///
/// 最終的な採否判断は本関数の 1 run 出力ではなく、5 run 中央値に基づき
/// `docs/perf/metal-gemm-tgid-swizzle-ab.md` で人間が確定する
/// （`gemm_fine_barrier_ab_bench.rs` と同じ位置づけ）。本関数の `verdict=`
/// ログ出力はその中間証跡（機械的 `grep verdict=` で単一 run の判定を
/// 追える参考表示）にとどまる。
// Linux CI（`cargo clippy --workspace --all-targets --all-features -- -D
// warnings`）は本 example の `example` ターゲット（`cfg(test)` 無効・
// `target_os = "macos"` 無効）を単独でも lint するため、macOS 実機専用の
// `main`（`macos_impl::main`）からしか呼ばれない本型は非 macOS かつ非
// テストの経路では文字通り到達不能になり dead_code 検知の対象になる。
// 本体は macOS 実機（`macos_impl::main`）と `single_run_verdict_tests`
// （Linux CI でも走る `#[cfg(test)]` ユニットテスト）の双方から使われて
// おり未使用ではなく cfg 分岐の組み合わせが原因の誤検知にあたるため、
// その 2 経路以外（非 macOS・非テストの `example` ターゲット単体）でのみ
// `dead_code` を抑止する（`gemm_fine_barrier_ab_bench.rs` と同型。
// codex-review 指摘 PR #1372）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SingleRunVerdict {
    AdoptCandidate,
    SizeConditionalCandidate,
    RejectCandidate,
    Undetermined,
}

// enum 本体（上）とは異なり、`as_str` は `single_run_verdict_tests`
// （Linux CI の `#[cfg(test)]` ビルド）からも呼ばれない（テストは
// `SingleRunVerdict` の等値比較のみを検証し、`as_str` の文字列表現までは
// 検証しない）。したがって非 macOS では test cfg の有無に関わらず常に
// 未使用となるため、`test` を除外条件に含めない
// （`gemm_fine_barrier_ab_bench.rs` と同型。codex-review 指摘 PR #1372
// のフォローアップ）。
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl SingleRunVerdict {
    fn as_str(self) -> &'static str {
        match self {
            SingleRunVerdict::AdoptCandidate => "adopt_candidate",
            SingleRunVerdict::SizeConditionalCandidate => "size_conditional_candidate",
            SingleRunVerdict::RejectCandidate => "reject_candidate",
            SingleRunVerdict::Undetermined => "undetermined",
        }
    }
}

/// 採否判断における小〜中サイズ（256/512/1024。フェーズ 3 は 512/1024 の
/// み対象）の許容劣化率下限（`docs/perf/metal-gemm-tgid-swizzle-ab.md`
/// §判断基準の「小〜中形状で有意な劣化（spread 相当を超える悪化）が
/// ない」を、`bench_harness::ab::STABILITY_SPREAD_GATE`（0.05）から導出した
/// 値として定数化したもの。新規閾値の導入ではない。
/// `gemm_fine_barrier_ab_bench.rs::SMALL_SIZE_REGRESSION_TOLERANCE_RATIO`
/// と同値・同じ導出根拠）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const SMALL_SIZE_REGRESSION_TOLERANCE_RATIO: f64 = 0.95;

/// `phase2_ratios`・`phase3_ratios`: それぞれ size ごとの `head_over_base`
/// （TFLOPS 比。head/base）。フェーズ 3 は 512/1024/2048/4096（256 を含まない。
/// `phase3_swizzle_ab_transfer_included` の対象範囲）。`gate_exceeded` は
/// フェーズ 2・フェーズ 3 いずれかの A/B 計測自体で安定性ゲート超過が
/// 1 件でも残ったか（ラウンド間ばらつきが大きく計測値を信頼できないため
/// `Undetermined` へ倒す安全側判断）。
///
/// 判断基準（`docs/perf/metal-gemm-tgid-swizzle-ab.md` §判断基準〈#795
/// 改定版〉）:
/// - `large_ok`: フェーズ 2・フェーズ 3 **双方**で size 2048/4096 の
///   `head_over_base` が改善（>1.0）している
/// - `small_ok`: フェーズ 2 の size 256/512/1024・フェーズ 3 の size
///   512/1024 が `SMALL_SIZE_REGRESSION_TOLERANCE_RATIO` 以上（劣化許容
///   範囲内）
/// - `large_ok && small_ok` → 採用候補（`AdoptCandidate`）
/// - `large_ok && !small_ok` → サイズ条件付き採用候補
///   （`SizeConditionalCandidate`。大形状のみ改善し小形状で有意な劣化が
///   ある場合。`tile.rs` へ `should_apply_swizzle` を追加する分岐）
/// - それ以外 → 不採用候補（`RejectCandidate`）
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn single_run_verdict(
    phase2_ratios: &[(usize, f64)],
    phase3_ratios: &[(usize, f64)],
    gate_exceeded: bool,
) -> SingleRunVerdict {
    if gate_exceeded {
        return SingleRunVerdict::Undetermined;
    }

    let large_improved = |ratios: &[(usize, f64)]| {
        ratios
            .iter()
            .filter(|(size, _)| *size == 2048 || *size == 4096)
            .all(|(_, ratio)| *ratio > 1.0)
    };
    let large_ok = large_improved(phase2_ratios) && large_improved(phase3_ratios);

    let small_within_tolerance = |ratios: &[(usize, f64)], sizes: &[usize]| {
        ratios
            .iter()
            .filter(|(size, _)| sizes.contains(size))
            .all(|(_, ratio)| *ratio >= SMALL_SIZE_REGRESSION_TOLERANCE_RATIO)
    };
    let small_ok = small_within_tolerance(phase2_ratios, &[256, 512, 1024])
        && small_within_tolerance(phase3_ratios, &[512, 1024]);

    if large_ok && small_ok {
        SingleRunVerdict::AdoptCandidate
    } else if large_ok {
        SingleRunVerdict::SizeConditionalCandidate
    } else {
        SingleRunVerdict::RejectCandidate
    }
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{AbConfig, run_ab, run_stability};
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};
    use std::time::Duration;

    /// `crates/backend-metal/examples/gemm_bench.rs` と同一値
    /// （決定的シード。過去 PoC・CPU 実装ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// フェーズ 1〜3 共通のラウンド数・cooldown・時間ベースウォームアップ
    /// 下限（イシュー #746 実装計画 §4.2 の初期値）。実機で安定性ゲートを
    /// 満たせない場合は手順書（`docs/perf/metal-bench-noise-protocol.md`）の
    /// 調整手順に従い増やす方向のみ許す（減らす調整は spread 実測 green が
    /// 条件。実装計画 §4.2）。一時調整した値はコミットしない運用
    /// （イシュー #1279 で `effective_*` println を追加し実効値を実行ログの
    /// 先頭へ残す。`gemm_fine_barrier_ab_bench.rs` と同じ運用）。
    const ROUNDS: usize = 6;
    const COOLDOWN: Duration = Duration::from_secs(2);
    const MIN_WARMUP: Duration = Duration::from_secs(1);

    fn tflops(size: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (size as f64).powi(3);
        flops / median_secs / 1e12
    }

    /// `AbResult` の base/head 中央値秒数から `head_over_base`（TFLOPS 比）を
    /// 計算する共有ヘルパ。フェーズ 2・フェーズ 3 で同一ロジックを使う
    /// （`result.b_over_a_ratio` は実行時間の比〈レイテンシ比〉であり
    /// TFLOPS 比はその逆数になる。取り違えると採否判定基準
    /// 〈`head_over_base` > 1.0 で改善〉が逆転するため、フェーズ間で
    /// 重複実装せず 1 箇所に集約する。イシュー #746 PR #763 の
    /// codex-review・Cursor Bugbot 指摘の再発防止。イシュー #795 計画 Step 1）。
    fn head_over_base_tflops_ratio(size: usize, median_a_secs: f64, median_b_secs: f64) -> f64 {
        tflops(size, median_b_secs) / tflops(size, median_a_secs)
    }

    /// フェーズ 0: base（`swizzle_enabled=false`）/head（`true`）の出力が
    /// ビット単位で一致することを計測前に自己検証する（本ファイル冒頭
    /// ドキュメンテーションコメント参照。イシュー #1279）。不一致の場合は
    /// `panic` し、フェーズ 1〜3 へは進まない（安全側判断）。
    fn phase0_bit_match_selfcheck(ctx: &MetalContext) {
        println!("--- フェーズ 0: base/head 出力 bit 一致の自己検証 ---");

        let base_gemm = MetalGemm::new_with_swizzle(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm =
            MetalGemm::new_with_swizzle(ctx, true).expect("head GEMM パイプラインの構築に失敗した");

        for size in [256usize, 512, 1024, 2048, 4096] {
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let base_out = base_gemm
                .dispatch_auto(ctx, &a, &b, size, size, size)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(ctx, &a, &b, size, size, size)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            // `assert_eq!` の `f32` 数値比較は IEEE 754 の `+0.0 == -0.0` を
            // 区別できず符号ビットの差異を見逃しうるため、`to_bits()` 経由の
            // ビットパターン比較で厳密に検証する
            // （`tests/gemm_swizzle_bit_match.rs::assert_bit_exact` と同じ理由。
            // `gemm_fine_barrier_bit_match.rs` codex-review P2 指摘
            // discussion r3943195893 の教訓を踏襲）。
            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
            let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                base_bits, head_bits,
                "size={size}: SWIZZLE_ENABLED の有無で出力がビット単位で一致しなかった。\
                 tgid→タイル座標写像以外の演算オペランド列が変わっている疑いがあるため、\
                 A/B 計測（性能比較）は無意味になる。shaders/gemm.metal の SWIZZLE_ENABLED \
                 挿入箇所を確認すること。"
            );
            println!("size={size} bit-exact match: OK");
        }
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
    ///
    /// 戻り値は size ごとの `head_over_base`（TFLOPS 比）と、いずれかの
    /// size で spread が安定性ゲートを超えたか（`verdict=` 判定材料。
    /// イシュー #1279 で `single_run_verdict` を消費する呼び出し元へ
    /// 集計結果を返すよう拡張）。
    fn phase2_swizzle_ab(ctx: &MetalContext) -> (Vec<(usize, f64)>, bool) {
        println!("--- フェーズ 2: tgid swizzle A/B（prepared 境界。base=off / head=on）---");

        let base_gemm = MetalGemm::new_with_swizzle(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm =
            MetalGemm::new_with_swizzle(ctx, true).expect("head GEMM パイプラインの構築に失敗した");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        let mut ratios: Vec<(usize, f64)> = Vec::new();
        let mut gate_exceeded = false;

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

            if result.spread_a > SPREAD_GATE || result.spread_b > SPREAD_GATE {
                gate_exceeded = true;
            }
            ratios.push((size, head_over_base));
        }

        println!(
            "--- フェーズ 2 完了。採否判定は #795 判断基準（prepared・転送込み両境界で\
             2048/4096 の中央値改善が必要）に従い、`docs/perf/metal-gemm-tgid-swizzle-ab.md`\
             へ記録すること。"
        );
        (ratios, gate_exceeded)
    }

    /// フェーズ 3（イシュー #795）: `MetalGemm::new_with_swizzle` で構築した
    /// base/head の 2 インスタンスへ `dispatch_auto`（ホストスライス入力。
    /// アップロード + GEMM + 読み戻しを 1 計測区間に含む転送込み境界）を
    /// [`run_ab`] で interleaved 計測する。フェーズ 2（prepared 境界。
    /// アップロード済みバッファを使い回す）では見えない、swizzle 適用に伴う
    /// grid 形状変化がアップロード/読み戻しコストと相互作用する効果
    /// （もしあれば）を捕捉するのが狙い（計画 Step 1）。
    ///
    /// サイズは 512〜4096（256 は #795 計画のフェーズ 3 対象範囲外。
    /// prepared 境界フェーズ 2 で既に全サイズ計測済みのため、フェーズ 3 は
    /// 転送込み境界での「大形状での効果」確認に絞る）。
    ///
    /// 戻り値はフェーズ 2 と同型（size ごとの `head_over_base`・gate 超過
    /// 有無。イシュー #1279 で `single_run_verdict` へ渡すため集計結果を
    /// 返すよう拡張）。
    fn phase3_swizzle_ab_transfer_included(ctx: &MetalContext) -> (Vec<(usize, f64)>, bool) {
        println!("--- フェーズ 3: tgid swizzle A/B（転送込み境界。base=off / head=on）---");

        let base_gemm = MetalGemm::new_with_swizzle(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm =
            MetalGemm::new_with_swizzle(ctx, true).expect("head GEMM パイプラインの構築に失敗した");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        let mut ratios: Vec<(usize, f64)> = Vec::new();
        let mut gate_exceeded = false;

        for size in [512usize, 1024, 2048, 4096] {
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let result = run_ab(
                &ab_config,
                &measurement_config,
                || {
                    base_gemm
                        .dispatch_auto(ctx, &a, &b, size, size, size)
                        .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
                },
                || {
                    head_gemm
                        .dispatch_auto(ctx, &a, &b, size, size, size)
                        .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
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
                "size={size} base_median_tflops={:.4} head_median_tflops={:.4} \
                 head_over_base={:.4} spread_base={:.4} spread_head={:.4} \
                 base_round_tflops={base_tflops:.4?} head_round_tflops={head_tflops:.4?}",
                base_median_tflops,
                head_median_tflops,
                head_over_base,
                result.spread_a,
                result.spread_b,
            );

            if result.spread_a > SPREAD_GATE || result.spread_b > SPREAD_GATE {
                gate_exceeded = true;
            }
            ratios.push((size, head_over_base));
        }

        println!(
            "--- フェーズ 3 完了。転送込み境界の実測もフェーズ 2 と合わせて \
             `docs/perf/metal-gemm-tgid-swizzle-ab.md` へ記録し、#795 判断基準の \
             根拠に含めること。"
        );
        (ratios, gate_exceeded)
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

        // イシュー #1279: 一時調整した ROUNDS/COOLDOWN/MIN_WARMUP をコミット
        // しない運用（`docs/perf/metal-bench-noise-protocol.md` の調整手順）
        // のため、実効値を実行ログの先頭に残す（env_info と併せて実測記録の
        // 証跡にする。`gemm_fine_barrier_ab_bench.rs` と同じ運用）。
        println!(
            "effective_rounds={ROUNDS} effective_cooldown_secs={} effective_min_warmup_secs={}",
            COOLDOWN.as_secs_f64(),
            MIN_WARMUP.as_secs_f64(),
        );

        phase0_bit_match_selfcheck(&ctx);

        // フェーズ 1 は本番既定（swizzle off）の `MetalGemm::new` で行う
        // （対照カーネルの計測プロトコル自体を検証する目的のため、
        // swizzle の有無に依存しない構成を使う）。
        let default_gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let phase1_ok = phase1_stability_selfcheck(&ctx, &default_gemm);
        if !phase1_ok {
            // イシュー #1279（`gemm_fine_barrier_ab_bench.rs`・
            // `gemm_transpose_route_ab_bench.rs` と同型）: 早期 return も
            // フェーズ 3 完了時と同じ `verdict=` 行を出力し、全終了経路で
            // ログを `grep verdict=` するだけで判定を一意に読み取れるように
            // する。
            println!(
                "verdict=undetermined (フェーズ 1 の安定性セルフチェックで \
                 spread ≤gate 相当を満たさないサイズが残ったため、フェーズ 2/3\
                 （A/B 判定）を実行せず判定不可のまま終了する)"
            );
            return;
        }

        let (phase2_ratios, phase2_gate_exceeded) = phase2_swizzle_ab(&ctx);
        let (phase3_ratios, phase3_gate_exceeded) = phase3_swizzle_ab_transfer_included(&ctx);

        // イシュー #1279 AC-3: 単一 run の参考 verdict を機械出力する
        // （`super::single_run_verdict` は macOS 非依存の純関数。最終採否は
        // 5 run 中央値に基づき `docs/perf/metal-gemm-tgid-swizzle-ab.md` へ
        // 人間可読な形で記録する——本行はログを `grep verdict=` するだけで
        // 単一 run の判定を追える参考表示にとどめる）。
        let gate_exceeded = phase2_gate_exceeded || phase3_gate_exceeded;
        let verdict = super::single_run_verdict(&phase2_ratios, &phase3_ratios, gate_exceeded);
        println!(
            "--- フェーズ 2/3 完了。採否判定基準（prepared・転送込み両境界で size 2048/4096\
             の中央値改善があり、かつ他サイズで劣化中央値 5% 超がないこと。#795 判断基準・\
             イシュー #1279 で 5 run 中央値運用へ拡張）に従い、`docs/perf/\
             metal-gemm-tgid-swizzle-ab.md` へ記録すること。"
        );
        println!(
            "verdict={} (単一 run の参考値。5 run 中央値で最終判断する)",
            verdict.as_str()
        );
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

#[cfg(test)]
mod single_run_verdict_tests {
    use super::*;

    #[test]
    fn adopts_when_large_sizes_improve_in_both_phases_and_small_sizes_hold() {
        let phase2 = vec![
            (256, 0.96),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.05),
            (4096, 1.1),
        ];
        let phase3 = vec![(512, 1.0), (1024, 1.0), (2048, 1.05), (4096, 1.1)];
        assert_eq!(
            single_run_verdict(&phase2, &phase3, false),
            SingleRunVerdict::AdoptCandidate
        );
    }

    #[test]
    fn size_conditional_when_large_sizes_improve_but_a_small_size_regresses() {
        let phase2 = vec![
            (256, 0.80),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.05),
            (4096, 1.1),
        ];
        let phase3 = vec![(512, 1.0), (1024, 1.0), (2048, 1.05), (4096, 1.1)];
        assert_eq!(
            single_run_verdict(&phase2, &phase3, false),
            SingleRunVerdict::SizeConditionalCandidate
        );
    }

    #[test]
    fn rejects_when_phase3_large_size_does_not_improve_even_if_phase2_does() {
        let phase2 = vec![
            (256, 1.0),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.05),
            (4096, 1.1),
        ];
        // フェーズ 3 の 4096 が改善していない（1.0 以下）ため、フェーズ 2
        // だけ改善していても #795 判断基準（両境界で改善必要）により不採用。
        let phase3 = vec![(512, 1.0), (1024, 1.0), (2048, 1.05), (4096, 1.0)];
        assert_eq!(
            single_run_verdict(&phase2, &phase3, false),
            SingleRunVerdict::RejectCandidate
        );
    }

    #[test]
    fn rejects_when_neither_phase_improves_large_sizes() {
        let phase2 = vec![
            (256, 1.0),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.0),
            (4096, 1.0),
        ];
        let phase3 = vec![(512, 1.0), (1024, 1.0), (2048, 1.0), (4096, 1.0)];
        assert_eq!(
            single_run_verdict(&phase2, &phase3, false),
            SingleRunVerdict::RejectCandidate
        );
    }

    #[test]
    fn undetermined_when_stability_gate_exceeded_even_if_ratios_look_good() {
        let phase2 = vec![
            (256, 1.0),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.2),
            (4096, 1.3),
        ];
        let phase3 = vec![(512, 1.0), (1024, 1.0), (2048, 1.2), (4096, 1.3)];
        assert_eq!(
            single_run_verdict(&phase2, &phase3, true),
            SingleRunVerdict::Undetermined
        );
    }
}
