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

/// フェーズ 2（`macos_impl::phase2_fine_barrier_ab`）の 1 run 分の集計から
/// 単一 run の参考 verdict を計算する純関数（`docs/perf/
/// metal-gemm-fine-barrier-ab.md` §判断基準をそのまま定数化したもので、
/// 新規閾値の導入ではない）。macOS 依存部分（Metal コンテキスト・計測）を
/// 一切持たないため `#[cfg(target_os = "macos")]` の外に置き、Linux CI
/// （`cargo test --workspace`）でもユニットテストとして検査対象にする
/// （`gemm_transpose_route_ab_bench.rs::phase2_route_ab` の総括ロジックを
/// 参考にした判断だが、あちらは verdict 計算がフェーズ 2 内部に閉じている
/// のに対し、本 example は #1278 AC-3 でクロスプラットフォーム検証可能な
/// 純関数へ切り出した点が差分）。
///
/// 最終的な採否判断は本関数の 1 run 出力ではなく、5 run 中央値に基づき
/// `docs/perf/metal-gemm-fine-barrier-ab.md` で人間が確定する（実装計画
/// §3.3）。本関数の `verdict=` ログ出力はその中間証跡（機械的
/// `grep verdict=` で単一 run の判定を追える参考表示）にとどまる。
// Linux CI（`cargo clippy --workspace --all-targets --all-features -- -D
// warnings`）は本 example の `example` ターゲット（`cfg(test)` 無効・
// `target_os = "macos"` 無効）を単独でも lint するため、macOS 実機専用の
// `main`（`macos_impl::main`）からしか呼ばれない本型は非 macOS かつ非
// テストの経路では文字通り到達不能になり dead_code 検知の対象になる。
// 本体は macOS 実機（`macos_impl::main`）と `single_run_verdict_tests`
// （Linux CI でも走る `#[cfg(test)]` ユニットテスト）の双方から使われて
// おり未使用ではなく cfg 分岐の組み合わせが原因の誤検知にあたるため、
// その 2 経路以外（非 macOS・非テストの `example` ターゲット単体）でのみ
// `dead_code` を抑止する（codex-review 指摘 PR #1372）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SingleRunVerdict {
    AdoptCandidate,
    RejectCandidate,
    Undetermined,
}

// enum 本体（上）とは異なり、`as_str` は `single_run_verdict_tests`
// （Linux CI の `#[cfg(test)]` ビルド）からも呼ばれない
// （テストは `SingleRunVerdict` の等値比較のみを検証し、`as_str` の文字列
// 表現までは検証しない）。したがって非 macOS では test cfg の有無に
// 関わらず常に未使用となるため、`test` を除外条件に含めない
// （codex-review 指摘 PR #1372 のフォローアップ: `not(any(.., test))` の
// ままだと Linux の `cargo test`〈`cfg(test)` 有効〉ビルドで dead_code
// が再発する）。
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl SingleRunVerdict {
    fn as_str(self) -> &'static str {
        match self {
            SingleRunVerdict::AdoptCandidate => "adopt_candidate",
            SingleRunVerdict::RejectCandidate => "reject_candidate",
            SingleRunVerdict::Undetermined => "undetermined",
        }
    }
}

/// 採否判断における小サイズ（256/512/1024）の許容劣化率下限
/// （`docs/perf/metal-gemm-fine-barrier-ab.md` §判断基準の「劣化中央値
/// 5% 超がない」を定数化したもの。`head_over_base` 比がこの値以上であれば
/// 劣化許容範囲内と判定する）。
///
/// `bench_harness::ab::STABILITY_SPREAD_GATE`（ラウンド間ばらつきの安定性
/// spread ゲート）とは意味が異なる別軸の閾値のため流用しない
/// （codex-review 指摘 PR #1372 discussion r3943195886）。
// enum `SingleRunVerdict` と同じ理由（非 macOS かつ非テストの `example`
// ターゲット単体でのみ dead_code を抑止。codex-review 指摘 PR #1372）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const SMALL_SIZE_REGRESSION_TOLERANCE_RATIO: f64 = 0.95;

/// `ratios`: size ごとの `head_over_base`（TFLOPS 比。head/base）。
/// `gate_exceeded`: フェーズ 2 の A/B 計測自体で安定性ゲート超過が 1 件
/// でも残ったか（ラウンド間ばらつきが大きく計測値を信頼できないため
/// `Undetermined` へ倒す安全側判断）。
///
/// 判断基準（`docs/perf/metal-gemm-fine-barrier-ab.md` §判断基準）:
/// size 2048/4096 の `head_over_base` に改善（>1.0）があり、かつ size
/// 256/512/1024 で劣化中央値が `SMALL_SIZE_REGRESSION_TOLERANCE_RATIO`
/// 超（5% 超）がない場合に採用候補とする。
// enum `SingleRunVerdict` と同じ理由（非 macOS かつ非テストの `example`
// ターゲット単体でのみ dead_code を抑止。codex-review 指摘 PR #1372）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn single_run_verdict(ratios: &[(usize, f64)], gate_exceeded: bool) -> SingleRunVerdict {
    if gate_exceeded {
        return SingleRunVerdict::Undetermined;
    }
    let improved_at_large = ratios
        .iter()
        .filter(|(size, _)| *size == 2048 || *size == 4096)
        .all(|(_, ratio)| *ratio > 1.0);
    let no_regression_at_small = ratios
        .iter()
        .filter(|(size, _)| *size == 256 || *size == 512 || *size == 1024)
        .all(|(_, ratio)| *ratio >= SMALL_SIZE_REGRESSION_TOLERANCE_RATIO);
    if improved_at_large && no_regression_at_small {
        SingleRunVerdict::AdoptCandidate
    } else {
        SingleRunVerdict::RejectCandidate
    }
}

/// [`round_extrema`] の戻り値。`phase1_round_stats` 行の `min_secs`／
/// `max_secs` とその 0 始まり index を保持する
/// （`gemm_transpose_route_ab_bench.rs::RoundExtrema` と同一契約。
/// イシュー #1484）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq)]
struct RoundExtrema {
    min_secs: f64,
    min_round_idx: usize,
    max_secs: f64,
    max_round_idx: usize,
}

/// `round_medians_secs`（[`bench_harness::ab::StabilityResult::
/// round_medians_secs`]。秒単位）から最小・最大ラウンドを求める純関数。
///
/// `gemm_transpose_route_ab_bench.rs::round_extrema`（イシュー #1251）と
/// 同一実装・同一契約（**秒基準**判定・同値タイは最初の出現・空スライス
/// は `None`）。本ファイルは transpose example と実行バイナリが分かれる
/// ため複製している（`bench-harness` クレートへの共通化は #1484 のスコープ
/// 外。イシュー本文参照）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn round_extrema(round_medians_secs: &[f64]) -> Option<RoundExtrema> {
    if round_medians_secs.is_empty() {
        return None;
    }
    let (mut min_idx, mut max_idx) = (0usize, 0usize);
    for (i, &v) in round_medians_secs.iter().enumerate().skip(1) {
        if v < round_medians_secs[min_idx] {
            min_idx = i;
        }
        if v > round_medians_secs[max_idx] {
            max_idx = i;
        }
    }
    Some(RoundExtrema {
        min_secs: round_medians_secs[min_idx],
        min_round_idx: min_idx,
        max_secs: round_medians_secs[max_idx],
        max_round_idx: max_idx,
    })
}

/// [`bench_harness::ab::StabilityResult::aux`]（[`bench_harness::ab::
/// AuxiliarySpread`]。イシュー #1483）を `phase1_round_stats` 行の末尾
/// キー群（`trimmed_spread_k1=`／`iqr_spread=`／`mad_spread=`）へ整形する
/// 純関数。`gemm_transpose_route_ab_bench.rs::format_aux_spread_keys` と
/// 同一契約（キー → フィールド対応・`NA` sentinel・判定への転用禁止。
/// イシュー #1484）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_aux_spread_keys(aux: &bench_harness::ab::AuxiliarySpread) -> String {
    let trimmed_str = match aux.trimmed {
        Some(v) => format!("{v:.4e}"),
        None => "NA".to_string(),
    };
    format!(
        "trimmed_spread_k1={trimmed_str} iqr_spread={:.4e} mad_spread={:.4e}",
        aux.iqr_over_median, aux.mad2_over_median
    )
}

/// `phase1_round_stats` 行（機械可読な 1 行）を組み立てる純関数
/// （イシュー #1484。新規追加。`gemm_transpose_route_ab_bench.rs` が
/// #1249/#1251 で確立した既存キー集合・書式をそのまま踏襲し、末尾に
/// [`format_aux_spread_keys`] の 3 キーを追記する）。
///
/// 呼び出し元（`phase1_stability_selfcheck`）の既存人間可読行
/// （`size=… spread=… (…) round_tflops=…`）は本関数の追加によって
/// 変更しない（本関数はその**直後**に新規出力される 1 行を生成するのみ）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_phase1_round_stats_line(
    size: usize,
    result: &bench_harness::ab::StabilityResult,
    gate: f64,
    within_gate: bool,
    extrema: &RoundExtrema,
) -> String {
    let median_secs = bench_harness::median_q1_q3(&result.round_medians_secs)
        .expect("run_stability が返す round_medians_secs は非空・非 NaN のため成功する")
        .median;
    let round_medians_secs_str = result
        .round_medians_secs
        .iter()
        .map(|s| format!("{s:.6e}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "phase1_round_stats size={size} rounds={} spread={:.4e} gate={gate:.4e} \
         within_gate={within_gate} median_secs={median_secs:.6e} \
         min_secs={:.6e} min_round_idx={} max_secs={:.6e} max_round_idx={} \
         round_medians_secs={round_medians_secs_str} {}",
        result.round_medians_secs.len(),
        result.spread,
        extrema.min_secs,
        extrema.min_round_idx,
        extrema.max_secs,
        extrema.max_round_idx,
        format_aux_spread_keys(&result.aux),
    )
}

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

        // イシュー #1278 AC-1: 512〜4096 まで拡張する（旧版は 256/512/1024
        // のみだったが、受け入れ条件は判断基準の対象サイズ（256〜4096）
        // 全域での bit 一致を要求する。256 は既存の下限確認として残す）。
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

            // イシュー #1484: 上記の既存人間可読行はバイト単位で変更せず、
            // 機械可読な `phase1_round_stats` 行を直後に追加出力する
            // （`gemm_transpose_route_ab_bench.rs` と同一キー集合。
            // `super::round_extrema`／`super::format_phase1_round_stats_line`
            // 契約参照）。
            if let Some(extrema) = super::round_extrema(&result.round_medians_secs) {
                println!(
                    "{}",
                    super::format_phase1_round_stats_line(
                        size,
                        &result,
                        SPREAD_GATE,
                        within_gate,
                        &extrema
                    )
                );
            }
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

        // フェーズ 2 総括（`verdict=`）の判定材料。`ratios` は size ごとの
        // `head_over_base`（TFLOPS 比）、`gate_exceeded` はいずれかの size で
        // `run_ab` 自体の spread が安定性ゲートを超えたか（超えた場合は
        // ラウンド間ばらつきが大きく計測値を信頼できないため
        // `Undetermined` に倒す。`gemm_transpose_route_ab_bench.rs::
        // phase2_route_ab` と同じ安全側判断）。
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

        // イシュー #1278 AC-3: 単一 run の参考 verdict を機械出力する
        // （`super::single_run_verdict` は macOS 非依存の純関数。最終採否は
        // 5 run 中央値に基づき `docs/perf/metal-gemm-fine-barrier-ab.md` へ
        // 人間可読な形で記録する——本行はログを `grep verdict=` するだけで
        // 単一 run の判定を追える参考表示にとどめる）。
        let verdict = super::single_run_verdict(&ratios, gate_exceeded);
        println!(
            "--- フェーズ 2 完了。採否判定基準（size 2048/4096 の中央値改善があり、かつ他\
             サイズで劣化中央値 5% 超がないこと。イシュー #809 計画 §3.1・#1278 で 5 run\
             中央値運用へ拡張）に従い、`docs/perf/metal-gemm-fine-barrier-ab.md` へ記録\
             すること。"
        );
        println!(
            "verdict={} (単一 run の参考値。5 run 中央値で最終判断する)",
            verdict.as_str()
        );
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

        // イシュー #1278: 一時調整した ROUNDS/COOLDOWN/MIN_WARMUP をコミット
        // しない運用（`docs/perf/metal-bench-noise-protocol.md` の調整手順）
        // のため、実効値を実行ログの先頭に残す（env_info と併せて実測記録の
        // 証跡にする）。
        println!(
            "effective_rounds={ROUNDS} effective_cooldown_secs={} effective_min_warmup_secs={}",
            COOLDOWN.as_secs_f64(),
            MIN_WARMUP.as_secs_f64(),
        );

        phase0_bit_match_selfcheck(&ctx);

        // フェーズ 1 は本番既定（fine barrier off）の `MetalGemm::new` で行う
        // （対照カーネルの計測プロトコル自体を検証する目的のため、fine
        // barrier の有無に依存しない構成を使う。`gemm_swizzle_ab_bench.rs`
        // と同じ判断）。
        let default_gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let phase1_ok = phase1_stability_selfcheck(&ctx, &default_gemm);
        if !phase1_ok {
            // codex-review 指摘対応（PR #1198・`gemm_transpose_route_ab_bench.rs`
            // と同型）: 早期 return もフェーズ 2 完了時と同じ `verdict=` 行を
            // 出力し、全終了経路でログを `grep verdict=` するだけで判定を
            // 一意に読み取れるようにする。
            println!(
                "verdict=undetermined (フェーズ 1 の安定性セルフチェックで \
                 spread ≤gate 相当を満たさないサイズが残ったため、フェーズ 2\
                 （A/B 判定）を実行せず判定不可のまま終了する)"
            );
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

#[cfg(test)]
mod single_run_verdict_tests {
    use super::*;

    #[test]
    fn adopts_when_large_sizes_improve_and_small_sizes_hold_within_5_percent() {
        let ratios = vec![
            (256, 0.96),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.05),
            (4096, 1.1),
        ];
        assert_eq!(
            single_run_verdict(&ratios, false),
            SingleRunVerdict::AdoptCandidate
        );
    }

    #[test]
    fn rejects_when_a_large_size_does_not_improve() {
        let ratios = vec![
            (256, 1.0),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.0),
            (4096, 1.1),
        ];
        assert_eq!(
            single_run_verdict(&ratios, false),
            SingleRunVerdict::RejectCandidate
        );
    }

    #[test]
    fn rejects_when_a_small_size_regresses_more_than_5_percent() {
        let ratios = vec![
            (256, 0.90),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.05),
            (4096, 1.1),
        ];
        assert_eq!(
            single_run_verdict(&ratios, false),
            SingleRunVerdict::RejectCandidate
        );
    }

    #[test]
    fn undetermined_when_stability_gate_exceeded_even_if_ratios_look_good() {
        let ratios = vec![
            (256, 1.0),
            (512, 1.0),
            (1024, 1.0),
            (2048, 1.2),
            (4096, 1.3),
        ];
        assert_eq!(
            single_run_verdict(&ratios, true),
            SingleRunVerdict::Undetermined
        );
    }
}

/// `round_extrema`／`format_aux_spread_keys`／`format_phase1_round_stats_line`
/// （イシュー #1484）の純関数ユニットテスト。macOS 依存部分を持たないため
/// Linux CI（`cargo test -p fandhe-ai-backend-metal --example
/// gemm_fine_barrier_ab_bench`）でも走る。
#[cfg(test)]
mod phase1_round_stats_tests {
    use super::{format_aux_spread_keys, format_phase1_round_stats_line, round_extrema};
    use bench_harness::ab::{AUXILIARY_TRIM_PER_SIDE, AuxiliarySpread, StabilityResult};

    #[test]
    fn auxiliary_trim_per_side_is_one() {
        assert_eq!(AUXILIARY_TRIM_PER_SIDE, 1);
    }

    #[test]
    fn round_extrema_empty_slice_is_none() {
        assert_eq!(round_extrema(&[]), None);
    }

    #[test]
    fn round_extrema_single_spike_finds_correct_index() {
        let samples = [1.0, 1.1, 0.9, 5.0, 1.05, 0.95];
        let extrema = round_extrema(&samples).expect("非空スライスは Some を返すはず");
        assert_eq!(extrema.max_secs, 5.0);
        assert_eq!(extrema.max_round_idx, 3);
        assert_eq!(extrema.min_secs, 0.9);
        assert_eq!(extrema.min_round_idx, 2);
    }

    #[test]
    fn format_aux_spread_keys_trimmed_none_renders_na_sentinel() {
        let aux = AuxiliarySpread {
            trimmed: None,
            iqr_over_median: 1.0e-1,
            mad2_over_median: 2.0e-1,
        };
        let keys = format_aux_spread_keys(&aux);
        assert_eq!(
            keys,
            "trimmed_spread_k1=NA iqr_spread=1.0000e-1 mad_spread=2.0000e-1"
        );
    }

    /// 既存キーの並び・書式が `gemm_transpose_route_ab_bench.rs` と同一で、
    /// 末尾に補助 3 キーがスペース区切りで追記されることを検証する
    /// （イシュー #1484）。
    #[test]
    fn format_phase1_round_stats_line_has_expected_keys_and_order() {
        let round_medians_secs = vec![1.0e-4, 1.1e-4, 0.9e-4, 1.05e-4];
        let result = StabilityResult {
            round_medians_secs: round_medians_secs.clone(),
            spread: 2.0e-1,
            aux: AuxiliarySpread {
                trimmed: Some(1.2345e-1),
                iqr_over_median: 2.3456e-1,
                mad2_over_median: 3.4567e-1,
            },
        };
        let extrema = round_extrema(&round_medians_secs).expect("非空スライスは Some を返すはず");
        let line = format_phase1_round_stats_line(512, &result, 5.0000e-2, false, &extrema);
        assert!(line.starts_with("phase1_round_stats size=512 rounds=4 spread=2.0000e-1 "));
        assert!(line.contains("gate=5.0000e-2 within_gate=false"));
        assert!(
            line.contains("round_medians_secs=1.000000e-4,1.100000e-4,9.000000e-5,1.050000e-4")
        );
        assert!(
            line.ends_with("trimmed_spread_k1=1.2345e-1 iqr_spread=2.3456e-1 mad_spread=3.4567e-1")
        );
    }
}
