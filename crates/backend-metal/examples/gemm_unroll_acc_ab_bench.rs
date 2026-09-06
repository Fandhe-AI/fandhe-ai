//! アキュムレータ系ループ条件付き loop unroll（イシュー #1282）の
//! **本番 `dispatch_auto` 経路**（実際の選択ロジック `tile::select_for_device`
//! を経由する構成）における A/B 再計測バイナリ（イシュー #1284）。
//!
//! `gemm_fine_barrier_ab_bench.rs`（#809/#1278）・`gemm_swizzle_ab_bench.rs`
//! （#540/#795/#1279）と同一プロトコル（`bench_harness::ab`・
//! `docs/perf/metal-bench-noise-protocol.md`）を踏襲する: フェーズ 0（bit
//! 一致の自己検証）→ フェーズ 1（安定性セルフチェック）→ フェーズ 2
//! （prepared 境界 A/B。本判定対象）→ フェーズ 3（`dispatch_auto` 転送込み
//! 境界 A/B。参考値）。
//!
//! ## 本 example が測る構造的な事実（実装計画 §1.2）
//!
//! `tile::select_with_occupancy_for_device`（`ctx.verified_m4_max_gpu_core_count()`
//! が `Some` の M4 Max 実測経路）は要求 4 形状（512/1024/2048/4096 の正方）
//! いずれも `acc_rows*acc_cols` 積が 8 の候補（`CANDIDATES[5]`／`[6]`／
//! `[1]`／`[2]`）を返す。`tile::UNROLL_ACC_MIN_PRODUCT`（16）未満のため、
//! 本番 `dispatch_auto` 経路はこれら 4 形状では **head（unroll 有効）でも
//! base と同一の非 unroll 版ループ**をコンパイルする
//! （`tile::unroll_acc_loops_for` の AND 条件。`docs/perf/
//! metal-gemm-n4096-kernel-gap.md` §7.9.3 参照）。したがって正方 4 形状の
//! A/B は「function constant 特殊化の有無による差 ≈ 1.0（ノイズ内）」を
//! 測る計測になり、`docs/perf/metal-gemm-n4096-kernel-gap.md` §7.5 で確認
//! した N=4096 +11.5%（`CANDIDATES[2]` を**無条件** unroll した実験値）は
//! 条件付き gating の下では設計上再現されない。
//!
//! 結線の実効が及ぶのは `tile::select_with_occupancy_for_device` の
//! `_ if m >= LARGE && n >= LARGE => CANDIDATES[0]` 分岐（`acc_rows*acc_cols
//! = 16`。非正方・非 tall/wide の大形状フォールバック）のみであるため、
//! 本 example は正方 4 形状に加え、この分岐へ到達する補助形状
//! （`(2048, 2048, 512)`・`(4096, 4096, 1024)`）も計測する
//! （`supplementary=true` として出力・判定に含める）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_unroll_acc_ab_bench --release
//! ```
//!
//! 実行前後に `pmset -g therm` でサーマル状態を、`uptime` で他プロセスの
//! GPU 負荷を記録すること（`docs/perf/metal-bench-noise-protocol.md`・
//! 実装計画 Phase 2 参照）。フェーズ 1 でいずれかの形状が安定性ゲート
//! （`bench_harness::ab::STABILITY_SPREAD_GATE`）を超過した場合、フェーズ
//! 2・3 は実行するが判定は `verdict=undetermined` に倒す（安全側判断）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs`
//! ドキュメンテーションコメント（同ディレクトリ）と同一。

/// フェーズ 2・3（`macos_impl`）の 1 run 分の集計から単一 run の参考
/// verdict を計算する純関数。macOS 依存部分（Metal コンテキスト・計測）を
/// 一切持たないため `#[cfg(target_os = "macos")]` の外に置き、Linux CI
/// （`cargo test --workspace`）でもユニットテストとして検査対象にする
/// （`gemm_fine_barrier_ab_bench.rs::single_run_verdict` と同型の設計）。
///
/// `gemm_fine_barrier_ab_bench.rs`（大形状の改善を要求）とは判断基準が
/// 異なる: 本 example の対象形状は「正方 4 形状（unroll 分岐を通らない
/// 想定。改善なしで同等が期待値）」と「補助 2 形状（unroll 分岐を通る）」
/// が混在するため、**全形状で改善を要求せず、全形状で非後退（5% 許容）の
/// みを要求する**（実装計画 §1.2(A)・§3.2「単一 run 参考 verdict」）。
///
/// 最終的な採否判断は本関数の 1 run 出力ではなく、5 run 中央値に基づき
/// `docs/perf/metal-gemm-n4096-kernel-gap.md` §7.10 で人間が確定する
/// （実装計画 §3.4）。本関数の `verdict=` ログ出力はその中間証跡
/// （機械的 `grep verdict=` で単一 run の判定を追える参考表示）にとどまる。
// Linux CI（`cargo clippy --workspace --all-targets --all-features -- -D
// warnings`）は本 example の `example` ターゲット（`cfg(test)` 無効・
// `target_os = "macos"` 無効）を単独でも lint するため、macOS 実機専用の
// `main`（`macos_impl::main`）からしか呼ばれない本型は非 macOS かつ非
// テストの経路では文字通り到達不能になり dead_code 検知の対象になる
// （`gemm_fine_barrier_ab_bench.rs` と同じ理由。codex-review 指摘 PR
// #1372 のパターンを踏襲）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SingleRunVerdict {
    AdoptCandidate,
    RejectCandidate,
    Undetermined,
}

// enum 本体（上）とは異なり、`as_str` は `single_run_verdict_tests`
// （Linux CI の `#[cfg(test)]` ビルド）からも呼ばれないため、非 macOS
// では test cfg の有無に関わらず常に未使用となる
// （`gemm_fine_barrier_ab_bench.rs` と同じ理由）。
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

/// 採否判断における全形状（正方 4 形状＋補助 2 形状）共通の非後退許容比率
/// 下限（`docs/perf/metal-gemm-fine-barrier-ab.md`
/// `SMALL_SIZE_REGRESSION_TOLERANCE_RATIO` と同一値の再利用。新規閾値の
/// 導入ではない。実装計画 §3.2 判定基準）。
///
/// `gemm_swizzle_ab_bench.rs::SMALL_SIZE_REGRESSION_TOLERANCE_RATIO` と
/// 同じ導出方式（`0.95` を独自にハードコードせず、単一真実源
/// `bench_harness::ab::STABILITY_SPREAD_GATE`〈0.05〉から導出する）へ
/// 揃える（イシュー #1284・codex-review P1 指摘対応。分散定義の解消）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const REGRESSION_TOLERANCE_RATIO: f64 = 1.0 - bench_harness::ab::STABILITY_SPREAD_GATE;

/// `ratios`: 形状ごとの `head_over_base`（TFLOPS 比。head/base）。正方 4
/// 形状・補助 2 形状のいずれも区別せず同じ基準を適用する
/// （`tile::UNROLL_ACC_MIN_PRODUCT` 分岐の構造上、正方形状は不変〈≈1.0〉が
/// 期待値であり改善を要求しないため）。
/// `gate_exceeded`: フェーズ 2／3 の A/B 計測自体で安定性ゲート超過が 1 件
/// でも残ったか（ラウンド間ばらつきが大きく計測値を信頼できないため
/// `Undetermined` へ倒す安全側判断）。
///
/// 判断基準（実装計画 §3.2・§9 Phase 3 Step 9）: 全形状の `head_over_base`
/// が [`REGRESSION_TOLERANCE_RATIO`] 以上（5% 超の劣化がない）場合に採用
/// 候補とする。改善（>1.0）は採用条件に含めない
/// （本ファイル冒頭ドキュメンテーションコメント §「本 example が測る
/// 構造的な事実」参照）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn single_run_verdict(ratios: &[(usize, bool, f64)], gate_exceeded: bool) -> SingleRunVerdict {
    if gate_exceeded {
        return SingleRunVerdict::Undetermined;
    }
    let no_regression = ratios
        .iter()
        .all(|(_, _, ratio)| *ratio >= REGRESSION_TOLERANCE_RATIO);
    if no_regression {
        SingleRunVerdict::AdoptCandidate
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

    /// 他 A/B bench と同一値（決定的シード。過去 PoC・CPU 実装ベンチと
    /// 同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// フェーズ 1〜3 共通のラウンド数・cooldown・時間ベースウォームアップ
    /// 下限（`gemm_fine_barrier_ab_bench.rs` と同一初期値。実機で安定性
    /// ゲートを満たせない場合は手順書
    /// （`docs/perf/metal-bench-noise-protocol.md`）の調整手順に従い
    /// 増やす方向のみ許す。コミット前に既定値へ戻す）。
    const ROUNDS: usize = 6;
    const COOLDOWN: Duration = Duration::from_secs(2);
    const MIN_WARMUP: Duration = Duration::from_secs(1);

    /// 計測対象形状。`m == n == k` の正方 4 形状（本番 `dispatch_auto`
    /// が exact-match 実測テーブルで `acc_rows*acc_cols=8` の候補を返し、
    /// unroll 分岐を通らない想定。実装計画 §1.2(A)）と、`CANDIDATES[0]`
    /// （`acc_rows*acc_cols=16`）へ到達する補助形状（同 §1.2(B)）を区別
    /// するため `supplementary` フラグを持つ。
    #[derive(Clone, Copy)]
    struct Shape {
        m: usize,
        n: usize,
        k: usize,
        supplementary: bool,
    }

    const SHAPES: &[Shape] = &[
        Shape {
            m: 512,
            n: 512,
            k: 512,
            supplementary: false,
        },
        Shape {
            m: 1024,
            n: 1024,
            k: 1024,
            supplementary: false,
        },
        Shape {
            m: 2048,
            n: 2048,
            k: 2048,
            supplementary: false,
        },
        Shape {
            m: 4096,
            n: 4096,
            k: 4096,
            supplementary: false,
        },
        // `tile::select_with_occupancy_for_device` の `m >= LARGE && n >=
        // LARGE`（非正方・非 tall/wide 大形状）分岐へ到達し `CANDIDATES[0]`
        // （acc 積 16）が選ばれる補助形状（実装計画 §1.2(B)）。
        Shape {
            m: 2048,
            n: 2048,
            k: 512,
            supplementary: true,
        },
        Shape {
            m: 4096,
            n: 4096,
            k: 1024,
            supplementary: true,
        },
    ];

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
    }

    /// `gemm_fine_barrier_ab_bench.rs::head_over_base_tflops_ratio` と同一
    /// ロジック（TFLOPS 比は実行時間比の逆数。取り違え防止のため 1 箇所に
    /// 集約する）。
    fn head_over_base_tflops_ratio(
        m: usize,
        n: usize,
        k: usize,
        median_a_secs: f64,
        median_b_secs: f64,
    ) -> f64 {
        tflops(m, n, k, median_b_secs) / tflops(m, n, k, median_a_secs)
    }

    /// 候補の `acc_rows*acc_cols` 積（`TileConfig` の `acc_rows`/`acc_cols`
    /// は `pub(crate)` のため、`crate` 外の example からは同じ式
    /// `(bm/wm)/8 * (bn/wn)/8` を独自に再計算する）。ログ出力の表示専用
    /// （`acc_product=` フィールド）であり、unroll 分岐の採否判定自体は
    /// `TileConfig::unroll_acc_loops`（イシュー #1284 で `pub` 化。単一
    /// 真実源）を直接呼ぶため本関数の値には依存しない。
    fn acc_product(cfg: tile::TileConfig) -> u32 {
        let acc_rows = (cfg.bm / cfg.wm) / 8;
        let acc_cols = (cfg.bn / cfg.wn) / 8;
        acc_rows * acc_cols
    }

    /// フェーズ 0: base（`unroll_acc_enabled=false`）/head（`true`）の出力が
    /// ビット単位で一致することを計測前に自己検証する（本ファイル冒頭
    /// ドキュメンテーションコメント参照。loop unroll はオペランド列を
    /// 変えないため理論上 base/head は完全一致するはずで、崩れている場合は
    /// A/B 計測へ進まない安全側判断）。
    fn phase0_bit_match_selfcheck(ctx: &MetalContext) {
        println!("--- フェーズ 0: base/head 出力 bit 一致の自己検証 ---");

        let base_gemm = MetalGemm::new_with_unroll_acc(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_unroll_acc(ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        for shape in SHAPES {
            let Shape { m, n, k, .. } = *shape;
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(m * k);
            let b = rng.fill_vec(k * n);

            let base_out = base_gemm
                .dispatch_auto(ctx, &a, &b, m, n, k)
                .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            let head_out = head_gemm
                .dispatch_auto(ctx, &a, &b, m, n, k)
                .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");

            // `assert_eq!` の `f32` 数値比較は IEEE 754 の `+0.0 == -0.0` を
            // 区別できず符号ビットの差異を見逃しうるため、`to_bits()` 経由の
            // ビットパターン比較で厳密に検証する（`gemm_swizzle_ab_bench.rs`
            // と同じ理由。イシュー #1284・codex-review P2 指摘）。
            let base_bits: Vec<u32> = base_out.iter().map(|v| v.to_bits()).collect();
            let head_bits: Vec<u32> = head_out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                base_bits, head_bits,
                "m={m} n={n} k={k}: UNROLL_ACC_ENABLED の有無で出力がビット単位で\
                 一致しなかった。演算オペランド列が変わっている疑いがあるため、\
                 A/B 計測（性能比較）は無意味になる。shaders/gemm.metal の\
                 UNROLL_ACC_ENABLED 挿入箇所を確認すること。"
            );
            println!("m={m} n={n} k={k} bit-exact match: OK");
        }
    }

    /// フェーズ 1: 本番既定経路（`MetalGemm::new`）の `dispatch_auto` を
    /// 形状ごとに [`run_stability`] で計測し、spread を出力する
    /// （`gemm_fine_barrier_ab_bench.rs` と同一構造）。
    fn phase1_stability_selfcheck(ctx: &MetalContext, gemm: &MetalGemm) -> bool {
        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        println!("--- フェーズ 1: 安定性セルフチェック（対照カーネル: 本番既定 dispatch_auto）---");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        let mut all_within_gate = true;
        for shape in SHAPES {
            let Shape { m, n, k, .. } = *shape;
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(m * k);
            let b = rng.fill_vec(k * n);

            let result = run_stability(&ab_config, &measurement_config, || {
                gemm.dispatch_auto(ctx, &a, &b, m, n, k)
                    .expect("Metal GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            })
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            let within_gate = result.spread <= SPREAD_GATE;
            all_within_gate &= within_gate;

            let round_tflops: Vec<f64> = result
                .round_medians_secs
                .iter()
                .map(|&secs| tflops(m, n, k, secs))
                .collect();
            println!(
                "m={m} n={n} k={k} spread={:.4} ({}) round_tflops={round_tflops:.4?}",
                result.spread,
                if within_gate { "OK" } else { "NG: gate 超過" }
            );
        }

        if !all_within_gate {
            println!(
                "--- フェーズ 1 判定: 一部形状が spread ≤{SPREAD_GATE:.2} 相当を満たさなかった。\
                 フェーズ 2・3（A/B 判定）は実行するが verdict は undetermined に倒す\
                 （安全側判断: 判定不可のまま採否を確定しない）。"
            );
        }
        all_within_gate
    }

    /// フェーズ 2: `MetalGemm::new_with_unroll_acc` で base
    /// （`unroll_acc_enabled=false`）/head（`true`）の 2 インスタンスを
    /// 同一プロセス内に構築し、[`run_ab`] で interleaved 計測する
    /// （prepared 境界。アップロード済みバッファを base/head で共有。
    /// **本 Issue の主判定対象**。`gemm_fine_barrier_ab_bench.rs::
    /// phase2_fine_barrier_ab` と同一構造）。
    ///
    /// `cfg` は `tile::select_for_device` で導出し、本番 `dispatch_auto`
    /// （内部で同じ関数を呼ぶ。`MetalGemm::dispatch_auto` 実装参照）と
    /// 同一の選択結果を使う。
    fn phase2_unroll_acc_ab(ctx: &MetalContext) -> (Vec<(usize, bool, f64)>, bool) {
        println!(
            "--- フェーズ 2: 条件付き loop unroll A/B（prepared 境界。base=off / head=on）---"
        );

        let base_gemm = MetalGemm::new_with_unroll_acc(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_unroll_acc(ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        // `ratios` の要素は `(m, supplementary, head_over_base)`。`m` は
        // 表示用（正方形状のため `n`/`k` と同値。補助形状も `m` で一意に
        // 識別できる: 2048/2048/512 と 4096/4096/1024 は他形状と `m` が
        // 重複しない）。
        let mut ratios: Vec<(usize, bool, f64)> = Vec::new();
        let mut gate_exceeded = false;

        for shape in SHAPES {
            let Shape {
                m,
                n,
                k,
                supplementary,
            } = *shape;
            let cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
            let acc_product = acc_product(cfg);
            // `tile::UNROLL_ACC_MIN_PRODUCT`（16）とのハードコード複製を
            // 避けるため、本番判定と同じ単一真実源
            // `TileConfig::unroll_acc_loops`（イシュー #1284 で `pub` 化）
            // を直接呼ぶ（codex-review P1 指摘対応）。
            let head_unroll_expected = cfg.unroll_acc_loops();

            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(m * k);
            let b = rng.fill_vec(k * n);

            let a_buf = MetalBuffer::new_with_data(ctx, &a)
                .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let b_buf = MetalBuffer::new_with_data(ctx, &b)
                .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
            let c_buf = MetalBuffer::new_zeroed(ctx, m * n)
                .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");

            let base_resolved = base_gemm
                .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
                .expect("base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
            let head_resolved = head_gemm
                .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
                .expect("head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");

            let result = run_ab(
                &ab_config,
                &measurement_config,
                || {
                    base_gemm
                        .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
                        .expect("base GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
                },
                || {
                    head_gemm
                        .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
                        .expect("head GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
                },
            )
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            let base_tflops: Vec<f64> = result
                .a_round_medians_secs
                .iter()
                .map(|&secs| tflops(m, n, k, secs))
                .collect();
            let head_tflops: Vec<f64> = result
                .b_round_medians_secs
                .iter()
                .map(|&secs| tflops(m, n, k, secs))
                .collect();

            let base_median_tflops = tflops(m, n, k, result.median_a_secs);
            let head_median_tflops = tflops(m, n, k, result.median_b_secs);
            let head_over_base =
                head_over_base_tflops_ratio(m, n, k, result.median_a_secs, result.median_b_secs);

            println!(
                "m={m} n={n} k={k} supplementary={supplementary} \
                 base_resolved=({}x{}, {}) head_resolved=({}x{}, {}) acc_product={acc_product} \
                 head_unroll_expected={head_unroll_expected} base_median_tflops={:.4} \
                 head_median_tflops={:.4} head_over_base={:.4} spread_base={:.4} \
                 spread_head={:.4} base_round_tflops={base_tflops:.4?} \
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
            ratios.push((m, supplementary, head_over_base));
        }

        println!(
            "--- フェーズ 2 完了（prepared 境界。本 Issue の主判定対象）。\
             `docs/perf/metal-gemm-n4096-kernel-gap.md` §7.10 へ記録すること。"
        );
        (ratios, gate_exceeded)
    }

    /// フェーズ 3: `dispatch_auto`（ホストスライス・転送込み）で base/head
    /// を [`run_ab`] 計測する参考値（イシュー #1284 の「本番 `dispatch_auto`
    /// 経路」の文言に直接対応。採否の主判定はフェーズ 2）。
    /// `gemm_swizzle_ab_bench.rs::phase3_swizzle_ab_transfer_included` と
    /// 同一構造。
    fn phase3_unroll_acc_ab_transfer_included(
        ctx: &MetalContext,
    ) -> (Vec<(usize, bool, f64)>, bool) {
        println!(
            "--- フェーズ 3: 条件付き loop unroll A/B（転送込み境界・参考値。base=off / head=on）---"
        );

        let base_gemm = MetalGemm::new_with_unroll_acc(ctx, false)
            .expect("base GEMM パイプラインの構築に失敗した");
        let head_gemm = MetalGemm::new_with_unroll_acc(ctx, true)
            .expect("head GEMM パイプラインの構築に失敗した");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        let mut ratios: Vec<(usize, bool, f64)> = Vec::new();
        let mut gate_exceeded = false;

        for shape in SHAPES {
            let Shape {
                m,
                n,
                k,
                supplementary,
            } = *shape;
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(m * k);
            let b = rng.fill_vec(k * n);

            let result = run_ab(
                &ab_config,
                &measurement_config,
                || {
                    base_gemm
                        .dispatch_auto(ctx, &a, &b, m, n, k)
                        .expect("base GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
                },
                || {
                    head_gemm
                        .dispatch_auto(ctx, &a, &b, m, n, k)
                        .expect("head GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
                },
            )
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            let base_tflops: Vec<f64> = result
                .a_round_medians_secs
                .iter()
                .map(|&secs| tflops(m, n, k, secs))
                .collect();
            let head_tflops: Vec<f64> = result
                .b_round_medians_secs
                .iter()
                .map(|&secs| tflops(m, n, k, secs))
                .collect();

            let base_median_tflops = tflops(m, n, k, result.median_a_secs);
            let head_median_tflops = tflops(m, n, k, result.median_b_secs);
            let head_over_base =
                head_over_base_tflops_ratio(m, n, k, result.median_a_secs, result.median_b_secs);

            println!(
                "m={m} n={n} k={k} supplementary={supplementary} \
                 base_median_tflops={:.4} head_median_tflops={:.4} head_over_base={:.4} \
                 spread_base={:.4} spread_head={:.4} base_round_tflops={base_tflops:.4?} \
                 head_round_tflops={head_tflops:.4?}",
                base_median_tflops,
                head_median_tflops,
                head_over_base,
                result.spread_a,
                result.spread_b,
            );

            if result.spread_a > SPREAD_GATE || result.spread_b > SPREAD_GATE {
                gate_exceeded = true;
            }
            ratios.push((m, supplementary, head_over_base));
        }

        println!(
            "--- フェーズ 3 完了（転送込み境界。参考値）。\
             `docs/perf/metal-gemm-n4096-kernel-gap.md` §7.10 へ記録すること。"
        );
        (ratios, gate_exceeded)
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

        // 一時調整した ROUNDS/COOLDOWN/MIN_WARMUP をコミットしない運用
        // （`docs/perf/metal-bench-noise-protocol.md` の調整手順）のため、
        // 実効値を実行ログの先頭に残す（env_info と併せて実測記録の
        // 証跡にする）。
        println!(
            "effective_rounds={ROUNDS} effective_cooldown_secs={} effective_min_warmup_secs={}",
            COOLDOWN.as_secs_f64(),
            MIN_WARMUP.as_secs_f64(),
        );

        phase0_bit_match_selfcheck(&ctx);

        // フェーズ 1 は本番既定（unroll acc off。`tile::UNROLL_ACC_ENABLED`
        // は本 example 実行時点でコミット済みの値のまま）の `MetalGemm::new`
        // で行う（対照カーネルの計測プロトコル自体を検証する目的のため、
        // unroll acc の有無に依存しない構成を使う。`gemm_fine_barrier_ab_bench.rs`
        // と同じ判断）。
        let default_gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let phase1_ok = phase1_stability_selfcheck(&ctx, &default_gemm);
        if !phase1_ok {
            println!(
                "--- フェーズ 1 が一部形状で安定性ゲートを満たさなかったが、\
                 フェーズ 2・3 は実行を継続する（verdict は undetermined に倒す）。"
            );
        }

        let (phase2_ratios, phase2_gate_exceeded) = phase2_unroll_acc_ab(&ctx);
        let (phase3_ratios, phase3_gate_exceeded) = phase3_unroll_acc_ab_transfer_included(&ctx);

        // フェーズ 2（本判定対象）の ratios を採否判断の主対象とする
        // （フェーズ 3 は転送込み境界の参考値。実装計画 §3.2「単一 run
        // 参考 verdict」・§3.4）。gate 超過はフェーズ 1〜3 いずれかで
        // 発生したものを合算する（安全側判断: どのフェーズで発生しても
        // Undetermined へ倒す）。
        let gate_exceeded = !phase1_ok || phase2_gate_exceeded || phase3_gate_exceeded;
        let verdict = super::single_run_verdict(&phase2_ratios, gate_exceeded);

        println!(
            "--- 実行完了。採否判定基準（正方 4 形状・補助 2 形状すべての \
             head_over_base が {:.2} 以上であること。改善は要求しない。実装計画\
             §1.2(A) 参照）に従い、5 run 中央値を \
             `docs/perf/metal-gemm-n4096-kernel-gap.md` §7.10 へ記録すること。\
             フェーズ 3（転送込み境界）の ratios は参考値: {phase3_ratios:?}",
            super::REGRESSION_TOLERANCE_RATIO,
        );
        println!(
            "verdict={} (フェーズ 2〈prepared 境界〉基準・単一 run の参考値。\
             5 run 中央値で最終判断する)",
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
        "backend-metal gemm_unroll_acc_ab_bench example requires macOS (Apple Silicon). \
         See docs/perf/metal-bench-noise-protocol.md and \
         docs/perf/metal-gemm-n4096-kernel-gap.md §7.10 for the real-hardware execution \
         procedure."
    );
}

#[cfg(test)]
mod single_run_verdict_tests {
    use super::*;

    #[test]
    fn adopts_when_all_shapes_hold_within_5_percent() {
        let ratios = vec![
            (512, false, 0.98),
            (1024, false, 1.0),
            (2048, false, 1.01),
            (4096, false, 0.99),
            (2048, true, 1.12),
            (4096, true, 1.08),
        ];
        assert_eq!(
            single_run_verdict(&ratios, false),
            SingleRunVerdict::AdoptCandidate
        );
    }

    #[test]
    fn rejects_when_a_square_shape_regresses_more_than_5_percent() {
        let ratios = vec![
            (512, false, 0.90),
            (1024, false, 1.0),
            (2048, false, 1.0),
            (4096, false, 1.0),
            (2048, true, 1.1),
            (4096, true, 1.1),
        ];
        assert_eq!(
            single_run_verdict(&ratios, false),
            SingleRunVerdict::RejectCandidate
        );
    }

    #[test]
    fn rejects_when_a_supplementary_shape_regresses_more_than_5_percent() {
        let ratios = vec![
            (512, false, 1.0),
            (1024, false, 1.0),
            (2048, false, 1.0),
            (4096, false, 1.0),
            (2048, true, 0.80),
            (4096, true, 1.1),
        ];
        assert_eq!(
            single_run_verdict(&ratios, false),
            SingleRunVerdict::RejectCandidate
        );
    }

    #[test]
    fn undetermined_when_stability_gate_exceeded_even_if_ratios_look_good() {
        let ratios = vec![
            (512, false, 1.0),
            (1024, false, 1.0),
            (2048, false, 1.0),
            (4096, false, 1.0),
            (2048, true, 1.2),
            (4096, true, 1.3),
        ];
        assert_eq!(
            single_run_verdict(&ratios, true),
            SingleRunVerdict::Undetermined
        );
    }
}
