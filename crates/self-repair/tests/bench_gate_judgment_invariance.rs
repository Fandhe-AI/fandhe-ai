//! TASK-3.2b（イシュー #138）の受け入れ条件「同一入力で付け替え前後の判定が一致する」を
//! 固定する回帰テスト。
//!
//! # 背景: 何の「付け替え前後」を比較するか
//!
//! TASK-3.2a（#137）は `self-repair` の検証フェーズベンチゲートの**計測系**
//! （Criterion/Burn 相当 → `bench-harness`）のみを付け替え、**判定ロジック**
//! （中央値算出・閾値判定・fail-closed 拒否）自体は変更しない契約だった
//! （`crates/self-repair/src/verify_bench.rs` モジュール冒頭ドキュメント・
//! `docs/spec/05-tasks.md` TASK-3.2）。v1 コード自体は本リポに存在しないため、
//! 本テストは「付け替え前」を次の 2 層で表現する:
//!
//! 1. **中央値定義そのものの参照実装**（[`reference_median`]）: `bench_harness::
//!    median_q1_q3`（median-of-halves 方式。`crates/bench-harness/src/stats.rs` に
//!    「この定義自体をテスト期待値に固定する」と明記されている）と独立に、同じ定義
//!    （ソート後 `idx = round(0.5*(n-1))` 番目）をテスト内に再実装し、新経路
//!    （[`guardrail::bench_gate::BenchSignal::from_measurements_pct`]）の算出値と
//!    突合する。中央値定義が線形補間方式等へ将来単独変更された場合に検出できる。
//! 2. **付け替え前から存在する判定結線との等価性**: `guardrail::median_gate::
//!    bench_signal_from_measurements`（TASK-4.4a・#112。v1 移植の判定結線
//!    `guardrail::decision::decide` への唯一の構築経路であり、TASK-3.2a より前から
//!    main に存在する）と、付け替え後に `self-repair` が実際に使う経路
//!    （`guardrail::bench_gate::BenchSignal::from_measurements_pct` →
//!    `guardrail::decision::BenchSignal::Measured` へ手動変換）に**同一の劣化率系列**を
//!    与え、中央値・`decide()` の `Verdict` が一致することを確認する。
//!
//! 閾値は組み込み default プリセット（`lines_max=200`・`bench_median_max_pct=5.0`・
//! `bench_runs_min=5`。TASK-4.3c・#117 確定値）を数値変更なしで使う
//! （ガードレール閾値の緩和はユーザー承認必須。`.claude/rules/security.md`）。
//!
//! `guardrail::decision::BenchSignal`（判定入力の型）と `guardrail::bench_gate::
//! BenchSignal`（計測結果 DTO。付け替え後の計測系が返す型）は同名だが別の型のため、
//! 本ファイルでは `DecisionBenchSignal`／`MeasuredBenchSignal` にエイリアスして明示的に
//! 区別する。

use guardrail::config::{self, PresetName};
use guardrail::decision::{
    BenchSignal as DecisionBenchSignal, DecisionInput, GateSignal, GateSignals, Verdict, decide,
};
use guardrail::median_gate::{self, MedianGateError};
use guardrail::{Thresholds, bench_gate};
use self_repair::verify_bench::{BenchGateError, MIN_BENCH_ITERATIONS, SelfRepairBenchGate};

/// [`bench_harness::median_q1_q3`]（median-of-halves 方式）と独立にテスト内で
/// 再実装した中央値定義（ソート後 `idx = round(0.5*(n-1))` 番目の要素）。
///
/// `bench_harness` を呼ばずに定義そのものを固定することで、将来
/// `bench_harness::median_q1_q3` 側の定義が変わった場合にこのテストが検出できる
/// （呼び出し経路の一致だけでなく定義自体の不変を確認する。モジュール冒頭
/// ドキュメント「背景」節 1. 参照）。
fn reference_median(samples: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let idx = (0.5 * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)]
}

fn default_thresholds() -> Thresholds {
    config::Thresholds::builtin(PresetName::Default)
}

fn all_passed_gates() -> GateSignals {
    GateSignals {
        build: GateSignal::Passed,
        test: GateSignal::Passed,
        clippy: GateSignal::Passed,
    }
}

/// 付け替え後の経路（`guardrail::bench_gate::BenchSignal`）が返す中央値を、
/// 判定本体（`guardrail::decision::decide`）が受け取る
/// `guardrail::decision::BenchSignal::Measured` へ変換する。
///
/// `self-repair` 側に `bench_gate::BenchSignal` → `decision::BenchSignal` の
/// 変換関数は存在しない（TASK-3.1c 未結線。`verify_bench.rs` モジュール冒頭
/// ドキュメント参照）ため、本テストは比較のためにここで手動変換する。
fn measured_from_post_path(median_pct: f64) -> DecisionBenchSignal {
    DecisionBenchSignal::Measured { median_pct }
}

// ---------------------------------------------------------------------
// ケース 1: 中央値定義そのものの参照実装との一致
// ---------------------------------------------------------------------

#[test]
fn median_matches_v1_reference_definition() {
    let series: Vec<Vec<f64>> = vec![
        // 奇数件（5 件）。`median_gate` の既存テストと同一系列（期待値 3.0 で
        // 既に検証済み。`crates/guardrail/src/median_gate.rs`
        // `median_is_recomputed_not_trusted_from_caller`）。
        vec![5.0, 1.0, 4.0, 2.0, 3.0],
        // 偶数件（6 件）。「中央 2 値の平均」方式とは値が乖離するため、
        // 定義の再乖離を検出する検出力を持つ（`bench_gate.rs`
        // `report_median_agrees_with_bench_signal_validate_for_even_count` と同種の理由）。
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        // 下限超過（7 件）・外れ値混入。`median_gate` の既存テストと同一系列
        // （期待値 4.0 で既に検証済み。`accepts_more_than_min_iterations`）。
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
        // 負値（改善）を含む混在系列。
        vec![-5.0, -1.0, 0.0, 2.0, 10.0],
        // 外れ値・改善・劣化が混在する 6 件系列。
        vec![-10.0, -2.0, 0.5, 1.5, 3.0, 50.0],
    ];

    for measurements in series {
        let expected = reference_median(&measurements);
        let signal = bench_gate::BenchSignal::from_measurements_pct(measurements.clone())
            .expect("5 件以上・有限値のみのため成功するはず");
        assert_eq!(
            signal.bench_median_pct, expected,
            "系列 {measurements:?} の中央値が参照実装と不一致"
        );
    }

    // 既存モジュールのテストで既に固定されている期待値との二重確認
    // （テスト内参照実装が既存契約と整合していることの裏付け）。
    assert_eq!(reference_median(&[5.0, 1.0, 4.0, 2.0, 3.0]), 3.0);
    assert_eq!(
        reference_median(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]),
        4.0
    );
}

// ---------------------------------------------------------------------
// ケース 2: 付け替え前後の中央値の一致
// ---------------------------------------------------------------------

#[test]
fn same_input_yields_identical_median_across_pre_and_post_paths() {
    let series_list: Vec<Vec<f64>> = vec![
        vec![5.0, 1.0, 4.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
        vec![-5.0, -1.0, 0.0, 2.0, 10.0],
    ];

    for series in series_list {
        // 付け替え前: `median_gate`（TASK-4.4a・#112。TASK-3.2a より前から main に存在）。
        let pre = median_gate::bench_signal_from_measurements(&series)
            .expect("5 件以上・有限値のみのため成功するはず");
        let DecisionBenchSignal::Measured {
            median_pct: pre_median,
        } = pre
        else {
            panic!("bench_signal_from_measurements は常に Measured を返すはず");
        };

        // 付け替え後: `bench_gate`（TASK-4.1d・#107。self-repair が実際に使う経路）。
        let post = bench_gate::BenchSignal::from_measurements_pct(series.clone())
            .expect("5 件以上・有限値のみのため成功するはず");

        assert_eq!(
            pre_median, post.bench_median_pct,
            "系列 {series:?} で付け替え前後の中央値が不一致"
        );
    }
}

// ---------------------------------------------------------------------
// ケース 3: 付け替え前後の Verdict の一致
// ---------------------------------------------------------------------

#[test]
fn same_input_yields_identical_verdict_across_pre_and_post_paths() {
    let thresholds = default_thresholds();

    // (系列, 期待する Verdict) の組。
    let cases: Vec<(Vec<f64>, Verdict)> = vec![
        // 中央値が閾値内（3.0 < 5.0）→ 両経路とも AutoApply。
        (vec![1.0, 2.0, 3.0, 4.0, 5.0], Verdict::AutoApply),
        // 中央値が閾値超過（7.5 > 5.0）→ 両経路とも Escalate。
        (vec![5.5, 6.5, 7.5, 8.5, 9.5], Verdict::Escalate),
        // 境界ちょうど（median == bench_median_max_pct == 5.0）
        // → 両経路とも AutoApply（「境界値ちょうどは罰しない」契約の不変確認。
        // `crates/guardrail/src/decision.rs` `bench_at_exact_threshold_is_within_limit`）。
        (
            vec![
                thresholds.bench_median_max_pct - 2.0,
                thresholds.bench_median_max_pct - 1.0,
                thresholds.bench_median_max_pct,
                thresholds.bench_median_max_pct + 1.0,
                thresholds.bench_median_max_pct + 2.0,
            ],
            Verdict::AutoApply,
        ),
        // 改善（負の中央値）→ 両経路とも AutoApply。
        (vec![-10.0, -5.0, -3.0, -1.0, 0.0], Verdict::AutoApply),
    ];

    for (series, expected_verdict) in cases {
        // 付け替え前経路。
        let pre_signal = median_gate::bench_signal_from_measurements(&series)
            .expect("5 件以上・有限値のみのため成功するはず");
        let pre_input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            pre_signal,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");
        let pre_decision = decide(&pre_input).expect("判定に失敗");

        // 付け替え後経路。
        let post_signal = bench_gate::BenchSignal::from_measurements_pct(series.clone())
            .expect("5 件以上・有限値のみのため成功するはず");
        let post_input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            measured_from_post_path(post_signal.bench_median_pct),
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");
        let post_decision = decide(&post_input).expect("判定に失敗");

        assert_eq!(
            pre_decision.verdict(),
            expected_verdict,
            "系列 {series:?} の付け替え前 Verdict が期待値と不一致"
        );
        assert_eq!(
            pre_decision.verdict(),
            post_decision.verdict(),
            "系列 {series:?} で付け替え前後の Verdict が不一致"
        );
    }
}

// ---------------------------------------------------------------------
// ケース 4: fail-closed 拒否の等価性
// ---------------------------------------------------------------------

#[test]
fn fail_closed_rejection_is_equivalent_across_paths() {
    // 下限未満（4 件）: REQ-4「5 回以上」への違反。
    let too_few = vec![1.0, 2.0, 3.0, 4.0];
    let pre_err = median_gate::bench_signal_from_measurements(&too_few)
        .expect_err("4 件は下限未満のため拒否されるはず");
    assert!(matches!(
        pre_err,
        MedianGateError::TooFewMeasurements { got: 4, min: 5 }
    ));
    let post_err = bench_gate::BenchSignal::from_measurements_pct(too_few)
        .expect_err("4 件は下限未満のため拒否されるはず");
    assert!(matches!(post_err, BenchGateError::InvalidSignal(_)));

    // NaN 混入（5 件）: 単発の異常計測が中央値を汚染するのを防ぐための fail-closed。
    let with_nan = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
    let pre_err = median_gate::bench_signal_from_measurements(&with_nan)
        .expect_err("NaN 混入は拒否されるはず");
    assert!(matches!(
        pre_err,
        MedianGateError::NonFiniteMeasurement { index: 2, .. }
    ));
    let post_err = bench_gate::BenchSignal::from_measurements_pct(with_nan)
        .expect_err("NaN 混入は拒否されるはず");
    assert!(matches!(post_err, BenchGateError::Measurement(_)));
}

// ---------------------------------------------------------------------
// ケース 5: bench-harness 実測系列でも付け替え前後の Verdict が一致する
// ---------------------------------------------------------------------

/// 空クロージャは `Instant::elapsed` がゼロを返しうり偶発的な `NonFiniteRatio` を
/// 招くため、`black_box` 経由で実測時間を確実に非ゼロにする軽量ワークロードを使う
/// （`crates/self-repair/tests/bench_gate_completion.rs` と同一パターン）。
fn cpu_workload() -> impl FnMut() {
    use std::hint::black_box;
    || {
        let mut acc: u64 = 0;
        for i in 0..1_000u64 {
            acc = black_box(acc.wrapping_add(black_box(i)));
        }
        black_box(acc);
    }
}

#[test]
fn harness_measured_series_flows_to_identical_verdicts() {
    let thresholds = default_thresholds();

    // 付け替え後の実測経路（TASK-3.2a・#137）。実機非依存の軽量 CPU ワークロード。
    let gate = SelfRepairBenchGate::new();
    let mut baseline = cpu_workload();
    let mut candidate = cpu_workload();
    let measured = gate
        .run(MIN_BENCH_ITERATIONS, &mut baseline, &mut candidate)
        .expect("軽量ワークロードでの bench-harness 経由計測は成功するはず");
    assert_eq!(measured.bench_measurements_pct.len(), MIN_BENCH_ITERATIONS);

    // 同一の実測系列を付け替え前経路（`median_gate`）へも通す。
    let pre_signal = median_gate::bench_signal_from_measurements(&measured.bench_measurements_pct)
        .expect("実測系列は 5 件以上・有限値のため成功するはず");
    let DecisionBenchSignal::Measured {
        median_pct: pre_median,
    } = pre_signal
    else {
        panic!("bench_signal_from_measurements は常に Measured を返すはず");
    };

    // タイミング実測値そのもの（符号・大小）にはアサートしない
    // （`bench_gate_decision_integration.rs` の flaky 化回避方針を踏襲）。
    // 「同一系列 → 同一中央値」の相対比較のみを確認する。
    assert_eq!(
        pre_median, measured.bench_median_pct,
        "実測系列で付け替え前後の中央値が不一致"
    );

    // 同一中央値を経て Verdict も一致することを確認する（相対比較）。
    let pre_input = DecisionInput::new(
        &thresholds,
        10,
        all_passed_gates(),
        false,
        false,
        pre_signal,
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    let post_input = DecisionInput::new(
        &thresholds,
        10,
        all_passed_gates(),
        false,
        false,
        measured_from_post_path(measured.bench_median_pct),
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");

    let pre_decision = decide(&pre_input).expect("判定に失敗");
    let post_decision = decide(&post_input).expect("判定に失敗");
    assert_eq!(
        pre_decision.verdict(),
        post_decision.verdict(),
        "実測系列で付け替え前後の Verdict が不一致"
    );
}
