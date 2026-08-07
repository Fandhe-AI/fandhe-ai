//! 計測（`bench_harness::run`）→ 5 回以上の劣化率系列 → 中央値（`median_gate`）→
//! 3 分岐判定（`decision::decide`）までの通し統合テスト（TASK-4.4a・イシュー #112・REQ-4
//! 受け入れ条件「中央値採用の計測が guardrail 判定経路で動作する」）。
//!
//! # アサーション方針（flaky 化回避）
//!
//! `bench_harness::run` に基づく実測（[`harness_measurement_produces_valid_series`]）は
//! self-hosted runner の負荷変動でタイミング値そのものが変わりうるため、実測値の符号・
//! 大小関係にはアサーションしない（計画書 §6）。実測パートは「系列が
//! [`guardrail::median_gate::MIN_BENCH_ITERATIONS`] 件以上・全要素有限・
//! `median_gate` の検証を通過する」という構造のみを検証する。
//! `decide()` の 3 分岐そのものの検証（[`median_series_flows_into_auto_apply_verdict`]・
//! [`median_series_flows_into_escalate_verdict`]）は決定的な合成系列を用い、
//! タイミング値に依存しない。
//!
//! ワークロード計測に `black_box` 経由の軽量ループを使う理由は
//! `bench-harness/src/protocol.rs` の `run_does_not_optimize_workload_away` と同じ
//! （空クロージャは `Instant::elapsed` がゼロ・非有限比率になり `median_gate` に拒否
//! されうるため。PR #305 のレビュー指摘を踏襲）。

use bench_harness::{MeasurementConfig, run};
use guardrail::Thresholds;
use guardrail::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals, decide};
use guardrail::median_gate::{self, MIN_BENCH_ITERATIONS, MedianGateError};
use std::hint::black_box;

fn thresholds() -> Thresholds {
    Thresholds {
        lines_max: 200,
        bench_median_max_pct: 5.0,
        bench_runs_min: 5,
    }
}

fn all_passed_gates() -> GateSignals {
    GateSignals {
        build: GateSignal::Passed,
        test: GateSignal::Passed,
        clippy: GateSignal::Passed,
    }
}

/// 単体テストの実行時間を抑えるため下限（20/20）ちょうどを使う
/// （`bench_harness::protocol::MIN_ITERATIONS`）。
fn fast_config() -> MeasurementConfig {
    MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず")
}

/// 計測対象外に消えないよう、ある程度の計算量を持つ軽量ワークロード
/// （`bench-harness` の `run_does_not_optimize_workload_away` と同じパターン）。
fn busy_workload() {
    let mut acc: u64 = 0;
    for i in 0..10_000u64 {
        acc = black_box(acc.wrapping_add(black_box(i)));
    }
    black_box(acc);
}

/// 受け入れ条件の中核: `bench_harness::run` を baseline／candidate 各
/// `MIN_BENCH_ITERATIONS` 回以上呼び、劣化率系列を構成し、`median_gate` の検証を
/// 通過することを確認する（「計測部分が新実装計測系〈bench-harness〉経由で動作する」の
/// 直接検証）。
#[test]
fn harness_measurement_produces_valid_series() {
    let config = fast_config();
    let mut measurements_pct = Vec::with_capacity(MIN_BENCH_ITERATIONS);

    for _ in 0..MIN_BENCH_ITERATIONS {
        let baseline = run(&config, busy_workload).expect("baseline 計測は成功するはず");
        let candidate = run(&config, busy_workload).expect("candidate 計測は成功するはず");

        // 劣化率（%）。baseline_median_secs は busy_workload により正の値になることが
        // `run_does_not_optimize_workload_away`（bench-harness 側）で担保されている。
        let pct = (candidate.median_secs / baseline.median_secs - 1.0) * 100.0;
        measurements_pct.push(pct);
    }

    assert_eq!(measurements_pct.len(), MIN_BENCH_ITERATIONS);
    assert!(
        measurements_pct.iter().all(|v: &f64| v.is_finite()),
        "全計測値が有限であるはず: {measurements_pct:?}"
    );

    let signal = median_gate::bench_signal_from_measurements(&measurements_pct)
        .expect("MIN_BENCH_ITERATIONS 件・全有限の系列は median_gate の検証を通過するはず");
    match signal {
        BenchSignal::Measured { median_pct } => assert!(median_pct.is_finite()),
        BenchSignal::NotRun => panic!("計測済みのため Measured であるはず"),
    }
}

/// 通し経路: 合成した劣化率系列（改善方向）→ `median_gate` → `decision::decide` が
/// `auto_apply` を返すこと。
#[test]
fn median_series_flows_into_auto_apply_verdict() {
    // 中央値は -2.0（改善）。閾値 5.0% を下回るため自動適用となるはず。
    let measurements_pct = [-5.0, -3.0, -2.0, -1.0, 0.0];
    let bench = median_gate::bench_signal_from_measurements(&measurements_pct)
        .expect("5 件・全有限の系列は成功するはず");
    assert_eq!(bench, BenchSignal::Measured { median_pct: -2.0 });

    let t = thresholds();
    let input = DecisionInput::new(&t, 10, all_passed_gates(), false, false, bench, Vec::new())
        .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "auto_apply");
    assert!(decision.reasons().is_empty());
}

/// 通し経路: 合成した劣化率系列（劣化方向・閾値超過）→ `median_gate` →
/// `decision::decide` が `escalate` を返すこと。
#[test]
fn median_series_flows_into_escalate_verdict() {
    let t = thresholds();
    // 中央値は 8.0（閾値 5.0 を超過）。
    let measurements_pct = [6.0, 7.0, 8.0, 9.0, 10.0];
    let bench = median_gate::bench_signal_from_measurements(&measurements_pct)
        .expect("5 件・全有限の系列は成功するはず");
    assert_eq!(bench, BenchSignal::Measured { median_pct: 8.0 });

    let input = DecisionInput::new(&t, 10, all_passed_gates(), false, false, bench, Vec::new())
        .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "escalate");
    assert_eq!(decision.reason_conditions(), vec!["bench_median_exceeded"]);
}

/// fail-closed 回帰 (a): 計測 4 件（下限未満）は `median_gate` の時点で拒否され、
/// `decision::decide` に到達しない（単発・少数計測での閾値判定を防ぐ REQ-4 の中核）。
#[test]
fn fewer_than_five_measurements_are_rejected_before_decision() {
    let measurements_pct = [100.0, 100.0, 100.0, 100.0];
    let err = median_gate::bench_signal_from_measurements(&measurements_pct)
        .expect_err("4 件は下限（5 件）未満のため拒否されるはず");
    assert_eq!(err, MedianGateError::TooFewMeasurements { got: 4, min: 5 });
}

/// fail-closed 回帰 (b): NaN／非有限値混入は `median_gate` の時点で拒否され、
/// 汚染された中央値が判定へ渡ることを防ぐ。
#[test]
fn non_finite_measurement_is_rejected_before_decision() {
    let measurements_pct = [1.0, 2.0, f64::NAN, 4.0, 5.0];
    let err = median_gate::bench_signal_from_measurements(&measurements_pct)
        .expect_err("NaN 混入は拒否されるはず");
    assert!(matches!(
        err,
        MedianGateError::NonFiniteMeasurement { index: 2, .. }
    ));

    let measurements_pct_inf = [1.0, 2.0, 3.0, 4.0, f64::INFINITY];
    let err_inf = median_gate::bench_signal_from_measurements(&measurements_pct_inf)
        .expect_err("inf 混入は拒否されるはず");
    assert!(matches!(
        err_inf,
        MedianGateError::NonFiniteMeasurement { index: 4, .. }
    ));
}

/// 境界値ちょうどはエスカレーションしないこと（`decision.rs` 既存契約の通し経路確認）。
#[test]
fn median_at_threshold_boundary_does_not_escalate() {
    let t = thresholds();
    // 中央値がちょうど閾値（5.0）になる系列。
    let measurements_pct = [3.0, 4.0, 5.0, 6.0, 7.0];
    let bench = median_gate::bench_signal_from_measurements(&measurements_pct)
        .expect("5 件・全有限の系列は成功するはず");
    assert_eq!(
        bench,
        BenchSignal::Measured {
            median_pct: t.bench_median_max_pct
        }
    );

    let input = DecisionInput::new(&t, 10, all_passed_gates(), false, false, bench, Vec::new())
        .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "auto_apply");
}
