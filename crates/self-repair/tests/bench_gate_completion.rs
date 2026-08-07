//! TASK-3.2a（イシュー #137）の受け入れ条件「ベンチゲートが bench-harness 経由で
//! 完走する」を検証する統合テスト。
//!
//! `self_repair::verify_bench::SelfRepairBenchGate`（既定の `HarnessBenchGate` runner）
//! を軽量 CPU ワークロードで実行し、`guardrail::bench_gate::BenchSignal`（5 標本以上・
//! 中央値記録。REQ-4）が得られることを確認する。実機（CUDA・Metal）依存はないため
//! `#[ignore]` 分離は不要（`.claude/rules/coding-rust.md`）。
//!
//! `guardrail::bench_gate::BenchGateRunner::measure` の計測設定（`bench_harness::
//! MeasurementConfig`）は `guardrail` が再輸出していないため本クレートから型名として
//! 参照できず（`verify_bench` モジュール冒頭ドキュメント参照）、本テストは独自の
//! `BenchGateRunner` モック実装を持たない。エラー伝播（fail-closed）の確認は、
//! guardrail 側が実際に返す検証エラー（反復回数下限違反）を用いる。
//!
//! `BenchSignal`・`BenchGateError` は `self_repair::verify_bench` が `guardrail::
//! bench_gate` から再輸出しているため、本テストは `guardrail` を直接 import しない
//! （`verify_bench` モジュール冒頭ドキュメント「依存方向」参照）。

use self_repair::verify_bench::{
    BenchGateError, BenchSignal, MIN_BENCH_ITERATIONS, SelfRepairBenchGate, VerifyBenchError,
};
use std::hint::black_box;

/// 空クロージャは `Instant::elapsed` がゼロを返しうり、`NonFiniteRatio` として
/// 偶発的に失敗しうる（`guardrail::bench_gate` の単体テストと同種の既知の注意点。
/// `crates/guardrail/src/bench_gate.rs` の `harness_bench_gate_measures_five_iterations_
/// with_lightweight_workloads` を参照）。`black_box` 経由で実測時間を確実に非ゼロにする
/// 軽量ワークロードへ寄せる。
fn cpu_workload() -> impl FnMut() {
    || {
        let mut acc: u64 = 0;
        for i in 0..1_000u64 {
            acc = black_box(acc.wrapping_add(black_box(i)));
        }
        black_box(acc);
    }
}

#[test]
fn bench_gate_completes_via_bench_harness_with_default_runner() {
    let gate = SelfRepairBenchGate::new();
    let mut baseline = cpu_workload();
    let mut candidate = cpu_workload();

    let signal: BenchSignal = gate
        .run(MIN_BENCH_ITERATIONS, &mut baseline, &mut candidate)
        .expect("軽量ワークロードでの bench-harness 経由計測は成功するはず");

    assert_eq!(signal.bench_measurements_pct.len(), MIN_BENCH_ITERATIONS);
    assert!(signal.bench_median_pct.is_finite());
    signal
        .validate()
        .expect("構築された BenchSignal は guardrail 側の検証を通過するはず");
}

/// 反復回数が下限（[`MIN_BENCH_ITERATIONS`]）未満の場合、guardrail 側の判定
/// （`HarnessBenchGate::measure`）が拒否し、`self-repair` 側はそれをそのまま
/// fail-closed で伝播することを確認する（判定ロジックを迂回する経路を作らない。
/// エラーを握り潰さず [`VerifyBenchError::Gate`] として透過することも併せて確認する）。
#[test]
fn bench_gate_rejects_fewer_than_min_iterations() {
    let gate = SelfRepairBenchGate::new();
    let mut baseline = cpu_workload();
    let mut candidate = cpu_workload();

    let err = gate
        .run(MIN_BENCH_ITERATIONS - 1, &mut baseline, &mut candidate)
        .expect_err("下限未満の反復回数は拒否されるはず");
    assert!(matches!(
        err,
        VerifyBenchError::Gate(BenchGateError::InvalidSignal(_))
    ));
}
