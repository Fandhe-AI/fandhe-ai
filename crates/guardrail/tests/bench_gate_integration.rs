//! ベンチゲート計測系の統合テスト（TASK-4.1d・#107）。
//!
//! `guardrail::bench_gate::HarnessBenchGate` が `bench-harness` を計測実行系として
//! 実際に呼び出し、劣化率系列 → `BenchSignal` → JSON スキーマ整合まで通しで動作することを
//! 実証する。決定的なダミーワークロード（計算量差を持つ 2 クロージャ）を用い、実機・GPU
//! 非依存で CI（self-hosted）実行可能とする（`#[ignore]` 不要。実装計画 #107 4 節）。
//!
//! アサーションは計測値の符号（candidate が baseline より遅い/速い）ではなく、
//! 構造（件数下限・有限性・中央値の再計算整合・JSON フィールド名）に対して行う。
//! 符号への依存はセルフホスト runner の負荷変動で flaky になりうるため避ける
//! （実装計画 #107 5 節・レビュー指摘）。

use bench_harness::MeasurementConfig;
use guardrail::bench_gate::{BenchGateRunner, BenchSignal, HarnessBenchGate, MIN_BENCH_ITERATIONS};
use std::hint::black_box;

/// 単体テストと同様に下限（20/20）ちょうどで実行時間を抑える。
fn fast_config() -> MeasurementConfig {
    MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず")
}

#[test]
fn bench_gate_runs_via_bench_harness_and_produces_valid_signal() {
    let gate = HarnessBenchGate;
    let config = fast_config();

    // baseline: 軽量な計算量のダミーワークロード。
    let mut baseline = || {
        let mut acc: u64 = 0;
        for i in 0..1_000u64 {
            acc = black_box(acc.wrapping_add(black_box(i)));
        }
        black_box(acc);
    };
    // candidate: baseline より計算量の大きいダミーワークロード
    // （実際の劣化率の符号はアサーションに使わないが、両ワークロードが異なる処理である
    // ことで「計測部分が新実装経由で動作する」受け入れ条件をより明確に実証する）。
    let mut candidate = || {
        let mut acc: u64 = 0;
        for i in 0..10_000u64 {
            acc = black_box(acc.wrapping_add(black_box(i)));
        }
        black_box(acc);
    };

    let signal = gate
        .measure(&config, MIN_BENCH_ITERATIONS, &mut baseline, &mut candidate)
        .expect("bench-harness 経由の計測は成功するはず");

    // REQ-4 受け入れ基準「5 回以上」の直接検証。
    assert!(signal.bench_measurements_pct.len() >= MIN_BENCH_ITERATIONS);
    assert!(signal.bench_measurements_pct.iter().all(|v| v.is_finite()));
    assert!(signal.bench_median_pct.is_finite());

    // BenchSignal::validate が独自に再計算する中央値と一致すること（改竄検知経路の自己整合性）。
    signal
        .validate()
        .expect("HarnessBenchGate が構築した BenchSignal は検証を通過するはず");
}

#[test]
fn bench_signal_json_roundtrip_preserves_schema_field_names() {
    // 判定レポート JSON（docs/guardrail-self-repair-cli.md 2.1 節）のフィールド名
    // `bench_measurements_pct` / `bench_median_pct` を serde_json でそのまま検証する。
    let signal = BenchSignal::from_measurements_pct(vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("5 件の劣化率系列は成功するはず");

    let json = serde_json::to_string(&signal).expect("シリアライズは成功するはず");
    let value: serde_json::Value = serde_json::from_str(&json).expect("JSON パースは成功するはず");

    assert!(value.get("bench_measurements_pct").is_some());
    assert!(value.get("bench_median_pct").is_some());
    let measurements = value["bench_measurements_pct"]
        .as_array()
        .expect("bench_measurements_pct は配列であるはず");
    assert!(measurements.len() >= MIN_BENCH_ITERATIONS);

    // `BenchSignal::from_json`（`--signals` 注入経路〈CLI 仕様書 1.2 節〉の契約検証パスと
    // 同じ入口）はパース直後に必ず validate を通す。手動 from_str + 手動 validate ではなく
    // この公開 API を経由することで「検証を経ずに生値へアクセスできる公開経路を設けない」
    // 契約自体を検証する。
    let roundtripped =
        BenchSignal::from_json(&json).expect("検証済み BenchSignal の JSON は成功するはず");
    assert_eq!(roundtripped, signal);
}

#[test]
fn bench_signal_rejects_measurements_below_min_iterations() {
    let err = BenchSignal::from_measurements_pct(vec![1.0, 2.0, 3.0])
        .expect_err("3 件は REQ-4 の下限（5 件）未満のため拒否されるはず");
    assert!(matches!(
        err,
        guardrail::bench_gate::BenchGateError::InvalidSignal(_)
    ));
}
