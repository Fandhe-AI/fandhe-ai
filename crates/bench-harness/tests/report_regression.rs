//! `report::BenchReport` の構造化出力・プロトコル遵守回帰テスト（TASK-8.1c・イシュー #29）。
//!
//! 受け入れ条件「出力形式が guardrail／self-repair から参照可能で、テストが green」を、
//! 公開 API（`bench_harness::BenchReport` 等）のみを import した消費側シミュレーションで
//! 直接検証する（`crates/guardrail`・`crates/self-repair` 自体は編集しない。イシュー #29
//! 実装計画の判断。実連携配線は TASK-3.2・TASK-8.2 の後続スコープ）。
//! 実機依存の計測は含まないため、全テストが self-hosted CI で実行可能である
//! （`.claude/rules/coding-rust.md`）。

use bench_harness::{BenchError, BenchReport, MeasurementConfig, median_q1_q3, run};

/// 手書き golden fixture（決定的固定値）。スキーマの意図しない変更を回帰検知する。
const GOLDEN_FIXTURE: &str = include_str!("fixtures/bench-report-v1.json");

fn expect_protocol_violation(result: Result<BenchReport, BenchError>, context: &str) {
    let err = result.expect_err(&format!("{context}: プロトコル違反として拒否されるはず"));
    assert!(
        matches!(err, BenchError::ProtocolViolation(_)),
        "{context}: BenchError::ProtocolViolation を期待したが {err:?} だった"
    );
}

#[test]
fn round_trip_is_deterministic() {
    // protocol::run の実測結果 → BenchReport → to_json → from_json で元と完全一致することを
    // 検証する（serde_json の f64 shortest-representation により round-trip はビット等価）。
    let config = MeasurementConfig::new(20, 20).unwrap();
    let measurement = run(&config, || {
        let mut acc: u64 = 0;
        for i in 0..1_000u64 {
            acc = std::hint::black_box(acc.wrapping_add(std::hint::black_box(i)));
        }
        std::hint::black_box(acc);
    })
    .expect("軽量ワークロードの計測は成功するはず");

    let report = BenchReport::from_measurement("round_trip_dummy", "cpu", &measurement)
        .expect("プロトコル遵守済み Measurement からの構築は成功するはず");

    let json = report.to_json().expect("to_json は成功するはず");
    let decoded = BenchReport::from_json(&json).expect("from_json は成功するはず");

    assert_eq!(
        report, decoded,
        "round-trip 後に BenchReport が完全一致するはず"
    );
}

#[test]
fn golden_fixture_parses_with_expected_values() {
    // スキーマの意図しない変更（フィールド追加・削除・型変更）を回帰検知する。
    let report = BenchReport::from_json(GOLDEN_FIXTURE).expect("golden fixture は解析できるはず");

    assert_eq!(report.schema_version, bench_harness::SCHEMA_VERSION);
    assert_eq!(report.name, "gemm_f32_4096");
    assert_eq!(report.backend, "cpu");
    assert_eq!(report.warmup, 20);
    assert_eq!(report.iters, 20);
    assert_eq!(report.median_secs, 1.095);
    assert_eq!(report.q1_secs, 1.045);
    assert_eq!(report.q3_secs, 1.145);
    assert_eq!(report.samples_secs.len(), 20);
}

#[test]
fn rejects_warmup_below_minimum() {
    let json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 19, "iters": 20,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    expect_protocol_violation(BenchReport::from_json(json), "warmup 19（下限未満）");
}

#[test]
fn rejects_iters_below_minimum() {
    let json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 20, "iters": 19,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    expect_protocol_violation(BenchReport::from_json(json), "iters 19（下限未満）");
}

#[test]
fn rejects_samples_len_mismatch_with_iters() {
    // iters=20 と宣言しつつ samples_secs が 19 要素のみの改竄・破損データを拒否する。
    let json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 20, "iters": 20,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    expect_protocol_violation(
        BenchReport::from_json(json),
        "samples_secs 要素数（19）が iters（20）と不一致",
    );
}

#[test]
fn rejects_samples_containing_null() {
    // 非有限値（NaN・±inf）は serde_json 上 null としてシリアライズされる（to_json 側で
    // 事前拒否する対象）。デシリアライズ経路でも null 混入データを受理しないことを確認する。
    let json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 20, "iters": 20,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, null]
    }"#;
    let err = BenchReport::from_json(json)
        .expect_err("samples_secs に null を含む JSON は拒否されるはず");
    assert!(matches!(err, BenchError::ProtocolViolation(_)));
}

#[test]
fn rejects_samples_containing_negative_value() {
    let json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 20, "iters": 20,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    expect_protocol_violation(BenchReport::from_json(json), "samples_secs に負値を含む");
}

#[test]
fn rejects_unknown_schema_version() {
    let json = r#"{
        "schema_version": "2", "name": "n", "backend": "cpu",
        "warmup": 20, "iters": 20,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    expect_protocol_violation(
        BenchReport::from_json(json),
        "schema_version 2（未知バージョン）",
    );
}

#[test]
fn rejects_q1_greater_than_median() {
    let json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 20, "iters": 20,
        "median_secs": 1.0, "q1_secs": 1.5, "q3_secs": 2.0,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    expect_protocol_violation(BenchReport::from_json(json), "q1_secs > median_secs");
}

#[test]
fn to_json_rejects_nan_before_serialization() {
    // serde_json は非有限 f64 を null として黙って出力するため（silent corruption）、
    // to_json はシリアライズ前に validate() で NaN を事前拒否する（report.rs ドキュメント参照）。
    let config = MeasurementConfig::new(20, 20).unwrap();
    let measurement = run(&config, || {}).expect("軽量ワークロードの計測は成功するはず");
    let mut report = BenchReport::from_measurement("nan_case", "cpu", &measurement).unwrap();
    report.median_secs = f64::NAN;

    let err = report
        .to_json()
        .expect_err("NaN を含む BenchReport の to_json は拒否されるはず");
    assert!(matches!(err, BenchError::ProtocolViolation(_)));
}

#[test]
fn guardrail_can_derive_median_from_public_api_only() {
    // 受け入れ条件「出力形式が guardrail／self-repair から参照可能」の直接検証。
    // bench_harness::{BenchReport, median_q1_q3} のみを import し、JSON から
    // samples_secs を取り出して中央値（guardrail の bench_median_pct 算出の基礎）を
    // 独立に再導出できることを確認する。
    let config = MeasurementConfig::new(20, 20).unwrap();
    let measurement = run(&config, || {
        let mut acc: u64 = 0;
        for i in 0..1_000u64 {
            acc = std::hint::black_box(acc.wrapping_add(std::hint::black_box(i)));
        }
        std::hint::black_box(acc);
    })
    .expect("軽量ワークロードの計測は成功するはず");

    let report = BenchReport::from_measurement("guardrail_consumer_sim", "cpu", &measurement)
        .expect("プロトコル遵守済み Measurement からの構築は成功するはず");
    let json = report.to_json().expect("to_json は成功するはず");

    // guardrail 側は bench-harness を lib 依存する想定（docs/guardrail-self-repair-cli.md 1.4 節）
    // のため、ここでは公開 API のみを使って消費側の挙動をシミュレートする。
    let decoded = BenchReport::from_json(&json).expect("from_json は成功するはず");
    let quartiles = median_q1_q3(&decoded.samples_secs)
        .expect("decoded.samples_secs は validate 済みのため非空・非 NaN");

    // decoded.samples_secs は protocol::run が同じ median_q1_q3 アルゴリズムで
    // median_secs を算出した際の元データそのものであるため、再導出した中央値は完全一致する。
    assert_eq!(quartiles.median, decoded.median_secs);
}
