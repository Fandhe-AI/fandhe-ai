//! REQ-8 段階的下限表の自動合否判定・統合テスト（TASK-8.2a・イシュー #152）。
//!
//! 受け入れ条件「計測結果に対し合否が自動判定される」を、fixture JSON
//! （`tests/fixtures/` の既存パターン踏襲。`report_regression.rs` 参照）から
//! `BenchReport::from_json` → `judge` → 合否確認という end-to-end 経路で検証する。
//! 実機依存の計測は含まないため、全テストが self-hosted CI で実行可能である
//! （`.claude/rules/coding-rust.md`）。公開 API（`bench_harness::{BackendDtype, Stage,
//! Verdict, judge, ...}`）のみを import し、`crates/guardrail`・`crates/self-repair`
//! 自体への実配線は行わない（`report_regression.rs` と同じスコープ判断。TASK-8.2 の
//! `guardrail` 側配線は本イシューのスコープ外）。

use bench_harness::{BackendDtype, BenchError, BenchReport, FloorJudgment, Stage, Verdict, judge};

/// TASK-8.1c 実装時に作成済みの golden fixture（cpu・`median_secs=1.095`）を
/// 「自作実装（own）」側の計測結果として流用する（既存パターン踏襲）。
const OWN_CPU_F32: &str = include_str!("fixtures/bench-report-v1.json");
/// 上記 own（`median_secs=1.095`）に対し PyTorch 比 5.48%（初期リリース下限 5% を上回る）
/// となるよう新規作成した「PyTorch 参照実装」側 fixture。
const PYTORCH_CPU_F32_PASS: &str = include_str!("fixtures/bench-report-pytorch-cpu-f32-pass.json");

fn pytorch_report_with_median(backend: &str, median_secs: f64) -> BenchReport {
    // own/pytorch とも q1<=median<=q3・samples_secs.len()==iters==20 という
    // BenchReport::validate（TASK-8.1c）の不変条件を満たす JSON をその場で組み立てる。
    let json = format!(
        r#"{{
            "schema_version": "1", "name": "pytorch_ref", "backend": "{backend}",
            "warmup": 20, "iters": 20,
            "median_secs": {median_secs}, "q1_secs": {q1}, "q3_secs": {q3},
            "samples_secs": [{s},{s},{s},{s},{s},{s},{s},{s},{s},{s},
                              {s},{s},{s},{s},{s},{s},{s},{s},{s},{s}]
        }}"#,
        q1 = median_secs * 0.95,
        q3 = median_secs * 1.05,
        s = median_secs,
    );
    BenchReport::from_json(&json).expect("組み立てた JSON は BenchReport::validate を満たすはず")
}

fn own_report(backend: &str, median_secs: f64) -> BenchReport {
    pytorch_report_with_median(backend, median_secs)
}

#[test]
fn cpu_f32_initial_release_passes_from_fixture_json() {
    // fixture JSON → BenchReport::from_json → judge の end-to-end 経路。
    let own = BenchReport::from_json(OWN_CPU_F32).expect("golden fixture は解析できるはず");
    let pytorch =
        BenchReport::from_json(PYTORCH_CPU_F32_PASS).expect("pytorch fixture は解析できるはず");

    let j = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease)
        .expect("有効な入力のため合否判定は成功するはず");

    assert_eq!(j.verdict, Verdict::Pass);
    assert_eq!(j.floor_percent, Some(5.0));
    assert_eq!(j.floor_provisional, Some(false));
    // 実測比率が Q1/Q3 由来のレンジに収まっていることを確認する
    // （REQ-8「単一のパーセンタイル値のみでの合否判定は行わない」の裏付け）。
    assert!(j.ratio_q1_percent <= j.measured_ratio_percent);
    assert!(j.measured_ratio_percent <= j.ratio_q3_percent);
}

#[test]
fn cpu_f32_initial_release_fails_when_below_floor() {
    let own = own_report("cpu", 1.0);
    // ratio = 0.04/1.0*100 = 4.0% < 初期リリース下限 5%。
    let pytorch = pytorch_report_with_median("cpu", 0.04);

    let j = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease).unwrap();
    assert_eq!(j.verdict, Verdict::Fail);
    assert_eq!(j.floor_percent, Some(5.0));
}

#[test]
fn cuda_f32_optimized_confirmed_floor_is_not_flagged_provisional() {
    let own = own_report("cuda", 1.0);
    // ratio = 0.5/1.0*100 = 50% >= 最適化後下限 25%（#393 で確定）。
    let pytorch = pytorch_report_with_median("cuda", 0.5);

    let j = judge(&own, &pytorch, BackendDtype::CudaF32, Stage::Optimized).unwrap();
    assert_eq!(j.verdict, Verdict::Pass);
    assert_eq!(j.floor_percent, Some(25.0));
    // CUDA 最適化後下限はイシュー #393（PR #444 実測・2026-08-10 ユーザー承認）で
    // 40%（暫定）→ 25%（確定）へ再確定済み。承認記録の限定条件（候補算出経路
    // `wmma_tf32` が #389 §5.3 の数値一致 parity 恒常 fail 対象と一致・#186 解決後の
    // 再確認）は `docs/perf/performance-floor-decision.md` §9 で追跡し、`provisional`
    // フラグでは表現しない（`.claude/rules/coding-rust.md`）。
    assert_eq!(j.floor_provisional, Some(false));
}

#[test]
fn cuda_f16_initial_release_records_ratio_without_verdict() {
    // CUDA f16 初期リリースは下限を設定しない行（REQ-8 脚注）。
    // 実測比率 1.9% 相当（tensor core 未使用）でも Fail にはならず NotApplicable として記録される。
    let own = own_report("cuda", 1.0);
    let pytorch = pytorch_report_with_median("cuda", 0.019);

    let j = judge(&own, &pytorch, BackendDtype::CudaF16, Stage::InitialRelease).unwrap();
    assert_eq!(j.verdict, Verdict::NotApplicable);
    assert_eq!(j.floor_percent, None);
    assert!((j.measured_ratio_percent - 1.9).abs() < 1e-9);
}

#[test]
fn metal_f32_both_stages_use_distinct_floors() {
    let own = own_report("metal", 1.0);
    // ratio = 0.25/1.0*100 = 25%。初期リリース下限 20% は満たすが、最適化後下限 30% は満たさない
    // （同一実測値でも段階が異なれば合否が変わることを確認する）。
    let pytorch = pytorch_report_with_median("metal", 0.25);

    let initial = judge(
        &own,
        &pytorch,
        BackendDtype::MetalF32,
        Stage::InitialRelease,
    )
    .unwrap();
    assert_eq!(initial.verdict, Verdict::Pass);
    assert_eq!(initial.floor_percent, Some(20.0));

    let optimized = judge(&own, &pytorch, BackendDtype::MetalF32, Stage::Optimized).unwrap();
    assert_eq!(optimized.verdict, Verdict::Fail);
    assert_eq!(optimized.floor_percent, Some(30.0));
}

#[test]
fn backend_mismatch_between_report_and_backend_dtype_is_rejected() {
    // own/pytorch の backend が "cuda" なのに BackendDtype::MetalF32（期待値 "metal"）を渡した場合、
    // Pass に倒さず必ずエラーになることを確認する（未知の組合せの fail-closed 判定）。
    let own = own_report("cuda", 1.0);
    let pytorch = pytorch_report_with_median("cuda", 0.5);

    let err = judge(
        &own,
        &pytorch,
        BackendDtype::MetalF32,
        Stage::InitialRelease,
    )
    .expect_err("backend 不一致は拒否されるはず");
    assert!(matches!(err, BenchError::ProtocolViolation(_)));
}

#[test]
fn invalid_bench_report_is_rejected_before_ratio_computation() {
    // BenchReport::validate（TASK-8.1c）を満たさない入力（warmup 下限未満）は、
    // 比率計算に進む前に own.validate() の時点で拒否される。
    let invalid_json = r#"{
        "schema_version": "1", "name": "n", "backend": "cpu",
        "warmup": 19, "iters": 20,
        "median_secs": 1.0, "q1_secs": 0.9, "q3_secs": 1.1,
        "samples_secs": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    }"#;
    // BenchReport::from_json 自体が拒否するため、judge に渡す BenchReport を構築できない
    // （検証を経ない判定経路が存在しないことの直接的な裏付け）。
    let err = BenchReport::from_json(invalid_json)
        .expect_err("warmup 19（下限未満）は BenchReport::from_json の時点で拒否されるはず");
    assert!(matches!(err, BenchError::ProtocolViolation(_)));
}

#[test]
fn floor_judgment_json_round_trip_via_public_api() {
    // guardrail からの参照可能性（本モジュールドキュメント参照）を、
    // 公開 API（FloorJudgment::to_json/from_json）のみを使った消費側シミュレーションで検証する。
    let own = own_report("metal", 1.0);
    let pytorch = pytorch_report_with_median("metal", 0.25);
    let j = judge(
        &own,
        &pytorch,
        BackendDtype::MetalF32,
        Stage::InitialRelease,
    )
    .unwrap();

    let json = j
        .to_json()
        .expect("有効な判定結果は JSON エンコードできるはず");
    let restored = FloorJudgment::from_json(&json).expect("往復後も検証を通るはず");
    assert_eq!(j, restored);
}
