//! 3 分岐判定の代表ケース統合テスト（TASK-4.1c・イシュー #106 受け入れ条件
//! 「3 分岐が代表ケースで正しく出力される」）。
//!
//! `guardrail check` CLI バイナリ（`--signals` 注入経由の
//! `CARGO_BIN_EXE_guardrail` テスト。`docs/guardrail-self-repair-cli.md`
//! §1.2）は #104（TASK-4.1a・CLI 骨格）が未着手のため本 PR 時点では存在しない。
//! 計画書 §5 が許容する代替経路（「未整備部分があれば unit テスト側でカバーし
//! 差分を PR に明記」）に従い、本ファイルは公開ライブラリ API
//! （[`guardrail::decision`]・[`guardrail::exit_code`]・[`guardrail::report`]）を
//! クレート境界を跨いで結合するテストとし、CLI バイナリ結線後は #104 が
//! `CARGO_BIN_EXE_guardrail` 経由のテストへ差し替える想定とする。
//!
//! 「シグナル収集 → `decide()` → レポート出力 → 終了コード」という TASK-4.1c
//! が接続する経路（計画書 §3 `main.rs`/`src/lib.rs` 行）を、CLI 未着手下では
//! ライブラリ API 呼び出し列として検証する。
//!
//! 閾値は #105（TASK-4.1b）が移植した [`guardrail::config::Thresholds`]
//! （値域検証済みの検証済み型）を経由して構築する（`decision` モジュールが
//! 受け取る契約 API。`Thresholds::from_raw` を必ず経由し不変条件を型で強制する）。

use guardrail::config::{PresetName, Thresholds, ThresholdsRaw};
use guardrail::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals, decide};
use guardrail::exit_code::GuardrailExitCode;
use guardrail::report::VerdictSection;

fn thresholds() -> Thresholds {
    Thresholds::from_raw(
        PresetName::Default,
        ThresholdsRaw {
            lines_max: 200,
            bench_max_pct: 5.0,
            bench_runs: 5,
        },
    )
    .expect("固定値の検証に失敗")
}

fn all_passed_gates() -> GateSignals {
    GateSignals {
        build: GateSignal::Passed,
        test: GateSignal::Passed,
        clippy: GateSignal::Passed,
    }
}

/// 代表ケース 1: 自動適用。5 条件すべて充足 → `verdict = "auto_apply"`・
/// `reasons` 空・終了コード `0`。
#[test]
fn auto_apply_case_end_to_end() {
    let t = thresholds();
    let input = DecisionInput::new(
        &t,
        10,
        all_passed_gates(),
        false,
        false,
        BenchSignal::Measured { median_pct: -1.0 },
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "auto_apply");
    assert!(decision.reasons().is_empty());

    let section = VerdictSection::from_decision(&decision);
    let json = serde_json::to_value(&section).expect("シリアライズに失敗");
    assert_eq!(json["verdict"], "auto_apply");
    assert!(json["reason_conditions"].as_array().unwrap().is_empty());

    let exit_code = GuardrailExitCode::from_verdict(decision.verdict());
    assert_eq!(exit_code.as_u8(), 0);
}

/// 代表ケース 2: エスカレーション（複数条件の単独逸脱を代表して 4 通り検証）。
/// 各逸脱単独で `verdict = "escalate"`・終了コード `10`。
#[test]
fn escalate_case_lines_changed_exceeds_threshold() {
    let t = thresholds();
    let input = DecisionInput::new(
        &t,
        t.lines_max() + 1,
        all_passed_gates(),
        false,
        false,
        BenchSignal::NotRun,
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "escalate");
    let section = VerdictSection::from_decision(&decision);
    assert_eq!(section.reason_conditions, vec!["lines_max_exceeded"]);
    assert_eq!(
        GuardrailExitCode::from_verdict(decision.verdict()).as_u8(),
        10
    );
}

#[test]
fn escalate_case_public_api_broken() {
    let t = thresholds();
    let input = DecisionInput::new(
        &t,
        10,
        all_passed_gates(),
        true,
        false,
        BenchSignal::NotRun,
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "escalate");
    assert_eq!(
        GuardrailExitCode::from_verdict(decision.verdict()).as_u8(),
        10
    );
}

#[test]
fn escalate_case_gate_skipped() {
    let gates = GateSignals {
        build: GateSignal::Passed,
        test: GateSignal::Skipped,
        clippy: GateSignal::Passed,
    };
    let t = thresholds();
    let input = DecisionInput::new(&t, 10, gates, false, false, BenchSignal::NotRun, Vec::new())
        .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "escalate");
    assert_eq!(
        GuardrailExitCode::from_verdict(decision.verdict()).as_u8(),
        10
    );
}

/// 境界値ちょうど・ベンチ改善（負値）はエスカレーションしないこと。
#[test]
fn boundary_and_improvement_bench_do_not_escalate() {
    let t = thresholds();

    let at_boundary = DecisionInput::new(
        &t,
        10,
        all_passed_gates(),
        false,
        false,
        BenchSignal::Measured {
            median_pct: t.bench_max_pct(),
        },
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    assert_eq!(
        decide(&at_boundary)
            .expect("判定に失敗")
            .verdict()
            .as_machine_id(),
        "auto_apply"
    );

    let improvement = DecisionInput::new(
        &t,
        10,
        all_passed_gates(),
        false,
        false,
        BenchSignal::Measured { median_pct: -5.0 },
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    assert_eq!(
        decide(&improvement)
            .expect("判定に失敗")
            .verdict()
            .as_machine_id(),
        "auto_apply"
    );
}

/// 代表ケース 3: 却下。build/test/clippy のいずれかが失敗 → `verdict = "reject"`・
/// 終了コード `20`。他条件（行数超過・API 破壊）が同時発生しても却下が最優先。
#[test]
fn reject_case_gate_failure_takes_priority_over_escalation_conditions() {
    let t = thresholds();
    let gates = GateSignals {
        build: GateSignal::Failed,
        test: GateSignal::Skipped,
        clippy: GateSignal::Skipped,
    };
    let input = DecisionInput::new(
        &t,
        t.lines_max() + 1000,
        gates,
        true,
        false,
        BenchSignal::NotRun,
        Vec::new(),
    )
    .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "reject");
    let section = VerdictSection::from_decision(&decision);
    assert_eq!(section.reason_conditions, vec!["gate_build_failed"]);
    assert_eq!(
        GuardrailExitCode::from_verdict(decision.verdict()).as_u8(),
        20
    );
}

/// 除外リスト受け口: `exclusion_rule_ids` 非空なら機械条件によらず
/// `Escalate`（却下時は却下維持＋record）。空なら判定不変。
#[test]
fn exclusion_rule_ids_force_escalate_when_signals_are_otherwise_clean() {
    let t = thresholds();
    let input = DecisionInput::new(
        &t,
        10,
        all_passed_gates(),
        false,
        false,
        BenchSignal::NotRun,
        vec!["arch-hyperparameter-change".to_string()],
    )
    .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "escalate");
    assert_eq!(
        decision.exclusion_rule_ids(),
        &["arch-hyperparameter-change".to_string()]
    );
}

#[test]
fn exclusion_rule_ids_do_not_downgrade_reject() {
    let gates = GateSignals {
        build: GateSignal::Failed,
        test: GateSignal::Skipped,
        clippy: GateSignal::Skipped,
    };
    let t = thresholds();
    let input = DecisionInput::new(
        &t,
        10,
        gates,
        false,
        false,
        BenchSignal::NotRun,
        vec!["test-tolerance-loosening".to_string()],
    )
    .expect("矛盾なし入力の構築に失敗");
    let decision = decide(&input).expect("判定に失敗");

    assert_eq!(decision.verdict().as_machine_id(), "reject");
    assert_eq!(
        decision.exclusion_rule_ids(),
        &["test-tolerance-loosening".to_string()]
    );
}

/// fail-closed: `DecisionInput::new` が拒否する矛盾入力（内部エラー相当）は
/// `Verdict` を経由せず `Err` として返り、CLI 層（#104）はこれを終了コード
/// `1` にマップする契約（本 PR は `GuardrailError` 型までを担保する）。
#[test]
fn inconsistent_input_is_rejected_before_reaching_decide() {
    let gates = GateSignals {
        build: GateSignal::Failed,
        test: GateSignal::Skipped,
        clippy: GateSignal::Skipped,
    };
    let t = thresholds();
    let result = DecisionInput::new(
        &t,
        10,
        gates,
        false,
        false,
        BenchSignal::Measured { median_pct: 1.0 },
        Vec::new(),
    );
    assert!(result.is_err());
}
