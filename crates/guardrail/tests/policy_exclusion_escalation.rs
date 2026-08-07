//! イシュー #124（TASK-5.2c）受け入れ条件の機械検証: ポリシー除外リスト
//! （REQ-5）が match した場合に**必ず**無条件エスカレーションとなることを、
//! 「評価（`ExclusionEvaluation::evaluate`）→ `effective_rule_ids()` →
//! `DecisionInput::new` → `decide`」の実配線で end-to-end に固定する。
//!
//! `blindspot_g2_regression.rs`／`blindspot_g5_regression.rs` は実 dataset
//! シグナル・`meta.toml` の `expected_exclusion_rule_ids` を直接
//! `DecisionInput::new` へ注入する評価粒度（判定ロジック `decide` 側の
//! 検証）に対し、本ファイルは一時 git リポジトリで実際に `git diff` を
//! 発生させ、除外リスト評価器自体の配線（本イシューの核心）を検証する
//! （粒度を分けて重複を避ける。計画 3.4 節）。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use guardrail::config::{self, PresetName};
use guardrail::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals, Verdict, decide};
use guardrail::policy_exclusion::{EvaluationContext, ExclusionEvaluation, builtin_defaults};

fn run(cwd: &Path, args: &[&str]) {
    let mut cmd = Command::new("git");
    cmd.args(args).current_dir(cwd);
    for (key, _) in std::env::vars() {
        if key.starts_with("GIT_") {
            cmd.env_remove(key);
        }
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("git {args:?} 起動に失敗 (cwd={cwd:?}): {e}"));
    assert!(
        output.status.success(),
        "git {args:?} が失敗 (cwd={cwd:?}): {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn commit_all(cwd: &Path, message: &str) {
    run(cwd, &["add", "-A"]);
    run(
        cwd,
        &[
            "-c",
            "user.email=guardrail-test@example.invalid",
            "-c",
            "user.name=guardrail-test",
            "commit",
            "-q",
            "-m",
            message,
        ],
    );
}

fn init_repo(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "guardrail-policy-exclusion-escalation-{name}-{}",
        std::process::id()
    ));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{dir:?} の削除に失敗: {e}"));
    }
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{dir:?} の作成に失敗: {e}"));
    run(&dir, &["init", "-q"]);
    dir
}

fn all_passed_gates() -> GateSignals {
    GateSignals {
        build: GateSignal::Passed,
        test: GateSignal::Passed,
        clippy: GateSignal::Passed,
    }
}

/// 評価 → 配線 → 判定を 1 関数にまとめ、各ケースから呼び出す。
fn evaluate_and_decide(repo_root: &Path, baseline: &str) -> Verdict {
    let thresholds = config::Thresholds::builtin(PresetName::Default);
    let rules = builtin_defaults().unwrap().rules;
    let ctx = EvaluationContext::from_repo(repo_root, baseline).unwrap();
    let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
    let exclusion_rule_ids = evaluation.effective_rule_ids();

    let input = DecisionInput::new(
        &thresholds,
        10,
        all_passed_gates(),
        false,
        false,
        BenchSignal::NotRun,
        exclusion_rule_ids,
    )
    .unwrap();
    decide(&input).unwrap().verdict()
}

/// ケース 1（G2 系）: モデルアーキテクチャファイルの変更のみ・全シグナル
/// green → `arch-hyperparameter-change` match → `Escalate`。
#[test]
fn architecture_change_alone_escalates() {
    let dir = init_repo("g2");
    fs::create_dir_all(dir.join("crates/tensor-core/src")).unwrap();
    fs::write(
        dir.join("crates/tensor-core/src/model.rs"),
        "pub const HIDDEN_DIM: usize = 8;\n",
    )
    .unwrap();
    commit_all(&dir, "baseline");

    fs::write(
        dir.join("crates/tensor-core/src/model.rs"),
        "pub const HIDDEN_DIM: usize = 10;\n",
    )
    .unwrap();

    assert_eq!(evaluate_and_decide(&dir, "HEAD"), Verdict::Escalate);

    fs::remove_dir_all(&dir).ok();
}

/// ケース 2（G5 系）: テスト許容誤差の単独緩和（`1e-6 → 1e-2`）→
/// `test-tolerance-loosening` match → `Escalate`。
#[test]
fn test_tolerance_loosening_alone_escalates() {
    let dir = init_repo("g5");
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("src/lib.rs"),
        "pub fn leaky_relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }\n\n\
         #[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn known_value() {\n        \
         assert!((leaky_relu(-1.0) - (-0.01)).abs() < 1e-6);\n    }\n}\n",
    )
    .unwrap();
    commit_all(&dir, "baseline");

    fs::write(
        dir.join("src/lib.rs"),
        "pub fn leaky_relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }\n\n\
         #[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn known_value() {\n        \
         assert!((leaky_relu(-1.0) - (-0.01)).abs() < 1e-2);\n    }\n}\n",
    )
    .unwrap();

    assert_eq!(evaluate_and_decide(&dir, "HEAD"), Verdict::Escalate);

    fs::remove_dir_all(&dir).ok();
}

/// ケース 3（依存変更系）: `Cargo.toml` の変更 → `dependency-change` match
/// → `Escalate`。
#[test]
fn dependency_change_alone_escalates() {
    let dir = init_repo("dep");
    fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    commit_all(&dir, "baseline");

    fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"x\"\n\n[dependencies]\nserde = \"1\"\n",
    )
    .unwrap();

    assert_eq!(evaluate_and_decide(&dir, "HEAD"), Verdict::Escalate);

    fs::remove_dir_all(&dir).ok();
}

/// ケース 4（fail-closed）: `unevaluated_rule_ids` が非空の合成
/// `ExclusionEvaluation` は `effective_rule_ids()` を経由すると必ず
/// `Escalate` へ倒れる（`MatchRule` に将来 variant が追加され、実装者が
/// 誤って「未評価」のまま返しても安全側で扱われる契約の直接確認）。
#[test]
fn unevaluated_rule_ids_alone_escalate_via_effective_rule_ids() {
    let thresholds = config::Thresholds::builtin(PresetName::Default);
    let evaluation = ExclusionEvaluation {
        matched_rule_ids: Vec::new(),
        unevaluated_rule_ids: vec!["future-unevaluated-rule".to_string()],
    };
    let input = DecisionInput::new(
        &thresholds,
        10,
        all_passed_gates(),
        false,
        false,
        BenchSignal::NotRun,
        evaluation.effective_rule_ids(),
    )
    .unwrap();
    let decision = decide(&input).unwrap();
    assert_eq!(decision.verdict(), Verdict::Escalate);
    assert_eq!(decision.reason_conditions(), vec!["policy_exclusion_match"]);
}

/// ケース 5（後方互換）: match なし・全シグナル green → `AutoApply`
/// （除外リストは安全側にしか作用しない。REQ-5 不変条件の配線経路確認）。
#[test]
fn no_match_and_clean_signals_yield_auto_apply() {
    let dir = init_repo("clean");
    fs::write(dir.join("README.md"), "baseline\n").unwrap();
    commit_all(&dir, "baseline");
    fs::write(dir.join("README.md"), "updated\n").unwrap();

    assert_eq!(evaluate_and_decide(&dir, "HEAD"), Verdict::AutoApply);

    fs::remove_dir_all(&dir).ok();
}

/// ケース 6（却下優先）: ゲート失敗 × match 同時成立 → `Reject` だが
/// `exclusion_rule_ids()` に match 記録が残る（取り込み判断の根拠追跡可能性。
/// security.md A09）。
#[test]
fn gate_failure_rejects_even_when_match_recorded() {
    let thresholds = config::Thresholds::builtin(PresetName::Default);
    let evaluation = ExclusionEvaluation {
        matched_rule_ids: vec!["arch-hyperparameter-change".to_string()],
        unevaluated_rule_ids: Vec::new(),
    };
    let gates = GateSignals {
        build: GateSignal::Failed,
        test: GateSignal::Skipped,
        clippy: GateSignal::Skipped,
    };
    let input = DecisionInput::new(
        &thresholds,
        10,
        gates,
        false,
        false,
        BenchSignal::NotRun,
        evaluation.effective_rule_ids(),
    )
    .unwrap();
    let decision = decide(&input).unwrap();
    assert_eq!(decision.verdict(), Verdict::Reject);
    assert_eq!(
        decision.exclusion_rule_ids(),
        &["arch-hyperparameter-change".to_string()]
    );
}
