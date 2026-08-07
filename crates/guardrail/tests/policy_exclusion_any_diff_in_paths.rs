//! `policy_exclusion::any_diff_in_paths` の受け入れ条件テスト（イシュー #122・
//! TASK-5.2a）。
//!
//! REQ-5 受け入れ基準 1（`docs/spec/04-requirements.md`）「対象パス差分で
//! ルールが match すること」を、組み込み既定値 `arch-hyperparameter-change`
//! （`**/src/model*.rs`・`**/src/nn/**`・`**/src/*model*/**` の 3 パターン）の
//! 各代表パスに対して確認する。単体テスト（`src/policy_exclusion/*.rs` 内
//! `#[cfg(test)]`）と重複しても、クレート外部 API（`guardrail::policy_exclusion`）
//! からの到達性を独立に保証する目的で維持する（`.claude/rules/coding-rust.md`
//! 「受け入れ基準に対応するテストを同一 PR に含める」）。

use guardrail::policy_exclusion::{Category, ExclusionEvaluation, builtin_defaults};

#[test]
fn arch_hyperparameter_change_matches_model_star_rs_pattern() {
    let config = builtin_defaults().unwrap();
    let changed = vec!["crates/tensor-core/src/model_mlp.rs".to_string()];
    let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
    assert_eq!(
        evaluation.matched_rule_ids,
        vec!["arch-hyperparameter-change"]
    );
}

#[test]
fn arch_hyperparameter_change_matches_nn_subtree_pattern() {
    let config = builtin_defaults().unwrap();
    let changed = vec!["crates/tensor-core/src/nn/layer.rs".to_string()];
    let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
    assert_eq!(
        evaluation.matched_rule_ids,
        vec!["arch-hyperparameter-change"]
    );
}

#[test]
fn arch_hyperparameter_change_matches_star_model_star_directory_pattern() {
    let config = builtin_defaults().unwrap();
    let changed = vec!["crates/tensor-core/src/mlp_model_v2/config.rs".to_string()];
    let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
    assert_eq!(
        evaluation.matched_rule_ids,
        vec!["arch-hyperparameter-change"]
    );
}

#[test]
fn non_target_paths_do_not_match() {
    // `matched_rule_ids` が空であることのみを安全（未マッチ確定）と誤読
    // しないよう、`unevaluated_rule_ids`（`test-tolerance-loosening`。評価
    // ロジックは #123 が引き継ぐ）が必ず non-empty で返ることも併せて固定する
    // （イシュー #122 レビュー指摘対応。`builtin_defaults()` からこのルールが
    // 再度欠落した場合に本テストが検知する）。
    let config = builtin_defaults().unwrap();
    let changed = vec![
        "README.md".to_string(),
        "crates/guardrail/src/lib.rs".to_string(),
        "docs/license-matrix.md".to_string(),
    ];
    let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
    assert!(evaluation.matched_rule_ids.is_empty());
    assert_eq!(
        evaluation.unevaluated_rule_ids,
        vec!["test-tolerance-loosening"]
    );
}

#[test]
fn builtin_rule_category_is_architecture_change() {
    let config = builtin_defaults().unwrap();
    assert_eq!(config.rules[0].category, Category::ArchitectureChange);
}

#[test]
fn builtin_rules_include_test_tolerance_category() {
    // `builtin_defaults()` は `arch-hyperparameter-change` のみでなく
    // `test-tolerance-loosening`（G5 ブラインドスポット対策）も必ず含む
    // ことをクレート外部 API 経由で固定する（イシュー #122 レビュー指摘対応）。
    let config = builtin_defaults().unwrap();
    assert!(
        config
            .rules
            .iter()
            .any(|rule| rule.category == Category::TestToleranceLoosening)
    );
}
