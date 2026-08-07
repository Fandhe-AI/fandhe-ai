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
    let config = builtin_defaults().unwrap();
    let changed = vec![
        "README.md".to_string(),
        "crates/guardrail/src/lib.rs".to_string(),
        "docs/license-matrix.md".to_string(),
    ];
    let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
    assert!(evaluation.matched_rule_ids.is_empty());
}

#[test]
fn builtin_rule_category_is_architecture_change() {
    let config = builtin_defaults().unwrap();
    assert_eq!(config.rules[0].category, Category::ArchitectureChange);
}
