//! `policy_exclusion::any_diff_in_paths` の受け入れ条件テスト（イシュー #122・
//! TASK-5.2a。#124・TASK-5.2c で `ExclusionEvaluation::evaluate` の
//! `EvaluationContext` 化に追随）。
//!
//! REQ-5 受け入れ基準 1（`docs/spec/04-requirements.md`）「対象パス差分で
//! ルールが match すること」を、組み込み既定値 `arch-hyperparameter-change`
//! （`**/src/model*.rs`・`**/src/nn/**`・`**/src/*model*/**` の 3 パターン）・
//! `dependency-change`（#124 で追加）の各代表パスに対して確認する。単体テスト
//! （`src/policy_exclusion/*.rs` 内 `#[cfg(test)]`）と重複しても、クレート
//! 外部 API（`guardrail::policy_exclusion`）からの到達性を独立に保証する
//! 目的で維持する（`.claude/rules/coding-rust.md`「受け入れ基準に対応する
//! テストを同一 PR に含める」）。
//!
//! `EvaluationContext` は `git` を経由せずフィールドを直接構築する
//! （`policy_exclusion::mod.rs` の単体テストと同じ手法）。`AnyDiffInPaths`
//! 系ルールのみに絞り込むのは、`TestAssertionRelaxationWithoutProdChange`
//! 系ルールは `git diff` を要求し、ダミー文脈（`repo_root=""`）では
//! `Err` になってしまうため（本ファイルは `any_diff_in_paths` 方式の到達性
//! のみを検証対象とする）。

use std::path::PathBuf;

use guardrail::policy_exclusion::{
    Category, EvaluationContext, ExclusionEvaluation, ExclusionRule, MatchRule, builtin_defaults,
};

fn changed_files_only_context(changed_files: Vec<String>) -> EvaluationContext {
    EvaluationContext {
        repo_root: PathBuf::new(),
        baseline: String::new(),
        changed_files,
    }
}

/// `builtin_defaults()` から `AnyDiffInPaths` 系ルール（`arch-hyperparameter-change`・
/// `dependency-change`）のみを取り出す。
fn any_diff_in_paths_rules() -> Vec<ExclusionRule> {
    builtin_defaults()
        .unwrap()
        .rules
        .into_iter()
        .filter(|r| matches!(r.match_rule, MatchRule::AnyDiffInPaths { .. }))
        .collect()
}

#[test]
fn arch_hyperparameter_change_matches_model_star_rs_pattern() {
    let rules = any_diff_in_paths_rules();
    let ctx = changed_files_only_context(vec!["crates/tensor-core/src/model_mlp.rs".to_string()]);
    let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
    assert_eq!(
        evaluation.matched_rule_ids,
        vec!["arch-hyperparameter-change"]
    );
}

#[test]
fn arch_hyperparameter_change_matches_nn_subtree_pattern() {
    let rules = any_diff_in_paths_rules();
    let ctx = changed_files_only_context(vec!["crates/tensor-core/src/nn/layer.rs".to_string()]);
    let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
    assert_eq!(
        evaluation.matched_rule_ids,
        vec!["arch-hyperparameter-change"]
    );
}

#[test]
fn arch_hyperparameter_change_matches_star_model_star_directory_pattern() {
    let rules = any_diff_in_paths_rules();
    let ctx = changed_files_only_context(vec![
        "crates/tensor-core/src/mlp_model_v2/config.rs".to_string(),
    ]);
    let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
    assert_eq!(
        evaluation.matched_rule_ids,
        vec!["arch-hyperparameter-change"]
    );
}

#[test]
fn dependency_change_matches_cargo_toml_pattern() {
    // #124・TASK-5.2c で追加した `dependency-change` カテゴリ（TASK-5.1b・
    // #120 で人間承認済み）の到達性を確認する。
    let rules = any_diff_in_paths_rules();
    let ctx = changed_files_only_context(vec!["crates/guardrail/Cargo.toml".to_string()]);
    let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
    assert_eq!(evaluation.matched_rule_ids, vec!["dependency-change"]);
}

#[test]
fn non_target_paths_do_not_match() {
    // `README.md`／`crates/guardrail/src/lib.rs` はいずれの `AnyDiffInPaths`
    // ルールの `paths` にも一致しない代表例（`docs/license-matrix.md` は
    // #124 で `dependency-change` の対象に追加されたため、ここでは対象外に
    // 含めない）。
    let rules = any_diff_in_paths_rules();
    let ctx = changed_files_only_context(vec![
        "README.md".to_string(),
        "crates/guardrail/src/lib.rs".to_string(),
    ]);
    let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
    assert!(evaluation.matched_rule_ids.is_empty());
    assert!(evaluation.unevaluated_rule_ids.is_empty());
}

#[test]
fn builtin_rule_category_is_architecture_change() {
    let config = builtin_defaults().unwrap();
    assert_eq!(config.rules[0].category, Category::ArchitectureChange);
}

#[test]
fn builtin_rules_include_test_tolerance_category() {
    // `builtin_defaults()` は `arch-hyperparameter-change` のみでなく
    // `test-tolerance-loosening`（G5 ブラインドスポット対策）・
    // `dependency-change`（#124 で追加）も必ず含むことをクレート外部 API
    // 経由で固定する（イシュー #122 レビュー指摘対応の踏襲）。
    let config = builtin_defaults().unwrap();
    assert!(
        config
            .rules
            .iter()
            .any(|rule| rule.category == Category::TestToleranceLoosening)
    );
    assert!(
        config
            .rules
            .iter()
            .any(|rule| rule.category == Category::DependencyChange)
    );
}
