//! ポリシー除外リスト（REQ-5）: 機械判定（[`crate::decision`]）では検知できない
//! 変更カテゴリを、パス／変更内容パターンにマッチした事実のみで無条件人間承認へ
//! 回す保守的な仕組み。
//!
//! `docs/spec/04-requirements.md`（REQ-5・2026-08-05 v2 注記）が定める 2 層目
//! 不変条件「除外リストは安全側にしか作用しない（見逃し方向に緩まない）」を
//! 実装が壊さないことは、`ExclusionEvaluation` が「match したルール id を追加
//! するだけで、1 層目の [`crate::decision::Verdict`] を上書き・緩和しない」設計
//! で担保する。実際の verdict 合成は [`crate::decision::decide`] 側
//! （`DecisionInput::new` の `exclusion_rule_ids` 引数）が担い、本モジュールは
//! [`ExclusionEvaluation::effective_rule_ids`] でその引数を構築する（#124・
//! TASK-5.2c で統合済み）。
//!
//! # モジュール構成（イシュー #124・TASK-5.2c 時点）
//! - [`path_match`][]: 自作パス glob マッチャ（`glob` クレート非依存。#122 管轄）。
//! - [`any_diff_in_paths`](fn@any_diff_in_paths): `arch-hyperparameter-change`・`dependency-change`
//!   の match 評価本体（#122 管轄。`dependency-change` への適用は #124）。
//! - [`load`][]: `policy-exclusion.toml` 専用ミニ TOML ローダ（[`load::load_from_str`]。
//!   #124 管轄。`crate::toml_lite` を再利用しない判断理由は同モジュールのドキュメント参照）。
//!
//! # fail-closed 統合（#124 が完了させた設計。旧イシュー #122 レビュー指摘対応）
//! [`MatchRule`] の全 variant は [`ExclusionEvaluation::evaluate`] が必ず
//! 評価する（`test_assertion_relaxation_without_prod_change` の評価ロジックは
//! #123 が実装し、#124 で評価器に配線済み）ため、**現時点で `evaluate` が
//! 返す [`ExclusionEvaluation::unevaluated_rule_ids`] は常に空集合**である
//! （`evaluate` の実装に「未評価」へ push する分岐が存在しない。
//! `all_builtin_rules_evaluate_without_error_on_unrelated_change` 等の
//! テストが空集合であることを暗黙に前提とする）。
//!
//! それでもフィールド自体は残す: 将来 [`MatchRule`] に新 variant を追加する
//! 際、`evaluate` の `match` は網羅列挙（`_ =>` 禁止）のためコンパイル
//! エラーで実装者に選択を迫れる。その際もし実装者が評価ロジックを即座に
//! 書けず「未評価」のまま `unevaluated_rule_ids` へ回す判断をしても、
//! [`ExclusionEvaluation::effective_rule_ids`]（`matched_rule_ids` と
//! `unevaluated_rule_ids` の合併）を経由する限り無条件エスカレーション側に
//! 倒れる（判定不能を安全側で扱う契約を型で保つ。
//! `.claude/rules/security.md` A08）。あくまで**将来の受け皿**であり、
//! 本 #124 時点で `evaluate` 自体が未評価分岐を持つわけではない。
//!
//! # 変更ファイル一覧・`test_assertion_relaxation_without_prod_change` の呼び出し
//! [`EvaluationContext::from_repo`] が `crate::exclusion_match::changed_files_for_policy_exclusion`
//! （`Cargo.lock` を除外しない取得口。`dependency-change` ルール対応。
//! `exclusion_match` モジュール参照）を呼んで変更ファイル一覧を構築する。
//! `git` 実行エラーは [`crate::error::GuardrailError`] として
//! [`ExclusionEvaluation::evaluate`] から呼び出し元へそのまま伝播し、
//! `false`（match なし）へ丸めない（fail-closed。`exclusion_match` モジュール
//! と同一契約）。
//!
//! # スコープ境界（`.claude/rules/out-of-scope-tracking.md`）
//! - `main.rs` `run_check` の実シグナル計測経路への配線: #103（TASK-4.1）。
//!   `EvaluationContext::from_repo` → `ExclusionEvaluation::evaluate` →
//!   `effective_rule_ids()` → `DecisionInput::new` の経路を通すことが契約
//! - `self-repair` からの lib 呼び出し統合: TASK-3.1・#131
//! - `guardrail eval` への除外リスト適用: REQ-4 注記により適用しない設計
//!   （変更しない）

pub mod any_diff_in_paths;
pub mod load;
pub mod path_match;

pub use any_diff_in_paths::any_diff_in_paths;
pub use load::load_from_str;
pub use path_match::{PathPattern, PatternError};

use std::path::{Path, PathBuf};

use crate::error::GuardrailError;
use crate::exclusion_match;

/// 除外ルールが人間承認へ回す変更カテゴリ（v1 `policy-exclusion.toml` §4 由来）。
///
/// 値は `#[non_exhaustive]` とはしない（カテゴリの追加自体がプロダクト判断
/// を伴い、`policy-exclusion.toml`・`builtin_defaults()` の同時変更＋
/// ユーザー承認を要するため。`deps-policy.md`・`security.md` と同じ運用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// モデルアーキテクチャ・ハイパーパラメータ変更（PoC-3 G2 ブラインドスポット対策）。
    ArchitectureChange,
    /// テスト許容誤差の単独緩和（PoC-3 G5 ブラインドスポット対策。#123 管轄）。
    TestToleranceLoosening,
    /// 許容依存一覧の追加・更新、および依存管理ファイルへの変更（REQ-1 の受け皿。
    /// TASK-5.1b・#120 で人間承認済み。`.claude/rules/deps-policy.md` 「依存の
    /// 追加・更新は AI 自律メンテナンスの自動適用対象外」を機械判定側でも強制する）。
    DependencyChange,
}

/// 除外ルール match 時に取る対応。TASK-5.2 時点では「無条件人間承認」の
/// 1 種類のみを定義する（REQ-5 が要求する対応はこれのみ。`04-requirements.md`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    HumanApproval,
}

/// 除外ルールの match 方式（REQ-5「パス／変更内容パターンにマッチした事実のみで
/// 発火する」の 2 方式）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchRule {
    /// `changed_files` のいずれかが `paths` のいずれかに一致すれば match
    /// （評価は [`any_diff_in_paths::any_diff_in_paths`]）。
    AnyDiffInPaths { paths: Vec<PathPattern> },
    /// テスト許容誤差（assertion）が単独で緩和されているか判定する方式
    /// （評価は [`crate::exclusion_match::test_assertion_relaxation_without_prod_change`]。
    /// #123 が実装したロジックへ #124 が配線した）。
    ///
    /// `assertion_patterns` は `policy-exclusion.toml` §4.1 の
    /// `["assert!", "abs() <", "1e-[0-9]"]` に対応する（値の変更はユーザー
    /// 承認必須。`.claude/rules/security.md`）。
    TestAssertionRelaxationWithoutProdChange { assertion_patterns: Vec<String> },
}

/// 1 件の除外ルール定義（TOML の `[[exclusion]]` 1 要素に対応。
/// ロードは [`load::load_from_str`]）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusionRule {
    pub id: String,
    pub category: Category,
    pub action: Action,
    pub match_rule: MatchRule,
}

/// 除外ルール一式。`policy-exclusion.toml` のロード（[`load::load_from_str`]）
/// を経ずとも、[`builtin_defaults`] で組み込み既定値を直接得られる。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyExclusionConfig {
    pub rules: Vec<ExclusionRule>,
}

/// 組み込み既定値: `docs/policy-exclusion-design.md` §4.1・`policy-exclusion.toml`
/// が定める 3 カテゴリ（`arch-hyperparameter-change`・`test-tolerance-loosening`・
/// `dependency-change`）をそのまま採用する。値の変更はユーザー承認必須のため
/// この実装では一切変更しない（`.claude/rules/security.md`「ガードレール
/// 閾値・ポリシー除外リストの変更は必ず人間の承認を経る」）。
///
/// `policy-exclusion.toml` との完全一致は
/// `tests/policy_exclusion_toml_consistency.rs`（回帰テスト）で保証する
/// （乖離＝判定迂回経路。ファイルヘッダコメント参照）。
///
/// `.claude/rules/coding-rust.md`「本番経路で `unwrap()`/`expect()` を使わない」
/// に例外を設けず、パターン構文検証（[`PathPattern::compile`]）の失敗は
/// `Result` で呼び出し元へ返す（自明にコンパイル時定数だからと `.expect()` へ
/// 逃げない。呼び出し元は `self-repair` からの lib 直接呼び出しも含むため、
/// この関数自体をパニックしない契約に保つ）。
pub fn builtin_defaults() -> Result<PolicyExclusionConfig, PatternError> {
    let arch_paths = ["**/src/model*.rs", "**/src/nn/**", "**/src/*model*/**"]
        .iter()
        .map(|p| PathPattern::compile(p))
        .collect::<Result<Vec<_>, _>>()?;
    let dependency_paths = [
        "**/Cargo.toml",
        "**/Cargo.lock",
        "deny.toml",
        "docs/license-matrix.md",
        ".claude/rules/deps-policy.md",
    ]
    .iter()
    .map(|p| PathPattern::compile(p))
    .collect::<Result<Vec<_>, _>>()?;

    Ok(PolicyExclusionConfig {
        rules: vec![
            ExclusionRule {
                id: "arch-hyperparameter-change".to_string(),
                category: Category::ArchitectureChange,
                action: Action::HumanApproval,
                match_rule: MatchRule::AnyDiffInPaths { paths: arch_paths },
            },
            ExclusionRule {
                id: "test-tolerance-loosening".to_string(),
                category: Category::TestToleranceLoosening,
                action: Action::HumanApproval,
                // `policy-exclusion.toml` §4.1 のパターン列と同一値
                // （変更はユーザー承認必須）。評価は
                // `exclusion_match::test_assertion_relaxation_without_prod_change`
                // に委ねる（本 variant 自体は `paths` を持たない。テスト
                // 許容誤差緩和の判定は「差分内容」で行い「パス」では
                // 行わないため）。
                match_rule: MatchRule::TestAssertionRelaxationWithoutProdChange {
                    assertion_patterns: vec![
                        "assert!".to_string(),
                        "abs() <".to_string(),
                        "1e-[0-9]".to_string(),
                    ],
                },
            },
            ExclusionRule {
                id: "dependency-change".to_string(),
                category: Category::DependencyChange,
                action: Action::HumanApproval,
                // TASK-5.1b・#120 で人間承認済みの値をそのまま実装へ反映する
                // （新たな値の判断は含まない。escalate が増える方向のみの
                // 安全側単調な変更。PR 本文参照）。
                match_rule: MatchRule::AnyDiffInPaths {
                    paths: dependency_paths,
                },
            },
        ],
    })
}

/// 除外リスト評価に必要なリポジトリ文脈（TASK-5.2c・#124）。
///
/// [`ExclusionEvaluation::evaluate`] は `MatchRule::AnyDiffInPaths` の判定に
/// `changed_files` を、`MatchRule::TestAssertionRelaxationWithoutProdChange`
/// の判定に `repo_root`／`baseline`（`exclusion_match` 側で独自に `git diff`
/// を再実行する）を使う。両方式が同じ `baseline` を基準にする契約を保つため、
/// 呼び出し元は本構造体を経由してのみ評価する。
pub struct EvaluationContext {
    pub repo_root: PathBuf,
    pub baseline: String,
    pub changed_files: Vec<String>,
}

impl EvaluationContext {
    /// `repo_root`（`baseline` との差分対象の作業木）から `EvaluationContext`
    /// を構築する。`git diff` の起動失敗・非ゼロ終了は
    /// [`GuardrailError`] として伝播する（fail-closed。`match` なし
    /// ＝自動適用方向へ丸めない）。
    ///
    /// `changed_files` は `crate::exclusion_match::changed_files_for_policy_exclusion`
    /// （`Cargo.lock` を除外しない取得口）を使う理由は同関数のドキュメント参照
    /// （`dependency-change` ルールが `Cargo.lock` 単独変更を見逃さないため）。
    pub fn from_repo(repo_root: &Path, baseline: &str) -> Result<Self, GuardrailError> {
        let changed_files =
            exclusion_match::changed_files_for_policy_exclusion(repo_root, baseline)?;
        Ok(EvaluationContext {
            repo_root: repo_root.to_path_buf(),
            baseline: baseline.to_string(),
            changed_files,
        })
    }
}

/// ルール集合の評価結果。match したルール id の一覧を保持する
/// （REQ-5 2026-08-05 v2 注記の `expected_exclusion_rule_ids` と同型）。
///
/// `matched_rule_ids`（match したと確定・評価済み）と
/// `unevaluated_rule_ids`（現時点の [`MatchRule`] 全 variant は必ず評価される
/// ため通常は空集合。将来 variant 追加時の受け皿。モジュール冒頭「fail-closed
/// 統合」参照）を型で分離することで、呼び出し側が両者を取り違えて
/// fail-open にならないようにする。判定への反映は必ず
/// [`ExclusionEvaluation::effective_rule_ids`] を経由すること。
///
/// `Default` は意図的に derive しない: 既定値
/// `{matched_rule_ids: [], unevaluated_rule_ids: []}` は「全評価済み・
/// 未マッチ（安全）」を意味してしまい、`unwrap_or_default()` 等のエラー
/// パス経由で本型が分離しようとした「未マッチ」と「未評価」の混同を
/// 再導入しうる（イシュー #122 レビュー指摘）。値が必要な場合は
/// [`ExclusionEvaluation::evaluate`] を明示的に呼び出すこと。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusionEvaluation {
    pub matched_rule_ids: Vec<String>,
    /// 評価ロジック未実装のため判定不能だったルール id。
    /// **#124 時点で [`ExclusionEvaluation::evaluate`] はこのフィールドを
    /// 常に空 `Vec` で返す**（[`MatchRule`] の全 variant が評価対象であり、
    /// `evaluate` の実装に「未評価」へ push する分岐が存在しないため）。
    /// 将来 [`MatchRule`] に variant を追加し、評価ロジックを即座に実装
    /// できない場合の受け皿として型のみ残す（モジュール冒頭「fail-closed
    /// 統合」参照）。
    pub unevaluated_rule_ids: Vec<String>,
}

impl ExclusionEvaluation {
    /// `ctx` に対して `rules` を評価する。
    ///
    /// - `MatchRule::AnyDiffInPaths` は `ctx.changed_files` × `paths` の
    ///   純粋関数評価（[`any_diff_in_paths::any_diff_in_paths`]）
    /// - `MatchRule::TestAssertionRelaxationWithoutProdChange` は
    ///   `ctx.repo_root`／`ctx.baseline` を使って `git diff` を実行する
    ///   （[`crate::exclusion_match::test_assertion_relaxation_without_prod_change`]）
    ///
    /// `git` 実行エラーは `Err` として伝播し、`false`（match なし＝自動適用
    /// 方向）へ丸めない（fail-closed。`.claude/rules/security.md` A08。
    /// `exclusion_match` モジュールと同一契約）。`match` は網羅列挙・`_ =>`
    /// 禁止を維持し、`MatchRule` に variant を追加する際は本関数の更新を
    /// コンパイルエラーで強制する。
    pub fn evaluate(
        rules: &[ExclusionRule],
        ctx: &EvaluationContext,
    ) -> Result<Self, GuardrailError> {
        let mut matched_rule_ids = Vec::new();
        // 意図的に常に空のまま返す（#124 時点で [`MatchRule`] の全 variant を
        // 下記 `match` が評価するため、「未評価」に該当するケースが存在しない。
        // `ExclusionEvaluation::unevaluated_rule_ids` のフィールドドキュメント参照）。
        let unevaluated_rule_ids = Vec::new();
        for rule in rules {
            match &rule.match_rule {
                MatchRule::AnyDiffInPaths { paths } => {
                    if any_diff_in_paths::any_diff_in_paths(&ctx.changed_files, paths) {
                        matched_rule_ids.push(rule.id.clone());
                    }
                }
                MatchRule::TestAssertionRelaxationWithoutProdChange { assertion_patterns } => {
                    if exclusion_match::test_assertion_relaxation_without_prod_change(
                        &ctx.repo_root,
                        &ctx.baseline,
                        assertion_patterns,
                    )? {
                        matched_rule_ids.push(rule.id.clone());
                    }
                }
            }
        }
        Ok(ExclusionEvaluation {
            matched_rule_ids,
            unevaluated_rule_ids,
        })
    }

    /// `matched_rule_ids ∪ unevaluated_rule_ids`（重複排除・出現順保持）を
    /// 返す。[`crate::decision::DecisionInput::new`] の `exclusion_rule_ids`
    /// 引数へそのまま渡す想定の fail-closed 統合 API（モジュール冒頭
    /// 「fail-closed 統合」参照）。「未評価＝判定不能」を無条件エスカレー
    /// ション側へ倒す契約をここ 1 箇所に閉じ込め、呼び出し側
    /// （`main.rs`・`self-repair`）が個別に `unevaluated_rule_ids` の扱いを
    /// 判断しなくてよいようにする。
    pub fn effective_rule_ids(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for id in self
            .matched_rule_ids
            .iter()
            .chain(self.unevaluated_rule_ids.iter())
        {
            if seen.insert(id.clone()) {
                result.push(id.clone());
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    /// `AnyDiffInPaths` 系ルールのみを評価するテスト向けに、`git` を経由
    /// せず `changed_files` を直接指定した `EvaluationContext` を組み立てる。
    /// `repo_root`／`baseline` はダミー値（`TestAssertionRelaxationWithoutProdChange`
    /// を含む `rules` を渡さない限り参照されない）。
    fn changed_files_only_context(changed_files: Vec<String>) -> EvaluationContext {
        EvaluationContext {
            repo_root: PathBuf::new(),
            baseline: String::new(),
            changed_files,
        }
    }

    #[test]
    fn builtin_defaults_has_arch_hyperparameter_change_rule() {
        let config = builtin_defaults().unwrap();
        assert_eq!(config.rules.len(), 3);
        let rule = &config.rules[0];
        assert_eq!(rule.id, "arch-hyperparameter-change");
        assert_eq!(rule.category, Category::ArchitectureChange);
        assert_eq!(rule.action, Action::HumanApproval);
        match &rule.match_rule {
            MatchRule::AnyDiffInPaths { paths } => {
                let raw: Vec<&str> = paths.iter().map(|p| p.as_str()).collect();
                assert_eq!(
                    raw,
                    vec!["**/src/model*.rs", "**/src/nn/**", "**/src/*model*/**"]
                );
            }
            other => panic!("expected AnyDiffInPaths, got {other:?}"),
        }
    }

    #[test]
    fn builtin_defaults_has_test_tolerance_loosening_rule_with_patterns() {
        let config = builtin_defaults().unwrap();
        let rule = &config.rules[1];
        assert_eq!(rule.id, "test-tolerance-loosening");
        assert_eq!(rule.category, Category::TestToleranceLoosening);
        assert_eq!(rule.action, Action::HumanApproval);
        match &rule.match_rule {
            MatchRule::TestAssertionRelaxationWithoutProdChange { assertion_patterns } => {
                assert_eq!(
                    assertion_patterns,
                    &vec![
                        "assert!".to_string(),
                        "abs() <".to_string(),
                        "1e-[0-9]".to_string()
                    ]
                );
            }
            other => panic!("expected TestAssertionRelaxationWithoutProdChange, got {other:?}"),
        }
    }

    #[test]
    fn builtin_defaults_has_dependency_change_rule() {
        let config = builtin_defaults().unwrap();
        let rule = &config.rules[2];
        assert_eq!(rule.id, "dependency-change");
        assert_eq!(rule.category, Category::DependencyChange);
        assert_eq!(rule.action, Action::HumanApproval);
        match &rule.match_rule {
            MatchRule::AnyDiffInPaths { paths } => {
                let raw: Vec<&str> = paths.iter().map(|p| p.as_str()).collect();
                assert_eq!(
                    raw,
                    vec![
                        "**/Cargo.toml",
                        "**/Cargo.lock",
                        "deny.toml",
                        "docs/license-matrix.md",
                        ".claude/rules/deps-policy.md",
                    ]
                );
            }
            other => panic!("expected AnyDiffInPaths, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_matches_arch_hyperparameter_change_on_model_path() {
        let config = builtin_defaults().unwrap();
        // `TestAssertionRelaxationWithoutProdChange` は git を要求するため、
        // ここでは `AnyDiffInPaths` 系（arch・dependency）のみを対象にする。
        let rules: Vec<_> = config
            .rules
            .into_iter()
            .filter(|r| matches!(r.match_rule, MatchRule::AnyDiffInPaths { .. }))
            .collect();
        let ctx =
            changed_files_only_context(vec!["crates/tensor-core/src/model_mlp.rs".to_string()]);
        let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
        assert_eq!(
            evaluation.matched_rule_ids,
            vec!["arch-hyperparameter-change"]
        );
        assert!(evaluation.unevaluated_rule_ids.is_empty());
    }

    #[test]
    fn evaluate_matches_dependency_change_on_cargo_lock_only_change() {
        // 計画 3.1 節「注意」: `Cargo.toml` を伴わない `Cargo.lock` 単独変更
        // でも `dependency-change` が発火することを固定する（見逃し方向の
        // fail-open 回帰検知）。
        let config = builtin_defaults().unwrap();
        let rules: Vec<_> = config
            .rules
            .into_iter()
            .filter(|r| matches!(r.match_rule, MatchRule::AnyDiffInPaths { .. }))
            .collect();
        let ctx = changed_files_only_context(vec!["Cargo.lock".to_string()]);
        let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
        assert_eq!(evaluation.matched_rule_ids, vec!["dependency-change"]);
    }

    #[test]
    fn evaluate_does_not_match_unrelated_changes() {
        let config = builtin_defaults().unwrap();
        let rules: Vec<_> = config
            .rules
            .into_iter()
            .filter(|r| matches!(r.match_rule, MatchRule::AnyDiffInPaths { .. }))
            .collect();
        let ctx =
            changed_files_only_context(vec!["README.md".to_string(), "src/lib.rs".to_string()]);
        let evaluation = ExclusionEvaluation::evaluate(&rules, &ctx).unwrap();
        assert!(evaluation.matched_rule_ids.is_empty());
        assert!(evaluation.unevaluated_rule_ids.is_empty());
    }

    #[test]
    fn effective_rule_ids_dedupes_and_preserves_order() {
        let evaluation = ExclusionEvaluation {
            matched_rule_ids: vec!["a".to_string(), "b".to_string()],
            unevaluated_rule_ids: vec!["b".to_string(), "c".to_string()],
        };
        assert_eq!(
            evaluation.effective_rule_ids(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn effective_rule_ids_empty_when_no_match() {
        let evaluation = ExclusionEvaluation {
            matched_rule_ids: Vec::new(),
            unevaluated_rule_ids: Vec::new(),
        };
        assert!(evaluation.effective_rule_ids().is_empty());
    }

    // --- ここから git を要する統合ミニテスト（#123 述語の配線確認） ---
    // `exclusion_match.rs` のテストヘルパー（`init_repo`／`commit_all`）と
    // 同一方針だが、`pub(crate)`/private の境界をまたがず本モジュール内で
    // 完結させるため最小構成で独立実装する。

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
            "guardrail-policy-exclusion-mod-{name}-{}",
            std::process::id()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{dir:?} の削除に失敗: {e}"));
        }
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{dir:?} の作成に失敗: {e}"));
        run(&dir, &["init", "-q"]);
        dir
    }

    /// 配線確認の目印テスト（#122 レビュー指摘・#123 引き継ぎの解消）:
    /// `test-tolerance-loosening` ルールが実際に評価され、`matched_rule_ids`
    /// へ計上されることを確認する（以前は schema のみで
    /// `unevaluated_rule_ids` に回っていた。モジュール冒頭「fail-closed 統合」
    /// 参照）。
    #[test]
    fn test_assertion_relaxation_rule_is_evaluated_and_matched() {
        let dir = init_repo("wired");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        let config = builtin_defaults().unwrap();
        let rule = config
            .rules
            .into_iter()
            .find(|r| r.id == "test-tolerance-loosening")
            .unwrap();
        let ctx = EvaluationContext::from_repo(&dir, "HEAD").unwrap();
        let evaluation = ExclusionEvaluation::evaluate(&[rule], &ctx).unwrap();
        assert_eq!(
            evaluation.matched_rule_ids,
            vec!["test-tolerance-loosening"]
        );
        assert!(evaluation.unevaluated_rule_ids.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn all_builtin_rules_evaluate_without_error_on_unrelated_change() {
        // 3 ルール全て（`test-tolerance-loosening` を含む）を実リポジトリで
        // 評価し、無関係な変更では一切 match しないことを確認する
        // （`EvaluationContext::from_repo` 経由の end-to-end 疎通確認）。
        let dir = init_repo("all-rules-clean");
        fs::write(dir.join("README.md"), "baseline\n").unwrap();
        commit_all(&dir, "baseline");
        fs::write(dir.join("README.md"), "updated\n").unwrap();

        let config = builtin_defaults().unwrap();
        let ctx = EvaluationContext::from_repo(&dir, "HEAD").unwrap();
        let evaluation = ExclusionEvaluation::evaluate(&config.rules, &ctx).unwrap();
        assert!(evaluation.matched_rule_ids.is_empty());
        assert!(evaluation.unevaluated_rule_ids.is_empty());

        fs::remove_dir_all(&dir).ok();
    }
}
