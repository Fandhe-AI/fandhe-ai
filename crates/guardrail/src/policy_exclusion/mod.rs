//! ポリシー除外リスト（REQ-5）: 機械判定（[`crate::decision`]）では検知できない
//! 変更カテゴリを、パス／変更内容パターンにマッチした事実のみで無条件人間承認へ
//! 回す保守的な仕組み。
//!
//! `docs/spec/04-requirements.md`（REQ-5・2026-08-05 v2 注記）が定める 2 層目
//! 不変条件「除外リストは安全側にしか作用しない（見逃し方向に緩まない）」を
//! 実装が壊さないことは、`ExclusionEvaluation` が「match したルール id を追加
//! するだけで、1 層目の [`crate::decision::Verdict`] を上書き・緩和しない」設計
//! で担保する（実際の verdict 合成は #124・TASK-5.2c が `decision` 側と統合する）。
//!
//! # モジュール構成（イシュー #122・TASK-5.2a 時点）
//! - [`path_match`][]: 自作パス glob マッチャ（`glob` クレート非依存。#122 管轄）。
//! - [`any_diff_in_paths`][]: `arch-hyperparameter-change` の match 評価本体
//!   （#122 管轄）。
//!
//! # スコープ境界（`.claude/rules/out-of-scope-tracking.md`）
//! - `policy-exclusion.toml` ファイル自体の移植・TOML ロード機構: #119（TASK-5.1a）・
//!   #124。`toml` クレートは許容依存 8 区分に非該当のため、`crate::toml_lite`
//!   （手書きミニパーサ）の再利用可否をユーザー承認込みで #124 が判断する
//! - `test_assertion_relaxation_without_prod_change` の評価ロジック: #123（TASK-5.2b）。
//!   本モジュールは [`MatchRule`] として schema（型定義）のみ用意し、評価対象から
//!   意図的に除外する（下記 [`ExclusionEvaluation::evaluate`] 参照。#123 との
//!   並行編集コンフリクトを避けるためファイルを分離してある）
//! - git diff からの変更ファイル一覧取得・[`crate::decision::Verdict`] との統合:
//!   #103（TASK-4.1）・#124（TASK-5.2c）

pub mod any_diff_in_paths;
pub mod path_match;

pub use any_diff_in_paths::any_diff_in_paths;
pub use path_match::{PathPattern, PatternError};

/// 除外ルールが人間承認へ回す変更カテゴリ（v1 `policy-exclusion.toml` §4 由来）。
///
/// 値は REQ-1 の受け皿として TASK-5.1 で追加予定の「許容依存一覧の追加・更新」
/// カテゴリも見越して `#[non_exhaustive]` とはしない（本イシューでは 2 カテゴリの
/// みを定義し、値の追加はカテゴリ新設タスク側の責務とする）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// モデルアーキテクチャ・ハイパーパラメータ変更（PoC-3 G2 ブラインドスポット対策）。
    ArchitectureChange,
    /// テスト許容誤差の単独緩和（PoC-3 G5 ブラインドスポット対策。#123 管轄）。
    TestToleranceLoosening,
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
    /// （評価は [`any_diff_in_paths::any_diff_in_paths`]。本イシューで実装）。
    AnyDiffInPaths { paths: Vec<PathPattern> },
    /// テスト許容誤差（assertion）が単独で緩和されているか判定する方式。
    /// schema（型）のみここで定義し、評価ロジックは #123（TASK-5.2b）に委ねる
    /// （計画 4 節「v1 の TASK-5.2-S1/S2 分割と同型」）。
    TestAssertionRelaxationWithoutProdChange,
}

/// 1 件の除外ルール定義（TOML の 1 テーブルに対応する想定。ロード自体は #124）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusionRule {
    pub id: String,
    pub category: Category,
    pub action: Action,
    pub match_rule: MatchRule,
}

/// 除外ルール一式。TOML ファイルのロード（#119・#124）を経ずとも、
/// [`builtin_defaults`] で組み込み既定値を直接得られる。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyExclusionConfig {
    pub rules: Vec<ExclusionRule>,
}

/// 組み込み既定値: `arch-hyperparameter-change`（v1 `policy-exclusion.toml` §4.1
/// の値をそのまま採用。値の変更はユーザー承認必須のためこの実装では一切変更
/// しない。`.claude/rules/security.md`「ガードレール閾値・ポリシー除外リストの
/// 変更は必ず人間の承認を経る」）。
///
/// `.claude/rules/coding-rust.md`「本番経路で `unwrap()`/`expect()` を使わない」
/// に例外を設けず、パターン構文検証（[`PathPattern::compile`]）の失敗は
/// `Result` で呼び出し元へ返す（自明にコンパイル時定数だからと `.expect()` へ
/// 逃げない。呼び出し元は `self-repair` からの lib 直接呼び出しも含むため、
/// この関数自体をパニックしない契約に保つ）。
pub fn builtin_defaults() -> Result<PolicyExclusionConfig, PatternError> {
    let paths = ["**/src/model*.rs", "**/src/nn/**", "**/src/*model*/**"]
        .iter()
        .map(|p| PathPattern::compile(p))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PolicyExclusionConfig {
        rules: vec![ExclusionRule {
            id: "arch-hyperparameter-change".to_string(),
            category: Category::ArchitectureChange,
            action: Action::HumanApproval,
            match_rule: MatchRule::AnyDiffInPaths { paths },
        }],
    })
}

/// ルール集合の評価結果。match したルール id の一覧を保持する
/// （REQ-5 2026-08-05 v2 注記の `expected_exclusion_rule_ids` と同型）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExclusionEvaluation {
    pub matched_rule_ids: Vec<String>,
}

impl ExclusionEvaluation {
    /// `changed_files` に対して `rules` を評価する。
    ///
    /// `MatchRule::AnyDiffInPaths` のみを評価し、
    /// `MatchRule::TestAssertionRelaxationWithoutProdChange` は評価対象から
    /// 意図的に除外する（#123 が実装を追加するまでは match させない。schema
    /// のみ先行定義する計画のため、ここで `unimplemented!()` 等により fail する
    /// と #123 未着手の間クレート全体の評価が壊れてしまい fail-closed の趣旨に
    /// 反する。未実装ルールは「まだ発火しない」で安全側に倒す）。
    pub fn evaluate(rules: &[ExclusionRule], changed_files: &[String]) -> Self {
        let matched_rule_ids = rules
            .iter()
            .filter(|rule| match &rule.match_rule {
                MatchRule::AnyDiffInPaths { paths } => {
                    any_diff_in_paths::any_diff_in_paths(changed_files, paths)
                }
                MatchRule::TestAssertionRelaxationWithoutProdChange => false,
            })
            .map(|rule| rule.id.clone())
            .collect();
        ExclusionEvaluation { matched_rule_ids }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_defaults_has_arch_hyperparameter_change_rule() {
        let config = builtin_defaults().unwrap();
        assert_eq!(config.rules.len(), 1);
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
    fn evaluate_matches_arch_hyperparameter_change_on_model_path() {
        let config = builtin_defaults().unwrap();
        let changed = vec!["crates/tensor-core/src/model_mlp.rs".to_string()];
        let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
        assert_eq!(
            evaluation.matched_rule_ids,
            vec!["arch-hyperparameter-change"]
        );
    }

    #[test]
    fn evaluate_does_not_match_unrelated_changes() {
        let config = builtin_defaults().unwrap();
        let changed = vec!["README.md".to_string(), "Cargo.toml".to_string()];
        let evaluation = ExclusionEvaluation::evaluate(&config.rules, &changed);
        assert!(evaluation.matched_rule_ids.is_empty());
    }

    #[test]
    fn evaluate_never_matches_test_assertion_relaxation_rule_yet() {
        // #123（TASK-5.2b）の評価ロジック実装までは常に match しないことの
        // 固定テスト（スコープ境界の回帰検知。実装が入れば #123 側でこのテストを
        // 更新する）。
        let rule = ExclusionRule {
            id: "test-tolerance-loosening".to_string(),
            category: Category::TestToleranceLoosening,
            action: Action::HumanApproval,
            match_rule: MatchRule::TestAssertionRelaxationWithoutProdChange,
        };
        let changed = vec!["crates/tensor-core/tests/regression.rs".to_string()];
        let evaluation = ExclusionEvaluation::evaluate(&[rule], &changed);
        assert!(evaluation.matched_rule_ids.is_empty());
    }
}
