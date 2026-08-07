//! `any_diff_in_paths` 方式の除外ルール評価（REQ-5 受け入れ基準 1・TASK-5.2a）。
//!
//! `arch-hyperparameter-change`（PoC-3 G2: 隠れ層次元数変更のブラインドスポット
//! 対策）はこの方式で発火する。「変更ファイル一覧のいずれか 1 つでも対象パス
//! パターンに一致すれば match」という保守的な設計であり、変更内容の意図
//! （ドキュメントのみか実質的なアーキテクチャ変更か）は一切解釈しない
//! （`docs/spec/04-requirements.md:125` の意図的トレードオフ）。
//!
//! `PolicyExclusionConfig` の TOML ロード（#119・#124）・git diff からの変更
//! ファイル一覧取得（#103）はこのモジュールの呼び出し元が担う。本モジュールは
//! 「変更ファイルパス一覧 × 検証済みパターン列 → match 判定」の純粋関数のみを
//! 提供する（計画 2 節「前提未完了への対処」）。

use super::path_match::PathPattern;

/// `changed_files` のいずれか 1 つでも `paths` のいずれかに一致すれば `true`。
///
/// 一致判定はパスの事実のみで行う（変更内容の diff hunk は見ない）。
/// 空の `changed_files` は match しない（変更が無ければ発火しようがない）。
pub fn any_diff_in_paths(changed_files: &[String], paths: &[PathPattern]) -> bool {
    changed_files
        .iter()
        .any(|file| paths.iter().any(|pattern| pattern.matches(file)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch_hyperparameter_change_paths() -> Vec<PathPattern> {
        // v1 `policy-exclusion.toml` §4.1 の組み込み既定値（`super::builtin_defaults`
        // が公開既定として保持する値と同一。ここでは評価ロジック単体のテストの
        // ためにリテラルで再構築する）。
        ["**/src/model*.rs", "**/src/nn/**", "**/src/*model*/**"]
            .iter()
            .map(|p| PathPattern::compile(p).unwrap())
            .collect()
    }

    #[test]
    fn matches_when_model_file_changed() {
        let paths = arch_hyperparameter_change_paths();
        let changed = vec!["crates/tensor-core/src/model_mlp.rs".to_string()];
        assert!(any_diff_in_paths(&changed, &paths));
    }

    #[test]
    fn matches_when_nn_subtree_file_changed() {
        let paths = arch_hyperparameter_change_paths();
        let changed = vec!["crates/x/src/nn/layer.rs".to_string()];
        assert!(any_diff_in_paths(&changed, &paths));
    }

    #[test]
    fn matches_when_star_model_star_directory_file_changed() {
        let paths = arch_hyperparameter_change_paths();
        let changed = vec!["crates/foo/src/mymodel/config.rs".to_string()];
        assert!(any_diff_in_paths(&changed, &paths));
    }

    #[test]
    fn matches_when_at_least_one_of_many_changed_files_hits() {
        let paths = arch_hyperparameter_change_paths();
        let changed = vec![
            "README.md".to_string(),
            "crates/guardrail/src/lib.rs".to_string(),
            "crates/tensor-core/src/model_mlp.rs".to_string(),
        ];
        assert!(any_diff_in_paths(&changed, &paths));
    }

    #[test]
    fn does_not_match_unrelated_paths() {
        let paths = arch_hyperparameter_change_paths();
        let changed = vec![
            "README.md".to_string(),
            "crates/guardrail/src/lib.rs".to_string(),
        ];
        assert!(!any_diff_in_paths(&changed, &paths));
    }

    #[test]
    fn does_not_match_when_no_changed_files() {
        let paths = arch_hyperparameter_change_paths();
        assert!(!any_diff_in_paths(&[], &paths));
    }

    #[test]
    fn matches_even_on_comment_only_change_to_model_file() {
        // 意図的トレードオフ（`04-requirements.md:125`）: パスのみで match するため、
        // `src/model*.rs` へのコメント追加等の挙動不変変更も match する（過剰
        // エスカレーションを許容する保守的設計。ファイル内容までは見ない）。
        let paths = arch_hyperparameter_change_paths();
        let changed = vec!["crates/tensor-core/src/model.rs".to_string()];
        assert!(any_diff_in_paths(&changed, &paths));
    }
}
