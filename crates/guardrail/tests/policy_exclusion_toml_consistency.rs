//! `policy-exclusion.toml`（リポジトリルート）と
//! `guardrail::policy_exclusion::builtin_defaults()` の完全一致を保証する
//! 回帰テスト（TASK-5.2c・イシュー #124）。
//!
//! 両者が乖離すると、CLI（`main.rs`）が実際にロードする TOML の内容と、
//! テスト・ドキュメントが前提とする組み込み既定値がずれ、レビュー時に
//! 気づかれない判定迂回経路になり得る（`policy-exclusion.toml` ヘッダコメント
//! 「乖離は判定迂回経路になり得る」・`.claude/rules/security.md` A08）。
//! `PolicyExclusionConfig`／`ExclusionRule`／`MatchRule` はいずれも
//! `#[derive(PartialEq, Eq)]` のため、`assert_eq!` 1 発でルール一覧全体
//! （id・category・action・match 方式・パス／パターンの並び）の一致を保証できる。

use std::path::PathBuf;

use guardrail::policy_exclusion::{builtin_defaults, load_from_str};

/// `CARGO_MANIFEST_DIR`（`crates/guardrail`）からリポジトリルートの
/// `policy-exclusion.toml` を解決する。ワークスペース構成
/// （`crates/guardrail/../../policy-exclusion.toml`）が変わらない限り安定する。
fn repo_root_policy_exclusion_toml_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("policy-exclusion.toml")
}

#[test]
fn repo_root_policy_exclusion_toml_matches_builtin_defaults() {
    let path = repo_root_policy_exclusion_toml_path();
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?} の読み込みに失敗: {e}"));

    let loaded = load_from_str(&content).unwrap_or_else(|e| panic!("{path:?} のパースに失敗: {e}"));
    let builtin = builtin_defaults().unwrap();

    assert_eq!(
        loaded, builtin,
        "policy-exclusion.toml と builtin_defaults() が乖離しています（判定迂回経路の懸念。\
         両者は同一 PR で同時変更しユーザー承認を得ること）"
    );
}

#[test]
fn repo_root_policy_exclusion_toml_defines_all_three_categories() {
    // 設計 §4「3 カテゴリすべて最低 1 件ずつ存在することを builtin_defaults()
    // 側でも一致させる」の直接確認（TOML 側からの到達性）。
    let path = repo_root_policy_exclusion_toml_path();
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?} の読み込みに失敗: {e}"));
    let loaded = load_from_str(&content).unwrap();
    assert_eq!(loaded.rules.len(), 3);

    use guardrail::policy_exclusion::Category;
    assert!(
        loaded
            .rules
            .iter()
            .any(|r| r.category == Category::ArchitectureChange)
    );
    assert!(
        loaded
            .rules
            .iter()
            .any(|r| r.category == Category::TestToleranceLoosening)
    );
    assert!(
        loaded
            .rules
            .iter()
            .any(|r| r.category == Category::DependencyChange)
    );
}
