//! ゲーミング（判定回避）疑いの検知（TASK-4.1c・イシュー #106・REQ-4）。
//!
//! 「本番コードとテスト（アサーション・許容誤差）の**同時**変更」を検知する
//! （v1 `rust-ai-library-v1/crates/guardrail/src/gaming.rs` 移植）。
//! [`crate::exclusion_match::test_assertion_relaxation_without_prod_change`]
//! （`test-tolerance-loosening` ルール。**単独**緩和＝本番コード変更**なし**
//! を match 条件とする）とは対偶の入力領域を担う: 削除行のアサーション・
//! 許容誤差緩和パターンへの一致 **かつ** 本番コード（`src/*.rs`）の変更が
//! 両方成立する場合のみ疑いとする。
//!
//! パターン列（`assert!`／`abs() <`／`1e-[0-9]`）は
//! `policy_exclusion::builtin_defaults()` の `test-tolerance-loosening`
//! ルールと同一値を用いる（既定値の変更はユーザー承認必須。
//! `.claude/rules/security.md`）。ここではガードレール閾値ではなく
//! パターン自体の定義元を一致させる目的でリテラルとして複製する
//! （`policy_exclusion` モジュールへの依存を増やさないための独立実装。
//! 値のドリフトは `policy_exclusion_toml_consistency.rs` 相当の回帰テストが
//! ないため、変更時は両モジュールを同時に見直すこと）。

use std::path::Path;

use crate::error::GuardrailError;
use crate::exclusion_match::{self, ProdTouch};

/// アサーション・許容誤差緩和の検知パターン（`policy-exclusion.toml` §4.1・
/// `policy_exclusion::builtin_defaults()` と同一値）。
const ASSERTION_RELAXATION_PATTERNS: [&str; 3] = ["assert!", "abs() <", "1e-[0-9]"];

/// `baseline` と現作業木の差分がゲーミング（判定回避）の疑いに該当するか
/// 判定する。「テストのアサーション・許容誤差の緩和」**かつ**「本番コード
/// （`src/*.rs`）の変更」が両方成立する場合のみ `true`。
///
/// `mod tests` 境界が非標準名等で特定できない
/// （[`ProdTouch::UnknownBoundary`]）場合は「本番コード変更ありと確認できな
/// かった」として `false`（疑いなし）方向に倒す。安全側判定は
/// `exclusion_match::test_assertion_relaxation_without_prod_change`
/// （`UnknownBoundary` を match=true＝エスカレーション方向に倒す）が別途
/// 無条件エスカレーションとして担うため、本関数側で二重に安全側へ倒す
/// 必要はない（`decide()` の判定順序 2. 除外リスト match が本関数の結果に
/// 優先して評価される。`decision.rs` モジュールコメント参照）。
pub(crate) fn gaming_suspected(repo_root: &Path, baseline: &str) -> Result<bool, GuardrailError> {
    let patterns: Vec<String> = ASSERTION_RELAXATION_PATTERNS
        .iter()
        .map(|s| s.to_string())
        .collect();

    if !exclusion_match::touches_test_assertion_with_patterns(repo_root, baseline, &patterns)? {
        return Ok(false);
    }

    Ok(matches!(
        exclusion_match::touches_prod_logic(repo_root, baseline)?,
        ProdTouch::Touched
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_exclusion::{self, MatchRule};
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    /// `ASSERTION_RELAXATION_PATTERNS`（本モジュールが複製する値）が
    /// `policy_exclusion::builtin_defaults()` の `test-tolerance-loosening`
    /// ルール定義とドリフトしていないことを固定する（モジュール冒頭
    /// コメントの「値のドリフトは...」に対応する回帰テスト）。
    #[test]
    fn assertion_relaxation_patterns_match_policy_exclusion_builtin_defaults() {
        let config = policy_exclusion::builtin_defaults().unwrap();
        let rule = config
            .rules
            .into_iter()
            .find(|r| r.id == "test-tolerance-loosening")
            .expect("test-tolerance-loosening ルールが存在するはず");
        let MatchRule::TestAssertionRelaxationWithoutProdChange { assertion_patterns } =
            rule.match_rule
        else {
            panic!("test-tolerance-loosening は TestAssertionRelaxationWithoutProdChange のはず");
        };
        assert_eq!(
            assertion_patterns,
            ASSERTION_RELAXATION_PATTERNS
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "gaming.rs の ASSERTION_RELAXATION_PATTERNS が policy_exclusion::builtin_defaults() \
             とドリフトしています"
        );
    }

    fn run(cwd: &Path, args: &[&str]) {
        let mut cmd = Command::new("git");
        cmd.args(args).current_dir(cwd);
        // 祖先プロセス（lefthook の pre-push フック等）から継承された
        // `GIT_DIR`／`GIT_WORK_TREE` 等を除去する（`exclusion_match::git_command`
        // と同一方針。除去しないとフィクスチャ用一時リポジトリの隔離が壊れる）。
        for (key, _) in std::env::vars() {
            if key.starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
        let output = cmd
            .output()
            .unwrap_or_else(|e| panic!("git {args:?} 起動に失敗: {e}"));
        assert!(
            output.status.success(),
            "git {args:?} が失敗: {}",
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
        let dir =
            std::env::temp_dir().join(format!("guardrail-gaming-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        run(&dir, &["init", "-q"]);
        dir
    }

    /// 本番コード変更＋許容誤差緩和の同時発生 → ゲーミング疑いあり。
    #[test]
    fn simultaneous_prod_and_tolerance_change_is_suspected() {
        let dir = init_repo("simultaneous");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn double(a: i32) -> i32 { a * 2 }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn double(a: i32) -> i32 { a * 3 }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        assert!(gaming_suspected(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    /// テスト単独の許容誤差緩和（本番コード変更なし）→ ゲーミング疑いなし
    /// （REQ-5 の `test-tolerance-loosening` ルール側が無条件エスカレー
    /// ションとして別途拾う対偶ケース）。
    #[test]
    fn test_only_relaxation_without_prod_change_is_not_suspected() {
        let dir = init_repo("test-only");
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

        assert!(!gaming_suspected(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    /// 本番コードのみの変更（アサーション緩和なし）→ ゲーミング疑いなし。
    #[test]
    fn prod_only_change_is_not_suspected() {
        let dir = init_repo("prod-only");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/lib.rs"), "pub fn noop() {}\n").unwrap();
        commit_all(&dir, "baseline");

        fs::write(dir.join("src/lib.rs"), "pub fn noop() { let _ = 1; }\n").unwrap();

        assert!(!gaming_suspected(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }
}
