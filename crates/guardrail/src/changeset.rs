//! `--baseline`/`--repo` の検証・解決（TASK-4.1c・イシュー #106）。
//!
//! [`crate::check`] の measured 経路の入口で呼ばれる。`--baseline` は
//! そのまま `git` の引数へ渡るため、注入攻撃（A03）を避けるべく事前に
//! 文字クラスを制限し、かつ `git rev-parse --verify` で実在確認する
//! （v1 `rust-ai-library-v1/crates/guardrail/src/changeset.rs` からの移植）。

use std::path::Path;

use crate::error::GuardrailError;
use crate::exclusion_match;

/// `baseline` に許可する文字クラス（git の ref 名として妥当な範囲に限定する
/// 保守的なホワイトリスト。英数字・`-`・`_`・`/`・`.` のみ）。
fn is_allowed_ref_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.')
}

/// `baseline` の構文検証（`git` へ渡す前段。実在確認は
/// [`resolve_baseline_commit`] が担う）。
fn validate_baseline_ref(baseline: &str) -> Result<(), GuardrailError> {
    if baseline.is_empty() {
        return Err(GuardrailError::UsageError(
            "--baseline must not be empty".to_string(),
        ));
    }
    if baseline.starts_with('-') {
        // `-` 始まりは git がオプションと誤認する経路（A03: 引数注入対策）。
        return Err(GuardrailError::UsageError(format!(
            "--baseline '{baseline}' must not start with '-'"
        )));
    }
    if !baseline.chars().all(is_allowed_ref_char) {
        return Err(GuardrailError::UsageError(format!(
            "--baseline '{baseline}' contains disallowed characters (allowed: alnum, '-', '_', '/', '.')"
        )));
    }
    Ok(())
}

/// `baseline` を構文検証したうえで、`repo_root` に実在するコミットを指すか
/// `git rev-parse --verify --quiet` で確認する。
///
/// 実在しない場合は [`exclusion_match::run_git`] が返す
/// [`GuardrailError::DiffFailed`] をそのまま呼び出し元へ伝播する
/// （fail-closed。`main.rs::report_error_and_exit` はこれを終了コード `1`
/// （内部エラー）へ写像し、`0`/`10`/`20` のいずれへも丸めない）。
pub(crate) fn resolve_baseline_commit(
    repo_root: &Path,
    baseline: &str,
) -> Result<(), GuardrailError> {
    validate_baseline_ref(baseline)?;
    let commit_ref = format!("{baseline}^{{commit}}");
    exclusion_match::run_git(
        repo_root,
        &["rev-parse", "--verify", "--quiet", &commit_ref],
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_baseline() {
        let err = validate_baseline_ref("").unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn rejects_dash_prefixed_baseline() {
        let err = validate_baseline_ref("--upload-pack=evil").unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn rejects_disallowed_characters() {
        let err = validate_baseline_ref("main; rm -rf /").unwrap_err();
        assert!(matches!(err, GuardrailError::UsageError(_)));
    }

    #[test]
    fn accepts_typical_ref_names() {
        assert!(validate_baseline_ref("main").is_ok());
        assert!(validate_baseline_ref("origin/main").is_ok());
        assert!(validate_baseline_ref("v1.2.3").is_ok());
        assert!(validate_baseline_ref("feature/106-foo").is_ok());
    }

    #[test]
    fn resolve_nonexistent_ref_propagates_error_not_panicking() {
        let dir =
            std::env::temp_dir().join(format!("guardrail-changeset-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut cmd = std::process::Command::new("git");
        cmd.arg("init").arg("-q").current_dir(&dir);
        // 祖先プロセス（lefthook の pre-push フック等）から継承された
        // `GIT_DIR`／`GIT_WORK_TREE` 等を除去する（`exclusion_match::git_command`
        // と同一方針。除去しないとフィクスチャ用一時リポジトリの隔離が壊れる）。
        for (key, _) in std::env::vars() {
            if key.starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
        let status = cmd.status().unwrap();
        assert!(status.success());

        let err = resolve_baseline_commit(&dir, "this-ref-does-not-exist").unwrap_err();
        assert!(matches!(err, GuardrailError::DiffFailed { .. }));

        std::fs::remove_dir_all(&dir).ok();
    }
}
