//! 公開 API 破壊の実測（TASK-4.1c・イシュー #106）。
//!
//! **変更された** `.rs` ファイル（`baseline` との差分に現れるファイルのみ。
//! `exclusion_match::changed_files` を再利用）の `pub fn`/`pub struct`/
//! `pub enum` 行シグネチャを、baseline 時点の内容と現作業木の内容とで行単位
//! 比較する。baseline に存在したシグネチャ行が現作業木から消えていれば
//! 「破壊的変更」とみなす（PoC-3 パリティ。`cargo public-api` 相当の意味論
//! 解析は行わない — スコープ外・`.claude/rules/out-of-scope-tracking.md`。
//! v1 `rust-ai-library-v1/crates/guardrail/src/checks/api_stability.rs`
//! 移植）。
//!
//! 差分に現れないファイルは baseline・現作業木で内容が同一であり、
//! シグネチャが消失しようがないため対象外にする（Review 指摘: baseline
//! ツリー全 `.rs` を毎回 `git show` すると `guardrail check` 1 回あたり
//! `git` 起動数がリポジトリ全体のファイル数に比例してしまう。変更ファイル
//! 数に比例させることで計測コストを差分規模に見合わせる）。
//!
//! 既知の限界（スコープ外として記録済み）:
//! - 引数名のみの変更・トレイト境界の変更等、シグネチャ行の文字列比較では
//!   検出できないシグネチャ変更（型は同じだが意味が変わる等）は対象外
//! - ファイルの改名（rename）: `git diff --name-only` の既定のリネーム検出
//!   閾値を超えなかった場合は旧パス（削除）・新パス（追加）の 2 エントリと
//!   して現れ、旧パスの全シグネチャが「消えた」として扱われる（escalate
//!   方向の誤検知のみで fail-open にはならない）。閾値を超えてリネームと
//!   認識された場合は新パスのみが現れ、旧パスは走査対象にならないため
//!   本チェックは発火しない（`exclusion_match::changed_files` は REQ-5
//!   除外リスト評価と共用のため、リネーム検出抑止のための変更は行わない）
//! - シグネチャと関数本体が同一物理行にある場合（1 行関数）、本体のみの
//!   変更もシグネチャ変更として検出される（行単位比較のため。escalate
//!   方向の誤検知のみで fail-open にはならない）

use std::path::Path;

use crate::error::GuardrailError;
use crate::exclusion_match;

/// シグネチャ行として扱う接頭辞（先頭空白除去後）。
const PUBLIC_SIGNATURE_PREFIXES: [&str; 3] = ["pub fn ", "pub struct ", "pub enum "];

/// `content` から公開 API シグネチャ行（トリム済み）を抽出する。
fn extract_public_signatures(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|line| line.trim())
        .filter(|line| {
            PUBLIC_SIGNATURE_PREFIXES
                .iter()
                .any(|prefix| line.starts_with(prefix))
        })
        .map(|line| line.to_string())
        .collect()
}

/// `baseline:file` の内容を取得する。`file` が `baseline` 時点に存在しない
/// （現作業木で新規追加されたファイル）場合は `Ok(None)` を返す
/// （新規ファイルは baseline 側にシグネチャを持ちようがなく「破壊」判定の
/// 対象になり得ないため）。
///
/// `baseline` の実在自体は呼び出し元（`check::run_measured` →
/// `changeset::resolve_baseline_commit`）が事前に確認済みの前提に立つ。
/// この前提のもとでは `git show <baseline>:<file>` の非ゼロ終了は
/// 「そのパスが baseline 時点に存在しない」以外の要因（`baseline` 自体が
/// 無効等）はほぼ生じないため、`DiffFailed` を「baseline に存在しない」
/// として扱う。前提が崩れる呼び出し順序変更を行う場合はこの判断の再検証が
/// 必要（`check.rs` の実行順序契約参照）。
fn show_file_at_baseline_if_present(
    repo_root: &Path,
    baseline: &str,
    file: &str,
) -> Result<Option<String>, GuardrailError> {
    match exclusion_match::run_git(repo_root, &["show", &format!("{baseline}:{file}")]) {
        Ok(content) => Ok(Some(content)),
        Err(GuardrailError::DiffFailed { .. }) => Ok(None),
        Err(other) => Err(other),
    }
}

/// `baseline` と現作業木を比較し、公開 API の破壊的変更（シグネチャ行の
/// 消失）を検出したら `true` を返す。
///
/// 走査対象は `exclusion_match::changed_files`（`baseline` との差分に現れる
/// ファイル。`Cargo.lock` 除外）のうち `.rs` ファイルのみ（モジュール冒頭
/// コメント参照）。`git` 呼び出しの失敗（`DiffFailed` 以外）は fail-closed
/// でそのまま伝播する（`false`＝「破壊なし」へ丸めない。
/// `.claude/rules/security.md` A08）。
pub(crate) fn api_broken(repo_root: &Path, baseline: &str) -> Result<bool, GuardrailError> {
    let changed_files = exclusion_match::changed_files(repo_root, baseline)?;
    for file in changed_files.iter().filter(|f| f.ends_with(".rs")) {
        let Some(baseline_content) = show_file_at_baseline_if_present(repo_root, baseline, file)?
        else {
            // baseline に存在しない＝現作業木での新規追加ファイル。
            // 消失し得るシグネチャがないため対象外。
            continue;
        };
        let baseline_sigs = extract_public_signatures(&baseline_content);
        if baseline_sigs.is_empty() {
            continue;
        }

        // ファイルが現作業木から削除されている場合は空文字列扱い（全シグネ
        // チャが消失＝破壊）。I/O エラー（権限等）は「読めなかった」を
        // 「削除された」と区別せず安全側（破壊あり方向）に倒す。
        let current_content = std::fs::read_to_string(repo_root.join(file)).unwrap_or_default();
        let current_sigs = extract_public_signatures(&current_content);

        if baseline_sigs.iter().any(|sig| !current_sigs.contains(sig)) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    fn run(cwd: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(cwd)
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
        let dir = std::env::temp_dir().join(format!(
            "guardrail-api-stability-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        run(&dir, &["init", "-q"]);
        dir
    }

    #[test]
    fn removing_pub_fn_is_detected_as_broken() {
        let dir = init_repo("remove-fn");
        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();

        assert!(api_broken(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn changing_signature_is_detected_as_broken() {
        let dir = init_repo("change-sig");
        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i64, b: i64) -> i64 { a + b }\n",
        )
        .unwrap();

        assert!(api_broken(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn adding_new_pub_fn_is_not_broken() {
        let dir = init_repo("add-fn");
        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
        )
        .unwrap();

        assert!(!api_broken(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn private_change_is_not_broken() {
        // シグネチャ行（`pub fn ...` の宣言行）自体は変更せず、本体・非公開
        // 関数のみを変更する（本チェックは行単位の文字列比較のため、
        // シグネチャと本体が同一行にあると本体の変更もシグネチャ変更として
        // 誤検知する。PoC-3 パリティの既知の限界＝モジュール冒頭コメント）。
        let dir = init_repo("private-change");
        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    helper(a, b)\n}\n\nfn helper(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    helper(a, b) + 1\n}\n\nfn helper(a: i32, b: i32) -> i32 { a + b - 1 }\n",
        )
        .unwrap();

        assert!(!api_broken(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deleted_file_with_pub_items_is_broken() {
        let dir = init_repo("delete-file");
        fs::write(dir.join("lib.rs"), "pub struct Foo;\n").unwrap();
        commit_all(&dir, "baseline");

        fs::remove_file(dir.join("lib.rs")).unwrap();

        assert!(api_broken(&dir, "HEAD").unwrap());
        fs::remove_dir_all(&dir).ok();
    }
}
