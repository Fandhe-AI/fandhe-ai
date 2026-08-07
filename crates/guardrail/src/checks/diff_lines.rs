//! 変更行数の実測（TASK-4.1c・イシュー #106）。
//!
//! `git diff --numstat` の追加・削除行数を合算する（v1
//! `rust-ai-library-v1/crates/guardrail/src/checks/diff_lines.rs` 移植）。
//! `Cargo.lock` は依存解決の機械的差分であり `decision::Reason::LinesMaxExceeded`
//! の実質を歪めるため除外する（`exclusion_match::changed_files` と同じ方針）。

use std::path::Path;

use crate::error::GuardrailError;
use crate::exclusion_match;

/// `baseline` と現作業木との差分行数（追加＋削除の合算）を返す。
///
/// バイナリファイル（`--numstat` が `-\t-\t<path>` を出力する）は集計に
/// 含めない（`"-".parse::<u64>()` は失敗するため `0` として扱う＝安全側。
/// 変更行数を過小評価する方向にしか作用せず、逆方向〈過大評価によるエスカ
/// レーション過多〉ではない点に注意。バイナリ変更の検知自体は本チェックの
/// スコープ外）。
pub(crate) fn lines_changed(repo_root: &Path, baseline: &str) -> Result<u64, GuardrailError> {
    let stdout = exclusion_match::run_git(
        repo_root,
        &["diff", "--numstat", baseline, "--", ".", ":!Cargo.lock"],
    )?;

    let mut total: u64 = 0;
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let added = parts.next().unwrap_or("0").parse::<u64>().unwrap_or(0);
        let removed = parts.next().unwrap_or("0").parse::<u64>().unwrap_or(0);
        total += added + removed;
    }
    Ok(total)
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
            "guardrail-diff-lines-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        run(&dir, &["init", "-q"]);
        dir
    }

    #[test]
    fn sums_added_and_removed_lines_excluding_cargo_lock() {
        let dir = init_repo("basic");
        fs::write(dir.join("a.txt"), "line1\nline2\nline3\n").unwrap();
        fs::write(dir.join("Cargo.lock"), "orig\n").unwrap();
        commit_all(&dir, "baseline");

        // a.txt: 1 行削除・2 行追加（3 変更行）。Cargo.lock は除外対象。
        fs::write(dir.join("a.txt"), "line1\nline2-changed\nline3\nline4\n").unwrap();
        fs::write(dir.join("Cargo.lock"), "orig\nnew-dep\n").unwrap();

        let changed = lines_changed(&dir, "HEAD").unwrap();
        assert_eq!(changed, 3, "Cargo.lock の変更は集計対象外のはず");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_changes_yields_zero() {
        let dir = init_repo("empty");
        fs::write(dir.join("a.txt"), "unchanged\n").unwrap();
        commit_all(&dir, "baseline");

        let changed = lines_changed(&dir, "HEAD").unwrap();
        assert_eq!(changed, 0);

        fs::remove_dir_all(&dir).ok();
    }
}
