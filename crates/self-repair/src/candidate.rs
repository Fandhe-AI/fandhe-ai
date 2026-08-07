//! 候補修正（[`CandidateFix`]）の表現と、attempt 順適用の共通ロジック
//! （TASK-3.1b・イシュー #133、TASK-3.1c・イシュー #134・REQ-3。移植元は v1
//! `Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/candidate.rs`。
//! `docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//!
//! [`crate::bug_fix::BugFixFixGenerator`]・
//! [`crate::feature_addition::FeatureAdditionFixGenerator`] の双方が「候補存在
//! 確認 → baseline 復元 → 候補適用」という同一の適用契約（候補枯渇の
//! hard-error 経路では baseline 復元によるファイル書き換えを発生させない）を
//! 持つ。両者から [`apply_candidate`] として共通利用する（構築時検証は種別
//! ごとに異なるため各モジュールに残す。`feature_addition.rs` の
//! `is_manifest_path` 等）。
//!
//! 種別を持たない汎用の [`crate::stages::FixGenerator`] 実装
//! [`CandidateFixGenerator`] も本モジュールが提供する（TASK-3.1c・#134）。
//! v1 の種別別 `FixGenerator`（`bug_fix.rs`・`feature_addition.rs`）は検出器
//! （種別ごとの `Detector`）と一体だったのに対し、決定的な候補列
//! （[`CandidateFix`]）さえ構築時に注入できれば種別非依存に組み立てられる
//! ループ利用者（`verify_gates`・`runner` の新 API 経路）向けに、[`apply_candidate`]
//! を直接ラップする薄い実装として追加する。既存の種別別 `FixGenerator`
//! （`bug_fix.rs`・`feature_addition.rs`）を置き換えるものではなく、
//! いずれも本モジュールの [`apply_candidate`]／[`validate_relative_path`] を
//! 共通基盤として利用する。
//!
//! # A03（インジェクション）対応
//! [`validate_relative_path`] は絶対パス・`..` 成分を含むパスを構築時に
//! 拒否する。これにより候補修正が workspace 外のファイルへ書き込む経路を
//! 閉じる（`.claude/rules/security.md` A03）。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::SelfRepairError;
use crate::stages::{Finding, FixGenerator, Proposal};

/// workspace 相対パスの検証（絶対パス・`..` 成分を拒否）。
///
/// `path` が絶対パスの場合、または `..`（親ディレクトリ参照）成分を含む
/// 場合は `Err` を返す。workspace 外への書き込みを構築時に封じる
/// （`apply_candidate` がファイルシステムへ触れる前に必ず経由させる。
/// `bug_fix.rs`/`feature_addition.rs` の `new` も構築時検証として呼ぶ）。
pub fn validate_relative_path(path: &Path) -> Result<(), String> {
    if path.is_absolute() {
        return Err(format!(
            "候補修正のパスは workspace 相対パスである必要があります（絶対パス: {}）",
            path.display()
        ));
    }
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(format!(
                "候補修正のパスに親ディレクトリ参照（..）を含めることはできません: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// 1 候補修正 = workspace 相対パスと置換後内容の組。
///
/// `description`・`files` とも [`crate::stages::Proposal::description`] と
/// 同じ理由（不変条件を持たない値）で `pub` フィールドとする。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateFix {
    /// 人間可読な修正内容の要約（[`Proposal::description`] へそのまま渡る）。
    pub description: String,
    /// (workspace 相対パス, 置換後の全内容) の列。
    pub files: Vec<(PathBuf, String)>,
}

/// `candidates[attempt - 1]` を `workspace` へ適用し [`Proposal`] を返す共通本体。
///
/// 手順（[`crate::bug_fix::BugFixFixGenerator::generate`]・
/// [`crate::feature_addition::FeatureAdditionFixGenerator::generate`]・
/// [`CandidateFixGenerator::generate`] が委譲する）:
/// 1. 候補（`attempt` 番目、1 始まり）の存在確認 — ファイルシステム副作用
///    より前に行う。候補枯渇（`attempt` が候補数を超える）を検出した場合は
///    baseline 復元を一切行わずに `Err` を返す。これを候補確認より後に
///    baseline 復元を行う実装にすると、候補枯渇時にも復元処理が実行され
///    「本来何もしないはずの hard-error 経路でファイルが書き換わる」
///    副作用が生じる（v1 PR #172 の指摘）。
/// 2. baseline 復元（前回試行の書き換えを白紙化）
/// 3. 候補適用（今回の書き換え）
pub fn apply_candidate(
    workspace: &Path,
    baseline: &HashMap<PathBuf, String>,
    candidates: &[CandidateFix],
    attempt: u32,
) -> Result<Proposal, SelfRepairError> {
    // 1. 候補存在確認（副作用より前）。
    let index =
        (attempt as usize)
            .checked_sub(1)
            .ok_or_else(|| SelfRepairError::FixGeneration {
                attempt,
                reason: "attempt は 1 始まりである必要があります".to_string(),
            })?;
    let candidate = candidates
        .get(index)
        .ok_or_else(|| SelfRepairError::FixGeneration {
            attempt,
            reason: format!(
                "候補修正が尽きました（attempt={attempt}・候補数={}）",
                candidates.len()
            ),
        })?;

    // 2. baseline 復元。
    for (relative_path, original_content) in baseline {
        let absolute_path = workspace.join(relative_path);
        fs::write(&absolute_path, original_content).map_err(|error| {
            SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "baseline 復元に失敗しました（{}）: {error}",
                    relative_path.display()
                ),
            }
        })?;
    }

    // 3. 候補適用。
    for (relative_path, content) in &candidate.files {
        let absolute_path = workspace.join(relative_path);
        fs::write(&absolute_path, content).map_err(|error| SelfRepairError::FixGeneration {
            attempt,
            reason: format!(
                "候補修正の適用に失敗しました（{}）: {error}",
                relative_path.display()
            ),
        })?;
    }

    Ok(Proposal {
        attempt,
        description: candidate.description.clone(),
    })
}

/// [`FixGenerator`] の種別非依存な実装。
///
/// 構築時に決定的な候補列（`candidates`）を受け取り、`generate` 呼び出しごと
/// に `attempt` 番目の候補を [`apply_candidate`] 経由でファイルシステムへ
/// 適用する。種別別の候補**選定**ロジック（バグ修正ならどのテスト失敗から
/// どんな差分を作るか等）は持たず、あくまで「候補列 → 適用」の機械的な
/// 変換のみを担う（モジュール冒頭ドキュメント参照）。
pub struct CandidateFixGenerator {
    workspace: PathBuf,
    baseline: HashMap<PathBuf, String>,
    candidates: Vec<CandidateFix>,
}

impl CandidateFixGenerator {
    /// `workspace` 配下で `candidates` を試行順に適用する
    /// [`CandidateFixGenerator`] を構築する。
    ///
    /// 構築時に全候補パスを [`validate_relative_path`] で検証し
    /// （A03 対応。ファイルシステムへ触れる前に構築自体を失敗させる）、
    /// 候補が参照する全ファイルの現内容を baseline としてスナップショット
    /// する（`apply_candidate` の再試行時復元に使う）。
    pub fn new(workspace: PathBuf, candidates: Vec<CandidateFix>) -> Result<Self, SelfRepairError> {
        let mut baseline = HashMap::new();
        for candidate in &candidates {
            for (relative_path, _content) in &candidate.files {
                validate_relative_path(relative_path)
                    .map_err(|reason| SelfRepairError::FixGeneration { attempt: 0, reason })?;
                if baseline.contains_key(relative_path) {
                    continue;
                }
                let absolute_path = workspace.join(relative_path);
                let original_content = fs::read_to_string(&absolute_path).map_err(|error| {
                    SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "baseline スナップショット取得に失敗しました（{}）: {error}",
                            relative_path.display()
                        ),
                    }
                })?;
                baseline.insert(relative_path.clone(), original_content);
            }
        }
        Ok(CandidateFixGenerator {
            workspace,
            baseline,
            candidates,
        })
    }
}

impl FixGenerator for CandidateFixGenerator {
    fn generate(&self, _finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
        // `Proposal.attempt` にループから渡された `attempt` をそのまま
        // 設定する（`apply_candidate` が返す `Proposal` がこの契約を満たす。
        // `runner.rs` の fail-closed 検査契約〈attempt 番号の単一の真実源〉
        // を満たすため、ここで値をすり替えない）。
        apply_candidate(&self.workspace, &self.baseline, &self.candidates, attempt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::RepairKind;
    use std::fs;

    fn write_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create_dir_all should succeed in test setup");
        }
        fs::write(path, content).expect("write should succeed in test setup");
    }

    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "self-repair-candidate-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create_dir_all should succeed in test setup");
        dir
    }

    #[test]
    fn validate_relative_path_rejects_absolute_path() {
        let result = validate_relative_path(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_relative_path_rejects_parent_dir_traversal() {
        let result = validate_relative_path(Path::new("../outside/file.rs"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_relative_path_accepts_plain_relative_path() {
        let result = validate_relative_path(Path::new("src/lib.rs"));
        assert!(result.is_ok());
    }

    #[test]
    fn apply_candidate_applies_in_attempt_order() {
        let dir = temp_workspace("order");
        write_file(&dir, "target.txt", "original");

        let mut baseline = HashMap::new();
        baseline.insert(PathBuf::from("target.txt"), "original".to_string());

        let candidates = vec![
            CandidateFix {
                description: "attempt 1".to_string(),
                files: vec![(PathBuf::from("target.txt"), "fix-1".to_string())],
            },
            CandidateFix {
                description: "attempt 2".to_string(),
                files: vec![(PathBuf::from("target.txt"), "fix-2".to_string())],
            },
        ];

        let proposal = apply_candidate(&dir, &baseline, &candidates, 1)
            .expect("attempt 1 should apply successfully");
        assert_eq!(proposal.attempt, 1);
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "fix-1"
        );

        let proposal = apply_candidate(&dir, &baseline, &candidates, 2)
            .expect("attempt 2 should apply successfully");
        assert_eq!(proposal.attempt, 2);
        // baseline 復元後に候補 2 が適用されるため "fix-1" は残らない。
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "fix-2"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_exhaustion_does_not_touch_filesystem() {
        // 候補枯渇時（attempt が候補数を超える）は baseline 復元すら
        // 発生しないことを確認する（v1 PR #172 指摘の回帰防止）。
        let dir = temp_workspace("exhaustion");
        write_file(&dir, "target.txt", "untouched");

        let mut baseline = HashMap::new();
        baseline.insert(
            PathBuf::from("target.txt"),
            "should-not-be-written".to_string(),
        );

        let candidates = vec![CandidateFix {
            description: "only attempt".to_string(),
            files: vec![(PathBuf::from("target.txt"), "fix-1".to_string())],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 2);
        assert!(result.is_err());
        // baseline 復元が実行されていれば "should-not-be-written" になる。
        // 候補確認が先に走り Err を返すため、ファイル内容は変化しない。
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "untouched"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_rejects_attempt_zero() {
        let dir = temp_workspace("attempt-zero");
        let baseline = HashMap::new();
        let candidates: Vec<CandidateFix> = Vec::new();
        let result = apply_candidate(&dir, &baseline, &candidates, 0);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_fix_generator_retains_loop_attempt_number() {
        let dir = temp_workspace("generator");
        write_file(&dir, "target.txt", "original");

        let candidates = vec![CandidateFix {
            description: "fix".to_string(),
            files: vec![(PathBuf::from("target.txt"), "fixed".to_string())],
        }];

        let generator = CandidateFixGenerator::new(dir.clone(), candidates)
            .expect("generator construction should succeed");
        let finding = Finding::new(RepairKind::BugFix, "dummy finding");
        let proposal = generator
            .generate(&finding, 1)
            .expect("generate should succeed");
        assert_eq!(proposal.attempt, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_fix_generator_construction_rejects_unsafe_path() {
        let dir = temp_workspace("unsafe-path");
        let candidates = vec![CandidateFix {
            description: "malicious".to_string(),
            files: vec![(PathBuf::from("../outside.txt"), "pwned".to_string())],
        }];
        let result = CandidateFixGenerator::new(dir.clone(), candidates);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
