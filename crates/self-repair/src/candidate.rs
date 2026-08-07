//! 候補修正（[`CandidateFix`]）の表現と、attempt 順適用の共通ロジック
//! （TASK-3.1b・イシュー #133・REQ-3。移植元は v1
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

use std::path::{Path, PathBuf};

use crate::error::SelfRepairError;
use crate::stages::Proposal;

/// 1 候補修正 = ファイルパス（workspace 相対）と置換後内容の組。
///
/// PoC-2 の「修正試行」1 回分に対応する（バグ修正・機能追加の双方で共用。
/// `docs/spec/03-poc/poc-2-ai-self-maintenance/README.md`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateFix {
    /// 修正内容の要約（[`crate::stages::Proposal::description`] にそのまま使われる）。
    pub description: String,
    /// 書き換え対象ファイル（workspace 相対パス）と新しい内容の組。
    pub files: Vec<(PathBuf, String)>,
}

/// `rel` が workspace 相対の安全なパスであることを検証する。
///
/// 絶対パス・`..` を含むパスは workspace 外への書き込みに使われうるため拒否する
/// （`.claude/rules/security.md` A03）。[`crate::bug_fix::BugFixFixGenerator::new`]・
/// [`crate::feature_addition::FeatureAdditionFixGenerator::new`] の双方が構築時
/// 検証として呼ぶ。
pub(crate) fn validate_relative_path(rel: &Path) -> Result<(), String> {
    if rel.is_absolute() {
        return Err(format!(
            "workspace 外パス（絶対パスは禁止）: {}",
            rel.display()
        ));
    }
    if rel
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(format!(
            "workspace 外パス（`..` を含むパスは禁止）: {}",
            rel.display()
        ));
    }
    Ok(())
}

/// `candidates[attempt - 1]` を `workspace` へ適用し [`Proposal`] を返す共通本体。
///
/// 手順（[`crate::bug_fix::BugFixFixGenerator::generate`]・
/// [`crate::feature_addition::FeatureAdditionFixGenerator::generate`] が
/// 委譲する）:
/// 1. 候補が存在するかを先に確認する（ファイルシステムへの副作用より前）。
///    存在確認を済ませてから baseline 復元へ進むことで、候補が尽きた
///    hard-error 経路（`max_attempts` 超過時）で baseline 復元による書き換えが
///    発生し、直前の適用済み修正が失われたまま変更後ツリーが残る事態を防ぐ。
/// 2. 前試行の変更を `baseline`（構築時点の内容）へ復元する。`baseline` に
///    含まれるパスは呼び出し元の `new` で検証済みであるため再検証しない。
/// 3. `candidate.files` を適用する。
pub(crate) fn apply_candidate(
    workspace: &Path,
    baseline: &[(PathBuf, String)],
    candidates: &[CandidateFix],
    attempt: u32,
) -> Result<Proposal, SelfRepairError> {
    let idx = attempt
        .checked_sub(1)
        .ok_or_else(|| SelfRepairError::FixGeneration {
            attempt,
            reason: "attempt は 1 始まりである必要があります".to_string(),
        })? as usize;

    let candidate = candidates
        .get(idx)
        .ok_or_else(|| SelfRepairError::FixGeneration {
            attempt,
            reason: format!(
                "候補修正が尽きました（試行={attempt}, 候補数={}）",
                candidates.len()
            ),
        })?;

    for (rel_path, content) in baseline {
        let abs = workspace.join(rel_path);
        std::fs::write(&abs, content).map_err(|source| SelfRepairError::FixGeneration {
            attempt,
            reason: format!(
                "baseline への復元に失敗しました（path={}）: {source}",
                rel_path.display()
            ),
        })?;
    }

    for (rel_path, content) in &candidate.files {
        let abs = workspace.join(rel_path);
        std::fs::write(&abs, content).map_err(|source| SelfRepairError::FixGeneration {
            attempt,
            reason: format!(
                "候補修正の適用に失敗しました（path={}）: {source}",
                rel_path.display()
            ),
        })?;
    }

    Ok(Proposal {
        attempt,
        description: candidate.description.clone(),
    })
}
