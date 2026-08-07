//! バグ修正種別（[`RepairKind::BugFix`]）の検出・修正生成ロジック
//! （TASK-3.1b・イシュー #133・REQ-3。移植元は v1
//! `Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/bug_fix.rs`。
//! `docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//!
//! PoC-2 検証題材 (a)（`docs/spec/03-poc/poc-2-ai-self-maintenance/README.md`）の
//! 「活性化関数の取り違え（relu → sigmoid すり替え）を既知正解値との誤差検証
//! テストで検出 → 修正試行 1（誤った修正・検証不合格で却下）→ 修正試行 2
//! （正しい復元・検証通過）→ 取り込み」というループを、[`crate::stages`] の
//! trait 実装として写像する。
//!
//! # 新 API 差し替え（TASK-3.1b の実体）
//! v1 は `guardrail::exec::CommandRunner` に依存していたが、v2 `guardrail` に
//! 同モジュールは存在しない（`crate::exec` の doc 参照）。本ファイルの
//! `BugFixDetector` は [`crate::exec::CommandRunner`] へ trait 境界を差し替える。
//! 修正生成本体は [`crate::candidate::apply_candidate`] へ委譲する
//! （`BugFixFixGenerator`/[`crate::feature_addition::FeatureAdditionFixGenerator`]
//! の共通契約。候補存在確認 → baseline 復元 → 候補適用の順序を守る）。
//!
//! # 修正生成の決定性（設計判断）
//! TASK-3.1 の成果物は「ループ実行モジュール」であり、修正案を発想する AI
//! エージェント本体はループの外側（呼び出し元）にいる。本実装は PoC-2 題材 (a)
//! の試行列を [`crate::candidate::CandidateFix`] の列として注入できる決定的な
//! `FixGenerator` であり、ループ機構そのものの完走を検証する。実 AI 生成修正の
//! 動的取得は TASK-3.3 以降・自己修復ループ運用時のスコープ
//! （`.claude/rules/out-of-scope-tracking.md` 準拠）。

use std::collections::HashMap;
use std::path::PathBuf;

use crate::candidate::{CandidateFix, apply_candidate, validate_relative_path};
use crate::error::SelfRepairError;
use crate::exec::CommandRunner;
use crate::kind::RepairKind;
use crate::stages::{DetectionOutcome, Detector, Finding, FixGenerator, Proposal};

/// バグ修正種別の検出器。対象ワークスペースで `cargo test --release` を実行し、
/// 既知正解値テストの失敗を [`Finding`] として報告する。
#[derive(Debug)]
pub struct BugFixDetector<R: CommandRunner> {
    workspace: PathBuf,
    runner: R,
}

impl<R: CommandRunner> BugFixDetector<R> {
    pub fn new(workspace: impl Into<PathBuf>, runner: R) -> Self {
        BugFixDetector {
            workspace: workspace.into(),
            runner,
        }
    }
}

impl<R: CommandRunner> Detector for BugFixDetector<R> {
    fn detect(&self, kind: RepairKind) -> Result<DetectionOutcome, SelfRepairError> {
        // fail-closed: 他種別からの誤った呼び出しを NoActionNeeded に丸めない
        // （stages.rs の Detector 契約）。
        if kind != RepairKind::BugFix {
            return Err(SelfRepairError::Detection {
                kind: "bug_fix",
                reason: format!(
                    "BugFixDetector は RepairKind::BugFix 専用です（要求された種別={kind:?}）"
                ),
            });
        }

        let output = self
            .runner
            .run("cargo", &["test", "--release"], &self.workspace)
            .map_err(|error| SelfRepairError::Detection {
                kind: "bug_fix",
                reason: error.message().to_string(),
            })?;

        if output.success() {
            Ok(DetectionOutcome::NoActionNeeded)
        } else {
            Ok(DetectionOutcome::Finding(Finding::new(
                RepairKind::BugFix,
                format!(
                    "既知正解値テストが失敗しました（cargo test --release）: {}",
                    output.log_tail()
                ),
            )))
        }
    }
}

/// バグ修正種別の修正生成器。事前に与えられた候補修正列（[`CandidateFix`]）を
/// attempt 順に適用する決定的な実装（PoC-2 の「修正試行 1 → 2」を写像する）。
#[derive(Debug)]
pub struct BugFixFixGenerator {
    workspace: PathBuf,
    /// 候補列が参照する全ファイルの、構築時点（= バグ注入済みの検出対象状態）の
    /// 内容。各試行の開始前にこの内容へ復元してから候補を適用することで、
    /// 前試行の変更が次の試行に持ち越されないようにする
    /// （PoC-2「失敗時は却下して再試行」の写像）。
    baseline: HashMap<PathBuf, String>,
    candidates: Vec<CandidateFix>,
}

impl BugFixFixGenerator {
    /// `candidates` が参照する全ファイルの現在の内容を baseline として読み込み、
    /// 構築する。読み込み対象ファイルは workspace 内に実在している必要がある
    /// （検出対象ワークスペースは既にバグ注入済みの状態であることが前提）。
    pub fn new(
        workspace: impl Into<PathBuf>,
        candidates: Vec<CandidateFix>,
    ) -> Result<Self, SelfRepairError> {
        let workspace = workspace.into();
        let mut baseline: HashMap<PathBuf, String> = HashMap::new();

        for candidate in &candidates {
            for (rel_path, _) in &candidate.files {
                validate_relative_path(rel_path)
                    .map_err(|reason| SelfRepairError::FixGeneration { attempt: 0, reason })?;
                if baseline.contains_key(rel_path) {
                    continue;
                }
                let abs = workspace.join(rel_path);
                let content = std::fs::read_to_string(&abs).map_err(|source| {
                    SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "baseline 読み込みに失敗しました（path={}）: {source}",
                            rel_path.display()
                        ),
                    }
                })?;
                baseline.insert(rel_path.clone(), content);
            }
        }

        Ok(BugFixFixGenerator {
            workspace,
            baseline,
            candidates,
        })
    }
}

impl FixGenerator for BugFixFixGenerator {
    fn generate(&self, _finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
        // 候補存在確認 → baseline 復元 → 候補適用の順序契約・エラー文言は
        // `crate::candidate::apply_candidate` に一本化済み。
        apply_candidate(&self.workspace, &self.baseline, &self.candidates, attempt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::ExecError;
    use crate::test_support::{
        ScriptedCommand, failing_test_response, passing_test_response, unique_temp_dir,
        write_workspace_file,
    };

    #[test]
    fn detect_returns_no_action_needed_when_tests_pass() {
        let detector = BugFixDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![passing_test_response()]),
        );
        let outcome = detector
            .detect(RepairKind::BugFix)
            .expect("検出段階は失敗しない");
        assert_eq!(outcome, DetectionOutcome::NoActionNeeded);
    }

    #[test]
    fn detect_returns_finding_when_tests_fail() {
        let detector = BugFixDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![failing_test_response(
                "test relu_matches_known_values ... FAILED",
            )]),
        );
        let outcome = detector
            .detect(RepairKind::BugFix)
            .expect("検出段階は失敗しない");
        match outcome {
            DetectionOutcome::Finding(finding) => {
                assert_eq!(finding.kind(), RepairKind::BugFix);
                assert!(finding.summary.contains("relu_matches_known_values"));
            }
            other => panic!("Finding が返るべき: {other:?}"),
        }
    }

    #[test]
    fn detect_fails_closed_on_spawn_error() {
        let detector = BugFixDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![(
                ("cargo", &["test", "--release"]),
                Err(ExecError::new("コマンド未インストール（scripted failure）")),
            )]),
        );
        let error = detector
            .detect(RepairKind::BugFix)
            .expect_err("spawn 失敗は SelfRepairError::Detection を返すこと");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn detect_rejects_other_repair_kinds() {
        // fail-closed 契約: BugFix 以外の要求を NoActionNeeded に丸めない。
        let detector = BugFixDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![passing_test_response()]),
        );
        let error = detector
            .detect(RepairKind::PerfRegression)
            .expect_err("他種別の要求は Detection エラーを返すこと");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn generate_applies_candidates_in_attempt_order_and_restores_baseline_between_attempts() {
        let dir = unique_temp_dir(
            "bug_fix_generate_applies_candidates_in_attempt_order_and_restores_baseline_between_attempts",
        );
        write_workspace_file(&dir, "src/lib.rs", "baseline content");

        let candidates = vec![
            CandidateFix {
                description: "試行1".to_string(),
                files: vec![(PathBuf::from("src/lib.rs"), "attempt1 content".to_string())],
            },
            CandidateFix {
                description: "試行2".to_string(),
                files: vec![(PathBuf::from("src/lib.rs"), "attempt2 content".to_string())],
            },
        ];
        let generator = BugFixFixGenerator::new(&dir, candidates).expect("FixGenerator 構築に失敗");
        let finding = Finding::new(RepairKind::BugFix, "dummy");

        let proposal1 = generator.generate(&finding, 1).expect("試行1は成功");
        assert_eq!(proposal1.description, "試行1");
        assert_eq!(
            std::fs::read_to_string(dir.join("src/lib.rs")).unwrap(),
            "attempt1 content"
        );

        // 試行2の直前に baseline（"baseline content"）へ復元されてから
        // 候補2（"attempt2 content"）が適用されることを確認する。
        let proposal2 = generator.generate(&finding, 2).expect("試行2は成功");
        assert_eq!(proposal2.description, "試行2");
        assert_eq!(
            std::fs::read_to_string(dir.join("src/lib.rs")).unwrap(),
            "attempt2 content"
        );
    }

    #[test]
    fn generate_fails_closed_when_candidates_exhausted() {
        let dir = unique_temp_dir("bug_fix_generate_fails_closed_when_candidates_exhausted");
        write_workspace_file(&dir, "src/lib.rs", "baseline content");

        let candidates = vec![CandidateFix {
            description: "唯一の候補".to_string(),
            files: vec![(PathBuf::from("src/lib.rs"), "attempt1 content".to_string())],
        }];
        let generator = BugFixFixGenerator::new(&dir, candidates).expect("FixGenerator 構築に失敗");
        let finding = Finding::new(RepairKind::BugFix, "dummy");

        // 試行1で唯一の候補（"attempt1 content"）を適用し、直前の適用結果として
        // ワークツリーに残す。
        generator.generate(&finding, 1).expect("試行1は成功");
        assert_eq!(
            std::fs::read_to_string(dir.join("src/lib.rs")).unwrap(),
            "attempt1 content"
        );

        // 試行2は候補が尽きているため FixGeneration エラーを返す。この hard-error
        // 経路では baseline 復元（ファイル書き換え）が発生してはならない。
        let error = generator
            .generate(&finding, 2)
            .expect_err("候補が尽きた場合は FixGeneration エラーを返すこと");
        assert!(matches!(
            error,
            SelfRepairError::FixGeneration { attempt: 2, .. }
        ));
        assert_eq!(
            std::fs::read_to_string(dir.join("src/lib.rs")).unwrap(),
            "attempt1 content",
            "候補枯渇時に baseline 復元によるファイル書き換えが発生してはならない"
        );
    }

    #[test]
    fn new_rejects_candidate_paths_outside_workspace() {
        let dir = unique_temp_dir("bug_fix_new_rejects_candidate_paths_outside_workspace");
        write_workspace_file(&dir, "src/lib.rs", "baseline content");

        let candidates = vec![CandidateFix {
            description: "workspace 外書き込みを試みる不正な候補".to_string(),
            files: vec![(
                PathBuf::from("../outside.rs"),
                "malicious content".to_string(),
            )],
        }];
        let error = BugFixFixGenerator::new(&dir, candidates)
            .expect_err("workspace 外パスは拒否されること");
        assert!(matches!(
            error,
            SelfRepairError::FixGeneration { attempt: 0, .. }
        ));
    }
}
