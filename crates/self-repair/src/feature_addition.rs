//! 機能追加種別（[`RepairKind::FeatureAddition`]）の検出・修正生成ロジック
//! （TASK-3.1b・イシュー #133・REQ-3。移植元は v1
//! `Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/feature_addition.rs`。
//! `docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//!
//! PoC-2 検証題材 (c)（`docs/spec/03-poc/poc-2-ai-self-maintenance/README.md:56-66`）の
//! 「LeakyReLU を追加してほしい、という機能追加チケットに対し、実装試行 1
//! （誤った実装・検証不合格で却下）→ 実装試行 2（正しい合成実装・検証通過）→
//! 取り込み」というループを、[`crate::stages`] の trait 実装として写像する。
//!
//! # 新 API 差し替え・[`crate::bug_fix`] との関係
//! `bug_fix.rs` と同じく、コマンド実行は [`crate::exec::CommandRunner`] へ
//! 差し替える（`crate::exec` の doc 参照）。`generate` 本体は
//! [`crate::candidate::apply_candidate`] へ委譲する
//! （[`crate::bug_fix::BugFixFixGenerator`] との共通契約。構築時検証（スコープ
//! 限定チェック）は種別ごとに異なるため統合しない）。
//!
//! # スコープ限定（受け入れ条件「対象範囲が既存組み込み演算の合成に限定」）
//! [`FeatureAdditionFixGenerator`] の構築時検証で機械的に強制する: (1) workspace
//! 外パス（絶対パス・`..` 含み）、(2) baseline に存在しないファイルへの
//! 書き込み（新規ファイル・新規モジュール追加の禁止）、(3) `Cargo.toml` の
//! 書き換え（依存クレート追加の禁止）、の 3 つを拒否する。
//!
//! v1 では (3) を「既存組み込み演算の合成に限定」という設計制約の一部として
//! 説明していたが、v2 では REQ-1/REQ-5「依存の追加・更新は人間承認必須」
//! （`.claude/rules/deps-policy.md`）の機械的強制として位置づけ直す
//! （実装計画セクション 3）。学習可能パラメータを持つ新規レイヤーの追加は
//! この合成の範囲を超えるため対象外である（PoC-2 発見事項 5）。この境界の
//! 正式なドキュメント化は後続タスクのスコープ（out-of-scope-tracking.md 準拠）。

use std::path::{Path, PathBuf};

use crate::candidate::{CandidateFix, apply_candidate, validate_relative_path};
use crate::error::SelfRepairError;
use crate::exec::CommandRunner;
use crate::kind::RepairKind;
use crate::stages::{DetectionOutcome, Detector, Finding, FixGenerator, Proposal};

/// 機能追加種別の検出器。対象ワークスペースで `cargo test --release` を実行し、
/// 要求チケットに付随する受け入れ基準テストの失敗を [`Finding`] として報告する。
///
/// [`crate::bug_fix::BugFixDetector`] と同型の実装（実行するコマンドも同一）だが、
/// 「検出」の意味づけが異なる: バグ修正は「既知正解値からのズレ」を検出するのに
/// 対し、機能追加は「受け入れ基準テストが通らない ＝ 要求機能が未充足」を検出
/// する。両者を同一の trait 実装として統合しない（種別ごとに `Finding` の summary
/// 文言・意味を明確に分ける。`kind` の自己申告検証は種別混同防止のため必須）。
#[derive(Debug)]
pub struct FeatureAdditionDetector<R: CommandRunner> {
    workspace: PathBuf,
    runner: R,
}

impl<R: CommandRunner> FeatureAdditionDetector<R> {
    pub fn new(workspace: impl Into<PathBuf>, runner: R) -> Self {
        FeatureAdditionDetector {
            workspace: workspace.into(),
            runner,
        }
    }
}

impl<R: CommandRunner> Detector for FeatureAdditionDetector<R> {
    fn detect(&self, kind: RepairKind) -> Result<DetectionOutcome, SelfRepairError> {
        // fail-closed: 他種別からの誤った呼び出しを NoActionNeeded に丸めない
        // （`BugFixDetector` と同じ方針）。
        if kind != RepairKind::FeatureAddition {
            return Err(SelfRepairError::Detection {
                kind: "feature_addition",
                reason: format!(
                    "FeatureAdditionDetector は RepairKind::FeatureAddition 専用です（要求された種別={kind:?}）"
                ),
            });
        }

        let output = self
            .runner
            .run("cargo", &["test", "--release"], &self.workspace)
            .map_err(|reason| SelfRepairError::Detection {
                kind: "feature_addition",
                reason,
            })?;

        if output.success() {
            Ok(DetectionOutcome::NoActionNeeded)
        } else {
            Ok(DetectionOutcome::Finding(Finding::new(
                RepairKind::FeatureAddition,
                format!(
                    "受け入れ基準テストが失敗しました（cargo test --release ＝要求機能が未充足）: {}",
                    output.log_tail()
                ),
            )))
        }
    }
}

/// 候補が参照するファイル名が `Cargo.toml`（大文字小文字を区別しない比較。
/// macOS 既定の APFS は大文字小文字を区別しないため、ファイルシステム非依存で
/// 判定する）であるかを判定する。
///
/// 依存クレートの追加は `Cargo.toml` の書き換えを伴うため、これを拒否することで
/// REQ-1/REQ-5「依存の追加・更新は人間承認必須」を機械的に強制する
/// （モジュール冒頭コメント参照）。
fn is_manifest_path(rel: &Path) -> bool {
    rel.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("Cargo.toml"))
}

/// 機能追加種別の修正生成器。事前に与えられた候補修正列
/// （[`CandidateFix`]）を attempt 順に適用する決定的な実装（PoC-2 題材 (c) の
/// 「実装試行 1 → 2」を写像する。[`crate::bug_fix::BugFixFixGenerator`] と
/// 同じ生成契約: 候補存在確認 → baseline 復元 → 候補適用の順序を守る）。
#[derive(Debug)]
pub struct FeatureAdditionFixGenerator {
    workspace: PathBuf,
    /// 候補列が参照する全ファイルの、構築時点（＝機能未追加状態）の内容。
    /// 各試行の開始前にこの内容へ復元してから候補を適用することで、前試行の
    /// 変更が次の試行に持ち越されないようにする（`BugFixFixGenerator` と同じ
    /// 設計）。
    baseline: Vec<(PathBuf, String)>,
    candidates: Vec<CandidateFix>,
}

impl FeatureAdditionFixGenerator {
    /// `candidates` が参照する全ファイルを検証・baseline として読み込み、構築する。
    ///
    /// 構築時検証（上から順に適用）:
    /// 1. workspace 相対の安全なパスであること（絶対パス・`..` を拒否）
    /// 2. `Cargo.toml` を書き換え対象にしないこと（依存クレート追加の禁止）
    /// 3. baseline（workspace 内の現在の内容）に実在するファイルであること
    ///    （新規ファイル・新規モジュール追加の禁止 ＝ 既存モジュール内の合成
    ///    実装への限定）
    ///
    /// 検証 1・2 は `read_to_string` を試みる前に明示チェックとして行う。
    /// 検証 3（新規ファイル拒否）を `read_to_string` の失敗に委ねると、
    /// エラー文言が「baseline 読み込みに失敗しました」という I/O エラー由来の
    /// ものになり、「新規ファイル追加は対象外」というスコープ違反の意図が
    /// 埋もれてしまうため、専用の存在チェックを先に行う。
    pub fn new(
        workspace: impl Into<PathBuf>,
        candidates: Vec<CandidateFix>,
    ) -> Result<Self, SelfRepairError> {
        let workspace = workspace.into();
        let mut baseline: Vec<(PathBuf, String)> = Vec::new();

        for candidate in &candidates {
            for (rel_path, _) in &candidate.files {
                validate_relative_path(rel_path)
                    .map_err(|reason| SelfRepairError::FixGeneration { attempt: 0, reason })?;

                if is_manifest_path(rel_path) {
                    return Err(SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "Cargo.toml の書き換えは対象外です（依存クレート追加は人間承認必須。deps-policy.md）: {}",
                            rel_path.display()
                        ),
                    });
                }

                if baseline.iter().any(|(p, _)| p == rel_path) {
                    continue;
                }

                let abs = workspace.join(rel_path);
                if !abs.is_file() {
                    return Err(SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "新規ファイル追加は対象外です（既存モジュール内の合成実装に限定。baseline に実在しないファイル）: {}",
                            rel_path.display()
                        ),
                    });
                }

                let content = std::fs::read_to_string(&abs).map_err(|source| {
                    SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "baseline 読み込みに失敗しました（path={}）: {source}",
                            rel_path.display()
                        ),
                    }
                })?;
                baseline.push((rel_path.clone(), content));
            }
        }

        Ok(FeatureAdditionFixGenerator {
            workspace,
            baseline,
            candidates,
        })
    }
}

impl FixGenerator for FeatureAdditionFixGenerator {
    fn generate(&self, _finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
        // 候補存在確認 → baseline 復元 → 候補適用の順序契約・エラー文言は
        // `crate::candidate::apply_candidate` に一本化済み。
        apply_candidate(&self.workspace, &self.baseline, &self.candidates, attempt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        ScriptedCommand, failing_test_response, passing_test_response, unique_temp_dir,
        write_workspace_file,
    };

    #[test]
    fn detect_returns_no_action_needed_when_tests_pass() {
        let detector = FeatureAdditionDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![passing_test_response()]),
        );
        let outcome = detector
            .detect(RepairKind::FeatureAddition)
            .expect("検出段階は失敗しない");
        assert_eq!(outcome, DetectionOutcome::NoActionNeeded);
    }

    #[test]
    fn detect_returns_finding_when_acceptance_test_fails() {
        let detector = FeatureAdditionDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![failing_test_response(
                "test leaky_relu_matches_known_values ... FAILED",
            )]),
        );
        let outcome = detector
            .detect(RepairKind::FeatureAddition)
            .expect("検出段階は失敗しない");
        match outcome {
            DetectionOutcome::Finding(finding) => {
                assert_eq!(finding.kind(), RepairKind::FeatureAddition);
                assert!(finding.summary.contains("leaky_relu_matches_known_values"));
            }
            other => panic!("Finding が返るべき: {other:?}"),
        }
    }

    #[test]
    fn detect_fails_closed_on_spawn_error() {
        let detector = FeatureAdditionDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![(
                ("cargo", &["test", "--release"]),
                Err("コマンド未インストール（scripted failure）".to_string()),
            )]),
        );
        let error = detector
            .detect(RepairKind::FeatureAddition)
            .expect_err("spawn 失敗は SelfRepairError::Detection を返すこと");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn detect_rejects_other_repair_kinds() {
        // fail-closed 契約: FeatureAddition 以外の要求を NoActionNeeded に丸めない。
        let detector = FeatureAdditionDetector::new(
            PathBuf::from("/does/not/matter"),
            ScriptedCommand::new(vec![passing_test_response()]),
        );
        let error = detector
            .detect(RepairKind::BugFix)
            .expect_err("他種別の要求は Detection エラーを返すこと");
        assert!(matches!(error, SelfRepairError::Detection { .. }));
    }

    #[test]
    fn generate_applies_candidates_in_attempt_order_and_restores_baseline_between_attempts() {
        let dir = unique_temp_dir(
            "feature_addition_generate_applies_candidates_in_attempt_order_and_restores_baseline_between_attempts",
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
        let generator =
            FeatureAdditionFixGenerator::new(&dir, candidates).expect("FixGenerator 構築に失敗");
        let finding = Finding::new(RepairKind::FeatureAddition, "dummy");

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
        let dir =
            unique_temp_dir("feature_addition_generate_fails_closed_when_candidates_exhausted");
        write_workspace_file(&dir, "src/lib.rs", "baseline content");

        let candidates = vec![CandidateFix {
            description: "唯一の候補".to_string(),
            files: vec![(PathBuf::from("src/lib.rs"), "attempt1 content".to_string())],
        }];
        let generator =
            FeatureAdditionFixGenerator::new(&dir, candidates).expect("FixGenerator 構築に失敗");
        let finding = Finding::new(RepairKind::FeatureAddition, "dummy");

        generator.generate(&finding, 1).expect("試行1は成功");
        assert_eq!(
            std::fs::read_to_string(dir.join("src/lib.rs")).unwrap(),
            "attempt1 content"
        );

        // 試行2は候補が尽きているため FixGeneration エラーを返す。この
        // hard-error 経路では baseline 復元（ファイル書き換え）が発生しては
        // ならない（`BugFixFixGenerator` と同じ契約）。
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
        let dir = unique_temp_dir("feature_addition_new_rejects_candidate_paths_outside_workspace");
        write_workspace_file(&dir, "src/lib.rs", "baseline content");

        let candidates = vec![CandidateFix {
            description: "workspace 外書き込みを試みる不正な候補".to_string(),
            files: vec![(
                PathBuf::from("../outside.rs"),
                "malicious content".to_string(),
            )],
        }];
        let error = FeatureAdditionFixGenerator::new(&dir, candidates)
            .expect_err("workspace 外パスは拒否されること");
        assert!(matches!(
            error,
            SelfRepairError::FixGeneration { attempt: 0, .. }
        ));
    }

    #[test]
    fn new_rejects_new_file_addition_not_present_in_baseline() {
        // 受け入れ条件: 新規ファイル追加候補は「既存モジュール内の合成実装」
        // の範囲を超えるため拒否する。
        let dir = unique_temp_dir(
            "feature_addition_new_rejects_new_file_addition_not_present_in_baseline",
        );
        write_workspace_file(&dir, "src/lib.rs", "baseline content");

        let candidates = vec![CandidateFix {
            description: "新規モジュール追加を試みる不正な候補".to_string(),
            files: vec![(
                PathBuf::from("src/new_module.rs"),
                "pub fn new_layer() {}".to_string(),
            )],
        }];
        let error = FeatureAdditionFixGenerator::new(&dir, candidates)
            .expect_err("baseline に存在しない新規ファイルは拒否されること");
        match error {
            SelfRepairError::FixGeneration { attempt: 0, reason } => {
                assert!(
                    reason.contains("新規ファイル追加は対象外"),
                    "エラー理由が新規ファイル拒否であることが明示されること: {reason}"
                );
            }
            other => panic!("FixGeneration エラーが返るべき: {other:?}"),
        }
    }

    #[test]
    fn new_rejects_cargo_toml_rewrite_candidate() {
        // 受け入れ条件: 依存クレート追加（Cargo.toml 書き換え）は人間承認必須
        // （deps-policy.md）であり本クレートは単独で許可しない。
        let dir = unique_temp_dir("feature_addition_new_rejects_cargo_toml_rewrite_candidate");
        write_workspace_file(&dir, "Cargo.toml", "[package]\nname = \"x\"");

        let candidates = vec![CandidateFix {
            description: "依存クレート追加を試みる不正な候補".to_string(),
            files: vec![(
                PathBuf::from("Cargo.toml"),
                "[dependencies]\nsome-crate = \"1.0\"".to_string(),
            )],
        }];
        let error = FeatureAdditionFixGenerator::new(&dir, candidates)
            .expect_err("Cargo.toml の書き換えは拒否されること");
        match error {
            SelfRepairError::FixGeneration { attempt: 0, reason } => {
                assert!(
                    reason.contains("Cargo.toml"),
                    "エラー理由が Cargo.toml 拒否であることが明示されること: {reason}"
                );
            }
            other => panic!("FixGeneration エラーが返るべき: {other:?}"),
        }
    }

    #[test]
    fn is_manifest_path_ignores_case() {
        // macOS（APFS）は大文字小文字を区別しないため、`is_manifest_path` は
        // ファイルシステムの実際の大小文字区別有無に依存せず、常に大文字小文字を
        // 無視して `Cargo.toml` を検出すること。
        assert!(is_manifest_path(Path::new("Cargo.toml")));
        assert!(is_manifest_path(Path::new("cargo.toml")));
        assert!(is_manifest_path(Path::new("CARGO.TOML")));
        assert!(is_manifest_path(Path::new("nested/dir/cargo.TOML")));
        assert!(!is_manifest_path(Path::new("Cargo.lock")));
        assert!(!is_manifest_path(Path::new("src/lib.rs")));
    }
}
