//! `self-repair` クレート全体で共有する型付きエラー（TASK-3.1a・イシュー #132）。
//!
//! `.claude/rules/coding-rust.md`「エラーは型付きエラーとし、本番経路で
//! `unwrap()` / `expect()` を使わない」に対応する。`thiserror` は
//! `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれず、依存追加は
//! ユーザー承認事項のため、v1 の `#[derive(thiserror::Error)]` を使わず
//! `std::fmt::Display` / `std::error::Error` を手書きで実装する
//! （`crates/guardrail/src/error.rs` と同一方針）。
//!
//! ここで扱うのは「段階の実行自体が失敗した」という**予期しないエラー**であり、
//! 「検証に落ちた」「取り込みを却下された」という**予期された否定的結果**とは
//! 区別する。後者は [`crate::outcome::LoopOutcome`] / 各段階の戻り値型
//! （[`crate::stages::VerificationOutcome`] / [`crate::outcome::AdoptionVerdict`]）
//! で表現し、`Result::Err` には含めない（`.claude/rules/security.md` A08:
//! 判定不能と却下を混同すると、判定不能を握りつぶして通過させる経路が
//! 生まれかねないため）。
use std::fmt;

/// 自己修復ループの各段階（[`crate::stages`] の 4 trait）の実行時エラー。
#[derive(Debug)]
pub enum SelfRepairError {
    /// [`crate::stages::Detector`] の実行自体が失敗した（例: ベースライン
    /// 情報の取得不能）。
    Detection { kind: &'static str, reason: String },

    /// [`crate::stages::FixGenerator`] の実行自体が失敗した。
    FixGeneration { attempt: u32, reason: String },

    /// [`crate::stages::VerificationGate`] の実行自体が失敗した（検証「不合格」
    /// ではなく、検証手段そのものが動作しなかったケース。例: ゲート実行系の
    /// spawn 失敗）。
    Verification { attempt: u32, reason: String },

    /// [`crate::stages::AdoptionJudge`] の実行自体が失敗した。
    Judgement { attempt: u32, reason: String },
}

impl fmt::Display for SelfRepairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelfRepairError::Detection { kind, reason } => {
                write!(f, "検出段階でエラーが発生しました（kind={kind}）: {reason}")
            }
            SelfRepairError::FixGeneration { attempt, reason } => {
                write!(
                    f,
                    "修正生成段階でエラーが発生しました（attempt={attempt}）: {reason}"
                )
            }
            SelfRepairError::Verification { attempt, reason } => {
                write!(
                    f,
                    "検証段階でエラーが発生しました（attempt={attempt}）: {reason}"
                )
            }
            SelfRepairError::Judgement { attempt, reason } => {
                write!(
                    f,
                    "取り込み判断段階でエラーが発生しました（attempt={attempt}）: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for SelfRepairError {}
