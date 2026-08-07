//! 試行回数・所要時間・判断根拠を保持する報告構造体（TASK-3.1a・イシュー #132）。
//!
//! [`crate::runner::SelfRepairLoop::run`] が返す [`LoopReport`] は、各試行の
//! 記録 [`AttemptRecord`] の列と最終結論を保持する。構造化ログとしての
//! シリアライズ・改竄検知可能な記録形式（署名・追記専用ストレージ等）は
//! TASK-3.4（イシュー #145）のスコープであり、本モジュールはその入力と
//! なる seam（値を保持するだけの構造体）までを提供する
//! （`.claude/rules/out-of-scope-tracking.md` 準拠。`.claude/rules/security.md`:
//! 試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を追跡可能にする、
//! という要求への対応は #145 側で行う）。

use std::time::Duration;

use crate::error::SelfRepairError;
use crate::kind::RepairKind;
use crate::outcome::LoopOutcome;

/// 1 試行（[`crate::stages::FixGenerator::generate`] 1 回分）の記録。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptRecord {
    /// 試行番号（1 始まり）。
    pub attempt: u32,
    /// この試行に要した所要時間（修正生成開始〜判断確定まで）。
    pub duration: Duration,
    /// この試行がどこまで到達し、なぜ採用に至らなかった／至ったか。
    pub outcome: AttemptOutcome,
}

/// 1 試行の到達段階・判断根拠。
///
/// `_ =>` を使わず全 variant を明示するため、[`LoopReport`] を読む側
/// （TASK-3.4 のログ出力・レビュー時の人間）は必ずどこで何が起きたかを
/// 判別できる（`.claude/rules/security.md` A05 と同じ fail-closed の考え方を
/// レポート表現にも適用する）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// 検証（[`crate::stages::VerificationGate`]）で不合格になった。
    VerificationFailed { reason: String },
    /// 検証は通過したが、取り込み判断が再試行可能な却下を返した。
    AdoptionRejectedRetryable { reason: String },
    /// 検証・取り込み判断の双方を通過し採用された（最終試行）。
    Adopted,
    /// 検証は通過したが、取り込み判断が人間レビューへ回すべきと判定した
    /// （最終試行。ループはこの時点で終了する）。`reason` は取り込み判断が
    /// 返したエスカレーション理由（`AdoptionRejectedRetryable`/
    /// `RejectedFinal` と同様に理由を保持する）。
    Escalated { reason: String },
    /// 再試行の余地なく却下が確定した（最終試行。ループはこの時点で終了する）。
    RejectedFinal { reason: String },
}

/// [`crate::runner::SelfRepairLoop::run`] 1 回分の全体報告。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopReport {
    /// 対象とした種別。
    pub kind: RepairKind,
    /// ループ全体の最終結論。
    pub outcome: LoopOutcome,
    /// 各試行の記録(検出段階のみで完了した場合は空)。
    pub attempts: Vec<AttemptRecord>,
    /// 検出開始から最終結論確定までの合計所要時間。
    pub total_duration: Duration,
}

impl LoopReport {
    /// 実施した試行回数（[`crate::stages::FixGenerator::generate`] の呼び出し
    /// 回数と一致する）。
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }
}

/// [`crate::runner::SelfRepairLoop::run`] が段階の実行自体の失敗
/// （[`SelfRepairError`]）で終了した場合の報告。
///
/// `Result<LoopReport, SelfRepairError>` のみでは
/// `fix_generator.generate` / `verification_gate.verify` / `adoption_judge.judge`
/// のいずれかが `Err` を返した瞬間、それまでにループが蓄積した試行記録
/// （`VerificationFailed`・`AdoptionRejectedRetryable` 等）が early return と
/// ともに失われる（v1 イシュー #40 レビュー指摘）。`.claude/rules/security.md`
/// の「ループ試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を
/// 追跡可能にする」という要求は、正常終了時だけでなくエラー終了時にも
/// 成り立つ必要があるため、エラー自体に加えてそれまでの [`AttemptRecord`]
/// 列を保持するラッパーを介して返す。
#[derive(Debug)]
pub struct LoopFailure {
    /// 段階の実行自体が失敗した理由。
    pub error: SelfRepairError,
    /// 失敗するまでに実施された試行の記録（[`LoopReport::attempts`] と同じ形式）。
    pub attempts: Vec<AttemptRecord>,
}

impl std::fmt::Display for LoopFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}（それまでの試行回数={}）",
            self.error,
            self.attempts.len()
        )
    }
}

impl std::error::Error for LoopFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
