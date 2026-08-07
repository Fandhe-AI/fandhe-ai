//! `guardrail check` の CI 呼び出し契約: 終了コードの一元定義（TASK-4.1c・
//! イシュー #106）。
//!
//! CI（`.github/workflows/ci.yml`）・自己修復ループ（`self-repair`）の双方が
//! `guardrail` CLI の終了コードのみを見て後続分岐できるようにするための契約。
//!
//! [`crate::decision::Verdict`] → 終了コードの変換は本モジュールの
//! [`GuardrailExitCode::from_verdict`] **のみ**に閉じ込め、他の経路から `0`
//! を返せないようにする（`.claude/rules/security.md` A08: 自己修復ループが
//! 取り込む変更の判定を迂回する経路を作らない）。内部エラー（シグナル入力の
//! 欠落・JSON 解析失敗等）は「判定不能」であり、自動適用（0）でもエスカレー
//! ション（10）でも却下（20）でもない別区分（1）として明確に分離する。`2` は
//! clap の usage エラー（引数解析失敗。#104 管轄）の既定値であるため本 enum
//! の対象外とする。
//!
//! `guardrail eval`（`EvalExitCode`。終了コード 0/30/1）は TASK-4.2/4.3 系
//! （イシュー #108 以降）の管轄であり本 PR のスコープ外（計画 §7）。

use std::process::ExitCode;

use crate::decision::Verdict;

/// CI から見た `guardrail check` の終了コード契約（`docs/guardrail-self-repair-cli.md`
/// §2.3）。
///
/// | 値 | 意味 |
/// |----|------|
/// | 0  | 自動適用（`Verdict::AutoApply`） |
/// | 10 | エスカレーション（`Verdict::Escalate`） |
/// | 20 | 却下（`Verdict::Reject`） |
/// | 1  | 内部エラー（判定不能。シグナル入力・出力書き出しの失敗等） |
/// | 2  | （本 enum の対象外）clap の usage エラー既定値 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailExitCode {
    AutoApply,
    Escalate,
    Reject,
    InternalError,
}

impl GuardrailExitCode {
    /// [`Verdict`] から対応する終了コード区分を導出する唯一の変換点
    /// （判定を迂回する経路なし。`match` は網羅列挙とし `_ =>` を使わない）。
    pub fn from_verdict(verdict: Verdict) -> Self {
        match verdict {
            Verdict::AutoApply => GuardrailExitCode::AutoApply,
            Verdict::Escalate => GuardrailExitCode::Escalate,
            Verdict::Reject => GuardrailExitCode::Reject,
        }
    }

    /// 実際のプロセス終了コードへの変換（CLI 層〈#104〉の `main` 戻り値に使う
    /// 想定の値）。
    pub fn as_u8(self) -> u8 {
        match self {
            GuardrailExitCode::AutoApply => 0,
            GuardrailExitCode::Escalate => 10,
            GuardrailExitCode::Reject => 20,
            GuardrailExitCode::InternalError => 1,
        }
    }

    /// `std::process::ExitCode` への変換（`fn main() -> ExitCode` 形式で
    /// CLI 層が返す想定。プロセス終了コード自体は `as_u8()` と同一）。
    pub fn to_process_exit_code(self) -> ExitCode {
        ExitCode::from(self.as_u8())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_apply_is_exit_zero() {
        assert_eq!(
            GuardrailExitCode::from_verdict(Verdict::AutoApply).as_u8(),
            0
        );
    }

    #[test]
    fn escalate_is_exit_ten() {
        assert_eq!(
            GuardrailExitCode::from_verdict(Verdict::Escalate).as_u8(),
            10
        );
    }

    #[test]
    fn reject_is_exit_twenty() {
        assert_eq!(GuardrailExitCode::from_verdict(Verdict::Reject).as_u8(), 20);
    }

    #[test]
    fn internal_error_is_exit_one() {
        assert_eq!(GuardrailExitCode::InternalError.as_u8(), 1);
    }

    #[test]
    fn all_four_variants_have_distinct_codes() {
        let codes = [
            GuardrailExitCode::AutoApply.as_u8(),
            GuardrailExitCode::Escalate.as_u8(),
            GuardrailExitCode::Reject.as_u8(),
            GuardrailExitCode::InternalError.as_u8(),
        ];
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(codes[i], codes[j], "終了コードが重複しています: {codes:?}");
            }
        }
    }

    /// `1`（内部エラー）が判定 3 分岐（0/10/20）のいずれとも重複しないこと
    /// （判定不能が自動適用〈0〉へ倒れない fail-closed 契約の固定）。
    #[test]
    fn internal_error_never_collides_with_a_verdict_code() {
        for verdict in [Verdict::AutoApply, Verdict::Escalate, Verdict::Reject] {
            assert_ne!(
                GuardrailExitCode::InternalError.as_u8(),
                GuardrailExitCode::from_verdict(verdict).as_u8()
            );
        }
    }
}
