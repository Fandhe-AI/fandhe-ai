//! `guardrail check` の CI 呼び出し契約: 終了コードの一元定義。
//!
//! `docs/guardrail-self-repair-cli.md` 2.3 節「fail-closed 設計（A08）」が
//! 要求するとおり、[`crate::decision::Verdict`] / eval 合否から終了コードへの
//! 変換経路は本モジュール（`GuardrailExitCode::from_verdict` /
//! `EvalExitCode::from_pass`）のみとし、他の経路から `0`（自動適用・評価合格）
//! を返せないようにする。`main.rs` はこれらの関数を経由してのみプロセス
//! 終了コードを決定する（自己修復ループが取り込む変更の判定を迂回する経路を
//! 作らない。`.claude/rules/security.md` A08）。
//!
//! `self-repair` から lib として呼び出される際も同じ `Verdict` を経由する
//! （3.4 節「guardrail 連携方式」。サブプロセス起動ではなく lib 直接呼び出し）。
//! `Verdict` 自体の定義は判定ロジック本体（[`crate::decision`]。TASK-4.1c・
//! イシュー #106）が正本であり、本モジュールは終了コードへの写像のみを担う。
//!
//! | 値 | 意味 |
//! |---|---|
//! | `0`  | 自動適用（`Verdict::AutoApply`） |
//! | `10` | エスカレーション（`Verdict::Escalate`） |
//! | `20` | 却下（`Verdict::Reject`） |
//! | `1`  | 内部エラー（判定不能。シグナル入力・出力書き出しの失敗等） |
//! | `2`  | usage エラー（clap 相当。#104 管轄） |

use std::process::ExitCode;

use crate::decision::Verdict;

/// `guardrail check` の終了コード契約（2.3 節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailExitCode {
    AutoApply,
    Escalate,
    Reject,
    InternalError,
    UsageError,
}

impl GuardrailExitCode {
    /// [`Verdict`] から終了コード区分への唯一の変換経路（判定を迂回する経路
    /// なし。`match` は網羅列挙とし `_ =>` を使わない）。
    ///
    /// 判定ロジック（#105/#106）を経ずにこの関数を直接呼ぶ経路（`main.rs`
    /// の `run_check`）が、TASK-4.1a 段階では常に `Verdict::Escalate` を渡す
    /// ことで「判定不能時に自動適用へ倒れない」契約を骨格段階から満たす
    /// （計画 2.2 節）。
    pub fn from_verdict(verdict: Verdict) -> Self {
        match verdict {
            Verdict::AutoApply => GuardrailExitCode::AutoApply,
            Verdict::Escalate => GuardrailExitCode::Escalate,
            Verdict::Reject => GuardrailExitCode::Reject,
        }
    }

    /// プロセス終了コード（`u8`）へ変換する。`main.rs` の戻り値に用いる。
    pub fn as_u8(self) -> u8 {
        match self {
            GuardrailExitCode::AutoApply => 0,
            GuardrailExitCode::Escalate => 10,
            GuardrailExitCode::Reject => 20,
            GuardrailExitCode::InternalError => 1,
            GuardrailExitCode::UsageError => 2,
        }
    }

    pub fn into_process_exit_code(self) -> ExitCode {
        ExitCode::from(self.as_u8())
    }
}

/// `guardrail eval` の終了コード契約（2.3 節）。`check`（`GuardrailExitCode`）
/// と重複しない値を選定済みで、CI が両者を区別できる（30 = 閾値未達）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalExitCode {
    Pass,
    ThresholdNotMet,
    InternalError,
    UsageError,
}

impl EvalExitCode {
    /// 見逃し率 0% かつ誤検知率 30% 以下（REQ-4 受け入れ基準）の合否から
    /// 終了コード区分への唯一の変換経路。評価ロジック本体（[`crate::eval::run`]。
    /// TASK-4.3a・イシュー #115）が返す [`crate::eval::report::EvalReport::pass`]
    /// を `main.rs::run_eval` がそのまま渡す。
    pub fn from_pass(pass: bool) -> Self {
        if pass {
            EvalExitCode::Pass
        } else {
            EvalExitCode::ThresholdNotMet
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            EvalExitCode::Pass => 0,
            EvalExitCode::ThresholdNotMet => 30,
            EvalExitCode::InternalError => 1,
            EvalExitCode::UsageError => 2,
        }
    }

    pub fn into_process_exit_code(self) -> ExitCode {
        ExitCode::from(self.as_u8())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_verdict_maps_all_variants() {
        assert_eq!(
            GuardrailExitCode::from_verdict(Verdict::AutoApply).as_u8(),
            0
        );
        assert_eq!(
            GuardrailExitCode::from_verdict(Verdict::Escalate).as_u8(),
            10
        );
        assert_eq!(GuardrailExitCode::from_verdict(Verdict::Reject).as_u8(), 20);
    }

    #[test]
    fn internal_and_usage_error_codes_are_distinct_from_auto_apply() {
        assert_eq!(GuardrailExitCode::InternalError.as_u8(), 1);
        assert_eq!(GuardrailExitCode::UsageError.as_u8(), 2);
        assert_ne!(GuardrailExitCode::InternalError.as_u8(), 0);
    }

    #[test]
    fn from_pass_maps_bool_to_eval_exit_code() {
        assert_eq!(EvalExitCode::from_pass(true).as_u8(), 0);
        assert_eq!(EvalExitCode::from_pass(false).as_u8(), 30);
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
