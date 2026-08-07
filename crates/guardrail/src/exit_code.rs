//! `Verdict` から終了コードへの変換を 1 箇所に閉じ込めるモジュール。
//!
//! `docs/guardrail-self-repair-cli.md` 2.3 節「fail-closed 設計（A08）」が
//! 要求するとおり、`Verdict` / eval 合否から終了コードへの変換経路は本モジュール
//! （`GuardrailExitCode::from_verdict` / `EvalExitCode::from_pass`）のみとし、
//! 他の経路から `0`（自動適用・評価合格）を返せないようにする。`main.rs` は
//! これらの関数を経由してのみプロセス終了コードを決定する。
//!
//! `self-repair` から lib として呼び出される際も同じ `Verdict` を経由する
//! （3.4 節「guardrail 連携方式」。サブプロセス起動ではなく lib 直接呼び出し）。

use std::process::ExitCode;

/// guardrail の 3 分岐判定結果（`docs/guardrail-self-repair-cli.md` 2.1 節）。
///
/// 5 条件の判定ロジック自体（#105）・除外リスト適用（#106）は本イシューの
/// スコープ外。TASK-4.1a では型のみを定義し、`check` サブコマンドは
/// 判定不能を表す `Escalate` を暫定固定で返す（fail-closed。計画 2.2 節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    AutoApply,
    Escalate,
    Reject,
}

/// `guardrail check` の終了コード契約（2.3 節）。
///
/// | 値 | 意味 |
/// |---|---|
/// | `0`  | 自動適用（`Verdict::AutoApply`） |
/// | `10` | エスカレーション（`Verdict::Escalate`） |
/// | `20` | 却下（`Verdict::Reject`） |
/// | `1`  | 内部エラー（判定不能） |
/// | `2`  | usage エラー |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailExitCode {
    AutoApply,
    Escalate,
    Reject,
    InternalError,
    UsageError,
}

impl GuardrailExitCode {
    /// `Verdict` から終了コード区分への唯一の変換経路。
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
    /// 終了コード区分への唯一の変換経路。評価ロジック本体は #108/#111 の
    /// スコープであり、TASK-4.1a では `eval` 自体が
    /// `GuardrailError::NotImplemented` で終了するため未使用だが、
    /// 契約を先に固定しておく（計画 4 節ステップ 2）。
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
}
