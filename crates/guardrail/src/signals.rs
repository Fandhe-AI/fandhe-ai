//! 判定シグナルの型（TASK-4.1b・イシュー #105）。
//!
//! PoC-3（`docs/spec/03-poc/poc-3-guardrail-validity/code/guardrail.sh`）が
//! git 差分・build/test/clippy・bench 実行から計測する各シグナルを型として
//! 定義する。実際のシグナル計測（git 差分からの変更行数・公開 API 破壊検出・
//! ゲーミング疑い検出、build/test/clippy ゲート実行）は本イシューのスコープ外
//! （#104・#106・#107 の管轄）であり、本クレートにはまだ実装されていない。
//! 本モジュールは実測経路が未実装の段階でも判定ロジック（`decision::decide`）
//! を CI 契約検証可能にするための「契約としてのシグナル入力型」であり、実測
//! モジュールのマージ時にそちらの出力型と統合される想定（フィールド名は
//! `guardrail.sh` の出力 JSON キーと 1:1 対応させているため統合コストは
//! 小さいはずである）。
//!
//! [`crate::decision::decide`] が [`Signals::to_decision_input`] の戻り値
//! （[`crate::decision::DecisionInput`]）を受け取り
//! [`crate::decision::Decision`] を返す。CLI からは判定レポートに包んで
//! 出力する契約（#106 が接続する）。

use serde::{Deserialize, Serialize};

use crate::config::Thresholds;
use crate::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals};
use crate::error::GuardrailError;

/// 単一変更セットに対する判定入力シグナル一式。
///
/// フィールドは PoC-3 の結果 JSON（`guardrail.sh` 出力）のキーと同名にして
/// いる（`bench_samples_pct` 等の判定に使わない付随情報は本イシューのスコープ
/// 外の表示層が別途保持する契約とし、ここには含めない）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Signals {
    /// 変更行数（挿入+削除。Cargo.lock は対象外。guardrail.sh 1 節）。
    pub lines_changed: u32,
    /// 変更行数の上限閾値（プリセット由来）。
    pub lines_max: u32,
    /// 公開 API シグネチャ破壊の簡易検出結果（guardrail.sh 2 節）。
    pub api_broken: bool,
    /// テスト・本番コード同時変更（ゲーミング疑い）の検出結果（guardrail.sh 3 節）。
    pub gaming_suspect: bool,
    /// `cargo build` の成否（guardrail.sh 4 節）。
    pub build_ok: bool,
    /// `cargo test --release` の成否。`build_ok` が false の場合は評価されない。
    pub test_ok: bool,
    /// `cargo clippy --all-targets -- -D warnings` の成否。
    pub clippy_ok: bool,
    /// bench を実行したか（build/test/clippy 全通過時のみ true）。
    pub bench_ran: bool,
    /// bench 劣化率の中央値（％。正の値が劣化・負の値が改善）。
    pub bench_median_pct: f64,
    /// bench 劣化許容上限（％。プリセット由来）。
    pub bench_max_pct: f64,
}

impl Signals {
    /// [`crate::decision::decide`] へ渡すための変換。フラットな真偽値
    /// （`build_ok`/`test_ok`/`clippy_ok`）から [`GateSignals`]
    /// （`Passed`/`Failed`/`Skipped`）を導出する。
    ///
    /// `guardrail.sh` の実行順序契約（build 失敗時は test/clippy を実行しない。
    /// test 失敗時は clippy を実行する）に合わせ、`build_ok` が false の場合
    /// のみ test/clippy を `Skipped` として扱う（それ以外は各フィールドの
    /// 真偽値をそのまま `Passed`/`Failed` へ写像する）。
    ///
    /// `DecisionInput::new` は「ゲート全通過でないのにベンチ計測済み」という
    /// 矛盾入力を fail-closed で拒否するため、本変換もそのままエラーを伝播する
    /// （security.md A08。判定を迂回しない）。
    ///
    /// `exclusion_rule_ids`（match したポリシー除外リストのルール `id`。空 =
    /// match なし）は呼び出し元が渡す必須引数（評価忘れの fail-open 経路を
    /// 型で封鎖する契約は `decision::DecisionInput::new` を参照）。
    pub fn to_decision_input<'a>(
        &self,
        thresholds: &'a Thresholds,
        exclusion_rule_ids: Vec<String>,
    ) -> Result<DecisionInput<'a>, GuardrailError> {
        let (test, clippy) = if self.build_ok {
            (
                bool_to_gate_signal(self.test_ok),
                bool_to_gate_signal(self.clippy_ok),
            )
        } else {
            (GateSignal::Skipped, GateSignal::Skipped)
        };
        let gates = GateSignals {
            build: bool_to_gate_signal(self.build_ok),
            test,
            clippy,
        };
        let bench = if self.bench_ran {
            BenchSignal::Measured {
                median_pct: self.bench_median_pct,
            }
        } else {
            BenchSignal::NotRun
        };

        DecisionInput::new(
            thresholds,
            self.lines_changed,
            gates,
            self.api_broken,
            self.gaming_suspect,
            bench,
            exclusion_rule_ids,
        )
    }
}

/// `bool`（成否）→ [`GateSignal`]（`Skipped` を含まない 2 値からの変換。
/// `Skipped` の判定は呼び出し元（[`Signals::to_decision_input`]）が行う）。
fn bool_to_gate_signal(ok: bool) -> GateSignal {
    if ok {
        GateSignal::Passed
    } else {
        GateSignal::Failed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PresetName;

    fn base_signals() -> Signals {
        Signals {
            lines_changed: 10,
            lines_max: 200,
            api_broken: false,
            gaming_suspect: false,
            build_ok: true,
            test_ok: true,
            clippy_ok: true,
            bench_ran: false,
            bench_median_pct: 0.0,
            bench_max_pct: 5.0,
        }
    }

    #[test]
    fn all_ok_signals_yield_auto_apply() {
        let thresholds =
            Thresholds::builtin(PresetName::Default).expect("組み込み既定値の検証に失敗");
        let signals = base_signals();
        let input = signals
            .to_decision_input(&thresholds, Vec::new())
            .expect("シグナル変換に失敗");
        let decision = crate::decision::decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), crate::decision::Verdict::AutoApply);
    }

    #[test]
    fn build_failure_skips_test_and_clippy() {
        let thresholds =
            Thresholds::builtin(PresetName::Default).expect("組み込み既定値の検証に失敗");
        let signals = Signals {
            build_ok: false,
            test_ok: true,
            clippy_ok: true,
            ..base_signals()
        };
        let input = signals
            .to_decision_input(&thresholds, Vec::new())
            .expect("シグナル変換に失敗");
        let decision = crate::decision::decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), crate::decision::Verdict::Reject);
        assert_eq!(decision.reason_conditions(), vec!["gate_build_failed"]);
    }

    #[test]
    fn bench_ran_true_with_build_failure_is_inconsistent() {
        let thresholds =
            Thresholds::builtin(PresetName::Default).expect("組み込み既定値の検証に失敗");
        let signals = Signals {
            build_ok: false,
            bench_ran: true,
            bench_median_pct: 1.0,
            ..base_signals()
        };
        let err = signals
            .to_decision_input(&thresholds, Vec::new())
            .unwrap_err();
        assert!(matches!(
            err,
            GuardrailError::InconsistentDecisionInput { .. }
        ));
    }

    #[test]
    fn bench_ran_true_with_all_gates_passed_yields_measured_bench() {
        let thresholds =
            Thresholds::builtin(PresetName::Default).expect("組み込み既定値の検証に失敗");
        let signals = Signals {
            bench_ran: true,
            bench_median_pct: 6.0,
            ..base_signals()
        };
        let input = signals
            .to_decision_input(&thresholds, Vec::new())
            .expect("シグナル変換に失敗");
        let decision = crate::decision::decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), crate::decision::Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["bench_median_exceeded"]);
    }
}
