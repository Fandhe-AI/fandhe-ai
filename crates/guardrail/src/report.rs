//! `guardrail check --output` が書き出す判定レポート JSON（v2 版）。
//!
//! `docs/guardrail-self-repair-cli.md` 2.1 節のスキーマをそのまま型として
//! 定義する。REQ-3 データ要件・REQ-6 の回帰テストセット根拠データとして
//! 再利用可能な形式を維持するため、フィールド名・型は文書と 1 対 1 対応させる。
//! シリアライズは `serde_json` に一任し、文字列連結で JSON を組み立てない
//! （2.5 節 A03 対策）。

use serde::{Deserialize, Serialize};

use crate::exit_code::Verdict;

/// シグナルの出所。`--signals`（1.2 節 CI 契約検証パス）経由なら `Injected`、
/// 実シグナル計測（本番相当経路。TASK-4.1a では未実装）なら `Measured`。
/// REQ-6 の回帰テストセット根拠データは `Measured` のみを採用する（2.1 節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSource {
    Measured,
    Injected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    Pass,
    Fail,
}

/// 判定レポート JSON のスキーマバージョン。2.1 節の記載どおり `"1"`（初版）で
/// 据え置く（新規設計文書のため既存消費者との互換負担がない）。
pub const SCHEMA_VERSION: &str = "1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: String,
    pub signal_source: SignalSource,
    pub change_id: Option<String>,
    pub lines_changed: u64,
    pub public_api_broken: bool,
    pub gaming_suspected: bool,
    pub build_result: GateOutcome,
    pub test_result: GateOutcome,
    pub clippy_result: GateOutcome,
    pub bench_measurements_pct: Vec<f64>,
    pub bench_median_pct: f64,
    pub applied_exclusion_rule_ids: Vec<String>,
    pub verdict: Verdict,
    pub reason: String,
}

/// 昇順ソート済みの計測値列から中央値を求める。TASK-4.1a では `report.rs` が
/// レポートの整形にのみ用い、判定ロジック（#105）には使わない。
///
/// 事前条件: `values` は空でないこと（呼び出し元の `signals.rs` が REQ-4
/// 「5 回以上」の受け入れ基準として非空・5 件以上を検証済み）。
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_length() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn median_of_even_length() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn report_round_trips_through_json() {
        let report = Report {
            schema_version: SCHEMA_VERSION.to_string(),
            signal_source: SignalSource::Injected,
            change_id: Some("abc123".to_string()),
            lines_changed: 10,
            public_api_broken: false,
            gaming_suspected: false,
            build_result: GateOutcome::Pass,
            test_result: GateOutcome::Pass,
            clippy_result: GateOutcome::Pass,
            bench_measurements_pct: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            bench_median_pct: 1.0,
            applied_exclusion_rule_ids: vec![],
            verdict: Verdict::Escalate,
            reason: "判定ロジック未実装（TASK-4.1b/#105 で移植）".to_string(),
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.verdict, Verdict::Escalate);
        assert_eq!(parsed.signal_source, SignalSource::Injected);
    }
}
