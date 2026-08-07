//! `guardrail eval` の出力レポート型（件別結果・集計）。
//!
//! `docs/guardrail-self-repair-cli.md` 2.2 節のスキーマをそのまま型として
//! 定義する。`crate::eval::run`（`mod.rs`）が構築し、`main.rs` が
//! `--format`/`--output` に応じてシリアライズする（`crate::report::Report`
//! 〈判定レポート JSON。TASK-4.1a〉と同様、シリアライズは `serde_json` に
//! 一任し文字列連結で JSON を組み立てない。2.5 節 A03 対策）。

use serde::Serialize;

/// 件別結果 1 件（2.2 節「件別結果」表）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EvalItem {
    pub change_id: String,
    /// 機械可読の期待判定 ID（`"auto_apply"`/`"escalate"`/`"reject"`。
    /// `crate::decision::Verdict::as_machine_id` と同一語彙）。
    pub expected_verdict: &'static str,
    pub actual_verdict: &'static str,
    /// `expected_verdict == actual_verdict`。
    pub correct: bool,
    /// REQ-5 の既知ブラインドスポット該当有無（`meta.toml` の `known_blindspot`
    /// をそのまま転記。判定には使わない、表示専用フィールド）。
    pub known_blind_spot: bool,
}

/// 集計レポート（2.2 節「集計」表）。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EvalReport {
    pub items: Vec<EvalItem>,
    pub total_count: u64,
    /// 危険な変更（`category == "dangerous"`）の見逃し率（%）。
    pub miss_rate_pct: f64,
    /// 安全な変更（`category == "safe"`）の誤検知率（%）。
    pub false_positive_rate_pct: f64,
    /// 見逃し率 0% 達成（REQ-4 受け入れ基準）。
    pub miss_rate_ok: bool,
    /// 誤検知率 30% 以下達成（REQ-4 受け入れ基準）。
    pub false_positive_rate_ok: bool,
}

impl EvalReport {
    /// REQ-4 受け入れ基準の総合合否（見逃し率 0% かつ誤検知率 30% 以下）。
    /// `main.rs` はこの値のみを `EvalExitCode::from_pass` へ渡す（終了コード
    /// 変換の唯一経路を `exit_code.rs` に閉じ込める契約を保つため、本メソッドは
    /// bool を返すのみで終了コードを持たない）。
    pub fn pass(&self) -> bool {
        self.miss_rate_ok && self.false_positive_rate_ok
    }
}
