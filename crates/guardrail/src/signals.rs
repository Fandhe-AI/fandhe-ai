//! `--signals` が受け取る計測済みシグナル JSON の入力型。
//!
//! `docs/guardrail-self-repair-cli.md` 1.2 節「`--signals` の迂回防止」・1.4 節
//! （bench-harness 付け替え後もスキーマは同一）に対応する CI 契約検証専用パス。
//! `cli.rs` の入口ガード（環境変数 `GUARDRAIL_ALLOW_INJECTED_SIGNALS=1`）を
//! 通過した後にのみ `main.rs` から読み込まれる。
//!
//! 2.5 節の方針どおり、シグナル JSON は「必須フィールド欠落の検出のみ」を行い、
//! 将来のフィールド追加に対する前方互換性を優先する（判定を安全側に倒す性質の
//! 設定ファイルではないため、`guardrail.toml` のような未知フィールド拒否は行わない。
//! serde の既定動作＝未知フィールドは無視、をそのまま用いる）。

use serde::{Deserialize, Serialize};

use crate::error::GuardrailError;
use crate::toml_lite::MAX_INPUT_BYTES;

/// v1 `signals.rs` 相当の 5 条件入力（`docs/guardrail-self-repair-cli.md` 2.4 節の
/// 5 条件に対応するシグナル）。`bench_measurements_pct` は REQ-4「5 回以上」の
/// 受け入れ基準に合わせて長さ検証を行う。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signals {
    pub lines_changed: u64,
    pub public_api_broken: bool,
    pub gaming_suspected: bool,
    pub build_result: GateResult,
    pub test_result: GateResult,
    pub clippy_result: GateResult,
    pub bench_measurements_pct: Vec<f64>,
}

/// 4 ゲート（build/test/clippy/bench）のうち pass/fail で表現できるものの結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateResult {
    Pass,
    Fail,
}

impl Signals {
    /// JSON 文字列からデシリアライズする。サイズ上限（A03 対策・2.5 節）は
    /// `toml_lite::MAX_INPUT_BYTES` を流用し、guardrail.toml と同一の 64 KiB
    /// 上限をシグナル JSON にも適用する（DoS 的な巨大入力の一律拒否）。
    pub fn from_json_str(raw: &str) -> Result<Self, GuardrailError> {
        if raw.len() > MAX_INPUT_BYTES {
            return Err(GuardrailError::InvalidInput(format!(
                "signals JSON exceeds {MAX_INPUT_BYTES} byte limit ({} bytes)",
                raw.len()
            )));
        }
        let signals: Signals = serde_json::from_str(raw)
            .map_err(|e| GuardrailError::InvalidInput(format!("invalid signals JSON: {e}")))?;
        // REQ-4「5 回以上」の受け入れ基準（1.2 節が参照する 2.1 節スキーマ注記）。
        if signals.bench_measurements_pct.len() < 5 {
            return Err(GuardrailError::InvalidInput(
                "bench_measurements_pct must contain at least 5 measurements (REQ-4)".to_string(),
            ));
        }
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> &'static str {
        r#"{
            "lines_changed": 42,
            "public_api_broken": false,
            "gaming_suspected": false,
            "build_result": "pass",
            "test_result": "pass",
            "clippy_result": "pass",
            "bench_measurements_pct": [1.0, 1.1, 0.9, 1.2, 1.05]
        }"#
    }

    #[test]
    fn parses_valid_signals() {
        let signals = Signals::from_json_str(valid_json()).unwrap();
        assert_eq!(signals.lines_changed, 42);
        assert_eq!(signals.bench_measurements_pct.len(), 5);
    }

    #[test]
    fn rejects_missing_required_field() {
        let raw = r#"{"lines_changed": 1}"#;
        let err = Signals::from_json_str(raw).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn rejects_fewer_than_five_bench_measurements() {
        let raw = r#"{
            "lines_changed": 1,
            "public_api_broken": false,
            "gaming_suspected": false,
            "build_result": "pass",
            "test_result": "pass",
            "clippy_result": "pass",
            "bench_measurements_pct": [1.0, 1.0]
        }"#;
        let err = Signals::from_json_str(raw).unwrap_err();
        assert!(matches!(err, GuardrailError::InvalidInput(_)));
    }

    #[test]
    fn ignores_unknown_fields_for_forward_compat() {
        let raw = r#"{
            "lines_changed": 1,
            "public_api_broken": false,
            "gaming_suspected": false,
            "build_result": "pass",
            "test_result": "pass",
            "clippy_result": "pass",
            "bench_measurements_pct": [1.0, 1.0, 1.0, 1.0, 1.0],
            "future_field": "ignored"
        }"#;
        assert!(Signals::from_json_str(raw).is_ok());
    }
}
