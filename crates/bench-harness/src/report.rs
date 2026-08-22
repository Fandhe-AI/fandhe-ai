//! 計測結果の構造化出力（TASK-8.1c・イシュー #29）。
//!
//! `protocol::run` が返す [`crate::Measurement`] を JSON へシリアライズし、
//! `guardrail`（判定レポート `bench_measurements_pct` / `bench_median_pct` 算出。
//! `docs/guardrail-self-repair-cli.md` 2.1 節）・`self-repair`（検証ゲート・TASK-3.2）
//! から参照可能にする。依存方向は `guardrail` → `bench-harness`（lib 依存。同 1.4 節）であり、
//! 本モジュールは消費側クレートを直接編集せず、公開 API（[`BenchReport::to_json`] /
//! [`BenchReport::from_json`]）を提供することで「参照可能」を満たす（イシュー #29 実装計画の
//! 判断。guardrail/self-repair 側の実連携配線は TASK-3.2・TASK-8.2 の後続スコープ）。
//!
//! ## 検証済み DTO として `Measurement` と分離する理由
//!
//! `MeasurementConfig`（`protocol` モジュール）に直接 `Deserialize` を付けると、
//! `new()` が担う下限検証（warmup／iters とも 20 回以上。TASK-8.1）をバイパスした
//! 外部入力からの構築経路が生まれ、fail-closed 方針（`.claude/rules/security.md` A08）に反する。
//! [`BenchReport`] は独立した DTO とし、[`BenchReport::from_json`] はパース直後に必ず
//! [`BenchReport::validate`] を通してから返す。検証を経ずに生値へアクセスできる公開経路は設けない。
//!
//! 計測実行メタデータ（タイムスタンプ・ホスト名等）は保持しない。回帰テストの決定性を
//! 優先するためであり、構造化ログとしての実行メタデータは TASK-3.4（self-repair ログ形式・
//! イシュー #145）のスコープとする。

use crate::Measurement;
use crate::protocol::MIN_ITERATIONS;
use crate::stats::BenchError;
use serde::{Deserialize, Serialize};

/// `BenchReport` の JSON スキーマバージョン。
///
/// guardrail 判定レポート JSON（`docs/guardrail-self-repair-cli.md` 2.1 節）の
/// `schema_version` 手法を踏襲し、[`BenchReport::validate`] が未知バージョンを
/// fail-closed で拒否することでスキーマの意図しない破壊的変更を検知可能にする。
pub const SCHEMA_VERSION: &str = "1";

/// 計測結果の構造化出力形式（1 計測対象分）。
///
/// 値の意味づけ（TFLOPS 換算・合否判定）は本クレートの関心事ではなく、
/// 呼び出し側（TASK-8.2・guardrail の判定ロジック）の責務とする
/// （`protocol::Measurement` ドキュメントコメントと同じ責務境界。`stats.rs` 冒頭コメント参照）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    /// スキーマバージョン（現行値は [`SCHEMA_VERSION`]）。
    pub schema_version: String,
    /// 計測対象識別子（例 `"gemm_f32_4096"`）。呼び出し側が自由に採番する。
    pub name: String,
    /// 計測を実行したバックエンド識別子（例 `"cpu"` / `"cuda"` / `"metal"`）。
    ///
    /// 列挙型としての固定化は TASK-8.2 側（guardrail 判定ロジック）の関心事のため、
    /// 本モジュールでは自由文字列として扱う。
    pub backend: String,
    pub warmup: usize,
    pub iters: usize,
    pub median_secs: f64,
    pub q1_secs: f64,
    pub q3_secs: f64,
    /// 全計測サンプル（秒）。guardrail の `bench_measurements_pct` 算出根拠
    /// （`docs/guardrail-self-repair-cli.md` 2.1 節）。
    pub samples_secs: Vec<f64>,
}

impl BenchReport {
    /// `protocol::run` の実測結果 [`Measurement`] から構築する。
    ///
    /// `Measurement` は `MeasurementConfig::new`（または `run` の防御的検証）を経由済みの
    /// 値のみを保持するため、構築時点で下限違反は原理的に発生しないが、[`Self::validate`] を
    /// 通してから返すことで「検証を経ない `BenchReport` は存在しない」という不変条件を
    /// 構築経路全体（`from_measurement` / `from_json`）で統一する。
    ///
    /// # Errors
    ///
    /// [`Self::validate`] が失敗した場合（通常は発生しない防御的経路）。
    pub fn from_measurement(
        name: impl Into<String>,
        backend: impl Into<String>,
        measurement: &Measurement,
    ) -> Result<Self, BenchError> {
        let report = Self {
            schema_version: SCHEMA_VERSION.to_string(),
            name: name.into(),
            backend: backend.into(),
            warmup: measurement.warmup,
            iters: measurement.iters,
            median_secs: measurement.median_secs,
            q1_secs: measurement.q1_secs,
            q3_secs: measurement.q3_secs,
            samples_secs: measurement.samples_secs.clone(),
        };
        report.validate()?;
        Ok(report)
    }

    /// TASK-8.1 の計測プロトコル遵守を fail-closed で検証する。
    ///
    /// 検証項目（`.claude/rules/security.md` A08: 判定の迂回経路を作らない）:
    /// - `schema_version` が [`SCHEMA_VERSION`] と一致する（不一致＝未知スキーマは拒否）
    /// - `warmup` / `iters` が TASK-8.1 下限（`MIN_ITERATIONS` 以上）を満たす
    /// - `samples_secs.len() == iters` かつ全サンプルが有限・非負
    /// - `q1_secs <= median_secs <= q3_secs` かつ 3 値とも有限
    ///
    /// # Errors
    ///
    /// 検証に失敗した場合 `BenchError::ProtocolViolation`（違反内容をメッセージに含む）。
    pub fn validate(&self) -> Result<(), BenchError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(BenchError::ProtocolViolation(format!(
                "未知の schema_version: 期待値 {SCHEMA_VERSION:?}, 実際 {:?}",
                self.schema_version
            )));
        }
        if self.warmup < MIN_ITERATIONS {
            return Err(BenchError::ProtocolViolation(format!(
                "warmup は {MIN_ITERATIONS} 回以上が必須（TASK-8.1）。指定値: {}",
                self.warmup
            )));
        }
        if self.iters < MIN_ITERATIONS {
            return Err(BenchError::ProtocolViolation(format!(
                "計測回数（iters）は {MIN_ITERATIONS} 回以上が必須（TASK-8.1）。指定値: {}",
                self.iters
            )));
        }
        if self.samples_secs.len() != self.iters {
            return Err(BenchError::ProtocolViolation(format!(
                "samples_secs の要素数（{}）が iters（{}）と不一致",
                self.samples_secs.len(),
                self.iters
            )));
        }
        if self.samples_secs.iter().any(|&s| !s.is_finite() || s < 0.0) {
            return Err(BenchError::ProtocolViolation(
                "samples_secs に非有限値または負値が含まれている".to_string(),
            ));
        }
        if !self.median_secs.is_finite() || !self.q1_secs.is_finite() || !self.q3_secs.is_finite() {
            return Err(BenchError::ProtocolViolation(
                "median_secs / q1_secs / q3_secs のいずれかが非有限値".to_string(),
            ));
        }
        if !(self.q1_secs <= self.median_secs && self.median_secs <= self.q3_secs) {
            return Err(BenchError::ProtocolViolation(format!(
                "q1_secs <= median_secs <= q3_secs を満たさない: q1={}, median={}, q3={}",
                self.q1_secs, self.median_secs, self.q3_secs
            )));
        }
        Ok(())
    }

    /// JSON へシリアライズする。
    ///
    /// シリアライズ**前**に [`Self::validate`] を実行する。serde_json は非有限 f64
    /// （NaN・±inf）を `null` として出力し情報が黙って壊れるため、事前に拒否する
    /// （`.claude/rules/security.md` A08: silent corruption の防止）。
    ///
    /// # Errors
    ///
    /// - [`Self::validate`] が失敗した場合、そのエラーをそのまま返す
    /// - JSON エンコードに失敗した場合 `BenchError::ProtocolViolation`
    ///   （`serde_json::Error` を包む。`BenchError` は公開 enum であり非破壊のため
    ///   新規 variant を追加せず既存 variant に委譲する）
    pub fn to_json(&self) -> Result<String, BenchError> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|e| BenchError::ProtocolViolation(format!("JSON エンコード失敗: {e}")))
    }

    /// JSON からデシリアライズし、[`Self::validate`] を通してから返す。
    ///
    /// パース段階のエラー（フォーマット不正・型不一致）と検証段階のエラー（プロトコル違反）を
    /// いずれも `BenchError::ProtocolViolation` に正規化する。検証を経ずに生値へアクセスできる
    /// 公開経路は設けない（外部フォーマットパースは長さ・形状の検証を先に行う方針。
    /// `.claude/rules/security.md` A03）。
    ///
    /// # Errors
    ///
    /// - JSON デコードに失敗した場合 `BenchError::ProtocolViolation`
    /// - デコードに成功しても [`Self::validate`] が失敗した場合、そのエラー
    pub fn from_json(json: &str) -> Result<Self, BenchError> {
        let report: Self = serde_json::from_str(json)
            .map_err(|e| BenchError::ProtocolViolation(format!("JSON デコード失敗: {e}")))?;
        report.validate()?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_measurement() -> Measurement {
        let samples_secs: Vec<f64> = (0..20).map(|i| 1.0 + i as f64 * 0.01).collect();
        Measurement {
            median_secs: 1.1,
            q1_secs: 1.05,
            q3_secs: 1.15,
            samples_secs,
            warmup: 20,
            iters: 20,
        }
    }

    #[test]
    fn from_measurement_succeeds_for_valid_measurement() {
        let report = BenchReport::from_measurement("gemm_f32_4096", "cpu", &valid_measurement())
            .expect("プロトコル遵守済み Measurement からの構築は成功するはず");
        assert_eq!(report.schema_version, SCHEMA_VERSION);
        assert_eq!(report.name, "gemm_f32_4096");
        assert_eq!(report.backend, "cpu");
    }

    #[test]
    fn to_json_rejects_nan() {
        let mut report =
            BenchReport::from_measurement("gemm_f32_4096", "cpu", &valid_measurement()).unwrap();
        report.median_secs = f64::NAN;
        let err = report
            .to_json()
            .expect_err("NaN を含む BenchReport の to_json は拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }
}
