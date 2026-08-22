//! ベンチゲート計測系（TASK-4.1d・#107）。
//!
//! v1 の guardrail ベンチゲート（4 ゲートの `cargo bench` 相当）は Criterion・Burn
//! 計測 API に依存していたが、v2 は完全自作コアのため計測実行系を `crates/bench-harness`
//! へ付け替える（REQ-3「検証ゲートの計測系付け替え」注記・REQ-4 受け入れ基準。
//! `docs/guardrail-self-repair-cli.md` 1.4 節）。依存方向は「`guardrail` → `bench-harness`
//! （lib 依存）」で固定し、判定ロジック自体（中央値採用・決定的シード）は変更しない
//! （同節）。本モジュールは判定レポート JSON（同 2.1 節）の `bench_measurements_pct`・
//! `bench_median_pct` フィールドを構成する計測実行系のみを担い、5 条件判定ロジック本体
//! （TASK-4.1b・#105）・3 分岐出力（TASK-4.1c・#106）は別モジュールの責務とする。
//!
//! ## 計測回数レイヤの区別（用語混同防止）
//!
//! REQ-4「5 回以上・変化率の中央値」（本モジュールの `MIN_BENCH_ITERATIONS`）と
//! TASK-8.1 の「warmup 20 回以上・計測 20 回以上」（`bench_harness::MeasurementConfig`
//! の下限。`bench-harness/src/lib.rs:17-25` に既知の規約ファイル記述不一致の注記あり）は
//! 別レイヤの規定である。本モジュールでいう「1 反復」は `bench_harness::run` 1 回の
//! 呼び出し（内部で warmup 20+・計測 20+ サンプルを取る）を指し、劣化率 `pct` はその
//! 1 反復から得られる baseline・candidate 各 1 つの `median_secs` の比から求める 1 つの
//! 標本である。この `pct` を 5 反復以上繰り返して `bench_measurements_pct` を構成し、
//! その中央値を `bench_median_pct` とする。
//!
//! ## スコープ境界
//!
//! - baseline／candidate のワークロード自体（実際に何を計測するか。CLI・self-repair 側の
//!   ビルド成果物起動等）を本モジュールは持たない。呼び出し側がクロージャとして渡す
//!   （`bench_harness::protocol` がバックエンド非依存クロージャ設計を取るのと同じ理由）。
//! - 反復回数の運用既定値の確定・決定的シードユーティリティの guardrail／self-repair
//!   双方向組み込みは TASK-4.4（#111）のスコープであり、本モジュールでは重複実装しない
//!   （実装計画 #107 の切り分け）。
//! - 判定ロジック（4.1b・#105）が本モジュールの [`BenchGateRunner`] trait をどう消費するかの
//!   結線は本イシュー時点では未完でも構わない。受け入れ条件（計測部分が新実装経由で動作する）
//!   は統合テスト（`tests/bench_gate_integration.rs`）で満たす。

use bench_harness::{MeasurementConfig, run};
use serde::{Deserialize, Serialize};
use std::fmt;

/// REQ-4「5 回以上・変化率の中央値」の下限（ベンチゲートの反復回数）。
///
/// `bench_harness::protocol::MIN_ITERATIONS`（1 反復内の warmup／計測サンプル数下限）とは
/// 別レイヤの下限である（本モジュール冒頭ドキュメント参照）。下限を回避する公開 API は
/// 設けない（`.claude/rules/security.md`: 閾値・許容値の単独緩和禁止）。
pub const MIN_BENCH_ITERATIONS: usize = 5;

/// ベンチゲート計測系のエラー。
///
/// 本番経路で `unwrap()` / `expect()` を使わない方針（`.claude/rules/coding-rust.md`）に
/// 従い、bench-harness 側のエラー（`bench_harness::BenchError`）は握り潰さず
/// [`BenchGateError::Measurement`] として伝播する（実装計画 #107 セキュリティ考慮事項）。
#[derive(Debug, Clone, PartialEq)]
pub enum BenchGateError {
    /// bench-harness 側の計測プロトコル違反・統計計算失敗をそのまま包む。
    Measurement(String),
    /// 劣化率算出に使う baseline の中央値が 0（またはそれに近い非有限値を生む値）であり、
    /// 除算結果が非有限（NaN／inf）になる場合に返す。`median_q1_q3` は NaN の混入は
    /// 自身で検査する（`bench_harness::stats::median_q1_q3` が `BenchError::NanSample`
    /// を返す）が、Infinite（±inf）は個別にチェックしないため通過しうる。この除算自体が
    /// 非有限値（Infinite）の発生源になりうるため、除算直後にこの境界を本モジュール側で
    /// 明示的に検査する（実装計画 #107・レビュー指摘）。
    NonFiniteRatio { baseline_median_secs: f64 },
    /// [`BenchSignal`] の反復回数が [`MIN_BENCH_ITERATIONS`] 未満、または
    /// `bench_median_pct` が `bench_measurements_pct` の実際の中央値と一致しない
    /// （`--signals` 注入経路〈CLI 仕様書 1.2 節〉からの改竄・不整合を fail-closed で拒否する。
    /// A08: 判定の迂回経路を作らない）。
    InvalidSignal(String),
}

impl fmt::Display for BenchGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchGateError::Measurement(msg) => write!(f, "ベンチ計測エラー: {msg}"),
            BenchGateError::NonFiniteRatio {
                baseline_median_secs,
            } => write!(
                f,
                "劣化率算出が非有限値になった（baseline_median_secs={baseline_median_secs}）"
            ),
            BenchGateError::InvalidSignal(msg) => write!(f, "ベンチシグナル不正: {msg}"),
        }
    }
}

impl std::error::Error for BenchGateError {}

impl From<bench_harness::BenchError> for BenchGateError {
    fn from(err: bench_harness::BenchError) -> Self {
        BenchGateError::Measurement(err.to_string())
    }
}

/// ベンチゲート計測結果（判定レポート JSON 2.1 節のフィールドに対応する DTO）。
///
/// `--signals` 経由（CLI 仕様書 1.2 節）で外部 JSON からデシリアライズされうるため、
/// `bench_harness::report::BenchReport` と同じ設計（検証を経ない値へアクセスできる公開
/// 経路を設けない）を踏襲する。[`Self::from_measurements_pct`]・`Self::validated` の
/// いずれの構築経路でも [`Self::validate`] を通す。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchSignal {
    /// ベンチ劣化率の計測値（%。5 件以上。REQ-4「5 回以上」の受け入れ基準に対応）。
    pub bench_measurements_pct: Vec<f64>,
    /// 上記計測値の中央値（判定に用いる値）。
    pub bench_median_pct: f64,
}

impl BenchSignal {
    /// 劣化率系列 `measurements_pct` から構築し、検証を通してから返す。
    ///
    /// `bench_median_pct` は `measurements_pct` から `bench_harness::median_q1_q3` で
    /// 再計算する（呼び出し側からの信頼値をそのまま使わない。中央値算出は独自実装しない
    /// 方針も兼ねる。実装計画 #107 3 節）。
    ///
    /// # Errors
    ///
    /// - `measurements_pct.len() < MIN_BENCH_ITERATIONS` の場合 [`BenchGateError::InvalidSignal`]
    /// - `median_q1_q3` が失敗した場合（空・NaN 混入。Infinite は個別チェックしないため
    ///   通過しうるが、後続の [`Self::validate`] が `bench_measurements_pct` の有限性を
    ///   検査して拒否する） [`BenchGateError::Measurement`]
    pub fn from_measurements_pct(measurements_pct: Vec<f64>) -> Result<Self, BenchGateError> {
        if measurements_pct.len() < MIN_BENCH_ITERATIONS {
            return Err(BenchGateError::InvalidSignal(format!(
                "bench_measurements_pct は {MIN_BENCH_ITERATIONS} 件以上が必須（REQ-4）。指定件数: {}",
                measurements_pct.len()
            )));
        }
        let quartiles = bench_harness::median_q1_q3(&measurements_pct)?;
        let signal = Self {
            bench_measurements_pct: measurements_pct,
            bench_median_pct: quartiles.median,
        };
        signal.validate()?;
        Ok(signal)
    }

    /// `--signals` 注入経路（CLI 仕様書 1.2 節）からのデシリアライズ後に必ず通す検証。
    ///
    /// 検証項目（A08: 判定の迂回経路を作らない。`bench_harness::report::BenchReport::validate`
    /// と同種の fail-closed 設計）:
    /// - `bench_measurements_pct.len() >= MIN_BENCH_ITERATIONS`
    /// - 全要素が有限値
    /// - `bench_median_pct` が `bench_measurements_pct` から再計算した中央値と一致する
    ///   （JSON 側の `bench_median_pct` を信頼せず、改竄・不整合を拒否する）
    ///
    /// # Errors
    ///
    /// 検証に失敗した場合 [`BenchGateError::InvalidSignal`]。
    pub fn validate(&self) -> Result<(), BenchGateError> {
        if self.bench_measurements_pct.len() < MIN_BENCH_ITERATIONS {
            return Err(BenchGateError::InvalidSignal(format!(
                "bench_measurements_pct は {MIN_BENCH_ITERATIONS} 件以上が必須（REQ-4）。指定件数: {}",
                self.bench_measurements_pct.len()
            )));
        }
        if self.bench_measurements_pct.iter().any(|v| !v.is_finite()) {
            return Err(BenchGateError::InvalidSignal(
                "bench_measurements_pct に非有限値（NaN／inf）が含まれている".to_string(),
            ));
        }
        if !self.bench_median_pct.is_finite() {
            return Err(BenchGateError::InvalidSignal(
                "bench_median_pct が非有限値（NaN／inf）".to_string(),
            ));
        }
        let recomputed = bench_harness::median_q1_q3(&self.bench_measurements_pct)
            .map_err(|e| BenchGateError::InvalidSignal(e.to_string()))?;
        if recomputed.median != self.bench_median_pct {
            return Err(BenchGateError::InvalidSignal(format!(
                "bench_median_pct（{}）が bench_measurements_pct から再計算した中央値（{}）と不一致",
                self.bench_median_pct, recomputed.median
            )));
        }
        Ok(())
    }

    /// `--signals` 注入経路（CLI 仕様書 1.2 節）の JSON からデシリアライズし、
    /// [`Self::validate`] を通してから返す。`bench_harness::report::BenchReport::from_json`
    /// と同じ設計（パース段階のエラー・検証段階のエラーをいずれも型付きエラーへ正規化し、
    /// 検証を経ずに生値へアクセスできる公開経路を設けない。`.claude/rules/security.md` A03）。
    ///
    /// # Errors
    ///
    /// - JSON デコードに失敗した場合 [`BenchGateError::InvalidSignal`]
    /// - デコードに成功しても [`Self::validate`] が失敗した場合、そのエラー
    pub fn from_json(json: &str) -> Result<Self, BenchGateError> {
        let signal: Self = serde_json::from_str(json)
            .map_err(|e| BenchGateError::InvalidSignal(format!("JSON デコード失敗: {e}")))?;
        signal.validate()?;
        Ok(signal)
    }
}

/// 判定ロジック（TASK-4.1b・#105）が消費するベンチゲート計測系の抽象契約。
///
/// baseline／candidate ワークロードを 5 回以上（[`MIN_BENCH_ITERATIONS`]）計測し、
/// 劣化率系列 [`BenchSignal`] を返す。Criterion・Burn への参照は一切持たない
/// （受け入れ条件「Burn 依存が排除され、計測部分が新実装経由で動作すること」）。
/// テストではモック実装を注入できるよう trait として切り出す。実装は必ず
/// [`BenchSignal::validate`] を通過する `BenchSignal` を返すこと（`BenchSignal` のフィールドは
/// 公開のため、フィールドリテラルで直接構築するモック実装は各自でこの契約を守る責務を持つ）。
pub trait BenchGateRunner {
    /// `baseline`・`candidate` それぞれのワークロードを `iterations` 回反復計測し、
    /// [`BenchSignal`] を返す。
    ///
    /// # Errors
    ///
    /// 計測・検証に失敗した場合 [`BenchGateError`]。
    fn measure(
        &self,
        config: &MeasurementConfig,
        iterations: usize,
        baseline: &mut dyn FnMut(),
        candidate: &mut dyn FnMut(),
    ) -> Result<BenchSignal, BenchGateError>;
}

/// [`BenchGateRunner`] の本番実装。`bench_harness::run` を計測実行系として呼び出す。
///
/// 1 反復 = baseline・candidate それぞれ 1 回の `bench_harness::run`（内部で warmup 20+・
/// 計測 20+ サンプルを取り中央値を返す。TASK-8.1 プロトコル）。劣化率
/// `pct = (candidate_median / baseline_median - 1.0) * 100.0` をその反復の 1 標本とし、
/// `iterations` 回（[`MIN_BENCH_ITERATIONS`] 以上）繰り返して [`BenchSignal`] を構成する。
#[derive(Debug, Default, Clone, Copy)]
pub struct HarnessBenchGate;

impl BenchGateRunner for HarnessBenchGate {
    fn measure(
        &self,
        config: &MeasurementConfig,
        iterations: usize,
        baseline: &mut dyn FnMut(),
        candidate: &mut dyn FnMut(),
    ) -> Result<BenchSignal, BenchGateError> {
        if iterations < MIN_BENCH_ITERATIONS {
            return Err(BenchGateError::InvalidSignal(format!(
                "反復回数（iterations）は {MIN_BENCH_ITERATIONS} 回以上が必須（REQ-4）。指定値: {iterations}"
            )));
        }

        let mut measurements_pct = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let baseline_measurement = run(config, &mut *baseline)?;
            let candidate_measurement = run(config, &mut *candidate)?;

            let baseline_median_secs = baseline_measurement.median_secs;
            // baseline の中央値が 0（またはそれに近い値）だと除算結果が非有限（NaN／inf）に
            // なる。NaN は後段の `bench_harness::median_q1_q3` が自身で検査し
            // `BenchError::NanSample` を返すが、Infinite（±inf）は個別にチェックせず
            // 通過しうるため、発生源（この除算）で明示的に検査してから系列へ加える
            // （実装計画 #107・レビュー指摘。A08: silent corruption の防止）。
            let pct = (candidate_measurement.median_secs / baseline_median_secs - 1.0) * 100.0;
            if !pct.is_finite() {
                return Err(BenchGateError::NonFiniteRatio {
                    baseline_median_secs,
                });
            }
            measurements_pct.push(pct);
        }

        BenchSignal::from_measurements_pct(measurements_pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_config() -> MeasurementConfig {
        // 単体テストは実行時間を抑えるため下限（20/20）ちょうどを使う。
        MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず")
    }

    #[test]
    fn bench_signal_rejects_fewer_than_min_iterations() {
        let err = BenchSignal::from_measurements_pct(vec![1.0, 2.0, 3.0, 4.0])
            .expect_err("4 件は下限（5 件）未満のため拒否されるはず");
        assert!(matches!(err, BenchGateError::InvalidSignal(_)));
    }

    #[test]
    fn bench_signal_accepts_min_iterations_and_computes_median() {
        let signal = BenchSignal::from_measurements_pct(vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("5 件は下限ちょうどのため成功するはず");
        assert_eq!(signal.bench_measurements_pct.len(), 5);
        assert_eq!(signal.bench_median_pct, 3.0);
    }

    #[test]
    fn bench_signal_from_json_roundtrips_and_validates() {
        let signal = BenchSignal::from_measurements_pct(vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("5 件は下限ちょうどのため成功するはず");
        let json = serde_json::to_string(&signal).expect("シリアライズは成功するはず");
        let parsed =
            BenchSignal::from_json(&json).expect("検証済み BenchSignal の JSON は成功するはず");
        assert_eq!(parsed, signal);
    }

    #[test]
    fn bench_signal_from_json_rejects_tampered_median() {
        // `--signals` 注入経路（CLI 仕様書 1.2 節）を想定: bench_median_pct が系列と
        // 矛盾する JSON を直接デコードした場合に from_json が拒否することを検証する。
        let json = r#"{"bench_measurements_pct":[1.0,2.0,3.0,4.0,5.0],"bench_median_pct":-999.0}"#;
        let err = BenchSignal::from_json(json)
            .expect_err("bench_median_pct の改竄は from_json でも拒否されるはず");
        assert!(matches!(err, BenchGateError::InvalidSignal(_)));
    }

    #[test]
    fn bench_signal_validate_rejects_tampered_median() {
        // `--signals` 注入経路を想定: JSON デシリアライズ直後に bench_median_pct が
        // 系列と矛盾する値へ改竄されているケース。validate は再計算値と照合して拒否する。
        let tampered = BenchSignal {
            bench_measurements_pct: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bench_median_pct: -999.0,
        };
        let err = tampered
            .validate()
            .expect_err("bench_median_pct の改竄は拒否されるはず");
        assert!(matches!(err, BenchGateError::InvalidSignal(_)));
    }

    #[test]
    fn bench_signal_validate_rejects_non_finite_measurement() {
        let invalid = BenchSignal {
            bench_measurements_pct: vec![1.0, 2.0, f64::NAN, 4.0, 5.0],
            bench_median_pct: 3.0,
        };
        let err = invalid.validate().expect_err("NaN 混入は拒否されるはず");
        assert!(matches!(err, BenchGateError::InvalidSignal(_)));
    }

    #[test]
    fn report_median_agrees_with_bench_signal_validate_for_even_count() {
        // `crate::report::median`（判定レポート JSON の `bench_median_pct` 生成経路。
        // `main.rs` の `--signals` 注入フロー）と `BenchSignal::validate`（本モジュールの
        // 改竄検知）が同じ中央値定義（`bench_harness::median_q1_q3`）に一本化されている
        // ことの回帰テスト（Bugbot 指摘・PR #305・#107）。件数が偶数（n=6）の場合、
        // 「中央 2 値の平均」方式（旧 `report::median`）と index 選択方式
        // （`median_q1_q3`）は値が乖離するため、この n で検証しないと定義の再乖離を
        // 検出できない（n=5 等の奇数件では両方式が一致してしまい検出力がない）。
        let measurements = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let median = crate::report::median(&measurements)
            .expect("非空・非 NaN のため report::median は成功するはず");
        let signal = BenchSignal {
            bench_measurements_pct: measurements,
            bench_median_pct: median,
        };
        signal
            .validate()
            .expect("report::median が算出した中央値は BenchSignal::validate を通過するはず");
    }

    #[test]
    fn harness_bench_gate_rejects_fewer_than_min_iterations() {
        let gate = HarnessBenchGate;
        let config = fast_config();
        let err = gate
            .measure(&config, 4, &mut || {}, &mut || {})
            .expect_err("反復回数 4 は下限（5 回）未満のため拒否されるはず");
        assert!(matches!(err, BenchGateError::InvalidSignal(_)));
    }

    #[test]
    fn harness_bench_gate_measures_five_iterations_with_lightweight_workloads() {
        // 空クロージャは `Instant::elapsed` がゼロを返しうるため、`baseline_median_secs`
        // が `NonFiniteRatio` として拒否され success-path テストが偶発的に失敗しうる
        // （Bugbot 指摘・PR #305）。統合テスト（bench_gate_integration.rs）と同様に
        // `black_box` 経由で実測時間が確実に非ゼロになる軽量ワークロードへ置き換える。
        use std::hint::black_box;
        let gate = HarnessBenchGate;
        let config = fast_config();
        let mut baseline = || {
            let mut acc: u64 = 0;
            for i in 0..1_000u64 {
                acc = black_box(acc.wrapping_add(black_box(i)));
            }
            black_box(acc);
        };
        let mut candidate = || {
            let mut acc: u64 = 0;
            for i in 0..1_000u64 {
                acc = black_box(acc.wrapping_add(black_box(i)));
            }
            black_box(acc);
        };
        let signal = gate
            .measure(&config, MIN_BENCH_ITERATIONS, &mut baseline, &mut candidate)
            .expect("軽量ダミーワークロードは成功するはず");

        assert_eq!(signal.bench_measurements_pct.len(), MIN_BENCH_ITERATIONS);
        assert!(signal.bench_median_pct.is_finite());
        signal
            .validate()
            .expect("構築直後の BenchSignal は検証を通過するはず");
    }
}
