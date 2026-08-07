//! 5 回以上計測の中央値を判定入力（[`crate::decision::BenchSignal`]）へ変換する結線
//! （TASK-4.4a・イシュー #112・REQ-4）。
//!
//! REQ-4 受け入れ基準（`docs/spec/04-requirements.md` L116）は「ベンチマーク計測は
//! 5 回以上実施し、変化率の中央値を採用すること。**単発計測での閾値判定は行わないこと**」
//! と定める。[`crate::decision`]（TASK-4.1c・#106）の `BenchSignal::Measured` は既に
//! 計測済みの `median_pct: f64` を受け取る受け口であり、回数下限の強制自体は
//! 「#105／#107 の計測系〈bench-harness 付け替え〉が担う契約」とドキュメントされている
//! （`crates/guardrail/src/decision.rs` の `BenchSignal` ドキュメント参照）。本モジュールは
//! その「回数下限・有限性の検証」を実装し、検証を通過した劣化率系列からのみ
//! `BenchSignal::Measured` を構築できる唯一の公開経路を提供する。
//!
//! # 依存関係・中央値定義の一元化
//!
//! 中央値の算出は本モジュールで独自実装せず、[`bench_harness::median_q1_q3`]
//! （median-of-halves 方式。PoC-v2-1 実測踏襲）を唯一の定義として呼び出す
//! （`.claude/rules/coding-rust.md`「バックエンド間許容誤差の定義を単独変更しない」と
//! 同種の「定義の二重管理防止」。計画書 §3.2）。
//!
//! # 未マージ PR との関係（実装時点の既知事項。イシュー #112 計画書 §2）
//!
//! 実装時点（イシュー #112）で `crates/guardrail` には以下 3 件の未マージ PR が並行して
//! 存在し、いずれも main 未マージのため本モジュールはそれらのファイルを編集・依存しない
//! （`.claude/rules/delegation-impl.md`「複数 Agent に同一ファイルを並行編集させない」の
//! 趣旨を踏襲し、未マージ PR のファイルとの衝突を避ける）:
//!
//! - PR #303（TASK-4.1a・CLI 骨格。`cli.rs`／`signals.rs`／`config.rs` 等）
//! - PR #305（TASK-4.1d・`bench_gate.rs`）。本モジュールと責務が重なる箇所があるが、
//!   `bench_gate::BenchGateRunner`／`HarnessBenchGate`（baseline／candidate の反復計測
//!   実行系・`--signals` JSON 注入向け DTO）は本モジュールが再実装しない範囲であり、
//!   マージ後は本モジュールとの統合（`bench_gate::BenchSignal` → `median_gate` を経由せず
//!   直接 `decision::BenchSignal` へ変換する等）を後続イシュー（親 #111 への追記）で
//!   検討する
//! - PR #306（TASK-4.1b・5 条件の閾値体系。`decision.rs`／`signals.rs`／`config.rs`）
//!
//! 本モジュールは上記いずれのファイルにも依存せず、main に既にマージ済みの
//! [`crate::decision::BenchSignal`]（公開 enum。フィールドは公開だが `Measured` の構築元を
//! 本モジュール経由に限定する運用上の契約とする。型レベルでの強制〈`decision::BenchSignal`
//! 自体のフィールド非公開化〉は `decision.rs` の変更を伴うため #306 マージ後の後続作業とし、
//! 本イシューでは行わない）と `bench_harness`（TASK-8.1・マージ済み）のみに依存する。
//!
//! 上記のフィールド非公開化・PR #305（`bench_gate.rs`）との統合は
//! `out-of-scope-tracking.md` の規約に従い親イシュー #111 へ追跡コメントを記録済み
//! （<https://github.com/Fandhe-AI/rust-ai-library/issues/111#issuecomment-5218294865>）。

use crate::decision::BenchSignal;
use std::fmt;

/// REQ-4「5 回以上」の下限（劣化率系列の反復回数）。
///
/// `bench_harness::protocol::MIN_ITERATIONS`（1 反復内の warmup／計測サンプル数下限。
/// 20/20）とは別レイヤの下限である。「1 反復」はここでは「劣化率 1 標本」を指し、
/// `bench_harness::run` 1 回の呼び出し（内部で warmup 20+・計測 20+ を取り中央値を返す）
/// から baseline・candidate 各 1 つの `median_secs` の比を取ったものが 1 標本となる
/// （PR #305 `bench_gate.rs` の設計と同一の用語整理。計画書 §2 のスコープ外事項参照）。
/// 下限を回避する公開 API は設けない（`.claude/rules/security.md`: 閾値の単独緩和禁止）。
pub const MIN_BENCH_ITERATIONS: usize = 5;

/// 本モジュールが検出する検証失敗の型付きエラー。
///
/// 本番経路で `unwrap()`/`expect()` を使わない方針（`.claude/rules/coding-rust.md`）に従う。
/// いずれの分岐も「判定不能」として扱われ、呼び出し側（CLI 層。将来 #303 マージ後）は
/// `GuardrailExitCode::InternalError`（終了コード `1`）へ写像する想定であり、
/// 自動適用（`0`）へフォールバックしない（`docs/guardrail-self-repair-cli.md` §2.3
/// fail-closed 契約。`.claude/rules/security.md` A08）。
#[derive(Debug, Clone, PartialEq)]
pub enum MedianGateError {
    /// 計測件数が [`MIN_BENCH_ITERATIONS`] 未満（REQ-4「5 回以上」への違反）。
    TooFewMeasurements { got: usize, min: usize },
    /// 系列中に非有限値（NaN／inf）が混入している。単発の異常計測が中央値を汚染して
    /// 誤った自動適用へつながることを防ぐため、系列全体を fail-closed で拒否する。
    NonFiniteMeasurement { index: usize, value: f64 },
    /// `bench_harness::median_q1_q3` 側の集計失敗（空スライス・NaN 混入）をそのまま伝播する。
    /// 上記 2 分岐で事前検査済みのため通常到達しないが、`bench_harness` 側の契約変更に
    /// 対しても黙って握り潰さず防御的に伝播する。
    Aggregation(String),
}

impl fmt::Display for MedianGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MedianGateError::TooFewMeasurements { got, min } => write!(
                f,
                "計測件数が下限未満です（REQ-4「5 回以上」）: 指定件数={got}, 下限={min}"
            ),
            MedianGateError::NonFiniteMeasurement { index, value } => write!(
                f,
                "計測系列に非有限値（NaN／inf）が混入しています: index={index}, value={value}"
            ),
            MedianGateError::Aggregation(msg) => write!(f, "中央値算出に失敗しました: {msg}"),
        }
    }
}

impl std::error::Error for MedianGateError {}

/// 劣化率系列（%。正 = 劣化、負 = 改善）を検証し、検証済み
/// [`crate::decision::BenchSignal::Measured`] を構築する唯一の公開経路。
///
/// REQ-4 受け入れ基準「単発計測での閾値判定は行わないこと」を実装レベルで担保する
/// ([`MIN_BENCH_ITERATIONS`] 未満は拒否）。中央値は呼び出し側からの信頼値を使わず、
/// 本関数が [`bench_harness::median_q1_q3`] で系列から再計算する。
///
/// # Errors
///
/// - `measurements_pct.len() < MIN_BENCH_ITERATIONS` の場合
///   [`MedianGateError::TooFewMeasurements`]
/// - 系列中に非有限値がある場合 [`MedianGateError::NonFiniteMeasurement`]
///   （どの要素かを特定できるよう `index` を保持する）
/// - `bench_harness::median_q1_q3` が失敗した場合 [`MedianGateError::Aggregation`]
///   （上記 2 検証を経ているため通常到達しない防御的分岐）
pub fn bench_signal_from_measurements(
    measurements_pct: &[f64],
) -> Result<BenchSignal, MedianGateError> {
    if measurements_pct.len() < MIN_BENCH_ITERATIONS {
        return Err(MedianGateError::TooFewMeasurements {
            got: measurements_pct.len(),
            min: MIN_BENCH_ITERATIONS,
        });
    }
    for (index, &value) in measurements_pct.iter().enumerate() {
        if !value.is_finite() {
            return Err(MedianGateError::NonFiniteMeasurement { index, value });
        }
    }

    let quartiles = bench_harness::median_q1_q3(measurements_pct)
        .map_err(|e| MedianGateError::Aggregation(e.to_string()))?;

    Ok(BenchSignal::Measured {
        median_pct: quartiles.median,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_fewer_than_min_iterations() {
        let err = bench_signal_from_measurements(&[1.0, 2.0, 3.0, 4.0])
            .expect_err("4 件は下限（5 件）未満のため拒否されるはず");
        assert_eq!(err, MedianGateError::TooFewMeasurements { got: 4, min: 5 });
    }

    #[test]
    fn rejects_empty_series() {
        let err =
            bench_signal_from_measurements(&[]).expect_err("空系列は下限未満のため拒否されるはず");
        assert_eq!(err, MedianGateError::TooFewMeasurements { got: 0, min: 5 });
    }

    #[test]
    fn accepts_min_iterations_and_computes_median() {
        let signal = bench_signal_from_measurements(&[1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("5 件は下限ちょうどのため成功するはず");
        assert_eq!(signal, BenchSignal::Measured { median_pct: 3.0 });
    }

    #[test]
    fn median_is_recomputed_not_trusted_from_caller() {
        // 呼び出し順に依存しない中央値算出（bench_harness::median_q1_q3 の
        // median-of-halves 方式に一致すること）を確認する。
        let signal = bench_signal_from_measurements(&[5.0, 1.0, 4.0, 2.0, 3.0])
            .expect("非空・非 NaN のため成功するはず");
        assert_eq!(signal, BenchSignal::Measured { median_pct: 3.0 });
    }

    #[test]
    fn rejects_nan_measurement() {
        let err = bench_signal_from_measurements(&[1.0, 2.0, f64::NAN, 4.0, 5.0])
            .expect_err("NaN 混入は拒否されるはず");
        assert!(matches!(
            err,
            MedianGateError::NonFiniteMeasurement { index: 2, .. }
        ));
    }

    #[test]
    fn rejects_infinite_measurement() {
        let err = bench_signal_from_measurements(&[1.0, 2.0, 3.0, 4.0, f64::INFINITY])
            .expect_err("inf 混入は拒否されるはず");
        assert!(matches!(
            err,
            MedianGateError::NonFiniteMeasurement { index: 4, .. }
        ));
    }

    #[test]
    fn accepts_more_than_min_iterations() {
        let signal = bench_signal_from_measurements(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0])
            .expect("7 件（下限超過）は成功するはず");
        // n=7: median idx = round(0.5*6)=3 -> ソート後 4 番目（0-index）の値。
        assert_eq!(signal, BenchSignal::Measured { median_pct: 4.0 });
    }
}
