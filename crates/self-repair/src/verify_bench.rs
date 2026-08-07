//! 検証フェーズのベンチゲート（TASK-3.2a・イシュー #137）。
//!
//! `self-repair` の検証（VerificationGate）4 ゲートのうちベンチゲートの計測系を、
//! `guardrail::bench_gate`（`bench-harness` 付け替え済み・TASK-4.1d・#107）経由で
//! 実行する。判定ロジック（5 回以上の計測・変化率の中央値採用。REQ-4）自体は
//! 一切変更せず、`guardrail::bench_gate::HarnessBenchGate` が内部で呼ぶ
//! `bench_harness::run`（warmup 20+・計測 20+・中央値採用。TASK-8.1 プロトコル）に
//! 完全に委ねる（実装計画 #137 §3.2）。
//!
//! # 依存方向: guardrail 経由（bench-harness へ直接依存しない）
//!
//! `self-repair` の `Cargo.toml` は `bench-harness` への依存を持たない
//! （`crates/self-repair/Cargo.toml` 参照。`lib.rs` の「`bench-harness` へ直接依存を
//! 重ねず `guardrail` 経由に一本化する」方針と同じ）。`guardrail::bench_gate::
//! BenchGateRunner::measure` の計測設定（`bench_harness::MeasurementConfig`）は
//! `self-repair` からは型名として参照できない（`guardrail` が `pub use` で
//! 再輸出していないため）。一方で判定結果型・エラー内部型（[`BenchSignal`]・
//! [`BenchGateError`]）は本モジュールが `guardrail::bench_gate` から
//! `pub use` で再輸出しており、呼び出し側がこれらを名指しするために
//! `guardrail::bench_gate` を直接 import する必要はない。本モジュールは計測設定を公開 API に露出させず、
//! `Default::default()`（spec 下限 20/20・`bench-harness/src/protocol.rs` 参照）を
//! 型推論のみで渡すことで、`bench_harness` を直接名指しせずに済ませる（TASK-3.3
//! で計測対象ワークロード・設定を定義する際の拡張点は、本モジュールを経由せず
//! `guardrail::bench_gate` 側への設定注入経路を新設する形になる想定であり、
//! 本イシューのスコープ外）。
//!
//! # TASK-3.1（検証フェーズ骨格）未完時の暫定配置
//!
//! 本モジュール実装時点（イシュー #137）で TASK-3.1c（#134: 検証フェーズの
//! `VerificationGate` trait 群）は main 未マージのため、本モジュールは独立した
//! ベンチゲート実装として配置する。TASK-3.1c マージ後は [`SelfRepairBenchGate`] を
//! `VerificationGate` の一実装として結線する想定であり、結線点は
//! [`SelfRepairBenchGate::run`] のシグネチャ（baseline／candidate クロージャを受けて
//! 判定用の劣化率系列を返す）に閉じている（実装計画 #137 §3.3・§9 リスク）。

pub use guardrail::bench_gate::{BenchGateError, BenchSignal, MIN_BENCH_ITERATIONS};
use guardrail::bench_gate::{BenchGateRunner, HarnessBenchGate};
use std::fmt;

/// 検証フェーズのベンチゲート実行時に自己修復ループへ返すエラー。
///
/// `guardrail::bench_gate::BenchGateError` を握り潰さず包む（本番経路で `unwrap()` /
/// `expect()` を使わない方針。`.claude/rules/coding-rust.md`）。ベンチゲート失敗は
/// 検証不合格として fail-closed に扱い、自動適用へフォールバックしない
/// （`docs/guardrail-self-repair-cli.md` 2.3 節の契約）。
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyBenchError {
    /// `guardrail::bench_gate` 側の計測・検証エラーをそのまま包む。
    Gate(BenchGateError),
}

impl fmt::Display for VerifyBenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyBenchError::Gate(err) => write!(f, "検証フェーズのベンチゲート失敗: {err}"),
        }
    }
}

impl std::error::Error for VerifyBenchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // `BenchGateError` は `std::error::Error` を実装済み（guardrail 側）。
            // ここで連結しないと `std::error::Error::source()` ベースのエラーチェーン
            // 走査（ロギング・診断ツール）が本型で途切れるため、内部エラーへ委譲する。
            VerifyBenchError::Gate(err) => Some(err),
        }
    }
}

impl From<BenchGateError> for VerifyBenchError {
    fn from(err: BenchGateError) -> Self {
        VerifyBenchError::Gate(err)
    }
}

/// 検証フェーズのベンチゲート（TASK-3.2a）。
///
/// `guardrail::bench_gate::HarnessBenchGate`（`BenchGateRunner` の本番実装。
/// `bench_harness::run` を計測実行系として呼ぶ）に固定して保持し、baseline
/// （修復前）／candidate（AI 生成の修復後）のワークロードクロージャを受けて
/// `guardrail::bench_gate::BenchSignal`（劣化率系列・中央値）を返す。
///
/// `BenchGateRunner` を汎用（ジェネリック）にせず `HarnessBenchGate` 固定とする
/// のは意図的な設計判断である。`BenchGateRunner::measure` は反復回数下限
/// （[`MIN_BENCH_ITERATIONS`]）の検査を実装側に委ねる契約であり、外部から任意の
/// runner を注入できる API は REQ-4 の下限検査を迂回しうる経路になる
/// （`.claude/rules/security.md`「判定の迂回経路を作らない」）。加えて
/// `BenchGateRunner` を外部クレートで実装するには `measure` シグネチャの
/// `bench_harness::MeasurementConfig` を型名として書く必要があるが、`guardrail`
/// はこの型を `pub use` で再輸出しておらず（モジュール冒頭ドキュメント「依存方向」
/// 参照）、`self-repair` からも他クレートからも実装不可能である。差し替え可能な
/// 注入点（テスト用モック・TASK-3.1c 結線）を設けるには `guardrail` 側で
/// `MeasurementConfig` の再輸出が必要だが、`guardrail` は本イシューでは編集しない
/// （実装計画 #137 §4「`crates/guardrail/` は編集しない」）。この制約は
/// out-of-scope-tracking.md に従いスコープ外事項として記録する（#138 との統合検討時）。
#[derive(Debug, Default, Clone, Copy)]
pub struct SelfRepairBenchGate {
    runner: HarnessBenchGate,
}

impl SelfRepairBenchGate {
    /// 本番用ベンチゲートを構築する（`bench_harness::run` を計測実行系として使う
    /// `HarnessBenchGate` を runner とする）。
    pub fn new() -> Self {
        Self::default()
    }

    /// baseline・candidate ワークロードを `iterations` 回（[`MIN_BENCH_ITERATIONS`] 以上）
    /// 計測し、[`BenchSignal`] を返す。
    ///
    /// 計測設定は `bench_harness::MeasurementConfig` の spec 下限（`Default`。
    /// warmup 20・計測 20。`bench-harness/src/protocol.rs`）を型推論のみで渡す
    /// （モジュール冒頭ドキュメント「依存方向」参照。下限自体は `HarnessBenchGate`
    /// が既に強制するため本モジュールで別途検査しない）。計測実行系そのものは
    /// `runner`（`bench_harness::run` を呼ぶ `HarnessBenchGate`）に委ね、本メソッドは
    /// 判定ロジックへ渡す前段のエラー正規化のみを担う（受け入れ条件「ベンチゲートが
    /// bench-harness 経由で完走する」は `HarnessBenchGate::measure` への委譲で満たす）。
    ///
    /// # Errors
    ///
    /// 計測・検証に失敗した場合 [`VerifyBenchError`]。
    pub fn run(
        &self,
        iterations: usize,
        baseline: &mut dyn FnMut(),
        candidate: &mut dyn FnMut(),
    ) -> Result<BenchSignal, VerifyBenchError> {
        self.runner
            .measure(&Default::default(), iterations, baseline, candidate)
            .map_err(VerifyBenchError::from)
    }
}
