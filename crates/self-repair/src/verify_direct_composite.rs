//! ベンチゲートの候補 diff 直接実測を組み込んだ真の 4 ゲート合成
//! （TASK-3.2a・イシュー #137）。
//!
//! [`crate::verify_composite::FeatureAdditionCompositeGate`]（既存。TASK-3.3c・
//! #142）は 2 つの制約を持つ:
//! 1. diff 由来 4 シグナル（`lines_changed`／`api_broken`／`gaming_suspect`／
//!    `exclusion_rule_ids`）が**構築時に一度だけ**固定され、試行ごとに再計測
//!    されない（同モジュール冒頭ドキュメント「diff 由来シグナルの扱い」）
//! 2. ベンチは baseline／candidate 双方が**同一の合成ワークロード**
//!    （`leaky_relu_like_workload`）であり、候補実装固有の性能劣化を検出できない
//!    （#139 reopen コメント・完走判定基準 5）
//!
//! 本モジュールの [`RepairCompositeGate`] はこの 2 点を解消する:
//! - `verify` 呼び出しのたび [`crate::diff_signals::measure_diff_signals`] で
//!   4 シグナルを実測し直す
//! - ベンチは [`crate::verify_bench_direct::DirectBenchRunner`] で候補 diff
//!   （`sandbox_root` の実際の作業木）を baseline commit との比較で直接実測する
//!
//! `FeatureAdditionCompositeGate` は既存の統合テスト
//! （`tests/feature_addition_loop_completion_task_3_3c.rs`）が依存するため
//! **変更しない**（実装計画 #137 §3.3。移行判断は #141／#142 の再実証完了後）。
//! 本モジュールは新規追加であり両ゲートは共存する。
//!
//! # 判定ロジックへの委譲（閾値・下限を再実装しない）
//! - build/test/clippy の逐次実行・fail-fast 判定は
//!   [`crate::verify_gates::CargoVerificationGate`] にそのまま委譲する
//! - ベンチの計測実行系（warmup 20+・計測 20+・[`crate::verify_bench::
//!   MIN_BENCH_ITERATIONS`] 以上の反復・中央値算出）は
//!   [`crate::verify_bench::SelfRepairBenchGate`]（→ `guardrail::bench_gate::
//!   HarnessBenchGate` → `bench_harness::run`）にそのまま委譲する
//! - 判定用 `guardrail::BenchSignal::Measured` への変換は
//!   `guardrail::median_gate::bench_signal_from_measurements`（`
//!   FeatureAdditionCompositeGate` と同一の変換関数）を再利用する
//! - `guardrail.toml`／`policy-exclusion.toml`／`MeasurementConfig` 下限・
//!   `MIN_BENCH_ITERATIONS`（5）はいずれも読み取るのみで変更しない
//!   （`.claude/rules/security.md`「ガードレール閾値・許容誤差の変更は
//!   ユーザー承認必須」）
//!
//! # 実行順序（ゲート全通過時のみベンチを計測。既存 2 ゲートと同じ契約）
//! diff 実測 → build/test/clippy → （通過時のみ）候補 diff 直接ベンチ実測。

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use crate::diff_signals::measure_diff_signals;
use crate::error::SelfRepairError;
use crate::exec::CommandRunner;
use crate::outcome::VerifiedEvidence;
use crate::stages::{Proposal, VerificationGate, VerificationOutcome};
use crate::verify_bench::BenchSignal as DirectBenchSignal;
use crate::verify_bench_direct::{DirectBenchRunner, DirectBenchSpec};
use crate::verify_gates::CargoVerificationGate;

/// build/test/clippy（[`CargoVerificationGate`]）と候補 diff 直接ベンチ実測
/// （[`DirectBenchRunner`]）を合成した [`VerificationGate`] 実装。
///
/// `R: CommandRunner + Clone` は試行ごとに `CargoVerificationGate<R>` を
/// **新規構築**するために `Clone` 境界を要求する（diff 由来シグナルを試行ごとに
/// 実測し直すため、構築済みの `CargoVerificationGate` を使い回せない。
/// `crate::exec::SystemCommandRunner` は本イシューで `Clone` を実装済み）。
pub struct RepairCompositeGate<R: CommandRunner + Clone> {
    /// 検証対象ワークスペース（build/test/clippy の cwd）。
    workspace: PathBuf,
    /// sandbox リポジトリのルート（diff・ポリシー除外評価・ベンチ実測の基点）。
    sandbox_root: PathBuf,
    /// diff の起点（検出前コミットの sha）。
    baseline_commit: String,
    policy_exclusion_path: PathBuf,
    bench_bin: String,
    workload_sources: Vec<String>,
    bench_iterations: usize,
    runner: R,
    /// 直近の `verify` が `Passed` を返した際の [`VerifiedEvidence`]（観測用途。
    /// `FeatureAdditionCompositeGate` と同じ理由・同じ設計）。
    last_evidence: Rc<RefCell<Option<VerifiedEvidence>>>,
    /// 直近の候補 diff 直接ベンチ実測の生系列（観測用途。`FeatureAdditionCompositeGate`
    /// の `last_bench_measurement` と同じ理由）。
    last_bench_measurement: Rc<RefCell<Option<DirectBenchSignal>>>,
}

/// [`RepairCompositeGate::new`] の設定パラメータ。フィールド数が多いため
/// 構造化して `clippy::too_many_arguments` を型レベルで回避する
/// （`.claude/rules/coding-rust.md`「`#[allow]` の安易な追加で黙らせない」。
/// `tests/revalidation_bug_fix.rs::ReportJsonInput` と同じ一貫した方針）。
pub struct RepairCompositeGateSpec<R: CommandRunner + Clone> {
    pub workspace: PathBuf,
    pub sandbox_root: PathBuf,
    pub baseline_commit: String,
    pub policy_exclusion_path: PathBuf,
    pub bench_bin: String,
    pub workload_sources: Vec<String>,
    pub bench_iterations: usize,
    pub runner: R,
}

impl<R: CommandRunner + Clone> RepairCompositeGate<R> {
    pub fn new(spec: RepairCompositeGateSpec<R>) -> Self {
        RepairCompositeGate {
            workspace: spec.workspace,
            sandbox_root: spec.sandbox_root,
            baseline_commit: spec.baseline_commit,
            policy_exclusion_path: spec.policy_exclusion_path,
            bench_bin: spec.bench_bin,
            workload_sources: spec.workload_sources,
            bench_iterations: spec.bench_iterations,
            runner: spec.runner,
            last_evidence: Rc::new(RefCell::new(None)),
            last_bench_measurement: Rc::new(RefCell::new(None)),
        }
    }

    /// 直近の `verify` が `Passed` を返した際に発行した [`VerifiedEvidence`]
    /// の複製（観測用途。`FeatureAdditionCompositeGate::last_evidence` と同じ）。
    pub fn last_evidence(&self) -> Option<VerifiedEvidence> {
        self.last_evidence.borrow().clone()
    }

    /// `last_evidence` の `Rc` 複製を返す（[`crate::runner::SelfRepairLoop::new`]
    /// へ値ごと渡す前に呼び出すことでループ実行後も観測できる。
    /// `FeatureAdditionCompositeGate::evidence_sink` と同じ）。
    pub fn evidence_sink(&self) -> Rc<RefCell<Option<VerifiedEvidence>>> {
        Rc::clone(&self.last_evidence)
    }

    /// 直近の候補 diff 直接ベンチ実測の生系列を観測するための `Rc` 複製
    /// （`evidence_sink` と同じ事前取得の理由）。`signal_source: "measured"`
    /// 付きレポート組み込み（#141／#142 のスコープ）はこの観測点を消費する
    /// 想定（out-of-scope-tracking.md §8）。
    pub fn bench_measurement_sink(&self) -> Rc<RefCell<Option<DirectBenchSignal>>> {
        Rc::clone(&self.last_bench_measurement)
    }
}

impl<R: CommandRunner + Clone> VerificationGate for RepairCompositeGate<R> {
    fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
        // 1. diff 由来 4 シグナルを試行ごとに実測する（候補適用直後の作業木を
        //    対象。`FeatureAdditionCompositeGate` の「構築時固定」制約を解消
        //    する本モジュールの主眼。モジュール冒頭ドキュメント参照）。
        let diff_signals = measure_diff_signals(
            &self.runner,
            &self.sandbox_root,
            &self.baseline_commit,
            &self.policy_exclusion_path,
        )
        .map_err(|error| SelfRepairError::Verification {
            attempt: proposal.attempt,
            reason: format!("diff 由来シグナルの実測に失敗しました: {error}"),
        })?;

        // 2. build/test/clippy（`CargoVerificationGate::verify` に委譲）。
        //    実測した diff 由来シグナルで試行ごとに新規構築する（`R: Clone`
        //    境界を要求する理由。型冒頭ドキュメント参照）。不合格ならベンチを
        //    計測せずここで終わる（ゲート全通過時のみ計測する順序契約。
        //    `verify_composite.rs` と同じ）。
        let cargo_gate = CargoVerificationGate::new(
            self.workspace.clone(),
            self.runner.clone(),
            diff_signals.lines_changed,
            diff_signals.api_broken,
            diff_signals.gaming_suspect,
            diff_signals.exclusion_rule_ids.clone(),
        );
        let evidence = match cargo_gate.verify(proposal)? {
            VerificationOutcome::Failed { reason } => {
                return Ok(VerificationOutcome::Failed { reason });
            }
            VerificationOutcome::Passed(evidence) => evidence,
        };

        // 3. 候補 diff 直接ベンチ実測（`DirectBenchRunner::measure` に委譲。
        //    外部タイミング方式・ワークロードソースのピン留め検査は
        //    `verify_bench_direct.rs` の責務であり、本モジュールでは再実装
        //    しない）。
        let bench_spec = DirectBenchSpec {
            sandbox_root: self.sandbox_root.clone(),
            baseline_commit: self.baseline_commit.clone(),
            bench_bin: self.bench_bin.clone(),
            workload_sources: self.workload_sources.clone(),
            bench_iterations: self.bench_iterations,
        };
        let measured = DirectBenchRunner
            .measure(&self.runner, &bench_spec)
            .map_err(|error| SelfRepairError::Verification {
                attempt: proposal.attempt,
                reason: format!("候補 diff 直接ベンチ実測に失敗しました: {error}"),
            })?;
        *self.last_bench_measurement.borrow_mut() = Some(measured.clone());

        // 4. 判定用 `guardrail::BenchSignal::Measured` へ変換する（中央値算出
        //    ロジックの再実装を避け `guardrail::median_gate` に一本化する。
        //    `verify_composite.rs` と同じ変換経路・同じ理由）。
        let bench = guardrail::median_gate::bench_signal_from_measurements(
            &measured.bench_measurements_pct,
        )
        .map_err(|error| SelfRepairError::Verification {
            attempt: proposal.attempt,
            reason: format!("ベンチ計測系列の判定変換に失敗しました: {error}"),
        })?;

        // 5. `gates`（3 ゲート実測）・diff 由来 4 シグナル（試行ごと実測値）は
        //    そのまま、`bench` のみ実測値へ差し替えた証跡を発行する。
        //    `gate_report` に `bench=measured-direct` を付与し、合成ワークロード
        //    版（`bench=measured`）と区別可能にする（#141／#142 が
        //    `signal_source` 付きレポートを組む際の識別点）。
        let merged = VerifiedEvidence::new(
            evidence.attempt(),
            evidence.proposal_summary().to_string(),
            format!("{} bench=measured-direct", evidence.gate_report()),
            evidence.gates(),
            bench,
            evidence.lines_changed(),
            evidence.api_broken(),
            evidence.gaming_suspect(),
            evidence.exclusion_rule_ids().to_vec(),
        );
        *self.last_evidence.borrow_mut() = Some(merged.clone());
        Ok(VerificationOutcome::Passed(merged))
    }
}
