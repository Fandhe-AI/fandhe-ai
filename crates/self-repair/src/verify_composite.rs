//! 機能追加種別のループ完走実証（TASK-3.3c・イシュー #142）専用の検証ゲート合成。
//!
//! [`crate::stages::VerificationGate`] は `verify` 1 回の呼び出しにつき単一の
//! ゲート実装しか受け取らない設計（`stages.rs` 契約）だが、build/test/clippy
//! 3 ゲート（[`crate::verify_gates::CargoVerificationGate`]）とベンチゲート
//! （[`crate::verify_bench::SelfRepairBenchGate`]）を 1 回のループ実行内で
//! 両方通す「4 ゲート合成」は、[`crate::outcome::VerifiedEvidence::new`] が
//! `pub(crate)`（本クレート外からは構築不能。`outcome.rs` 参照）であるため、
//! 本クレート内（`tests/` の統合テストではなく `src/`）に実装する必要がある。
//! TASK-3.2（#136 系）が「4 ゲート合成」自体を明示的にスコープ外としていた
//! 結線点（`verify_gates.rs` モジュール冒頭ドキュメント参照）を、本実証
//! （TASK-3.3c）のために初めて満たす。
//!
//! `crates/self-repair/Cargo.toml`・`verify_gates.rs`・`report.rs` は変更せず
//! 新規ファイルとして追加する（並行実装中の #141〈TASK-3.3b〉との編集衝突を
//! 避けるため。実装計画リスク節）。
//!
//! # diff 由来シグナルの扱い（試行ごとではなくゲート単位で固定）
//! `lines_changed`／`api_broken`／`gaming_suspect`／`exclusion_rule_ids` は
//! [`crate::verify_gates::CargoVerificationGate`] と同じ契約（呼び出し元が
//! 実測した値を構築時に必須引数として渡す。未計測値を fail-open な既定値で
//! 埋めない。`.claude/rules/security.md` A08）を引き継ぐ。本モジュール自身は
//! diff 解析を行わない（実測は呼び出し元＝統合テストハーネスの責務）。
//!
//! **重要**: これらの値は [`FeatureAdditionCompositeGate`] インスタンス
//! 構築時に一度だけ渡され、[`crate::runner::SelfRepairLoop`] が同一インス
//! タンスを全試行で使い回す（`verify` は `&self` を取り、試行ごとに再構築
//! されない）ため、**試行ごとに再計測されるのではなくゲート単位で固定**
//! される。呼び出し元が複数の異なる候補（diff サイズの異なる複数試行）を
//! 扱う場合、いずれの候補の値を渡すかを呼び出し元が決める必要がある。
//! `crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs`
//! は「最終的に採用される候補（最後の試行）の diff を渡し、それ以前の
//! 誤実装候補は検証ゲート（build/test/clippy）不合格で `AdoptionJudge` へ
//! 到達しないため diff 由来シグナルが判定に使われない」という構成に依存
//! している（`stages.rs` の呼び出し順序契約: 検証 `Failed` の場合は
//! `AdoptionJudge::judge` を呼ばない）。誤実装候補が検証を通過しうる構成で
//! 本ゲートを再利用する場合はこの前提が崩れるため、試行ごとに異なる
//! ゲートインスタンスを構築するなど別の設計が必要になる。
//!
//! # ベンチ実行順序
//! ベンチは 3 ゲート全通過時のみ計測する（`guardrail::decision` の「ゲート
//! 全通過時のみベンチを計測する」契約と同じ順序。`verify_gates.rs` が固定で
//! 発行する `guardrail::BenchSignal::NotRun` を、全通過時に限り実測
//! `BenchSignal::Measured` へ差し替える）。

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use crate::error::SelfRepairError;
use crate::exec::CommandRunner;
use crate::outcome::VerifiedEvidence;
use crate::stages::{Proposal, VerificationGate, VerificationOutcome};
use crate::verify_bench::SelfRepairBenchGate;
use crate::verify_gates::CargoVerificationGate;

/// build/test/clippy（[`CargoVerificationGate`]）とベンチ
/// （[`SelfRepairBenchGate`]）を合成した [`VerificationGate`] 実装。
///
/// `R: CommandRunner` は [`CargoVerificationGate`] と同じ理由でテスト時の
/// 差し替えを可能にする総称化（本番経路は [`crate::exec::SystemCommandRunner`]）。
pub struct FeatureAdditionCompositeGate<R: CommandRunner> {
    cargo_gate: CargoVerificationGate<R>,
    bench_gate: SelfRepairBenchGate,
    /// ベンチ計測の反復回数（[`crate::verify_bench::MIN_BENCH_ITERATIONS`] 以上
    /// であることは `SelfRepairBenchGate::run` 側が強制するため、本型では
    /// 再検査しない）。
    bench_iterations: usize,
    /// 直近の `verify` が発行した [`VerifiedEvidence`]（`Passed` の場合のみ）。
    ///
    /// [`crate::runner::SelfRepairLoop::run`] は `AttemptOutcome::Adopted` に
    /// 証跡そのものを保持しない（`report.rs` 参照）ため、実証ハーネス
    /// （TASK-3.3c・#142 統合テスト）が「ベンチが実際に `Measured` として
    /// 計測されたか」を事後確認するための観測点として保持する。取り込み
    /// 判断自体はこのフィールドを経由せず `verify` の戻り値（型で検証迂回を
    /// 封じた `VerifiedEvidence`）を直接使う（A08: 判定の迂回経路を作らない。
    /// 本フィールドは読み取り専用の観測用途に限る）。
    ///
    /// `Rc` で包むのは、[`crate::runner::SelfRepairLoop::new`] が本型を
    /// 値として所有権ごと受け取るため（呼び出し元は `evidence_sink()` で
    /// `Rc` の複製を `SelfRepairLoop::new` 呼び出し前に取得しておくことで、
    /// ループ実行後も観測できる）。
    last_evidence: Rc<RefCell<Option<VerifiedEvidence>>>,
    /// 直近の `verify` が計測した生のベンチ劣化率系列
    /// （[`crate::verify_bench::BenchSignal`]。`bench_measurements_pct`
    /// 5 件以上・`bench_median_pct`）。`last_evidence` が保持する
    /// `guardrail::BenchSignal`（3 分岐判定用の列挙型。`Measured` の場合
    /// `median_pct` のみ）は判定に使う要約値のみで、完走ログに「何回・
    /// どの値を計測したか」を記録するための生系列を保持しないため、
    /// 別途保持する（`verify_composite.rs` 冒頭「ベンチ実行順序」節・
    /// `bench` 変換ドキュメント参照）。
    last_bench_measurement: Rc<RefCell<Option<crate::verify_bench::BenchSignal>>>,
}

impl<R: CommandRunner> FeatureAdditionCompositeGate<R> {
    /// `lines_changed`／`api_broken`／`gaming_suspect`／`exclusion_rule_ids` は
    /// 呼び出し元が実測した diff 由来シグナルをそのまま渡す必須引数とする
    /// （[`CargoVerificationGate::new`] と同じ設計判断。モジュール冒頭ドキュメント
    /// 参照）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workspace: PathBuf,
        runner: R,
        lines_changed: u64,
        api_broken: bool,
        gaming_suspect: bool,
        exclusion_rule_ids: Vec<String>,
        bench_iterations: usize,
    ) -> Self {
        FeatureAdditionCompositeGate {
            cargo_gate: CargoVerificationGate::new(
                workspace,
                runner,
                lines_changed,
                api_broken,
                gaming_suspect,
                exclusion_rule_ids,
            ),
            bench_gate: SelfRepairBenchGate::new(),
            bench_iterations,
            last_evidence: Rc::new(RefCell::new(None)),
            last_bench_measurement: Rc::new(RefCell::new(None)),
        }
    }

    /// 直近の `verify` が `Passed` を返した際に発行した [`VerifiedEvidence`]
    /// の複製（観測用途。型フィールド doc 参照）。`Failed`／未実行の場合は
    /// `None`。
    pub fn last_evidence(&self) -> Option<VerifiedEvidence> {
        self.last_evidence.borrow().clone()
    }

    /// `last_evidence` の `Rc` 複製を返す。呼び出し元がこの型を
    /// [`crate::runner::SelfRepairLoop::new`] へ値ごと渡す（所有権が
    /// ループへ移る）前に呼び出すことで、ループ実行後も観測点を保持できる
    /// （型フィールド doc 参照）。
    pub fn evidence_sink(&self) -> Rc<RefCell<Option<VerifiedEvidence>>> {
        Rc::clone(&self.last_evidence)
    }

    /// `last_bench_measurement` の `Rc` 複製を返す（`evidence_sink` と同じ
    /// 事前取得の理由。型フィールド doc 参照）。
    pub fn bench_measurement_sink(&self) -> Rc<RefCell<Option<crate::verify_bench::BenchSignal>>> {
        Rc::clone(&self.last_bench_measurement)
    }
}

impl<R: CommandRunner> VerificationGate for FeatureAdditionCompositeGate<R> {
    fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
        // 1. build/test/clippy（`CargoVerificationGate::verify` に委譲）。
        //    不合格ならベンチを計測せずここで終わる（ゲート全通過時のみ計測
        //    する順序契約。モジュール冒頭ドキュメント参照）。
        let evidence = match self.cargo_gate.verify(proposal)? {
            VerificationOutcome::Failed { reason } => {
                return Ok(VerificationOutcome::Failed { reason });
            }
            VerificationOutcome::Passed(evidence) => evidence,
        };

        // 2. ベンチ（`SelfRepairBenchGate::run` に委譲）。sandbox の実バイナリを
        //    2 系統ビルドして計測する構成は本実証のスコープを超えるため、
        //    `bench_gate_completion.rs` の既存慣行（合成ワークロード計測）を
        //    踏襲し、leaky_relu の forward 計算相当の軽量 CPU ワークロードで
        //    baseline/candidate を計測する（実装計画 4 節）。
        let mut baseline_workload = leaky_relu_like_workload();
        let mut candidate_workload = leaky_relu_like_workload();
        let measured = self
            .bench_gate
            .run(
                self.bench_iterations,
                &mut baseline_workload,
                &mut candidate_workload,
            )
            .map_err(|error| SelfRepairError::Verification {
                attempt: proposal.attempt,
                reason: format!("ベンチゲートが失敗しました: {error}"),
            })?;
        *self.last_bench_measurement.borrow_mut() = Some(measured.clone());

        // `SelfRepairBenchGate::run` が返す型（`crate::verify_bench::BenchSignal`
        // = `guardrail::bench_gate::BenchSignal`。劣化率系列 DTO）と
        // `VerifiedEvidence::new` が要求する型（`guardrail::BenchSignal`
        // = `guardrail::decision::BenchSignal`。3 分岐判定の列挙型）は
        // 同名だが別モジュール定義の別型である
        // （`crates/guardrail/src/bench_gate.rs` の `BenchSignal` 構造体 と
        // `crates/guardrail/src/decision.rs` の `BenchSignal` 列挙型。
        // `crates/guardrail/tests/bench_gate_decision_integration.rs` が示す
        // 変換経路と同じ `guardrail::median_gate::bench_signal_from_measurements`
        // で変換する。中央値算出ロジックを本クレートで再実装しない）。
        let bench = guardrail::median_gate::bench_signal_from_measurements(
            &measured.bench_measurements_pct,
        )
        .map_err(|error| SelfRepairError::Verification {
            attempt: proposal.attempt,
            reason: format!("ベンチ計測系列の判定変換に失敗しました: {error}"),
        })?;

        // 3. `gates`（3 ゲート実測）はそのまま、`bench` のみ実測値へ差し替えた
        //    証跡を発行する。他の 4 シグナル（lines_changed 等）は構築時に
        //    受け取った値をそのまま `CargoVerificationGate` 経由で保持している
        //    ため、ここでは `evidence` から読み戻すだけで再計測しない。
        let merged = VerifiedEvidence::new(
            evidence.attempt(),
            evidence.proposal_summary().to_string(),
            format!("{} bench=measured", evidence.gate_report()),
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

/// leaky_relu の forward 計算相当の軽量 CPU ワークロード。
///
/// `crates/self-repair/tests/bench_gate_completion.rs::cpu_workload` と同じ
/// 理由（`Instant::elapsed` がゼロを返し `NonFiniteRatio` として偶発的に
/// 失敗しうる）で `black_box` を経由し実測時間を確実に非ゼロにする。反復数
/// 10,000 は `crates/guardrail/tests/bench_gate_decision_integration.rs::
/// busy_workload` の既存慣行と同じ規模であり、閾値（`bench_median_max_pct`
/// 既定 5.0%）に対しタイマー・スケジューリングノイズの寄与を相対的に
/// 小さく保つ（5 回連続実行での実測観察: 反復 1,000 では単発サンプルが
/// 5% 付近まで振れることがあったため、閾値に対する安全マージンを確保する
/// 目的で規模を引き上げた。ガードレール閾値・許容誤差そのものは変更しない）。
fn leaky_relu_like_workload() -> impl FnMut() {
    || {
        let mut acc = 0.0f32;
        for i in 0..10_000u32 {
            let x = std::hint::black_box(i as f32 - 5_000.0);
            let y = if x >= 0.0 { x } else { 0.1 * x };
            acc = std::hint::black_box(acc + y);
        }
        std::hint::black_box(acc);
    }
}
