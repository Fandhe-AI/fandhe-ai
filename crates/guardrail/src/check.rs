//! `guardrail check` の計測・判定オーケストレーション本体（TASK-4.1c・
//! イシュー #106）。
//!
//! `main.rs`（バイナリ。統合層のみ）から呼ばれ、injected 経路（`--signals`）・
//! measured 経路（本番相当計測）の両方を実装する。**いずれの経路も必ず
//! [`crate::decision::decide`] を経由して [`Report`] を構築する**
//! （`.claude/rules/security.md` A08「判定の迂回経路を作らない」。判定
//! ロジックへの到達不能を修正するのが本イシューの目的そのもの — #103 の
//! クローズ再検証コメント参照）。
//!
//! # スコープ境界（`.claude/rules/out-of-scope-tracking.md`）
//! - **bench 実計測**: measured 経路では常に [`BenchSignal::NotRun`] を
//!   構築する。CLI から起動できる bench ワークロードが v2 に未定義なため
//!   （`bench_gate.rs` は呼び出し側クロージャ前提で未結線。CI 側の bench
//!   ゲートは `verification-gate-bench.yml` に分離済み）。`decide()` は
//!   `NotRun` を「逸脱なし」として扱うため（`decision.rs` テスト
//!   `all_clean_yields_auto_apply` 参照）、bench 未計測でも auto_apply へ
//!   到達できる。CLI 結線自体は別イシュー（親 #111 系）で扱う
//! - **公開 API 破壊検出**: [`crate::checks::api_stability`] は PoC-3 パリティ
//!   （`pub fn`/`pub struct`/`pub enum` の行シグネチャ比較）に留め、
//!   `cargo public-api` 相当の意味論解析は行わない
//!
//! # 除外リスト評価の実行契約
//! ポリシー除外リスト評価（[`EvaluationContext::from_repo`] →
//! [`ExclusionEvaluation::evaluate`]）は measured 経路でゲート成否に**関わらず
//! 必ず実行**する（REQ-5・security.md「取り込み判断の根拠を追跡可能にする」）。
//! injected 経路（`--signals`）は diff 前提の評価であり実行できないため
//! `exclusion_rule_ids` は常に空 `Vec` を渡す（機械判定器単体の契約検証パス
//! という §1.1 の役割に整合）。
//!
//! # 信頼境界（measured 経路）
//! [`run_measured`] は `--repo` で指定された対象リポジトリ上で
//! `cargo build`/`cargo test --release`/`cargo clippy`（[`crate::gates`]）を
//! 実際に**実行**する。つまり対象リポジトリのビルドスクリプト・テスト
//! コード自体を実行する設計であり（v1・CI 側の bench ゲートと同等）、
//! `guardrail check`（measured 経路）は運用者が検証意図を持つリポジトリ
//! （CI・`self-repair` が扱う自リポジトリのワークツリー）のみを対象とする
//! 前提に立つ。任意の未信頼リポジトリを `--repo` に指定して実行する運用は
//! 想定しない。

use std::path::Path;

use crate::changeset;
use crate::checks;
use crate::config::Config;
use crate::decision::{BenchSignal, DecisionInput, GateSignal, GateSignals, decide};
use crate::error::GuardrailError;
use crate::exec::{CommandRunner, SystemCommandRunner};
use crate::gaming;
use crate::gates;
use crate::median_gate;
use crate::policy_exclusion::{self, EvaluationContext, ExclusionEvaluation};
use crate::report::{GateOutcome, Report, ReportInputs, SignalSource};
use crate::signals::{GateResult, Signals};

fn gate_signal_from_result(result: GateResult) -> GateSignal {
    match result {
        GateResult::Pass => GateSignal::Passed,
        GateResult::Fail => GateSignal::Failed,
    }
}

fn gate_outcome_from_signal(signal: GateSignal) -> GateOutcome {
    match signal {
        GateSignal::Passed => GateOutcome::Pass,
        // `Skipped`（先行ゲート失敗による未実行）もレポート上は「実行して
        // 合格したわけではない」という点で `Fail` 相当として表示する
        // （`GateOutcome` に skip 概念を持たせない §2.1 スキーマの制約）。
        GateSignal::Failed | GateSignal::Skipped => GateOutcome::Fail,
    }
}

/// injected 経路（`--signals` JSON。1.2 節「CI 契約検証パス」）。
///
/// `--signals` は環境変数 `GUARDRAIL_ALLOW_INJECTED_SIGNALS=1` の入口ガード
/// （`cli.rs`）を通過済みの前提で呼ばれる。除外リスト評価は行わない
/// （モジュール冒頭「除外リスト評価の実行契約」参照）。
pub fn run_injected(
    signals: &Signals,
    config: &Config,
    change_id: Option<String>,
) -> Result<Report, GuardrailError> {
    let gates = GateSignals {
        build: gate_signal_from_result(signals.build_result),
        test: gate_signal_from_result(signals.test_result),
        clippy: gate_signal_from_result(signals.clippy_result),
    };
    let all_passed = gates.build == GateSignal::Passed
        && gates.test == GateSignal::Passed
        && gates.clippy == GateSignal::Passed;

    // `DecisionInput::new` の実行順序契約（「ベンチはゲート全通過時のみ計測
    // する」）を守るため、ゲートが全通過でない場合は入力された
    // `bench_measurements_pct` を無視し `NotRun` を渡す（そのまま渡すと
    // `InconsistentDecisionInput` で拒否される）。
    let bench = if all_passed {
        median_gate::bench_signal_from_measurements(&signals.bench_measurements_pct).map_err(
            |e| GuardrailError::InvalidInput(format!("bench_measurements_pct の検証に失敗: {e}")),
        )?
    } else {
        BenchSignal::NotRun
    };
    let bench_median_pct = match bench {
        BenchSignal::Measured { median_pct } => median_pct,
        BenchSignal::NotRun => 0.0,
    };
    // レポート上の `bench_measurements_pct` は `bench_median_pct` の算出元
    // 系列である契約（§2.1「bench_median_pct は bench_measurements_pct の
    // 中央値である」）を保つ。ゲート未全通過で `bench` を `NotRun` に強制
    // した場合は計測自体が行われていないため、入力された
    // `signals.bench_measurements_pct`（実行されなかった計測値）をレポート
    // にそのまま転記せず空にする（Cursor Bugbot 指摘 #337
    // discussion_r3738322999 対応）。
    let bench_measurements_pct = if all_passed {
        signals.bench_measurements_pct.clone()
    } else {
        Vec::new()
    };

    let input = DecisionInput::new(
        &config.thresholds,
        signals.lines_changed,
        gates,
        signals.public_api_broken,
        signals.gaming_suspected,
        bench,
        Vec::new(),
    )?;
    let decision = decide(&input)?;

    let inputs = ReportInputs {
        signal_source: SignalSource::Injected,
        change_id,
        lines_changed: signals.lines_changed,
        public_api_broken: signals.public_api_broken,
        gaming_suspected: signals.gaming_suspected,
        build_result: gate_outcome_from_signal(gates.build),
        test_result: gate_outcome_from_signal(gates.test),
        clippy_result: gate_outcome_from_signal(gates.clippy),
        bench_measurements_pct,
        bench_median_pct,
    };
    Ok(Report::from_decision(inputs, &decision))
}

/// `repo_root` の `policy-exclusion.toml` を読み込む。存在しない場合は
/// 組み込み既定値（[`policy_exclusion::builtin_defaults`]）にフォールバック
/// する（`config.rs::resolve` の `guardrail.toml` 探索順序と同じ設計）。
fn load_policy_exclusion_config(
    repo_root: &Path,
) -> Result<policy_exclusion::PolicyExclusionConfig, GuardrailError> {
    let candidate = repo_root.join("policy-exclusion.toml");
    if candidate.is_file() {
        let raw = std::fs::read_to_string(&candidate).map_err(|source| GuardrailError::Io {
            path: candidate.clone(),
            source,
        })?;
        policy_exclusion::load_from_str(&raw)
    } else {
        policy_exclusion::builtin_defaults().map_err(|e| {
            GuardrailError::InvalidInput(format!(
                "policy-exclusion.toml 組み込み既定値の構築に失敗: {e}"
            ))
        })
    }
}

/// measured 経路（本番相当計測。`docs/guardrail-self-repair-cli.md` 1.2 節）。
/// `main.rs` からはこの薄いラッパー（`SystemCommandRunner` 固定）のみを呼ぶ。
pub fn run_measured(
    repo_root: &Path,
    baseline: &str,
    config: &Config,
    change_id: Option<String>,
) -> Result<Report, GuardrailError> {
    run_measured_with(&SystemCommandRunner, repo_root, baseline, config, change_id)
}

/// [`run_measured`] の本体。`runner` を注入可能にすることで、
/// `cargo build`/`test`/`clippy` を実起動せずに measured 経路の
/// オーケストレーション（実行順序・`decide()` への到達・`Report` 合成）を
/// 単体テストで固定できる（下記 `#[cfg(test)]` 参照。TASK-4.1c・イシュー #106）。
///
/// 実行順序: `baseline` 実在確認 → 除外リスト評価（ゲート成否に関わらず
/// 必ず実行。モジュール冒頭「除外リスト評価の実行契約」参照）→ 変更行数・
/// 公開 API 破壊・ゲーミング疑いの実測 → build/test/clippy ゲート実行 →
/// `decide()`。
pub(crate) fn run_measured_with(
    runner: &dyn CommandRunner,
    repo_root: &Path,
    baseline: &str,
    config: &Config,
    change_id: Option<String>,
) -> Result<Report, GuardrailError> {
    changeset::resolve_baseline_commit(repo_root, baseline)?;

    let exclusion_config = load_policy_exclusion_config(repo_root)?;
    let eval_ctx = EvaluationContext::from_repo(repo_root, baseline)?;
    let evaluation = ExclusionEvaluation::evaluate(&exclusion_config.rules, &eval_ctx)?;
    let exclusion_rule_ids = evaluation.effective_rule_ids();

    let lines_changed = checks::diff_lines::lines_changed(repo_root, baseline)?;
    let api_broken = checks::api_stability::api_broken(repo_root, baseline)?;
    let gaming_suspect = gaming::gaming_suspected(repo_root, baseline)?;

    let gates = gates::run_gates(runner, repo_root)?;

    // D1（計画 3 節）: bench 実計測は本イシューのスコープ外。`decide()` は
    // `NotRun` を逸脱なしとして扱うため（`decision.rs` テスト参照）、
    // auto_apply への到達可能性は損なわれない。
    let bench = BenchSignal::NotRun;

    let input = DecisionInput::new(
        &config.thresholds,
        lines_changed,
        gates,
        api_broken,
        gaming_suspect,
        bench,
        exclusion_rule_ids,
    )?;
    let decision = decide(&input)?;

    let inputs = ReportInputs {
        signal_source: SignalSource::Measured,
        change_id,
        lines_changed,
        public_api_broken: api_broken,
        gaming_suspected: gaming_suspect,
        build_result: gate_outcome_from_signal(gates.build),
        test_result: gate_outcome_from_signal(gates.test),
        clippy_result: gate_outcome_from_signal(gates.clippy),
        bench_measurements_pct: Vec::new(),
        bench_median_pct: 0.0,
    };
    Ok(Report::from_decision(inputs, &decision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::tests_support::ScriptedRunner;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    fn run_git(cwd: &Path, args: &[&str]) {
        let mut cmd = Command::new("git");
        cmd.args(args).current_dir(cwd);
        // 祖先プロセス（lefthook の pre-push フック等）から継承された
        // `GIT_DIR`／`GIT_WORK_TREE` 等を除去する。除去しないと `current_dir`
        // 指定を無視して呼び出し元プロセスのリポジトリ（本 worktree の
        // `.git`）に対して動作してしまい、フィクスチャ用一時リポジトリの
        // 隔離が壊れる（`exclusion_match::git_command` と同一方針。#337
        // レビュー対応時に発覚した並行 push 環境下でのテストフレーク）。
        for (key, _) in std::env::vars() {
            if key.starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
        let output = cmd
            .output()
            .unwrap_or_else(|e| panic!("git {args:?} 起動に失敗: {e}"));
        assert!(
            output.status.success(),
            "git {args:?} が失敗: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn commit_all(cwd: &Path, message: &str) {
        run_git(cwd, &["add", "-A"]);
        run_git(
            cwd,
            &[
                "-c",
                "user.email=guardrail-test@example.invalid",
                "-c",
                "user.name=guardrail-test",
                "commit",
                "-q",
                "-m",
                message,
            ],
        );
    }

    fn init_repo(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("guardrail-check-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        run_git(&dir, &["init", "-q"]);
        dir
    }

    fn default_config() -> Config {
        Config {
            thresholds: crate::config::Thresholds::builtin(crate::config::PresetName::Default),
        }
    }

    /// measured 経路の reject 分岐: `cargo build` 失敗（`ScriptedRunner`
    /// モック）→ ゲート実行結果によらず `decide()` の判定順序 1.（却下最優先）
    /// が確定することを、`cargo` を実起動せず固定する。
    #[test]
    fn run_measured_with_failing_build_yields_reject() {
        let dir = init_repo("reject");
        fs::write(dir.join("README.md"), "baseline\n").unwrap();
        commit_all(&dir, "baseline");
        fs::write(dir.join("README.md"), "updated\n").unwrap();

        let runner = ScriptedRunner::new(vec![false]);
        let config = default_config();
        let report = run_measured_with(&runner, &dir, "HEAD", &config, None).expect("判定に失敗");

        assert_eq!(report.verdict, crate::decision::Verdict::Reject);
        assert_eq!(report.signal_source, SignalSource::Measured);
        assert!(
            report
                .reason_conditions
                .iter()
                .any(|c| c == "gate_build_failed")
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// measured 経路の escalate 分岐: 全ゲート pass だが変更行数が閾値
    /// （既定 200 行）を超過 → `decide()` の機械エスカレーション条件に到達
    /// することを固定する。
    #[test]
    fn run_measured_with_large_diff_yields_escalate() {
        let dir = init_repo("escalate");
        fs::write(dir.join("a.txt"), "baseline\n").unwrap();
        commit_all(&dir, "baseline");
        // 変更行数を確実に閾値（200 行）超にするため、300 行の新規内容へ
        // 置き換える（追加 300 行の diff）。
        let big_content: String = (0..300).map(|i| format!("line-{i}\n")).collect();
        fs::write(dir.join("a.txt"), big_content).unwrap();

        let runner = ScriptedRunner::new(vec![true, true, true]);
        let config = default_config();
        let report = run_measured_with(&runner, &dir, "HEAD", &config, None).expect("判定に失敗");

        assert_eq!(report.verdict, crate::decision::Verdict::Escalate);
        assert!(
            report
                .reason_conditions
                .iter()
                .any(|c| c == "lines_max_exceeded")
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// measured 経路の escalate 分岐（除外リスト match 経由）: `Cargo.toml`
    /// 変更（`dependency-change` ルール。組み込み既定値）が
    /// `decide()` の判定順序 2.（無条件エスカレーション）を発火させ、
    /// `reason_conditions` が `policy_exclusion_match` のみ（機械判定条件が
    /// 一切参照されず短絡する契約。`decision.rs` 判定順序契約参照）に
    /// なることを固定する。除外リスト評価がゲート成否によらず必ず実行され
    /// `DecisionInput` へ実際に到達していることの証跡でもある。
    #[test]
    fn run_measured_with_dependency_change_yields_escalate_via_exclusion_match() {
        let dir = init_repo("exclusion");
        fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        commit_all(&dir, "baseline");
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"x\"\nversion = \"0.2.0\"\n",
        )
        .unwrap();

        let runner = ScriptedRunner::new(vec![true, true, true]);
        let config = default_config();
        let report = run_measured_with(&runner, &dir, "HEAD", &config, None).expect("判定に失敗");

        assert_eq!(report.verdict, crate::decision::Verdict::Escalate);
        assert_eq!(report.reason_conditions, vec!["policy_exclusion_match"]);
        assert_eq!(
            report.applied_exclusion_rule_ids,
            vec!["dependency-change".to_string()]
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// measured 経路の auto_apply 分岐: 全ゲート pass・閾値内・除外リスト
    /// 非該当のコメントのみの小変更 → `decide()` が逸脱なしと判定する
    /// ことを固定する。
    #[test]
    fn run_measured_with_clean_small_change_yields_auto_apply() {
        let dir = init_repo("auto-apply");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");
        fs::write(
            dir.join("src/lib.rs"),
            "// コメント追加\npub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();

        let runner = ScriptedRunner::new(vec![true, true, true]);
        let config = default_config();
        let report = run_measured_with(&runner, &dir, "HEAD", &config, None).expect("判定に失敗");

        assert_eq!(report.verdict, crate::decision::Verdict::AutoApply);
        assert!(report.reason_conditions.is_empty());
        assert!(report.applied_exclusion_rule_ids.is_empty());

        fs::remove_dir_all(&dir).ok();
    }
}
