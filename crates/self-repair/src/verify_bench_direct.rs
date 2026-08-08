//! ベンチゲートの候補 diff 直接実測（TASK-3.2a・イシュー #137）。
//!
//! [`crate::verify_bench::SelfRepairBenchGate`]（既存。判定ロジック・
//! `bench-harness` 呼び出しは一切変更しない）は baseline／candidate の
//! ワークロードクロージャを**呼び出し側が用意する**契約であり、それ自体は
//! 「何を計測するか」を持たない（`bench_gate.rs`「スコープ境界」参照）。
//! これまでの唯一の消費者（`crate::verify_composite::FeatureAdditionCompositeGate`）
//! は baseline・candidate 双方に**同一の合成ワークロード**を渡しており、
//! 候補実装固有の性能劣化を検出できない（#139 reopen コメント・完走判定基準 5）。
//!
//! 本モジュールは「候補 diff に対する直接実測」を実現する [`DirectBenchRunner`]
//! を提供する。設計方針（外部タイミング方式。実装計画 #137 §3.1）:
//! 1. `baseline_commit` を `git worktree add --detach` で隔離した作業木として
//!    実体化し、`cargo build --release --bin <bench_bin>` する
//! 2. 候補 diff 適用済みの `sandbox_root` 作業木をそのまま同じ bin で
//!    `cargo build --release` する
//! 3. 両バイナリを 1 回 exec するだけのクロージャを構成し、
//!    [`SelfRepairBenchGate::run`]（→ `HarnessBenchGate` → `bench_harness::run`。
//!    warmup 20+・計測 20+・`MIN_BENCH_ITERATIONS` 以上の反復・中央値判定）へ
//!    そのまま委譲する。**計測・判定ロジックは一切再実装しない**（受け入れ条件
//!    「ベンチゲートが bench-harness 経由で完走する」を既存経路で満たす）
//!
//! # ゲーミング防止（A08）
//! 計測は信頼側（本プロセス）が握り、候補コード（AI 生成 diff）が計測時間を
//! 自己申告する経路を作らない（sandbox 内バイナリの標準出力はパースしない。
//! 実装計画 §3.1 対案比較）。加えて、候補 diff がベンチワークロードのソース
//! （`workload_sources`）自体を改変して「軽くして速く見せる」ゲーミングを
//! [`DirectBenchRunner::measure`] の最初の実質ステップで fail-closed に拒否する
//! （[`pinned_sources_untouched`]。実装計画 §3.2）。ビルド設定（`.cargo/config.toml`
//! 等）経由の間接的ゲーミングは残余リスクであり、既存の `gaming_suspect`
//! ヒューリスティック・ポリシー除外評価・guardrail 3 分岐判定に委ねる
//! （本モジュールでは検出しない。out-of-scope-tracking.md 準拠でスコープ外）。
//!
//! # A03（インジェクション）対応
//! `crate::diff_signals::validate_commit_ref` と同じ 16 進 commit sha 検証を
//! `baseline_commit` に適用する。すべての外部プロセス起動は固定引数配列
//! （`CommandRunner::run`）でシェルを経由しない。

use std::cell::Cell;
use std::path::{Path, PathBuf};

use crate::exec::CommandRunner;
use crate::verify_bench::{BenchSignal, SelfRepairBenchGate, VerifyBenchError};

/// [`DirectBenchRunner::measure`] の実測時エラー。`crate::diff_signals::
/// DiffSignalsError` と同じ理由で `attempt` を持たない独立型とし、呼び出し元
/// （`crate::verify_direct_composite::RepairCompositeGate::verify`）で
/// `crate::error::SelfRepairError::Verification` へ変換する。
#[derive(Debug, Clone, PartialEq)]
pub enum DirectBenchError {
    /// `baseline_commit` の形式不正・ビルド失敗・バイナリ不在等の準備段階エラー。
    Setup(String),
    /// ベンチワークロードソース（[`DirectBenchSpec::workload_sources`]）が
    /// 候補 diff によって改変されている（ゲーミング疑い。モジュール冒頭
    /// 「ゲーミング防止」参照）。
    WorkloadPinningViolation { touched: Vec<String> },
    /// baseline／candidate バイナリの実行（exec）自体が失敗した。
    ExecutionFailed(String),
    /// [`SelfRepairBenchGate::run`] への委譲が失敗した（`bench-harness` 側の
    /// 計測・判定エラーをそのまま包む）。
    Gate(VerifyBenchError),
}

impl std::fmt::Display for DirectBenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectBenchError::Setup(msg) => write!(f, "候補 diff 直接実測の準備に失敗: {msg}"),
            DirectBenchError::WorkloadPinningViolation { touched } => write!(
                f,
                "ベンチワークロードソースが候補 diff によって改変されています（ピン留め違反）: {touched:?}"
            ),
            DirectBenchError::ExecutionFailed(msg) => {
                write!(f, "ベンチワークロードバイナリの実行に失敗: {msg}")
            }
            DirectBenchError::Gate(err) => write!(f, "ベンチゲート機構への委譲に失敗: {err}"),
        }
    }
}

impl std::error::Error for DirectBenchError {}

impl From<VerifyBenchError> for DirectBenchError {
    fn from(err: VerifyBenchError) -> Self {
        DirectBenchError::Gate(err)
    }
}

/// [`DirectBenchRunner::measure`] の設定 DTO。
#[derive(Debug, Clone)]
pub struct DirectBenchSpec {
    /// 候補 diff 適用済みの sandbox 作業木ルート（使い捨てリポジトリ。実リポジトリ
    /// には触れない）。
    pub sandbox_root: PathBuf,
    /// diff の起点（バグ修正／機能追加検出前のコミット sha）。baseline 実体化は
    /// この commit を `git worktree add --detach` する。
    pub baseline_commit: String,
    /// 計測対象ワークロードの `[[bin]]` 名（`cargo build --release --bin
    /// <bench_bin>` で使う）。
    pub bench_bin: String,
    /// ゲーミング防止のためピン留めするワークロードソースファイル
    /// （`sandbox_root` 相対パス。`git diff --name-only` で候補 diff がこれらを
    /// 変更していないか検査する）。
    pub workload_sources: Vec<String>,
    /// ベンチゲート機構（[`SelfRepairBenchGate::run`]）へ渡す反復回数
    /// （[`crate::verify_bench::MIN_BENCH_ITERATIONS`] 以上であることは
    /// `HarnessBenchGate::measure` 側が強制するため、本モジュールでは別途
    /// 検査しない）。
    pub bench_iterations: usize,
}

/// worktree 実体化・release ビルド・外部タイミング計測を組み合わせ、
/// [`SelfRepairBenchGate`] へ委譲する runner（モジュール冒頭ドキュメント参照）。
#[derive(Debug, Default, Clone, Copy)]
pub struct DirectBenchRunner;

/// `sandbox/target/self-repair-baseline-<pid>` 配下に確保する baseline worktree
/// の RAII ガード。`Drop` でベストエフォートに `git worktree remove --force` +
/// `git worktree prune` を試み、後始末漏れがあっても `target/`（`.gitignore`
/// 済み）配下のため tracked 汚染は生じない（実装計画 §9 リスク）。
struct WorktreeGuard<'a, R: CommandRunner> {
    runner: &'a R,
    sandbox_root: &'a Path,
    worktree_path: PathBuf,
}

impl<R: CommandRunner> Drop for WorktreeGuard<'_, R> {
    fn drop(&mut self) {
        let path_str = self.worktree_path.to_string_lossy().into_owned();
        let _ = self.runner.run(
            "git",
            &["worktree", "remove", "--force", &path_str],
            self.sandbox_root,
        );
        let _ = self
            .runner
            .run("git", &["worktree", "prune"], self.sandbox_root);
    }
}

/// `crate::diff_signals::validate_commit_ref` と同一の検証（16 進 commit sha。
/// コマンドラインオプション偽装の遮断。両モジュールとも `pub(crate)` 実装を
/// 共有せずそれぞれ独立に持つ理由: `diff_signals` はテスト移植元
/// （`revalidation_bug_fix.rs`）由来の独立モジュールとして構成し、`#141`/`#142`
/// との並行編集衝突を避けるため単一ファイルへの相互依存を増やさない設計判断
/// （実装計画 §3.3）。
fn validate_commit_ref(baseline_commit: &str) -> Result<(), DirectBenchError> {
    let is_valid = (7..=40).contains(&baseline_commit.len())
        && baseline_commit
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && (b.is_ascii_digit() || b.is_ascii_lowercase()));
    if is_valid {
        Ok(())
    } else {
        Err(DirectBenchError::Setup(format!(
            "baseline_commit は 7〜40 桁の小文字 16 進文字列である必要があります: {baseline_commit:?}"
        )))
    }
}

/// 候補 diff がベンチワークロードソース（`spec.workload_sources`）自体を
/// 変更していないか検査する（ゲーミング防止。モジュール冒頭ドキュメント参照）。
fn pinned_sources_untouched<R: CommandRunner>(
    runner: &R,
    spec: &DirectBenchSpec,
) -> Result<(), DirectBenchError> {
    let mut args = vec!["diff", "--name-only", spec.baseline_commit.as_str(), "--"];
    for source in &spec.workload_sources {
        args.push(source.as_str());
    }
    let output = runner
        .run("git", &args, &spec.sandbox_root)
        .map_err(|error| DirectBenchError::Setup(format!("ピン留め検査の起動に失敗: {error}")))?;
    if !output.success() {
        return Err(DirectBenchError::Setup(format!(
            "ピン留め検査（git diff --name-only）が失敗しました: {}",
            output.log_tail()
        )));
    }
    // `output.log_tail()` は 256 KiB 超過時に先頭側を切り詰める（`exec.rs`
    // 参照）。切り詰められたファイル一覧を「改変なし」と誤解析すると
    // ピン留め違反（ゲーミング）を見逃す fail-open 経路になるため、
    // `diff_signals::run_git` と同じ方針で fail-closed に拒否する
    // （Codex レビュー #137 指摘の同種クラス。`args` は `workload_sources`
    // のみを対象とするため通常は上限に達しないが、防御的に検査する）。
    if output.truncated() {
        return Err(DirectBenchError::Setup(
            "ピン留め検査（git diff --name-only）の出力が 256 KiB 上限で切り詰められました。\
             ピン留め違反の見逃しを避けるため fail-open な部分解析はせず拒否します（fail-closed）"
                .to_string(),
        ));
    }
    let touched: Vec<String> = output
        .log_tail()
        .lines()
        .map(str::to_string)
        .filter(|line| !line.is_empty())
        .collect();
    if touched.is_empty() {
        Ok(())
    } else {
        Err(DirectBenchError::WorkloadPinningViolation { touched })
    }
}

/// `spec.baseline_commit` を `sandbox/target/self-repair-baseline-<pid>` へ
/// `git worktree add --detach` で実体化する。
fn add_baseline_worktree<'a, R: CommandRunner>(
    runner: &'a R,
    spec: &'a DirectBenchSpec,
) -> Result<WorktreeGuard<'a, R>, DirectBenchError> {
    let worktree_path = spec
        .sandbox_root
        .join("target")
        .join(format!("self-repair-baseline-{}", std::process::id()));
    let worktree_str = worktree_path.to_string_lossy().into_owned();
    let output = runner
        .run(
            "git",
            &[
                "worktree",
                "add",
                "--detach",
                &worktree_str,
                spec.baseline_commit.as_str(),
            ],
            &spec.sandbox_root,
        )
        .map_err(|error| {
            DirectBenchError::Setup(format!("git worktree add の起動に失敗: {error}"))
        })?;
    if !output.success() {
        return Err(DirectBenchError::Setup(format!(
            "git worktree add が失敗しました: {}",
            output.log_tail()
        )));
    }
    Ok(WorktreeGuard {
        runner,
        sandbox_root: &spec.sandbox_root,
        worktree_path,
    })
}

/// `workspace_root` で `cargo build --release --bin <bench_bin>` を実行し、
/// 成果物パス（`target/release/<bench_bin>`）を返す。
fn build_release_bin<R: CommandRunner>(
    runner: &R,
    workspace_root: &Path,
    bench_bin: &str,
) -> Result<PathBuf, DirectBenchError> {
    let output = runner
        .run(
            "cargo",
            &["build", "--release", "--bin", bench_bin],
            workspace_root,
        )
        .map_err(|error| DirectBenchError::Setup(format!("cargo build の起動に失敗: {error}")))?;
    if !output.success() {
        return Err(DirectBenchError::Setup(format!(
            "cargo build --release --bin {bench_bin} が失敗しました: {}",
            output.log_tail()
        )));
    }
    let bin_path = workspace_root.join("target/release").join(bench_bin);
    if !bin_path.is_file() {
        return Err(DirectBenchError::Setup(format!(
            "ビルド成功後もバイナリが見つかりません: {}",
            bin_path.display()
        )));
    }
    Ok(bin_path)
}

/// バイナリを引数なしで 1 回 exec するクロージャを構成する。exec 失敗・非 0
/// 終了は `failed` フラグへ記録し、以降の呼び出しは即座に return する
/// （`SelfRepairBenchGate::run` のクロージャ境界は `&mut dyn FnMut()`（戻り値なし）
/// のため、失敗はここでは伝播できず、呼び出し元が実行後にフラグを検査して
/// `Err` へ変換する。`Cell` は単一スレッド内での使用〈`bench_harness::run` は
/// クロージャを呼び出し元スレッドで直接呼ぶ。`FnMut` 境界であり `Send` は不要〉
/// のため `RefCell` より軽量な `Cell<bool>` で十分）。
fn exec_once_closure(bin_path: PathBuf, failed: &Cell<bool>) -> impl FnMut() + '_ {
    move || {
        if failed.get() {
            return;
        }
        match std::process::Command::new(&bin_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(status) if status.success() => {}
            _ => failed.set(true),
        }
    }
}

impl DirectBenchRunner {
    /// 候補 diff を [`DirectBenchSpec`] の指示に従い直接実測する（モジュール
    /// 冒頭ドキュメントの処理順序参照）。
    ///
    /// # Errors
    ///
    /// `baseline_commit` の形式不正・ピン留め違反・worktree／ビルド失敗・
    /// バイナリ実行失敗・ベンチゲート機構の失敗のいずれかで [`DirectBenchError`]。
    pub fn measure<R: CommandRunner>(
        &self,
        runner: &R,
        spec: &DirectBenchSpec,
    ) -> Result<BenchSignal, DirectBenchError> {
        validate_commit_ref(&spec.baseline_commit)?;
        pinned_sources_untouched(runner, spec)?;

        let worktree_guard = add_baseline_worktree(runner, spec)?;
        let baseline_bin =
            build_release_bin(runner, &worktree_guard.worktree_path, &spec.bench_bin)?;
        let candidate_bin = build_release_bin(runner, &spec.sandbox_root, &spec.bench_bin)?;

        let baseline_failed = Cell::new(false);
        let candidate_failed = Cell::new(false);
        let mut baseline_closure = exec_once_closure(baseline_bin, &baseline_failed);
        let mut candidate_closure = exec_once_closure(candidate_bin, &candidate_failed);

        let bench_gate = SelfRepairBenchGate::new();
        let signal = bench_gate
            .run(
                spec.bench_iterations,
                &mut baseline_closure,
                &mut candidate_closure,
            )
            .map_err(DirectBenchError::from)?;

        // worktree はここまでの計測完了後に明示的に破棄する（`Drop` に委ねても
        // 安全だが、後続の呼び出し元処理より前にベストエフォート後始末を確実に
        // 走らせるため明示的にスコープを閉じる）。
        drop(worktree_guard);

        if baseline_failed.get() || candidate_failed.get() {
            return Err(DirectBenchError::ExecutionFailed(
                "baseline または candidate バイナリの実行が 1 回以上失敗しました（fail-closed）"
                    .to_string(),
            ));
        }

        Ok(signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{CommandOutput, ExecError};
    use std::cell::RefCell;

    struct ScriptedRunner {
        /// (先頭サブコマンド, 成功可否, stdout) の対応表。
        results: Vec<(&'static str, bool, &'static str)>,
        calls: RefCell<Vec<String>>,
    }

    impl ScriptedRunner {
        fn new(results: Vec<(&'static str, bool, &'static str)>) -> Self {
            ScriptedRunner {
                results,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl CommandRunner for ScriptedRunner {
        fn run(
            &self,
            program: &str,
            args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            self.calls
                .borrow_mut()
                .push(format!("{program} {}", args.join(" ")));
            let subcommand = args.first().copied().unwrap_or("");
            let (success, stdout) = self
                .results
                .iter()
                .find(|(name, _, _)| *name == subcommand)
                .map(|(_, success, stdout)| (*success, *stdout))
                .unwrap_or((true, ""));
            Ok(CommandOutput::from_captured(
                success,
                stdout.as_bytes().to_vec(),
            ))
        }
    }

    fn spec() -> DirectBenchSpec {
        DirectBenchSpec {
            sandbox_root: PathBuf::from("/sandbox"),
            baseline_commit: "abc1234".to_string(),
            bench_bin: "bench_workload".to_string(),
            workload_sources: vec!["src/bin/bench_workload.rs".to_string()],
            bench_iterations: 5,
        }
    }

    #[test]
    fn measure_rejects_invalid_baseline_commit_before_any_spawn() {
        let runner = ScriptedRunner::new(vec![]);
        let mut bad_spec = spec();
        bad_spec.baseline_commit = "--evil".to_string();
        let err = DirectBenchRunner
            .measure(&runner, &bad_spec)
            .expect_err("不正な commit は拒否されるはず");
        assert!(matches!(err, DirectBenchError::Setup(_)));
        assert!(runner.calls.borrow().is_empty(), "git を一切起動しないはず");
    }

    #[test]
    fn measure_rejects_workload_pinning_violation_before_worktree_or_build() {
        let runner = ScriptedRunner::new(vec![("diff", true, "src/bin/bench_workload.rs\n")]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("ワークロード改竄は拒否されるはず");
        match err {
            DirectBenchError::WorkloadPinningViolation { touched } => {
                assert_eq!(touched, vec!["src/bin/bench_workload.rs".to_string()]);
            }
            other => panic!("expected WorkloadPinningViolation, got {other:?}"),
        }
        // ピン留め検査（diff）のみ呼ばれ、worktree/build には到達しない。
        assert_eq!(runner.calls.borrow().len(), 1);
    }

    #[test]
    fn measure_fails_closed_when_baseline_worktree_add_fails() {
        let runner = ScriptedRunner::new(vec![("diff", true, ""), ("worktree", false, "")]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("worktree add 失敗は拒否されるはず");
        assert!(matches!(err, DirectBenchError::Setup(_)));
    }

    #[test]
    fn measure_fails_closed_when_release_build_fails() {
        let runner = ScriptedRunner::new(vec![
            ("diff", true, ""),
            ("worktree", true, ""),
            ("build", false, ""),
        ]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("build 失敗は拒否されるはず");
        assert!(matches!(err, DirectBenchError::Setup(_)));
    }
}
