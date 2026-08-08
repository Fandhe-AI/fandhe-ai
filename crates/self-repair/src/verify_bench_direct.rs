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
//! （[`pinned_sources_untouched`]。実装計画 §3.2）。
//!
//! [`pinned_sources_untouched`] は `workload_sources` に列挙されたベンチ
//! ターゲット実体だけでなく、`sandbox_root` 配下の全差分ファイルから
//! **マニフェスト（`Cargo.toml`／`Cargo.lock`）・ビルドスクリプト（`build.rs`）・
//! Cargo 設定（`.cargo/config.toml`／`.cargo/config`）に該当するものすべて**を
//! ピン留め対象へ含める（`[[bin]].path` の付け替え・`[profile.release]`
//! 改変・`build.rs` 経由の間接的なビルド条件操作でのゲーミング迂回を防ぐ。
//! Codex レビュー #355 P1 指摘対応）。上記に該当しない未知のビルド系ファイル
//! 種別を経由した迂回は既存の `gaming_suspect` ヒューリスティック・ポリシー
//! 除外評価・guardrail 3 分岐判定に委ねる残余リスクとする（out-of-scope-
//! tracking.md 準拠でスコープ外）。
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
    /// 変更していないか検査する）。マニフェスト・ビルドスクリプト・Cargo 設定は
    /// ここに列挙せずとも [`pinned_sources_untouched`] が自動でピン留め対象に
    /// 含める（モジュール冒頭「ゲーミング防止」参照）。
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

/// `path`（`sandbox_root` 相対）が `spec.workload_sources` に明示列挙されて
/// いるか、マニフェスト（`Cargo.toml`／`Cargo.lock`）・ビルドスクリプト
/// （`build.rs`）・Cargo 設定（`.cargo/config.toml`／`.cargo/config`）の
/// いずれかに該当するかを判定する（ピン留め対象の全体像。モジュール冒頭
/// 「ゲーミング防止」参照。Codex レビュー #355 P1 指摘対応）。ディレクトリ
/// 位置を問わず判定する（`crates/*/Cargo.toml` 等のワークスペース内配置も
/// 拾う）ことで、候補 diff が `[[bin]].path` の付け替えや `[profile.release]`
/// 改変・`build.rs` 経由の間接的なビルド条件操作でゲーミングを迂回する経路を
/// 塞ぐ。
fn is_pinned_path(path: &str, workload_sources: &[String]) -> bool {
    if workload_sources.iter().any(|source| source == path) {
        return true;
    }
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if matches!(file_name, "Cargo.toml" | "Cargo.lock" | "build.rs") {
        return true;
    }
    // `.cargo/config.toml`（新形式）・`.cargo/config`（旧形式。cargo は両対応）
    // のいずれかで終わるパスをピン留め対象とする。ワークスペースルート以外
    // （ネストした `.cargo/` 等）に配置される場合も拾えるよう suffix 一致で
    // 判定する。
    path.ends_with(".cargo/config.toml") || path.ends_with(".cargo/config")
}

/// `sandbox_root` の作業木で未追跡ファイルを含む全ファイルを index へ反映する
/// （`git add -A -- .`）。`crate::diff_signals::stage_untracked_files` と同一の
/// 理由・同一のコマンドだが、両モジュールを跨いだ `pub(crate)` 共有はしない
/// （`verify_bench_direct.rs` 冒頭「A03 対応」節・`validate_commit_ref` と同じ
/// 独立モジュール構成の判断）。`git diff <baseline_commit>` は**未追跡（新規
/// 追加）ファイルを出力に含めない**ため、候補 diff が新規 `build.rs`・
/// `.cargo/config.toml` を追加してピン留め検査を迂回するのを防ぐには、diff の
/// 前に index へ反映しておく必要がある（`diff_signals.rs` が解決したのと同じ
/// 問題クラス。Codex レビュー #355 P1 指摘対応。追跡済みファイルへの変更のみ
/// では新規追加ファイルを見逃す fail-open 経路になる）。
fn stage_untracked_files<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
) -> Result<(), DirectBenchError> {
    let output = runner
        .run("git", &["add", "-A", "--", "."], sandbox_root)
        .map_err(|error| {
            DirectBenchError::Setup(format!(
                "git add -A の起動に失敗（未追跡ファイル反映）: {error}"
            ))
        })?;
    if !output.success() {
        return Err(DirectBenchError::Setup(format!(
            "git add -A が失敗しました（未追跡ファイル反映）: {}",
            output.log_tail()
        )));
    }
    Ok(())
}

/// 候補 diff がベンチワークロードソース（`spec.workload_sources`）に加え、
/// マニフェスト・ビルドスクリプト・Cargo 設定（[`is_pinned_path`]）を
/// 変更していないか検査する（ゲーミング防止。モジュール冒頭ドキュメント参照）。
/// `spec.workload_sources` のみへ絞った `git diff -- <pathspec>` ではなく
/// `sandbox_root` 全体の差分を取得したうえで [`is_pinned_path`] によって
/// 事後フィルタする（候補 diff がどこにマニフェスト・ビルド設定を追加・変更しても
/// 見逃さないため。Codex レビュー #355 P1 指摘対応）。
fn pinned_sources_untouched<R: CommandRunner>(
    runner: &R,
    spec: &DirectBenchSpec,
) -> Result<(), DirectBenchError> {
    stage_untracked_files(runner, &spec.sandbox_root)?;
    let args = ["diff", "--name-only", spec.baseline_commit.as_str()];
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
    // （Codex レビュー #137 指摘の同種クラス。本検査は sandbox_root 全体の
    // 差分を取得するため `workload_sources` のみを対象にしていた旧実装より
    // 上限に達しやすく、この防御の重要性が増している）。
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
        .filter(|line| !line.is_empty())
        .filter(|line| is_pinned_path(line, &spec.workload_sources))
        .map(str::to_string)
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
        // ピン留め検査（未追跡ファイル反映の add・diff）のみ呼ばれ、
        // worktree/build には到達しない。
        assert_eq!(runner.calls.borrow().len(), 2);
    }

    /// Codex レビュー #355 P1 指摘の回帰テスト: `workload_sources` に明示
    /// 列挙していない `Cargo.toml`（`[[bin]].path` の付け替え・
    /// `[profile.release]` 改変等の温床）が候補 diff で改変された場合も
    /// ピン留め違反として拒否されることを確認する。
    #[test]
    fn measure_rejects_manifest_change_even_when_not_listed_in_workload_sources() {
        let runner = ScriptedRunner::new(vec![("diff", true, "Cargo.toml\n")]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("Cargo.toml 改変は拒否されるはず");
        match err {
            DirectBenchError::WorkloadPinningViolation { touched } => {
                assert_eq!(touched, vec!["Cargo.toml".to_string()]);
            }
            other => panic!("expected WorkloadPinningViolation, got {other:?}"),
        }
    }

    /// 同上（`build.rs`・ネストした `.cargo/config.toml` も同様に検出する）。
    #[test]
    fn measure_rejects_build_script_and_cargo_config_changes() {
        let runner = ScriptedRunner::new(vec![(
            "diff",
            true,
            "build.rs\n.cargo/config.toml\nunrelated/README.md\n",
        )]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("build.rs・.cargo/config.toml 改変は拒否されるはず");
        match err {
            DirectBenchError::WorkloadPinningViolation { touched } => {
                assert_eq!(
                    touched,
                    vec!["build.rs".to_string(), ".cargo/config.toml".to_string()]
                );
            }
            other => panic!("expected WorkloadPinningViolation, got {other:?}"),
        }
    }

    /// ピン留め対象に該当しない無関係ファイルのみの変更は許容される
    /// （回帰防止: フィルタが過剰検知しないことの確認）。
    #[test]
    fn measure_allows_unrelated_file_changes() {
        let runner = ScriptedRunner::new(vec![
            ("diff", true, "src/lib.rs\n"),
            ("worktree", true, ""),
            ("build", false, ""),
        ]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("build 失敗で止まるはず（ピン留めでは止まらない）");
        assert!(
            matches!(err, DirectBenchError::Setup(_)),
            "ピン留め違反ではなく build 失敗として扱われるはず: {err:?}"
        );
    }

    /// Codex レビュー #355 P1 指摘の回帰テスト: 候補 diff が `Cargo.toml`・
    /// `build.rs` 等を**新規追加**した場合も `git diff` の前に `git add -A`
    /// で index 反映してから検査するため見逃さないことを確認する（未追跡
    /// ファイルは `git diff <baseline_commit>` 単体では検出できない。
    /// `stage_untracked_files` ドキュメント参照）。
    #[test]
    fn measure_stages_untracked_files_before_pinning_diff() {
        let runner = ScriptedRunner::new(vec![("diff", true, "src/lib.rs\n")]);
        let _ = DirectBenchRunner.measure(&runner, &spec());
        let calls = runner.calls.borrow();
        assert_eq!(
            calls.first().map(String::as_str),
            Some("git add -A -- ."),
            "git diff の前に git add -A で未追跡ファイルを index 反映するはず"
        );
    }

    /// 未追跡ファイル反映（`git add -A`）自体が失敗した場合も fail-closed に
    /// 拒否する（`Setup` エラー。ピン留め違反の見逃しより安全側に倒す）。
    #[test]
    fn measure_fails_closed_when_staging_untracked_files_fails() {
        let runner = ScriptedRunner::new(vec![("add", false, "")]);
        let err = DirectBenchRunner
            .measure(&runner, &spec())
            .expect_err("git add -A 失敗は拒否されるはず");
        assert!(matches!(err, DirectBenchError::Setup(_)));
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
