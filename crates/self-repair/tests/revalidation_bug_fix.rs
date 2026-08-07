//! TASK-3.3b（イシュー #141・REQ-3）: バグ修正種別のループ完走実証。
//!
//! `docs/self-repair-revalidation-plan.md`（#140・人間承認済み）4.1 節の推奨題材
//! 「`Var::relu`（`crates/autodiff/src/var.rs`）の実装本体を sigmoid 相当の演算
//! グラフにすり替える」を、リポジトリ外の一時 sandbox（`git clone --local`）へ
//! 注入し、`self_repair::SelfRepairLoop`（検出 → 修正試行 1〈誤り・却下〉→
//! 修正試行 2〈正解・採用〉→ `guardrail::decide` 経由の取り込み判断）が人間介在
//! なしで完走することを実証する統合テストである。
//!
//! # 4 ゲート合成についての設計上の制約（重要）
//! `crate::verify_gates::CargoVerificationGate`（build/test/clippy の 3 ゲート）と
//! `crate::verify_bench::SelfRepairBenchGate`（ベンチゲート）の合成（4 ゲート化）は
//! `crates/self-repair/src/` 本体へまだ結線されていない（#136 系のスコープ。
//! `lib.rs` モジュールコメント参照）。本テストは `tests/` 配下の統合テストクレート
//! （`self_repair` を外部クレートとして利用する）であり、
//! `self_repair::outcome::VerifiedEvidence::new` は `pub(crate)` のため
//! ここから新しい `VerifiedEvidence`（bench シグナルを差し替えたもの）を構築でき
//! ない（A08 の型レベル境界。`outcome.rs` のドキュメント参照）。
//!
//! このため [`RevalidationVerificationGate`]（本ファイル内の
//! `self_repair::stages::VerificationGate` 実装）は次の 2 段構成を取る:
//! 1. `CargoVerificationGate::verify` で build/test/clippy 3 ゲートを実行し、
//!    その戻り値（`bench` フィールドは常に `NotRun`。`verify_gates.rs` の
//!    ドキュメントが「全ゲート通過 + NotRun」を矛盾としないことを明記）を
//!    そのまま `guardrail::decide` へ渡す唯一の経路として使う。
//! 2. 3 ゲート通過後、**別途**ベンチゲート機構（`SelfRepairBenchGate`）を合成
//!    ワークロード（下記）で実行し、機構自体が `bench_runs_min` 回以上・中央値
//!    判定で完走し、劣化率が `bench_median_max_pct` 以内であることを fail-closed
//!    に確認する（閾値超過時は `VerificationOutcome::Failed` として次の試行へ
//!    回す）。**この計測はベンチゲート機構の完走確認であり、候補 diff（bug fix
//!    のワークツリー差分）そのものの性能劣化率実測ではない**（sandbox の
//!    候補ファイルはリポジトリ外の別プロセス空間にあり、本テストプロセスへ
//!    動的リンクして直接呼び出す経路がないため）。`loop-report.json`／README
//!    ではこの区別を明示し、「候補 diff に対する劣化率実測」は #136 系
//!    （4 ゲート合成の `src/` 本体への昇格）の範囲として out-of-scope に記録する
//!    （`.claude/rules/out-of-scope-tracking.md`）。
//!
//! # ガードレール閾値の扱い
//! `bench_median_max_pct`／`bench_runs_min`／`lines_max` はいずれも本テストで
//! 再定義しない。sandbox にクローンされたリポジトリ直下の `guardrail.toml`
//! （TASK-4.3c・#117 確定値）を `guardrail::config::resolve` で読み込み、その値を
//! そのまま使う（`.claude/rules/security.md`「ガードレール閾値の変更はユーザー
//! 承認必須」）。
//!
//! # diff 由来シグナルの実測
//! `lines_changed`／`api_broken`／`gaming_suspect`／`exclusion_rule_ids` は
//! fail-open な既定値で埋めず、各試行の直前に sandbox の実際の git diff から
//! 実測する（`verify_gates.rs` の契約。詳細は本ファイル内のヘルパー関数を参照）。
//!
//! # 実行時間・分離方針
//! sandbox は本ワークスペース全体ではなく `crates/autodiff` 1 クレートのみを
//! 検証対象とする（`cargo build`/`test --release`/`clippy` をワークスペース
//! メンバーディレクトリ内で実行すると、そのメンバーのみがデフォルトビルド対象と
//! なる cargo の既定動作を利用する）。実 workspace 全体を対象にした完走実証は
//! TASK-3.3 系の別スコープであり本テストでは行わない
//! （`verify_gates_integration.rs` の先例コメントと同じ整理）。
//! それでも cold cache では相応に時間がかかるため通常 CI ジョブでは実行しない
//! （`#[ignore]`。`.claude/rules/coding-rust.md`「実機依存テストは #[ignore] で
//! 分離」と同じ運用をコンパイル時間の観点でも適用する）。実機（CUDA・Metal）
//! 依存はなく CPU バックエンドのみで完走する。

use std::cell::RefCell;
use std::env;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use self_repair::stages::{Proposal, VerificationGate, VerificationOutcome};
use self_repair::verify_bench::{MIN_BENCH_ITERATIONS, SelfRepairBenchGate};
use self_repair::{
    BugFixDetector, BugFixFixGenerator, CandidateFix, CargoVerificationGate,
    GuardrailAdoptionJudge, RepairKind, SelfRepairError, SelfRepairLoop, SystemCommandRunner,
};

/// `crates/autodiff/src/var.rs` の workspace 相対パス（sandbox 内でも同一）。
const VAR_RS_RELATIVE: &str = "crates/autodiff/src/var.rs";

/// ベンチゲート機構の試行ごとの計測ログ（attempt 番号・劣化率系列・中央値）。
/// `RevalidationVerificationGate` と呼び出し元テスト本体が `Rc` で共有する
/// （型の可読性のため `clippy::type_complexity` を構造的に回避する。
/// `.claude/rules/coding-rust.md`「`#[allow]` の安易な追加で黙らせない」）。
type BenchLog = Rc<RefCell<Vec<(u32, Vec<f64>, f64)>>>;

/// 試行ごとに実測したポリシー除外リスト match（attempt 番号・match したルール
/// id 列）の共有ログ。用途・共有理由は [`BenchLog`] と同じ。
type ExclusionRuleIdsLog = Rc<RefCell<Vec<(u32, Vec<String>)>>>;

/// リポジトリルート（このテストバイナリをビルドした `self-repair` の
/// `Cargo.toml` の 2 階層上）。`cargo test` が設定する CWD に依存せず常に
/// 同じ場所を指すよう、実行時の `current_dir()` ではなくビルド時定数
/// （`CARGO_MANIFEST_DIR`）から導出する。
fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root should be resolvable from CARGO_MANIFEST_DIR")
}

/// `git` を `cwd` で起動し、非 0 終了時は panic する（テストセットアップ専用の
/// ヘルパー。本番経路〈`crates/self-repair/src`〉の
/// `.claude/rules/coding-rust.md`「unwrap/expect 禁止」はここには適用されない
/// ——既存の統合テスト（`verify_gates_integration.rs`）と同じ運用）。
fn git(cwd: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗しました: {error}"));
    if !output.status.success() {
        panic!(
            "git {args:?} が失敗しました（exit={:?}）: stdout={} stderr={}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// sandbox（`git clone --local`）を構築する。メイン working copy・共有 git
/// 状態には一切触れない（並列イシュー実行時のグローバル状態保護。プロンプト
/// 手順の隔離契約）。以降の git 操作はすべて sandbox 内に閉じる。
fn create_sandbox() -> PathBuf {
    let sandbox = env::temp_dir().join(format!(
        "self-repair-revalidation-bug-fix-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&sandbox);
    git(
        Path::new("."),
        &[
            "clone",
            "--local",
            // `/tmp`（sandbox の作成先）とリポジトリが別ファイルシステム
            // （別マウント）上にありうる環境（例: worktree がバインドマウント
            // 上にある CI/コンテナ環境）では、既定のハードリンク複製が
            // `Invalid cross-device link` で失敗する。`--no-hardlinks` で
            // 常にファイルコピーへフォールバックし、実行環境に依存せず
            // sandbox を構築できるようにする。
            "--no-hardlinks",
            "--quiet",
            repo_root().to_str().expect("repo root should be UTF-8"),
            sandbox.to_str().expect("sandbox path should be UTF-8"),
        ],
    );
    sandbox
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// `var.rs` を行単位で読み込み、`relu` メソッド本体の forward 呼び出し行を
/// 探す。行の内容は呼び出し元が `forward_call_needle` で指定する
/// （バグ注入前は `let value = eval::relu(&self.value());`、注入後は
/// `let value = eval::sigmoid(&self.value());` と、状態によって内容が異なる
/// ため）。文字列一括置換（`str::replace`）ではなく「`pub fn relu` 宣言の
/// 直後」という行番号ベースで探索範囲を絞るのは、同一テキスト
/// （`eval::sigmoid(&self.value())`）が `sigmoid` メソッド本体にも存在し、
/// 素朴な全文検索では `relu` メソッド以外の行を誤って書き換えうるため。
fn find_relu_forward_line(lines: &[String], forward_call_needle: &str) -> usize {
    let relu_fn_idx = lines
        .iter()
        .position(|line| line.contains("pub fn relu(&self)"))
        .expect("Var::relu の宣言行が見つかりません（var.rs の構造が変わった可能性）");
    lines[relu_fn_idx..]
        .iter()
        .position(|line| line.contains(forward_call_needle))
        .map(|offset| relu_fn_idx + offset)
        .unwrap_or_else(|| {
            panic!(
                "Var::relu 本体に想定の forward 呼び出し行が見つかりません \
                 （探索対象: {forward_call_needle:?}）"
            )
        })
}

/// バグ注入: `relu` メソッド本体の forward 呼び出しを `eval::relu` → `eval::sigmoid`
/// へすり替える（`Op::Relu` の登録自体は変更しない。実装計画 4.1 節の推奨題材）。
/// これにより forward 値は sigmoid 相当になる一方、backward は `Op::Relu` の
/// 勾配式（`x > 0` の指示関数）のまま計算されるため、既知正解値テスト
/// （`crates/autodiff/tests/tape_recording.rs` 等）が forward 段階で失敗する。
fn inject_bug(original: &str) -> String {
    let mut lines: Vec<String> = original.lines().map(str::to_string).collect();
    let idx = find_relu_forward_line(&lines, "let value = eval::relu(&self.value());");
    assert!(
        lines[idx].contains("eval::relu"),
        "対象行が想定と異なります: {}",
        lines[idx]
    );
    lines[idx] = lines[idx].replace("eval::relu(&self.value())", "eval::sigmoid(&self.value())");
    let mut joined = lines.join("\n");
    joined.push('\n');
    joined
}

/// 誤った修正候補（attempt 1）: 注入されたバグ（`eval::sigmoid`）を別の誤り
/// （`eval::tanh`）に置き換えるだけで、依然として `Op::Relu` の勾配式と forward
/// 値が一致しない。既知正解値テストは失敗し続けるため、検証ゲートで却下される
/// （PoC-2「修正試行 1: 誤った修正・検証不合格で却下」の写像）。
fn wrong_fix_content(injected: &str) -> String {
    let mut lines: Vec<String> = injected.lines().map(str::to_string).collect();
    let idx = find_relu_forward_line(&lines, "let value = eval::sigmoid(&self.value());");
    assert!(
        lines[idx].contains("eval::sigmoid"),
        "対象行が想定と異なります（バグ注入後の内容ではない可能性）: {}",
        lines[idx]
    );
    lines[idx] = lines[idx].replace("eval::sigmoid(&self.value())", "eval::tanh(&self.value())");
    let mut joined = lines.join("\n");
    joined.push('\n');
    joined
}

/// sandbox の作業木へファイルを書き込み、コミットする（バグ注入コミット専用。
/// 以降の候補適用は `BugFixFixGenerator` が作業木へ直接書き込むのみでコミット
/// しない——コミットするのは diff の起点〈baseline〉を確定するこの 1 回のみ）。
fn write_and_commit(sandbox: &Path, relative: &str, content: &str, message: &str) -> String {
    fs::write(sandbox.join(relative), content)
        .unwrap_or_else(|error| panic!("{relative} の書き込みに失敗しました: {error}"));
    git(sandbox, &["add", "--", relative]);
    git(
        sandbox,
        &[
            "-c",
            "user.email=self-repair-revalidation@example.invalid",
            "-c",
            "user.name=self-repair-revalidation",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
    );
    git(sandbox, &["rev-parse", "HEAD"]).trim().to_string()
}

/// `baseline`（バグ注入コミットの sha）と現在の作業木との diff から
/// `lines_changed`（追加行数＋削除行数の合計）・変更ファイル一覧を実測する。
/// `git diff --numstat` は「追加\t削除\tパス」の TSV を返す（`Cargo.lock` 等の
/// 除外はしない——本題材は `var.rs` 単一ファイルのみを変更する想定であり、
/// 除外ロジックの要否自体が本題材のスコープ外）。
fn diff_numstat(sandbox: &Path, baseline: &str) -> (u64, Vec<String>) {
    let stdout = git(sandbox, &["diff", "--numstat", baseline, "--"]);
    let mut lines_changed: u64 = 0;
    let mut files = Vec::new();
    for line in stdout.lines() {
        let mut cols = line.splitn(3, '\t');
        let added = cols.next().unwrap_or("0");
        let deleted = cols.next().unwrap_or("0");
        let path = cols.next().unwrap_or("").to_string();
        lines_changed += added.parse::<u64>().unwrap_or(0);
        lines_changed += deleted.parse::<u64>().unwrap_or(0);
        if !path.is_empty() {
            files.push(path);
        }
    }
    (lines_changed, files)
}

/// 公開関数シグネチャ（`pub fn` を含む行）が diff の追加・削除行に現れないかを
/// 検査する（本題材は関数本体〈1 行〉のみの変更であり、シグネチャ行自体は
/// 変更されない想定）。`true` = 破壊的変更の疑いあり。
fn api_signature_touched(sandbox: &Path, baseline: &str) -> bool {
    let stdout = git(
        sandbox,
        &["diff", "--no-color", "-U0", baseline, "--", "*.rs"],
    );
    stdout.lines().any(|line| {
        let content = line
            .strip_prefix('+')
            .or_else(|| line.strip_prefix('-'))
            .unwrap_or("");
        // 差分ヘッダ行（`+++`/`---`）を誤検出しないよう、`+`/`-` 直後が
        // さらに `+`/`-` の場合は除外する。
        !content.starts_with(['+', '-']) && content.trim_start().starts_with("pub fn")
    })
}

/// 変更ファイル一覧に「本番コード」と「テストコード」の双方が同時に含まれるかを
/// 検査する（ゲーミング疑いの簡易ヒューリスティック。本題材は `var.rs`
/// 単一ファイルのみのため `false` になる想定。実装計画 3 節 6 項）。
fn gaming_suspect(changed_files: &[String]) -> bool {
    let touches_test = changed_files
        .iter()
        .any(|path| path.contains("/tests/") || path.ends_with("_test.rs"));
    let touches_prod = changed_files.iter().any(|path| {
        !path.contains("/tests/") && !path.ends_with("_test.rs") && path.ends_with(".rs")
    });
    touches_test && touches_prod
}

/// `guardrail` のポリシー除外リスト評価（REQ-5）を sandbox 上で実行する。
/// sandbox 直下の `policy-exclusion.toml`（TASK-5.1・確定値。本テストでは
/// 一切変更しない）をロードし、`baseline` と現作業木の diff に対して評価する。
fn evaluate_exclusion_rules(sandbox: &Path, baseline: &str) -> Vec<String> {
    let toml = fs::read_to_string(sandbox.join("policy-exclusion.toml"))
        .expect("policy-exclusion.toml の読み込みに失敗しました");
    let config = guardrail::load_policy_exclusion(&toml)
        .expect("policy-exclusion.toml のパースに失敗しました");
    let ctx = guardrail::EvaluationContext::from_repo(sandbox, baseline)
        .expect("EvaluationContext::from_repo の構築に失敗しました");
    let evaluation = guardrail::ExclusionEvaluation::evaluate(&config.rules, &ctx)
        .expect("ポリシー除外リストの評価に失敗しました");
    evaluation.effective_rule_ids()
}

/// ベンチゲート機構の合成ワークロード（relu forward 相当の要素毎演算）。
/// 候補 diff の実測ではない（本ファイル冒頭ドキュメント参照）ため、
/// baseline・candidate で同一の計算を行う軽量クロージャとする
/// （`bench_harness::run` の warmup/計測反復に対して短時間で完走させるため）。
fn synthetic_relu_workload() {
    let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.01).collect();
    let mut sum = 0.0f32;
    for value in &data {
        sum += value.max(0.0);
    }
    std::hint::black_box(sum);
}

/// build/test/clippy 3 ゲート（`CargoVerificationGate`）＋ベンチゲート機構
/// 完走確認（`SelfRepairBenchGate`）を合成する `VerificationGate` 実装
/// （本ファイル冒頭ドキュメント「4 ゲート合成についての設計上の制約」参照）。
struct RevalidationVerificationGate {
    /// 検証対象ワークスペース（sandbox の `crates/autodiff`）。
    workspace: PathBuf,
    /// sandbox リポジトリのルート（diff・ポリシー除外評価はここを基点にする）。
    sandbox_root: PathBuf,
    /// diff の起点（バグ注入コミットの sha）。
    baseline_commit: String,
    bench_gate: SelfRepairBenchGate,
    thresholds: guardrail::Thresholds,
    bench_iterations: usize,
    /// ベンチゲート機構の計測結果を試行ごとに記録する（`loop-report.json` へ
    /// 「機構完走確認」として書き出すための seam。`RefCell` は `verify` が
    /// `&self` を取る `VerificationGate` trait のシグネチャ制約による。`Rc` は
    /// `RevalidationVerificationGate` が `SelfRepairLoop::new` に所有権ごと
    /// 渡された後も、呼び出し元がログ内容を読み出せるよう共有ハンドルとして
    /// 保持するため（`Rc::clone` で複製した片方をループ実行前に呼び出し元へ
    /// 残しておく）。
    bench_log: BenchLog,
    /// 試行ごとに実測したポリシー除外リスト match（`Rc<RefCell<..>>` の理由は
    /// `bench_log` と同じ）。
    exclusion_rule_ids_log: ExclusionRuleIdsLog,
}

impl VerificationGate for RevalidationVerificationGate {
    fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError> {
        // diff 由来シグナルは試行ごとに実測する（候補適用直後の作業木を対象。
        // fail-open な既定値で埋めない。`verify_gates.rs` の契約）。
        let (lines_changed, changed_files) =
            diff_numstat(&self.sandbox_root, &self.baseline_commit);
        let api_broken = api_signature_touched(&self.sandbox_root, &self.baseline_commit);
        let suspect = gaming_suspect(&changed_files);
        let exclusion_rule_ids =
            evaluate_exclusion_rules(&self.sandbox_root, &self.baseline_commit);
        self.exclusion_rule_ids_log
            .borrow_mut()
            .push((proposal.attempt, exclusion_rule_ids.clone()));

        let cargo_gate = CargoVerificationGate::new(
            self.workspace.clone(),
            SystemCommandRunner::new(),
            lines_changed,
            api_broken,
            suspect,
            exclusion_rule_ids,
        );

        let outcome = cargo_gate.verify(proposal)?;
        let evidence = match outcome {
            VerificationOutcome::Failed { reason } => {
                return Ok(VerificationOutcome::Failed { reason });
            }
            VerificationOutcome::Passed(evidence) => evidence,
        };

        // 3 ゲート通過後にのみベンチゲート機構を完走させる（PoC-2「検証落ちは
        // 取り込み判断へ進めず再試行」と同じく、既に不合格が確定した試行で
        // 追加の計測コストを払わない）。
        let mut baseline_workload = synthetic_relu_workload;
        let mut candidate_workload = synthetic_relu_workload;
        let bench_signal = self
            .bench_gate
            .run(
                self.bench_iterations,
                &mut baseline_workload,
                &mut candidate_workload,
            )
            .map_err(|error| SelfRepairError::Verification {
                attempt: proposal.attempt,
                reason: format!("ベンチゲート機構の完走に失敗しました: {error}"),
            })?;

        self.bench_log.borrow_mut().push((
            proposal.attempt,
            bench_signal.bench_measurements_pct.clone(),
            bench_signal.bench_median_pct,
        ));

        if bench_signal.bench_median_pct > self.thresholds.bench_median_max_pct {
            return Ok(VerificationOutcome::Failed {
                reason: format!(
                    "ベンチゲート機構（合成ワークロード）の劣化率中央値が閾値を超過しました: \
                     {:.4}% > {:.4}%",
                    bench_signal.bench_median_pct, self.thresholds.bench_median_max_pct
                ),
            });
        }

        Ok(VerificationOutcome::Passed(evidence))
    }
}

/// 完走ログの出力先ディレクトリを決める。環境変数
/// `SELF_REPAIR_REVALIDATION_OUT`（プロンプト手順・実装計画 3 節ステップ 3 が
/// 指定する起動方法）が設定されていればそれを使い、未設定時はリポジトリ直下の
/// `docs/self-repair-revalidation/bug-fix` へフォールバックする（`cargo test`
/// の CWD に依存しないよう `repo_root()` から導出する）。
fn output_dir() -> PathBuf {
    match env::var("SELF_REPAIR_REVALIDATION_OUT") {
        Ok(value) => {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                path
            } else {
                // `cargo test` はテストバイナリの CWD をパッケージ
                // （`crates/self-repair`）のマニフェストディレクトリに設定する
                // （workspace ルートではない）。プロンプト手順・実装計画 3 節が
                // 想定する「リポジトリルートからの相対パス」を実行時 CWD に
                // 依存せず解決するため、`repo_root()` を基点として結合する。
                repo_root().join(path)
            }
        }
        Err(_) => repo_root().join("docs/self-repair-revalidation/bug-fix"),
    }
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis() as u64
}

/// 完走ログ（`LoopReport`）を JSON へ変換する。
///
/// `self_repair::LoopReport`／`AttemptRecord`／`LoopOutcome`／`AttemptOutcome` は
/// いずれも `Serialize` を実装していない（`report.rs`・`outcome.rs` のドキュメント
/// 参照。構造化ログ出力そのものは TASK-3.4・#145 のスコープ）ため、本関数が
/// フィールドを手動で `serde_json::Value` へ写す。試行回数・所要時間・判断根拠
/// （#141 の受け入れ条件）に加え、ベンチゲート機構の完走ログ（合成ワークロード
/// である旨のラベル付き。本ファイル冒頭ドキュメント参照）も含める。
#[allow(clippy::too_many_arguments)]
fn build_report_json(
    report: &self_repair::LoopReport,
    bench_log: &[(u32, Vec<f64>, f64)],
    thresholds: &guardrail::Thresholds,
    exclusion_rule_ids_by_attempt: &[(u32, Vec<String>)],
    sandbox_commit: &str,
    started_at_unix_ms: u128,
) -> serde_json::Value {
    let outcome_json = match &report.outcome {
        self_repair::LoopOutcome::NoActionNeeded => serde_json::json!({"kind": "no_action_needed"}),
        self_repair::LoopOutcome::Adopted => serde_json::json!({"kind": "adopted"}),
        self_repair::LoopOutcome::Escalated { reason } => {
            serde_json::json!({"kind": "escalated", "reason": reason})
        }
        self_repair::LoopOutcome::Rejected { stage, reason } => {
            serde_json::json!({"kind": "rejected", "stage": stage, "reason": reason})
        }
        self_repair::LoopOutcome::Exhausted => serde_json::json!({"kind": "exhausted"}),
    };

    let attempts_json: Vec<serde_json::Value> = report
        .attempts
        .iter()
        .map(|attempt| {
            let (stage, reason): (&str, Option<&str>) = match &attempt.outcome {
                self_repair::report::AttemptOutcome::VerificationFailed { reason } => {
                    ("verification_failed", Some(reason.as_str()))
                }
                self_repair::report::AttemptOutcome::AdoptionRejectedRetryable { reason } => {
                    ("adoption_rejected_retryable", Some(reason.as_str()))
                }
                self_repair::report::AttemptOutcome::Adopted => ("adopted", None),
                self_repair::report::AttemptOutcome::Escalated { reason } => {
                    ("escalated", Some(reason.as_str()))
                }
                self_repair::report::AttemptOutcome::RejectedFinal { reason } => {
                    ("rejected_final", Some(reason.as_str()))
                }
            };
            let bench = bench_log
                .iter()
                .find(|(attempt_no, _, _)| *attempt_no == attempt.attempt)
                .map(|(_, measurements, median)| {
                    serde_json::json!({
                        "measurements_pct": measurements,
                        "median_pct": median,
                        "note": "ベンチゲート機構の完走確認（合成ワークロード）。候補 diff の性能劣化率実測ではない。",
                    })
                });
            let exclusion_rule_ids = exclusion_rule_ids_by_attempt
                .iter()
                .find(|(attempt_no, _)| *attempt_no == attempt.attempt)
                .map(|(_, ids)| ids.clone())
                .unwrap_or_default();
            serde_json::json!({
                "attempt": attempt.attempt,
                "duration_ms": duration_ms(attempt.duration),
                "stage_reached": stage,
                "reason": reason,
                "exclusion_rule_ids": exclusion_rule_ids,
                "bench_gate_mechanism": bench,
            })
        })
        .collect();

    serde_json::json!({
        "task": "TASK-3.3b",
        "issue": 141,
        "kind": report.kind.as_machine_id(),
        "outcome": outcome_json,
        "attempt_count": report.attempt_count(),
        "attempts": attempts_json,
        "total_duration_ms": duration_ms(report.total_duration),
        "thresholds": {
            "lines_max": thresholds.lines_max,
            "bench_median_max_pct": thresholds.bench_median_max_pct,
            "bench_runs_min": thresholds.bench_runs_min,
        },
        "sandbox_bug_injection_commit": sandbox_commit,
        "started_at_unix_ms": started_at_unix_ms,
        "scope_notes": [
            "self-repair run/verify-log CLI バイナリ未実装のため、lib 直接呼び出し（SelfRepairLoop::run）経由での完走実証である（CLI 経由での再実施は #145 マージ後の後続イシューのスコープ）。",
            "JSON Lines ハッシュチェーンログ・verify-log 検証は TASK-3.4（#145）のスコープ。",
            "ベンチゲート値は機構の完走確認（合成ワークロード）であり、候補 diff に対する劣化率実測ではない（4 ゲート合成の src/ 本体への昇格は #136 系のスコープ）。",
        ],
    })
}

/// TASK-3.3b（#141）の受け入れ条件本体: バグ修正種別のループが人間介在なしで
/// 完走し、完走ログ（試行回数・所要時間・判断根拠）が記録されることを確認する。
#[test]
#[ignore = "実 workspace メンバー（autodiff）に対する cargo build/test --release/clippy を \
            複数試行分・繰り返し実行するため長時間かかる。通常 CI ジョブでは実行しない \
            （.claude/rules/coding-rust.md の実機依存テスト分離と同じ運用をコンパイル \
            時間の観点で適用）。実行: cargo test -p self-repair --test revalidation_bug_fix \
            -- --ignored --nocapture"]
fn bug_fix_loop_completes_without_human_intervention() {
    let started_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_millis();
    let loop_start = Instant::now();

    let sandbox = create_sandbox();

    // バグ注入 → sandbox 内コミット（diff の起点。メイン working copy には
    // 一切触れない）。
    let var_rs_path = sandbox.join(VAR_RS_RELATIVE);
    let original_content =
        fs::read_to_string(&var_rs_path).expect("var.rs の読み込みに失敗しました（注入前）");
    let injected_content = inject_bug(&original_content);
    let injected_commit = write_and_commit(
        &sandbox,
        VAR_RS_RELATIVE,
        &injected_content,
        "test: inject relu->sigmoid activation mismatch (revalidation harness, not for merge)",
    );

    let autodiff_dir = sandbox.join("crates/autodiff");

    // 検出段階: 既知正解値テストの失敗を検出する。
    let detector = BugFixDetector::new(autodiff_dir.clone(), SystemCommandRunner::new());

    // 修正生成段階: attempt 1（誤り・却下される想定）→ attempt 2（正解・採用）。
    let candidates = vec![
        CandidateFix {
            description: "誤った修正: eval::sigmoid を eval::tanh に置換（依然として \
                           Op::Relu の勾配式と forward 値が一致しない誤り）"
                .to_string(),
            files: vec![(
                PathBuf::from("src/var.rs"),
                wrong_fix_content(&injected_content),
            )],
        },
        CandidateFix {
            description: "正しい修正: relu 実装（eval::relu）を復元".to_string(),
            files: vec![(PathBuf::from("src/var.rs"), original_content.clone())],
        },
    ];
    let fix_generator = BugFixFixGenerator::new(autodiff_dir.clone(), candidates)
        .expect("BugFixFixGenerator の構築に失敗しました");

    // 検証段階の閾値: sandbox 直下の guardrail.toml（TASK-4.3c 確定値）を
    // そのまま読み込む（閾値の再定義・緩和はしない）。
    let thresholds = guardrail::config::resolve(None, &sandbox, guardrail::PresetName::Default)
        .expect("guardrail.toml の解決に失敗しました")
        .thresholds;
    let bench_iterations = MIN_BENCH_ITERATIONS.max(thresholds.bench_runs_min as usize);

    // `RevalidationVerificationGate` は所有権ごと `SelfRepairLoop::new` へ渡す
    // （`VerificationGate` trait 境界は値渡し）ため、ループ実行後にも試行ごとの
    // 計測ログを読み出せるよう `Rc::clone` した片方を呼び出し元に残しておく。
    let bench_log: BenchLog = Rc::new(RefCell::new(Vec::new()));
    let exclusion_rule_ids_log: ExclusionRuleIdsLog = Rc::new(RefCell::new(Vec::new()));

    let verification_gate = RevalidationVerificationGate {
        workspace: autodiff_dir.clone(),
        sandbox_root: sandbox.clone(),
        baseline_commit: injected_commit.clone(),
        bench_gate: SelfRepairBenchGate::new(),
        thresholds,
        bench_iterations,
        bench_log: Rc::clone(&bench_log),
        exclusion_rule_ids_log: Rc::clone(&exclusion_rule_ids_log),
    };

    // 取り込み判断段階: `guardrail::decide` を経由するアダプタ（迂回経路なし）。
    let adoption_judge = GuardrailAdoptionJudge::new(thresholds);

    let max_attempts = NonZeroU32::new(2).expect("2 は非ゼロ");
    let loop_runner = SelfRepairLoop::new(
        detector,
        fix_generator,
        verification_gate,
        adoption_judge,
        max_attempts,
    );

    let run_result = loop_runner.run(RepairKind::BugFix);

    let output_dir_path = output_dir();
    fs::create_dir_all(&output_dir_path).expect("完走ログ出力先ディレクトリの作成に失敗しました");
    let output_path = output_dir_path.join("loop-report.json");

    match &run_result {
        Ok(report) => {
            let json = build_report_json(
                report,
                &bench_log.borrow(),
                &thresholds,
                &exclusion_rule_ids_log.borrow(),
                &injected_commit,
                started_at_unix_ms,
            );
            fs::write(
                &output_path,
                serde_json::to_string_pretty(&json).expect("JSON シリアライズに失敗しました"),
            )
            .expect("loop-report.json の書き込みに失敗しました");
        }
        Err(failure) => {
            let json = serde_json::json!({
                "task": "TASK-3.3b",
                "issue": 141,
                "outcome": {"kind": "loop_failure"},
                "error": failure.error.to_string(),
                "attempt_count": failure.attempts.len(),
                "started_at_unix_ms": started_at_unix_ms,
            });
            fs::write(
                &output_path,
                serde_json::to_string_pretty(&json).expect("JSON シリアライズに失敗しました"),
            )
            .expect("loop-report.json（失敗時）の書き込みに失敗しました");
        }
    }

    let total_elapsed = loop_start.elapsed();
    cleanup(&sandbox);

    let report = run_result.unwrap_or_else(|failure| {
        panic!(
            "自己修復ループが段階実行自体のエラーで終了しました（人間介在なし完走の \
             受け入れ条件を満たさない）: {failure}"
        )
    });

    assert_eq!(
        report.outcome,
        self_repair::LoopOutcome::Adopted,
        "最終 verdict は AutoApply（LoopOutcome::Adopted）である必要があります: {:?}",
        report.outcome
    );
    assert_eq!(
        report.attempt_count(),
        2,
        "attempt 1（誤り・却下）→ attempt 2（正解・採用）の 2 試行で完走する想定です"
    );
    match &report.attempts[0].outcome {
        self_repair::report::AttemptOutcome::VerificationFailed { .. } => {}
        other => panic!("attempt 1 は検証不合格（VerificationFailed）である想定です: {other:?}"),
    }
    match &report.attempts[1].outcome {
        self_repair::report::AttemptOutcome::Adopted => {}
        other => panic!("attempt 2 は採用（Adopted）である想定です: {other:?}"),
    }
    assert!(
        total_elapsed > Duration::ZERO,
        "所要時間が記録されていること"
    );
    for attempt in &report.attempts {
        assert!(
            attempt.duration >= Duration::ZERO,
            "各試行の所要時間が記録されていること（attempt={}）",
            attempt.attempt
        );
    }
}
