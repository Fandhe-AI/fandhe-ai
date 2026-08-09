//! `self-repair` バイナリのエントリポイント（TASK-3.4 残作業・イシュー #145・
//! `run` サブコマンドはイシュー #142 差し戻し分・完走判定基準 1）。
//!
//! `docs/guardrail-self-repair-cli.md` 3.1 節「self-repair run」・3.2 節
//! 「self-repair verify-log」の CLI エントリポイント。`run` は検出 → 修正
//! 試行 → 検証 → 取り込み判断の 1 ループを [`self_repair::SelfRepairLoop`]
//! へそのまま委譲する薄い統合であり（3.4 節「guardrail を lib として直接
//! 呼び出す」設計をループ全体にも敷衍する）、`verify-log` は試行ログ
//! （JSON Lines・SHA-256 ハッシュチェーン。3.3 節）の整合性検証（改竄検知）を、
//! `cargo test` を経由せず監査担当者が直接実行できる手段として提供する
//! （`.claude/rules/security.md`「ループ試行ログは改竄検知可能な形式で記録し、
//! 取り込み判断の根拠を追跡可能にする」への対応）。
//!
//! `run` の段階実装（検出・修正生成・検証・取り込み判断）はいずれも
//! `self_repair` lib の既存実装（`BugFixDetector`／`FeatureAdditionDetector`・
//! `BugFixFixGenerator`／`FeatureAdditionFixGenerator`・`RepairCompositeGate`・
//! `GuardrailAdoptionJudge`）をそのまま呼ぶのみで、判定ロジックの二重実装・
//! 迂回経路は持たない（`.claude/rules/security.md` A08）。`verify_chain` も
//! 同様に単一実装のみを呼ぶ（[`report_verify_log_error`] 参照）。
//!
//! # `--kind perf-regression` は本イシューのスコープ外
//! `PerfRegressionDetector`/`PerfRegressionFixGenerator` は `BenchMeasurer`／
//! 戦略リストという異なる構築契約を持ち（`crate::perf_regression`）、
//! `CommandRunner` ベースの `BugFix`/`FeatureAddition` 系検出器と対称でない。
//! #142 の対象題材（機能追加）・#141 の対象題材（バグ修正）のいずれも
//! `perf-regression` を必要としないため CLI への結線を行っていない。
//! `RepairKind` 型自体は 3 variant を持つが、`--kind` の受理値は
//! `bug-fix`／`feature-addition` の 2 つのみとし、`perf-regression` は
//! `cli::parse_repair_kind` が usage エラー（exit 2）として拒否する
//! （values を受理してから実行時に内部エラーを返す従来実装は「3 種別を
//! 受理する」契約を満たさないという PR #361 codex-review P1 指摘への対応。
//! `cli.rs::parse_repair_kind` doc 参照）。下記 `RepairKind::PerfRegression`
//! 分岐は CLI 経由では到達しない防御的なケースとしてのみ残す。
//! `out-of-scope-tracking.md` 準拠でユーザーへの追跡起票要否を確認する事項
//! として記録する（実装計画 §2「スコープ判断」）。
//!
//! # 終了コード契約
//! `docs/guardrail-self-repair-cli.md` 3.5 節の `run` 3 分岐契約
//! （0/10/20/1）を [`exit_code_for_outcome`] のみが担う（fail-closed。
//! `.claude/rules/security.md` A08「判定の迂回経路を作らない」）。同節は
//! `LoopOutcome::Exhausted`／`NoActionNeeded` を明示しないため、本イシュー
//! 差し戻し分で「完走失敗を正常終了〈0〉と誤認させない」方針のもと `1`
//! （内部エラー区分）へ写像すると定め、`docs/guardrail-self-repair-cli.md`
//! 3.5 節へ追記した（実装計画 §3.1）。`verify-log` 固有の契約は
//! [`report_verify_log_error`] 側のドキュメント参照:
//!
//! | 値 | 意味（`run`） | 意味（`verify-log`） |
//! |---|---|---|
//! | `0` | 自動適用（`Verdict::AutoApply`）かつ `--repo` への反映も成功 | チェーン整合（改竄なし） |
//! | `10` | エスカレーション | （該当なし） |
//! | `20` | 却下 | （該当なし） |
//! | `1` | 内部エラー（`LoopFailure`・`Exhausted`・`NoActionNeeded`・段階構築失敗・sandbox 構築失敗・`--log` 書き込み失敗〈`Adopted` でも反映しない。下記「`--log` 一次記録契約」節参照〉／`--output` 書き込み失敗〈`--log` は成功済みのため反映は行う。同節参照〉）／自動適用後の `--repo` への反映失敗（下記参照） | 検証不合格・内部エラー |
//! | `2` | usage エラー（`--kind perf-regression` を含む。`cli.rs::parse_repair_kind` 参照） | usage エラー |
//!
//! `LoopOutcome` → 終了コードの基本写像（0/10/20/1）は [`exit_code_for_outcome`]
//! が単独で担うが、`run_run` はその後段で `LoopOutcome::Adopted`（exit 0）の
//! 場合のみ [`self_repair::sandbox::reflect_adopted_diff`] を呼び、隔離
//! sandbox（[`self_repair::sandbox::RunSandbox`]）内で検証済みの差分を
//! `--repo` の作業ツリーへ競合検査つきで反映する（PR #361 codex-review P0
//! 指摘対応: `--repo` に人間の作業リポジトリを直接渡すと、非採用に終わった
//! 候補の変更が未コミットの作業ツリーへ残置され `git add -A` が無関係な
//! 変更まで staged にしてしまう問題があった。`sandbox.rs` モジュール冒頭
//! ドキュメント参照）。この反映が失敗した場合（`--repo` がダーティで競合
//! する等）は、ループ自体は `Adopted` で完走していても `--repo` への反映が
//! できていないためプロセス全体としては完了しておらず、終了コードを `1` へ
//! 上書きする（`exit_code_for_outcome` が返す `0` を後段で書き換える唯一の
//! 経路であり、`LoopOutcome` の解釈自体には手を加えない。`--log`／`--output`
//! はループの真の結果〈Adopted〉を記録済みのまま変更しない）。
//!
//! # `--log` 一次記録契約（PR #361 codex-review P1・Medium 指摘対応）
//! `--repo` への反映は「監査ログ（`--log`）が採用結果を一次記録として
//! 残せている」ことのみを前提とする。`--output`（任意の複製レポート JSON）の
//! 書き込み成否は反映可否に影響しない（`--output` は 3.1 節で「未指定時は
//! 標準出力へ要約を出す」任意の副次出力であり、一次記録は常に `--log` が
//! 担う）。[`execute_loop`] は [`finish_with_report`] が返す `persisted`
//! フラグ（`--log` 書き込み〈自己検証込み〉が成功したか。`--output` の成否は
//! 含まない）を見て、`persisted == false` の場合は `LoopOutcome::Adopted`
//! であっても `run_run` へ `None` を返す（ログを残せていないまま
//! `reflect_adopted_diff` が実リポジトリへ差分反映してしまう経路を断つ）。
//! `--log` 書き込み失敗（エラー経路）では `--repo` に一切触れない。
//! `--output` 書き込み失敗は終了コードを非 0（`1`）にする（`docs/
//! guardrail-self-repair-cli.md` 3.5 節参照）が、`--log` が成功していれば
//! 反映は行う（PR #361 codex-review Medium 指摘: `--output` の失敗を `--log`
//! と同列に扱うと、監査ログには記録済みの正当な採用差分の反映まで過剰に
//! ブロックしてしまっていた）。

use std::cell::RefCell;
use std::ffi::OsString;
use std::num::NonZeroU32;
use std::path::Path;
use std::process::{Command as ProcessCommand, ExitCode};
use std::rc::Rc;

use self_repair::candidate::load_candidates_from_json;
use self_repair::cli::{self, Command, RunArgs, UsageError, VerifyLogArgs};
use self_repair::isolation::{ExecIsolation, NetworkIsolation, candidate_home_dirs};
use self_repair::outcome::VerifiedEvidence;
use self_repair::sandbox::{RunSandbox, reflect_adopted_diff};
use self_repair::stages::{Detector, FixGenerator};
use self_repair::verify_bench::BenchSignal as DirectBenchSignal;
use self_repair::{
    BugFixDetector, BugFixFixGenerator, CandidateFix, FeatureAdditionDetector,
    FeatureAdditionFixGenerator, GuardrailAdoptionJudge, LogError, LoopFailure, LoopOutcome,
    LoopReport, RepairCompositeGate, RepairCompositeGateSpec, RepairKind, SelfRepairError,
    SelfRepairLoop, SystemCommandRunner, VerifyChainSummary, verify_chain,
};

/// 候補実行用の隔離ディレクトリ（`isolation::candidate_home_dirs` が返す
/// `home`／`tmp` の共通の親。`sandbox_root` の**外側**〈兄弟ディレクトリ〉）
/// の後始末を保証する RAII guard（イシュー #414 レビュー対応）。
///
/// `candidate_home_dirs` は `sandbox_root`（`git add -A` で diff 計測される
/// git worktree。`RunSandbox::root()`）の内側に置くと、候補の `build.rs`／
/// テストが `$HOME`／`$TMPDIR` へ書き込んだファイルが diff シグナルを汚染
/// するため意図的に外側へ置く（`isolation.rs::candidate_home_dirs` doc
/// 参照）。その結果 `RunSandbox::Drop`（`sandbox_root` 配下のみ削除）では
/// 回収されなくなるため、`run_run` がこの guard を局所変数として保持し、
/// 早期 return を含む全経路で関数末尾のスコープ終了時に `Drop` させる。
struct IsolationDirsGuard {
    base: std::path::PathBuf,
}

impl IsolationDirsGuard {
    /// `candidate_home_dirs` が返す `home` パスの親ディレクトリ（`home`／
    /// `tmp` 双方の共通の親。`isolation.rs::candidate_home_dirs` 契約）を
    /// 後始末対象として保持する。
    fn new(candidate_home: &Path) -> Self {
        let base = candidate_home
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| candidate_home.to_path_buf());
        IsolationDirsGuard { base }
    }
}

impl Drop for IsolationDirsGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

/// [`RepairCompositeGate::evidence_sink`] の戻り値型（`execute_loop` が
/// ループ実行前に取得し、実行後の証跡観測に使う。型名で意図を明示するための
/// エイリアス）。
type EvidenceSink = Rc<RefCell<Option<VerifiedEvidence>>>;
/// [`RepairCompositeGate::bench_measurement_sink`] の戻り値型（[`EvidenceSink`]
/// と同じ理由のエイリアス）。
type BenchMeasurementSink = Rc<RefCell<Option<DirectBenchSignal>>>;

fn main() -> ExitCode {
    // `std::env::args()` ではなく `args_os()` を使う理由は `cli` モジュール
    // 冒頭ドキュメント参照（非 UTF-8 引数での panic を避けるため。PR #356
    // codex-review P1 指摘対応）。
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();

    match cli::parse(args) {
        Ok(Command::VerifyLog(verify_log_args)) => run_verify_log(verify_log_args),
        Ok(Command::Run(run_args)) => run_run(run_args),
        Err(err) => report_usage_error_and_exit(&err),
    }
}

/// `self-repair verify-log` の実行フロー。`verify_chain` の結果を stdout/stderr
/// への報告と終了コードへ変換するのみの薄い統合とする（`guardrail::main::run_check`
/// と同一方針。ロジックは lib 側 [`self_repair::verify_chain`] に一本化）。
fn run_verify_log(args: VerifyLogArgs) -> ExitCode {
    match verify_chain(&args.log) {
        // レコード 0 件は「チェーン検証エラーなし」であって「改竄されていない
        // ことの証明」ではない（末尾切り詰めで空になったログも同じ結果になる。
        // `verify_chain` モジュール冒頭ドキュメント参照）。ログ全削除による
        // 改竄と正当な空ログを区別できないため、既定では fail-closed に
        // exit 1 とする（PR #356 codex-review P1 指摘対応: 終了コードのみを
        // 見る監査自動化がこれまでの無条件 exit 0 を「検証成功」として
        // 見逃していた）。空ログを正当な運用として扱いたい場合のみ
        // `--allow-empty-log` を明示指定させ、その場合は `WARN:` 付きで
        // exit 0 のまま外部アンカー突合（docs/self-repair-log-format.md §7）を
        // 促す。
        Ok(summary) if summary.record_count == 0 && !args.allow_empty_log => {
            eprintln!(
                "self-repair verify-log: ログチェーンにレコードがありません（log={}, records=0）。空ログか末尾切り詰め（ログ全削除を含む改竄）かは本コマンド単体では区別できないため fail-closed で不合格とします。空ログを許容する場合は --allow-empty-log を指定してください",
                args.log.display()
            );
            ExitCode::from(1)
        }
        Ok(summary) if summary.record_count == 0 => {
            println!(
                "WARN: ログチェーンにレコードがありません（log={}, records=0）。--allow-empty-log が指定されているため exit 0 とします。空ログか末尾切り詰めかは本コマンド単体では区別できません。外部アンカー運用（docs/self-repair-log-format.md 7 節）との突合を確認してください",
                args.log.display()
            );
            ExitCode::from(0)
        }
        // `record_count > 0` の分岐でのみ到達する。`verify_chain` の実装上
        // `record_count > 0` は `last_seq.is_some()` と等価であり、この分岐に
        // 限れば `None` は生じない（`unwrap_or_default()` で握り潰すと万一の
        // 不整合時に `last_seq=0` を誤表示しうるため、`Some` を明示的に照合する）。
        Ok(VerifyChainSummary {
            record_count,
            last_seq: Some(last_seq),
            last_hash,
        }) => {
            // 監査担当者が外部アンカー（書き込み直後に別経路へ記録した最終
            // hash・seq）と突合できるよう、成功メッセージにレコード件数・
            // 最終 seq・最終 hash を含める（security.md A08 の意図。
            // `verify_chain` モジュール冒頭ドキュメント参照）。
            println!(
                "OK: ログチェーンの整合性を確認しました（log={}, records={}, last_seq={}, last_hash={}）",
                args.log.display(),
                record_count,
                last_seq,
                last_hash
            );
            ExitCode::from(0)
        }
        // record_count > 0 かつ last_seq == None は verify_chain の不変条件上
        // 到達しないはずだが、将来の実装変更でこの不変条件が崩れた場合に
        // 静かに誤った「OK」を出さないよう、fail-closed でエラー扱いにする
        // （security.md A08「判定の迂回経路を作らない」と同じ思想）。
        Ok(summary) => {
            eprintln!(
                "self-repair verify-log: 内部不整合を検知しました（log={}, records={}, last_seq=None）。verify_chain の実装を確認してください",
                args.log.display(),
                summary.record_count
            );
            ExitCode::from(1)
        }
        Err(err) => report_verify_log_error(&err),
    }
}

/// [`LogError`] を人間可読なメッセージとして stderr へ出力し、終了コード `1`
/// （検証不合格・内部エラーの両方を含む fail-closed 契約。モジュール冒頭
/// ドキュメント参照）へ変換する。`ChainViolation`（改竄・欠落検知）の内容は
/// `seq`・理由のみを含み、`LogError::Display`（`logging.rs`）実装同様
/// payload 本文は出力しない。
fn report_verify_log_error(err: &LogError) -> ExitCode {
    eprintln!("self-repair verify-log: {err}");
    ExitCode::from(1)
}

/// CLI 引数の usage エラーを stderr へ出力し、終了コード `2` へ変換する
/// （guardrail の usage エラー区分と整合。モジュール冒頭ドキュメント参照）。
fn report_usage_error_and_exit(err: &UsageError) -> ExitCode {
    eprintln!("self-repair: {err}");
    ExitCode::from(2)
}

/// `self-repair run` の実行フロー（3.1 節・実装計画 §3.1）。
///
/// 0. `--candidates` の候補コード実行に対する明示的な承認
///    （`--allow-candidate-exec`）は `cli::parse_run` が構築時に必須検証
///    しており（[`RunArgs::allow_candidate_exec`] doc・PR #361 codex-review
///    P0 指摘対応）、未指定の場合は本関数へ到達する前に usage エラー
///    （exit 2）で拒否される。以降のステップはすべてこの承認済み前提の
///    もとで候補コードを実行する（`docs/guardrail-self-repair-cli.md`
///    「候補実行の信頼境界」節参照）。
/// 1. `--repo` の HEAD を `baseline_commit` として解決する
/// 2. `--candidates` の JSON を [`CandidateFix`] へ変換する
/// 3. `--config` を [`guardrail::config::resolve`] で解決する（`guardrail.toml`
///    閾値・除外リストの緩和はここでは一切行わない。`.claude/rules/security.md`）
/// 4. `--kind` に応じ検出器・修正生成器の具体型を選ぶ（[`RepairCompositeGate`]・
///    [`GuardrailAdoptionJudge`] は種別非依存のため共有する）
/// 5. [`SelfRepairLoop::run`] を 1 回実行し、`--log`（常に）・`--output`
///    （任意）へ結果を書き出す
///
/// 各ステップの失敗は usage エラー（引数自体の問題）ではなく実行時エラー
/// として扱い、[`exit_code_for_outcome`] とは別に内部エラー区分の終了コード
/// `1` を返す（3.5 節）。
fn run_run(args: RunArgs) -> ExitCode {
    let baseline_commit = match resolve_baseline_commit(&args.repo) {
        Ok(sha) => sha,
        Err(message) => {
            eprintln!("self-repair run: {message}");
            return ExitCode::from(1);
        }
    };

    let candidates: Vec<CandidateFix> = match load_candidates_from_json(&args.candidates) {
        Ok(candidates) => candidates,
        Err(err) => {
            eprintln!("self-repair run: {err}");
            return ExitCode::from(1);
        }
    };

    let config = match guardrail::config::resolve(
        args.config.as_deref(),
        &args.repo,
        guardrail::PresetName::Default,
    ) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("self-repair run: guardrail.toml の解決に失敗しました: {err}");
            return ExitCode::from(1);
        }
    };

    // `--repo`（人間の作業リポジトリ）の作業ツリー・index には一切触れず、
    // ループ全体（候補適用・4 ゲート検証・`git add -A` を含む）を隔離
    // sandbox 内で完結させる（PR #361 codex-review P0 指摘対応。`sandbox.rs`
    // モジュール冒頭ドキュメント参照）。
    let mut sandbox = match RunSandbox::create(&args.repo, &baseline_commit) {
        Ok(sandbox) => sandbox,
        Err(message) => {
            eprintln!("self-repair run: 検証用 sandbox の構築に失敗しました: {message}");
            return ExitCode::from(1);
        }
    };
    let sandbox_root = sandbox.root().to_path_buf();

    // 候補実行の縦深防御（イシュー #414・A08）: 環境変数遮断・書き込み先制限
    // は候補実行経路（`gate_spec.runner`・`BugFixDetector`・
    // `FeatureAdditionDetector`）で既定有効とする。`--isolate-network` 指定時は
    // `unshare` の可用性を probe で事前確認し、失敗時は黙って劣化させず
    // 内部エラー区分（exit 1）で拒否する（`isolation` モジュール冒頭
    // ドキュメント・`docs/self-repair-candidate-isolation.md` 参照）。
    if args.isolate_network
        && let Err(message) = ExecIsolation::probe_unshare_net()
    {
        eprintln!(
            "self-repair run: --isolate-network が指定されましたが、この実行環境では \
             ネットワーク隔離を利用できません（fail-closed のためネットワーク隔離なしへは \
             劣化させず実行を中止します）: {message}"
        );
        return ExitCode::from(1);
    }
    // `candidate_home_dirs` は `sandbox_root`（`git add -A` で diff 計測される
    // git worktree）の**外側**（兄弟ディレクトリ）を返す契約（イシュー #414
    // レビュー対応。`isolation.rs::candidate_home_dirs` doc 参照）。そのため
    // `sandbox`（`RunSandbox`）の `Drop` では回収されず、本関数側で
    // `IsolationDirsGuard` により後始末する（早期 return を含む全経路で
    // 構築後は必ず `Drop` される。関数末尾の `Adopted` 反映失敗時に
    // `sandbox.keep()` で調査用に sandbox 本体を残す場合でも、隔離
    // ディレクトリ自体は候補実行専用の使い捨てであり調査対象ではないため
    // 引き続き削除する）。
    let (candidate_home, candidate_tmp) = candidate_home_dirs(&sandbox_root);
    let _isolation_dirs_guard = IsolationDirsGuard::new(&candidate_home);
    for dir in [&candidate_home, &candidate_tmp] {
        if let Err(error) = std::fs::create_dir_all(dir) {
            eprintln!(
                "self-repair run: 候補実行用の隔離ディレクトリ作成に失敗しました（{}）: {error}",
                dir.display()
            );
            return ExitCode::from(1);
        }
    }
    let mut exec_isolation = ExecIsolation::new(candidate_home, candidate_tmp);
    if args.isolate_network {
        exec_isolation = exec_isolation.with_network_isolation(NetworkIsolation::UnshareNet);
    }

    // diff・ベンチの計測系（`RepairCompositeGate`）は種別非依存であり、
    // `verify_direct_composite.rs` モジュール冒頭ドキュメントどおり build/
    // test/clippy の逐次実行・fail-fast 判定・ベンチ判定への変換を再実装
    // しない（このゲート 1 インスタンスを BugFix/FeatureAddition 双方の
    // 分岐で共有する）。`workspace`／`sandbox_root` はいずれも隔離 sandbox
    // を指す（`--repo` を直接渡さない。`sandbox.rs` 参照）。
    let policy_exclusion_path = args
        .policy_exclusion
        .clone()
        .unwrap_or_else(|| sandbox_root.join("policy-exclusion.toml"));
    // ポリシー除外設定は候補適用前（sandbox 構築直後・候補未適用の時点）に
    // 一度だけロードし、以降は不変値としてループ全体で使い回す
    // （`self_repair::diff_signals::load_policy_exclusion_config` doc「呼び出し
    // 契約」参照）。この時点ではまだ `SelfRepairLoop::run` を呼んでおらず
    // sandbox には候補が一切適用されていないため、既定パス（sandbox 内）の
    // 場合ここで読むのは `baseline_commit` 時点の内容（`RunSandbox::create`
    // が `git clone --local` で反映した直後の状態。`--repo` の作業ツリー上の
    // 未コミット編集は反映されない）であり、候補が sandbox 内の同ファイルを
    // 書き換えても以降の判定には反映されない（PR #361 codex-review P1
    // 指摘対応。`verify_direct_composite.rs` モジュール冒頭「ポリシー除外
    // 設定の固定」参照）。
    let policy_exclusion =
        match self_repair::diff_signals::load_policy_exclusion_config(&policy_exclusion_path) {
            Ok(config) => config,
            Err(err) => {
                eprintln!("self-repair run: policy-exclusion.toml の解決に失敗しました: {err}");
                return ExitCode::from(1);
            }
        };
    let gate_spec = RepairCompositeGateSpec {
        workspace: sandbox_root.clone(),
        sandbox_root: sandbox_root.clone(),
        baseline_commit: baseline_commit.clone(),
        policy_exclusion,
        bench_bin: args.bench_bin.clone(),
        workload_sources: args.workload_sources.clone(),
        bench_iterations: self_repair::verify_bench::MIN_BENCH_ITERATIONS,
        runner: SystemCommandRunner::isolated(exec_isolation.clone()),
    };
    let verification_gate = RepairCompositeGate::new(gate_spec);
    // `SelfRepairLoop::new` はゲートを値ごと（所有権を）受け取るため、ループ
    // 実行後もベンチ実測・4 シグナルを観測できるよう `Rc` 複製を先に取得して
    // おく（`--output` の `adopted_evidence` フィールド用。
    // `tests/feature_addition_loop_completion_task_3_3c.rs`（旧 lib 直接呼び出し
    // ハーネス）と同じ観測手段を CLI 本体側へ移した。`verify_direct_composite.rs`
    // の `evidence_sink`／`bench_measurement_sink` doc 参照）。
    let evidence_sink = verification_gate.evidence_sink();
    let bench_measurement_sink = verification_gate.bench_measurement_sink();
    let adoption_judge = GuardrailAdoptionJudge::new(config.thresholds);

    // `SelfRepairLoop<D, F, V, J>` は型パラメータのため、実行時に選んだ
    // `--kind` に応じた具体型（`D`/`F`）の選択は分岐ごとに行う
    // （`cli.rs` モジュール冒頭ドキュメント参照）。検出器・修正生成器の
    // workspace も sandbox を指す（`--repo` を直接渡さない）。
    let (exit_code, outcome) = match args.kind {
        RepairKind::BugFix => {
            let detector = BugFixDetector::new(
                sandbox_root.clone(),
                SystemCommandRunner::isolated(exec_isolation.clone()),
            );
            let fix_generator = match BugFixFixGenerator::new(sandbox_root.clone(), candidates) {
                Ok(generator) => generator,
                Err(err) => return report_run_setup_error(&err),
            };
            execute_loop(
                args.kind,
                detector,
                fix_generator,
                verification_gate,
                adoption_judge,
                args.max_attempts,
                &args.log,
                args.output.as_deref(),
                evidence_sink,
                bench_measurement_sink,
            )
        }
        RepairKind::FeatureAddition => {
            let detector = FeatureAdditionDetector::new(
                sandbox_root.clone(),
                SystemCommandRunner::isolated(exec_isolation.clone()),
            );
            let fix_generator =
                match FeatureAdditionFixGenerator::new(sandbox_root.clone(), candidates) {
                    Ok(generator) => generator,
                    Err(err) => return report_run_setup_error(&err),
                };
            execute_loop(
                args.kind,
                detector,
                fix_generator,
                verification_gate,
                adoption_judge,
                args.max_attempts,
                &args.log,
                args.output.as_deref(),
                evidence_sink,
                bench_measurement_sink,
            )
        }
        // `cli::parse_repair_kind` が `perf-regression` を usage エラー
        // （exit 2）として拒否するため、`args.kind` がこの分岐に到達する
        // ことは CLI 経由では起こらない（PR #361 codex-review P1 指摘
        // 対応。モジュール冒頭ドキュメント「`--kind perf-regression` は
        // 本イシューのスコープ外」参照）。`RepairKind` は 3 variant を持つ
        // 型のため `match` を網羅させる目的でのみ残す防御的なケースであり、
        // `unreachable!()`/`panic!()`（exit 101。coding-rust.md「本番経路で
        // `unwrap()`/`expect()` を使わない」と同種の禁則）にはせず、内部
        // エラー区分の exit 1 を返す。
        RepairKind::PerfRegression => {
            eprintln!(
                "self-repair run: --kind perf-regression は未対応です（cli::parse_repair_kind が usage エラーとして拒否するため本来到達しないはずの分岐です。#141/#142 のいずれも本種別を必要としないため未実装のまま。out-of-scope-tracking.md 準拠で追跡要否をユーザーへ確認する）"
            );
            (ExitCode::from(1), None)
        }
    };

    // `LoopOutcome::Adopted` の場合のみ、検証済み差分を `--repo` の作業
    // ツリーへ競合検査つきで反映する（`reflect_adopted_diff`。`sandbox.rs`
    // モジュール冒頭ドキュメント参照）。非採用・エラー経路では `--repo` に
    // 一切触れない（sandbox の自動削除〈`Drop`〉に任せる）。
    //
    // `outcome` は `execute_loop` が `--log`／`--output` の書き込み
    // （`finish_with_report` の `persisted` フラグ）に成功した場合のみ
    // `Some` を返す（失敗時は `Adopted` であっても `None` へ落とす）。
    // そのため本条件は実質「exit_code が Adopted の正常系（0）へ写像
    // され得た、かつ監査ログを一次記録として残せた」場合のみ真になり、
    // ログ書き込み失敗時に `--repo` へ反映してしまう経路を閉じる
    // （PR #361 codex-review P1 指摘対応）。
    if matches!(outcome, Some(LoopOutcome::Adopted)) {
        match reflect_adopted_diff(&args.repo, &sandbox_root, &baseline_commit) {
            Ok(()) => exit_code,
            Err(message) => {
                // 反映に失敗した sandbox は調査対象として残す（`Drop` の
                // 自動削除を抑止する）。ログ・`--output` は既にループの
                // 真の結果（Adopted）を記録済みだが、`--repo` への反映が
                // できなかった以上、プロセス全体としては完了していない
                // ため終了コードは内部エラー区分の 1 で上書きする
                // （`docs/guardrail-self-repair-cli.md` 3.5 節「採用差分の
                // 作業ツリー反映失敗」参照）。
                eprintln!(
                    "self-repair run: 採用された差分を --repo の作業ツリーへ反映できませんでした: {message}（調査用に sandbox を保持します: {}）",
                    sandbox_root.display()
                );
                sandbox.keep();
                ExitCode::from(1)
            }
        }
    } else {
        exit_code
    }
}

/// `--repo` の HEAD を `git rev-parse HEAD` で解決する（`RepairCompositeGateSpec::
/// baseline_commit` の入力）。
///
/// sandbox の使い捨て git リポジトリに対して呼ばれる場合（実証ハーネスの
/// 想定用途）も、`lefthook.yml` の `pre-push` フック経由で本バイナリが
/// 起動されるケースを含め、githooks(5) が子プロセスへ継承させる `GIT_*`
/// 環境変数（`GIT_DIR`／`GIT_WORK_TREE`／`GIT_INDEX_FILE` 等）が `current_dir`
/// より優先されてしまう。継承された `GIT_DIR` を除去せずに `--repo` の
/// sandbox で `git rev-parse` を実行すると、実リポジトリの HEAD を誤って
/// 参照しうる（`tests/feature_addition_loop_completion_task_3_3c.rs::
/// sandboxed_git_command` の doc に記録された 2026-08-07 実測の事故と同種の
/// リスク。advisor 指摘: 本番経路〈`main.rs`〉にも同じ隔離を適用する必要が
/// ある）。sandbox 用テストヘルパーと同じ方式で `GIT_*` を明示的に除去する。
fn resolve_baseline_commit(repo: &Path) -> Result<String, String> {
    let mut command = ProcessCommand::new("git");
    command.args(["rev-parse", "HEAD"]).current_dir(repo);
    for (key, _) in std::env::vars_os() {
        if let Some(key_str) = key.to_str()
            && key_str.starts_with("GIT_")
        {
            command.env_remove(key_str);
        }
    }
    let output = command.output().map_err(|error| {
        format!(
            "git rev-parse HEAD の起動に失敗しました（repo={}）: {error}",
            repo.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse HEAD が失敗しました（repo={}）: {}",
            repo.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// `BugFixFixGenerator::new`／`FeatureAdditionFixGenerator::new`（候補パス
/// 検証。`crate::candidate::validate_relative_path` 等）の構築時エラーを
/// 内部エラー区分（exit 1）へ変換する。usage エラー（exit 2）にしない理由:
/// エラーの原因は引数の構文ではなく `--candidates` の中身（外部入力）が
/// workspace の構造検証に違反したことであり、`UsageError` の意味（未知引数・
/// 値欠落）とは異なる。
fn report_run_setup_error(err: &SelfRepairError) -> ExitCode {
    eprintln!("self-repair run: 段階の構築に失敗しました: {err}");
    ExitCode::from(1)
}

/// [`SelfRepairLoop::new`]・[`SelfRepairLoop::run`] を実行し、結果を `--log`
/// （常に）・`--output`（任意）へ書き出したうえで終了コードへ変換する
/// （`run_run` の `RepairKind::BugFix`／`RepairKind::FeatureAddition` 分岐が
/// 共有する本体。`D`/`F` のみ型パラメータとし、`V`＝[`RepairCompositeGate`]・
/// `J`＝[`GuardrailAdoptionJudge`] は種別非依存のため固定する）。
///
/// 戻り値の第 2 要素（`Option<LoopOutcome>`）は `run_run` が
/// [`self_repair::sandbox::reflect_adopted_diff`] を呼ぶべきか（`Adopted` か
/// どうか）を判定するために使う。`None` になるのは次の 2 パターン:
/// 1. `LoopFailure` の場合（段階の実行自体が失敗しており採用判断に到達
///    していないため）
/// 2. `SelfRepairLoop::run` は正常終了（`Ok(LoopReport)`）したが
///    `finish_with_report` の `--log`／`--output` 書き込みが失敗した場合
///    （[`outcome_for_reflection`] が `persisted == false` を検知し
///    `Adopted` であっても `None` へ落とす。PR #361 codex-review P1
///    指摘対応。監査ログを一次記録として残せていない状態のまま
///    `--repo` へ反映してしまう経路を断つ）
#[allow(clippy::too_many_arguments)]
fn execute_loop<D, F>(
    kind: RepairKind,
    detector: D,
    fix_generator: F,
    verification_gate: RepairCompositeGate<SystemCommandRunner>,
    adoption_judge: GuardrailAdoptionJudge,
    max_attempts: NonZeroU32,
    log_path: &Path,
    output_path: Option<&Path>,
    evidence_sink: EvidenceSink,
    bench_measurement_sink: BenchMeasurementSink,
) -> (ExitCode, Option<LoopOutcome>)
where
    D: Detector,
    F: FixGenerator,
{
    let self_repair_loop = SelfRepairLoop::new(
        detector,
        fix_generator,
        verification_gate,
        adoption_judge,
        max_attempts,
    );

    match self_repair_loop.run(kind) {
        Ok(report) => {
            let outcome = report.outcome.clone();
            let (exit_code, persisted) = finish_with_report(
                &report,
                log_path,
                output_path,
                evidence_sink.borrow().clone(),
                bench_measurement_sink.borrow().clone(),
            );
            (exit_code, outcome_for_reflection(outcome, persisted))
        }
        Err(failure) => (finish_with_failure(&failure, log_path, output_path), None),
    }
}

/// `run_run` の `--repo` 反映判定（`matches!(outcome, Some(LoopOutcome::Adopted))`）
/// が参照する `outcome` を決める純粋関数（`execute_loop` から切り出し
/// 単体テスト可能にする）。
///
/// `persisted == false`（`--log` の書き込み失敗。`--output` の成否は
/// 含まない。[`finish_with_report`] の戻り値）の場合は `outcome` が
/// `LoopOutcome::Adopted` であっても `None` へ落とし `run_run` へ伝えない
/// （PR #361 codex-review P1 指摘対応）。`run_run` は
/// `Some(LoopOutcome::Adopted)` の場合にのみ `reflect_adopted_diff` で
/// `--repo` の作業ツリーへ差分反映するため、監査ログを一次記録として
/// 残せていない状態のまま実リポジトリへ反映してしまう経路をここで断つ
/// （「`--log` は一次記録として常に残す・エラー経路では `--repo` に触れない」
/// 契約。モジュール冒頭ドキュメント「`--log` 一次記録契約」節参照）。
/// `--output` の書き込み失敗単独では `persisted` は `false` にならない
/// （PR #361 codex-review Medium 指摘対応。`finish_with_report` doc 参照）。
fn outcome_for_reflection(outcome: LoopOutcome, persisted: bool) -> Option<LoopOutcome> {
    if persisted { Some(outcome) } else { None }
}

/// 正常終了（[`LoopReport`]）を `--log`（[`self_repair::LogWriter::append_report`]。
/// 追記後に [`verify_chain`] で自己検証する。実装計画 §3.1「ログ出力」）・
/// `--output`（任意。JSON 化）へ書き出し、[`exit_code_for_outcome`] で
/// 終了コードへ変換する。`evidence`／`bench_measurement` は `execute_loop` が
/// ループ実行前に取得した `Rc<RefCell<..>>`（`RepairCompositeGate::
/// evidence_sink`／`bench_measurement_sink`）から複製した値であり、
/// `verify` が最後に `Passed` を返した際の判断根拠を保持する
/// （`AttemptOutcome::Adopted` 自体は証跡を保持しないため。`report.rs` 参照）。
///
/// 戻り値の第 2 要素（`persisted`）は `--log`（一次記録・自己検証込み）の
/// 書き込みが成功したかのみを示す（`--output` の書き込み成否は含めない。
/// `execute_loop` が `--repo` への差分反映可否を判定する唯一の材料。
/// `ExitCode` は不透明型で値の比較ができないため〈`std::process::ExitCode`
/// は `PartialEq` を実装しない〉、「exit 0 相当か」を `ExitCode` から逆算
/// せずこの明示フラグで表す。PR #361 codex-review P1 指摘対応）。`--output`
/// の書き込みが失敗しても `persisted` は `true` のまま終了コードのみ `1` に
/// する（PR #361 codex-review Medium 指摘対応: `--output` は任意の複製
/// レポートに過ぎず、その書き込み失敗を理由に `--log` へ記録済みの正当な
/// 採用差分の反映まで抑止すべきではない。モジュール冒頭「`--log` 一次記録
/// 契約」節参照）。
fn finish_with_report(
    report: &LoopReport,
    log_path: &Path,
    output_path: Option<&Path>,
    evidence: Option<VerifiedEvidence>,
    bench_measurement: Option<DirectBenchSignal>,
) -> (ExitCode, bool) {
    if let Err(message) = append_report_and_verify(report, log_path) {
        eprintln!("self-repair run: ログ出力に失敗しました: {message}");
        return (ExitCode::from(1), false);
    }
    match output_path {
        // 3.1 節「未指定時は標準出力へテキスト要約を出す」契約
        // （`--output` は任意・`--log` は必須という非対称の埋め合わせ:
        // JSON Lines ログは常に残すが、その場で消費できれば足りる
        // `LoopReport` は指定時のみファイル化する）。
        None => {
            println!(
                "self-repair run: kind={} outcome={:?} attempts={}",
                report.kind.as_machine_id(),
                report.outcome,
                report.attempt_count()
            );
        }
        Some(output_path) => {
            if let Err(message) = write_report_json(
                report,
                output_path,
                evidence.as_ref(),
                bench_measurement.as_ref(),
            ) {
                eprintln!("self-repair run: --output 書き出しに失敗しました: {message}");
                // `--log`（一次記録）は既に書き込み・自己検証済みのため、
                // `--output`（任意の複製レポート）の書き込み失敗は反映
                // （`reflect_adopted_diff`）を抑止する理由にしない
                // （PR #361 codex-review Medium 指摘対応。モジュール冒頭
                // 「`--log` 一次記録契約」節・[`finish_with_report`] doc
                // 参照）。終了コードは非 0（1）のまま保つ。
                return (ExitCode::from(1), true);
            }
        }
    }
    (exit_code_for_outcome(&report.outcome), true)
}

/// 異常終了（[`LoopFailure`]。段階の実行自体が失敗）を `--log`
/// （[`self_repair::LogWriter::append_failure`]）・`--output`（任意）へ
/// 書き出し、内部エラー区分の exit 1 を返す（3.5 節「内部エラー」）。
fn finish_with_failure(
    failure: &LoopFailure,
    log_path: &Path,
    output_path: Option<&Path>,
) -> ExitCode {
    if let Err(message) = append_failure_and_verify(failure, log_path) {
        eprintln!("self-repair run: ログ出力に失敗しました: {message}");
        return ExitCode::from(1);
    }
    match output_path {
        None => eprintln!("self-repair run: {failure}"),
        Some(output_path) => {
            if let Err(message) = write_failure_json(failure, output_path) {
                eprintln!("self-repair run: --output 書き出しに失敗しました: {message}");
                return ExitCode::from(1);
            }
            eprintln!("self-repair run: {failure}");
        }
    }
    ExitCode::from(1)
}

/// `--log` を開いて [`LoopReport`] を追記し、書き込み直後に [`verify_chain`]
/// で自己検証する（実装計画 §3.1「ログ出力: … 書き込み後に verify_chain で
/// 自己検証する」）。
fn append_report_and_verify(report: &LoopReport, log_path: &Path) -> Result<(), LogError> {
    let mut writer = self_repair::LogWriter::open(log_path)?;
    writer.append_report(report)?;
    verify_chain(log_path)?;
    Ok(())
}

/// [`append_report_and_verify`] の [`LoopFailure`] 版（`append_failure` を使う）。
fn append_failure_and_verify(failure: &LoopFailure, log_path: &Path) -> Result<(), LogError> {
    let mut writer = self_repair::LogWriter::open(log_path)?;
    writer.append_failure(failure)?;
    verify_chain(log_path)?;
    Ok(())
}

/// [`self_repair::LoopOutcome`] を 3.5 節の終了コードへ変換する唯一の関数
/// （fail-closed。他の経路から `0` を返さない。`.claude/rules/security.md` A08）。
/// 全 variant を明示 match し `_ =>` を使わない（`outcome.rs`／`report.rs`
/// と同じ fail-closed 方針。新 variant 追加時にコンパイルエラーで検出する）。
fn exit_code_for_outcome(outcome: &self_repair::LoopOutcome) -> ExitCode {
    use self_repair::LoopOutcome;
    match outcome {
        LoopOutcome::Adopted => ExitCode::from(0),
        LoopOutcome::Escalated { .. } => ExitCode::from(10),
        LoopOutcome::Rejected { .. } => ExitCode::from(20),
        // 3.5 節は 3 分岐＋LoopFailure(1) のみを定義し `NoActionNeeded`
        // （検出段階で修正不要と判定・そもそもループ未開始）を含まない。
        // 「取り込まれなかった」点は却下と共通するが正常完走の主張はできない
        // ため、内部エラー区分（exit 1）へ写像する（本関数のみが行う写像。
        // 実装計画 §3.1）。
        LoopOutcome::NoActionNeeded => ExitCode::from(1),
        // `Exhausted`（試行上限到達）も 3.5 節が未定義のため、完走判定基準 1
        // 「1 回起動・追加の人間入力なしで exit 0」を満たせなかった場合を
        // 「エラーではない正常終了」と誤認させないよう、同じく exit 1 へ
        // 写像する（`docs/guardrail-self-repair-cli.md` 3.5 節へ追記済み）。
        LoopOutcome::Exhausted => ExitCode::from(1),
    }
}

/// `--output` 向けの [`LoopReport`] JSON 化（v1 `tools/self-repair/src/report.rs`
/// を踏襲。`LoopReport` 自体は `serde::Serialize` を実装しないため
/// `serde_json::json!` で手組みする。`tests/feature_addition_loop_completion_
/// task_3_3c.rs::write_loop_report` と同型の変換方針）。
fn write_report_json(
    report: &LoopReport,
    output_path: &Path,
    evidence: Option<&VerifiedEvidence>,
    bench_measurement: Option<&DirectBenchSignal>,
) -> Result<(), String> {
    let attempts_json: Vec<serde_json::Value> = report
        .attempts
        .iter()
        .map(|attempt| {
            serde_json::json!({
                "attempt": attempt.attempt,
                "duration_ms": attempt.duration.as_millis(),
                "outcome": format!("{:?}", attempt.outcome),
            })
        })
        .collect();

    // 完走判定基準 5・6（`docs/self-repair-revalidation-plan.md` §5）が求める
    // 「候補 diff 直接実測」「signal_source: measured」の証跡を、`execute_loop`
    // が取得した `evidence_sink`／`bench_measurement_sink` の複製から埋める
    // （`RepairCompositeGate::verify` は毎試行 diff を実測するため
    // `bench=measured-direct` を含む `gate_report` になる。`verify_direct_composite.rs`
    // 参照）。`evidence` は「最後に `Passed` を返した verify 呼び出し」の
    // スナップショットであり、`report.outcome == Adopted` の場合はその採用に
    // 至った試行の証跡と一致する。
    let adopted_evidence_json = evidence.map(|evidence| {
        let bench_median_pct = match evidence.bench() {
            guardrail::BenchSignal::Measured { median_pct } => Some(*median_pct),
            guardrail::BenchSignal::NotRun => None,
        };
        serde_json::json!({
            "attempt": evidence.attempt(),
            "gate_report": evidence.gate_report(),
            "bench_median_pct": bench_median_pct,
            "bench_measurements_pct": bench_measurement
                .map(|measurement| measurement.bench_measurements_pct.clone()),
            "lines_changed": evidence.lines_changed(),
            "api_broken": evidence.api_broken(),
            "gaming_suspect": evidence.gaming_suspect(),
            "exclusion_rule_ids": evidence.exclusion_rule_ids(),
        })
    });

    let doc = serde_json::json!({
        "kind": report.kind.as_machine_id(),
        "outcome": format!("{:?}", report.outcome),
        "attempt_count": report.attempt_count(),
        "attempts": attempts_json,
        "total_duration_ms": report.total_duration.as_millis(),
        "adopted_evidence": adopted_evidence_json,
        // `--signals` 契約検証パス（`guardrail check`。2.1 節）を経由しない
        // 実シグナル計測経路であることを明示する（同節の `signal_source`
        // フィールドと同じ語彙。REQ-6 の回帰テストセット根拠データとしての
        // 採用可否をこのフィールドで機械判定可能にする）。
        "signal_source": "measured",
    });
    write_json_with_trailing_newline(&doc, output_path)
}

/// [`write_report_json`] の [`LoopFailure`] 版。
fn write_failure_json(failure: &LoopFailure, output_path: &Path) -> Result<(), String> {
    let attempts_json: Vec<serde_json::Value> = failure
        .attempts
        .iter()
        .map(|attempt| {
            serde_json::json!({
                "attempt": attempt.attempt,
                "duration_ms": attempt.duration.as_millis(),
                "outcome": format!("{:?}", attempt.outcome),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "error": failure.error.to_string(),
        "attempts": attempts_json,
    });
    write_json_with_trailing_newline(&doc, output_path)
}

/// `.editorconfig` の `insert_final_newline` 慣行に合わせ、末尾改行付きで
/// pretty-print JSON を書き出す共通処理。
fn write_json_with_trailing_newline(doc: &serde_json::Value, path: &Path) -> Result<(), String> {
    let mut pretty = serde_json::to_string_pretty(doc)
        .map_err(|error| format!("JSON シリアライズに失敗しました: {error}"))?;
    pretty.push('\n');
    std::fs::write(path, pretty)
        .map_err(|error| format!("{} への書き込みに失敗しました: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`outcome_for_reflection`] が `persisted == true` の場合は `outcome`
    /// をそのまま透過することを確認する（`--log`／`--output` 書き込み成功時、
    /// `run_run` が `Adopted` を正しく観測できることの前提）。
    #[test]
    fn outcome_for_reflection_passes_through_when_persisted() {
        let result = outcome_for_reflection(LoopOutcome::Adopted, true);
        assert_eq!(result, Some(LoopOutcome::Adopted));
    }

    /// PR #361 codex-review P1 指摘の回帰防止: `persisted == false`
    /// （`--log`／`--output` 書き込み失敗）の場合、`outcome` が
    /// `LoopOutcome::Adopted` であっても `None` へ落とすことを確認する。
    /// `run_run` はこの `None` により `reflect_adopted_diff`
    /// （`--repo` の作業ツリーへの反映）を呼ばない
    /// （`matches!(outcome, Some(LoopOutcome::Adopted))` が偽になる）。
    #[test]
    fn outcome_for_reflection_suppresses_adopted_when_not_persisted() {
        let result = outcome_for_reflection(LoopOutcome::Adopted, false);
        assert_eq!(result, None);
    }

    /// 非 `Adopted`（`Escalated`）の場合は `persisted` に関わらず
    /// `run_run` の反映条件（`Some(LoopOutcome::Adopted)` 一致）を満たさない
    /// ため、`outcome_for_reflection` 自体は透過してよいことを確認する
    /// （反映抑止は `run_run` 側の `matches!` が担い、本関数は「ログ永続化
    /// 失敗を握り潰さない」ことのみを責務とする）。
    #[test]
    fn outcome_for_reflection_passes_through_non_adopted_when_persisted() {
        let result = outcome_for_reflection(
            LoopOutcome::Escalated {
                reason: "test".to_string(),
            },
            true,
        );
        assert_eq!(
            result,
            Some(LoopOutcome::Escalated {
                reason: "test".to_string()
            })
        );
    }

    /// [`finish_with_report`] は `--log` の書き込みが失敗した場合（ここでは
    /// 親ディレクトリが存在しないパスを渡して発生させる）、`outcome` が
    /// `Adopted` であっても `persisted == false` を返すことを確認する
    /// （`outcome_for_reflection` 単体テストの前提となる、実際のログ書き込み
    /// 経路との結合確認）。
    #[test]
    fn finish_with_report_reports_not_persisted_on_log_write_failure() {
        let report = LoopReport {
            kind: RepairKind::FeatureAddition,
            outcome: LoopOutcome::Adopted,
            attempts: Vec::new(),
            total_duration: std::time::Duration::from_millis(0),
        };
        // 存在しないディレクトリ配下のパスを渡し、`LogWriter` の追記処理
        // （ファイル生成）を確実に失敗させる（`logging.rs::LogWriter::
        // append_stages` が `OpenOptions::create(true)` で開こうとし、親
        // ディレクトリ不在により `NotFound` で失敗する）。
        let log_path = std::env::temp_dir().join(format!(
            "self-repair-main-test-nonexistent-dir-{}/trial.jsonl",
            std::process::id()
        ));

        let (_, persisted) = finish_with_report(&report, &log_path, None, None, None);
        assert!(!persisted);
    }

    /// PR #361 codex-review Medium 指摘の回帰防止: `--log` の書き込みは
    /// 成功するが `--output` の書き込みが失敗する場合、`persisted == true`
    /// （`--log` を一次記録として残せているため `run_run` は反映
    /// `reflect_adopted_diff` を実行してよい）であることを確認する。
    /// `--log` 側は実在する一時ディレクトリを渡して確実に成功させ、
    /// `--output` 側のみ親ディレクトリが存在しないパスを渡して失敗させる
    /// （`finish_with_report_reports_not_persisted_on_log_write_failure` と
    /// 対になる、`--log` 成功・`--output` 失敗の分岐を単独で踏むテスト）。
    #[test]
    fn finish_with_report_persists_when_only_output_write_fails() {
        let report = LoopReport {
            kind: RepairKind::FeatureAddition,
            outcome: LoopOutcome::Adopted,
            attempts: Vec::new(),
            total_duration: std::time::Duration::from_millis(0),
        };
        let log_dir = std::env::temp_dir().join(format!(
            "self-repair-main-test-output-fails-log-dir-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&log_dir).expect("log_dir の作成に失敗");
        let log_path = log_dir.join("trial.jsonl");
        let output_path = std::env::temp_dir().join(format!(
            "self-repair-main-test-nonexistent-dir-{}/report.json",
            std::process::id()
        ));

        let (exit_code, persisted) =
            finish_with_report(&report, &log_path, Some(&output_path), None, None);
        assert!(
            persisted,
            "--log が成功していれば --output 失敗単独では persisted は false にならないはず"
        );
        // `ExitCode` は `PartialEq` を実装しないため `Debug` 表示で比較する
        // （`persisted == true` でも終了コードは非 0 のまま保つ契約の確認）。
        assert_eq!(format!("{exit_code:?}"), format!("{:?}", ExitCode::from(1)));

        let _ = std::fs::remove_dir_all(&log_dir);
    }
}
