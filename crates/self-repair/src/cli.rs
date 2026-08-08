//! `self-repair` バイナリの自作コマンドライン引数パーサ（TASK-3.4 残作業・
//! イシュー #145）。
//!
//! `clap` は `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれず、
//! 依存追加はユーザー承認事項のため、`crates/guardrail/src/cli.rs` と同じ
//! 方針で `std::env::args` ベースの自作パースを行う。
//!
//! 結線済みのサブコマンドは `verify-log`（`docs/guardrail-self-repair-cli.md`
//! 3.2 節）と `run`（同 3.1 節。イシュー #142 差し戻し分・完走判定基準 1
//! で実装）の 2 つ。`run` は #131 が担当だったが未実装のまま closed になり、
//! 他に追跡イシューがないため #142 のスコープとして本モジュールへ拡張した
//! （実装計画 §2「スコープ判断」参照）。`--kind` で `RepairKind` の 3
//! variant を受理する種別非依存の実装とし、`self-repair` バイナリを必要と
//! する他イシュー（#141 のバグ修正種別再実証等）からも再利用できるようにする
//! （guardrail の `check`/`eval` 二本立てと同じ拡張パターンで
//! [`Command::VerifyLog`] と併存させる）。
//!
//! # 引数の受け渡しは `OsString`（PR #356 codex-review P1 指摘対応）
//!
//! `std::env::args()` は非 UTF-8 引数が渡ると panic する（exit 101。fail-closed
//! 契約の usage エラー exit 2・内部エラー exit 1 のいずれとも異なる非文書化の
//! 終了コードになってしまう）。ファイルシステム上は有効な非 UTF-8 パスの
//! ログファイルを `--log` に渡すケースを panic させないため、本モジュールは
//! `OsString` を受け取り、サブコマンド名・フラグ名のみを `to_str()` で UTF-8
//! 検証する（不正なら usage エラーへ変換。exit コードへの写像は `main.rs` に
//! 一本化）。`--log` の値自体は UTF-8 検証せず `PathBuf` へそのまま渡す。

use std::ffi::{OsStr, OsString};
use std::num::NonZeroU32;
use std::path::PathBuf;

use crate::kind::RepairKind;

/// CLI 引数の解析・検証に失敗した（未知引数・値欠落・不正なサブコマンド等）。
/// `docs/guardrail-self-repair-cli.md` 3.2 節には verify-log 固有の終了コード
/// 定義がないため、guardrail の usage エラー区分（終了コード `2`）に整合する
/// 契約を実装計画（イシュー #145 差し戻し分・完走判定基準 4）側で新たに定め、
/// `main.rs::report_error_and_exit` がこの型を終了コードへ写像する
/// （写像は 1 箇所に閉じ込め、fail-closed 契約を保つ。`.claude/rules/security.md` A08）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UsageError {}

/// `self-repair verify-log` の引数（3.2 節）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyLogArgs {
    pub log: PathBuf,
    /// `--allow-empty-log`（任意フラグ・値なし）。レコード 0 件のログを
    /// 明示的に許容する場合のみ指定する。指定なしでは空ログを exit 1
    /// （検証不合格扱い）にする（PR #356 codex-review P1 指摘対応:
    /// 空ログを無条件 exit 0 で通すと、ログ全削除による改竄を終了コードのみ
    /// 見る監査スクリプトが「検証成功」として見逃す経路になっていたため。
    /// `main.rs::run_verify_log` 参照）。
    pub allow_empty_log: bool,
}

/// `self-repair run`（3.1 節）の引数。`--kind` で対象種別を選び、以降の
/// 段階構築（検出・修正生成・検証・取り込み判断）を `main.rs::run_run` が
/// 種別ごとに分岐する（`SelfRepairLoop<D, F, V, J>` は型パラメータのため、
/// 実行時に選んだ種別に応じた具体型を選ぶ分岐は `cli.rs` ではなく `main.rs`
/// 側の責務）。
///
/// 3.1 節の表にない `--candidates`／`--bench-bin`／`--workload-source`／
/// `--policy-exclusion` は、同節が候補生成手段・ベンチ仕様の受け渡しを
/// 未定義としているため、本イシュー（#142 差し戻し分）で新たに定めて
/// `docs/guardrail-self-repair-cli.md` へ追記する（実装計画 §3.1）。
/// `--allow-candidate-exec`（必須フラグ）は PR #361 codex-review P0
/// 指摘対応で追加した（[`RunArgs::allow_candidate_exec`] 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArgs {
    /// 対象種別（必須）。
    pub kind: RepairKind,
    /// 対象リポジトリのルート（既定 `.`）。`RepairCompositeGateSpec` の
    /// `workspace`／`sandbox_root` 双方に使う（検証対象ワークスペースと
    /// diff・ベンチ実測の基点は同一という前提。実証ハーネスの sandbox 構成
    /// と同じ）。
    pub repo: PathBuf,
    /// 修正試行回数の上限（既定 5。`docs/self-repair-revalidation-plan.md`
    /// §5 基準 3 の承認済み提案値をそのまま既定値として採用する）。
    pub max_attempts: NonZeroU32,
    /// JSON Lines ログの出力先（必須。3.3 節）。
    pub log: PathBuf,
    /// guardrail 判定閾値の設定ファイル（任意。未指定時は `guardrail::config::resolve`
    /// が `--repo` 直下探索 → 組み込み既定値の順で解決する。2.4 節と共通）。
    pub config: Option<PathBuf>,
    /// `LoopReport`／`LoopFailure` JSON の書き出し先（任意）。
    pub output: Option<PathBuf>,
    /// 事前生成された候補修正列（JSON。必須）。`candidate::load_candidates_from_json`
    /// が `CandidateFix` へ変換する。候補生成手段自体（AI 生成・人手作成）は
    /// 本 CLI のスコープ外とし、事前に確定済みの候補列を受け取るのみとする
    /// （実装計画 §3.1「追加引数」）。
    pub candidates: PathBuf,
    /// ベンチワークロードの `[[bin]]` 名（必須。`RepairCompositeGateSpec::bench_bin`）。
    pub bench_bin: String,
    /// ゲーミング防止のためピン留めするワークロードソース（`--repo` 相対。
    /// 1 回以上必須。`RepairCompositeGateSpec::workload_sources`）。
    pub workload_sources: Vec<String>,
    /// REQ-5 除外ルール設定ファイル（任意。未指定時は `<repo>/policy-exclusion.toml`）。
    /// `main.rs::run_run` が候補適用前に一度だけロードし、
    /// `RepairCompositeGateSpec::policy_exclusion`（不変値）へ渡す
    /// （`docs/guardrail-self-repair-cli.md` 3.8 節「除外設定の固定」参照。
    /// PR #361 codex-review P1 指摘対応: 試行ごとの再読込は候補による判定
    /// 迂回を許すため廃止した）。
    pub policy_exclusion: Option<PathBuf>,
    /// `--candidates` に含まれる候補コードの実行に対する明示的な承認
    /// （必須フラグ。既定 false）。PR #361 codex-review P0 指摘対応:
    /// `--candidates` の候補コードは検証フェーズ（`RepairCompositeGate`。
    /// `verify_gates.rs`／`verify_direct_composite.rs`）で `cargo build`／
    /// `cargo test`／`cargo clippy` としてホスト権限のまま実行される。
    /// `RunSandbox`（`sandbox.rs`）はファイルシステム上の作業ツリー分離
    /// （`--repo` の作業ツリー・index を汚さない）のみを提供し、プロセス・
    /// 権限・ネットワークの隔離は行わない（`sandbox.rs` モジュール冒頭
    /// ドキュメント参照）。悪意ある候補（`build.rs`・テストコード）が任意
    /// コード実行しうるため、OS レベル隔離を実装するまでの間は「信頼済み
    /// 候補に限り、明示的な承認なしには実行しない」設計とし
    /// （`docs/guardrail-self-repair-cli.md`「候補実行の信頼境界」節参照）、
    /// `--allow-candidate-exec` を指定しない限り `parse_run` が usage エラー
    /// （exit 2）として拒否する。フラグは常に `true` の状態でのみ
    /// `RunArgs` を構築できるため（`parse_run` 末尾の検証）、値そのものは
    /// 情報としてのみ保持する。
    pub allow_candidate_exec: bool,
}

/// 本バイナリが受理するサブコマンド。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    VerifyLog(VerifyLogArgs),
    Run(RunArgs),
}

/// `std::env::args_os()` の実引数（プログラム名を除く）を受け取ってパースする。
/// `main.rs` から呼ばれ、返った [`Command`] に応じて実行フローを分岐する。
/// `OsString` を受け取る理由はモジュール冒頭ドキュメント参照（非 UTF-8 引数で
/// panic させないため）。
pub fn parse<I, S>(args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut it = args.into_iter().map(|s| s.as_ref().to_os_string());
    let subcommand = it
        .next()
        .ok_or_else(|| UsageError("missing subcommand (verify-log)".to_string()))?;
    let subcommand = subcommand.to_str().ok_or_else(|| {
        UsageError("subcommand must be valid UTF-8 (expected run or verify-log)".to_string())
    })?;

    match subcommand {
        "verify-log" => parse_verify_log(it).map(Command::VerifyLog),
        "run" => parse_run(it).map(Command::Run),
        other => Err(UsageError(format!(
            "unknown subcommand '{other}' (expected run or verify-log)"
        ))),
    }
}

/// `run` サブコマンドが受理する既知フラグ一覧（`KNOWN_FLAGS` と同じ役割。
/// `take_value` の値位置フラグ誤消費検出に使う）。
const RUN_KNOWN_FLAGS: &[&str] = &[
    "--kind",
    "--repo",
    "--max-attempts",
    "--log",
    "--config",
    "--output",
    "--candidates",
    "--bench-bin",
    "--workload-source",
    "--policy-exclusion",
    "--allow-candidate-exec",
];

/// `--kind` の値を [`RepairKind`] へ変換する（3.1 節「v1 `RepairKind`:
/// `BugFix`/`PerfRegression`/`FeatureAddition` を継承」の文字列表現）。
///
/// `RepairKind` は 3 variant を持つが、CLI が受理する値は `bug-fix`／
/// `feature-addition` の 2 つのみとする。`perf-regression` は
/// `PerfRegressionDetector`/`PerfRegressionFixGenerator` が `BenchMeasurer`・
/// 戦略リストという他 2 種別（`CommandRunner` ベース）と非対称な構築契約を
/// 持ち、CLI 側の結線（`main.rs::run_run`）が未実装のため、値として受理して
/// から実行時に内部エラー（exit 1）を返す従来の実装は「3 種別を受理する」
/// という契約を満たせていなかった（PR #361 codex-review P1 指摘）。結線が
/// 完成するまでは usage エラー（exit 2）として値域から明示的に除外し、
/// `main.rs` 側の実行時未対応分岐は防御的な到達不能ケースとしてのみ残す
/// （`main.rs` モジュール冒頭ドキュメント「`--kind perf-regression` は
/// 本イシューのスコープ外」参照）。
fn parse_repair_kind(value: &str) -> Result<RepairKind, UsageError> {
    match value {
        "bug-fix" => Ok(RepairKind::BugFix),
        "feature-addition" => Ok(RepairKind::FeatureAddition),
        "perf-regression" => Err(UsageError(
            "'--kind perf-regression' is not yet supported by the CLI \
             (PerfRegressionDetector/PerfRegressionFixGenerator have an asymmetric \
             construction contract and are not wired into `self-repair run` yet; \
             tracking-issue follow-up is pending user approval per \
             out-of-scope-tracking.md; expected bug-fix|feature-addition)"
                .to_string(),
        )),
        other => Err(UsageError(format!(
            "unknown value '{other}' for '--kind' (expected bug-fix|feature-addition)"
        ))),
    }
}

fn parse_run<I>(args: I) -> Result<RunArgs, UsageError>
where
    I: Iterator<Item = OsString>,
{
    let mut kind: Option<RepairKind> = None;
    let mut repo: Option<PathBuf> = None;
    let mut max_attempts: Option<NonZeroU32> = None;
    let mut log: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut candidates: Option<PathBuf> = None;
    let mut bench_bin: Option<String> = None;
    let mut workload_sources: Vec<String> = Vec::new();
    let mut policy_exclusion: Option<PathBuf> = None;
    let mut allow_candidate_exec = false;

    let mut it = args.peekable();
    while let Some(flag) = it.next() {
        let flag_str = flag.to_str().ok_or_else(|| {
            UsageError(format!(
                "argument '{}' must be valid UTF-8",
                flag.to_string_lossy()
            ))
        })?;
        match flag_str {
            "--kind" => {
                let value = take_value(&mut it, flag_str, RUN_KNOWN_FLAGS)?;
                let value_str = value
                    .to_str()
                    .ok_or_else(|| UsageError("--kind must be valid UTF-8".to_string()))?;
                kind = Some(parse_repair_kind(value_str)?);
            }
            "--repo" => {
                repo = Some(PathBuf::from(take_value(
                    &mut it,
                    flag_str,
                    RUN_KNOWN_FLAGS,
                )?))
            }
            "--max-attempts" => {
                let value = take_value(&mut it, flag_str, RUN_KNOWN_FLAGS)?;
                let value_str = value
                    .to_str()
                    .ok_or_else(|| UsageError("--max-attempts must be valid UTF-8".to_string()))?;
                let parsed: u32 = value_str.parse().map_err(|_| {
                    UsageError(format!(
                        "--max-attempts must be a positive integer (got '{value_str}')"
                    ))
                })?;
                max_attempts = Some(NonZeroU32::new(parsed).ok_or_else(|| {
                    UsageError("--max-attempts must be >= 1 (0 is rejected)".to_string())
                })?);
            }
            "--log" => {
                log = Some(PathBuf::from(take_value(
                    &mut it,
                    flag_str,
                    RUN_KNOWN_FLAGS,
                )?))
            }
            "--config" => {
                config = Some(PathBuf::from(take_value(
                    &mut it,
                    flag_str,
                    RUN_KNOWN_FLAGS,
                )?))
            }
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    &mut it,
                    flag_str,
                    RUN_KNOWN_FLAGS,
                )?))
            }
            "--candidates" => {
                candidates = Some(PathBuf::from(take_value(
                    &mut it,
                    flag_str,
                    RUN_KNOWN_FLAGS,
                )?))
            }
            "--bench-bin" => {
                let value = take_value(&mut it, flag_str, RUN_KNOWN_FLAGS)?;
                let value_str = value
                    .to_str()
                    .ok_or_else(|| UsageError("--bench-bin must be valid UTF-8".to_string()))?;
                bench_bin = Some(value_str.to_string());
            }
            "--workload-source" => {
                let value = take_value(&mut it, flag_str, RUN_KNOWN_FLAGS)?;
                let value_str = value.to_str().ok_or_else(|| {
                    UsageError("--workload-source must be valid UTF-8".to_string())
                })?;
                workload_sources.push(value_str.to_string());
            }
            "--policy-exclusion" => {
                policy_exclusion = Some(PathBuf::from(take_value(
                    &mut it,
                    flag_str,
                    RUN_KNOWN_FLAGS,
                )?))
            }
            "--allow-candidate-exec" => allow_candidate_exec = true,
            unknown => {
                return Err(UsageError(format!(
                    "unknown argument '{unknown}' for 'self-repair run'"
                )));
            }
        }
    }

    let kind = kind.ok_or_else(|| UsageError("missing required argument '--kind'".to_string()))?;
    let log = log.ok_or_else(|| UsageError("missing required argument '--log'".to_string()))?;
    let candidates = candidates
        .ok_or_else(|| UsageError("missing required argument '--candidates'".to_string()))?;
    let bench_bin = bench_bin
        .ok_or_else(|| UsageError("missing required argument '--bench-bin'".to_string()))?;
    if workload_sources.is_empty() {
        return Err(UsageError(
            "missing required argument '--workload-source' (specify at least once)".to_string(),
        ));
    }
    // PR #361 codex-review P0 指摘対応: `--candidates` は必須のため
    // `self-repair run` の全呼び出しが候補コードを実行しうる。sandbox clone
    // （`sandbox.rs`）はファイルシステム上の作業ツリー分離のみでプロセス・
    // 権限・ネットワークを隔離しないため、明示的な承認（`--allow-candidate-exec`）
    // なしにはここで usage エラーとして拒否する（fail-closed。`RunArgs` は
    // このフラグが true の場合にのみ構築されるため、以降の実行経路
    // 〈`main.rs::run_run`〉に到達する前に構造的に拒否が保証される）。
    // `RunArgs` フィールド doc・`docs/guardrail-self-repair-cli.md`
    // 「候補実行の信頼境界」節参照。
    if !allow_candidate_exec {
        return Err(UsageError(
            "refusing to run: '--candidates' code is executed via cargo build/test/clippy \
             with host process privileges (the sandbox clone only isolates the filesystem \
             work tree, not process/privilege/network access). Pass --allow-candidate-exec \
             only for trusted candidates to acknowledge this and proceed"
                .to_string(),
        ));
    }
    let repo = repo.unwrap_or_else(|| PathBuf::from("."));
    let max_attempts = max_attempts.unwrap_or_else(|| {
        NonZeroU32::new(5).expect("5 は非ゼロ（docs/self-repair-revalidation-plan.md §5 基準 3）")
    });

    Ok(RunArgs {
        kind,
        repo,
        max_attempts,
        log,
        config,
        output,
        candidates,
        bench_bin,
        workload_sources,
        policy_exclusion,
        allow_candidate_exec,
    })
}

/// `verify-log` サブコマンドが受理する既知フラグ一覧（`take_value` が値位置の
/// フラグ誤消費を検出するために参照する。PR #356 codex-review P1 指摘対応）。
const KNOWN_FLAGS: &[&str] = &["--log", "--allow-empty-log"];

fn parse_verify_log<I>(args: I) -> Result<VerifyLogArgs, UsageError>
where
    I: Iterator<Item = OsString>,
{
    let mut log: Option<PathBuf> = None;
    let mut allow_empty_log = false;

    let mut it = args.peekable();
    while let Some(flag) = it.next() {
        let flag_str = flag.to_str().ok_or_else(|| {
            UsageError(format!(
                "argument '{}' must be valid UTF-8",
                flag.to_string_lossy()
            ))
        })?;
        match flag_str {
            "--log" => log = Some(PathBuf::from(take_value(&mut it, flag_str, KNOWN_FLAGS)?)),
            "--allow-empty-log" => allow_empty_log = true,
            unknown => {
                return Err(UsageError(format!(
                    "unknown argument '{unknown}' for 'self-repair verify-log'"
                )));
            }
        }
    }

    let log = log.ok_or_else(|| UsageError("missing required argument '--log'".to_string()))?;
    Ok(VerifyLogArgs {
        log,
        allow_empty_log,
    })
}

/// `--flag value` 形式で次のトークンを値として取り出す。値欠落は usage エラー。
/// 値は UTF-8 検証しない（`--log` のファイルパスが非 UTF-8 でもそのまま
/// `OsString` として受理する。モジュール冒頭ドキュメント参照）。
///
/// 次のトークンが `known_flags`（`verify-log` が受理する既知フラグ）と
/// UTF-8 完全一致する場合は値として消費せず、値欠落の usage エラーとして扱う
/// （PR #356 codex-review P1 指摘対応: `--log --allow-empty-log` のように値を
/// 省略された `--log` が後続の既知フラグをファイル名として誤飲し、
/// `--allow-empty-log` 指定自体を消失させたまま I/O エラー〈exit 1〉に
/// フォールスルーしていた。値位置に非 UTF-8 や `--` 始まりでも既知フラグでない
/// 実ファイル名を許容する必要がある場合は `--log=...` 形式や `--` エスケープを
/// 別途用意する想定とし、本関数は現行の空白区切り構文のみを対象にする）。
fn take_value<I>(
    it: &mut std::iter::Peekable<I>,
    flag: &str,
    known_flags: &[&str],
) -> Result<OsString, UsageError>
where
    I: Iterator<Item = OsString>,
{
    if let Some(next) = it.peek()
        && let Some(next_str) = next.to_str()
        && known_flags.contains(&next_str)
    {
        return Err(UsageError(format!("missing value for '{flag}'")));
    }
    it.next()
        .ok_or_else(|| UsageError(format!("missing value for '{flag}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_verify_log_with_log_arg() {
        let cmd = parse(["verify-log", "--log", "trial.jsonl"]).unwrap();
        let Command::VerifyLog(args) = cmd else {
            panic!("expected Command::VerifyLog")
        };
        assert_eq!(args.log, PathBuf::from("trial.jsonl"));
        assert!(!args.allow_empty_log);
    }

    #[test]
    fn parses_verify_log_with_allow_empty_log_flag() {
        let cmd = parse(["verify-log", "--log", "trial.jsonl", "--allow-empty-log"]).unwrap();
        let Command::VerifyLog(args) = cmd else {
            panic!("expected Command::VerifyLog")
        };
        assert!(args.allow_empty_log);
    }

    #[test]
    fn rejects_missing_log_arg() {
        let err = parse(["verify-log"]).unwrap_err();
        assert!(err.0.contains("--log"));
    }

    #[test]
    fn rejects_missing_value_for_log() {
        let err = parse(["verify-log", "--log"]).unwrap_err();
        assert!(err.0.contains("--log"));
    }

    /// PR #356 codex-review P1 指摘対応: `--log` の値位置に既知フラグ
    /// `--allow-empty-log` が来た場合、ファイル名として誤消費せず値欠落の
    /// usage エラーにする（`take_value` 参照）。
    #[test]
    fn rejects_known_flag_as_log_value() {
        let err = parse(["verify-log", "--log", "--allow-empty-log"]).unwrap_err();
        assert!(err.0.contains("--log"));
    }

    #[test]
    fn rejects_unknown_argument() {
        let err = parse(["verify-log", "--bogus", "x"]).unwrap_err();
        assert!(err.0.contains("--bogus"));
    }

    #[test]
    fn rejects_missing_subcommand() {
        let err = parse(Vec::<&str>::new()).unwrap_err();
        assert!(err.0.contains("subcommand"));
    }

    #[test]
    fn rejects_unknown_subcommand() {
        let err = parse(["bogus"]).unwrap_err();
        assert!(err.0.contains("bogus"));
    }

    /// PR #356 codex-review P1 指摘対応: 非 UTF-8 の `--log` 値は panic せず
    /// `PathBuf` としてそのまま受理される（ファイルシステム上は有効な非 UTF-8
    /// パスの拒否は `verify_chain` の I/O エラー経路〈exit 1〉に委ねる）。
    #[cfg(unix)]
    #[test]
    fn accepts_non_utf8_log_value_without_panicking() {
        use std::os::unix::ffi::OsStrExt;

        let non_utf8_log = OsStr::from_bytes(b"log-\xff\xfe.jsonl").to_os_string();
        let cmd = parse([
            OsString::from("verify-log"),
            OsString::from("--log"),
            non_utf8_log.clone(),
        ])
        .unwrap();
        let Command::VerifyLog(args) = cmd else {
            panic!("expected Command::VerifyLog")
        };
        assert_eq!(args.log, PathBuf::from(non_utf8_log));
    }

    /// PR #356 codex-review P1 指摘対応: サブコマンド名自体が非 UTF-8 の場合は
    /// panic ではなく usage エラー（exit 2）に写像される。
    #[cfg(unix)]
    #[test]
    fn rejects_non_utf8_subcommand_without_panicking() {
        use std::os::unix::ffi::OsStrExt;

        let non_utf8_subcommand = OsStr::from_bytes(b"\xff\xfe").to_os_string();
        let err = parse([non_utf8_subcommand]).unwrap_err();
        assert!(err.0.contains("UTF-8"));
    }

    /// `run` の必須引数（`--kind`／`--log`／`--candidates`／`--bench-bin`／
    /// `--workload-source` 1 回以上）をすべて満たす最小構成が通ること・
    /// 未指定の任意引数が既定値（`--repo` = `.`／`--max-attempts` = 5）に
    /// なることを確認する。
    #[test]
    fn parses_run_with_required_args_and_defaults() {
        let cmd = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
            "--allow-candidate-exec",
        ])
        .unwrap();
        let Command::Run(args) = cmd else {
            panic!("expected Command::Run")
        };
        assert_eq!(args.kind, RepairKind::FeatureAddition);
        assert_eq!(args.repo, PathBuf::from("."));
        assert_eq!(args.max_attempts, NonZeroU32::new(5).unwrap());
        assert_eq!(args.log, PathBuf::from("trial.jsonl"));
        assert_eq!(args.candidates, PathBuf::from("candidates.json"));
        assert_eq!(args.bench_bin, "bench_workload");
        assert_eq!(
            args.workload_sources,
            vec!["src/bin/bench_workload.rs".to_string()]
        );
        assert_eq!(args.config, None);
        assert_eq!(args.output, None);
        assert_eq!(args.policy_exclusion, None);
        assert!(args.allow_candidate_exec);
    }

    /// PR #361 codex-review P0 指摘対応の回帰防止: `--allow-candidate-exec`
    /// を指定しない場合、他の必須引数を全て満たしていても usage エラー
    /// （exit 2 相当。`main.rs::report_usage_error_and_exit` 参照）として
    /// 拒否されることを確認する（`--candidates` の候補コードが明示的な
    /// 承認なしに実行されない設計。`RunArgs::allow_candidate_exec` doc 参照）。
    #[test]
    fn rejects_run_without_allow_candidate_exec() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
        ])
        .unwrap_err();
        assert!(err.0.contains("--allow-candidate-exec"));
    }

    /// `--workload-source` を複数回指定すると累積されること（ゲーミング防止の
    /// ピン留め対象が複数ファイルにまたがるケースを想定）。
    #[test]
    fn parses_run_with_repeated_workload_source_and_overrides() {
        let cmd = parse([
            "run",
            "--kind",
            "bug-fix",
            "--repo",
            "/tmp/sandbox",
            "--max-attempts",
            "3",
            "--log",
            "trial.jsonl",
            "--config",
            "guardrail.toml",
            "--output",
            "report.json",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
            "--workload-source",
            "src/lib.rs",
            "--policy-exclusion",
            "policy-exclusion.toml",
            "--allow-candidate-exec",
        ])
        .unwrap();
        let Command::Run(args) = cmd else {
            panic!("expected Command::Run")
        };
        assert_eq!(args.kind, RepairKind::BugFix);
        assert!(args.allow_candidate_exec);
        assert_eq!(args.repo, PathBuf::from("/tmp/sandbox"));
        assert_eq!(args.max_attempts, NonZeroU32::new(3).unwrap());
        assert_eq!(args.config, Some(PathBuf::from("guardrail.toml")));
        assert_eq!(args.output, Some(PathBuf::from("report.json")));
        assert_eq!(
            args.workload_sources,
            vec![
                "src/bin/bench_workload.rs".to_string(),
                "src/lib.rs".to_string()
            ]
        );
        assert_eq!(
            args.policy_exclusion,
            Some(PathBuf::from("policy-exclusion.toml"))
        );
    }

    #[test]
    fn rejects_run_missing_kind() {
        let err = parse([
            "run",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(err.0.contains("--kind"));
    }

    #[test]
    fn rejects_run_unknown_kind_value() {
        let err = parse([
            "run",
            "--kind",
            "bogus",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(err.0.contains("--kind"));
    }

    /// PR #361 codex-review P1 指摘の回帰防止: `--kind perf-regression` は
    /// `RepairKind` 型としては存在するが、`main.rs::run_run` に結線されて
    /// いない（実行時は常に内部エラーを返していた）ため、値としても usage
    /// エラー（exit 2）として拒否し、「3 種別を受理する」という誤った契約を
    /// 公開しないことを確認する（`parse_repair_kind` doc 参照）。
    #[test]
    fn rejects_run_perf_regression_kind_as_unsupported() {
        let err = parse([
            "run",
            "--kind",
            "perf-regression",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(err.0.contains("perf-regression"));
        assert!(err.0.contains("not yet supported"));
    }

    /// 上記回帰防止テストと対になる確認: 未知の `--kind` 値（`bogus`）の
    /// エラーメッセージから `perf-regression` が消えている（値域として
    /// 案内しなくなった）ことを固定する。二つの `UsageError` 文字列が
    /// 将来のリファクタで再び食い違うのを防ぐ（advisor 指摘）。
    #[test]
    fn unknown_kind_value_message_no_longer_advertises_perf_regression() {
        let err = parse([
            "run",
            "--kind",
            "bogus",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(!err.0.contains("perf-regression"));
    }

    #[test]
    fn rejects_run_missing_log() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(err.0.contains("--log"));
    }

    #[test]
    fn rejects_run_missing_candidates() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(err.0.contains("--candidates"));
    }

    #[test]
    fn rejects_run_missing_bench_bin() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--workload-source",
            "s",
        ])
        .unwrap_err();
        assert!(err.0.contains("--bench-bin"));
    }

    #[test]
    fn rejects_run_missing_workload_source() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
        ])
        .unwrap_err();
        assert!(err.0.contains("--workload-source"));
    }

    #[test]
    fn rejects_run_zero_max_attempts() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
            "--max-attempts",
            "0",
        ])
        .unwrap_err();
        assert!(err.0.contains("--max-attempts"));
    }

    #[test]
    fn rejects_run_non_numeric_max_attempts() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
            "--max-attempts",
            "abc",
        ])
        .unwrap_err();
        assert!(err.0.contains("--max-attempts"));
    }

    #[test]
    fn rejects_run_unknown_argument() {
        let err = parse([
            "run",
            "--kind",
            "feature-addition",
            "--log",
            "trial.jsonl",
            "--candidates",
            "candidates.json",
            "--bench-bin",
            "b",
            "--workload-source",
            "s",
            "--bogus",
            "x",
        ])
        .unwrap_err();
        assert!(err.0.contains("--bogus"));
    }
}
