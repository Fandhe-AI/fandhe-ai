//! 試行ごとの diff 由来シグナル実測（TASK-3.2a・イシュー #137）。
//!
//! [`crate::verify_gates::CargoVerificationGate`] は `lines_changed`／
//! `api_broken`／`gaming_suspect`／`exclusion_rule_ids` を構築時の必須引数として
//! 受け取るのみで自ら計測しない（`verify_gates.rs` モジュール冒頭「スコープ境界」
//! 参照）。この実測ロジックはこれまで `tests/revalidation_bug_fix.rs`
//! （`diff_numstat`／`api_signature_touched`／`gaming_suspect`／
//! `evaluate_exclusion_rules`）にのみ存在し、テスト専用の使い捨て実装だった。
//!
//! 本モジュールはそのロジックを `crates/self-repair/src/` 本体へ昇格し、
//! [`crate::verify_direct_composite::RepairCompositeGate`] が**試行ごとに**
//! 呼び出せるようにする（実装計画 #137 §3.3・§4）。これにより
//! `crate::verify_composite::FeatureAdditionCompositeGate`（構築時にシグナルを
//! 固定する既存合成ゲート）が持つ「試行ごとの再計測ができない」という設計制約
//! （`verify_composite.rs` モジュール冒頭ドキュメント参照）を、新設の合成ゲート側
//! では解消する。
//!
//! `revalidation_bug_fix.rs` 自体は編集しない（#141 との並行編集衝突を避ける。
//! `.claude/rules/delegation-impl.md`）。ロジックの複製元として参照するのみ。
//!
//! # fail-closed 契約（A08）
//! git 呼び出し・ポリシー除外リストのロード/評価がいずれか失敗した場合、
//! 未計測値を fail-open な既定値（0 行・破壊なし・ゲーミング疑いなし）で
//! 埋めず [`Err`] を返す（`.claude/rules/security.md` A08「判定の迂回経路を
//! 作らない」。`verify_gates.rs` の契約と同じ）。
//!
//! # A03（インジェクション）対応
//! `baseline_commit` は git コマンドライン引数へ渡す前に 7〜40 桁の 16 進文字列
//! であることを検証する（[`validate_commit_ref`]）。検証しない場合、先頭に
//! `-` を含む値（例: `--upload-pack=...`）を git のオプションとして誤解釈させる
//! 攻撃（コマンドラインオプション偽装）が成立しうる
//! （`.claude/rules/security.md` A03）。パス引数は `--` 区切りで分離し、
//! シェル（`sh -c` 等）は一切経由しない（`crate::exec` モジュール冒頭ドキュメント
//! と同じ設計）。

use std::path::Path;

use crate::exec::CommandRunner;

/// 試行ごとに実測した diff 由来 4 シグナル（`guardrail::DecisionInput::new` の
/// `lines_changed`／`api_broken`／`gaming_suspect`／`exclusion_rule_ids` 引数に
/// 1 対 1 対応する。`outcome.rs` の `VerifiedEvidence` フィールドと同じ意味）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSignals {
    /// `baseline_commit` から現在の作業木までの追加行数＋削除行数の合計
    /// （`git diff --numstat` の実測値）。
    pub lines_changed: u64,
    /// 追加・削除された行に `pub fn` 宣言が含まれるか（PoC-3 パリティの簡易
    /// ヒューリスティック。`guardrail::checks::api_stability` と同種の判定だが、
    /// 当該モジュールは `guardrail` の非公開実装のため本モジュールが独立して
    /// 実装する）。
    pub api_broken: bool,
    /// 変更ファイル一覧に本番コードとテストコードの双方が同時に含まれるか
    /// （ゲーミング疑いの簡易ヒューリスティック）。
    pub gaming_suspect: bool,
    /// match したポリシー除外リストのルール `id` 一覧（空 = match なし）。
    pub exclusion_rule_ids: Vec<String>,
}

/// [`measure_diff_signals`] の実測時エラー。
///
/// `crate::error::SelfRepairError` は `attempt` を必須フィールドとして要求する
/// （`error.rs` 参照）が、本関数は `attempt` を知らない（呼び出し元
/// `RepairCompositeGate::verify` が `Proposal::attempt` を知る）ため、独立した
/// エラー型として返し、呼び出し元で `SelfRepairError::Verification` へ変換する
/// （`crate::verify_gates::CargoVerificationGate::run_gate` が `ExecError` を
/// 変換する契約と同じ設計）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSignalsError {
    message: String,
}

impl DiffSignalsError {
    fn new(message: impl Into<String>) -> Self {
        DiffSignalsError {
            message: message.into(),
        }
    }

    /// 人間可読なエラー内容（`SelfRepairError::Verification.reason` へ埋め込む
    /// 用途）。
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for DiffSignalsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "diff 由来シグナルの実測に失敗しました: {}", self.message)
    }
}

impl std::error::Error for DiffSignalsError {}

/// `baseline_commit` が git のコマンドラインオプションとして誤解釈されない
/// 16 進 commit sha（短縮形含む）であることを検証する（モジュール冒頭
/// 「A03（インジェクション）対応」参照）。
fn validate_commit_ref(baseline_commit: &str) -> Result<(), DiffSignalsError> {
    let is_valid = (7..=40).contains(&baseline_commit.len())
        && baseline_commit
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && (b.is_ascii_digit() || b.is_ascii_lowercase()));
    if is_valid {
        Ok(())
    } else {
        Err(DiffSignalsError::new(format!(
            "baseline_commit は 7〜40 桁の小文字 16 進文字列である必要があります: {baseline_commit:?}"
        )))
    }
}

/// `runner` で `git` を `sandbox_root` を cwd として起動し、標準出力（UTF-8
/// lossy 変換済み）を返す。非 0 終了・spawn 失敗はいずれも [`Err`]（fail-closed。
/// stderr は診断用にメッセージへ含める）。
fn run_git<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    args: &[&str],
) -> Result<String, DiffSignalsError> {
    let output = runner
        .run("git", args, sandbox_root)
        .map_err(|error| DiffSignalsError::new(format!("git {args:?} の起動に失敗: {error}")))?;
    if !output.success() {
        return Err(DiffSignalsError::new(format!(
            "git {args:?} が失敗しました: {}",
            output.log_tail()
        )));
    }
    Ok(output.log_tail().to_string())
}

/// `baseline_commit` と現在の作業木の diff から `lines_changed`（追加行数＋
/// 削除行数の合計）・変更ファイル一覧を実測する。移植元:
/// `tests/revalidation_bug_fix.rs::diff_numstat`（モジュール冒頭ドキュメント
/// 参照）。バイナリファイル等で `added`/`deleted` 列の解析に失敗した場合、
/// fail-open な 0 加算にせず [`Err`] を返す（移植元コメントの fail-closed 方針を
/// そのまま踏襲。本番経路のため `panic!` ではなく型付きエラーにする）。
fn diff_numstat<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
) -> Result<(u64, Vec<String>), DiffSignalsError> {
    let stdout = run_git(
        runner,
        sandbox_root,
        &["diff", "--numstat", baseline_commit, "--"],
    )?;
    let mut lines_changed: u64 = 0;
    let mut files = Vec::new();
    for line in stdout.lines() {
        let mut cols = line.splitn(3, '\t');
        let added = cols.next().ok_or_else(|| {
            DiffSignalsError::new(format!("git diff --numstat の行が想定外です: {line:?}"))
        })?;
        let deleted = cols.next().ok_or_else(|| {
            DiffSignalsError::new(format!("git diff --numstat の行が想定外です: {line:?}"))
        })?;
        let path = cols.next().unwrap_or("").to_string();
        lines_changed += added.parse::<u64>().map_err(|error| {
            DiffSignalsError::new(format!(
                "added 列の解析に失敗しました（fail-open で 0 に丸めない）: {added:?}: {error}"
            ))
        })?;
        lines_changed += deleted.parse::<u64>().map_err(|error| {
            DiffSignalsError::new(format!(
                "deleted 列の解析に失敗しました（fail-open で 0 に丸めない）: {deleted:?}: {error}"
            ))
        })?;
        if !path.is_empty() {
            files.push(path);
        }
    }
    Ok((lines_changed, files))
}

/// 公開関数シグネチャ（`pub fn` を含む行）が diff の追加・削除行に現れないかを
/// 検査する。移植元: `tests/revalidation_bug_fix.rs::api_signature_touched`。
fn api_signature_touched<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
) -> Result<bool, DiffSignalsError> {
    let stdout = run_git(
        runner,
        sandbox_root,
        &["diff", "--no-color", "-U0", baseline_commit, "--", "*.rs"],
    )?;
    Ok(stdout.lines().any(|line| {
        let content = line
            .strip_prefix('+')
            .or_else(|| line.strip_prefix('-'))
            .unwrap_or("");
        // 差分ヘッダ行（`+++`/`---`）を誤検出しないよう、`+`/`-` 直後がさらに
        // `+`/`-` の場合は除外する。
        !content.starts_with(['+', '-']) && content.trim_start().starts_with("pub fn")
    }))
}

/// 変更ファイル一覧に「本番コード」と「テストコード」の双方が同時に含まれる
/// かを検査する（ゲーミング疑いの簡易ヒューリスティック）。移植元:
/// `tests/revalidation_bug_fix.rs::gaming_suspect`。
fn gaming_suspect_from_files(changed_files: &[String]) -> bool {
    let touches_test = changed_files
        .iter()
        .any(|path| path.contains("/tests/") || path.ends_with("_test.rs"));
    let touches_prod = changed_files.iter().any(|path| {
        !path.contains("/tests/") && !path.ends_with("_test.rs") && path.ends_with(".rs")
    });
    touches_test && touches_prod
}

/// `guardrail` のポリシー除外リスト評価（REQ-5）を `sandbox_root` 上で実行する。
/// `policy_exclusion_path` の `policy-exclusion.toml` をロードし、
/// `baseline_commit` と現作業木の diff に対して評価する。移植元:
/// `tests/revalidation_bug_fix.rs::evaluate_exclusion_rules`（`std::fs::read_to_string`
/// を用いる点は同一。`guardrail::EvaluationContext::from_repo` 自体が内部で git を
/// 起動するため、`CommandRunner` 注入はここでは効かない——これは guardrail 側
/// 公開 API の既存契約であり、本モジュールで変更しない）。
fn evaluate_exclusion_rules(
    sandbox_root: &Path,
    baseline_commit: &str,
    policy_exclusion_path: &Path,
) -> Result<Vec<String>, DiffSignalsError> {
    let toml = std::fs::read_to_string(policy_exclusion_path).map_err(|error| {
        DiffSignalsError::new(format!(
            "{} の読み込みに失敗しました: {error}",
            policy_exclusion_path.display()
        ))
    })?;
    let config = guardrail::load_policy_exclusion(&toml).map_err(|error| {
        DiffSignalsError::new(format!("policy-exclusion.toml のパースに失敗: {error}"))
    })?;
    let ctx = guardrail::EvaluationContext::from_repo(sandbox_root, baseline_commit).map_err(
        |error| DiffSignalsError::new(format!("EvaluationContext の構築に失敗: {error}")),
    )?;
    let evaluation = guardrail::ExclusionEvaluation::evaluate(&config.rules, &ctx)
        .map_err(|error| DiffSignalsError::new(format!("除外リスト評価に失敗: {error}")))?;
    Ok(evaluation.effective_rule_ids())
}

/// `sandbox_root` 上で `baseline_commit` からの diff・ポリシー除外リストを
/// **試行ごとに**実測し、4 シグナルをまとめて返す
/// （[`crate::verify_direct_composite::RepairCompositeGate::verify`] から
/// 候補適用直後の作業木に対して毎試行呼ばれる想定。モジュール冒頭ドキュメント
/// 参照）。
///
/// # Errors
///
/// `baseline_commit` の形式検証・git 呼び出し・ポリシー除外リストのロード／
/// 評価のいずれかに失敗した場合 [`DiffSignalsError`]（fail-closed。未計測値を
/// 既定値で埋めない）。
pub fn measure_diff_signals<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
    policy_exclusion_path: &Path,
) -> Result<DiffSignals, DiffSignalsError> {
    validate_commit_ref(baseline_commit)?;
    let (lines_changed, changed_files) = diff_numstat(runner, sandbox_root, baseline_commit)?;
    let api_broken = api_signature_touched(runner, sandbox_root, baseline_commit)?;
    let gaming_suspect = gaming_suspect_from_files(&changed_files);
    let exclusion_rule_ids =
        evaluate_exclusion_rules(sandbox_root, baseline_commit, policy_exclusion_path)?;
    Ok(DiffSignals {
        lines_changed,
        api_broken,
        gaming_suspect,
        exclusion_rule_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{CommandOutput, ExecError};
    use std::cell::RefCell;

    /// スクリプト化した `CommandRunner` テストダブル。`args` の先頭から
    /// `git diff --numstat ...` / `git diff -U0 ...` 等を区別して固定応答を
    /// 返す（`verify_gates.rs` のテストダブルと同種の設計）。
    struct ScriptedGit {
        numstat_stdout: String,
        signature_stdout: String,
        calls: RefCell<Vec<Vec<String>>>,
    }

    impl ScriptedGit {
        fn new(numstat_stdout: &str, signature_stdout: &str) -> Self {
            ScriptedGit {
                numstat_stdout: numstat_stdout.to_string(),
                signature_stdout: signature_stdout.to_string(),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl CommandRunner for ScriptedGit {
        fn run(
            &self,
            _program: &str,
            args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            self.calls
                .borrow_mut()
                .push(args.iter().map(|s| s.to_string()).collect());
            let stdout = if args.contains(&"--numstat") {
                self.numstat_stdout.clone()
            } else if args.contains(&"-U0") {
                self.signature_stdout.clone()
            } else {
                String::new()
            };
            Ok(CommandOutput::from_captured(true, stdout.into_bytes()))
        }
    }

    /// spawn 自体が失敗するテストダブル。
    struct FailingRunner;
    impl CommandRunner for FailingRunner {
        fn run(
            &self,
            _program: &str,
            _args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            Err(ExecError::new("spawn failed (scripted)"))
        }
    }

    #[test]
    fn validate_commit_ref_accepts_valid_short_and_full_sha() {
        assert!(validate_commit_ref("abc1234").is_ok());
        assert!(validate_commit_ref(&"a".repeat(40)).is_ok());
    }

    #[test]
    fn validate_commit_ref_rejects_option_like_value() {
        let err = validate_commit_ref("--upload-pack=evil").expect_err("拒否されるはず");
        assert!(err.message().contains("baseline_commit"));
    }

    #[test]
    fn validate_commit_ref_rejects_uppercase_and_short_values() {
        assert!(validate_commit_ref("ABCDEFA").is_err(), "大文字は拒否");
        assert!(
            validate_commit_ref("abc123").is_err(),
            "6 桁は下限未満で拒否"
        );
    }

    #[test]
    fn diff_numstat_parses_added_and_deleted_columns() {
        let runner = ScriptedGit::new("3\t1\tsrc/lib.rs\n0\t2\ttests/foo.rs\n", "");
        let (lines_changed, files) =
            diff_numstat(&runner, Path::new("/sandbox"), "abc1234").expect("解析成功");
        assert_eq!(lines_changed, 6);
        assert_eq!(files, vec!["src/lib.rs", "tests/foo.rs"]);
    }

    #[test]
    fn diff_numstat_fails_closed_on_unparseable_column() {
        let runner = ScriptedGit::new("not-a-number\t1\tsrc/lib.rs\n", "");
        let err = diff_numstat(&runner, Path::new("/sandbox"), "abc1234")
            .expect_err("fail-open で 0 に丸めず拒否されるはず");
        assert!(err.message().contains("added"));
    }

    #[test]
    fn api_signature_touched_detects_pub_fn_in_diff() {
        let runner = ScriptedGit::new(
            "",
            "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn new_api() {}\n",
        );
        let touched =
            api_signature_touched(&runner, Path::new("/sandbox"), "abc1234").expect("成功");
        assert!(touched);
    }

    #[test]
    fn api_signature_touched_ignores_diff_header_lines() {
        let runner = ScriptedGit::new("", "+++ b/src/lib.rs\n--- a/src/lib.rs\n+let x = 1;\n");
        let touched =
            api_signature_touched(&runner, Path::new("/sandbox"), "abc1234").expect("成功");
        assert!(!touched, "ヘッダ行・非 pub fn 行は検出されないはず");
    }

    #[test]
    fn gaming_suspect_from_files_true_when_prod_and_test_both_touched() {
        let files = vec!["src/lib.rs".to_string(), "tests/foo_test.rs".to_string()];
        assert!(gaming_suspect_from_files(&files));
    }

    #[test]
    fn gaming_suspect_from_files_false_when_only_prod_touched() {
        let files = vec!["src/lib.rs".to_string()];
        assert!(!gaming_suspect_from_files(&files));
    }

    #[test]
    fn measure_diff_signals_rejects_invalid_baseline_commit_before_spawning() {
        let runner = FailingRunner;
        let err = measure_diff_signals(
            &runner,
            Path::new("/sandbox"),
            "--evil",
            Path::new("/sandbox/policy-exclusion.toml"),
        )
        .expect_err("不正な baseline_commit は git 起動前に拒否されるはず");
        assert!(err.message().contains("baseline_commit"));
    }

    #[test]
    fn measure_diff_signals_propagates_spawn_failure() {
        let err = measure_diff_signals(
            &FailingRunner,
            Path::new("/sandbox"),
            "abc1234",
            Path::new("/sandbox/policy-exclusion.toml"),
        )
        .expect_err("spawn 失敗は Err として伝播するはず");
        assert!(err.message().contains("起動に失敗"));
    }
}
