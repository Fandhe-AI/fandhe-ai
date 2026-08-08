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
///
/// # ログ切り詰めの扱い（Codex レビュー #137 指摘）
/// [`crate::exec::CommandOutput::log_tail`] は 256 KiB 超過時に**先頭側**を
/// 切り詰める（`exec.rs` の `MAX_CAPTURED_LOG_BYTES` 参照）。本関数の戻り値は
/// `diff_numstat`／`api_signature_touched` が `git diff --numstat`／
/// `git diff -U0` の構造化出力として逐次パースするため、先頭側が欠落した
/// ログをそのまま解析すると `lines_changed` の過少計上・`api_broken` の
/// 見逃しに繋がる（fail-closed 契約違反。モジュール冒頭ドキュメント参照）。
/// `output.truncated()` が真の場合は解析を試みず [`Err`] を返す。
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
    if output.truncated() {
        return Err(DiffSignalsError::new(format!(
            "git {args:?} の出力が 256 KiB 上限で切り詰められました。構造化出力の \
             先頭側欠落により lines_changed／api_broken を誤計測しうるため、\
             fail-open な部分解析はせず拒否します（fail-closed。\
             .claude/rules/security.md A08）"
        )));
    }
    Ok(output.log_tail().to_string())
}

/// `sandbox_root` の作業木で未追跡ファイルを含む全ファイルを index へ反映する
/// （`git add -A -- .`）。
///
/// `diff_numstat`／`api_signature_touched`（本モジュール）と
/// `guardrail::EvaluationContext::from_repo`（`evaluate_exclusion_rules` が
/// 内部で呼ぶ。`git diff <baseline_commit>` を独自に起動する既存契約であり
/// `CommandRunner` 注入が効かない点はモジュール冒頭ドキュメント参照）はいずれも
/// `git diff <baseline_commit> --` 系のコマンドで差分を取得するが、この形式は
/// **未追跡（新規追加）ファイルを出力に含めない**（Codex レビュー #137 指摘）。
/// 候補が新規 `.rs` ファイルを追加した場合、`lines_changed`／`api_broken`／
/// `gaming_suspect` の計測対象から外れ、変更規模・公開 API ガードを迂回しうる
/// （fail-open）。
///
/// `sandbox_root` は `git worktree add --detach` で作られた候補適用専用の
/// 隔離作業木（呼び出し元 `RepairCompositeGate::verify` の契約。
/// `crate::verify_bench_direct` モジュール冒頭ドキュメント参照）であり、実リポの
/// 作業木・index には影響しない。ここで index へ反映しておけば、以降に走る
/// 全ての `git diff <baseline_commit>` 系コマンド（本モジュール内・guardrail
/// 内部呼び出しの双方）が新規ファイルを diff 対象に含めるようになる。
/// staging 失敗（git 未初期化・権限エラー等）は fail-open で無視せず [`Err`]
/// を返す。
fn stage_untracked_files<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
) -> Result<(), DiffSignalsError> {
    let output = runner
        .run("git", &["add", "-A", "--", "."], sandbox_root)
        .map_err(|error| {
            DiffSignalsError::new(format!(
                "git add -A の起動に失敗（未追跡ファイル反映）: {error}"
            ))
        })?;
    if !output.success() {
        return Err(DiffSignalsError::new(format!(
            "git add -A が失敗しました（未追跡ファイル反映）: {}",
            output.log_tail()
        )));
    }
    Ok(())
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

/// パス文字列が「テストコード」を指すかを判定する。`Path::components()` で
/// `tests` コンポーネントの有無を見るため、`tests/foo.rs`（リポジトリ直下）・
/// `crates/self-repair/tests/foo.rs`（中間パス）のいずれの相対パス表現でも
/// 一致する（`path.contains("/tests/")` は先頭が `/` を含まない `tests/foo.rs`
/// を取りこぼすため使わない）。
fn is_test_path(path: &str) -> bool {
    Path::new(path)
        .components()
        .any(|component| component.as_os_str() == "tests")
        || path.ends_with("_test.rs")
}

/// 変更ファイル一覧に「本番コード」と「テストコード」の双方が同時に含まれる
/// かを検査する（ゲーミング疑いの簡易ヒューリスティック）。移植元:
/// `tests/revalidation_bug_fix.rs::gaming_suspect`。`touches_test` /
/// `touches_prod` は `is_test_path` を共通の判定基準として使うため、
/// ルート直下 `tests/foo.rs` のようなパスが誤って両方 `false`（または
/// `touches_prod` 側で誤って `true`）になるブラインドスポットを持たない。
fn gaming_suspect_from_files(changed_files: &[String]) -> bool {
    let touches_test = changed_files.iter().any(|path| is_test_path(path));
    let touches_prod = changed_files
        .iter()
        .any(|path| path.ends_with(".rs") && !is_test_path(path));
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
    // 未追跡（新規追加）ファイルも `git diff <baseline_commit>` の対象に含める
    // ため、以降の全 diff 計測に先立って index へ反映する（`stage_untracked_files`
    // ドキュメント参照。Codex レビュー #137 指摘）。
    stage_untracked_files(runner, sandbox_root)?;
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
    fn gaming_suspect_from_files_true_when_root_level_tests_dir_touched() {
        // リポジトリ直下の `tests/foo.rs`（先頭に `/` が付かない相対パス）は
        // `path.contains("/tests/")` では取りこぼし、かつ `.rs` 拡張子ゆえに
        // `touches_prod` 側で誤って本番コード扱いされていた回帰ケース
        // （PR #355 codex-review 指摘。P1）。
        let files = vec!["src/lib.rs".to_string(), "tests/foo.rs".to_string()];
        assert!(gaming_suspect_from_files(&files));
    }

    #[test]
    fn gaming_suspect_from_files_false_when_only_root_level_tests_dir_touched() {
        let files = vec!["tests/foo.rs".to_string()];
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

    /// `stage_untracked_files` → `diff_numstat`／`api_signature_touched` の
    /// 一連が、実 git リポジトリ上で**未追跡（新規追加）ファイル**を diff
    /// 計測対象に含めることを確認する（Codex レビュー #137 指摘「未追跡
    /// ファイルが差分シグナルから完全に除外される」の再発防止）。
    /// `ScriptedGit` はスクリプト化した固定応答を返すのみで、`git diff
    /// --numstat` が実際に未追跡ファイルを出力しないという実 git の挙動
    /// そのものは検証できないため、`SystemCommandRunner` を使った実 git
    /// リポジトリで検証する。
    #[test]
    fn stage_untracked_files_makes_new_file_visible_to_diff_numstat_and_api_signature_touched() {
        use crate::exec::SystemCommandRunner;

        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-untracked-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(&sandbox).expect("sandbox ディレクトリ作成に失敗");

        let runner = SystemCommandRunner::new();
        let run_ok = |args: &[&str]| {
            let output = runner
                .run("git", args, &sandbox)
                .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗: {error}"));
            assert!(
                output.success(),
                "git {args:?} が失敗しました: {}",
                output.log_tail()
            );
        };

        // baseline commit（追跡ファイルを 1 つ持つ）。
        std::fs::write(sandbox.join("tracked.rs"), "// tracked\n")
            .expect("tracked.rs 書き込み失敗");
        run_ok(&["init", "-q"]);
        run_ok(&["add", "-A"]);
        run_ok(&[
            "-c",
            "user.email=self-repair-137-diff-signals@example.invalid",
            "-c",
            "user.name=self-repair-137-diff-signals",
            "commit",
            "-q",
            "-m",
            "baseline",
        ]);
        let baseline_output = runner
            .run("git", &["rev-parse", "HEAD"], &sandbox)
            .expect("git rev-parse HEAD の起動に失敗");
        assert!(
            baseline_output.success(),
            "git rev-parse HEAD が失敗しました"
        );
        let baseline_commit = baseline_output.log_tail().trim().to_string();

        // 未追跡（新規追加）ファイルを作成する。コミットにもステージングにも
        // 含めない。`pub fn` を含み `api_signature_touched` の検出対象にもする。
        std::fs::create_dir_all(sandbox.join("src")).expect("src ディレクトリ作成に失敗");
        std::fs::write(
            sandbox.join("src/new_api.rs"),
            "pub fn something() -> i32 {\n    1\n}\n",
        )
        .expect("src/new_api.rs 書き込み失敗");

        // stage 前: 実 git の既知の挙動として未追跡ファイルは diff --numstat
        // に現れない（このアサーションはテスト対象コードの前提確認であり、
        // 本テストの主眼は stage 後の挙動）。
        let (lines_changed_before, files_before) =
            diff_numstat(&runner, &sandbox, &baseline_commit).expect("diff_numstat 実行に失敗");
        assert_eq!(
            lines_changed_before, 0,
            "stage 前は未追跡ファイルが diff --numstat に現れないはず（前提確認）"
        );
        assert!(
            files_before.is_empty(),
            "stage 前の変更ファイル一覧は空のはず（前提確認）"
        );

        // `stage_untracked_files` 適用後は新規ファイルが diff 計測対象に入る。
        stage_untracked_files(&runner, &sandbox).expect("stage_untracked_files に失敗");
        let (lines_changed_after, files_after) =
            diff_numstat(&runner, &sandbox, &baseline_commit).expect("diff_numstat 実行に失敗");
        assert!(
            lines_changed_after > 0,
            "stage 後は新規ファイルの追加行が lines_changed に計上されるはず"
        );
        assert!(
            files_after.iter().any(|path| path == "src/new_api.rs"),
            "stage 後は新規ファイルが変更ファイル一覧に含まれるはず: {files_after:?}"
        );
        let api_broken = api_signature_touched(&runner, &sandbox, &baseline_commit)
            .expect("api_signature_touched 実行に失敗");
        assert!(
            api_broken,
            "新規ファイルの pub fn は api_signature_touched で検出されるはず"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }
}
