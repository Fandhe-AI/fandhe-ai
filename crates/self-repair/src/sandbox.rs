//! `self-repair run` の実リポジトリ隔離機構（PR #361 codex-review P0 指摘対応・
//! イシュー #142）。
//!
//! # 背景（何が壊れていたか）
//! `main.rs::run_run` は `--repo` の実リポジトリを
//! [`crate::verify_direct_composite::RepairCompositeGateSpec`] の `workspace`・
//! `sandbox_root` 双方へ**直接**渡していた。しかし [`crate::verify_direct_composite::
//! RepairCompositeGate::verify`] は検証のたび `git add -A`（[`crate::diff_signals::
//! measure_diff_signals`] 経由）を実行し、[`crate::candidate::apply_candidate`]
//! は候補ファイルを sandbox の作業木へ直接上書きする。この 2 つはいずれも
//! 「使い捨ての隔離 sandbox」を前提とした設計（`tests/revalidation_bug_fix.rs`・
//! `tests/feature_addition_loop_completion_task_3_3c.rs` の統合テストが実際に
//! 使い捨て sandbox を経由している）であり、`--repo` に人間の作業リポジトリを
//! そのまま渡すと、非採用（`Rejected`/`Escalated`/試行上限到達）に終わった候補の
//! 変更が未コミットの作業ツリーへ残置され、`git add -A` が無関係な変更まで
//! staged にしてしまう。
//!
//! # 本モジュールの役割
//! [`RunSandbox::create`] は `--repo` を `baseline_commit` の状態で
//! `git clone --local`（`tests/revalidation_bug_fix.rs::create_sandbox`・
//! `tests/feature_addition_loop_completion_task_3_3c.rs::unique_sandbox_dir` と
//! 同じ隔離パターンを `src/` 側へ昇格したもの）した独立 sandbox として構築する。
//! ループ全体（候補適用・4 ゲート検証・`git add -A` を含む）はこの sandbox
//! 内で完結し、`--repo` の作業ツリー・index には一切触れない。
//!
//! `git worktree add` ではなく `git clone --local` を選ぶ理由: `git worktree add`
//! は実リポジトリの `.git/worktrees/<name>` にメタデータを書き込むため、
//! 「非採用・エラー経路では元リポジトリに一切触れない」という要求を
//! （sandbox 作成の時点で既に）満たせない。`git clone --local` は完全に独立した
//! `.git` を作るため、より強い隔離を保証できる。
//!
//! sandbox 先パスは事前削除せず排他作成し（既存パスなら `Err`・内容には触れない）、
//! 作成済みの空ディレクトリへ clone する（イシュー #2388）。後始末は自分が作った
//! ディレクトリだけを対象にする。
//!
//! [`reflect_adopted_diff`] は [`crate::outcome::LoopOutcome::Adopted`] の場合
//! のみ呼ばれ、sandbox の作業木と `baseline_commit` の差分を `--repo` の作業
//! ツリーへ `git apply --check` の競合検査を経て反映する（index へは触れない。
//! `.claude/rules/security.md` A08「判定の迂回経路を作らない」と同種の
//! fail-closed 方針: 反映先がダーティで競合する場合は一切適用しない）。

use std::env;
use std::fs;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// `git` を `cwd` で起動するコマンドを構築する。継承されうる `GIT_*` 環境変数
/// （`GIT_DIR`／`GIT_WORK_TREE`／`GIT_INDEX_FILE` 等）を明示的に除去してから
/// 起動する。
///
/// `current_dir(cwd)` だけでは sandbox 隔離を保証できない。`lefthook.yml` の
/// `pre-push.jobs.test` 等 githooks(5) 経由で本バイナリが起動されるケースを
/// 含め、git はフック起動時に `GIT_*` 環境変数を子プロセスへ設定し、それが
/// `Command::new("git")` まで継承されうる。継承された `GIT_DIR` は
/// `current_dir` より優先されるため、これを除去しないと sandbox 内のつもりの
/// `git clone`／`git add`／`git diff` が実リポジトリの `.git` を対象にしうる
/// （`tests/feature_addition_loop_completion_task_3_3c.rs::sandboxed_git_command`・
/// `main.rs::resolve_baseline_commit` と同一の事故パターン・同一の対処）。
fn git_command(cwd: &Path, args: &[&str]) -> Command {
    let mut command = Command::new("git");
    command.args(args).current_dir(cwd);
    for (key, _) in env::vars_os() {
        if let Some(key_str) = key.to_str()
            && key_str.starts_with("GIT_")
        {
            command.env_remove(key_str);
        }
    }
    command
}

/// [`git_command`] を実行し、標準出力（バイト列。`git diff --binary` の
/// バイナリ差分を UTF-8 変換せずそのまま扱うため）を返す。非 0 終了は
/// エラーメッセージへ変換する（`main.rs` の内部エラー区分〈exit 1〉が
/// そのまま stderr へ出力する前提の平文メッセージ）。
fn run_git(cwd: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = git_command(cwd, args).output().map_err(|error| {
        format!(
            "git {args:?} の起動に失敗しました（cwd={}）: {error}",
            cwd.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?} が失敗しました（cwd={}, exit={:?}）: {}",
            cwd.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

/// プロセス内で単調増加する sandbox 名の連番（同一プロセス・同一ナノ秒でも
/// 名前が衝突しないようにする）。
static SANDBOX_SEQ: AtomicU64 = AtomicU64::new(0);

/// 通常経路（`RunSandbox::create`）で `AlreadyExists` 時に新しい名前で再試行する上限回数。
const MAX_SANDBOX_CREATE_ATTEMPTS: usize = 8;

/// `env::temp_dir()` 配下に、プロセス ID・ナノ秒タイムスタンプ・プロセス内連番
/// で一意化した sandbox パス候補を生成する。名前は予測可能でありうるため、
/// 一意性だけに頼らず、呼び出し側が排他作成（`create_sandbox_dir`）と
/// `AlreadyExists` 時の再試行を組み合わせる前提である。
fn unique_sandbox_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let seq = SANDBOX_SEQ.fetch_add(1, Ordering::Relaxed);
    env::temp_dir().join(format!(
        "self-repair-run-sandbox-{}-{nanos}-{seq}",
        std::process::id()
    ))
}

/// `self-repair run` ループ全体を隔離して実行するための使い捨て sandbox。
/// [`RunSandbox::create`] が `--repo` を `baseline_commit` の状態で clone し、
/// `root()` を [`crate::verify_direct_composite::RepairCompositeGateSpec`] の
/// `workspace`／`sandbox_root`、検出器・修正生成器の workspace として使う。
///
/// `Drop` で自プロセスが作成した sandbox ディレクトリのみを削除する
/// （[`RunSandbox::keep`] を呼んだ場合を除く。反映失敗時に調査のため sandbox
/// を残す用途）。
pub struct RunSandbox {
    root: PathBuf,
    keep: bool,
}

/// `root` を排他的に作成する（既存のディレクトリ・ファイル・シンボリックリンク
/// 〈dangling を含む〉はすべて `AlreadyExists`）。Unix では他ユーザーから
/// sandbox（`--repo` のソース一式と候補差分を含む）を読ませないため `0o700`
/// で作る（umask で更に絞られうる）。親ディレクトリは作らない（非再帰）。
fn create_exclusive_dir(root: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        fs::DirBuilder::new().mode(0o700).create(root)
    }
    #[cfg(not(unix))]
    {
        fs::create_dir(root)
    }
}

/// `root` を排他作成し、成功した直後に cleanup guard を兼ねる `RunSandbox` を
/// 構築する。作成に失敗した場合は何も構築せず何も削除しない（他者のパスを
/// 消さない）。
fn create_sandbox_dir(root: PathBuf) -> io::Result<RunSandbox> {
    create_exclusive_dir(&root)?;
    Ok(RunSandbox { root, keep: false })
}

/// `--repo` を正規化し UTF-8 文字列として返す。ディレクトリ作成より前に
/// 行うため、ここでの失敗は後始末を要さない。
fn resolve_repo(repo: &Path) -> Result<String, String> {
    let repo_abs = fs::canonicalize(repo).map_err(|error| {
        format!(
            "--repo の解決に失敗しました（repo={}）: {error}",
            repo.display()
        )
    })?;
    repo_abs
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| "--repo のパスが UTF-8 ではありません".to_string())
}

/// 作成済みの空 sandbox ディレクトリへ `git clone` し、`baseline_commit` へ
/// detached checkout する。`?` で早期 return しても `sandbox` の `Drop` が
/// 自分の作った `root` のみを削除する。
fn initialize(
    sandbox: RunSandbox,
    repo_str: &str,
    baseline_commit: &str,
) -> Result<RunSandbox, String> {
    let root_str = sandbox
        .root
        .to_str()
        .ok_or_else(|| "sandbox パスが UTF-8 ではありません".to_string())?
        .to_string();
    run_git(
        Path::new("."),
        &[
            "clone",
            "--local",
            "--no-hardlinks",
            "--quiet",
            repo_str,
            &root_str,
        ],
    )?;
    run_git(
        sandbox.root(),
        &["checkout", "--quiet", "--detach", baseline_commit],
    )?;
    Ok(sandbox)
}

/// パス注入版の単発作成（再試行なし。通常経路は `create_with_candidates`）。テスト専用。`root`（sandbox 先パス）を
/// 呼び出し元から注入できる形にしたのは、テストで既知のパスを使い、初期化
/// 失敗時に「そのパスが削除されているか」を決定的に検証するため
/// （`tests` モジュール `create_removes_sandbox_directory_when_initialization_fails_after_clone`
/// 参照）。
///
/// # 契約（イシュー #2388）
/// - 事前削除はしない。`root` を排他作成し、既存パス（ディレクトリ・ファイル・
///   シンボリックリンク）なら再試行せず、既存の内容に触れずに `Err` を返す。
///   `git clone` は空の既存宛先を受け付けるため、事前削除は不要である。
/// - `create_dir` 成功直後に cleanup guard（`RunSandbox`）を構築する。
///   以降 clone・checkout のどちらが失敗しても、`Drop` が自分の作った
///   ディレクトリだけを削除する（PR #361 codex-review 第 3 波 P2 指摘の
///   「clone 済み sandbox の残置」防止を、構築位置を前倒しして維持）。
///   作成自体に失敗した経路では何も消さない。
#[cfg(test)]
fn create_at(root: PathBuf, repo: &Path, baseline_commit: &str) -> Result<RunSandbox, String> {
    let repo_str = resolve_repo(repo)?;
    if root.to_str().is_none() {
        return Err("sandbox パスが UTF-8 ではありません".to_string());
    }
    let sandbox = create_sandbox_dir(root.clone()).map_err(|error| {
        format!(
            "sandbox ディレクトリの排他作成に失敗しました（既存のパスには触れていません。path={}）: {error}",
            root.display()
        )
    })?;
    initialize(sandbox, &repo_str, baseline_commit)
}

/// 候補パスを順に排他作成し、最初に成功したものを sandbox として初期化する。
/// `AlreadyExists` のときだけ次の候補へ進み（既存パスには触れない）、それ以外の
/// I/O エラーは再試行せず即座に `Err` とする。候補を使い切った場合も `Err`。
/// `RunSandbox::create`（候補は `unique_sandbox_path()` の有限個）から呼ばれる。
fn create_with_candidates(
    candidates: impl IntoIterator<Item = PathBuf>,
    repo: &Path,
    baseline_commit: &str,
) -> Result<RunSandbox, String> {
    let repo_str = resolve_repo(repo)?;
    let mut attempts = 0usize;
    for root in candidates {
        attempts += 1;
        if root.to_str().is_none() {
            return Err("sandbox パスが UTF-8 ではありません".to_string());
        }
        match create_sandbox_dir(root.clone()) {
            Ok(sandbox) => return initialize(sandbox, &repo_str, baseline_commit),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "sandbox ディレクトリの作成に失敗しました（path={}）: {error}",
                    root.display()
                ));
            }
        }
    }
    Err(format!(
        "一意な sandbox パスを確保できませんでした（{attempts} 回衝突）"
    ))
}

impl RunSandbox {
    /// `repo` を `baseline_commit` の状態で `git clone --local --no-hardlinks`
    /// した独立 sandbox を構築する。`--no-hardlinks` は `env::temp_dir()` と
    /// `repo` が別ファイルシステム（別マウント）上にある環境（CI・コンテナ・
    /// worktree がバインドマウント上にある環境）で既定のハードリンク複製が
    /// `Invalid cross-device link` で失敗するのを避けるため常にファイルコピー
    /// へフォールバックする（`tests/revalidation_bug_fix.rs::create_sandbox`
    /// と同じ理由）。clone 直後に `baseline_commit` へ明示的に detached
    /// checkout し直すのは、`repo` が現在別ブランチ・別コミットを指している
    /// 場合でも sandbox が必ず `baseline_commit` の内容と一致することを保証
    /// するため（`git clone` の既定挙動〈`repo` の HEAD が指す先〉に依存
    /// しない）。
    ///
    /// sandbox 先は事前削除せず排他作成する（Unix では `0o700`）。名前が既存
    /// パスと衝突した場合は新しい名前で最大 8 回まで再試行し、既存パスには
    /// 触れない。
    pub fn create(repo: &Path, baseline_commit: &str) -> Result<Self, String> {
        create_with_candidates(
            (0..MAX_SANDBOX_CREATE_ATTEMPTS).map(|_| unique_sandbox_path()),
            repo,
            baseline_commit,
        )
    }

    /// sandbox のルートパス（`RepairCompositeGateSpec::workspace`／
    /// `sandbox_root`、検出器・修正生成器の workspace に使う）。
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `Drop` 時の自動削除を抑止する（反映失敗〈[`reflect_adopted_diff`] の
    /// エラー〉時に、調査のため sandbox を残す用途）。
    pub fn keep(&mut self) {
        self.keep = true;
    }
}

impl Drop for RunSandbox {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

/// [`crate::outcome::LoopOutcome::Adopted`] の場合のみ呼ばれる: sandbox の
/// 作業木と `baseline_commit` の差分を `repo`（`--repo` の実リポジトリ）の
/// 作業ツリーへ反映する。
///
/// # 決定的な差分生成
/// `crate::verify_direct_composite::RepairCompositeGate::verify`（→
/// [`crate::diff_signals::measure_diff_signals`]）は検証のたび sandbox 内で
/// `git add -A` を実行するため、ループ終了時点の sandbox の index は「最後の
/// 試行がどのゲートで止まったか」に応じて staged 状態・未 staged 状態のいずれ
/// もありうる。`git diff <baseline_commit>`（index を経由しない作業木直接比較）
/// だけでは新規ファイルの扱いが経路依存になりうるため、反映前に sandbox 内で
/// 明示的に `git add -A` してから `git diff --cached` を取ることで、経路に
/// 依らず完全かつ決定的な patch を得る（`--repo` の index には触れない。
/// `git add -A` は sandbox 内で完結する）。
///
/// # 反映は競合検査つき（fail-closed）
/// `git apply --check`（`--index` を付けない。`repo` の index には触れず作業木
/// のみを対象にする）で反映可否を先に検査し、失敗時は `repo` の作業ツリーへ
/// 一切触れずにエラーを返す（呼び出し元 `main.rs` が sandbox のパスを
/// エラーメッセージに含めて調査可能にする）。検査を通過した場合のみ実際に
/// 適用する。
///
/// # 空 diff は `Err` を返す（fail-closed。PR #361 codex-review 第 4 波 P1 指摘）
/// `LoopOutcome::Adopted` は「baseline との非空差分が存在する」ことを前提とする
/// 状態のはずだが、その前提はこの関数の外側（採用判定・検証ゲート）では型で
/// 保証されていない。過去の実装は空 diff を「反映不要」として `Ok(())` を返して
/// いたが、これは「反映すべき差分がない」ケースと「反映に成功した」ケースを
/// 呼び出し元から区別不能にし、`--repo` を一切変更しないまま exit 0（修復成功）
/// を報告しうる欠陥だった。本実装は空 diff を契約違反として明示的に `Err` を
/// 返し、`main.rs::run_run` が内部エラー区分（exit 1）へ写像する。
pub fn reflect_adopted_diff(
    repo: &Path,
    sandbox: &Path,
    baseline_commit: &str,
) -> Result<(), String> {
    run_git(sandbox, &["add", "--all"])?;
    let patch = run_git(
        sandbox,
        &[
            "diff",
            "--binary",
            "--no-ext-diff",
            "--cached",
            baseline_commit,
        ],
    )?;
    if patch.is_empty() {
        // 空 diff は契約違反として明示的に失敗させる（fail-closed）。
        //
        // `LoopOutcome::Adopted` は「採用された変更が存在する」ことを前提とする
        // 状態のはずだが、その前提はこの関数の外側（採用判定・検証ゲート）では
        // 型で保証されていない。ここで `Ok(())` を返すと、呼び出し元
        // `main.rs::run_run` は「反映すべき差分がない」ことと「反映に成功した」
        // ことを区別できず、`--repo` を一切変更しないまま exit 0（修復成功）を
        // 報告してしまう（PR #361 codex-review 第 4 波 P1 指摘）。
        // それを防ぐため、空 diff の Adopted は呼び出し元がエラーとして検知
        // できるよう明示的に `Err` を返す。
        return Err(
            "反映対象の差分が空です（Adopted 候補は baseline と非空の差分を持つ前提が破れて\
             います）。--repo は変更していません"
                .to_string(),
        );
    }
    apply_patch(repo, &patch, true)?;
    apply_patch(repo, &patch, false)
}

/// `git apply`（`check_only` 時は `--check`）を `patch`（stdin 経由）に対して
/// 実行する。`--index` を付けないため `repo` の index には触れず作業ツリーの
/// みを変更する（`reflect_adopted_diff` モジュール冒頭ドキュメント参照）。
fn apply_patch(repo: &Path, patch: &[u8], check_only: bool) -> Result<(), String> {
    let mut args: Vec<&str> = vec!["apply", "--binary"];
    if check_only {
        args.push("--check");
    }
    args.push("-");

    let mut command = git_command(repo, &args);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("git apply の起動に失敗しました: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "git apply の標準入力を取得できませんでした".to_string())?
        .write_all(patch)
        .map_err(|error| format!("git apply への patch 書き込みに失敗しました: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("git apply の完了待機に失敗しました: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git apply{}が失敗しました（repo={}）: {}",
            if check_only { " --check" } else { "" },
            repo.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    /// テスト専用の隔離 git リポジトリを構築する（`main.rs::resolve_baseline_commit`
    /// と同じ `GIT_*` 除去方式）。`init` の初期ブランチ名を明示指定し、
    /// 環境の `init.defaultBranch` 設定に依存しないようにする。
    /// `dir` は呼び出し元が `unique_temp_dir` で作成済みの空ディレクトリとする。
    fn init_repo(dir: &Path) {
        for args in [
            vec!["init", "--quiet", "--initial-branch=main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
        ] {
            let status = git_command(dir, &args)
                .status()
                .expect("git コマンドの起動に失敗");
            assert!(status.success(), "git {args:?} が失敗しました");
        }
    }

    fn git_commit_all(dir: &Path, message: &str) {
        let status = git_command(dir, &["add", "--all"])
            .status()
            .expect("git add の起動に失敗");
        assert!(status.success());
        let status = git_command(dir, &["commit", "--quiet", "-m", message])
            .status()
            .expect("git commit の起動に失敗");
        assert!(status.success());
    }

    fn head_commit(dir: &Path) -> String {
        String::from_utf8(run_git(dir, &["rev-parse", "HEAD"]).expect("HEAD 解決に失敗"))
            .expect("HEAD sha は UTF-8 のはず")
            .trim()
            .to_string()
    }

    fn init_baseline_repo(name: &str) -> (crate::test_support::TempDirGuard, String) {
        let guard = unique_temp_dir(name);
        let repo = guard.path().to_path_buf();
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");
        let baseline = head_commit(&repo);
        (guard, baseline)
    }

    fn dir_entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read_dir に失敗")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn create_at_rejects_existing_directory_without_touching_contents() {
        let (repo, baseline) = init_baseline_repo("excl-dir-src");
        let parent = unique_temp_dir("excl-dir");
        let existing = parent.path().join("existing");
        fs::create_dir(&existing).expect("mkdir");
        fs::write(existing.join("marker.txt"), "keep\n").expect("marker");
        let before = dir_entries(&existing);

        let result = create_at(existing.clone(), repo.path(), &baseline);
        assert!(result.is_err(), "既存ディレクトリは拒否されるはず");
        assert_eq!(
            fs::read_to_string(existing.join("marker.txt")).expect("marker 読み取り"),
            "keep\n"
        );
        assert_eq!(dir_entries(&existing), before);
    }

    #[cfg(unix)]
    #[test]
    fn create_at_rejects_existing_symlink_without_touching_target() {
        let (repo, baseline) = init_baseline_repo("excl-link-src");
        let parent = unique_temp_dir("excl-link");
        let target = parent.path().join("target");
        fs::create_dir(&target).expect("mkdir");
        fs::write(target.join("marker.txt"), "keep\n").expect("marker");
        let link = parent.path().join("link");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        let result = create_at(link.clone(), repo.path(), &baseline);
        assert!(result.is_err(), "既存 symlink は拒否されるはず");
        assert!(
            fs::symlink_metadata(&link)
                .expect("link metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_to_string(target.join("marker.txt")).expect("marker 読み取り"),
            "keep\n"
        );
        assert!(!target.join(".git").exists());
        assert_eq!(dir_entries(&target), vec!["marker.txt".to_string()]);
    }

    #[cfg(unix)]
    #[test]
    fn create_at_rejects_dangling_symlink() {
        let (repo, baseline) = init_baseline_repo("excl-dangling-src");
        let parent = unique_temp_dir("excl-dangling");
        let missing = parent.path().join("missing-target");
        let link = parent.path().join("link");
        std::os::unix::fs::symlink(&missing, &link).expect("symlink");

        let result = create_at(link.clone(), repo.path(), &baseline);
        assert!(result.is_err(), "dangling symlink は拒否されるはず");
        assert!(!missing.exists(), "リンク先パスが作られてはならない");
        assert!(
            fs::symlink_metadata(&link)
                .expect("link metadata")
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn create_at_rejects_existing_file() {
        let (repo, baseline) = init_baseline_repo("excl-file-src");
        let parent = unique_temp_dir("excl-file");
        let file = parent.path().join("f");
        fs::write(&file, "keep\n").expect("write");
        let result = create_at(file.clone(), repo.path(), &baseline);
        assert!(result.is_err());
        assert_eq!(fs::read_to_string(&file).expect("read"), "keep\n");
    }

    #[test]
    fn create_at_clones_into_freshly_created_empty_directory() {
        let (repo, baseline) = init_baseline_repo("excl-fresh-src");
        let parent = unique_temp_dir("excl-fresh");
        let root = parent.path().join("sandbox");

        let sandbox = create_at(root.clone(), repo.path(), &baseline)
            .expect("未作成パスへの create_at は成功するはず");
        assert_eq!(
            fs::read_to_string(root.join("a.txt")).expect("a.txt"),
            "baseline\n"
        );
        assert!(root.join(".git").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&root).expect("metadata").permissions().mode();
            assert_eq!(mode & 0o077, 0, "group/other に権限があってはならない");
        }
        drop(sandbox);
        assert!(!root.exists());
    }

    #[test]
    fn create_with_candidates_skips_existing_candidate_untouched() {
        let (repo, baseline) = init_baseline_repo("cand-skip-src");
        let parent = unique_temp_dir("cand-skip");
        let first = parent.path().join("first");
        fs::create_dir(&first).expect("mkdir");
        fs::write(first.join("marker.txt"), "keep\n").expect("marker");
        let second = parent.path().join("second");

        let sandbox =
            create_with_candidates(vec![first.clone(), second.clone()], repo.path(), &baseline)
                .expect("2 番目の候補で成功するはず");
        assert_eq!(sandbox.root(), second.as_path());
        assert_eq!(dir_entries(&first), vec!["marker.txt".to_string()]);
        assert_eq!(
            fs::read_to_string(first.join("marker.txt")).expect("marker"),
            "keep\n"
        );
    }

    #[test]
    fn create_with_candidates_errors_when_all_candidates_exist() {
        let (repo, baseline) = init_baseline_repo("cand-all-src");
        let parent = unique_temp_dir("cand-all");
        let mut candidates = Vec::new();
        for i in 0..3 {
            let dir = parent.path().join(format!("c{i}"));
            fs::create_dir(&dir).expect("mkdir");
            fs::write(dir.join("marker.txt"), "keep\n").expect("marker");
            candidates.push(dir);
        }
        let result = create_with_candidates(candidates.clone(), repo.path(), &baseline);
        assert!(result.is_err());
        for dir in candidates {
            assert_eq!(dir_entries(&dir), vec!["marker.txt".to_string()]);
        }
    }

    #[test]
    fn create_generates_distinct_roots_for_consecutive_calls() {
        let (repo, baseline) = init_baseline_repo("distinct-src");
        let first = RunSandbox::create(repo.path(), &baseline).expect("create 1");
        let second = RunSandbox::create(repo.path(), &baseline).expect("create 2");
        assert_ne!(first.root(), second.root());
    }

    /// P2 回帰防止（PR #361 codex-review 第 3 波指摘）: `git clone` 成功後に
    /// `git checkout --detach` が失敗した場合でも、clone 済みの sandbox
    /// ディレクトリが残置されないことを確認する。
    ///
    /// `RunSandbox::create` は内部で `unique_sandbox_path()`（PID・ナノ秒
    /// タイムスタンプ由来）を使うため決定的なパス検証ができない。本テストは
    /// `create_at` を直接呼び、既知の `root` パスを注入することで
    /// 「初期化失敗後にそのパスが確実に存在しない」ことを決定的に検証する
    /// （`create_at` doc コメント参照）。
    #[test]
    fn create_removes_sandbox_directory_when_initialization_fails_after_clone() {
        let repo_guard = unique_temp_dir("create-checkout-fails-source");
        let repo = repo_guard.path().to_path_buf();
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");

        // `create_at` は既存パスを拒否する（排他作成）ため、ガード配下の未作成の子パスを使う。
        let root_parent = unique_temp_dir("sandbox-create-checkout-fails");
        let root = root_parent.path().join("sandbox");
        // 存在しない commit sha を渡し、clone 成功後の `git checkout --detach`
        // を確実に失敗させる（40 桁の 16 進数だが実在しないオブジェクト）。
        let bogus_baseline_commit = "0".repeat(40);

        let result = create_at(root.clone(), &repo, &bogus_baseline_commit);
        assert!(
            result.is_err(),
            "存在しない baseline commit への checkout は失敗するはず"
        );
        assert!(
            !root.exists(),
            "checkout 失敗時は clone 済みの sandbox ディレクトリが残置されてはならない: {}",
            root.display()
        );
    }

    /// P0 不変条件 (a): `RunSandbox::create` は `--repo` の作業ツリー・index に
    /// 一切触れない（未コミット変更のある `--repo` で構築しても状態が不変）。
    #[test]
    fn create_does_not_touch_source_repo_working_tree_or_index() {
        let repo_guard = unique_temp_dir("create-source");
        let repo = repo_guard.path().to_path_buf();
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");
        let baseline = head_commit(&repo);

        // 未コミットの変更（作業ツリー・index の両方）を作る。
        fs::write(repo.join("a.txt"), "dirty-working-tree\n").expect("a.txt 上書きに失敗");
        fs::write(repo.join("b.txt"), "staged-new-file\n").expect("b.txt 書き込みに失敗");
        let status = git_command(&repo, &["add", "b.txt"])
            .status()
            .expect("git add の起動に失敗");
        assert!(status.success());

        let status_before =
            run_git(&repo, &["status", "--porcelain"]).expect("git status の取得に失敗");

        let mut sandbox = RunSandbox::create(&repo, &baseline).expect("RunSandbox::create に失敗");

        let status_after =
            run_git(&repo, &["status", "--porcelain"]).expect("git status の取得に失敗");
        assert_eq!(
            status_before, status_after,
            "RunSandbox::create の前後で --repo の作業ツリー・index が変化してはならない"
        );
        assert_eq!(
            head_commit(&repo),
            baseline,
            "RunSandbox::create は --repo の HEAD を進めてはならない"
        );

        // sandbox 側は baseline commit の内容（dirty な変更を含まない）。
        let sandboxed_content = fs::read_to_string(sandbox.root().join("a.txt"))
            .expect("sandbox の a.txt 読み取りに失敗");
        assert_eq!(
            sandboxed_content, "baseline\n",
            "sandbox は --repo の未コミット変更を含まず baseline commit の内容のはず"
        );

        sandbox.keep();
        let _ = fs::remove_dir_all(sandbox.root());
    }

    /// P0 不変条件 (b): `reflect_adopted_diff` は clean な `--repo` へ差分のみを
    /// 適用し、`--repo` の index は変化しない（`git add -A` が実リポの index を
    /// 汚さないことの確認。sandbox 内の `git add -A` は sandbox 専用）。
    #[test]
    fn reflect_adopted_diff_applies_only_working_tree_changes_without_staging() {
        let repo_guard = unique_temp_dir("reflect-clean");
        let repo = repo_guard.path().to_path_buf();
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");
        let baseline = head_commit(&repo);

        let mut sandbox = RunSandbox::create(&repo, &baseline).expect("RunSandbox::create に失敗");
        fs::write(sandbox.root().join("a.txt"), "adopted-change\n")
            .expect("sandbox の a.txt 上書きに失敗");

        reflect_adopted_diff(&repo, sandbox.root(), &baseline)
            .expect("clean な --repo への反映は成功するはず");

        let reflected =
            fs::read_to_string(repo.join("a.txt")).expect("--repo の a.txt 読み取りに失敗");
        assert_eq!(
            reflected, "adopted-change\n",
            "採用された差分が --repo の作業ツリーへ反映されているはず"
        );
        let index_status = run_git(&repo, &["diff", "--cached", "--name-only"])
            .expect("git diff --cached の取得に失敗");
        assert!(
            index_status.is_empty(),
            "反映は作業ツリーのみを変更し index は空のままのはず: {}",
            String::from_utf8_lossy(&index_status)
        );

        sandbox.keep();
        let _ = fs::remove_dir_all(sandbox.root());
    }

    /// P0 不変条件 (c): 反映先（`--repo`）が競合する形でダーティな場合、
    /// `reflect_adopted_diff` は適用せずエラーを返し、`--repo` の作業ツリーは
    /// 不変のまま（`git apply --check` の fail-closed 検査）。
    #[test]
    fn reflect_adopted_diff_rejects_conflicting_dirty_repo_without_touching_it() {
        let repo_guard = unique_temp_dir("reflect-conflict");
        let repo = repo_guard.path().to_path_buf();
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");
        let baseline = head_commit(&repo);

        let mut sandbox = RunSandbox::create(&repo, &baseline).expect("RunSandbox::create に失敗");
        // sandbox 側では baseline の行を書き換える。
        fs::write(sandbox.root().join("a.txt"), "adopted-change\n")
            .expect("sandbox の a.txt 上書きに失敗");

        // --repo 側は同じ行を別内容へ書き換えた未コミット変更（競合するダーティ
        // 状態）を持つ。
        fs::write(repo.join("a.txt"), "conflicting-local-edit\n")
            .expect("--repo の a.txt 上書きに失敗");
        let dirty_before =
            fs::read_to_string(repo.join("a.txt")).expect("--repo の a.txt 読み取りに失敗");

        let result = reflect_adopted_diff(&repo, sandbox.root(), &baseline);
        assert!(
            result.is_err(),
            "競合するダーティな --repo への反映は失敗するはず"
        );

        let dirty_after =
            fs::read_to_string(repo.join("a.txt")).expect("--repo の a.txt 読み取りに失敗");
        assert_eq!(
            dirty_before, dirty_after,
            "反映失敗時は --repo の作業ツリーが一切変化してはならない"
        );

        sandbox.keep();
        let _ = fs::remove_dir_all(sandbox.root());
    }

    /// P1 回帰防止（PR #361 codex-review 第 4 波指摘）: sandbox の作業木が
    /// `baseline_commit` と同一（空 diff）の場合、`reflect_adopted_diff` は
    /// `Ok(())` を返さず fail-closed に `Err` を返す。空 diff の `Adopted` を
    /// 「反映不要の成功」として扱うと、`--repo` を一切変更しないまま
    /// exit 0（修復成功）を報告できてしまうため（`main.rs::run_run` は
    /// この `Err` を内部エラー区分 exit 1 へ写像する）。
    #[test]
    fn reflect_adopted_diff_rejects_empty_diff_instead_of_reporting_success() {
        let repo_guard = unique_temp_dir("reflect-empty-diff-source");
        let repo = repo_guard.path().to_path_buf();
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");
        let baseline = head_commit(&repo);

        // sandbox は作成直後（baseline と同一内容）のまま何も変更しない。
        let mut sandbox = RunSandbox::create(&repo, &baseline).expect("RunSandbox::create に失敗");

        let before =
            fs::read_to_string(repo.join("a.txt")).expect("--repo の a.txt 読み取りに失敗");

        let result = reflect_adopted_diff(&repo, sandbox.root(), &baseline);
        assert!(
            result.is_err(),
            "sandbox と baseline が同一（空 diff）の場合は Err を返すはず"
        );

        let after = fs::read_to_string(repo.join("a.txt")).expect("--repo の a.txt 読み取りに失敗");
        assert_eq!(
            before, after,
            "空 diff のエラー経路では --repo の作業ツリーが一切変化してはならない"
        );

        sandbox.keep();
        let _ = fs::remove_dir_all(sandbox.root());
    }
}
