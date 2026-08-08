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
//! [`reflect_adopted_diff`] は [`crate::outcome::LoopOutcome::Adopted`] の場合
//! のみ呼ばれ、sandbox の作業木と `baseline_commit` の差分を `--repo` の作業
//! ツリーへ `git apply --check` の競合検査を経て反映する（index へは触れない。
//! `.claude/rules/security.md` A08「判定の迂回経路を作らない」と同種の
//! fail-closed 方針: 反映先がダーティで競合する場合は一切適用しない）。

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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

/// `env::temp_dir()` 配下に、プロセス ID とナノ秒タイムスタンプで一意化した
/// sandbox パスを生成する（同一プロセス内で `self-repair run` 相当の処理を
/// 連続実行しても衝突しないよう、PID のみに依存した `tests/` 側の簡易方式
/// より強い一意性を持たせる。本モジュールは本番経路〈`src/`〉であり、
/// テスト専用ヘルパーより衝突耐性を優先する）。
fn unique_sandbox_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    env::temp_dir().join(format!(
        "self-repair-run-sandbox-{}-{nanos}",
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

/// [`RunSandbox::create`] の本体。`root`（sandbox 先パス）を呼び出し元から
/// 注入できる形にしたのは、テストで `unique_sandbox_path()`（PID・ナノ秒
/// タイムスタンプ由来で決定不能）ではなく既知のパスを使い、初期化失敗時に
/// 「そのパスが削除されているか」を決定的に検証するため
/// （`tests` モジュール `create_removes_sandbox_directory_when_initialization_fails_after_clone`
/// 参照）。
///
/// # clone 成功後の初期化失敗で一時ディレクトリが残置される問題（P2）
/// 旧実装は `git clone` → `git checkout --detach` の両方が成功したあとで
/// はじめて `RunSandbox { root, keep: false }` を構築していた。そのため
/// `checkout` が失敗すると `RunSandbox`（`Drop` で `root` を削除する唯一の
/// 主体）が一度も存在せず、clone 済みの sandbox ディレクトリが `Err` 経路で
/// 残置されていた（PR #361 codex-review 第 3 波 P2 指摘）。
///
/// 本実装は `clone` 成功直後に `RunSandbox { root, keep: false }` を構築し、
/// 以降の初期化ステップ（`checkout`）は構築済みの `sandbox`（cleanup guard
/// を兼ねる）に対して行う。`checkout` が失敗して `?` で早期 return する際は
/// ローカル変数 `sandbox` が関数末尾でドロップされ、`Drop for RunSandbox` が
/// `root` を削除する。`RunSandbox` 自体が cleanup guard であるため、専用の
/// 別型は導入しない。
fn create_at(root: PathBuf, repo: &Path, baseline_commit: &str) -> Result<RunSandbox, String> {
    let repo_abs = fs::canonicalize(repo).map_err(|error| {
        format!(
            "--repo の解決に失敗しました（repo={}）: {error}",
            repo.display()
        )
    })?;
    let repo_str = repo_abs
        .to_str()
        .ok_or_else(|| "--repo のパスが UTF-8 ではありません".to_string())?;

    // 同一パスが前回実行の残骸として残っていないことを保証してから clone
    // する（`git clone` は既存の空でない宛先ディレクトリを拒否するため）。
    let _ = fs::remove_dir_all(&root);
    let root_str = root
        .to_str()
        .ok_or_else(|| "sandbox パスが UTF-8 ではありません".to_string())?;

    run_git(
        Path::new("."),
        &[
            "clone",
            "--local",
            "--no-hardlinks",
            "--quiet",
            repo_str,
            root_str,
        ],
    )?;

    // clone 成功直後に `RunSandbox` を構築する（上記ドキュメント参照）。
    // 以降 `?` で早期 return しても `sandbox` の `Drop` が `root` を削除する。
    let sandbox = RunSandbox { root, keep: false };
    run_git(
        sandbox.root(),
        &["checkout", "--quiet", "--detach", baseline_commit],
    )?;

    Ok(sandbox)
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
    pub fn create(repo: &Path, baseline_commit: &str) -> Result<Self, String> {
        create_at(unique_sandbox_path(), repo, baseline_commit)
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
/// [`crate::verify_direct_composite::RepairCompositeGate::verify`]（→
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
        // 採用された候補が baseline と差分を持たない（理論上は
        // `RepairCompositeGate` の 4 シグナル実測・build/test/clippy が
        // すべて空 diff で通過することはないはずだが、念のため反映不要として
        // fail-closed に早期 return する。`repo` には一切触れない）。
        return Ok(());
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

    /// テスト専用の隔離 git リポジトリを構築する（`main.rs::resolve_baseline_commit`
    /// と同じ `GIT_*` 除去方式）。`init` の初期ブランチ名を明示指定し、
    /// 環境の `init.defaultBranch` 設定に依存しないようにする。
    fn init_repo(dir: &Path) {
        fs::create_dir_all(dir).expect("repo ディレクトリ作成に失敗");
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

    fn unique_test_dir(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "self-repair-sandbox-test-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
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
        let repo = unique_test_dir("create-checkout-fails-source");
        init_repo(&repo);
        fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
        git_commit_all(&repo, "baseline commit");

        let root = unique_test_dir("create-checkout-fails-sandbox");
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

        let _ = fs::remove_dir_all(&repo);
        let _ = fs::remove_dir_all(&root);
    }

    /// P0 不変条件 (a): `RunSandbox::create` は `--repo` の作業ツリー・index に
    /// 一切触れない（未コミット変更のある `--repo` で構築しても状態が不変）。
    #[test]
    fn create_does_not_touch_source_repo_working_tree_or_index() {
        let repo = unique_test_dir("create-source");
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
        let _ = fs::remove_dir_all(&repo);
    }

    /// P0 不変条件 (b): `reflect_adopted_diff` は clean な `--repo` へ差分のみを
    /// 適用し、`--repo` の index は変化しない（`git add -A` が実リポの index を
    /// 汚さないことの確認。sandbox 内の `git add -A` は sandbox 専用）。
    #[test]
    fn reflect_adopted_diff_applies_only_working_tree_changes_without_staging() {
        let repo = unique_test_dir("reflect-clean");
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
        let _ = fs::remove_dir_all(&repo);
    }

    /// P0 不変条件 (c): 反映先（`--repo`）が競合する形でダーティな場合、
    /// `reflect_adopted_diff` は適用せずエラーを返し、`--repo` の作業ツリーは
    /// 不変のまま（`git apply --check` の fail-closed 検査）。
    #[test]
    fn reflect_adopted_diff_rejects_conflicting_dirty_repo_without_touching_it() {
        let repo = unique_test_dir("reflect-conflict");
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
        let _ = fs::remove_dir_all(&repo);
    }
}
