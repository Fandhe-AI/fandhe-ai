//! TASK-4.2a（イシュー #109）受け入れ条件の機械検証:
//! `tests/fixtures/labeled-changes/` の 15 件以上の変更セットが、
//! 新実装コードベース（`tests/fixtures/labeled-changes/baseline`。v2 自作
//! コア上に再構築したミニ MLP ワークロード）へ適用可能であることを検証する。
//!
//! **std-only の理由**: TASK-4.1（guardrail CLI 移植・#103）が
//! `crates/guardrail/src/lib.rs`・`Cargo.toml` を並行して実装中のため、
//! 本イシューではこれらのファイルに一切触れない（`delegation-impl.md`
//! 「複数 Agent に同一ファイルを並行編集させない」）。本テストは
//! `guardrail` の依存に serde/toml 等を追加せず、`std::fs`／
//! `std::process::Command`（git CLI 呼び出し）のみで完結させる。
//! `meta.toml` の 2 層ラベルの意味検証（TOML パース必須）は TASK-4.2b
//! （#110）のスコープであり、本テストは構造検証（ファイル存在・
//! change_id の文字クラス）に留める。
//!
//! **検証範囲**（`fixtures/labeled-changes/README.md`「検証範囲の注記」
//! と対応）:
//! 1. 構造検証: `changes/` 配下が 15 ディレクトリ以上、各ディレクトリに
//!    `meta.toml`／`change.patch`／`poc3-result.json` が存在し、
//!    change_id（ディレクトリ名）が安全な文字クラスを満たす
//!    （パストラバーサル対策。A03。`.claude/rules/security.md`）。
//! 2. patch 適用可能性: 全 15 件が baseline へクリーン適用できる。
//! 3. 行数境界: G4 のみ 200 行超・他 14 件は 200 行以内
//!    （判定順序契約の行数閾値エスカレーション条件の再現）。
//! 4. build/test 境界の実証（コンパイル時間を抑えるため境界の代表例のみ）:
//!    D2 は適用後 `cargo build` 失敗、D1/D4/D5/G1 は適用後 `cargo test`
//!    失敗。全ゲート通過が期待される残り 10 件（S1〜S5・D3・G2・G3・G5）
//!    の build/test 実行は本テストのスコープ外とする（README「検証範囲の
//!    注記」参照。実装時にローカルで 1 回ずつ確認済み）。
//!
//! **セキュリティ（A03 インジェクション対策）**: patch 適用は実リポジトリ
//! 外の隔離作業ディレクトリ（workspace `target/labeled-changes-fixture/`）
//! でのみ行い、`git`/`cargo` は引数配列で起動する（シェル文字列展開なし）。
//! change_id の文字クラス検証でディレクトリトラバーサルを遮断する。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `changes/` 配下の変更セットは最低 15 件（安全・危険・グレー各 5 件以上。
/// README「15 件一覧」参照）。
const MIN_CHANGE_COUNT: usize = 15;

/// 判定順序契約の行数閾値（README「重要な注記」表）。
const LINES_MAX: usize = 200;

/// `cargo test` が失敗することを期待する変更セット（危険題材のバグ注入・
/// ゲーミング題材）。
const EXPECT_TEST_FAILURE: &[&str] = &[
    "D1-relu-sigmoid-swap",
    "D4-leaky-relu-sign-bug",
    "D5-lr-bug",
    "G1-gaming",
];

/// `cargo build` が失敗することを期待する変更セット（コンパイル不能題材）。
const EXPECT_BUILD_FAILURE: &[&str] = &["D2-private-method"];

/// `crates/guardrail/tests/fixtures/labeled-changes` への絶対パス。
fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labeled-changes")
}

/// workspace ルート（`crates/guardrail` の 2 階層上）。共有
/// `CARGO_TARGET_DIR`（`target/labeled-changes-fixture/`）の算出に使う。
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/guardrail の親ディレクトリ crates/ が存在する")
        .parent()
        .expect("crates/ の親ディレクトリ（workspace ルート）が存在する")
        .to_path_buf()
}

/// フィクスチャ専用の作業ディレクトリ。実リポジトリの `crates/*` 配下を
/// 汚さないよう workspace の `target/` 配下に隔離する（README「パッチの
/// 再生成・検証手順」と同じ隔離方針）。
fn fixture_work_root() -> PathBuf {
    workspace_root()
        .join("target")
        .join("labeled-changes-fixture")
}

/// 全パッチ適用テストで共有する `CARGO_TARGET_DIR`。cargo 呼び出しごとに
/// 依存クレート（tensor-core・autodiff・bench-harness）を再ビルドしない
/// ようにし、CI 所要時間を抑える。
fn shared_cargo_target_dir() -> PathBuf {
    fixture_work_root().join("cargo-target")
}

/// change_id（`changes/` 配下のディレクトリ名）の文字クラス契約。
/// 英数字始まり・`[A-Za-z0-9._-]` のみ・64 字以内（パストラバーサル対策。
/// v1 `changeset.rs::validate_change_id` と同一契約の先取り）。
fn is_valid_change_id(id: &str) -> bool {
    if id.is_empty() || id.len() > 64 {
        return false;
    }
    let mut chars = id.chars();
    let first = chars.next().expect("空文字列は上で弾いている");
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    id.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// `changes/` 配下の change_id 一覧を列挙する（隠しファイル・
/// ディレクトリ以外の混入物は除外）。
fn list_change_ids() -> Vec<String> {
    let changes_dir = fixtures_root().join("changes");
    let mut ids: Vec<String> = fs::read_dir(&changes_dir)
        .unwrap_or_else(|e| panic!("changes/ ディレクトリの読み取りに失敗: {changes_dir:?}: {e}"))
        .filter_map(|entry| {
            let entry = entry.expect("read_dir エントリの取得に失敗");
            if entry.file_type().expect("file_type の取得に失敗").is_dir() {
                Some(entry.file_name().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();
    ids.sort();
    ids
}

/// 構造検証: 15 件以上・各ディレクトリに 3 ファイル揃っている・
/// change_id の文字クラスが安全。
#[test]
fn changes_directory_has_required_structure() {
    let ids = list_change_ids();
    assert!(
        ids.len() >= MIN_CHANGE_COUNT,
        "changes/ 配下が {MIN_CHANGE_COUNT} 件未満: 実際は {} 件（{ids:?}）",
        ids.len()
    );

    let changes_dir = fixtures_root().join("changes");
    for id in &ids {
        assert!(
            is_valid_change_id(id),
            "change_id '{id}' が文字クラス契約（英数字始まり・[A-Za-z0-9._-]・64 字以内）を満たさない"
        );

        let dir = changes_dir.join(id);
        for required in ["meta.toml", "change.patch", "poc3-result.json"] {
            let path = dir.join(required);
            assert!(
                path.is_file(),
                "change_id '{id}' に必須ファイル '{required}' が存在しない: {path:?}"
            );
        }
    }
}

/// unified diff（`change.patch`）本文から変更行数（`+`/`-` で始まる行。
/// `+++`/`---` のファイルヘッダ行は除く）を数える。README「重要な注記」
/// 表の行数境界検証に使う。
fn count_changed_lines(patch_text: &str) -> usize {
    patch_text
        .lines()
        .filter(|line| {
            (line.starts_with('+') && !line.starts_with("+++"))
                || (line.starts_with('-') && !line.starts_with("---"))
        })
        .count()
}

/// 行数境界: G4 のみ 200 行超・他 14 件は 200 行以内。
#[test]
fn line_count_boundary_matches_labels() {
    let changes_dir = fixtures_root().join("changes");
    for id in list_change_ids() {
        let patch_path = changes_dir.join(&id).join("change.patch");
        let patch_text = fs::read_to_string(&patch_path)
            .unwrap_or_else(|e| panic!("{patch_path:?} の読み取りに失敗: {e}"));
        let lines_changed = count_changed_lines(&patch_text);

        if id == "G4-large-comment-refactor" {
            assert!(
                lines_changed > LINES_MAX,
                "G4 は変更行数が {LINES_MAX} 行を超える想定だが実際は {lines_changed} 行"
            );
        } else {
            assert!(
                lines_changed <= LINES_MAX,
                "{id} は変更行数が {LINES_MAX} 行以内の想定だが実際は {lines_changed} 行"
            );
        }
    }
}

/// `command` を `cwd` で実行し、成否と結合出力（stdout+stderr）を返す。
/// 引数は配列で渡すため、シェル文字列展開によるインジェクションの余地は
/// ない（A03 対策。`.claude/rules/security.md`）。
///
/// `program == "git"` の場合、親プロセス（本テストを起動した
/// `cargo test`。lefthook の pre-push フック経由で実行されると git 自身が
/// 子プロセスへ `GIT_DIR`/`GIT_WORK_TREE`/`GIT_INDEX_FILE` 等を継承させる）
/// から漏れ込んだ `GIT_*` 環境変数を明示的に除去する。除去しないと
/// `current_dir(cwd)` を無視してこのリポジトリ自体（本 worktree の
/// `.git`）に対して `git init`/`git commit` が実行され、フィクスチャの
/// 隔離が壊れる（モジュールコメントに明記済みの隔離方針を実際に満たす）。
fn run(cwd: &Path, program: &str, args: &[&str], envs: &[(&str, &str)]) -> (bool, String) {
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd);
    if program == "git" {
        for (key, _) in std::env::vars() {
            if key.starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("コマンド起動に失敗: {program} {args:?} (cwd={cwd:?}): {e}"));
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), combined)
}

/// `baseline/Cargo.toml` の相対 path 依存（`../../../../../{tensor-core,
/// autodiff,bench-harness}`）を絶対パスへ書き換えたテキストを返す。
/// フィクスチャ作業ディレクトリは baseline 本来の階層深度（5 階層下）を
/// 保持しないため、コミット済みファイル自体は変更せず、コピー後にこの
/// 関数で書き換えたテキストへ差し替える（コミット対象の `Cargo.toml` は
/// 元の相対パス表記のまま維持する。README「パッチの再生成・検証手順」
/// の運用と同じ）。
fn rewrite_cargo_toml_paths(original: &str, workspace_root: &Path) -> String {
    let mut out = original.to_string();
    for crate_name in ["tensor-core", "autodiff", "bench-harness"] {
        let rel = format!("../../../../../{crate_name}");
        let abs = workspace_root.join("crates").join(crate_name);
        out = out.replace(
            &format!("path = \"{rel}\""),
            &format!("path = \"{}\"", abs.display()),
        );
    }
    out
}

/// baseline を隔離作業ディレクトリへコピーし、`git init` 済みの初期
/// コミットを作る。`change.patch` はこのコミットに対して適用する
/// （README の「パッチの再生成・検証手順」と同じ手順を自動化したもの）。
fn prepare_baseline_worktree(dest: &Path) {
    if dest.exists() {
        fs::remove_dir_all(dest).unwrap_or_else(|e| panic!("{dest:?} の削除に失敗: {e}"));
    }
    fs::create_dir_all(dest).unwrap_or_else(|e| panic!("{dest:?} の作成に失敗: {e}"));

    let baseline_dir = fixtures_root().join("baseline");
    copy_dir_recursive(&baseline_dir, dest);

    let cargo_toml_path = dest.join("Cargo.toml");
    let original = fs::read_to_string(&cargo_toml_path)
        .unwrap_or_else(|e| panic!("{cargo_toml_path:?} の読み取りに失敗: {e}"));
    let rewritten = rewrite_cargo_toml_paths(&original, &workspace_root());
    fs::write(&cargo_toml_path, rewritten)
        .unwrap_or_else(|e| panic!("{cargo_toml_path:?} への書き込みに失敗: {e}"));

    // `GIT_DIR`/`GIT_WORK_TREE` 等の継承を避け、`dest` を明示的な作業木として
    // 初期化する（v1 `labeled_dataset.rs` と同じ隔離方針）。
    let (ok, out) = run(dest, "git", &["init", "-q"], &[]);
    assert!(ok, "git init に失敗: {out}");
    let (ok, out) = run(dest, "git", &["add", "-A"], &[]);
    assert!(ok, "git add に失敗: {out}");
    let (ok, out) = run(
        dest,
        "git",
        &[
            "-c",
            "user.email=guardrail-fixture@example.invalid",
            "-c",
            "user.name=guardrail-fixture",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        &[],
    );
    assert!(ok, "git commit に失敗: {out}");
}

/// `src`（ディレクトリ）を `dest` 配下へ再帰コピーする。`target/` は
/// コピー対象から除外する（baseline 自体に `.gitignore` で除外設定済み
/// だが、フィクスチャの `baseline/target/` が実装時のローカルビルドで
/// 生成されている場合に無駄なコピーを避けるため明示的にも除外する）。
fn copy_dir_recursive(src: &Path, dest: &Path) {
    for entry in fs::read_dir(src).unwrap_or_else(|e| panic!("{src:?} の読み取りに失敗: {e}"))
    {
        let entry = entry.expect("read_dir エントリの取得に失敗");
        let file_name = entry.file_name();
        if file_name == "target" {
            continue;
        }
        let src_path = entry.path();
        let dest_path = dest.join(&file_name);
        let file_type = entry.file_type().expect("file_type の取得に失敗");
        if file_type.is_dir() {
            fs::create_dir_all(&dest_path)
                .unwrap_or_else(|e| panic!("{dest_path:?} の作成に失敗: {e}"));
            copy_dir_recursive(&src_path, &dest_path);
        } else {
            fs::copy(&src_path, &dest_path)
                .unwrap_or_else(|e| panic!("{src_path:?} -> {dest_path:?} のコピーに失敗: {e}"));
        }
    }
}

/// 受け入れ条件の本体: 全 15 件の `change.patch` が baseline へ
/// クリーン適用できることを検証する。
#[test]
fn all_patches_apply_cleanly_to_baseline() {
    let ids = list_change_ids();
    assert!(ids.len() >= MIN_CHANGE_COUNT);

    for id in &ids {
        let work_dir = fixture_work_root().join(format!("apply-check-{id}"));
        prepare_baseline_worktree(&work_dir);

        let patch_path = fixtures_root()
            .join("changes")
            .join(id)
            .join("change.patch");
        let patch_path_str = patch_path
            .to_str()
            .unwrap_or_else(|| panic!("{patch_path:?} が有効な UTF-8 パスではない"));

        let (ok, out) = run(&work_dir, "git", &["apply", "--check", patch_path_str], &[]);
        assert!(
            ok,
            "change_id '{id}' の change.patch が baseline へ適用できない: {out}"
        );
    }
}

/// build/test 境界の実証: `EXPECT_BUILD_FAILURE`／`EXPECT_TEST_FAILURE`
/// に列挙した代表例のみ、実際に patch を適用して `cargo build`／
/// `cargo test` を実行し、期待どおりの失敗が起きることを確認する。
/// 全 15 件を実行しない理由は README「検証範囲の注記」を参照。
#[test]
fn build_and_test_boundary_for_representative_changes() {
    let target_dir = shared_cargo_target_dir();
    let target_dir_str = target_dir
        .to_str()
        .unwrap_or_else(|| panic!("{target_dir:?} が有効な UTF-8 パスではない"));

    for id in EXPECT_BUILD_FAILURE {
        let work_dir = fixture_work_root().join(format!("build-check-{id}"));
        prepare_baseline_worktree(&work_dir);
        apply_patch_or_panic(&work_dir, id);

        let (ok, out) = run(
            &work_dir,
            "cargo",
            &["build"],
            &[("CARGO_TARGET_DIR", target_dir_str)],
        );
        assert!(
            !ok,
            "change_id '{id}' は cargo build が失敗する想定だが成功した: {out}"
        );
    }

    for id in EXPECT_TEST_FAILURE {
        let work_dir = fixture_work_root().join(format!("test-check-{id}"));
        prepare_baseline_worktree(&work_dir);
        apply_patch_or_panic(&work_dir, id);

        let (ok, out) = run(
            &work_dir,
            "cargo",
            &["test"],
            &[("CARGO_TARGET_DIR", target_dir_str)],
        );
        assert!(
            !ok,
            "change_id '{id}' は cargo test が失敗する想定だが成功した: {out}"
        );
    }
}

/// `work_dir`（`prepare_baseline_worktree` 済み）へ `id` の
/// `change.patch` を実際に適用する。適用失敗はテスト前提が崩れている
/// ため即座に panic する。
fn apply_patch_or_panic(work_dir: &Path, id: &str) {
    let patch_path = fixtures_root()
        .join("changes")
        .join(id)
        .join("change.patch");
    let patch_path_str = patch_path
        .to_str()
        .unwrap_or_else(|| panic!("{patch_path:?} が有効な UTF-8 パスではない"));
    let (ok, out) = run(work_dir, "git", &["apply", patch_path_str], &[]);
    assert!(ok, "change_id '{id}' の change.patch 適用に失敗: {out}");
}
