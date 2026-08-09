//! `self-repair run --isolate-network` の probe-guarded 統合テスト（イシュー
//! #414・実装計画 §4 ステップ 5「統合（`tests/` 配下）」）。
//!
//! `crate::isolation::ExecIsolation::probe_unshare_net` は `unshare --user
//! --map-root-user --net true` の可用性に依存し、実行環境（self-hosted
//! runner・開発コンテナ・ユーザーのローカル環境）によって成否が変わる
//! （user namespace が禁止されている container/CI 環境がありうるため。
//! `docs/self-repair-candidate-isolation.md` 2 節）。本ファイルは probe の
//! 成否を実行時に判定し、
//! - probe が失敗する環境（本リポジトリの CI・開発コンテナが想定する既定）
//!   では `self-repair run --isolate-network` が fail-closed（exit 1・
//!   `--isolate-network` を含む診断メッセージ）で拒否されることを検証する
//! - probe が成功する環境（user namespace が許可されたホスト）では、
//!   その旨を明示出力してテストを early-return する（`#[ignore]` 分離は
//!   使わず、実行環境依存の分岐をテスト本体に持たせることで CI コンテナで
//!   green を維持する。実装計画 §4 ステップ 5 の方針どおり）
//!
//! `--repo`／`--candidates` は `main.rs::run_run` が `--isolate-network` の
//! probe チェックへ到達する前段（`resolve_baseline_commit`・
//! `load_candidates_from_json`・`guardrail::config::resolve`・
//! `RunSandbox::create`）をすべて満たす必要があるため、最小限の実 git
//! リポジトリ・非空の候補 JSON（`load_candidates_from_json` は空配列を
//! それ自体エラーとして拒否するため、`candidate.rs::load_candidates_from_json_parses_valid_input`
//! と同じ最小形の 1 件を用意する）を用意する。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn self_repair_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_self-repair"))
}

fn unique_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "self-repair-isolate-network-probe-test-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .expect("git コマンドの起動に失敗");
    assert!(
        status.success(),
        "git {args:?} が失敗しました（dir={}）",
        dir.display()
    );
}

/// `unshare --user --map-root-user --net true` を直接試行し、この実行環境で
/// network namespace 分離が可能かを判定する（`isolation.rs::
/// ExecIsolation::probe_unshare_net` と同じコマンドをテスト側で独立に再実行
/// し、本体の probe 結果に応じてアサーションを分岐するため）。
fn host_supports_unshare_net() -> bool {
    Command::new("unshare")
        .args(["--user", "--map-root-user", "--net", "true"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// 最小の git リポジトリ（コミット 1 つ）を構築し、`--repo` として使えるパス
/// を返す。
fn prepare_minimal_repo() -> PathBuf {
    let repo = unique_dir("repo");
    fs::create_dir_all(&repo).expect("repo ディレクトリ作成に失敗");
    git(&repo, &["init", "--quiet", "--initial-branch=main"]);
    git(&repo, &["config", "user.email", "test@example.com"]);
    git(&repo, &["config", "user.name", "Test"]);
    fs::write(repo.join("a.txt"), "baseline\n").expect("a.txt 書き込みに失敗");
    git(&repo, &["add", "--all"]);
    git(&repo, &["commit", "--quiet", "-m", "baseline"]);
    repo
}

/// `load_candidates_from_json` が受理する最小の非空候補列を書き込む
/// （`candidate.rs::load_candidates_from_json_parses_valid_input` と同じ
/// 最小形。空配列は `load_candidates_from_json` 自体がそれ単体でエラーとして
/// 拒否するため、`--isolate-network` の probe チェック〈`RunSandbox::create`
/// より後段〉へ到達するには非空候補が必須）。
fn write_minimal_candidates(path: &Path) {
    fs::write(
        path,
        r#"[{"description": "probe-test", "files": [{"path": "a.txt", "content": "candidate\n"}]}]"#,
    )
    .expect("candidates.json 書き込みに失敗");
}

/// イシュー #414 の中核契約: probe が失敗する環境では `--isolate-network` が
/// 黙って劣化せず fail-closed（exit 1）で拒否される。
#[test]
fn run_with_isolate_network_fails_closed_when_unshare_net_unavailable() {
    if host_supports_unshare_net() {
        eprintln!(
            "skip: この実行環境は unshare --user --map-root-user --net を許可しているため、\
             fail-closed 拒否パスを再現できません（`docs/self-repair-candidate-isolation.md` \
             2 節が想定する user namespace 禁止環境ではないための early-return）"
        );
        return;
    }

    let repo = prepare_minimal_repo();
    let candidates = repo.join("candidates.json");
    write_minimal_candidates(&candidates);
    let log = repo.join("trial.jsonl");

    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "feature-addition",
            "--repo",
            repo.to_str().expect("repo パスは UTF-8 のはず"),
            "--log",
            log.to_str().expect("log パスは UTF-8 のはず"),
            "--candidates",
            candidates.to_str().expect("candidates パスは UTF-8 のはず"),
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
            "--allow-candidate-exec",
            "--isolate-network",
        ])
        .output()
        .expect("failed to run self-repair binary");

    let _ = fs::remove_dir_all(&repo);

    assert_eq!(
        output.status.code(),
        Some(1),
        "probe 失敗時は内部エラー区分（exit 1）で拒否されるはず。stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--isolate-network"),
        "fail-closed の理由が --isolate-network に言及するはず: {stderr}"
    );
}

/// `--isolate-network` を指定しない場合は probe 自体を実行しないため、
/// probe が使えない環境でも（他の必須引数が揃っていれば）probe 由来の
/// exit 1 には到達しない回帰確認（`main.rs::run_run` の `if args.isolate_network`
/// ガード）。候補実行自体は完走させない（`--bench-bin` の実ワークロードが
/// 存在しないため後続ステップで別の exit 1 になりうるが、`--isolate-network`
/// 由来のメッセージを含まないことのみを確認する）。
#[test]
fn run_without_isolate_network_does_not_invoke_probe() {
    let repo = prepare_minimal_repo();
    let candidates = repo.join("candidates.json");
    write_minimal_candidates(&candidates);
    let log = repo.join("trial.jsonl");

    let output = self_repair_bin()
        .args([
            "run",
            "--kind",
            "feature-addition",
            "--repo",
            repo.to_str().expect("repo パスは UTF-8 のはず"),
            "--log",
            log.to_str().expect("log パスは UTF-8 のはず"),
            "--candidates",
            candidates.to_str().expect("candidates パスは UTF-8 のはず"),
            "--bench-bin",
            "bench_workload",
            "--workload-source",
            "src/bin/bench_workload.rs",
            "--allow-candidate-exec",
        ])
        .output()
        .expect("failed to run self-repair binary");

    let _ = fs::remove_dir_all(&repo);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("--isolate-network が指定されましたが"),
        "--isolate-network 未指定時は probe 拒否メッセージが出ないはず: {stderr}"
    );
}
