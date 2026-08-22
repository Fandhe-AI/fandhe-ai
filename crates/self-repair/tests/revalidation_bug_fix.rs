//! TASK-3.3b（イシュー #141）: バグ修正種別の自己修復ループを
//! `self-repair run` CLI 経由で 1 回実行し、実測結果を記録する。
//!
//! REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）「自作コアに
//! 対する自己修復ループの人間介在なし完走を新実装リポで再実証する」を、
//! `docs/self-repair-revalidation-plan.md` §4.1 の推奨題材（`Var::relu`
//! 〈`crates/autodiff/src/var.rs`〉の実装本体を sigmoid 相当の演算グラフへ
//! すり替える）で 1 ループ実測する。題材選定・完走判定基準は TASK-3.3a
//! （#140・PR #322）で人間承認済み。
//!
//! # CLI 経由への移行（#139 (c) 差し戻し対応。#141 は 2 度目の差し戻し）
//! 旧版（PR #341）は **lib API 直接呼び出しの実証ハーネス**として
//! [`self_repair::SelfRepairLoop`] を直接構築していたが、#139 reopen コメント
//! の判断 (c) 差し戻しにより、完走判定基準 1・2・4・5・6
//! （`docs/self-repair-revalidation-plan.md` §5）を CLI 経由・
//! `self-repair verify-log` 外部コマンド経由・候補 diff 直接実測・
//! `signal_source: "measured"` 付きで満たすことが求められた（#139 ユーザー
//! 判断コメント・2026-08-08）。機能追加種別（#142）は
//! `tests/feature_addition_loop_completion_task_3_3c.rs`（PR #361）で同様の
//! 移行を先行完了しており、本ファイルはその構成をバグ修正種別へ写像する。
//! `env!("CARGO_BIN_EXE_self-repair")` で実バイナリを **1 回だけ起動**し、
//! 続けて `self-repair verify-log` を別プロセスとして起動する。
//!
//! # 「準備リポジトリ」（`--repo` への入力）の構成
//! `self-repair run` の `--repo` は `RunSandbox::create`（`sandbox.rs`）が
//! `git clone --local` する対象であり、以降の build/test/clippy/bench は
//! すべてそのクローン先（`--repo` そのものではなく内部隔離 sandbox）を
//! ワークスペースとして実行される（`main.rs::run_run` 参照）。本ハーネスは
//! 実 workspace 全体ではなく `crates/autodiff` 1 クレートのみを検証対象と
//! するため（旧版・`verify_gates_integration.rs` と同じスコーピング判断。
//! 実行時間の観点）、`crates/autodiff` を一意な一時ディレクトリへ再帰コピー
//! し、ワークスペース継承（`*.workspace = true`）をすべて実体値へ展開して
//! 独立 crate 化したうえで、単独の git リポジトリとして `git init` する
//! （[`prepare_standalone_autodiff_repo`]）。この「準備リポジトリ」の HEAD が
//! `--repo` の `baseline_commit`（バグ注入済み状態）になる。
//!
//! # ベンチワークロード（候補 diff 直接実測。基準 2・5）
//! [`crate::verify_bench_direct::DirectBenchRunner`] が `--bench-bin`／
//! `--workload-source` で指定した bin（`src/bin/bench_workload.rs`）を
//! baseline・候補適用後の双方でビルド・実行し、実行時間を直接比較する
//! （`RepairCompositeGate`。TASK-3.2a・#137）。本ハーネスが用意する
//! `bench_workload.rs` は `fandhe_ai_autodiff::Var::relu` の forward+backward を反復する
//! 決定的ワークロードであり、`tests/fixtures/feature-addition-leaky-relu/
//! baseline/src/bin/bench_workload.rs` と同じ計測プロトコル（xorshift64 決定的
//! 入力・`black_box`）を踏襲する。旧版（`SelfRepairBenchGate` による合成
//! ワークロード完走確認のみ）とは異なり、本バージョンは候補 diff（`var.rs` の
//! 実装差分）そのものの性能劣化率を実測する（基準 5 の充足）。
//!
//! # シグナルは実測のみ（捏造しない）
//! `lines_changed`／`api_broken`／`gaming_suspect`／`exclusion_rule_ids`・ベンチは
//! いずれも CLI バイナリ内部の [`self_repair::RepairCompositeGate`] が試行ごとに
//! sandbox 内の使い捨て git リポジトリを対象に実測する。未計測値を fail-open な
//! 既定値で埋めない（`.claude/rules/security.md` A08）。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// 準備リポジトリ（sandbox ディレクトリ）を、`Drop` により確実に削除する
/// RAII ガード（レビュー指摘への対応。旧版〈PR #341〉が持っていた
/// `SandboxGuard` を CLI 経由への全面書き換え時に落としていたのを復元する）。
/// 本テストは複数の `assert!` を通過した後にのみ手動クリーンアップを行う
/// 構成だったため、途中の `assert!` 失敗（panic）で sandbox（`crates/autodiff`
/// のフルコピー＋ビルド成果物）が `/tmp` に残置される退行があった。PID
/// ベースの一時パス（`unique_sandbox_dir`）のため次回実行時には自己回収
/// されるが、失敗のたびに一時的なディスク消費が発生するため、通常経路・
/// panic 経路のいずれでも確実に削除する。
struct SandboxGuard(PathBuf);

impl Drop for SandboxGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `--candidates` JSON 出力先（一時ファイル）を、`Drop` により確実に削除する
/// RAII ガード。`SandboxGuard` と同じ理由（panic 時の残置防止）で、
/// 候補 JSON 側にも同型のガードを用意する。
struct CandidatesFileGuard(PathBuf);

impl Drop for CandidatesFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// バグ注入・修正対象ファイル（`crates/autodiff/src/var.rs`。準備リポジトリ
/// 相対パスは `src/var.rs`）。
const TARGET_FILE: &str = "src/var.rs";
/// ベンチワークロード bin（`RepairCompositeGate` の候補 diff 直接実測が
/// ピン留め・ビルド対象とする）。
const WORKLOAD_SOURCE: &str = "src/bin/bench_workload.rs";
const BENCH_BIN: &str = "bench_workload";

/// リポジトリルート（`crates/self-repair` の 2 階層上）。
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/self-repair の親は crates/")
        .parent()
        .expect("crates/ の親はリポジトリルート")
        .to_path_buf()
}

/// テストごとに衝突しない一時ディレクトリ（`self_repair::test_support` と同じ
/// `temp_dir() + process::id()` 方式。本ファイルは `tests/` 配下の独立クレート
/// のため `pub(crate)` ヘルパーを再利用できず、`feature_addition_loop_completion_
/// task_3_3c.rs` と同型のヘルパーを再実装する）。
fn unique_sandbox_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "self-repair-revalidation-bug-fix-task-3-3b-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("sandbox ディレクトリ作成に失敗");
    dir
}

/// 単一ファイル用の一時パス（`--candidates` JSON 出力先）。
fn unique_temp_file(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "self-repair-revalidation-bug-fix-task-3-3b-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// `src` 配下を再帰的に `dst` へコピーする（`target/`・`.git/` は対象外）。
fn copy_dir_recursive(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("コピー先ディレクトリ作成に失敗");
    for entry in std::fs::read_dir(src).expect("コピー元ディレクトリの読み取りに失敗")
    {
        let entry = entry.expect("ディレクトリエントリの読み取りに失敗");
        let file_type = entry.file_type().expect("file_type 取得に失敗");
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path);
        } else {
            std::fs::copy(&src_path, &dst_path).expect("ファイルコピーに失敗");
        }
    }
}

/// `crates/autodiff` のコピー（`sandbox`）の `Cargo.toml` を、ルート workspace
/// への継承（`*.workspace = true`）から切り離し、単独 crate として独立
/// ビルド可能にする（実装計画 §3.2 手順 2）。
///
/// - `version`/`edition`/`license`/`publish` はルート `Cargo.toml`
///   `[workspace.package]` の実体値へ展開する
/// - `tensor-core`／`bench-harness`（dev-dependency）への相対 path 依存は、
///   コピー先では深度が保たれないため実リポジトリへの絶対パスへ書き換える
/// - `serde`/`serde_json`（dev-dependency）は `[workspace.dependencies]` の
///   `=x.y.z` 固定値へ実体化する
/// - 末尾へ空の `[workspace]` テーブルを追記し、親 workspace の巻き込みから
///   切り離す（`tests/fixtures/feature-addition-leaky-relu/baseline/Cargo.toml`
///   と同一方針）
///
/// 各置換は `assert_ne!` ガード付きで、実リポジトリ側の `Cargo.toml` 構造が
/// 変化した場合に fail-fast で検出する（レビュー指摘・実装計画 §7 リスク
/// 対策）。
fn detach_autodiff_cargo_toml(sandbox: &Path) {
    let cargo_toml = sandbox.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).expect("sandbox Cargo.toml 読み取りに失敗");

    let with_package_fields = content
        .replace("version.workspace = true", "version = \"0.3.0\"")
        .replace("edition.workspace = true", "edition = \"2024\"")
        .replace(
            "license.workspace = true",
            "license = \"MIT OR Apache-2.0\"",
        )
        .replace("publish.workspace = true", "publish = false");
    assert_ne!(
        with_package_fields, content,
        "package フィールドの workspace 継承が 1 件も置換されなかった \
         （crates/autodiff/Cargo.toml の記法が変わっていないか確認）"
    );

    let tensor_core_abs = repo_root().join("crates/tensor-core");
    let bench_harness_abs = repo_root().join("crates/bench-harness");
    let with_path_deps = with_package_fields
        .replace(
            r#"path = "../tensor-core""#,
            &format!(r#"path = "{}""#, tensor_core_abs.display()),
        )
        .replace(
            r#"path = "../bench-harness""#,
            &format!(r#"path = "{}""#, bench_harness_abs.display()),
        );
    assert_ne!(
        with_path_deps, with_package_fields,
        "path 依存の書き換えが 1 件も適用されなかった \
         （crates/autodiff/Cargo.toml の記法が変わっていないか確認）"
    );

    // TASK-12.1d（#164）: `Tape::new_with_ops(ops)` の破壊的変更（ops 必須化）に伴い、
    // `bench_workload_source()`（下記）が生成する `src/bin/bench_workload.rs`
    // は `Tape::new_with_ops(...)` へ渡す `BackendOps` 実装を要する。`autodiff` 自身は
    // `backend-cpu` へ依存しない（設計上の不変条件）ため、この bin 専用の
    // 依存として `[dependencies]` セクションへ直接追記する（`[dependencies]`
    // は `dev-dependencies` と異なり通常の `cargo build --release --bin
    // bench_workload` でも解決される）。`tensor-core` 行の直後（同じ
    // `[dependencies]` テーブル内）へ挿入することでテーブル境界を跨がない。
    let backend_cpu_abs = repo_root().join("crates/backend-cpu");
    let tensor_core_line = format!(r#"path = "{}""#, tensor_core_abs.display());
    let with_backend_cpu_dep = with_path_deps.replacen(
        &tensor_core_line,
        &format!(
            "{tensor_core_line}\nbackend-cpu = {{ path = \"{}\" }}",
            backend_cpu_abs.display()
        ),
        1,
    );
    assert_ne!(
        with_backend_cpu_dep, with_path_deps,
        "backend-cpu 依存の追記が適用されなかった（tensor-core 行のテキストが変わっていないか確認）"
    );
    let with_path_deps = with_backend_cpu_dep;

    let with_dev_deps = with_path_deps
        .replace(
            "serde.workspace = true",
            "serde = { version = \"=1.0.229\", features = [\"derive\"] }",
        )
        .replace("serde_json.workspace = true", "serde_json = \"=1.0.151\"");
    assert_ne!(
        with_dev_deps, with_path_deps,
        "serde/serde_json の workspace 継承が 1 件も置換されなかった \
         （crates/autodiff/Cargo.toml の記法が変わっていないか確認）"
    );

    let detached = format!("{with_dev_deps}\n[workspace]\n");
    std::fs::write(&cargo_toml, detached).expect("sandbox Cargo.toml 書き換えに失敗");
}

/// 候補 diff 直接実測（TASK-3.2a・#137）向けの決定的ベンチワークロード
/// （`WORKLOAD_SOURCE`）を準備リポジトリへ追加する。`fandhe_ai_autodiff::Var::relu` の
/// forward+backward を反復する（`tests/fixtures/feature-addition-leaky-relu/
/// baseline/src/bin/bench_workload.rs` と同じ計測プロトコル: xorshift64 決定的
/// 入力・`black_box`・1 プロセス実行あたり 10ms 以上の作業量）。
fn bench_workload_source() -> &'static str {
    r#"//! 候補 diff 直接実測（TASK-3.2a・イシュー #137）向けの決定的ベンチワークロード。
//!
//! `crate::verify_bench_direct::DirectBenchRunner` が baseline commit・候補
//! 適用済み作業木の双方でこの bin を `cargo build --release --bin
//! bench_workload` し、生成物を「1 回 exec するだけ」で外部タイミング計測する。
//! このファイル自体は `DirectBenchSpec::workload_sources` によってピン留めされ、
//! 候補 diff が改変すると計測前に fail-closed で拒否される（ゲーミング防止）。
//!
//! `fandhe_ai_autodiff::Var::relu` の forward+backward を反復する（TASK-3.3b・#141 の
//! バグ注入対象と同一 API）。決定的シード・作業量の方針は
//! `tests/fixtures/feature-addition-leaky-relu/baseline/src/bin/bench_workload.rs`
//! と同一（本ハーネス〈`tests/revalidation_bug_fix.rs`〉が生成する）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::Tensor;

const SEED: u64 = 42;
const ELEMENTS: usize = 4096;

fn deterministic_inputs(count: usize) -> Vec<f32> {
    let mut state = SEED ^ 0x9E37_79B9_7F4A_7C15;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        values.push((unit * 4.0 - 2.0) as f32);
    }
    values
}

fn run_once(inputs: &[f32]) -> f32 {
    let tensor = Tensor::new(inputs.to_vec(), &[inputs.len()])
        .expect("bench_workload: shape とデータ長は一致させている");
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));

    let x = tape.var(&tensor);
    let y = x.relu();
    let grad = tape
        .backward(&y)
        .expect("bench_workload: relu backward は常に成功する");
    grad.get(&x)
        .expect("bench_workload: x は同一 tape 上のノード")
        .map(|g| {
            g.contiguous()
                .as_slice()
                .map(|s| s.iter().sum())
                .unwrap_or(0.0)
        })
        .unwrap_or(0.0)
}

fn main() {
    let inputs = deterministic_inputs(ELEMENTS);
    let mut acc = 0.0f32;
    for _ in 0..4000u32 {
        acc = std::hint::black_box(acc + run_once(std::hint::black_box(&inputs)));
    }
    std::hint::black_box(acc);
}
"#
}

/// `var.rs` を行単位で読み込み、`relu` メソッド本体の forward 呼び出し行を
/// 探す（`pub fn relu(&self)` 宣言の直後という行番号ベースの探索。同一
/// テキストが `sigmoid` メソッド本体にも存在するため素朴な全文検索は使わない
/// ——旧版〈PR #341〉のヘルパーをそのまま再利用する）。
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
/// へすり替える（`Op::Relu` の登録自体は変更しない。#140 承認済み題材）。
/// forward 値は sigmoid 相当になる一方、backward は `Op::Relu` の勾配式のまま
/// 計算されるため、既知正解値テスト（`crates/autodiff/tests/backward.rs` 等）が
/// forward 段階で失敗する。
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
/// 値が一致しない。既知正解値テストは失敗し続けるため検証ゲートで却下される
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

/// sandbox 内でのみ完結する `git` コマンドを構築する。`current_dir(sandbox)`
/// だけでは実リポジトリへの誤動作を防げない（githooks(5) 経由で継承されうる
/// `GIT_*` 環境変数が `current_dir` より優先されるため。
/// `feature_addition_loop_completion_task_3_3c.rs::sandboxed_git_command`・
/// `sandbox.rs::git_command` と同一の事故パターン・同一の対処。#149 実測）。
fn sandboxed_git_command(sandbox: &Path, args: &[&str]) -> Command {
    let mut command = Command::new("git");
    command.args(args).current_dir(sandbox);
    for (key, _) in std::env::vars_os() {
        if let Some(key_str) = key.to_str()
            && key_str.starts_with("GIT_")
        {
            command.env_remove(key_str);
        }
    }
    command
}

/// `sandbox`（準備リポジトリ）を使い捨て git リポジトリ化し、現在の内容
/// （バグ注入済み）を `baseline_commit` として記録する。`self-repair run` の
/// `--repo` はこの sandbox を指す（`RunSandbox::create` が更にこれを
/// `git clone --local` して内部隔離 sandbox を構築する。`sandbox.rs` doc 参照）。
fn git_init_baseline(sandbox: &Path) {
    let run = |args: &[&str]| {
        let output = sandboxed_git_command(sandbox, args)
            .output()
            .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗: {error}"));
        assert!(
            output.status.success(),
            "git {args:?} が失敗しました: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    run(&["init", "-q"]);
    run(&["add", "-A"]);
    run(&[
        "-c",
        "user.email=self-repair-task-3-3b@example.invalid",
        "-c",
        "user.name=self-repair-task-3-3b",
        "commit",
        "-q",
        "-m",
        "baseline (relu->sigmoid activation mismatch, revalidation harness)",
    ]);
}

fn self_repair_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_self-repair"))
}

/// 準備リポジトリの `Cargo.lock` を baseline commit 前に生成する。
///
/// 実行時に判明した挙動: `crates/autodiff` は workspace member のため単独の
/// `Cargo.lock` を持たない（ルート `Cargo.lock` を参照する）。detach 後の
/// 準備リポジトリで `cargo generate-lockfile` を実行せずに `git init` すると、
/// `Cargo.lock` は baseline commit に含まれず、attempt 1 の 3 ゲート実行
/// （`cargo build` 等）が初めて生成する未追跡ファイルになる。
/// [`crate::verify_bench_direct::pinned_sources_untouched`] は
/// `baseline_commit` との差分に現れる `Cargo.lock` を無条件でピン留め違反
/// （マニフェスト改変によるゲーミング疑い）とみなすため、これを baseline
/// 側に確定させておく必要がある（`tests/fixtures/feature-addition-leaky-relu/
/// baseline/Cargo.lock` が baseline から commit 済みであるのと同じ理由）。
fn generate_lockfile(sandbox: &Path) {
    let mut command = Command::new("cargo");
    command.args(["generate-lockfile"]).current_dir(sandbox);
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("cargo generate-lockfile の起動に失敗: {error}"));
    assert!(
        output.status.success(),
        "cargo generate-lockfile が失敗しました: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// `crates/autodiff` を一意な一時ディレクトリへコピーし、workspace 継承の
/// 実体化・ベンチワークロード追加・バグ注入・git 初期化までを行った「準備
/// リポジトリ」を構築する。`sandbox` は呼び出し元が [`unique_sandbox_dir`]
/// で確保済みのディレクトリを渡す（`SandboxGuard` を本関数呼び出し前に
/// 取得できるようにするため。関数内で新規作成すると、内部の `assert!`／
/// `assert_ne!`〈`detach_autodiff_cargo_toml`・`generate_lockfile`・
/// `git_init_baseline`〉が panic した場合に sandbox がガード対象外のまま
/// 残置してしまう）。戻り値は `(injected_var_rs_content,
/// original_var_rs_content)`（`original` は attempt 2〈正解復元〉の候補内容
/// として使う）。
fn prepare_standalone_autodiff_repo(sandbox: &Path) -> (String, String) {
    let fixture_src = repo_root().join("crates/autodiff");
    copy_dir_recursive(&fixture_src, sandbox);
    detach_autodiff_cargo_toml(sandbox);

    let bin_dir = sandbox.join("src/bin");
    std::fs::create_dir_all(&bin_dir).expect("src/bin ディレクトリ作成に失敗");
    std::fs::write(bin_dir.join("bench_workload.rs"), bench_workload_source())
        .expect("bench_workload.rs の書き込みに失敗");

    // 実 `crates/autodiff` は `.gitignore` を持たず、ルート workspace の
    // `.gitignore`（`/target` 除外）に依存している。準備リポジトリは単独 git
    // リポジトリのため、この除外を明示的に持たせないと `cargo build`/`test`
    // が生成する `target/` 配下のバイナリ成果物が `git add -A`（diff 由来
    // シグナル実測。`diff_signals.rs`）に巻き込まれ `git diff --numstat` が
    // バイナリ差分（"-"）を返して `lines_changed` 実測が fail-closed に
    // panic する（実装時に実測して判明。`tests/fixtures/feature-addition-
    // leaky-relu/baseline/.gitignore` と同一内容）。
    std::fs::write(sandbox.join(".gitignore"), "/target\n").expect(".gitignore の書き込みに失敗");

    let var_rs_path = sandbox.join(TARGET_FILE);
    let original_content =
        std::fs::read_to_string(&var_rs_path).expect("var.rs の読み込みに失敗しました（注入前）");
    let injected_content = inject_bug(&original_content);
    std::fs::write(&var_rs_path, &injected_content).expect("var.rs へのバグ注入書き込みに失敗");

    generate_lockfile(sandbox);
    git_init_baseline(sandbox);

    (injected_content, original_content)
}

/// TASK-3.3b（#141）の受け入れ条件本体: バグ修正種別のループが
/// `self-repair run` CLI 経由・1 回起動・人間介在なしで完走し、`signal_source:
/// "measured"`・候補 diff 直接実測ベンチ（基準 2・5）・改竄検知ログ（`verify-log`
/// CLI・基準 4）を伴うことを確認する。
#[test]
#[ignore = "cargo build/test --release/clippy を試行1・2の2系列分・crates/autodiff \
            全テストスイート込みで実行するため長時間かかる。通常 CI ジョブでは実行しない。\
            実行: cargo test -p self-repair --test revalidation_bug_fix -- --ignored --nocapture"]
fn bug_fix_loop_reaches_adopted_with_measured_evidence() {
    let overall_start = Instant::now();

    // --- 準備リポジトリの構築（バグ注入済み baseline） ---
    // sandbox は `prepare_standalone_autodiff_repo` 呼び出し前に確保し、
    // 直後にガードを取得する。関数内部の `assert!`／`assert_ne!` が panic
    // しても sandbox が確実に削除されるようにするため（レビュー指摘対応。
    // `SandboxGuard`／`prepare_standalone_autodiff_repo` doc 参照）。
    let sandbox = unique_sandbox_dir("prep");
    let _sandbox_guard = SandboxGuard(sandbox.clone());
    let (injected_content, original_content) = prepare_standalone_autodiff_repo(&sandbox);

    // --- 候補列（#140 承認済み題材）を `--candidates` JSON へ書き出す ---
    let candidates_json = serde_json::json!([
        {
            "description": "試行1（誤った修正: eval::sigmoid を eval::tanh に置換。\
                             依然として Op::Relu の勾配式と forward 値が一致しない）",
            "files": [{"path": TARGET_FILE, "content": wrong_fix_content(&injected_content)}],
        },
        {
            "description": "試行2（正しい修正: relu 実装〈eval::relu〉を復元）",
            "files": [{"path": TARGET_FILE, "content": original_content}],
        },
    ]);
    let candidates_path = unique_temp_file("candidates.json");
    std::fs::write(
        &candidates_path,
        serde_json::to_string_pretty(&candidates_json).expect("候補 JSON のシリアライズに失敗"),
    )
    .expect("候補 JSON の書き込みに失敗");
    // 以降の assert! が panic しても候補 JSON が確実に削除されるよう、
    // 書き込み直後にガードを取得する（CandidatesFileGuard doc 参照）。
    let _candidates_guard = CandidatesFileGuard(candidates_path.clone());

    // --- `self-repair run` を 1 回だけ起動する（完走判定基準 1） ---
    let target_out_dir = repo_root().join("target/self-repair-revalidation/bug-fix");
    std::fs::create_dir_all(&target_out_dir).expect("完走ログ出力先ディレクトリ作成に失敗");
    let log_path = target_out_dir.join("loop-log.jsonl");
    // `LogWriter::open` は既存ファイルへ追記継続するため、固定ファイル名を
    // 実行のたび削除してから開き「このディレクトリはこの 1 回の実行を記述
    // する」契約を `loop-report.json` の上書きと揃える。
    let _ = std::fs::remove_file(&log_path);
    let output_path = target_out_dir.join("loop-report.json");

    let guardrail_config_path = repo_root().join("guardrail.toml");
    let policy_exclusion_path = repo_root().join("policy-exclusion.toml");
    let run_args: Vec<String> = vec![
        "run".to_string(),
        "--kind".to_string(),
        "bug-fix".to_string(),
        "--repo".to_string(),
        sandbox.display().to_string(),
        "--max-attempts".to_string(),
        "5".to_string(),
        "--log".to_string(),
        log_path.display().to_string(),
        "--output".to_string(),
        output_path.display().to_string(),
        "--candidates".to_string(),
        candidates_path.display().to_string(),
        "--bench-bin".to_string(),
        BENCH_BIN.to_string(),
        "--workload-source".to_string(),
        WORKLOAD_SOURCE.to_string(),
        "--config".to_string(),
        guardrail_config_path.display().to_string(),
        "--policy-exclusion".to_string(),
        policy_exclusion_path.display().to_string(),
        // 本テストは #140 承認済み題材の候補（ハーネス自身が決定的に生成する
        // 信頼済み入力。`docs/guardrail-self-repair-cli.md` §3.7「候補実行の
        // 信頼境界」参照）を渡す実証ハーネスであり、`--candidates` の候補
        // コードを sandbox 内で `cargo build`/`cargo test`/`cargo clippy` として
        // ホスト権限で実行することを承認する（`feature_addition_loop_
        // completion_task_3_3c.rs` と同じ根拠）。
        "--allow-candidate-exec".to_string(),
    ];
    let run_output = self_repair_bin()
        .args(&run_args)
        .output()
        .expect("self-repair run の起動に失敗");
    let run_exit_code = run_output.status.code();
    assert!(
        matches!(run_exit_code, Some(0) | Some(10) | Some(20)),
        "self-repair run は 3 分岐（0/10/20）のいずれかの終端 verdict に到達するはず（内部エラー・usage エラーで終わらない）: exit={run_exit_code:?}, stdout={}, stderr={}",
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );

    // --- `self-repair verify-log` を外部コマンド経由で検証する（完走判定基準 4） ---
    let verify_output = self_repair_bin()
        .args(["verify-log", "--log", log_path.to_str().unwrap()])
        .output()
        .expect("self-repair verify-log の起動に失敗");
    assert_eq!(
        verify_output.status.code(),
        Some(0),
        "loop-log.jsonl のハッシュチェーン検証（CLI 経由）に失敗しました: stdout={}, stderr={}",
        String::from_utf8_lossy(&verify_output.stdout),
        String::from_utf8_lossy(&verify_output.stderr)
    );

    // --- `--output` JSON を読み込み、実測結果を検証する ---
    let report_text =
        std::fs::read_to_string(&output_path).expect("loop-report.json の読み込みに失敗");
    let mut report_json: serde_json::Value =
        serde_json::from_str(&report_text).expect("loop-report.json のパースに失敗");

    assert!(
        report_json["outcome"]
            .as_str()
            .is_some_and(|s| s.starts_with("Adopted")),
        "relu 実装の復元のみ（既存 pub fn シグネチャは不変）のため api_broken=false と \
         正しく判定され Adopted が実測結果のはず: {report_json}"
    );
    assert_eq!(
        run_exit_code,
        Some(0),
        "Adopted の終了コードは 0 のはず（docs/guardrail-self-repair-cli.md 3.5 節）"
    );
    assert_eq!(
        report_json["attempt_count"], 2,
        "試行1（検証不合格）→試行2（検証通過・採用）の 2 試行系列であるはず: {report_json}"
    );
    let attempts = report_json["attempts"]
        .as_array()
        .expect("attempts は配列のはず");
    assert!(
        attempts[0]["outcome"]
            .as_str()
            .is_some_and(|s| s.contains("VerificationFailed")),
        "試行1は既知正解値テスト不合格で検証不合格になるはず（完走判定基準 2 の一部）: {:?}",
        attempts[0]
    );
    assert!(
        attempts[1]["outcome"]
            .as_str()
            .is_some_and(|s| s.contains("Adopted")),
        "試行2は検証通過後に採用されるはず: {:?}",
        attempts[1]
    );
    assert_eq!(
        report_json["signal_source"], "measured",
        "--signals 契約検証パスを経由しない実シグナル計測であるはず（完走判定基準 6）: {report_json}"
    );

    // ベンチが `NotRun` に丸められず候補 diff 直接実測されたことを確認する
    // （完走判定基準 5。試行2は build/test/clippy 3 ゲートを通過しているため、
    // `RepairCompositeGate` の実行順序契約〈全ゲート通過後に限りベンチを
    // 計測する〉によりベンチも実測済みになる）。
    let adopted_evidence = &report_json["adopted_evidence"];
    assert!(
        !adopted_evidence.is_null(),
        "試行2は検証（3 ゲート＋ベンチ）を通過している以上、その証跡が記録されているはず: {report_json}"
    );
    let bench_median_pct = adopted_evidence["bench_median_pct"]
        .as_f64()
        .expect("bench_median_pct は NotRun に丸められず有限値のはず（完走判定基準 5）");
    assert!(bench_median_pct.is_finite());
    assert!(
        adopted_evidence["gate_report"]
            .as_str()
            .is_some_and(|s| s.contains("bench=measured-direct")),
        "gate_report は候補 diff 直接実測（RepairCompositeGate）由来であるはず: {adopted_evidence}"
    );
    let bench_measurements_pct = adopted_evidence["bench_measurements_pct"]
        .as_array()
        .expect("bench_measurements_pct（生の計測系列）が記録されているはず");
    assert!(
        bench_measurements_pct.len() >= self_repair::verify_bench::MIN_BENCH_ITERATIONS,
        "5 回以上の計測系列であるはず（REQ-4 受け入れ基準）: {bench_measurements_pct:?}"
    );
    assert!(
        adopted_evidence["api_broken"].as_bool() == Some(false),
        "relu 実装本体の復元のみで既存シグネチャは消失していないため api_broken=false のはず: {adopted_evidence}"
    );

    // 実行コマンドライン（監査・再現性のための記録）を付与する。`--repo`／
    // `--candidates` は使い捨て sandbox・一時ファイルの絶対パスであり実行の
    // たび変わる既知の制約のため相対化しない（`feature_addition_loop_
    // completion_task_3_3c.rs` と同じ方針）。
    let root = repo_root();
    let relativize = |absolute: &Path| -> String {
        absolute
            .strip_prefix(&root)
            .map(|relative| relative.display().to_string())
            .unwrap_or_else(|_| absolute.display().to_string())
    };
    let invocation_for_record = format!(
        "self-repair run --kind bug-fix --repo <sandbox> --max-attempts 5 --log {} --output {} --candidates <candidates.json> --bench-bin {} --workload-source {} --config {} --policy-exclusion {} --allow-candidate-exec",
        relativize(&log_path),
        relativize(&output_path),
        BENCH_BIN,
        WORKLOAD_SOURCE,
        relativize(&guardrail_config_path),
        relativize(&policy_exclusion_path),
    );
    report_json["invocation"] = serde_json::json!(invocation_for_record);
    report_json["harness_wall_time_ms"] = serde_json::json!(overall_start.elapsed().as_millis());
    report_json["issue"] = serde_json::json!(141);
    report_json["task"] = serde_json::json!("TASK-3.3b");
    report_json["notes"] = serde_json::json!([
        "self-repair run CLI（#142 差し戻し分で実装。PR #361）を 1 回起動する経路へ移行済み（基準 1）。",
        "self-repair verify-log CLI（外部コマンド経由。#145）でハッシュチェーン検証済み（基準 4・充足）。",
        "ベンチは RepairCompositeGate（TASK-3.2a・#137）による候補 diff 直接実測であり、baseline/候補それぞれの var.rs 実装差分を直接計測する（旧版の合成ワークロード完走確認とは異なる。基準 2・5・充足）。",
        "signal_source=measured（基準 6・充足）。",
        "検証対象は crates/autodiff 1 クレート（準備リポジトリとして workspace 継承から切り離した独立コピー）に限定し、実 workspace 全体は対象外（実行時間の理由。verify_gates_integration.rs と同じスコーピング）。",
    ]);
    let mut pretty = serde_json::to_string_pretty(&report_json).expect("JSON シリアライズに失敗");
    // `.editorconfig` の `insert_final_newline` 慣行に合わせ、commit 対象と
    // なりうる出力（`docs/` 側）に末尾改行を付与する。
    pretty.push('\n');
    std::fs::write(&output_path, &pretty).expect("loop-report.json の再書き込みに失敗");

    // リポジトリに commit 済みの記録（`docs/self-repair-revalidation/bug-fix/`）
    // の更新は環境変数 `SELF_REPAIR_TASK_3_3B_WRITE_DOCS=1` を明示指定した
    // 場合のみ行う（既定を `docs/` 直書きにすると `cargo test --workspace` を
    // 実行するたびに sandbox の一時パスを含む `invocation` が変わり、意図しない
    // tracked diff が毎回発生してしまうため。`feature_addition_loop_completion_
    // task_3_3c.rs` と同じ方針）。`self-repair run` を再実行せず、実際に 1 回
    // だけ起動した本実行の証跡をそのまま複製する。
    if std::env::var("SELF_REPAIR_TASK_3_3B_WRITE_DOCS").as_deref() == Ok("1") {
        let docs_out_dir = repo_root().join("docs/self-repair-revalidation/bug-fix");
        std::fs::create_dir_all(&docs_out_dir).expect("docs 出力先ディレクトリ作成に失敗");
        std::fs::copy(&output_path, docs_out_dir.join("loop-report.json"))
            .expect("loop-report.json の docs へのコピーに失敗");
        std::fs::copy(&log_path, docs_out_dir.join("loop-log.jsonl"))
            .expect("loop-log.jsonl の docs へのコピーに失敗");
    }

    // sandbox・候補 JSON は `_sandbox_guard`／`_candidates_guard` の Drop で
    // 削除される（正常終了・assert! panic のいずれの経路でも確実に削除する
    // ため、明示的な remove_dir_all／remove_file 呼び出しはここに置かない）。
}
