//! ベンチゲートの候補 diff 直接実測（TASK-3.2a・イシュー #137）の統合テスト。
//!
//! [`self_repair::RepairCompositeGate`] を、`fixtures/feature-addition-leaky-relu/
//! baseline`（`crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs`
//! と同じ保守対象 fixture）を注入した使い捨て sandbox 上で 1 回 `verify` する。
//!
//! `feature_addition_loop_completion_task_3_3c.rs` はループ全体
//! （検出 → 修正試行 → 検証 → 取り込み）の完走実証であるのに対し、本テストは
//! `RepairCompositeGate::verify` 単体（候補 diff 直接ベンチ実測の受け入れ条件
//! 「ベンチゲートが bench-harness 経由で完走する」の実証）に焦点を絞る。
//! sandbox 構築ヘルパー（fixture コピー・path 依存の絶対パス書き換え・
//! `GIT_*` を除去した git init/commit）は `feature_addition_loop_completion_
//! task_3_3c.rs` の先例をそのまま踏襲する（実装計画 #137 §9 リスク
//! 「revalidation_bug_fix.rs の先例に倣う」と同じ再利用方針）。
//!
//! # 実行時間・分離方針
//! 各テストは release ビルドを baseline／candidate の 2 系統（Case A/C）行う
//! ため cold cache では相応に時間がかかる。通常 CI ジョブでは実行しない
//! （`#[ignore]`。`revalidation_bug_fix.rs`・`feature_addition_loop_completion_
//! task_3_3c.rs` と同じ運用）。実行:
//! `cargo test -p self-repair --test verify_direct_composite_integration -- --ignored --nocapture`

use std::path::{Path, PathBuf};
use std::process::Command;

use self_repair::stages::{Proposal, VerificationGate, VerificationOutcome};
use self_repair::{RepairCompositeGate, RepairCompositeGateSpec, SystemCommandRunner};

/// baseline フィクスチャの相対パス（`CARGO_MANIFEST_DIR` は本クレート
/// 〈`crates/self-repair`〉のルート）。
const FIXTURE_REL: &str = "tests/fixtures/feature-addition-leaky-relu/baseline";
/// ベンチワークロード bin のソース（ピン留め対象。`sandbox_root` 相対）。
const WORKLOAD_SOURCE: &str = "src/bin/bench_workload.rs";
const BENCH_BIN: &str = "bench_workload";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/self-repair の親は crates/")
        .parent()
        .expect("crates/ の親はリポジトリルート")
        .to_path_buf()
}

fn unique_sandbox_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "self-repair-verify-direct-composite-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("sandbox ディレクトリ作成に失敗");
    dir
}

/// sandbox を確実に削除する RAII ガード（`revalidation_bug_fix.rs::SandboxGuard`
/// と同じ理由・同じ設計。パニック時の unwind 経由でも削除される）。
struct SandboxGuard(PathBuf);

impl Drop for SandboxGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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

/// `feature_addition_loop_completion_task_3_3c.rs::rewrite_path_deps_to_absolute`
/// と同一ロジック（コピー先で相対深度が保たれないための絶対パス書き換え）。
fn rewrite_path_deps_to_absolute(sandbox: &Path) {
    let cargo_toml = sandbox.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).expect("sandbox Cargo.toml 読み取りに失敗");
    let tensor_core_abs = repo_root().join("crates/tensor-core");
    let autodiff_abs = repo_root().join("crates/autodiff");
    let rewritten = content
        .replace(
            r#"path = "../../../../../tensor-core""#,
            &format!(r#"path = "{}""#, tensor_core_abs.display()),
        )
        .replace(
            r#"path = "../../../../../autodiff""#,
            &format!(r#"path = "{}""#, autodiff_abs.display()),
        );
    assert_ne!(
        rewritten, content,
        "path 依存の書き換えが 1 件も適用されなかった（fixture Cargo.toml の記法が変わっていないか確認）"
    );
    std::fs::write(&cargo_toml, rewritten).expect("sandbox Cargo.toml 書き換えに失敗");
}

/// `feature_addition_loop_completion_task_3_3c.rs::sandboxed_git_command` と
/// 同一ロジック（`GIT_*` 環境変数の除去による sandbox 隔離。同モジュール
/// ドキュメント参照）。
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

fn run_git(sandbox: &Path, args: &[&str]) {
    let output = sandboxed_git_command(sandbox, args)
        .output()
        .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗: {error}"));
    assert!(
        output.status.success(),
        "git {args:?} が失敗しました: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn git_rev_parse_head(sandbox: &Path) -> String {
    let output = sandboxed_git_command(sandbox, &["rev-parse", "HEAD"])
        .output()
        .expect("git rev-parse HEAD の起動に失敗");
    assert!(output.status.success(), "git rev-parse HEAD が失敗しました");
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn git_init_baseline(sandbox: &Path) -> String {
    run_git(sandbox, &["init", "-q"]);
    run_git(sandbox, &["add", "-A"]);
    run_git(
        sandbox,
        &[
            "-c",
            "user.email=self-repair-137@example.invalid",
            "-c",
            "user.name=self-repair-137",
            "commit",
            "-q",
            "-m",
            "baseline (bench_workload fixture)",
        ],
    );
    git_rev_parse_head(sandbox)
}

/// sandbox 化した fixture のコピーを構築し、`baseline` commit sha を返す。
fn build_sandbox(name: &str) -> (SandboxGuard, PathBuf, String) {
    let sandbox = unique_sandbox_dir(name);
    let guard = SandboxGuard(sandbox.clone());
    let fixture_src = repo_root().join("crates/self-repair").join(FIXTURE_REL);
    copy_dir_recursive(&fixture_src, &sandbox);
    rewrite_path_deps_to_absolute(&sandbox);
    let baseline_commit = git_init_baseline(&sandbox);
    (guard, sandbox, baseline_commit)
}

fn gate_spec(
    sandbox: &Path,
    baseline_commit: &str,
) -> RepairCompositeGateSpec<SystemCommandRunner> {
    // `main.rs::run_run` と同じ契約: ポリシー除外設定は候補適用前に一度だけ
    // ロードし、不変値として `RepairCompositeGateSpec` へ渡す（PR #361
    // codex-review P1 指摘対応。`self_repair::diff_signals::
    // load_policy_exclusion_config` doc「呼び出し契約」参照）。
    let policy_exclusion = self_repair::diff_signals::load_policy_exclusion_config(
        &repo_root().join("policy-exclusion.toml"),
    )
    .expect("policy-exclusion.toml のロードに失敗");
    RepairCompositeGateSpec {
        workspace: sandbox.to_path_buf(),
        sandbox_root: sandbox.to_path_buf(),
        baseline_commit: baseline_commit.to_string(),
        policy_exclusion,
        bench_bin: BENCH_BIN.to_string(),
        workload_sources: vec![WORKLOAD_SOURCE.to_string()],
        bench_iterations: self_repair::verify_bench::MIN_BENCH_ITERATIONS,
        runner: SystemCommandRunner::new(),
    }
}

/// baseline の `#[cfg(test)] mod tests {` 開始位置を示すマーカー
/// （`feature_addition_loop_completion_task_3_3c.rs::MOD_TESTS_MARKER` と同一。
/// 既存テストへ触れず新規関数のみを挿入するための挿入位置）。
const MOD_TESTS_MARKER: &str = "#[cfg(test)]\nmod tests {";

/// 正実装の `leaky_relu` を `activations.rs` へ追加した候補内容を組み立てる
/// （`feature_addition_loop_completion_task_3_3c.rs::candidate2_correct_content`
/// と同一の合成方法。baseline は `leaky_relu` 未実装のため `tests/
/// leaky_relu_acceptance.rs` がコンパイルできず、これを解消しないと
/// `cargo test --release`〈`CargoVerificationGate` の test ゲート〉が失敗する）。
fn leaky_relu_candidate_content(baseline: &str) -> String {
    let with_import = baseline.replacen(
        "use autodiff::Var;\nuse autodiff::nn::activation::{Relu, Sigmoid};\n",
        "use autodiff::Tape;\nuse autodiff::Var;\nuse autodiff::nn::activation::{Relu, Sigmoid};\nuse tensor_core::Tensor;\n",
        1,
    );
    assert_ne!(
        with_import, baseline,
        "baseline の import ブロックが想定形式と異なる（fixture 更新時に本関数も更新すること）"
    );
    assert!(
        with_import.contains(MOD_TESTS_MARKER),
        "baseline に #[cfg(test)] mod tests が見つからない"
    );
    let leaky_relu_fn = "fn constant<'t>(tape: &'t Tape, value: f32) -> Var<'t> {\n\
         \x20   let t = Tensor::full(&[1], value).expect(\"constant: shape [1] は常に妥当\");\n\
         \x20   tape.var(&t)\n\
         }\n\n\
         /// Leaky ReLU 活性化関数（case_a: issue #137 統合テスト候補）。\n\
         pub fn leaky_relu<'t>(tape: &'t Tape, x: &Var<'t>, negative_slope: f64) -> Var<'t> {\n\
         \x20   let positive = relu(x);\n\
         \x20   let neg_one = constant(tape, -1.0);\n\
         \x20   let neg_positive = positive.mul(&neg_one).expect(\"leaky_relu: broadcast 可能\");\n\
         \x20   let negative_input = x.add(&neg_positive).expect(\"leaky_relu: 同一 shape\");\n\
         \x20   let slope = constant(tape, negative_slope as f32);\n\
         \x20   let negative_part = negative_input.mul(&slope).expect(\"leaky_relu: broadcast 可能\");\n\
         \x20   positive.add(&negative_part).expect(\"leaky_relu: 同一 shape\")\n\
         }\n";
    with_import.replacen(
        MOD_TESTS_MARKER,
        &format!("{leaky_relu_fn}\n{MOD_TESTS_MARKER}"),
        1,
    )
}

/// Case A（完走・受け入れ条件対応）: 候補 diff（`activations.rs` への
/// `leaky_relu` 正実装追加。`WORKLOAD_SOURCE` には触れない）を適用した sandbox
/// で `RepairCompositeGate::verify` が `Passed` を返し、ベンチが実測
/// （[`self_repair::verify_bench::BenchSignal`]・5 件以上・有限値）である
/// ことを確認する。
#[test]
#[ignore = "release ビルド 2 系統（baseline worktree・candidate）＋ cargo build/test/clippy を \
            実行するため長時間かかる。通常 CI ジョブでは実行しない。実行: \
            cargo test -p self-repair --test verify_direct_composite_integration \
            case_a -- --ignored --nocapture"]
fn case_a_harmless_candidate_diff_completes_with_measured_bench() {
    let (_guard, sandbox, baseline_commit) = build_sandbox("case-a");

    // 候補 diff: leaky_relu の正実装を追加する（`cargo test --release` の
    // 受け入れ基準テストを通過させ、build/test/clippy 3 ゲートが Passed に
    // なるようにする。ベンチワークロード〈`WORKLOAD_SOURCE`〉には触れない）。
    let activations_path = sandbox.join("src/activations.rs");
    let original = std::fs::read_to_string(&activations_path).expect("activations.rs 読み取り失敗");
    let patched = leaky_relu_candidate_content(&original);
    std::fs::write(&activations_path, patched).expect("activations.rs 書き込み失敗");

    let gate = RepairCompositeGate::new(gate_spec(&sandbox, &baseline_commit));
    let evidence_sink = gate.evidence_sink();
    let bench_sink = gate.bench_measurement_sink();

    let outcome = gate
        .verify(&Proposal {
            attempt: 1,
            description: "case_a: harmless comment".to_string(),
        })
        .expect("verify 自体はエラーにならないはず");

    match outcome {
        VerificationOutcome::Passed(evidence) => {
            // 新規 `pub fn leaky_relu` を追加するのみで既存 `pub fn relu`／
            // `pub fn sigmoid` のシグネチャは変更していないため、
            // `api_signature_touched`（`guardrail::checks::api_stability::
            // api_broken` と同一意味論。イシュー #142 差し戻し分で修正済み。
            // `diff_signals.rs` モジュール冒頭ドキュメント参照）は `false` に
            // なる想定（本テストの主眼は「試行ごとに実測されること」であり、
            // 値そのものは新規追加のみの候補で破壊なしとなることを併せて
            // 確認する）。
            assert!(evidence.lines_changed() > 0, "実装追加分の diff があるはず");
            assert!(
                !evidence.api_broken(),
                "既存シグネチャを維持したままの新規 pub fn 追加は破壊とみなさない想定"
            );
            assert!(!evidence.gaming_suspect(), "本番コードのみの変更のはず");
            assert!(evidence.gate_report().contains("bench=measured-direct"));
        }
        VerificationOutcome::Failed { reason } => {
            panic!("Case A は Passed を想定: {reason}")
        }
    }

    assert!(
        evidence_sink.borrow().is_some(),
        "evidence_sink に観測されているはず"
    );
    let bench = bench_sink
        .borrow()
        .clone()
        .expect("bench_measurement_sink に候補 diff 直接実測の結果が観測されているはず");
    assert!(
        bench.bench_measurements_pct.len() >= self_repair::verify_bench::MIN_BENCH_ITERATIONS,
        "反復回数は下限（5 回）以上のはず: {:?}",
        bench.bench_measurements_pct
    );
    assert!(
        bench.bench_median_pct.is_finite(),
        "劣化率中央値は有限値のはず"
    );
}

/// Case B（ゲーミング防止）: 候補 diff がベンチワークロードソース
/// （`bench_workload.rs`）自体を改変した場合、`RepairCompositeGate::verify`
/// が `Err`（fail-closed）を返すことを確認する（実装計画 §3.2）。
#[test]
#[ignore = "cargo build/test/clippy を実行するため長時間かかる（ピン留め違反は \
            3 ゲート通過後に検出される設計のため）。通常 CI ジョブでは実行しない。実行: \
            cargo test -p self-repair --test verify_direct_composite_integration \
            case_b -- --ignored --nocapture"]
fn case_b_workload_source_tampering_is_rejected_fail_closed() {
    let (_guard, sandbox, baseline_commit) = build_sandbox("case-b");

    // build/test/clippy 3 ゲートを通過させるため、Case A と同じ leaky_relu
    // 正実装をまず適用する（ピン留め違反は 3 ゲート通過後に検出される設計
    // のため。`verify_direct_composite.rs` の実行順序ドキュメント参照）。
    let activations_path = sandbox.join("src/activations.rs");
    let original_activations =
        std::fs::read_to_string(&activations_path).expect("activations.rs 読み取り失敗");
    let patched_activations = leaky_relu_candidate_content(&original_activations);
    std::fs::write(&activations_path, patched_activations).expect("activations.rs 書き込み失敗");

    // 候補がベンチワークロードのソース自体を改変（「軽くして速く見せる」
    // ゲーミングの単純化した再現。実際の内容変更ではなくコメント追加のみで
    // 十分——ピン留め検査はファイル内容の diff 有無のみを見る）。
    let workload_path = sandbox.join(WORKLOAD_SOURCE);
    let original = std::fs::read_to_string(&workload_path).expect("bench_workload.rs 読み取り失敗");
    let tampered = format!("// case_b: tampered by candidate (issue #137)\n{original}");
    std::fs::write(&workload_path, tampered).expect("bench_workload.rs 書き込み失敗");

    let gate = RepairCompositeGate::new(gate_spec(&sandbox, &baseline_commit));
    let result = gate.verify(&Proposal {
        attempt: 1,
        description: "case_b: tampered workload".to_string(),
    });

    let error = result.expect_err("ワークロードソース改竄は fail-closed で拒否されるはず");
    let message = error.to_string();
    assert!(
        message.contains("直接ベンチ実測に失敗"),
        "ベンチ実測段階のエラーであるはず: {message}"
    );
}

/// `relu` を「関数的には等価だが 10 倍の再計算を行う」実装へ置き換えた候補
/// diff を組み立てる（Case C 専用。完走判定基準 5「候補 diff に対する直接
/// 実測」の実証: baseline と候補で異なる計算量の実装を比較し、劣化率が
/// 検出されることを確認するため）。`Relu.forward` の冪等性
/// （`relu(relu(x)) == relu(x)`。負値は 0 のまま、非負値はそのまま通る）を
/// 利用し、forward 値・勾配のいずれも変えずにテープのノード数のみを 10 倍に
/// 増やす。これにより `relu_matches_known_values`／`sigmoid_matches_known_values`
/// （`activations.rs` 自身の既知値テスト）と `leaky_relu_matches_known_values`
/// （`leaky_relu` は `relu` の合成のため連鎖律の指示関数の積が単一の指示関数と
/// 一致し、勾配も不変）はいずれも通過したまま、build/test/clippy 3 ゲートを
/// 通過させつつ、候補 diff（`relu` の実装のみ）に固有の性能劣化を発生させる。
fn slow_relu_candidate_content(baseline_with_leaky_relu: &str) -> String {
    let original_relu_body = "pub fn relu<'t>(x: &Var<'t>) -> Var<'t> {\n    Relu.forward(x)\n}\n";
    assert!(
        baseline_with_leaky_relu.contains(original_relu_body),
        "baseline の relu 実装が想定形式と異なる（fixture 更新時に本関数も更新すること）"
    );
    let slow_relu_body = "pub fn relu<'t>(x: &Var<'t>) -> Var<'t> {\n    \
         // case_c（issue #137 統合テスト候補）: relu(relu(x)) == relu(x) の冪等性を\n    \
         // 利用し、値・勾配を変えずにテープのノード数のみ 10 倍にする（候補固有の\n    \
         // 性能劣化を意図的に注入する検証専用の実装）。\n    \
         let mut out = Relu.forward(x);\n    \
         for _ in 0..9 {\n        \
         out = Relu.forward(&out);\n    \
         }\n    \
         out\n}\n";
    baseline_with_leaky_relu.replacen(original_relu_body, slow_relu_body, 1)
}

/// Case C（候補固有の劣化検出。完走判定基準 5 対応）: baseline と候補で
/// 異なる `relu` 実装（候補は関数的に等価だが 10 倍のノード数）を比較すると、
/// `DirectBenchRunner`（候補 diff 直接実測）の中央値劣化率が
/// `guardrail::Thresholds::builtin(PresetName::Default).bench_median_max_pct`
/// （読み取るのみ・閾値は変更しない）を大きく超過することを確認する
/// （実装計画 §5 Step 5 Case C・#139 reopen コメントの直接目的）。
#[test]
#[ignore = "release ビルド 2 系統（baseline worktree・candidate）＋ cargo build/test/clippy を \
            実行するため長時間かかる。通常 CI ジョブでは実行しない。実行: \
            cargo test -p self-repair --test verify_direct_composite_integration \
            case_c -- --ignored --nocapture"]
fn case_c_candidate_specific_slowdown_is_detected_by_direct_measurement() {
    let (_guard, sandbox, baseline_commit) = build_sandbox("case-c");

    let activations_path = sandbox.join("src/activations.rs");
    let original_activations =
        std::fs::read_to_string(&activations_path).expect("activations.rs 読み取り失敗");
    let with_leaky_relu = leaky_relu_candidate_content(&original_activations);
    let with_slow_relu = slow_relu_candidate_content(&with_leaky_relu);
    std::fs::write(&activations_path, with_slow_relu).expect("activations.rs 書き込み失敗");

    let gate = RepairCompositeGate::new(gate_spec(&sandbox, &baseline_commit));
    let outcome = gate
        .verify(&Proposal {
            attempt: 1,
            description: "case_c: relu with 10x redundant recomputation".to_string(),
        })
        .expect("verify 自体はエラーにならないはず（既知値テストは変わらず通過する）");

    let VerificationOutcome::Passed(_) = outcome else {
        panic!("Case C は build/test/clippy を通過し Passed になる想定（既知値は不変）");
    };

    let bench_sink = {
        // `RepairCompositeGate` は `verify` 内で `last_bench_measurement` へ
        // 書き込む（`evidence_sink`/`bench_measurement_sink` は `verify` 呼び出し
        // 前に取得する契約だが、本テストは単発 `verify` のみのため事後取得でも
        // 同じ `Rc` を経由して観測できる）。
        gate.bench_measurement_sink()
    };
    let bench = bench_sink
        .borrow()
        .clone()
        .expect("bench_measurement_sink に候補 diff 直接実測の結果が観測されているはず");

    let thresholds = guardrail::Thresholds::builtin(guardrail::PresetName::Default);
    assert!(
        bench.bench_median_pct > thresholds.bench_median_max_pct,
        "候補固有の 10 倍再計算は閾値（{}%）を大きく超える劣化率として検出される想定: 実測中央値={}%",
        thresholds.bench_median_max_pct,
        bench.bench_median_pct
    );
}
