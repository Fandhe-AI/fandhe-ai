//! TASK-3.1c（イシュー #134）の受け入れ条件「検証 3 ゲート（build/test/clippy）が
//! 新 workspace で動作する」を検証する統合テスト。
//!
//! 実 cargo（`self_repair::SystemCommandRunner`）を用いて、一時ディレクトリに
//! 実行時生成した最小 fixture workspace（単一 bin クレート・外部依存なし）に対し
//! [`self_repair::CargoVerificationGate`] の 3 ゲートを実行する。実機（CUDA・
//! Metal）依存はないため `#[ignore]` 分離は不要（`.claude/rules/coding-rust.md`）。
//! 本リポ実 workspace 全体を対象とした完走実証は TASK-3.3（#139 系）のスコープで
//! あり本テストでは行わない（実装計画 6 章）。

use self_repair::stages::{Proposal, VerificationGate, VerificationOutcome};
use self_repair::{CargoVerificationGate, SystemCommandRunner};
use std::fs;

mod common;

use common::temp_dir::TempDirGuard;

/// 一時ディレクトリに fixture workspace を作る。`tests/common/` の `TempDirGuard`
/// （一意名＋排他作成・Drop で削除。イシュー #2382）で確保し、返すガードを保持する間だけ存在する。
/// 本リポの workspace（親 `Cargo.toml` の `[workspace]`）配下に置くと fixture の Cargo.toml が
/// ワークスペースメンバーとして誤認識されうるため、リポジトリ外の一時ディレクトリに作る。
fn fixture_workspace(name: &str, main_rs: &str) -> TempDirGuard {
    let guard = TempDirGuard::new(&format!("verify-gates-fixture-{name}"));
    let dir = guard.path().to_path_buf();
    // 自身が排他作成したディレクトリの中なので、`create_dir` で十分。
    fs::create_dir(dir.join("src")).expect("create_dir should succeed in test setup");

    fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    )
    .expect("write Cargo.toml should succeed in test setup");
    fs::write(dir.join("src/main.rs"), main_rs)
        .expect("write src/main.rs should succeed in test setup");

    guard
}

#[test]
fn all_gates_pass_for_valid_fixture_workspace() {
    let workspace = fixture_workspace(
        "pass",
        "fn main() {\n    println!(\"hello from fixture\");\n}\n\n\
         #[cfg(test)]\nmod tests {\n    #[test]\n    fn trivial() {\n        assert_eq!(1 + 1, 2);\n    }\n}\n",
    );

    let gate = CargoVerificationGate::new(
        workspace.path().to_path_buf(),
        SystemCommandRunner::new(),
        0,
        false,
        false,
        Vec::new(),
    );
    let proposal = Proposal {
        attempt: 1,
        description: "fixture: 全ゲート通過".to_string(),
    };

    let outcome = gate
        .verify(&proposal)
        .expect("verify should not error for a valid fixture");

    match outcome {
        VerificationOutcome::Passed(evidence) => {
            assert_eq!(evidence.attempt(), 1);
            assert_eq!(evidence.gate_report(), "build=pass test=pass clippy=pass");
        }
        VerificationOutcome::Failed { reason } => {
            panic!("expected all gates to pass, got Failed: {reason}")
        }
    }
}

#[test]
fn build_gate_fails_for_fixture_with_compile_error() {
    // 意図的な構文エラー（未定義変数の参照）で build ゲートを不合格にする。
    let workspace = fixture_workspace(
        "build-fail",
        "fn main() {\n    let _ = undefined_symbol_for_self_repair_test;\n}\n",
    );

    let gate = CargoVerificationGate::new(
        workspace.path().to_path_buf(),
        SystemCommandRunner::new(),
        0,
        false,
        false,
        Vec::new(),
    );
    let proposal = Proposal {
        attempt: 1,
        description: "fixture: build 失敗".to_string(),
    };

    let outcome = gate
        .verify(&proposal)
        .expect("verify should not error even when the gate fails");

    match outcome {
        VerificationOutcome::Failed { reason } => {
            assert!(reason.contains("build"));
        }
        VerificationOutcome::Passed(_) => {
            panic!("expected build gate to fail for a fixture with a compile error")
        }
    }
}
