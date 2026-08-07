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
use std::path::{Path, PathBuf};

/// 一時ディレクトリに fixture workspace を作る（`bench_gate_completion.rs` 等
/// 既存の統合テストと同様、`temp_dir()` + `process::id()` で並列テスト実行時の
/// 衝突を避ける）。本リポの workspace（親 `Cargo.toml` の `[workspace]`）配下に
/// 置くと fixture の Cargo.toml がワークスペースメンバーとして誤認識されうる
/// ため、リポジトリ外の一時ディレクトリに作る。
fn fixture_workspace(name: &str, main_rs: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "self-repair-verify-gates-fixture-{name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");

    fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    )
    .expect("write Cargo.toml should succeed in test setup");
    fs::write(dir.join("src/main.rs"), main_rs)
        .expect("write src/main.rs should succeed in test setup");

    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn all_gates_pass_for_valid_fixture_workspace() {
    let workspace = fixture_workspace(
        "pass",
        "fn main() {\n    println!(\"hello from fixture\");\n}\n\n\
         #[cfg(test)]\nmod tests {\n    #[test]\n    fn trivial() {\n        assert_eq!(1 + 1, 2);\n    }\n}\n",
    );

    let gate = CargoVerificationGate::new(workspace.clone(), SystemCommandRunner::new());
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

    cleanup(&workspace);
}

#[test]
fn build_gate_fails_for_fixture_with_compile_error() {
    // 意図的な構文エラー（未定義変数の参照）で build ゲートを不合格にする。
    let workspace = fixture_workspace(
        "build-fail",
        "fn main() {\n    let _ = undefined_symbol_for_self_repair_test;\n}\n",
    );

    let gate = CargoVerificationGate::new(workspace.clone(), SystemCommandRunner::new());
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

    cleanup(&workspace);
}
