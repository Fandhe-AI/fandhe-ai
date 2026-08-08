//! TASK-3.3c（イシュー #142）: 機能追加種別の自己修復ループを
//! `self-repair run` CLI 経由で 1 回実行し、実測結果を記録する。
//!
//! REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）「自作コアに
//! 対する自己修復ループの人間介在なし完走を新実装リポで再実証する」を、PoC-2
//! 検証題材 (c)（`leaky_relu` 新規実装）で 1 ループ実測する。題材選定・完走
//! 判定基準は TASK-3.3a（#140・PR #322）で人間承認済み
//! （`docs/self-repair-revalidation-plan.md` §4.2）。
//!
//! # CLI 経由への移行（#139 (c) 差し戻し対応）と、実測で判明した基準 1 未充足
//! 旧版（PR #338）は **lib API 直接呼び出しの実証ハーネス**として
//! [`self_repair::SelfRepairLoop`] を直接構築していたが、#139 reopen コメント
//! の判断 (c) 差し戻しにより、完走判定基準 1・4・5
//! （`docs/self-repair-revalidation-plan.md` §5）を CLI 経由・
//! `self-repair verify-log` 外部コマンド経由・候補 diff 直接実測で満たすこと
//! が求められた（実装計画 #142 §3.2）。本ファイルは
//! `env!("CARGO_BIN_EXE_self-repair")` で実バイナリを **1 回だけ起動**し、
//! 続けて `self-repair verify-log` を別プロセスとして起動する。sandbox 準備
//! （fixture コピー・path 依存絶対化・`GIT_*` 除去 git init）・候補生成内容
//! （`candidate1_wrong_content`／`candidate2_correct_content`。#140 承認済み
//! 題材）は旧版から変更していない。
//!
//! 実装後に実行して判明した事実として、**基準 1（1 回起動・exit 0 到達）は
//! 現行のガードレール判定では未充足**である。旧版の lib 直接呼び出しハーネスは
//! 構築時固定の diff 由来シグナル（`FeatureAdditionCompositeGate`）を使って
//! おり `api_broken` を独自算出していたため Adopted に到達していたが、
//! CLI 経由（`RepairCompositeGate`・#137）は `diff_signals.rs::
//! api_signature_touched` の実測を使う。これは追加・削除いずれの `pub fn`
//! も「API 破壊」とみなすヒューリスティック（`tests/
//! verify_direct_composite_integration.rs` の doc で「新規 pub fn 追加は
//! ヒューリスティック上検出される想定」と既に明記されている既存仕様）であり、
//! PoC-2 題材 (c) の受け入れ基準は `pub fn leaky_relu` の追加を必須とするため、
//! この構成では機能追加種別が自動適用（Adopted・exit 0）へ到達する経路が
//! 存在しない。詳細は
//! [`feature_addition_loop_reaches_escalation_with_measured_evidence`] の
//! ドキュメンテーションコメント参照。ガードレール判定・除外リストの変更は
//! ユーザー承認必須（`.claude/rules/security.md`）のため本イシューでは
//! 変更していない。
//!
//! # シグナルは実測のみ（捏造しない）
//! `lines_changed`／`exclusion_rule_ids`／`gaming_suspect`／ベンチは、CLI
//! バイナリ内部の [`self_repair::RepairCompositeGate`]（TASK-3.2a・#137）が
//! 試行ごとに sandbox 内の使い捨て git リポジトリを対象に実測する
//! （`git diff --numstat`・`git diff --unified=0`・`guardrail::policy_exclusion`・
//! `DirectBenchRunner` による候補 diff 直接ベンチ実測）。旧版
//! （`FeatureAdditionCompositeGate`。合成ワークロード・構築時固定シグナル）
//! とは異なり、ベンチは baseline／候補それぞれの実装差分を直接計測する
//! （完走判定基準 5）。未計測値を fail-open な既定値で埋めない
//! （`.claude/rules/security.md` A08）。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// baseline フィクスチャの相対パス（`crates/self-repair/tests/fixtures/…`。
/// `CARGO_MANIFEST_DIR` は本クレート〈`crates/self-repair`〉のルート）。
const FIXTURE_REL: &str = "tests/fixtures/feature-addition-leaky-relu/baseline";
/// 機能追加候補が書き換える唯一のファイル（baseline に実在し、新規ファイル
/// 追加にならないこと。`FeatureAdditionFixGenerator::new` の構築時検証対象）。
const TARGET_FILE: &str = "src/activations.rs";
/// `TARGET_FILE` 内の `#[cfg(test)] mod tests` 開始位置を示すマーカー。
/// `candidate_content`（新規関数の挿入位置）と `mod_tests_start_line`
/// （境界検出の純関数テスト。`boundary_detection_tests`）の両方が同一の
/// マーカーを参照することを保証する。
const MOD_TESTS_MARKER: &str = "#[cfg(test)]\nmod tests {";
/// ベンチワークロード bin（`RepairCompositeGate` の候補 diff 直接実測が
/// ピン留め・ビルド対象とする。`tests/verify_direct_composite_integration.rs`
/// と同一の fixture 内 bin を再利用する）。
const WORKLOAD_SOURCE: &str = "src/bin/bench_workload.rs";
const BENCH_BIN: &str = "bench_workload";

/// sandbox の baseline（`TARGET_FILE`）へ `leaky_relu` を追加した候補内容を
/// 組み立てる。既存のインポート・`relu`/`sigmoid`・`#[cfg(test)] mod tests`
/// （既存の `assert!`／`abs() <`／`1e-6` 等を含む既存テスト）は一切変更せず、
/// 新規関数の挿入のみを行う（差分を「新規追加」のみに限定することで、
/// `guardrail::policy_exclusion` の `test-tolerance-loosening` ルール
/// （既存の許容誤差リテラルの緩和を検知するルール）が意図せず match しない
/// ようにする。候補が既存テストを巻き込んで書き換えると、ルールが「テスト
/// 側の変更」と「本番コード側の変更」の境界を判別できず安全側
/// エスカレーションへ倒れる。実装時に実測して判明した挙動であり、
/// `crates/guardrail/src/exclusion_match.rs` の `test_assertion_relaxation_
/// without_prod_change` doc 参照）。
fn candidate_content(leaky_relu_fn: &str) -> String {
    let baseline = std::fs::read_to_string(
        repo_root()
            .join("crates/self-repair")
            .join(FIXTURE_REL)
            .join(TARGET_FILE),
    )
    .expect("baseline fixture ファイルは実在する");

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
    with_import.replacen(
        MOD_TESTS_MARKER,
        &format!("{leaky_relu_fn}\n{MOD_TESTS_MARKER}"),
        1,
    )
}

/// `constant` ヘルパー（両候補で共通）: shape `[1]` の定数 `Var` を `tape` へ
/// 葉ノードとして登録する。`Var::add`/`Var::mul` の broadcast により任意
/// shape の `Var` と組み合わせられる
/// （`crates/guardrail/tests/fixtures/labeled-changes/baseline/src/
/// activations.rs::constant` と同一の合成手法）。
const CONSTANT_HELPER: &str = r#"fn constant<'t>(tape: &'t Tape, value: f32) -> Var<'t> {
    let t = Tensor::full(&[1], value).expect("constant: shape [1] は常に妥当");
    tape.var(&t)
}

"#;

/// 試行 1: 誤実装（PoC-2 題材 (c) の「実装試行 1」。符号分岐を欠き、正の
/// 入力にも negative_slope を一様適用してしまうバグ。既知正解値の
/// `x=0.5 → 0.5` に対し `0.05` を返すため受け入れ基準テストで不合格になる）。
fn candidate1_wrong_content() -> String {
    let leaky_relu_fn = format!(
        "{CONSTANT_HELPER}/// バグ: 符号分岐（`x >= 0` なら `x` をそのまま通す）を欠き、全要素へ\n\
         /// `negative_slope` を一様適用してしまう（PoC-2 題材 (c) の「実装試行 1」）。\n\
         pub fn leaky_relu<'t>(tape: &'t Tape, x: &Var<'t>, negative_slope: f64) -> Var<'t> {{\n\
         \x20   let slope = constant(tape, negative_slope as f32);\n\
         \x20   x.mul(&slope).expect(\"leaky_relu: shape [1] は常に broadcast 可能\")\n\
         }}\n"
    );
    candidate_content(&leaky_relu_fn)
}

/// 試行 2: 正実装（`relu(x) + negative_slope * (x - relu(x))` として合成。
/// `crates/guardrail/tests/fixtures/labeled-changes/baseline/src/activations.rs`
/// と同一の合成方法。既存組み込み演算の合成のみで構成され、新規レイヤー・
/// 新規依存を追加しない）。
fn candidate2_correct_content() -> String {
    let leaky_relu_fn = format!(
        "{CONSTANT_HELPER}/// Leaky ReLU 活性化関数: `x >= 0` なら `x`、`x < 0` なら `negative_slope * x`。\n\
         /// `relu(x) + negative_slope * (x - relu(x))` として合成する\n\
         /// （`x - relu(x)` は `x` の負部分のみを残す）。\n\
         pub fn leaky_relu<'t>(tape: &'t Tape, x: &Var<'t>, negative_slope: f64) -> Var<'t> {{\n\
         \x20   let positive = relu(x);\n\
         \x20   let neg_one = constant(tape, -1.0);\n\
         \x20   let neg_positive = positive\n\
         \x20       .mul(&neg_one)\n\
         \x20       .expect(\"leaky_relu: shape [1] は常に broadcast 可能\");\n\
         \x20   let negative_input = x\n\
         \x20       .add(&neg_positive)\n\
         \x20       .expect(\"leaky_relu: relu(x) と x は同一 shape\");\n\
         \x20   let slope = constant(tape, negative_slope as f32);\n\
         \x20   let negative_part = negative_input\n\
         \x20       .mul(&slope)\n\
         \x20       .expect(\"leaky_relu: shape [1] は常に broadcast 可能\");\n\
         \x20   positive\n\
         \x20       .add(&negative_part)\n\
         \x20       .expect(\"leaky_relu: positive と negative_part は同一 shape\")\n\
         }}\n"
    );
    candidate_content(&leaky_relu_fn)
}

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
/// のため `pub(crate)` ヘルパーを再利用できず、同型のヘルパーを再実装する）。
fn unique_sandbox_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "self-repair-feature-addition-task-3-3c-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("sandbox ディレクトリ作成に失敗");
    dir
}

/// 単一ファイル用の一時パス（`--candidates` JSON 出力先。`unique_sandbox_dir`
/// と同じ `temp_dir() + process::id()` 方式だがディレクトリではなくファイル）。
fn unique_temp_file(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "self-repair-feature-addition-task-3-3c-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// `src` 配下を再帰的に `dst` へコピーする（`target/`・`.git/` は対象外。
/// baseline フィクスチャは `.gitignore` に `/target` のみを持つため通常
/// 存在しないが、ローカルでビルド済みの場合に備えて明示的に除外する）。
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

/// sandbox の `Cargo.toml` が持つ `tensor-core`/`autodiff` への相対 path 依存
/// （`../../../../../tensor-core` 等）を、コピー先では深度が保たれないため
/// 絶対パスへ書き換える（実装計画 §5 ステップ 1・advisor 指摘）。
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

/// sandbox 内でのみ完結する `git` コマンドを構築する。
///
/// `current_dir(sandbox)` だけでは実リポジトリへの誤動作を防げない。本
/// クレートの `git commit`／`git diff` は `lefthook.yml` の `pre-push.jobs.test`
/// （`cargo test --workspace`）経由で間接的に実行されうるが、githooks(5) の
/// 仕様どおり git はフック起動時に `GIT_DIR`／`GIT_WORK_TREE`／
/// `GIT_INDEX_FILE` 等の `GIT_*` 環境変数を子プロセスへ設定し、それらは
/// `cargo test` → 本テストバイナリ → `Command::new("git")` まで継承され
/// うる。継承された `GIT_DIR` は `current_dir` より優先されるため、
/// これを除去しないと sandbox 内のつもりの `git init`／`git commit` が
/// 実リポジトリの `.git` を対象にしてしまう（2026-08-07 実測: 本関数
/// 対応前に sandbox の `git commit` が実リポジトリの現在ブランチ HEAD へ
/// 実際にコミットしてしまう事故が発生した。#149 PR 対応時に発見・修正。
/// 該当コミットは `git reset --hard` で復旧済み）。sandbox を実リポジトリ
/// から完全に独立させるため、継承されうる `GIT_*` 環境変数をすべて明示的
/// に除去してから起動する。`self-repair run` バイナリ本体（`main.rs::
/// resolve_baseline_commit`）にも同じ隔離を実装済み（advisor 指摘: 本番経路
/// にも同じリスクがあるため）。
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

/// `sandbox` を使い捨て git リポジトリ化し、現在の内容を `baseline` コミット
/// として記録する（`RepairCompositeGate` の diff 由来シグナル実測用。実
/// リポジトリの git 履歴とは独立しており push・fetch は一切行わない。
/// [`sandboxed_git_command`] のドキュメンテーションコメント参照）。
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
        "user.email=self-repair-task-3-3c@example.invalid",
        "-c",
        "user.name=self-repair-task-3-3c",
        "commit",
        "-q",
        "-m",
        "baseline (leaky_relu unimplemented)",
    ]);
}

/// baseline 内容から `MOD_TESTS_MARKER`（`#[cfg(test)]\nmod tests {`）の
/// 開始行番号（1-indexed）を求める。`git diff` ハンクヘッダの行番号
/// （1-indexed）と直接比較できるようにするための変換。`candidate_content`
/// の挿入位置判定と同一のマーカー定数（`MOD_TESTS_MARKER`）を参照する
/// ことで、境界の意味が 2 箇所でずれる回帰を防ぐ。
fn mod_tests_start_line(baseline: &str) -> usize {
    let marker_offset = baseline
        .find(MOD_TESTS_MARKER)
        .expect("baseline に #[cfg(test)] mod tests マーカーが見つからない");
    baseline[..marker_offset].matches('\n').count() + 1
}

/// `git diff --unified=0` の出力を解析し、旧ファイル側で変更された行範囲の
/// いずれかが `boundary_line`（1-indexed。`#[cfg(test)] mod tests` の開始行）
/// **以降の既存行**と重なるかを判定する（ハンクヘッダ `@@ -a,b +c,d @@` の
/// `-a,b` を旧ファイル側の変更行範囲として読む）。
///
/// 純追加ハンク（`b == 0`）は旧ファイルの行を 1 行も変更しない（`a` 行の
/// 直後に新規行を挿入するのみ）。本ハーネスの候補生成は `mod tests` marker
/// の直前（`a == boundary_line - 1`）に新規関数を挿入する構成であり、これは
/// `mod tests` 自体の内容には触れない「隣接挿入」であるため境界内とは
/// 判定しない（`a >= boundary_line` の場合のみ、挿入位置が `mod tests`
/// 本文の内部にあるとみなし境界内と判定する）。
///
/// # 本関数の現在の利用範囲
/// `RepairCompositeGate`（`crate::diff_signals::gaming_suspect_from_files` 等）
/// への移行後、本ハーネス自体は本関数を診断目的では呼ばなくなった
/// （ゲーミング疑いの実測は CLI バイナリ内部の `diff_signals.rs` が担う）。
/// 境界判定ロジックの純関数レベルでの回帰検知（`boundary_detection_tests`）
/// としての価値は変わらないため、テスト対象としてそのまま残す。
fn diff_touches_boundary(unified_diff: &str, boundary_line: usize) -> bool {
    unified_diff
        .lines()
        .filter(|line| line.starts_with("@@"))
        .any(|hunk_header| {
            let Some(old_range) = hunk_header.split(' ').find(|token| token.starts_with('-'))
            else {
                return false;
            };
            let old_range = old_range.trim_start_matches('-');
            let mut parts = old_range.splitn(2, ',');
            let start: usize = match parts.next().and_then(|s| s.parse().ok()) {
                Some(value) => value,
                None => return false,
            };
            let count: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            if count == 0 {
                start >= boundary_line
            } else {
                start + count > boundary_line
            }
        })
}

/// `mod_tests_start_line`／`diff_touches_boundary` の境界判定を、実際の
/// git 実行なしで検証する（純関数の単体テスト。advisor 指摘: 挙動を green
/// 化する前後で「調整しただけ」か「正しい境界判定」かを区別できるようにする）。
#[cfg(test)]
mod boundary_detection_tests {
    use super::{diff_touches_boundary, mod_tests_start_line};

    /// 実 fixture（`crates/self-repair/tests/fixtures/…/src/activations.rs`）
    /// の `#[cfg(test)]` は 24 行目にある（`grep -n` で実測済み）。
    #[test]
    fn mod_tests_start_line_matches_real_fixture() {
        let baseline = std::fs::read_to_string(
            super::repo_root()
                .join("crates/self-repair")
                .join(super::FIXTURE_REL)
                .join(super::TARGET_FILE),
        )
        .expect("baseline fixture ファイルは実在する");
        assert_eq!(mod_tests_start_line(&baseline), 24);
    }

    /// 本ハーネスが実際に生成する形（`mod tests` marker 直前への隣接挿入）
    /// は境界内と判定しない。
    #[test]
    fn adjacent_insertion_before_marker_is_not_boundary_touch() {
        let diff = "@@ -23,0 +24,15 @@\n";
        assert!(!diff_touches_boundary(diff, 24));
    }

    /// `mod tests` 本文の内部への挿入は境界内と判定する。
    #[test]
    fn insertion_inside_mod_tests_is_boundary_touch() {
        let diff = "@@ -30,0 +45,3 @@\n";
        assert!(diff_touches_boundary(diff, 24));
    }

    /// 既存行 1 行の改変（`assert!` 緩和等を想定）は境界内と判定する。
    #[test]
    fn single_line_change_inside_mod_tests_is_boundary_touch() {
        let diff = "@@ -30 +30 @@\n";
        assert!(diff_touches_boundary(diff, 24));
    }

    /// 境界直前で終わる改変（`mod tests` に到達しない）は境界内と判定しない。
    #[test]
    fn change_ending_before_boundary_is_not_boundary_touch() {
        let diff = "@@ -20,3 +20,3 @@\n";
        assert!(!diff_touches_boundary(diff, 24));
    }

    /// 境界を跨ぐ改変は境界内と判定する。
    #[test]
    fn change_spanning_boundary_is_boundary_touch() {
        let diff = "@@ -20,11 +20,11 @@\n";
        assert!(diff_touches_boundary(diff, 24));
    }

    /// ハンクを含まない diff（無変更）は境界内と判定しない。
    #[test]
    fn empty_diff_is_not_boundary_touch() {
        assert!(!diff_touches_boundary("", 24));
    }
}

fn self_repair_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_self-repair"))
}

/// 完走判定基準 1（`docs/self-repair-revalidation-plan.md` §5「1 回起動・
/// exit 0」）が現行実装では未充足であることが判明したため（実行して実測。
/// 下記コメント参照）、`#[ignore]` で分離する（旧版〈PR #338〉には
/// `#[ignore]` がなかったが、通過させたい主張〈完走〉と実際の到達点
/// 〈エスカレーション〉が食い違う状態で通常 CI に緑を出さないための意図的な
/// 分離である。実行: `cargo test -p self-repair --test
/// feature_addition_loop_completion_task_3_3c -- --ignored --nocapture`）。
///
/// # 基準 1 未充足の原因（実測で判明。ガードレール判定は変更していない）
/// `crates/self-repair/src/diff_signals.rs::api_signature_touched` は追加・
/// 削除いずれの `pub fn` 行も「API 破壊」として検出するヒューリスティックで
/// あり（`tests/verify_direct_composite_integration.rs::
/// case_a_harmless_candidate_diff_completes_with_measured_bench` の doc が
/// 「新規 pub fn 追加はヒューリスティック上検出される想定」と明記済み）、
/// `judge.rs::api_broken_yields_escalate` は `api_broken=true` を無条件で
/// Escalate へ写像する。PoC-2 題材 (c) の受け入れ基準（`tests/
/// leaky_relu_acceptance.rs`）は `pub fn leaky_relu` の追加を要求するため、
/// この構成では機能追加種別の候補が Adopted（exit 0）へ到達する経路が
/// 存在しない。ヒューリスティックの精緻化（追加と削除を区別する等）・
/// 除外リストの追加はいずれもガードレール判定・除外リストの変更であり
/// ユーザー承認必須（`.claude/rules/security.md`）のため本イシューでは
/// 行わない。基準 1 の扱い（許容する／題材か判定器を見直す）はユーザー判断
/// 事項として summary へ記録する。
#[test]
#[ignore = "基準1(exit 0)未充足: api_signature_touched が新規 pub fn 追加を \
            api_broken=true とし judge.rs が無条件 Escalate するため、PoC-2 \
            題材(c)は現行ガードレール判定で Adopted に到達不能（実測確認済み）。\
            #142 reopen 事項としてユーザー判断待ち。他の基準(2/4/5/6)はこの \
            テストで検証済み"]
fn feature_addition_loop_reaches_escalation_with_measured_evidence() {
    let overall_start = Instant::now();

    // --- sandbox 準備 ---
    let fixture_src = repo_root().join("crates/self-repair").join(FIXTURE_REL);
    let sandbox = unique_sandbox_dir("sandbox");
    copy_dir_recursive(&fixture_src, &sandbox);
    rewrite_path_deps_to_absolute(&sandbox);
    git_init_baseline(&sandbox);

    // --- 候補列（#140 承認済み題材）を `--candidates` JSON へ書き出す ---
    let candidates_json = serde_json::json!([
        {
            "description": "試行1（誤実装・符号分岐なし）",
            "files": [{"path": TARGET_FILE, "content": candidate1_wrong_content()}],
        },
        {
            "description": "試行2（正実装・既存組み込み演算の合成）",
            "files": [{"path": TARGET_FILE, "content": candidate2_correct_content()}],
        },
    ]);
    let candidates_path = unique_temp_file("candidates.json");
    std::fs::write(
        &candidates_path,
        serde_json::to_string_pretty(&candidates_json).expect("候補 JSON のシリアライズに失敗"),
    )
    .expect("候補 JSON の書き込みに失敗");

    // --- `self-repair run` を 1 回だけ起動する（完走判定基準 1） ---
    let target_out_dir = repo_root().join("target/self-repair-revalidation/feature-addition");
    std::fs::create_dir_all(&target_out_dir).expect("完走ログ出力先ディレクトリ作成に失敗");
    let log_path = target_out_dir.join("loop-log.jsonl");
    // `LogWriter::open` は既存ファイルへ追記継続するため、固定ファイル名を
    // 実行のたび削除してから開き「このディレクトリはこの 1 回の実行を記述
    // する」契約を `loop-report.json` の上書きと揃える（旧版と同じ方針）。
    let _ = std::fs::remove_file(&log_path);
    let output_path = target_out_dir.join("loop-report.json");

    let policy_exclusion_path = repo_root().join("policy-exclusion.toml");
    let run_args: Vec<String> = vec![
        "run".to_string(),
        "--kind".to_string(),
        "feature-addition".to_string(),
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
        "--policy-exclusion".to_string(),
        policy_exclusion_path.display().to_string(),
    ];
    let run_output = self_repair_bin()
        .args(&run_args)
        .output()
        .expect("self-repair run の起動に失敗");
    // --- exit code は基準 1 が未充足（本関数冒頭ドキュメント参照）なため
    //     `Some(0)` を主張しない。段階の実行自体が失敗（usage エラー=2・
    //     内部エラー=1）していないこと（＝終端 verdict に到達したこと）のみ
    //     確認し、実際の値は後段の `--output` JSON 検証で照合する。
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

    // 基準 1 が未充足である実測結果（Escalated。本関数冒頭ドキュメント参照）を
    // そのまま固定する。ここを `Adopted` に書き換えて green 化することは
    // #139 で既に拒否された「(b) 条件付き充足の独断選択」と同種の判断のため
    // 行わない。
    assert!(
        report_json["outcome"]
            .as_str()
            .is_some_and(|s| s.starts_with("Escalated")),
        "現行のガードレール判定では api_broken=true により Escalated が実測結果のはず（本関数冒頭ドキュメント参照。判定を変えていないため予期せぬ Adopted/Rejected はコード側の回帰を疑う）: {report_json}"
    );
    assert_eq!(
        run_exit_code,
        Some(10),
        "Escalated の終了コードは 10 のはず（3.5 節）"
    );
    assert_eq!(
        report_json["attempt_count"], 2,
        "試行1（検証不合格）→試行2（検証通過するが api_broken でエスカレーション）の 2 試行系列であるはず: {report_json}"
    );
    let attempts = report_json["attempts"]
        .as_array()
        .expect("attempts は配列のはず");
    assert!(
        attempts[0]["outcome"]
            .as_str()
            .is_some_and(|s| s.contains("VerificationFailed")),
        "試行1は受け入れ基準テスト不合格で検証不合格になるはず（完走判定基準 2 の一部）: {:?}",
        attempts[0]
    );
    assert!(
        attempts[1]["outcome"]
            .as_str()
            .is_some_and(|s| s.contains("Escalated")),
        "試行2は検証通過後に api_broken でエスカレーションされるはず: {:?}",
        attempts[1]
    );
    assert_eq!(
        report_json["signal_source"], "measured",
        "--signals 契約検証パスを経由しない実シグナル計測であるはず（完走判定基準 6）: {report_json}"
    );

    // ベンチが `NotRun` に丸められず候補 diff 直接実測されたことを確認する
    // （完走判定基準 5。試行2は build/test/clippy 3 ゲートを通過している
    // ため、`RepairCompositeGate` の実行順序契約〈全ゲート通過後に限り
    // ベンチを計測する〉によりベンチも実測済みになる。エスカレーションは
    // ベンチの後段〈取り込み判断〉で発生するため、ベンチ実測自体は完了して
    // いる。`verify_direct_composite.rs` 参照）。
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
        adopted_evidence["api_broken"].as_bool() == Some(true),
        "エスカレーション理由（api_broken）と証跡が一致するはず: {adopted_evidence}"
    );

    // sandbox は Escalated（未採用）で終わっているため、`FeatureAdditionFixGenerator`
    // による候補適用は最終的に baseline へ復元された状態のまま（`generate` は
    // 各試行の開始前に baseline へ復元してから候補を適用する契約。
    // `feature_addition.rs` 参照）。よって「採用後の sandbox で受け入れ基準
    // テストが通る」ことの再確認はここでは行わない（旧版は Adopted 前提の
    // 検証だったため実施していたが、Escalated では意味を持たない）。

    // 実行コマンドライン（監査・再現性のための記録。実装計画 §5 ステップ 8）
    // を付与し、CLI 未実装だった旧版の notes を実測結果（基準 1 未充足を
    // 含む）へ更新する。
    // `--log`／`--output`／`--policy-exclusion` はリポジトリルート配下の
    // 固定パスのため、記録用にはルート相対へ変換する（`--repo`／`--candidates`
    // は使い捨て sandbox・一時ファイルの絶対パスであり本質的に非決定的な
    // ため変換しない。この 2 点が実行のたび変わることは旧版から明示済みの
    // 既知の制約であり、`docs/` 直書きを既定にしない理由でもある。
    // モジュール冒頭ドキュメント「実行のたび sandbox の一時パスを含む」参照）。
    let root = repo_root();
    let relativize = |absolute: &Path| -> String {
        absolute
            .strip_prefix(&root)
            .map(|relative| relative.display().to_string())
            .unwrap_or_else(|_| absolute.display().to_string())
    };
    let invocation_for_record = format!(
        "self-repair run --kind feature-addition --repo <sandbox> --max-attempts 5 --log {} --output {} --candidates <candidates.json> --bench-bin {} --workload-source {} --policy-exclusion {}",
        relativize(&log_path),
        relativize(&output_path),
        BENCH_BIN,
        WORKLOAD_SOURCE,
        relativize(&policy_exclusion_path),
    );
    report_json["invocation"] = serde_json::json!(invocation_for_record);
    report_json["harness_wall_time_ms"] = serde_json::json!(overall_start.elapsed().as_millis());
    report_json["issue"] = serde_json::json!(142);
    report_json["task"] = serde_json::json!("TASK-3.3c");
    report_json["notes"] = serde_json::json!([
        "self-repair run CLI（3.1 節。#142 差し戻し分で実装）を 1 回起動する経路へ移行済み。",
        "self-repair verify-log CLI（外部コマンド経由。#145）でハッシュチェーン検証済み（完走判定基準 4・充足）。",
        "ベンチは RepairCompositeGate（TASK-3.2a・#137）による候補 diff 直接実測であり、baseline/candidate 双方に合成ワークロードを使う旧版（FeatureAdditionCompositeGate）とは異なる（完走判定基準 5・充足）。",
        "signal_source=measured（完走判定基準 6・充足）。",
        "基準 1（exit 0 / Adopted）は未充足: diff_signals.rs::api_signature_touched が新規 pub fn 追加を api_broken=true とし、judge.rs が無条件 Escalate するため、PoC-2 題材 (c) は現行のガードレール判定では自動適用へ到達しない（実測: outcome=Escalated, exit=10, reason=『公開 API の破壊的変更が検出されました』）。ヒューリスティック・除外リストの変更はユーザー承認必須のため本イシューでは行わない。#142 reopen 事項としてユーザー判断待ち。",
    ]);
    let mut pretty = serde_json::to_string_pretty(&report_json).expect("JSON シリアライズに失敗");
    // `.editorconfig` の `insert_final_newline` 慣行に合わせ、commit 対象と
    // なりうる出力（`docs/` 側）に末尾改行を付与する。
    pretty.push('\n');
    std::fs::write(&output_path, &pretty).expect("loop-report.json の再書き込みに失敗");

    // リポジトリに commit 済みの記録（`docs/self-repair-revalidation/
    // feature-addition/`）の更新は環境変数 `SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1`
    // を明示指定した場合のみ行う（既定を `docs/` 直書きにすると、
    // `cargo test --workspace` を実行するたび〈CI・並行実装中の他イシューの
    // テスト実行を含む〉に sandbox の一時パスを含む `invocation` が変わり、
    // 意図しない tracked diff が毎回発生してしまうため。旧版と同じ方針）。
    // `self-repair run` を再実行せず、実際に 1 回だけ起動した本実行の証跡を
    // そのまま複製する（「1 回起動」の完走判定基準 1 を docs 反映のためだけに
    // 破らないようにするため）。
    if std::env::var("SELF_REPAIR_TASK_3_3C_WRITE_DOCS").as_deref() == Ok("1") {
        let docs_out_dir = repo_root().join("docs/self-repair-revalidation/feature-addition");
        std::fs::create_dir_all(&docs_out_dir).expect("docs 出力先ディレクトリ作成に失敗");
        std::fs::copy(&output_path, docs_out_dir.join("loop-report.json"))
            .expect("loop-report.json の docs へのコピーに失敗");
        std::fs::copy(&log_path, docs_out_dir.join("loop-log.jsonl"))
            .expect("loop-log.jsonl の docs へのコピーに失敗");
    }

    let _ = std::fs::remove_dir_all(&sandbox);
    let _ = std::fs::remove_file(&candidates_path);
}
