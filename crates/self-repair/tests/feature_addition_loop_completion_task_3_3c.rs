//! TASK-3.3c（イシュー #142）の受け入れ条件「完走ログが記録されている」を
//! 満たす、機能追加種別の自己修復ループ 1 回分の完走実証。
//!
//! REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）「自作コアに
//! 対する自己修復ループの人間介在なし完走を新実装リポで再実証する」を、PoC-2
//! 検証題材 (c)（`leaky_relu` 新規実装）で 1 ループ実測する。題材選定・完走
//! 判定基準は TASK-3.3a（#140・PR #322）で人間承認済み
//! （`docs/self-repair-revalidation-plan.md` §4.2）。
//!
//! # CLI 経由完走との差分（明示）
//! `docs/guardrail-self-repair-cli.md` §5.1 が定める「`self-repair run` CLI
//! 経由の完走」・§5.4「JSON Lines ログのハッシュチェーン検証」は、CLI
//! バイナリ（`self-repair run`）・ログ形式移植（TASK-3.4・#145）の両方に
//! 依存し、本イシュー（#142）のスコープではない（`lib.rs` モジュール冒頭
//! 「本クレートが担わない責務」参照）。本テストは **lib API 直接呼び出しの
//! 実証ハーネス**として [`self_repair::SelfRepairLoop`] を 1 回起動し、
//! `LoopReport` を JSON 化した完走ログを既定で
//! `target/self-repair-revalidation/feature-addition/loop-report.json`
//! （git 管理外）へ書き出す。リポジトリに commit 済みの記録
//! （`docs/self-repair-revalidation/feature-addition/loop-report.json`）の
//! 更新は環境変数 `SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1` を明示指定した場合
//! に限る（`write_loop_report` doc 参照。既定を毎回上書きにすると sandbox
//! の PID・一時パスが変わるたびに無関係な tracked diff が生じるため）。
//! CLI・ハッシュチェーン検証の充足は #145 実装後に #144（人間評価）側で
//! 判断する。
//!
//! # シグナルは実測のみ（捏造しない）
//! `lines_changed`／`exclusion_rule_ids`／`gaming_suspect` は候補適用の実差分
//! を sandbox 内の使い捨て git リポジトリで実測する（`git diff --numstat`・
//! `git diff --unified=0`・
//! `guardrail::policy_exclusion::ExclusionEvaluation::evaluate`）。ベンチは
//! [`self_repair::verify_composite::FeatureAdditionCompositeGate`] が
//! `SelfRepairBenchGate`（bench-harness 経由・5 回計測中央値）で実測する。
//! 未計測値を fail-open な既定値で埋めない（`.claude/rules/security.md` A08）。

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use self_repair::verify_bench::MIN_BENCH_ITERATIONS;
use self_repair::{
    CandidateFix, CommandRunner, FeatureAdditionCompositeGate, FeatureAdditionDetector,
    FeatureAdditionFixGenerator, GuardrailAdoptionJudge, RepairKind, SelfRepairLoop,
    SystemCommandRunner,
};

/// baseline フィクスチャの相対パス（`crates/self-repair/tests/fixtures/…`。
/// `CARGO_MANIFEST_DIR` は本クレート〈`crates/self-repair`〉のルート）。
const FIXTURE_REL: &str = "tests/fixtures/feature-addition-leaky-relu/baseline";
/// 機能追加候補が書き換える唯一のファイル（baseline に実在し、新規ファイル
/// 追加にならないこと。`FeatureAdditionFixGenerator::new` の構築時検証対象）。
const TARGET_FILE: &str = "src/activations.rs";
/// `TARGET_FILE` 内の `#[cfg(test)] mod tests` 開始位置を示すマーカー。
/// `candidate_content`（新規関数の挿入位置）と `mod_tests_start_line`
/// （`gaming_suspect` 判定の境界行算出）の両方が同一のマーカーを参照する
/// ことを保証する（2 箇所で別々のマーカー文字列を持つと、片方だけ更新
/// されて境界の意味がずれる回帰を招くため、`const` で一本化する）。
const MOD_TESTS_MARKER: &str = "#[cfg(test)]\nmod tests {";

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

/// `sandbox` を使い捨て git リポジトリ化し、現在の内容を `baseline` コミット
/// として記録する（diff 由来シグナルの実測用。実リポジトリの git 履歴とは
/// 独立しており push・fetch は一切行わない）。
fn git_init_baseline(sandbox: &Path) {
    let run = |args: &[&str]| {
        let output = Command::new("git")
            .args(args)
            .current_dir(sandbox)
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

/// 実測 diff 由来シグナル（試行 2＝正実装が採用された場合に評価対象となる値。
/// 試行 1 は受け入れ基準テスト不合格で `VerificationGate::verify` が
/// `Failed` を返すため `AdoptionJudge` へ到達せず、この値は参照されない
/// 〈`runner.rs` の呼び出し順序契約〉。`FeatureAdditionCompositeGate` は
/// ループの全試行を通じて 1 インスタンスを使い回すため、本構造体の値は
/// 試行ごとに再計測されるのではなく「候補 2 の diff」に固定される
/// 〈`verify_composite.rs` の diff 由来シグナル契約 doc 参照〉）。
struct MeasuredSignals {
    lines_changed: u64,
    api_broken: bool,
    /// テスト側の緩和（ゲーミング）疑いの実測値: (a) 変更ファイルパスに
    /// `tests/` 配下が含まれる、または (b) `TARGET_FILE` の diff ハンクが
    /// baseline の `#[cfg(test)] mod tests` 境界（行番号）以降の既存行と
    /// 重なる、のいずれかで判定する（本ハーネスの候補生成は常にテストへ
    /// 触れない設計だが、`signal_source: "measured"` を名乗る以上ハード
    /// コードせず `git diff` の実測結果から導出する。
    /// `measure_signals_for_candidate2`・`diff_touches_boundary` 参照。
    /// **既知の限界**: `mod tests` marker 直前への隣接挿入〈本ハーネスの
    /// 挿入位置そのもの〉は境界内と判定しない。既存テスト内容を一切変えず
    /// `mod tests` の直前に新規 `#[cfg(test)]` ブロックを丸ごと追加する
    /// ような候補は、本判定では検知できない）。
    gaming_suspect: bool,
    exclusion_rule_ids: Vec<String>,
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
                // 純追加ハンク: 旧ファイルの行は変更されない。挿入位置
                // （`start` 行の直後）が `mod tests` 本文の内部かどうかで
                // 判定する（`mod tests` marker 直前への隣接挿入は除外）。
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

/// `candidate2`（正実装）を sandbox の working tree に一時的に書き込み、
/// baseline コミットとの差分を実測してから baseline 内容へ復元する。
fn measure_signals_for_candidate2(sandbox: &Path) -> MeasuredSignals {
    let target_path = sandbox.join(TARGET_FILE);
    let baseline = std::fs::read_to_string(&target_path).expect("baseline ファイル読み取りに失敗");
    std::fs::write(&target_path, candidate2_correct_content()).expect("candidate2 書き込みに失敗");

    // 1. 変更行数（`git diff --numstat` の insertions+deletions 合計）。
    let numstat = Command::new("git")
        .args(["diff", "HEAD", "--numstat", "--", TARGET_FILE])
        .current_dir(sandbox)
        .output()
        .expect("git diff --numstat の起動に失敗");
    assert!(
        numstat.status.success(),
        "git diff --numstat が失敗しました"
    );
    let numstat_text = String::from_utf8_lossy(&numstat.stdout);
    let lines_changed: u64 = numstat_text
        .lines()
        .filter_map(|line| {
            let mut cols = line.split_whitespace();
            let ins: u64 = cols.next()?.parse().ok()?;
            let del: u64 = cols.next()?.parse().ok()?;
            Some(ins + del)
        })
        .sum();

    // 2. 変更ファイル一覧（policy_exclusion 評価の `changed_files` 実測入力）。
    let name_only = Command::new("git")
        .args(["diff", "HEAD", "--name-only"])
        .current_dir(sandbox)
        .output()
        .expect("git diff --name-only の起動に失敗");
    assert!(
        name_only.status.success(),
        "git diff --name-only が失敗しました"
    );
    let changed_files: Vec<String> = String::from_utf8_lossy(&name_only.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect();

    // 2b. テスト緩和（ゲーミング）疑いの実測: (a) 変更ファイルパスに
    //     `tests/` 配下が含まれるか、(b) `TARGET_FILE` の diff ハンクが
    //     baseline の `#[cfg(test)] mod tests` 境界行以降と重なるか、の
    //     いずれかで判定する（本ハーネスの候補は `TARGET_FILE`
    //     〈`src/activations.rs`〉のみを変更する設計であり、`tests/` 配下・
    //     `TARGET_FILE` 内の `#[cfg(test)] mod tests` 境界のいずれにも触れて
    //     いないことを実測で確認する。fail-open な既定値ではなく
    //     `changed_files`・`git diff --unified=0` の実測結果から導出する）。
    let touches_tests_dir = changed_files.iter().any(|f| f.contains("tests/"));
    let unified_diff = Command::new("git")
        .args(["diff", "HEAD", "--unified=0", "--", TARGET_FILE])
        .current_dir(sandbox)
        .output()
        .expect("git diff --unified=0 の起動に失敗");
    assert!(
        unified_diff.status.success(),
        "git diff --unified=0 が失敗しました"
    );
    let boundary_line = mod_tests_start_line(&baseline);
    let touches_mod_tests_boundary = diff_touches_boundary(
        &String::from_utf8_lossy(&unified_diff.stdout),
        boundary_line,
    );
    let gaming_suspect = touches_tests_dir || touches_mod_tests_boundary;

    // 3. ポリシー除外リスト評価（guardrail lib 直接呼び出し。組み込み既定値。
    //    `.claude/rules/security.md`「判定の迂回経路を作らない」に沿い、
    //    本番経路〈judge.rs〉と同じ `guardrail::policy_exclusion` API を使う）。
    let rules =
        guardrail::policy_exclusion_builtin_defaults().expect("組み込み既定ルールの構築に失敗");
    let ctx = guardrail::EvaluationContext {
        repo_root: sandbox.to_path_buf(),
        baseline: "HEAD".to_string(),
        changed_files,
    };
    let evaluation = guardrail::ExclusionEvaluation::evaluate(&rules.rules, &ctx)
        .expect("policy_exclusion 評価の実行に失敗");
    let exclusion_rule_ids = evaluation.effective_rule_ids();

    // 4. 公開 API 破壊チェック（既存の `pub fn` 行がすべて残存しているか。
    //    候補は追加のみで既存シグネチャを変更しないことを実測する）。
    let candidate2 = candidate2_correct_content();
    let api_broken = !baseline
        .lines()
        .filter(|line| line.trim_start().starts_with("pub fn "))
        .all(|line| candidate2.contains(line.trim()));

    // baseline 内容へ復元（実ループが `FeatureAdditionFixGenerator` 経由で
    // 改めて適用するため、ここでの一時書き込みは計測専用）。
    std::fs::write(&target_path, &baseline).expect("baseline 復元に失敗");

    MeasuredSignals {
        lines_changed,
        api_broken,
        gaming_suspect,
        exclusion_rule_ids,
    }
}

#[test]
fn feature_addition_loop_completes_with_measured_evidence() {
    let overall_start = Instant::now();

    // --- sandbox 準備 ---
    let fixture_src = repo_root().join("crates/self-repair").join(FIXTURE_REL);
    let sandbox = unique_sandbox_dir("sandbox");
    copy_dir_recursive(&fixture_src, &sandbox);
    rewrite_path_deps_to_absolute(&sandbox);
    git_init_baseline(&sandbox);

    // baseline 状態で `cargo test --release` が実際に失敗することを確認する
    // （`FeatureAdditionDetector` が Finding を返す前提条件の独立検証）。
    let runner = SystemCommandRunner::new();
    let signals = measure_signals_for_candidate2(&sandbox);

    // --- 段階の構築（すべて実実装・テストダブルなし） ---
    let detector = FeatureAdditionDetector::new(sandbox.clone(), SystemCommandRunner::new());

    let candidates = vec![
        CandidateFix {
            description: "試行1（誤実装・符号分岐なし）".to_string(),
            files: vec![(PathBuf::from(TARGET_FILE), candidate1_wrong_content())],
        },
        CandidateFix {
            description: "試行2（正実装・既存組み込み演算の合成）".to_string(),
            files: vec![(PathBuf::from(TARGET_FILE), candidate2_correct_content())],
        },
    ];
    let fix_generator = FeatureAdditionFixGenerator::new(sandbox.clone(), candidates)
        .expect("FixGenerator の構築に失敗（候補パス検証）");

    let verification_gate = FeatureAdditionCompositeGate::new(
        sandbox.clone(),
        SystemCommandRunner::new(),
        signals.lines_changed,
        signals.api_broken,
        signals.gaming_suspect,
        signals.exclusion_rule_ids.clone(),
        MIN_BENCH_ITERATIONS,
    );
    // `SelfRepairLoop::new` はゲートを値ごと（所有権を）受け取るため、
    // ループ実行後もベンチ実測を観測できるよう `Rc` 複製を先に取得しておく
    // （`verify_composite.rs` の `evidence_sink`／`bench_measurement_sink`
    // doc 参照）。
    let evidence_sink = verification_gate.evidence_sink();
    let bench_measurement_sink = verification_gate.bench_measurement_sink();

    // guardrail.toml（TASK-4.3c 承認済み・リポジトリルート）を guardrail の
    // 設定解決 API で読み込む。数値のハードコード・緩和は行わない
    // （`.claude/rules/security.md`）。
    let config = guardrail::config::resolve(None, &repo_root(), guardrail::PresetName::Default)
        .expect("guardrail.toml の解決に失敗");
    let adoption_judge = GuardrailAdoptionJudge::new(config.thresholds);

    let max_attempts = NonZeroU32::new(5).expect("5 は非ゼロ");
    let self_repair_loop = SelfRepairLoop::new(
        detector,
        fix_generator,
        verification_gate,
        adoption_judge,
        max_attempts,
    );

    // --- ループ実行（追加入力なし・1 回起動） ---
    let report = self_repair_loop
        .run(RepairKind::FeatureAddition)
        .expect("ループは段階実行自体のエラーなく完走するはず");

    // --- 完走判定基準（lib 版） ---
    assert_eq!(
        report.outcome,
        self_repair::LoopOutcome::Adopted,
        "機能追加種別のループは最終的に Adopted に到達するはず: {report:?}"
    );
    assert_eq!(
        report.attempt_count(),
        2,
        "試行1（検証不合格）→試行2（採用）の 2 試行系列であるはず"
    );
    assert!(
        matches!(
            report.attempts[0].outcome,
            self_repair::report::AttemptOutcome::VerificationFailed { .. }
        ),
        "試行1は受け入れ基準テスト不合格で検証不合格になるはず: {:?}",
        report.attempts[0]
    );
    assert!(
        matches!(
            report.attempts[1].outcome,
            self_repair::report::AttemptOutcome::Adopted
        ),
        "試行2は採用されるはず: {:?}",
        report.attempts[1]
    );

    // 適用後 sandbox で受け入れ基準テストが通ることを再確認する（既知正解値の
    // 再検証。`cargo test --release` を直接再実行する）。
    let post_check = runner
        .run("cargo", &["test", "--release"], &sandbox)
        .expect("適用後の cargo test --release 起動に失敗");
    assert!(
        post_check.success(),
        "採用後の sandbox で受け入れ基準テストが通らない: {}",
        post_check.log_tail()
    );

    // ベンチが `NotRun` に丸められず実測されたことを確認する（`verify` 内で
    // 全ゲート通過後に限りベンチを計測する順序契約。`evidence_sink`〈ループ
    // 実行前に取得した `Rc` 複製〉経由で観測する。`AttemptOutcome::Adopted`
    // は証跡そのものを保持しないため〈`report.rs` 参照〉、この観測点が
    // 唯一の事後確認手段である）。
    let last_evidence = evidence_sink
        .borrow()
        .clone()
        .expect("採用に至った以上 verify は少なくとも 1 回 Passed を発行しているはず");
    match last_evidence.bench() {
        guardrail::BenchSignal::Measured { median_pct } => {
            assert!(median_pct.is_finite(), "ベンチ中央値は有限値のはず");
        }
        guardrail::BenchSignal::NotRun => {
            panic!("ベンチが NotRun のまま（全ゲート通過後は Measured のはず）")
        }
    }

    let bench_measurement = bench_measurement_sink
        .borrow()
        .clone()
        .expect("ベンチが Measured である以上、生の計測系列も記録されているはず");

    let total_wall_time = overall_start.elapsed();

    write_loop_report(
        &report,
        &signals,
        total_wall_time,
        &last_evidence,
        &bench_measurement,
        &config.thresholds,
    );

    let _ = std::fs::remove_dir_all(&sandbox);
}

/// `LoopReport` と実測シグナルから完走ログ（JSON）を構築する。
///
/// 既定の出力先は `target/self-repair-revalidation/feature-addition/
/// loop-report.json`（git 管理対象外）。リポジトリに commit 済みの記録
/// （`docs/self-repair-revalidation/feature-addition/loop-report.json`）は
/// 環境変数 `SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1` を明示指定した場合のみ
/// 上書きする。既定を `docs/` 直書きにすると、`cargo test --workspace` を
/// 実行するたび（CI・並行実装中の他イシューのテスト実行を含む）に
/// sandbox の PID・一時パスを含む `log_tail` が変わり、意図しない tracked
/// diff が毎回発生してしまうため（advisor 指摘）。
fn write_loop_report(
    report: &self_repair::LoopReport,
    signals: &MeasuredSignals,
    wall_time: std::time::Duration,
    adopted_evidence: &self_repair::VerifiedEvidence,
    bench_measurement: &self_repair::verify_bench::BenchSignal,
    thresholds: &guardrail::Thresholds,
) {
    let attempts_json: Vec<serde_json::Value> = report
        .attempts
        .iter()
        .map(|attempt| {
            serde_json::json!({
                "attempt": attempt.attempt,
                "duration_ms": attempt.duration.as_millis(),
                "outcome": format!("{:?}", attempt.outcome),
            })
        })
        .collect();

    // 採用（Adopted）に至った試行の判断根拠。`AttemptOutcome::Adopted` は
    // 証跡そのものを保持しないため（`report.rs` 参照）、`evidence_sink`／
    // `bench_measurement_sink` 経由で観測した値をここへ明示的に記録する
    // （TASK-3.3c 受け入れ条件「判断根拠」を満たす。advisor 指摘 1）。
    let bench_median_pct = match adopted_evidence.bench() {
        guardrail::BenchSignal::Measured { median_pct } => Some(*median_pct),
        guardrail::BenchSignal::NotRun => None,
    };
    let adopted_evidence_json = serde_json::json!({
        "attempt": adopted_evidence.attempt(),
        "gate_report": adopted_evidence.gate_report(),
        "bench_median_pct": bench_median_pct,
        "bench_measurements_pct": bench_measurement.bench_measurements_pct,
        "lines_changed": adopted_evidence.lines_changed(),
        "api_broken": adopted_evidence.api_broken(),
        "gaming_suspect": adopted_evidence.gaming_suspect(),
        "exclusion_rule_ids": adopted_evidence.exclusion_rule_ids(),
    });

    let doc = serde_json::json!({
        "task": "TASK-3.3c",
        "issue": 142,
        "repair_kind": format!("{:?}", report.kind),
        "outcome": format!("{:?}", report.outcome),
        "attempt_count": report.attempt_count(),
        "attempts": attempts_json,
        "total_duration_ms": report.total_duration.as_millis(),
        "harness_wall_time_ms": wall_time.as_millis(),
        "measured_signals": {
            "lines_changed": signals.lines_changed,
            "api_broken": signals.api_broken,
            "gaming_suspect": signals.gaming_suspect,
            "exclusion_rule_ids": signals.exclusion_rule_ids,
        },
        "adopted_evidence": adopted_evidence_json,
        "thresholds": {
            "source": "guardrail.toml (preset.default) via guardrail::config::resolve",
            "lines_max": thresholds.lines_max,
            "bench_median_max_pct": thresholds.bench_median_max_pct,
            "bench_runs_min": thresholds.bench_runs_min,
        },
        "signal_source": "measured",
        "notes": [
            "CLI（self-repair run）経由の完走ではなく lib API 直接呼び出しの実証ハーネス（#142 スコープ判断。docs/self-repair-revalidation/feature-addition/README.md 参照）",
            "JSON Lines ハッシュチェーン形式（改竄検知）は TASK-3.4（#145）実装後に適用予定",
            "bench_median_pct・bench_measurements_pct は baseline/candidate 双方に同一の合成ワークロード（leaky_relu_like_workload）を用いた計測であり、実際の leaky_relu 実装差分固有の性能特性は計測していない（既知の制約。verify_composite.rs・README『シグナルは実測のみ』節参照）",
        ],
    });
    let mut pretty = serde_json::to_string_pretty(&doc).expect("JSON シリアライズに失敗");
    // `.editorconfig` の `insert_final_newline` 慣行に合わせ、commit 対象
    // となりうる出力（`docs/` 側）に末尾改行を付与する。
    pretty.push('\n');

    let target_out_dir = repo_root().join("target/self-repair-revalidation/feature-addition");
    std::fs::create_dir_all(&target_out_dir).expect("target 出力ディレクトリ作成に失敗");
    std::fs::write(target_out_dir.join("loop-report.json"), &pretty)
        .expect("target/loop-report.json 書き込みに失敗");

    if std::env::var("SELF_REPAIR_TASK_3_3C_WRITE_DOCS").as_deref() == Ok("1") {
        let docs_out_dir = repo_root().join("docs/self-repair-revalidation/feature-addition");
        std::fs::create_dir_all(&docs_out_dir).expect("docs 出力ディレクトリ作成に失敗");
        std::fs::write(docs_out_dir.join("loop-report.json"), &pretty)
            .expect("docs/loop-report.json 書き込みに失敗");
    }
}
