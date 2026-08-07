//! ポリシー除外リスト（REQ-5・`policy-exclusion.toml`。TASK-5.1a・#119 が
//! 定義するルール定義ファイル）のうち `test-tolerance-loosening` ルール
//! （match 方式 `test_assertion_relaxation_without_prod_change`）の match 述語。
//!
//! # 背景（PoC-3 発見事項 5・REQ-5 受け入れ基準 2）
//! REQ-4 のゲーミング検知（`gaming::check`。本モジュールでは未移植）は
//! 「本番コードとテストの**同時**変更」のみを対象とするため、テスト許容誤差の
//! **単独**緩和（例: `1e-6 → 1e-2`。本番コード変更なし）は機械判定をすり抜け、
//! `auto-apply` されてしまう既知ブラインドスポット G5
//! （`tests/fixtures/labeled-changes/changes/G5-test-only-loosen`）である。
//! 本モジュールはこのブラインドスポットを補うための match 述語のみを提供する。
//!
//! # 呼び出し元・責務境界
//! [`test_assertion_relaxation_without_prod_change`] は
//! [`crate::policy_exclusion::ExclusionEvaluation::evaluate`]（TASK-5.2c・#124
//! が統合した `MatchRule` 評価器）から呼ばれ、判定結果は `decision::decide`
//! （既に受け口を持つ `DecisionInput::exclusion_rule_ids`）へ渡って**無条件**
//! エスカレーションに使われる（`policy_exclusion::ExclusionEvaluation::effective_rule_ids`
//! 経由）。本モジュール自体は `decide()` を呼ばず、`MatchRule` 列挙・
//! TOML ロードも持たない（`policy_exclusion` モジュール本体の責務）。
//!
//! [`changed_files_for_policy_exclusion`] も同じ理由で `policy_exclusion`
//! （`EvaluationContext::from_repo`）専用の変更ファイル一覧取得口として提供する
//! （`Cargo.lock` を除外しない。下記関数ドキュメント参照）。
//!
//! `gaming::check`（REQ-4 側・本番・テスト**同時**変更を対象とする側。
//! 未移植）とは入力領域が排他的（本モジュールは「本番コード変更**なし**」側
//! のみを担う）。
//!
//! # #124 統合状態（イシュー #122／#123 からの引き継ぎ完了）
//! `test_assertion_relaxation_without_prod_change` の判定結果は
//! `ExclusionEvaluation::evaluate`（`policy_exclusion/mod.rs`）の
//! `matched_rule_ids` へ計上済み。かつては `unevaluated_rule_ids` へ回っていたが
//! （#122 レビュー時点の暫定対応）、本統合により `MatchRule::TestAssertionRelaxationWithoutProdChange`
//! は評価対象になった。
//!
//! # 変更ファイルパスの正規化契約
//! [`changed_files`]／[`changed_files_for_policy_exclusion`] が返すパスは
//! リポジトリルート相対・先頭 `/` なし・`./` なし・`/` 区切り（git の
//! `--name-only` 出力形式）である。`git_command` に `-c core.quotePath=false`
//! を指定し、非 ASCII を含むパスが 8 進数エスケープ表記
//! （例: `"foo/\346\227\245.rs"`）へ変換されて `PathPattern::matches` の
//! 単純な文字列比較が無言で不一致になる経路を塞ぐ（Review 指摘 Low・#122。
//! quotePath は「7bit 非 ASCII をエスケープしない」設定であり、正規化の一部
//! として明示的に無効化する）。
//!
//! 「リポジトリルート相対」は `git_command` が `-C repo_root` で作業木を
//! 固定し、かつ `diff --name-only <baseline> -- .` の pathspec `.` が
//! （シェル展開ではなく）`-C` 適用後のカレントディレクトリ＝`repo_root`
//! を指す前提に立つ。呼び出し元が常に `repo_root`（サブディレクトリでは
//! ない）を渡す契約を守る限りこの前提は成立する（現状の呼び出し元
//! （`self-repair`・テスト群）はいずれもリポジトリルートを渡す。将来
//! `main.rs`（#103）がサブディレクトリから呼ぶ構成へ変える場合は、この
//! 前提の再検証が必要）。
//!
//! # fail-closed 契約（`.claude/rules/security.md` A08）
//! `git diff` の起動失敗・非ゼロ終了は必ず [`crate::error::GuardrailError`]
//! として伝播し、`false`（match なし＝除外リスト素通り→自動適用方向）へ
//! 丸めない。空 stdout は「疑いなし」であって「エラーなし」と同義ではない
//! ため区別する。

use std::path::Path;
use std::process::Command;

use crate::error::GuardrailError;

/// `repo_root` を作業木として `git` を起動する共通ヘルパー。
///
/// 祖先プロセス（lefthook の pre-push フック等）から継承された `GIT_DIR`／
/// `GIT_WORK_TREE`／`GIT_INDEX_FILE` 等の `GIT_*` 環境変数を明示的に除去する。
/// 除去しないと `-C repo_root` の指定を無視して呼び出し元プロセスの
/// リポジトリ（例: 本 worktree の `.git`）に対して動作してしまい、
/// 検査対象の取り違えが起こる（`tests/labeled_changes_fixtures.rs::run` と
/// 同一方針）。
///
/// 加えて、diff 出力の形式を変化させうる利用者の `~/.gitconfig`／リポジトリ
/// ローカル設定（`color.ui = always`・`diff.external` 外部ツール等）を
/// `-c` で明示的に無効化する。`-c` はコマンドライン引数であり設定ファイル
/// より優先順位が高いため、継承した `GIT_*` 除去だけでは防げないローカル
/// `.git/config` 経由の汚染も上書きできる。無効化しないと削除行が `^-`
/// で始まらなくなり `is_removed_content_line` のパターンマッチが無言で
/// 全滅し、本モジュールが明示的に避けようとしている fail-open
/// （match なし＝除外リスト素通り→自動適用方向）そのものを招く
/// （Review 指摘 Low・#123）。
///
/// `core.quotePath=false` も同じ趣旨で無効化する: 既定（true）だと非 ASCII
/// パスが 8 進数エスケープ表記で出力され、[`changed_files`] 等の呼び出し元が
/// `PathPattern::matches` で単純な文字列比較を行う際に無言で不一致になる
/// （パス正規化契約の一部。モジュール冒頭コメント参照。#124）。
fn git_command(repo_root: &Path) -> Command {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(repo_root);
    for (key, _) in std::env::vars() {
        if key.starts_with("GIT_") {
            cmd.env_remove(key);
        }
    }
    cmd.arg("-c").arg("color.ui=never");
    cmd.arg("-c").arg("color.diff=never");
    cmd.arg("-c").arg("color.status=never");
    cmd.arg("-c").arg("diff.external=");
    cmd.arg("-c").arg("core.pager=cat");
    cmd.arg("-c").arg("core.quotePath=false");
    cmd
}

/// `git_command` を `args` で実行し、成功時は stdout（UTF-8 損失変換込み）を
/// 返す。起動失敗・非ゼロ終了は [`GuardrailError::DiffSpawn`]／
/// [`GuardrailError::DiffFailed`] としてエラー伝播する（fail-closed。
/// 「match なし」へ丸めない。モジュール冒頭コメント参照）。
///
/// `pub(crate)`: [`crate::checks::diff_lines`]／[`crate::checks::api_stability`]／
/// [`crate::gaming`]（TASK-4.1c・イシュー #106）も本関数を経由して `git` を
/// 起動する（`-c core.quotePath=false` 等の diff 出力汚染対策・`GIT_*`
/// 環境変数除去を単一実装に保つため。モジュール冒頭「変更ファイルパスの
/// 正規化契約」参照）。
pub(crate) fn run_git(repo_root: &Path, args: &[&str]) -> Result<String, GuardrailError> {
    let command_display = format!("git {}", args.join(" "));
    let output = git_command(repo_root)
        .args(args)
        .output()
        .map_err(|source| GuardrailError::DiffSpawn {
            command: command_display.clone(),
            source,
        })?;
    if !output.status.success() {
        return Err(GuardrailError::DiffFailed {
            command: command_display,
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `run_git` の diff 系呼び出しの戻り値が unified diff の想定形状
/// （`diff --git ` ヘッダを含む）かを確認する。空 stdout（差分なし）は
/// 許容する。
///
/// `git_command` の `-c diff.external=`／呼び出し側の `--no-ext-diff`・
/// `--no-textconv` で既知の変形経路は防いでいるが、`.gitattributes` の
/// 未知の diff ドライバ設定等、列挙しきれない経路が残りうる。想定外の
/// 形状であれば「削除行なし」と静かに丸めず
/// [`GuardrailError::DiffUnexpectedFormat`] として伝播する（fail-closed。
/// Review 指摘 Low・#123）。
fn validate_unified_diff_shape(command: &str, stdout: &str) -> Result<(), GuardrailError> {
    if stdout.is_empty() || stdout.lines().any(|line| line.starts_with("diff --git ")) {
        return Ok(());
    }
    Err(GuardrailError::DiffUnexpectedFormat {
        command: command.to_string(),
        reason: "unified diff 形式（'diff --git ' ヘッダ）を含まない出力".to_string(),
    })
}

/// unified diff の削除行（`^-` かつファイルヘッダ `^--` ではない）判定。
/// `guardrail.sh:86` の `grep -E '^-[^-]'`（v1 相当。PoC-3）と同じ契約。
fn is_removed_content_line(line: &str) -> bool {
    line.starts_with('-') && !line.starts_with("--")
}

/// `line` が `patterns` のいずれかにリテラル部分文字列として一致するか判定する。
///
/// 正規表現エンジンは使わない（外部入力である `assertion_patterns` を
/// パターン起点のコード実行・ReDoS 経路に晒さないため。A03）。
/// トークン `"1e-[0-9]"` のみ特別扱いし、`"1e-"` の直後に ASCII 数字が
/// 続く箇所があれば一致とみなす（`policy-exclusion.toml` のルール定義
/// （#119）と同一契約）。空文字列パターンは防御的に無視する（load 時検証を
/// すり抜けた場合でも「全行一致」という fail-open を防ぐため）。
pub(crate) fn line_matches_any_pattern(line: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| {
        if pattern.is_empty() {
            return false;
        }
        if pattern == "1e-[0-9]" {
            return contains_1e_minus_digit(line);
        }
        line.contains(pattern.as_str())
    })
}

/// `line` 中に `1e-` の直後へ ASCII 数字が続く箇所があるか走査する。
fn contains_1e_minus_digit(line: &str) -> bool {
    let bytes = line.as_bytes();
    let needle = b"1e-";
    if bytes.len() < needle.len() + 1 {
        return false;
    }
    for start in 0..=(bytes.len() - needle.len() - 1) {
        if &bytes[start..start + needle.len()] == needle
            && bytes[start + needle.len()].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

/// `git diff --name-only` の共通実装。`exclude_cargo_lock` で `Cargo.lock`
/// を pathspec 除外するかを切り替える（[`changed_files`]・
/// [`changed_files_for_policy_exclusion`] の 2 通りの呼び出し口を持つ理由は
/// 各関数のドキュメント参照）。
fn changed_files_impl(
    repo_root: &Path,
    baseline: &str,
    exclude_cargo_lock: bool,
) -> Result<Vec<String>, GuardrailError> {
    let mut args = vec!["diff", "--name-only", baseline, "--", "."];
    if exclude_cargo_lock {
        args.push(":!Cargo.lock");
    }
    let stdout = run_git(repo_root, &args)?;
    Ok(stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect())
}

/// `baseline` と現作業木との差分で変更されたファイル一覧を返す
/// （`Cargo.lock` は判定対象外。依存解決の機械的差分をルール判定に
/// 混入させないため）。[`touches_prod_logic`]（テスト許容誤差単独緩和判定。
/// G5 対策）専用の取得口であり、`Cargo.lock` の扱いはその用途に閉じる。
pub(crate) fn changed_files(
    repo_root: &Path,
    baseline: &str,
) -> Result<Vec<String>, GuardrailError> {
    changed_files_impl(repo_root, baseline, true)
}

/// [`crate::policy_exclusion::EvaluationContext::from_repo`] 専用の変更
/// ファイル一覧取得口。[`changed_files`] と異なり `Cargo.lock` を除外**しない**。
///
/// `dependency-change` ルール（TASK-5.1b・#120 で人間承認済み。
/// `policy-exclusion.toml` 参照）は `paths` に `Cargo.lock` 自体を含むため、
/// ここで除外してしまうと `Cargo.toml` を伴わない `Cargo.lock` 単独差分
/// （依存の間接更新等）が黙って `any_diff_in_paths` の判定対象から漏れ、
/// 無条件エスカレーションが発火しない fail-open 経路になる（計画 3.1 節
/// 「注意」・`.claude/rules/security.md` A08）。
pub(crate) fn changed_files_for_policy_exclusion(
    repo_root: &Path,
    baseline: &str,
) -> Result<Vec<String>, GuardrailError> {
    changed_files_impl(repo_root, baseline, false)
}

/// `baseline` と現作業木との差分（`-U0`）のうち、削除行が `patterns`
/// のいずれかに一致する箇所が 1 つでもあるか判定する。
/// 「テストのアサーション・許容誤差の緩和」検知の第一条件を担う。
pub(crate) fn touches_test_assertion_with_patterns(
    repo_root: &Path,
    baseline: &str,
    patterns: &[String],
) -> Result<bool, GuardrailError> {
    let args = [
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "-U0",
        baseline,
        "--",
        ".",
        ":!Cargo.lock",
    ];
    let stdout = run_git(repo_root, &args)?;
    validate_unified_diff_shape(&format!("git {}", args.join(" ")), &stdout)?;
    Ok(stdout
        .lines()
        .filter(|line| is_removed_content_line(line))
        .any(|line| line_matches_any_pattern(line, patterns)))
}

/// `line` が `mod tests` 宣言（`mod tests {` 等）そのものか判定する。
/// 単純な前方一致だと `mod tests_helpers {`／`mod testsuite {` のような
/// 別モジュールを誤検知するため、`"mod tests"` の直後が識別子継続文字
/// （英数字・`_`）でないことを境界条件として要求する（Review 指摘 Low。
/// #123）。
fn is_mod_tests_declaration(line: &str) -> bool {
    let trimmed = line.trim_start();
    match trimmed.strip_prefix("mod tests") {
        Some(rest) => !matches!(
            rest.chars().next(),
            Some(c) if c.is_alphanumeric() || c == '_'
        ),
        None => false,
    }
}

/// `file`（現作業木上のテキスト）中で `mod tests` 宣言が現れる行番号
/// （1-origin・新ファイル側の座標系）を返す。見つからない場合は `None`
/// （呼び出し元 [`touches_prod_logic`] は `mod tests` 宣言の別名（例:
/// 単数形 `mod test`）等の非標準な境界を「未確認」として扱い、
/// [`ProdTouch::UnknownBoundary`] へ写像する。Review 指摘 Medium・#123）。
///
/// [`hunk_new_start_lines`] と同じ「現作業木＝new 側」の座標系で行番号を
/// 数える必要がある（両者の座標系が食い違うと `touches_prod_logic` の
/// 比較が意味を失う。Review 指摘 Medium。#123）。
fn current_mod_tests_line(repo_root: &Path, file: &str) -> Option<u32> {
    let path = repo_root.join(file);
    let content = std::fs::read_to_string(&path).ok()?;
    content
        .lines()
        .enumerate()
        .find(|(_, line)| is_mod_tests_declaration(line))
        .map(|(idx, _)| (idx + 1) as u32)
}

/// `baseline` と現作業木との差分（`-U0`）における `file` のハンク開始行
/// （**new 側＝現作業木側**の行番号）一覧を返す。unified diff ヘッダ
/// `@@ -oldStart[,oldCount] +newStart[,newCount] @@` の `newStart` を
/// パースする（正規表現は使わず手書きパース。A03 の趣旨と同じくパターン
/// 起点の処理を避ける）。
///
/// old 側ではなく new 側を使う理由: [`current_mod_tests_line`] は現作業木
/// （new 側）のファイル内容から行番号を数えるため、比較対象のハンク開始行も
/// 同じ座標系でなければならない。削除を伴うハンク（`oldCount > newCount`）
/// では old 側行番号と new 側行番号がずれるため、old 側を使うと
/// `mod tests` 宣言の直前で本番コードを削除するハンクを見逃す
/// （実測による再現・Review 指摘 Medium。#123）。unified diff の規約上、
/// `newStart` は「その hunk 直前で最後に残った行」を指す（0 カウントの
/// 純削除ハンクでも同様）ため、`newStart < mod_tests_line` は
/// 「`mod tests` の外（＝本番コード領域）で変更が起きた」を安全側に
/// 判定できる。
fn hunk_new_start_lines(
    repo_root: &Path,
    baseline: &str,
    file: &str,
) -> Result<Vec<u32>, GuardrailError> {
    let args = [
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "-U0",
        baseline,
        "--",
        file,
    ];
    let stdout = run_git(repo_root, &args)?;
    validate_unified_diff_shape(&format!("git {}", args.join(" ")), &stdout)?;
    let mut starts = Vec::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("@@ -") {
            // rest は "oldStart[,oldCount] +newStart[,newCount] @@ ..." の形。
            // "+" 以降のトークン（カンマ区切りの数値の前半）を取り出す。
            let new_part = rest
                .split_whitespace()
                .find_map(|tok| tok.strip_prefix('+'))
                .unwrap_or("");
            let new_start_str = new_part.split(',').next().unwrap_or("");
            if let Ok(new_start) = new_start_str.parse::<u32>() {
                starts.push(new_start);
            }
        }
    }
    Ok(starts)
}

/// `changed_files` のうち「本番コード（`src/*.rs`。`tests/` 配下除く）」に
/// 該当するものへ絞り込む。
fn is_prod_source_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let in_src = normalized.starts_with("src/") || normalized.contains("/src/");
    let in_tests_dir = normalized.starts_with("tests/") || normalized.contains("/tests/");
    normalized.ends_with(".rs") && in_src && !in_tests_dir
}

/// [`touches_prod_logic`] の判定結果。3 値にするのは、`mod tests` 境界が
/// 非標準名等で特定できない「未確認」状態と、境界が確認できたうえで
/// 本番コード変更が確認できない状態とを区別するため（Bugbot 指摘 Medium・
/// #123。詳細は [`touches_prod_logic`] のドキュメント参照）。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ProdTouch {
    /// 変更された本番コード候補ファイルすべてで `mod tests` 境界を確認でき、
    /// いずれのハンクも `mod tests` 内側（＝本番コード変更なし）だった。
    NotTouched,
    /// `mod tests` 境界の外側（＝本番コード領域）で変更を確認した。
    Touched,
    /// 少なくとも 1 ファイルで `mod tests` 境界を確認できず、本番コード
    /// 変更の有無を安全に判定できない（`Touched` 未検出の場合のみ成立）。
    UnknownBoundary,
}

/// `baseline` と現作業木との差分が本番コード（`src/*.rs`。`tests/` 配下
/// 除く）に触れているか判定する。
///
/// 判定方法: 変更ファイルのうち本番コード候補ファイルそれぞれについて、
/// 差分ハンクの **new 側**（現作業木側）開始行が、現作業木の当該ファイル中
/// `mod tests` 宣言行より前（＝ `mod tests` の外＝本番コード領域）にあるかを
/// 見る。ハンク開始行・`mod tests` 宣言行の双方を同じ new 側座標系で数える
/// ことで、削除を伴うハンク（`mod tests` 直前の本番コード削除等）でも
/// 座標系のずれによる見逃しが起きない（[`hunk_new_start_lines`] のドキュメント
/// 参照。Review 指摘 Medium・#123 で実測確認済み）。
///
/// `mod tests` 宣言が見つからないファイル（別名のテストモジュール・
/// テストモジュールを持たないファイル・削除されたファイル等）は
/// [`ProdTouch::UnknownBoundary`] を返す（`Touched` が他ファイルで確定
/// 済みなら優先する）。呼び出し元 [`test_assertion_relaxation_without_prod_change`]
/// は `UnknownBoundary` を「本番コード変更なしと確認できなかった」＝
/// match（エスカレーション）方向に倒す。旧実装は安全側番兵で
/// `touches_prod_logic` 自体を `true`（Touched 相当）へ丸めていたが、
/// これは overall predicate `!touches_prod_logic` を `false`（match なし＝
/// エスカレーションなし）にしてしまい、コメントが意図した「安全側」とは
/// **逆**の結果（境界不明のファイルが自動適用をすり抜ける）を招いていた
/// （Bugbot 指摘 Medium・#123。REQ-5 の不変条件「除外リストは安全側にしか
/// 作用しない」を守るには、未確認は Touched とは区別し、末端の match 判定を
/// 呼び出し元で明示的にエスカレーション方向へ倒す必要がある）。
///
/// 既知の制約（スコープ外・`.claude/rules/out-of-scope-tracking.md`）:
/// 本判定はファイル中「最初に現れる」`mod tests` 宣言のみを境界として扱う。
/// 複数の `mod tests` を持つファイル・`mod tests` より後ろに本番コードが
/// 続く構成には対応しない。
pub(crate) fn touches_prod_logic(
    repo_root: &Path,
    baseline: &str,
) -> Result<ProdTouch, GuardrailError> {
    let files = changed_files(repo_root, baseline)?;
    let mut unknown_boundary = false;
    for file in files.iter().filter(|f| is_prod_source_path(f)) {
        let hunk_starts = hunk_new_start_lines(repo_root, baseline, file)?;
        match current_mod_tests_line(repo_root, file) {
            Some(mod_tests_line) => {
                if hunk_starts.iter().any(|&start| start < mod_tests_line) {
                    return Ok(ProdTouch::Touched);
                }
            }
            None => {
                unknown_boundary = true;
            }
        }
    }
    if unknown_boundary {
        Ok(ProdTouch::UnknownBoundary)
    } else {
        Ok(ProdTouch::NotTouched)
    }
}

/// `test-tolerance-loosening` ルール（`policy-exclusion.toml`。#119）の
/// match 述語本体。REQ-5 受け入れ基準 2「テスト許容誤差の単独緩和で
/// ルールが match する」を実装する。
///
/// `repo_root` の作業木を `baseline` ref と比較し、以下の**両方**を満たす
/// 場合に `true` を返す:
/// 1. 削除行に既存アサーション・許容誤差リテラル（`assertion_patterns`。
///    典型例: `assert!`／`abs() <`／`1e-[0-9]`。`policy-exclusion.toml` の
///    ルール定義と同一契約）が含まれる（テストの緩和が起きている）
/// 2. 本番コード（`src/*.rs`。`tests/` 配下除く）の変更を確認できない
///    （[`ProdTouch::Touched`] ではない。境界確認済みで確実に変更なしの
///    場合＝[`ProdTouch::NotTouched`] だけでなく、`mod tests` 境界が非標準
///    名等で特定できない [`ProdTouch::UnknownBoundary`] の場合も match
///    （エスカレーション）方向に倒す。安全性を証明できない場合は
///    「安全」と誤判定して自動適用をすり抜けさせない。Bugbot 指摘 Medium・
///    #123。詳細は [`touches_prod_logic`] のドキュメント参照）
///
/// 短絡評価: 条件 1 が偽の場合は `touches_prod_logic` の git 呼び出しを
/// 省略する（無用な子プロセス起動を避ける）。
pub fn test_assertion_relaxation_without_prod_change(
    repo_root: &Path,
    baseline: &str,
    assertion_patterns: &[String],
) -> Result<bool, GuardrailError> {
    if !touches_test_assertion_with_patterns(repo_root, baseline, assertion_patterns)? {
        return Ok(false);
    }
    Ok(!matches!(
        touches_prod_logic(repo_root, baseline)?,
        ProdTouch::Touched
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// `run_git` と同じ `GIT_*` 除去方針を踏襲したテスト専用 git 実行
    /// ヘルパー（`tests/labeled_changes_fixtures.rs::run` と同一方針）。
    fn run(cwd: &Path, args: &[&str]) {
        let mut cmd = Command::new("git");
        cmd.args(args).current_dir(cwd);
        for (key, _) in std::env::vars() {
            if key.starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
        let output = cmd
            .output()
            .unwrap_or_else(|e| panic!("git {args:?} 起動に失敗 (cwd={cwd:?}): {e}"));
        assert!(
            output.status.success(),
            "git {args:?} が失敗 (cwd={cwd:?}): {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn commit_all(cwd: &Path, message: &str) {
        run(cwd, &["add", "-A"]);
        run(
            cwd,
            &[
                "-c",
                "user.email=guardrail-test@example.invalid",
                "-c",
                "user.name=guardrail-test",
                "commit",
                "-q",
                "-m",
                message,
            ],
        );
    }

    /// 隔離作業ディレクトリを作り `git init` 済みリポジトリを用意する
    /// （`std::env::temp_dir()` 配下。`tempfile` クレートは使わない。
    /// `labeled_changes_fixtures.rs` の std-only 手法を踏襲）。
    fn init_repo(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "guardrail-exclusion-match-{name}-{}",
            std::process::id()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{dir:?} の削除に失敗: {e}"));
        }
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{dir:?} の作成に失敗: {e}"));
        run(&dir, &["init", "-q"]);
        dir
    }

    fn default_patterns() -> Vec<String> {
        vec![
            "assert!".to_string(),
            "abs() <".to_string(),
            "1e-[0-9]".to_string(),
        ]
    }

    /// ケース 1: テスト単独緩和（`mod tests` 内の削除行に `1e-6`）→ `true`。
    /// 受け入れ条件の最小再現。
    #[test]
    fn test_only_tolerance_loosening_matches() {
        let dir = init_repo("case1");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(matched, "テスト単独の許容誤差緩和で match しない");
    }

    /// ケース 2: 許容誤差緩和＋本番コード変更（G1 相当）→ `false`
    /// （`without_prod_change` 条件）。
    #[test]
    fn tolerance_loosening_with_prod_change_does_not_match() {
        let dir = init_repo("case2");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn double(a: i32) -> i32 { a * 2 }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn double(a: i32) -> i32 { a * 3 }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(!matched, "本番コード変更を伴う場合は match してはいけない");
    }

    /// ケース 3: 本番コードのみの変更 → `false`。
    #[test]
    fn prod_only_change_does_not_match() {
        let dir = init_repo("case3");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn double(a: i32) -> i32 { a * 2 }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn double(a: i32) -> i32 { a * 3 }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(
            !matched,
            "テストの削除行にパターン一致がない場合は match しない"
        );
    }

    /// ケース 4: 新規テスト追加のみ（削除行なし）→ `false`
    /// （PoC-3 発見事項 3: S2 誤検知解消の維持）。
    #[test]
    fn new_test_addition_only_does_not_match() {
        let dir = init_repo("case4");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n\n    #[test]\n    \
             fn works_more() {\n        assert!((2.0f32 - 2.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(!matched, "削除行のない新規テスト追加のみでは match しない");
    }

    /// ケース 5: `"1e-[0-9]"` の特別扱い（`1e-x` は不一致・`1e-5` は一致）・
    /// 空文字列パターンは無視。
    #[test]
    fn one_e_minus_digit_token_and_empty_pattern_handling() {
        let patterns = vec!["1e-[0-9]".to_string(), String::new()];
        assert!(
            line_matches_any_pattern("-        assert!(diff < 1e-5);", &patterns),
            "1e-5 は 1e-[0-9] トークンに一致するはず"
        );
        assert!(
            !line_matches_any_pattern("-        assert!(diff < 1e-x);", &patterns),
            "1e-x は数字でないため 1e-[0-9] トークンに一致しないはず"
        );
        assert!(
            !line_matches_any_pattern("-        anything at all", &["".to_string()]),
            "空文字列パターンは全行一致を防ぐため無視されるはず"
        );
    }

    /// `validate_unified_diff_shape` の直接検証: 空 stdout・正常な unified
    /// diff は許容し、`diff --git ` ヘッダを含まない出力（ANSI カラー化等で
    /// ヘッダが変形された想定）は `DiffUnexpectedFormat` として拒否する
    /// （Bugbot 指摘 Low・#123）。
    #[test]
    fn validate_unified_diff_shape_accepts_normal_output_and_rejects_malformed_output() {
        assert!(validate_unified_diff_shape("git diff", "").is_ok());
        assert!(
            validate_unified_diff_shape("git diff", "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n")
                .is_ok()
        );
        assert!(matches!(
            validate_unified_diff_shape("git diff", "\x1b[1mdiff --git a/x b/x\x1b[m\n"),
            Err(GuardrailError::DiffUnexpectedFormat { .. })
        ));
    }

    /// ケース 6: `tests/` ディレクトリ配下のみの変更（`src/` 外）での
    /// アサーション緩和 → `true`。
    #[test]
    fn tests_directory_only_change_matches() {
        let dir = init_repo("case6");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("tests")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();
        fs::write(
            dir.join("tests/integration.rs"),
            "#[test]\nfn works() {\n    assert!((1.0f32 - 1.0).abs() < 1e-6);\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("tests/integration.rs"),
            "#[test]\nfn works() {\n    assert!((1.0f32 - 1.0).abs() < 1e-2);\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(matched, "tests/ 配下のみの緩和は match するはず");
    }

    /// ケース 8: `mod tests` 直前の本番コード削除＋許容誤差緩和（座標系
    /// 不整合の回帰テスト。Review 指摘 Medium・#123）→ `false`。
    ///
    /// old 側の削除行数を k とすると、削除が `mod tests` 直前で終わる場合
    /// old 側ハンク開始行も new 側の `mod tests` 行番号もともに
    /// `M - k`（M は削除前の `mod tests` 行番号）になり、old 側座標系で
    /// 比較すると `M-k < M-k` は常に偽になる（本番コード削除を見逃す）。
    /// new 側座標系（[`hunk_new_start_lines`]）で比較することで、本番コード
    /// 削除を伴う変更が正しく `false`（match しない＝エスカレーション対象
    /// のまま）になることを検証する。
    #[test]
    fn prod_deletion_immediately_before_mod_tests_does_not_match() {
        let dir = init_repo("case8");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn a() {}\npub fn b() {}\npub fn c() {}\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        // `mod tests` 直前の本番関数 b・c を削除しつつ、許容誤差も緩和する
        // （REQ-4 ゲーミング検知の入力領域＝本番・テスト同時変更に該当する
        // ケース。本ルールは match してはいけない）。
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn a() {}\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(
            !matched,
            "mod tests 直前の本番コード削除を伴う場合は match してはいけない \
             （座標系不整合の回帰。#123）"
        );
    }

    /// ケース 9: `mod tests` の単純前方一致では `mod tests_helpers` の
    /// ような別モジュールを誤検知することの回帰確認（Review 指摘 Low・
    /// #123）。境界文字を要求する [`is_mod_tests_declaration`] を直接検証する。
    #[test]
    fn mod_tests_declaration_requires_word_boundary() {
        assert!(is_mod_tests_declaration("mod tests {"));
        assert!(is_mod_tests_declaration("    mod tests {"));
        assert!(is_mod_tests_declaration("mod tests;"));
        assert!(!is_mod_tests_declaration("mod tests_helpers {"));
        assert!(!is_mod_tests_declaration("mod testsuite {"));
        assert!(!is_mod_tests_declaration("mod other {"));
    }

    /// ケース 10: `mod tests` 境界が非標準名（単数形 `mod test`）で特定
    /// できない場合、テスト単独の許容誤差緩和が `true`（エスカレーション）
    /// になることの回帰確認（Bugbot 指摘 Medium・#123）。
    ///
    /// 旧実装は境界不明を `touches_prod_logic` 自体の `true`（本番コード
    /// 変更あり相当）へ丸めていたため、overall predicate の
    /// `!touches_prod_logic` が `false` になり、コメントが主張する
    /// 「安全側」とは逆に、非標準なテストモジュール名を持つファイルが
    /// 本ルールによるエスカレーションをすり抜けていた。
    #[test]
    fn unknown_test_boundary_escalates_instead_of_suppressing_match() {
        let dir = init_repo("case10");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod test {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod test {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        let prod_touch = touches_prod_logic(&dir, "HEAD").unwrap();
        assert_eq!(
            prod_touch,
            ProdTouch::UnknownBoundary,
            "単数形 mod test は境界として認識されず UnknownBoundary になるはず"
        );

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(
            matched,
            "mod tests 境界が非標準名で特定できない場合も安全側として \
             エスカレーション（match=true）すべき（Bugbot 指摘 Medium・#123）"
        );
    }

    /// ケース 11: 出力形式を変化させうる利用者の git 設定（`color.ui`・
    /// `diff.external`）を有効化しても、削除行パターンマッチングが無効化
    /// されないことの回帰確認（Bugbot 指摘 Low・#123）。
    #[test]
    fn hostile_git_config_does_not_defeat_pattern_matching() {
        let dir = init_repo("case11");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-6);\n    }\n}\n",
        )
        .unwrap();
        commit_all(&dir, "baseline");

        // diff 出力を変形しうるローカル設定を意図的に有効化する
        // （`-c` はこれらの設定より優先されるはずだが、優先されない場合は
        // 削除行が `^-` で始まらなくなりパターンマッチが無言で全滅する）。
        run(&dir, &["config", "--local", "color.ui", "always"]);
        run(&dir, &["config", "--local", "color.diff", "always"]);
        run(&dir, &["config", "--local", "diff.external", "cat"]);

        fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {\n        \
             assert!((1.0f32 - 1.0).abs() < 1e-2);\n    }\n}\n",
        )
        .unwrap();

        let matched =
            test_assertion_relaxation_without_prod_change(&dir, "HEAD", &default_patterns())
                .unwrap();
        assert!(
            matched,
            "利用者の gitconfig（color.ui=always・diff.external）が \
             パターンマッチングを無効化してはいけない（Bugbot 指摘 Low・#123）"
        );
    }

    /// ケース 7: 不正 baseline ref 等で git diff が失敗 → `Err(DiffFailed)`
    /// （`false` に丸めないことを検証。fail-closed）。
    #[test]
    fn invalid_baseline_ref_propagates_error_instead_of_false() {
        let dir = init_repo("case7");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/lib.rs"), "pub fn noop() {}\n").unwrap();
        commit_all(&dir, "baseline");

        let result = test_assertion_relaxation_without_prod_change(
            &dir,
            "this-ref-does-not-exist",
            &default_patterns(),
        );
        match result {
            Err(GuardrailError::DiffFailed { .. }) => {}
            other => panic!("DiffFailed を期待したが実際は: {other:?}"),
        }
    }

    /// `core.quotePath=false`（`git_command`）が非 ASCII パスの 8 進数
    /// エスケープを抑止することを固定する（計画 3.2 節「非 ASCII ファイル名
    /// を含む一時 git リポジトリでのユニットテスト」・モジュール冒頭
    /// 「変更ファイルパスの正規化契約」参照）。無効化されていない場合、
    /// `changed_files_for_policy_exclusion` は `"src/\346\227\245model.rs"`
    /// のような 8 進数エスケープ表記を返し、`PathPattern::matches` の単純な
    /// 文字列比較が無言で不一致になる（fail-open。#124）。
    #[test]
    fn non_ascii_path_is_not_octal_escaped() {
        let dir = init_repo("quote-path");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/lib.rs"), "pub fn noop() {}\n").unwrap();
        commit_all(&dir, "baseline");

        let non_ascii_file = "src/日本語model.rs";
        fs::write(dir.join(non_ascii_file), "pub fn noop2() {}\n").unwrap();
        // `git diff <baseline>` は index に存在しないパス（純粋な未追跡ファイル）
        // を差分に含めない。`add` で index に載せた時点で「baseline のツリーに
        // 存在しない新規パス」として diff に現れるようになる（commit は不要）。
        run(&dir, &["add", "-A"]);

        let changed = changed_files_for_policy_exclusion(&dir, "HEAD").unwrap();
        assert_eq!(
            changed,
            vec![non_ascii_file.to_string()],
            "非 ASCII パスが 8 進数エスケープ表記へ変換されている（core.quotePath \
             の無効化が効いていない）"
        );
    }
}
