//! `#[cfg(test)]` 専用のテストダブル集約（TASK-3.1b・イシュー #133）。
//!
//! [`crate::bug_fix`]・[`crate::feature_addition`] のユニットテストが実 cargo を
//! 起動せず検出・修正生成ロジックのみを検証するために使う
//! [`ScriptedCommand`]（[`crate::exec::CommandRunner`] 実装）を 1 箇所に集約する
//! （v1 `tools/self-repair/src/test_support.rs` と同じ集約方針。逐語複製を防ぐ）。
//! 一時ディレクトリは `tempfile`（許容依存 8 区分外・依存追加はユーザー承認
//! 事項）を使わず、v2 既存慣行（`crates/guardrail/tests/eval_harness.rs` 等）に
//! 倣い `std::env::temp_dir()` + `std::process::id()` による一意ディレクトリで
//! 代替する（実装計画セクション 2）。
//!
//! 公開 API には含めない（`lib.rs` で `#[cfg(test)] pub(crate) mod test_support;`
//! として登録し、テストビルド時のみ他モジュールの `#[cfg(test)]` から参照可能）。

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::exec::{CommandOutput, CommandRunner, ExecError};

/// `ScriptedCommand::new` の入力 1 要素分（`(program, args)` と返す結果の組）。
/// clippy::type_complexity 対応で型エイリアスへ切り出す。
pub(crate) type ScriptedResponse<'a> = ((&'a str, &'a [&'a str]), Result<CommandOutput, ExecError>);

/// [`crate::exec::CommandRunner`] のテストダブル。`(program, args)` の組を
/// キーに事前設定した結果を返す。実 cargo を起動せず検出・生成ロジックのみを
/// 検証する（実 cargo 実行は [`crate::exec`] のスモークテストで別途検証する）。
///
/// spawn 失敗を模擬する場合は `Err(ExecError::new(reason))` を渡す
/// （`crate::exec::CommandRunner::run` の戻り値型に揃える。TASK-3.1c・#134）。
pub(crate) struct ScriptedCommand {
    responses: HashMap<(String, Vec<String>), Result<CommandOutput, ExecError>>,
    calls: RefCell<Vec<(String, Vec<String>)>>,
}

impl ScriptedCommand {
    pub(crate) fn new(responses: Vec<ScriptedResponse<'_>>) -> Self {
        let responses = responses
            .into_iter()
            .map(|((program, args), result)| {
                (
                    (
                        program.to_string(),
                        args.iter().map(|s| s.to_string()).collect(),
                    ),
                    result,
                )
            })
            .collect();
        ScriptedCommand {
            responses,
            calls: RefCell::new(Vec::new()),
        }
    }

    /// `run` の累計呼び出し回数（fail-closed 経路が余計な再実行をしないこと
    /// の検証に使う想定）。本イシュー（#133）時点では利用テストがなく
    /// `cargo clippy` の non-test 判定外でも到達できない dead_code 扱いとなる
    /// ため、意図的に `#[allow]` する（`crate::outcome::VerifiedEvidence::new`
    /// と同じ経過的な扱い。検証ゲート実実行〔#134〕が呼び出し回数を検証する
    /// テストを追加し次第、外れる想定）。
    #[allow(dead_code)]
    pub(crate) fn call_count(&self) -> usize {
        self.calls.borrow().len()
    }
}

impl CommandRunner for ScriptedCommand {
    fn run(&self, program: &str, args: &[&str], _cwd: &Path) -> Result<CommandOutput, ExecError> {
        let key = (
            program.to_string(),
            args.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        );
        self.calls.borrow_mut().push(key.clone());
        self.responses.get(&key).cloned().unwrap_or_else(|| {
            Err(ExecError::new(format!(
                "スクリプト未設定の呼び出し: {key:?}"
            )))
        })
    }
}

/// `cargo test --release` が成功したことを表すスクリプト応答（両種別の
/// `Detector` テストが共用する）。
pub(crate) fn passing_test_response() -> (
    (&'static str, &'static [&'static str]),
    Result<CommandOutput, ExecError>,
) {
    (
        ("cargo", &["test", "--release"]),
        Ok(CommandOutput::from_captured(
            true,
            b"test result: ok".to_vec(),
        )),
    )
}

/// `cargo test --release` が失敗したことを表すスクリプト応答。失敗ログ文言
/// （どのテストが FAILED か）は種別ごとに異なるため引数化する。
pub(crate) fn failing_test_response(
    log: &'static str,
) -> (
    (&'static str, &'static [&'static str]),
    Result<CommandOutput, ExecError>,
) {
    (
        ("cargo", &["test", "--release"]),
        Ok(CommandOutput::from_captured(false, log.as_bytes().to_vec())),
    )
}

/// `dir` 配下に `rel`（相対パス）のフィクスチャファイルを書き込む。親
/// ディレクトリが存在しない場合は作成する。`BugFixFixGenerator`/
/// `FeatureAdditionFixGenerator` の baseline・候補適用テストが共用する。
///
/// 本番経路ではないテストユーティリティのため `expect` の使用を許容する
/// （`.claude/rules/coding-rust.md` の unwrap/expect 禁止は本番経路が対象）。
pub(crate) fn write_workspace_file(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("親ディレクトリ作成に失敗");
    }
    std::fs::write(path, content).expect("フィクスチャファイル書き込みに失敗");
}

/// テストごとに衝突しない一時ディレクトリを作成し、その絶対パスを返す。
///
/// `tempfile` クレート（許容依存 8 区分外）を導入せず、`std::env::temp_dir()`
/// とプロセス ID・単調増加カウンタの組み合わせでテスト間の一意性を確保する
/// （同一プロセス内で複数テストが並行実行されても衝突しない。
/// `crates/guardrail/tests/eval_harness.rs` と同じ一意化パターン）。呼び出し元は
/// テスト終了時に明示的な削除を行わない（OS の一時領域クリーンアップに委ねる。
/// CI runner は使い捨てのため許容する）。
pub(crate) fn unique_temp_dir(test_name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "self-repair-test-{}-{}-{seq}",
        std::process::id(),
        test_name
    ));
    std::fs::create_dir_all(&dir).expect("一時ディレクトリ作成に失敗");
    dir
}
