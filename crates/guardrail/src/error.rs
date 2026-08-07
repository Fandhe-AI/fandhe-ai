//! `guardrail` クレート全体で共有する型付きエラー。
//!
//! `.claude/rules/coding-rust.md`「エラーは型付きエラーとし、本番経路で
//! `unwrap()` / `expect()` を使わない」に対応する。`thiserror` は
//! `.claude/rules/deps-policy.md` の許容依存 8 区分に含まれず、依存追加は
//! ユーザー承認事項のため（`docs/spec/05-tasks.md` TASK-4.1a 計画 2.1 節）、
//! `std::fmt::Display` / `std::error::Error` を手書きする。
//!
//! `main.rs` はここで定義するバリアントを `exit_code.rs` の終了コード契約
//! （`docs/guardrail-self-repair-cli.md` 2.3 節）に写像する。本エラー自体は
//! 終了コードを持たず、「何が起きたか」のみを表す（写像は 1 箇所に閉じ込め、
//! fail-closed 契約を保つため。`.claude/rules/security.md` A08）。
//!
//! `InconsistentDecisionInput` は [`crate::decision::DecisionInput::new`]
//! （TASK-4.1c・イシュー #106）が検出する入力矛盾（`GateSignals`/`BenchSignal`
//! の実行順序契約違反）専用のバリアントであり、CLI 引数パース・設定ファイル
//! 検証由来のエラー（TASK-4.1a・イシュー #104 管轄）とは別 PR で追加された
//! （`.claude/rules/delegation-impl.md`: 同一ファイルの並行編集回避のため
//! 各 PR は必要なバリアントのみを追加する運用）。TASK-4.1b（イシュー #105）が
//! 移植する閾値体系の値域検証は `config.rs`（#104 が定義する `InvalidInput`）
//! 経由でエラーを返す契約に合流し、本モジュールへ専用バリアントを追加しない。

use std::fmt;
use std::path::PathBuf;

/// `guardrail` の実行時に起こりうるエラー。
///
/// バリアントは呼び出し元（`main.rs`）が終了コードへ変換する際の分岐単位
/// と一致させてある。新しい失敗モードを追加する際は、変換表
/// （`docs/guardrail-self-repair-cli.md` 2.3 節）とセットで見直すこと。
#[derive(Debug)]
pub enum GuardrailError {
    /// CLI 引数の解析・検証に失敗した（未知引数・値欠落・不正なプリセット名等）。
    /// 終了コード契約（2.3 節）上は clap 相当の usage エラー区分（`2`）に対応する。
    UsageError(String),

    /// `--config` の TOML・`--signals` の JSON など外部入力の読み込み・
    /// 検証に失敗した（未知フィールド・値域外・サイズ上限超過・必須フィールド欠落等）。
    /// 内部エラー区分（終了コード `1`）に対応する。
    InvalidInput(String),

    /// ファイル I/O に失敗した（読み込み・書き込み双方）。
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    /// `--signals` が環境変数 `GUARDRAIL_ALLOW_INJECTED_SIGNALS=1` なしで
    /// 指定された（1.2 節「`--signals` の迂回防止」入口ガード）。usage エラー
    /// 区分（終了コード `2`）に対応する。
    InjectedSignalsNotAllowed,

    /// 判定・評価ロジック本体が未実装であることを示す（TASK-4.1a のスコープ外。
    /// eval 評価ロジックは #108/#111 で実装する）。
    /// 内部エラー区分（終了コード `1`）に対応する。
    NotImplemented(&'static str),

    /// [`crate::decision::DecisionInput::new`] が検出した判定入力の矛盾。
    ///
    /// 例: build/test/clippy が全通過していないにもかかわらずベンチ計測結果
    /// （`BenchSignal::Measured`）が渡された場合（PoC-3 の実行順序契約
    /// 「ベンチはゲート全通過時のみ計測する」への違反。呼び出し側バグとみなし
    /// 判定を続行せず拒否する。`.claude/rules/security.md` A08）。
    InconsistentDecisionInput { reason: String },
}

impl fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuardrailError::UsageError(msg) => write!(f, "usage error: {msg}"),
            GuardrailError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            GuardrailError::Io { path, source } => {
                write!(f, "io error at {}: {source}", path.display())
            }
            GuardrailError::InjectedSignalsNotAllowed => write!(
                f,
                "usage error: --signals requires GUARDRAIL_ALLOW_INJECTED_SIGNALS=1"
            ),
            GuardrailError::NotImplemented(what) => write!(f, "not implemented: {what}"),
            GuardrailError::InconsistentDecisionInput { reason } => {
                write!(f, "判定入力が矛盾しています: {reason}")
            }
        }
    }
}

impl std::error::Error for GuardrailError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GuardrailError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
