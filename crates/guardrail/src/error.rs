//! `guardrail` クレート全体で共有する型付きエラー。
//!
//! coding-rust.md / security.md の方針により、本番経路（`config`・`decision`・
//! `signals` の各検証処理）は `unwrap()` / `expect()` を使わず、失敗理由を
//! ここで定義した variant で呼び出し元へ返す（security.md A08: 判定を
//! 迂回して成功終了させない）。CLI 層（#104）は本エラーを
//! `GuardrailExitCode::Internal`（終了コード `1`）へ変換し、判定不能を
//! 自動適用（`0`）と明確に分離する（`docs/guardrail-self-repair-cli.md` §2.3
//! fail-closed 設計）。
//!
//! v1（`Fandhe-AI/rust-ai-library-v1/crates/guardrail/src/error.rs`）は
//! `thiserror` で 20 種超の variant（設定ファイル I/O・CLI・git・ゲート実行・
//! eval 等）を定義するが、`thiserror` は v2 の許容依存 8 区分
//! （`.claude/rules/deps-policy.md`）に含まれず追加はユーザー承認が必須のため
//! （TASK-4.1b・イシュー #105 は自動運転につき追加を回避する）、本クレートでは
//! `std::fmt::Display` / `std::error::Error` を手書き実装する。variant は
//! TASK-4.1b（閾値体系の値域検証）・TASK-4.1c（#106・判定入力の矛盾検出）が
//! それぞれ必要とする分のみを移植し、ファイル I/O 系 variant（v1 の
//! `ConfigIo`/`ConfigTooLarge`/`ConfigParse` 等）は CLI・設定ファイルパースを
//! 担当する #104（TASK-4.1a）が追加する。

use std::fmt;

/// `guardrail` クレート全体のエラー型。
///
/// fail-closed 方針（security.md A05）に基づき、いずれの variant も
/// 「実行を拒否する」ことを意味し、緩和した既定値へのフォールバックは行わない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailError {
    /// 構文は正しいが値域検証（`lines_max` 正整数・`bench_max_pct` 有限正数・
    /// `bench_runs >= 5` 等、REQ-4 の初期推奨閾値の型的制約）に反した。
    ConfigInvalidValue { preset: String, reason: String },

    /// プリセット名（`strict`/`default`/`loose`）以外の文字列が指定された。
    UnknownPreset { preset: String },

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
            GuardrailError::ConfigInvalidValue { preset, reason } => {
                write!(f, "設定値が不正です: preset={preset} ({reason})")
            }
            GuardrailError::UnknownPreset { preset } => {
                write!(
                    f,
                    "未定義のプリセットです: {preset}（strict/default/loose のいずれでもありません）"
                )
            }
            GuardrailError::InconsistentDecisionInput { reason } => {
                write!(f, "判定入力が矛盾しています: {reason}")
            }
        }
    }
}

impl std::error::Error for GuardrailError {}
