//! `guardrail` クレート共通のエラー型。
//!
//! 本イシュー（#106・TASK-4.1c）では [`crate::decision::DecisionInput::new`] が
//! 検出する入力矛盾（`GateSignals`/`BenchSignal` の実行順序契約違反）のみを
//! 定義する。CLI 引数パース・設定ファイル検証由来のエラー（#104・TASK-4.1a
//! 管轄）、5 条件の実測エラー（#105・TASK-4.1b 管轄）は本 PR のスコープ外
//! であり、それぞれの実装 PR で本 enum へバリアントを追記する想定の受け口
//! とする（`.claude/rules/delegation-impl.md`: 同一ファイルの並行編集回避の
//! ため、本 PR は本イシューが必要とするバリアントのみを追加する）。
use std::fmt;

/// `guardrail` クレート全体のエラー型。
///
/// 本番経路で `unwrap()`/`expect()` を使わない方針（`.claude/rules/coding-rust.md`
/// 「コード品質」）に対応する型付きエラー。CLI 層（#104）は本エラーを
/// `GuardrailExitCode::Internal`（終了コード `1`）へ変換し、判定不能を
/// 自動適用（`0`）と明確に分離する（`docs/guardrail-self-repair-cli.md` §2.3
/// fail-closed 設計）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailError {
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
            GuardrailError::InconsistentDecisionInput { reason } => {
                write!(f, "判定入力が矛盾しています: {reason}")
            }
        }
    }
}

impl std::error::Error for GuardrailError {}
