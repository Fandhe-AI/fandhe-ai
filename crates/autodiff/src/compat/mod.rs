//! numpy/Keras 慣習の互換 API 層（REQ-9・TASK-9.2a・#95）の**非推奨シム**。
//!
//! **TASK-9.4（#411）で `compat` の唯一のサポート対象実装は `fandhe_ai::compat`
//! へ移設した**（`docs/compat-api-scope.md` §0「サポート境界」・
//! `crates/facade/src/compat/`）。以後の新規開発・ドキュメントは
//! `fandhe_ai::compat::{array, Sequential}` を正とする。
//!
//! 本モジュールは移設前の `fandhe_ai_autodiff::compat::{array, Sequential,
//! SequentialVars}` を利用する既存コードのソース互換性のみを保つ
//! **移行期間中の非推奨シム**である（codex-review PR #424 P1 是正。
//! 移設それ自体が破壊的変更である一方、公開 API パスの即時撤去は
//! `.claude/rules/coding-rust.md`／ベース側レビュー基準「公開 API の
//! 破壊的変更は P1」に反するため、移行期間を設ける）。
//!
//! - 実装は `fandhe_ai::compat` 側と論理的に重複する（意図的な重複。
//!   `autodiff` は `facade` に依存できない〈依存方向が逆〉ため、
//!   委譲ではなくコードの複製によってのみソース互換を保てる）
//! - `Sequential::predict` は移設前と同じ挙動（`default_ops::naive_ops()`
//!   による naive CPU 参照実装。具体バックエンドクレートに依存しない）を
//!   維持する。旧 `predict_with_ops`（任意 `BackendOps` 注入経路）は
//!   `fandhe_ai::compat::Sequential` 側で既に撤去済み（REQ-12「任意
//!   `BackendOps` 実装を注入できる公開 API を設けない」）のため、本シムでも
//!   復元しない（撤去自体は破壊的変更として維持する。`docs/
//!   compat-api-scope.md` §2）
//! - 撤去予定: `fandhe_ai::compat` への移行が完了し利用実績が確認でき次第、
//!   別イシューで本モジュールごと削除する（`.claude/rules/
//!   out-of-scope-tracking.md` 対象。撤去時期は正本 spec の改定を要さない
//!   実装リポ側の判断）
//!
//! **配置の経緯（TASK-9.2a 時点の確定・履歴）**: `docs/compat-api-scope.md`
//! §4.1 参照。`Sequential` は `nn::Linear`/`nn::Module` に依存し、
//! `tensor-core` は `autodiff` に依存できない（下位クレートが上位クレートへ
//! 依存すると循環する）ため、当時は `autodiff` 配下以外に置く選択肢が
//! なかった。TASK-9.3（facade 新設・#410）以降は `facade` が compat の
//! 正式な置き場所となった。

mod array;
mod sequential;

#[allow(deprecated)] // 非推奨シム自身の再エクスポート（本モジュール doc 参照）。
pub use array::{ArrayData, array};
#[allow(deprecated)]
pub use sequential::{Sequential, SequentialVars};
