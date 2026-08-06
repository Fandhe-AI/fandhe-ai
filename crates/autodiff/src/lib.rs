//! 動的テープ式の自動微分エンジン。
//!
//! `tensor-core` が定義するテンソル型・演算グラフの上に、順伝播で実行した演算を
//! 動的テープへ記録し逆伝播で勾配を計算する（REQ-1 v2）。互換 API 層
//! （`compat::array` 等。REQ-9）はこのテープ機構を薄くラップして呼び出す想定であり、
//! 本クレート自体には互換レイヤ固有のロジックを持ち込まない。
//!
//! TASK-1.5a（#16）でテープ構造（`Tape`/`TapeId`/`NodeId`）・
//! forward 演算群（`Var::matmul`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/
//! `max`/`mse_loss`）の値計算とノード記録を実装した（spec 根拠:
//! `docs/spec/05-tasks.md` TASK-1.5、`docs/public-api-design.md` §3）。
//! `Op`（`tape.rs`・非公開）が各演算の入力 `NodeId` を保持する構造にする
//! ことで、後続タスクが発生順に記録されたノード列を逆走査できる下地とする。
//!
//! TASK-1.5b（#17）で各演算の勾配関数（VJP: vector-Jacobian product）と
//! `Op` 単位のディスパッチ入口 `vjp()`（`grad.rs`・非公開）を実装した。
//! 数値微分との突合テスト（受け入れ条件）は `grad.rs` 内のユニット
//! テストに含む。
//!
//! TASK-1.5c（本イシュー・#18）で勾配伝播 API（`Tape::backward`・
//! `Gradients`。`backward.rs`）を実装した。テープを発生順とは逆順に
//! 走査して `grad::vjp()` を呼び、複数経路から同一ノードへ流入する
//! 勾配を合算する（PoC-v2-2 の `accumulate()` 相当）。合成関数
//! end-to-end 勾配の受け入れ条件検証は `tests/backward.rs` に含む。
//!
//! **残スコープ**（本イシューでは実装しない）:
//! - PoC-v2-2 数値突合の回帰テスト（TASK-1.5d・#19）
//!
//! forward の値計算は `backend-cpu`（TASK-1.6・#20 以降。並行実装中で
//! 未完）が完成するまでの暫定参照実装（`eval.rs`、非公開）で行い、
//! TASK-1.9（バックエンド抽象層への接続）で backend 経由の実行に
//! 差し替える（PoC-v2-2 と同じ構成）。`grad.rs`・`backward.rs` も同じ
//! `eval.rs` のヘルパーを再利用するため、差し替えの影響範囲は
//! forward/backward 双方でこの 1 ファイルに閉じる。

mod backward;
mod error;
mod eval;
mod grad;
mod tape;
mod var;

pub use backward::Gradients;
pub use error::AutodiffError;
pub use tape::{NodeId, Tape, TapeId};
pub use var::Var;
