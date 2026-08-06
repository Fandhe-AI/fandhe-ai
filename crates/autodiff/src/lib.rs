//! 動的テープ式の自動微分エンジン。
//!
//! `tensor-core` が定義するテンソル型・演算グラフの上に、順伝播で実行した演算を
//! 動的テープへ記録し逆伝播で勾配を計算する（REQ-1 v2）。互換 API 層
//! （`compat::array` 等。REQ-9）はこのテープ機構を薄くラップして呼び出す想定であり、
//! 本クレート自体には互換レイヤ固有のロジックを持ち込まない。
//!
//! TASK-1.5a（本イシュー・#16）でテープ構造（`Tape`/`TapeId`/`NodeId`）・
//! forward 演算群（`Var::matmul`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/
//! `max`/`mse_loss`）の値計算とノード記録を実装した（spec 根拠:
//! `docs/spec/05-tasks.md` TASK-1.5、`docs/public-api-design.md` §3）。
//! `Op`（`tape.rs`・非公開）が各演算の入力 `NodeId` を保持する構造にする
//! ことで、後続タスクが発生順に記録されたノード列を逆走査できる下地とする。
//!
//! **残スコープ**（本イシューでは実装しない）:
//! - 各演算の backward 実装（TASK-1.5b・#17）
//! - `Tape::backward`・`Gradients`（勾配取得 API。TASK-1.5c・#18）
//! - PoC-v2-2 数値突合の回帰テスト（TASK-1.5d・#19）
//!
//! forward の値計算は `backend-cpu`（TASK-1.6・#20 以降。並行実装中で
//! 未完）が完成するまでの暫定参照実装（`eval.rs`、非公開）で行い、
//! TASK-1.9（バックエンド抽象層への接続）で backend 経由の実行に
//! 差し替える（PoC-v2-2 と同じ構成）。

mod error;
mod eval;
mod tape;
mod var;

pub use error::AutodiffError;
pub use tape::{NodeId, Tape, TapeId};
pub use var::Var;
