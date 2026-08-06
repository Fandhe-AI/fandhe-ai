//! テンソル型・演算グラフ／カーネル融合機構の完全自作コア。
//!
//! Burn 等の既存 ML フレームワークに依存せず、テンソルの形状・ストレージ表現と
//! 演算グラフ（カーネル融合機構）を本クレートで自作する（REQ-1 v2。
//! `.claude/rules/coding-rust.md`）。`autodiff` クレートはこのテンソル表現の上に
//! 動的テープを構築し、`backend-cpu` / `backend-cuda` / `backend-metal` は
//! ここで定義する演算グラフのノードを各バックエンドのカーネルへ変換して実行する。
//!
//! TASK-1.4a でテンソル型のデータ構造（`Tensor<T>` の stride レイアウト・
//! `Arc<Storage<T>>` 所有権モデル・生成系／zero-copy view API）を実装済み。
//! TASK-1.4b（本クレート現段階・#12）では NumPy 互換ブロードキャスト
//! （`broadcast_shape`・`Tensor::broadcast_to`／`broadcast_with`。stride 0
//! による zero-copy view）を追加する。演算時 shape 検査の統合（#13・
//! TASK-1.4c）・PoC-v2-1 数値突合の総合テスト（#14・TASK-1.4d）・演算グラフ
//! 本体（カーネル融合機構）は後続タスクで追加する（spec 根拠:
//! `docs/spec/05-tasks.md` TASK-1.4、`docs/public-api-design.md` §2）。

mod broadcast;
mod element;
mod error;
mod tensor;

pub use broadcast::broadcast_shape;
pub use element::Element;
pub use error::ShapeError;
pub use tensor::Tensor;
