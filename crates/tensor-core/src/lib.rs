//! テンソル型・演算グラフ／カーネル融合機構の完全自作コア。
//!
//! Burn 等の既存 ML フレームワークに依存せず、テンソルの形状・ストレージ表現と
//! 演算グラフ（カーネル融合機構）を本クレートで自作する（REQ-1 v2。
//! `.claude/rules/coding-rust.md`）。`autodiff` クレートはこのテンソル表現の上に
//! 動的テープを構築し、`backend-cpu` / `backend-cuda` / `backend-metal` は
//! ここで定義する演算グラフのノードを各バックエンドのカーネルへ変換して実行する。
//!
//! TASK-1.4a（本クレート現段階）ではテンソル型のデータ構造（`Tensor<T>` の
//! stride レイアウト・`Arc<Storage<T>>` 所有権モデル・生成系／zero-copy view
//! API）を実装する。ブロードキャスト機構（#12・TASK-1.4b）・演算時 shape 検査
//! （#13・TASK-1.4c）・演算グラフ本体（カーネル融合機構）は後続タスクで追加する
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.4、`docs/public-api-design.md` §2）。

mod element;
mod error;
mod tensor;

pub use element::Element;
pub use error::ShapeError;
pub use tensor::Tensor;
