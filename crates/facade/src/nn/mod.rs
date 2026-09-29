//! `nn` 公開面の入口（イシュー #1955）。
//!
//! facade 独自の [`crate::nn::Module`]（#2395。生の `Tape` を出さず [`crate::TapeRef`] を取る
//! 薄い trait。非公開 `mod module` と `pub use` で公開する）と、
//! 子モジュール [`crate::nn::rnn`]（`Rnn`／`Lstm`／`Gru` の
//! Sequence レベル API。`fandhe_ai_autodiff::nn` からの純再エクスポート）
//! を提供する。`ModuleList`／`Sequential` は #2396 まで保留。それ以外の `nn` 層（`Linear`・活性化関数・正規化・
//! Conv・pooling・Embedding・MultiheadAttention 等）は本モジュール
//! ではなく `compat::Sequential::add_*`（`crate::compat::sequential`）
//! 経由で到達する契約のまま変更しない（[`crate::nn::rnn`] モジュール
//! doc「`Sequential::add_*` を設けない理由」参照）。
mod module;
pub mod rnn;

pub use module::Module;
