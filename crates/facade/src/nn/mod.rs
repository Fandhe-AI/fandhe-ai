//! `nn` 公開面の入口（イシュー #1955）。
//!
//! 現時点の子モジュールは [`crate::nn::rnn`]（`Rnn`／`Lstm`／`Gru` の
//! Sequence レベル API。`fandhe_ai_autodiff::nn` からの純再エクスポート）
//! のみである。それ以外の `nn` 層（`Linear`・活性化関数・正規化・
//! Conv・pooling・Embedding・MultiheadAttention 等）は本モジュール
//! ではなく `compat::Sequential::add_*`（`crate::compat::sequential`）
//! 経由で到達する契約のまま変更しない（[`crate::nn::rnn`] モジュール
//! doc「`Sequential::add_*` を設けない理由」参照）。
pub mod rnn;
