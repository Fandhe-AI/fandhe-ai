//! `nn` 公開面の入口（イシュー #1955）。
//!
//! facade 独自の [`crate::nn::Module`]（#2395。生の `Tape` を出さず [`crate::TapeRef`] を取る
//! 薄い trait。非公開 `mod module` と `pub use` で公開する）と、
//! 子モジュール [`crate::nn::rnn`]（`Rnn`／`Lstm`／`Gru` の
//! Sequence レベル API。`fandhe_ai_autodiff::nn` からの純再エクスポート）
//! に加え、facade 側コンテナ `ModuleList`／`Sequential`（#2396。同じく非公開
//! `mod container` と `pub use`）と、`ModuleDict`／`summary`（#2402。autodiff #2134 の鏡写し）を提供する。それ以外の `nn` 層（`Linear`・活性化関数・正規化・
//! Conv・pooling・Embedding・MultiheadAttention 等）は本モジュール
//! ではなく `compat::Sequential::add_*`（`crate::compat::sequential`）
//! 経由で到達する契約のまま変更しない（[`crate::nn::rnn`] モジュール
//! doc「`Sequential::add_*` を設けない理由」参照）。
mod container;
mod module;
pub mod rnn;

pub use container::{ModuleList, Sequential};
// #2402: autodiff `ModuleDict`／`summary`（#2134）の鏡写し。上の行とは別の行で宣言する
// （`api_surface.rs` のトークン走査が両行の完全一致を別々に検査するため）。
pub use container::{ModuleDict, summary};
pub use module::Module;
// `compat::Sequential::add_module`（#2398）が独自層を autodiff 側 `Module` へ包む crate 内専用の
// アダプタ。`pub(crate)` のため公開面には現れない。
pub(crate) use module::FacadeModuleAdapter;
