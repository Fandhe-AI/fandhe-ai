//! `nn` 公開面の入口（イシュー #1955）。
//!
//! facade 独自の [`crate::nn::Module`]（#2395。生の `Tape` を出さず [`crate::TapeRef`] を取る
//! 薄い trait。非公開 `mod module` と `pub use` で公開する）と、
//! 子モジュール [`crate::nn::init`]（`torch.nn.init.*` 相当の初期化関数群。#2504。
//! `fandhe_ai_autodiff::nn::init` からの純再エクスポート）と [`crate::nn::rnn`]（`Rnn`／`Lstm`／`Gru` の
//! Sequence レベル API。`fandhe_ai_autodiff::nn` からの純再エクスポート）
//! に加え、facade 側コンテナ `ModuleList`／`Sequential`（#2396。同じく非公開
//! `mod container` と `pub use`）と、`ModuleDict`／`summary`（#2402。autodiff #2134 の鏡写し）を提供する。それ以外の `nn` 層（`Linear`・活性化関数・正規化・
//! Conv・pooling・Embedding・MultiheadAttention 等）は本モジュール
//! ではなく `compat::Sequential::add_*`（`crate::compat::sequential`）
//! 経由で到達する契約のまま変更しない（[`crate::nn::rnn`] モジュール
//! doc「`Sequential::add_*` を設けない理由」参照）。
//! 例外として [`crate::nn::kv_cache`]（#2579。`KvCache`・`StatefulAttention` の純再エクスポート）を
//! 持つ。MHA 本体は引き続き `Sequential::add_*` 経由のみ。
//! **#2587（hooks の承認形）**: facade 版 [`crate::nn::ForwardHooked`]（`nn::Module` 実装を包み
//! `forward` 直後に観察専用 hook を 1 回呼ぶラッパー。非公開 `mod forward_hook` と `pub use`）と、その
//! hook の引数型 [`crate::nn::ForwardHookCtx`]（autodiff からの純再エクスポート）を提供する
//! （`docs/autodiff-forward-backward-hooks-design.md` §14.4・§18。`pub mod hooks` は設けない）。
//!
//! **例外（#2532・#2533。ルート #2499 の一括承認）**: `Transformer`・
//! `TransformerDecoderLayer`・`TransformerConfig` は型として再エクスポートする
//! （`docs/autodiff-transformer-decoder-decision.md` §承認事項の承認形）。構築と学習は
//! `compat::Sequential::add_transformer_decoder_layer`・`add_transformer` 経由で行う。
//! 単体構築（`Transformer::new`・`TransformerDecoderLayer::new`）には
//! `FeedForwardActivation` 等（facade 未公開）が必要で、facade だけでは行えない
//! （承認形の範囲外のため `FeedForwardActivation` や `Tape` への委譲メソッドは追加していない）。
mod container;
mod forward_hook;
pub mod init;
pub mod kv_cache;
mod module;
pub mod rnn;

pub use container::{ModuleList, Sequential};
// #2402: autodiff `ModuleDict`／`summary`（#2134）の鏡写し。上の行とは別の行で宣言する
// （`api_surface.rs` のトークン走査が両行の完全一致を別々に検査するため）。
pub use container::{ModuleDict, summary};
pub use module::Module;
// #2587: forward hook の facade 版ラッパー（P5′）と ctx 型（autodiff からの純再エクスポート）。
// 各 1 文 1 行で宣言する（`api_surface.rs` が行単位で固定する）。
pub use fandhe_ai_autodiff::nn::ForwardHookCtx;
pub use forward_hook::ForwardHooked;
// #2532・#2533: Transformer／decoder 1 層の再エクスポート。上の行とは別の独立した 1 行で宣言する
// （`api_surface.rs::facade_reexports_transformer_decoder_items_only_in_approved_shape` が
// この 1 文の完全一致を要求する。#2533 で `Transformer` を加えた 3 名形）。
pub use fandhe_ai_autodiff::nn::{Transformer, TransformerConfig, TransformerDecoderLayer};
// `compat::Sequential::add_module`（#2398）が独自層を autodiff 側 `Module` へ包む crate 内専用の
// アダプタ。`pub(crate)` のため公開面には現れない。
pub(crate) use module::FacadeModuleAdapter;
