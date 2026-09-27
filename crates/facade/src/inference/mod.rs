//! バッチ推論・phase 計測の内部実装モジュール（イシュー #2192・親
//! #2131）。`crate::compat::sequential::Sequential` の DataLoader 反復
//! 推論（[`batch::run_loader_inference`] 相当。実体は `Sequential` の
//! inherent メソッドとして `compat::sequential` 側に置く）が
//! [`batch`]（フェーズ計測の型・thread-local・入力トレイト）を利用する。
//!
//! **facade 公開は保留**（`docs::facade-predict-batches-phase-metrics-decision.md`
//! §5「承認事項」）。このため本モジュールは `pub mod` にせず**非公開**
//! （`crate::lib.rs` の `mod inference;`）とし、`inference` という
//! モジュール名自体も facade の公開面に現れない。承認後は本モジュールを
//! `pub mod` へ昇格し、`batch` 配下の該当型・関数を `pub` にする想定。
pub(crate) mod batch;
