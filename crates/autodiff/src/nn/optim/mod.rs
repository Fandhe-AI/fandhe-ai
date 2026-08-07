//! optimizer（親イシュー #192「optimizer（SGD・AdamW）・gradient
//! clipping の実装」・REQ-9・M3）。
//!
//! `nn`（`Linear`・活性化・損失。`nn/mod.rs` 参照）で組んだ計算グラフの
//! 逆伝播結果（`Tape::backward` が返す `Gradients`）を消費し、
//! パラメータの `Tensor<f32>` を更新後の値へ差し替える入口を置く。
//! 既存の学習ループ（`tests/nn_train_convergence.rs::sgd_step`・
//! `tests/poc_v2_2_parity.rs::sgd_step`）が採る不変更新パターン
//! （`Linear` はパラメータを不変に保持し、`Linear::from_parameters` で
//! 更新後の値を持つ新しい `Linear` に差し替える）にそのまま接続できる
//! よう、optimizer の `step()` は `(param, grad)` の参照列を受け取り
//! 更新後 `Tensor<f32>` の列を返す形にする（呼び出し元が層を再構築
//! する。`adamw.rs` の doc 参照）。
//!
//! 本イシュー（#194）で AdamW（[`AdamW`]・[`AdamWConfig`]）を実装する。
//! SGD は #193、gradient clipping・LR スケジューラは #195 のスコープ。
//! `nn/mod.rs` の `Module` trait 未定義方針と同様、共通 `Optimizer`
//! trait の定義は本イシューでは行わない（並行実装される #193/#195 と
//! 一方的に API を固定しないため。親 #192 の統合時に判断する）。

mod adamw;

pub use adamw::{AdamW, AdamWConfig};
