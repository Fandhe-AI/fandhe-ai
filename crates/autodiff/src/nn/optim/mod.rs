//! optimizer 周辺の最小構成部品（親イシュー #192「optimizer（SGD・AdamW）・
//! gradient clipping の実装」・REQ-9・M3）。
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
//! 本イシュー（#194）で AdamW（[`AdamW`]・[`AdamWConfig`]）を実装した。
//! SGD（momentum・dampening・weight decay・nesterov 対応。PyTorch
//! `torch.optim.SGD` 準拠）は `crate::optim`（`Tape`/`Var` から独立した
//! 純粋な optimizer 群を置く別モジュール。#193）に実装済みで、AdamW
//! （本モジュール）とはモジュールの置き場所が異なる。統合は親 #192
//! 完了時に判断する。
//!
//! **適用順序契約**: 1 学習ステップは必ず
//! `backward → (AMP 導入後の unscale) → clip → optimizer step`
//! の順で実行する。損失スケーリング（AMP）は現時点で未実装のため
//! unscale ステップは存在しないが、将来 AMP を導入する際も
//! 「clip は unscale 後の生勾配に対してのみ適用する」契約を崩さない
//! （clip 前に scale が残っていると `max_norm` の意味が変わり、
//! 意図しない過剰クリップ・過小クリップを招くため。仕様突合
//! 2026-08-06・#192 本文）。本モジュールはこの契約を doc として固定し、
//! [`clip`] にテスト（`nn_optim_clip.rs`）で正順・逆順の不一致を回帰化する。
//!
//! gradient clipping・LR スケジューラ（本イシュー・#195）を追加した。
//! [`clip::clip_grad_norm`]／[`lr_scheduler::LrScheduler`] は
//! `Gradients`/`Var` に依存しない純関数・純データ構造として実装し、
//! `crate::optim::Sgd`（#193）・[`AdamW`]（#194）等の optimizer 実装
//! からそのまま呼び出せる形にする（疎結合設計）。
//! `nn/mod.rs` の `Module` trait 未定義方針と同様、共通 `Optimizer`
//! trait の定義は本イシューでは行わない（並行実装される #193/#194 と
//! 一方的に API を固定しないため。親 #192 の統合時に判断する）。

mod adamw;

pub mod clip;
pub mod lr_scheduler;

pub use adamw::{AdamW, AdamWConfig};
pub use clip::{ClipGradResult, clip_grad_norm, global_grad_norm};
pub use lr_scheduler::{ConstantLr, LrScheduler, StepLr};
