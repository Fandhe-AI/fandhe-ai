//! optimizer 周辺の最小構成部品（親イシュー #192、第 3 分割・本イシュー #195）。
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
//! SGD momentum（#193）・AdamW（#194）本体は本イシューのスコープ外。
//! [`clip::clip_grad_norm`]／[`lr_scheduler::LrScheduler`] は
//! `Gradients`/`Var` に依存しない純関数・純データ構造として実装し、
//! optimizer 実装（#193/#194）からそのまま呼び出せる形にする
//! （疎結合設計。並行実装される兄弟イシューとの衝突を避ける狙い）。

pub mod clip;
pub mod lr_scheduler;

pub use clip::{ClipGradResult, clip_grad_norm, global_grad_norm};
pub use lr_scheduler::{ConstantLr, LrScheduler, StepLr};
