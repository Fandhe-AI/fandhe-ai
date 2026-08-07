//! optimizer 群（親イシュー #192「optimizer（SGD・AdamW）・gradient
//! clipping の実装」、REQ-9・TASK-9.1 系の後続）。
//!
//! `nn`（`crate::nn`）が `Tape`/`Var` に直接依存する層プリミティブを
//! 置くのに対し、本モジュールは「パラメータ `Tensor<f32>` の集合＋
//! 対応する勾配 `Tensor<f32>` の集合」から更新後パラメータを返す純粋な
//! 計算（`Tape`/`Var` に非依存）を置く。`Tape` はステップごとに生成・
//! 破棄される運用（`tape.rs`「学習ループでの運用」節）のため、
//! optimizer はテープのライフサイクルをまたいで存在できる状態
//! （momentum バッファ等）を自身で保持し、呼び出し元
//! （`nn::Linear::from_parameters` で更新後パラメータを差し替える側）は
//! 毎 step 同じ optimizer インスタンスへ `step()` を呼ぶ運用になる
//! （`nn_train_convergence.rs` のテストローカル `sgd_step` を optimizer
//! 本体へ格上げしたもの）。
//!
//! #193（本イシュー）で第 1 分割として [`Sgd`]/[`SgdConfig`]
//! （momentum・dampening・weight decay・nesterov 対応。PyTorch
//! `torch.optim.SGD` 準拠）を実装した。AdamW（#194）・gradient
//! clipping／LR スケジューラ（#195）は本モジュール配下への後続分割。
//! 共通 `Optimizer` trait の導入は AdamW 実装時に必要性を判断する
//! （過剰な抽象化を避ける。#193 計画 §8）。

mod sgd;

pub use sgd::{Sgd, SgdConfig};
