//! 自作 NN モジュール（TASK-9.1、REQ-9・M3、親イシュー #90）。
//!
//! `Tape`/`Var`（`tape.rs`/`var.rs`）に直接依存する自作コア側の部品群
//! （PyTorch `nn.Module` 相当）を置く。**互換 API 層
//! （`facade::compat::array`/`facade::compat::Sequential`。REQ-9・
//! TASK-9.2・TASK-9.4）とは区別する**: `nn` はこのクレートの一部として
//! `Var`/`Tape` の内部契約（クロステープ検査・shape 検査・`Tape` の
//! ステップ単位ライフサイクル）を直接扱う実装であり、compat 層
//! （TASK-9.4・#411 で `facade::compat` へ移設済み。numpy/Keras 慣習の
//! 薄いラッパー）とは区別される。compat 層は本モジュールの [`Module`]
//! trait・`Linear`・`activation` を呼ぶだけで、`Var`/`Tape` の内部契約
//! には触れない（`lib.rs` クレート doc の「互換レイヤ固有のロジックを
//! 持ち込まない」は compat 層本体の話であり、本モジュールには適用
//! されない）。
//!
//! TASK-9.1a（#91）で第 1 分割として `Linear`（全結合層）を実装した。
//! TASK-9.1b（#92）で活性化関数（[`activation`]）を追加した。#190
//! （親 #189）で MSE 損失（[`loss`]）を追加し、#191 で CrossEntropy
//! 損失（同じく [`loss`]）を追加した。#194（親 #192）で optimizer の
//! 第 1 弾として AdamW（[`optim::AdamW`]）を追加した。#195（親 #192）で
//! gradient clipping・LR スケジューラ最小セット（[`optim::clip`]・
//! [`optim::lr_scheduler`]）を追加した。SGD 本体は `crate::optim`
//! （本モジュールとは別モジュール。#193）で実装済み。TASK-9.2a（#95）
//! で共通 [`Module`] trait（`module.rs`）を確定し、`Sequential`
//! （当時 `crate::compat::Sequential`。TASK-9.4・#411 で
//! `facade::compat::Sequential` へ移設）がこれを介して `Linear`・
//! 活性化関数を均一に扱えるようにした。共通 `Optimizer` trait の定義は
//! 本イシューでは
//! 行わない（`optim` 配下が揃った時点で確定する）。

mod init;
mod linear;
mod module;

pub mod activation;
pub mod loss;
pub mod optim;

pub use linear::{Linear, LinearVars};
pub use module::Module;
