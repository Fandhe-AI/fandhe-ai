//! 自作 NN モジュール（TASK-9.1、REQ-9・M3、親イシュー #90）。
//!
//! `Tape`/`Var`（`tape.rs`/`var.rs`）に直接依存する自作コア側の部品群
//! （PyTorch `nn.Module` 相当）を置く。**互換 API 層
//! （`compat::array`/`compat::Sequential`。REQ-9・TASK-9.2）とは区別する**:
//! `nn` はこのクレートの一部として `Var`/`Tape` の内部契約
//! （クロステープ検査・shape 検査・`Tape` のステップ単位ライフサイクル）
//! を直接扱う実装であり、`compat` 層はその上に numpy/Keras 慣習の
//! 薄いラッパーを被せる別レイヤ（配置は未定。TASK-9.1a 計画 §3.1・§8）。
//! `lib.rs` クレート doc の「互換レイヤ固有のロジックを持ち込まない」は
//! `compat` 層本体の話であり、本モジュールには適用されない。
//!
//! TASK-9.1a（本イシュー・#91）で第 1 分割として `Linear`（全結合層）を
//! 実装する。活性化関数（#92）・損失関数（#189）・optimizer（#192）・
//! `compat::Sequential`（#94）は後続イシューのスコープ。

mod init;
mod linear;

pub use linear::{Linear, LinearVars};
