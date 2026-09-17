//! RNN／LSTM／GRU の Sequence レベル API（`fandhe_ai_autodiff::nn`
//! の `Rnn`／`Lstm`／`Gru`）を facade へ再エクスポートする**純再
//! エクスポートモジュール**（`crate::data`・`crate::optim` と同型。
//! facade 独自の型・関数は持ち込まない）。
//!
//! イシュー #1955（承認: 2026-09-17 ユーザー承認コメント「選択肢
//! C」）。承認スコープは「`compat::Sequential` への `add_rnn`／
//! `add_lstm`／`add_gru` は追加しない・`fandhe_ai::nn::rnn` として
//! 素の再エクスポートで公開する」（`docs/compat-api-scope.md` §5
//! 経路 2）。
//!
//! # `Sequential::add_*` を設けない理由
//!
//! `compat::Sequential`（`crate::compat::sequential`）は
//! `Var → Var` の平坦鎖（1 テンソル入力・1 テンソル出力を各層が
//! 順に受け渡す構成）を前提とする。RNN 系の学習経路
//! （`Rnn::forward_seq`／`Lstm::forward_seq`／`Gru::forward_seq`）は
//! これと構造的に不整合である:
//!
//! - 入力は `Var` ではなく `&Tensor<f32>`（`[T,B,D]`。時系列方向へ
//!   `Tensor` レベルでスライスするため。`fandhe_ai_autodiff::nn::rnn`
//!   モジュール doc「`Module` trait との関係」参照）
//! - 隠れ状態（LSTM は `c` も）を明示的に引き回す必要がある
//! - 戻り値が複数の `Var`（`outputs`・`h_n`・場合により `c_n`・
//!   学習済みパラメータ `params`）を持つ構造体である
//!
//! よって `Rnn`／`Lstm`／`Gru` は `compat::Sequential` の層としては
//! 扱わず、独立した `fandhe_ai::nn::rnn` モジュールとして公開する
//! （`docs/autodiff-rnn-cell-tape-design.md` 決定 10 参照）。
//!
//! # `forward_seq` の呼び出し方（`Tape` 委譲メソッド）
//!
//! `Rnn::forward_seq` 等は第 1 引数に生の
//! `fandhe_ai_autodiff::Tape` を取るが、facade の [`crate::Tape`] は
//! newtype（内部フィールドは `pub(crate)`）であり利用者はこれを
//! 取り出せない（REQ-12・`crate::lib.rs` モジュール doc「公開面の
//! 設計」）。このため本モジュール自体は `forward_seq` を橋渡しする
//! 関数を持たず、[`crate::Tape::rnn_forward_seq`]／
//! [`crate::Tape::lstm_forward_seq`]／[`crate::Tape::gru_forward_seq`]
//! （`&self.0` を渡すだけの薄い委譲。`crate::Tape::
//! step_device_param_store` 等と同型）を入口として使う。
//!
//! # 再エクスポート対象を 8 型に限定する理由
//!
//! `forward_seq` の呼び出し・戻り値の利用に必要な型（本体 3 型・
//! 戻り値型 2 型・`params` フィールド型 3 型）のみを再エクスポート
//! する。以下は意図して対象外とする:
//!
//! - `RnnCell`／`LstmCell`／`GruCell`: `forward_seq` の入出力には
//!   現れない（`Rnn::cell()` の戻り値型を facade から名指しできない
//!   ため、`from_cell`／`from_parameters` による外部重み持ち込みは
//!   facade からは行えない）
//! - `Module` trait: `Rnn`／`Lstm`／`Gru` の `named_parameters`／
//!   `state_dict`／`forward_host`（`&dyn BackendOps` 引数を取る）は
//!   facade へ `BackendOps` を露出させずには到達できない（REQ-12）
//!
//! # 利用例
//!
//! ```
//! use fandhe_ai::nn::rnn::Rnn;
//! use fandhe_ai::Tensor;
//!
//! let tape = fandhe_ai::tape();
//! let rnn = Rnn::new(/* input_size */ 3, /* hidden_size */ 4, /* bias */ true, /* seed */ 0)
//!     .unwrap();
//! // x: [T=2, B=1, D=3]
//! let x = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 1, 3]).unwrap();
//! let out = tape.rnn_forward_seq(&rnn, &x, None).unwrap();
//!
//! let loss = out.outputs.last().unwrap().sum(None).unwrap();
//! let grads = tape.backward(&loss).unwrap();
//! // 学習に使ったパラメータの勾配は `out.params` 経由で取得する。
//! assert!(grads.get(&out.params.weight_ih).unwrap().is_some());
//! ```

pub use fandhe_ai_autodiff::nn::{Gru, GruCellVars, Lstm, LstmCellVars, LstmSeqOutput};
pub use fandhe_ai_autodiff::nn::{Rnn, RnnCellVars, RnnSeqOutput};
