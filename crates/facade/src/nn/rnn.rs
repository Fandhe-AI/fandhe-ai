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
//! # 再エクスポート対象を 14 型に限定する理由
//!
//! `forward_seq` の呼び出し・戻り値の利用に必要な型（単層の本体 3 型・
//! 戻り値型 2 型・`params` フィールド型 3 型）に加え、イシュー #2535
//! （親 #2534・ルート #2499 の一括承認。`docs/autodiff-rnn-stacked-
//! config-decision.md` §8・§9）で多層・双方向・層間 dropout 版の
//! `RnnConfig`（構築オプション）・`StackedRnn`／`StackedLstm`／
//! `StackedGru`（本体 3 型）・`StackedRnnSeqOutput`／
//! `StackedLstmSeqOutput`（戻り値型 2 型）の計 6 型を追加した。
//! 多層版の入口は [`crate::Tape::stacked_rnn_forward_seq`]／
//! [`crate::Tape::stacked_lstm_forward_seq`]／
//! [`crate::Tape::stacked_gru_forward_seq`]。以下は意図して対象外とする:
//!
//! - `RnnCell`／`LstmCell`／`GruCell`: `forward_seq` の入出力には
//!   現れない（`Rnn::cell()` の戻り値型を facade から名指しできない
//!   ため、`from_cell`／`from_parameters` による外部重み持ち込みは
//!   facade からは行えない）
//! - `Module` trait: `Rnn`／`Lstm`／`Gru` の `named_parameters`／
//!   `state_dict`／`forward_host`（`&dyn BackendOps` 引数を取る）は
//!   facade へ `BackendOps` を露出させずには到達できない（REQ-12）
//! - `Rnn`／`Lstm`／`Gru::with_config`: `docs/autodiff-rnn-stacked-
//!   config-decision.md` §2.1 で不採用（多層は `Stacked*` を使う）
//!
//! # 既知の制限（eval モード）
//!
//! `Stacked*::new` は training=true で構築され、`set_training` は
//! autodiff の `Module` trait にしかないため facade からは到達できない。
//! `dropout > 0` の場合 facade 経由の forward は常に学習モードとなる。
//! 推論用途では `dropout = 0.0` で構築する。
//!
//! # 可変長系列（packed sequence。イシュー #2679・親 #2625）
//!
//! `torch.nn.utils.rnn` 相当の [`crate::nn::rnn::PackedSequence`]・出力型 4 種
//! （[`crate::nn::rnn::PackedRnnSeqOutput`]・[`crate::nn::rnn::PackedLstmSeqOutput`]・
//! [`crate::nn::rnn::StackedPackedRnnSeqOutput`]・[`crate::nn::rnn::StackedPackedLstmSeqOutput`]）と
//! 自由関数 8 本（`pack_padded_sequence`・`pad_packed_sequence`・`rnn_forward_packed`・
//! `gru_forward_packed`・`lstm_forward_packed`・`stacked_rnn_forward_packed`・
//! `stacked_gru_forward_packed`・`stacked_lstm_forward_packed`）を、承認（ルート #2499 の
//! コメント。`docs/autodiff-packed-sequence-decision.md` §7）に従い純再エクスポートする。
//! 自由関数は生の `Tape` を取らず（tape は入力 `Var` から得る）、`Tape` 委譲メソッドは不要。
//! `Var`／`Tape` への委譲メソッド・`Rnn::forward_packed`・`Sequential::add_*` は追加しない
//! （承認形にないため。`PackedSequenceHoldDoctestGuard` が未承認経路を固定する）。
//! `PackedSequence::new` は型の一部（検証付きコンストラクタ）として到達可能になる。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::nn::rnn::{Rnn, pack_padded_sequence, pad_packed_sequence, rnn_forward_packed};
//!
//! let tape = fandhe_ai::tape();
//! let rnn = Rnn::new(2, 3, true, 0).unwrap();
//! // padded 入力 [T=3, B=2, D=2]。系列長は [3, 2]（非増加）。
//! let x = Tensor::new((0..12).map(|v| v as f32 * 0.1).collect(), &[3, 2, 2]).unwrap();
//! let packed = pack_padded_sequence(&tape.var(&x), &[3, 2], false, true).unwrap();
//! let out = rnn_forward_packed(&rnn, &packed, None).unwrap();
//! // 有効な時刻のみ詰められた出力（3 + 2 = 5 行）を padded へ戻す。
//! let (padded, lengths) = pad_packed_sequence(&out.output, false, 0.0, None).unwrap();
//! assert_eq!(lengths, vec![3, 2]);
//! assert_eq!(padded.to_tensor().shape(), &[3usize, 2, 3]);
//! ```
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
//!
//! 2 層・双方向の `StackedLstm`（イシュー #2535）:
//!
//! ```
//! use fandhe_ai::nn::rnn::{RnnConfig, StackedLstm};
//! use fandhe_ai::Tensor;
//!
//! let tape = fandhe_ai::tape();
//! let cfg = RnnConfig::new().with_num_layers(2).with_bidirectional(true);
//! let lstm = StackedLstm::new(3, 4, true, 0, cfg).unwrap();
//! // x: [T=2, B=1, D=3]
//! let x = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 1, 3]).unwrap();
//! let out = tape.stacked_lstm_forward_seq(&lstm, &x, None, None).unwrap();
//! // h_n は num_layers * num_directions = 4 本。出力は [B, 2H]。
//! assert_eq!(out.h_n.len(), 4);
//! assert_eq!(out.outputs[0].to_tensor().shape(), &[1usize, 8]);
//!
//! let loss = out.outputs.last().unwrap().sum(None).unwrap();
//! let grads = tape.backward(&loss).unwrap();
//! assert!(grads.get(&out.params[0].weight_ih).unwrap().is_some());
//! ```

pub use fandhe_ai_autodiff::nn::packed_sequence::{
    PackedLstmSeqOutput, PackedRnnSeqOutput, PackedSequence,
};
pub use fandhe_ai_autodiff::nn::packed_sequence::{
    StackedPackedLstmSeqOutput, StackedPackedRnnSeqOutput,
};
pub use fandhe_ai_autodiff::nn::packed_sequence::{
    gru_forward_packed, lstm_forward_packed, rnn_forward_packed,
};
pub use fandhe_ai_autodiff::nn::packed_sequence::{pack_padded_sequence, pad_packed_sequence};
pub use fandhe_ai_autodiff::nn::packed_sequence::{
    stacked_gru_forward_packed, stacked_lstm_forward_packed, stacked_rnn_forward_packed,
};
pub use fandhe_ai_autodiff::nn::{Gru, GruCellVars, Lstm, LstmCellVars, LstmSeqOutput};
pub use fandhe_ai_autodiff::nn::{Rnn, RnnCellVars, RnnSeqOutput};
pub use fandhe_ai_autodiff::nn::{RnnConfig, StackedGru, StackedLstm, StackedRnn};
pub use fandhe_ai_autodiff::nn::{StackedLstmSeqOutput, StackedRnnSeqOutput};
