//! 自作 NN モジュール（TASK-9.1、REQ-9・M3、親イシュー #90）。
//!
//! `Tape`/`Var`（`tape.rs`/`var.rs`）に直接依存する自作コア側の部品群
//! （PyTorch `nn.Module` 相当）を置く。**互換 API 層
//! （`fandhe_ai::compat::array`/`fandhe_ai::compat::Sequential`。REQ-9・
//! TASK-9.2・TASK-9.4）とは区別する**: `nn` はこのクレートの一部として
//! `Var`/`Tape` の内部契約（クロステープ検査・shape 検査・`Tape` の
//! ステップ単位ライフサイクル）を直接扱う実装であり、compat 層
//! （TASK-9.4・#411 で `fandhe_ai::compat` へ移設済み。numpy/Keras 慣習の
//! 薄いラッパー）とは区別される。compat 層は本モジュールの [`Module`]
//! trait・`Linear`・`activation` を呼ぶだけで、`Var`/`Tape` の内部契約
//! には触れない（`lib.rs` クレート doc の「互換レイヤ固有のロジックを
//! 持ち込まない」は compat 層本体の話であり、本モジュールには適用
//! されない）。
//!
//! TASK-9.1a（#91）で第 1 分割として `Linear`（全結合層）を実装した。
//! TASK-9.1b（#92）で活性化関数（[`activation`]）を追加した。#190
//! （親 #189）で MSE 損失（[`loss`]）を追加し、#191 で CrossEntropy
//! 損失（同じく [`loss`]）を追加した。#1737（親 #1609）で二値交差
//! エントロピー損失（`loss::BceLoss`／`loss::BceWithLogitsLoss`）を
//! 追加した。#194（親 #192）で optimizer の
//! 第 1 弾として AdamW（[`optim::AdamW`]）を追加した。#195（親 #192）で
//! gradient clipping・LR スケジューラ最小セット（[`optim::clip`]・
//! [`optim::lr_scheduler`]）を追加した。SGD 本体は `crate::optim`
//! （本モジュールとは別モジュール。#193）で実装済み。TASK-9.2a（#95）
//! で共通 [`Module`] trait（`module.rs`）を確定し、`Sequential`
//! （当時 `crate::compat::Sequential`。TASK-9.4・#411 で
//! `fandhe_ai::compat::Sequential` へ移設）がこれを介して `Linear`・
//! 活性化関数を均一に扱えるようにした。共通 `Optimizer` trait の定義は
//! 本イシューでは
//! 行わない（`optim` 配下が揃った時点で確定する）。イシュー #1596 で
//! [`RmsNorm`]／[`LayerNorm`]（`norm` モジュール）を追加し、既存の RMSNorm 行
//! カーネル（`backend-cpu`／`backend-cuda`／`backend-metal` の
//! `rmsnorm.rs`）を `BackendOps::rmsnorm` 経由で・LayerNorm を新設
//! カーネル経由でそれぞれ接続した（`docs/norm-ops-design.md`）。
//! イシュー #1604 で Embedding 層（`embedding` モジュール。
//! `Var::embedding` を薄くラップする `nn::Embedding`／`EmbeddingVars`）を
//! 追加した。`Module` trait は実装しない（`embedding.rs` モジュール
//! doc 参照。id 入力の型が `Module::forward` の f32 `Var` 契約と
//! 一致しないうえ、`compat::Sequential` の学習可能パラメータ収集が
//! `as_linear` フック限定のため）。イシュー #1640 で `MultiheadAttention`
//! Module（`attention` モジュール）を追加した。新規 `Op`／`BackendOps`
//! メソッドは追加せず、`nn::Linear` 4 層（q/k/v/out projection）と
//! 既存 `Var` 演算（`matmul`／`reshape`／`permute`／`masked_fill`／
//! `softmax`）の合成として実装している。`Module` trait は self-attention
//! （`q=k=v=input`）として実装する（`attention.rs` モジュール doc
//! 参照）。イシュー #1738（親 #1609）で `NllLoss`（`Var::nll_loss` の
//! ラッパー。PyTorch `nn.NLLLoss` 相当）・`KlDivLoss`（`Var::kl_div_loss`／
//! `kl_div_loss_with_log_target` のラッパー。PyTorch `nn.KLDivLoss`
//! 相当。同じく [`loss`]）を追加した。
//! 参照）。イシュー #1739 で Huber／SmoothL1 損失（[`loss::HuberLoss`]／
//! [`loss::SmoothL1Loss`]）を追加した。`MseLoss`／`CrossEntropyLoss` と
//! 同じ「`BackendOps` の専用融合カーネルを優先し `Unsupported` のとき
//! のみホスト参照実装へフォールバックする」設計を踏襲する
//! （`loss.rs` モジュール doc 参照）。イシュー #1758 で [`Module`] trait
//! に `set_training`／`training`（PyTorch `Module.training` 相当）・
//! `named_parameters`（PyTorch `Module.named_parameters()` 相当）の
//! defaulted メソッドを追加した。本クレート内実装（`Linear`・活性化
//! 関数群・`RmsNorm`／`LayerNorm`・`Softmax`／`LogSoftmax`・
//! `MultiheadAttention`・`Rnn`／`Lstm`／`Gru`）はいずれもモード非依存
//! のため `set_training`／`training` は既定（no-op／常に `true`）の
//! まま——モードの正はコンテナ（`fandhe_ai_facade::compat::sequential::
//! Sequential`）が保持するフラグとする契約（`module.rs` の trait doc
//! 参照）。`named_parameters` は `Linear`／`RmsNorm`／`LayerNorm`／
//! `MultiheadAttention`／`Rnn`／`Lstm`／`Gru` でオーバーライドし、
//! struct フィールド名／accessor 名をそのまま使う命名契約（PyTorch の
//! packed 命名は追わない）とした。イシュー #1759（親 #1617）で
//! [`Module`] trait doc が「将来の `ModuleList`／汎用 `Sequential`」
//! として予告していたコンテナを `container` モジュールへ実装した
//! （`ModuleList`・`Sequential`）。`fandhe_ai_facade::compat::sequential::
//! Sequential`（Linear／活性化関数の閉集合限定ビルダー）は本モジュール
//! の `Sequential` を `inner` として合成する薄いラッパーへ再構成した
//! （`container.rs` モジュール doc「配置」節参照）。`ModuleList`／
//! `Sequential` は facade から再エクスポートしない（`Module` trait
//! 自体が非公開のため）。イシュー #1603 で [`Dropout`]（`dropout`
//! モジュール）を追加した。`Var::dropout` の薄いラッパーであると
//! 同時に、`set_training`／`training` を実際にオーバーライドする
//! **最初の実装**（`dropout.rs` モジュール doc・`module.rs` の trait
//! doc「今後 Dropout・BatchNorm 等のモード依存層を追加する際は…」
//! 参照）。イシュー #1770（親 #1645）で [`Conv2d`]／[`Conv1d`]（`conv`
//! モジュール）を追加した。`Var::conv2d`／`Var::conv1d`（#1764・#1765）
//! を薄くラップするのみで新規 `Op`／`BackendOps`／VJP は追加しない。
//! `Module` trait へ `as_conv2d`／`as_conv1d`（各 `_mut` 版込み）フックを
//! 追加し、`fandhe_ai_facade::compat::sequential::Sequential` の学習
//! 経路（`bind`／`trainable_parameters`／`apply_parameters` 等）へ
//! 接続した（`docs/compat-api-scope.md` §5 手続き・親 #1645 コメントで
//! ユーザー承認済み）。イシュー #2066（親 #2058）で [`GroupNorm`]／
//! [`InstanceNorm`]（`normalization` モジュール）を追加した。既存の
//! 最終軸限定 [`LayerNorm`]（`norm` モジュール）を「軸削減 reshape →
//! `layer_norm`（affine なし）→ 逆 reshape」で呼ぶ合成のみで新規
//! `Op`／`BackendOps`／VJP は追加しない（`normalization.rs` モジュール
//! doc「軸削減公式」参照）。PyTorch 既定と異なり **affine（学習可能な
//! per-channel `weight`／`bias`）を持たない**（`normalization.rs`
//! モジュール doc「affine 非対応」節参照。`.claude/rules/
//! coding-rust.md` の勾配長軸縮約契約との抵触を避けるため）。`Module`
//! trait への統合（`as_group_norm`／`as_instance_norm`）はあるが、
//! `docs/compat-api-scope.md` §5 の facade 公開面拡張承認が未取得の
//! ため `compat::Sequential::add_group_norm`／`add_instance_norm` は
//! 追加していない。イシュー #2065（親 #2059）で [`Flatten`]（`flatten`
//! モジュール）を追加した。`Var::flatten`（#1597）を薄くラップする
//! のみで新規 `Op`／`BackendOps`／VJP は追加しない。同イシューで
//! `fandhe_ai_facade::compat::sequential::Sequential` に
//! `add_softmax`／`add_log_softmax`／`add_gelu`／`add_gelu_tanh`／
//! `add_softplus`／`add_flatten` の 6 `pub fn` を追加した（`Softmax`／
//! `LogSoftmax`／`Gelu`／`GeluTanh`／`Softplus` 自体は `nn::activation`
//! に既存実装済み。`docs/compat-api-scope.md` §5「適用記録（経路2。
//! イシュー #2065）」参照）。イシュー #2068（親 #2059）で
//! [`TransformerEncoderLayer`]（`transformer_encoder_layer` モジュール）
//! を追加した。既存の [`MultiheadAttention`]・[`LayerNorm`]・
//! `nn::Linear` 2 層の合成（post-norm 固定・新規 `Op`／`BackendOps`／
//! VJP なし）として PyTorch `nn.TransformerEncoderLayer` 相当の 1 層を
//! 実装する（`transformer_encoder_layer.rs` モジュール doc 参照）。
//! `fandhe_ai_facade::compat::sequential::Sequential` に
//! `add_transformer_encoder` を追加した（`docs/compat-api-scope.md`
//! §5「適用記録（経路2。イシュー #2068）」参照）。イシュー #2134
//! （親 #2131）で [`Module`] trait に `children`／`named_modules`／
//! `parameter_count`／`type_name`（PyTorch `Module.children()`／
//! `named_modules()`／`sum(p.numel() for p in model.parameters())`
//! 相当）の defaulted メソッド 4 件を追加し、`ModuleList`・
//! `Sequential`・`MultiheadAttention`・`TransformerEncoderLayer` に
//! `children` をオーバーライドした。新規コンテナ [`container::
//! ModuleDict`]（PyTorch `nn.ModuleDict` 相当）・自由関数
//! [`container::summary`]（`print(model)` 相当の簡易表示）を
//! `container` モジュールへ追加した。いずれも数値経路（`Op`／
//! `BackendOps`／VJP）を追加しない CPU ホスト側の introspection
//! 機構であり、facade へは再エクスポートしない（`container.rs`
//! モジュール doc「facade への非公開」節参照）。
//! イシュー #2084（親 #2059）で KV キャッシュ付き attention
//! （[`attention::KvCache`]・`MultiheadAttentionVars::
//! forward_with_cache`・[`attention::StatefulAttention`]）を
//! 追加した。既存 `Var` 演算（`cat`／`var_no_grad`）と
//! `nn/attention.rs` の既存合成（`project`／`split_heads`／
//! `sdpa_compose`）のみで実装し、新規 `Op`／`BackendOps`／
//! カーネル／依存は追加しない（`docs/kv-cache-design.md`）。
//! K-1（本 autodiff 内部実装）は 2026-09-24 にユーザー承認済み
//! （`docs/kv-cache-design.md` §6 承認事項 1）。facade 公開
//! （`add_stateful_attention` 相当・K-2）は別途承認が必要で未承認
//! のため保留する（`crates/facade/tests/api_surface.rs` の否定
//! ガードで固定。`docs/kv-cache-design.md` §6 承認事項 2）。
//! イシュー #2140（親 #2131）で [`init`] を `pub mod` 化し、PyTorch
//! `torch.nn.init.*` 相当の初期化関数群（`uniform`／`normal`／
//! `constant`／`xavier_uniform`／`xavier_normal`／`kaiming_uniform`／
//! `kaiming_normal`／`orthogonal`／`trunc_normal`）を追加した。個別
//! シード方式の既存 `pub(crate)` ヘルパー（`uniform_init` 等）とは独立
//! に、プロセスグローバル決定的 RNG（`tensor-core::rng::manual_seed`）
//! へ従属する（`init.rs` モジュール doc「`nn::init`」節参照）。facade
//! への再エクスポートは別途ユーザー承認（`docs/compat-api-scope.md`
//! §5 経路 2）を要する公開面拡張のため、本イシュー時点では未承認の
//! まま保留し `crates/facade/**` は変更していない
//! （`docs/facade-nn-init-exposure-decision.md` 参照）。

mod attention;
mod batch_norm;
mod container;
mod conv;
mod dropout;
mod embedding;
mod flatten;
pub mod init;
mod linear;
mod module;
mod norm;
mod normalization;
mod pooling;
mod rnn;
mod transformer_encoder_layer;

pub mod activation;
pub mod loss;
pub mod optim;

pub use attention::{
    KvCache, MultiheadAttention, MultiheadAttentionVars, StatefulAttention,
    multihead_attention_forward_low_precision,
};
pub use batch_norm::{
    BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM, BatchNorm1d, BatchNorm2d, BatchNormVars,
};
pub use container::{ModuleDict, ModuleList, Sequential, summary};
pub use conv::{
    Conv1d, Conv1dVars, Conv2d, Conv2dVars, ConvTranspose2d, ConvTranspose2dVars,
    conv2d_forward_low_precision,
};
pub use dropout::Dropout;
pub use embedding::{Embedding, EmbeddingVars};
pub use flatten::Flatten;
pub use linear::{Linear, LinearVars, linear_forward_low_precision};
pub use module::Module;
pub use norm::{
    LAYER_NORM_DEFAULT_EPS, LayerNorm, LayerNormVars, RMS_NORM_DEFAULT_EPS, RmsNorm, RmsNormVars,
};
pub use normalization::{
    GROUP_NORM_DEFAULT_EPS, GroupNorm, INSTANCE_NORM_DEFAULT_EPS, InstanceNorm,
};
pub use pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d,
};
pub use rnn::{
    Gru, GruCell, GruCellVars, Lstm, LstmCell, LstmCellVars, LstmSeqOutput, Rnn, RnnCell,
    RnnCellVars, RnnSeqOutput,
};
pub use transformer_encoder_layer::{
    FeedForwardActivation, TransformerEncoderLayer, TransformerEncoderLayerVars,
};
