//! 参照モデルの PyTorch 対応表（イシュー #2201・親 #2190）の集約モジュール。
//!
//! `Mlp`／`LeNet` 本体は #2974 で `fandhe_ai::models` として公開済み。ここの `mlp`／`lenet` は
//! 公開面に含めない PyTorch 層対応表（`*ParamMap` と対応表関数）だけを持つ。
//! `crates/facade/examples/reference_models.rs`（runnable example）だけがこのモジュールを
//! `mod models;` で取り込み、統合テスト側は本ファイルを経由せず `#[path]` で個別に
//! 直接取り込む（各ファイルは単独で完結する）。
//!
//! `resnet.rs`／`transformer.rs`（イシュー #2202）は意図的に本モジュールへ登録していない。
//! 学習 script が `crates/facade/examples/main.rs`（`reference_models.rs` とは別の
//! runnable example）であり、`#[path]` による個別取り込み方式を採るため
//! （`docs/reference-models-decision.md` #2202 節「配置・取り込み方」参照）。

pub mod lenet;
pub mod mlp;
