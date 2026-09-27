//! 参照モデル定義（イシュー #2201・親 #2190）の集約モジュール。
//!
//! `crates/facade/examples/reference_models.rs`（runnable example）
//! だけがこのモジュールを `mod models;` で取り込む。`mlp`／`lenet` の
//! 各ファイルはそれぞれ単独で完結しており（`mlp.rs`／`lenet.rs`
//! モジュール doc「位置づけ」節参照）、統合テスト側は本ファイルを経由
//! せず `#[path]` で個別に直接取り込む。
//!
//! `resnet.rs`／`transformer.rs`（イシュー #2202）は意図的に本モジュール
//! へ登録していない。学習 script が `crates/facade/examples/main.rs`
//! （`reference_models.rs` とは別の runnable example）であり、
//! `#[path]` による個別取り込み方式を採るため
//! （`docs/reference-models-decision.md` #2202 節「配置・取り込み方」
//! 参照）。

pub mod lenet;
pub mod mlp;
