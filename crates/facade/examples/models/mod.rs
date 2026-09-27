//! 参照モデル定義（イシュー #2201・親 #2190）の集約モジュール。
//!
//! `crates/facade/examples/reference_models.rs`（runnable example）
//! だけがこのモジュールを `mod models;` で取り込む。`mlp`／`lenet` の
//! 各ファイルはそれぞれ単独で完結しており（`mlp.rs`／`lenet.rs`
//! モジュール doc「位置づけ」節参照）、統合テスト側は本ファイルを経由
//! せず `#[path]` で個別に直接取り込む。

pub mod lenet;
pub mod mlp;
