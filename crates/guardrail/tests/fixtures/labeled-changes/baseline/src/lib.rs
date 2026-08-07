//! `guardrail` 判定器の評価データセット（TASK-4.2a・イシュー #109）の判定対象
//! サンドボックス。v1 リポ（イシュー #269・PR #276）の PoC-3 検証コード
//! （Burn `=0.21.0` ベース）を、v2 自作コア（`tensor-core`・`autodiff`）上の
//! ミニ MLP 学習ワークロードとして再構築したもの。
//!
//! Burn は依存禁止リスト対象（`.claude/rules/deps-policy.md`）のため v1 の
//! `baseline/` をそのまま移植できず、`changes/*/change.patch` も本ワーク
//! ロード向けに再構築している（`README.md`「重要な注記」節参照）。
//!
//! **意図的に欠陥・ゲーミングを注入したテストデータ**であり、本番の Rust
//! コード（`crates/*`）へのマージ・流用は禁止する（`README.md`「baseline の
//! 隔離」節参照）。

pub mod activations;
pub mod compat;
pub mod model;
pub mod train;
