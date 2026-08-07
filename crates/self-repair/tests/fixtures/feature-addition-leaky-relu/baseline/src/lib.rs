//! 機能追加種別の自己修復ループ完走実証（TASK-3.3c・イシュー #142）の保守対象
//! サンドボックス。
//!
//! PoC-2 検証題材 (c)（`docs/spec/03-poc/poc-2-ai-self-maintenance/README.md:56-66`）
//! の「LeakyReLU を追加してほしい」という機能追加チケットを、v2 自作コア
//! （`tensor-core`・`autodiff`）上で再現する baseline（leaky_relu 未実装状態）
//! である。`crates/guardrail/tests/fixtures/labeled-changes/baseline` と同じ
//! 隔離方針（空 `[workspace]` テーブル）を踏襲するが、あちらは判定器
//! （guardrail）の評価データセットであるのに対し、本 crate はループ全体
//! （検出 → 修正試行 → 検証 → 取り込み）の完走実証の保守対象という別目的の
//! ため、`crates/self-repair/tests/fixtures/` 配下に独立して置く。
//!
//! **意図的に未完成の状態（leaky_relu 欠落）を保持したテストフィクスチャ**
//! であり、本番の Rust コード（`crates/*`）へのマージ・流用は禁止する。

pub mod activations;
