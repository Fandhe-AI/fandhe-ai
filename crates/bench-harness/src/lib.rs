//! ベンチマーク計測基盤。
//!
//! `backend-cpu` / `backend-cuda` / `backend-metal` の性能計測・回帰検出を担う。
//! 実機（DGX Spark GB10・Metal 実機）依存のベンチは `#[ignore]` で分離する
//! （`.claude/rules/coding-rust.md`）。ベンチ本体（`criterion`。`dev-dependencies` 限定）は
//! 許容依存 8 区分の「ベンチ」区分に対応する（`.claude/rules/deps-policy.md`）。
//! カーネル側の手動境界検査省略の口実として性能計測結果を用いない（REQ-8）。
//!
//! ## TASK-8.1a: 計測プロトコル（本イシュー #27 の実装範囲）
//!
//! [`protocol`] モジュールが warmup 20 回以上・計測 20 回以上・中央値採用・Q1/Q3 記録という
//! `docs/spec/05-tasks.md` TASK-8.1 の計測プロトコルを実装する。分位点の定義は PoC-v2-1 参照実装
//! （`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/rust/src/bin/gemm_bench.rs:17-25`）を踏襲し、
//! [`stats`] モジュールが純粋関数として提供する。ワークロードはクロージャで受け取る
//! バックエンド非依存設計とし、バックエンド抽象層（TASK-1.9）の完成を待たずに実装している。
//!
//! ## 未実装（後続イシューのスコープ）
//!
//! - TASK-8.1b（イシュー #28）: バックエンド別同期統一（CUDA `stream.synchronize()` /
//!   Metal コマンドバッファ完了待ち）・決定的シード（xorshift64*）
//! - TASK-8.1c（イシュー #29）: 構造化出力（JSON 等）・プロトコル遵守回帰テスト
//!
//! 上記スコープ外事項は本クレートの型設計（`serde` 非依存・同期はワークロードクロージャの
//! 責務とするコメント明記）で後から拡張しやすい形に留めている。

mod protocol;
mod stats;

pub use protocol::{Measurement, MeasurementConfig, run};
pub use stats::{BenchError, Quartiles, median_q1_q3};
