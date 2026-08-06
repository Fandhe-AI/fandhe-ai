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
//! ### `.claude/rules/coding-rust.md` との回数記述の不一致（既知・要フォローアップ）
//!
//! `.claude/rules/coding-rust.md`（テスト・ベンチ節）は「ベンチは 5 回計測の中央値を採用し」と
//! 記載しているが、正本 `docs/spec/05-tasks.md` TASK-8.1 は warmup 20 回以上・計測 20 回以上を
//! 定めており、本クレートは正本（TASK-8.1）の 20/20 に従う（[`protocol::MeasurementConfig`] の
//! 下限 `MIN_ITERATIONS = 20` 参照）。仕様正本と実装リポ側規約ファイルの不一致であり、
//! `.claude/rules/out-of-scope-tracking.md` の規約上はユーザー承認を得たうえで Issue 化する
//! か `coding-rust.md` 側を訂正すべき事項である（Review 指摘。イシュー #27 時点では
//! ユーザー承認を経ていないため rule ファイル自体は変更せず、ここに不一致を明記するに留める）。
//!
//! ## TASK-8.1b: 同期方式統一・決定的シード（イシュー #28。main へ先行マージ済み）
//!
//! - [`rng`]: 決定的シードの自作 PRNG（xorshift64*）。ベンチ入力・回帰テストの
//!   再現性を「同一シード → 同一入力系列」で担保する
//! - [`sync`]: バックエンド間で統一する同期方式の契約と 3 バックエンド実装。
//!   「ホスト転送を伴わない完了待ち」への統一（REQ-8）を担う
//!
//! [`protocol`] モジュール（本イシュー #27）は計測コアとして独立しており、
//! [`sync`] の `SyncPoint` 実装をワークロードクロージャ内から呼び出す形で
//! 組み合わせる想定（[`sync`] モジュールドキュメント参照）。両モジュールの
//! 結合（計測区間終端での `wait_idle` 呼び出し）はバックエンド抽象層
//! （TASK-1.9）側の実装で行う。
//!
//! ## 未実装（後続イシューのスコープ）
//!
//! - TASK-8.1c（イシュー #29）: 構造化出力（JSON 等）・プロトコル遵守回帰テスト
//!
//! 上記スコープ外事項は本クレートの型設計（`serde` 非依存・同期はワークロードクロージャの
//! 責務とするコメント明記）で後から拡張しやすい形に留めている。

mod protocol;
pub mod rng;
mod stats;
pub mod sync;

pub use protocol::{Measurement, MeasurementConfig, run};
pub use stats::{BenchError, Quartiles, median_q1_q3};
