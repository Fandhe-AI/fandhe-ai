//! ベンチマーク計測基盤。
//!
//! `backend-cpu` / `backend-cuda` / `backend-metal` の性能計測・回帰検出を担う。
//! 計測は 5 回計測の中央値を採用し、実機（DGX Spark GB10・Metal 実機）依存のベンチは
//! `#[ignore]` で分離する（`.claude/rules/coding-rust.md`）。ベンチ本体（`criterion`。
//! `dev-dependencies` 限定）は許容依存 8 区分の「ベンチ」区分に対応する
//! （`.claude/rules/deps-policy.md`）。カーネル側の手動境界検査省略の口実として
//! 性能計測結果を用いない（REQ-8）。
//!
//! - [`rng`]: 決定的シードの自作 PRNG（TASK-8.1b）。ベンチ入力・回帰テストの
//!   再現性を「同一シード → 同一入力系列」で担保する
//! - [`sync`]: バックエンド間で統一する同期方式の契約と 3 バックエンド実装
//!   （TASK-8.1b）。「ホスト転送を伴わない完了待ち」への統一（REQ-8）を担う
//!
//! 計測コア（TASK-8.1a）・構造化出力／回帰テスト（TASK-8.1c）は別イシューで
//! 追加する（spec 根拠: `docs/spec/05-tasks.md` TASK-8.1、REQ-8）。

pub mod rng;
pub mod sync;
