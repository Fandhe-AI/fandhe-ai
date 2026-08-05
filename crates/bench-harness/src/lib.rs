//! ベンチマーク計測基盤。
//!
//! `backend-cpu` / `backend-cuda` / `backend-metal` の性能計測・回帰検出を担う。
//! 計測は 5 回計測の中央値を採用し、実機（DGX Spark GB10・Metal 実機）依存のベンチは
//! `#[ignore]` で分離する（`.claude/rules/coding-rust.md`）。ベンチ本体（`criterion`。
//! `dev-dependencies` 限定）は許容依存 8 区分の「ベンチ」区分に対応する
//! （`.claude/rules/deps-policy.md`）。カーネル側の手動境界検査省略の口実として
//! 性能計測結果を用いない（REQ-8）。
//!
//! 雛形段階（TASK-1.1 部分実装。許容依存の `Cargo.toml` 反映を除く。反映はユーザー承認を
//! 要するため別イシューで対応する）では型・実装を持たない（spec 根拠: `docs/spec/05-tasks.md`
//! TASK-1.1、REQ-8）。
