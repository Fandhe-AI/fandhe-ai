# speculative decoding・連続バッチングの facade 公開（#2934）実機未実測の申し送り

`docs/facade-speculative-decoding-batching-design.md` §18 参照。#2934 は `fandhe_ai::inference` への純再エクスポート
（`generate_speculative`・`SpeculativeConfig`・`BatchScheduler`・`RequestId`・`SchedulerLimits`）で、新規カーネルも
デバイス上の挙動の変更もない。このためデバイス上の確認は #2890 の既存 `#[ignore]` テストが担い、本 issue では
新しい `#[ignore]` テストを足していない。本実装エージェント実行環境は x86_64 Linux で CUDA／Metal に到達できず、
実機は未実測である。

## 測定コマンド

測定コマンド・期待結果・判定（REQ-2 統一複合判定。tolerance・baseline は新設しない）は
`docs/perf/logs/speculative-batching-2890/README.md` を参照する。

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture metal
```

facade パス経由の結合テスト `crates/facade/tests/speculative_batching_facade.rs` は CPU のみで通常 CI が実行する。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-09 | DGX Spark GB10 | `cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture cuda` | pass 2 / fail 0（`cuda_scheduler_greedy_matches_cpu` ok・`cuda_speculative_greedy_matches_cpu` ok。running 2 tests） | main `8bbeb874ceb4748cbcf01b52e7162d02812bf8ba`・`rustc 1.97.0 (2d8144b78 2026-07-07)`。`speculative-batching-2890` と同一の実行結果 |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
