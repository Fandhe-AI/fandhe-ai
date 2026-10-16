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
| 2026-10-09 | Apple Silicon（M4 Max） | `cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture metal` | pass 2 / fail 0（`metal_scheduler_greedy_matches_cpu` ok・`metal_speculative_greedy_matches_cpu` ok。running 2 tests） | main `8bbeb874ceb4748cbcf01b52e7162d02812bf8ba`・Apple M4 Max・macOS 27.0（26A428）・`rustc 1.98.1 (48a229cea 2026-09-01)`。`speculative-batching-2890` と同一の実行結果 |

## #3098 追記

- 追加分（`inference/mod.rs` の成功例 doctest、`speculative_batching_facade.rs` の公開面だけの掃引、`api_surface.rs` の実在検査）は CPU のみで、デバイス挙動は変えていない。
- 新しい `#[ignore]` テストは足していない。実機（CUDA／Metal）の再計測は Phase 8 で扱う。実機 parity は上記の既存の測定コマンドで確認できる。
- 上記の 2026-10-09 の実測行は変更していない。

## #3107 追記

`BatchScheduler::step_batched`（テンソル単位バッチ。#3106）のバッチ経路の実機 parity を `speculative_batching_backend_parity.rs` に足した。実機は未実測で、Phase 8（#3171・#3172）へ申し送る。実測値は推測で書かない。性能保証はない。

追加した `#[ignore]` テスト:

- `cuda_scheduler_batched_greedy_matches_cpu`（CUDA。DGX Spark GB10）
- `metal_scheduler_batched_greedy_matches_cpu`（Metal。`cfg(target_os = "macos")` 限定）

属性なしの `cpu_backend_ops_matches_naive_ops_for_scheduler_batched_greedy`（CPU 対 Naive）は通常 CI が実行する。facade パス経由の CPU 結合テストは `crates/facade/tests/speculative_batching_step_batched_facade.rs`。

```sh
# CUDA（DGX Spark GB10 等の実機上で。バッチ経路だけに絞る）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture cuda_scheduler_batched

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture metal_scheduler_batched
```

絞り込みなしの `... -- --ignored --nocapture cuda`（metal も同様）は既存の 2 件に本テストを加えた 3 件を実行する。

期待結果と判定:

- 各 `forward_step` の logits は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance・baseline は新設しない
- 呼び出し数・b・行ごとの `valid_len`・ids・shape が device と CPU で一致する（片側だけ B = 1 へ戻った場合の検出を兼ねる）。layers = 2 では padding を伴う呼び出しが 1 件以上ある
- 仮説 H1（バックエンドをまたぐ Greedy の token 列一致）と、同じバックエンドの単独 `generate` との token 列一致が成り立つ。破れても判定は緩めず、設計記録 §10 論点 2 として承認依頼に戻す

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | `... -- --ignored --nocapture cuda_scheduler_batched` | 未実測 | |
| 未実測 | Apple Silicon | `... -- --ignored --nocapture metal_scheduler_batched` | 未実測 | |

上記の 2026-10-09 の実測行は変更していない。
