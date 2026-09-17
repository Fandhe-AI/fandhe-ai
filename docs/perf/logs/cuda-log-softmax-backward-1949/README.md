# CUDA `log_softmax` backward カーネル（イシュー #1949）実機実測 申し送り

本エージェント実行環境に DGX Spark GB10 実機への到達手段がないため、
以下の `#[ignore]` テストは未実施のまま記入欄を残す。

## 実行コマンド

```sh
# crate 直接（`CudaLogSoftmaxBackward` 直接 parity・形状網羅・空 shape・
# run-to-run bit 同一性）
cargo test -p fandhe-ai-backend-cuda --release --test log_softmax_backward_parity -- --ignored --nocapture

# facade（forward 非後退確認・matmul→log_softmax→mse_loss backward の
# CUDA vs CPU 突合）
cargo test -p fandhe-ai --release --test softmax_backend_parity -- --ignored --nocapture
```

## 事前登録判定規則

1. `#[ignore]` 対象テスト全 pass（REQ-2 統一複合判定・
   `fandhe_ai_backend_cpu::parity::assert_parity`。tolerance 定数は
   不変）。
2. run-to-run bit 同一（`log_softmax_backward_is_run_to_run_bit_identical`）。
3. 既存 `make test-ignored-cuda`（本イシューが追加した対象を含む）の
   非後退。既知 FAIL 一覧は
   `docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.2 を
   参照する。FAIL は是正せず記録のみ・事後緩和はしない。

性能採否判定（本番結線の性能面での妥当性）は本イシューのスコープ外
（機能結線が目的。カーネルは既に本番 `BackendOps::log_softmax_backward`
経由で結線済みであり、本記入欄は実機での正しさ検証のみを対象とする）。

## 記入欄

（GB10 実機実測完了後に追記する。生ログ・env_info は本ディレクトリへ
配置し、内部ホスト名は含めない。）
