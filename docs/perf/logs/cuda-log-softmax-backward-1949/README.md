# CUDA `log_softmax` backward カーネル（イシュー #1949）実機実測 申し送り

> **2026-09-18 DGX Spark GB10 で実測済み**（4/4・3/3 pass。下記「記入欄」参照）。以下の申し送り文は PR 時点の記録としてそのまま残す。

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
   不変）。`log_softmax_backward_cancelling_upstream_grad_matches_cpu`
   （PR #1994 codex-review 指摘の相殺反例 `logits=[0,0,0,0]`・
   `g=[1e20,-1e20,1,0]` の実機回帰。`Σ_dim(g)` を lane 0 逐次和へ是正
   済み）を含む。
2. run-to-run bit 同一（`log_softmax_backward_is_run_to_run_bit_identical`）。
3. 既存 `make test-ignored-cuda`（本イシューが追加した対象を含む）の
   非後退。既知 FAIL 一覧は
   `docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.2 を
   参照する。FAIL は是正せず記録のみ・事後緩和はしない。

性能採否判定（本番結線の性能面での妥当性）は本イシューのスコープ外
（機能結線が目的。カーネルは既に本番 `BackendOps::log_softmax_backward`
経由で結線済みであり、本記入欄は実機での正しさ検証のみを対象とする）。

## 記入欄

**2026-09-18（UTC 01:47Z）DGX Spark GB10 で実測済み**（転送元 origin/main `536c56a8`。
環境は `env_info.txt`〈`hostname: masked`〉。ログ内のユーザー名・絶対パスは `<home>` へ
置換済み）。

| ログ | コマンド | 結果 |
|---|---|---|
| `log_softmax_backward_parity.log` | `cargo test -p fandhe-ai-backend-cuda --release --test log_softmax_backward_parity -- --ignored --nocapture` | 4 pass／0 fail（`log_softmax_backward_matches_cpu_across_shapes`・`backend_ops_log_softmax_backward_matches_cpu_reference_across_shapes`・`log_softmax_backward_cancelling_upstream_grad_matches_cpu`・`log_softmax_backward_is_run_to_run_bit_identical`。0.27 s） |
| `facade-softmax_backend_parity.log` | `cargo test -p fandhe-ai --release --test softmax_backend_parity -- --ignored --nocapture` | 3 pass／0 fail（`cuda_softmax_forward_matches_cpu`・`cuda_log_softmax_forward_matches_cpu`・`cuda_log_softmax_backward_matches_cpu`。0.47 s） |

事前登録判定規則の判定:

1. `#[ignore]` 対象テスト全 pass（相殺反例 `log_softmax_backward_cancelling_upstream_grad_matches_cpu` を含む）: **充足**
2. run-to-run bit 同一: **充足**
3. `make test-ignored-cuda` 相当の非後退: 同日の全 `#[ignore]` 群実行は 330 pass／12 FAIL
   （`docs/perf/logs/cuda-realdevice-phase4-2026-09-18/README.md`）。本イシューの 4 件は
   並列実行のまま ok（by-name 新規 ok）。FAIL 12 件はいずれも本イシューの対象外
   （既知 9・既知外 3〈同 README §2.2〉）で是正せず記録のみ。全群の事前登録規則
   「FAIL ≤ 既知 10」は 12 > 10 で字義どおり不充足として同 README に記録（事後緩和なし）

tolerance／baseline は変更していない。
