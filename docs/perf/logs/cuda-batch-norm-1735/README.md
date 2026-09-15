# イシュー #1735 実測記録先（CUDA BatchNorm1d／2d）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10 実機
への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧のみを提供し、実測は GB10 実機を持つセッションへ申し送る
（`docs/perf/logs/cuda-conv2d-1766/README.md` と同型の運用）。

## 目的

イシュー #1735「CUDA BatchNorm1d／2d（train／eval・running stats）」の
実機正しさ検証記録先。以下の `#[ignore]` テストを GB10 実機で実行して
green を確認することが目的。

## 実行コマンド

```bash
# 1) crates/backend-cuda の train／infer bit 同一・parity（形状網羅。
#    rank 2/3/4・warp 幅 32 の端数・weight/bias 有無・極値・NaN 伝播・
#    run-to-run 決定性・backend-cpu 直接突合）
cargo test -p fandhe-ai-backend-cuda --release --test batch_norm_parity -- --ignored --nocapture

# 2) crates/facade の batch_norm forward／backward REQ-2 複合判定
#    （train forward・infer forward・train backward〈dW／db／dx〉・
#    rank 4〈BatchNorm2d〉forward）
cargo test -p fandhe-ai --release --test batch_norm_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）にも上記が
自動的に含まれる。

## 保存すべきログ

- `batch_norm_parity-ignored.log`: 上記 1) の生出力（形状ごとの
  parity・run-to-run 決定性の判定内訳）
- `batch_norm_backend_parity-ignored.log`: 上記 2) の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **train／infer forward の REQ-2 複合判定**: CPU 参照実装（テスト
  専用 naive `f64` 参照実装・`backend-cpu::run_batch_norm_train_f32`・
  `fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（`docs/batch-norm-
  ops-design.md` §3.5「REQ-2 複合判定は将来の GPU 側との突合に用いる」
  の適用。bit 一致は主張・assert しない）
- **backward（dW／db／dx）の REQ-2 複合判定**: `matmul → batch_norm
  → mse_loss` の backward を CUDA tape と CPU tape で突合し全 fail
  0 件であること（forward は CUDA 融合カーネル・backward はホスト
  VJP という組合せの結線確認。VJP 自体は #1732 でホスト側に確定済み）
- **run-to-run 決定性**: 同一入力で 2 回起動しても bit 同一であること
  （`batch_norm_train_is_run_to_run_deterministic`）
- **NaN 伝播**: NaN が属するチャネルのみへ伝播し他チャネルを汚染しな
  いこと（`batch_norm_train_propagates_nan_only_for_channel_with_nan`）
- 上記いずれも FAIL の場合は本番結線（`ops.rs::CudaBackendOps::
  batch_norm_train`／`batch_norm_infer`）を見直す（イシュー再オープン）
