# facade 公開 `backward_create_graph` の GPU 実機検証（イシュー #2545）

## 背景

- facade `Tape::backward_create_graph`／`CreateGraphResult` の公開（決定記録 `docs/autodiff-higher-order-grad-decision.md` §17・§18）。新規カーネルはなく、`Var` 演算の合成のみ
- #2062 の実測ログ（`docs/perf/logs/create-graph-remaining-ops-2062/README.md`）が「実機用テストは未整備」としていた点を、本テストで埋める
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance は変更しない

## 実行コマンド

CUDA（DGX Spark GB10。CUDA 専用 2 テストをテスト名で絞り、フィルタ解釈に依存しないよう別コマンドで実行する）:

```
cargo test -p fandhe-ai --test create_graph_facade -- --ignored create_graph_hessian_cuda_matches_cpu
cargo test -p fandhe-ai --test create_graph_facade -- --ignored create_graph_rejects_device_mismatch
```

Metal（Apple Silicon。`create_graph_hessian_metal_matches_cpu` は macOS のみ。`--ignored` 単独だと CUDA 専用の
`create_graph_hessian_cuda_matches_cpu`・`create_graph_rejects_device_mismatch` も実行され CUDA なし環境で失敗するため、
テスト名で絞る）:

```
cargo test -p fandhe-ai --test create_graph_facade -- --ignored metal
```

## 記入欄

| 環境 | 実行日 | コミット | 結果 | fail 要素数 |
|------|--------|---------|------|-------------|
| GB10（CUDA） | 2026-10-19 | 9e2f7ddcc（[`cuda/rev-stamp.txt`](./cuda/rev-stamp.txt)） | `create_graph_hessian_cuda_matches_cpu` pass（1/1・0.55s。ログ [`cuda/create_graph_hessian_cuda_matches_cpu.log`](./cuda/create_graph_hessian_cuda_matches_cpu.log)）。`create_graph_rejects_device_mismatch` pass（1/1・0.21s。型付きエラー検査で parity ではない。ログ [`cuda/create_graph_rejects_device_mismatch.log`](./cuda/create_graph_rejects_device_mismatch.log)）。#3168。各 `--exact --nocapture --test-threads=1`。GPU 経路かホストフォールバックかは確かめていない | 0 |
| Mac mini（Apple M4・Metal） | 2026-10-19 | bf894108a（[`metal/rev-stamp.txt`](./metal/rev-stamp.txt)） | `create_graph_hessian_metal_matches_cpu` pass（1/1・0.14s。`--exact --nocapture --test-threads=1`。ログ [`metal/create_graph_hessian_metal_matches_cpu.log`](./metal/create_graph_hessian_metal_matches_cpu.log)）。#3169（集計 [`../metal-autodiff-parity-3169/`](../metal-autodiff-parity-3169/README.md)）。GPU 経路かホストフォールバックかは確かめていない | 0 |
