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
| GB10（CUDA） | 未実施 | - | - | - |
| M4 Max（Metal） | 未実施 | - | - | - |
