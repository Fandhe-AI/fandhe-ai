# facade 公開 `TapeF64::backward_create_graph`／`CreateGraphResultF64` の GPU 実機検証（イシュー #3786）

## 背景

- f64 テープの `create_graph` 機構の第 2 段（`matmul`・`sum`・`mean` と閉包性）と facade への公開を実装した（#3786・親 #3719。設計記録 `docs/autodiff-functional-transforms-design.md` §35・§38）。新規カーネルはなく、既存の `TypedOps<f64>`（CPU・CUDA）とホスト参照実装（Metal は常にホスト計算）を子テープ上の `VarF64` 演算として呼ぶだけ
- 本環境（Linux・実機なし）では CPU のみ実行できた。CUDA（GB10）・Metal の parity は `#[ignore]` テストとして用意した。**実機は未実測**で、下の記入欄は空のまま
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity_f64`）。tolerance・baseline は変更しない
- 比較対象は同じ loss の二階微分（`Σ grad` を入力で微分したもの）を CPU と実機で求めた値

## 実行コマンド

CUDA（DGX Spark GB10）:

```
cargo test -p fandhe-ai --test f64_create_graph_facade -- --ignored f64_create_graph_cuda_matches_cpu
```

Metal（Apple Silicon。`f64_create_graph_metal_matches_cpu` は macOS のみ。`--ignored` 単独だと CUDA 専用テストも
実行され CUDA なし環境で失敗するため、テスト名で絞る）:

```
cargo test -p fandhe-ai --test f64_create_graph_facade -- --ignored f64_create_graph_metal_matches_cpu
```

## 記入欄

| 環境 | 実行日 | コミット | 結果 | fail 要素数 |
|------|--------|---------|------|-------------|
| `<cuda-node>`（CUDA） | 未実測 | 未実測 | 未実測 | 未実測 |
| `<metal-node>`（Metal） | 未実測 | 未実測 | 未実測 | 未実測 |
