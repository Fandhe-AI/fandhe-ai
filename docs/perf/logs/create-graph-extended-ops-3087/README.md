# facade 拡張 Op の create_graph GPU 実機検証（イシュー #3087）

## 背景

- facade の `Tape::backward_create_graph`・`hvp` で、#3077〜#3086 で対象に加えた Op（LayerNorm・Conv2d・MaxPool2d を代表）が CPU と同じ二階の結果になることを実機で確かめる。決定記録は `docs/autodiff-higher-order-grad-decision.md` §32
- 新規カーネルはなく `Var` 演算の合成のみ。線形代数は forward が GPU からホスト f64 にフォールバックするので代表にしていない
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance は変更しない
- 実測は本イシューの範囲外で #3168・#3169 へ申し送る

## 実行コマンド

CUDA（DGX Spark GB10）:

```
cargo test -p fandhe-ai --test create_graph_facade -- --ignored create_graph_extended_ops_cuda_matches_cpu
```

Metal（Apple Silicon。`--ignored` 単独だと CUDA 専用テストも実行され失敗するため、テスト名で絞る）:

```
cargo test -p fandhe-ai --test create_graph_facade -- --ignored metal
```

## 記入欄

| 環境 | 実行日 | コミット | 結果 | fail 要素数 |
|------|--------|---------|------|-------------|
| GB10（CUDA） | 未実施 | - | - | - |
| M4 Max（Metal） | 未実施 | - | - | - |
