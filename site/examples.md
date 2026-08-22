# Examples

`crates/facade/examples/` に実際にコンパイル・実行確認済みの example
として置いてあるコードを転記したページです。すべて CPU バックエンドで
完結するため、実機（CUDA/Metal）がない環境（GitHub ホステッド CI 含む）
でもそのまま `cargo run` できます。

## ページ一覧・実行コマンド

| ページ | 内容 | 実行コマンド |
|---|---|---|
| [学習ループ](/examples/training-loop/) | `compat::Sequential` + 手動 SGD の学習ループ | `cargo run -p facade --example training_loop` |
| [推論](/examples/inference/) | `predict()` と 明示的な `Tape` 経由 `forward()` の 2 経路 | `cargo run -p facade --example inference` |
| [GEMM ベンチ](/examples/gemm-bench/) | `Var::matmul` の 5 回計測中央値デモ | `cargo run --release -p facade --example gemm_bench` |

各ページのコードブロックは、対応する `crates/facade/examples/*.rs` の
「`use` 以降の実行コード部分」（冒頭のモジュールドキュメンテーション
コメントを除く）とバイト一致です。
