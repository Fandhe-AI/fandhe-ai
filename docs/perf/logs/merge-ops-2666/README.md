# 結合演算（Concatenate・Add・Multiply・Average。#2666）CUDA／Metal 実機未実測の申し送り

`docs/facade-functional-api-decision.md` §17「#2666 実装記録」参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、次の 4 テストは `#[ignore]` のまま未実測である。

| テスト | 場所 | 対象 |
|---|---|---|
| `cuda_merge_ops_match_cpu_reference`・`metal_merge_ops_match_cpu_reference`（`cfg(target_os = "macos")`） | `crates/facade/tests/merge_ops_backend_parity.rs` | `merge_ops` 4 関数の forward・全入力 backward |
| `cuda_merge_graph_matches_cpu_reference`・`metal_merge_graph_matches_cpu_reference`（`cfg(target_os = "macos")`） | `crates/facade/src/compat/functional/merge_tests.rs`（`--lib`。実装が `#[cfg(test)]` 限定の `pub(crate)` のため結合テストでは実行できない） | fixture の `chained_merges_two_inputs`（結合の出力を別の結合へ入れる連鎖・2 入力・2 出力） |

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test merge_ops_backend_parity -- --ignored --nocapture cuda_merge_ops_match_cpu_reference
cargo test -p fandhe-ai --lib -- --ignored --nocapture cuda_merge_graph_matches_cpu_reference

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test merge_ops_backend_parity -- --ignored --nocapture metal_merge_ops_match_cpu_reference
cargo test -p fandhe-ai --lib -- --ignored --nocapture metal_merge_graph_matches_cpu_reference
```

CPU 版は属性なしテスト（`cpu_matches_naive_reference`・`hand_computed_expectations`・
`matches_pytorch_reference_for_merge_graphs`・`crates/autodiff/tests/merge_ops_parity.rs` の PyTorch 2.14.0 実行値突合）で
検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない（新規 `Op`・`BackendOps` メソッド・VJP なし）。実機で走るのは **既存の `Var` 演算**
  （`concat`・`add`・`mul`・`div` と各 VJP）であり、確認対象は新規カーネルの parity ではなく既存カーネルの新しい呼び出し形
  （多数件の左畳み込み・結合の出力を別の結合へ入れる連鎖・同一ノードの fan-out による勾配合流）である。
- いずれも CPU tape と実機 tape の出力・入力勾配が REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
  満たす見込み。形状は小さく、gemm を呼ばないため Metal の split-K 経路は発動しない。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
