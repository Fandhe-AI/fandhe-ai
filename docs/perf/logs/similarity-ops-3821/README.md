# CosineSimilarity・PairwiseDistance（#3821）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
新規カーネルは無い。2 演算は `BackendOps` に対応メソッドを持たない host の `f64` 計算（新 `Op` 2 件）で、入力を host へ
実体化してから計算するため、バックエンド間で bit 一致するはずである。実測は Phase 10 の #3860（`<cuda-node>`）と #3861（`<metal-node>`）。

## 対象テスト（`crates/facade/tests/similarity_ops_backend_parity.rs`）

| テスト | 条件 | 判定 |
|---|---|---|
| `cuda_bit_exact_forward_matches_cpu_reference` | `#[ignore]` | bit 完全一致（NaN はクラス一致。REQ-2 統一複合判定より強い） |
| `cuda_bit_exact_backward_matches_cpu_reference` | `#[ignore]` | 同上（dx1・dx2） |
| `metal_bit_exact_forward_matches_cpu_reference` | `#[ignore]`・macOS 限定 | 同上 |
| `metal_bit_exact_backward_matches_cpu_reference` | `#[ignore]`・macOS 限定 | 同上 |

tolerance・baseline は変えない。bit 一致を外れた場合は tolerance を緩めず所有者へ戻す。

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test similarity_ops_backend_parity -- --ignored --exact cuda_bit_exact_forward_matches_cpu_reference
cargo test -p fandhe-ai --test similarity_ops_backend_parity -- --ignored --exact cuda_bit_exact_backward_matches_cpu_reference
# <metal-node>
cargo test -p fandhe-ai --test similarity_ops_backend_parity -- --ignored --exact metal_bit_exact_forward_matches_cpu_reference
cargo test -p fandhe-ai --test similarity_ops_backend_parity -- --ignored --exact metal_bit_exact_backward_matches_cpu_reference
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
