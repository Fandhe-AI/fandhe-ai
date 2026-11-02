# Hardshrink・Softshrink（#3817）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
新規カーネルは無く、既存の `masked_fill`・`clamp`・`sub` 経路の確認である。実測は Phase 10 の #3860（`<cuda-node>`）と #3861（`<metal-node>`）。

## 対象テスト（`crates/facade/tests/shrink_ops_backend_parity.rs`）

| テスト | 条件 | 判定 |
|---|---|---|
| `cuda_bit_exact_forward_matches_cpu_reference` | `#[ignore]` | CPU と bit 一致 |
| `cuda_bit_exact_backward_matches_cpu_reference` | `#[ignore]` | CPU と bit 一致 |
| `metal_bit_exact_forward_matches_cpu_reference` | `#[ignore]`・macOS 限定 | CPU と bit 一致 |
| `metal_bit_exact_backward_matches_cpu_reference` | `#[ignore]`・macOS 限定 | CPU と bit 一致 |

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test shrink_ops_backend_parity -- --ignored --exact cuda_bit_exact_forward_matches_cpu_reference
cargo test -p fandhe-ai --test shrink_ops_backend_parity -- --ignored --exact cuda_bit_exact_backward_matches_cpu_reference
# <metal-node>
cargo test -p fandhe-ai --test shrink_ops_backend_parity -- --ignored --exact metal_bit_exact_forward_matches_cpu_reference
cargo test -p fandhe-ai --test shrink_ops_backend_parity -- --ignored --exact metal_bit_exact_backward_matches_cpu_reference
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
