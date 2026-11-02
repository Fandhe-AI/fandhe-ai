# Bilinear（#3818）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
新規カーネルは無く、既存の `matmul`・`mul`・`sum`・`add` 経路の確認である。実測は Phase 10 の #3860（`<cuda-node>`）と #3861（`<metal-node>`）。

## 対象テスト（`crates/facade/tests/bilinear_ops_backend_parity.rs`）

| テスト | 条件 | 判定 |
|---|---|---|
| `cuda_bilinear_forward_matches_cpu_reference` | `#[ignore]` | REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満） |
| `cuda_bilinear_backward_matches_cpu_reference` | `#[ignore]` | 同上（dx1・dx2・dw・db） |
| `metal_bilinear_forward_matches_cpu_reference` | `#[ignore]`・macOS 限定 | 同上 |
| `metal_bilinear_backward_matches_cpu_reference` | `#[ignore]`・macOS 限定 | 同上 |

bit 一致は主張しない（`matmul` の FMA 契約・CUDA の TF32 opt-in・`sum` の f64 チャンク 2 段を通る合成のため）。
tolerance・baseline は変えない。複合判定を外れた場合は tolerance を緩めず所有者へ戻す。

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test bilinear_ops_backend_parity -- --ignored --exact cuda_bilinear_forward_matches_cpu_reference
cargo test -p fandhe-ai --test bilinear_ops_backend_parity -- --ignored --exact cuda_bilinear_backward_matches_cpu_reference
# <metal-node>
cargo test -p fandhe-ai --test bilinear_ops_backend_parity -- --ignored --exact metal_bilinear_forward_matches_cpu_reference
cargo test -p fandhe-ai --test bilinear_ops_backend_parity -- --ignored --exact metal_bilinear_backward_matches_cpu_reference
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
