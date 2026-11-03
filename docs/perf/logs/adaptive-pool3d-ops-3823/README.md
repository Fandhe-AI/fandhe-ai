# 3D 適応プーリング（#3823）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
`adaptive_pool3d_avg`／`adaptive_pool3d_max` は CUDA／Metal に専用カーネルが無く、既定の `Unsupported` から
ホストの共有カーネルへフォールバックする。ここで確認するのは「GPU tape でもフォールバック経路が CPU tape と
bit 一致すること」で、GPU カーネル自体の parity ではない。実測は Phase 10 の #3860（`<cuda-node>`）と
#3861（`<metal-node>`）。

## 対象テスト

| ファイル | テスト | 条件 | 判定 |
|---|---|---|---|
| `crates/facade/tests/adaptive_pool3d_ops_backend_parity.rs` | `cuda_adaptive_pool3d_ops_match_cpu_reference` | `#[ignore]` | CPU と bit 一致（Max の値・索引、Avg の forward、両 backward） |
| 同上 | `metal_adaptive_pool3d_ops_match_cpu_reference` | `#[ignore]`・macOS 限定 | CPU と bit 一致 |

層経路（`Sequential::add_adaptive_*_pool3d`。#3824）の実機テスト:

| ファイル | テスト | 条件 | 判定 |
|---|---|---|---|
| `crates/facade/tests/compat_sequential_adaptive_pool3d_backend_parity.rs` | `cuda_compat_sequential_adaptive_pool3d_matches_cpu` | `#[ignore]` | CPU と REQ-2 統一複合判定（`assert_parity`）で一致（forward と入力勾配） |
| 同上 | `metal_compat_sequential_adaptive_pool3d_matches_cpu` | `#[ignore]`・macOS 限定 | 同上 |

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test adaptive_pool3d_ops_backend_parity -- --ignored --exact cuda_adaptive_pool3d_ops_match_cpu_reference
# <metal-node>
cargo test -p fandhe-ai --test adaptive_pool3d_ops_backend_parity -- --ignored --exact metal_adaptive_pool3d_ops_match_cpu_reference
```

```sh
# <cuda-node>
cargo test -p fandhe-ai --test compat_sequential_adaptive_pool3d_backend_parity -- --ignored --exact cuda_compat_sequential_adaptive_pool3d_matches_cpu
# <metal-node>
cargo test -p fandhe-ai --test compat_sequential_adaptive_pool3d_backend_parity -- --ignored --exact metal_compat_sequential_adaptive_pool3d_matches_cpu
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
