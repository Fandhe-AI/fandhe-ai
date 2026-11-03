# FractionalMaxPool2d／3d（#3826）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
`fractional_max_pool2d_windows`／`fractional_max_pool3d_windows` は CUDA／Metal に専用カーネルが無く、既定の
`Unsupported` からホストの共有カーネルへフォールバックする。窓表はホストで 1 回だけ整数に確定する（GPU 側で
式を計算し直さない）。ここで確認するのは「同じシードの下で GPU tape でもフォールバック経路が CPU tape と
bit 一致すること」で、GPU カーネル自体の parity ではない。実測は Phase 10 の #3860（`<cuda-node>`）と
#3861（`<metal-node>`）。

## 対象テスト

| ファイル | テスト | 条件 | 判定 |
|---|---|---|---|
| `crates/facade/tests/fractional_max_pool_ops_backend_parity.rs` | `cuda_fractional_max_pool_ops_match_cpu_reference` | `#[ignore]` | CPU と bit 一致（2d・3d の値・索引・入力勾配。各 tape の直前に同じシードで `manual_seed`） |
| 同上 | `metal_fractional_max_pool_ops_match_cpu_reference` | `#[ignore]`・macOS 限定 | CPU と bit 一致 |

層経路（`Sequential::add_fractional_max_pool*`。#3827）の実機テストは次の表に追記した。

| ファイル | テスト | 条件 | 判定 |
|---|---|---|---|
| `crates/facade/tests/compat_sequential_fractional_max_pool_backend_parity.rs` | `cuda_compat_sequential_fractional_max_pool_matches_cpu` | `#[ignore]` | REQ-2 統一複合判定（`assert_parity`）。forward と入力勾配。各デバイスの直前に同じシードで `manual_seed` |
| 同上 | `metal_compat_sequential_fractional_max_pool_matches_cpu` | `#[ignore]`・macOS 限定 | 同上 |

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test fractional_max_pool_ops_backend_parity -- --ignored --exact cuda_fractional_max_pool_ops_match_cpu_reference
# <metal-node>
cargo test -p fandhe-ai --test fractional_max_pool_ops_backend_parity -- --ignored --exact metal_fractional_max_pool_ops_match_cpu_reference
```

```sh
# <cuda-node>（層経路。#3827）
cargo test -p fandhe-ai --test compat_sequential_fractional_max_pool_backend_parity -- --ignored --exact cuda_compat_sequential_fractional_max_pool_matches_cpu
# <metal-node>（層経路。#3827）
cargo test -p fandhe-ai --test compat_sequential_fractional_max_pool_backend_parity -- --ignored --exact metal_compat_sequential_fractional_max_pool_matches_cpu
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
| `<cuda-node>`（層経路 #3827） | 未実測 | - | - |
| `<metal-node>`（層経路 #3827） | 未実測 | - | - |
