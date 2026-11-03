# LPPool（#3825）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
新規カーネルは無く、既存の `pow_scalar`・`avg_pool`・`sign`・`abs`・`relu`・`mul` 経路（GPU tape ではホストフォールバックを含む）の確認である。実測は Phase 10 の #3860（`<cuda-node>`）と #3861（`<metal-node>`）。

## 対象テスト

| ファイル | テスト | 条件 | 判定 |
|---|---|---|---|
| `crates/facade/tests/lp_pool_ops_backend_parity.rs` | `cuda_lp_pool_ops_match_cpu_reference` | `#[ignore]` | CPU と REQ-2 統一複合判定（forward と入力勾配、1d／2d／3d） |
| 同上 | `metal_lp_pool_ops_match_cpu_reference` | `#[ignore]`・macOS 限定 | 同上 |
| `crates/facade/tests/compat_sequential_lp_pool_backend_parity.rs` | `cuda_lp_pool_layers_match_cpu` | `#[ignore]` | CPU と REQ-2 統一複合判定（3d 層） |
| 同上 | `metal_lp_pool_layers_match_cpu` | `#[ignore]`・macOS 限定 | 同上 |

tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）は変更しない。`train_gpu_parity.rs` の CASES への登録も Phase 10 で行う。

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test lp_pool_ops_backend_parity -- --ignored --exact cuda_lp_pool_ops_match_cpu_reference
cargo test -p fandhe-ai --test compat_sequential_lp_pool_backend_parity -- --ignored --exact cuda_lp_pool_layers_match_cpu
# <metal-node>
cargo test -p fandhe-ai --test lp_pool_ops_backend_parity -- --ignored --exact metal_lp_pool_ops_match_cpu_reference
cargo test -p fandhe-ai --test compat_sequential_lp_pool_backend_parity -- --ignored --exact metal_lp_pool_layers_match_cpu
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
