# `Var::batch_norm` rank 5・BatchNorm3d（#3819・#3820）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
新規カーネルは無く、既存の `batch_norm_train`／`batch_norm_infer` 経路が rank 5 入力（`(n, c, spatial = D*H*W)` へ畳まれる）でも成立することの確認である。実測は #3860（`<cuda-node>`）と #3861（`<metal-node>`）。

## 対象テスト（`crates/facade/tests/batch_norm_backend_parity.rs`）

| テスト | 条件 | 判定 |
|---|---|---|
| `cuda_batch_norm_train_forward_rank5_matches_cpu` | `#[ignore]` | REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満） |
| `metal_batch_norm_train_forward_rank5_matches_cpu` | `#[ignore]` | 同上 |

## 対象テスト（#3820。`crates/facade/tests/compat_sequential_layers_backend_parity.rs`）

| テスト | 条件 | 判定 |
|---|---|---|
| `cuda_batch_norm3d_matches_cpu` | `#[ignore]` | REQ-2 統一複合判定 |
| `metal_batch_norm3d_matches_cpu`（`cfg(target_os = "macos")`） | `#[ignore]` | 同上 |

GPU と CPU の bit 一致は主張しない。同一バックエンド内で rank 3 reshape と bit 一致するかは、必要なら実測時に追加で確認する。
tolerance・baseline は変えない。複合判定を外れた場合は tolerance を緩めず所有者へ戻す。

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test batch_norm_backend_parity -- --ignored --exact cuda_batch_norm_train_forward_rank5_matches_cpu
# <metal-node>
cargo test -p fandhe-ai --test batch_norm_backend_parity -- --ignored --exact metal_batch_norm_train_forward_rank5_matches_cpu
```

```sh
# <cuda-node>（#3820）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored --exact cuda_batch_norm3d_matches_cpu
# <metal-node>（#3820）
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity -- --ignored --exact metal_batch_norm3d_matches_cpu
```

`running 1 test` を確認する（0 本実行を pass の根拠にしない）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
