# ChannelShuffle（#3822）の GPU 実機 parity 申し送り

状態: **未実測**。実装環境に CUDA・Metal の実機がなく、測っていない値を pass と書かない。
新規カーネルは無く、既存の `contiguous`・`reshape`・`permute` 経路の確認である。実測は Phase 10 の #3860（`<cuda-node>`）と #3861（`<metal-node>`）。

## 対象テスト

| ファイル | テスト | 条件 | 判定 |
|---|---|---|---|
| `crates/facade/tests/channel_shuffle_backend_parity.rs` | `cuda_channel_shuffle_forward_matches_cpu`・`cuda_channel_shuffle_backward_matches_cpu` | `#[ignore]` | CPU と bit 一致 |
| 同上 | `metal_channel_shuffle_forward_matches_cpu`・`metal_channel_shuffle_backward_matches_cpu` | `#[ignore]`・macOS 限定 | CPU と bit 一致 |
| `crates/facade/tests/compat_sequential_channel_shuffle_backend_parity.rs` | `cuda_channel_shuffle_layers_bit_exact` | `#[ignore]` | CPU と bit 一致 |
| 同上 | `metal_channel_shuffle_layers_bit_exact` | `#[ignore]`・macOS 限定 | CPU と bit 一致 |

## 実行コマンド（1 本ずつ別プロセス）

```sh
# <cuda-node>
cargo test -p fandhe-ai --test channel_shuffle_backend_parity -- --ignored --exact cuda_channel_shuffle_forward_matches_cpu
cargo test -p fandhe-ai --test channel_shuffle_backend_parity -- --ignored --exact cuda_channel_shuffle_backward_matches_cpu
cargo test -p fandhe-ai --test compat_sequential_channel_shuffle_backend_parity -- --ignored --exact cuda_channel_shuffle_layers_bit_exact
# <metal-node>
cargo test -p fandhe-ai --test channel_shuffle_backend_parity -- --ignored --exact metal_channel_shuffle_forward_matches_cpu
cargo test -p fandhe-ai --test channel_shuffle_backend_parity -- --ignored --exact metal_channel_shuffle_backward_matches_cpu
cargo test -p fandhe-ai --test compat_sequential_channel_shuffle_backend_parity -- --ignored --exact metal_channel_shuffle_layers_bit_exact
```

cfg により 0 本しか実行されなかった結果を pass の根拠にしない（`running 1 test` を確認する）。

## 記入欄

| ノード | 測定日 | 測定コミット | 結果 |
|---|---|---|---|
| `<cuda-node>` | 未実測 | - | - |
| `<metal-node>` | 未実測 | - | - |
