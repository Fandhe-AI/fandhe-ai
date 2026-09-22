# `ModelCheckpoint::to_file` device forward parity 実機ランブック（イシュー #2073）

イシュー #2073（親 #2059・`phase:1`）は `compat::callbacks::
ModelCheckpoint::to_file`（safetensors ファイル保存の薄い結線）を
実装した。CPU（Linux 実行可能）経路の正しさは
`crates/facade/tests/compat_sequential_checkpoint_file.rs` の統合
テスト（bit 完全一致検証）で確認済みだが、CUDA／Metal 実機での
device forward parity テスト（`#[ignore]`。R6）は**本セッションが
CUDA／Metal 実機に到達できないため未実施のまま申し送る**
（`.claude/rules/out-of-scope-tracking.md` 対象）。

## 現状（本イシュー #2073 時点）

- `ModelCheckpoint::to_file` 自体はホスト側の状態機械（新規 `Op`／
  `BackendOps`／VJP／カーネルなし）であり、バックエンド差は
  「保存対象の state_dict をどのデバイスで学習したか」にしか現れない
  （`file_path` に何を書くかは常にホスト `Tensor<f32>`）。実機テスト
  は「保存済みファイルを読み戻し、対象デバイス上で forward した出力が
  CPU forward と REQ-2 統一複合判定で一致するか」のみを確認する
  （state_dict 自体の bit 一致は CPU 側テストで既に確認済み）
- Linux（GPU・driver 不在）で実行可能な CPU 側テストはすべて pass 済み
  （下記コマンド参照）

## 実行コマンド（後続セッションでの実施を想定）

```sh
# Linux（実装セッション）で完了済みの CPU 側テスト:
cargo test -p fandhe-ai --lib compat::callbacks
cargo test -p fandhe-ai --test compat_sequential_checkpoint_file
cargo test -p fandhe-ai --test api_surface
cargo test -p fandhe-ai --test interop_safetensors_roundtrip

# DGX Spark GB10（CUDA 実機）:
cargo test -p fandhe-ai --release --test compat_sequential_checkpoint_file -- \
  --ignored --nocapture --test-threads=1 checkpoint_file_roundtrip_on_cuda

# Mac（Metal 実機）:
cargo test -p fandhe-ai --release --test compat_sequential_checkpoint_file -- \
  --ignored --nocapture --test-threads=1 checkpoint_file_roundtrip_on_metal
```

## 結果記入欄（未実測）

| 環境 | 実行日 | 結果 | 備考 |
|------|--------|------|------|
| CUDA（DGX Spark GB10） | 未実施 | - | - |
| Metal（Mac 実機） | 未実施 | - | - |
