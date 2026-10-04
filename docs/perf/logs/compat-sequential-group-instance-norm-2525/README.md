# compat::Sequential の GroupNorm／InstanceNorm 実機 parity 申し送り（イシュー #2525）

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_group_instance_norm.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_group_norm` → `add_instance_norm`）の `forward` 出力を、CPU `predict` と
  REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）で比較する。
- 新規 `Op`／`BackendOps`／カーネルはなく、既存 `layer_norm` カーネルの正しさに依存する。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_group_instance_norm_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_group_instance_norm_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_group_instance_norm_matches_cpu | 未実測 | |
| M4 Max | metal_group_instance_norm_matches_cpu | 未実測 | |
