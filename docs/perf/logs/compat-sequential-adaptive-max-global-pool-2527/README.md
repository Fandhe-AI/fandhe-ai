# compat::Sequential の AdaptiveMaxPool／GlobalPool 実機 parity 申し送り（イシュー #2527）

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_adaptive_max_global_pool.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_adaptive_max_pool2d` → `add_global_pool(Max, true)` → `add_global_pool(Avg, false)`）の `forward` 出力を、
  CPU `predict` と **bit 完全一致**で比較する（純粋な選択演算。`GlobalPool(Avg)` は既存 `adaptive_avg_pool*` の f64 アキュムレータ契約のまま。
  GPU 側はホストフォールバック経由）。
- 新規 `Op`／`BackendOps`／カーネルはない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_adaptive_max_global_pool_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_adaptive_max_global_pool_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_adaptive_max_global_pool_bit_exact | 未実測 | |
| M4 Max | metal_adaptive_max_global_pool_bit_exact | 未実測 | |
