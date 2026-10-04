# compat::Sequential の PixelShuffle／PixelUnshuffle 実機 parity 申し送り（イシュー #2526）

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_pixel_shuffle.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_pixel_shuffle` → `add_pixel_unshuffle` → `add_pixel_shuffle`）の `forward` 出力を、
  CPU `predict` と **bit 完全一致**で比較する（算術を含まない contiguous／reshape／permute の合成のため）。
- 新規 `Op`／`BackendOps`／カーネルはない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_pixel_shuffle_layers_bit_exact | 未実測 | |
| M4 Max | metal_pixel_shuffle_layers_bit_exact | 未実測 | |
