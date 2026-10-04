# compat::Sequential の Dropout2d／AlphaDropout／EmbeddingBag 実機 parity 申し送り（イシュー #2528）

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_dropout_embedding_bag.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_dropout2d(0.5)` → `add_alpha_dropout(0.25)`）の `forward` 出力を、同じ `manual_seed` から実行した CPU
  `predict` と **bit 完全一致**で比較する（`Op::Dropout` の再利用。GPU 側はホストフォールバック経由。マスクはホストで生成）。
- `add_embedding_bag(6, 4, mode, Some(0), 21)`（`mode` は Sum／Mean／Max の 3 種）の `forward` 出力を CPU `predict` と **bit 完全一致**で比較する
  （既存の `embedding_bag_backend_parity.rs` は Sum／Mean を REQ-2 統一複合判定・Max を bit 完全一致としているため、実機で不一致が出た場合は
  同テストの判定方式との差を確認する。本テストで tolerance を緩めない）。
- 新規 `Op`／`BackendOps`／カーネルはない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_dropout_embedding_bag_bit_exact | 未実測 | |
| M4 Max | metal_dropout_embedding_bag_bit_exact | 未実測 | |
