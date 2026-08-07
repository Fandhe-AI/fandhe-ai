# safetensors ローダーテスト用フィクスチャの出自

`#73`（TASK-7.1a・REQ-7）の `tests/st_load.rs` が参照する固定 fixture。
`docs/spec/03-poc/poc-v2-6-interop/code/fixtures/`（PoC-v2-6・PR 由来）から
そのままコピーした（コピー元は変更・削除していない）。CI（self-hosted）は
`docs/spec`（submodule）を checkout しないため（`crates/tensor-core/tests/
tensor_views.rs` 冒頭コメント参照）、テストが依存する fixture は本クレート
配下に複製して持つ。

出所（PoC-v2-6 `code/fixtures/README.md` からの転記）: v1 実装リポ
`Fandhe-AI/rust-ai-library`（コミット `e259c369b7349ecf06eb1ca81886555dddee2262`、
取得日 2026-08-05）の `crates/rust-ai-library/tests/fixtures/pytorch-reference/`。

- PyTorch 2.13.0+cpu・`torch.manual_seed(42)`（学習）・`torch.manual_seed(0)`
  （データセット生成）で決定的に学習した MLP（`2->8->8->1`、ReLU x2 +
  Sigmoid、キー名 `fc1/fc2/fc3.{weight,bias}`）
- `st_reference.json` は入力 8 件・出力・`final_training_loss` を含む
  参照値（PyTorch 実行結果）
- `weight_shapes.json` は各レイヤーの PyTorch 側 shape
  （`fc1.weight: [8, 2]` 等。`[out_features, in_features]`）

## sha256（改竄検知用）

```
00db455b36df8ea50c65d4cb656faf25dd8231abc0dc66422505b1f4f949e1dd  model.safetensors
6f0eac0de5b88fa719b01eeb0740fc405c8165d6c81b57ef41907940e1ee0b8c  st_reference.json
6fb4f826fb56ee2656df48100d555c6500db94e93b84f125d180b3cff53373ff  weight_shapes.json
```

PoC-v2-6 fixtures README（`docs/spec/03-poc/poc-v2-6-interop/code/fixtures/
README.md`）記載値と一致することを確認済み。
