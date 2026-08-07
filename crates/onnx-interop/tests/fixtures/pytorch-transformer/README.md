# `transformer.onnx` end-to-end 参照値フィクスチャの出自

イシュー #87（TASK-7.4a・REQ-7）の `tests/onnx_transformer_e2e.rs` が参照する
PyTorch 実行結果の固定 fixture。

## `reference.json`（939,579 bytes）

- 出所: v1 実装リポ `Fandhe-AI/rust-ai-library-v1` commit
  `a14568897521f7bea6eac93218fe917cf2a25f04` の
  `crates/rust-ai-library/tests/fixtures/pytorch-transformer/reference.json`
- sha256: `84c0d0055ccd6a4cf32c4b5f9a0b6f6b1028e3344ded2f1d763ac426b41915c8`
- モデル構成（`config` フィールド。同梱 `transformer.onnx` のエクスポート元
  PyTorch モデルの構成）: `d_model=512`・`n_heads=8`・`d_ff=2048`・
  `num_layers=1`・`activation=gelu`・`norm_first=false`（post-norm）・
  `dropout=0.0`・`batch=2`・`seq_len=16`
- `input_shape`/`output_shape`: `[2, 16, 512]`（`[batch, seq_len, d_model]`）
- `input`/`output`: 上記 shape のネスト配列（`Vec<Vec<Vec<f32>>>` としてそのまま
  deserialize 可能）。入力値そのものが JSON に保存されているため、乱数再現に
  依存せず PyTorch/Python 環境なしで参照突合できる（自動運転での再現性確保）
- `onnxruntime_vs_pytorch_*` 系フィールド: v1 側で onnxruntime 実行結果と
  PyTorch 実行結果を突合したセルフチェックのメタデータ（本クレートのテストは
  参照しない。`config` 同様に無視して deserialize する）

## `transformer.onnx` 本体との対応

対になる `transformer.onnx`（12MB 超・非コミット）は同 commit の
`crates/rust-ai-library/src/interop/onnx_model/transformer.onnx`
（sha256 `6f6430e6b99408c949635da16ed7d6e7cdc2a500db050ae80c660b3b8b057b0f`）。
取得手順は `tests/fixtures/README.md` の `transformer.onnx` 節を参照する。

## 再取得手順

```bash
gh api -H "Accept: application/vnd.github.raw" \
  "repos/Fandhe-AI/rust-ai-library-v1/contents/crates/rust-ai-library/tests/fixtures/pytorch-transformer/reference.json?ref=a14568897521f7bea6eac93218fe917cf2a25f04" \
  > reference.json
sha256sum reference.json
# 期待値: 84c0d0055ccd6a4cf32c4b5f9a0b6f6b1028e3344ded2f1d763ac426b41915c8
```
