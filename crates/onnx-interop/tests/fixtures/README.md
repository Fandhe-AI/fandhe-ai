# onnx-interop テストフィクスチャの出自

TASK-7.2a（イシュー #77）向けに `docs/spec`（正本 submodule）の
`03-poc/poc-v2-6-interop/code/fixtures/` からコピーしたもの。生成条件の詳細は
コピー元の `docs/spec/03-poc/poc-v2-6-interop/code/fixtures/README.md`（正本）を
参照する。本ファイルは本クレートのテストが依存する事実（出自・sha256）のみを
記録する。

## `model.onnx`（1,204 bytes）

- 出自: PoC-6・PoC-v2-6（PyTorch エクスポート済み MLP。`2->8->8->1`、
  Gemm/Relu x2 + Gemm/Sigmoid、initializer キー名 `fc1/fc2/fc3.{weight,bias}`）
- sha256: `917c8c3e87e6c4c4af95d0dbb8193196e00bd6ed4f9a63d76cad1c14a04b7446`
- グラフ構造（`onnx_decode.rs` の期待値の根拠。本クレートの `graph::build_graph`
  で実際に decode して確認済み）:
  - node（6）: `["/fc1/Gemm", "/relu/Relu", "/fc2/Gemm", "/relu_1/Relu", "/fc3/Gemm", "/sigmoid/Sigmoid"]`
  - op_type 列: `["Gemm", "Relu", "Gemm", "Relu", "Gemm", "Sigmoid"]`
  - initializer（6）: `fc1.weight` shape=[8,2]・`fc1.bias` shape=[8]・
    `fc2.weight` shape=[8,8]・`fc2.bias` shape=[8]・`fc3.weight` shape=[1,8]・
    `fc3.bias` shape=[1]（すべて F32）
  - input=`["input"]`、output=`["output"]`

## `slice_repro.onnx`（527 bytes）

- 出自: PoC-v2-6 新規作成分（`onnx.helper` で構築した動的境界 Slice の最小
  再現グラフ。v1 の `burn-onnx` 失敗パターン
  〈Shape -> Gather -> Unsqueeze -> Concat -> 動的境界 Slice〉を模す）
- sha256: `7b032bd48acd85064e97c16c47275afb7c59124a19bb1d76dfc39a3f2e32f2b9`
- グラフ構造:
  - node（5）: `["n_shape", "n_gather", "n_unsqueeze", "n_concat", "n_slice"]`
  - op_type 列: `["Shape", "Gather", "Unsqueeze", "Concat", "Slice"]`
  - initializer（4、すべて I64）: `const_axes` shape=[2] data=[0,1]・
    `const_4` shape=[1] data=[4]・`const_starts` shape=[2] data=[0,0]・
    `const_gather_idx` shape=[1] data=[0]
  - input=`["x"]`、output=`["output"]`

## `transformer.onnx`（コミットしない）

12MB 超のバイナリのためリポジトリにコミットしない（`docs/spec` の
PoC-v2-6 README と同方針）。

- 出所: v1 実装リポ `Fandhe-AI/rust-ai-library-v1`（コミット
  `a14568897521f7bea6eac93218fe917cf2a25f04`）の
  `crates/rust-ai-library/src/interop/onnx_model/transformer.onnx`
- サイズ: 12,632,320 bytes
- sha256: `6f6430e6b99408c949635da16ed7d6e7cdc2a500db050ae80c660b3b8b057b0f`
- 実測値（`node=165`・`initializer=12`・オペ 20 種別）の根拠:
  `docs/spec/03-poc/poc-v2-6-interop/evidence/transformer_probe.log`

取得手順（受け入れ条件確認・`#[ignore]` テストの実行）:

```bash
# 1. 上記出所から取得し sha256 を検証する
sha256sum <取得した transformer.onnx のパス>
# 期待値: 6f6430e6b99408c949635da16ed7d6e7cdc2a500db050ae80c660b3b8b057b0f

# 2. 環境変数でパスを指定して #[ignore] テストを実行する
ONNX_INTEROP_TRANSFORMER_ONNX=<取得したパス> \
  cargo test -p onnx-interop -- --ignored --nocapture
```

`tests/onnx_decode.rs` の `transformer_onnx_decodes_expected_graph_structure`
（decode・`build_graph` のみ）に加え、`tests/onnx_transformer_e2e.rs`
（イシュー #87・TASK-7.4a）が同じ環境変数を使い `decode → build_graph →
onnx::interp::run` の全経路 end-to-end 推論を実行し、PyTorch 参照出力
（`tests/fixtures/pytorch-transformer/reference.json`。出自は同ディレクトリの
`README.md` 参照）と REQ-7 判定式で数値一致を確認する。上記コマンドの
`--ignored` 一括実行で両テストとも対象になる。

環境変数未設定時はこのテストがスキップされる。`--ignored` を渡さない通常の
`cargo test` では `#[ignore]` によりそもそも実行対象外となり、`--ignored`
（`make test-ignored` 等）を渡した場合でも環境変数未設定なら早期 return で
スキップする（panic しない）。CI は本リポにコミットされた `model.onnx` /
`slice_repro.onnx` のテストのみを常時実行する
（`.claude/rules/ci.md` の実機依存分離と同じ運用）。
