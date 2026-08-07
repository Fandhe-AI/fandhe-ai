# ONNX 経路テスト用フィクスチャの出自

`#80`（TASK-7.2d・REQ-7）の `tests/onnx_poc_v2_6_match.rs`・
`tests/onnx_slice_dynamic_bounds.rs` が参照する固定 fixture。
`docs/spec/03-poc/poc-v2-6-interop/code/fixtures/`（PoC-v2-6・正本 submodule）
からそのままコピーした（コピー元は変更・削除していない）。CI（self-hosted）は
`docs/spec`（submodule）を checkout しないため（`tests/fixtures/pytorch-reference/
README.md` と同じ制約）、テストが依存する fixture は本クレート配下に複製して持つ。

## `onnx_reference.json` / `onnx_weight_shapes.json`

出所（PoC-v2-6 `code/fixtures/README.md` からの転記）: 本 spec リポ
`03-poc/poc-6-python-interop/code/fixtures/`（PoC-6・PR 由来の既存フィクスチャ）。

- 同じ `export_model.py`・同じ乱数シードで学習した MLP（`2->8->8->1`、ReLU x2 +
  Sigmoid、キー名 `fc1/fc2/fc3.{weight,bias}`）だが、`tests/fixtures/
  pytorch-reference/`（safetensors 経路）とは別実行のため参照値が僅かに
  異なる（1e-6〜1e-9 オーダー）。**ONNX 経路のテストは本ファイルの参照値
  とのみ突合し、safetensors 経路の `st_reference.json` とクロス比較しない**
  （PoC-v2-6 fixtures README の注記どおり）。
- `onnx_reference.json`: 入力 8 件・出力 8 件・`final_training_loss` を含む
  参照値（PyTorch 実行結果）
- `onnx_weight_shapes.json`: 各レイヤーの PyTorch 側 shape（`fc1.weight:
  [8, 2]` 等。`[out_features, in_features]`。ONNX `Gemm` の `transB=1` 相当）

## `onnx_weights.json`（新規生成）

`model.onnx`（PoC-v2-6 fixture）の initializer（fc1/fc2/fc3 の weight/bias、
計 6 テンソル）を、ONNX proto デコード層（#77 TASK-7.2a）が本イシュー着手時点
で main 未マージのため、一回きりのオフラインスクリプトで抽出した JSON
fixture。`model.onnx` そのものはサイズ・出自管理の都合でコミットせず、
抽出済みの重み値のみを JSON でコミットする（`onnx-interop` クレートは
protobuf ライブラリに依存しないため、テストコード内に独自の protobuf
パーサを実装しない。#77 との重複・DRY 違反を避ける）。

- 生成スクリプト: `/tmp` 配下の一回きりスクリプト
  （`extract_onnx_weights.py`。protobuf 依存なしの手書き最小 varint/
  length-delimited デコーダで `ModelProto.graph.initializer` の
  `TensorProto`（`dims`・`data_type`・`name`・`raw_data`）を読む）
- 生成コマンド: `python3 extract_onnx_weights.py
  docs/spec/03-poc/poc-v2-6-interop/code/fixtures/model.onnx
  onnx_weights.json`
- 検証: 抽出した重みで `fc1(Gemm+bias, transB 相当)→Relu→fc2→Relu→
  fc3→Sigmoid` を Python で素朴に forward 計算し、`onnx_reference.json`
  の 8 出力全件と最大相対誤差 `1.09e-06`（閾値 1e-3 を十分下回る）で
  一致することを確認済み（f32 と f64 の演算精度差・丸め順序差に由来する
  誤差で、REQ-7 判定式は Rust 側 f32 演算で別途満たす）
- 出力形式: `{"weights": {"<key>": [f32...]}, "shapes": {"<key>":
  [dim...]}}`。`weights` の値は `raw_data`（little-endian f32）を
  行優先（row-major）でフラット化した配列で、`shapes` の次元順と対応する
  （safetensors 経路と同じ `[out_features, in_features]` ネイティブ
  レイアウト。転置は呼び出し側の責務）
- `onnx_weight_shapes.json` と `onnx_weights.json` の `shapes` は一致する
  （抽出時に検証済み）

`#77`（decode API）マージ後に `model.onnx` 直接デコード経路へ切り替える場合は
本 fixture を `model.onnx` の複製に置き換え可能だが、二重整備を避けるため
本イシューでは行わない（`#86` TASK-7.4 の end-to-end 実測で改めて扱う）。

## `slice_repro_reference.json`

出所（PoC-v2-6 `code/fixtures/README.md` からの転記）: 新規作成
（`code/py/make_minimal_slice_repro.py`）。v1 の `burn-onnx` 失敗パターン
（`tracel-ai/burn#5295`。Shape → Gather → Unsqueeze → Concat → 動的境界
Slice）を模した最小グラフ（`onnx==1.22.0` の `onnx.helper`）の参照値。
onnxruntime を使わず、グラフの意味論（先頭 4 列を切り出す）を numpy で
直接計算して求めた。`input_shape: [5, 6]`・`output_shape: [5, 4]`。

`slice_repro.onnx`（バイナリグラフ本体）は本クレートのテストが直接
参照しないため複製を見送る（`#78`/`#86` でインタープリタ経由の実測を行う
際に別途複製する）。所在は `docs/spec/03-poc/poc-v2-6-interop/code/
fixtures/slice_repro.onnx`。

## sha256（改竄検知用）

```
c1070fb357b6e2fefee9db81f39747d3618f71c01d94cbb55a7401bcdf651529  onnx_reference.json
6fb4f826fb56ee2656df48100d555c6500db94e93b84f125d180b3cff53373ff  onnx_weight_shapes.json
d0463734f568121f3eeda5f2d80e515141a3dd023a1dc57c454e233f7372dc1b  onnx_weights.json
bbd71d80b7525af2a77bce348a9d528bda9254476d7eab99b01eabdb8ef1824b  slice_repro_reference.json
```

`onnx_reference.json`・`onnx_weight_shapes.json`・`slice_repro_reference.json`
は PoC-v2-6 fixtures README（`docs/spec/03-poc/poc-v2-6-interop/code/
fixtures/README.md`）記載のコピー元と一致することを確認済み
（`sha256sum` 比較。`onnx_weights.json` は新規生成のため対応する
PoC-v2-6 側の値は存在しない）。
