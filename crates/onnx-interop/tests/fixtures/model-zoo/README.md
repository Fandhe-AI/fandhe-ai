# ONNX Model Zoo フィクスチャの出自（イシュー #2081・REQ-7）

`onnx/models`（ONNX Model Zoo。opset 12 系）が公開する第三者モデルによる
import 実証・parity 検証（`tests/model_zoo_parity.rs`・
`examples/model_zoo_probe.rs`・`crates/facade/tests/interop_onnx_model_zoo.rs`）
用の fixture。出自を commit SHA 固定で記録し、非コミット分は sha256 検証手順を
残す（`tests/fixtures/README.md` の `transformer.onnx` と同じ運用）。

## 出所（共通）

- リポジトリ: [`onnx/models`](https://github.com/onnx/models)
- commit: `4f43949841cb55a0b98dc8fcd045431ccafd9f96`（**必ずこの commit SHA 固定
  URL を使う**。`main` ブランチは可変で再現性が無いため使わない）
- 取得 URL パターン: `https://media.githubusercontent.com/media/onnx/models/4f43949841cb55a0b98dc8fcd045431ccafd9f96/<path>.tar.gz`
  （Git LFS 実体を返す raw エンドポイント。通常の `raw.githubusercontent.com`
  は LFS ポインタファイルしか返さないため使わない）
- ライセンス: リポジトリ全体 Apache-2.0。加えてモデル個別の SPDX 記載
  （下記各節）を確認する

## tier A（コミット済み）: `mnist-12`

`crates/onnx-interop/tests/fixtures/model-zoo/mnist-12/` に展開済みファイルを
コミットしている（26,143 B の `.onnx` + 3,157 B / 66 B の `test_data_set_0`。
tar.gz 自体〈26,741 B〉はコミットせず展開後のファイルのみをコミットする）。

- パス: `validated/vision/classification/mnist/model/mnist-12`
- tar.gz LFS oid（sha256）: `a53a59dcaca8804a0f6dfda9a3cf2e082979589391dbde73640b60684f1d24e9`（26,741 B）
- ライセンス: MIT（`onnx/models` 同 commit の
  `validated/vision/classification/mnist/README.md` の SPDX 記載）
- 展開後ファイルの sha256:
  - `mnist-12/mnist-12.onnx`: `5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd`（26,143 B）
  - `mnist-12/test_data_set_0/input_0.pb`: `d44b08082c3ded89e081f699a9d604239818c805ee8b5d03cd80f338e641c720`（3,157 B）
  - `mnist-12/test_data_set_0/output_0.pb`: `153a5b1d96f9a544fc398f8c1837b994bbf5f26d3d12a7eff2ec63f7fb2317e1`（66 B。
    Model Zoo 同梱の公式サンプル出力。生成フレームワークはモデルカード未記載の
    ため「参照値」とのみ呼び、「ORT 生成」等と断定しない）
- グラフ構造（本クレートの `build_graph` で実測。`examples/model_zoo_probe.rs`
  で再現可能）:
  - `opset_import`: `[("", 12)]`
  - node=12・initializer=8
  - input=`["Input3"]`（shape `[1,1,28,28]`）、output=`["Plus214_Output_0"]`
    （shape `[1,10]`）
  - op ヒストグラム: `Add×3・Conv×2・MatMul×1・MaxPool×2・Relu×2・Reshape×2`
  - 実行結果（HEAD。`decode → build_graph → run` 全経路）:
    `InterpError::UnsupportedOp("Conv")`（`Conv` は未対応 op。追跡先はイシュー
    #2199。ただし #2199 の受け入れ条件は pads／strides／dilations のみで
    `auto_pad`〈本モデルが使う `SAME_UPPER`〉には触れていないため、`auto_pad`
    対応は別途追跡が必要 — `docs/onnx-model-zoo-parity.md` §5 参照）

## tier B（非コミット。`ONNX_INTEROP_MODEL_ZOO_DIR` 経由の `#[ignore]` テスト）

以下 3 モデルはサイズが大きい（5〜103 MB）ため非コミットとし、
`tests/model_zoo_parity.rs` の `#[ignore]` テストが環境変数
`ONNX_INTEROP_MODEL_ZOO_DIR` 経由で参照する。

取得手順（各モデル共通）:

```bash
# <path> は下表の Model Zoo パス、<name> はディレクトリ名（例: squeezenet1.0-12）
curl -sS -o <name>.tar.gz \
  "https://media.githubusercontent.com/media/onnx/models/4f43949841cb55a0b98dc8fcd045431ccafd9f96/<path>.tar.gz"
sha256sum <name>.tar.gz   # 下表の tar.gz oid と一致することを確認
mkdir -p "$ONNX_INTEROP_MODEL_ZOO_DIR"
tar tzf <name>.tar.gz   # 展開前に <name>/ 配下に閉じていることを確認（path traversal 対策）
tar xzf <name>.tar.gz -C "$ONNX_INTEROP_MODEL_ZOO_DIR"
sha256sum "$ONNX_INTEROP_MODEL_ZOO_DIR"/<name>/<name>.onnx   # 下表の .onnx sha256 と照合
```

| モデル | Model Zoo パス | ライセンス | tar.gz LFS oid（sha256） | `.onnx` sha256 |
|---|---|---|---|---|
| `squeezenet1.0-12` | `validated/vision/classification/squeezenet/model/squeezenet1.0-12` | Apache-2.0 | `8a2dcc5a…0cb917`（詳細は取得後に自前で再計算し本 README を更新すること） | `dec81a86…992bd` |
| `mobilenetv2-12` | `validated/vision/classification/mobilenet/model/mobilenetv2-12` | Apache-2.0 | `5f83b422…37762d` | `c0c3f76d…32ad5` |
| `resnet50-v1-12` | `validated/vision/classification/resnet/model/resnet50-v1-12` | Apache-2.0 | `9391137c…599ea` | `3f03fdef…d1526` |

上記 tier B の oid／sha256 は計画立案時（Plan フェーズ）に記録された値であり、
本 PR の実装時点では実ファイルのダウンロード検証を行っていない（103 MB の
resnet50-v1-12 を含み容量都合で本セッションでは再取得しなかった）。**tier B
の `#[ignore]` テストを初めて実行する開発者は、取得したファイルの sha256 を
必ず自分で再計算し、この表と食い違えば実測値でこの README を更新すること**
（fail-closed。上記省略記法 `…` の値をそのまま信頼しない）。

グラフ構造の実測値（計画立案時点。Plan フェーズで `build_graph` まで到達
確認済み。`run` はいずれも HEAD で `UnsupportedOp("Conv")`）:

- `squeezenet1.0-12`: opset=12・node=66・initializer=53・
  input=`["data_0"]`（`[1,3,224,224]`）・output=`["softmaxout_1"]`
  （`[1,1000,1,1]`）・op ヒストグラム
  `Concat×8・Conv×26・Dropout×1・GlobalAveragePool×1・MaxPool×3・Relu×26・Softmax×1`
- `mobilenetv2-12`: opset=12・node=105・initializer=177・
  input=`["input"]`（`[batch_size,3,224,224]`）・output=`["output"]`
  （`[batch_size,1000]`）・op ヒストグラム
  `Add×10・Clip×35・Concat×1・Constant×1・Conv×52・Gather×1・Gemm×1・GlobalAveragePool×1・Reshape×1・Shape×1・Unsqueeze×1`
- `resnet50-v1-12`: opset=12・node=175・initializer=299・
  input=`["data"]`（`[N,3,224,224]`）・output=`["resnetv17_dense0_fwd"]`
  （`[N,1000]`）・op ヒストグラム
  `Add×16・BatchNormalization×53・Conv×53・Flatten×1・Gemm×1・GlobalAveragePool×1・MaxPool×1・Relu×49`。
  `test_data_set_0`〜`test_data_set_2` の 3 セットが同梱される

## tier C（doc 記録のみ。未取得）

- `bertsquad-12`（`text/machine_comprehension/bert-squad/model/bertsquad-12`。
  tar.gz 403,082,198 B）は容量都合で本 issue のスコープでは取得しない。
  取得手順・実行コマンドの申し送りは
  `docs/perf/logs/onnx-model-zoo-parity-2081/README.md` を参照

## 被覆台帳・判定基準

`docs/onnx-model-zoo-parity.md` を参照（本 README は fixture の出自・sha256
のみを記録し、判定基準・期待値表の正本ではない）。
