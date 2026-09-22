# イシュー #2081 申し送り記録先（ONNX Model Zoo import 実証 example・parity 検証）

## 位置づけ

本 issue の成果（`docs/onnx-model-zoo-parity.md`・
`crates/onnx-interop/tests/model_zoo_parity.rs`・
`crates/onnx-interop/examples/model_zoo_probe.rs`・
`crates/facade/tests/interop_onnx_model_zoo.rs`）は CI で常時実行可能な
tier A（`mnist-12`。コミット済み）のハーネスに閉じている。以下は本 issue の
スコープ外として申し送る事項。

## CUDA／Metal 実機 parity: 構造的 N/A（実測不要）

`docs/onnx-gpu-execution-decision.md` に対する `crates/onnx-interop/tests/
model_zoo_parity.rs` の判定は **ONNX インタープリタのホスト CPU 実行のみ**
（`onnx::interp::run`。GPU dispatcher `run_with_ops` は使わない）を対象と
しており、REQ-2（バックエンド間数値一致）が要求する GPU 経路・実測
baseline が存在しない。したがって本 issue に関する限り CUDA／Metal 実機
実測は「未実測」ではなく「構造的に対象外」である。

ONNX import モデルを GPU 経由で実行する場合の実機 parity は別 issue
#2077（`onnx::interp::run_with_ops`。§6 契約 (c)）が管轄し、その実測記録は
`docs/perf/logs/onnx-gpu-execution-2077/README.md` にある（本 issue が
新たに追記すべき内容はない。Model Zoo モデルを #2077 の GPU dispatcher
経由で実行する組み合わせ検証は、#2077 が実機実測を終え、かつ Model Zoo
モデルが要求する `Conv` 等が実装された後に再検討する）。

## tier B（squeezenet1.0-12・mobilenetv2-12・resnet50-v1-12）の未検証 sha256

`crates/onnx-interop/tests/fixtures/model-zoo/README.md` の tier B 節に
記録した tar.gz LFS oid／`.onnx` sha256 は計画立案時点の値であり、本
実装セッションでは実ファイルのダウンロード再検証を行っていない
（103 MB の resnet50-v1-12 を含み容量都合で見送った）。tier B の
`#[ignore]` テストを初めて実行する開発者は、取得したファイルの sha256 を
必ず自分で再計算し、記録値と食い違えば同 README を実測値で更新すること
（fail-closed。該当箇所は `crates/onnx-interop/tests/fixtures/model-zoo/
README.md` に明記済み）。

## tier C: bertsquad-12 の取得・棚卸し未実施

`text/machine_comprehension/bert-squad/model/bertsquad-12`
（tar.gz 403,082,198 B）は容量都合で本 issue のスコープでは取得しなかった。
取得・op 棚卸しの手順:

```bash
curl -sS -o bertsquad-12.tar.gz \
  "https://media.githubusercontent.com/media/onnx/models/4f43949841cb55a0b98dc8fcd045431ccafd9f96/text/machine_comprehension/bert-squad/model/bertsquad-12.tar.gz"
sha256sum bertsquad-12.tar.gz   # 事前記録値なし。初回取得者が本 README へ追記すること
tar tzf bertsquad-12.tar.gz     # 展開前に <name>/ 配下に閉じていることを確認
tar xzf bertsquad-12.tar.gz -C "$ONNX_INTEROP_MODEL_ZOO_DIR"
cargo run -p fandhe-ai-onnx-interop --example model_zoo_probe -- \
  "$ONNX_INTEROP_MODEL_ZOO_DIR"/bertsquad-12
```

棚卸し後、`op_histogram` を `docs/onnx-model-zoo-parity.md` §1 の対象
モデル表・`crates/onnx-interop/tests/fixtures/model-zoo/README.md` へ追記し、
必要なら `tests/model_zoo_parity.rs` の `ZOO_MODELS` へ tier C エントリを
追加すること（`RunExpectation` は再プローブ結果に合わせて確定する）。

## PyTorch／ONNX Runtime を用いた三点突合（cross-check）

Model Zoo 同梱 `output_0.pb` の生成フレームワークはモデルカードに記載が
なく未確認のため、本 issue では「Model Zoo 公式参照値」とのみ呼び、
「ORT 生成」等と断定していない（`docs/onnx-model-zoo-parity.md` §3）。
PyTorch／ONNX Runtime を Python で再実行し三点（fandhe-ai・Model Zoo 参照
値・PyTorch or ORT 再実行）で突合するタスクは、依存ポリシー上
workspace 外の一時的な Python 環境を要するため本 issue のスコープ外
とし、実施する場合は別途承認を得て実施すること。

## 未対応 op の追跡（起票候補。ユーザー承認後に起票）

以下はスコープ外の実装対象候補として issue 化する余地があるが、本
issue のスコープでは起票していない（`.claude/rules/out-of-scope-tracking.md`
に従い、ユーザー承認後に着手すること）:

1. **`Dropout`（推論時 identity）の import 対応** — `squeezenet1.0-12` の
   e2e 実行に必須。既存 sibling（#2199／#2200／#2186）に含まれない
2. **`Conv` の `auto_pad`（`SAME_UPPER` 等）対応** — `mnist-12` の e2e
   実行に必須。#2199 の受け入れ条件は pads／strides／dilations のみで
   `auto_pad` には言及していないため、#2199 へのコメント追記候補
3. **group／depthwise conv** — `mobilenetv2-12` に必須。#2199 は Phase 2
   として明記済み

## sibling（#2199／#2200／#2186）マージ後の期待値反転

`Conv`・`MaxPool`・`BatchNormalization`・`GlobalAveragePool`・`Flatten`
等が実装され `run` が Model Zoo モデルに対してさらに先へ進んだ場合、
`crates/onnx-interop/tests/model_zoo_parity.rs::ZOO_MODELS` の該当
`RunExpectation` を更新すること（手順は `docs/onnx-model-zoo-parity.md`
§6）。中間状態で別のエラー（属性欠落・shape 不一致等）に変わった場合も
`is_err()` 等の緩い判定へ逃げず、新しい `RunExpectation` variant を
追加して完全一致で固定すること（catch-all variant は設けない）。
