# ONNX Model Zoo からの import 実証・parity 検証 実施計画（イシュー #2081）

## 0. 位置づけ

`fandhe_ai_onnx_interop::onnx::interp`（`decode → build_graph → run`）と
facade `fandhe_ai::interop::onnx::OnnxModel` は、これまで自前生成 fixture
（`model.onnx`・`slice_repro.onnx`・非コミット `transformer.onnx`）に対して
のみ検証されており、第三者が公開する実モデルに対する到達性は未立証
だった。本 issue は [ONNX Model Zoo](https://github.com/onnx/models)
（`onnx/models`。opset 12 系）の公式モデルを用いた import・実行の到達性
検証ハーネスと被覆台帳を整備し、REQ-7（`docs/spec/04-requirements.md:
172-187`）の第三者 fixture による裏付けを提供する（`04-requirements.md:
345`「対応オペ範囲の制約」の HEAD 基準の実測更新根拠も兼ねる）。

**本 issue 実装時点（HEAD）の結論: 選定モデルは 1 件も end-to-end
実行できない。** 対象モデル（`mnist-12`・`squeezenet1.0-12`・
`mobilenetv2-12`・`resnet50-v1-12`）はいずれも `Conv`（未対応 op。追跡先
イシュー #2199）で `run` が止まる。よって本 issue の成果物は「ハーネス・
コミット済み基準データ・被覆台帳・fail-closed な期待値表」であり、
green parity（全要素一致の実測）ではない。未対応 op が実装され `run` が
先へ進んだ場合の反転手順は §6 に記す。

## 1. 対象モデル

| モデル | Model Zoo パス | opset | node | initializer | 入力 shape | 出力 shape |
|---|---|---|---|---|---|---|
| `mnist-12`（コミット済み） | `validated/vision/classification/mnist/model/mnist-12` | 12 | 12 | 8 | `Input3` `[1,1,28,28]` | `Plus214_Output_0` `[1,10]` |
| `squeezenet1.0-12`（非コミット） | `validated/vision/classification/squeezenet/model/squeezenet1.0-12` | 12 | 66 | 53 | `data_0` `[1,3,224,224]` | `softmaxout_1` `[1,1000,1,1]` |
| `mobilenetv2-12`（非コミット） | `validated/vision/classification/mobilenet/model/mobilenetv2-12` | 12 | 105 | 177 | `input` `[batch_size,3,224,224]` | `output` `[batch_size,1000]` |
| `resnet50-v1-12`（非コミット） | `validated/vision/classification/resnet/model/resnet50-v1-12` | 12 | 175 | 299 | `data` `[N,3,224,224]` | `resnetv17_dense0_fwd` `[N,1000]` |

op ヒストグラム・出自・sha256・ライセンス・取得手順の正本は
`crates/onnx-interop/tests/fixtures/model-zoo/README.md`（本 doc から
二重管理しない）。`bertsquad-12`（403 MB）は tier C として容量都合で
本 issue では取得しなかった（申し送りは
`docs/perf/logs/onnx-model-zoo-parity-2081/README.md`）。

### 前処理・入力データの標準化

- `mnist-12`: 28×28 グレースケール・白地黒背景・値域 `[0,1]`・batch 固定 1
- `squeezenet1.0-12`／`mobilenetv2-12`／`resnet50-v1-12`: ImageNet 224×224
  center crop・`[0,1]` 正規化後 `mean=[0.485,0.456,0.406]`・
  `std=[0.229,0.224,0.225]`・HWC→CHW（`resnet50-v1-12` は 256 リサイズ後
  crop。各モデルカード準拠の記録であり、本 issue では自前実装しない）

**本 issue では前処理を自前実装しない。** Model Zoo 同梱
`test_data_set_0/input_0.pb` を「標準化済み入力」としてそのまま feed する
（`tests/model_zoo_parity.rs`・`examples/model_zoo_probe.rs` の共通方針）。
上記は再現手順の記録（モデルカード準拠）であり、PyTorch／ONNX Runtime を
Python で再実行する前処理の cross-check は本 issue のスコープ外
（§8・申し送り先は上記 perf/logs README）。

## 2. fixture 方針

- **tier A（コミット済み）**: `mnist-12` のみ。`crates/onnx-interop/tests/
  fixtures/model-zoo/mnist-12/`（`.onnx` 26,143 B・`test_data_set_0/
  {input_0.pb, output_0.pb}` 3,157 B / 66 B）
- **tier B（非コミット。`ONNX_INTEROP_MODEL_ZOO_DIR` 経由の `#[ignore]`
  テスト）**: `squeezenet1.0-12`・`mobilenetv2-12`・`resnet50-v1-12`
- **tier C（doc 記録のみ）**: `bertsquad-12`

`sha2` 等の checksum クレートは許容依存外（deps-policy.md）のため、
`tests/model_zoo_parity.rs` はファイルバイト長の完全一致を無依存の
最小限の整合性ガードとして使う
（`mnist12_fixture_byte_lengths_match_recorded_sizes`）。sha256 の正は
`crates/onnx-interop/tests/fixtures/model-zoo/README.md` に記録し、
取得時に人手（`sha256sum`）で検証する。

## 3. 参照値の出自

`test_data_set_0/output_0.pb`（tier A・B とも同梱）は Model Zoo が公式に
配布する参照出力である。**生成フレームワークはモデルカードに記載が
なく未確認のため、本 doc では「Model Zoo 公式参照値」とのみ呼び、
「ONNX Runtime 生成」等と断定しない。** PyTorch／ONNX Runtime による
三点突合（cross-check）は本 issue のスコープ外（§8）。

## 4. 判定基準

### REQ-7 事前固定式（`Parity` 判定）

`abs_err / (|ref| + 1e-6) <= 1e-3`（`crates/onnx-interop/tests/
onnx_interp.rs` 等と同一の事前固定基準）。`run` が全経路を通過し出力を
得られたモデルに対してのみ適用する。

### REQ-2 との混同禁止（構造的 N/A）

`.claude/rules/coding-rust.md` の REQ-2 バックエンド間数値一致 OR 複合
判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）は **本 issue の
検証対象外**である。`onnx::interp::run`（本 issue が使う経路）はホスト
CPU 実行のみで GPU 経路・実測 baseline が存在しないため、REQ-2 は構造的
N/A である（GPU dispatcher `run_with_ops` を経由する parity は別 issue
#2077 が管轄。`docs/onnx-gpu-execution-decision.md`）。この「記入」を
もって受け入れ条件 5（判定基準の doc への記入）を満たす。

### 将来の判定不能ケース

未対応 op が実装され Model Zoo モデルが `Parity` 判定まで到達した際、
softmax の裾（〜1e-8）等で REQ-7 式が Model Zoo 参照値と合わない
可能性がある。その場合は「（比較データ側の）判定不能」として記録し、
tolerance 自体は変更せず、必要なら spec リポへの提案（ユーザー承認後）
で扱う。

## 5. 被覆台帳

| op | 対応状況（HEAD） | 追跡先 |
|---|---|---|
| `Conv` | 未対応（pads／strides／dilations 対応予定） | #2199（`auto_pad`〈`mnist-12` が使う `SAME_UPPER`〉は受け入れ条件に含まれず別追跡が必要） |
| `MaxPool` | 未対応 | #2199 |
| `BatchNormalization` | 未対応 | #2200 |
| `Flatten` | 未対応 | #2200 |
| `GlobalAveragePool` | 未対応 | #2200 |
| `Clip`（min/max 入力形） | 未対応 | #2186 |
| `Dropout`（推論時 identity） | 未対応 | 追跡先なし（起票候補。`docs/perf/logs/onnx-model-zoo-parity-2081/README.md` §「未対応 op の追跡」） |
| group／depthwise conv（`Conv` の `group` 属性） | 未対応 | #2199（Phase 2 と明記されており本 issue 選定モデルの `mobilenetv2-12` に必須） |
| `auto_pad`（`SAME_UPPER` 等） | 未対応 | 追跡先なし（起票候補。同上） |

## 6. 期待値反転手順

`crates/onnx-interop/tests/model_zoo_parity.rs::ZOO_MODELS` の各エントリは
`RunExpectation`（`Parity` または `UnsupportedOp(op)`）を持つ
fail-closed な期待値表である。

- **sibling（#2199／#2200／#2186 等）が先にマージされた場合**: そちらの
  PR 側で対象モデルの `run` が先へ進んだ時点（＝別の `UnsupportedOp` で
  止まる、または成功する）で `RunExpectation` を更新する義務を負う
- **本 issue の PR が後にマージされる場合**: マージ直前（実際の作業は
  git 履歴を参照）の再プローブ（`cargo run -p
  fandhe-ai-onnx-interop --example model_zoo_probe -- <モデルディレクトリ>`）
  で HEAD の実際の挙動を確認してから期待値表を確定する
- **中間状態の扱い**: 例えば #2199 が `auto_pad` 非対応のままマージ
  されると、`mnist-12` の `run` は `UnsupportedOp("Conv")` ではなく
  属性欠落・shape 不一致等の別エラーで止まりうる。この場合も
  `is_err()` のような緩い判定へ逃げず、当該エラーの variant／メッセージ
  を完全一致で期待する新しい `RunExpectation` variant を追加する
  （catch-all variant は設けない）
- **成功に転じた場合**: `RunExpectation::Parity` へ変更し、
  `output_0.pb`（tier A）または `ONNX_INTEROP_MODEL_ZOO_DIR` 配下の
  `output_0.pb`（tier B）との REQ-7 全要素一致を要求する
  （`crates/facade/tests/interop_onnx_model_zoo.rs` 側は facade 出力と
  内部クレート出力の bit 同一検査へ切替える）

## 7. 実行手順

```bash
# 常時実行（tier A・CI 対象）
cargo test -p fandhe-ai-onnx-interop --test model_zoo_parity
cargo test -p fandhe-ai --test interop_onnx_model_zoo

# import 実証 example（tier A）
cargo run -p fandhe-ai-onnx-interop --example model_zoo_probe -- \
  crates/onnx-interop/tests/fixtures/model-zoo/mnist-12

# tier B（非コミット。取得手順は tests/fixtures/model-zoo/README.md 参照）
ONNX_INTEROP_MODEL_ZOO_DIR=<展開先ディレクトリ> \
  cargo test -p fandhe-ai-onnx-interop --test model_zoo_parity -- --ignored --nocapture

# workspace 全体（make test-ignored 相当。ONNX_INTEROP_MODEL_ZOO_DIR 未設定なら
# tier B は早期 return でスキップし非破壊）
cargo test --workspace --all-features -- --ignored
```

CI は本リポにコミットされた `mnist-12`（tier A）のテストのみ常時実行する
（`.claude/rules/ci.md` の実機依存分離と同じ運用。tier B／C は
`#[ignore]` 分離・環境変数ゲート）。

## 8. スコープ外・申し送り・承認事項

以下は実施しない（列挙のみ）。詳細な申し送りは
`docs/perf/logs/onnx-model-zoo-parity-2081/README.md` を参照する:

- CUDA／Metal 実機 parity: ONNX インタープリタはホスト CPU 実行のみの
  ため構造的 N/A（#2077 が実装され GPU dispatcher 経由の実行が可能に
  なった時点で再検討）
- `bertsquad-12`（tier C）の取得・op 棚卸し（容量都合で未実施）
- PyTorch／ONNX Runtime を用いた三点突合（cross-check）
- 未対応 op（`Dropout`・group conv・`auto_pad`）の新規 issue 起票
  （起票候補として記録するがユーザー承認後に実施する）
- spec 変更提案（REQ-7 式が softmax 裾で Model Zoo 参照出力と合わない
  場合の扱い。発生時に (b) 形式で提案しユーザー承認後に投稿）
- 依存追加なし・新規 `unsafe` なし・facade 公開面拡張なし（`OnnxModel`
  へのアクセサ追加等は行わない）・ガードレール閾値・テスト許容誤差の
  変更なし・`docs/spec/`（正本 submodule）への変更なし
