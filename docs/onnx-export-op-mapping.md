# ONNX export op マッピング（イシュー #1773）

`onnx-interop` 内部（`crate::onnx::export_ops`）の export 側 op マッピング
（`ExportOp` -> `NodeProto`）の対応表・契約を記録する。**本モジュール自体**
（`ExportOp`／`ExportNode`／`to_node_proto` 等）は facade へ一切公開しない
（`crates/facade/tests/api_surface.rs` が `ExportError`／`onnx::export::` の
`pub` シグネチャへの露出を機械検査で固定）。ただし、本モジュールが担う
allowlist fail-closed 検査（`check_exportable`）は #2018 で facade
`OnnxModel::to_bytes`／`to_path`（`export::build_model_proto` 経由）から
間接的に実行される（import 済みモデルの roundtrip export に限定。
`compat::Sequential`／`nn` -> `ExportNode` 橋渡しは対象外のまま。
`docs/compat-api-scope.md` §1.3・`docs/facade-onnx-export-exposure-
decision.md` §14）。

## 0. スコープ境界

- 本 issue の対象は「`interp.rs` が読む 23 op の逆方向（内部 op -> `NodeProto`）」
  のみ。autodiff `Op`／`Tape` -> `ExportOp` の橋渡し（`compat::Sequential`／
  `nn` -> `ExportNode`）は #1653 のスコープのまま対象外（#2018 でも実装しない）。
  facade 公開可否自体は #1775 が判断済みで、#2018 で
  `OnnxModel::to_bytes`／`to_path`（import 済みモデルの roundtrip export
  ラッパー限定）として実装済み（`docs/facade-onnx-export-exposure-
  decision.md` §14）。
- import -> export -> import の総合 roundtrip・未対応 op を含むモデルの
  fail-closed 確認という総合テストは #1774 で実装済み（§6 参照）。本ドキュメント・
  実装の単体テストは op 単位の対称性検証に限定する。

## 1. 契約

- **属性は既定値であっても常に全て書き出す**（省略すると「属性欠落＝既定値」
  という対称性テストが空虚に pass してしまうため）。唯一の例外は
  `ExportOp::Transpose::perm`: `None`（省略時の意味論＝rank 依存の軸反転）を
  静的な既定値で埋めず、属性自体を省略する。
- `NodeProto.domain` は常に空文字列（既定 opset）。
- 出力は全 op 単一（`interp.rs::require_single_output` と対称）。
- opset 形態は `ExportOptions::default().opset_version=17` に合わせ、
  `Reshape.shape`・`Squeeze`/`Unsqueeze.axes`・`Slice.starts/ends/axes/steps`
  は入力テンソル名として表す（attr 形〈opset<13〉は export しない）。
- `check_exportable`（層 B）は `build_model_proto` の先頭で必ず呼ばれ、
  `op_type` が allowlist（`SUPPORTED_OP_TYPES`）に含まれ、かつ `domain` が
  空文字列であることを検査する。違反は `ExportError::UnsupportedOp` で
  fail-closed に拒否する（無言 skip はしない）。

## 2. 対応表

| op_type | Tier | 入力（`[]` は省略可） | 属性 name: 型 = 既定値 | `interp.rs` 読み取り箇所 |
|---------|------|------|------|------|
| Gemm | 1 | A, B, [C] | alpha: FLOAT=1.0 / beta: FLOAT=1.0 / transA: INT=0 / transB: INT=0 | `compute_gemm` |
| MatMul | 1 | A, B | なし | `compute_matmul` |
| Add | 1 | A, B | なし | `compute_add` |
| Mul | 1 | A, B | なし | `compute_mul` |
| Div | 1 | A, B | なし | `compute_div` |
| Relu | 1 | X | なし | `compute_relu` |
| Sigmoid | 1 | X | なし | `compute_sigmoid` |
| Softmax | 1 | X | axis: INT=-1 | `compute_softmax` |
| Reshape | 1 | data, shape（入力） | allowzero: INT=0 | `compute_reshape` |
| Mod | 3 | A, B | fmod: INT=0 | `compute_mod` |
| Sqrt | 3 | X | なし | `compute_sqrt` |
| Erf | 3 | X | なし | `compute_erf` |
| Shape | 2 | data | なし | `compute_shape` |
| Gather | 2 | data, indices | axis: INT=0 | `compute_gather` |
| Unsqueeze | 2 | data, axes（入力・opset>=13 形） | なし | `compute_unsqueeze` |
| Squeeze | 3 | data, [axes]（入力・opset>=13 形） | なし | `compute_squeeze` |
| Concat | 2 | inputs...（1 個以上・可変長） | axis: INT（必須） | `compute_concat` |
| Slice | 2 | data, starts, ends, [axes], [steps]（全入力・opset>=13 形） | なし | `compute_slice` |
| Transpose | 3 | data | perm: INTS（省略可。省略時は rank 依存の軸反転） | `compute_transpose` |
| Cast | 3 | input | to: INT（1/7/9/10 のみ。export 側は値検査しない） | `compute_cast` |
| Constant | 3 | なし | value〈TENSOR〉／value_float〈FLOAT〉／value_floats〈FLOATS〉／value_int〈INT〉／value_ints〈INTS〉のいずれか 1 つ | `compute_constant` |
| LayerNormalization | 3 | X, Scale, [B] | axis: INT=-1 / epsilon: FLOAT=1e-5 | `compute_layer_normalization` |
| Conv | 3 | X, W, [B] | auto_pad: STRING="NOTSET" / dilations: INTS=[1,1] / group: INT=1 / kernel_shape: INTS（`W.shape()[2..4]`） / pads: INTS=[0,0,0,0] / strides: INTS=[1,1] | `compute_conv` |

Tier 1（必須・受入条件）・Tier 2（`slice_repro.onnx` fixture roundtrip に必要）・
Tier 3（推奨・全実装済み）。23 op すべて実装済み（削減なし。`Conv` はイシュー
#2076・親 #2034 で追加）。

## 3. `Constant` の `value`（TENSOR）契約

`ConstantAttr::Tensor(RawTensor)` を書き出す際、埋め込む `TensorProto.name` は
ノードの出力名（`node.outputs[0]`）を決定的に用いる。`TensorProto.name` は
import 側（`decode_tensor`）がエラーメッセージにしか使わないため、この選択に
実害はない。

## 4. arity 検査（層 A）

必須入力の個数・非空文字列、省略可入力は末尾のみ・空文字列可、出力は全 op
厳密 1（`interp.rs::require_single_output` と対称）。違反は
`ExportError::InputArityMismatch`／`OutputArityMismatch`／
`EmptyRequiredInput` で拒否する。

「最小個数」判定と「必須入力の空文字列拒否」判定は別軸である
（`check_arity` の `min_inputs`／`all_variadic_required` 引数）。`Slice` は
`ends`（第 3 入力）が必須のため `min_inputs=3`（`data, starts, ends` の 3 個
未満は arity 違反・`axes`/`steps` の 2 個は省略可入力として空文字列を許容）。
`Concat` は可変長入力の全要素が必須（ONNX 仕様上どの要素も省略できない）
のため、`min_inputs=1`（1 個以上の arity 検査）とは独立に
`all_variadic_required=true` を渡し、実際に渡された全入力位置の空文字列を
拒否する。

## 5. 対象外事項

- autodiff `Op`／`Tape`／`Sequential` -> `ExportOp` の橋渡し（#1653。橋渡しの
  配置候補は `docs/facade-onnx-export-exposure-decision.md` §3.2。#2018 の
  対象外だったが #2036 で `onnx::export_nn`〈本クレート内部限定・facade
  未接続〉として実装済み。§7 参照）。本モジュール自体（`ExportOp`／
  `ExportNode`）の facade 再エクスポートは対象外のまま（`OnnxModel::
  to_bytes`／`to_path` という薄いラッパー経由の間接実行のみ。#1775 が
  判断済み・#2018 で実装）
- `value_info`／`TypeProto` 非出力による外部ツール（`onnx.checker`）妥当性
  （#1772 既知事項）
- opset<13 の attr 形（`Squeeze`/`Unsqueeze` の `axes` 属性）での export・
  `Shape` の `start`/`end`・`LayerNormalization` の `stash_type`・`Constant`
  の `value_string(s)`／sparse（import 側も不参照）

## 6. roundtrip テスト（#1774）

import -> export -> import の総合 roundtrip（`decode -> build_graph ->
build_model_proto -> encode -> decode -> build_graph`）の構造一致・数値一致・
未対応 op の fail-closed 確認は
`crates/onnx-interop/tests/onnx_export_roundtrip.rs` に実装済み。既存の
`tests/onnx_export.rs`（#1772・組み立て自体の単体テスト）・
`tests/onnx_export_ops.rs`（#1773・op 単位の対称性・`check_exportable` の
allowlist／domain 検査）とは別ファイルで、以下を固定する:

- **fixture roundtrip の構造一致（bit 同一）**: `model.onnx`・`slice_repro.onnx`
  それぞれについて、roundtrip 後の `Graph` がフィールド単位・bit 同一
  （`to_bits()` 一致。属性・initializer 込み）で元の `Graph` と一致すること。
  比較が空虚でないことを fixture README の既知構造（node 数・op_type 列・
  initializer 名と shape／値）を直接 assert して担保する。
- **export の不動点性**: `export(G)` のバイト列と `export(import(export(G)))`
  のバイト列が完全一致すること（`export.rs` の initializer 名ソート契約から
  導かれる決定的な性質）。
- **typed data -> `raw_data` 正規化の bit 保存**: `float_data`／`int64_data`
  ベースの手組み initializer（NaN・-0.0・非正規化数込み）を import した結果と、
  export -> 再 import した結果が bit 完全一致すること。
- **`interp::run` の bit 同一**: 同一 feed に対し、import 元の `Graph` と
  export 後に再 import した `Graph` の `run` 結果が bit 完全一致すること
  （`model.onnx`・`slice_repro.onnx` の全参照入力・`transformer.onnx` の
  1 入力）。
- **未対応 op を含むモデルの fail-closed**: allowlist 外の op_type・既定
  opset 以外の domain を含む decode 経由のモデルが `build_model_proto` で
  `ExportError::UnsupportedOp` を返すこと（node_name／op_type／domain の
  完全一致）。複数違反がある場合はトポロジカル順で最初のノードが報告される
  ことも固定する。

`transformer.onnx`（12MB・非コミット）を使うテストは `#[ignore]` で分離し、
`tests/fixtures/README.md` と同じ運用（`ONNX_INTEROP_TRANSFORMER_ONNX` 環境
変数でパス指定）で実行する:

```bash
ONNX_INTEROP_TRANSFORMER_ONNX=<path> \
  cargo test -p onnx-interop --test onnx_export_roundtrip -- --ignored --nocapture
```

比較はすべて **bit 同一**の別軸契約であり、REQ-2 バックエンド間数値一致複合
判定・REQ-7 事前固定判定式（`abs_err / (|ref| + 1e-6) <= 1e-3`）とは混同しない
（tolerance は導入も変更もしていない）。本モジュール自体への facade 新規
公開面はなし（#2018 の facade 公開面は `OnnxModel::to_bytes`／`to_path`・
`OnnxExportOptions` の 3 件のみで、`ExportOp`／`ExportNode` 等は含まない）。

## 7. `nn::Module` 層列 -> `ExportOp`／`Graph` の橋渡し（`export_nn`。#2036）

`docs/facade-onnx-export-exposure-decision.md` §15 の設計確定を受け、
`onnx::export_nn` モジュール（`crates/onnx-interop/src/onnx/export_nn.rs`）
が `fandhe_ai_autodiff::nn::Module` の層列（`&[Box<dyn Module>]`）から
[`export_parts_from_layers`]／[`graph_from_layers`] で本クレートの
`Graph`（本ファイル §1〜§6 が担う `build_model_proto` の入力形）を組み立てる。
本モジュールは `super::export`／`super::export_ops` の**利用者**であり
（`ExportOp::Gemm`／`ExportOp::Relu` を構築して `to_node_proto` へ渡す）、
op マッピング表自体（§2）は変更しない。

- **対応層（イシュー #2076・親 #2034 で Softmax・LayerNorm・
  GELU（erf 版）・Conv2d へ拡大。`Sigmoid` は §15.7 項 5〈数値契約〉が
  承認保留のため対象外のまま——`Module::as_sigmoid` フックは追加して
  いない）**: `Module::as_linear()` が `Some` →
  `ExportOp::Gemm(GemmAttrs { alpha: 1.0, beta: 1.0, trans_a: false,
  trans_b: false })`（weight `[in, out]` のまま・転置しない）。
  `Module::as_relu()` が `true` → `ExportOp::Relu`。`Module::as_softmax()` が `Some` →
  `ExportOp::Softmax { axis: dim as i64 }`（`Softmax::dim()`。`pub` へ
  変更済み）。`Module::as_layer_norm()` が `Some` → `ExportOp::
  LayerNormalization(LayerNormAttrs { axis: -1, epsilon: eps() })`
  （`weight() == None`〈`without_affine`〉は ONNX `LayerNormalization` の
  `Scale` 必須制約により `ExportError::InvalidLayerParameter` で拒否）。
  `Module::as_gelu()` が `true` → `Mul(x, 1/√2) -> Erf -> Add(1.0) ->
  Mul(x, ·) -> Mul(0.5)` の 5 演算ノード（+ `Constant` 3 ノード。ONNX に
  GELU 単体演算が opset 17 に無いための合成。`GeluTanh` は対象外）。
  `Module::as_conv2d()` が `Some` → `ExportOp::Conv(ConvAttrs)`
  （`kernel_shape = weight.shape()[2..4]`／`strides`／`dilations`／
  `group`／`pads = [ph, pw, ph, pw]`〈対称〉／`auto_pad = "NOTSET"`）。
  それ以外は `ExportError::UnsupportedLayer { index, layer_kind }` で
  fail-closed に拒否する（`layer_kind` は既存 8 フック〈`as_conv2d` は
  対応層化したため報告対象から除く・`as_conv1d`／`as_rms_norm`／
  `as_batch_norm1d`／`as_batch_norm2d`／`as_embedding`／
  `as_multihead_attention`〉で判別できる範囲のみ具体名を報告し、
  それ以外は `"unknown"`）。
- **`autodiff` API バージョン制約について**: `crates/onnx-interop/
  Cargo.toml` の `fandhe-ai-autodiff = { version = "=0.9.0", .. }`
  （通常依存。§15.7 承認事項 1 を適用済み）は単一クレート
  `cargo publish --dry-run` 時に registry の `fandhe-ai-autodiff
  =0.9.0` でビルド検証するが、`docs/facade-onnx-export-exposure-
  decision.md` §15.2 項 5・§17.3 が整理するとおり単一クレート dry-run
  の失敗は既知の非ブロッカーであり、7 パッケージ一括 dry-run
  （`docs/crates-io-publishing-order.md` §8.1）はローカル解決（同一
  workspace 内の未公開バージョンを解決する）で成立する。本 issue で
  追加した `Module::as_gelu`／`as_softmax` は crates.io
  未公開の新 API のため単一クレート dry-run では解決できないが、上記
  整理により issue のブロッカーにはならない。
- **名前規約**: graph input `"input"`／output `"output"`・中間テンソル
  `layer{i}_out`・ノード名 `layer{i}`・initializer 名 `{i}.weight`／
  `{i}.bias`（`nn::Sequential::named_parameters()`／`state_dict()` の
  `"{index}.{name}"` 契約と同一形式）。
- **検証順序**: `ExportNode`／`RawTensor` を 1 つも構築する前に全層を
  検証する（空層列 → `ExportError::EmptyModel`・shape 不整合 →
  `ExportError::InvalidLayerParameter`・テンソル名重複 →
  `ExportError::DuplicateTensorName`。いずれも本 issue で `ExportError`
  へ追加した variant）。
- **正しさの検証**: `crates/onnx-interop/tests/onnx_export_nn.rs` が
  2 層 MLP（`Linear -> Relu -> Linear`。ブロックタイル境界〈KC=256〉を
  跨ぐ大形状を含む）の export → `interp::run` 出力を、
  `Module::forward_host`（tape 不要経路）・`Module::forward`（tape 経路）
  という 2 通りの独立な手動 forward と bit 完全一致で突き合わせて検証する
  （前提: GEMM 出力に厳密な `±0.0` が現れない入力。`export_nn.rs` モジュール
  冒頭ドキュメント参照）。
- **facade 結線**: 本モジュール自体（`export_parts_from_layers`／
  `graph_from_layers`／`NnExportParts`）は非公開クレート `onnx-interop`
  の内部限定のまま facade へは再エクスポートしない。facade
  `OnnxModel::from_sequential(&Sequential)`（§15.7 承認事項 3）は
  イシュー #2037 で `graph_from_layers` への薄い委譲として実装済み
  （`docs/facade-onnx-export-exposure-decision.md` §17）。
