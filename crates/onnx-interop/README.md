# fandhe-ai-onnx-interop

[fandhe-ai](https://github.com/Fandhe-AI/fandhe-ai) の ONNX（手書き prost
デコード・グラフ解釈器・export）／safetensors 相互運用（REQ-7）を担う
内部クレートです。

**このクレートへの直接依存・直接利用はサポート対象外です。** 唯一のサポート
される公開 API 面は
[`fandhe-ai`](https://crates.io/crates/fandhe-ai)
（composition root・compat 公開面）であり、本クレートはその内部実装として
crates.io にも公開されています（依存解決のため）。

ONNX import／export は `fandhe_ai::interop::onnx`（`OnnxModel`／
`OnnxValue`／`OnnxError`・`OnnxModel::{from_bytes, from_path, run,
to_bytes, to_path, from_sequential}`・`OnnxExportOptions`）として
`fandhe-ai` から公開済みです（import はイシュー #2017・export はイシュー
#2018）。export には、import したモデルの書き出しと、`compat::Sequential`
からの書き出し（`from_sequential` でモデルを作り、`to_bytes`／`to_path` で
書き出す。対応層は 7 種）の 2 つがあります。export した ONNX は、中間値の `value_info` が空で、グラフの入出力の `ValueInfoProto` にも名前だけで `TypeProto`（型・形状）を持たないため、`onnx.checker` 等の外部ツールによる厳密な妥当性検証は保証しません。また `OnnxExportOptions` の opset 指定は宣言値の設定だけでバージョン変換は行わず、ノードの入力・属性はそのまま複製します（例: `axes` 属性を持つ opset 12 の `Unsqueeze` を既定設定で書き出すと、opset 17 宣言と旧形式ノードが混在します）。指定先の opset の仕様に既存ノードが適合している必要があります。import したモデルの再エクスポートは、元の `opset_import`／`ir_version` を保持せず `OnnxExportOptions`（既定 `opset_version=17`）の値で出力します。safetensors save／load も
`fandhe_ai::interop::safetensors`（`LoadError`／`SaveError`／
`load_safetensors_f32`／`load_safetensors_f32_from_bytes`／
`require_keys`／`save_safetensors_f32`／`save_safetensors_f32_to_bytes`）
として `fandhe-ai` から公開済みです（イシュー #2019）。dtype 混在ファイルの読込
（`SafetensorsTensor`／`SafetensorsDtype`／`load_safetensors_typed`／
`load_safetensors_typed_from_bytes`／`require_keys_typed`）は
イシュー #3845、書出側（`save_safetensors_typed`／`save_safetensors_typed_to_bytes`）は
イシュー #3846 で追加しました。

本クレートは ONNX（protobuf）・safetensors という外部フォーマットの
パーサーを含みます。信頼できない入力のパースは長さ・形状検証を先行させる
方針で実装していますが、上記のとおり直接利用はサポート対象外です。

## 利用方法

```toml
[dependencies]
fandhe-ai = "0.15.0"
```

開発版を試す場合は Git 依存でも参照できます。

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/fandhe-ai" }
```

インストール・最小コード例は
[`fandhe-ai` の README](https://github.com/Fandhe-AI/fandhe-ai/blob/main/README.md#最小コード例)
を参照してください。

## ドキュメント・リポジトリ

利用者向けドキュメントサイト（GitHub Pages）: https://fandhe-ai.github.io/fandhe-ai/（Getting Started / Guides / Examples / API Reference）。API リファレンスは https://docs.rs/fandhe-ai

- ソース: <https://github.com/Fandhe-AI/fandhe-ai/tree/main/crates/onnx-interop>
- サポート境界の詳細: [`docs/compat-api-scope.md`](https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/compat-api-scope.md)

## ライセンス

MIT または Apache License 2.0（デュアルライセンス）。
[LICENSE-MIT](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-MIT) ／
[LICENSE-APACHE](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-APACHE)
