# fandhe-ai-onnx-interop

[fandhe-ai](https://github.com/Fandhe-AI/fandhe-ai) の ONNX（手書き prost
デコード・グラフ解釈器・export）／safetensors 相互運用（REQ-7）を担う
内部クレートです。

**このクレートへの直接依存・直接利用はサポート対象外です。** 唯一のサポート
される公開 API 面は
[`fandhe-ai`](https://crates.io/crates/fandhe-ai)
（composition root・compat 公開面）であり、本クレートはその内部実装として
crates.io にも公開されています（依存解決のため）。

ONNX import は `fandhe-ai::interop::onnx`（`OnnxModel`／`OnnxValue`／
`OnnxError`）として `fandhe-ai` から公開済みです（イシュー #2017）。
safetensors save／load も `fandhe-ai::interop::safetensors`
（`LoadError`／`SaveError`／`load_safetensors_f32`／
`load_safetensors_f32_from_bytes`／`require_keys`／
`save_safetensors_f32`／`save_safetensors_f32_to_bytes`）として
`fandhe-ai` から公開済みです（イシュー #2019）。ONNX export への
再エクスポートは引き続き提供されていません（facade 公開面の拡張は
別途ユーザー承認が必要な段階 0 のまま。
[`docs/facade-onnx-export-exposure-decision.md`](https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/facade-onnx-export-exposure-decision.md)
参照）。

本クレートは ONNX（protobuf）・safetensors という外部フォーマットの
パーサーを含みます。信頼できない入力のパースは長さ・形状検証を先行させる
方針で実装していますが、上記のとおり直接利用はサポート対象外です。

## 利用方法

```toml
[dependencies]
fandhe-ai = "0.9.0"
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
