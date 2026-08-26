# fandhe-ai-backend-metal

[fandhe-ai](https://github.com/Fandhe-AI/fandhe-ai) の Metal バックエンド
（`objc2` 系直接バインディング。macOS 限定）を担う内部クレートです。`fandhe-ai`
から `Device::Metal` 指定時に実行時解決されます。

**このクレートへの直接依存・直接利用はサポート対象外です。** 唯一のサポート
される公開 API 面は
[`fandhe-ai`](https://crates.io/crates/fandhe-ai)
（composition root・compat 公開面）であり、本クレートはその内部実装として
crates.io にも公開されています（依存解決のため）。

## 利用方法

```toml
[dependencies]
fandhe-ai = "0.3.0"
```

開発版を試す場合は Git 依存でも参照できます。

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/fandhe-ai" }
```

インストール・最小コード例は
[`fandhe-ai` の README](https://github.com/Fandhe-AI/fandhe-ai/blob/main/README.md#最小コード例)
を参照してください。バックエンド切替（`Device::Metal` 指定）の例は同 README の
「バックエンド」節を参照してください。

## ドキュメント・リポジトリ

利用者向けドキュメントサイト（GitHub Pages）: https://fandhe-ai.github.io/fandhe-ai/（Getting Started / Guides / Examples / API Reference）。API リファレンスは https://docs.rs/fandhe-ai

- ソース: <https://github.com/Fandhe-AI/fandhe-ai/tree/main/crates/backend-metal>
- Metal 実装方式（`wgpu` 不採用判断）の詳細: [`docs/backend-metal-wgpu-decision.md`](https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/backend-metal-wgpu-decision.md)

## ライセンス

MIT または Apache License 2.0（デュアルライセンス）。
[LICENSE-MIT](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-MIT) ／
[LICENSE-APACHE](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-APACHE)
