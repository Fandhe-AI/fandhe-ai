# fandhe-ai-tensor-core

[rust-ai-library](https://github.com/Fandhe-AI/rust-ai-library) のテンソル型・
演算グラフ／カーネル融合機構を担う完全自作コアの内部クレートです。

**このクレートへの直接依存・直接利用はサポート対象外です。** 唯一のサポート
される公開 API 面は
[`fandhe-ai`](https://crates.io/crates/fandhe-ai)
（composition root・compat 公開面）であり、本クレートはその内部実装として
crates.io にも公開されています（依存解決のため）。

## 利用方法

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/rust-ai-library" }
```

インストール・最小コード例は
[Getting Started](https://fandhe-ai.github.io/rust-ai-library/getting-started/)
を参照してください。

## ドキュメント・リポジトリ

- [Getting Started](https://fandhe-ai.github.io/rust-ai-library/getting-started/)・[API Reference](https://fandhe-ai.github.io/rust-ai-library/api/)
- ソース: <https://github.com/Fandhe-AI/rust-ai-library/tree/main/crates/tensor-core>
- サポート境界の詳細: [`docs/compat-api-scope.md`](https://github.com/Fandhe-AI/rust-ai-library/blob/main/docs/compat-api-scope.md)

## ライセンス

MIT または Apache License 2.0（デュアルライセンス）。
[LICENSE-MIT](https://github.com/Fandhe-AI/rust-ai-library/blob/main/LICENSE-MIT) ／
[LICENSE-APACHE](https://github.com/Fandhe-AI/rust-ai-library/blob/main/LICENSE-APACHE)
