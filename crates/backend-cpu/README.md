# fandhe-ai-backend-cpu

[fandhe-ai](https://github.com/Fandhe-AI/fandhe-ai) の CPU バックエンド
（`rayon` 並列カーネル）を担う内部クレートです。既定バックエンドとして
`fandhe-ai` から結線されます。

**このクレートへの直接依存・直接利用はサポート対象外です。** 唯一のサポート
される公開 API 面は
[`fandhe-ai`](https://crates.io/crates/fandhe-ai)
（composition root・compat 公開面）であり、本クレートはその内部実装として
crates.io にも公開されています（依存解決のため）。

## 利用方法

```toml
[dependencies]
fandhe-ai = "0.4.0"
```

開発版を試す場合は Git 依存でも参照できます。

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/fandhe-ai" }
```

インストール・最小コード例は
[`fandhe-ai` の README](https://github.com/Fandhe-AI/fandhe-ai/blob/main/README.md#最小コード例)
を参照してください。

## 並列度（`RAYON_NUM_THREADS`）

`gemm_blis` 並列 GEMM の既定並列度は、`RAYON_NUM_THREADS` 環境変数が
未指定の場合、プラットフォーム判定（macOS `hw.perflevel0.logicalcpu`／
Linux sysfs `cpu_capacity`）で検出した物理大コア数へ限定されます
（イシュー #1363。異種コア構成〈big.LITTLE 系〉での非単調性仮説の検証が
目的）。`RAYON_NUM_THREADS` を明示指定した場合はその値がそのまま
採用され、上限はかかりません。判定不能な環境（同種コア構成・Linux/macOS
以外・sandbox 等）では従来どおり `rayon` の既定並列度がそのまま使われ
ます。詳細は `docs/perf/cpu-gemm-default-thread-limit.md` を参照して
ください。

## ドキュメント・リポジトリ

利用者向けドキュメントサイト（GitHub Pages）: https://fandhe-ai.github.io/fandhe-ai/（Getting Started / Guides / Examples / API Reference）。API リファレンスは https://docs.rs/fandhe-ai

- ソース: <https://github.com/Fandhe-AI/fandhe-ai/tree/main/crates/backend-cpu>
- バックエンド設計の詳細: [`docs/backend-switching-design.md`](https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/backend-switching-design.md)

## ライセンス

MIT または Apache License 2.0（デュアルライセンス）。
[LICENSE-MIT](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-MIT) ／
[LICENSE-APACHE](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-APACHE)
