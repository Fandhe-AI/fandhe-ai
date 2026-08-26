# fandhe-ai

*A from-scratch Rust AI/ML library — no Burn/candle/tch dependency.*

Rust 製 AI/ML ライブラリ [fandhe-ai](https://github.com/Fandhe-AI/fandhe-ai) の
composition root であり、**唯一のサポートされる公開 API 面**です。テンソル・autodiff・
バックエンド抽象層を完全自作コアとして実装した内部クレート群（`fandhe-ai-tensor-core`・
`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・
`fandhe-ai-backend-metal`）を結線し、`Device` 指定によるバックエンド選択と
compat 公開面（`compat::array`／`compat::Sequential`）を提供します。

## インストール

```toml
[dependencies]
fandhe-ai = "0.3.0"
```

開発版を試す場合は Git 依存でも参照できます。

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/fandhe-ai" }
```

## 最小コード例

```rust
use fandhe_ai::compat::{Sequential, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = array(vec![
        vec![0.1_f32, 0.2, 0.3, 0.4],
        vec![0.5_f32, 0.6, 0.7, 0.8],
    ])?;

    let model = Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)?
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)?;

    let output = model.predict(&input)?;

    println!("output shape: {:?}", output.shape());
    Ok(())
}
```

`cargo run -p fandhe-ai --example getting_started` で実行確認済みです
（出力: `output shape: [2, 2]`）。

## バックエンド

バックエンド切替は feature フラグを使わない cfg ベースです。既定の
`fandhe_ai::tape()` は常に利用可能な CPU バックエンドを結線し、`Device::Cuda(_)`／
`Device::Metal` を明示指定したい場合は `fandhe_ai::tape_for(Device)` を使います。
CUDA・Metal は実行時にデバイスの存在を検証し、利用できない場合はエラーを返します
（自動フォールバックはしません）。バックエンド間の数値一致は「相対誤差 1e-3 未満
または絶対誤差 1e-5 未満」の複合判定で担保します。

## ドキュメント・リポジトリ

利用者向けドキュメントサイト（GitHub Pages）: https://fandhe-ai.github.io/fandhe-ai/（Getting Started / Guides / Examples / API Reference）。API リファレンスは https://docs.rs/fandhe-ai

- ソース: <https://github.com/Fandhe-AI/fandhe-ai/tree/main/crates/facade>
- サポート境界の詳細: [`docs/compat-api-scope.md`](https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/compat-api-scope.md)

## ライセンス

MIT または Apache License 2.0（デュアルライセンス）。
[LICENSE-MIT](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-MIT) ／
[LICENSE-APACHE](https://github.com/Fandhe-AI/fandhe-ai/blob/main/LICENSE-APACHE)
