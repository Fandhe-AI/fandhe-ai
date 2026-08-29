# Getting Started

## インストール

本ライブラリは Rust の `stable` チャンネルを前提としています
（リポジトリ直下の `rust-toolchain.toml` が単一真実源です）。crates.io（v0.4.0・
2026-08-29 公開済み）から利用できます。

```toml
[dependencies]
fandhe-ai = "0.4.0"
```

公開ドキュメントは以下のとおりです。

- https://docs.rs/fandhe-ai

開発版を試す場合は Git 依存で参照できます。

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/fandhe-ai" }
```

リポジトリを clone してワークスペース内から利用する場合は、path 依存でも
参照できます。

```toml
[dependencies]
fandhe-ai = { path = "../fandhe-ai/crates/facade" }
```

利用者が直接依存すべきクレートは `fandhe-ai` だけです。`fandhe-ai-tensor-core`・
`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・
`fandhe-ai-backend-metal` は内部クレートであり、直接の依存・利用はサポート
対象外です（詳細は [API Reference](/api/) を参照）。

## 最小コード例

`compat::array`（numpy `np.array` 慣習のテンソル生成）と
`compat::Sequential`（Keras `Sequential` 慣習のレイヤー積み上げ）を使うと、
数行でモデルを組み立てて推論できます。以下は
`crates/facade/examples/getting_started.rs`（`cargo run --example
getting_started` で実行確認済み）と同一のコードです。

```rust
use fandhe_ai::compat::{Sequential, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // numpy `np.array` 慣習でテンソルを組み立てる（2 行 4 列のバッチ入力）。
    // 実データ・shape 検査は `fandhe_ai_tensor_core::Tensor::new` へ委譲される
    // （compat 層は薄いラッパーに徹する。REQ-9）。
    let input = array(vec![
        vec![0.1_f32, 0.2, 0.3, 0.4],
        vec![0.5_f32, 0.6, 0.7, 0.8],
    ])?;

    // Keras `Sequential` 慣習でレイヤーを積み上げる（対象は Linear・
    // ReLU/Sigmoid/Tanh の 4 種限定。`docs/compat-api-scope.md` §1）。
    // `add_linear` は `in_features == 0` を拒否するため `Result` を返し
    // `?` で連鎖できる（`Linear::new` への委譲。`fandhe_ai_autodiff::nn::linear`）。
    let model = Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)?
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)?;

    // 推論の入口。内部で `fandhe_ai::tape()`（既定 CPU・`CpuBackendOps`・
    // 融合有効）を構築し forward するだけの 1 ステップ呼び出し
    // （`Sequential::predict` のドキュメントコメント参照）。
    let output = model.predict(&input)?;

    println!("output shape: {:?}", output.shape());
    Ok(())
}
```

このコードブロックは `crates/facade/examples/getting_started.rs` の
実行コード部分（冒頭のモジュールドキュメンテーションコメントを除く
`use` 以降）と同一です（`cargo run --example getting_started` で実行
確認済み。出力は `output shape: [2, 2]`）。

`add_linear` の第 3 引数はパラメータ初期化のシード値です。同じシードを
渡せば毎回同じ初期値になる決定的な構築になります。

学習（勾配取得・パラメータ更新）が必要な場合は `Sequential::bind` が返す
`SequentialVars` 経由で `LinearVars`（勾配取得の入口）へアクセスできます。
詳細は [API Reference の compat API](/api/compat/) を
参照してください。

## バックエンド切替

バックエンド切替は feature フラグを使わない **cfg ベース**です。既定の
`fandhe_ai::tape()` は常に利用可能な CPU バックエンドを結線します。明示的に
デバイスを指定したい場合は `fandhe_ai::tape_for(Device)` を使います。

```rust
use fandhe_ai::{Device, tape_for};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tape = match tape_for(Device::Cuda(0)) {
        Ok(tape) => {
            println!("connected to Device::Cuda(0)");
            tape
        }
        Err(err) => {
            // driver 不在・範囲外 ordinal 等は fail-fast で `BackendError`
            // が返る（`panic!`/`unwrap()` しない。`.claude/rules/coding-rust.md`）。
            // ここでは CPU へフォールバックして example の実行を継続する。
            println!("Device::Cuda(0) unavailable ({err}); falling back to Device::Cpu");
            tape_for(Device::Cpu)?
        }
    };

    let input = tape.var(&fandhe_ai::Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], &[1, 4])?);
    let loss = input.sum(None)?;
    let grads = tape.backward(&loss)?;
    // 入力は loss に直接寄与しているため勾配が必ず存在するはずだが、本番経路で
    // `unwrap()`/`expect()` を使わない方針（`.claude/rules/coding-rust.md`）に
    // 合わせ `?` で型付きエラーとして伝播する。
    let input_grad = grads
        .get(&input)?
        .ok_or("input has no gradient after backward")?;

    println!("input grad shape: {:?}", input_grad.shape());
    Ok(())
}
```

このコードブロックは `crates/facade/examples/backend_switching.rs` の
実行コード部分（冒頭のモジュールドキュメンテーションコメントを除く
`use` 以降）と同一です（`cargo run --example backend_switching` で実行
確認済み。
出力は 1 行目が `Device::Cuda(0) unavailable (...); falling back to
Device::Cpu`〈GitHub ホステッド CI・CUDA 非搭載環境の場合〉、2 行目が
`input grad shape: [1, 4]`）。

`Device::Cuda(ordinal)`／`Device::Metal`（macOS 限定）はいずれも構築時に
デバイスの存在検証を行い、ドライバ不在・範囲外 ordinal の場合はエラーを
返す fail-fast 設計です。`fandhe-ai` はデバイスが利用できないときに自動的に
別のバックエンドへフォールバックすることはしません。フォールバックが
必要な場合は、上記の例のように呼び出し側で `Result` を見て分岐して
ください。

`Device::available()` 相当の自動デバイス検出・列挙 API は現時点では
提供していません。
