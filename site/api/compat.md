# compat API

`facade::compat` は自作コア（`autodiff` の `Tape`／`Var`／`nn`）の上に
被せた薄いラッパーです。数値ロジック・shape 検査は一切持ち込まず、
`tensor-core::Tensor::new` や `autodiff::nn::Module` へ委譲します。

対象レイヤーは Linear・ReLU・Sigmoid・Tanh の 3 種限定です。

## `compat::array`

numpy `np.array` 慣習でテンソルを組み立てます。

```rust
use facade::compat::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1-D: `Vec<f32>` → shape [n]
    let v = array(vec![1.0_f32, 2.0, 3.0])?;
    println!("1-D shape: {:?}", v.shape());

    // 2-D: `Vec<Vec<f32>>` → 行優先で平坦化し shape [rows, cols]
    let m = array(vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]])?;
    println!("2-D shape: {:?}", m.shape());

    Ok(())
}
```

このコードブロックは `crates/facade/examples/array_shapes.rs` と
バイト同一です（`cargo run --example array_shapes` で実行確認済み。
出力は `1-D shape: [3]` ／ `2-D shape: [2, 2]`）。

行長が不揃い（jagged）な 2-D 入力は、計算前に検証してエラーを返します。

## `compat::Sequential`

Keras `Sequential` 慣習でレイヤーを積み上げるビルダーです。`add_*` は
`self` を消費し `Self` を返すメソッドチェーンで層を追加します。

| メソッド | 役割 |
|---|---|
| `add_linear(in_features, out_features, seed)` | 全結合層を追加する（bias あり既定）。`in_features == 0` は拒否されるため `Result` を返す |
| `add_relu()` / `add_sigmoid()` / `add_tanh()` | 活性化層を追加する（shape 不変のため `Result` を返さない） |
| `predict(&input)` | 推論の入口。既定バックエンド（CPU）で 1 ステップ分の `Tape` を構築して forward し、追跡を外した `Tensor<f32>` を返す |
| `forward(&tape, &input)` | 呼び出し元が用意した `Tape` 上で forward する（外部 `Tape` に記録をつなげたいとき向け） |

### 学習（勾配取得・パラメータ更新）

`Sequential::bind(&tape)` はこのステップの `Tape` へ全 `Linear` 層の
`weight`／`bias` を葉ノードとして登録し、`SequentialVars` を返します。

| `SequentialVars` のメソッド | 役割 |
|---|---|
| `forward(&tape, &input)` | 学習用 forward（`bind` 済みの葉ノードを使う） |
| `trainable_vars()` | 学習可能パラメータの `Var` 参照列（層順に weight → bias） |
| `trainable_grads(&grads)` | `Tape::backward` の結果から同じ順序で勾配参照列を取得する |

`Sequential::trainable_parameters()` / `Sequential::apply_parameters(updated)`
と組み合わせることで、`autodiff::optim::Sgd` や `autodiff::nn::optim::AdamW`
の位置対応契約にそのまま渡せます。`apply_parameters` は shape を変えない
更新専用の契約であり、層幅を変えるリサイズ用途はサポート対象外です。

### `facade::Tape` newtype について

`Sequential::forward`／`Sequential::bind` はいずれも `autodiff::Tape` では
なく `facade::Tape`（`facade` 側の newtype）を引数に取ります。これは
`facade` が任意の `BackendOps` 実装を注入できる公開 API をあえて設けない
という設計判断（サポート境界の一部）によるもので、`Tape` は
`var`／`backward` の 2 メソッドのみを公開しています。`Tape` を構築できる
のは `facade::tape()` / `facade::tape_for(Device)` だけです。

コード例・動作確認済みの一次ソースは
[Getting Started](/getting-started/) を参照してください。
