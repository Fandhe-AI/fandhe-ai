# 学習ループ

`Tape`／`Var` と `fandhe_ai::optim::Sgd` で書く、PyTorch 風の手書き学習ループの
最小例です。`fandhe-ai` の公開面（`fandhe_ai::*`）だけで完結するため、
`fandhe-ai` だけに依存するプロジェクトへそのまま写して動きます。

学習ループを書かずに済ませたい場合（Keras 風）は `Sequential::compile`／`fit`
を使います（API は [compat API](/api/compat/) を参照。実行可能な example は
イシュー #4338 で追加予定です）。

```rust
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{Sgd, SgdConfig};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

const STEPS: usize = 100;
const LR: f32 = 0.05;

/// `x`（`[BATCH, D_IN]`）・`y`（`[BATCH, D_OUT]`）を `[0, 1)` の一様乱数で
/// 生成する（`manual_seed` で系列を固定してから `rand` を呼ぶ）。
fn gen_regression_data(
    seed: u64,
) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    fandhe_ai::manual_seed(seed);
    let x = fandhe_ai::rand(&[BATCH, D_IN])?;
    let y = fandhe_ai::rand(&[BATCH, D_OUT])?;
    Ok((x, y))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)?
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)?;

    let (x_data, y_data) = gen_regression_data(SEED_DATA)?;
    let mut optimizer = Sgd::new(SgdConfig::new(LR))?;

    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;

    for step in 0..STEPS {
        // 1 ステップ分の Tape 上で forward → backward → optimizer.step までを
        // 行い、更新後テンソル列（所有 `Vec<Tensor<f32>>`）をブロック外へ
        // 持ち出す。ブロックを抜けると `bound`（`&model` を借用）と `tape` が
        // drop され、直後の `apply_parameters`（`&mut model`）と借用が競合しない。
        let updated: Vec<Tensor<f32>> = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound.forward(&tape, &x)?;
            let loss = pred.mse_loss(&y)?;
            let loss_value = loss
                .to_tensor()
                .get(&[])
                .ok_or("loss はスカラー shape [] のはず")?;
            if step == 0 {
                initial_loss = Some(loss_value);
            }
            final_loss = loss_value;

            // PyTorch の `loss.backward()` に相当。
            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let param_refs = model.trainable_parameters();

            // PyTorch の `optimizer.step()` に相当（更新後パラメータを返す）。
            optimizer.step(&param_refs, &grad_refs)?
        };
        model.apply_parameters(updated)?;
    }

    let initial_loss = initial_loss.ok_or("STEPS > 0 のため初期 loss は必ず記録される")?;
    println!("initial loss: {initial_loss}");
    println!("final loss: {final_loss}");
    let loss_decreased = final_loss < initial_loss;
    println!("loss decreased: {loss_decreased}");

    // 学習退行（勾配計算・パラメータ更新の不具合で loss が非有限化
    // する、または減少しなくなる）を表示するだけで見逃さないよう、
    // 満たさない場合は Err を返して `cargo run` を失敗終了させる
    // （イシュー #875 レビュー指摘）。
    if !final_loss.is_finite() {
        return Err(format!("final loss が有限値ではない: {final_loss}").into());
    }
    if !loss_decreased {
        return Err(format!(
            "loss が減少しなかった（退行の可能性）: initial={initial_loss}, final={final_loss}"
        )
        .into());
    }

    Ok(())
}
```

このコードブロックは `crates/facade/examples/training_loop.rs` の
実行コード部分（冒頭のモジュールドキュメンテーションコメントを除く
`use` 以降）と同一です（`cargo run -p fandhe-ai --example training_loop`
で実行確認済み。出力は次の 3 行）。

```
initial loss: 0.15434094
final loss: 0.06244343
loss decreased: true
```

（決定的シードのため同一環境では再現しますが、下 1〜2 桁は CPU・
コンパイラ・並列実行順序に依存しうるため、確認済み環境での実測値として
扱ってください。`loss decreased: true` は環境非依存で成立します。）

## 借用スコープに関する注意

[`Sequential::bind`](/api/compat/) が返す `SequentialVars` は
`&model`（不変借用）と `&tape` の両方を借用します。このハンドルが
生きている間は `Sequential::apply_parameters`（`&mut self` を要求）を
呼べません。上記の例では 1 ステップ分の処理をブロックスコープ
（`let updated: Vec<Tensor<f32>> = { ... };`）で囲み、ブロックを抜けて
借用を解放してから `apply_parameters` を呼ぶ構成にしています。

## `Tape` の再利用（`reset`）について

上記の例はステップごとに `fandhe_ai::tape()` を新規生成・破棄する
運用です。ステップ内で `Tape` を再利用したい場合（reuse GEMM・
tape 構築コストが問題になる学習ループ）は、`Tape::reset`（イシュー
#1048）でノード列を葉プレフィックス（最初の演算より前に登録した葉）
まで切り詰め、同一 `Tape` を使い回せます。

```rust
let mut tape = fandhe_ai::tape();
let x = tape.var(&x_data); // 演算前に登録した葉は reset を跨いで保持される
let w = tape.var(&w_data);

for _ in 0..STEPS {
    // `tape.leaf(index)` で保持済みの葉 `Var` を再取得する（コピーなし）。
    let x = tape.leaf(0).unwrap();
    let w = tape.leaf(1).unwrap();
    let y = x.matmul(&w)?.relu();
    let loss = y.mse_loss(&target)?;
    let grads = tape.backward(&loss)?;
    // ... grads を使ってパラメータを更新 ...
    tape.reset(); // 演算ノードを切り詰め、次ステップ用に再利用する
}
```

`reset` は `&mut self` を要求するため、reset 前に取得した `Var` を
reset 後のスコープへ持ち越すことはコンパイル時に弾かれます（借用
検査による静的な安全性）。`reset` 前に得た `Gradients` を reset 後の
`Var` に対して読もうとした場合は、`Gradients::get` が
`Err(AutodiffError::TapeMismatch)` を返します（世代番号による
実行時 fail-closed 検査）。

決定的シード・数値一致の判定方針は
[数値一致契約](/guides/numerical-parity/)を参照してください。
