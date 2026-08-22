# 学習ループ

`compat::Sequential` を使った学習ループの最小例です。`autodiff::optim::
{Sgd, AdamW}` は `facade` の公開面（`docs/compat-api-scope.md` §0）に
含まれない内部 API のため、この例は `param - lr * grad` を自前で計算
する手動 SGD にしています。optimizer 実装自体の解説は本ページの対象外
です。

```rust
use bench_harness::rng::Xorshift64Star;
use facade::Tensor;
use facade::compat::Sequential;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

const STEPS: usize = 100;
const LR: f32 = 0.05;

/// `x`（`[BATCH, D_IN]`）・`y`（`[BATCH, D_OUT]`）を Xorshift64Star から
/// 生成する（`crates/facade/tests/compat_sequential_train.rs::gen_regression_data`
/// と同一生成順）。
fn gen_regression_data(
    seed: u64,
) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    Ok((
        Tensor::new(x, &[BATCH, D_IN])?,
        Tensor::new(y, &[BATCH, D_OUT])?,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)?
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)?;

    let (x_data, y_data) = gen_regression_data(SEED_DATA)?;

    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;

    for step in 0..STEPS {
        // 1 ステップ分の Tape 上で forward → backward → 手動 SGD 更新値の
        // 計算までを行い、更新後テンソル列（所有 `Vec<Tensor<f32>>`）を
        // ブロック外へ持ち出す。ブロックを抜けると `bound`（`&model` を
        // 借用）と `tape` が drop され、直後の `apply_parameters`
        // （`&mut model`）呼び出しと借用が競合しない。
        let updated: Vec<Tensor<f32>> = {
            let tape = facade::tape();
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

            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let param_refs = model.trainable_parameters();

            let mut next_params = Vec::with_capacity(param_refs.len());
            for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                // `contiguous()` の戻り値は同一文の間だけ生存すればよい
                // （`crates/facade/tests/compat_sequential_train.rs::dense_vec`
                // と同じ一時変数寿命の使い方）ため 1 文で `to_vec()` まで行う。
                let param_data = param.contiguous().as_slice().ok_or(
                    "trainable_parameters() の要素は contiguous() 直後は必ず as_slice() が Some",
                )?.to_vec();
                let grad_data = grad
                    .contiguous()
                    .as_slice()
                    .ok_or("trainable_grads() の要素は contiguous() 直後は必ず as_slice() が Some")?
                    .to_vec();
                let sgd_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - LR * g)
                    .collect();
                next_params.push(Tensor::from_slice(&sgd_data, param.shape())?);
            }
            next_params
        };
        model.apply_parameters(updated)?;
    }

    let initial_loss = initial_loss.ok_or("STEPS > 0 のため初期 loss は必ず記録される")?;
    println!("initial loss: {initial_loss}");
    println!("final loss: {final_loss}");
    println!("loss decreased: {}", final_loss < initial_loss);

    Ok(())
}
```

このコードブロックは `crates/facade/examples/training_loop.rs` の
実行コード部分（冒頭のモジュールドキュメンテーションコメントを除く
`use` 以降）と同一です（`cargo run -p facade --example training_loop`
で実行確認済み。出力は次の 3 行）。

```
initial loss: 0.5237394
final loss: 0.055154312
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

決定的シード・数値一致の判定方針は
[数値一致契約](/guides/numerical-parity/)を参照してください。
