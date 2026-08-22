# 推論

`compat::Sequential` による推論の 2 経路を示す最小例です。

- **経路 1（`Sequential::predict`）**: 内部で `facade::tape()`
  （composition root。既定 CPU・`CpuBackendOps`・融合有効）を構築して
  forward するだけの 1 ステップ呼び出し。最も簡単な推論入口です。
- **経路 2（外部 `Tape` + `Sequential::forward` + `Var::to_tensor`）**:
  呼び出し側が `Tape` を持ち回りたい用途（grad check・複数層を跨いだ
  計算グラフを自分で組みたい場合等）向けです。

```rust
use facade::compat::{Sequential, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)?
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)?;

    // numpy `np.array` 慣習でバッチ入力を組み立てる（2 行 4 列）。
    let input = array(vec![
        vec![0.1_f32, 0.2, 0.3, 0.4],
        vec![0.5_f32, 0.6, 0.7, 0.8],
    ])?;

    // 経路 1: predict()。
    let predicted = model.predict(&input)?;
    println!("predict() output shape: {:?}", predicted.shape());
    let predicted_00 = predicted
        .get(&[0, 0])
        .ok_or("predict() の出力 shape は [2, 2] のはず（index [0, 0] は範囲内）")?;
    println!("predict() output[0, 0] = {predicted_00}");

    // 経路 2: 外部 Tape + forward + to_tensor。
    let tape = facade::tape();
    let input_var = tape.var(&input);
    let output_var = model.forward(&tape, &input_var)?;
    let output = output_var.to_tensor();
    println!("forward() output shape: {:?}", output.shape());

    // 同一モデル・同一入力のため、2 経路の出力はビット一致するはず
    // （`Sequential::predict` のドキュメンテーションコメント「`predict`
    // は内部で `tape()` を構築して `forward` を呼ぶだけ」と整合する。
    // `.claude/rules/coding-rust.md`「許容誤差を単独で緩和しない」の
    // 趣旨に沿い、ここでは 2 経路が同一計算であることを完全一致で示す）。
    let predicted_data = predicted
        .contiguous()
        .as_slice()
        .ok_or("contiguous() 直後は必ず as_slice() が Some")?
        .to_vec();
    let output_data = output
        .contiguous()
        .as_slice()
        .ok_or("contiguous() 直後は必ず as_slice() が Some")?
        .to_vec();
    let bit_exact = predicted_data
        .iter()
        .zip(output_data.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("predict() と forward() の出力はビット一致: {bit_exact}");

    Ok(())
}
```

このコードブロックは `crates/facade/examples/inference.rs` の実行
コード部分（冒頭のモジュールドキュメンテーションコメントを除く `use`
以降）と同一です（`cargo run -p facade --example inference` で実行
確認済み。出力は次の 4 行）。

```
predict() output shape: [2, 2]
predict() output[0, 0] = -0.17685153
forward() output shape: [2, 2]
predict() と forward() の出力はビット一致: true
```

（`output[0, 0]` の下 1〜2 桁は CPU・コンパイラ・並列実行順序に依存
しうるため確認済み環境での実測値として扱ってください。`predict()` と
`forward()` が同一出力になること自体〈ビット一致: true〉は同一 ops・
同一演算列である限り環境非依存で成立します。）

2 経路が同一出力になるのは、`Sequential::predict` が内部で
`facade::tape()` を構築して `forward` を呼ぶだけの薄いラッパーである
ためです（`crates/facade/src/compat/sequential.rs` の
`Sequential::predict` ドキュメンテーションコメント参照）。
