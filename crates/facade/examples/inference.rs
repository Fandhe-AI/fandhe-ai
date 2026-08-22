//! `compat::Sequential` による推論の 2 経路を示す最小例（イシュー #875）。
//!
//! サイト原稿（`site/examples/inference.md`）に転記するコード例の一次
//! ソース（`getting_started.rs`〈#874〉と同じ理由で二重実装を避ける。
//! `.claude/rules/code-comment-style.md`）。本 example の実行成功
//! （`cargo run -p fandhe-ai --example inference`）が原稿の受け入れ条件
//! （コード例がコンパイル・動作確認済みであること）を担保する。
//!
//! - **経路 1（`Sequential::predict`）**: 内部で [`fandhe_ai::tape`]
//!   （composition root。既定 CPU・`CpuBackendOps`・融合有効）を構築し
//!   forward するだけの 1 ステップ呼び出し（最も簡単な推論入口）。
//! - **経路 2（外部 `Tape` + `Sequential::forward` + `Var::to_tensor`）**:
//!   呼び出し側が `Tape` を持ち回りたい用途（grad check・複数層を跨いだ
//!   計算グラフを自分で組みたい場合等）向け。`Sequential::forward` の
//!   ドキュメンテーションコメント（`crates/facade/src/compat/
//!   sequential.rs`）参照。

use fandhe_ai::compat::{Sequential, array};

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
    let tape = fandhe_ai::tape();
    let input_var = tape.var(&input);
    let output_var = model.forward(&tape, &input_var)?;
    let output = output_var.to_tensor();
    println!("forward() output shape: {:?}", output.shape());

    // 同一モデル・同一入力のため、2 経路の出力は shape・要素数・値の
    // すべてがビット一致するはず（`Sequential::predict` のドキュメン
    // テーションコメント「`predict` は内部で `tape()` を構築して
    // `forward` を呼ぶだけ」と整合する。`.claude/rules/coding-rust.md`
    // 「許容誤差を単独で緩和しない」の趣旨に沿い、ここでは 2 経路が
    // 同一計算であることを完全一致で示す）。
    //
    // shape が一致しない場合、後続の `zip` は短い側までしか比較せず
    // 不一致を握り潰してしまう（イシュー #875 レビュー指摘）ため、
    // 値比較の前に shape 自体の一致を検査して不一致なら Err で失敗
    // させる。
    if predicted.shape() != output.shape() {
        return Err(format!(
            "predict() と forward() の出力 shape が不一致: {:?} != {:?}",
            predicted.shape(),
            output.shape()
        )
        .into());
    }
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
    // 出力ビット不一致は数値退行を意味するため、表示だけで成功終了
    // させず Err を返す（`cargo run` の終了コードで退行を検出できる
    // ようにする。イシュー #875 レビュー指摘）。
    if !bit_exact {
        return Err("predict() と forward() の出力がビット一致しない（数値退行の可能性）".into());
    }

    Ok(())
}
