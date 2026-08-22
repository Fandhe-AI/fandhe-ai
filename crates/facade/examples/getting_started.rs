//! `facade` 最小利用例（イシュー #874）。
//!
//! サイト原稿（`site/getting-started.md`）に転記するコード例の一次
//! ソース。「原稿のコード例は実際にコンパイル・動作
//! 確認済みであること」という #874 の受け入れ条件を、この example の
//! `cargo run --example getting_started` 実行成功で担保する
//! （原稿側は本ファイルの内容をそのまま貼り付けるだけにし、二重実装
//! しない。`.claude/rules/code-comment-style.md`「陳腐化しやすい実装
//! 詳細の重複を避ける」）。
//!
//! `facade` が唯一のサポートされる公開 API 面であり、`compat::array`／
//! `compat::Sequential` はいずれもその薄いラッパー（REQ-9）。本番経路で
//! `unwrap()`/`expect()` を使わない方針（`.claude/rules/coding-rust.md`）
//! に合わせ、`main` は `Result` を返し `?` で伝播する。

use facade::compat::{Sequential, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // numpy `np.array` 慣習でテンソルを組み立てる（2 行 4 列のバッチ入力）。
    // 実データ・shape 検査は `tensor_core::Tensor::new` へ委譲される
    // （compat 層は薄いラッパーに徹する。REQ-9）。
    let input = array(vec![
        vec![0.1_f32, 0.2, 0.3, 0.4],
        vec![0.5_f32, 0.6, 0.7, 0.8],
    ])?;

    // Keras `Sequential` 慣習でレイヤーを積み上げる（対象は Linear・
    // ReLU/Sigmoid/Tanh の 4 種限定。`docs/compat-api-scope.md` §1）。
    // `add_linear` は `in_features == 0` を拒否するため `Result` を返し
    // `?` で連鎖できる（`Linear::new` への委譲。`autodiff::nn::linear`）。
    let model = Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)?
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)?;

    // 推論の入口。内部で `facade::tape()`（既定 CPU・`CpuBackendOps`・
    // 融合有効）を構築し forward するだけの 1 ステップ呼び出し
    // （`Sequential::predict` のドキュメントコメント参照）。
    let output = model.predict(&input)?;

    println!("output shape: {:?}", output.shape());
    Ok(())
}
