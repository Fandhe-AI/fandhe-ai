//! `compat::array` の 1-D/2-D 入力例（イシュー #874）。
//!
//! サイト原稿（`site/api/compat.md`）に転記するコード例の一次ソース
//! （`getting_started.rs` と同じ理由で二重実装を避ける。
//! `.claude/rules/code-comment-style.md`）。`compat::array` は
//! numpy `np.array` 慣習でテンソルを組み立てる薄いラッパーで、shape 検査は
//! `fandhe_ai_tensor_core::Tensor::new` へ委譲される（REQ-9）。

use fandhe_ai::compat::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1-D: `Vec<f32>` → shape [n]
    let v = array(vec![1.0_f32, 2.0, 3.0])?;
    println!("1-D shape: {:?}", v.shape());

    // 2-D: `Vec<Vec<f32>>` → 行優先で平坦化し shape [rows, cols]
    let m = array(vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]])?;
    println!("2-D shape: {:?}", m.shape());

    Ok(())
}
