//! 活性化関数モジュール（baseline 状態: `leaky_relu` 未実装）。
//!
//! `autodiff::Var` の演算プリミティブ（`nn::activation` の `Relu`/`Sigmoid`）
//! への薄いラッパー。この baseline では `leaky_relu` が存在しないため、
//! `../tests/leaky_relu_acceptance.rs`（受け入れ基準テスト）はコンパイルが
//! 通らず `cargo test --release` が失敗する
//! （`self_repair::FeatureAdditionDetector::detect` の検出対象。
//! `crates/self-repair/src/feature_addition.rs` の doc 参照）。

use autodiff::Var;
use autodiff::nn::activation::{Relu, Sigmoid};

/// ReLU 活性化関数: `max(0, x)`。`nn::activation::Relu` の薄いラッパー。
pub fn relu<'t>(x: &Var<'t>) -> Var<'t> {
    Relu.forward(x)
}

/// Sigmoid 活性化関数: `1 / (1 + exp(-x))`。既知値検証のリファレンスとしても
/// 使用する。`nn::activation::Sigmoid` の薄いラッパー。
pub fn sigmoid<'t>(x: &Var<'t>) -> Var<'t> {
    Sigmoid.forward(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::Tape;
    use tensor_core::Tensor;

    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        let c = t.contiguous();
        c.as_slice()
            .expect("test fixture: contiguous 直後は必ず as_slice が Some")
            .to_vec()
    }

    fn known_input(tape: &Tape) -> Var<'_> {
        // 既知値: 負・ゼロ・正を含む 2x3 の固定入力
        // （`crates/guardrail/tests/fixtures/labeled-changes/baseline/src/
        // activations.rs` と同一の既知値セットを使う）。
        let t = Tensor::new(vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        tape.var(&t)
    }

    #[test]
    fn relu_matches_known_values() {
        let tape = Tape::new();
        let x = known_input(&tape);
        let y = dense_vec(&relu(&x).to_tensor());
        let expected = [0.0f32, 0.0, 0.0, 0.5, 1.0, 3.0];
        for (got, want) in y.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "relu 出力が既知正解値と不一致: got={got}, want={want}"
            );
        }
    }

    #[test]
    fn sigmoid_matches_known_values() {
        let tape = Tape::new();
        let x = known_input(&tape);
        let y = dense_vec(&sigmoid(&x).to_tensor());
        let expected = [
            0.11920292f32,
            0.37754068,
            0.5,
            0.62245933,
            0.7310586,
            0.95257413,
        ];
        for (got, want) in y.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "sigmoid 出力が既知正解値と不一致: got={got}, want={want}"
            );
        }
    }
}
