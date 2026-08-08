//! 活性化関数モジュール。
//!
//! `autodiff::Var` の演算プリミティブ（`matmul`/`add`/`mul`/`relu`/`tanh`
//! 等。`nn::activation` の `Relu`/`Sigmoid` はその薄いラッパー）への
//! 追加の薄いラッパーとして実装する。`relu`/`sigmoid` は `nn::activation`
//! を再委譲するのみ、`leaky_relu` は `Var` がスカラー四則（`mul_scalar`/
//! `sub` 等）を公開しないため、shape `[1]` の定数 `Var`（broadcast 対象。
//! `Var::add`/`Var::mul` の broadcast 規則は `nn::linear.rs::LinearVars::forward`
//! の bias 加算と同じ）を経由して合成する。数値精度回帰テスト（本モジュール
//! 末尾）の対象であり、TASK-4.2a 変更セット（D1・D2・D4・G1・G5・S2・S5）の
//! バグ注入・機能追加対象でもある。

use autodiff::Tape;
use autodiff::Var;
use autodiff::nn::activation::{Relu, Sigmoid};
use tensor_core::Tensor;

/// `value` を全要素に持つ shape `[1]` の定数を `tape` へ葉ノードとして
/// 登録する。`Var::add`/`Var::mul` の broadcast（`broadcast_shape`）に
/// より任意 shape の `Var` と組み合わせられるため、`leaky_relu` の
/// スカラー係数をこの経路で表現する。
fn constant<'t>(tape: &'t Tape, value: f32) -> Var<'t> {
    let t = Tensor::full(&[1], value).expect("constant: shape [1] は常に妥当");
    tape.var(&t)
}

/// ReLU 活性化関数: `max(0, x)`。`nn::activation::Relu` の薄いラッパー。
pub fn relu<'t>(x: &Var<'t>) -> Var<'t> {
    Relu.forward(x)
}

/// Sigmoid 活性化関数: `1 / (1 + exp(-x))`。既知値検証のリファレンス
/// としても使用する。`nn::activation::Sigmoid` の薄いラッパー。
pub fn sigmoid<'t>(x: &Var<'t>) -> Var<'t> {
    Sigmoid.forward(x)
}

/// Leaky ReLU 活性化関数: `x >= 0` なら `x`、`x < 0` なら
/// `negative_slope * x`。`relu(x) + negative_slope * (x - relu(x))`
/// として合成する（`x - relu(x)` は `x` の負部分のみを残す）。
pub fn leaky_relu<'t>(tape: &'t Tape, x: &Var<'t>, negative_slope: f64) -> Var<'t> {
    let positive = relu(x);
    let neg_one = constant(tape, -1.0);
    let neg_positive = positive
        .mul(&neg_one)
        .expect("leaky_relu: shape [1] は常に broadcast 可能");
    let negative_input = x
        .add(&neg_positive)
        .expect("leaky_relu: relu(x) と x は同一 shape");
    let slope = constant(tape, negative_slope as f32);
    let negative_part = negative_input
        .mul(&slope)
        .expect("leaky_relu: shape [1] は常に broadcast 可能");
    positive
        .add(&negative_part)
        .expect("leaky_relu: positive と negative_part は同一 shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `autodiff::eval`（private）に相当する稠密化ヘルパー。テストは
    /// クレート外部（`tests/fixtures/…/baseline` は独立 crate）のため
    /// `autodiff` の公開 API（`Tensor::contiguous`/`as_slice`）のみで
    /// 完結させる（`autodiff/tests/nn_train_convergence.rs::flat_get`
    /// と同じ立場の再実装）。
    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        let c = t.contiguous();
        c.as_slice()
            .expect("dense_vec: contiguous() 直後は必ず as_slice が Some")
            .to_vec()
    }

    fn known_input(tape: &Tape) -> Var<'_> {
        // 既知値: 負・ゼロ・正を含む 2x3 の固定入力。
        let t = Tensor::new(vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        tape.var(&t)
    }

    /// 数値精度回帰テスト: ReLU の既知正解値との誤差を検証する。
    /// TASK-4.2a 検証題材（活性化関数の取り違えバグ。D1/D2/G1）の検出ゲート。
    #[test]
    fn relu_matches_known_values() {
        let tape = Tape::new_with_ops(Box::new(backend_cpu::CpuBackendOps::new()));
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

    /// 数値精度回帰テスト: Sigmoid の既知正解値との誤差を検証する。
    #[test]
    fn sigmoid_matches_known_values() {
        let tape = Tape::new_with_ops(Box::new(backend_cpu::CpuBackendOps::new()));
        let x = known_input(&tape);
        let y = dense_vec(&sigmoid(&x).to_tensor());
        // 参照値は f64 で `1 / (1 + exp(-x))` を計算した既知正解値。
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

    /// 数値精度回帰テスト: Leaky ReLU（機能追加題材）の既知正解値との誤差を
    /// 検証する。TASK-4.2a 検証題材（D4・G5）の検出ゲート。
    #[test]
    fn leaky_relu_matches_known_values() {
        let tape = Tape::new_with_ops(Box::new(backend_cpu::CpuBackendOps::new()));
        let x = known_input(&tape);
        let y = dense_vec(&leaky_relu(&tape, &x, 0.1).to_tensor());
        let expected = [-0.2f32, -0.05, 0.0, 0.5, 1.0, 3.0];
        for (got, want) in y.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "leaky_relu 出力が既知正解値と不一致: got={got}, want={want}"
            );
        }
    }
}
