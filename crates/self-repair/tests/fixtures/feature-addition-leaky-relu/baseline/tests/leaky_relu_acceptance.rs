//! 機能追加チケットの受け入れ基準テスト（TASK-3.3c・イシュー #142）。
//!
//! PoC-2 検証題材 (c) の要求「LeakyReLU を追加してほしい」の受け入れ基準を
//! 既知正解値として表す。baseline（`leaky_relu` 未実装）ではコンパイルが
//! 通らず失敗し、
//! `self_repair::FeatureAdditionDetector::detect`（実行するのは
//! `cargo test --release`）がこれを検出する
//! （`crates/self-repair/src/feature_addition.rs` の doc 参照）。

use autodiff::Tape;
use self_repair_feature_addition_leaky_relu_baseline::activations::leaky_relu;
use tensor_core::Tensor;

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    let c = t.contiguous();
    c.as_slice()
        .expect("test fixture: contiguous 直後は必ず as_slice が Some")
        .to_vec()
}

#[test]
fn leaky_relu_matches_known_values() {
    let tape = Tape::new();
    let x_t = Tensor::new(vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let x = tape.var(&x_t);

    let negative_slope = 0.1;
    let y = dense_vec(&leaky_relu(&tape, &x, negative_slope).to_tensor());
    // x >= 0 は x のまま、x < 0 は negative_slope * x（PoC-2 題材 (c) の
    // 受け入れ基準。`crates/guardrail/tests/fixtures/labeled-changes/
    // baseline/src/activations.rs::leaky_relu_matches_known_values` と同一
    // 既知値）。
    let expected = [-0.2f32, -0.05, 0.0, 0.5, 1.0, 3.0];
    for (got, want) in y.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "leaky_relu 出力が既知正解値と不一致: got={got}, want={want}"
        );
    }
}
