//! 回帰テスト: 学習が収束し、推論が期待どおりの出力を出すことを確認する。
//! TASK-4.2a 検証題材（D1・D2・D4・D5・G1、バグ注入・機能追加）の
//! end-to-end 検証ゲート。

use autodiff::Tape;
use guardrail_labeled_changes_baseline::model::Mlp;
use guardrail_labeled_changes_baseline::train::{train, xor_dataset};

/// テストの再現性確保のため、モデル初期化を決定的シードで行う。XOR
/// タスクは小規模ネットワークゆえに初期値次第で局所解に陥りうるため、
/// 回帰テストとしての安定性を優先しシードを固定する。
const FIXED_SEED: u64 = 42;

/// 学習が収束し、XOR 拡張タスクの loss が閾値以下になることを確認する。
#[test]
fn training_converges() {
    let model = Mlp::new(FIXED_SEED).expect("test fixture: shape は事前に妥当");
    let (x, y) = xor_dataset(8);
    let (_model, final_loss) = train(model, x, y, 3000, 5e-2).expect("学習ループは失敗しない");

    assert!(
        final_loss < 0.05,
        "学習が収束しなかった: final_loss={final_loss} (閾値 0.05 未満を要求)"
    );
}

/// 推論結果が XOR の期待値（0, 1, 1, 0）に近いことを確認する。
#[test]
fn inference_matches_expected_xor() {
    let model = Mlp::new(FIXED_SEED).expect("test fixture: shape は事前に妥当");
    let (x, y) = xor_dataset(8);
    let (model, _loss) = train(model, x, y, 3000, 5e-2).expect("学習ループは失敗しない");

    let tape = Tape::new(Box::new(backend_cpu::CpuBackendOps::new()));
    let infer_x =
        tensor_core::Tensor::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])
            .expect("test fixture: shape とデータ長は一致する");
    let infer_x_var = tape.var(&infer_x);
    let (pred, _, _, _) = model
        .forward(&tape, &infer_x_var)
        .expect("forward は失敗しない");
    let pred_tensor = pred.to_tensor();
    let expected = [0.0f32, 1.0, 1.0, 0.0];

    for (i, want) in expected.iter().enumerate() {
        let got = pred_tensor
            .get(&[i, 0])
            .expect("test fixture: pred の shape は [4, 1]");
        assert!(
            (got - want).abs() < 0.15,
            "推論結果が期待値から乖離: 入力{i} got={got}, want={want}"
        );
    }
}
