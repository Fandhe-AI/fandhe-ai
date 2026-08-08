//! 受け入れ条件「Sequential でのモデル構築・推論が動作する」
//! （TASK-9.2a・イシュー #95）を公開 API（`autodiff::compat`）経由で
//! 検証する統合テスト。ユニットテスト（`compat/sequential.rs`・
//! `compat/array.rs` 内）はクレート内部の中間状態を扱うのに対し、
//! 本ファイルは crate 外部から見える公開 API のみを使う点が異なる。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

mod common;

use autodiff::Tape;
use autodiff::compat::{Sequential, array};
use autodiff::nn::Linear;
use autodiff::nn::activation::Sigmoid;
use tensor_core::Tensor;

const SEED1: u64 = 7001;
const SEED2: u64 = 7002;

/// 受け入れ条件本体: `Sequential::new().add_linear(..).add_relu()...` の
/// メソッドチェーンで 2 層 MLP を構築し、`predict` が期待 shape の
/// `Tensor<f32>` を返すこと。
#[test]
fn sequential_builds_two_layer_mlp_and_predicts() {
    let model = Sequential::new()
        .add_linear(8, 16, SEED1)
        .unwrap()
        .add_relu()
        .add_linear(16, 4, SEED2)
        .unwrap();

    let batch = 5;
    let input = array(vec![vec![0.25_f32; 8]; batch]).unwrap();
    let output = model.predict(&input, common::naive_ops()).unwrap();

    assert_eq!(output.shape(), &[batch, 4]);
}

/// `compat::array` から生成したテンソルをそのまま `Sequential::predict`
/// に渡す経路（numpy 慣習の入口 → Keras 慣習の入口、という互換 API 層
/// 2 系統の連携）を確認する。
#[test]
fn array_output_feeds_directly_into_sequential_predict() {
    let model = Sequential::new()
        .add_linear(3, 2, SEED1)
        .unwrap()
        .add_tanh();

    let input = array([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let output = model.predict(&input, common::naive_ops()).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // tanh の出力は (-1, 1) に収まる。
    for v in output_dense(&output) {
        assert!((-1.0..=1.0).contains(&v));
    }
}

/// 数値一致（同一経路の完全一致）: 同一シードで `nn::Linear` +
/// `nn::activation` を直接組んだ手動 forward と `Sequential::forward` の
/// 出力がビット一致すること。同一演算列のため tolerance は新設しない
/// （`.claude/rules/coding-rust.md` の許容誤差を単独で緩和しない方針に
/// 沿い、ここでは緩和ではなく完全一致で判定する）。
#[test]
fn sequential_forward_matches_manual_nn_forward_bit_exact() {
    let linear1 = Linear::new(4, 6, true, SEED1).unwrap();
    let linear2 = Linear::new(6, 3, true, SEED2).unwrap();

    let input_tensor = array(vec![vec![0.1_f32, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]]).unwrap();

    // 手動経路。
    let manual_tape = Tape::new(common::naive_ops());
    let manual_input = manual_tape.var(&input_tensor);
    let h = linear1.bind(&manual_tape).forward(&manual_input).unwrap();
    let h = Sigmoid.forward(&h);
    let manual_output = linear2.bind(&manual_tape).forward(&h).unwrap();

    // Sequential 経路（別インスタンスだが同一シードのため同一パラメータ）。
    let model = Sequential::new()
        .add_linear(4, 6, SEED1)
        .unwrap()
        .add_sigmoid()
        .add_linear(6, 3, SEED2)
        .unwrap();
    let seq_tape = Tape::new(common::naive_ops());
    let seq_input = seq_tape.var(&input_tensor);
    let seq_output = model.forward(&seq_tape, &seq_input).unwrap();

    assert_eq!(
        output_dense(&manual_output.to_tensor()),
        output_dense(&seq_output.to_tensor())
    );
}

/// Tape 経由 forward: `Sequential::forward` を外部 `Tape` 上で実行し
/// `Tape::backward` まで通ることを確認する（グラフ記録の整合確認）。
#[test]
fn sequential_forward_on_external_tape_reaches_backward() {
    let model = Sequential::new()
        .add_linear(4, 3, SEED1)
        .unwrap()
        .add_relu();

    let tape = Tape::new(common::naive_ops());
    let input_tensor = array([[0.1_f32, -0.2, 0.3, -0.4]]).unwrap();
    let input = tape.var(&input_tensor);
    let output = model.forward(&tape, &input).unwrap();
    let loss = output.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let input_grad = grads
        .get(&input)
        .unwrap()
        .expect("入力ノードは loss に寄与している");
    assert_eq!(input_grad.shape(), input_tensor.shape());
}

/// `array` の jagged 入力エラーは `Sequential` 側の構築失敗と混同されず、
/// 独立して伝播すること（外部入力検証は forward 到達前に完了する契約。
/// `.claude/rules/security.md` A03）。
#[test]
fn jagged_array_input_is_rejected_before_sequential_predict() {
    let jagged = vec![vec![1.0_f32, 2.0], vec![3.0]];
    assert!(array(jagged).is_err());
}

fn output_dense(t: &Tensor<f32>) -> Vec<f32> {
    // `tests/backward.rs` 等と同じく、非連続 stride を考慮せず単純な
    // 添字アクセスで足りる shape のみを扱うテストのため、`Tensor` の
    // 公開 API（`shape`/インデックス相当）を使わず素朴に検証したい
    // 箇所は呼び出し側で直接比較する。ここでは全要素が連続レイアウト
    // で確定している `to_tensor()` 直後の値のみを対象にするため、
    // `storage` を経由しない単純な複製で十分。
    (0..t.numel())
        .map(|i| {
            let shape = t.shape();
            let mut idx = vec![0usize; shape.len()];
            let mut rem = i;
            for d in (0..shape.len()).rev() {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            t.get(&idx).expect("計算済み添字は shape 内に収まる")
        })
        .collect()
}
