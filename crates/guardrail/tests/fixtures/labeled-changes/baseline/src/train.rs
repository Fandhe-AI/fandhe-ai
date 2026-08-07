//! 学習ループとデータセット生成のユーティリティ。回帰テスト
//! （`tests/regression.rs`）とベンチマーク（`benches/forward_bench.rs`）の
//! 両方から利用する。決定的シードは `bench_harness::rng::Xorshift64Star`
//! で駆動する（`autodiff/tests/nn_train_convergence.rs` と同じ利用パターン。
//! `.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード設定
//! ユーティリティを使う」）。
//!
//! TASK-4.2a 検証題材（D5・学習率バグ）の変更対象。

use autodiff::optim::{Sgd, SgdConfig};
use autodiff::{AutodiffError, Tape};
use tensor_core::Tensor;

use crate::model::Mlp;

/// XOR 拡張の合成データセットを生成する。`repeat` 回複製してミニバッチ
/// 相当にする。
pub fn xor_dataset(repeat: usize) -> (Tensor<f32>, Tensor<f32>) {
    let base_inputs: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets: [f32; 4] = [0.0, 1.0, 1.0, 0.0];

    let mut inputs: Vec<f32> = Vec::new();
    let mut targets: Vec<f32> = Vec::new();
    for _ in 0..repeat {
        for i in 0..4 {
            inputs.push(base_inputs[i][0]);
            inputs.push(base_inputs[i][1]);
            targets.push(base_targets[i]);
        }
    }
    let n = targets.len();
    let x = Tensor::new(inputs, &[n, 2]).expect("xor_dataset: shape とデータ長は一致する");
    let y = Tensor::new(targets, &[n, 1]).expect("xor_dataset: shape とデータ長は一致する");
    (x, y)
}

/// `epochs` 回の学習を実行し、学習済みモデルと最終 loss を返す。
/// パラメータ更新は `autodiff::optim::Sgd`（TASK-9.4 相当・許容依存
/// 追加なしの自作コア optimizer）を使う。`Mlp::with_layers` で
/// 各ステップの更新後パラメータへ差し替える（`nn/linear.rs` の
/// `Linear::from_parameters` を都度呼び直す設計と同じ理由。
/// `model.rs::Mlp::with_layers` doc 参照）。
pub fn train(
    mut model: Mlp,
    x: Tensor<f32>,
    y: Tensor<f32>,
    epochs: usize,
    lr: f32,
) -> Result<(Mlp, f32), AutodiffError> {
    let mut optim = Sgd::new(SgdConfig::new(lr))?;
    let mut last_loss = f32::MAX;

    for _ in 0..epochs {
        let tape = Tape::new();
        let x_var = tape.var(&x);
        let y_var = tape.var(&y);

        // `model.forward` が返す `l1v`/`l2v`/`l3v` はこの forward 呼び出し
        // 自身が bind したノードであり、計算グラフに接続されている
        // （`model.rs::Mlp::forward` doc 参照。ここで独立に `bind()` し
        // 直すと非接続ノードの勾配を引くことになり `grads.get` が
        // `None` を返す退行を招く）。
        let (pred, l1v, l2v, l3v) = model.forward(&tape, &x_var)?;
        let loss = pred.mse_loss(&y_var)?;
        last_loss = loss
            .to_tensor()
            .get(&[])
            .expect("train: mse_loss はスカラー shape [] を返す");

        let grads = tape.backward(&loss)?;

        let l1_bias = l1v.bias.expect("train: Mlp::new は bias=true で構築する");
        let l2_bias = l2v.bias.expect("train: Mlp::new は bias=true で構築する");
        let l3_bias = l3v.bias.expect("train: Mlp::new は bias=true で構築する");

        let params: [&Tensor<f32>; 6] = [
            model.fc1().weight(),
            model.fc1().bias().expect("train: bias=true で構築"),
            model.fc2().weight(),
            model.fc2().bias().expect("train: bias=true で構築"),
            model.fc3().weight(),
            model.fc3().bias().expect("train: bias=true で構築"),
        ];
        let grad_refs: [&Tensor<f32>; 6] = [
            grads
                .get(&l1v.weight)?
                .expect("train: weight は forward で必ず使用されるため勾配が存在する"),
            grads
                .get(&l1_bias)?
                .expect("train: bias は forward で必ず使用されるため勾配が存在する"),
            grads
                .get(&l2v.weight)?
                .expect("train: weight は forward で必ず使用されるため勾配が存在する"),
            grads
                .get(&l2_bias)?
                .expect("train: bias は forward で必ず使用されるため勾配が存在する"),
            grads
                .get(&l3v.weight)?
                .expect("train: weight は forward で必ず使用されるため勾配が存在する"),
            grads
                .get(&l3_bias)?
                .expect("train: bias は forward で必ず使用されるため勾配が存在する"),
        ];
        let updated = optim.step(&params, &grad_refs)?;

        let fc1 =
            autodiff::nn::Linear::from_parameters(updated[0].clone(), Some(updated[1].clone()))?;
        let fc2 =
            autodiff::nn::Linear::from_parameters(updated[2].clone(), Some(updated[3].clone()))?;
        let fc3 =
            autodiff::nn::Linear::from_parameters(updated[4].clone(), Some(updated[5].clone()))?;
        model = Mlp::with_layers(fc1, fc2, fc3);
    }

    Ok((model, last_loss))
}
