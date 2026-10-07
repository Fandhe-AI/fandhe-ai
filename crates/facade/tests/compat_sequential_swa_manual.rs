//! イシュー #2658（親 #2657）の `AveragedModel`（#2679 で
//! `fandhe_ai::optim::AveragedModel` として公開済み。`docs/autodiff-swa-decision.md` §7）と
//! 公開 compat API（`compat::Sequential`）の手動結線の契約テスト。
//! `compat::Sequential` の `named_parameters()`／`state_dict()`／`load_state_dict()` と
//! `AveragedModel::from_named`／`update_named`／`averaged_state_dict` を結線する。
//!
//! 検査項目: 各 step 後スナップショットの `f64` 算術平均と SWA の平均が統一複合
//! 判定（相対 1e-3 未満 または 絶対 1e-5 未満）で一致すること・平均重みへ差し替えた
//! 後に退避値で戻すと bit 完全一致すること・評価のための差し替えが学習状態を
//! 変えないこと。flaky になりうる accuracy 比較は入れない。ホスト計算のみのため
//! `#[ignore]` 分離は行わない。

use std::collections::HashMap;

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{AdamW, AdamWConfig, AveragedModel};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_tensor_core::Tensor;

const REL_TOL: f64 = 1e-3;
const ABS_RESCUE: f64 = 1e-5;
const STEPS: usize = 6;

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, 0x5A01)
        .unwrap()
        .add_relu()
        .add_linear(8, 3, 0x5A02)
        .unwrap()
}

fn data() -> (Tensor<f32>, Tensor<i32>) {
    let x: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.37).sin()).collect();
    let y: Vec<i32> = vec![0, 1, 2, 0, 1, 2];
    (
        Tensor::new(x, &[6, 4]).unwrap(),
        Tensor::new(y, &[6]).unwrap(),
    )
}

fn snapshot(model: &Sequential) -> HashMap<String, Vec<f32>> {
    model
        .state_dict()
        .into_iter()
        .map(|(k, v)| (k, v.as_slice().unwrap().to_vec()))
        .collect()
}

#[test]
fn swa_average_matches_f64_mean_and_evaluation_swap_is_state_neutral() {
    let (x_train, y_train) = data();
    let mut model = build_model();
    let mut opt = AdamW::new(AdamWConfig {
        lr: 5e-2,
        ..AdamWConfig::default()
    })
    .unwrap();
    let mut swa = AveragedModel::from_named(model.named_parameters()).unwrap();
    let mut snapshots: Vec<HashMap<String, Vec<f32>>> = Vec::new();

    for _ in 0..STEPS {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(&x_train);
        let pred = bound.forward(&tape, &x).unwrap();
        let loss = pred
            .cross_entropy_loss(&y_train, 1, Reduction::Mean)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let grad_refs = bound.trainable_grads(&grads).unwrap();
        let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> = model
            .trainable_parameters()
            .into_iter()
            .zip(grad_refs)
            .collect();
        let updated = opt.step(&params_and_grads).unwrap();
        model.apply_parameters(updated).unwrap();

        swa.update_named(model.named_parameters()).unwrap();
        snapshots.push(snapshot(&model));
    }
    assert_eq!(swa.n_averaged(), STEPS as u64);

    // (1) SWA 平均 = スナップショットの f64 算術平均（統一複合判定）。
    for (name, avg) in swa.averaged_state_dict() {
        let avg = avg.as_slice().unwrap().to_vec();
        for (k, &a) in avg.iter().enumerate() {
            let mean: f64 =
                snapshots.iter().map(|s| s[&name][k] as f64).sum::<f64>() / STEPS as f64;
            let diff = (a as f64 - mean).abs();
            let scale = (a as f64).abs().max(mean.abs()).max(1e-12);
            assert!(
                diff / scale < REL_TOL || diff < ABS_RESCUE,
                "{name}[{k}]: swa={a} mean={mean}"
            );
        }
    }

    // (2) 評価のための差し替え → 退避値で復帰すると bit 完全一致。
    let before = snapshot(&model);
    let backup = model.state_dict();
    model.load_state_dict(swa.averaged_state_dict()).unwrap();
    let during = snapshot(&model);
    assert_ne!(before, during, "平均重みへ差し替わっていない");
    let _ = model.predict(&x_train).unwrap();
    model.load_state_dict(backup).unwrap();
    let after = snapshot(&model);
    for (name, vals) in &before {
        let a = &after[name];
        assert_eq!(vals.len(), a.len());
        for (x, y) in vals.iter().zip(a) {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}: 復帰後が bit 一致しない");
        }
    }
}
