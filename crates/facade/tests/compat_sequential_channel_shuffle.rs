//! `compat::Sequential::add_channel_shuffle` と `Var::channel_shuffle`（イシュー #3822・
//! 設計 #3746・承認 #3768 コメント 6554564412。
//! `docs/nn-shrink-bilinear-channel-shuffle-decision.md` §3）の facade 公開面を CPU で検証する
//! 統合テスト。無状態層で、内部実装（`nn::ChannelShuffle`）の薄い委譲である。
//!
//! - 構築検査（`groups == 0` の即時拒否）と forward 時の遅延検査（rank・整除性）
//! - 数値: 手計算例・`Var` 経路と層経路の bit 一致
//! - 学習経路（`fit` が前後に Linear を挟んで通る）・常駐経路（CPU 既定経路）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄の型付き拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（承認済みの現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_channel_shuffle_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn ramp(shape: &[usize], scale: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(
        (0..n).map(|i| ((i as f32) * 0.37).sin() * scale).collect(),
        shape,
    )
}

/// `Linear(4 → 6) → ChannelShuffle(3) → Linear(6 → 2)`（入力 `[4, 1, 4]` は rank 3。
/// Linear は最終軸に作用するため C=1 は shuffle できない。ここでは Flatten を使わず
/// `[N, 6, 1]` へ整える代わりに conv で C を作る）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 6, [1, 1], [1, 1], [0, 0], [1, 1], 1, 11)
        .unwrap()
        .add_channel_shuffle(3)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(24, 2, 12)
        .unwrap()
}

#[test]
fn zero_groups_is_rejected_at_construction() {
    assert!(matches!(
        Sequential::new().add_channel_shuffle(0),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn rank_and_divisibility_are_checked_at_forward() {
    let model = Sequential::new().add_channel_shuffle(2).unwrap();
    assert!(model.predict(&ramp(&[4, 4], 1.0)).is_err());
    assert!(model.predict(&ramp(&[1, 3, 2, 2], 1.0)).is_err());
}

#[test]
fn hand_computed_example() {
    let model = Sequential::new().add_channel_shuffle(2).unwrap();
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], &[1, 4, 1, 1]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[1, 4, 1, 1]);
    assert_eq!(
        bits(&y),
        bits(&tensor(vec![0.0, 2.0, 1.0, 3.0], &[1, 4, 1, 1]))
    );
}

#[test]
fn var_method_matches_layer_path_bit_exact_and_rejects_bad_input() {
    let x = ramp(&[2, 6, 3, 3], 1.0);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_var = xv.channel_shuffle(3).unwrap();
    let via_layer = Sequential::new()
        .add_channel_shuffle(3)
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(bits(&via_var.to_tensor()), bits(&via_layer));
    assert!(matches!(
        xv.channel_shuffle(0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(xv.channel_shuffle(4).is_err());
}

#[test]
fn mixed_model_predict_matches_tape_forward() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 2, 2], 1.0);
    let via_predict = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 2]);
    assert_eq!(bits(&via_predict), bits(&via_forward));
}

#[test]
fn fit_updates_parameters_through_channel_shuffle() {
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    // Conv2d(weight, bias) + Linear(weight, bias) の 4 件。層自体は学習パラメータを持たない。
    assert_eq!(model.trainable_parameters().len(), 4);
    let x = ramp(&[4, 1, 2, 2], 1.0);
    let y = ramp(&[4, 2], 0.5);
    let before: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    let history = model.fit(&x, &y, FitConfig::new(3, 2)).unwrap();
    assert!(history.loss.iter().all(|l| l.is_finite()));
    let after: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    assert_ne!(before, after, "fit でパラメータが更新されるはず");
}

#[test]
fn save_and_load_round_trip_is_bit_identical() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 2, 2], 1.0);
    let guard = TempDirGuard::new("channel_shuffle_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("channel_shuffle"), "{manifest}");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = Sequential::new().add_channel_shuffle(2).unwrap();
    let guard = TempDirGuard::new("channel_shuffle_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    assert!(original.contains("\"groups\":2"), "{original}");
    // 必須キーの欠落（未知キー）。
    std::fs::write(&path, original.replace("\"groups\"", "\"groupsx\"")).unwrap();
    assert!(load_model(&dir).is_err());
    // groups=0 は add_channel_shuffle 経由で Autodiff の型付きエラーになる。
    std::fs::write(&path, original.replace("\"groups\":2", "\"groups\":0")).unwrap();
    assert!(load_model(&dir).is_err());
}

#[test]
fn onnx_export_rejects_channel_shuffle_layer() {
    let model = Sequential::new().add_channel_shuffle(2).unwrap();
    let err = OnnxModel::from_sequential(&model).unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
}
