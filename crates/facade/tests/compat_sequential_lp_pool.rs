//! `compat::Sequential::add_lp_pool1d`／`2d`／`3d` と `Var::lp_pool*d`（イシュー #3825・設計 #3748・
//! 承認 #3768。`docs/autodiff-pool3d-ops-decision.md` §19）の facade 公開面を CPU で検証する統合テスト。
//!
//! - 構築検査（`norm_type`・`kernel_size` の即時拒否）と forward 時の遅延検査（rank）
//! - 数値: 手計算例・`Var` 経路と層経路の bit 一致
//! - 学習経路（`fit` が非常駐の tape 経路で通る）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄の型付き拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（承認済みの現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_lp_pool_backend_parity.rs`（`#[ignore]`）。

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

fn ramp(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(
        (0..n)
            .map(|i| 0.5 + ((i as f32) * 0.37).sin().abs())
            .collect(),
        shape,
    )
}

/// `Conv2d(1→2) → LPPool2d(2, [2,2]) → Flatten → Linear`（入力 `[N, 1, 4, 4]`）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 2, [1, 1], [1, 1], [0, 0], [1, 1], 1, 11)
        .unwrap()
        .add_lp_pool2d(2.0, [2, 2], None)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(8, 2, 12)
        .unwrap()
}

#[test]
fn invalid_arguments_are_rejected_at_construction() {
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            Sequential::new().add_lp_pool1d(bad, 2, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(Sequential::new().add_lp_pool2d(bad, [2, 2], None).is_err());
        assert!(
            Sequential::new()
                .add_lp_pool3d(bad, [2, 2, 2], None)
                .is_err()
        );
    }
    assert!(Sequential::new().add_lp_pool1d(2.0, 0, None).is_err());
    assert!(Sequential::new().add_lp_pool2d(2.0, [2, 0], None).is_err());
    assert!(
        Sequential::new()
            .add_lp_pool3d(2.0, [2, 2, 2], Some([1, 0, 1]))
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_lp_pool3d(2.0, [usize::MAX, 2, 2], None)
            .is_err()
    );
}

#[test]
fn rank_is_checked_at_forward() {
    let model = Sequential::new().add_lp_pool2d(2.0, [2, 2], None).unwrap();
    assert!(model.predict(&ramp(&[4, 4])).is_err());
    assert!(model.predict(&ramp(&[1, 1, 4])).is_err());
}

#[test]
fn hand_computed_examples() {
    let m1 = Sequential::new().add_lp_pool1d(1.0, 2, None).unwrap();
    let y = m1
        .predict(&tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]))
        .unwrap();
    assert_eq!(bits(&y), bits(&tensor(vec![3.0, 7.0], &[1, 1, 2])));
    let m2 = Sequential::new().add_lp_pool2d(2.0, [2, 2], None).unwrap();
    let y = m2
        .predict(&tensor(vec![1.0, 2.0, 2.0, 4.0], &[1, 1, 2, 2]))
        .unwrap();
    assert!((y.as_slice().unwrap()[0] - 5.0).abs() < 1e-6);
}

#[test]
fn var_method_matches_layer_path_bit_exact() {
    let x = ramp(&[1, 2, 4, 4, 4]);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_var = xv
        .lp_pool3d(1.5, [2, 2, 2], Some([1, 2, 2]), false)
        .unwrap();
    let via_layer = Sequential::new()
        .add_lp_pool3d(1.5, [2, 2, 2], Some([1, 2, 2]))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(bits(&via_var.to_tensor()), bits(&via_layer));
    let x1 = ramp(&[2, 3, 8]);
    let v1 = tape.var(&x1).lp_pool1d(2.0, 3, Some(2), false).unwrap();
    let l1 = Sequential::new()
        .add_lp_pool1d(2.0, 3, Some(2))
        .unwrap()
        .predict(&x1)
        .unwrap();
    assert_eq!(bits(&v1.to_tensor()), bits(&l1));
}

#[test]
fn mixed_model_predict_matches_tape_forward() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 4, 4]);
    let via_predict = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 2]);
    assert_eq!(bits(&via_predict), bits(&via_forward));
}

#[test]
fn fit_updates_parameters_through_lp_pool() {
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    assert_eq!(model.trainable_parameters().len(), 4);
    let x = ramp(&[4, 1, 4, 4]);
    let y = ramp(&[4, 2]);
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
    let model = mixed_model().add_linear(2, 2, 13).unwrap();
    let x = ramp(&[2, 1, 4, 4]);
    let guard = TempDirGuard::new("lp_pool_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("lp_pool2d"), "{manifest}");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );
    // 1d・3d（stride の Some／None）も往復する。
    let m13 = Sequential::new()
        .add_lp_pool1d(1.5, 2, Some(1))
        .unwrap()
        .add_lp_pool1d(3.0, 2, None)
        .unwrap();
    let dir13 = guard.path().join("m13");
    save_model(&m13, &dir13).expect("保存できるはず");
    let l13 = load_model(&dir13).expect("復元できるはず");
    let x1 = ramp(&[1, 2, 9]);
    assert_eq!(
        bits(&m13.predict(&x1).unwrap()),
        bits(&l13.predict(&x1).unwrap())
    );
    let m3 = Sequential::new()
        .add_lp_pool3d(2.0, [2, 2, 2], Some([1, 2, 1]))
        .unwrap();
    let dir3 = guard.path().join("m3");
    save_model(&m3, &dir3).expect("保存できるはず");
    let l3 = load_model(&dir3).expect("復元できるはず");
    let x3 = ramp(&[1, 1, 3, 4, 3]);
    assert_eq!(
        bits(&m3.predict(&x3).unwrap()),
        bits(&l3.predict(&x3).unwrap())
    );
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = Sequential::new().add_lp_pool1d(2.0, 2, None).unwrap();
    let guard = TempDirGuard::new("lp_pool_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    assert!(original.contains("\"norm_type\""), "{original}");
    std::fs::write(&path, original.replace("\"norm_type\"", "\"norm_typex\"")).unwrap();
    assert!(load_model(&dir).is_err());
    // norm_type を 0 にすると add_lp_pool1d 経由で型付きエラーになる。
    let zeroed = original.replacen("\"norm_type\":2", "\"norm_type\":0", 1);
    assert_ne!(zeroed, original, "norm_type の表記が想定と違う: {original}");
    std::fs::write(&path, zeroed).unwrap();
    assert!(load_model(&dir).is_err());
}

#[test]
fn onnx_export_rejects_lp_pool_layers() {
    for model in [
        Sequential::new().add_lp_pool1d(2.0, 2, None).unwrap(),
        Sequential::new().add_lp_pool2d(2.0, [2, 2], None).unwrap(),
        Sequential::new()
            .add_lp_pool3d(2.0, [2, 2, 2], None)
            .unwrap(),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
