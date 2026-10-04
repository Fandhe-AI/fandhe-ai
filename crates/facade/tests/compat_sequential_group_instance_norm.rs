//! `compat::Sequential::add_group_norm`／`add_instance_norm`（イシュー #2525・親 #2520・
//! ルート #2499 の 2026-10-04 一括承認。`docs/norm-ops-design.md` §11）の facade 公開面を
//! CPU で検証する統合テスト。
//!
//! 2 層は学習可能パラメータ（affine）を持たない無状態層で、内部実装（`nn::GroupNorm`／
//! `nn::InstanceNorm`。#2066）の薄い委譲である。本ファイルは次を facade 公開 API だけで固定する。
//!
//! - 構築検査（`groups == 0`・非有限／負の `eps`）と forward 時の遅延検査（チャネル非整除・rank）
//! - 数値: `Var::reshape → layer_norm → reshape` の手組みと bit 一致・`InstanceNorm == GroupNorm(C)`
//! - `predict`（host 経路）と `forward`（外部 `Tape` 経路）の bit 一致
//! - 学習経路（`bind`／`trainable_parameters`／`trainable_grads`／`apply_parameters`／`fit`）
//! - 常駐経路は fail-closed（`BackendError::Unsupported`。決定は `docs/norm-ops-design.md` §11.x）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄の型付き拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_group_instance_norm_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, Tensor};

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

/// `Conv2d → GroupNorm → ReLU → InstanceNorm → Flatten → Linear` の混在モデル
/// （入力 `[2, 1, 4, 4]`・出力 `[2, 3]`）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 11)
        .unwrap()
        .add_group_norm(2, 1e-5)
        .unwrap()
        .add_relu()
        .add_instance_norm(1e-5)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(64, 3, 12)
        .unwrap()
}

/// 常駐対応の対照モデル（GroupNorm を含まない）。
fn resident_ok_model() -> Sequential {
    Sequential::new()
        .add_flatten(1, 3)
        .add_linear(16, 4, 21)
        .unwrap()
        .add_relu()
        .add_linear(4, 2, 22)
        .unwrap()
}

/// 常駐拒否の確認対象（GroupNorm／InstanceNorm をそれぞれ含む）。
fn resident_ng_models() -> Vec<Sequential> {
    vec![
        Sequential::new()
            .add_flatten(1, 3)
            .add_linear(16, 4, 21)
            .unwrap()
            .add_group_norm(2, 1e-5)
            .unwrap(),
        Sequential::new()
            .add_instance_norm(1e-5)
            .unwrap()
            .add_flatten(1, 3)
            .add_linear(16, 4, 21)
            .unwrap(),
    ]
}

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn constructors_reject_invalid_arguments() {
    assert!(matches!(
        Sequential::new().add_group_norm(0, 1e-5),
        Err(AutodiffError::InvalidArgument(_))
    ));
    for eps in [f32::NAN, f32::INFINITY, -1e-5] {
        assert!(matches!(
            Sequential::new().add_group_norm(2, eps),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            Sequential::new().add_instance_norm(eps),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}

#[test]
fn indivisible_channels_are_rejected_at_forward() {
    let model = Sequential::new().add_group_norm(2, 1e-5).unwrap();
    let err = model.predict(&ramp(&[1, 3, 2, 2], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn rank_mismatch_is_rejected_at_forward() {
    let g = Sequential::new().add_group_norm(1, 1e-5).unwrap();
    let err = g.predict(&ramp(&[4], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)), "{err:?}");
    let i = Sequential::new().add_instance_norm(1e-5).unwrap();
    let err = i.predict(&ramp(&[2, 4], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)), "{err:?}");
}

// ---------------------------------------------------------------------
// 数値
// ---------------------------------------------------------------------

#[test]
fn group_norm_matches_manual_reshape_layer_norm_bit_exact() {
    let (n, c, h, w, g) = (2usize, 4usize, 3usize, 3usize, 2usize);
    let x = ramp(&[n, c, h, w], 2.0);
    let model = Sequential::new().add_group_norm(g, 1e-5).unwrap();
    let got = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let manual = xv
        .reshape(&[n * g, (c / g) * h * w])
        .unwrap()
        .layer_norm(None, None, 1e-5)
        .unwrap()
        .reshape(&[n, c, h, w])
        .unwrap()
        .to_tensor();
    assert_eq!(bits(&got), bits(&manual));
}

#[test]
fn instance_norm_equals_group_norm_with_groups_eq_channels() {
    let x = ramp(&[2, 4, 3, 3], 2.0);
    let a = Sequential::new()
        .add_instance_norm(1e-5)
        .unwrap()
        .predict(&x)
        .unwrap();
    let b = Sequential::new()
        .add_group_norm(4, 1e-5)
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn group_norm_output_has_zero_mean_and_unit_variance_per_group() {
    let x = ramp(&[1, 4, 3, 3], 3.0);
    let y = Sequential::new()
        .add_group_norm(2, 1e-5)
        .unwrap()
        .predict(&x)
        .unwrap();
    let v = y.contiguous().as_slice().unwrap().to_vec();
    for group in v.chunks(18) {
        let mean = group.iter().sum::<f32>() / 18.0;
        let var = group.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / 18.0;
        assert!(mean.abs() < 1e-4, "mean={mean}");
        assert!((var - 1.0).abs() < 1e-3, "var={var}");
    }
}

#[test]
fn mixed_model_predict_matches_tape_forward() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 4, 4], 1.0);
    let via_predict = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 3]);
    assert_eq!(bits(&via_predict), bits(&via_forward));
}

// ---------------------------------------------------------------------
// 学習経路
// ---------------------------------------------------------------------

#[test]
fn stateless_layers_do_not_add_trainable_parameters_and_grads_match() {
    let model = mixed_model();
    // Conv2d(weight, bias) + Linear(weight, bias) の 4 件だけ。
    let param_count = model.trainable_parameters().len();
    assert_eq!(param_count, 4);

    let x = ramp(&[2, 1, 4, 4], 1.0);
    let target = ramp(&[2, 3], 0.5);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    assert_eq!(bound.trainable_vars().len(), param_count);
    let grads = tape.backward(&loss).unwrap();
    let grad_refs = bound.trainable_grads(&grads).unwrap();
    assert_eq!(grad_refs.len(), param_count);
    for g in grad_refs {
        assert!(g.as_slice().unwrap().iter().all(|v| v.is_finite()));
    }
}

#[test]
fn apply_parameters_succeeds_with_stateless_layers() {
    let mut model = mixed_model();
    let updated: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| {
            tensor(
                p.contiguous()
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|v| v + 0.25)
                    .collect(),
                p.shape(),
            )
        })
        .collect();
    model.apply_parameters(updated).unwrap();
}

#[test]
fn fit_updates_parameters_through_norm_layers() {
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let x = ramp(&[4, 1, 4, 4], 1.0);
    let y = ramp(&[4, 3], 0.5);
    let before: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    let history = model.fit(&x, &y, FitConfig::new(3, 2)).unwrap();
    assert_eq!(history.loss.len(), 3);
    assert!(
        history.loss.iter().all(|l| l.is_finite()),
        "{:?}",
        history.loss
    );
    let after: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    assert_ne!(before, after, "fit でパラメータが更新されるはず");
}

// ---------------------------------------------------------------------
// 常駐経路（fail-closed）
// ---------------------------------------------------------------------

#[test]
fn resident_path_rejects_group_and_instance_norm() {
    let x = ramp(&[2, 1, 4, 4], 1.0);
    for model in resident_ng_models() {
        let tape = fandhe_ai::tape();
        let err = model.init_device_param_store(&tape).unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");

        // 常駐対応モデルのストアを借りて predict／forward の入口も拒否されることを見る。
        let store_tape = fandhe_ai::tape();
        let mut store = resident_ok_model()
            .init_device_param_store(&store_tape)
            .unwrap();
        drop(store_tape);
        let err = model.predict_resident(&store, &x).unwrap_err();
        assert!(
            matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
            "{err:?}"
        );
        let tape2 = fandhe_ai::tape();
        let xv = tape2.var(&x);
        let err = model
            .forward_resident(&tape2, &xv, &mut store)
            .expect_err("forward_resident も拒否されるはず");
        assert!(
            matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
            "{err:?}"
        );
    }
}

#[test]
fn resident_path_still_accepts_model_without_norm_layers() {
    let model = resident_ok_model();
    let x = ramp(&[2, 1, 4, 4], 1.0);
    let tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&tape).unwrap();
    drop(tape);
    let via_resident = model.predict_resident(&store, &x).unwrap();
    assert_eq!(bits(&via_resident), bits(&model.predict(&x).unwrap()));
}

// ---------------------------------------------------------------------
// save_model／load_model
// ---------------------------------------------------------------------

#[test]
fn save_and_load_round_trip_is_bit_identical() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 4, 4], 1.0);
    let guard = TempDirGuard::new("group_instance_norm_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("group_norm"), "{manifest}");
    assert!(manifest.contains("instance_norm"), "{manifest}");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = Sequential::new().add_group_norm(2, 1e-5).unwrap();
    let guard = TempDirGuard::new("group_instance_norm_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    for (from, to) in [
        // 必須キーの欠落（eps → epsx で未知キー＋欠落）。
        ("\"eps\"", "\"epsx\""),
        ("\"groups\"", "\"groupz\""),
    ] {
        assert!(original.contains(from), "{original}");
        std::fs::write(&path, original.replace(from, to)).unwrap();
        assert!(load_model(&dir).is_err(), "{from} → {to} は拒否されるはず");
    }
}

// ---------------------------------------------------------------------
// ONNX export（対象外。現行挙動の固定）
// ---------------------------------------------------------------------

#[test]
fn onnx_export_rejects_group_and_instance_norm() {
    for model in [
        Sequential::new().add_group_norm(2, 1e-5).unwrap(),
        Sequential::new().add_instance_norm(1e-5).unwrap(),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
