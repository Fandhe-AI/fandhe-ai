//! `compat::Sequential::add_adaptive_max_pool2d`／`add_adaptive_max_pool1d`／`add_global_pool` と
//! `Var::adaptive_max_pool2d`／`adaptive_max_pool1d`（イシュー #2527・親 #2520・ルート #2499 の
//! 一括承認。`docs/autodiff-adaptive-max-global-pool-decision.md` §6・§8）の facade 公開面を
//! CPU で検証する統合テスト。
//!
//! 3 層は学習可能パラメータを持たない無状態層で、内部実装（`nn::AdaptiveMaxPool2d`／
//! `AdaptiveMaxPool1d`／`GlobalPool`。#2160）の薄い委譲である。facade 公開 API だけで次を固定する。
//!
//! - 構築検査（`output_size` の 0 の即時拒否）と forward 時の遅延検査（rank 不一致）
//! - 数値: `Var` 直接呼び出しと層経路の bit 一致・`GlobalPool` と 1 出力 adaptive pooling の bit 一致
//! - 学習経路（`bind`／`trainable_parameters`／`trainable_vars`／`trainable_grads`／
//!   `apply_parameters`／`fit`）と backward の bit 一致
//! - 常駐経路は Pooling 層として `Unsupported`（fail-closed）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄の型付き拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_adaptive_max_global_pool_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, GlobalPoolMode, Tensor};

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

/// `Conv2d → ReLU → AdaptiveMaxPool2d([2, 2]) → Flatten → Linear` の混在モデル
/// （入力 `[2, 1, 5, 5]`・出力 `[2, 3]`）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 11)
        .unwrap()
        .add_relu()
        .add_adaptive_max_pool2d([2, 2])
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(16, 3, 12)
        .unwrap()
}

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn zero_output_size_is_rejected_at_construction() {
    for r in [
        Sequential::new().add_adaptive_max_pool2d([0, 2]),
        Sequential::new().add_adaptive_max_pool2d([2, 0]),
        Sequential::new().add_adaptive_max_pool1d(0),
    ] {
        assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
    }
}

#[test]
fn rank_mismatch_is_checked_at_forward() {
    let m2 = Sequential::new().add_adaptive_max_pool2d([2, 2]).unwrap();
    assert!(m2.predict(&ramp(&[1, 1, 4], 1.0)).is_err());
    assert!(m2.predict(&ramp(&[4, 4], 1.0)).is_err());
    let m1 = Sequential::new().add_adaptive_max_pool1d(2).unwrap();
    assert!(m1.predict(&ramp(&[1, 1, 4, 4], 1.0)).is_err());
    let g = Sequential::new().add_global_pool(GlobalPoolMode::Max, true);
    assert!(g.predict(&ramp(&[2, 3], 1.0)).is_err());
    assert!(g.predict(&ramp(&[1, 1, 2, 2, 2], 1.0)).is_err());
}

// ---------------------------------------------------------------------
// 数値
// ---------------------------------------------------------------------

#[test]
fn layer_path_matches_direct_var_call_bit_exact() {
    // 割り切れる形・重なり窓・拡大の 3 形状。
    for (in_hw, out) in [([4, 4], [2, 2]), ([5, 7], [2, 3]), ([2, 3], [3, 4])] {
        let x = ramp(&[2, 3, in_hw[0], in_hw[1]], 1.0);
        let model = Sequential::new().add_adaptive_max_pool2d(out).unwrap();
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let (direct, _idx) = xv.adaptive_max_pool2d(out).unwrap();
        let via_predict = model.predict(&x).unwrap();
        let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
        assert_eq!(bits(&via_predict), bits(&direct.to_tensor()));
        assert_eq!(bits(&via_forward), bits(&direct.to_tensor()));
    }
    for (l, out) in [(8, 4), (7, 3), (3, 5)] {
        let x = ramp(&[2, 3, l], 1.0);
        let model = Sequential::new().add_adaptive_max_pool1d(out).unwrap();
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let (direct, _idx) = xv.adaptive_max_pool1d(out).unwrap();
        assert_eq!(bits(&model.predict(&x).unwrap()), bits(&direct.to_tensor()));
    }
}

#[test]
fn global_pool_shapes_and_values_for_all_modes() {
    for (rank, shape) in [(3, vec![2, 3, 5]), (4, vec![2, 3, 4, 5])] {
        let x = ramp(&shape, 1.0);
        for mode in [GlobalPoolMode::Avg, GlobalPoolMode::Max] {
            for keepdims in [true, false] {
                let y = Sequential::new()
                    .add_global_pool(mode, keepdims)
                    .predict(&x)
                    .unwrap();
                let expected_shape: Vec<usize> = match (rank, keepdims) {
                    (3, true) => vec![2, 3, 1],
                    (4, true) => vec![2, 3, 1, 1],
                    _ => vec![2, 3],
                };
                assert_eq!(y.shape(), expected_shape.as_slice());
                // 1 出力の adaptive pooling（keepdims 形）と bit 一致する。
                let reference = match (rank, mode) {
                    (3, GlobalPoolMode::Avg) => {
                        Sequential::new().add_adaptive_avg_pool1d(1).unwrap()
                    }
                    (3, _) => Sequential::new().add_adaptive_max_pool1d(1).unwrap(),
                    (_, GlobalPoolMode::Avg) => {
                        Sequential::new().add_adaptive_avg_pool2d([1, 1]).unwrap()
                    }
                    (_, _) => Sequential::new().add_adaptive_max_pool2d([1, 1]).unwrap(),
                }
                .predict(&x)
                .unwrap();
                assert_eq!(
                    y.contiguous().as_slice().unwrap().len(),
                    reference.contiguous().as_slice().unwrap().len()
                );
                assert_eq!(bits(&y), bits(&reference), "{mode:?} keepdims={keepdims}");
            }
        }
    }
}

#[test]
fn stateless_pooling_is_train_eval_independent() {
    let x = ramp(&[1, 2, 4, 4], 1.0);
    let mut model = Sequential::new()
        .add_adaptive_max_pool2d([2, 2])
        .unwrap()
        .add_global_pool(GlobalPoolMode::Avg, false);
    let eval = model.predict(&x).unwrap();
    model.train();
    assert_eq!(bits(&model.predict(&x).unwrap()), bits(&eval));
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

    let x = ramp(&[2, 1, 5, 5], 1.0);
    let target = ramp(&[2, 3], 0.5);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    assert_eq!(bound.trainable_vars().len(), param_count);
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(bound.trainable_grads(&grads).unwrap().len(), param_count);
}

#[test]
fn backward_input_grad_matches_direct_graph_bit_exact() {
    let x = ramp(&[2, 2, 5, 5], 1.0);
    let grad_of = |layer: bool, global: bool| -> Vec<u32> {
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let y = match (layer, global) {
            (true, false) => Sequential::new()
                .add_adaptive_max_pool2d([2, 3])
                .unwrap()
                .forward(&tape, &xv)
                .unwrap(),
            (false, false) => xv.adaptive_max_pool2d([2, 3]).unwrap().0,
            (true, true) => Sequential::new()
                .add_global_pool(GlobalPoolMode::Max, false)
                .forward(&tape, &xv)
                .unwrap(),
            (false, true) => xv
                .adaptive_max_pool2d([1, 1])
                .unwrap()
                .0
                .reshape(&[2, 2])
                .unwrap(),
        };
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        bits(grads.get(&xv).unwrap().expect("入力の勾配があるはず"))
    };
    assert_eq!(grad_of(true, false), grad_of(false, false));
    assert_eq!(grad_of(true, true), grad_of(false, true));
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
fn fit_reduces_loss_through_adaptive_max_pool() {
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let x = ramp(&[4, 1, 5, 5], 1.0);
    let y = ramp(&[4, 3], 0.5);
    let before: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    let history = model.fit(&x, &y, FitConfig::new(8, 2)).unwrap();
    assert_eq!(history.loss.len(), 8);
    assert!(
        history.loss.iter().all(|l| l.is_finite()),
        "{:?}",
        history.loss
    );
    assert!(
        history.loss.last().unwrap() < history.loss.first().unwrap(),
        "loss が減少するはず: {:?}",
        history.loss
    );
    let after: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    assert_ne!(before, after);
}

// ---------------------------------------------------------------------
// 常駐経路（Pooling 層として fail-closed）
// ---------------------------------------------------------------------

/// 常駐経路が拒否すべき、3 層をそれぞれ含むモデルと入力。
fn resident_rejected_cases() -> Vec<(Sequential, Tensor<f32>)> {
    let linear_head =
        |s: Sequential, feat: usize| s.add_flatten(1, 3).add_linear(feat, 2, 5).unwrap();
    vec![
        (
            linear_head(
                Sequential::new().add_adaptive_max_pool2d([2, 2]).unwrap(),
                8,
            ),
            ramp(&[2, 2, 4, 4], 1.0),
        ),
        (
            Sequential::new()
                .add_adaptive_max_pool1d(2)
                .unwrap()
                .add_flatten(1, 2)
                .add_linear(6, 2, 5)
                .unwrap(),
            ramp(&[2, 3, 4], 1.0),
        ),
        (
            linear_head(
                Sequential::new().add_global_pool(GlobalPoolMode::Max, true),
                2,
            ),
            ramp(&[2, 2, 4, 4], 1.0),
        ),
    ]
}

#[test]
fn resident_path_rejects_pooling_layers_fail_closed() {
    for (model, x) in resident_rejected_cases() {
        let tape = fandhe_ai::tape();
        let err = model.init_device_param_store(&tape).unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");

        // predict_resident／forward_resident も、別モデル由来の store で到達しても拒否される。
        let store_tape = fandhe_ai::tape();
        let mut store = Sequential::new()
            .add_linear(2, 2, 1)
            .unwrap()
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

// ---------------------------------------------------------------------
// save_model／load_model
// ---------------------------------------------------------------------

#[test]
fn save_and_load_round_trip_is_bit_identical() {
    let model = Sequential::new()
        .add_conv2d(1, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 11)
        .unwrap()
        .add_adaptive_max_pool2d([3, 2])
        .unwrap()
        .add_global_pool(GlobalPoolMode::Avg, true)
        .add_global_pool(GlobalPoolMode::Max, false)
        .add_linear(4, 3, 12)
        .unwrap();
    let model1d = Sequential::new()
        .add_adaptive_max_pool1d(3)
        .unwrap()
        .add_global_pool(GlobalPoolMode::Max, true);
    let x = ramp(&[2, 1, 5, 5], 1.0);
    let x1 = ramp(&[2, 2, 7], 1.0);
    for (name, m, input) in [("2d", &model, &x), ("1d", &model1d, &x1)] {
        let guard = TempDirGuard::new(&format!("adaptive_max_global_pool_round_trip_{name}"));
        let dir = guard.path().join("m");
        save_model(m, &dir).expect("保存できるはず");
        let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        assert!(manifest.contains("global_pool"), "{manifest}");
        let loaded = load_model(&dir).expect("復元できるはず");
        assert_eq!(
            bits(&m.predict(input).unwrap()),
            bits(&loaded.predict(input).unwrap())
        );
    }
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = Sequential::new().add_global_pool(GlobalPoolMode::Max, true);
    let guard = TempDirGuard::new("adaptive_max_global_pool_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    for (from, to) in [
        ("\"max\"", "\"min\""),
        ("\"keepdims\"", "\"keepdimsx\""),
        ("true", "1"),
    ] {
        assert!(original.contains(from), "{from} not in {original}");
        std::fs::write(&path, original.replace(from, to)).unwrap();
        assert!(load_model(&dir).is_err(), "{from} → {to} は拒否されるはず");
    }
}

// ---------------------------------------------------------------------
// ONNX export（対象外。現行挙動の固定）
// ---------------------------------------------------------------------

#[test]
fn onnx_export_rejects_adaptive_max_and_global_pool_layers() {
    for model in [
        Sequential::new().add_adaptive_max_pool2d([2, 2]).unwrap(),
        Sequential::new().add_adaptive_max_pool1d(2).unwrap(),
        Sequential::new().add_global_pool(GlobalPoolMode::Avg, true),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
