//! `compat::Sequential::add_mish`／`add_hardtanh`／`add_relu6`／`add_glu`／`add_prelu`
//! （イシュー #2529・親 #2520・ルート #2499 の 2026-10-04 一括承認。
//! `docs/autodiff-activation-ops-decision.md` §2.1・§9）の facade 公開面を CPU で検証する統合テスト。
//!
//! 5 層は内部実装（`nn::activation::{Mish, Hardtanh, Relu6, Glu, PRelu}`。#2146）の薄い委譲で、
//! 状態を持つのは `PRelu`（`weight`）だけ。facade 公開 API だけで次を固定する。
//!
//! - 構築検査（Hardtanh の NaN・`min >= max`、PRelu の `num_parameters == 0`）と forward 時の検査
//!   （Glu の軸・奇数長）
//! - 数値: `Var::mish` 等の直接呼び出しと層経路（`predict`／`forward`）の bit 一致
//! - 学習経路（PRelu: `bind`／`trainable_parameters`／`trainable_vars`／`trainable_grads`／
//!   `apply_parameters`／`state_dict`／`fit`）
//! - 常駐経路: 無状態 4 層は通過・PRelu は `Unsupported`（fail-closed）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄・保存不能値（Hardtanh の ±inf）の
//!   型付き拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_activation_layers_backend_parity.rs`（`#[ignore]`）。

#![cfg(unix)]

use fandhe_ai::compat::{
    FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model,
};
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

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn invalid_hardtanh_and_prelu_arguments_are_rejected_at_construction() {
    for (lo, hi) in [(f32::NAN, 1.0), (-1.0, f32::NAN), (1.0, 1.0), (2.0, -2.0)] {
        assert!(
            matches!(
                Sequential::new().add_hardtanh(lo, hi),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "({lo}, {hi})"
        );
    }
    // ±inf は構築できる（保存は拒否される。下の save テスト参照）。
    assert!(
        Sequential::new()
            .add_hardtanh(f32::NEG_INFINITY, 1.0)
            .is_ok()
    );
    assert!(matches!(
        Sequential::new().add_prelu(0, 0.25),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // init は検証せず IEEE のまま扱う。
    assert!(Sequential::new().add_prelu(1, f32::NAN).is_ok());
}

#[test]
fn glu_axis_and_odd_length_are_rejected_at_forward_time() {
    let ok = Sequential::new().add_glu(1);
    assert_eq!(ok.predict(&ramp(&[2, 4], 1.0)).unwrap().shape(), &[2, 2]);
    // 奇数長・範囲外の軸は構築ではなく forward で型付きエラーになる。
    assert!(ok.predict(&ramp(&[2, 3], 1.0)).is_err());
    assert!(
        Sequential::new()
            .add_glu(5)
            .predict(&ramp(&[2, 4], 1.0))
            .is_err()
    );
}

// ---------------------------------------------------------------------
// 数値（Var 直接呼び出しとの bit 一致）
// ---------------------------------------------------------------------

#[test]
fn stateless_layers_match_var_methods_bit_for_bit() {
    let x = ramp(&[3, 4], 4.0);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);

    let cases: Vec<(&str, Sequential, Tensor<f32>)> = vec![
        (
            "mish",
            Sequential::new().add_mish(),
            xv.mish().unwrap().to_tensor(),
        ),
        (
            "hardtanh",
            Sequential::new().add_hardtanh(-1.0, 2.0).unwrap(),
            xv.hardtanh(-1.0, 2.0).unwrap().to_tensor(),
        ),
        (
            "relu6",
            Sequential::new().add_relu6(),
            xv.relu6().unwrap().to_tensor(),
        ),
        (
            "glu",
            Sequential::new().add_glu(1),
            xv.glu(1).unwrap().to_tensor(),
        ),
    ];
    for (name, model, direct) in cases {
        let via_predict = model.predict(&x).unwrap();
        let tape2 = fandhe_ai::tape();
        let xv2 = tape2.var(&x);
        let via_forward = model.forward(&tape2, &xv2).unwrap().to_tensor();
        assert_eq!(bits(&via_predict), bits(&direct), "{name}: predict");
        assert_eq!(bits(&via_forward), bits(&direct), "{name}: forward");
    }
}

#[test]
fn prelu_matches_var_prelu_with_same_weight() {
    let x = ramp(&[3, 4], 4.0);
    let model = Sequential::new().add_prelu(4, 0.2).unwrap();
    let tape = fandhe_ai::tape();
    let w = tape.var(&model.trainable_parameters()[0].clone());
    let direct = tape.var(&x).prelu(&w).unwrap().to_tensor();
    assert_eq!(bits(&model.predict(&x).unwrap()), bits(&direct));
    // 単一パラメータ（全チャネル共有）。
    let shared = Sequential::new().add_prelu(1, 0.25).unwrap();
    let y = shared
        .predict(&tensor(vec![-4.0, 2.0, -8.0, 1.0], &[2, 2]))
        .unwrap();
    assert_eq!(y.host_slice().into_owned(), vec![-1.0, 2.0, -2.0, 1.0]);
}

// ---------------------------------------------------------------------
// 学習経路（PRelu）
// ---------------------------------------------------------------------

/// `Linear → PRelu → Linear` の最小学習モデル（入力 `[B, 2]`・出力 `[B, 1]`）。
fn prelu_model() -> Sequential {
    Sequential::new()
        .add_linear(2, 3, 11)
        .unwrap()
        .add_prelu(3, 0.25)
        .unwrap()
        .add_linear(3, 1, 12)
        .unwrap()
}

#[test]
fn prelu_is_tracked_as_one_trainable_parameter_in_layer_order() {
    let model = Sequential::new()
        .add_linear(2, 3, 1)
        .unwrap()
        .add_mish()
        .add_prelu(3, 0.1)
        .unwrap()
        .add_hardtanh(-1.0, 1.0)
        .unwrap()
        .add_linear(3, 2, 2)
        .unwrap();
    // Linear(weight, bias) + PRelu(weight) + Linear(weight, bias)。
    let params = model.trainable_parameters();
    assert_eq!(params.len(), 5);
    assert_eq!(params[2].shape(), &[3]);
    assert_eq!(params[2].host_slice().into_owned(), vec![0.1; 3]);

    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let vars = bound.trainable_vars();
    assert_eq!(vars.len(), 5);
    assert_eq!(vars[2].to_tensor().shape(), &[3]);

    // state_dict のキーは `{index}.weight`。
    let sd = model.state_dict();
    assert!(sd.contains_key("2.weight"), "{:?}", sd.keys());
    assert_eq!(sd["2.weight"].shape(), &[3]);
}

#[test]
fn prelu_model_grads_flow_and_apply_parameters_updates_weight() {
    let mut model = prelu_model();
    let x = ramp(&[5, 2], 2.0);
    let target = ramp(&[5, 1], 0.5);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let g = bound.trainable_grads(&grads).unwrap();
    // 位置対応（bind 順 = trainable_parameters 順 = trainable_vars 順）。
    assert_eq!(g.len(), 5);
    assert_eq!(g[2].shape(), &[3]);
    assert!(
        g[2].contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .any(|v| *v != 0.0),
        "PRelu weight に勾配が届くはず（汎用 forward へのフォールバックで葉が作り直されていない）"
    );
    drop(bound);

    let before = bits(model.trainable_parameters()[2]);
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
    assert_ne!(before, bits(model.trainable_parameters()[2]));

    // shape を変える置換は拒否される。
    let mut bad: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| (*p).clone())
        .collect();
    bad[2] = tensor(vec![0.0; 4], &[4]);
    assert!(model.apply_parameters(bad).is_err());
}

#[test]
fn state_dict_round_trips_prelu_weight() {
    let mut model = prelu_model();
    let mut sd = model.state_dict();
    sd.insert("1.weight".to_string(), tensor(vec![0.5, -0.5, 0.125], &[3]));
    model.load_state_dict(sd).unwrap();
    assert_eq!(
        model.state_dict()["1.weight"].host_slice().into_owned(),
        vec![0.5, -0.5, 0.125]
    );
}

#[test]
fn fit_updates_prelu_weight_and_decreases_loss() {
    let mut model = prelu_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let x = ramp(&[8, 2], 2.0);
    let y = ramp(&[8, 1], 0.5);
    let before = bits(model.trainable_parameters()[2]);
    let history = model.fit(&x, &y, FitConfig::new(30, 8)).unwrap();
    assert_eq!(history.loss.len(), 30);
    assert!(
        history.loss.iter().all(|l| l.is_finite()),
        "{:?}",
        history.loss
    );
    assert!(
        history.loss.last().unwrap() < history.loss.first().unwrap(),
        "loss が下がるはず: {:?}",
        history.loss
    );
    assert_ne!(before, bits(model.trainable_parameters()[2]));
}

// ---------------------------------------------------------------------
// 常駐経路
// ---------------------------------------------------------------------

#[test]
fn resident_path_accepts_stateless_layers_and_matches_predict() {
    let model = Sequential::new()
        .add_linear(3, 4, 5)
        .unwrap()
        .add_mish()
        .add_hardtanh(-1.0, 1.0)
        .unwrap()
        .add_relu6()
        .add_glu(1)
        .add_linear(2, 2, 6)
        .unwrap();
    let x = ramp(&[2, 3], 3.0);
    let tape = fandhe_ai::tape();
    let store = model
        .init_device_param_store(&tape)
        .expect("無状態の活性化層は常駐経路を通過するはず");
    let resident = model.predict_resident(&store, &x).unwrap();
    let host = model.predict(&x).unwrap();
    assert_eq!(bits(&resident), bits(&host));
}

#[test]
fn resident_path_rejects_prelu_fail_closed() {
    let model = prelu_model();
    let x = ramp(&[2, 2], 1.0);

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

// ---------------------------------------------------------------------
// save_model／load_model
// ---------------------------------------------------------------------

fn all_layers_model() -> Sequential {
    let mut m = Sequential::new()
        .add_linear(3, 8, 1)
        .unwrap()
        .add_mish()
        .add_hardtanh(-1.5, 2.5)
        .unwrap()
        .add_relu6()
        .add_prelu(8, 0.3)
        .unwrap()
        .add_glu(1)
        .add_linear(4, 2, 2)
        .unwrap();
    m.eval();
    m
}

#[test]
fn save_and_load_round_trip_is_bit_identical() {
    let model = all_layers_model();
    let x = ramp(&[4, 3], 3.0);
    let guard = TempDirGuard::new("activation_layers_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    for kind in ["mish", "hardtanh", "relu6", "prelu", "glu"] {
        assert!(
            manifest.contains(&format!("\"kind\":\"{kind}\"")),
            "{kind} in {manifest}"
        );
    }
    let loaded = load_model(&dir).expect("復元できるはず");
    for (k, v) in model.state_dict() {
        assert_eq!(bits(&v), bits(&loaded.state_dict()[&k]), "{k}");
    }
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );
}

#[test]
fn save_rejects_non_finite_hardtanh_bounds() {
    let model = Sequential::new()
        .add_hardtanh(f32::NEG_INFINITY, 1.0)
        .unwrap();
    let guard = TempDirGuard::new("hardtanh_inf_rejected");
    let dir = guard.path().join("m");
    let err = save_model(&model, &dir).unwrap_err();
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err:?}"
    );
    assert!(!dir.exists(), "拒否時は dir に何も作らない");
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = all_layers_model();
    let guard = TempDirGuard::new("activation_layers_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    for (from, to) in [
        // hardtanh: 型違い・キー過不足・NaN 相当の逆転境界
        ("\"min_val\":-1.5", "\"min_val\":\"-1.5\""),
        ("\"min_val\":-1.5", "\"min_val\":-1.5,\"extra\":1"),
        ("\"min_val\":-1.5", "\"min_value\":-1.5"),
        ("\"min_val\":-1.5", "\"min_val\":9.0"),
        // glu: 型違い
        ("\"dim\":1", "\"dim\":\"1\""),
        ("\"dim\":1", "\"dim\":-1"),
        // prelu: 0・型違い・weight 形状と不整合・巨大値
        ("\"num_parameters\":8", "\"num_parameters\":0"),
        ("\"num_parameters\":8", "\"num_parameters\":\"8\""),
        ("\"num_parameters\":8", "\"num_parameters\":9"),
        (
            "\"num_parameters\":8",
            "\"num_parameters\":18446744073709551615",
        ),
        ("\"num_parameters\":8", "\"num_parameters\":4294967296"),
    ] {
        assert!(original.contains(from), "{from} not in {original}");
        std::fs::write(&path, original.replacen(from, to, 1)).unwrap();
        assert!(load_model(&dir).is_err(), "{from} → {to} は拒否されるはず");
    }
}

// ---------------------------------------------------------------------
// ONNX export（対象外。現行挙動の固定）
// ---------------------------------------------------------------------

#[test]
fn onnx_export_rejects_new_activation_layers() {
    for model in [
        Sequential::new().add_mish(),
        Sequential::new().add_hardtanh(-1.0, 1.0).unwrap(),
        Sequential::new().add_relu6(),
        Sequential::new().add_glu(1),
        Sequential::new().add_prelu(2, 0.25).unwrap(),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
