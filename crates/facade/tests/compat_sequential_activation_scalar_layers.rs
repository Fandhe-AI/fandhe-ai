//! `compat::Sequential` の活性化 9 層（`add_selu`／`add_celu`／`add_softsign`／`add_hardsigmoid`／
//! `add_log_sigmoid`／`add_softmin`／`add_tanhshrink`／`add_threshold`／`add_rrelu`。イシュー #2679・
//! 親 #2625。承認はルート #2499 のコメント issuecomment-6033824965。
//! `docs/autodiff-activation-scalar-ops-decision.md` §7・`docs/autodiff-softmin-threshold-ops-decision.md` §7）の
//! facade 公開面を CPU で検証する統合テスト。
//!
//! 9 層はいずれも学習可能パラメータを持たない無状態層（内部実装は
//! `nn::activation::{Selu, Celu, Softsign, Hardsigmoid, LogSigmoid}` と
//! `nn::softmin_threshold::{Softmin, Tanhshrink, Threshold, RRelu}`）。facade 公開 API だけで次を固定する。
//!
//! - 構築検査（CELU の `alpha == 0`／非有限、RReLU の `lower > upper`／非有限）と forward 時の検査（Softmin の軸）
//! - 数値: 独立に書いた `f64` の閉形式参照と REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）で一致
//! - 学習経路: パラメータを増やさず（`trainable_parameters`／`trainable_vars`／`trainable_grads` が層を素通し）、
//!   `Linear` と組んだ `fit` で loss が下がる
//! - `set_training`／`eval` の RReLU への伝播（推論時は固定傾き `(lower + upper) / 2`）
//! - 常駐経路: 無状態層は通過し `predict` と bit 一致する・未対応層（`PRelu`）を混ぜると従来どおり `Unsupported`
//! - `save_model` は未対応（manifest の kind が未承認）のため `ModelIoError::UnsupportedModel` で拒否し、
//!   保存先に何も作らない
//!
//! 実機 parity は `compat_sequential_activation_scalar_layers_backend_parity.rs`（`#[ignore]`）。

#![cfg(unix)]

use std::sync::{Mutex, MutexGuard};

use fandhe_ai::compat::{FitConfig, Loss, ModelIoError, Optimizer, Sequential, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, Tensor};
use fandhe_ai_backend_cpu::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

mod common;
use common::temp_dir::TempDirGuard;

/// グローバル RNG（`manual_seed`）を触るテストの直列化ロック。
fn rng_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

fn ramp(shape: &[usize], scale: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(
        (0..n).map(|i| ((i as f32) * 0.37).sin() * scale).collect(),
        shape,
    )
}

/// REQ-2 統一複合判定。閾値は backend-cpu の定数を参照する（値は変更しない）。
fn assert_close(label: &str, actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 長さ");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let a = f64::from(*a);
        let diff = (a - e).abs();
        let scale = a.abs().max(e.abs()).max(1e-12);
        assert!(
            diff / scale < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD,
            "{label}[{i}]: actual={a} expected={e} diff={diff}"
        );
    }
}

// ---------------------------------------------------------------------
// 独立に書いた f64 の閉形式参照（PyTorch の定義式）
// ---------------------------------------------------------------------

const SELU_ALPHA: f64 = 1.673_263_242_354_377_3;
const SELU_SCALE: f64 = 1.050_700_987_355_480_5;

fn ref_selu(x: f64) -> f64 {
    SELU_SCALE * (x.max(0.0) + (SELU_ALPHA * (x.exp() - 1.0)).min(0.0))
}

fn ref_celu(x: f64, alpha: f64) -> f64 {
    x.max(0.0) + (alpha * ((x / alpha).exp() - 1.0)).min(0.0)
}

fn ref_softsign(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

fn ref_hardsigmoid(x: f64) -> f64 {
    ((x + 3.0).clamp(0.0, 6.0)) / 6.0
}

fn ref_log_sigmoid(x: f64) -> f64 {
    x.min(0.0) - (1.0 + (-x.abs()).exp()).ln()
}

fn ref_tanhshrink(x: f64) -> f64 {
    x - x.tanh()
}

fn ref_threshold(x: f64, threshold: f64, value: f64) -> f64 {
    if x > threshold { x } else { value }
}

fn input_values(x: &Tensor<f32>) -> Vec<f64> {
    values(x).into_iter().map(f64::from).collect()
}

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn invalid_celu_and_rrelu_arguments_are_rejected_at_construction() {
    for alpha in [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            matches!(
                Sequential::new().add_celu(alpha),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "celu alpha={alpha}"
        );
    }
    // 負の alpha は PyTorch と同じく受理する。
    assert!(Sequential::new().add_celu(-1.0).is_ok());
    for (lower, upper) in [
        (0.5, 0.1),
        (f32::NAN, 0.3),
        (0.1, f32::NAN),
        (f32::NEG_INFINITY, 0.3),
        (0.1, f32::INFINITY),
    ] {
        assert!(
            matches!(
                Sequential::new().add_rrelu(lower, upper),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "rrelu ({lower}, {upper})"
        );
    }
    assert!(Sequential::new().add_rrelu(0.125, 0.125).is_ok());
}

#[test]
fn softmin_axis_is_checked_at_forward_time_only() {
    let ok = Sequential::new().add_softmin(1);
    assert_eq!(ok.predict(&ramp(&[2, 4], 1.0)).unwrap().shape(), &[2, 4]);
    // 範囲外の軸は構築ではなく forward で型付きエラーになる（`add_softmax` と同じ遅延検査契約）。
    assert!(
        Sequential::new()
            .add_softmin(5)
            .predict(&ramp(&[2, 4], 1.0))
            .is_err()
    );
}

// ---------------------------------------------------------------------
// 数値（f64 閉形式参照との統一複合判定）と predict／forward の一致
// ---------------------------------------------------------------------

#[test]
fn layers_match_closed_form_references() {
    let x = ramp(&[3, 5], 6.0);
    let xs = input_values(&x);

    let cases: Vec<(&str, Sequential, Vec<f64>)> = vec![
        (
            "selu",
            Sequential::new().add_selu(),
            xs.iter().map(|v| ref_selu(*v)).collect(),
        ),
        (
            "celu(1.5)",
            Sequential::new().add_celu(1.5).unwrap(),
            xs.iter().map(|v| ref_celu(*v, 1.5)).collect(),
        ),
        (
            "celu(-0.75)",
            Sequential::new().add_celu(-0.75).unwrap(),
            xs.iter().map(|v| ref_celu(*v, -0.75)).collect(),
        ),
        (
            "softsign",
            Sequential::new().add_softsign(),
            xs.iter().map(|v| ref_softsign(*v)).collect(),
        ),
        (
            "hardsigmoid",
            Sequential::new().add_hardsigmoid(),
            xs.iter().map(|v| ref_hardsigmoid(*v)).collect(),
        ),
        (
            "log_sigmoid",
            Sequential::new().add_log_sigmoid(),
            xs.iter().map(|v| ref_log_sigmoid(*v)).collect(),
        ),
        (
            "tanhshrink",
            Sequential::new().add_tanhshrink(),
            xs.iter().map(|v| ref_tanhshrink(*v)).collect(),
        ),
        (
            "threshold(0.5, -2.0)",
            Sequential::new().add_threshold(0.5, -2.0),
            xs.iter().map(|v| ref_threshold(*v, 0.5, -2.0)).collect(),
        ),
    ];
    for (name, mut model, expected) in cases {
        model.eval();
        let via_predict = model.predict(&x).unwrap();
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
        assert_close(name, &values(&via_predict), &expected);
        // predict（`forward_host` 未提供の層は tape 経路へフォールバック）と forward は同じ値になる。
        assert_eq!(bits(&via_predict), bits(&via_forward), "{name}");
    }
}

#[test]
fn softmin_matches_softmax_of_negated_input_and_sums_to_one() {
    let x = ramp(&[4, 6], 3.0);
    let model = Sequential::new().add_softmin(1);
    let y = model.predict(&x).unwrap();
    let ys = values(&y);
    let xs = input_values(&x);
    let mut expected = Vec::with_capacity(xs.len());
    for row in xs.chunks(6) {
        let m = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let exps: Vec<f64> = row.iter().map(|v| (-(v - m)).exp()).collect();
        let sum: f64 = exps.iter().sum();
        expected.extend(exps.iter().map(|e| e / sum));
    }
    assert_close("softmin", &ys, &expected);
    for row in ys.chunks(6) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "行和 {sum}");
    }
}

// ---------------------------------------------------------------------
// 学習経路（無状態層はパラメータを増やさず勾配を通す）
// ---------------------------------------------------------------------

/// 各層を `Linear(2→3) → 層 → Linear(3→1)` に挟んだ最小学習モデルを返す。
fn sandwich(layer: &str) -> Sequential {
    let base = Sequential::new().add_linear(2, 3, 11).unwrap();
    let mid = match layer {
        "selu" => base.add_selu(),
        "celu" => base.add_celu(1.0).unwrap(),
        "softsign" => base.add_softsign(),
        "hardsigmoid" => base.add_hardsigmoid(),
        "log_sigmoid" => base.add_log_sigmoid(),
        "softmin" => base.add_softmin(1),
        "tanhshrink" => base.add_tanhshrink(),
        "threshold" => base.add_threshold(-10.0, 0.0),
        "rrelu" => base.add_rrelu(0.125, 0.3).unwrap(),
        other => panic!("未知の層: {other}"),
    };
    mid.add_linear(3, 1, 12).unwrap()
}

const LAYERS: [&str; 9] = [
    "selu",
    "celu",
    "softsign",
    "hardsigmoid",
    "log_sigmoid",
    "softmin",
    "tanhshrink",
    "threshold",
    "rrelu",
];

#[test]
fn stateless_layers_add_no_parameters_and_pass_gradients_through() {
    // RReLU は学習モードで共有 RNG を消費するため、シード再現性検査と排他する。
    let _guard = rng_lock();
    fandhe_ai::manual_seed(2679);
    for layer in LAYERS {
        let model = sandwich(layer);
        // Linear(weight, bias) × 2 のみ。9 層はパラメータを持たない。
        assert_eq!(model.trainable_parameters().len(), 4, "{layer}");
        assert_eq!(model.state_dict().len(), 4, "{layer}");

        let x = ramp(&[5, 2], 2.0);
        let target = ramp(&[5, 1], 0.5);
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        assert_eq!(bound.trainable_vars().len(), 4, "{layer}");
        let pred = bound.forward(&tape, &tape.var(&x)).unwrap();
        let loss = pred.mse_loss(&tape.var(&target)).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let g = bound.trainable_grads(&grads).unwrap();
        assert_eq!(g.len(), 4, "{layer}");
        // 層を挟んだ手前の Linear（添字 0・1）へ勾配が届く。
        assert!(
            g[0].contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .any(|v| *v != 0.0),
            "{layer}: 層より前の Linear に勾配が届いていない"
        );
    }
}

#[test]
fn fit_decreases_loss_with_each_layer() {
    let _guard = rng_lock();
    fandhe_ai::manual_seed(2679);
    for layer in LAYERS {
        let mut model = sandwich(layer);
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
            .unwrap();
        let x = ramp(&[8, 2], 2.0);
        let y = ramp(&[8, 1], 0.5);
        let history = model.fit(&x, &y, FitConfig::new(40, 8)).unwrap();
        assert_eq!(history.loss.len(), 40, "{layer}");
        assert!(
            history.loss.iter().all(|l| l.is_finite()),
            "{layer}: {:?}",
            history.loss
        );
        assert!(
            history.loss.last().unwrap() < history.loss.first().unwrap(),
            "{layer}: loss が下がるはず: {:?}",
            history.loss
        );
    }
}

// ---------------------------------------------------------------------
// RReLU の学習／推論モード
// ---------------------------------------------------------------------

#[test]
fn rrelu_eval_uses_fixed_mean_slope_and_train_uses_random_slope_in_bounds() {
    let _guard = rng_lock();
    let (lower, upper) = (0.125_f32, 0.3_f32);
    let x = tensor(vec![-4.0, -2.0, -1.0, 0.0, 1.0, 3.0], &[2, 3]);

    let mut model = Sequential::new().add_rrelu(lower, upper).unwrap();
    // 追加時点は学習モード（`Dropout` と同じ初期値）。
    assert!(model.training());
    model.eval();
    assert!(!model.training());
    let slope = f64::from(lower + upper) / 2.0;
    let expected: Vec<f64> = input_values(&x)
        .iter()
        .map(|v| if *v >= 0.0 { *v } else { *v * slope })
        .collect();
    assert_close(
        "rrelu eval",
        &values(&model.predict(&x).unwrap()),
        &expected,
    );

    model.train();
    assert!(model.training());
    fandhe_ai::manual_seed(7);
    let a = model.predict(&x).unwrap();
    fandhe_ai::manual_seed(7);
    let b = model.predict(&x).unwrap();
    assert_eq!(bits(&a), bits(&b), "同じシードなら学習時の傾きも決定的");
    for (xv, yv) in values(&x).into_iter().zip(values(&a)) {
        if xv >= 0.0 {
            assert_eq!(xv.to_bits(), yv.to_bits(), "非負の入力は恒等");
        } else {
            let ratio = yv / xv;
            assert!(
                ratio >= lower - 1e-6 && ratio <= upper + 1e-6,
                "x={xv} y={yv} 傾き {ratio} が [{lower}, {upper}] の外"
            );
        }
    }
}

// ---------------------------------------------------------------------
// 常駐経路（無状態層は通過し、未対応層を混ぜると従来どおり拒否）
// ---------------------------------------------------------------------

#[test]
fn resident_path_accepts_stateless_layers_and_matches_predict() {
    let mut model = Sequential::new()
        .add_linear(3, 4, 5)
        .unwrap()
        .add_selu()
        .add_celu(1.0)
        .unwrap()
        .add_softsign()
        .add_hardsigmoid()
        .add_log_sigmoid()
        .add_tanhshrink()
        .add_threshold(-5.0, 0.0)
        .add_rrelu(0.125, 0.3)
        .unwrap()
        .add_linear(4, 3, 6)
        .unwrap()
        .add_softmin(1);
    // RReLU を決定的にするため推論モードへ。
    model.eval();
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
fn resident_path_still_rejects_unsupported_layers_mixed_with_new_layers() {
    let model = Sequential::new()
        .add_linear(2, 3, 1)
        .unwrap()
        .add_selu()
        .add_prelu(3, 0.25)
        .unwrap()
        .add_softmin(1);
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
}

// ---------------------------------------------------------------------
// 保存・ONNX（未対応を型付きエラーで拒否する）
// ---------------------------------------------------------------------

#[test]
fn save_model_rejects_each_layer_without_touching_dir() {
    for layer in LAYERS {
        let model = sandwich(layer);
        let guard = TempDirGuard::new(&format!("scalar_layers_save_{layer}"));
        let dir = guard.path().join("m");
        let err = save_model(&model, &dir).unwrap_err();
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{layer}: {err:?}"
        );
        assert!(!dir.exists(), "{layer}: 拒否時は保存先に何も作らない");
    }
}

#[test]
fn onnx_export_rejects_each_layer_as_unsupported() {
    for layer in LAYERS {
        let model = sandwich(layer);
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(
            matches!(err, OnnxError::UnsupportedLayer { .. }),
            "{layer}: {err:?}"
        );
    }
}
