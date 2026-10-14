//! `compat::Sequential::parametrize_weight_norm`／`parametrize_spectral_norm`（イシュー #3048・
//! 親 #3018。決定記録 `docs/facade-weight-reparam-wiring-decision.md` §5〜§10・§13）の CPU 結合テスト。
//!
//! 作成済みの `Linear` 層を後から weight_norm／spectral_norm へ再パラメータ化する公開面を、
//! facade 公開 API だけで次のとおり固定する。
//!
//! - 勾配一致（受入条件の本体）: 1 step の損失と `trainable_grads` の各勾配を、`Var::weight_norm`／
//!   `Var::spectral_norm` を直接使った参照計算と bit 一致で比較する（spectral_norm は複数 step で
//!   参照側の状態進行とも一致）
//! - 適用直後の有効重みが元の重みと一致する（REQ-2 の統一複合判定。定数は変更しない）
//! - `named_parameters` の並び・optimizer 往復・`fit` で再パラメータ化が保たれる
//! - 常駐経路・低精度（AMP）・`save_model`・ONNX export が型付きエラーで拒否される
//! - 拒否系（範囲外添字・非 Linear・二重適用・compile 後）
//!
//! 保存往復（新 kind）・保留ガード反転・ソース走査インベントリは #3050 の担当（本テストの対象外）。

use fandhe_ai::compat::{
    AmpConfig, AmpDType, FitConfig, Loss, ModelIoError, Optimizer, Sequential, save_model,
};
use fandhe_ai::interop::onnx::OnnxModel;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_backend_cpu::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

mod common;
use common::temp_dir::TempDirGuard;

const IN: usize = 3;
const OUT: usize = 4;
const SEED: u64 = 0x5eed_3048;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn host(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().host_slice().into_owned()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    host(t).iter().map(|v| v.to_bits()).collect()
}

fn ramp(shape: &[usize], scale: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(
        (0..n).map(|i| ((i as f32) * 0.37).sin() * scale).collect(),
        shape,
    )
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: 要素数");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        // REQ-2 統一複合判定。閾値は正本定数（`backend-cpu::parity`）を参照する。
        let diff = f64::from((a - e).abs());
        let rel = diff / f64::from(e.abs()).max(f64::from(f32::MIN_POSITIVE));
        assert!(
            rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD,
            "{what}[{i}]: actual={a} expected={e}"
        );
    }
}

fn base_model() -> Sequential {
    Sequential::new().add_linear(IN, OUT, SEED).unwrap()
}

fn param_names(model: &Sequential) -> Vec<String> {
    model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

fn named(model: &Sequential, key: &str) -> Tensor<f32> {
    model
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == key)
        .unwrap_or_else(|| panic!("{key} が無い"))
        .1
        .clone()
}

/// 参照計算の損失（`mse`）。モデル側と同じ入力・ターゲットを使う。
fn input() -> Tensor<f32> {
    ramp(&[5, IN], 1.0)
}

fn target() -> Tensor<f32> {
    ramp(&[5, OUT], 0.5)
}

fn assert_invalid_argument<T>(r: Result<T, AutodiffError>, what: &str) {
    let err = r.err();
    assert!(
        matches!(err, Some(AutodiffError::InvalidArgument(_))),
        "{what}: {err:?}"
    );
}

// ---------------------------------------------------------------------
// weight_norm
// ---------------------------------------------------------------------

#[test]
fn weight_norm_named_parameters_order_and_initial_equivalence() {
    let plain = base_model();
    let w0 = named(&plain, "0.weight");
    let b0 = named(&plain, "0.bias");
    let x = input();
    let y_plain = plain.predict(&x).unwrap();

    for dim in [None, Some(0), Some(1)] {
        let model = base_model().parametrize_weight_norm(0, dim).unwrap();
        assert_eq!(
            param_names(&model),
            [
                "0.parametrizations.weight.original0",
                "0.parametrizations.weight.original1",
                "0.bias"
            ],
            "dim={dim:?}"
        );
        // v は元の重み、bias は不変。
        assert_eq!(
            bits(&named(&model, "0.parametrizations.weight.original1")),
            bits(&w0)
        );
        assert_eq!(bits(&named(&model, "0.bias")), bits(&b0));
        // g は norm_except_dim の手計算（dim 以外の全軸の L2）と一致する。
        let g = named(&model, "0.parametrizations.weight.original0");
        let w = host(&w0);
        let expected_g: Vec<f32> = match dim {
            None => vec![w.iter().map(|v| v * v).sum::<f32>().sqrt()],
            Some(0) => (0..IN)
                .map(|i| (0..OUT).map(|j| w[i * OUT + j].powi(2)).sum::<f32>().sqrt())
                .collect(),
            Some(_) => (0..OUT)
                .map(|j| (0..IN).map(|i| w[i * OUT + j].powi(2)).sum::<f32>().sqrt())
                .collect(),
        };
        assert_close(&host(&g), &expected_g, "g");
        // 適用直後の有効重みは元の重みと一致する（predict の出力で確認）。
        assert_close(
            &host(&model.predict(&x).unwrap()),
            &host(&y_plain),
            &format!("predict dim={dim:?}"),
        );
        // trainable_parameters も named_parameters と同じ順序。
        let tp: Vec<Vec<u32>> = model
            .trainable_parameters()
            .iter()
            .map(|t| bits(t))
            .collect();
        let np: Vec<Vec<u32>> = model
            .named_parameters()
            .iter()
            .map(|(_, t)| bits(t))
            .collect();
        assert_eq!(tp, np);
    }
}

#[test]
fn weight_norm_forward_and_grads_match_var_reference_bit_exact() {
    for dim in [None, Some(0), Some(1)] {
        let model = base_model().parametrize_weight_norm(0, dim).unwrap();
        let g = named(&model, "0.parametrizations.weight.original0");
        let v = named(&model, "0.parametrizations.weight.original1");
        let b = named(&model, "0.bias");
        let (x, t) = (input(), target());

        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let xv = tape.var(&x);
        let tv = tape.var(&t);
        let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let model_grads: Vec<Vec<u32>> = bound
            .trainable_grads(&grads)
            .unwrap()
            .iter()
            .map(|t| bits(t))
            .collect();
        assert_eq!(model_grads.len(), 3, "[g, v, bias]");

        let rtape = fandhe_ai::tape();
        let (gv, vv, bv) = (rtape.var(&g), rtape.var(&v), rtape.var(&b));
        let rx = rtape.var(&x);
        let rt = rtape.var(&t);
        let w = vv.weight_norm(&gv, dim).unwrap();
        let ref_loss = rx
            .matmul(&w)
            .unwrap()
            .add(&bv)
            .unwrap()
            .mse_loss(&rt)
            .unwrap();
        let ref_grads = rtape.backward(&ref_loss).unwrap();
        let expected: Vec<Vec<u32>> = [&gv, &vv, &bv]
            .iter()
            .map(|p| bits(ref_grads.get(p).unwrap().unwrap()))
            .collect();

        assert_eq!(
            loss.to_tensor().get(&[]).unwrap().to_bits(),
            ref_loss.to_tensor().get(&[]).unwrap().to_bits(),
            "loss dim={dim:?}"
        );
        assert_eq!(model_grads, expected, "grads dim={dim:?}");
    }
}

#[test]
fn weight_norm_with_relu_fusion_matches_reference_within_tolerance() {
    let model = Sequential::new()
        .add_linear(IN, OUT, SEED)
        .unwrap()
        .add_relu()
        .parametrize_weight_norm(0, Some(1))
        .unwrap();
    let g = named(&model, "0.parametrizations.weight.original0");
    let v = named(&model, "0.parametrizations.weight.original1");
    let b = named(&model, "0.bias");
    let (x, t) = (input(), tensor(vec![0.1; 5 * OUT], &[5, OUT]));

    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&t);
    let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let model_grads: Vec<Vec<f32>> = bound
        .trainable_grads(&grads)
        .unwrap()
        .iter()
        .map(|t| host(t))
        .collect();

    let rtape = fandhe_ai::tape();
    let (gv, vv, bv) = (rtape.var(&g), rtape.var(&v), rtape.var(&b));
    let (rx, rt) = (rtape.var(&x), rtape.var(&t));
    let w = vv.weight_norm(&gv, Some(1)).unwrap();
    let ref_loss = rx
        .matmul(&w)
        .unwrap()
        .add(&bv)
        .unwrap()
        .relu()
        .mse_loss(&rt)
        .unwrap();
    let ref_grads = rtape.backward(&ref_loss).unwrap();
    for (i, p) in [&gv, &vv, &bv].iter().enumerate() {
        assert_close(
            &model_grads[i],
            &host(ref_grads.get(p).unwrap().unwrap()),
            &format!("relu grad[{i}]"),
        );
    }
}

// ---------------------------------------------------------------------
// spectral_norm
// ---------------------------------------------------------------------

const N_ITER: usize = 2;
const EPS: f32 = 1e-12;

#[test]
fn spectral_norm_named_parameters_and_initial_scale() {
    let plain = base_model();
    let w0 = named(&plain, "0.weight");
    let model = base_model()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    assert_eq!(
        param_names(&model),
        ["0.parametrizations.weight.original", "0.bias"]
    );
    assert_eq!(
        bits(&named(&model, "0.parametrizations.weight.original")),
        bits(&w0)
    );
    assert_eq!(model.trainable_parameters().len(), 2);
}

#[test]
fn spectral_norm_forward_and_grads_match_var_reference_over_steps() {
    // 参照側の初期状態は同じ seed の autodiff `Linear` から取る（`add_linear(IN, OUT, SEED)` は
    // `Linear::new(IN, OUT, true, SEED)` と同一の初期化・u0／v0 導出）。
    let mut state = Linear::new(IN, OUT, true, SEED)
        .unwrap()
        .with_spectral_norm(N_ITER, EPS, SEED)
        .unwrap()
        .spectral_norm_state()
        .unwrap();
    let model = base_model()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    let orig = named(&model, "0.parametrizations.weight.original");
    let b = named(&model, "0.bias");
    let (x, t) = (input(), target());

    for step in 0..3 {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let xv = tape.var(&x);
        let tv = tape.var(&t);
        let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let model_grads: Vec<Vec<u32>> = bound
            .trainable_grads(&grads)
            .unwrap()
            .iter()
            .map(|t| bits(t))
            .collect();
        assert_eq!(model_grads.len(), 2, "[original, bias]");

        let rtape = fandhe_ai::tape();
        let (ov, bv) = (rtape.var(&orig), rtape.var(&b));
        let (rx, rt) = (rtape.var(&x), rtape.var(&t));
        let w = ov.spectral_norm(&mut state, true).unwrap();
        let ref_loss = rx
            .matmul(&w)
            .unwrap()
            .add(&bv)
            .unwrap()
            .mse_loss(&rt)
            .unwrap();
        let ref_grads = rtape.backward(&ref_loss).unwrap();
        let expected: Vec<Vec<u32>> = [&ov, &bv]
            .iter()
            .map(|p| bits(ref_grads.get(p).unwrap().unwrap()))
            .collect();

        assert_eq!(
            loss.to_tensor().get(&[]).unwrap().to_bits(),
            ref_loss.to_tensor().get(&[]).unwrap().to_bits(),
            "loss step={step}"
        );
        assert_eq!(model_grads, expected, "grads step={step}");
    }
}

#[test]
fn spectral_norm_eval_does_not_advance_state() {
    let mut model = base_model()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    model.eval();
    let x = input();
    let first = bits(&model.predict(&x).unwrap());
    let second = bits(&model.predict(&x).unwrap());
    assert_eq!(first, second, "eval の forward は状態を進めない");

    // eval で作った出力は、参照側 `training = false`・初期状態の spectral_norm と bit 一致する。
    let mut state = Linear::new(IN, OUT, true, SEED)
        .unwrap()
        .with_spectral_norm(N_ITER, EPS, SEED)
        .unwrap()
        .spectral_norm_state()
        .unwrap();
    let orig = named(&model, "0.parametrizations.weight.original");
    let b = named(&model, "0.bias");
    let tape = fandhe_ai::tape();
    let (ov, bv, xv) = (tape.var(&orig), tape.var(&b), tape.var(&x));
    let w = ov.spectral_norm(&mut state, false).unwrap();
    let y = xv.matmul(&w).unwrap().add(&bv).unwrap();
    assert_eq!(first, bits(&y.to_tensor()));
}

// ---------------------------------------------------------------------
// optimizer 往復・fit
// ---------------------------------------------------------------------

#[test]
fn optimizer_round_trip_keeps_parametrization() {
    for weight_norm in [true, false] {
        let mut model = if weight_norm {
            base_model().parametrize_weight_norm(0, Some(1)).unwrap()
        } else {
            base_model()
                .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
                .unwrap()
        };
        let names_before = param_names(&model);
        let (x, t) = (input(), target());
        let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
        let mut losses = Vec::new();
        for _ in 0..4 {
            let updated = {
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                let (xv, tv) = (tape.var(&x), tape.var(&t));
                let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
                losses.push(loss.to_tensor().get(&[]).unwrap());
                let grads = tape.backward(&loss).unwrap();
                let grad_refs = bound.trainable_grads(&grads).unwrap();
                let param_refs = model.trainable_parameters();
                sgd.step(&param_refs, &grad_refs).unwrap()
            };
            model.apply_parameters(updated).unwrap();
            // 更新後も再パラメータ化が保たれる（層を作り直して黙って消えない）。
            assert_eq!(param_names(&model), names_before);
        }
        assert!(
            losses.last().unwrap() < &losses[0],
            "weight_norm={weight_norm}: loss は減少するはず {losses:?}"
        );
    }
}

#[test]
fn fit_after_parametrize_updates_parameters() {
    let mut model = Sequential::new()
        .add_linear(IN, 6, SEED)
        .unwrap()
        .add_relu()
        .add_linear(6, OUT, SEED + 1)
        .unwrap()
        .parametrize_weight_norm(0, Some(1))
        .unwrap()
        .parametrize_spectral_norm(2, N_ITER, EPS, SEED)
        .unwrap();
    let names_before = param_names(&model);
    assert_eq!(
        names_before,
        [
            "0.parametrizations.weight.original0",
            "0.parametrizations.weight.original1",
            "0.bias",
            "2.parametrizations.weight.original",
            "2.bias",
        ]
    );
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let before: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    let history = model
        .fit(&input(), &target(), FitConfig::new(3, 5))
        .unwrap();
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
    assert_ne!(before, after);
    assert_eq!(param_names(&model), names_before);
}

// ---------------------------------------------------------------------
// 非対応経路の型付き拒否
// ---------------------------------------------------------------------

#[test]
fn resident_paths_are_rejected_with_unsupported() {
    let model = base_model().parametrize_weight_norm(0, Some(1)).unwrap();
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");

    // 常駐対応モデルのストアを借りて predict／forward の入口も拒否されることを見る。
    let plain_tape = fandhe_ai::tape();
    let mut store = base_model().init_device_param_store(&plain_tape).unwrap();
    drop(plain_tape);
    let err = model.predict_resident(&store, &input()).unwrap_err();
    assert!(
        matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
        "{err:?}"
    );
    let tape2 = fandhe_ai::tape();
    let xv = tape2.var(&input());
    let err = model.forward_resident(&tape2, &xv, &mut store).unwrap_err();
    assert!(
        matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
        "{err:?}"
    );
}

#[test]
fn amp_forward_is_rejected_for_parametrized_linear() {
    let mut model = base_model()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap();
    assert_invalid_argument(
        model.fit(&input(), &target(), FitConfig::new(1, 5)),
        "amp fit",
    );
}

#[test]
fn save_model_is_rejected_and_leaves_nothing() {
    for model in [
        base_model().parametrize_weight_norm(0, Some(1)).unwrap(),
        base_model()
            .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
            .unwrap(),
    ] {
        let guard = TempDirGuard::new("parametrize_linear_save");
        let dir = guard.path().join("m");
        let err = save_model(&model, &dir).unwrap_err();
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{err:?}"
        );
        assert!(!dir.exists(), "拒否時は dir を作らない");
    }
}

#[test]
fn onnx_export_is_rejected_for_parametrized_linear() {
    for model in [
        base_model().parametrize_weight_norm(0, Some(1)).unwrap(),
        base_model()
            .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
            .unwrap(),
    ] {
        assert!(OnnxModel::from_sequential(&model).is_err());
    }
    // 非再パラメータ化は従来どおり export できる。
    assert!(OnnxModel::from_sequential(&base_model()).is_ok());
}

// ---------------------------------------------------------------------
// predict（tape 経路）
// ---------------------------------------------------------------------

#[test]
fn predict_runs_via_tape_and_spectral_state_advances_only_in_train() {
    let x = input();
    // train（既定）: predict ごとに状態が進む。参照（autodiff Linear の state）と 2 回分一致する。
    let model = base_model()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    let mut state = Linear::new(IN, OUT, true, SEED)
        .unwrap()
        .with_spectral_norm(N_ITER, EPS, SEED)
        .unwrap()
        .spectral_norm_state()
        .unwrap();
    let orig = named(&model, "0.parametrizations.weight.original");
    let b = named(&model, "0.bias");
    for call in 0..2 {
        let got = bits(&model.predict(&x).unwrap());
        let tape = fandhe_ai::tape();
        let (ov, bv, xv) = (tape.var(&orig), tape.var(&b), tape.var(&x));
        let w = ov.spectral_norm(&mut state, true).unwrap();
        let y = xv.matmul(&w).unwrap().add(&bv).unwrap();
        assert_eq!(got, bits(&y.to_tensor()), "predict call={call}");
    }
}

// ---------------------------------------------------------------------
// 拒否系
// ---------------------------------------------------------------------

#[test]
fn invalid_targets_are_rejected() {
    // 範囲外添字。
    assert_invalid_argument(
        base_model().parametrize_weight_norm(1, None),
        "out of range",
    );
    assert_invalid_argument(
        base_model().parametrize_spectral_norm(7, 1, EPS, 0),
        "out of range (spectral)",
    );
    // 非 Linear。
    let with_relu = || {
        Sequential::new()
            .add_linear(IN, OUT, SEED)
            .unwrap()
            .add_relu()
    };
    assert_invalid_argument(with_relu().parametrize_weight_norm(1, None), "relu");
    assert_invalid_argument(
        with_relu().parametrize_spectral_norm(1, 1, EPS, 0),
        "relu (spectral)",
    );
    // 二重適用（同種・異種）。
    let wn = base_model().parametrize_weight_norm(0, None).unwrap();
    assert_invalid_argument(wn.parametrize_weight_norm(0, None), "double wn");
    let wn = base_model().parametrize_weight_norm(0, None).unwrap();
    assert_invalid_argument(wn.parametrize_spectral_norm(0, 1, EPS, 0), "wn then sn");
    // 不正 dim・n_power_iterations・eps。
    assert_invalid_argument(base_model().parametrize_weight_norm(0, Some(2)), "dim=2");
    assert_invalid_argument(
        base_model().parametrize_spectral_norm(0, 0, EPS, 0),
        "n_power_iterations=0",
    );
    assert_invalid_argument(
        base_model().parametrize_spectral_norm(0, 1, f32::NAN, 0),
        "eps=NaN",
    );
    assert_invalid_argument(
        base_model().parametrize_spectral_norm(0, 1, -1.0, 0),
        "eps<0",
    );
}

#[test]
fn parametrize_after_compile_is_rejected() {
    let mut model = base_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    assert_invalid_argument(model.parametrize_weight_norm(0, None), "after compile");
    let mut model = base_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    assert_invalid_argument(
        model.parametrize_spectral_norm(0, 1, EPS, 0),
        "after compile (spectral)",
    );
}

#[test]
fn add_linear_without_parametrize_is_unchanged() {
    let model = base_model();
    assert_eq!(param_names(&model), ["0.weight", "0.bias"]);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    assert_eq!(bound.trainable_vars().len(), 2);
}
