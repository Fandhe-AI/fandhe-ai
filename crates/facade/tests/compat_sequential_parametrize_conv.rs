//! `compat::Sequential::parametrize_weight_norm`／`parametrize_spectral_norm` の Conv 系 6 層への結線
//! （イシュー #3049・親 #3018。決定記録 `docs/facade-weight-reparam-wiring-decision.md` §5〜§10・§13。
//! Linear 版は `compat_sequential_parametrize_linear.rs`〈#3048〉）の CPU 結合テスト。
//!
//! 作成済みの `Conv1d`／`Conv2d`／`Conv3d`／`ConvTranspose1d`／`ConvTranspose2d`／`ConvTranspose3d`
//! 層を後から再パラメータ化する公開面を、facade 公開 API だけで次のとおり固定する（6 層共通のマクロ）。
//!
//! - 勾配一致（受入条件の本体）: 1 step の損失と `trainable_grads` の各勾配を、`Var::weight_norm`／
//!   `Var::spectral_norm` で有効重みを組んで同じ畳み込みへ渡す参照計算と bit 一致で比較する
//!   （groups>1・stride／padding 非自明・bias あり。spectral_norm は複数 step で参照側の状態進行とも一致）
//! - 適用直後の有効重みが元の重みと一致する（REQ-2 の統一複合判定。定数は変更しない）
//! - `named_parameters` の並び・optimizer 往復（`apply_parameters` で再パラメータ化が消えない）・`fit`
//! - 常駐経路が型付きエラーで拒否される（Conv2d は AMP・ONNX export も。`save_model` は #3050 で対応済み）
//! - 拒否系（範囲外添字・非対象層・二重適用・`dim` 範囲外・compile 後）
//!
//! 軸の方針: `weight_norm` の `dim` は層種に依らず利用者指定、spectral_norm の行軸は層から導出
//! （Conv 系 0・ConvTranspose 系 1。層単体テスト `crates/autodiff/tests/nn_conv_reparam.rs` が固定）。
//!
//! 保存往復（新 kind）は `compat_sequential_model_io_reparam.rs`（#3050）が固定する。

use fandhe_ai::compat::{
    AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::interop::onnx::OnnxModel;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};
use fandhe_ai_autodiff::nn::{
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
};
use fandhe_ai_backend_cpu::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

mod common;
use common::temp_dir::TempDirGuard;

const SEED: u64 = 0x5eed_3049;
const N_ITER: usize = 2;
const EPS: f32 = 1e-12;

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

fn assert_close(actual: &Tensor<f32>, expected: &Tensor<f32>, what: &str) {
    let (a, e) = (host(actual), host(expected));
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    for (i, (a, e)) in a.iter().zip(&e).enumerate() {
        // REQ-2 統一複合判定。閾値は正本定数（`backend-cpu::parity`）を参照する。
        let diff = f64::from((a - e).abs());
        let rel = diff / f64::from(e.abs()).max(f64::from(f32::MIN_POSITIVE));
        assert!(
            rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD,
            "{what}[{i}]: actual={a} expected={e}"
        );
    }
}

/// 参照計算用の autodiff 生 `Tape`（層の `bind` が `fandhe_ai_autodiff::Tape` を要求するため、
/// facade の `tape()` ラッパーではなく同じ CPU バックエンドで直接構築する）。
fn ref_tape() -> fandhe_ai_autodiff::Tape {
    fandhe_ai_autodiff::Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
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

fn assert_invalid_argument<T>(r: Result<T, AutodiffError>, what: &str) {
    let err = r.err();
    assert!(
        matches!(err, Some(AutodiffError::InvalidArgument(_))),
        "{what}: {err:?}"
    );
}

macro_rules! conv_parametrize_tests {
    (
        $modname:ident, $base:expr, $reference:expr, $input_shape:expr, $wn_dim:expr
    ) => {
        mod $modname {
            use super::*;

            fn base_model() -> Sequential {
                $base
            }

            fn input() -> Tensor<f32> {
                ramp(&$input_shape, 1.0)
            }

            fn target() -> Tensor<f32> {
                ramp(base_model().predict(&input()).unwrap().shape(), 0.5)
            }

            /// モデル経路の (損失, `trainable_grads` の各ビット列)。
            fn model_step(model: &Sequential) -> (Vec<u32>, Vec<Vec<u32>>) {
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                let (xv, tv) = (tape.var(&input()), tape.var(&target()));
                let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
                let grads = tape.backward(&loss).unwrap();
                let g = bound.trainable_grads(&grads).unwrap();
                (bits(&loss.to_tensor()), g.iter().map(|t| bits(t)).collect())
            }

            #[test]
            fn weight_norm_named_parameters_and_initial_equivalence() {
                let plain = base_model();
                let y_plain = plain.predict(&input()).unwrap();
                let w0 = named(&plain, "0.weight");
                let b0 = named(&plain, "0.bias");
                assert_eq!(param_names(&plain), ["0.weight", "0.bias"]);

                let model = base_model().parametrize_weight_norm(0, $wn_dim).unwrap();
                assert_eq!(
                    param_names(&model),
                    [
                        "0.parametrizations.weight.original0",
                        "0.parametrizations.weight.original1",
                        "0.bias"
                    ]
                );
                assert_eq!(
                    bits(&named(&model, "0.parametrizations.weight.original1")),
                    bits(&w0)
                );
                assert_eq!(bits(&named(&model, "0.bias")), bits(&b0));
                assert_close(&model.predict(&input()).unwrap(), &y_plain, "weight_norm");
            }

            #[test]
            fn weight_norm_one_step_loss_and_grads_bit_match_var_reference() {
                let model = base_model().parametrize_weight_norm(0, $wn_dim).unwrap();
                let (loss_bits, grads) = model_step(&model);
                assert_eq!(grads.len(), 3, "g・v・bias");

                let layer = $reference;
                let g_init = {
                    let m = base_model().parametrize_weight_norm(0, $wn_dim).unwrap();
                    named(&m, "0.parametrizations.weight.original0")
                };
                let tape = ref_tape();
                let mut vars = layer.bind(&tape);
                let v = vars.weight;
                let g = tape.var(&g_init);
                vars.weight = v.weight_norm(&g, $wn_dim).unwrap();
                let xv = tape.var(&input());
                let loss = vars
                    .forward(&xv)
                    .unwrap()
                    .mse_loss(&tape.var(&target()))
                    .unwrap();
                let rg = tape.backward(&loss).unwrap();
                let bias = vars.bias.as_ref().unwrap();
                let want = vec![
                    bits(rg.get(&g).unwrap().unwrap()),
                    bits(rg.get(&v).unwrap().unwrap()),
                    bits(rg.get(bias).unwrap().unwrap()),
                ];
                assert_eq!(loss_bits, bits(&loss.to_tensor()));
                assert_eq!(grads, want);
            }

            #[test]
            fn spectral_norm_multi_step_grads_bit_match_var_reference() {
                let model = base_model()
                    .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
                    .unwrap();
                assert_eq!(
                    param_names(&model),
                    ["0.parametrizations.weight.original", "0.bias"]
                );
                let layer = $reference;
                let mut state = $reference
                    .with_spectral_norm(N_ITER, EPS, SEED)
                    .unwrap()
                    .spectral_norm_state()
                    .unwrap();
                for step in 0..3 {
                    let (loss_bits, grads) = model_step(&model);
                    let tape = ref_tape();
                    let mut vars = layer.bind(&tape);
                    let raw = vars.weight;
                    vars.weight = raw.spectral_norm(&mut state, true).unwrap();
                    let xv = tape.var(&input());
                    let loss = vars
                        .forward(&xv)
                        .unwrap()
                        .mse_loss(&tape.var(&target()))
                        .unwrap();
                    let rg = tape.backward(&loss).unwrap();
                    let bias = vars.bias.as_ref().unwrap();
                    let want = vec![
                        bits(rg.get(&raw).unwrap().unwrap()),
                        bits(rg.get(bias).unwrap().unwrap()),
                    ];
                    assert_eq!(loss_bits, bits(&loss.to_tensor()), "step={step}");
                    assert_eq!(grads, want, "step={step}");
                }
            }

            #[test]
            fn predict_runs_via_tape_and_spectral_state_advances_only_in_train() {
                let x = input();
                let model = base_model()
                    .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
                    .unwrap();
                let layer = $reference;
                let mut state = $reference
                    .with_spectral_norm(N_ITER, EPS, SEED)
                    .unwrap()
                    .spectral_norm_state()
                    .unwrap();
                let orig = named(&model, "0.parametrizations.weight.original");
                let b = named(&model, "0.bias");
                for call in 0..2 {
                    let got = bits(&model.predict(&x).unwrap());
                    let tape = ref_tape();
                    let mut vars = layer.bind(&tape);
                    vars.weight = tape.var(&orig).spectral_norm(&mut state, true).unwrap();
                    vars.bias = Some(tape.var(&b));
                    let y = vars.forward(&tape.var(&x)).unwrap();
                    assert_eq!(got, bits(&y.to_tensor()), "predict call={call}");
                }
            }

            #[test]
            fn optimizer_round_trip_keeps_parametrization() {
                for weight_norm in [true, false] {
                    let mut model = if weight_norm {
                        base_model().parametrize_weight_norm(0, $wn_dim).unwrap()
                    } else {
                        base_model()
                            .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
                            .unwrap()
                    };
                    let names_before = param_names(&model);
                    let params_before: Vec<Vec<u32>> = model
                        .trainable_parameters()
                        .iter()
                        .map(|p| bits(p))
                        .collect();
                    let mut sgd = Sgd::new(SgdConfig::new(0.01)).unwrap();
                    for _ in 0..3 {
                        let updated = {
                            let tape = fandhe_ai::tape();
                            let bound = model.bind(&tape);
                            let (xv, tv) = (tape.var(&input()), tape.var(&target()));
                            let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
                            assert!(loss.to_tensor().get(&[]).unwrap().is_finite());
                            let grads = tape.backward(&loss).unwrap();
                            let grad_refs = bound.trainable_grads(&grads).unwrap();
                            let param_refs = model.trainable_parameters();
                            assert_eq!(param_refs.len(), grad_refs.len());
                            sgd.step(&param_refs, &grad_refs).unwrap()
                        };
                        model.apply_parameters(updated).unwrap();
                        // 更新後も再パラメータ化が保たれる（層を作り直して黙って消えない）。
                        assert_eq!(param_names(&model), names_before);
                    }
                    let params_after: Vec<Vec<u32>> = model
                        .trainable_parameters()
                        .iter()
                        .map(|p| bits(p))
                        .collect();
                    assert_ne!(params_before, params_after, "weight_norm={weight_norm}");
                }
            }

            #[test]
            fn fit_after_parametrize_updates_parameters() {
                let mut model = base_model().parametrize_weight_norm(0, $wn_dim).unwrap();
                let names_before = param_names(&model);
                model
                    .compile(Optimizer::Sgd(SgdConfig::new(0.01)), Loss::Mse)
                    .unwrap();
                let before: Vec<Vec<u32>> = model
                    .trainable_parameters()
                    .iter()
                    .map(|p| bits(p))
                    .collect();
                let mut x_shape = $input_shape.to_vec();
                x_shape[0] = 4;
                let x = ramp(&x_shape, 1.0);
                let out_shape = {
                    let mut s = base_model().predict(&x).unwrap().shape().to_vec();
                    s[0] = 4;
                    s
                };
                let history = model
                    .fit(&x, &ramp(&out_shape, 0.5), FitConfig::new(2, 2))
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

            #[test]
            fn resident_paths_are_rejected_with_unsupported() {
                // Conv は元から常駐非対応。再パラメータ化後も型付き `Unsupported` のまま。
                for model in [
                    base_model().parametrize_weight_norm(0, $wn_dim).unwrap(),
                    base_model()
                        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
                        .unwrap(),
                ] {
                    let tape = fandhe_ai::tape();
                    let err = model.init_device_param_store(&tape).unwrap_err();
                    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
                }
            }

            #[test]
            fn save_model_round_trips_state_bit_identically() {
                // #3050: 保存・復元に対応した（新 kind の網羅・改竄拒否は
                // `compat_sequential_model_io_reparam.rs`）。
                for model in [
                    base_model().parametrize_weight_norm(0, $wn_dim).unwrap(),
                    base_model()
                        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
                        .unwrap(),
                ] {
                    let guard =
                        TempDirGuard::new(concat!("parametrize_conv_save_", stringify!($modname)));
                    let dir = guard.path().join("m");
                    save_model(&model, &dir).expect("保存できるはず");
                    let loaded = load_model(&dir).expect("復元できるはず");
                    let (a, b) = (model.named_parameters(), loaded.named_parameters());
                    assert_eq!(a.len(), b.len());
                    for ((na, ta), (nb, tb)) in a.into_iter().zip(b) {
                        assert_eq!(na, nb);
                        assert_eq!(bits(ta), bits(tb), "{na}");
                    }
                }
            }

            #[test]
            fn invalid_targets_are_rejected() {
                assert_invalid_argument(base_model().parametrize_weight_norm(1, None), "range");
                assert_invalid_argument(
                    base_model().parametrize_spectral_norm(7, 1, EPS, 0),
                    "range (spectral)",
                );
                // 非対象層（ReLU）。
                let with_relu = || base_model().add_relu();
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
                // 不正 dim（weight の rank 以上）・n_power_iterations・eps。
                // `dim` の範囲外は `norm_except_dim` が `Shape(AxisOutOfRange)` の型付きエラーで拒否する。
                let err = base_model().parametrize_weight_norm(0, Some(9)).err();
                assert!(
                    matches!(err, Some(AutodiffError::Shape(_))),
                    "dim=9: {err:?}"
                );
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
                let compiled = || {
                    let mut model = base_model();
                    model
                        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
                        .unwrap();
                    model
                };
                assert_invalid_argument(compiled().parametrize_weight_norm(0, None), "compile");
                assert_invalid_argument(
                    compiled().parametrize_spectral_norm(0, 1, EPS, 0),
                    "compile (spectral)",
                );
            }

            #[test]
            fn add_conv_without_parametrize_is_unchanged() {
                let model = base_model();
                assert_eq!(param_names(&model), ["0.weight", "0.bias"]);
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                assert_eq!(bound.trainable_vars().len(), 2);
            }
        }
    };
}

conv_parametrize_tests!(
    conv2d,
    Sequential::new()
        .add_conv2d(4, 6, [3, 2], [1, 2], [1, 0], [1, 1], 2, SEED)
        .unwrap(),
    Conv2d::new(4, 6, [3, 2], [1, 2], [1, 0], [1, 1], 2, true, SEED).unwrap(),
    [2, 4, 6, 7],
    Some(0)
);
conv_parametrize_tests!(
    conv1d,
    Sequential::new()
        .add_conv1d(4, 6, 3, 2, 1, 1, 2, SEED)
        .unwrap(),
    Conv1d::new(4, 6, 3, 2, 1, 1, 2, true, SEED).unwrap(),
    [2, 4, 9],
    Some(0)
);
conv_parametrize_tests!(
    conv3d,
    Sequential::new()
        .add_conv3d(4, 6, [2, 2, 2], [1, 1, 1], [1, 0, 0], [1, 1, 1], 2, SEED)
        .unwrap(),
    Conv3d::new(
        4,
        6,
        [2, 2, 2],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        2,
        true,
        SEED
    )
    .unwrap(),
    [1, 4, 3, 4, 4],
    None
);
// ConvTranspose 系の `weight_norm` は PyTorch 既定の `dim=0`（入力チャネル軸）と、出力チャネル単位の
// `Some(1)` の両方が利用者指定で選べる（決定記録 §13。ここでは `Some(1)` を基準に固定する）。
conv_parametrize_tests!(
    conv_transpose1d,
    Sequential::new()
        .add_conv_transpose1d(4, 6, 3, 2, 1, 1, 1, 2, SEED)
        .unwrap(),
    ConvTranspose1d::new(4, 6, 3, 2, 1, 1, 1, 2, true, SEED).unwrap(),
    [2, 4, 5],
    Some(1)
);
conv_parametrize_tests!(
    conv_transpose2d,
    Sequential::new()
        .add_conv_transpose2d(4, 6, [3, 2], [2, 1], [1, 0], [1, 0], [1, 1], 2, SEED)
        .unwrap(),
    ConvTranspose2d::new(4, 6, [3, 2], [2, 1], [1, 0], [1, 0], [1, 1], 2, true, SEED).unwrap(),
    [2, 4, 5, 4],
    Some(1)
);
conv_parametrize_tests!(
    conv_transpose3d,
    Sequential::new()
        .add_conv_transpose3d(
            4,
            6,
            [2, 2, 2],
            [2, 1, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            2,
            SEED
        )
        .unwrap(),
    ConvTranspose3d::new(
        4,
        6,
        [2, 2, 2],
        [2, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        2,
        true,
        SEED
    )
    .unwrap(),
    [1, 4, 2, 3, 3],
    Some(0)
);

// ---------------------------------------------------------------------
// Conv2d 固有: AMP・ONNX export（他の Conv は AMP を無視する／export 未対応のため対象外）
// ---------------------------------------------------------------------

fn conv2d_model() -> Sequential {
    Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, SEED)
        .unwrap()
}

#[test]
fn amp_forward_is_rejected_for_parametrized_conv2d() {
    let mut model = conv2d_model()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap();
    let x = ramp(&[2, 2, 4, 4], 1.0);
    let y = ramp(&[2, 3, 4, 4], 0.5);
    assert_invalid_argument(model.fit(&x, &y, FitConfig::new(1, 2)), "amp fit");
}

#[test]
fn onnx_export_is_rejected_for_parametrized_conv2d() {
    for model in [
        conv2d_model().parametrize_weight_norm(0, Some(0)).unwrap(),
        conv2d_model()
            .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
            .unwrap(),
    ] {
        assert!(OnnxModel::from_sequential(&model).is_err());
    }
    // 非再パラメータ化は従来どおり export できる。
    assert!(OnnxModel::from_sequential(&conv2d_model()).is_ok());
}

#[test]
fn conv_then_relu_composite_keeps_equivalence_after_weight_norm() {
    // Conv→ReLU の複合でも適用直後の出力が元と一致する（REQ-2 統一複合判定。tolerance は不変）。
    let build = || conv2d_model().add_relu();
    let x = ramp(&[2, 2, 4, 4], 1.0);
    let want = build().predict(&x).unwrap();
    let got = build()
        .parametrize_weight_norm(0, Some(0))
        .unwrap()
        .predict(&x)
        .unwrap();
    assert_close(&got, &want, "conv->relu");
}
