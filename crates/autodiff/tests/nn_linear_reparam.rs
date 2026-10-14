//! `nn::Linear` の weight_norm／spectral_norm 再パラメータ化（イシュー #3048・親 #3018。決定記録
//! `docs/facade-weight-reparam-wiring-decision.md` §5〜§10）の層単体テスト。
//!
//! facade の結合テスト（`crates/facade/tests/compat_sequential_parametrize_linear.rs`）が
//! `Sequential` 越しの勾配一致を担うのに対し、ここでは層自身の契約を固定する。
//!
//! - 命名・順序（`named_parameters`）と `set_parameter` の受理／拒否
//! - 失敗した適用（二重適用・不正引数）で層が変化しないこと
//! - spectral_norm の状態: train の forward 1 回ごとに進み、eval と失敗 forward では進まない
//! - host 経路（`forward_host`／`supports_forward_host`）が有効重みの誤読を防ぐこと

mod common;

use fandhe_ai_autodiff::nn::{Linear, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, Tensor};

const SEED: u64 = 0x3048;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn linear() -> Linear {
    Linear::new(3, 4, true, SEED).unwrap()
}

fn names(layer: &Linear) -> Vec<String> {
    layer
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

#[test]
fn named_parameters_order_for_each_parametrization() {
    assert_eq!(names(&linear()), ["weight", "bias"]);
    assert_eq!(
        names(&linear().with_weight_norm(Some(1)).unwrap()),
        [
            "parametrizations.weight.original0",
            "parametrizations.weight.original1",
            "bias"
        ]
    );
    assert_eq!(
        names(&linear().with_spectral_norm(1, 1e-12, SEED).unwrap()),
        ["parametrizations.weight.original", "bias"]
    );
    let no_bias = Linear::new(3, 4, false, SEED)
        .unwrap()
        .with_spectral_norm(1, 1e-12, SEED)
        .unwrap();
    assert_eq!(names(&no_bias), ["parametrizations.weight.original"]);
}

#[test]
fn set_parameter_accepts_new_keys_only_and_keeps_state() {
    let mut layer = linear().with_weight_norm(Some(1)).unwrap();
    let g_shape = layer.weight_norm_g().unwrap().shape().to_vec();
    assert_eq!(g_shape, [1, 4]);

    layer
        .set_parameter(
            "parametrizations.weight.original0",
            t(vec![2.0; 4], &[1, 4]),
        )
        .unwrap();
    assert_eq!(
        layer.weight_norm_g().unwrap().host_slice().to_vec(),
        vec![2.0; 4]
    );
    layer
        .set_parameter(
            "parametrizations.weight.original1",
            t(vec![0.5; 12], &[3, 4]),
        )
        .unwrap();
    assert_eq!(layer.weight().host_slice().to_vec(), vec![0.5; 12]);
    assert!(layer.is_parametrized(), "再パラメータ化状態は保持される");
    assert_eq!(layer.weight_norm_dim(), Some(Some(1)));

    // `weight` キーは拒否、shape 不一致は拒否。
    assert!(matches!(
        layer.set_parameter("weight", t(vec![0.0; 12], &[3, 4])),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        layer.set_parameter(
            "parametrizations.weight.original0",
            t(vec![1.0; 3], &[1, 3])
        ),
        Err(AutodiffError::Shape(_))
    ));
    // 非再パラメータ化の層は新キーを受理しない。
    let mut plain = linear();
    assert!(matches!(
        plain.set_parameter("parametrizations.weight.original0", t(vec![1.0], &[1])),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn failed_parametrize_leaves_layer_unchanged() {
    let mut layer = linear();
    assert!(layer.parametrize_weight_norm(Some(2)).is_err());
    assert!(layer.parametrize_spectral_norm(0, 1e-12, SEED).is_err());
    assert!(
        layer
            .parametrize_spectral_norm(1, f32::INFINITY, SEED)
            .is_err()
    );
    assert!(!layer.is_parametrized());
    assert_eq!(names(&layer), ["weight", "bias"]);

    layer.parametrize_weight_norm(None).unwrap();
    assert!(layer.parametrize_weight_norm(None).is_err());
    assert!(layer.parametrize_spectral_norm(1, 1e-12, SEED).is_err());
    assert_eq!(
        layer.weight_norm_dim(),
        Some(None),
        "二重適用の拒否後も不変"
    );
    assert_eq!(layer.weight_norm_g().unwrap().rank(), 0);
}

fn run_forward(layer: &Linear, x: &Tensor<f32>) -> Result<Vec<u32>, AutodiffError> {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let vars = layer.bind(&tape);
    let g = layer.bind_weight_norm_g(&tape);
    let y = layer.forward_bound(&vars, g.as_ref(), &xv, false)?;
    Ok(y.to_tensor()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

#[test]
fn spectral_state_advances_only_in_train_and_not_on_failed_forward() {
    let x = t((0..6).map(|i| i as f32 * 0.1).collect(), &[2, 3]);
    let mut layer = linear().with_spectral_norm(1, 1e-12, SEED).unwrap();

    // train（既定）: forward ごとに状態が進む。
    let s0 = layer.spectral_norm_state().unwrap();
    run_forward(&layer, &x).unwrap();
    let s1 = layer.spectral_norm_state().unwrap();
    assert_ne!(s0, s1, "train の forward は状態を進める");
    run_forward(&layer, &x).unwrap();
    assert_ne!(s1, layer.spectral_norm_state().unwrap());

    // 失敗 forward（in 軸不一致）は状態を進めない。
    let before = layer.spectral_norm_state().unwrap();
    let bad = t(vec![0.0; 8], &[2, 4]);
    assert!(run_forward(&layer, &bad).is_err());
    assert_eq!(before, layer.spectral_norm_state().unwrap());

    // eval: 状態は進まず、出力も再現する。
    layer.set_training(false);
    assert!(!layer.training());
    let frozen = layer.spectral_norm_state().unwrap();
    let y1 = run_forward(&layer, &x).unwrap();
    let y2 = run_forward(&layer, &x).unwrap();
    assert_eq!(frozen, layer.spectral_norm_state().unwrap());
    assert_eq!(y1, y2);

    // train へ戻すと再び進む。
    layer.set_training(true);
    run_forward(&layer, &x).unwrap();
    assert_ne!(frozen, layer.spectral_norm_state().unwrap());
}

#[test]
fn host_paths_never_read_raw_weight_as_effective_weight() {
    let ops = common::naive_ops();
    let x = t(vec![0.1; 6], &[2, 3]);
    for layer in [
        linear().with_weight_norm(Some(1)).unwrap(),
        linear().with_spectral_norm(1, 1e-12, SEED).unwrap(),
    ] {
        assert!(!layer.supports_forward_host());
        assert!(matches!(
            layer.forward_host(&*ops, &x),
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
        assert!(matches!(
            layer.forward_host_with_activation(&*ops, &x, fandhe_ai_tensor_core::Activation::Relu),
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
    }
    // 非再パラメータ化は従来どおり host 経路に対応する。
    let plain = linear();
    assert!(plain.supports_forward_host());
    assert!(plain.forward_host(&*ops, &x).is_ok());
}

#[test]
fn module_forward_matches_bound_forward_for_weight_norm() {
    // `Module::forward`（bind + g 登録 + forward_bound）と初期の素の層の出力が一致する。
    let x = t((0..6).map(|i| i as f32 * 0.2 - 0.3).collect(), &[2, 3]);
    let plain = linear();
    let wn = linear().with_weight_norm(Some(1)).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let y_plain = plain.forward(&tape, &xv).unwrap().to_tensor();
    let y_wn = wn.forward(&tape, &xv).unwrap().to_tensor();
    let (a, b) = (y_plain.host_slice(), y_wn.host_slice());
    for (p, q) in a.iter().zip(b.iter()) {
        let diff = (p - q).abs();
        assert!(
            diff / p.abs().max(f32::MIN_POSITIVE) < 1e-3 || diff < 1e-5,
            "{p} vs {q}"
        );
    }
}
