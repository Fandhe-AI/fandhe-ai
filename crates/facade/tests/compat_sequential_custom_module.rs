//! `compat::Sequential::add_module`（イシュー #2398。#2338 承認事項 4）の統合テスト。
//! facade だけに依存して書いた独自層（`fandhe_ai::nn::Module` 実装）を `compat::Sequential` へ積み、
//! 推論・`state_dict` 往復・`apply_parameters` が動くこと、およびパラメータ持ちの独自層は
//! 学習経路（`fit`・`bind`）と常駐経路で型付き `Err` により fail-closed に拒否され
//! 「学習されないまま成功する」状態にならないことを固定する。
//! `fandhe_ai_autodiff` は import しない。tolerance は使わずビット一致で比較する。

use std::cell::Cell;
use std::rc::Rc;

use fandhe_ai::compat::{AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, SgdConfig as FacadeSgdConfig, TapeRef, Tensor, Var, tape};

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, shape).expect("テスト入力の構築")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("host tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `x @ weight + bias` を返す 3→2 のパラメータ持ち独自層。モードは共有フラグで観測できる。
struct Affine {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    training: bool,
    seen: Rc<Cell<bool>>,
}

impl Affine {
    fn new() -> Self {
        Self::observed(Rc::new(Cell::new(true)))
    }
    fn observed(seen: Rc<Cell<bool>>) -> Self {
        Self {
            weight: t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, -0.05], &[2]),
            training: true,
            seen,
        }
    }
}

impl fandhe_ai::nn::Module for Affine {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let w = tape.var(&self.weight);
        let b = tape.var(&self.bias);
        input.matmul(&w)?.add(&b)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("weight".into(), &self.weight), ("bias".into(), &self.bias)]
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let slot = match name {
            "weight" => &mut self.weight,
            "bias" => &mut self.bias,
            _ => return Err(AutodiffError::InvalidArgument(format!("unknown `{name}`"))),
        };
        if slot.shape() != value.shape() {
            return Err(AutodiffError::InvalidArgument("shape".into()));
        }
        *slot = value;
        Ok(())
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.seen.set(training);
    }
    fn training(&self) -> bool {
        self.training
    }
}

/// 無状態の独自 tanh 層（パラメータなし。既定実装のみ）。
struct CustomTanh;
impl fandhe_ai::nn::Module for CustomTanh {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(input.tanh())
    }
}

const SEED_A: u64 = 11;
const SEED_B: u64 = 22;

fn input() -> Tensor<f32> {
    t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3])
}

fn target2() -> Tensor<f32> {
    t32(&[0.1, -0.2, 0.3, 0.4], &[2, 2])
}

/// `[linear(3,3), Affine(3→2), tanh]`。
fn custom_model(seed: u64) -> Sequential {
    Sequential::new()
        .add_linear(3, 3, seed)
        .unwrap()
        .add_module(Affine::new())
        .add_tanh()
}

fn compile_sgd(m: &mut Sequential) {
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
}

fn assert_invalid_argument<T: std::fmt::Debug>(r: Result<T, AutodiffError>, what: &str) {
    match r {
        Err(AutodiffError::InvalidArgument(_)) => {}
        other => panic!("{what}: InvalidArgument を期待したが {other:?}"),
    }
}

#[test]
fn forward_and_predict_match_manual_composition_bitwise() {
    let model = custom_model(SEED_A);
    let x = input();

    // 手動合成: linear 層は同 seed の単独モデルで、独自層は同じ演算列で合成する。
    let lin_only = Sequential::new().add_linear(3, 3, SEED_A).unwrap();
    let t = tape();
    let xv = t.var(&x);
    let h = lin_only.forward(&t, &xv).unwrap();
    let aff = Affine::new();
    let y = h
        .matmul(&t.var(&aff.weight))
        .unwrap()
        .add(&t.var(&aff.bias))
        .unwrap()
        .tanh();
    let expected = y.to_tensor();

    let t2 = tape();
    let x2 = t2.var(&x);
    let got = model.forward(&t2, &x2).unwrap().to_tensor();
    assert_eq!(bits(&got), bits(&expected));
    assert_eq!(bits(&model.predict(&x).unwrap()), bits(&expected));
}

#[test]
fn state_dict_uses_index_prefixed_keys_and_round_trips() {
    let src = custom_model(SEED_A);
    let sd = src.state_dict();
    assert!(sd.contains_key("1.weight"), "keys: {:?}", sd.keys());
    assert!(sd.contains_key("1.bias"));

    let mut dst = custom_model(SEED_B);
    assert_ne!(
        bits(&dst.predict(&input()).unwrap()),
        bits(&src.predict(&input()).unwrap())
    );
    dst.load_state_dict(sd).unwrap();
    assert_eq!(
        bits(&dst.predict(&input()).unwrap()),
        bits(&src.predict(&input()).unwrap())
    );
}

#[test]
fn load_state_dict_rejects_bad_key_and_shape_without_mutation() {
    let mut model = custom_model(SEED_A);
    let before = model.state_dict();
    let before_out = bits(&model.predict(&input()).unwrap());

    let mut bad_key = model.state_dict();
    bad_key.insert("9.weight".into(), t32(&[0.0], &[1]));
    assert!(model.load_state_dict(bad_key).is_err());

    let mut bad_shape = model.state_dict();
    bad_shape.insert("1.weight".into(), t32(&[0.0, 0.0], &[2]));
    assert!(model.load_state_dict(bad_shape).is_err());

    assert_eq!(bits(&model.predict(&input()).unwrap()), before_out);
    for (k, v) in &before {
        assert_eq!(bits(&model.state_dict()[k]), bits(v), "{k}");
    }
}

#[test]
fn parameters_are_listed_in_layer_order_and_apply_parameters_updates_custom_layer() {
    let mut model = custom_model(SEED_A);
    let names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, ["0.weight", "0.bias", "1.weight", "1.bias"]);

    let params: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();
    assert_eq!(params.len(), 4, "独自層のパラメータが層順で含まれる");

    let mut updated = params.clone();
    updated[2] = t32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    model.apply_parameters(updated).unwrap();
    assert_eq!(
        bits(&model.state_dict()["1.weight"]),
        bits(&t32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
    );
}

#[test]
fn fit_rejects_parametric_custom_layer_without_side_effects() {
    let mut model = custom_model(SEED_A);
    compile_sgd(&mut model);
    let before = model.state_dict();
    let training_before = model.training();

    assert_invalid_argument(model.fit(&input(), &target2(), FitConfig::new(1, 2)), "fit");
    assert!(model.is_compiled(), "compiled は書き戻される");
    assert_eq!(model.training(), training_before, "モードは不変");
    for (k, v) in &before {
        assert_eq!(bits(&model.state_dict()[k]), bits(v), "{k}");
    }

    assert_invalid_argument(
        model.fit_with_callbacks(&input(), &target2(), FitConfig::new(1, 2), None, &mut []),
        "fit_with_callbacks",
    );

    let mut amp = custom_model(SEED_A);
    amp.compile_with_amp(
        Optimizer::Sgd(SgdConfig::new(0.1)),
        Loss::Mse,
        AmpConfig::new(AmpDType::F16),
    )
    .unwrap();
    assert_invalid_argument(
        amp.fit(&input(), &target2(), FitConfig::new(1, 2)),
        "amp fit",
    );
}

#[test]
fn bind_rejects_forward_and_trainable_grads_for_parametric_custom_layer() {
    let model = custom_model(SEED_A);
    let t = tape();
    let x = t.var(&input());
    {
        let bound = model.bind(&t);
        assert_invalid_argument(bound.forward(&t, &x), "SequentialVars::forward");
    }

    // 外部 tape で forward + backward した勾配を `trainable_grads` へ渡しても拒否される。
    let t2 = tape();
    let x2 = t2.var(&input());
    let out = model.forward(&t2, &x2).unwrap();
    let loss = out.mul(&out).unwrap().sum(None).unwrap();
    let grads = t2.backward(&loss).unwrap();
    let bound = model.bind(&t2);
    assert_invalid_argument(bound.trainable_grads(&grads), "trainable_grads");
}

fn resident_is_unsupported<T: std::fmt::Debug>(r: Result<T, AutodiffError>, what: &str) {
    match r {
        Err(AutodiffError::Backend(fandhe_ai::BackendError::Unsupported(_))) => {}
        other => panic!("{what}: Backend(Unsupported) を期待したが {other:?}"),
    }
}

#[test]
fn resident_entries_reject_parametric_custom_layer() {
    let model = custom_model(SEED_A);
    let t = tape();
    match model.init_device_param_store(&t) {
        Err(fandhe_ai::BackendError::Unsupported(_)) => {}
        other => panic!(
            "init_device_param_store: Unsupported を期待したが {:?}",
            other.err()
        ),
    }

    // store は Linear のみのモデルから作る（入口検査が先に走ることの確認）。
    let lin = Sequential::new().add_linear(3, 3, SEED_A).unwrap();
    let mut store = lin.init_device_param_store(&t).unwrap();
    let x = t.var(&input());
    resident_is_unsupported(
        model.forward_resident(&t, &x, &mut store),
        "forward_resident",
    );
    resident_is_unsupported(model.predict_resident(&store, &input()), "predict_resident");
}

/// 無状態の独自層は活性化層と同じ扱いで `fit` を通過し、同 seed の `add_tanh` 版と bit 一致する。
#[test]
fn stateless_custom_layer_trains_identically_to_builtin_activation() {
    let build = |custom: bool| {
        let m = Sequential::new().add_linear(3, 4, SEED_A).unwrap();
        let m = if custom {
            m.add_module(CustomTanh)
        } else {
            m.add_tanh()
        };
        m.add_linear(4, 2, SEED_B).unwrap()
    };
    let mut a = build(true);
    let mut b = build(false);
    compile_sgd(&mut a);
    compile_sgd(&mut b);
    a.fit(&input(), &target2(), FitConfig::new(3, 2)).unwrap();
    b.fit(&input(), &target2(), FitConfig::new(3, 2)).unwrap();
    let (sa, sb) = (a.state_dict(), b.state_dict());
    assert_eq!(sa.len(), sb.len());
    for (k, v) in &sb {
        assert_eq!(bits(&sa[k]), bits(v), "{k}");
    }
}

#[test]
fn stateless_custom_layer_passes_resident_train_step() {
    let model = Sequential::new()
        .add_linear(3, 4, SEED_A)
        .unwrap()
        .add_module(CustomTanh)
        .add_linear(4, 2, SEED_B)
        .unwrap();
    let init = tape();
    let mut store = model.init_device_param_store(&init).unwrap();
    drop(init);

    let t = tape();
    let x = t.var(&input());
    let pred = model.forward_resident(&t, &x, &mut store).unwrap();
    let loss = pred.mul(&pred).unwrap().sum(None).unwrap();
    let grads = t.backward_device_param_store(&loss, &store).unwrap();
    t.step_device_param_store(&mut store, &grads, &FacadeSgdConfig::new(0.1))
        .unwrap();
}

#[test]
fn set_training_propagates_to_custom_layer() {
    let seen = Rc::new(Cell::new(true));
    let mut model = Sequential::new()
        .add_linear(3, 3, SEED_A)
        .unwrap()
        .add_module(Affine::observed(Rc::clone(&seen)));
    model.set_training(false);
    assert!(!seen.get());
    model.set_training(true);
    assert!(seen.get());
}

#[test]
fn evaluate_still_works_with_parametric_custom_layer() {
    let mut model = custom_model(SEED_A);
    compile_sgd(&mut model);
    let loss = model.evaluate(&input(), &target2(), 2).unwrap();
    assert!(loss.is_finite());
}
