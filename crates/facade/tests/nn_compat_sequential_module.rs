//! `compat::Sequential` への facade `nn::Module` 実装（イシュー #3815・親 #3741。
//! 承認根拠は #3768 の所有者回答コメント 6554564412、設計は
//! `docs/facade-nn-module-exposure-decision.md` §23）の統合テスト。
//! facade だけに依存し（`fandhe_ai_autodiff` は import しない）、tolerance は使わず bit 一致で比較する。
//! 実機依存の経路はない（CPU 既定のみ）ため `#[ignore]` テストはない。利用者シナリオの受入テストは
//! #3816 が担い、本ファイルは契約（委譲の等価性・fail-closed の維持）を固定する。

use fandhe_ai::compat::{
    FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes,
};
use fandhe_ai::nn::Module;
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, TapeRef, Tensor, Var, tape};

mod common;
use common::temp_dir::TempDirGuard;

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

fn input() -> Tensor<f32> {
    t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3])
}

const SEED: u64 = 7;

/// Linear(3,4) → ReLU → Linear(4,2)。Linear→ReLU 融合の先読み走査が働く構成。
fn mlp() -> Sequential {
    Sequential::new()
        .add_linear(3, 4, SEED)
        .unwrap()
        .add_relu()
        .add_linear(4, 2, SEED + 1)
        .unwrap()
}

/// 学習時にマスクがかかる Dropout を含む構成（eval では恒等）。
fn with_dropout() -> Sequential {
    Sequential::new()
        .add_linear(3, 4, SEED)
        .unwrap()
        .add_relu()
        .add_dropout(0.5)
        .unwrap()
        .add_linear(4, 2, SEED + 1)
        .unwrap()
}

fn assert_invalid_argument<T: std::fmt::Debug>(r: Result<T, AutodiffError>, what: &str) {
    match r {
        Err(AutodiffError::InvalidArgument(_)) => {}
        other => panic!("{what}: InvalidArgument を期待したが {other:?}"),
    }
}

/// UFCS で trait 版 forward を呼ぶ（inherent の `forward(&Tape)` がメソッド解決で優先されるため）。
fn trait_forward<'t>(
    m: &impl Module,
    tape: TapeRef<'t>,
    x: &Var<'t>,
) -> Result<Var<'t>, AutodiffError> {
    Module::forward(m, tape, x)
}

/// 外側の独自層: 組み込みブロックを UFCS で呼び、入力を足す残差。
struct Residual {
    block: Sequential,
}

impl Module for Residual {
    fn forward<'t>(&self, tape: TapeRef<'t>, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let y = Module::forward(&self.block, tape, x)?;
        y.add(x)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.block
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (format!("block.{n}"), t))
            .collect()
    }
    fn set_training(&mut self, training: bool) {
        self.block.set_training(training);
    }
    fn training(&self) -> bool {
        self.block.training()
    }
}

#[test]
fn module_forward_matches_inherent_forward_bit_exact() {
    for build in [mlp as fn() -> Sequential, || {
        let mut m = with_dropout();
        m.set_training(false);
        m
    }] {
        let model = build();

        let t1 = tape();
        let x1 = t1.var(&input());
        let out1 = trait_forward(&model, TapeRef::from(&t1), &x1).unwrap();
        let loss1 = out1.mul(&out1).unwrap().sum(None).unwrap();
        let g1 = t1.backward(&loss1).unwrap();

        let t2 = tape();
        let x2 = t2.var(&input());
        let out2 = model.forward(&t2, &x2).unwrap();
        let loss2 = out2.mul(&out2).unwrap().sum(None).unwrap();
        let g2 = t2.backward(&loss2).unwrap();

        assert_eq!(bits(&out1.to_tensor()), bits(&out2.to_tensor()));
        assert_eq!(
            bits(g1.get(&x1).unwrap().expect("入力勾配")),
            bits(g2.get(&x2).unwrap().expect("入力勾配"))
        );
    }
}

#[test]
fn custom_module_calls_builtin_block_via_ufcs() {
    // 残差のため 3→3 のブロックを使う。
    let block = Sequential::new().add_linear(3, 3, SEED).unwrap().add_relu();
    let res = Residual { block };
    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let out = Module::forward(&res, tr, &x).unwrap();

    // 手組み: inherent forward の出力 + 入力。
    let t2 = tape();
    let x2 = t2.var(&input());
    let manual = res.block.forward(&t2, &x2).unwrap().add(&x2).unwrap();
    assert_eq!(bits(&out.to_tensor()), bits(&manual.to_tensor()));

    // 形が合わないブロック（3→2）は型付き Err（黙った成功にならない）。
    let bad = Residual { block: mlp() };
    assert!(Module::forward(&bad, tr, &x).is_err());
}

#[test]
fn training_mode_propagates_into_nested_compat_sequential() {
    // 外側が facade の nn::Sequential。
    let mut outer = fandhe_ai::nn::Sequential::new().add(with_dropout());
    assert!(Module::training(&outer));
    outer.set_training(false);
    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let a = Module::forward(&outer, tr, &x).unwrap();
    let b = Module::forward(&outer, tr, &x).unwrap();
    assert_eq!(bits(&a.to_tensor()), bits(&b.to_tensor()), "eval は決定的");

    // 外側が compat::Sequential::add_module。
    let mut outer2 = Sequential::new().add_module(with_dropout());
    outer2.set_training(false);
    let c = outer2.forward(&t, &x).unwrap();
    assert_eq!(bits(&a.to_tensor()), bits(&c.to_tensor()));

    // training() は既定の true ではなく委譲された値を返す。
    let mut inner = with_dropout();
    Module::set_training(&mut inner, false);
    assert!(!Module::training(&inner));
    Module::set_training(&mut inner, true);
    assert!(Module::training(&inner));
}

#[test]
fn nested_named_parameters_keys_and_order() {
    let outer = Sequential::new().add_relu().add_module(mlp());
    let keys: Vec<String> = Module::named_parameters(&outer)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(keys, ["1.0.weight", "1.0.bias", "1.2.weight", "1.2.bias"]);
    // 3*4 + 4 + 4*2 + 2
    assert_eq!(Module::parameter_count(&outer), 26);
    assert_eq!(Module::parameter_count(&mlp()), 26);
}

#[test]
fn nested_state_dict_safetensors_roundtrip() {
    let src = Sequential::new().add_module(mlp());
    let sd = Module::state_dict(&src);
    let bytes = save_safetensors_f32_to_bytes(&sd, None).expect("save");
    let loaded = load_safetensors_f32_from_bytes(&bytes).expect("load");

    let mut dst = Sequential::new().add_module(
        Sequential::new()
            .add_linear(3, 4, SEED + 100)
            .unwrap()
            .add_relu()
            .add_linear(4, 2, SEED + 101)
            .unwrap(),
    );
    Module::load_state_dict(&mut dst, loaded).unwrap();
    for (k, v) in &sd {
        assert_eq!(bits(&Module::state_dict(&dst)[k]), bits(v), "{k}");
    }
}

#[test]
fn nested_load_state_dict_rejects_and_rolls_back() {
    let mut model = Sequential::new().add_module(mlp());
    let before = Module::state_dict(&model);
    let unchanged = |m: &Sequential| {
        let now = Module::state_dict(m);
        before.iter().all(|(k, v)| bits(&now[k]) == bits(v))
    };

    let mut unknown = before.clone();
    unknown.insert("0.9.weight".into(), t32(&[0.0], &[1]));
    assert_invalid_argument(Module::load_state_dict(&mut model, unknown), "未知キー");
    assert!(unchanged(&model));

    let mut bad_shape = before.clone();
    bad_shape.insert("0.0.weight".into(), t32(&[0.0; 4], &[2, 2]));
    assert_invalid_argument(Module::load_state_dict(&mut model, bad_shape), "shape");
    assert!(unchanged(&model));

    // set_parameter 単体の拒否（範囲外 index・不正 index・shape 不一致）。
    for name in ["7.0.weight", "x.weight", "0.0.nope"] {
        assert_invalid_argument(
            Module::set_parameter(&mut model, name, t32(&[0.0; 12], &[3, 4])),
            name,
        );
    }
    // shape 不一致は層側の型付き Shape エラー（InvalidArgument ではない）で拒否される。
    assert!(matches!(
        Module::set_parameter(&mut model, "0.0.weight", t32(&[0.0; 4], &[2, 2])),
        Err(AutodiffError::Shape(_))
    ));
    assert!(unchanged(&model));
}

#[test]
fn freeze_on_compat_sequential_module_is_fail_closed() {
    let mut with_params = mlp();
    assert_invalid_argument(
        Module::set_requires_grad(&mut with_params, false),
        "set_requires_grad",
    );
    assert_invalid_argument(Module::freeze(&mut with_params), "freeze");

    let mut stateless = Sequential::new().add_relu();
    assert!(Module::freeze(&mut stateless).is_ok());

    let mut outer = fandhe_ai::nn::Sequential::new().add(mlp());
    assert_invalid_argument(
        Module::freeze(&mut outer),
        "外側 nn::Sequential 経由の freeze",
    );
}

#[test]
fn compat_sequential_module_is_leaf_in_introspection() {
    let seq = mlp();
    assert!(Module::children(&seq).is_empty());
    assert!(Module::named_modules(&seq).is_empty());

    let outer = fandhe_ai::nn::Sequential::new().add(mlp());
    let s = fandhe_ai::nn::summary(&outer);
    assert!(s.contains("Submodules: 1"), "{s}");
}

#[test]
fn outer_fit_rejects_parametric_nested_compat_sequential() {
    let x = input();
    let y = t32(&[0.1, -0.2, 0.3, 0.4], &[2, 2]);

    let nested = || Sequential::new().add_linear(3, 2, SEED).unwrap();
    let mut outer = Sequential::new().add_module(nested());
    outer
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    assert_invalid_argument(outer.fit(&x, &y, FitConfig::new(1, 2)), "外側 fit");

    let probe = tape();
    match outer.init_device_param_store(&probe) {
        Err(fandhe_ai::BackendError::Unsupported(_)) => {}
        other => panic!("常駐経路: Unsupported を期待したが {:?}", other.err()),
    }

    // 内側単体の fit は従来どおり成功する。
    let mut inner = nested();
    inner
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    inner.fit(&x, &y, FitConfig::new(1, 2)).unwrap();
}

#[test]
fn nested_save_model_and_onnx_export_fail_closed() {
    let outer = Sequential::new().add_module(mlp());
    let guard = TempDirGuard::new("nested-save-3815");
    match save_model(&outer, guard.path()) {
        Err(ModelIoError::UnsupportedModel { .. }) => {}
        other => panic!("UnsupportedModel を期待したが {other:?}"),
    }
    // 保存は拒否されたため復元もできない（load_model は Err）。
    assert!(load_model(guard.path()).is_err());
    match OnnxModel::from_sequential(&outer) {
        Err(OnnxError::UnsupportedLayer { .. }) => {}
        other => panic!("UnsupportedLayer を期待したが {:?}", other.err()),
    }
}
