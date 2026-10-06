//! イシュー #2655（親 #2654）: `fandhe_ai_autodiff::nn::optim::Rprop` の
//! 受け入れテスト。
//!
//! 受け入れ条件（`nn_optim_adamax.rs` 冒頭コメントと同じ規律）: Rprop は
//! `Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため VJP・
//! parity テストの字義どおりの適用はできない（optimizer は計算グラフ上の
//! 演算ではない）。代わりに実 PyTorch 実行値 fixture との統一複合判定・
//! NaN／±0／inf／アンダーフロー積のエッジケース・決定性（bit 完全一致）・
//! MLP 収束で検証する。参照値は `tests/fixtures/rprop-pytorch-reference/
//! rprop_reference.json`（実 PyTorch 2.14.0+cpu 実行値。README 参照）。
//!
//! **契約: CI は `docs/spec`（submodule）を checkout しない**。本ファイルは
//! 本クレート配下の fixture のみを参照する。

mod common;

use std::fs;
use std::path::PathBuf;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::optim::{Rprop, RpropConfig};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    param_a_shape: Vec<usize>,
    param_b_shape: Vec<usize>,
    steps: usize,
    init_a: Vec<f32>,
    init_b: Vec<f32>,
    grads_a: Vec<Vec<f32>>,
    grads_b: Vec<Vec<f32>>,
    cases: std::collections::BTreeMap<String, Case>,
    edge: Edge,
}

#[derive(Deserialize)]
struct Case {
    hyperparams: Hyperparams,
    steps: Vec<StepValues>,
}

#[derive(Deserialize)]
struct Hyperparams {
    lr: f32,
    eta_minus: f32,
    eta_plus: f32,
    step_size_min: f32,
    step_size_max: f32,
    lr_change_after_step: Option<usize>,
    lr_after: Option<f32>,
}

#[derive(Deserialize)]
struct StepValues {
    param_a: Vec<f32>,
    param_b: Vec<f32>,
}

/// NaN／inf を JSON で表せないため `f32` のビットパターン（`u32`）で保存した
/// エッジケース系列。
#[derive(Deserialize)]
struct Edge {
    hyperparams: Hyperparams,
    init_bits: Vec<u32>,
    grads_bits: Vec<Vec<u32>>,
    params_bits: Vec<Vec<u32>>,
}

fn config_of(hp: &Hyperparams) -> RpropConfig {
    RpropConfig {
        lr: hp.lr,
        eta_minus: hp.eta_minus,
        eta_plus: hp.eta_plus,
        step_size_min: hp.step_size_min,
        step_size_max: hp.step_size_max,
    }
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rprop-pytorch-reference/rprop_reference.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    let fixture: Fixture = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("fixture のパースに失敗（JSON 構造が壊れている）: {e}"));
    assert_eq!(
        fixture.init_a.len(),
        fixture.param_a_shape.iter().product::<usize>()
    );
    assert_eq!(
        fixture.init_b.len(),
        fixture.param_b_shape.iter().product::<usize>()
    );
    assert_eq!(fixture.grads_a.len(), fixture.steps);
    assert_eq!(fixture.grads_b.len(), fixture.steps);
    for case in fixture.cases.values() {
        assert_eq!(case.steps.len(), fixture.steps);
    }
    assert_eq!(
        fixture.edge.grads_bits.len(),
        fixture.edge.params_bits.len()
    );
    fixture
}

fn assert_close(actual: f32, expected: f32, context: &str) {
    assert!(
        common::req2_close(actual as f64, expected as f64),
        "{context}: actual={actual} expected={expected}"
    );
}

fn index_of(shape: &[usize], i: usize) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = vec![0usize; shape.len()];
    let mut rem = i;
    for d in (0..shape.len()).rev() {
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    idx
}

/// 受け入れ条件の本体: 4 ケース（既定・etas・tight_bounds・lr_change）×
/// 10 step の全パラメータ要素を PyTorch 実測値と突合する。
#[test]
fn rprop_matches_pytorch_reference() {
    let fixture = load_fixture();

    for (case_name, case) in &fixture.cases {
        let hp = &case.hyperparams;
        let mut opt = Rprop::new(config_of(hp))
            .unwrap_or_else(|e| panic!("case {case_name}: Rprop::new 失敗: {e}"));

        let mut param_a = Tensor::new(fixture.init_a.clone(), &fixture.param_a_shape).unwrap();
        let mut param_b = Tensor::new(fixture.init_b.clone(), &fixture.param_b_shape).unwrap();

        for step in 0..fixture.steps {
            if hp.lr_change_after_step == Some(step) {
                opt.set_lr(hp.lr_after.unwrap()).unwrap();
            }
            let grad_a =
                Tensor::new(fixture.grads_a[step].clone(), &fixture.param_a_shape).unwrap();
            let grad_b =
                Tensor::new(fixture.grads_b[step].clone(), &fixture.param_b_shape).unwrap();

            let updated = opt
                .step(&[(&param_a, &grad_a), (&param_b, &grad_b)])
                .unwrap_or_else(|e| panic!("case {case_name} step {step}: step() 失敗: {e}"));
            param_a = updated[0].clone();
            param_b = updated[1].clone();

            let expected = &case.steps[step];
            for i in 0..expected.param_a.len() {
                assert_close(
                    param_a.get(&index_of(&fixture.param_a_shape, i)).unwrap(),
                    expected.param_a[i],
                    &format!("case={case_name} step={step} param_a[{i}]"),
                );
            }
            for i in 0..expected.param_b.len() {
                assert_close(
                    param_b.get(&index_of(&fixture.param_b_shape, i)).unwrap(),
                    expected.param_b[i],
                    &format!("case={case_name} step={step} param_b[{i}]"),
                );
            }
        }
    }
}

/// NaN・`±0.0`・`±inf`・積がアンダーフローする微小値を含む系列を PyTorch
/// 実測値（ビットパターン）と突合する。期待値が NaN なら NaN クラス一致、
/// それ以外は統一複合判定。
#[test]
fn rprop_matches_pytorch_reference_edge_cases() {
    let fixture = load_fixture();
    let edge = &fixture.edge;
    let n = edge.init_bits.len();
    let mut opt = Rprop::new(config_of(&edge.hyperparams)).unwrap();
    let mut param = Tensor::new(
        edge.init_bits.iter().map(|b| f32::from_bits(*b)).collect(),
        &[n],
    )
    .unwrap();

    for (step, (grad_bits, expected_bits)) in
        edge.grads_bits.iter().zip(&edge.params_bits).enumerate()
    {
        let grad =
            Tensor::new(grad_bits.iter().map(|b| f32::from_bits(*b)).collect(), &[n]).unwrap();
        param = opt.step(&[(&param, &grad)]).unwrap().remove(0);
        for (i, eb) in expected_bits.iter().enumerate() {
            let expected = f32::from_bits(*eb);
            let actual = param.get(&[i]).unwrap();
            let ctx = format!("edge step={step} param[{i}]");
            if expected.is_nan() {
                assert!(actual.is_nan(), "{ctx}: actual={actual} expected=NaN");
            } else {
                assert_close(actual, expected, &ctx);
            }
        }
    }
}

/// 同一入力で 2 回独立に `Rprop::step` を 10 回呼び、結果がビット完全一致する。
#[test]
fn rprop_step_is_deterministic() {
    fn run() -> Vec<u32> {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let mut param = Tensor::new(vec![1.0, -1.0, 0.5], &[3]).unwrap();
        for step in 0..10 {
            let grad = Tensor::new(vec![0.1 * step as f32 - 0.3, -0.2, 0.05], &[3]).unwrap();
            let out = opt.step(&[(&param, &grad)]).unwrap();
            param = out.into_iter().next().unwrap();
        }
        (0..3).map(|i| param.get(&[i]).unwrap().to_bits()).collect()
    }
    assert_eq!(run(), run(), "同一入力で Rprop::step の結果が一致しない");
}

/// Rprop が `Linear`+`Relu`+`MseLoss` の 2 層 MLP を収束させることを確認する。
#[test]
fn mlp_converges_with_rprop() {
    const BATCH: usize = 4;
    const D_IN: usize = 8;
    const D_HIDDEN: usize = 16;
    const D_OUT: usize = 4;
    const STEPS: usize = 100;

    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x_data = Tensor::new(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]).unwrap();
    let y_data = Tensor::new(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]).unwrap();

    let relu = Relu;
    let mut l1 = Linear::new(D_IN, D_HIDDEN, true, 0x1111_1111).unwrap();
    let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, 0x2222_2222).unwrap();

    let mut opt = Rprop::new(RpropConfig::default()).unwrap();

    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for _ in 0..STEPS {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let l1v = l1.bind(&tape);
        let l2v = l2.bind(&tape);

        let h1 = l1v.forward(&x).unwrap();
        let a1 = relu.forward(&h1);
        let h2 = l2v.forward(&a1).unwrap();
        let loss = h2.mse_loss(&y).unwrap();

        let loss_value = loss.to_tensor().get(&[]).unwrap();
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = loss_value;

        let grads = tape.backward(&loss).unwrap();
        let l1_weight_grad = grads.get(&l1v.weight).unwrap().unwrap();
        let l1_bias_grad = grads.get(l1v.bias.as_ref().unwrap()).unwrap().unwrap();
        let l2_weight_grad = grads.get(&l2v.weight).unwrap().unwrap();
        let l2_bias_grad = grads.get(l2v.bias.as_ref().unwrap()).unwrap().unwrap();

        let updated = opt
            .step(&[
                (l1.weight(), l1_weight_grad),
                (l1.bias().unwrap(), l1_bias_grad),
                (l2.weight(), l2_weight_grad),
                (l2.bias().unwrap(), l2_bias_grad),
            ])
            .unwrap();

        l1 = Linear::from_parameters(updated[0].clone(), Some(updated[1].clone())).unwrap();
        l2 = Linear::from_parameters(updated[2].clone(), Some(updated[3].clone())).unwrap();
    }

    let initial_loss = initial_loss.unwrap();
    assert!(
        final_loss < 0.5 * initial_loss,
        "Rprop での収束が不十分: initial={initial_loss} final={final_loss}"
    );
}
