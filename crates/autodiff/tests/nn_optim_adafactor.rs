//! イシュー #2656（親 #2654）: `fandhe_ai_autodiff::nn::optim::Adafactor` の
//! 受け入れテスト。
//!
//! 受け入れ条件（`nn_optim_rprop.rs` 冒頭コメントと同じ規律）: Adafactor は
//! `Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため VJP・
//! parity テストの字義どおりの適用はできない（optimizer は計算グラフ上の
//! 演算ではない）。代わりに実 PyTorch 実行値 fixture との統一複合判定・
//! NaN／±0／inf／極小値のエッジケース・決定性（bit 完全一致）・MLP 収束で
//! 検証する。参照値は `tests/fixtures/adafactor-pytorch-reference/
//! adafactor_reference.json`（実 `torch.optim.Adafactor` 2.14.0+cpu 実行値。
//! README 参照）。
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
use fandhe_ai_autodiff::nn::optim::{Adafactor, AdafactorConfig};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    shapes: Vec<Vec<usize>>,
    steps: usize,
    init: Vec<Vec<f32>>,
    /// `grads[step][param]`
    grads: Vec<Vec<Vec<f32>>>,
    cases: std::collections::BTreeMap<String, Case>,
    edge: Edge,
}

#[derive(Deserialize)]
struct Case {
    hyperparams: Hyperparams,
    /// `steps[step][param]`
    steps: Vec<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct Hyperparams {
    lr: f32,
    beta2_decay: f32,
    eps1: f32,
    eps2: f32,
    d: f32,
    weight_decay: f32,
    lr_change_after_step: Option<usize>,
    lr_after: Option<f32>,
}

/// NaN／inf を JSON で表せないため `f32` のビットパターン（`u32`）で保存した
/// エッジケース系列。
#[derive(Deserialize)]
struct Edge {
    hyperparams: Hyperparams,
    shapes: Vec<Vec<usize>>,
    init_bits: Vec<Vec<u32>>,
    /// `grads_bits[step][param]`
    grads_bits: Vec<Vec<Vec<u32>>>,
    params_bits: Vec<Vec<Vec<u32>>>,
}

fn config_of(hp: &Hyperparams) -> AdafactorConfig {
    AdafactorConfig {
        lr: hp.lr,
        beta2_decay: hp.beta2_decay,
        eps1: hp.eps1,
        eps2: hp.eps2,
        d: hp.d,
        weight_decay: hp.weight_decay,
    }
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/adafactor-pytorch-reference/adafactor_reference.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    let fixture: Fixture = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("fixture のパースに失敗（JSON 構造が壊れている）: {e}"));
    assert_eq!(fixture.init.len(), fixture.shapes.len());
    for (init, shape) in fixture.init.iter().zip(&fixture.shapes) {
        assert_eq!(init.len(), shape.iter().product::<usize>());
    }
    assert_eq!(fixture.grads.len(), fixture.steps);
    for step_grads in &fixture.grads {
        assert_eq!(step_grads.len(), fixture.shapes.len());
        for (g, shape) in step_grads.iter().zip(&fixture.shapes) {
            assert_eq!(g.len(), shape.iter().product::<usize>());
        }
    }
    for case in fixture.cases.values() {
        assert_eq!(case.steps.len(), fixture.steps);
        for step in &case.steps {
            assert_eq!(step.len(), fixture.shapes.len());
        }
    }
    assert_eq!(
        fixture.edge.grads_bits.len(),
        fixture.edge.params_bits.len()
    );
    assert_eq!(fixture.edge.init_bits.len(), fixture.edge.shapes.len());
    fixture
}

fn assert_close(actual: f32, expected: f32, context: &str) {
    assert!(
        common::req2_close(actual as f64, expected as f64),
        "{context}: actual={actual} expected={expected}"
    );
}

/// rank 0（空 shape）も扱える添字復元。
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

fn flat(t: &Tensor<f32>) -> Vec<f32> {
    let n: usize = t.shape().iter().product();
    (0..n)
        .map(|i| t.get(&index_of(t.shape(), i)).unwrap())
        .collect()
}

/// 受け入れ条件の本体: 7 ケース × 10 step × 4 パラメータ（rank 2・1・3・0）
/// の全要素を PyTorch 実測値と突合する。
#[test]
fn adafactor_matches_pytorch_reference() {
    let fixture = load_fixture();

    for (case_name, case) in &fixture.cases {
        let hp = &case.hyperparams;
        let mut opt = Adafactor::new(config_of(hp))
            .unwrap_or_else(|e| panic!("case {case_name}: Adafactor::new 失敗: {e}"));
        let mut params: Vec<Tensor<f32>> = fixture
            .init
            .iter()
            .zip(&fixture.shapes)
            .map(|(v, s)| Tensor::new(v.clone(), s).unwrap())
            .collect();

        for step in 0..fixture.steps {
            if hp.lr_change_after_step == Some(step) {
                opt.set_lr(hp.lr_after.unwrap()).unwrap();
            }
            let grads: Vec<Tensor<f32>> = fixture.grads[step]
                .iter()
                .zip(&fixture.shapes)
                .map(|(v, s)| Tensor::new(v.clone(), s).unwrap())
                .collect();
            let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                params.iter().zip(grads.iter()).collect();
            let updated = opt
                .step(&pairs)
                .unwrap_or_else(|e| panic!("case {case_name} step {step}: step() 失敗: {e}"));
            params = updated;

            for (k, expected) in case.steps[step].iter().enumerate() {
                let got = flat(&params[k]);
                assert_eq!(got.len(), expected.len());
                for i in 0..expected.len() {
                    assert_close(
                        got[i],
                        expected[i],
                        &format!("case={case_name} step={step} param{k}[{i}]"),
                    );
                }
            }
        }
    }
}

/// NaN・`±0.0`・`±inf`・極小値を含む系列（rank 1 と rank 2）を PyTorch
/// 実測値（ビットパターン）と突合する。期待値が NaN なら NaN クラス一致、
/// inf は符号込み一致、それ以外は統一複合判定。
#[test]
fn adafactor_matches_pytorch_reference_edge_cases() {
    let fixture = load_fixture();
    let edge = &fixture.edge;
    let mut opt = Adafactor::new(config_of(&edge.hyperparams)).unwrap();
    let mut params: Vec<Tensor<f32>> = edge
        .init_bits
        .iter()
        .zip(&edge.shapes)
        .map(|(bits, s)| Tensor::new(bits.iter().map(|b| f32::from_bits(*b)).collect(), s).unwrap())
        .collect();

    for (step, (grad_bits, expected_bits)) in
        edge.grads_bits.iter().zip(&edge.params_bits).enumerate()
    {
        let grads: Vec<Tensor<f32>> = grad_bits
            .iter()
            .zip(&edge.shapes)
            .map(|(bits, s)| {
                Tensor::new(bits.iter().map(|b| f32::from_bits(*b)).collect(), s).unwrap()
            })
            .collect();
        let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> = params.iter().zip(grads.iter()).collect();
        params = opt.step(&pairs).unwrap();
        for (k, exp_k) in expected_bits.iter().enumerate() {
            let got = flat(&params[k]);
            for (i, eb) in exp_k.iter().enumerate() {
                let expected = f32::from_bits(*eb);
                let actual = got[i];
                let ctx = format!("edge step={step} param{k}[{i}]");
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{ctx}: actual={actual} expected=NaN");
                } else if expected.is_infinite() {
                    assert_eq!(actual, expected, "{ctx}");
                } else {
                    assert_close(actual, expected, &ctx);
                }
            }
        }
    }
}

/// 同一入力で 2 回独立に `Adafactor::step` を 10 回呼び、結果がビット完全一致する。
#[test]
fn adafactor_step_is_deterministic() {
    fn run() -> Vec<u32> {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let mut param = Tensor::new(vec![1.0, -1.0, 0.5, 0.25, -0.3, 0.8], &[2, 3]).unwrap();
        for step in 0..10 {
            let grad = Tensor::new(
                vec![
                    0.1 * step as f32 - 0.3,
                    -0.2,
                    0.05,
                    0.4,
                    -0.1 * step as f32,
                    0.7,
                ],
                &[2, 3],
            )
            .unwrap();
            param = opt.step(&[(&param, &grad)]).unwrap().remove(0);
        }
        flat(&param).iter().map(|x| x.to_bits()).collect()
    }
    assert_eq!(
        run(),
        run(),
        "同一入力で Adafactor::step の結果が一致しない"
    );
}

/// Adafactor が `Linear`+`Relu`+`MseLoss` の 2 層 MLP を収束させることを
/// 確認する。既定 `lr=1e-2` は 100 step では更新量が小さい（`rho` が
/// `lr` で頭打ちかつ `alpha` が param の RMS に比例する）ため `lr=0.1` を使う。
#[test]
fn mlp_converges_with_adafactor() {
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

    let mut opt = Adafactor::new(AdafactorConfig {
        lr: 0.1,
        ..AdafactorConfig::default()
    })
    .unwrap();

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
        "Adafactor での収束が不十分: initial={initial_loss} final={final_loss}"
    );
}
