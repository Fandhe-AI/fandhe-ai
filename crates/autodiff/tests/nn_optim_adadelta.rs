//! イシュー #2171（親 #2131「PyTorch／TF 置き換えの API 網羅」）:
//! `fandhe_ai_autodiff::nn::optim::Adadelta` の受け入れテスト。
//!
//! 受け入れ条件（`nn_optim_rmsprop.rs` 冒頭コメントと同じ規律）:
//! Adadelta は `Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数
//! のため VJP・parity テストの字義どおりの適用はできない。代わりに実
//! PyTorch 実行値 fixture との統一複合判定（`.claude/rules/
//! coding-rust.md` 既存 tolerance）・閉形式（t=1）一致・決定性（bit
//! 完全一致）で検証する。参照値は `tests/fixtures/adadelta-pytorch-
//! reference/adadelta_reference.json`（実 PyTorch 2.14.0+cpu 実行値。
//! README 参照）から読み込む。
//!
//! **契約: CI は `docs/spec`（submodule）を checkout しない**
//! （`nn_optim_adamw.rs` 冒頭コメントと同じ制約）。本ファイルは
//! `tests/fixtures/adadelta-pytorch-reference/`（本クレート配下に
//! 複製済み）のみを参照し、`docs/spec` 配下のいかなるファイルにも
//! 依存しない。

mod common;

use std::fs;
use std::path::PathBuf;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::optim::{Adadelta, AdadeltaConfig};
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
}

#[derive(Deserialize)]
struct Case {
    hyperparams: Hyperparams,
    steps: Vec<StepValues>,
}

#[derive(Deserialize)]
struct Hyperparams {
    lr: f32,
    rho: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Deserialize)]
struct StepValues {
    param_a: Vec<f32>,
    param_b: Vec<f32>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/adadelta-pytorch-reference/adadelta_reference.json");
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
    fixture
}

fn assert_close(actual: f32, expected: f32, context: &str) {
    assert!(
        common::req2_close(actual as f64, expected as f64),
        "{context}: actual={actual} expected={expected}"
    );
}

/// 受け入れ条件の本体: 4 ケース（既定・lr_rho・weight_decay・全部）×
/// 10 step の全パラメータ要素を PyTorch 実測値と突合する。
#[test]
fn adadelta_matches_pytorch_reference() {
    let fixture = load_fixture();

    for (case_name, case) in &fixture.cases {
        let cfg = AdadeltaConfig {
            lr: case.hyperparams.lr,
            rho: case.hyperparams.rho,
            eps: case.hyperparams.eps,
            weight_decay: case.hyperparams.weight_decay,
        };
        let mut opt = Adadelta::new(cfg)
            .unwrap_or_else(|e| panic!("case {case_name}: Adadelta::new 失敗: {e}"));

        let mut param_a = Tensor::new(fixture.init_a.clone(), &fixture.param_a_shape).unwrap();
        let mut param_b = Tensor::new(fixture.init_b.clone(), &fixture.param_b_shape).unwrap();

        for step in 0..fixture.steps {
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

/// 受け入れ条件「再現可能」の直接検証: 同一入力で 2 回独立に
/// `Adadelta::step` を 10 回呼び、結果がビット完全一致すること。
#[test]
fn adadelta_step_is_deterministic() {
    fn run() -> Vec<f32> {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        let mut param = Tensor::new(vec![1.0, -1.0, 0.5], &[3]).unwrap();
        for step in 0..10 {
            let grad = Tensor::new(vec![0.1 * step as f32, -0.2, 0.05], &[3]).unwrap();
            let out = opt.step(&[(&param, &grad)]).unwrap();
            param = out.into_iter().next().unwrap();
        }
        (0..3).map(|i| param.get(&[i]).unwrap()).collect()
    }

    let run1 = run();
    let run2 = run();
    assert_eq!(run1, run2, "同一入力で Adadelta::step の結果が一致しない");
}

/// Adadelta が `Linear`+`Relu`+`MseLoss` の 2 層 MLP を収束させることを
/// 確認する（`tests/nn_optim_rmsprop.rs::mlp_converges_with_rmsprop` と
/// 同型の収束テスト。optimizer 差し替え版）。
#[test]
fn mlp_converges_with_adadelta() {
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

    let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();

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
        "Adadelta での収束が不十分: initial={initial_loss} final={final_loss}"
    );
}
