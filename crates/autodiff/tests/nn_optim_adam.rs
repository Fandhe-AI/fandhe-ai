//! イシュー #1742（親 #1610）: `fandhe_ai_autodiff::nn::optim::Adam`
//! （coupled L2 weight decay）の受け入れテスト。
//!
//! `Adam` は `AdamW` の decay 適用箇所（勾配へ加算 vs パラメータへ直接
//! 減算）を分岐する薄い派生であり（`crates/autodiff/src/nn/optim/adam.rs`
//! doc・`docs/compat-feature-gap.md` §2.9）、新規 PyTorch 参照値 fixture
//! は追加しない。代わりに以下 3 段で受け入れを担保する:
//!
//! 1. `weight_decay=0` ケースでは `Adam` と `torch.optim.Adam` は定義上
//!    `torch.optim.AdamW(weight_decay=0)` と完全に一致するため、既存の
//!    `adamw-pytorch-reference/adamw_reference.json`（実 PyTorch 2.13.0+cpu
//!    実行値）の `weight_decay_zero` ケースへ `Adam` を直接突合する
//!    （[`adam_matches_pytorch_reference_weight_decay_zero_case`]）。
//! 2. 全 3 ケースのハイパーパラメータで `weight_decay` を 0 に強制し、
//!    `Adam` と `AdamW` が bit 完全一致することを固定する
//!    （[`adam_wd_zero_bit_matches_adamw_wd_zero`]）。
//! 3. `weight_decay>0` は PyTorch `_single_tensor_adam` の定義
//!    （`grad = grad.add(param, alpha=weight_decay)`）に基づく恒等式
//!    `Adam(wd).step(p, g) == AdamW(wd=0).step(p, mul_add(wd, p, g))`
//!    を bit 完全一致で固定する（[`adam_coupled_l2_bit_matches_adamw_on_shifted_grads`]）。
//!
//! **数値判定の規律**: 1 のみバックエンド間統一複合判定「相対誤差 1e-3
//! 未満 または 絶対誤差 1e-5 未満」（`.claude/rules/coding-rust.md`）を
//! 使う（新設 tolerance の緩和ではない）。2・3 は bit 完全一致
//! （`assert_eq!`）で、既存複合判定より厳しい規律である。
//!
//! **契約: CI（self-hosted）は `docs/spec`（submodule）を checkout
//! しない**（`nn_optim_adamw.rs` 冒頭コメントと同じ制約）。本ファイルは
//! `tests/fixtures/adamw-pytorch-reference/`（既存・複製済み）のみを
//! 参照し、`docs/spec` 配下のいかなるファイルにも依存しない。

mod common;

use std::fs;
use std::path::PathBuf;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig, AdamW, AdamWConfig};
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

#[derive(Deserialize, Clone, Copy)]
struct Hyperparams {
    lr: f32,
    beta1: f32,
    beta2: f32,
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
        .join("tests/fixtures/adamw-pytorch-reference/adamw_reference.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    let fixture: Fixture = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("fixture のパースに失敗（JSON 構造が壊れている）: {e}"));
    // A03: 外部由来（テスト fixture とはいえ）データの要素数・shape を
    // 使う前に検証する（`nn_optim_adamw.rs::load_fixture` と同一規律）。
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

/// 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
/// （`.claude/rules/coding-rust.md`）。既存 tolerance の値を変更せず
/// そのまま使う。
fn assert_close(actual: f32, expected: f32, context: &str) {
    let abs_err = (actual - expected).abs();
    let rel_err = if expected != 0.0 {
        abs_err / expected.abs()
    } else {
        abs_err
    };
    assert!(
        rel_err < 1e-3 || abs_err < 1e-5,
        "{context}: actual={actual} expected={expected} (abs_err={abs_err}, rel_err={rel_err})"
    );
}

/// `Tensor::get` は多次元添字を要求するため、行優先の平坦添字 `i` を
/// `shape` から多次元添字へ復元する（`nn_optim_adamw.rs::index_of` と
/// 同一パターン）。
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

/// 受け入れ段 1: `weight_decay=0` ケースで実 PyTorch 実行値
/// （`weight_decay_zero` ケース。`hyperparams.weight_decay == 0.0`）と
/// `Adam` の 10 step 全パラメータ要素を突合する。`weight_decay=0` では
/// `torch.optim.Adam` と `torch.optim.AdamW` は定義上完全に一致するため、
/// 新規 fixture を用意せず既存 `AdamW` 参照値をそのまま `Adam` の受け入れ
/// 根拠として使える。
#[test]
fn adam_matches_pytorch_reference_weight_decay_zero_case() {
    let fixture = load_fixture();
    let case = fixture
        .cases
        .get("weight_decay_zero")
        .expect("fixture に weight_decay_zero ケースが存在するはず");
    assert_eq!(
        case.hyperparams.weight_decay, 0.0,
        "test fixture: weight_decay_zero ケースの前提"
    );

    let cfg = AdamConfig {
        lr: case.hyperparams.lr,
        beta1: case.hyperparams.beta1,
        beta2: case.hyperparams.beta2,
        eps: case.hyperparams.eps,
        weight_decay: case.hyperparams.weight_decay,
    };
    let mut opt = Adam::new(cfg).unwrap_or_else(|e| panic!("Adam::new 失敗: {e}"));

    let mut param_a = Tensor::new(fixture.init_a.clone(), &fixture.param_a_shape).unwrap();
    let mut param_b = Tensor::new(fixture.init_b.clone(), &fixture.param_b_shape).unwrap();

    for step in 0..fixture.steps {
        let grad_a = Tensor::new(fixture.grads_a[step].clone(), &fixture.param_a_shape).unwrap();
        let grad_b = Tensor::new(fixture.grads_b[step].clone(), &fixture.param_b_shape).unwrap();

        let updated = opt
            .step(&[(&param_a, &grad_a), (&param_b, &grad_b)])
            .unwrap_or_else(|e| panic!("step {step}: step() 失敗: {e}"));
        param_a = updated[0].clone();
        param_b = updated[1].clone();

        let expected = &case.steps[step];
        for i in 0..expected.param_a.len() {
            assert_close(
                param_a.get(&index_of(&fixture.param_a_shape, i)).unwrap(),
                expected.param_a[i],
                &format!("step={step} param_a[{i}]"),
            );
        }
        for i in 0..expected.param_b.len() {
            assert_close(
                param_b.get(&index_of(&fixture.param_b_shape, i)).unwrap(),
                expected.param_b[i],
                &format!("step={step} param_b[{i}]"),
            );
        }
    }
}

/// 受け入れ段 2: fixture の全 3 ケースのハイパーパラメータで
/// `weight_decay` を 0 に強制し、`Adam` と `AdamW` が同一入力・同一 10
/// step で bit 完全一致することを固定する（`adam.rs` の decay 分岐が
/// `weight_decay == 0.0` で「演算自体を skip し生の `g` を使う」ため、
/// `AdamW` の decay 乗算〈`* 1.0`〉と bit 単位で揃うという構造的根拠を
/// 直接検証する）。
#[test]
fn adam_wd_zero_bit_matches_adamw_wd_zero() {
    let fixture = load_fixture();

    for (case_name, case) in &fixture.cases {
        let hp = Hyperparams {
            weight_decay: 0.0,
            ..case.hyperparams
        };

        let mut adam = Adam::new(AdamConfig {
            lr: hp.lr,
            beta1: hp.beta1,
            beta2: hp.beta2,
            eps: hp.eps,
            weight_decay: hp.weight_decay,
        })
        .unwrap_or_else(|e| panic!("case {case_name}: Adam::new 失敗: {e}"));
        let mut adamw = AdamW::new(AdamWConfig {
            lr: hp.lr,
            beta1: hp.beta1,
            beta2: hp.beta2,
            eps: hp.eps,
            weight_decay: hp.weight_decay,
        })
        .unwrap_or_else(|e| panic!("case {case_name}: AdamW::new 失敗: {e}"));

        let mut param_a_adam = Tensor::new(fixture.init_a.clone(), &fixture.param_a_shape).unwrap();
        let mut param_b_adam = Tensor::new(fixture.init_b.clone(), &fixture.param_b_shape).unwrap();
        let mut param_a_adamw =
            Tensor::new(fixture.init_a.clone(), &fixture.param_a_shape).unwrap();
        let mut param_b_adamw =
            Tensor::new(fixture.init_b.clone(), &fixture.param_b_shape).unwrap();

        for step in 0..fixture.steps {
            let grad_a =
                Tensor::new(fixture.grads_a[step].clone(), &fixture.param_a_shape).unwrap();
            let grad_b =
                Tensor::new(fixture.grads_b[step].clone(), &fixture.param_b_shape).unwrap();

            let updated_adam = adam
                .step(&[(&param_a_adam, &grad_a), (&param_b_adam, &grad_b)])
                .unwrap_or_else(|e| panic!("case {case_name} step {step}: Adam::step 失敗: {e}"));
            let updated_adamw = adamw
                .step(&[(&param_a_adamw, &grad_a), (&param_b_adamw, &grad_b)])
                .unwrap_or_else(|e| panic!("case {case_name} step {step}: AdamW::step 失敗: {e}"));

            param_a_adam = updated_adam[0].clone();
            param_b_adam = updated_adam[1].clone();
            param_a_adamw = updated_adamw[0].clone();
            param_b_adamw = updated_adamw[1].clone();

            for i in 0..fixture.init_a.len() {
                let idx = index_of(&fixture.param_a_shape, i);
                assert_eq!(
                    param_a_adam.get(&idx).unwrap(),
                    param_a_adamw.get(&idx).unwrap(),
                    "case={case_name} step={step} param_a[{i}]: weight_decay=0 で Adam と AdamW は bit 一致するはず"
                );
            }
            for i in 0..fixture.init_b.len() {
                let idx = index_of(&fixture.param_b_shape, i);
                assert_eq!(
                    param_b_adam.get(&idx).unwrap(),
                    param_b_adamw.get(&idx).unwrap(),
                    "case={case_name} step={step} param_b[{i}]: weight_decay=0 で Adam と AdamW は bit 一致するはず"
                );
            }
        }
    }
}

/// 受け入れ段 3: PyTorch `_single_tensor_adam` の定義に基づく恒等式
/// `Adam(wd).step(p, g) == AdamW(wd=0).step(p, mul_add(wd, p, g))` を
/// bit 完全一致で固定する（`grad.add(param, alpha=weight_decay)` を
/// 呼び出し側で先に適用してから decoupled 経路〈decay 乗算なし〉へ渡す
/// のと、`adam.rs` が内部で行う coupled 経路が同じ演算列になることの
/// 直接検証。`weight_decay>0` に対する受け入れの主根拠）。
#[test]
fn adam_coupled_l2_bit_matches_adamw_on_shifted_grads() {
    let weight_decay = 0.1f32;
    let cfg_adam = AdamConfig {
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay,
    };
    let cfg_adamw = AdamWConfig {
        lr: cfg_adam.lr,
        beta1: cfg_adam.beta1,
        beta2: cfg_adam.beta2,
        eps: cfg_adam.eps,
        weight_decay: 0.0,
    };

    let mut adam = Adam::new(cfg_adam).unwrap();
    let mut adamw = AdamW::new(cfg_adamw).unwrap();

    let mut param_adam = Tensor::new(vec![0.5, -1.2, 3.0, 0.0], &[4]).unwrap();
    let mut param_adamw = Tensor::new(vec![0.5, -1.2, 3.0, 0.0], &[4]).unwrap();

    let mut rng = Xorshift64Star::new(0xADA_1742);
    for step in 0..10 {
        let raw_grad_data = rng.fill_vec(4);
        let raw_grad = Tensor::new(raw_grad_data.clone(), &[4]).unwrap();

        // AdamW(wd=0) 側は呼び出し元で `g' = mul_add(wd, p, g)`（実装と
        // 同一の演算）を先に適用してから渡す（PyTorch
        // `grad.add(param, alpha=weight_decay)` の呼び出し順再現）。
        let shifted_grad_data: Vec<f32> = (0..4)
            .map(|i| {
                let p = param_adamw.get(&[i]).unwrap();
                f32::mul_add(weight_decay, p, raw_grad_data[i])
            })
            .collect();
        let shifted_grad = Tensor::new(shifted_grad_data, &[4]).unwrap();

        let updated_adam = adam
            .step(&[(&param_adam, &raw_grad)])
            .unwrap_or_else(|e| panic!("step {step}: Adam::step 失敗: {e}"));
        let updated_adamw = adamw
            .step(&[(&param_adamw, &shifted_grad)])
            .unwrap_or_else(|e| panic!("step {step}: AdamW::step 失敗: {e}"));

        param_adam = updated_adam.into_iter().next().unwrap();
        param_adamw = updated_adamw.into_iter().next().unwrap();

        for i in 0..4 {
            assert_eq!(
                param_adam.get(&[i]).unwrap(),
                param_adamw.get(&[i]).unwrap(),
                "step={step} index={i}: coupled Adam と shifted-grad AdamW(wd=0) は bit 一致するはず"
            );
        }
    }
}

/// 受け入れ条件「再現可能」の直接検証（`nn_optim_adamw.rs::
/// adamw_step_is_deterministic` と同型）: 同一入力で 2 回独立に
/// `Adam::step` を 10 回呼び、結果がビット完全一致すること。
#[test]
fn adam_step_is_deterministic() {
    fn run() -> Vec<f32> {
        let cfg = AdamConfig {
            weight_decay: 0.05,
            ..AdamConfig::default()
        };
        let mut opt = Adam::new(cfg).unwrap();
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
    assert_eq!(run1, run2, "同一入力で Adam::step の結果が一致しない");
}

/// Adam（`lr=0.01, weight_decay=1e-4`）が `Linear`+`Relu`+`MseLoss` の
/// 2 層 MLP を収束させることを確認する（`nn_optim_adamw.rs::
/// mlp_converges_with_adamw` と同型の収束テスト。`weight_decay>0` を
/// 指定し coupled 経路を実際に通す）。
#[test]
fn mlp_converges_with_adam() {
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

    let cfg = AdamConfig {
        lr: 0.01,
        weight_decay: 1e-4,
        ..AdamConfig::default()
    };
    let mut opt = Adam::new(cfg).unwrap();

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
        "Adam での収束が不十分: initial={initial_loss} final={final_loss}"
    );
}
