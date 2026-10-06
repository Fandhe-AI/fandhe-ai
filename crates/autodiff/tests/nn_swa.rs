//! イシュー #2658（親 #2657）の [`fandhe_ai_autodiff::nn::AveragedModel`]・
//! [`fandhe_ai_autodiff::nn::optim::SwaLr`] の統合テスト。
//!
//! - **実 PyTorch 2.14.0 の実行値 fixture**
//!   （`tests/fixtures/swa-pytorch-reference/swa_reference.json`・生成条件は同ディレクトリの
//!   `README.md`）と統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
//!   `common::req2_close`）で突合する。tolerance 定数は新設・変更しない。
//!   `AveragedModel` は PyTorch の除算形と演算順が異なるため bit 一致は主張しない。
//! - `Module` 統合（`nn_ema.rs` と同型。`Linear → Relu → Linear` の `nn::Sequential`）。
//! - ホスト計算のみで実機（CUDA／Metal）非依存のため `#[ignore]` 分離は行わない。

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use common::req2_close;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::optim::{
    CosineAnnealingLr, LrScheduler, SequentialLr, SwaAnneal, SwaLr,
};
use fandhe_ai_autodiff::nn::{AveragedModel, Linear, Module, Sequential};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    swalr_cases: Vec<LrCase>,
    swalr_chain_case: ChainCase,
    avg_cases: Vec<AvgCase>,
}

#[derive(Deserialize)]
struct LrCase {
    name: String,
    base_lr: f32,
    swa_lr: f32,
    anneal_epochs: usize,
    strategy: String,
    lrs: Vec<f64>,
}

#[derive(Deserialize)]
struct ChainCase {
    base_lr: f32,
    swa_lr: f32,
    t_max: usize,
    switch_step: usize,
    anneal_epochs: usize,
    lrs: Vec<f64>,
}

#[derive(Deserialize)]
struct AvgCase {
    name: String,
    shapes: Vec<Vec<usize>>,
    init: Vec<Vec<u32>>,
    steps: Vec<AvgStep>,
}

#[derive(Deserialize)]
struct AvgStep {
    params: Vec<Vec<u32>>,
    averaged: Vec<Vec<u32>>,
    n_averaged: u64,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/swa-pytorch-reference/swa_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn tensors(bits: &[Vec<u32>], shapes: &[Vec<usize>]) -> Vec<Tensor<f32>> {
    bits.iter()
        .zip(shapes)
        .map(|(b, s)| Tensor::new(from_bits(b), s).unwrap())
        .collect()
}

fn named_refs(ts: &[Tensor<f32>]) -> Vec<(String, &Tensor<f32>)> {
    ts.iter()
        .enumerate()
        .map(|(i, t)| (i.to_string(), t))
        .collect()
}

fn anneal_of(s: &str) -> SwaAnneal {
    match s {
        "cos" => SwaAnneal::Cos,
        "linear" => SwaAnneal::Linear,
        other => panic!("未知の strategy: {other}"),
    }
}

/// スケジューラ値の検査用: 相対誤差 1e-6（期待値 0 は完全一致）。#2176 の先例と同じ。
/// スケジューラ値の検査であり、REQ-2 の tolerance 定数とは無関係。
fn lr_close_1e6(actual: f32, expected: f64) -> bool {
    let a = actual as f64;
    if expected == 0.0 {
        return a == 0.0;
    }
    ((a - expected) / expected).abs() < 1e-6
}

#[test]
fn fixture_was_generated_by_torch_2_14_0() {
    assert!(
        load_fixture().torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 の実行値でなければならない"
    );
}

#[test]
fn averaged_model_matches_pytorch_fixture() {
    let fx = load_fixture();
    assert!(fx.avg_cases.len() >= 5);
    let mut worst = 0.0f64;
    for case in &fx.avg_cases {
        let init = tensors(&case.init, &case.shapes);
        let mut avg = AveragedModel::from_named(named_refs(&init)).unwrap();
        assert_eq!(avg.n_averaged(), 0);
        for (si, step) in case.steps.iter().enumerate() {
            let params = tensors(&step.params, &case.shapes);
            avg.update_named(named_refs(&params)).unwrap();
            assert_eq!(
                avg.n_averaged(),
                step.n_averaged,
                "{}: step {si}",
                case.name
            );
            for (pi, expected_bits) in step.averaged.iter().enumerate() {
                let got = avg
                    .averaged(&pi.to_string())
                    .unwrap()
                    .as_slice()
                    .unwrap()
                    .to_vec();
                let expected = from_bits(expected_bits);
                assert_eq!(got.len(), expected.len());
                for (k, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        req2_close(g as f64, e as f64),
                        "{}: step {si} param {pi}[{k}]: got {g} expected {e}",
                        case.name
                    );
                    let diff = (g as f64 - e as f64).abs();
                    let rel = diff / (g as f64).abs().max((e as f64).abs()).max(1e-12);
                    worst = worst.max(rel.min(diff));
                }
            }
        }
    }
    eprintln!("AveragedModel vs PyTorch: 最大 min(相対, 絶対) 誤差 = {worst:e}");
}

#[test]
fn swalr_matches_pytorch_fixture() {
    let fx = load_fixture();
    assert!(fx.swalr_cases.len() >= 10);
    let mut worst = 0.0f64;
    for c in &fx.swalr_cases {
        let s = SwaLr::new(c.base_lr, c.swa_lr, c.anneal_epochs, anneal_of(&c.strategy)).unwrap();
        for (step, &expected) in c.lrs.iter().enumerate() {
            let got = s.lr_at(step);
            assert!(
                req2_close(got as f64, expected),
                "{}: step {step}: got {got} expected {expected}",
                c.name
            );
            assert!(
                lr_close_1e6(got, expected),
                "{}: step {step}: got {got} expected {expected} (rel 1e-6)",
                c.name
            );
            worst = worst.max(((got as f64 - expected) / expected).abs());
        }
    }
    eprintln!("SwaLr vs PyTorch: 最大相対誤差 = {worst:e}");
}

#[test]
fn swalr_chained_after_main_scheduler_matches_pytorch_loop() {
    let fx = load_fixture();
    let c = &fx.swalr_chain_case;
    let main = CosineAnnealingLr::new(c.base_lr, c.t_max, 0.0).unwrap();
    // SWA 段の base_lr は主スケジューラの切替時点の値。
    let swa_base = main.lr_at(c.switch_step);
    let swa = SwaLr::new(swa_base, c.swa_lr, c.anneal_epochs, SwaAnneal::Cos).unwrap();
    let seq = SequentialLr::new(vec![Box::new(main), Box::new(swa)], vec![c.switch_step]).unwrap();
    for (step, &expected) in c.lrs.iter().enumerate() {
        let got = seq.lr_at(step);
        assert!(
            req2_close(got as f64, expected),
            "chain step {step}: got {got} expected {expected}"
        );
        assert!(
            lr_close_1e6(got, expected),
            "chain step {step}: got {got} expected {expected} (rel 1e-6)"
        );
    }
}

#[test]
fn swalr_rejects_invalid_learning_rates() {
    for bad in [0.0f32, -0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            SwaLr::new(bad, 0.01, 5, SwaAnneal::Cos).is_err(),
            "base_lr={bad}"
        );
        assert!(
            SwaLr::new(0.1, bad, 5, SwaAnneal::Cos).is_err(),
            "swa_lr={bad}"
        );
    }
}

#[test]
fn swalr_boundaries_and_extremes() {
    let s = SwaLr::new(0.1, 0.01, 5, SwaAnneal::Cos).unwrap();
    assert_eq!(s.lr_at(0), 0.1);
    for step in [5usize, 6, 100, usize::MAX] {
        assert_eq!(s.lr_at(step).to_bits(), 0.01f32.to_bits());
    }
    let z = SwaLr::new(0.1, 0.01, 0, SwaAnneal::Linear).unwrap();
    for step in [0usize, 1, usize::MAX] {
        assert_eq!(z.lr_at(step).to_bits(), 0.01f32.to_bits());
    }
    // anneal_epochs = usize::MAX でも panic せず、先頭は base_lr。
    let huge = SwaLr::new(0.1, 0.01, usize::MAX, SwaAnneal::Cos).unwrap();
    assert_eq!(huge.lr_at(0), 0.1);
    assert!(huge.lr_at(1).is_finite());
    let _ = huge.lr_at(usize::MAX);
}

#[test]
fn swalr_midpoint_is_mean_for_both_strategies() {
    for anneal in [SwaAnneal::Linear, SwaAnneal::Cos] {
        let s = SwaLr::new(0.1, 0.02, 10, anneal).unwrap();
        let mid = s.lr_at(5);
        assert!((mid - 0.06).abs() < 1e-7, "{anneal:?}: {mid}");
    }
}

#[test]
fn swalr_is_deterministic() {
    let s = SwaLr::new(0.1, 0.01, 7, SwaAnneal::Cos).unwrap();
    for step in 0..12 {
        assert_eq!(s.lr_at(step).to_bits(), s.lr_at(step).to_bits());
    }
}

// ---- Module 統合（nn_ema.rs と同型） ----

fn build_model() -> Sequential {
    Sequential::new()
        .add(Linear::new(4, 8, true, 0x1111).unwrap())
        .add(Relu)
        .add(Linear::new(8, 2, true, 0x2222).unwrap())
}

fn forward_output(model: &Sequential) -> Vec<f32> {
    let x = Tensor::new(vec![0.1, -0.2, 0.3, -0.4], &[1, 4]).unwrap();
    let tape = Tape::new();
    let var = tape.var(&x);
    let out = model.forward(&tape, &var).unwrap();
    out.value().as_slice().unwrap().to_vec()
}

fn bump_linear_weights(model: &mut Sequential, offset: f32) {
    for layer in model.layers_mut() {
        if let Some(linear) = layer.as_linear() {
            let w = linear.weight();
            let bumped: Vec<f32> = w.as_slice().unwrap().iter().map(|v| v + offset).collect();
            let new_w = Tensor::new(bumped, w.shape()).unwrap();
            layer.set_parameter("weight", new_w).unwrap();
        }
    }
}

#[test]
fn module_update_matches_hand_computed_equal_weight_average() {
    let mut model = build_model();
    let mut avg = AveragedModel::from_module(&model).unwrap();
    let mut snapshots: Vec<HashMap<String, Vec<f32>>> = Vec::new();
    for step in 0..4 {
        bump_linear_weights(&mut model, 0.05 * (step as f32 + 1.0));
        avg.update_from_module(&model).unwrap();
        snapshots.push(
            model
                .state_dict()
                .into_iter()
                .map(|(k, v)| (k, v.as_slice().unwrap().to_vec()))
                .collect(),
        );
    }
    assert_eq!(avg.n_averaged(), 4);
    for (name, got) in avg.averaged_state_dict() {
        let got = got.as_slice().unwrap().to_vec();
        for (k, &g) in got.iter().enumerate() {
            let mean: f64 =
                snapshots.iter().map(|s| s[&name][k] as f64).sum::<f64>() / snapshots.len() as f64;
            assert!(req2_close(g as f64, mean), "{name}[{k}]: {g} vs {mean}");
        }
    }
}

#[test]
fn apply_then_restore_round_trips_to_bit_identical_weights() {
    let mut model = build_model();
    let mut avg = AveragedModel::from_module(&model).unwrap();
    bump_linear_weights(&mut model, 0.5);
    avg.update_from_module(&model).unwrap();
    bump_linear_weights(&mut model, 0.5);
    avg.update_from_module(&model).unwrap();

    let before = model.state_dict();
    let out_before = forward_output(&model);

    let backup = avg.apply(&mut model).unwrap();
    for (name, t) in avg.averaged_state_dict() {
        assert_eq!(
            model.state_dict()[&name].as_slice().unwrap(),
            t.as_slice().unwrap()
        );
    }
    assert_ne!(out_before, forward_output(&model));

    AveragedModel::restore(&mut model, backup).unwrap();
    let after = model.state_dict();
    for (name, t) in &before {
        assert_eq!(t.as_slice().unwrap(), after[name].as_slice().unwrap());
    }
    assert_eq!(out_before, forward_output(&model));
}

#[test]
fn apply_with_mismatched_shape_model_fails_and_leaves_model_unchanged() {
    let avg = AveragedModel::from_module(&build_model()).unwrap();
    let mut other = Sequential::new()
        .add(Linear::new(3, 8, true, 0x3333).unwrap())
        .add(Relu)
        .add(Linear::new(8, 2, true, 0x4444).unwrap());
    let before = other.state_dict();
    assert!(avg.apply(&mut other).is_err());
    let after = other.state_dict();
    for (name, t) in &before {
        assert_eq!(t.as_slice().unwrap(), after[name].as_slice().unwrap());
    }
}
