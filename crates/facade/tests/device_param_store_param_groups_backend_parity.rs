//! イシュー #3837（`DeviceParamStore` の常駐更新の param groups 対応。設計
//! `docs/autodiff-param-groups-decision.md` §19.5 の 3 段目）の実機 parity テスト。
//!
//! `device_param_store_backend_parity.rs`（#936）と同構成: 決定的シードで 100 step 学習し、
//! 最終パラメータのみを統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で CPU の
//! 同一 groups 付き常駐学習と突合する。SGD・Adam・AdamW を CUDA・Metal で回す。
//!
//! 実機（CUDA・Metal）依存のため理由付き `#[ignore]` で通常 CI から分離する。実測は
//! #3862（CUDA。GB10）・#3863（Metal。Mac mini）が持ち、本ファイルは未実測のまま。
//! store は facade の公開入口 `Sequential::init_device_param_store_with_param_groups`（#3838）で作る
//! （公開経路そのものを実機で測る）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::SgdConfig;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{AdamConfig, AdamWConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::ParamGroup;
use fandhe_ai_autodiff::optim::DeviceParamStore;
use fandhe_ai_tensor_core::Tensor;

const STEPS: usize = 100;

#[derive(Clone, Copy)]
enum Opt {
    Sgd,
    Adam,
    AdamW,
}

fn model() -> Sequential {
    Sequential::new()
        .add_linear(8, 16, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(16, 4, 0x2222_2222)
        .unwrap()
}

fn groups() -> Vec<ParamGroup> {
    vec![
        ParamGroup::new(vec![0], 5e-3, 0.05),
        ParamGroup::new(vec![1, 2], 2e-2, 0.0),
    ]
}

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    assert!(
        fandhe_ai_tensor_core::tolerance::composite_close_f32(actual, expected),
        "{ctx}: actual={actual} expected={expected}"
    );
}

/// `ops` へ結線した autodiff Tape で store を作り、`tape_for(device)` の facade Tape で
/// `STEPS` 回学習して最終パラメータを返す。
fn train(device: fandhe_ai::Device, opt: Opt) -> Vec<Tensor<f32>> {
    let m = model();
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x = Tensor::new(rng.fill_vec(4 * 8), &[4, 8]).unwrap();
    let y = Tensor::new(rng.fill_vec(4 * 4), &[4, 4]).unwrap();
    let store_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store: DeviceParamStore = m
        .init_device_param_store_with_param_groups(&store_tape, &groups())
        .unwrap();
    for _ in 0..STEPS {
        let tape = fandhe_ai::tape_for(device).unwrap();
        let (xv, yv) = (tape.var(&x), tape.var(&y));
        let pred = m.forward_resident(&tape, &xv, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &yv).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        match opt {
            Opt::Sgd => tape
                .step_device_param_store(
                    &mut store,
                    &grads,
                    &SgdConfig::new(0.02).with_momentum(0.9).with_nesterov(true),
                )
                .unwrap(),
            Opt::Adam => tape
                .step_device_param_store_adam(&mut store, &grads, &AdamConfig::default())
                .unwrap(),
            Opt::AdamW => tape
                .step_device_param_store_adamw(&mut store, &grads, &AdamWConfig::default())
                .unwrap(),
        }
    }
    fandhe_ai::tape_for(device)
        .unwrap()
        .sync_device_param_store_to_host(&store)
        .unwrap()
}

fn cpu_reference(opt: Opt) -> Vec<Tensor<f32>> {
    train(fandhe_ai::Device::Cpu, opt)
}

fn compare(a: &[Tensor<f32>], b: &[Tensor<f32>], ctx: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let (xc, yc) = (x.contiguous(), y.contiguous());
        for (j, (p, q)) in xc
            .as_slice()
            .unwrap()
            .iter()
            .zip(yc.as_slice().unwrap())
            .enumerate()
        {
            assert_close(*p, *q, &format!("{ctx} slot {i} elem {j}"));
        }
    }
}

#[test]
#[ignore = "CUDA 実機（GB10）依存。#3862 で実測"]
fn cuda_param_groups_match_cpu_for_sgd_adam_adamw() {
    for (opt, name) in [
        (Opt::Sgd, "sgd"),
        (Opt::Adam, "adam"),
        (Opt::AdamW, "adamw"),
    ] {
        let dev = train(fandhe_ai::Device::Cuda(0), opt);
        compare(&dev, &cpu_reference(opt), &format!("cuda/{name}"));
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。#3863 で実測"]
fn metal_param_groups_match_cpu_for_sgd_adam_adamw() {
    for (opt, name) in [
        (Opt::Sgd, "sgd"),
        (Opt::Adam, "adam"),
        (Opt::AdamW, "adamw"),
    ] {
        let dev = train(fandhe_ai::Device::Metal, opt);
        compare(&dev, &cpu_reference(opt), &format!("metal/{name}"));
    }
}
