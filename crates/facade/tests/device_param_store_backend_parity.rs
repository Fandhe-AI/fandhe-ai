//! イシュー #936（デバイス常駐更新の parity テストとベンチ非後退確認）の
//! 受け入れ条件 1（parity テスト追加）を、`crates/facade/tests/
//! device_param_store_train.rs` の CPU tape 限定テストでは検証できない
//! 実バックエンド（Metal／CUDA）横断で満たす統合テスト。
//!
//! `fandhe_ai::tape_for(Device::Metal)`／`tape_for(Device::Cuda(0))`
//! （composition root。`crates/facade/src/lib.rs::tape_for`）経由で
//! `Sequential::init_device_param_store`／`forward_resident`／
//! `Tape::step_device_param_store` を回し、CPU ホスト参照実装
//! （`fandhe_ai_autodiff::optim::Sgd::step` + `Sequential::
//! apply_parameters`）の最終パラメータと突合する。
//!
//! **判定方式**: 設計文書 `docs/device-resident-update-design.md` §5.3・
//! §7「#936 への引き渡し事項」が定める「決定的シードで 100 step 程度学習
//! し、最終パラメータのみを統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）で比較する」方式
//! （中間ステップごとの判定は必須としない）。
//!
//! **実機ゲーティング**: `cfg(target_os = "macos")` は非 macOS の CI での
//! コンパイル対象除外にしかならず実機の有無までは保証しないため
//! （`.claude/rules/ci.md`「実機依存」節・`.claude/rules/coding-rust.md`
//! 「テスト・ベンチ」節）、Metal・CUDA いずれも理由付き `#[ignore]` で
//! 通常 CI（GitHub ホステッド ubuntu-latest）から分離する
//! （`crates/backend-metal/tests/sgd_device_parity.rs`・
//! `crates/backend-cuda/tests/sgd_device_real_device.rs` と同じ方針）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

const STEPS: usize = 100;
const LR: f32 = 0.02;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// 統一複合判定（`.claude/rules/coding-rust.md`）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// `device` へ結線した `tape_for` 経由でデバイス常駐経路を `steps` 回
/// 回し、最終的にホストへ同期したパラメータ列を返す。momentum・
/// weight_decay・nesterov を有効化した構成で回す（`crates/backend-cpu/
/// tests/sgd_device_parity.rs::full_combo_matches_host_reference_across_100_steps`
/// と同じ意図で、全オプション経路を累積誤差の観点で保険的に検証する）。
fn train_device_resident(device: Device, steps: usize, lr: f32) -> Vec<Tensor<f32>> {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(lr)
        .with_momentum(0.9)
        .with_weight_decay(0.01)
        .with_nesterov(true);

    for _ in 0..steps {
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward(&loss).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    let final_tape = fandhe_ai::tape_for(device).unwrap();
    final_tape.sync_device_param_store_to_host(&store).unwrap()
}

/// `Sgd::step` + `apply_parameters`（ホスト参照実装。CPU）で同一初期化・
/// 同一データ・同一ハイパーパラメータの学習ループを回し、最終パラメータ
/// 列を返す。
fn train_host_reference(steps: usize, lr: f32) -> Vec<Tensor<f32>> {
    let mut model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let sgd_config = SgdConfig::new(lr)
        .with_momentum(0.9)
        .with_weight_decay(0.01)
        .with_nesterov(true);
    let mut sgd = Sgd::new(sgd_config).unwrap();

    for _ in 0..steps {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    model.trainable_parameters().into_iter().cloned().collect()
}

fn assert_params_match(device_params: &[Tensor<f32>], host_params: &[Tensor<f32>], ctx: &str) {
    assert_eq!(device_params.len(), host_params.len());
    for (i, (d, h)) in device_params.iter().zip(host_params.iter()).enumerate() {
        assert_eq!(d.shape(), h.shape(), "{ctx}: param {i} shape mismatch");
        let d_slice = d.contiguous();
        let h_slice = h.contiguous();
        let d_data = d_slice.as_slice().unwrap();
        let h_data = h_slice.as_slice().unwrap();
        for (j, (dv, hv)) in d_data.iter().zip(h_data.iter()).enumerate() {
            assert_close(*dv, *hv, &format!("{ctx}: param {i} element {j}"));
        }
    }
}

/// Metal 実機（macOS）での 100 step 累積・最終値判定 parity。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn device_resident_matches_host_sgd_on_metal_across_100_steps() {
    let device_params = train_device_resident(Device::Metal, STEPS, LR);
    let host_params = train_host_reference(STEPS, LR);
    assert_params_match(&device_params, &host_params, "Metal vs host (100 steps)");
}

/// CUDA 実機（DGX Spark GB10 等）での 100 step 累積・最終値判定 parity。
///
/// ```sh
/// cargo test -p fandhe-ai --test device_param_store_backend_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn device_resident_matches_host_sgd_on_cuda_across_100_steps() {
    let device_params = train_device_resident(Device::Cuda(0), STEPS, LR);
    let host_params = train_host_reference(STEPS, LR);
    assert_params_match(&device_params, &host_params, "CUDA vs host (100 steps)");
}
