//! イシュー #2175（RmsProp・Adagrad・LAMB を `DeviceParamStore` の常駐
//! step へ結線する）の受け入れ条件検証（RmsProp 版）。
//!
//! `crates/facade/tests/device_param_store_adam_train.rs` と同じ設計
//! （各 step の勾配を `Tape::param_grads_to_host` で読み出し、直前の
//! 同期パラメータとともにホスト `RmsProp::step`〈独立インスタンス〉へ
//! 与えた結果と `step_device_param_store_rmsprop` 実行後の
//! `sync_device_param_store_to_host` を **bit 完全一致**で突合する）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fandhe_ai::tape()` 既定 CPU 経由）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::RmsPropConfig;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::RmsProp;
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn assert_bits_eq(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    let a_data = a.as_slice().unwrap_or(&[]);
    let e_data = e.as_slice().unwrap_or(&[]);
    for (i, (av, ev)) in a_data.iter().zip(e_data.iter()).enumerate() {
        assert_eq!(
            av.to_bits(),
            ev.to_bits(),
            "{ctx}: element {i}: actual={av} expected={ev}"
        );
    }
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

fn run_train_parity(config: RmsPropConfig) {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let mut host_params: Vec<Tensor<f32>> =
        init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let mut host_rmsprop = RmsProp::new(config).unwrap();

    for _ in 0..6 {
        let tape = fandhe_ai::tape();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        assert!(scalar(&loss.to_tensor()).is_finite());

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        let grad_host = tape.param_grads_to_host(&store, &grads).unwrap();

        tape.step_device_param_store_rmsprop(&mut store, &grads, &config)
            .unwrap();
        let device_synced = tape.sync_device_param_store_to_host(&store).unwrap();

        let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
            host_params.iter().zip(grad_host.iter()).collect();
        let host_out = host_rmsprop.step(&pairs).unwrap();

        for (i, (dev, host)) in device_synced.iter().zip(host_out.iter()).enumerate() {
            assert_bits_eq(dev, host, &format!("param slot {i}"));
        }
        host_params = host_out;
    }
}

#[test]
fn rmsprop_device_step_matches_host_reference_per_step() {
    run_train_parity(RmsPropConfig {
        lr: 1e-3,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.01,
        momentum: 0.0,
        centered: false,
    });
}

#[test]
fn rmsprop_centered_and_momentum_device_step_matches_host_reference_per_step() {
    run_train_parity(RmsPropConfig {
        lr: 1e-3,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        momentum: 0.9,
        centered: true,
    });
}

/// 状態種別ガード: 初回 `step_rmsprop` 確定後の `alpha` 途中変更を
/// 拒否する（`lr` のみ可変）。
#[test]
fn rmsprop_rejects_hyperparameter_change_mid_training() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let base = RmsPropConfig {
        lr: 1e-2,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        momentum: 0.0,
        centered: false,
    };

    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    tape.step_device_param_store_rmsprop(&mut store, &grads, &base)
        .unwrap();

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = tape2.var(&y_data);
    let pred2 = model.forward_resident(&tape2, &x2, &mut store).unwrap();
    let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
    let grads2 = tape2.backward_device_param_store(&loss2, &store).unwrap();
    let mut changed = base;
    changed.alpha = 0.5;
    let err = tape2
        .step_device_param_store_rmsprop(&mut store, &grads2, &changed)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));

    // `lr` のみを変えた再試行はそのまま成功する（`pending` は拒否時に
    // 消費されないため）。
    let mut lr_changed = base;
    lr_changed.lr = 5e-3;
    tape2
        .step_device_param_store_rmsprop(&mut store, &grads2, &lr_changed)
        .unwrap();
}

/// 状態種別ガード（続き）: `step()`（SGD）で既に使われているストアへ
/// `step_rmsprop` を呼ぶと拒否される。
#[test]
fn sgd_then_rmsprop_on_same_store_is_rejected() {
    use fandhe_ai::optim::SgdConfig;

    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let sgd_config = SgdConfig::new(0.1);
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    tape.step_device_param_store(&mut store, &grads, &sgd_config)
        .unwrap();

    let rmsprop_config = RmsPropConfig::default();
    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = tape2.var(&y_data);
    let pred2 = model.forward_resident(&tape2, &x2, &mut store).unwrap();
    let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
    let grads2 = tape2.backward_device_param_store(&loss2, &store).unwrap();
    let err = tape2
        .step_device_param_store_rmsprop(&mut store, &grads2, &rmsprop_config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));
}

/// `RmsProp::new`（ホスト参照実装）と同一基準のハイパーパラメータ
/// 検証: `lr = NaN` は拒否され、デバイス側パラメータは未変更のまま
/// 残る。
#[test]
fn rmsprop_rejects_nan_lr_and_leaves_params_unchanged() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let before = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let bad_config = RmsPropConfig {
        lr: f32::NAN,
        ..RmsPropConfig::default()
    };
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    let err = tape
        .step_device_param_store_rmsprop(&mut store, &grads, &bad_config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));

    let after = tape.sync_device_param_store_to_host(&store).unwrap();
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_bits_eq(
            a,
            b,
            &format!("param slot {i} unchanged after rejected step"),
        );
    }
}
