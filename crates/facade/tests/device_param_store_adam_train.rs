//! イシュー #1959（Adam／AdamW を `DeviceParamStore` の常駐 step へ
//! 結線する）の受け入れ条件検証。
//!
//! `crates/facade/tests/device_param_store_train.rs`（SGD 版）と同じ
//! `Sequential::forward_resident`／`Tape::backward_device_param_store`
//! ループを使うが、optimizer 部分の正しさを分離して検証するため、
//! **各 step の勾配を `Tape::param_grads_to_host`（#1479）で読み出し、
//! 直前の同期パラメータとともにホスト `Adam::step`／`AdamW::step` へ
//! 与えた結果**と、`step_device_param_store_adam`／`_adamw` 実行後の
//! `sync_device_param_store_to_host` を **bit 完全一致**で突合する
//! （`crates/backend-cpu/tests/adam_device_parity.rs` と同じ CPU bit
//! 一致契約。forward／backward 自体の再現ではなく optimizer 演算列の
//! 一致のみを検証する設計）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fandhe_ai::tape()` 既定 CPU 経由）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{AdamConfig, AdamWConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::{Adam, AdamW};
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

/// `Sequential::init_device_param_store` → `forward_resident` →
/// `tape.backward_device_param_store` → `Tape::step_device_param_store_adam`
/// を `steps` 回繰り返し、各 step ごとにデバイス側の更新後パラメータを
/// ホスト `Adam::step`（独立インスタンス）の結果と bit 完全一致で突合
/// する。
#[test]
fn adam_device_step_matches_host_reference_per_step() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    // 初期パラメータ列（`trainable_parameters()` と同じ位置対応契約。
    // 以後のホスト側 Adam ステップの起点として使う）。
    let mut host_params: Vec<Tensor<f32>> =
        init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut host_adam = Adam::new(config).unwrap();

    for _ in 0..6 {
        let tape = fandhe_ai::tape();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        assert!(scalar(&loss.to_tensor()).is_finite());

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        // resident 経由・host 経由を横断して全パラメータの勾配を読み出す
        // （イシュー #1479）。`host_params` と同じ位置対応。
        let grad_host = tape.param_grads_to_host(&store, &grads).unwrap();

        tape.step_device_param_store_adam(&mut store, &grads, &config)
            .unwrap();
        let device_synced = tape.sync_device_param_store_to_host(&store).unwrap();

        let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
            host_params.iter().zip(grad_host.iter()).collect();
        let host_out = host_adam.step(&pairs).unwrap();

        for (i, (dev, host)) in device_synced.iter().zip(host_out.iter()).enumerate() {
            assert_bits_eq(dev, host, &format!("param slot {i}"));
        }
        host_params = host_out;
    }
}

/// `AdamW`（decoupled weight decay）版。`Adam` 版と同じループ構造だが
/// `step_device_param_store_adamw`／ホスト `AdamW::step` を使う。
#[test]
fn adamw_device_step_matches_host_reference_per_step() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let mut host_params: Vec<Tensor<f32>> =
        init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = AdamWConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut host_adamw = AdamW::new(config).unwrap();

    for _ in 0..6 {
        let tape = fandhe_ai::tape();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        assert!(scalar(&loss.to_tensor()).is_finite());

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        let grad_host = tape.param_grads_to_host(&store, &grads).unwrap();

        tape.step_device_param_store_adamw(&mut store, &grads, &config)
            .unwrap();
        let device_synced = tape.sync_device_param_store_to_host(&store).unwrap();

        let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
            host_params.iter().zip(grad_host.iter()).collect();
        let host_out = host_adamw.step(&pairs).unwrap();

        for (i, (dev, host)) in device_synced.iter().zip(host_out.iter()).enumerate() {
            assert_bits_eq(dev, host, &format!("param slot {i}"));
        }
        host_params = host_out;
    }
}

/// 状態種別ガード: 初回 `step_adam` 確定後の `beta1`／`beta2`／`eps`／
/// `weight_decay` 途中変更を拒否する（`lr` のみ可変）。
#[test]
fn adam_rejects_hyperparameter_change_mid_training() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let base = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };

    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    tape.step_device_param_store_adam(&mut store, &grads, &base)
        .unwrap();

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = tape2.var(&y_data);
    let pred2 = model.forward_resident(&tape2, &x2, &mut store).unwrap();
    let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
    let grads2 = tape2.backward_device_param_store(&loss2, &store).unwrap();
    let mut changed = base;
    changed.beta1 = 0.8;
    let err = tape2
        .step_device_param_store_adam(&mut store, &grads2, &changed)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));

    // 上記の拒否（`beta1` 不一致）はエラーを返す前に `pending`
    // （forward で登録済みの葉ノード集合）を消費しないため、同一
    // `tape2`／`grads2` を使って `lr` のみを変えた再試行がそのまま
    // 成功する（新たな `forward_resident` を呼ぶ必要はない）。
    let mut lr_changed = base;
    lr_changed.lr = 5e-4;
    tape2
        .step_device_param_store_adam(&mut store, &grads2, &lr_changed)
        .unwrap();
}

/// 状態種別ガード（続き・実装計画 §5.2）: `step_adam` で確定した
/// ストアへ `step_adamw`（`kind` 不一致）を呼ぶと拒否される。
#[test]
fn adam_then_adamw_on_same_store_is_rejected() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let adam_config = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    tape.step_device_param_store_adam(&mut store, &grads, &adam_config)
        .unwrap();

    let adamw_config = AdamWConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = tape2.var(&y_data);
    let pred2 = model.forward_resident(&tape2, &x2, &mut store).unwrap();
    let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
    let grads2 = tape2.backward_device_param_store(&loss2, &store).unwrap();
    let err = tape2
        .step_device_param_store_adamw(&mut store, &grads2, &adamw_config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));
}

/// 状態種別ガード（続き・実装計画 §2.3「SGD → Adam 方向」）: `step()`
/// （SGD）で既に使われているストアへ `step_adam` を呼ぶと拒否される。
#[test]
fn sgd_then_adam_on_same_store_is_rejected() {
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

    let adam_config = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = tape2.var(&y_data);
    let pred2 = model.forward_resident(&tape2, &x2, &mut store).unwrap();
    let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
    let grads2 = tape2.backward_device_param_store(&loss2, &store).unwrap();
    let err = tape2
        .step_device_param_store_adam(&mut store, &grads2, &adam_config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));
}

/// `Adam::new`（ホスト参照実装）と同一基準のハイパーパラメータ検証:
/// `lr = NaN` は拒否され、デバイス側パラメータは未変更のまま残る。
#[test]
fn adam_rejects_nan_lr_and_leaves_params_unchanged() {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let before = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let bad_config = AdamConfig {
        lr: f32::NAN,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    let err = tape
        .step_device_param_store_adam(&mut store, &grads, &bad_config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::InvalidArgument(_)
    ));

    // ハイパーパラメータ検証は `pending`（forward で登録済みの葉ノード
    // 集合）を消費する更新フェーズより前に失敗するため、デバイス側
    // パラメータは未変更のまま残る。
    let after = tape.sync_device_param_store_to_host(&store).unwrap();
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_bits_eq(
            a,
            b,
            &format!("param slot {i} unchanged after rejected step"),
        );
    }
}
