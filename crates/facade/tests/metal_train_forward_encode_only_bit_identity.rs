//! train reuse（`forward_resident` → MSE → backward → SGD）を K step 実行し、
//! Metal の train forward encode-only 合流（イシュー #2113・opt-in・既定
//! OFF）の ON／OFF で step ごとの loss と最終パラメータが `to_bits` 完全
//! 一致することを固定する（A/B の checksum 完全一致の前提ゲート）。
//!
//! フラグはプロセスワイドのためテストは 1 本にまとめ static Mutex で直列化
//! する。モデル形状・シードは `mnist_scale_train_reuse_bench.rs` と同値。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test metal_train_forward_encode_only_bit_identity -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const SEED_X: u64 = 0xDA7A_0001;
const SEED_Y: u64 = 0xDA7A_0002;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const STEPS: usize = 8;
const LR: f32 = 0.01;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

/// フラグを一時設定し drop で元へ戻す RAII ガード。
struct FlagGuard(bool);

impl FlagGuard {
    fn set(enabled: bool) -> Self {
        let original = fandhe_ai_backend_metal::__train_forward_encode_only_enabled();
        fandhe_ai_backend_metal::__set_train_forward_encode_only_enabled(enabled);
        Self(original)
    }
}

impl Drop for FlagGuard {
    fn drop(&mut self) {
        fandhe_ai_backend_metal::__set_train_forward_encode_only_enabled(self.0);
    }
}

/// `encode_only` を設定して STEPS step 学習し、(step ごとの loss bits,
/// 最終パラメータ bits) を返す。
fn run(encode_only: bool) -> (Vec<u32>, Vec<Vec<u32>>) {
    let _flag = FlagGuard::set(encode_only);
    let device = Device::Metal;
    let model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap();
    let x_data = tensor(
        Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN),
        &[BATCH, D_IN],
    );
    let y_data = tensor(
        Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT),
        &[BATCH, D_OUT],
    );

    let init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let _ = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);
    let mut losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let v = loss
            .to_tensor()
            .get(&[])
            .expect("loss は shape [] スカラー");
        assert!(v.is_finite(), "loss not finite: {v}");
        losses.push(v.to_bits());
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    let final_tape = fandhe_ai::tape_for(device).unwrap();
    let params = final_tape
        .sync_device_param_store_to_host(&store)
        .unwrap()
        .iter()
        .map(|t| {
            t.contiguous()
                .as_slice()
                .expect("contiguous() 直後は as_slice() が Some")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect();
    (losses, params)
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn train_reuse_forward_encode_only_is_bit_identical() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let (loss_off, params_off) = run(false);
    let (loss_on, params_on) = run(true);
    assert_eq!(
        loss_off, loss_on,
        "step ごとの loss が bit 完全一致するはず"
    );
    assert_eq!(
        params_off, params_on,
        "最終パラメータが bit 完全一致するはず"
    );
}
