//! イシュー #3246: `Op::LinearResident` の VJP が、入力が勾配不要
//! （`requires_grad=false`。`var_no_grad`）のとき d_input の GEMM を省いても、
//! weight・bias の勾配と更新後パラメータが `var` 入力と bit 同一であることを
//! facade の公開面（`forward_resident`・`backward_device_param_store`・
//! `step_device_param_store`）越しに検証する統合テスト。
//!
//! 根拠: `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（#3245）§4・§8。
//! 同一バックエンド内の比較のため tolerance は使わず `to_bits()` 一致で判定する。
//!
//! **実機ゲーティング**: CPU は通常 CI で実行する。Metal・CUDA は実機依存のため
//! 理由付き `#[ignore]` で CI から分離する（`.claude/rules/ci.md`「実機依存」節）。
//! Metal のディスパッチ回数の減少は本テストでは検証しない（bit 一致のみ）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig, Tape};
use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const STEPS: usize = 5;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap()
}

/// 入力を `no_grad` で登録して STEPS 回学習し、更新後パラメータの bit 列と、
/// 最終 step の入力 `Gradients::get` が `GradientTrackingDisabled` だったかを返す。
fn run(make_tape: &dyn Fn() -> Tape, no_grad: bool) -> (Vec<u32>, bool) {
    let model = build_model();
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x_data = tensor(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]);
    let y_data = tensor(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]);
    let init_tape = make_tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    let config = SgdConfig::new(0.02);
    let mut disabled = false;
    for _ in 0..STEPS {
        let tape = make_tape();
        let x = if no_grad {
            tape.var_no_grad(&x_data)
        } else {
            tape.var(&x_data)
        };
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        disabled = matches!(grads.get(&x), Err(AutodiffError::GradientTrackingDisabled));
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }
    let params = make_tape().sync_device_param_store_to_host(&store).unwrap();
    let bits = params
        .iter()
        .flat_map(|t| {
            t.contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect();
    (bits, disabled)
}

fn assert_bit_identical(make_tape: &dyn Fn() -> Tape) {
    let (with_grad, dis_a) = run(make_tape, false);
    let (no_grad, dis_b) = run(make_tape, true);
    assert!(!dis_a, "var 入力は勾配が返る");
    assert!(dis_b, "var_no_grad 入力の get は GradientTrackingDisabled");
    assert_eq!(
        with_grad, no_grad,
        "d_input 省略で更新後パラメータが変わった"
    );
}

#[test]
fn nograd_input_dinput_skip_is_bit_identical_on_cpu() {
    assert_bit_identical(&fandhe_ai::tape);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn nograd_input_dinput_skip_is_bit_identical_on_metal() {
    assert_bit_identical(&|| fandhe_ai::tape_for(Device::Metal).unwrap());
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。CI では実行しない"]
fn nograd_input_dinput_skip_is_bit_identical_on_cuda() {
    assert_bit_identical(&|| fandhe_ai::tape_for(Device::Cuda(0)).unwrap());
}
