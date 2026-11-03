//! `compat::Sequential::add_lp_pool1d`／`2d`／`3d`（イシュー #3825）の CUDA／Metal parity テスト
//! （`compat_sequential_channel_shuffle_backend_parity.rs` と同型）。
//!
//! 層は `pow_scalar`／`avg_pool`／`sign`／`mul` の合成のため、対象デバイスと CPU を REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。tolerance は変更しない）で比較する。CPU 側の正しさは
//! `compat_sequential_lp_pool.rs` が Linux で担う。実機実測は未実施で、GB10・M4 Max セッション
//! （#3860・#3861）へ申し送る（`docs/perf/logs/lp-pool-ops-3825/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};

fn input() -> Tensor<f32> {
    let shape = [2, 3, 4, 4, 4];
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|i| 0.25 + (i % 17) as f32 * 0.1).collect(),
        &shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

fn device_forward(model: &Sequential, x: &Tensor<f32>, device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

fn run_parity(device: Device) {
    let model = Sequential::new()
        .add_lp_pool3d(2.0, [2, 2, 2], None)
        .unwrap();
    let x = input();
    let cpu = model.predict(&x).unwrap();
    let dev = device_forward(&model, &x, device);
    assert_eq!(dev.shape(), cpu.shape());
    fandhe_ai_backend_cpu::parity::assert_parity("lp_pool3d layer", &dense(&dev), &dense(&cpu));
}

#[test]
fn cpu_reference_runs_without_device() {
    let model = Sequential::new()
        .add_lp_pool3d(2.0, [2, 2, 2], None)
        .unwrap();
    assert_eq!(model.predict(&input()).unwrap().shape(), &[2, 3, 2, 2, 2]);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #3860 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_lp_pool_layers_match_cpu() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #3861 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_lp_pool_layers_match_cpu() {
    run_parity(Device::Metal);
}
