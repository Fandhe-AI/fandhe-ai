//! `compat::Sequential::add_adaptive_max_pool2d`／`add_adaptive_max_pool1d`／`add_global_pool`
//! （イシュー #2527）の CUDA／Metal parity テスト（`compat_sequential_pixel_shuffle_backend_parity.rs`
//! と同型）。
//!
//! 3 層は純粋な選択演算（Avg の `GlobalPool` は既存 `adaptive_avg_pool*` の f64 アキュムレータ契約）
//! のため、対象デバイスと CPU を **bit 完全一致**で比較する（tolerance は変更しない）。GPU 側は
//! ホストフォールバック経由。CPU 側の正しさは `compat_sequential_adaptive_max_global_pool.rs` が
//! Linux で担う。実機実測は未実施で、GB10・M4 Max セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-adaptive-max-global-pool-2527/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, GlobalPoolMode, Tensor};

fn input() -> Tensor<f32> {
    let shape = [2, 3, 7, 6];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| ((i as f32) * 0.37).sin()).collect(), &shape)
        .expect("test fixture: shape とデータ長は一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn model() -> Sequential {
    Sequential::new()
        .add_adaptive_max_pool2d([3, 4])
        .unwrap()
        .add_global_pool(GlobalPoolMode::Max, true)
        .add_global_pool(GlobalPoolMode::Avg, false)
}

fn device_forward(model: &Sequential, x: &Tensor<f32>, device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

fn run_bit_exact(device: Device) {
    let m = model();
    let x = input();
    let cpu = m.predict(&x).unwrap();
    let dev = device_forward(&m, &x, device);
    assert_eq!(dense(&dev), dense(&cpu));
}

#[test]
fn cpu_reference_runs_without_device() {
    assert_eq!(model().predict(&input()).unwrap().shape(), &[2, 3]);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2527 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_adaptive_max_global_pool_bit_exact() {
    run_bit_exact(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2527 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_adaptive_max_global_pool_bit_exact() {
    run_bit_exact(Device::Metal);
}
