//! `compat::Sequential::add_channel_shuffle`（イシュー #3822）の CUDA／Metal parity テスト
//! （`compat_sequential_pixel_shuffle_backend_parity.rs` と同型）。
//!
//! 層は contiguous／reshape／permute の合成（算術を含まないコピー）のため、対象デバイスと
//! CPU を **bit 完全一致**で比較する（tolerance は変更しない）。CPU 側の正しさは
//! `compat_sequential_channel_shuffle.rs` が Linux で担う。実機実測は未実施で、GB10・M4 Max
//! セッションへ申し送る（`docs/perf/logs/channel-shuffle-3822/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};

fn input() -> Tensor<f32> {
    let shape = [2, 6, 3, 3];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32 * 0.1 - 0.9).collect(), &shape)
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

fn device_forward(model: &Sequential, x: &Tensor<f32>, device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

fn run_bit_exact(device: Device) {
    let model = Sequential::new()
        .add_channel_shuffle(3)
        .unwrap()
        .add_channel_shuffle(2)
        .unwrap();
    let x = input();
    let cpu = model.predict(&x).unwrap();
    let dev = device_forward(&model, &x, device);
    assert_eq!(dense(&dev), dense(&cpu));
}

#[test]
fn cpu_reference_runs_without_device() {
    let model = Sequential::new().add_channel_shuffle(3).unwrap();
    assert_eq!(model.predict(&input()).unwrap().shape(), &[2, 6, 3, 3]);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #3860 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_channel_shuffle_layers_bit_exact() {
    run_bit_exact(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #3861 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_channel_shuffle_layers_bit_exact() {
    run_bit_exact(Device::Metal);
}
