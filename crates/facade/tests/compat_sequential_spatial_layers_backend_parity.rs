//! `compat::Sequential::add_upsample`／`add_zero_pad2d`／`add_identity`（イシュー #2522）の
//! CUDA／Metal parity テスト（`compat_sequential_activation_shape_backend_parity.rs` と同型）。
//!
//! CPU 側の正しさは `compat_sequential_spatial_layers.rs` が Linux で担う。本ファイルは
//! `ZeroPad2d`／`Identity`／`Upsample(Nearest)`（算術を含まない整数添字・view 系）を
//! 対象デバイスと CPU で **bit 完全一致**、`Upsample(Bilinear)` を REQ-2 統一複合判定
//! （`assert_parity`）で比較する。tolerance は変更しない。実機実測は未実施で、
//! GB10・M4 Max セッションへ申し送る（`docs/perf/logs/compat-sequential-spatial-2522/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, InterpolateMode, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn input() -> Tensor<f32> {
    let shape = [2, 2, 3, 3];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32 * 0.1 - 0.9).collect(), &shape)
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

fn run_bit_exact(device: Device) {
    let model = Sequential::new()
        .add_zero_pad2d([1, 2, 0, 1])
        .add_upsample(vec![8, 8], InterpolateMode::Nearest)
        .unwrap()
        .add_identity();
    let x = input();
    let cpu = model.predict(&x).unwrap();
    let dev = device_forward(&model, &x, device);
    assert_eq!(dense(&dev), dense(&cpu));
}

fn run_bilinear_parity(device: Device) {
    let model = Sequential::new()
        .add_zero_pad2d([1, 1, 1, 1])
        .add_upsample(
            vec![7, 9],
            InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .unwrap()
        .add_identity();
    let x = input();
    let cpu = model.predict(&x).unwrap();
    let dev = device_forward(&model, &x, device);
    assert_parity(
        "compat::Sequential(ZeroPad2d→Upsample(Bilinear)→Identity) device vs CPU",
        &dense(&dev),
        &dense(&cpu),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2522 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_spatial_layers_nearest_bit_exact() {
    run_bit_exact(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_spatial_layers_bilinear_matches_cpu() {
    run_bilinear_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2522 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_spatial_layers_nearest_bit_exact() {
    run_bit_exact(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須"]
fn metal_spatial_layers_bilinear_matches_cpu() {
    run_bilinear_parity(Device::Metal);
}
