//! `compat::Sequential::add_conv_transpose2d`（イシュー #2523）の CUDA／Metal parity テスト
//! （`compat_sequential_spatial_layers_backend_parity.rs` と同型）。
//!
//! CPU 側の正しさは `compat_sequential_conv_transpose2d.rs` が Linux で担う。本ファイルは
//! facade の `Sequential` 経由の forward と backward（重み・バイアス勾配）を対象デバイスと CPU で
//! REQ-2 統一複合判定（`assert_parity`。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較する。
//! tolerance は変更しない。実機実測は未実施で GB10・M4 Max セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-conv-transpose2d-2523/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn input() -> Tensor<f32> {
    let shape = [2, 4, 3, 3];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32 * 0.05 - 0.9).collect(), &shape)
        .expect("test fixture: shape とデータ長は一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

fn model() -> Sequential {
    Sequential::new()
        .add_conv_transpose2d(4, 6, [3, 3], [2, 2], [1, 1], [1, 1], [1, 1], 2, 0x2523)
        .expect("test fixture: 有効な引数")
}

/// (forward 出力, 全パラメータ勾配の連結) を `device` 上で計算する。
fn run(device: Device) -> (Vec<f32>, Vec<f32>) {
    let model = model();
    let x = input();
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let y = bound.forward(&tape, &xv).unwrap();
    let out = dense(&y.to_tensor());
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let g: Vec<f32> = bound
        .trainable_grads(&grads)
        .unwrap()
        .iter()
        .flat_map(|t| dense(t))
        .collect();
    (out, g)
}

fn run_parity(device: Device) {
    let (cpu_out, cpu_grad) = run(Device::Cpu);
    let (dev_out, dev_grad) = run(device);
    assert_parity(
        "compat::Sequential(ConvTranspose2d) forward device vs CPU",
        &dev_out,
        &cpu_out,
    );
    assert_parity(
        "compat::Sequential(ConvTranspose2d) backward device vs CPU",
        &dev_grad,
        &cpu_grad,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2523 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_conv_transpose2d_forward_backward_matches_cpu() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2523 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_conv_transpose2d_forward_backward_matches_cpu() {
    run_parity(Device::Metal);
}
