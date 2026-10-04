//! `compat::Sequential::add_group_norm`／`add_instance_norm`（イシュー #2525）の
//! CUDA／Metal parity テスト（`compat_sequential_spatial_layers_backend_parity.rs` と同型）。
//!
//! CPU 側の正しさは `compat_sequential_group_instance_norm.rs` が Linux で担う。本ファイルは
//! `tape_for(device)` 上の `Sequential::forward`（`reshape → layer_norm → reshape` の合成。
//! 新規 `Op`／カーネルなし）を CPU `predict` と REQ-2 統一複合判定（`assert_parity`）で
//! 比較する。tolerance は変更しない。実機実測は未実施で、GB10・M4 Max セッションへ申し送る
//! （`docs/perf/logs/compat-sequential-group-instance-norm-2525/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn input() -> Tensor<f32> {
    let shape = [2, 4, 3, 3];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32 * 0.13 - 2.0).collect(), &shape)
        .expect("test fixture: shape とデータ長は一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

fn run_parity(device: Device) {
    let model = Sequential::new()
        .add_group_norm(2, 1e-5)
        .unwrap()
        .add_instance_norm(1e-5)
        .unwrap();
    let x = input();
    let cpu = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let dev = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_parity(
        "compat::Sequential(GroupNorm→InstanceNorm) device vs CPU",
        &dense(&dev),
        &dense(&cpu),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2525 の申し送り先（GB10 セッション）へ引き継ぐ"]
fn cuda_group_instance_norm_matches_cpu() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2525 の申し送り先（Mac セッション）へ引き継ぐ"]
fn metal_group_instance_norm_matches_cpu() {
    run_parity(Device::Metal);
}
