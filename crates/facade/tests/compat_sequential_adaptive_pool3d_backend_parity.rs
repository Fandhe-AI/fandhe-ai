//! `compat::Sequential::add_adaptive_avg_pool3d`／`add_adaptive_max_pool3d`（イシュー #3824）の層経路の
//! CUDA／Metal parity テスト（`compat_sequential_pool3d_backend_parity.rs` と同型）。
//!
//! facade `Sequential` 経由の forward と入力勾配を対象デバイスと CPU で REQ-2 統一複合判定
//! （`assert_parity`。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較する。tolerance は変更しない。
//! 3D 適応プーリングの GPU カーネルは存在せずホストフォールバックするため、期待結果はホスト
//! フォールバック経路の確認になる。CPU 側の正しさは `compat_sequential_adaptive_pool3d.rs` が担う。
//! 実機実測は Phase 10 の #3860・#3861 が担い、本 issue では未実測
//! （`docs/perf/logs/adaptive-pool3d-ops-3823/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

/// 全要素が異なる ramp（max の索引タイブレークに依存させない）。NCDHW。
fn input() -> Tensor<f32> {
    let shape = [2, 3, 5, 6, 6];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32 * 0.011 - 1.3).collect(), &shape)
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
        .add_adaptive_max_pool3d([3, 4, 3])
        .and_then(|m| m.add_adaptive_avg_pool3d([2, 2, 2]))
        .expect("test fixture: 有効な引数")
}

/// (forward 出力, 入力勾配) を `device` 上で計算する。
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
    let gx = grads.get(&xv).unwrap().expect("入力勾配があるはず");
    (out, dense(gx))
}

fn run_parity(device: Device) {
    let (cpu_out, cpu_grad) = run(Device::Cpu);
    let (dev_out, dev_grad) = run(device);
    assert_parity(
        "compat::Sequential(AdaptivePool3d) forward device vs CPU",
        &dev_out,
        &cpu_out,
    );
    assert_parity(
        "compat::Sequential(AdaptivePool3d) input grad device vs CPU",
        &dev_grad,
        &cpu_grad,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実測は Phase 10 の #3860 の申し送り先"]
fn cuda_compat_sequential_adaptive_pool3d_matches_cpu() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実測は Phase 10 の #3861 の申し送り先"]
fn metal_compat_sequential_adaptive_pool3d_matches_cpu() {
    run_parity(Device::Metal);
}
