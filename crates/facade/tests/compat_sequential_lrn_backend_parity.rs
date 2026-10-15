//! `compat::Sequential::add_local_response_norm`（イシュー #3045・親 #3018）の層経路の
//! CUDA／Metal parity テスト（`compat_sequential_fold_unfold_backend_parity.rs` と同型）。
//!
//! 演算単体は `lrn_weight_reparam_backend_parity.rs`、本ファイルは facade `Sequential` 経由の forward と
//! 入力勾配（無パラメータ層のため入力 `Var` の勾配）を対象デバイスと CPU で REQ-2 統一複合判定
//! （`assert_parity`。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較する。tolerance は変更しない。
//! CUDA／Metal は LRN の 3 フックを override せず `Unsupported` → ホストフォールバック経路を通るため、
//! 新規カーネルの parity ではなく「層経由でもフォールバックが CPU と一致する」ことの確認になる。
//! CPU 側の正しさは `compat_sequential_lrn.rs` が担う。実機実測は #3171・#3172 が担い、
//! 本 issue では未実測（`docs/perf/logs/compat-sequential-local-response-norm-3045/README.md`）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

/// ramp 入力 NCHW `[2, 7, 3, 2]`（チャネル 7 に対し `size = 5` で窓クリップが効く）。
fn input() -> Tensor<f32> {
    let shape = [2, 7, 3, 2];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| i as f32 * 0.037 - 1.2).collect(), &shape)
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
        .add_local_response_norm(5, 1e-4, 0.75, 2.0)
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
        "compat::Sequential(LRN) forward device vs CPU",
        &dev_out,
        &cpu_out,
    );
    assert_parity(
        "compat::Sequential(LRN) input grad device vs CPU",
        &dev_grad,
        &cpu_grad,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実測は #3171 の申し送り先（GB10 セッション）"]
fn cuda_local_response_norm_forward_backward_matches_cpu() {
    run_parity(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実測は #3172 の申し送り先（Mac セッション）"]
fn metal_local_response_norm_forward_backward_matches_cpu() {
    run_parity(Device::Metal);
}
