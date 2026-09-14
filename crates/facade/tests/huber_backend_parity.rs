//! `Var::huber_loss`／`smooth_l1_loss`（イシュー #1739）の facade 到達
//! 経路（既存 `Var` 再エクスポート経由。`crates/facade/src/lib.rs` への
//! 新規 `pub use`／`pub fn` は追加していない）の受け入れ条件対応テスト
//! （`activation_gelu_softplus_backend_parity.rs` と同型の構成）。
//!
//! - 属性なし: `fandhe_ai::tape()`（CPU 融合カーネル。
//!   `CpuBackendOps::huber_loss`）と `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps` → `eval::huber_loss` フォールバック）の forward／
//!   backward を `assert_parity`（REQ-2 統一複合判定）で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`
//!   （`cfg(target_os = "macos")` 限定）と CPU の突合。本エージェント
//!   実行環境に実機への到達手段がないため未実測のまま Mac／GB10
//!   セッションへ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::{Reduction, Var};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`scalar_unary_
/// transcendental_backend_parity.rs::VarSource` と同型）。
trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

// --- forward/backward parity（属性なし: CPU 融合 vs NaiveOps フォール
// バック。REQ-2 統一複合判定） ---

#[test]
fn cpu_huber_loss_forward_and_backward_match_naive_fallback() {
    let shape = [3usize, 4];
    let pred_val = leaf(1739, &shape);
    let target_val = leaf(1740, &shape);

    let cpu_tape = fandhe_ai::tape();
    let pred_cpu = cpu_tape.make_var(&pred_val);
    let target_cpu = cpu_tape.make_var(&target_val);
    let loss_cpu = pred_cpu
        .huber_loss(&target_cpu, 1.0, Reduction::Mean)
        .unwrap();
    let out_cpu = loss_cpu.to_tensor();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dpred_cpu = grads_cpu.get(&pred_cpu).unwrap().expect("到達する").clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let pred_naive = naive_tape.make_var(&pred_val);
    let target_naive = naive_tape.make_var(&target_val);
    let loss_naive = pred_naive
        .huber_loss(&target_naive, 1.0, Reduction::Mean)
        .unwrap();
    let out_naive = loss_naive.to_tensor();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dpred_naive = grads_naive.get(&pred_naive).unwrap().expect("到達する");

    assert_parity(
        "huber_loss forward: CPU fused vs NaiveOps fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
    assert_parity(
        "huber_loss backward (dPred): CPU fused vs NaiveOps fallback",
        &contiguous_slice(&dpred_cpu),
        &contiguous_slice(dpred_naive),
    );
}

#[test]
fn cpu_smooth_l1_loss_forward_and_backward_match_naive_fallback() {
    let shape = [3usize, 4];
    let pred_val = leaf(1741, &shape);
    let target_val = leaf(1742, &shape);

    let cpu_tape = fandhe_ai::tape();
    let pred_cpu = cpu_tape.make_var(&pred_val);
    let target_cpu = cpu_tape.make_var(&target_val);
    let loss_cpu = pred_cpu
        .smooth_l1_loss(&target_cpu, 2.0, Reduction::Sum)
        .unwrap();
    let out_cpu = loss_cpu.to_tensor();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dpred_cpu = grads_cpu.get(&pred_cpu).unwrap().expect("到達する").clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let pred_naive = naive_tape.make_var(&pred_val);
    let target_naive = naive_tape.make_var(&target_val);
    let loss_naive = pred_naive
        .smooth_l1_loss(&target_naive, 2.0, Reduction::Sum)
        .unwrap();
    let out_naive = loss_naive.to_tensor();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dpred_naive = grads_naive.get(&pred_naive).unwrap().expect("到達する");

    assert_parity(
        "smooth_l1_loss forward: CPU fused vs NaiveOps fallback",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
    assert_parity(
        "smooth_l1_loss backward (dPred): CPU fused vs NaiveOps fallback",
        &contiguous_slice(&dpred_cpu),
        &contiguous_slice(dpred_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn huber_forward_on(device: Device, pred: &Tensor<f32>, target: &Tensor<f32>) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let pred_v = tape.make_var(pred);
    let target_v = tape.make_var(target);
    pred_v
        .huber_loss(&target_v, 1.0, Reduction::Mean)
        .unwrap()
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`scalar_unary_transcendental_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_huber_loss_forward_matches_cpu() {
    let pred = leaf(1743, &[3, 4]);
    let target = leaf(1744, &[3, 4]);
    let metal_out = huber_forward_on(Device::Metal, &pred, &target);
    let cpu_out = huber_forward_on(Device::Cpu, &pred, &target);
    assert_parity(
        "huber_loss forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_huber_loss_forward_matches_cpu() {
    let pred = leaf(1745, &[3, 4]);
    let target = leaf(1746, &[3, 4]);
    let cuda_out = huber_forward_on(Device::Cuda(0), &pred, &target);
    let cpu_out = huber_forward_on(Device::Cpu, &pred, &target);
    assert_parity(
        "huber_loss forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}
