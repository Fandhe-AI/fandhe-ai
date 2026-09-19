//! `nn::GroupNorm`／`InstanceNorm`（イシュー #2066・親 #2058）の facade
//! 横断 parity テスト（`norm_backend_parity.rs`・`mha_backend_parity.rs`
//! と同型）。
//!
//! GroupNorm／InstanceNorm は `docs/compat-api-scope.md` の Tier 1／
//! Tier 2 列挙に個別の行を持たず、facade 公開面拡張（`compat::
//! Sequential::add_group_norm`／`add_instance_norm`）は同 §5 の承認が
//! 未取得のため、本テストは facade を経由せず
//! `fandhe_ai_autodiff::nn::{GroupNorm, InstanceNorm}` を直接呼ぶ
//! （`mha_backend_parity.rs` が `MultiheadAttentionVars::new` を直接
//! 呼ぶのと同型。`fandhe_ai::Tape`〈facade newtype〉と
//! `fandhe_ai_autodiff::Tape`〈生の型〉を横断する `VarSource` trait は
//! 共有）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`BackendOps::
//!   layer_norm` を実機カーネルでオーバーライド済み）上の
//!   `GroupNorm::forward`／`InstanceNorm::forward` forward・backward
//!   を、無引数 `fandhe_ai_autodiff::Tape::new()`（`NaiveOps` →
//!   `eval::layer_norm_rows` へフォールバックする経路）と REQ-2 統一
//!   複合判定で突き合わせる。
//! - `#[ignore]`: `fandhe_ai::tape_for(Device::Metal)`／
//!   `tape_for(Device::Cuda(0))` 上の同経路を CPU tape と突き合わせる
//!   （実機必須）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{GroupNorm, InstanceNorm};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`norm_backend_parity.rs`
/// の `VarSource` と同じ理由・同じ構成）。
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

/// `[N=2, C=4, S=3]` の決定的な固定値（`norm_backend_parity.rs::leaf`
/// と同じ方針。facade へ dev-dep を増やさないため乱数ユーティリティは
/// 使わない）。
fn leaf() -> Tensor<f32> {
    let data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.31 + 0.05).collect();
    Tensor::new(data, &[2, 4, 3]).expect("leaf: shape 一致")
}

fn target() -> Tensor<f32> {
    let data: Vec<f32> = (0..24)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1 - 0.5)
        .collect();
    Tensor::new(data, &[2, 4, 3]).expect("target: shape 一致")
}

// --- GroupNorm ---

/// `GroupNorm::forward` forward の CPU（融合カーネル経由）と NaiveOps
/// （ホスト参照実装フォールバック）の parity。
#[test]
fn cpu_group_norm_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let cpu_out = GroupNorm::new(2, 1e-5)
        .unwrap()
        .forward(&cpu_tape.make_var(&leaf()))
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = GroupNorm::new(2, 1e-5)
        .unwrap()
        .forward(&naive_tape.make_var(&leaf()))
        .unwrap()
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::layer_norm）vs NaiveOps フォールバック（GroupNorm）",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `GroupNorm::forward → mse_loss` backward（`dx`）の CPU（`Op::
/// Reshape`／`Op::LayerNorm` の VJP 合成）と NaiveOps の parity。
#[test]
fn cpu_group_norm_backward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf());
    let t_cpu = cpu_tape.make_var(&target());
    let y_cpu = GroupNorm::new(2, 1e-5).unwrap().forward(&x_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf());
    let t_naive = naive_tape.make_var(&target());
    let y_naive = GroupNorm::new(2, 1e-5).unwrap().forward(&x_naive).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_parity(
        "GroupNorm backward（dx）: CpuBackendOps vs NaiveOps",
        dx_cpu.as_slice().expect("contiguous"),
        dx_naive.as_slice().expect("contiguous"),
    );
}

fn group_norm_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    GroupNorm::new(2, 1e-5)
        .unwrap()
        .forward(&tape.make_var(&leaf()))
        .unwrap()
        .to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_group_norm_forward_matches_cpu() {
    let metal_out = group_norm_forward_on(Device::Metal);
    let cpu_out = group_norm_forward_on(Device::Cpu);

    assert_parity(
        "GroupNorm forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_group_norm_forward_matches_cpu() {
    let cuda_out = group_norm_forward_on(Device::Cuda(0));
    let cpu_out = group_norm_forward_on(Device::Cpu);

    assert_parity(
        "GroupNorm forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

// --- InstanceNorm ---

/// `InstanceNorm::forward` forward の CPU と NaiveOps の parity
/// （`groups = channels` の `GroupNorm` と同型の経路を通る）。
#[test]
fn cpu_instance_norm_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let cpu_out = InstanceNorm::new(1e-5)
        .unwrap()
        .forward(&cpu_tape.make_var(&leaf()))
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = InstanceNorm::new(1e-5)
        .unwrap()
        .forward(&naive_tape.make_var(&leaf()))
        .unwrap()
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::layer_norm）vs NaiveOps フォールバック（InstanceNorm）",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `InstanceNorm::forward → mse_loss` backward（`dx`）の CPU と
/// NaiveOps の parity。
#[test]
fn cpu_instance_norm_backward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf());
    let t_cpu = cpu_tape.make_var(&target());
    let y_cpu = InstanceNorm::new(1e-5).unwrap().forward(&x_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf());
    let t_naive = naive_tape.make_var(&target());
    let y_naive = InstanceNorm::new(1e-5).unwrap().forward(&x_naive).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_parity(
        "InstanceNorm backward（dx）: CpuBackendOps vs NaiveOps",
        dx_cpu.as_slice().expect("contiguous"),
        dx_naive.as_slice().expect("contiguous"),
    );
}

fn instance_norm_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    InstanceNorm::new(1e-5)
        .unwrap()
        .forward(&tape.make_var(&leaf()))
        .unwrap()
        .to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_instance_norm_forward_matches_cpu() {
    let metal_out = instance_norm_forward_on(Device::Metal);
    let cpu_out = instance_norm_forward_on(Device::Cpu);

    assert_parity(
        "InstanceNorm forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_instance_norm_forward_matches_cpu() {
    let cuda_out = instance_norm_forward_on(Device::Cuda(0));
    let cpu_out = instance_norm_forward_on(Device::Cpu);

    assert_parity(
        "InstanceNorm forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}
