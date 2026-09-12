//! RMSNorm／LayerNorm（イシュー #1596）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。`crates/facade/src/` は無変更——`docs/
//! compat-api-scope.md` §1.2「facade 到達経路は既存 `Var` 再エクスポート
//! 経由」）の受け入れ条件対応テスト。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`BackendOps::
//!   rmsnorm`／`layer_norm` を実機カーネルでオーバーライド済み）上の
//!   `Var::rms_norm`／`layer_norm` forward・backward を、無引数
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps` → ホスト参照実装
//!   `eval::rmsnorm_rows`／`layer_norm_rows` へフォールバックする経路）
//!   と REQ-2 統一複合判定で突き合わせる（`fusion_default_parity.rs`・
//!   `softmax_backend_parity.rs`〈イシュー #1594〉と同型）。
//! - `#[ignore]`: `fandhe_ai::tape_for(Device::Metal)`／
//!   `tape_for(Device::Cuda(0))` 上の同経路を CPU tape と突き合わせる
//!   （実機必須）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`fusion_default_parity.rs`
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

fn leaf() -> Tensor<f32> {
    // 決定的な固定値（`fusion_default_parity.rs` と同じ方針。facade へ
    // dev-dep を増やさないため乱数ユーティリティは使わない）。
    Tensor::new(
        vec![0.5, -1.2, 2.0, -0.3, 1.1, -0.7, 0.2, -2.1, 0.9, -0.4],
        &[2, 5],
    )
    .expect("leaf: shape 一致")
}

fn weight() -> Tensor<f32> {
    Tensor::new(vec![1.5, -0.8, 1.0, 0.3, -1.2], &[5]).expect("weight: shape 一致")
}

fn bias() -> Tensor<f32> {
    Tensor::new(vec![0.1, -0.2, 0.3, 0.0, -0.1], &[5]).expect("bias: shape 一致")
}

// --- RmsNorm ---

/// `rms_norm(weight, eps)` forward の CPU（融合カーネル経由）と NaiveOps
/// （ホスト参照実装フォールバック）の parity。
#[test]
fn cpu_rms_norm_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let w_cpu = cpu_tape.make_var(&weight());
    let cpu_out = cpu_tape
        .make_var(&leaf())
        .rms_norm(Some(&w_cpu), 1e-6)
        .expect("rms_norm: weight shape [5] は hidden=5 と一致")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let w_naive = naive_tape.make_var(&weight());
    let naive_out = naive_tape
        .make_var(&leaf())
        .rms_norm(Some(&w_naive), 1e-6)
        .expect("rms_norm: weight shape [5] は hidden=5 と一致")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::rmsnorm）vs NaiveOps フォールバック（rms_norm）",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `matmul → rms_norm → mse_loss` backward（`dW`）の CPU（`Op::RmsNorm`
/// の VJP。融合カーネル forward + ホスト VJP）と NaiveOps（forward・
/// backward ともホスト参照実装）の parity。
#[test]
fn cpu_rms_norm_backward_matches_naive_reference() {
    let w_lin = Tensor::new(
        vec![
            0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3, 0.9, 0.2, -0.5, 0.1, 0.3, -0.2, -0.4,
            0.6, -0.1, 0.2, 0.5, 0.3, -0.2, 0.4, -0.6, 0.1,
        ],
        &[5, 5],
    )
    .expect("valid tensor");
    let target = Tensor::new(
        vec![0.2, 0.6, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2, 0.1, -0.2],
        &[2, 5],
    )
    .expect("valid tensor");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf());
    let w_lin_cpu = cpu_tape.make_var(&w_lin);
    let w_rms_cpu = cpu_tape.make_var(&weight());
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu
        .matmul(&w_lin_cpu)
        .unwrap()
        .rms_norm(Some(&w_rms_cpu), 1e-6)
        .unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_rms_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf());
    let w_lin_naive = naive_tape.make_var(&w_lin);
    let w_rms_naive = naive_tape.make_var(&weight());
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive
        .matmul(&w_lin_naive)
        .unwrap()
        .rms_norm(Some(&w_rms_naive), 1e-6)
        .unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_rms_naive).unwrap().expect("到達する");

    assert_parity(
        "rms_norm backward（dW）: CpuBackendOps vs NaiveOps",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );
}

// --- LayerNorm ---

/// `layer_norm(weight, bias, eps)` forward の CPU（新設カーネル経由）と
/// NaiveOps の parity。
#[test]
fn cpu_layer_norm_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let w_cpu = cpu_tape.make_var(&weight());
    let b_cpu = cpu_tape.make_var(&bias());
    let cpu_out = cpu_tape
        .make_var(&leaf())
        .layer_norm(Some(&w_cpu), Some(&b_cpu), 1e-5)
        .expect("layer_norm: weight/bias shape [5] は hidden=5 と一致")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let w_naive = naive_tape.make_var(&weight());
    let b_naive = naive_tape.make_var(&bias());
    let naive_out = naive_tape
        .make_var(&leaf())
        .layer_norm(Some(&w_naive), Some(&b_naive), 1e-5)
        .expect("layer_norm: weight/bias shape [5] は hidden=5 と一致")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::layer_norm）vs NaiveOps フォールバック（layer_norm）",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `matmul → layer_norm → mse_loss` backward（`dW`／`dB`）の CPU
/// （`Op::LayerNorm` の VJP）と NaiveOps の parity。
#[test]
fn cpu_layer_norm_backward_matches_naive_reference() {
    let w_lin = Tensor::new(
        vec![
            0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3, 0.9, 0.2, -0.5, 0.1, 0.3, -0.2, -0.4,
            0.6, -0.1, 0.2, 0.5, 0.3, -0.2, 0.4, -0.6, 0.1,
        ],
        &[5, 5],
    )
    .expect("valid tensor");
    let target = Tensor::new(
        vec![0.2, 0.6, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2, 0.1, -0.2],
        &[2, 5],
    )
    .expect("valid tensor");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf());
    let w_lin_cpu = cpu_tape.make_var(&w_lin);
    let w_ln_cpu = cpu_tape.make_var(&weight());
    let b_ln_cpu = cpu_tape.make_var(&bias());
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu
        .matmul(&w_lin_cpu)
        .unwrap()
        .layer_norm(Some(&w_ln_cpu), Some(&b_ln_cpu), 1e-5)
        .unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_ln_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_ln_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf());
    let w_lin_naive = naive_tape.make_var(&w_lin);
    let w_ln_naive = naive_tape.make_var(&weight());
    let b_ln_naive = naive_tape.make_var(&bias());
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive
        .matmul(&w_lin_naive)
        .unwrap()
        .layer_norm(Some(&w_ln_naive), Some(&b_ln_naive), 1e-5)
        .unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_ln_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_ln_naive).unwrap().expect("到達する");

    assert_parity(
        "layer_norm backward（dW）: CpuBackendOps vs NaiveOps",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );
    assert_parity(
        "layer_norm backward（dB）: CpuBackendOps vs NaiveOps",
        db_cpu.as_slice().expect("contiguous"),
        db_naive.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn rms_norm_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let w = tape.make_var(&weight());
    tape.make_var(&leaf())
        .rms_norm(Some(&w), 1e-6)
        .expect("rms_norm: weight shape [5] は hidden=5 と一致")
        .to_tensor()
}

fn layer_norm_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let w = tape.make_var(&weight());
    let b = tape.make_var(&bias());
    tape.make_var(&leaf())
        .layer_norm(Some(&w), Some(&b), 1e-5)
        .expect("layer_norm: weight/bias shape [5] は hidden=5 と一致")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある。`#[ignore]`
// は実行のみをスキップしコンパイルはスキップしないため、Linux（CI の
// ubuntu-latest）では `cfg` ゲートがないと E0599 でビルド不能になる
// （`softmax_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_rms_norm_forward_matches_cpu() {
    let metal_out = rms_norm_forward_on(Device::Metal);
    let cpu_out = rms_norm_forward_on(Device::Cpu);

    assert_parity(
        "rms_norm forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_layer_norm_forward_matches_cpu() {
    let metal_out = layer_norm_forward_on(Device::Metal);
    let cpu_out = layer_norm_forward_on(Device::Cpu);

    assert_parity(
        "layer_norm forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_rms_norm_forward_matches_cpu() {
    let cuda_out = rms_norm_forward_on(Device::Cuda(0));
    let cpu_out = rms_norm_forward_on(Device::Cpu);

    assert_parity(
        "rms_norm forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_layer_norm_forward_matches_cpu() {
    let cuda_out = layer_norm_forward_on(Device::Cuda(0));
    let cpu_out = layer_norm_forward_on(Device::Cpu);

    assert_parity(
        "layer_norm forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}
