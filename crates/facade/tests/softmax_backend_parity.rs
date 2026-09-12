//! softmax／log_softmax（イシュー #1594）の facade 到達経路（既存
//! `Var` 再エクスポート経由）の受け入れ条件対応テスト。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`。`BackendOps::
//!   softmax`／`log_softmax` を融合カーネルでオーバーライド済み）上の
//!   `Var::softmax`／`log_softmax` forward・backward を、無引数
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。デフォルト
//!   `Unsupported` のためホスト参照実装 `eval::softmax_along`／
//!   `log_softmax_along` へフォールバックする経路）と REQ-2 統一複合
//!   判定で突き合わせる（`fusion_default_parity.rs` と同型）。
//! - `#[ignore]`: `fandhe_ai::tape_for(Device::Metal)`／
//!   `tape_for(Device::Cuda(0))` 上の同経路を CPU tape と突き合わせる
//!   （`device_param_store_backend_parity.rs` と同型。実機必須）。

use bench_harness::rng::Xorshift64Star;
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
    // 決定的シード（`bench-harness::Xorshift64Star`。facade tests は
    // dev-dep として `bench-harness` を既に使用済み——`fusion_default_
    // parity.rs` と異なり本ファイルは新規外部依存を増やさない）。
    let data = Xorshift64Star::new(0xC0FFEE).fill_vec(2 * 5);
    Tensor::new(data, &[2, 5]).expect("leaf: shape 一致")
}

/// `softmax(dim=1)` forward の CPU（融合カーネル経由）と NaiveOps
/// （ホスト参照実装フォールバック）の parity。
#[test]
fn cpu_softmax_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&leaf())
        .softmax(1)
        .expect("softmax: dim=1 は rank=2 の範囲内")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&leaf())
        .softmax(1)
        .expect("softmax: dim=1 は rank=2 の範囲内")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::softmax）vs NaiveOps フォールバック（softmax）",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `log_softmax(dim=1)` forward の CPU（融合カーネル経由）と NaiveOps
/// の parity。
#[test]
fn cpu_log_softmax_forward_matches_naive_reference() {
    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&leaf())
        .log_softmax(1)
        .expect("log_softmax: dim=1 は rank=2 の範囲内")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&leaf())
        .log_softmax(1)
        .expect("log_softmax: dim=1 は rank=2 の範囲内")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::log_softmax デフォルト）vs NaiveOps（log_softmax_along）",
        cpu_out.as_slice().expect("contiguous"),
        naive_out.as_slice().expect("contiguous"),
    );
}

/// `matmul → softmax → mse_loss` backward（`d_input`）の CPU
/// （`Op::Softmax` の VJP。融合カーネル forward + ホスト VJP）と
/// NaiveOps（forward・backward ともホスト参照実装）の parity。
#[test]
fn cpu_softmax_backward_matches_naive_reference() {
    let w = Tensor::new(
        vec![0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3, 0.9],
        &[5, 2],
    )
    .expect("valid tensor");
    let target = Tensor::new(vec![0.2, 0.6, 0.3, 0.4], &[2, 2]).expect("valid tensor");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf());
    let w_cpu = cpu_tape.make_var(&w);
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().softmax(1).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf());
    let w_naive = naive_tape.make_var(&w);
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive.matmul(&w_naive).unwrap().softmax(1).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "softmax backward（dW）: CpuBackendOps vs NaiveOps",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn softmax_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    tape.make_var(&leaf())
        .softmax(1)
        .expect("softmax: dim=1 は rank=2 の範囲内")
        .to_tensor()
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_softmax_forward_matches_cpu() {
    let metal_out = softmax_forward_on(Device::Metal);
    let cpu_out = softmax_forward_on(Device::Cpu);

    assert_parity(
        "softmax forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_softmax_forward_matches_cpu() {
    let cuda_out = softmax_forward_on(Device::Cuda(0));
    let cpu_out = softmax_forward_on(Device::Cpu);

    assert_parity(
        "softmax forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}
