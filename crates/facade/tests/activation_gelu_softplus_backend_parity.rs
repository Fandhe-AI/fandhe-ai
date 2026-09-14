//! `Var::gelu`／`gelu_tanh`／`softplus`（イシュー #1713）の facade 到達
//! 経路（既存 `Var` 再エクスポート経由。`crates/facade/src/lib.rs` への
//! 新規 `pub use`／`pub fn` は追加していない）の受け入れ条件対応
//! テスト。`scalar_unary_transcendental_backend_parity.rs`（#1711）と
//! 同型の構成を踏襲する。
//!
//! バックエンド 3 クレート（`backend-cpu`／`backend-cuda`／
//! `backend-metal`）・`ScalarUnaryOp` enum・dispatch・CPU 参照実装・VJP
//! は #1592／#1634／#1635／#1636／#1713 で実装済みのため、本ファイルは
//! `Var` の新規 3 メソッドから既存カーネルへ到達できることのみを検証
//! する（新規カーネル実装は含まない）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::scalar_unary`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps` → `eval::scalar::
//!   unary` フォールバック）で forward／backward を突き合わせる。
//!   CPU tape と `NaiveOps` はどちらも最終的に
//!   `ScalarUnaryOp::apply` を呼ぶため（`backend-cpu/tests/
//!   scalar_op_parity.rs` で bit 同一を確認済み）forward は bit 同一
//!   （`assert_eq!`）で検証する（tolerance 変更なしのより強い検証）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の forward を CPU
//!   tape と REQ-2 複合判定（`assert_parity`。GELU／Softplus は超越
//!   関数のため bit 同一は主張しない）で突き合わせる。本エージェント
//!   実行環境に実機への到達手段がないため未実測のまま Mac／GB10
//!   セッションへ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`softmax_backend_parity.rs`
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

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

/// `[-1, 1)` の一様乱数（GELU／Softplus とも定義域制約なしのため使う）。
fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

// --- forward parity（属性なし: CPU vs NaiveOps。bit 同一） ---

#[test]
fn cpu_gelu_forward_matches_naive_reference_bit_exact() {
    let shape = [2usize, 3];
    let x_val = leaf(31, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape.make_var(&x_val).gelu().unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape.make_var(&x_val).gelu().unwrap().to_tensor();

    assert_eq!(
        contiguous_slice(&out_cpu),
        contiguous_slice(&out_naive),
        "gelu: CPU tape も NaiveOps も ScalarUnaryOp::apply を呼ぶため bit 同一のはず"
    );
}

#[test]
fn cpu_gelu_tanh_forward_matches_naive_reference_bit_exact() {
    let shape = [2usize, 3];
    let x_val = leaf(32, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape.make_var(&x_val).gelu_tanh().unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape.make_var(&x_val).gelu_tanh().unwrap().to_tensor();

    assert_eq!(
        contiguous_slice(&out_cpu),
        contiguous_slice(&out_naive),
        "gelu_tanh: CPU tape も NaiveOps も ScalarUnaryOp::apply を呼ぶため bit 同一のはず"
    );
}

#[test]
fn cpu_softplus_forward_matches_naive_reference_bit_exact() {
    let shape = [2usize, 3];
    let x_val = leaf(33, &shape);

    let cpu_tape = fandhe_ai::tape();
    let out_cpu = cpu_tape
        .make_var(&x_val)
        .softplus(1.0, 20.0)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let out_naive = naive_tape
        .make_var(&x_val)
        .softplus(1.0, 20.0)
        .unwrap()
        .to_tensor();

    assert_eq!(
        contiguous_slice(&out_cpu),
        contiguous_slice(&out_naive),
        "softplus: CPU tape も NaiveOps も ScalarUnaryOp::apply を呼ぶため bit 同一のはず"
    );
}

// --- backward 合成（属性なし: CPU vs NaiveOps） ---

/// `matmul → gelu → sum` の CPU と NaiveOps の parity（`dW`／`dx`）。
#[test]
fn cpu_matmul_gelu_backward_matches_naive_reference() {
    let x_shape = [3usize, 2];
    let w_shape = [2usize, 4];

    let x_val = leaf(41, &x_shape);
    let w_val = leaf(42, &w_shape);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_val);
    let w_cpu = cpu_tape.make_var(&w_val);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().gelu().unwrap();
    let loss_cpu = y_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_val);
    let w_naive = naive_tape.make_var(&w_val);
    let y_naive = x_naive.matmul(&w_naive).unwrap().gelu().unwrap();
    let loss_naive = y_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "matmul→gelu backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_parity(
        "matmul→gelu backward（dW）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

/// `matmul → softplus → sum` の CPU と NaiveOps の parity（`dW`／`dx`）。
#[test]
fn cpu_matmul_softplus_backward_matches_naive_reference() {
    let x_shape = [3usize, 2];
    let w_shape = [2usize, 4];

    let x_val = leaf(43, &x_shape);
    let w_val = leaf(44, &w_shape);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_val);
    let w_cpu = cpu_tape.make_var(&w_val);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().softplus(1.0, 20.0).unwrap();
    let loss_cpu = y_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_val);
    let w_naive = naive_tape.make_var(&w_val);
    let y_naive = x_naive
        .matmul(&w_naive)
        .unwrap()
        .softplus(1.0, 20.0)
        .unwrap();
    let loss_naive = y_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "matmul→softplus backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_parity(
        "matmul→softplus backward（dW）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn unary_forward_on(
    device: Device,
    x: &Tensor<f32>,
    apply: impl for<'a> Fn(&'a Var<'a>) -> Var<'a>,
) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let v = tape.make_var(x);
    apply(&v).to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`scalar_unary_transcendental_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_gelu_forward_matches_cpu() {
    let x = leaf(31, &[2, 3]);
    let metal_out = unary_forward_on(Device::Metal, &x, |v| v.gelu().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.gelu().unwrap());
    assert_parity(
        "gelu forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_gelu_tanh_forward_matches_cpu() {
    let x = leaf(32, &[2, 3]);
    let metal_out = unary_forward_on(Device::Metal, &x, |v| v.gelu_tanh().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.gelu_tanh().unwrap());
    assert_parity(
        "gelu_tanh forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_softplus_forward_matches_cpu() {
    let x = leaf(33, &[2, 3]);
    let metal_out = unary_forward_on(Device::Metal, &x, |v| v.softplus(1.0, 20.0).unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.softplus(1.0, 20.0).unwrap());
    assert_parity(
        "softplus forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_gelu_forward_matches_cpu() {
    let x = leaf(31, &[2, 3]);
    let cuda_out = unary_forward_on(Device::Cuda(0), &x, |v| v.gelu().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.gelu().unwrap());
    assert_parity(
        "gelu forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_gelu_tanh_forward_matches_cpu() {
    let x = leaf(32, &[2, 3]);
    let cuda_out = unary_forward_on(Device::Cuda(0), &x, |v| v.gelu_tanh().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.gelu_tanh().unwrap());
    assert_parity(
        "gelu_tanh forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_softplus_forward_matches_cpu() {
    let x = leaf(33, &[2, 3]);
    let cuda_out = unary_forward_on(Device::Cuda(0), &x, |v| v.softplus(1.0, 20.0).unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.softplus(1.0, 20.0).unwrap());
    assert_parity(
        "softplus forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}
