//! `Var::sub`／`div`／`pow`／`sqrt`（イシュー #1710・親 #1593）の facade
//! 到達経路（既存 `Var` 再エクスポート経由。新規 `pub use`／`pub fn`
//! は追加していない）の受け入れ条件対応テスト（`softmax_backend_
//! parity.rs`・`index_ops_backend_parity.rs` と同型）。
//!
//! 4 演算はいずれも `Var::scalar_binary`／`scalar_unary`
//! （`ScalarBinaryOp`／`ScalarUnaryOp`。イシュー #1634）への薄い委譲
//! であり、forward・VJP 係数の解析値と数値微分（中央差分）の突合は
//! `crates/autodiff/src/grad.rs` の `#[cfg(test)]` 側（`unary_numeric_
//! grad_cases`／`binary_numeric_grad_cases` の `Sqrt`／`Sub`／`Div`／
//! `Pow` 行）で既に検証済みのため、本ファイルでは facade 到達経路
//! （`fandhe_ai::tape()`／`tape_for(Device)`）に限定した 3 バックエンド
//! parity のみを担う。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::scalar_unary`／
//!   `scalar_binary`）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。
//!   既定 `Unsupported` のためホスト参照実装 `eval::scalar::unary`／
//!   `binary` へフォールバックする経路）の forward・backward を
//!   REQ-2 統一複合判定で突き合わせる。`Sub`／`Div`／`Sqrt` は IEEE 754
//!   丸め契約により bit 同一（`crates/backend-{cuda,metal}/tests/
//!   scalar_op_parity.rs` モジュール doc と同じ主張）のため forward の
//!   bit 同一も併記する（超越関数の `Pow` は複合判定のみ）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（既存 parity テストと同じ
/// 構成）。
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

/// 一般値（`sub`。0 近傍・符号混在を含む）。
fn general_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel); // [-1, 1)
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

/// 正の値のみ（`div` の分母・`pow` の底・`sqrt` の入力。0 除算・
/// 定義域外を避ける。`scalar_op_parity.rs::positive_data` と同じ
/// オフセット方針）。
fn positive_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v.abs() + 0.1) // [0.1, 1.1)
        .collect();
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_sub_forward_matches_naive_reference_bit_exact() {
    let a_data = general_leaf(0xC0FFEE, &[2, 3]);
    let b_data = general_leaf(0xBEEF, &[2, 3]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&a_data)
        .sub(&cpu_tape.make_var(&b_data))
        .expect("同 shape の sub は成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&a_data)
        .sub(&naive_tape.make_var(&b_data))
        .expect("同 shape の sub は成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "sub: IEEE 754 丸め契約により CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

#[test]
fn cpu_div_forward_matches_naive_reference_bit_exact() {
    let a_data = general_leaf(0x1234, &[2, 3]);
    let b_data = positive_leaf(0x5678, &[2, 3]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&a_data)
        .div(&cpu_tape.make_var(&b_data))
        .expect("同 shape の div は成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&a_data)
        .div(&naive_tape.make_var(&b_data))
        .expect("同 shape の div は成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "div: IEEE 754 丸め契約により CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

#[test]
fn cpu_pow_forward_matches_naive_reference() {
    let a_data = positive_leaf(0xA1B2, &[2, 3]);
    let b_data = general_leaf(0xC3D4, &[2, 3]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&a_data)
        .pow(&cpu_tape.make_var(&b_data))
        .expect("同 shape の pow は成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&a_data)
        .pow(&naive_tape.make_var(&b_data))
        .expect("同 shape の pow は成功するはず")
        .to_tensor();

    assert_parity(
        "pow forward: CpuBackendOps vs NaiveOps フォールバック",
        &contiguous_slice(&cpu_out),
        &contiguous_slice(&naive_out),
    );
}

#[test]
fn cpu_sqrt_forward_matches_naive_reference_bit_exact() {
    let x_data = positive_leaf(0xF00D, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&x_data)
        .sqrt()
        .expect("sqrt は shape 不変で常に成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&x_data)
        .sqrt()
        .expect("sqrt は shape 不変で常に成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "sqrt: IEEE 754 丸め契約により CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

/// `matmul → sub → mse_loss` backward（`d_input`／`dW`／`dbias`）の
/// CPU（`Op::ScalarBinary` の VJP。#1634）と NaiveOps の parity
/// （broadcast bias 縮約を含む合成経路）。
#[test]
fn cpu_scalar_ops_backward_matches_naive_reference_with_broadcast() {
    let w = Tensor::new(
        vec![0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3, 0.9],
        &[5, 2],
    )
    .expect("valid tensor");
    let bias = Tensor::new(vec![0.3, 0.7], &[2]).expect("valid tensor"); // sub の broadcast rhs
    let target = Tensor::new(vec![0.2, 0.6, 0.3, 0.4, 0.1, 0.5], &[3, 2]).expect("valid tensor");
    let x_data = general_leaf(0x9E37_79B9, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let w_cpu = cpu_tape.make_var(&w);
    let bias_cpu = cpu_tape.make_var(&bias);
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().sub(&bias_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");
    let dbias_cpu = grads_cpu.get(&bias_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let w_naive = naive_tape.make_var(&w);
    let bias_naive = naive_tape.make_var(&bias);
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive.matmul(&w_naive).unwrap().sub(&bias_naive).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");
    let dbias_naive = grads_naive.get(&bias_naive).unwrap().expect("到達する");

    assert_parity(
        "sub backward（dW）: CpuBackendOps vs NaiveOps",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );
    assert_parity(
        "sub backward（dbias, broadcast 縮約）: CpuBackendOps vs NaiveOps",
        dbias_cpu.as_slice().expect("contiguous"),
        dbias_naive.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn sub_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = general_leaf(0x1111, &[3, 4]);
    let b = general_leaf(0x2222, &[3, 4]);
    tape.make_var(&a)
        .sub(&tape.make_var(&b))
        .expect("同 shape の sub は成功するはず")
        .to_tensor()
}

fn div_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = general_leaf(0x3333, &[3, 4]);
    let b = positive_leaf(0x4444, &[3, 4]);
    tape.make_var(&a)
        .div(&tape.make_var(&b))
        .expect("同 shape の div は成功するはず")
        .to_tensor()
}

fn pow_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = positive_leaf(0x5555, &[3, 4]);
    let b = general_leaf(0x6666, &[3, 4]);
    tape.make_var(&a)
        .pow(&tape.make_var(&b))
        .expect("同 shape の pow は成功するはず")
        .to_tensor()
}

fn sqrt_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = positive_leaf(0x7777, &[3, 4]);
    tape.make_var(&x)
        .sqrt()
        .expect("sqrt は shape 不変で常に成功するはず")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある。`#[ignore]`
// は実行のみをスキップしコンパイルはスキップしないため、Linux（CI の
// ubuntu-latest）では `cfg` ゲートがないと E0599 でビルド不能になる。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sub_forward_matches_cpu_bit_exact() {
    let metal_out = sub_forward_on(Device::Metal);
    let cpu_out = sub_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "sub: IEEE 754 丸め契約により Metal と CPU は bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_div_forward_matches_cpu_bit_exact() {
    let metal_out = div_forward_on(Device::Metal);
    let cpu_out = div_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "div: IEEE 754 丸め契約により Metal と CPU は bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_pow_forward_matches_cpu() {
    let metal_out = pow_forward_on(Device::Metal);
    let cpu_out = pow_forward_on(Device::Cpu);
    assert_parity(
        "pow forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sqrt_forward_matches_cpu_bit_exact() {
    let metal_out = sqrt_forward_on(Device::Metal);
    let cpu_out = sqrt_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "sqrt: IEEE 754 丸め契約により Metal と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sub_forward_matches_cpu_bit_exact() {
    let cuda_out = sub_forward_on(Device::Cuda(0));
    let cpu_out = sub_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "sub: IEEE 754 丸め契約により CUDA と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_div_forward_matches_cpu_bit_exact() {
    let cuda_out = div_forward_on(Device::Cuda(0));
    let cpu_out = div_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "div: IEEE 754 丸め契約により CUDA と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_pow_forward_matches_cpu() {
    let cuda_out = pow_forward_on(Device::Cuda(0));
    let cpu_out = pow_forward_on(Device::Cpu);
    assert_parity(
        "pow forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sqrt_forward_matches_cpu_bit_exact() {
    let cuda_out = sqrt_forward_on(Device::Cuda(0));
    let cpu_out = sqrt_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "sqrt: IEEE 754 丸め契約により CUDA と CPU は bit 同一のはず"
    );
}
