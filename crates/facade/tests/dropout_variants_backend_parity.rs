//! `nn::Dropout2d`／`nn::AlphaDropout`（イシュー #2161・親 #2131）の
//! 3 バックエンド受け入れ条件対応テスト（`dropout_backend_parity.rs`
//! と同型。facade 公開は保留のため `fandhe_ai_autodiff::nn` を直接
//! 経由する——facade 自体からの到達性は本ファイルの対象外）。
//!
//! - `Dropout2d`: マスク生成後は `BackendOps::mul` 1 回のみ（`Dropout`
//!   と同じ `Op::Dropout` 経路）のため forward／backward とも bit 完全
//!   一致を主張する。
//! - `AlphaDropout`: `BackendOps::mul`（`x * noise`）→
//!   `BackendOps::add`（`+ b`）の 2 回のみのため、こちらも bit 完全
//!   一致を主張する。`Max` を経由しない縮約なし演算のため丸め方式の
//!   バックエンド差は生じない。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／
//!   backward を bit 同一＋`assert_parity` 併記で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` を CPU `tape_for`
//!   と比較する（各バックエンド呼び出し直前に `manual_seed` を打ち直す）。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{AlphaDropout, Dropout2d};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

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

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- 属性なし（CPU vs NaiveOps） -----------------------------------------

#[test]
fn cpu_dropout2d_forward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [2usize, 3, 2, 2];
    let x_data = leaf(31, &shape);
    let d = Dropout2d::new(0.4).expect("有効な p");

    fandhe_ai::manual_seed(9201);
    let cpu_tape = fandhe_ai::tape();
    let cpu_x = cpu_tape.make_var(&x_data);
    let cpu_out = d.forward(&cpu_x).expect("rank 4 入力").to_tensor();

    fandhe_ai::manual_seed(9201);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_x = naive_tape.make_var(&x_data);
    let naive_out = d.forward(&naive_x).expect("rank 4 入力").to_tensor();

    let cpu_slice = contiguous_slice(&cpu_out);
    let naive_slice = contiguous_slice(&naive_out);
    assert_parity(
        "dropout2d forward: CPU vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "dropout2d forward: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
fn cpu_dropout2d_backward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [2usize, 2, 2, 2];
    let x_data = leaf(32, &shape);
    let d = Dropout2d::new(0.5).expect("有効な p");

    fandhe_ai::manual_seed(9202);
    let cpu_tape = fandhe_ai::tape();
    let cpu_x = cpu_tape.make_var(&x_data);
    let cpu_out = d.forward(&cpu_x).expect("rank 4 入力");
    let cpu_loss = cpu_out.sum(None).expect("全軸縮約は失敗しない");
    let cpu_grads = cpu_tape
        .backward(&cpu_loss)
        .expect("x は requires_grad の葉");
    let cpu_dx = cpu_grads
        .get(&cpu_x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    fandhe_ai::manual_seed(9202);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_x = naive_tape.make_var(&x_data);
    let naive_out = d.forward(&naive_x).expect("rank 4 入力");
    let naive_loss = naive_out.sum(None).expect("全軸縮約は失敗しない");
    let naive_grads = naive_tape
        .backward(&naive_loss)
        .expect("x は requires_grad の葉");
    let naive_dx = naive_grads
        .get(&naive_x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    let cpu_slice = contiguous_slice(cpu_dx);
    let naive_slice = contiguous_slice(naive_dx);
    assert_parity(
        "dropout2d backward: CPU vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "dropout2d backward: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
fn cpu_alpha_dropout_forward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [4usize, 5];
    let x_data = leaf(33, &shape);
    let d = AlphaDropout::new(0.3).expect("有効な p");

    fandhe_ai::manual_seed(9203);
    let cpu_tape = fandhe_ai::tape();
    let cpu_x = cpu_tape.make_var(&x_data);
    let cpu_out = d.forward(&cpu_x).expect("training 既定 true").to_tensor();

    fandhe_ai::manual_seed(9203);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_x = naive_tape.make_var(&x_data);
    let naive_out = d.forward(&naive_x).expect("training 既定 true").to_tensor();

    let cpu_slice = contiguous_slice(&cpu_out);
    let naive_slice = contiguous_slice(&naive_out);
    assert_parity(
        "alpha_dropout forward: CPU vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "alpha_dropout forward: mul → add の 2 回のみのため bit 同一のはず"
    );
}

#[test]
fn cpu_alpha_dropout_backward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [3usize, 4];
    let x_data = leaf(34, &shape);
    let d = AlphaDropout::new(0.5).expect("有効な p");

    fandhe_ai::manual_seed(9204);
    let cpu_tape = fandhe_ai::tape();
    let cpu_x = cpu_tape.make_var(&x_data);
    let cpu_out = d.forward(&cpu_x).expect("training 既定 true");
    let cpu_loss = cpu_out.sum(None).expect("全軸縮約は失敗しない");
    let cpu_grads = cpu_tape
        .backward(&cpu_loss)
        .expect("x は requires_grad の葉");
    let cpu_dx = cpu_grads
        .get(&cpu_x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    fandhe_ai::manual_seed(9204);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_x = naive_tape.make_var(&x_data);
    let naive_out = d.forward(&naive_x).expect("training 既定 true");
    let naive_loss = naive_out.sum(None).expect("全軸縮約は失敗しない");
    let naive_grads = naive_tape
        .backward(&naive_loss)
        .expect("x は requires_grad の葉");
    let naive_dx = naive_grads
        .get(&naive_x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    let cpu_slice = contiguous_slice(cpu_dx);
    let naive_slice = contiguous_slice(naive_dx);
    assert_parity(
        "alpha_dropout backward: CPU vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "alpha_dropout backward: 勾配は upstream ⊙ noise のみのため bit 同一のはず"
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---------------------------------

fn dropout2d_forward_on(device: Device, seed: u64, x: &Tensor<f32>, d: &Dropout2d) -> Tensor<f32> {
    fandhe_ai::manual_seed(seed);
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let v = tape.make_var(x);
    d.forward(&v).expect("rank 4 入力").to_tensor()
}

fn alpha_dropout_forward_on(
    device: Device,
    seed: u64,
    x: &Tensor<f32>,
    d: &AlphaDropout,
) -> Tensor<f32> {
    fandhe_ai::manual_seed(seed);
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let v = tape.make_var(x);
    d.forward(&v).expect("training 既定 true").to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_dropout2d_forward_matches_cpu() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let x = leaf(41, &[2, 3, 2, 2]);
    let d = Dropout2d::new(0.5).expect("有効な p");
    let metal_out = dropout2d_forward_on(Device::Metal, 9301, &x, &d);
    let cpu_out = dropout2d_forward_on(Device::Cpu, 9301, &x, &d);

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "dropout2d forward: Metal tape_for vs CPU tape_for",
        &metal_slice,
        &cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "dropout2d: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_alpha_dropout_forward_matches_cpu() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let x = leaf(42, &[4, 5]);
    let d = AlphaDropout::new(0.4).expect("有効な p");
    let metal_out = alpha_dropout_forward_on(Device::Metal, 9302, &x, &d);
    let cpu_out = alpha_dropout_forward_on(Device::Cpu, 9302, &x, &d);

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "alpha_dropout forward: Metal tape_for vs CPU tape_for",
        &metal_slice,
        &cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "alpha_dropout: mul → add の 2 回のみのため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_dropout2d_forward_matches_cpu() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let x = leaf(43, &[2, 3, 2, 2]);
    let d = Dropout2d::new(0.5).expect("有効な p");
    let cuda_out = dropout2d_forward_on(Device::Cuda(0), 9303, &x, &d);
    let cpu_out = dropout2d_forward_on(Device::Cpu, 9303, &x, &d);

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "dropout2d forward: CUDA tape_for vs CPU tape_for",
        &cuda_slice,
        &cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "dropout2d: マスク乗算は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_alpha_dropout_forward_matches_cpu() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let x = leaf(44, &[4, 5]);
    let d = AlphaDropout::new(0.4).expect("有効な p");
    let cuda_out = alpha_dropout_forward_on(Device::Cuda(0), 9304, &x, &d);
    let cpu_out = alpha_dropout_forward_on(Device::Cpu, 9304, &x, &d);

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "alpha_dropout forward: CUDA tape_for vs CPU tape_for",
        &cuda_slice,
        &cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "alpha_dropout: mul → add の 2 回のみのため bit 同一のはず"
    );
}
