//! permute／broadcast_to（`expand`）／squeeze／unsqueeze／flatten
//! （イシュー #1597）の facade 到達経路（既存 `Var` 再エクスポート
//! 経由。新規 `pub use`／`pub fn` は追加していない）の受け入れ条件
//! 対応テスト（`softmax_backend_parity.rs` と同型）。
//!
//! 本 issue の 5 演算自体はホスト `Tensor` の stride 再解釈（view）で
//! あり `BackendOps` を経由しないため（`Var::permute`／`broadcast_to`
//! doc 参照）、ここでは下流の融合経路（`add`）・GEMM カーネル
//! （`matmul`）が view を正しく消費することを検証する。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で
//!   `permute → matmul`／`broadcast_to → add`／
//!   `squeeze/unsqueeze/flatten → sum` の forward／backward を REQ-2
//!   統一複合判定で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と `assert_parity` で比較する（GEMM カーネルが異なるため bit
//!   同一は主張しない）。

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

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    // 決定的シード（`bench-harness::Xorshift64Star`。facade tests は
    // dev-dep として `bench-harness` を既に使用済み）。
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

/// `permute → matmul` forward の CPU（GEMM の NT/TN 入口。#1213／#1215
/// の高速経路が strides 判定で到達しうる形状）と NaiveOps（ホスト
/// `eval::matmul` 参照実装）の parity。
#[test]
fn cpu_permute_matmul_forward_matches_naive_reference() {
    let x_shape = [2usize, 3];
    let w_shape = [2usize, 2];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(2, &w_shape));
    let out_cpu = x_cpu
        .permute(&[1, 0])
        .expect("permute([1,0]) は常に成功する（rank 2）")
        .matmul(&w_cpu)
        .expect("matmul: [3,2] x [2,2] は形状適合")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(2, &w_shape));
    let out_naive = x_naive
        .permute(&[1, 0])
        .expect("permute([1,0]) は常に成功する（rank 2）")
        .matmul(&w_naive)
        .expect("matmul: [3,2] x [2,2] は形状適合")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::gemm 経由 permute→matmul）vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

/// `permute → matmul` backward（`dW`）の CPU と NaiveOps の parity。
#[test]
fn cpu_permute_matmul_backward_matches_naive_reference() {
    let x_shape = [2usize, 3];
    let w_shape = [2usize, 2];
    let target_shape = [3usize, 2];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(2, &w_shape));
    let t_cpu = cpu_tape.make_var(&leaf(3, &target_shape));
    let y_cpu = x_cpu.permute(&[1, 0]).unwrap().matmul(&w_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(2, &w_shape));
    let t_naive = naive_tape.make_var(&leaf(3, &target_shape));
    let y_naive = x_naive.permute(&[1, 0]).unwrap().matmul(&w_naive).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "permute→matmul backward（dW）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

/// `broadcast_to → add` forward・backward の CPU（融合 elementwise
/// 経路）と NaiveOps の parity。
#[test]
fn cpu_broadcast_to_add_matches_naive_reference() {
    let x_shape = [3usize];
    let y_shape = [2usize, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(4, &x_shape));
    let y_cpu = cpu_tape.make_var(&leaf(5, &y_shape));
    let z_cpu = x_cpu
        .broadcast_to(&[2, 3])
        .expect("先頭軸新設の broadcast_to は常に成功する")
        .add(&y_cpu)
        .unwrap();
    let loss_cpu = z_cpu.sum(None).unwrap();
    let forward_cpu = z_cpu.to_tensor();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(4, &x_shape));
    let y_naive = naive_tape.make_var(&leaf(5, &y_shape));
    let z_naive = x_naive
        .broadcast_to(&[2, 3])
        .unwrap()
        .add(&y_naive)
        .unwrap();
    let loss_naive = z_naive.sum(None).unwrap();
    let forward_naive = z_naive.to_tensor();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_parity(
        "broadcast_to→add forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&forward_cpu),
        &contiguous_slice(&forward_naive),
    );
    assert_parity(
        "broadcast_to→add backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
}

/// `squeeze → unsqueeze → flatten → sum`（すべて `reshape` への委譲）
/// forward の CPU と NaiveOps の parity。
#[test]
fn cpu_squeeze_unsqueeze_flatten_matches_naive_reference() {
    let x_shape = [2usize, 1, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let out_cpu = x_cpu
        .squeeze(Some(1))
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .flatten(1, 2)
        .unwrap()
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(6, &x_shape));
    let out_naive = x_naive
        .squeeze(Some(1))
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .flatten(1, 2)
        .unwrap()
        .to_tensor();

    assert_parity(
        "squeeze→unsqueeze→flatten forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn permute_matmul_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[2, 3]));
    let w = tape.make_var(&leaf(2, &[2, 2]));
    x.permute(&[1, 0])
        .expect("permute([1,0]) は常に成功する（rank 2）")
        .matmul(&w)
        .expect("matmul: [3,2] x [2,2] は形状適合")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`softmax_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_permute_matmul_forward_matches_cpu() {
    let metal_out = permute_matmul_forward_on(Device::Metal);
    let cpu_out = permute_matmul_forward_on(Device::Cpu);

    assert_parity(
        "permute→matmul forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_permute_matmul_forward_matches_cpu() {
    let cuda_out = permute_matmul_forward_on(Device::Cuda(0));
    let cpu_out = permute_matmul_forward_on(Device::Cpu);

    assert_parity(
        "permute→matmul forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}
