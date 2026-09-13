//! `Var::einsum`（イシュー #1620）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない）の
//! 受け入れ条件対応テスト（`shape_ops_backend_parity.rs` と同型）。
//!
//! `crate::einsum` は新規カーネルを追加せず既存の `matmul`／`sum`／
//! `permute`／`reshape`／`mul` への分解として実装しているため、ここでは
//! 「該当バックエンドすべてに実装」を分解によって自動的に充足して
//! いることを、GEMM 経路（`"ji,jk->ik"`。実際に非恒等 permute →
//! `Op::Contiguous` を経由する）で確認する。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／backward
//!   を REQ-2 統一複合判定で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定・M4 Max 実機実測済み）／`tape_for(Device::Cuda(0))`
//!   （本エージェント実行環境に CUDA 実機なしのため未実測。GB10 実機
//!   セッションへ引き継ぐ）の同経路を CPU tape と `assert_parity` で
//!   比較する（GEMM カーネルが異なるため bit 同一は主張しない）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`shape_ops_backend_parity.rs`
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

/// `"ji,jk->ik"`（a 側の目標順が現在の添字順と異なり実際に非恒等
/// `permute` → `Op::Contiguous` を経由する GEMM 経路。`crate::
/// einsum::einsum_matmul_path` doc 参照）forward の CPU
/// （`CpuBackendOps::gemm` 経由）と NaiveOps（ホスト `eval::matmul`
/// 参照実装）の parity。
#[test]
fn cpu_einsum_forward_matches_naive_reference() {
    let a_shape = [3usize, 2]; // [j, i]
    let b_shape = [3usize, 4]; // [j, k]

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(1, &a_shape));
    let b_cpu = cpu_tape.make_var(&leaf(2, &b_shape));
    let out_cpu = Var::einsum("ji,jk->ik", &[&a_cpu, &b_cpu])
        .expect("einsum: 形状適合（j 次元一致）")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(1, &a_shape));
    let b_naive = naive_tape.make_var(&leaf(2, &b_shape));
    let out_naive = Var::einsum("ji,jk->ik", &[&a_naive, &b_naive])
        .expect("einsum: 形状適合（j 次元一致）")
        .to_tensor();

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps 経由 einsum \"ji,jk->ik\"）vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

/// 同経路 backward（`da`）の CPU と NaiveOps の parity。
#[test]
fn cpu_einsum_backward_matches_naive_reference() {
    let a_shape = [3usize, 2];
    let b_shape = [3usize, 4];
    let target_shape = [2usize, 4];

    let cpu_tape = fandhe_ai::tape();
    let a_cpu = cpu_tape.make_var(&leaf(1, &a_shape));
    let b_cpu = cpu_tape.make_var(&leaf(2, &b_shape));
    let t_cpu = cpu_tape.make_var(&leaf(3, &target_shape));
    let y_cpu = Var::einsum("ji,jk->ik", &[&a_cpu, &b_cpu]).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let da_cpu = grads_cpu.get(&a_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let a_naive = naive_tape.make_var(&leaf(1, &a_shape));
    let b_naive = naive_tape.make_var(&leaf(2, &b_shape));
    let t_naive = naive_tape.make_var(&leaf(3, &target_shape));
    let y_naive = Var::einsum("ji,jk->ik", &[&a_naive, &b_naive]).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let da_naive = grads_naive.get(&a_naive).unwrap().expect("到達する");

    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps 経由 einsum \"ji,jk->ik\"）backward vs NaiveOps",
        &contiguous_slice(da_cpu),
        &contiguous_slice(da_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn einsum_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&leaf(1, &[3, 2]));
    let b = tape.make_var(&leaf(2, &[3, 4]));
    Var::einsum("ji,jk->ik", &[&a, &b])
        .expect("einsum: 形状適合（j 次元一致）")
        .to_tensor()
}

fn einsum_backward_da_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = tape.make_var(&leaf(1, &[3, 2]));
    let b = tape.make_var(&leaf(2, &[3, 4]));
    let t = tape.make_var(&leaf(3, &[2, 4]));
    let y = Var::einsum("ji,jk->ik", &[&a, &b]).unwrap();
    let loss = y.mse_loss(&t).unwrap();
    let grads = tape.backward(&loss).unwrap();
    grads.get(&a).unwrap().expect("到達する").clone()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`shape_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_einsum_forward_matches_cpu() {
    let metal_out = einsum_forward_on(Device::Metal);
    let cpu_out = einsum_forward_on(Device::Cpu);

    assert_parity(
        "einsum \"ji,jk->ik\" forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_einsum_backward_matches_cpu() {
    let metal_da = einsum_backward_da_on(Device::Metal);
    let cpu_da = einsum_backward_da_on(Device::Cpu);

    assert_parity(
        "einsum \"ji,jk->ik\" backward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_da),
        &contiguous_slice(&cpu_da),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_einsum_forward_matches_cpu() {
    let cuda_out = einsum_forward_on(Device::Cuda(0));
    let cpu_out = einsum_forward_on(Device::Cpu);

    assert_parity(
        "einsum \"ji,jk->ik\" forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_einsum_backward_matches_cpu() {
    let cuda_da = einsum_backward_da_on(Device::Cuda(0));
    let cpu_da = einsum_backward_da_on(Device::Cpu);

    assert_parity(
        "einsum \"ji,jk->ik\" backward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_da),
        &contiguous_slice(&cpu_da),
    );
}
