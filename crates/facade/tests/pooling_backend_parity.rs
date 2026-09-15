//! `Var::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d`（イシュー
//! #1728）の facade 到達経路（既存 `Var` 再エクスポート経由。facade
//! 新規公開面なし）の受け入れ条件対応テスト
//! （`interpolate_backend_parity.rs` と同型）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。`eval::*` 参照
//!   実装へ強制フォールバック）で forward／backward を bit 同一で
//!   突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU
//!   tape と比較する。本 PR 時点では CUDA／Metal とも専用カーネル
//!   未実装（`Unsupported` → ホストフォールバック）のため CPU と
//!   bit 完全一致する契約（設計 doc §10）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする
/// （`interpolate_backend_parity.rs::VarSource` と同じ理由・同じ構成）。
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

// --- max_pool2d（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_max_pool2d_forward_matches_naive_reference() {
    let shape = [1usize, 1, 4, 4];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &shape));
    let (out_cpu, idx_cpu) = x_cpu
        .max_pool2d([2, 2], None, [0, 0], [1, 1], false)
        .expect("max_pool2d: 常に成功する");
    let out_cpu = out_cpu.to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &shape));
    let (out_naive, idx_naive) = x_naive
        .max_pool2d([2, 2], None, [0, 0], [1, 1], false)
        .expect("max_pool2d: 常に成功する");
    let out_naive = out_naive.to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::max_pool2d）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "max_pool2d: 純粋な選択演算のため bit 同一のはず"
    );
    assert_eq!(
        idx_cpu.host_slice().into_owned(),
        idx_naive.host_slice().into_owned(),
        "max_pool2d: 索引も bit 同一のはず"
    );
}

#[test]
fn cpu_max_pool2d_backward_matches_naive_reference() {
    let shape = [1usize, 1, 3, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(2, &shape));
    let (out_cpu, _idx) = x_cpu
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(2, &shape));
    let (out_naive, _idx2) = x_naive
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    let dx_cpu_slice = contiguous_slice(dx_cpu);
    let dx_naive_slice = contiguous_slice(dx_naive);
    assert_parity(
        "max_pool2d backward（dx）: CpuBackendOps vs NaiveOps",
        &dx_cpu_slice,
        &dx_naive_slice,
    );
    assert_eq!(
        dx_cpu_slice, dx_naive_slice,
        "max_pool2d backward: scatter_add ベース VJP のため bit 同一のはず"
    );
}

// --- avg_pool2d（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_avg_pool2d_forward_matches_naive_reference() {
    let shape = [1usize, 2, 3, 4];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(3, &shape));
    let out_cpu = x_cpu
        .avg_pool2d([2, 2], Some([1, 1]), [1, 1], false, true)
        .expect("avg_pool2d: 常に成功する")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(3, &shape));
    let out_naive = x_naive
        .avg_pool2d([2, 2], Some([1, 1]), [1, 1], false, true)
        .expect("avg_pool2d: 常に成功する")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::avg_pool2d）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "avg_pool2d: f64 縮約契約が一致するため bit 同一のはず"
    );
}

#[test]
fn cpu_avg_pool2d_backward_matches_naive_reference() {
    let shape = [1usize, 1, 4, 4];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(4, &shape));
    let out_cpu = x_cpu
        .avg_pool2d([2, 2], Some([2, 2]), [0, 0], false, false)
        .unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(4, &shape));
    let out_naive = x_naive
        .avg_pool2d([2, 2], Some([2, 2]), [0, 0], false, false)
        .unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    let dx_cpu_slice = contiguous_slice(dx_cpu);
    let dx_naive_slice = contiguous_slice(dx_naive);
    assert_parity(
        "avg_pool2d backward（dx）: CpuBackendOps vs NaiveOps",
        &dx_cpu_slice,
        &dx_naive_slice,
    );
}

// --- adaptive_avg_pool2d（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_adaptive_avg_pool2d_forward_and_backward_matches_naive_reference() {
    let shape = [1usize, 1, 5, 7];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(5, &shape));
    let out_cpu = x_cpu.adaptive_avg_pool2d([2, 3]).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(5, &shape));
    let out_naive = x_naive.adaptive_avg_pool2d([2, 3]).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_parity(
        "adaptive_avg_pool2d forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu.to_tensor()),
        &contiguous_slice(&out_naive.to_tensor()),
    );
    assert_parity(
        "adaptive_avg_pool2d backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn max_pool2d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 1, 4, 4]));
    x.max_pool2d([2, 2], None, [0, 0], [1, 1], false)
        .expect("max_pool2d: 常に成功する")
        .0
        .to_tensor()
}

fn avg_pool2d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 3, 4]));
    x.avg_pool2d([2, 2], Some([1, 1]), [1, 1], false, true)
        .expect("avg_pool2d: 常に成功する")
        .to_tensor()
}

fn adaptive_avg_pool2d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 1, 5, 7]));
    x.adaptive_avg_pool2d([2, 3])
        .expect("adaptive_avg_pool2d: 常に成功する")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`interpolate_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_max_pool2d_forward_matches_cpu() {
    let metal_out = max_pool2d_forward_on(Device::Metal);
    let cpu_out = max_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "max_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_avg_pool2d_forward_matches_cpu() {
    let metal_out = avg_pool2d_forward_on(Device::Metal);
    let cpu_out = avg_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "avg_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_adaptive_avg_pool2d_forward_matches_cpu() {
    let metal_out = adaptive_avg_pool2d_forward_on(Device::Metal);
    let cpu_out = adaptive_avg_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "adaptive_avg_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_max_pool2d_forward_matches_cpu() {
    let cuda_out = max_pool2d_forward_on(Device::Cuda(0));
    let cpu_out = max_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "max_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_avg_pool2d_forward_matches_cpu() {
    let cuda_out = avg_pool2d_forward_on(Device::Cuda(0));
    let cpu_out = avg_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "avg_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_adaptive_avg_pool2d_forward_matches_cpu() {
    let cuda_out = adaptive_avg_pool2d_forward_on(Device::Cuda(0));
    let cpu_out = adaptive_avg_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "adaptive_avg_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}
