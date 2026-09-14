//! `Var::sort`／`argsort`／`topk`（イシュー #1733 で実装済み。CUDA／
//! Metal カーネルはイシュー #1741）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない）の
//! 受け入れ条件対応テスト（`unique_backend_parity.rs` と同型）。
//!
//! `sort`／`topk` は選択演算（`values` は算術演算を含まない `input`
//! の並べ替え）のため CPU-NaiveOps・CPU-Metal・CPU-CUDA 間で
//! **bit 完全一致**を検証する。forward（`values`／`index`）に加え、
//! backward（`d_input`）も scatter ベースの VJP（`crates/autodiff/src/
//! grad.rs`）のため bit 一致を確認する。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward・
//!   backward を突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する（bit 同一）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_tensor_core::Tensor;

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
    let mut data = Xorshift64Star::new(seed).fill_vec(numel);
    // NaN（両符号）・±0 を注入し同値・特殊値の扱いも検証する
    // （`crates/backend-cuda/src/sort_model.rs` のテストと同じ方針）。
    if numel >= 4 {
        data[0] = f32::NAN;
        data[1] = -f32::NAN;
        data[2] = 0.0;
        data[3] = -0.0;
    }
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn contiguous_i32(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- 属性なし: CPU vs NaiveOps ---

/// `Var::sort` forward・backward の CPU（`BackendOps::sort`）と
/// NaiveOps（ホスト `eval::sort` 参照実装）の parity。
#[test]
fn cpu_sort_matches_naive_reference() {
    let x_data = leaf(1, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let (values_cpu, index_cpu) = x_cpu.sort(1, false).expect("sort は常に成功する");
    let loss_cpu = values_cpu.sum(None).expect("sum");
    let grads_cpu = cpu_tape.backward(&loss_cpu).expect("backward");
    let dx_cpu = grads_cpu.get(&x_cpu).expect("get").expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let (values_naive, index_naive) = x_naive.sort(1, false).expect("sort は常に成功する");
    let loss_naive = values_naive.sum(None).expect("sum");
    let grads_naive = naive_tape.backward(&loss_naive).expect("backward");
    let dx_naive = grads_naive.get(&x_naive).expect("get").expect("到達する");

    assert_eq!(
        contiguous_f32_bits(&values_cpu.value()),
        contiguous_f32_bits(&values_naive.value())
    );
    assert_eq!(contiguous_i32(&index_cpu), contiguous_i32(&index_naive));
    assert_eq!(
        contiguous_f32_bits(dx_cpu),
        contiguous_f32_bits(dx_naive),
        "sort backward: scatter ベース VJP は bit 同一のはず"
    );
}

/// `Var::argsort`（非微分・`index` のみ）の CPU-NaiveOps parity。
#[test]
fn cpu_argsort_matches_naive_reference() {
    let x_data = leaf(2, &[4, 3]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let index_cpu = x_cpu.argsort(0, true).expect("argsort は常に成功する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let index_naive = x_naive.argsort(0, true).expect("argsort は常に成功する");

    assert_eq!(contiguous_i32(&index_cpu), contiguous_i32(&index_naive));
}

/// `Var::topk` forward・backward の CPU-NaiveOps parity。
#[test]
fn cpu_topk_matches_naive_reference() {
    let x_data = leaf(3, &[2, 6]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let (values_cpu, index_cpu) = x_cpu.topk(3, 1, true).expect("topk は常に成功する");
    let loss_cpu = values_cpu.sum(None).expect("sum");
    let grads_cpu = cpu_tape.backward(&loss_cpu).expect("backward");
    let dx_cpu = grads_cpu.get(&x_cpu).expect("get").expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let (values_naive, index_naive) = x_naive.topk(3, 1, true).expect("topk は常に成功する");
    let loss_naive = values_naive.sum(None).expect("sum");
    let grads_naive = naive_tape.backward(&loss_naive).expect("backward");
    let dx_naive = grads_naive.get(&x_naive).expect("get").expect("到達する");

    assert_eq!(
        contiguous_f32_bits(&values_cpu.value()),
        contiguous_f32_bits(&values_naive.value())
    );
    assert_eq!(contiguous_i32(&index_cpu), contiguous_i32(&index_naive));
    assert_eq!(
        contiguous_f32_bits(dx_cpu),
        contiguous_f32_bits(dx_naive),
        "topk backward: scatter ベース VJP は bit 同一のはず"
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn sort_on(device: Device, seed: u64, shape: &[usize]) -> (Tensor<f32>, Tensor<i32>) {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(seed, shape));
    let (values, index) = x.sort(1, false).expect("sort は常に成功する");
    (values.value().clone(), index)
}

fn topk_on(device: Device, seed: u64, shape: &[usize], k: usize) -> (Tensor<f32>, Tensor<i32>) {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(seed, shape));
    let (values, index) = x.topk(k, 1, true).expect("topk は常に成功する");
    (values.value().clone(), index)
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`unique_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sort_matches_cpu() {
    let (metal_values, metal_index) = sort_on(Device::Metal, 10, &[3, 40]);
    let (cpu_values, cpu_index) = sort_on(Device::Cpu, 10, &[3, 40]);

    assert_eq!(
        contiguous_f32_bits(&metal_values),
        contiguous_f32_bits(&cpu_values)
    );
    assert_eq!(contiguous_i32(&metal_index), contiguous_i32(&cpu_index));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_topk_matches_cpu() {
    let (metal_values, metal_index) = topk_on(Device::Metal, 11, &[2, 50], 5);
    let (cpu_values, cpu_index) = topk_on(Device::Cpu, 11, &[2, 50], 5);

    assert_eq!(
        contiguous_f32_bits(&metal_values),
        contiguous_f32_bits(&cpu_values)
    );
    assert_eq!(contiguous_i32(&metal_index), contiguous_i32(&cpu_index));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sort_matches_cpu() {
    let (cuda_values, cuda_index) = sort_on(Device::Cuda(0), 20, &[3, 40]);
    let (cpu_values, cpu_index) = sort_on(Device::Cpu, 20, &[3, 40]);

    assert_eq!(
        contiguous_f32_bits(&cuda_values),
        contiguous_f32_bits(&cpu_values)
    );
    assert_eq!(contiguous_i32(&cuda_index), contiguous_i32(&cpu_index));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_topk_matches_cpu() {
    let (cuda_values, cuda_index) = topk_on(Device::Cuda(0), 21, &[2, 50], 5);
    let (cpu_values, cpu_index) = topk_on(Device::Cpu, 21, &[2, 50], 5);

    assert_eq!(
        contiguous_f32_bits(&cuda_values),
        contiguous_f32_bits(&cpu_values)
    );
    assert_eq!(contiguous_i32(&cuda_index), contiguous_i32(&cpu_index));
}
