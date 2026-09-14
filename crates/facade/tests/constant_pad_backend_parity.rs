//! `pad`（イシュー #1756）の facade 到達経路（既存 `Var` 再エクスポート
//! 経由。新規 `pub use`／`pub fn` は追加していない）の受け入れ条件
//! 対応テスト（`index_ops_backend_parity.rs` と同型）。
//!
//! `pad` は `BackendOps` を経由する非融合ノード（`Op::Pad`。
//! `push_eager`）で、算術を含まない純粋なコピー演算のため CPU／CUDA／
//! Metal の 3 バックエンドとも **bit 完全一致**で検証する。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／
//!   backward を bit 同一で突き合わせる。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`index_ops_backend_parity.rs::
/// VarSource` と同じ理由・同じ構成）。
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

// --- pad（属性なし: CPU vs NaiveOps） ---

/// `pad` forward の CPU（`BackendOps::pad`）と NaiveOps（ホスト
/// `eval::pad` 参照実装）の parity。算術を含まないため bit 同一を
/// 検証する。
#[test]
fn cpu_pad_forward_matches_naive_reference() {
    let shape = [2usize, 3];
    let pads = [(1usize, 0usize), (0usize, 2usize)];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &shape));
    let out_cpu = x_cpu
        .pad(&pads, -1.5)
        .expect("pad: 常に成功する")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &shape));
    let out_naive = x_naive
        .pad(&pads, -1.5)
        .expect("pad: 常に成功する")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps::pad）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_eq!(
        cpu_slice, naive_slice,
        "pad: 算術を含まない純粋なコピー演算のため bit 同一のはず"
    );
}

/// `pad` backward（narrow view 連鎖）の CPU と NaiveOps の parity。
#[test]
fn cpu_pad_backward_matches_naive_reference() {
    let shape = [2usize, 2];
    let pads = [(1usize, 0usize), (0usize, 1usize)];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(2, &shape));
    let out_cpu = x_cpu.pad(&pads, 0.0).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(2, &shape));
    let out_naive = x_naive.pad(&pads, 0.0).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    let dx_cpu_slice = contiguous_slice(dx_cpu);
    let dx_naive_slice = contiguous_slice(dx_naive);
    assert_parity(
        "pad backward（dx）: CpuBackendOps vs NaiveOps",
        &dx_cpu_slice,
        &dx_naive_slice,
    );
    assert_eq!(
        dx_cpu_slice, dx_naive_slice,
        "pad backward: narrow view 連鎖のため bit 同一のはず"
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn pad_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[2, 3]));
    x.pad(&[(1, 0), (0, 2)], -1.5)
        .expect("pad: 常に成功する")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`index_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_pad_forward_matches_cpu() {
    let metal_out = pad_forward_on(Device::Metal);
    let cpu_out = pad_forward_on(Device::Cpu);

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "pad forward: Metal tape_for vs CPU tape_for",
        &metal_slice,
        &cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "pad: 算術を含まない純粋なコピー演算のため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_pad_forward_matches_cpu() {
    let cuda_out = pad_forward_on(Device::Cuda(0));
    let cpu_out = pad_forward_on(Device::Cpu);

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "pad forward: CUDA tape_for vs CPU tape_for",
        &cuda_slice,
        &cpu_slice,
    );
    assert_eq!(
        cuda_slice, cpu_slice,
        "pad: 算術を含まない純粋なコピー演算のため bit 同一のはず"
    );
}
