//! `nn::PixelShuffle`／`nn::PixelUnshuffle`（イシュー #2162・親
//! #2131）のバックエンド間 parity（REQ-2）対応テスト。
//! `spatial_layers_backend_parity.rs`（#2159）と同型。facade は本 2 層
//! を再エクスポートしていない（`docs/autodiff-pixel-shuffle-
//! decision.md` §6 承認事項・`PixelShuffleHoldDoctestGuard`）ため、
//! `fandhe_ai_autodiff::nn::*` を直接 `use` する（dev-dependencies に
//! 既に含まれている）。
//!
//! - 属性なし: `CpuBackendOps`（`fandhe_ai::tape()` 経由）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward・
//!   backward を突き合わせる。純粋なコピー・view の合成のみのため
//!   **bit 完全一致で判定**する（`ZeroPad2d`・`Identity`・`Unflatten`
//!   と同じ契約）。
//! - `#[ignore]`: `cuda_*`（`Device::Cuda(0)`）・`metal_*`（`cfg(
//!   target_os = "macos")`・`Device::Metal`）で同じ検証を CPU と対称
//!   に置く。実機未実測のまま出荷し `docs/perf/logs/pixel-shuffle-
//!   2162/README.md` へ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{PixelShuffle, PixelUnshuffle};
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

fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス一致しない（actual={a}, expected={e}）"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
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

fn print_fold_bits(label: &str, data: &[f32]) {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in data.iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

// --- PixelShuffle（bit 完全一致） ---

#[test]
fn cpu_pixel_shuffle_forward_bit_exact_vs_naive() {
    let layer = PixelShuffle::new(2).unwrap();
    let x_shape = [2usize, 8, 3, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

    assert_bits_eq(
        "PixelShuffle forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_pixel_shuffle_backward_bit_exact_vs_naive() {
    let layer = PixelShuffle::new(2).unwrap();
    let x_shape = [2usize, 8, 3, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(2, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(2, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_bits_eq(
        "PixelShuffle backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
}

// --- PixelUnshuffle（bit 完全一致） ---

#[test]
fn cpu_pixel_unshuffle_forward_bit_exact_vs_naive() {
    let layer = PixelUnshuffle::new(2).unwrap();
    let x_shape = [2usize, 2, 6, 6];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(3, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(3, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

    assert_bits_eq(
        "PixelUnshuffle forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_pixel_unshuffle_backward_bit_exact_vs_naive() {
    let layer = PixelUnshuffle::new(2).unwrap();
    let x_shape = [2usize, 2, 6, 6];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(4, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(4, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_bits_eq(
        "PixelUnshuffle backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。bit 完全一致契約） ---

fn pixel_shuffle_forward_on(device: Device) -> Tensor<f32> {
    let layer = PixelShuffle::new(2).unwrap();
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[2, 8, 3, 3]));
    layer.forward(&x).unwrap().to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_pixel_shuffle_forward_matches_cpu() {
    let metal_out = pixel_shuffle_forward_on(Device::Metal);
    let cpu_out = pixel_shuffle_forward_on(Device::Cpu);

    assert_bits_eq(
        "PixelShuffle forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "metal_pixel_shuffle_forward_matches_cpu[out]",
        &contiguous_slice(&metal_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_pixel_shuffle_forward_matches_cpu() {
    let cuda_out = pixel_shuffle_forward_on(Device::Cuda(0));
    let cpu_out = pixel_shuffle_forward_on(Device::Cpu);

    assert_bits_eq(
        "PixelShuffle forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "cuda_pixel_shuffle_forward_matches_cpu[out]",
        &contiguous_slice(&cuda_out),
    );
}

fn pixel_unshuffle_forward_on(device: Device) -> Tensor<f32> {
    let layer = PixelUnshuffle::new(2).unwrap();
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(3, &[2, 2, 6, 6]));
    layer.forward(&x).unwrap().to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_pixel_unshuffle_forward_matches_cpu() {
    let metal_out = pixel_unshuffle_forward_on(Device::Metal);
    let cpu_out = pixel_unshuffle_forward_on(Device::Cpu);

    assert_bits_eq(
        "PixelUnshuffle forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "metal_pixel_unshuffle_forward_matches_cpu[out]",
        &contiguous_slice(&metal_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_pixel_unshuffle_forward_matches_cpu() {
    let cuda_out = pixel_unshuffle_forward_on(Device::Cuda(0));
    let cpu_out = pixel_unshuffle_forward_on(Device::Cpu);

    assert_bits_eq(
        "PixelUnshuffle forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "cuda_pixel_unshuffle_forward_matches_cpu[out]",
        &contiguous_slice(&cuda_out),
    );
}
