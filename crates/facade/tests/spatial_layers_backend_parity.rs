//! `nn::ConvTranspose1d`／`nn::Upsample`／`nn::ZeroPad2d`／
//! `nn::Identity`／`nn::Unflatten`（イシュー #2159・親 #2131）の
//! バックエンド間 parity（REQ-2）対応テスト。`conv_transpose2d_
//! backend_parity.rs`・`activation_ops_backend_parity.rs` と同型。
//! facade は本 5 層を再エクスポートしていない（`docs/autodiff-
//! spatial-layers-decision.md` §6 承認事項 1・
//! `SpatialLayersHoldDoctestGuard`）ため、`fandhe_ai_autodiff::nn::*`
//! を直接 `use` する（dev-dependencies に既に含まれている）。
//!
//! - 属性なし: `CpuBackendOps`（`ConvTranspose1d` は `.bind()` が
//!   `&fandhe_ai_autodiff::Tape` を要求するため [`raw_tape_for`] 経由。
//!   それ以外は `fandhe_ai::tape()` 経由）と `fandhe_ai_autodiff::
//!   Tape::new()`（`NaiveOps`）で forward・backward を突き合わせる。
//!   - **bit 完全一致で判定**: `ZeroPad2d`・`Identity`・`Unflatten`・
//!     `Upsample(Nearest／NearestExact)`（いずれも算術を含まない純粋
//!     なコピーまたは view）。
//!   - **REQ-2 統一複合判定で判定**: `ConvTranspose1d`（`gemm_batched`
//!     を経由するため）・`Upsample(Bilinear)`。
//! - `#[ignore]`: `cuda_*`（`Device::Cuda(0)`）・`metal_*`（`cfg(
//!   target_os = "macos")`・`Device::Metal`）で同じ検証を CPU と対称
//!   に置く。実機未実測のまま出荷し `docs/perf/logs/spatial-layers-
//!   2159/README.md` へ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{ConvTranspose1d, Identity, Unflatten, Upsample, ZeroPad2d};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, InterpolateMode, Tensor};

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

/// `nn::ConvTranspose1d::bind` は `&fandhe_ai_autodiff::Tape` を要求
/// するが、facade の [`fandhe_ai::Tape`] は内部の `fandhe_ai_autodiff::
/// Tape`（`pub(crate)` タプルフィールド）を到達不能に保つ newtype
/// のため（REQ-12「任意 `BackendOps` 実装を注入できる公開 API を
/// 設けない」・`crates/facade/src/lib.rs::Tape` doc 参照）、`bind` を
/// 呼ぶテストでは facade の `tape()`／`tape_for` を経由せず、
/// `fandhe_ai::tape_for` の内部実装（`resolve_ops`）と同じ構成で
/// `fandhe_ai_autodiff::Tape` を直接組み立てる。CPU は常に成功する
/// ため `fandhe_ai::tape_for(Device::Cpu)` と等価（`CpuBackendOps` を
/// 直接束ねるだけ）。CUDA／Metal は `#[ignore]` テスト専用で実機の
/// 存在を前提とするため、`resolve_ops` が行う `DeviceProvider::
/// select` 事前検証は省略する（存在しない場合は `BackendOps` 側の
/// 呼び出しが失敗し、テストは実機未接続として fail する）。
fn raw_tape_for(device: Device) -> fandhe_ai_autodiff::Tape {
    let ops: Box<dyn BackendOps + Send> = match device {
        Device::Cpu => Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()),
        Device::Cuda(ordinal) => Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(ordinal)),
        #[cfg(target_os = "macos")]
        Device::Metal => Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
    };
    fandhe_ai_autodiff::Tape::new_with_ops(ops)
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

// --- ZeroPad2d（bit 完全一致） ---

#[test]
fn cpu_zero_pad2d_forward_bit_exact_vs_naive() {
    let layer = ZeroPad2d::new([1, 2, 0, 1]);
    let x_shape = [1usize, 2, 3, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

    assert_bits_eq(
        "ZeroPad2d forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- Identity（bit 完全一致） ---

#[test]
fn cpu_identity_forward_bit_exact_vs_naive() {
    let layer = Identity::new();
    let x_shape = [2usize, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(2, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(2, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

    assert_bits_eq(
        "Identity forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- Unflatten（bit 完全一致） ---

#[test]
fn cpu_unflatten_forward_bit_exact_vs_naive() {
    let layer = Unflatten::new(1, vec![3, 4]).unwrap();
    let x_shape = [2usize, 12];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(3, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(3, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

    assert_bits_eq(
        "Unflatten forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- Upsample(Nearest／NearestExact)（bit 完全一致） ---

#[test]
fn cpu_upsample_nearest_forward_bit_exact_vs_naive() {
    for mode in [InterpolateMode::Nearest, InterpolateMode::NearestExact] {
        let layer = Upsample::with_scale_factor(vec![2.0, 2.0], mode).unwrap();
        let x_shape = [1usize, 1, 3, 3];

        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&leaf(4, &x_shape));
        let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&leaf(4, &x_shape));
        let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

        assert_bits_eq(
            "Upsample(Nearest系) forward: CpuBackendOps vs NaiveOps",
            &contiguous_slice(&out_cpu),
            &contiguous_slice(&out_naive),
        );
    }
}

// --- Upsample(Bilinear)（REQ-2 複合判定） ---

#[test]
fn cpu_upsample_bilinear_forward_matches_naive_reference() {
    let mode = InterpolateMode::Bilinear {
        align_corners: false,
    };
    let layer = Upsample::with_size(vec![6, 6], mode).unwrap();
    let x_shape = [1usize, 1, 3, 3];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(5, &x_shape));
    let out_cpu = layer.forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(5, &x_shape));
    let out_naive = layer.forward(&x_naive).unwrap().to_tensor();

    assert_parity(
        "Upsample(Bilinear) forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

// --- ConvTranspose1d（REQ-2 複合判定。forward／backward） ---

#[test]
fn cpu_conv_transpose1d_forward_matches_naive_reference() {
    let layer = ConvTranspose1d::new(2, 3, 3, 2, 1, 1, 1, 1, true, 9).unwrap();
    let x_shape = [1usize, 2, 5];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let out_cpu = layer.bind(&cpu_tape).forward(&x_cpu).unwrap().to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(6, &x_shape));
    let out_naive = layer
        .bind(&naive_tape)
        .forward(&x_naive)
        .unwrap()
        .to_tensor();

    assert_parity(
        "ConvTranspose1d forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_conv_transpose1d_backward_matches_naive_reference() {
    let layer = ConvTranspose1d::new(2, 3, 3, 2, 1, 1, 1, 1, true, 10).unwrap();
    let x_shape = [1usize, 2, 5];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let vars_cpu = layer.bind(&cpu_tape);
    let x_cpu = cpu_tape.make_var(&leaf(7, &x_shape));
    let out_cpu = vars_cpu.forward(&x_cpu).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");
    let dw_cpu = grads_cpu.get(&vars_cpu.weight).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let vars_naive = layer.bind(&naive_tape);
    let x_naive = naive_tape.make_var(&leaf(7, &x_shape));
    let out_naive = vars_naive.forward(&x_naive).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive
        .get(&vars_naive.weight)
        .unwrap()
        .expect("到達する");

    assert_parity(
        "ConvTranspose1d backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_parity(
        "ConvTranspose1d backward（dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn conv_transpose1d_forward_on(device: Device) -> Tensor<f32> {
    let layer = ConvTranspose1d::new(2, 3, 3, 2, 1, 1, 1, 1, true, 9).unwrap();
    let tape = raw_tape_for(device);
    let x = tape.make_var(&leaf(6, &[1, 2, 5]));
    layer.bind(&tape).forward(&x).unwrap().to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv_transpose1d_forward_matches_cpu() {
    let metal_out = conv_transpose1d_forward_on(Device::Metal);
    let cpu_out = conv_transpose1d_forward_on(Device::Cpu);

    assert_parity(
        "ConvTranspose1d forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "metal_conv_transpose1d_forward_matches_cpu[out]",
        &contiguous_slice(&metal_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv_transpose1d_forward_matches_cpu() {
    let cuda_out = conv_transpose1d_forward_on(Device::Cuda(0));
    let cpu_out = conv_transpose1d_forward_on(Device::Cpu);

    assert_parity(
        "ConvTranspose1d forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "cuda_conv_transpose1d_forward_matches_cpu[out]",
        &contiguous_slice(&cuda_out),
    );
}

fn upsample_bilinear_forward_on(device: Device) -> Tensor<f32> {
    let mode = InterpolateMode::Bilinear {
        align_corners: false,
    };
    let layer = Upsample::with_size(vec![6, 6], mode).unwrap();
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(5, &[1, 1, 3, 3]));
    layer.forward(&x).unwrap().to_tensor()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_upsample_bilinear_forward_matches_cpu() {
    let metal_out = upsample_bilinear_forward_on(Device::Metal);
    let cpu_out = upsample_bilinear_forward_on(Device::Cpu);

    assert_parity(
        "Upsample(Bilinear) forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "metal_upsample_bilinear_forward_matches_cpu[out]",
        &contiguous_slice(&metal_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_upsample_bilinear_forward_matches_cpu() {
    let cuda_out = upsample_bilinear_forward_on(Device::Cuda(0));
    let cpu_out = upsample_bilinear_forward_on(Device::Cpu);

    assert_parity(
        "Upsample(Bilinear) forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        "cuda_upsample_bilinear_forward_matches_cpu[out]",
        &contiguous_slice(&cuda_out),
    );
}
