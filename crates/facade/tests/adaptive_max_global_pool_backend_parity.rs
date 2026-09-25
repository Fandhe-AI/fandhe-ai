//! `nn::{AdaptiveMaxPool2d, AdaptiveMaxPool1d, GlobalPool}`（イシュー
//! #2160）の facade 到達経路（`fandhe_ai_autodiff::nn` 経由。**内部
//! クレート限定・facade 未公開**——`crate::adaptive_max_pool_ops`
//! モジュール doc・`crates/facade/src/lib.rs::
//! AdaptiveMaxGlobalPoolHoldDoctestGuard` 参照）の受け入れ条件対応
//! テスト（`pooling_backend_parity.rs` と同型）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。`eval::*` 参照
//!   実装へ強制フォールバック）で forward／backward を bit 完全一致で
//!   突き合わせる（純粋な選択演算のため `assert_parity` の許容誤差
//!   ではなく `assert_eq!` の bit 一致を主張する）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU
//!   tape と比較する。本 PR 時点では CUDA／Metal とも専用カーネル
//!   未実装（`Unsupported` → ホストフォールバック）のため CPU と
//!   bit 完全一致する契約。未実測は `docs/perf/logs/
//!   adaptive-max-global-pool-2160/README.md` へ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{AdaptiveMaxPool1d, AdaptiveMaxPool2d, GlobalPool, GlobalPoolMode};
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする
/// （`pooling_backend_parity.rs::VarSource` と同じ理由・同じ構成）。
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

fn contiguous_slice_i32(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- AdaptiveMaxPool2d（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_adaptive_max_pool2d_forward_matches_naive_reference() {
    let shape = [1usize, 1, 5, 7];
    let layer = AdaptiveMaxPool2d::new([2, 3]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &shape));
    let (out_cpu, idx_cpu) = layer
        .forward(&x_cpu)
        .expect("adaptive_max_pool2d: 常に成功する");
    let out_cpu = out_cpu.to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &shape));
    let (out_naive, idx_naive) = layer
        .forward(&x_naive)
        .expect("adaptive_max_pool2d: 常に成功する");
    let out_naive = out_naive.to_tensor();

    assert_eq!(
        contiguous_slice(&out_cpu),
        contiguous_slice(&out_naive),
        "adaptive_max_pool2d: 純粋な選択演算のため bit 同一のはず"
    );
    assert_eq!(
        contiguous_slice_i32(&idx_cpu),
        contiguous_slice_i32(&idx_naive),
        "adaptive_max_pool2d: 索引も bit 同一のはず"
    );
}

#[test]
fn cpu_adaptive_max_pool2d_backward_matches_naive_reference() {
    let shape = [1usize, 1, 5, 7];
    let layer = AdaptiveMaxPool2d::new([2, 3]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(2, &shape));
    let (out_cpu, _idx) = layer.forward(&x_cpu).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(2, &shape));
    let (out_naive, _idx2) = layer.forward(&x_naive).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_eq!(
        contiguous_slice(dx_cpu),
        contiguous_slice(dx_naive),
        "adaptive_max_pool2d backward: scatter_add ベース VJP のため bit 同一のはず"
    );
}

// --- AdaptiveMaxPool1d（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_adaptive_max_pool1d_forward_and_backward_matches_naive_reference() {
    let shape = [1usize, 1, 7];
    let layer = AdaptiveMaxPool1d::new(3).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(3, &shape));
    let (out_cpu, idx_cpu) = layer.forward(&x_cpu).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(3, &shape));
    let (out_naive, idx_naive) = layer.forward(&x_naive).unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

    assert_eq!(
        contiguous_slice(&out_cpu.to_tensor()),
        contiguous_slice(&out_naive.to_tensor()),
        "adaptive_max_pool1d forward: bit 同一のはず"
    );
    assert_eq!(
        contiguous_slice_i32(&idx_cpu),
        contiguous_slice_i32(&idx_naive),
        "adaptive_max_pool1d: 索引も bit 同一のはず"
    );
    assert_eq!(
        contiguous_slice(dx_cpu),
        contiguous_slice(dx_naive),
        "adaptive_max_pool1d backward: bit 同一のはず"
    );
}

// --- GlobalPool（属性なし: CPU vs NaiveOps。Avg／Max・rank 3／4） ---

#[test]
fn cpu_global_pool_forward_and_backward_matches_naive_reference() {
    for mode in [GlobalPoolMode::Avg, GlobalPoolMode::Max] {
        for keepdims in [true, false] {
            let layer = GlobalPool::new(mode, keepdims);

            for shape in [vec![1usize, 2, 3, 4], vec![2usize, 3, 5]] {
                let cpu_tape = fandhe_ai::tape();
                let x_cpu = cpu_tape.make_var(&leaf(7, &shape));
                let out_cpu = layer.forward(&x_cpu).unwrap();
                let loss_cpu = out_cpu.sum(None).unwrap();
                let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
                let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");

                let naive_tape = fandhe_ai_autodiff::Tape::new();
                let x_naive = naive_tape.make_var(&leaf(7, &shape));
                let out_naive = layer.forward(&x_naive).unwrap();
                let loss_naive = out_naive.sum(None).unwrap();
                let grads_naive = naive_tape.backward(&loss_naive).unwrap();
                let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");

                assert_eq!(
                    contiguous_slice(&out_cpu.to_tensor()),
                    contiguous_slice(&out_naive.to_tensor()),
                    "GlobalPool({mode:?}, keepdims={keepdims}) shape={shape:?} forward: bit 同一のはず"
                );
                assert_eq!(
                    contiguous_slice(dx_cpu),
                    contiguous_slice(dx_naive),
                    "GlobalPool({mode:?}, keepdims={keepdims}) shape={shape:?} backward: bit 同一のはず"
                );
            }
        }
    }
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn adaptive_max_pool2d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 1, 5, 7]));
    let layer = AdaptiveMaxPool2d::new([2, 3]).unwrap();
    layer
        .forward(&x)
        .expect("adaptive_max_pool2d: 常に成功する")
        .0
        .to_tensor()
}

fn global_pool_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 3, 4]));
    let layer = GlobalPool::new(GlobalPoolMode::Max, true);
    layer
        .forward(&x)
        .expect("GlobalPool: 常に成功する")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`pooling_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_adaptive_max_pool2d_forward_matches_cpu() {
    let metal_out = adaptive_max_pool2d_forward_on(Device::Metal);
    let cpu_out = adaptive_max_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "adaptive_max_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_global_pool_forward_matches_cpu() {
    let metal_out = global_pool_forward_on(Device::Metal);
    let cpu_out = global_pool_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "GlobalPool: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_adaptive_max_pool2d_forward_matches_cpu() {
    let cuda_out = adaptive_max_pool2d_forward_on(Device::Cuda(0));
    let cpu_out = adaptive_max_pool2d_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "adaptive_max_pool2d: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_global_pool_forward_matches_cpu() {
    let cuda_out = global_pool_forward_on(Device::Cuda(0));
    let cpu_out = global_pool_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "GlobalPool: BackendOps 未実装フォールバックのため CPU と bit 完全一致のはず"
    );
}
