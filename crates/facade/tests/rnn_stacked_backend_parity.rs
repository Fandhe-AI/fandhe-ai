//! `nn::rnn_stacked::{StackedRnn, StackedLstm, StackedGru}`（イシュー
//! #2164・親 #2131）のバックエンド間 parity（REQ-2）対応テスト。
//! `spatial_layers_backend_parity.rs` と同型: facade は本 3 型を
//! 再エクスポートしていない（`docs/autodiff-rnn-stacked-config-
//! decision.md` §8・`RnnConfigHoldDoctestGuard`）ため、
//! `fandhe_ai_autodiff::nn::*` を直接 `use` する。
//!
//! - 属性なし: `CpuBackendOps` と `NaiveOps`（`fandhe_ai_autodiff::
//!   Tape::new()`）で forward・backward を突き合わせる（REQ-2 統一
//!   複合判定。`forward_seq`／`bind` は `&fandhe_ai_autodiff::Tape` を
//!   要求するため、`spatial_layers_backend_parity.rs::raw_tape_for` と
//!   同型のヘルパーを使う）。対象は L=2・双方向・`dropout=0`（eval 相当）
//!   の RNN／LSTM／GRU。
//! - `#[ignore]`: `cuda_*`（`Device::Cuda(0)`）・`metal_*`（`cfg(
//!   target_os = "macos")`）で同じ検証を CPU と対称に置く。実機未実測
//!   のまま出荷し `docs/perf/logs/rnn-stacked-2164/README.md` へ
//!   申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::nn::{RnnConfig, StackedGru, StackedLstm, StackedRnn};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// [`crates/facade/tests/spatial_layers_backend_parity.rs::
/// raw_tape_for`] と同型（`StackedRnn`／`StackedLstm`／`StackedGru::
/// forward_seq`／`.bind()` は `&fandhe_ai_autodiff::Tape` を要求する
/// ため、facade の `fandhe_ai::Tape` newtype 越しには呼べない。
/// REQ-12「任意 `BackendOps` 実装を注入できる公開 API を設けない」に
/// 従う facade 側制約の回避ではなく、内部クレート型を直接テストする
/// ためのテスト専用配線）。
fn raw_tape_for(device: Device) -> fandhe_ai_autodiff::Tape {
    let ops: Box<dyn BackendOps + Send> = match device {
        Device::Cpu => Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()),
        Device::Cuda(ordinal) => Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(ordinal)),
        #[cfg(target_os = "macos")]
        Device::Metal => Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
    };
    fandhe_ai_autodiff::Tape::new_with_ops(ops)
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

const T: usize = 3;
const B: usize = 2;
const D: usize = 2;
const H: usize = 3;

fn config() -> RnnConfig {
    RnnConfig::new().with_num_layers(2).with_bidirectional(true)
}

// --- StackedRnn（forward・backward。REQ-2 複合判定） ---

#[test]
fn cpu_stacked_rnn_forward_and_backward_matches_naive_reference() {
    let x_shape = [T, B, D];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let stacked_cpu = StackedRnn::new(D, H, true, 11, config()).unwrap();
    let out_cpu = stacked_cpu
        .forward_seq(&cpu_tape, &leaf(1, &x_shape), None)
        .unwrap();
    let loss_cpu = out_cpu.outputs.last().copied().unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu
        .get(&out_cpu.params[0].weight_ih)
        .unwrap()
        .expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let stacked_naive = StackedRnn::new(D, H, true, 11, config()).unwrap();
    let out_naive = stacked_naive
        .forward_seq(&naive_tape, &leaf(1, &x_shape), None)
        .unwrap();
    let loss_naive = out_naive
        .outputs
        .last()
        .copied()
        .unwrap()
        .sum(None)
        .unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive
        .get(&out_naive.params[0].weight_ih)
        .unwrap()
        .expect("到達する");

    for (t, (a, e)) in out_cpu
        .outputs
        .iter()
        .zip(out_naive.outputs.iter())
        .enumerate()
    {
        assert_parity(
            &format!("StackedRnn forward[t={t}]: CpuBackendOps vs NaiveOps"),
            &contiguous_slice(&a.to_tensor()),
            &contiguous_slice(&e.to_tensor()),
        );
    }
    assert_parity(
        "StackedRnn backward（layer0 dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

// --- StackedLstm（forward。REQ-2 複合判定） ---

#[test]
fn cpu_stacked_lstm_forward_matches_naive_reference() {
    let x_shape = [T, B, D];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let stacked_cpu = StackedLstm::new(D, H, true, 21, config()).unwrap();
    let out_cpu = stacked_cpu
        .forward_seq(&cpu_tape, &leaf(2, &x_shape), None, None)
        .unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let stacked_naive = StackedLstm::new(D, H, true, 21, config()).unwrap();
    let out_naive = stacked_naive
        .forward_seq(&naive_tape, &leaf(2, &x_shape), None, None)
        .unwrap();

    for (t, (a, e)) in out_cpu
        .outputs
        .iter()
        .zip(out_naive.outputs.iter())
        .enumerate()
    {
        assert_parity(
            &format!("StackedLstm forward[t={t}]: CpuBackendOps vs NaiveOps"),
            &contiguous_slice(&a.to_tensor()),
            &contiguous_slice(&e.to_tensor()),
        );
    }
}

// --- StackedGru（forward。REQ-2 複合判定） ---

#[test]
fn cpu_stacked_gru_forward_matches_naive_reference() {
    let x_shape = [T, B, D];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let stacked_cpu = StackedGru::new(D, H, true, 31, config()).unwrap();
    let out_cpu = stacked_cpu
        .forward_seq(&cpu_tape, &leaf(3, &x_shape), None)
        .unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let stacked_naive = StackedGru::new(D, H, true, 31, config()).unwrap();
    let out_naive = stacked_naive
        .forward_seq(&naive_tape, &leaf(3, &x_shape), None)
        .unwrap();

    for (t, (a, e)) in out_cpu
        .outputs
        .iter()
        .zip(out_naive.outputs.iter())
        .enumerate()
    {
        assert_parity(
            &format!("StackedGru forward[t={t}]: CpuBackendOps vs NaiveOps"),
            &contiguous_slice(&a.to_tensor()),
            &contiguous_slice(&e.to_tensor()),
        );
    }
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn stacked_rnn_forward_on(device: Device) -> Vec<f32> {
    let tape = raw_tape_for(device);
    let stacked = StackedRnn::new(D, H, true, 11, config()).unwrap();
    let out = stacked
        .forward_seq(&tape, &leaf(1, &[T, B, D]), None)
        .unwrap();
    let mut data = Vec::new();
    for step in &out.outputs {
        data.extend_from_slice(&contiguous_slice(&step.to_tensor()));
    }
    data
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_stacked_rnn_forward_matches_cpu() {
    let metal_out = stacked_rnn_forward_on(Device::Metal);
    let cpu_out = stacked_rnn_forward_on(Device::Cpu);

    assert_parity(
        "StackedRnn forward: Metal raw_tape_for vs CPU raw_tape_for",
        &metal_out,
        &cpu_out,
    );
    print_fold_bits("metal_stacked_rnn_forward_matches_cpu[out]", &metal_out);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_stacked_rnn_forward_matches_cpu() {
    let cuda_out = stacked_rnn_forward_on(Device::Cuda(0));
    let cpu_out = stacked_rnn_forward_on(Device::Cpu);

    assert_parity(
        "StackedRnn forward: CUDA raw_tape_for vs CPU raw_tape_for",
        &cuda_out,
        &cpu_out,
    );
    print_fold_bits("cuda_stacked_rnn_forward_matches_cpu[out]", &cuda_out);
}
