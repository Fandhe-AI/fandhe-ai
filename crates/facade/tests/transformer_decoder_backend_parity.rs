//! `nn::{TransformerDecoderLayer, Transformer}`（イシュー #2165・親
//! #2131・#2068 の対）のバックエンド間 parity（REQ-2）対応テスト。
//! `rnn_stacked_backend_parity.rs` と同型: facade は本 2 型を
//! 再エクスポートしていない（`docs/autodiff-transformer-decoder-
//! decision.md` §承認事項・`TransformerDecoderHoldDoctestGuard`）ため、
//! `fandhe_ai_autodiff::nn::*` を直接 `use` する。
//!
//! - 属性なし: `CpuBackendOps` と `NaiveOps`（`fandhe_ai_autodiff::
//!   Tape::new()`）で forward・backward を突き合わせる（REQ-2 統一
//!   複合判定。`bind`／`forward` は `&fandhe_ai_autodiff::Tape` を
//!   要求するため、`rnn_stacked_backend_parity.rs::raw_tape_for` と
//!   同型のヘルパーを使う）。対象は `TransformerDecoderLayer` 単体
//!   （`S != L` の memory・causal あり）と小さい `Transformer`
//!   （encoder 2 層・decoder 2 層）。
//! - `#[ignore]`: `cuda_*`（`Device::Cuda(0)`）・`metal_*`（`cfg(
//!   target_os = "macos")`）で同じ検証を CPU と対称に置く。実機未実測
//!   のまま出荷し `docs/perf/logs/transformer-decoder-2165/README.md`
//!   へ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::nn::{
    FeedForwardActivation, Transformer, TransformerConfig, TransformerDecoderLayer,
};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// [`crates/facade/tests/rnn_stacked_backend_parity.rs::raw_tape_for`]
/// と同型（`TransformerDecoderLayer::bind`／`Transformer::bind` は
/// `&fandhe_ai_autodiff::Tape` を要求するため、facade の `fandhe_ai::Tape`
/// newtype 越しには呼べない。REQ-12「任意 `BackendOps` 実装を注入
/// できる公開 API を設けない」に従う facade 側制約の回避ではなく、
/// 内部クレート型を直接テストするためのテスト専用配線）。
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

const D_MODEL: usize = 4;
const NUM_HEADS: usize = 2;
const DIM_FF: usize = 8;
const B: usize = 2;
const L: usize = 3;
const S: usize = 5;

// --- TransformerDecoderLayer（forward・backward。REQ-2 複合判定） ---

#[test]
fn cpu_transformer_decoder_layer_forward_and_backward_matches_naive_reference() {
    let tgt_shape = [B, L, D_MODEL];
    let mem_shape = [B, S, D_MODEL];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let layer_cpu = TransformerDecoderLayer::new(
        D_MODEL,
        NUM_HEADS,
        DIM_FF,
        FeedForwardActivation::Relu,
        1e-5,
        11,
    )
    .unwrap();
    let bound_cpu = layer_cpu.bind(&cpu_tape);
    let tgt_cpu = cpu_tape.var(&leaf(1, &tgt_shape));
    let mem_cpu = cpu_tape.var(&leaf(2, &mem_shape));
    let out_cpu = bound_cpu
        .forward(&tgt_cpu, &mem_cpu, None, None, true, false)
        .unwrap();
    let loss_cpu = out_cpu.mean(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu
        .get(&bound_cpu.self_attn.q.weight)
        .unwrap()
        .expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let layer_naive = TransformerDecoderLayer::new(
        D_MODEL,
        NUM_HEADS,
        DIM_FF,
        FeedForwardActivation::Relu,
        1e-5,
        11,
    )
    .unwrap();
    let bound_naive = layer_naive.bind(&naive_tape);
    let tgt_naive = naive_tape.var(&leaf(1, &tgt_shape));
    let mem_naive = naive_tape.var(&leaf(2, &mem_shape));
    let out_naive = bound_naive
        .forward(&tgt_naive, &mem_naive, None, None, true, false)
        .unwrap();
    let loss_naive = out_naive.mean(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive
        .get(&bound_naive.self_attn.q.weight)
        .unwrap()
        .expect("到達する");

    assert_parity(
        "TransformerDecoderLayer forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu.to_tensor()),
        &contiguous_slice(&out_naive.to_tensor()),
    );
    assert_parity(
        "TransformerDecoderLayer backward（self_attn.q dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
    print_fold_bits(
        "cpu_transformer_decoder_layer_forward_and_backward_matches_naive_reference[out]",
        &contiguous_slice(&out_cpu.to_tensor()),
    );
}

// --- Transformer（encoder 2 層・decoder 2 層。forward。REQ-2 複合判定） ---

fn transformer_config() -> TransformerConfig {
    TransformerConfig::new(D_MODEL, NUM_HEADS)
        .with_num_encoder_layers(2)
        .with_num_decoder_layers(2)
        .with_dim_feedforward(DIM_FF)
}

#[test]
fn cpu_transformer_forward_matches_naive_reference() {
    let src_shape = [B, S, D_MODEL];
    let tgt_shape = [B, L, D_MODEL];

    let cpu_tape = raw_tape_for(Device::Cpu);
    let model_cpu = Transformer::new(&transformer_config(), 7).unwrap();
    let bound_cpu = model_cpu.bind(&cpu_tape);
    let src_cpu = cpu_tape.var(&leaf(3, &src_shape));
    let tgt_cpu = cpu_tape.var(&leaf(4, &tgt_shape));
    let out_cpu = bound_cpu
        .forward(&src_cpu, &tgt_cpu, None, None, None, true)
        .unwrap();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let model_naive = Transformer::new(&transformer_config(), 7).unwrap();
    let bound_naive = model_naive.bind(&naive_tape);
    let src_naive = naive_tape.var(&leaf(3, &src_shape));
    let tgt_naive = naive_tape.var(&leaf(4, &tgt_shape));
    let out_naive = bound_naive
        .forward(&src_naive, &tgt_naive, None, None, None, true)
        .unwrap();

    assert_parity(
        "Transformer forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu.to_tensor()),
        &contiguous_slice(&out_naive.to_tensor()),
    );
    print_fold_bits(
        "cpu_transformer_forward_matches_naive_reference[out]",
        &contiguous_slice(&out_cpu.to_tensor()),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn transformer_decoder_layer_forward_on(device: Device) -> Vec<f32> {
    let tape = raw_tape_for(device);
    let layer = TransformerDecoderLayer::new(
        D_MODEL,
        NUM_HEADS,
        DIM_FF,
        FeedForwardActivation::Relu,
        1e-5,
        11,
    )
    .unwrap();
    let bound = layer.bind(&tape);
    let tgt = tape.var(&leaf(1, &[B, L, D_MODEL]));
    let memory = tape.var(&leaf(2, &[B, S, D_MODEL]));
    let out = bound
        .forward(&tgt, &memory, None, None, true, false)
        .unwrap();
    contiguous_slice(&out.to_tensor())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_transformer_decoder_layer_forward_matches_cpu() {
    let metal_out = transformer_decoder_layer_forward_on(Device::Metal);
    let cpu_out = transformer_decoder_layer_forward_on(Device::Cpu);

    assert_parity(
        "TransformerDecoderLayer forward: Metal raw_tape_for vs CPU raw_tape_for",
        &metal_out,
        &cpu_out,
    );
    print_fold_bits(
        "metal_transformer_decoder_layer_forward_matches_cpu[out]",
        &metal_out,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_transformer_decoder_layer_forward_matches_cpu() {
    let cuda_out = transformer_decoder_layer_forward_on(Device::Cuda(0));
    let cpu_out = transformer_decoder_layer_forward_on(Device::Cpu);

    assert_parity(
        "TransformerDecoderLayer forward: CUDA raw_tape_for vs CPU raw_tape_for",
        &cuda_out,
        &cpu_out,
    );
    print_fold_bits(
        "cuda_transformer_decoder_layer_forward_matches_cpu[out]",
        &cuda_out,
    );
}

fn transformer_forward_on(device: Device) -> Vec<f32> {
    let tape = raw_tape_for(device);
    let model = Transformer::new(&transformer_config(), 7).unwrap();
    let bound = model.bind(&tape);
    let src = tape.var(&leaf(3, &[B, S, D_MODEL]));
    let tgt = tape.var(&leaf(4, &[B, L, D_MODEL]));
    let out = bound.forward(&src, &tgt, None, None, None, true).unwrap();
    contiguous_slice(&out.to_tensor())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_transformer_forward_matches_cpu() {
    let metal_out = transformer_forward_on(Device::Metal);
    let cpu_out = transformer_forward_on(Device::Cpu);

    assert_parity(
        "Transformer forward: Metal raw_tape_for vs CPU raw_tape_for",
        &metal_out,
        &cpu_out,
    );
    print_fold_bits("metal_transformer_forward_matches_cpu[out]", &metal_out);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_transformer_forward_matches_cpu() {
    let cuda_out = transformer_forward_on(Device::Cuda(0));
    let cpu_out = transformer_forward_on(Device::Cpu);

    assert_parity(
        "Transformer forward: CUDA raw_tape_for vs CPU raw_tape_for",
        &cuda_out,
        &cpu_out,
    );
    print_fold_bits("cuda_transformer_forward_matches_cpu[out]", &cuda_out);
}
